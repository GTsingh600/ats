"""Deterministic baseline planner used by inference and tests."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

try:
    from .engine import SimulationOutcome, simulate_plan
    from .models import (
        ATCOptimizationObservation,
        FlightRecord,
        PriorityClass,
        RunwaySpec,
        SlotAssignment,
    )
    from .tasks import task_catalog
except ImportError:
    from engine import SimulationOutcome, simulate_plan
    from models import (
        ATCOptimizationObservation,
        FlightRecord,
        PriorityClass,
        RunwaySpec,
        SlotAssignment,
    )
    from tasks import task_catalog


SEPARATION_BY_WAKE: Dict[Tuple[str, str], int] = {
    ("H", "H"): 4,
    ("H", "M"): 5,
    ("H", "L"): 6,
    ("M", "H"): 3,
    ("M", "M"): 3,
    ("M", "L"): 4,
    ("L", "H"): 3,
    ("L", "M"): 3,
    ("L", "L"): 3,
}

PRIORITY_RANK = {
    PriorityClass.EMERGENCY: 0,
    PriorityClass.MEDICAL: 1,
    PriorityClass.CONNECTION: 2,
    PriorityClass.NORMAL: 3,
}


def _capacity_spacing(runway: RunwaySpec) -> int:
    base_gap = max(2, round(60 / runway.hourly_capacity))
    return max(2, round(base_gap * runway.weather_penalty))


def _flight_sort_key(flight: FlightRecord) -> Tuple[int, int, int, float, int]:
    return (
        PRIORITY_RANK[flight.priority],
        flight.scheduled_minute,
        0 if flight.operation.value == "arrival" else 1,
        -flight.connection_risk,
        -flight.passengers,
    )


def _outcome_key(outcome: SimulationOutcome) -> Tuple[float, float, float, float, float, float]:
    metrics = outcome.metrics
    return (
        metrics.overall_score,
        metrics.priority_handling,
        metrics.conflict_free_ratio,
        metrics.fairness,
        metrics.delay_efficiency,
        -float(metrics.total_delay_minutes),
    )


def build_heuristic_plan(observation: ATCOptimizationObservation) -> List[SlotAssignment]:
    """Create a safe, deterministic seed schedule."""

    runway_lookup = {runway.runway_id: runway for runway in observation.runways}
    runway_state: Dict[str, Tuple[int, str]] = {
        runway.runway_id: (-999, "M") for runway in observation.runways
    }
    airline_delay_totals: Dict[str, List[int]] = defaultdict(list)

    assignments: List[SlotAssignment] = []
    for flight in sorted(observation.flights, key=_flight_sort_key):
        best_choice: Tuple[float, str, int] | None = None
        for runway_id in flight.allowed_runways:
            runway = runway_lookup[runway_id]
            last_minute, last_wake = runway_state[runway_id]
            gap = max(
                _capacity_spacing(runway),
                SEPARATION_BY_WAKE[(last_wake, flight.wake_class.value)],
            )
            earliest_safe = max(flight.earliest_minute, last_minute + gap, flight.scheduled_minute)
            candidate_time = min(max(earliest_safe, flight.earliest_minute), flight.latest_minute)
            delay = abs(candidate_time - flight.scheduled_minute)
            airline_avg_delay = (
                sum(airline_delay_totals[flight.airline]) / len(airline_delay_totals[flight.airline])
                if airline_delay_totals[flight.airline]
                else 0.0
            )
            objective = (
                delay
                + 6 * PRIORITY_RANK[flight.priority]
                + 12 * flight.connection_risk
                + 0.25 * airline_avg_delay
                + (3 if runway_id.endswith("L") and flight.priority == PriorityClass.NORMAL else 0)
            )
            candidate = (objective, runway_id, candidate_time)
            if best_choice is None or candidate < best_choice:
                best_choice = candidate

        assert best_choice is not None
        _, chosen_runway, chosen_time = best_choice
        assignments.append(
            SlotAssignment(
                flight_id=flight.flight_id,
                runway=chosen_runway,
                assigned_minute=chosen_time,
                hold_minutes=max(0, chosen_time - flight.scheduled_minute),
            )
        )
        airline_delay_totals[flight.airline].append(abs(chosen_time - flight.scheduled_minute))
        runway_state[chosen_runway] = (chosen_time, flight.wake_class.value)

    assignments.sort(key=lambda item: item.assigned_minute)
    return assignments


def build_refined_plan(
    observation: ATCOptimizationObservation,
    seed_plan: List[SlotAssignment] | None = None,
    max_passes: int = 2,
) -> List[SlotAssignment]:
    """Greedily improve the seed plan using the deterministic simulator."""

    task = task_catalog().get(observation.task_id)
    if task is None:
        return seed_plan or build_heuristic_plan(observation)

    ordered_flights = sorted(observation.flights, key=_flight_sort_key)
    ordered_ids = [flight.flight_id for flight in ordered_flights]
    current_plan = list(seed_plan) if seed_plan is not None else build_heuristic_plan(observation)
    assignment_map = {assignment.flight_id: assignment for assignment in current_plan}

    for flight in ordered_flights:
        assignment_map.setdefault(
            flight.flight_id,
            SlotAssignment(
                flight_id=flight.flight_id,
                runway=flight.allowed_runways[0],
                assigned_minute=flight.scheduled_minute,
                hold_minutes=0,
            ),
        )

    best_plan = [assignment_map[flight_id] for flight_id in ordered_ids]
    best_outcome = simulate_plan(task, best_plan)

    for _ in range(max_passes):
        improved = False
        for flight in ordered_flights:
            current_assignment = assignment_map[flight.flight_id]
            best_local_assignment = current_assignment
            for runway_id in flight.allowed_runways:
                for minute in range(flight.earliest_minute, flight.latest_minute + 1):
                    if (
                        runway_id == current_assignment.runway
                        and minute == current_assignment.assigned_minute
                    ):
                        continue
                    candidate_assignment = SlotAssignment(
                        flight_id=flight.flight_id,
                        runway=runway_id,
                        assigned_minute=minute,
                        hold_minutes=max(0, minute - flight.scheduled_minute),
                    )
                    assignment_map[flight.flight_id] = candidate_assignment
                    candidate_plan = [assignment_map[item_id] for item_id in ordered_ids]
                    candidate_outcome = simulate_plan(task, candidate_plan)
                    if _outcome_key(candidate_outcome) > _outcome_key(best_outcome):
                        best_outcome = candidate_outcome
                        best_local_assignment = candidate_assignment
                        improved = True
                assignment_map[flight.flight_id] = best_local_assignment

            assignment_map[flight.flight_id] = best_local_assignment
            best_plan = [assignment_map[item_id] for item_id in ordered_ids]

        if not improved:
            break

    best_plan.sort(key=lambda item: (item.assigned_minute, item.flight_id))
    return best_plan
