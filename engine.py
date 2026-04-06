"""Operational simulator and reward shaping for ATC optimization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import pstdev
from typing import Dict, Iterable, List, Tuple

try:
    from .models import (
        FlightRecord,
        PriorityClass,
        RunwaySpec,
        SlotAssignment,
        TaskDefinition,
        TaskMetrics,
    )
except ImportError:
    from models import (
        FlightRecord,
        PriorityClass,
        RunwaySpec,
        SlotAssignment,
        TaskDefinition,
        TaskMetrics,
    )


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

PRIORITY_DELAY_TOLERANCE: Dict[PriorityClass, int] = {
    PriorityClass.NORMAL: 35,
    PriorityClass.CONNECTION: 20,
    PriorityClass.MEDICAL: 10,
    PriorityClass.EMERGENCY: 5,
}


@dataclass
class SimulationOutcome:
    """Full simulator output used by the environment and graders."""

    metrics: TaskMetrics
    diagnostics: List[str]
    recommendations: List[str]
    normalized_score: float


def empty_metrics() -> TaskMetrics:
    """Return a zeroed metric object."""

    return TaskMetrics()


def _capacity_spacing(runway: RunwaySpec) -> int:
    base_gap = max(2, round(60 / runway.hourly_capacity))
    return max(2, round(base_gap * runway.weather_penalty))


def _delay_for(flight: FlightRecord, assigned_minute: int) -> int:
    return abs(assigned_minute - flight.scheduled_minute)


def simulate_plan(task: TaskDefinition, proposal: Iterable[SlotAssignment]) -> SimulationOutcome:
    """Evaluate a candidate runway plan for a task."""

    flights_by_id = {flight.flight_id: flight for flight in task.flights}
    runways_by_id = {runway.runway_id: runway for runway in task.runways}
    assignments = list(proposal)

    diagnostics: List[str] = []
    recommendations: List[str] = []
    unknown_assignments = 0
    invalid_assignments = 0

    assignment_map: Dict[str, SlotAssignment] = {}
    duplicate_assignments = max(0, len(assignments) - len({item.flight_id for item in assignments}))
    for assignment in assignments:
        if assignment.flight_id in assignment_map:
            diagnostics.append(
                f"{assignment.flight_id} appears more than once; only the last assignment is used."
            )
        assignment_map[assignment.flight_id] = assignment
    if duplicate_assignments:
        invalid_assignments += duplicate_assignments

    scheduled_by_runway: Dict[str, List[Tuple[int, SlotAssignment, FlightRecord]]] = defaultdict(list)
    delays: Dict[str, int] = {}
    per_airline_delays: Dict[str, List[int]] = defaultdict(list)
    fuel_burn = 0.0
    priority_violations = 0
    capacity_violations = 0

    for assignment in assignment_map.values():
        flight = flights_by_id.get(assignment.flight_id)
        if flight is None:
            unknown_assignments += 1
            diagnostics.append(f"Unknown flight id {assignment.flight_id} in proposal.")
            continue

        if assignment.runway not in flight.allowed_runways:
            invalid_assignments += 1
            diagnostics.append(
                f"{flight.flight_id} cannot use runway {assignment.runway}; allowed runways are {', '.join(flight.allowed_runways)}."
            )
            continue

        if assignment.runway not in runways_by_id:
            invalid_assignments += 1
            diagnostics.append(f"Runway {assignment.runway} is not available in this task.")
            continue

        runway = runways_by_id[assignment.runway]
        if flight.operation not in runway.allowed_operations:
            invalid_assignments += 1
            diagnostics.append(
                f"Runway {assignment.runway} cannot handle {flight.operation.value} operations for {flight.flight_id}."
            )
            continue

        if assignment.assigned_minute < flight.earliest_minute or assignment.assigned_minute > flight.latest_minute:
            invalid_assignments += 1
            diagnostics.append(
                f"{flight.flight_id} is assigned outside its feasible window [{flight.earliest_minute}, {flight.latest_minute}]."
            )
            continue

        delay = _delay_for(flight, assignment.assigned_minute)
        if assignment.hold_minutes > 0 and abs(assignment.hold_minutes - delay) > 5:
            diagnostics.append(
                f"{flight.flight_id} declares hold_minutes={assignment.hold_minutes}, but the actual delay is {delay}."
            )
        delays[flight.flight_id] = delay
        per_airline_delays[flight.airline].append(delay)
        fuel_burn += delay * flight.fuel_burn_per_minute
        scheduled_by_runway[assignment.runway].append((assignment.assigned_minute, assignment, flight))

        tolerance = PRIORITY_DELAY_TOLERANCE[flight.priority]
        if delay > tolerance:
            priority_violations += 1
            diagnostics.append(
                f"{flight.flight_id} exceeds the {flight.priority.value} delay tolerance of {tolerance} minutes."
            )

    missing_assignments = max(0, len(task.flights) - len(delays))
    for flight in task.flights:
        if flight.flight_id not in delays:
            diagnostics.append(f"{flight.flight_id} has no valid runway assignment.")

    conflict_count = 0
    for runway_id, runway_items in scheduled_by_runway.items():
        runway_spec = runways_by_id[runway_id]
        runway_items.sort(key=lambda item: item[0])
        required_gap_floor = _capacity_spacing(runway_spec)
        for idx in range(1, len(runway_items)):
            prev_time, _, prev_flight = runway_items[idx - 1]
            curr_time, _, curr_flight = runway_items[idx]
            wake_gap = SEPARATION_BY_WAKE[(prev_flight.wake_class.value, curr_flight.wake_class.value)]
            required_gap = max(required_gap_floor, wake_gap)
            actual_gap = curr_time - prev_time
            if actual_gap < required_gap:
                conflict_count += 1
                capacity_violations += 1
                diagnostics.append(
                    f"Runway {runway_id} has {prev_flight.flight_id}->{curr_flight.flight_id} spaced {actual_gap} minutes apart; needs {required_gap}."
                )

    total_delay = sum(delays.values())
    max_delay = max(delays.values(), default=0)

    completeness = max(
        0.0,
        min(
            1.0,
            (len(delays) - unknown_assignments - invalid_assignments) / max(1, len(task.flights)),
        ),
    )
    conflict_free_ratio = max(
        0.0,
        1.0 - (conflict_count / max(1, len(task.flights) - 1)),
    )
    priority_handling = max(
        0.0,
        1.0 - (priority_violations / max(1, sum(1 for flight in task.flights if flight.priority != PriorityClass.NORMAL))),
    )
    delay_efficiency = max(0.0, 1.0 - (total_delay / task.delay_budget))
    fuel_efficiency = max(0.0, 1.0 - (fuel_burn / task.fuel_budget))

    airline_averages = {
        airline: round(sum(values) / len(values), 2)
        for airline, values in per_airline_delays.items()
        if values
    }
    spread = pstdev(list(airline_averages.values())) if len(airline_averages) > 1 else 0.0
    fairness = max(0.0, 1.0 - (spread / task.fairness_tolerance))

    if completeness < 1.0:
        recommendations.append("Cover every flight in the scenario before refining the sequence.")
    if conflict_count > 0:
        recommendations.append("Increase spacing on the affected runway or move some flights to a parallel runway.")
    if duplicate_assignments > 0:
        recommendations.append("Avoid duplicate flight entries; submit one definitive assignment per flight.")
    if priority_violations > 0:
        recommendations.append("Pull medical, emergency, and connection-sensitive flights closer to the front of the sequence.")
    if fairness < 0.7:
        recommendations.append("Redistribute delay so one airline is not absorbing most of the disruption.")
    if fuel_efficiency < 0.7:
        recommendations.append("Reduce airborne or taxi delay on the highest-burn arrivals and heavy aircraft.")
    if not recommendations:
        recommendations.append("The plan is operationally strong; minor gains remain in passenger delay reduction.")

    normalized_score = (
        0.24 * completeness
        + 0.24 * conflict_free_ratio
        + 0.18 * priority_handling
        + 0.16 * delay_efficiency
        + 0.10 * fairness
        + 0.08 * fuel_efficiency
    )
    normalized_score *= completeness
    if conflict_count > 0:
        normalized_score *= conflict_free_ratio
    normalized_score = max(0.0, min(1.0, normalized_score))

    metrics = TaskMetrics(
        schedule_completeness=round(completeness, 4),
        conflict_free_ratio=round(conflict_free_ratio, 4),
        priority_handling=round(priority_handling, 4),
        delay_efficiency=round(delay_efficiency, 4),
        fairness=round(fairness, 4),
        fuel_efficiency=round(fuel_efficiency, 4),
        agent_judgment=0.0,
        overall_score=round(normalized_score, 4),
        total_delay_minutes=total_delay,
        max_delay_minutes=max_delay,
        estimated_fuel_burn=round(fuel_burn, 2),
        conflict_count=conflict_count,
        capacity_violations=capacity_violations,
        priority_violations=priority_violations,
        missing_assignments=missing_assignments,
        invalid_assignments=invalid_assignments + unknown_assignments,
        per_airline_average_delay=airline_averages,
    )
    return SimulationOutcome(
        metrics=metrics,
        diagnostics=diagnostics[:12],
        recommendations=recommendations[:6],
        normalized_score=normalized_score,
    )
