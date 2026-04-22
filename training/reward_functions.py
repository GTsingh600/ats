"""Per-agent reward functions for GRPO training.

Design principles:
  1. Potential-based reward shaping (Ng et al. 1999) gives dense signal.
  2. Group-relative advantage (GRPO) avoids a learned value baseline.
  3. Role decomposition keeps AMAN/DMAN/GENERATOR/SUPERVISOR rewards separate.
  4. Theory-of-mind bonuses reward proactive coordination.
  5. Supervisor alignment shaping trains preference following.

The helpers in this module are intentionally defensive because newer TRL
versions may pass completions and metadata in slightly different shapes.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from ..engine import simulate_plan
    from .dataset import (
        _coerce_completion_text,
        parse_aman_action,
        parse_dman_action,
        parse_generator_action,
    )
    from ..models import OperationType, PriorityClass, SlotAssignment, TaskDefinition
    from ..tasks import task_catalog
    from ..multi_agent.generator import ChallengeGenerator
    from ..multi_agent.models import SupervisorProfileName
    from ..multi_agent.supervisor import SupervisorAgent
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine import simulate_plan
    from training.dataset import (
        _coerce_completion_text,
        parse_aman_action,
        parse_dman_action,
        parse_generator_action,
    )
    from models import OperationType, PriorityClass, SlotAssignment, TaskDefinition
    from tasks import task_catalog
    from multi_agent.generator import ChallengeGenerator
    from multi_agent.models import SupervisorProfileName
    from multi_agent.supervisor import SupervisorAgent

_CATALOG = None
_SUPERVISOR = SupervisorAgent()
_GENERATOR = ChallengeGenerator()
_TRACE_REWARDS = os.getenv("ATC_REWARD_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}
_DEFAULT_SUPERVISOR_PROFILE = SupervisorProfileName.SAFETY_STRICT


def _debug_reward_trace(role: str, components: Dict[str, float]) -> None:
    if not _TRACE_REWARDS:
        return
    print(f"\n[REWARD TRACE] role={role}")
    for key, value in components.items():
        print(f"  {key}: {value:.4f}")


def _get_catalog() -> Dict[str, TaskDefinition]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = task_catalog()
    return _CATALOG


def _metadata_list(value: Any, length: int, default: Any) -> List[Any]:
    if isinstance(value, list):
        if not value:
            return [default] * length
        if len(value) >= length:
            return value[:length]
        return value + [value[-1]] * (length - len(value))
    if value is None:
        return [default] * length
    return [value] * length


def _safe_supervisor_profile(profile_value: Any) -> SupervisorProfileName:
    try:
        return SupervisorProfileName(profile_value)
    except Exception:
        return _DEFAULT_SUPERVISOR_PROFILE


def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalized_cross_conflict_penalty(
    our_slots: List[SlotAssignment],
    other_slots: List[SlotAssignment],
    conflict_count: int,
    weight: float = 0.35,
) -> float:
    if conflict_count <= 0 or not our_slots or not other_slots:
        return 0.0

    our_by_runway: Dict[str, int] = {}
    other_by_runway: Dict[str, int] = {}

    for slot in our_slots:
        our_by_runway[slot.runway] = our_by_runway.get(slot.runway, 0) + 1
    for slot in other_slots:
        other_by_runway[slot.runway] = other_by_runway.get(slot.runway, 0) + 1

    max_cross_conflicts = 0
    for runway in set(our_by_runway) & set(other_by_runway):
        max_cross_conflicts += our_by_runway[runway] * other_by_runway[runway]

    if max_cross_conflicts <= 0:
        return 0.0

    normalized = min(1.0, conflict_count / max_cross_conflicts)
    return weight * normalized


def aman_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: Optional[List[str]] = None,
    dman_slots_json: Optional[List[str]] = None,
    atfm_deadlines_json: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """GRPO reward function for AMAN (Arrival Manager)."""
    catalog = _get_catalog()
    rewards: List[float] = []
    n = len(completions)

    task_id = _metadata_list(task_id, n, "")
    supervisor_profile = _metadata_list(
        supervisor_profile if supervisor_profile is not None else kwargs.get("supervisor_profile"),
        n,
        _DEFAULT_SUPERVISOR_PROFILE.value,
    )
    dman_slots_json = _metadata_list(
        dman_slots_json if dman_slots_json is not None else kwargs.get("dman_slots_json"),
        n,
        "[]",
    )
    atfm_deadlines_json = _metadata_list(
        atfm_deadlines_json if atfm_deadlines_json is not None else kwargs.get("atfm_deadlines_json"),
        n,
        "{}",
    )

    for completion, tid, profile_str, dman_json, _atfm_json in zip(
        completions, task_id, supervisor_profile, dman_slots_json, atfm_deadlines_json
    ):
        task = catalog.get(tid)
        if task is None:
            rewards.append(-1.0)
            continue

        aman_action = parse_aman_action(completion)
        if aman_action is None:
            rewards.append(-0.8)
            continue

        dman_slots = _parse_slots_json(dman_json)
        profile = _safe_supervisor_profile(profile_str)
        merged = aman_action.arrival_slots + dman_slots
        outcome = simulate_plan(task, merged)

        arrivals = [f for f in task.flights if f.operation == OperationType.ARRIVAL]
        arr_map = {s.flight_id: s for s in aman_action.arrival_slots}

        delay_total = 0
        missing = 0
        emg_ok = emg_miss = 0

        for f in arrivals:
            slot = arr_map.get(f.flight_id)
            if slot:
                delay = abs(slot.assigned_minute - f.scheduled_minute)
                delay_total += delay
                if f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL):
                    if delay <= 5:
                        emg_ok += 1
                    else:
                        emg_miss += 1
            else:
                missing += 1

        arr_count = max(1, len(arrivals))
        budget = task.delay_budget / 2.0
        delay_eff = max(0.0, 1.0 - delay_total / max(1, budget))
        coverage = 1.0 - missing / arr_count
        emg_score = emg_ok / max(1, emg_ok + emg_miss) if (emg_ok + emg_miss) else 1.0

        cross_penalty = 0.0
        if {s.runway for s in aman_action.arrival_slots} & {s.runway for s in dman_slots}:
            cross_penalty = _normalized_cross_conflict_penalty(
                aman_action.arrival_slots,
                dman_slots,
                outcome.metrics.conflict_count,
            )

        tom_bonus = _compute_tom_bonus_aman(aman_action, dman_slots, task)
        sup_align = _supervisor_alignment(outcome, task, profile)
        rationale_score = _score_rationale_quality(aman_action.rationale, task, outcome)

        # Counterfactual credit: how much does AMAN's plan improve over naive baseline?
        # cf_outcome = naive arrivals + real DMAN slots; advantage = real - counterfactual.
        cf_outcome = simulate_plan(task, _naive_arrival_slots(task) + dman_slots)
        cf_advantage = max(-1.0, min(1.0, outcome.normalized_score - cf_outcome.normalized_score))

        json_fmt = _json_format_score(completion)
        reward = (
            0.26 * delay_eff
            + 0.20 * emg_score
            + 0.17 * coverage
            + 0.12 * cf_advantage
            + 0.10 * tom_bonus
            + 0.05 * sup_align
            + 0.05 * rationale_score
            + 0.05 * json_fmt
            - cross_penalty
        )

        # Layered safety gates — hard ceilings that cannot be bought off by efficiency
        if outcome.metrics.conflict_count > 0:
            reward = min(reward, 0.30)   # conflict-free gate
        if emg_miss > 0:
            reward = min(reward, 0.40)   # emergency hard gate
        if coverage < 0.50:
            reward = max(-0.5, reward - 0.30)  # coverage floor penalty

        reward = round(max(-1.0, min(1.0, reward)), 4)
        _debug_reward_trace(
            role="AMAN",
            components={
                "delay_eff": delay_eff,
                "emg_score": emg_score,
                "coverage": coverage,
                "cf_advantage": cf_advantage,
                "tom_bonus": tom_bonus,
                "sup_align": sup_align,
                "rationale_score": rationale_score,
                "json_fmt": json_fmt,
                "cross_penalty": -cross_penalty,
                "conflict_gate": int(outcome.metrics.conflict_count > 0),
                "emg_gate": int(emg_miss > 0),
                "coverage_floor": int(coverage < 0.50),
                "final_reward": reward,
            },
        )
        rewards.append(reward)

    return rewards


def dman_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: Optional[List[str]] = None,
    aman_slots_json: Optional[List[str]] = None,
    atfm_deadlines_json: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """GRPO reward function for DMAN (Departure Manager)."""
    catalog = _get_catalog()
    rewards: List[float] = []
    n = len(completions)

    task_id = _metadata_list(task_id, n, "")
    supervisor_profile = _metadata_list(
        supervisor_profile if supervisor_profile is not None else kwargs.get("supervisor_profile"),
        n,
        _DEFAULT_SUPERVISOR_PROFILE.value,
    )
    aman_slots_json = _metadata_list(
        aman_slots_json if aman_slots_json is not None else kwargs.get("aman_slots_json"),
        n,
        "[]",
    )
    atfm_deadlines_json = _metadata_list(
        atfm_deadlines_json if atfm_deadlines_json is not None else kwargs.get("atfm_deadlines_json"),
        n,
        "{}",
    )

    for completion, tid, profile_str, aman_json, atfm_json in zip(
        completions, task_id, supervisor_profile, aman_slots_json, atfm_deadlines_json
    ):
        task = catalog.get(tid)
        if task is None:
            rewards.append(-1.0)
            continue

        dman_action = parse_dman_action(completion)
        if dman_action is None:
            rewards.append(-0.8)
            continue

        aman_slots = _parse_slots_json(aman_json)
        atfm = json.loads(atfm_json) if atfm_json else {}
        profile = _safe_supervisor_profile(profile_str)
        merged = aman_slots + dman_action.departure_slots
        outcome = simulate_plan(task, merged)

        departures = [f for f in task.flights if f.operation == OperationType.DEPARTURE]
        dep_map = {s.flight_id: s for s in dman_action.departure_slots}

        delay_total = 0
        missing = 0
        emg_ok = emg_miss = 0
        atfm_ok = atfm_viol = 0

        for f in departures:
            slot = dep_map.get(f.flight_id)
            if slot:
                delay = abs(slot.assigned_minute - f.scheduled_minute)
                delay_total += delay
                if f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL):
                    if delay <= 5:
                        emg_ok += 1
                    else:
                        emg_miss += 1
                deadline = atfm.get(f.flight_id)
                if deadline is not None:
                    if slot.assigned_minute <= deadline:
                        atfm_ok += 1
                    else:
                        atfm_viol += 1
            else:
                missing += 1

        dep_count = max(1, len(departures))
        budget = task.delay_budget / 2.0
        delay_eff = max(0.0, 1.0 - delay_total / max(1, budget))
        coverage = 1.0 - missing / dep_count
        emg_score = emg_ok / max(1, emg_ok + emg_miss) if (emg_ok + emg_miss) else 1.0
        atfm_score = atfm_ok / max(1, atfm_ok + atfm_viol) if (atfm_ok + atfm_viol) else 1.0

        cross_penalty = 0.0
        if {s.runway for s in dman_action.departure_slots} & {s.runway for s in aman_slots}:
            cross_penalty = _normalized_cross_conflict_penalty(
                dman_action.departure_slots,
                aman_slots,
                outcome.metrics.conflict_count,
            )

        tom_bonus = _compute_tom_bonus_dman(dman_action, aman_slots, task)
        sup_align = _supervisor_alignment(outcome, task, profile)
        rationale_score = _score_rationale_quality(dman_action.rationale, task, outcome)

        # Counterfactual credit: how much does DMAN's plan improve over naive baseline?
        # cf_outcome = real AMAN slots + naive departures; advantage = real - counterfactual.
        cf_outcome = simulate_plan(task, aman_slots + _naive_departure_slots(task))
        cf_advantage = max(-1.0, min(1.0, outcome.normalized_score - cf_outcome.normalized_score))

        json_fmt = _json_format_score(completion)
        reward = (
            0.23 * delay_eff
            + 0.17 * atfm_score
            + 0.16 * emg_score
            + 0.12 * coverage
            + 0.12 * cf_advantage
            + 0.10 * tom_bonus
            + 0.05 * sup_align
            + 0.03 * rationale_score
            + 0.02 * json_fmt
            - cross_penalty
        )

        # Layered safety gates
        if outcome.metrics.conflict_count > 0:
            reward = min(reward, 0.30)   # conflict-free gate
        if emg_miss > 0:
            reward = min(reward, 0.40)   # emergency hard gate
        if coverage < 0.50:
            reward = max(-0.5, reward - 0.30)  # coverage floor penalty

        reward = round(max(-1.0, min(1.0, reward)), 4)
        _debug_reward_trace(
            role="DMAN",
            components={
                "delay_eff": delay_eff,
                "atfm_score": atfm_score,
                "emg_score": emg_score,
                "coverage": coverage,
                "cf_advantage": cf_advantage,
                "tom_bonus": tom_bonus,
                "sup_align": sup_align,
                "rationale_score": rationale_score,
                "json_fmt": json_fmt,
                "cross_penalty": -cross_penalty,
                "conflict_gate": int(outcome.metrics.conflict_count > 0),
                "emg_gate": int(emg_miss > 0),
                "coverage_floor": int(coverage < 0.50),
                "final_reward": reward,
            },
        )
        rewards.append(reward)

    return rewards


def generator_reward_fn(
    completions: List[str],
    task_id: List[str],
    controller_scores: Optional[List[float]] = None,
    **kwargs: Any,
) -> List[float]:
    """GRPO reward function for GENERATOR role."""
    catalog = _get_catalog()
    rewards: List[float] = []
    n = len(completions)

    task_id = _metadata_list(task_id, n, "")
    controller_scores = _metadata_list(
        controller_scores if controller_scores is not None else kwargs.get("controller_scores"),
        n,
        0.5,
    )

    for completion, tid, ctrl_score in zip(completions, task_id, controller_scores):
        task = catalog.get(tid)
        if task is None:
            rewards.append(-1.0)
            continue

        gen_action = parse_generator_action(completion)
        if gen_action is None:
            rewards.append(-0.5)
            continue

        _, is_solvable = _GENERATOR.mutate(task, gen_action)
        reward = _GENERATOR.compute_reward(_safe_float(ctrl_score), is_solvable)
        rewards.append(round(max(-1.0, min(1.0, reward)), 4))

    return rewards


def supervisor_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: Optional[List[str]] = None,
    merged_plan_json: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """GRPO reward for SUPERVISOR role."""
    catalog = _get_catalog()
    rewards: List[float] = []
    n = len(completions)

    task_id = _metadata_list(task_id, n, "")
    supervisor_profile = _metadata_list(
        supervisor_profile if supervisor_profile is not None else kwargs.get("supervisor_profile"),
        n,
        _DEFAULT_SUPERVISOR_PROFILE.value,
    )
    merged_plan_json = _metadata_list(
        merged_plan_json if merged_plan_json is not None else kwargs.get("merged_plan_json"),
        n,
        "[]",
    )

    for completion, tid, profile_str, plan_json in zip(
        completions, task_id, supervisor_profile, merged_plan_json
    ):
        task = catalog.get(tid)
        if task is None:
            rewards.append(-1.0)
            continue

        try:
            plan_data = json.loads(plan_json)
            slots = [SlotAssignment(**s) for s in plan_data]
        except Exception:
            rewards.append(-0.5)
            continue

        outcome = simulate_plan(task, slots)
        profile = _safe_supervisor_profile(profile_str)
        score = _SUPERVISOR.score_plan(outcome, task, profile)

        supervisor_claimed = _extract_supervisor_score(completion)
        calibration_bonus = 0.0
        if supervisor_claimed is not None:
            calibration_bonus = max(0.0, 0.2 - abs(supervisor_claimed - score))

        rewards.append(round(min(1.0, score + calibration_bonus), 4))

    return rewards


def _json_format_score(completion: Any) -> float:
    """Returns 1.0 if completion contains valid JSON with expected keys, else 0.0."""
    text = _coerce_completion_text(completion)
    try:
        # Strip markdown code fences if present
        stripped = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        data = json.loads(stripped)
        if isinstance(data, dict) and (
            "arrival_slots" in data or "departure_slots" in data or "rationale" in data
        ):
            return 1.0
        return 0.5  # valid JSON but unexpected shape
    except Exception:
        # Partial credit if JSON block is present but malformed
        if "{" in text and ("slots" in text or "rationale" in text):
            return 0.2
        return 0.0


def _score_rationale_quality(
    rationale: str,
    task: TaskDefinition,
    outcome,
) -> float:
    """Rule-based rationale quality scorer — fully verifiable, no LLM required.

    Rewards agents for rationales that demonstrate awareness of the task's key
    constraints. Keeps rewards in RLVR territory (deterministic & verifiable).

    Returns a score in [0.0, 1.0].
    """
    if not rationale or not rationale.strip():
        return 0.0

    text = rationale.lower()
    score = 0.0

    # Acknowledge emergency / medical flights by ID
    emergency_flights = [
        f for f in task.flights
        if f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
    ]
    for ef in emergency_flights:
        if ef.flight_id.lower() in text:
            score += 0.20
            break  # one bonus regardless of how many emergencies are mentioned

    # Acknowledge wake turbulence separation constraints
    if any(kw in text for kw in ("wake", "turbulence", "separation", "heavy", "light")):
        score += 0.15

    # Acknowledge runway capacity or weather penalty
    if any(kw in text for kw in ("capacity", "weather", "penalty", "runway")):
        score += 0.10

    # Mention at least one flight ID from the task (shows engagement, not boilerplate)
    flight_ids_lower = {f.flight_id.lower() for f in task.flights}
    if any(fid in text for fid in flight_ids_lower):
        score += 0.10

    # Acknowledge a detected conflict if one exists
    if outcome.metrics.conflict_count > 0 and any(
        kw in text for kw in ("conflict", "spac", "gap", "collision")
    ):
        score += 0.15

    # Mention delay or fuel concern (shows cost awareness)
    if any(kw in text for kw in ("delay", "fuel", "hold", "burn", "cost")):
        score += 0.10

    # Penalty for suspiciously short rationales (< 15 chars after strip)
    if len(rationale.strip()) < 15:
        score = max(0.0, score - 0.30)

    return round(min(1.0, score), 4)


def _naive_arrival_slots(task: TaskDefinition) -> List[SlotAssignment]:
    """Baseline arrivals plan: each flight at scheduled_minute on first allowed runway.

    Used for COMA-style counterfactual credit assignment — measures how much better
    AMAN's plan is versus a naive do-nothing scheduler.
    """
    slots = []
    for f in task.flights:
        if f.operation == OperationType.ARRIVAL and f.allowed_runways:
            minute = max(f.earliest_minute, min(f.latest_minute, f.scheduled_minute))
            slots.append(SlotAssignment(
                flight_id=f.flight_id,
                runway=f.allowed_runways[0],
                assigned_minute=minute,
                hold_minutes=0,
            ))
    return slots


def _naive_departure_slots(task: TaskDefinition) -> List[SlotAssignment]:
    """Baseline departures plan: each flight at scheduled_minute on first allowed runway."""
    slots = []
    for f in task.flights:
        if f.operation == OperationType.DEPARTURE and f.allowed_runways:
            minute = max(f.earliest_minute, min(f.latest_minute, f.scheduled_minute))
            slots.append(SlotAssignment(
                flight_id=f.flight_id,
                runway=f.allowed_runways[0],
                assigned_minute=minute,
                hold_minutes=0,
            ))
    return slots


def _parse_slots_json(json_str: str) -> List[SlotAssignment]:
    try:
        data = json.loads(json_str)
        return [SlotAssignment(**item) for item in data]
    except Exception:
        return []


def _compute_tom_bonus_aman(aman_action, dman_slots, task) -> float:
    """Theory-of-mind bonus: AMAN pre-emptively left gap for DMAN emergency."""
    if not dman_slots:
        return 0.0
    flights_by_id = {f.flight_id: f for f in task.flights}
    emg_deps = [
        s for s in dman_slots
        if flights_by_id.get(s.flight_id) and flights_by_id[s.flight_id].priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
    ]
    if not emg_deps:
        return 0.0
    aman_runway_times = [(s.runway, s.assigned_minute) for s in aman_action.arrival_slots]
    for dep in emg_deps:
        gap_ok = all(
            abs(m - dep.assigned_minute) >= 3
            for rwy, m in aman_runway_times
            if rwy == dep.runway
        )
        if gap_ok:
            return 1.0
    return 0.1


def _compute_tom_bonus_dman(dman_action, aman_slots, task) -> float:
    """Theory-of-mind bonus: DMAN cleared gap for AMAN emergency arrival."""
    if not aman_slots:
        return 0.0
    flights_by_id = {f.flight_id: f for f in task.flights}
    emg_arrs = [
        s for s in aman_slots
        if flights_by_id.get(s.flight_id) and flights_by_id[s.flight_id].priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
    ]
    if not emg_arrs:
        return 0.0
    dman_runway_times = [(s.runway, s.assigned_minute) for s in dman_action.departure_slots]
    for arr in emg_arrs:
        gap_ok = all(
            abs(m - arr.assigned_minute) >= 3
            for rwy, m in dman_runway_times
            if rwy == arr.runway
        )
        if gap_ok:
            return 1.0
    return 0.1


def _supervisor_alignment(outcome, task, profile: SupervisorProfileName) -> float:
    return _SUPERVISOR.score_plan(outcome, task, profile)


def _extract_supervisor_score(completion: Any) -> Optional[float]:
    """Parse supervisor's claimed score from JSON-like completion content."""
    text = _coerce_completion_text(completion)
    try:
        data = json.loads(text)
        return float(data.get("score", None))
    except Exception:
        match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
        if match:
            return float(match.group(1))
    return None
