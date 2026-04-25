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
_ROLE_TOKEN_BUDGETS = {
    "AMAN": int(os.getenv("ATC_ROLE_TOKENS_AMAN", "384")),
    "DMAN": int(os.getenv("ATC_ROLE_TOKENS_DMAN", "384")),
    "GENERATOR": int(os.getenv("ATC_ROLE_TOKENS_GENERATOR", "224")),
    "SUPERVISOR": int(os.getenv("ATC_ROLE_TOKENS_SUPERVISOR", "160")),
}


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


def _weighted_active_mean(parts: List[tuple[float, float, bool]]) -> float:
    num = 0.0
    den = 0.0
    for value, weight, active in parts:
        if not active:
            continue
        num += float(value) * float(weight)
        den += float(weight)
    return num / den if den > 1e-9 else 0.0


def _approx_token_count(text: str) -> int:
    # Simple, fast approximation suitable for relative penalties.
    return max(0, len(text.split()))


def _length_budget_penalty(role: str, completion: Any) -> float:
    text = _coerce_completion_text(completion)
    budget = max(32, _ROLE_TOKEN_BUDGETS.get(role, 256))
    n = _approx_token_count(text)
    if n <= budget:
        return 0.0
    overflow = n - budget
    return min(0.25, overflow / max(1.0, float(budget)) * 0.25)


def _controller_failure_reward(completion: Any, role: str) -> float:
    """Spread parse-failure rewards to avoid flat-floor collapse in GRPO."""
    text = _coerce_completion_text(completion)
    base_credit = _parse_partial_credit(completion)  # [0, 0.18]
    lower = -0.95
    spread = 0.60
    reward = lower + spread * base_credit
    # Deterministic micro-jitter keeps rewards non-identical within bad batches.
    if text:
        bucket = sum(ord(c) for c in text[:240]) % 41
        jitter = (bucket / 40.0 - 0.5) * 0.08
        reward += jitter
    reward -= _length_budget_penalty(role, text)
    return round(max(-1.0, min(-0.2, reward)), 4)


def _slot_structural_quality(
    task: TaskDefinition,
    slots: List[SlotAssignment],
    *,
    operation: OperationType,
) -> float:
    """Role-local quality signal to reduce reward collapse in near-tie plans."""
    expected = [f for f in task.flights if f.operation == operation]
    if not expected:
        return 0.5
    by_id = {s.flight_id: s for s in slots}
    assigned = 0
    runway_ok = 0
    window_ok = 0
    for f in expected:
        s = by_id.get(f.flight_id)
        if s is None:
            continue
        assigned += 1
        if s.runway in f.allowed_runways:
            runway_ok += 1
        if f.earliest_minute <= s.assigned_minute <= f.latest_minute:
            window_ok += 1
    assign_ratio = assigned / max(1, len(expected))
    runway_ratio = runway_ok / max(1, assigned)
    window_ratio = window_ok / max(1, assigned)
    return max(0.0, min(1.0, 0.45 * assign_ratio + 0.30 * runway_ratio + 0.25 * window_ratio))


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
            rewards.append(_controller_failure_reward(completion, "AMAN"))
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
        emg_total = sum(
            1 for f in arrivals if f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
        )
        budget = task.delay_budget / 2.0
        # If all flights are unscheduled, delay_total=0 which would give delay_eff=1.0
        # (rewarding an empty plan). Clamp to 0 when nothing is scheduled.
        _all_arr_missing = missing == len(arrivals) and len(arrivals) > 0
        delay_eff = 0.0 if _all_arr_missing else max(0.0, 1.0 - delay_total / max(1, budget))
        coverage = 1.0 - missing / arr_count
        emg_score = emg_ok / max(1, emg_ok + emg_miss) if emg_total > 0 else 0.5

        cross_penalty = 0.0
        if {s.runway for s in aman_action.arrival_slots} & {s.runway for s in dman_slots}:
            cross_penalty = 1.55 * _normalized_cross_conflict_penalty(
                aman_action.arrival_slots,
                dman_slots,
                outcome.metrics.conflict_count,
            )

        tom_bonus = _compute_tom_bonus_aman(aman_action, dman_slots, task)
        sup_align = _supervisor_alignment(outcome, task, profile)
        rationale_score = _score_rationale_quality(aman_action.rationale, task, outcome)
        slot_quality = _slot_structural_quality(
            task,
            aman_action.arrival_slots,
            operation=OperationType.ARRIVAL,
        )

        # Counterfactual credit: how much does AMAN's plan improve over naive baseline?
        # cf_outcome = naive arrivals + real DMAN slots; advantage = real - counterfactual.
        cf_outcome = simulate_plan(task, _naive_arrival_slots(task) + dman_slots)
        cf_advantage = max(-1.0, min(1.0, outcome.normalized_score - cf_outcome.normalized_score))

        json_fmt = _json_format_score(completion)
        long_horizon = _weighted_active_mean(
            [
                (outcome.metrics.connection_impact_score, 0.45, True),
                (outcome.metrics.fairness, 0.25, True),
                (outcome.metrics.fuel_efficiency, 0.20, True),
                (outcome.metrics.schedule_completeness, 0.10, True),
            ]
        )
        reward = _weighted_active_mean(
            [
                (delay_eff, 0.22, True),
                (coverage, 0.18, True),
                (cf_advantage, 0.15, True),
                (tom_bonus, 0.10, True),
                (emg_score, 0.14, emg_total > 0),
                (sup_align, 0.06, True),
                (rationale_score, 0.05, True),
                (json_fmt, 0.04, True),
                (long_horizon, 0.06, True),
                (slot_quality, 0.06, True),
            ]
        ) - cross_penalty - _length_budget_penalty("AMAN", completion)

        if outcome.metrics.conflict_count == 0 and coverage >= 0.95 and missing == 0:
            reward = min(1.0, reward + 0.12)

        # Layered safety gates — hard ceilings that cannot be bought off by efficiency
        if outcome.metrics.conflict_count > 0:
            reward = min(reward, 0.18)   # conflict-free gate (stronger penalty)
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
                "slot_quality": slot_quality,
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
            rewards.append(_controller_failure_reward(completion, "DMAN"))
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
        emg_total = sum(
            1 for f in departures if f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
        )
        budget = task.delay_budget / 2.0
        _all_dep_missing = missing == len(departures) and len(departures) > 0
        delay_eff = 0.0 if _all_dep_missing else max(0.0, 1.0 - delay_total / max(1, budget))
        coverage = 1.0 - missing / dep_count
        emg_score = emg_ok / max(1, emg_ok + emg_miss) if emg_total > 0 else 0.5
        atfm_count = atfm_ok + atfm_viol
        atfm_score = atfm_ok / max(1, atfm_count) if atfm_count > 0 else 0.5

        cross_penalty = 0.0
        if {s.runway for s in dman_action.departure_slots} & {s.runway for s in aman_slots}:
            cross_penalty = 1.55 * _normalized_cross_conflict_penalty(
                dman_action.departure_slots,
                aman_slots,
                outcome.metrics.conflict_count,
            )

        tom_bonus = _compute_tom_bonus_dman(dman_action, aman_slots, task)
        sup_align = _supervisor_alignment(outcome, task, profile)
        rationale_score = _score_rationale_quality(dman_action.rationale, task, outcome)
        slot_quality = _slot_structural_quality(
            task,
            dman_action.departure_slots,
            operation=OperationType.DEPARTURE,
        )

        # Counterfactual credit: how much does DMAN's plan improve over naive baseline?
        # cf_outcome = real AMAN slots + naive departures; advantage = real - counterfactual.
        cf_outcome = simulate_plan(task, aman_slots + _naive_departure_slots(task))
        cf_advantage = max(-1.0, min(1.0, outcome.normalized_score - cf_outcome.normalized_score))

        json_fmt = _json_format_score(completion)
        long_horizon = _weighted_active_mean(
            [
                (outcome.metrics.connection_impact_score, 0.45, True),
                (outcome.metrics.fairness, 0.25, True),
                (outcome.metrics.fuel_efficiency, 0.20, True),
                (outcome.metrics.schedule_completeness, 0.10, True),
            ]
        )
        reward = _weighted_active_mean(
            [
                (delay_eff, 0.19, True),
                (coverage, 0.14, True),
                (cf_advantage, 0.14, True),
                (tom_bonus, 0.10, True),
                (atfm_score, 0.14, atfm_count > 0),
                (emg_score, 0.12, emg_total > 0),
                (sup_align, 0.05, True),
                (rationale_score, 0.04, True),
                (json_fmt, 0.03, True),
                (long_horizon, 0.05, True),
                (slot_quality, 0.07, True),
            ]
        ) - cross_penalty - _length_budget_penalty("DMAN", completion)

        if outcome.metrics.conflict_count == 0 and coverage >= 0.95 and missing == 0:
            reward = min(1.0, reward + 0.12)

        # Layered safety gates
        if outcome.metrics.conflict_count > 0:
            reward = min(reward, 0.18)   # conflict-free gate (stronger penalty)
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
                "slot_quality": slot_quality,
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

        try:
            _, is_solvable = _GENERATOR.mutate(task, gen_action)
            reward = _GENERATOR.compute_reward(_safe_float(ctrl_score), is_solvable)
        except Exception as exc:
            # LLM may emit invalid param types (e.g. flight_id as int).
            # Penalise but don't crash training — the source bug is also fixed
            # in generator.py, but this guard handles any future schema drift.
            if _TRACE_REWARDS:
                print(f"[REWARD TRACE] generator mutate failed: {exc}")
            reward = -0.4 + _parse_partial_credit(completion)
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
        stripped = re.sub(r"```[a-zA-Z]*\s*", "", text)
        stripped = re.sub(r"```", "", stripped).strip()
        # Normalise Python literals before parsing
        stripped = re.sub(r"\bTrue\b", "true", stripped)
        stripped = re.sub(r"\bFalse\b", "false", stripped)
        stripped = re.sub(r"\bNone\b", "null", stripped)
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


def _parse_partial_credit(completion: Any) -> float:
    """Partial credit for completions that fail JSON parsing.

    GRPO advantage = (r - mean) / std. If ALL completions in a group get
    identical rewards, std=0 → advantage=0 → zero gradient. This function
    ensures rewards in [-0.80, -0.62] vary even when parsing fails completely,
    including when the model outputs natural language instead of JSON.

    Credit tiers (cumulative, capped at 0.18):
      +0.03  any substantive text (> 20 chars)
      +0.02  longer response (> 100 chars, shows more effort)
      +0.04  has { (attempted JSON)
      +0.05  has slot key (arrival_slots / departure_slots / slots)
      +0.04  has flight_id key or pattern
      +0.03  has runway key or pattern
      +0.02  has rationale key
      -0.04  very short (< 20 chars, penalise lazy/empty outputs)
    """
    text = _coerce_completion_text(completion)
    if not text or not text.strip():
        return 0.0

    tl = text.lower()
    score = 0.0

    # Natural-language responses still get some credit so they vary from empty
    if len(text.strip()) > 20:
        score += 0.03
    if len(text.strip()) > 100:
        score += 0.02

    # JSON structure markers
    if "{" in text:
        score += 0.04
    if any(k in tl for k in ("arrival_slots", "departure_slots", '"slots"', "'slots'")):
        score += 0.05
    if "flight_id" in tl:
        score += 0.04
    if "runway" in tl:
        score += 0.03
    if "rationale" in tl:
        score += 0.02

    if len(text.strip()) < 20:
        score = max(0.0, score - 0.04)

    return round(min(0.18, score), 4)


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


def adapt_reward_fn(completions: List[Any], **kwargs) -> List[float]:
    """Reward ADAPT meta-agent for quality of structural domain transfer."""
    from multi_agent.adapt import (
        _build_adapt_heuristic,
        apply_adapt_mapping,
        build_adapt_observation,
        parse_adapt_action,
    )
    from multi_agent.inference import _build_aman_heuristic, _build_dman_heuristic

    rewards: List[float] = []
    n = len(completions)

    domain_task_jsons = kwargs.get("domain_task_json", [None] * n)
    supervisor_profiles = kwargs.get("supervisor_profile", [None] * n)

    if not isinstance(domain_task_jsons, list):
        domain_task_jsons = [domain_task_jsons] * n
    if not isinstance(supervisor_profiles, list):
        supervisor_profiles = [supervisor_profiles] * n

    for i, completion in enumerate(completions):
        dtjson = domain_task_jsons[i] if i < len(domain_task_jsons) else None
        profile = supervisor_profiles[i] if i < len(supervisor_profiles) else None

        if not dtjson:
            rewards.append(-0.60)
            continue

        try:
            domain_task = TaskDefinition.model_validate_json(dtjson)
        except Exception:
            rewards.append(-0.60)
            continue

        action = parse_adapt_action(completion)
        if action is None:
            rewards.append(-0.50)
            continue

        try:
            prof_enum = SupervisorProfileName(profile) if profile else SupervisorProfileName.SAFETY_STRICT
        except ValueError:
            prof_enum = SupervisorProfileName.SAFETY_STRICT

        try:
            adapt_obs = build_adapt_observation(task=domain_task, profile=prof_enum)
        except Exception:
            rewards.append(-0.40)
            continue

        # Baseline with heuristic ADAPT mapping
        try:
            heuristic_action = _build_adapt_heuristic(adapt_obs, domain_task)
            heuristic_mapped = apply_adapt_mapping(domain_task, heuristic_action)
            env_base = MultiAgentATCEnvironment(seed=0)
            aman_obs_b, dman_obs_b = env_base.reset(0, prof_enum, heuristic_mapped)
            aman_act_b = _build_aman_heuristic(aman_obs_b)
            dman_act_b = _build_dman_heuristic(dman_obs_b, env_base._state.atfm_deadlines)
            env_base.step_bid(aman_act_b, dman_act_b)
            baseline_composite = env_base.finalize().composite_score
        except Exception:
            baseline_composite = 0.40

        # LLM ADAPT mapping
        try:
            llm_mapped = apply_adapt_mapping(domain_task, action)
            env_llm = MultiAgentATCEnvironment(seed=0)
            aman_obs_l, dman_obs_l = env_llm.reset(0, prof_enum, llm_mapped)
            aman_act_l = _build_aman_heuristic(aman_obs_l)
            dman_act_l = _build_dman_heuristic(dman_obs_l, env_llm._state.atfm_deadlines)
            env_llm.step_bid(aman_act_l, dman_act_l)
            downstream = env_llm.finalize().composite_score
        except Exception:
            downstream = 0.0

        entity_types = set(adapt_obs.entity_types)
        mapped_types = set(action.entity_wake_map.keys()) | set(action.entity_priority_map.keys())
        coverage = len(entity_types & mapped_types) / max(1, len(entity_types))

        rationale = action.rationale or ""
        has_numbers = bool(re.search(r"\d+\.\d+", rationale))
        rationale_bonus = 0.05 if (len(rationale) >= 30 and has_numbers) else 0.0

        improvement = downstream - baseline_composite
        reward = (
            0.70 * downstream
            + 0.15 * max(-1.0, min(1.0, improvement))
            + 0.10 * coverage
            + rationale_bonus
        )
        reward = max(-1.0, min(1.0, reward))
        _debug_reward_trace(
            role="ADAPT",
            components={
                "downstream_composite": downstream,
                "baseline_composite": baseline_composite,
                "improvement": improvement,
                "coverage": coverage,
                "rationale_bonus": rationale_bonus,
                "total_reward": reward,
            },
        )
        rewards.append(reward)

    return rewards
