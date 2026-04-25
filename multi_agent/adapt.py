"""ADAPT — Adaptive Decision Agent for Problem Transfer.

ADAPT receives scheduling problems from UNKNOWN domains. It has no prior
knowledge of any domain. It analyses each task's structural signals:

  - Time window tightness  (latest - earliest relative to planning horizon)
  - connection_risk         (probability of cascading failure if delayed)
  - fuel_burn_per_minute    (resource intensity proxy)
  - passengers              (impact weight proxy)
  - notes                   (free-text urgency signal extraction)

From these domain-agnostic signals it infers ATC parameter mappings
(wake_class and priority) that let AMAN/DMAN solve ANY scheduling problem
unchanged — without ever being told what domain it is.

Core functions:
  build_adapt_observation  — build ADAPTObservation with structural entity profiles
  _compute_entity_profiles — derive per-entity-type statistics from task data
  _build_adapt_heuristic   — structural heuristic fallback (no keyword matching)
  apply_adapt_mapping      — apply ADAPTAction → ATC-ready TaskDefinition
  parse_adapt_action       — parse LLM JSON completion into ADAPTAction
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    from ..models import FlightRecord, PriorityClass, TaskDefinition, WakeClass
    from .models import (
        ADAPTAction,
        ADAPTObservation,
        AgentRole,
        SupervisorProfileName,
        SUPERVISOR_PROFILES,
    )
except ImportError:
    from models import FlightRecord, PriorityClass, TaskDefinition, WakeClass
    from multi_agent.models import (
        ADAPTAction,
        ADAPTObservation,
        AgentRole,
        SupervisorProfileName,
        SUPERVISOR_PROFILES,
    )


# ── Urgency signal words (domain-agnostic) ────────────────────────────────────
# These do NOT encode domain knowledge — they are general English urgency markers
# that appear in notes regardless of domain.

_URGENCY_WORDS = frozenset({
    "emergency", "critical", "urgent", "immediate", "priority",
    "stat", "asap", "deadline", "cannot wait", "must vacate",
    "surge", "life-threatening", "requires immediate",
})


# ── Structural profile computation ────────────────────────────────────────────

def _compute_entity_profiles(task: TaskDefinition) -> Dict[str, Dict[str, Any]]:
    """Compute domain-agnostic structural signals per entity type.

    Groups flights by FlightRecord.airline (entity type tag) and computes
    statistics that are meaningful across any scheduling domain:

    - avg_window_minutes   : avg(latest - earliest) — small = high urgency
    - time_pressure        : 1 - avg_window / planning_horizon — higher = tighter
    - avg_connection_risk  : avg connection_risk — higher = higher-priority
    - avg_fuel_burn        : avg fuel_burn_per_minute — resource intensity proxy
    - avg_passengers       : avg passengers — impact weight proxy
    - urgency_in_notes     : any urgency-word hit in combined notes
    - operation_mix        : {"arrival": N, "departure": N}
    - count                : number of entities of this type
    """
    groups: Dict[str, List[FlightRecord]] = defaultdict(list)
    for f in task.flights:
        groups[f.airline or "UNKNOWN"].append(f)

    horizon = max(1, task.planning_horizon_minutes)
    profiles: Dict[str, Dict[str, Any]] = {}

    for entity_type, flights in groups.items():
        windows = [f.latest_minute - f.earliest_minute for f in flights]
        risks   = [f.connection_risk for f in flights]
        burns   = [f.fuel_burn_per_minute for f in flights]
        pax     = [f.passengers for f in flights]

        ops: Dict[str, int] = {"arrival": 0, "departure": 0}
        for f in flights:
            k = f.operation.value if hasattr(f.operation, "value") else str(f.operation)
            if k in ops:
                ops[k] += 1

        combined_notes = " ".join((f.notes or "").lower() for f in flights)
        urgency_hit = any(w in combined_notes for w in _URGENCY_WORDS)

        n = max(1, len(flights))
        avg_window = sum(windows) / n
        avg_risk   = sum(risks)   / n
        avg_burn   = sum(burns)   / n
        avg_pax    = sum(pax)     / n

        # Normalized time pressure: 0 = fully flexible, 1 = zero slack
        time_pressure = max(0.0, min(1.0, 1.0 - avg_window / horizon))

        profiles[entity_type] = {
            "count":               n,
            "avg_window_minutes":  round(avg_window, 1),
            "time_pressure":       round(time_pressure, 4),
            "avg_connection_risk": round(avg_risk, 4),
            "avg_fuel_burn":       round(avg_burn, 2),
            "avg_passengers":      round(avg_pax, 1),
            "urgency_in_notes":    urgency_hit,
            "operation_mix":       ops,
        }

    return profiles


# ── Priority / wake-class tier ordering ──────────────────────────────────────

_PRIORITY_TIERS = [
    PriorityClass.EMERGENCY.value,
    PriorityClass.MEDICAL.value,
    PriorityClass.CONNECTION.value,
    PriorityClass.NORMAL.value,
]
_WAKE_TIERS = [WakeClass.HEAVY.value, WakeClass.MEDIUM.value, WakeClass.LIGHT.value]


def _demote_priority(priority: str) -> str:
    """Return the next-lower priority tier (floor at normal)."""
    idx = _PRIORITY_TIERS.index(priority) if priority in _PRIORITY_TIERS else len(_PRIORITY_TIERS) - 1
    return _PRIORITY_TIERS[min(idx + 1, len(_PRIORITY_TIERS) - 1)]


def _demote_wake(wake: str) -> str:
    """Return the next-lower wake class (floor at light)."""
    idx = _WAKE_TIERS.index(wake) if wake in _WAKE_TIERS else len(_WAKE_TIERS) - 1
    return _WAKE_TIERS[min(idx + 1, len(_WAKE_TIERS) - 1)]


# ── Structural heuristic fallback ─────────────────────────────────────────────

def _build_adapt_heuristic(obs: ADAPTObservation, task: TaskDefinition) -> ADAPTAction:
    """Structural heuristic with priority-budget enforcement.

    Step 1 — Raw scoring (per entity type, no domain knowledge):
      urgency_score = 0.50 × time_pressure + 0.40 × connection_risk + 0.10 × urgency_flag

      Wake class raw thresholds:   ≥ 0.70 → H  |  ≥ 0.35 → M  |  else → L
      Priority raw thresholds:
        cr ≥ 0.80  or  (tp ≥ 0.95 and urgency)  → emergency
        cr ≥ 0.50  or  tp ≥ 0.80                 → medical
        cr ≥ 0.20  or  tp ≥ 0.60                 → connection
        else                                      → normal

    Step 2 — Budget enforcement (prevents all-emergency starvation):
      Entities sorted by urgency_score desc; budgets applied top-down:
        max_emergency = 1               (exactly one entity type may be emergency)
        max_heavy     = max(1, N // 3)  (at most 1/3 of types as H)
        max_medical   = max(1, ceil(N / 3))  (after emergency slot is claimed)
      Entities that exceed a budget are demoted one tier automatically.

    This ensures AMAN/DMAN always see a realistic priority distribution and
    avoids the starvation failure mode where AMAN yields all capacity to
    emergencies and DMAN has no runway time left.
    """
    profiles: Dict[str, Dict[str, Any]] = obs.entity_profiles or _compute_entity_profiles(task)

    # ── Step 1: raw scores ────────────────────────────────────────────────────
    raw: List[tuple] = []   # (entity_type, urgency_score, raw_wake, raw_priority)

    for entity_type, p in profiles.items():
        tp      = p["time_pressure"]
        cr      = p["avg_connection_risk"]
        urgency = bool(p["urgency_in_notes"])

        urgency_score = 0.50 * tp + 0.40 * cr + 0.10 * float(urgency)

        if urgency_score >= 0.70:
            raw_wake = WakeClass.HEAVY.value
        elif urgency_score >= 0.35:
            raw_wake = WakeClass.MEDIUM.value
        else:
            raw_wake = WakeClass.LIGHT.value

        if cr >= 0.80 or (tp >= 0.95 and urgency):
            raw_priority = PriorityClass.EMERGENCY.value
        elif cr >= 0.50 or tp >= 0.80:
            raw_priority = PriorityClass.MEDICAL.value
        elif cr >= 0.20 or tp >= 0.60:
            raw_priority = PriorityClass.CONNECTION.value
        else:
            raw_priority = PriorityClass.NORMAL.value

        raw.append((entity_type, urgency_score, raw_wake, raw_priority, p))

    # ── Step 2: budget enforcement — sort by urgency desc ────────────────────
    n_types = max(1, len(raw))
    max_emergency = 1
    max_heavy     = max(1, n_types // 3)
    max_medical   = max(1, -(-n_types // 3))   # ceil(N / 3)

    raw.sort(key=lambda x: x[1], reverse=True)

    entity_wake_map: Dict[str, str] = {}
    entity_priority_map: Dict[str, str] = {}
    rationale_parts: List[str] = []

    emergency_used = 0
    medical_used   = 0
    heavy_used     = 0

    for entity_type, urgency_score, raw_wake, raw_priority, p in raw:
        # ── wake budget ───────────────────────────────────────────────────────
        wake = raw_wake
        if wake == WakeClass.HEAVY.value:
            if heavy_used < max_heavy:
                heavy_used += 1
            else:
                wake = _demote_wake(wake)

        # ── priority budget ───────────────────────────────────────────────────
        priority = raw_priority

        if priority == PriorityClass.EMERGENCY.value:
            if emergency_used < max_emergency:
                emergency_used += 1
            else:
                priority = _demote_priority(priority)   # → medical

        if priority == PriorityClass.MEDICAL.value:
            if medical_used < max_medical:
                medical_used += 1
            else:
                priority = _demote_priority(priority)   # → connection

        entity_wake_map[entity_type]     = wake
        entity_priority_map[entity_type] = priority

        demoted_wake = " (budget-demoted)" if wake != raw_wake else ""
        demoted_pri  = " (budget-demoted)" if priority != raw_priority else ""
        rationale_parts.append(
            f"{entity_type}: tp={p['time_pressure']:.2f} cr={p['avg_connection_risk']:.2f} "
            f"score={urgency_score:.2f} "
            f"→ wake={wake}{demoted_wake} priority={priority}{demoted_pri}"
        )

    return ADAPTAction(
        entity_wake_map=entity_wake_map,
        entity_priority_map=entity_priority_map,
        rationale=(
            f"Structural budget-enforced inference "
            f"(budget: emergency≤{max_emergency} heavy≤{max_heavy} medical≤{max_medical}): "
            + "; ".join(rationale_parts)
        ),
    )


# ── Observation builder ───────────────────────────────────────────────────────

def build_adapt_observation(
    task: TaskDefinition,
    profile: SupervisorProfileName,
    domain_name: Optional[str] = None,
    domain_description: Optional[str] = None,
) -> ADAPTObservation:
    """Build ADAPTObservation from any TaskDefinition.

    domain_name and domain_description default to the task's own fields,
    so ADAPT works on ATC tasks, ICU tasks, or any future domain without
    caller changes.
    """
    domain_name        = domain_name        or task.airport
    domain_description = domain_description or task.description or ""

    entity_types: List[str]  = sorted({f.airline for f in task.flights if f.airline})
    resource_names: List[str] = [r.runway_id for r in task.runways]

    # Structural profile computation — domain agnostic
    entity_profiles = _compute_entity_profiles(task)

    has_emergencies = any(
        f.priority == PriorityClass.EMERGENCY
        or any(w in (f.notes or "").lower() for w in ("emergency", "critical", "life-threatening"))
        for f in task.flights
    )
    has_hard_deadlines = any(
        any(w in (f.notes or "").upper() for w in ("DEADLINE", "AUTH", "MUST VACATE", "ATFM"))
        for f in task.flights
    )

    # Legacy sample_entities string (still rendered in to_prompt_text via entity_profiles)
    sample_lines: List[str] = []
    for f in task.flights[:5]:
        sample_lines.append(
            f"  [{f.airline}] {f.flight_id}: {f.operation.value} "
            f"window=[{f.earliest_minute},{f.latest_minute}] "
            f"pax={f.passengers} fuel={f.fuel_burn_per_minute}/min"
            f" risk={f.connection_risk:.2f}"
        )
    sample_entities = "\n".join(sample_lines) if sample_lines else "(none)"

    sup_desc = SUPERVISOR_PROFILES[profile]["description"]

    return ADAPTObservation(
        role=AgentRole.ADAPT,
        domain_id=task.task_id,
        domain_name=domain_name,
        domain_description=domain_description,
        entity_types=entity_types,
        resource_names=resource_names,
        sample_entities=sample_entities,
        entity_profiles=entity_profiles,
        supervisor_profile_name=profile,
        supervisor_description=sup_desc,
        has_emergencies=has_emergencies,
        has_hard_deadlines=has_hard_deadlines,
        entity_count=len(task.flights),
    )


# ── Mapping application ───────────────────────────────────────────────────────

def apply_adapt_mapping(task: TaskDefinition, action: ADAPTAction) -> TaskDefinition:
    """Apply ADAPTAction to task — override wake_class and priority per entity type.

    Entity type is stored in FlightRecord.airline (no schema changes needed).
    Unknown entity types are passed through unchanged.
    Invalid enum values in the action are silently skipped.
    """
    adapted_flights: List[FlightRecord] = []
    for flight in task.flights:
        entity_type = flight.airline or ""
        updates: Dict[str, Any] = {}

        if entity_type in action.entity_wake_map:
            try:
                updates["wake_class"] = WakeClass(action.entity_wake_map[entity_type])
            except ValueError:
                pass

        if entity_type in action.entity_priority_map:
            try:
                updates["priority"] = PriorityClass(action.entity_priority_map[entity_type])
            except ValueError:
                pass

        adapted_flights.append(
            flight.model_copy(update=updates) if updates else flight
        )

    return task.model_copy(update={"flights": adapted_flights})


# ── Completion parser ─────────────────────────────────────────────────────────

def _coerce_text(completion: Any) -> str:
    if completion is None:
        return ""
    if isinstance(completion, bytes):
        return completion.decode("utf-8", errors="ignore")
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        for key in ("content", "text", "completion", "generated_text"):
            if key in completion:
                return _coerce_text(completion[key])
        try:
            return json.dumps(completion)
        except Exception:
            return str(completion)
    if isinstance(completion, list):
        parts = [_coerce_text(item) for item in completion]
        return "\n".join(p for p in parts if p)
    return str(completion)


def parse_adapt_action(completion: Any) -> Optional[ADAPTAction]:
    """Parse LLM JSON completion into ADAPTAction. Returns None on failure."""
    text = _coerce_text(completion)
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return ADAPTAction(
            entity_wake_map=data.get("entity_wake_map", {}),
            entity_priority_map=data.get("entity_priority_map", {}),
            rationale=data.get("rationale", ""),
        )
    except Exception:
        return None
