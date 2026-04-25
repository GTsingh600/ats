"""Grounded curriculum: rule-based solver, Level-0 gate, bands, adaptive state, logging.

Used by ``build_episode_dataset(use_grounded_curriculum=True)`` and optional
warm-start for task sampling across training runs.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from engine import simulate_plan
from models import FlightRecord, OperationType, RunwaySpec, SlotAssignment, TaskDefinition
from tasks_grounded import GROUNDED_CURRICULUM_TASKS, GROUNDED_LEVEL_BY_TASK_ID

from constants import SEPARATION_BY_WAKE


def _capacity_spacing_minutes(runway: RunwaySpec) -> int:
    base_gap = max(2, round(60 / max(1, runway.hourly_capacity)))
    return max(2, round(base_gap * runway.weather_penalty))


def _required_gap(prev: FlightRecord, curr: FlightRecord, runway: RunwaySpec) -> int:
    wkey = (prev.wake_class.value, curr.wake_class.value)
    wake_gap = SEPARATION_BY_WAKE.get(wkey, 3)
    return max(_capacity_spacing_minutes(runway), wake_gap)


def _assignments_for_runway_order(
    flights: List[FlightRecord],
    runway: RunwaySpec,
    order: Tuple[int, ...],
) -> Optional[List[SlotAssignment]]:
    """Greedy earliest-feasible times for a fixed permutation (indices into flights)."""
    seq = [flights[i] for i in order]
    times: List[int] = []
    t = 0
    for idx, flight in enumerate(seq):
        lo = int(flight.earliest_minute)
        hi = int(flight.latest_minute)
        if idx == 0:
            cand = max(lo, min(hi, int(flight.scheduled_minute)))
        else:
            prev = seq[idx - 1]
            need = t + _required_gap(prev, flight, runway)
            cand = max(lo, need)
        if cand > hi:
            return None
        times.append(cand)
        t = cand
    rw = runway.runway_id
    return [
        SlotAssignment(
            flight_id=seq[i].flight_id,
            runway=rw,
            assigned_minute=times[i],
            hold_minutes=max(0, abs(times[i] - int(seq[i].scheduled_minute))),
        )
        for i in range(len(seq))
    ]


def _solve_single_runway(
    runway: RunwaySpec,
    flights: List[FlightRecord],
) -> Optional[List[SlotAssignment]]:
    if not flights:
        return []
    n = len(flights)
    if n > 6:
        return None
    best: Optional[List[SlotAssignment]] = None
    best_delay = 10**9
    for order in permutations(range(n)):
        slots = _assignments_for_runway_order(flights, runway, order)
        if slots is None:
            continue
        delay = sum(s.hold_minutes for s in slots)
        if delay < best_delay:
            best_delay = delay
            best = slots
    return best


def solve_grounded_rule_based(task: TaskDefinition) -> Optional[List[SlotAssignment]]:
    """Feasible full-plan search: independent mixed runways; one permutation search per runway."""
    runways_by_id = {r.runway_id: r for r in task.runways}
    by_runway: Dict[str, List[FlightRecord]] = {rid: [] for rid in runways_by_id}
    for f in task.flights:
        if not f.allowed_runways:
            return None
        rid = f.allowed_runways[0]
        if len(f.allowed_runways) > 1:
            return None
        if rid not in by_runway:
            return None
        by_runway[rid].append(f)

    all_slots: List[SlotAssignment] = []
    for rid, rw in runways_by_id.items():
        flist = by_runway.get(rid, [])
        part = _solve_single_runway(rw, flist)
        if part is None and flist:
            return None
        if part:
            all_slots.extend(part)
    if len(all_slots) != len(task.flights):
        return None
    return all_slots


def rule_based_plan_succeeds(task: TaskDefinition, min_normalized_score: float = 0.5) -> bool:
    slots = solve_grounded_rule_based(task)
    if not slots:
        return False
    out = simulate_plan(task, slots)
    return (
        out.metrics.conflict_count == 0
        and out.metrics.missing_assignments == 0
        and out.metrics.invalid_assignments == 0
        and float(out.normalized_score) >= min_normalized_score
    )


def validate_level0_gate(min_fraction: float = 0.8) -> Tuple[bool, List[str]]:
    """Level-0 tasks must pass the rule solver (≥80% of L0 definitions = all must pass if one L0)."""
    l0_tasks = [t for t in GROUNDED_CURRICULUM_TASKS if GROUNDED_LEVEL_BY_TASK_ID.get(t.task_id, -1) == 0]
    if not l0_tasks:
        return False, ["no_level_0_tasks"]
    ok = [t.task_id for t in l0_tasks if rule_based_plan_succeeds(t)]
    frac = len(ok) / len(l0_tasks)
    if frac >= min_fraction:
        return True, ok
    bad = [t.task_id for t in l0_tasks if t.task_id not in ok]
    return False, bad


def training_band_from_success_rate(rate: float) -> str:
    if rate >= 0.70:
        return "calibration"
    if rate >= 0.25:
        return "learning"
    return "challenge"


def band_from_proxy_score(normalized_score: float) -> str:
    """Map engine score to band labels when learner stats are unavailable (dataset metadata)."""
    if normalized_score >= 0.70:
        return "calibration"
    if normalized_score >= 0.25:
        return "learning"
    return "challenge"


@dataclass
class GroundedCurriculumState:
    """Rolling learner performance + discrete level for promote/regress."""

    active_level: int = 0
    window: int = 32
    promote_threshold: float = 0.72
    regress_threshold: float = 0.32
    success_by_level: Dict[int, Deque[float]] = field(default_factory=dict)
    success_by_task_id: Dict[str, Deque[float]] = field(default_factory=dict)
    failure_modes: Dict[str, int] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.active_level = max(0, min(self.active_level, len(GROUNDED_CURRICULUM_TASKS) - 1))

    def record_episode(
        self,
        task_id: str,
        success: float,
        failure_mode: str = "",
    ) -> None:
        level = int(GROUNDED_LEVEL_BY_TASK_ID.get(task_id, 0))
        if level not in self.success_by_level:
            self.success_by_level[level] = deque(maxlen=self.window)
        self.success_by_level[level].append(success)
        if task_id not in self.success_by_task_id:
            self.success_by_task_id[task_id] = deque(maxlen=self.window)
        self.success_by_task_id[task_id].append(success)
        if failure_mode:
            self.failure_modes[failure_mode] = self.failure_modes.get(failure_mode, 0) + 1
        self._maybe_adjust_level()

    def _level_mean(self, level: int) -> Optional[float]:
        q = self.success_by_level.get(level)
        if not q or len(q) < 4:
            return None
        return sum(q) / len(q)

    def _maybe_adjust_level(self) -> None:
        m = self._level_mean(self.active_level)
        if m is None:
            return
        old = self.active_level
        if m >= self.promote_threshold and self.active_level < len(GROUNDED_CURRICULUM_TASKS) - 1:
            self.active_level += 1
            self.events.append({"type": "promote", "from": old, "to": self.active_level, "mean_success": m})
        elif m <= self.regress_threshold and self.active_level > 0:
            self.active_level -= 1
            self.events.append({"type": "regress", "from": old, "to": self.active_level, "mean_success": m})

    def band_success_rates(self) -> Dict[str, float]:
        """Aggregate recent successes by calibration / learning / challenge (from per-sample success bits)."""
        # Approximate: map each level's mean to its typical band (levels 0-1 cal, 2-3 learn, 4-5 challenge).
        bands = {"calibration": [], "learning": [], "challenge": []}
        for lev, q in self.success_by_level.items():
            if not q:
                continue
            rate = sum(q) / len(q)
            if lev <= 1:
                bands["calibration"].append(rate)
            elif lev <= 3:
                bands["learning"].append(rate)
            else:
                bands["challenge"].append(rate)
        return {
            k: (sum(v) / len(v) if v else 0.0) for k, v in bands.items()
        }

    def to_json(self) -> str:
        payload = {
            "active_level": self.active_level,
            "success_by_level": {str(k): list(v) for k, v in self.success_by_level.items()},
            "success_by_task_id": {k: list(v) for k, v in self.success_by_task_id.items()},
            "failure_modes": dict(self.failure_modes),
            "events": self.events[-200:],
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json_file(cls, path: Path) -> "GroundedCurriculumState":
        data = json.loads(path.read_text())
        st = cls(active_level=int(data.get("active_level", 0)))
        for k, vals in data.get("success_by_level", {}).items():
            st.success_by_level[int(k)] = deque(vals, maxlen=st.window)
        for tid, vals in data.get("success_by_task_id", {}).items():
            st.success_by_task_id[tid] = deque(vals, maxlen=st.window)
        st.failure_modes = dict(data.get("failure_modes", {}))
        st.events = list(data.get("events", []))
        return st


def pick_grounded_task(
    rng,
    episode_id: int,
    state: Optional[GroundedCurriculumState] = None,
) -> TaskDefinition:
    """Sample task: anti-forgetting revisit to lower levels; focus near active_level."""
    if state is None:
        state = GroundedCurriculumState()
    max_l = len(GROUNDED_CURRICULUM_TASKS) - 1
    active = max(0, min(state.active_level, max_l))
    # 25% uniform revisit any level <= active; else active level with small jitter.
    if rng.random() < 0.25:
        level = rng.randint(0, active)
    else:
        jitter = rng.choice([-1, 0, 0, 0, 1])
        level = max(0, min(active + jitter, max_l))
    return GROUNDED_CURRICULUM_TASKS[level]


def append_curriculum_log(output_dir: Path, row: Dict[str, Any]) -> None:
    path = output_dir / "grounded_curriculum_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def grounded_task_proxy_score(task: TaskDefinition) -> float:
    slots = solve_grounded_rule_based(task)
    if not slots:
        return 0.0
    return float(simulate_plan(task, slots).normalized_score)


# Module import: reject broken Level-0 definitions early.
_L0_OK, _L0_DETAIL = validate_level0_gate(0.8)
if not _L0_OK:
    import warnings

    warnings.warn(
        f"Grounded Level-0 gate failed (need ≥80% rule-solver success): {_L0_DETAIL}",
        stacklevel=2,
    )
