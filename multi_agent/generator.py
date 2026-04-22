"""Adversarial self-play scenario generator.

The generator mutates existing TaskDefinitions to create harder coordination
challenges. It is rewarded when AMAN + DMAN fail to coordinate (zero-sum),
driving recursive skill amplification.

Adaptive curriculum:
  - Tracks rolling agent performance (EMA over last 10 episodes)
  - Escalates mutation intensity when agents score > ESCALATION_THRESHOLD
  - Eases back when agents score < FLOOR_THRESHOLD (prevents unsolvable traps)
  - Solvability guard: mutations are validated; impossible tasks penalised

Mutation catalogue:
  TIGHTEN_WINDOW          → squeeze flight's [earliest, latest] by N minutes each side
  INJECT_EMERGENCY        → insert a new EMERGENCY/MEDICAL flight into the scenario
  INCREASE_WEATHER_PENALTY → degrade runway capacity (forces AMAN/DMAN onto fewer slots)
  ADD_ATFM_DEADLINE       → add hard network slot constraint to a departure
  CLOSE_RUNWAY_WINDOW     → runway unavailable for T minutes around peak hour
  ADD_CONFLICTING_FLIGHT  → inject Heavy arrival just before a Light departure (wake trap)
"""

from __future__ import annotations

import copy
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

try:
    from ..engine import simulate_plan
    from ..models import (
        FlightRecord,
        OperationType,
        PriorityClass,
        RunwaySpec,
        SlotAssignment,
        TaskDefinition,
        WakeClass,
    )
    from .models import GeneratorAction, GeneratorMutation, MutationType
except ImportError:
    from engine import simulate_plan
    from models import (
        FlightRecord,
        OperationType,
        PriorityClass,
        RunwaySpec,
        SlotAssignment,
        TaskDefinition,
        WakeClass,
    )
    from multi_agent.models import GeneratorAction, GeneratorMutation, MutationType


ESCALATION_THRESHOLD = 0.65   # escalate when agents consistently above this
FLOOR_THRESHOLD      = 0.30   # ease back when agents consistently below this
EMA_ALPHA            = 0.2    # exponential moving average smoothing factor
MAX_MUTATIONS_PER_EPISODE = 3 # cap mutations to keep scenarios solvable
MIN_WINDOW_WIDTH     = 8      # never squeeze window below 8 minutes

# LLM outputs non-canonical priority strings; map them to valid PriorityClass values
_PRIORITY_ALIASES: Dict[str, str] = {
    "high": "emergency",
    "urgent": "emergency",
    "critical": "emergency",
    "med": "medical",
    "low": "normal",
    "standard": "normal",
    "routine": "normal",
    "conn": "connection",
}
MASTERY_WINDOW       = 10     # rolling window for per-scenario success rate
MASTERY_THRESHOLD    = 0.55   # rate below this → mutation considered "weak"/underused


class ChallengeGenerator:
    """Adversarial curriculum generator for multi-agent ATC training.

    Maintains an EMA of recent controller scores and escalates mutation
    intensity accordingly — creating an auto-curriculum without manual tuning.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._ema_score: float = 0.5       # start at mid-difficulty assumption
        self._difficulty_level: int = 1    # 1=easy mutations → 6=max chaos
        self._score_history: Deque[float] = deque(maxlen=10)
        self._mutation_history: List[Dict] = []
        # Per-scenario mastery: track agent success rate per task_id and mutation_type
        self._task_mastery: Dict[str, Deque[float]] = {}
        self._mutation_mastery: Dict[str, Deque[float]] = {}
        # PAIRED: store baseline score for last mutated task to compute regret reward
        self._last_heuristic_score: float = 0.5
        self._last_mutated_task: Optional[TaskDefinition] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, controller_score: float) -> None:
        """Update EMA and adjust difficulty level after each episode."""
        self._score_history.append(controller_score)
        self._ema_score = EMA_ALPHA * controller_score + (1 - EMA_ALPHA) * self._ema_score

        if self._ema_score > ESCALATION_THRESHOLD and self._difficulty_level < 6:
            self._difficulty_level += 1
        elif self._ema_score < FLOOR_THRESHOLD and self._difficulty_level > 1:
            self._difficulty_level -= 1

    def record(
        self,
        task_id: str,
        mutations_used: Optional[List[str]],
        composite_score: float,
    ) -> None:
        """Record per-scenario mastery after an episode completes.

        Called by the training loop alongside update(). Maintains rolling success
        rate per task_id and per mutation_type so the generator can identify which
        mutations the agents have mastered and which still challenge them.
        """
        if task_id not in self._task_mastery:
            self._task_mastery[task_id] = deque(maxlen=MASTERY_WINDOW)
        self._task_mastery[task_id].append(composite_score)

        for mtype in (mutations_used or []):
            if mtype not in self._mutation_mastery:
                self._mutation_mastery[mtype] = deque(maxlen=MASTERY_WINDOW)
            self._mutation_mastery[mtype].append(composite_score)

    def get_weak_mutations(self, threshold: float = MASTERY_THRESHOLD) -> List[str]:
        """Return mutation types where agents still score below threshold on average.

        Used by _sample_mutations() to boost under-explored challenge types.
        """
        weak = []
        for mtype, scores in self._mutation_mastery.items():
            if len(scores) >= 3 and (sum(scores) / len(scores)) < threshold:
                weak.append(mtype)
        return weak

    def mastery_report(self) -> Dict:
        """Summary of per-task and per-mutation success rates for logging."""
        task_means = {
            tid: round(sum(v) / len(v), 3)
            for tid, v in self._task_mastery.items()
            if v
        }
        mut_means = {
            mtype: round(sum(v) / len(v), 3)
            for mtype, v in self._mutation_mastery.items()
            if v
        }
        return {
            "task_mastery": task_means,
            "mutation_mastery": mut_means,
            "weak_mutations": self.get_weak_mutations(),
            "difficulty_level": self._difficulty_level,
            "ema_score": self.ema_score,
        }

    def mutate(
        self,
        base_task: TaskDefinition,
        generator_action: Optional[GeneratorAction] = None,
    ) -> Tuple[TaskDefinition, bool]:
        """Apply mutations to base_task. Returns (mutated_task, is_solvable).

        If generator_action is provided (LLM-driven), apply those mutations.
        Otherwise fall back to rule-based mutations matching current difficulty.
        Also stores the scheduled-baseline score on the mutated task for PAIRED
        regret computation in compute_reward().
        """
        task = self._deep_copy_task(base_task)

        if generator_action and generator_action.mutations:
            mutations = generator_action.mutations[:MAX_MUTATIONS_PER_EPISODE]
        else:
            mutations = self._sample_mutations(task)

        for mut in mutations:
            task = self._apply_mutation(task, mut)
            self._mutation_history.append({
                "type": mut.mutation_type.value,
                "level": self._difficulty_level,
                "ema": round(self._ema_score, 3),
            })

        solvable = self._check_solvability(task)

        # PAIRED: cache heuristic baseline on mutated task so compute_reward()
        # can compute regret = controller_score - baseline_score.
        self._last_mutated_task = task
        self._last_heuristic_score = self._score_scheduled_baseline(task) if solvable else 0.0

        return task, solvable

    def compute_reward(self, controller_score: float, is_solvable: bool) -> float:
        """Generator reward: PAIRED regret-based with solvability guard.

        Unsolvable scenarios get penalised — generator must create HARD but
        not IMPOSSIBLE challenges (mirrors real ATC simulator design).

        When a baseline score is available (after mutate() was called this episode),
        uses PAIRED regret = controller_score − heuristic_baseline so the generator
        is rewarded for staying at the frontier of agent capability, not just for
        making tasks arbitrarily hard.
        """
        if not is_solvable:
            return -1.0

        if self._last_mutated_task is not None:
            # PAIRED regret: positive when agents beat baseline (bad for generator),
            # negative when agents fall behind baseline (generator rewarded).
            regret = controller_score - self._last_heuristic_score
            return round(max(-1.0, min(1.0, -regret)), 4)

        # Fallback if called without prior mutate() (e.g. unit tests)
        return round(max(-1.0, min(1.0, 1.0 - controller_score)), 4)

    @property
    def difficulty_level(self) -> int:
        return self._difficulty_level

    @property
    def ema_score(self) -> float:
        return round(self._ema_score, 3)

    # ── Mutation sampling ─────────────────────────────────────────────────────

    def _sample_mutations(self, task: TaskDefinition) -> List[GeneratorMutation]:
        """Rule-based mutation selection based on current difficulty level.

        Mutations whose type appears in get_weak_mutations() are boosted 3x in
        the sampling pool, steering the curriculum toward underexplored challenges.
        """
        mutations: List[GeneratorMutation] = []

        level = self._difficulty_level
        n_mutations = min(MAX_MUTATIONS_PER_EPISODE, max(1, level // 2))

        # Mutation pool weighted by difficulty
        pool: List[MutationType] = []
        if level >= 1:
            pool += [MutationType.TIGHTEN_WINDOW] * 3
        if level >= 2:
            pool += [MutationType.ADD_ATFM_DEADLINE] * 2
        if level >= 3:
            pool += [MutationType.INCREASE_WEATHER_PENALTY] * 2
        if level >= 4:
            pool += [MutationType.INJECT_EMERGENCY] * 2
        if level >= 5:
            pool += [MutationType.ADD_CONFLICTING_FLIGHT] * 2
        if level >= 6:
            pool += [MutationType.CLOSE_RUNWAY_WINDOW] * 1

        # Boost weak mutations: add 2 extra copies (3x total) for each type that
        # the agents haven't mastered yet, so the curriculum focuses there.
        weak = set(self.get_weak_mutations())
        pool += [mt for mt in pool if mt.value in weak]

        selected = self._rng.choices(pool, k=n_mutations)

        departures = [f for f in task.flights if f.operation == OperationType.DEPARTURE]
        arrivals   = [f for f in task.flights if f.operation == OperationType.ARRIVAL]

        for mtype in selected:
            if mtype == MutationType.TIGHTEN_WINDOW:
                target = self._rng.choice(task.flights)
                squeeze = self._rng.randint(2, 4 + level)
                mutations.append(GeneratorMutation(
                    mutation_type=mtype,
                    target_flight_id=target.flight_id,
                    params={"squeeze_minutes": squeeze},
                    rationale=f"Tighten {target.flight_id} window by {squeeze} min each side",
                ))

            elif mtype == MutationType.ADD_ATFM_DEADLINE and departures:
                target = self._rng.choice(departures)
                buffer = self._rng.randint(5, 10)
                mutations.append(GeneratorMutation(
                    mutation_type=mtype,
                    target_flight_id=target.flight_id,
                    params={"deadline_offset": buffer},
                    rationale=f"ATFM slot: {target.flight_id} must depart by scheduled+{buffer}",
                ))

            elif mtype == MutationType.INCREASE_WEATHER_PENALTY:
                target_rwy = self._rng.choice(task.runways)
                delta = round(self._rng.uniform(0.1, 0.3), 2)
                mutations.append(GeneratorMutation(
                    mutation_type=mtype,
                    target_runway_id=target_rwy.runway_id,
                    params={"penalty_delta": delta},
                    rationale=f"Weather degrades {target_rwy.runway_id} capacity by {delta}x",
                ))

            elif mtype == MutationType.INJECT_EMERGENCY and arrivals:
                base = self._rng.choice(arrivals)
                window_center = self._rng.randint(
                    base.earliest_minute, min(base.latest_minute, base.earliest_minute + 20)
                )
                mutations.append(GeneratorMutation(
                    mutation_type=mtype,
                    params={
                        "flight_id":  f"EMG{self._rng.randint(100,999)}",
                        "priority":   "emergency",
                        "minute":     window_center,
                        "runway":     self._rng.choice(task.runways).runway_id,
                    },
                    rationale="Inject emergency diversion at peak arrival window",
                ))

            elif mtype == MutationType.ADD_CONFLICTING_FLIGHT:
                if arrivals:
                    anchor = self._rng.choice(arrivals)
                    # Heavy arrival 4 min before a Light departure → wake trap
                    mutations.append(GeneratorMutation(
                        mutation_type=mtype,
                        params={
                            "flight_id":  f"WKT{self._rng.randint(100,999)}",
                            "wake_class": "H",
                            "operation":  "arrival",
                            "minute":     max(0, anchor.earliest_minute - 4),
                            "runway":     self._rng.choice(anchor.allowed_runways),
                        },
                        rationale="Heavy arrival 4 min before window — forces 6-min wake gap violation",
                    ))

            elif mtype == MutationType.CLOSE_RUNWAY_WINDOW:
                target_rwy = self._rng.choice(task.runways)
                duration = self._rng.randint(10, 20)
                mutations.append(GeneratorMutation(
                    mutation_type=mtype,
                    target_runway_id=target_rwy.runway_id,
                    params={"close_duration": duration},
                    rationale=f"Runway {target_rwy.runway_id} closed for {duration} min (inspection)",
                ))

        return mutations

    # ── Mutation application ──────────────────────────────────────────────────

    def _apply_mutation(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        if mut.mutation_type == MutationType.TIGHTEN_WINDOW:
            return self._tighten_window(task, mut)
        elif mut.mutation_type == MutationType.ADD_ATFM_DEADLINE:
            return task  # handled at environment level via atfm_deadlines dict
        elif mut.mutation_type == MutationType.INCREASE_WEATHER_PENALTY:
            return self._increase_weather(task, mut)
        elif mut.mutation_type == MutationType.INJECT_EMERGENCY:
            return self._inject_emergency(task, mut)
        elif mut.mutation_type == MutationType.ADD_CONFLICTING_FLIGHT:
            return self._add_conflicting_flight(task, mut)
        elif mut.mutation_type == MutationType.CLOSE_RUNWAY_WINDOW:
            return self._close_runway_window(task, mut)
        return task

    def _tighten_window(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        squeeze = mut.params.get("squeeze_minutes", 3)
        updated = []
        for f in task.flights:
            if f.flight_id == mut.target_flight_id:
                new_earliest = f.earliest_minute + squeeze
                new_latest   = f.latest_minute   - squeeze
                if new_latest - new_earliest >= MIN_WINDOW_WIDTH:
                    f = f.model_copy(update={
                        "earliest_minute": new_earliest,
                        "latest_minute":   new_latest,
                    })
            updated.append(f)
        return task.model_copy(update={"flights": updated})

    def _increase_weather(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        delta = mut.params.get("penalty_delta", 0.15)
        updated = []
        for rwy in task.runways:
            if rwy.runway_id == mut.target_runway_id:
                new_penalty = round(min(2.0, rwy.weather_penalty + delta), 2)
                rwy = rwy.model_copy(update={"weather_penalty": new_penalty})
            updated.append(rwy)
        return task.model_copy(update={"runways": updated})

    def _inject_emergency(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        p = mut.params
        fid = p.get("flight_id", "EMG001")
        minute = int(p.get("minute", 20))
        priority_str = str(p.get("priority", "emergency")).lower().strip()
        priority_str = _PRIORITY_ALIASES.get(priority_str, priority_str)
        try:
            priority = PriorityClass(priority_str)
        except ValueError:
            priority = PriorityClass.EMERGENCY
        runway_id = p.get("runway", task.runways[0].runway_id)
        runway = next((r for r in task.runways if r.runway_id == runway_id), task.runways[0])

        new_flight = FlightRecord(
            flight_id=fid,
            airline="GOV",
            operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=minute,
            earliest_minute=max(0, minute - 2),
            latest_minute=minute + 6,   # tight 8-min window
            allowed_runways=[runway_id],
            passengers=8,
            fuel_burn_per_minute=7.5,
            priority=priority,
            notes=f"Generator-injected {priority.value} diversion",
        )
        return task.model_copy(update={"flights": list(task.flights) + [new_flight]})

    def _add_conflicting_flight(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        p = mut.params
        fid = p.get("flight_id", "WKT001")
        minute = int(p.get("minute", 10))
        try:
            wake = WakeClass(str(p.get("wake_class", "H")).upper())
        except ValueError:
            wake = WakeClass.HEAVY
        try:
            operation = OperationType(str(p.get("operation", "arrival")).lower())
        except ValueError:
            operation = OperationType.ARRIVAL
        runway_id = p.get("runway", task.runways[0].runway_id)

        new_flight = FlightRecord(
            flight_id=fid,
            airline="FRT",
            operation=operation,
            wake_class=wake,
            scheduled_minute=minute,
            earliest_minute=max(0, minute - 1),
            latest_minute=minute + 5,
            allowed_runways=[runway_id],
            passengers=1,
            fuel_burn_per_minute=6.0,
            priority=PriorityClass.NORMAL,
            notes="Generator-injected wake-turbulence trap (Heavy)",
        )
        return task.model_copy(update={"flights": list(task.flights) + [new_flight]})

    def _close_runway_window(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        """Close runway by reducing hourly_capacity to 0 for target — simulated via
        extreme weather penalty (capacity reduction proxy)."""
        duration = mut.params.get("close_duration", 15)
        delta = min(1.9, 0.05 * duration)  # longer closure → bigger penalty
        updated = []
        for rwy in task.runways:
            if rwy.runway_id == mut.target_runway_id:
                rwy = rwy.model_copy(update={
                    "weather_penalty": min(2.0, rwy.weather_penalty + delta),
                    "notes": rwy.notes + f" [CLOSED {duration}min — generator]",
                })
            updated.append(rwy)
        return task.model_copy(update={"runways": updated})

    # ── Solvability check ─────────────────────────────────────────────────────

    def _check_solvability(self, task: TaskDefinition) -> bool:
        """Verify at least one valid assignment exists for each flight.

        Heuristic: check no flight's window is tighter than minimum separation
        and no runway is overloaded beyond theoretical maximum.
        """
        for f in task.flights:
            if f.latest_minute - f.earliest_minute < 2:
                return False
            if not f.allowed_runways:
                return False

        # Check runway capacity: total flights per runway must fit in horizon
        from collections import defaultdict
        runway_demand: Dict[str, int] = defaultdict(int)
        for f in task.flights:
            for rwy_id in f.allowed_runways:
                runway_demand[rwy_id] += 1

        for rwy in task.runways:
            demand = runway_demand.get(rwy.runway_id, 0)
            effective_capacity = rwy.hourly_capacity / rwy.weather_penalty
            horizon_hours = task.planning_horizon_minutes / 60.0
            max_ops = effective_capacity * horizon_hours
            if demand > max_ops * 1.5:  # allow some slack (flights share runways)
                return False

        return True

    def _score_scheduled_baseline(self, task: TaskDefinition) -> float:
        """Compute the 'do nothing' baseline score for PAIRED regret calculation.

        Assigns every flight to its scheduled_minute (clamped to window) on the
        first allowed runway — the simplest possible plan with no optimisation.
        The difference between controller_score and this baseline is the regret
        signal: generator is rewarded for scenarios where agents can't beat naive.
        """
        slots = []
        for f in task.flights:
            if not f.allowed_runways:
                continue
            minute = max(f.earliest_minute, min(f.latest_minute, f.scheduled_minute))
            slots.append(SlotAssignment(
                flight_id=f.flight_id,
                runway=f.allowed_runways[0],
                assigned_minute=minute,
                hold_minutes=0,
            ))
        try:
            outcome = simulate_plan(task, slots)
            return outcome.normalized_score
        except Exception:
            return 0.5  # fallback: treat as mid-quality baseline

    def _deep_copy_task(self, task: TaskDefinition) -> TaskDefinition:
        return TaskDefinition.model_validate(task.model_dump())
