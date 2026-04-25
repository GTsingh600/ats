"""Continuous difficulty curriculum (scalar d in [0, 1]).

Internal state is always continuous; discrete labels appear only in logs
(`difficulty_bucket_index`, bin names). Grounded tasks use fixed structural
templates with per-task anchors; sampling picks the template nearest to d and
optional timing offsets verified by the rule-based solver.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from models import TaskDefinition
from tasks_grounded import (
    GROUNDED_CURRICULUM_TASKS,
    GROUNDED_TASK_DIFFICULTY_ANCHOR,
)

from training.curriculum_grounded import rule_based_plan_succeeds

# Logging / validation only — not used for internal decisions beyond coverage checks.
DIFFICULTY_BUCKETS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.25),
    (0.25, 0.5),
    (0.5, 0.75),
    (0.75, 1.0),
)


def scenario_features(d: float) -> Dict[str, float]:
    """Smooth structural features used for logging / diagnostics (monotonic in d)."""
    d = max(0.0, min(1.0, float(d)))
    # Smooth squashes so derivatives exist at boundaries.
    s = d * d * (3.0 - 2.0 * d)  # smoothstep
    return {
        "aircraft_count": 2.0 + 5.0 * s,
        "conflict_density": math.sin(0.5 * math.pi * s) ** 2,
        "timing_overlap": s * s,
        "uncertainty": 1.0 - math.cos(0.5 * math.pi * s),
    }


def beta_ab(mu: float, c: float) -> Tuple[float, float]:
    mu = max(1e-4, min(1.0 - 1e-4, float(mu)))
    c = max(0.5, float(c))
    return mu * c, (1.0 - mu) * c


def bucket_index(d: float) -> int:
    d = max(0.0, min(1.0, float(d)))
    for i, (lo, hi) in enumerate(DIFFICULTY_BUCKETS):
        if d < hi or i == len(DIFFICULTY_BUCKETS) - 1:
            if d >= lo or i == 0:
                return i
    return len(DIFFICULTY_BUCKETS) - 1


def balanced_d_sequence(n_episodes: int, rng: random.Random) -> List[float]:
    """Equal count per difficulty bucket; d ~ Uniform(bucket) — smooth coverage of [0,1]."""
    n = int(n_episodes)
    nb = len(DIFFICULTY_BUCKETS)
    if n < nb:
        raise ValueError(f"n_episodes must be >= {nb} for per-bucket balance, got {n}")
    base = n // nb
    rem = n % nb
    out: List[float] = []
    for i, (lo, hi) in enumerate(DIFFICULTY_BUCKETS):
        cnt = base + (1 if i < rem else 0)
        span = max(1e-9, hi - lo)
        for _ in range(cnt):
            out.append(lo + rng.random() * span)
    rng.shuffle(out)
    return out


def validate_bucket_balance(n_episodes: int, min_fraction: float = 0.98) -> None:
    """Reject dataset configs that cannot give near-equal bucket mass."""
    nb = len(DIFFICULTY_BUCKETS)
    n = int(n_episodes)
    if n < nb:
        raise ValueError(f"n_episodes must be >= {nb} (one per bucket minimum), got {n}")
    target = n / nb
    min_count = int(math.floor(target * min_fraction))
    if min_count < 1:
        raise ValueError(
            f"n_episodes={n} too small for four-way bucket balance at min_fraction={min_fraction} "
            f"(need larger n so each bucket gets at least one sample)."
        )


def nearest_grounded_task(d: float) -> TaskDefinition:
    """Deprecated for curriculum control; use ``softmax_pick_grounded_task`` (kept for tests/tools)."""
    anchors = GROUNDED_TASK_DIFFICULTY_ANCHOR
    d = max(0.0, min(1.0, float(d)))
    best = GROUNDED_CURRICULUM_TASKS[0]
    best_dist = 1.0
    for t in GROUNDED_CURRICULUM_TASKS:
        a = anchors[t.task_id]
        dist = abs(a - d)
        if dist < best_dist:
            best_dist = dist
            best = t
    return best


def softmax_pick_grounded_task(
    d: float,
    rng: random.Random,
    *,
    k: int = 4,
    temperature: float = 7.0,
) -> TaskDefinition:
    """Probabilistic template choice: softmax over top-k smallest anchor distances."""
    d = max(0.0, min(1.0, float(d)))
    anchors = GROUNDED_TASK_DIFFICULTY_ANCHOR
    ranked = sorted(
        GROUNDED_CURRICULUM_TASKS,
        key=lambda t: abs(anchors[t.task_id] - d),
    )
    k = max(1, min(int(k), len(ranked)))
    top = ranked[:k]
    dists = [abs(anchors[t.task_id] - d) for t in top]
    tau = max(0.35, float(temperature))
    weights = [math.exp(-dist / tau) for dist in dists]
    s = sum(weights) or 1.0
    weights = [w / s for w in weights]
    u = rng.random()
    acc = 0.0
    for t, w in zip(top, weights):
        acc += w
        if u <= acc:
            return t
    return top[-1]


def structural_signature_tuple(task: TaskDefinition) -> Tuple[Union[str, int, float, Tuple[str, ...]], ...]:
    """Hashable coarse structure (logging + batch diversity)."""
    rw = tuple(
        (
            r.runway_id,
            int(r.hourly_capacity),
            round(float(r.weather_penalty), 4),
            tuple(op.value for op in r.allowed_operations),
        )
        for r in sorted(task.runways, key=lambda x: x.runway_id)
    )
    fl = tuple(
        (
            f.flight_id,
            f.operation.value,
            f.wake_class.value,
            int(f.earliest_minute),
            int(f.latest_minute),
            int(f.scheduled_minute),
        )
        for f in sorted(task.flights, key=lambda x: x.flight_id)
    )
    return rw + fl


def difficulty_proxy_task(task: TaskDefinition) -> float:
    """Scalar proxy in [0,1] for structural hardness (must correlate with target d)."""
    n = max(1, len(task.flights))
    nfl = min(1.0, (n - 1) / 9.0)
    rw = max(1, len(task.runways))
    rw_score = min(1.0, (rw - 1) / 3.0)
    mixed = sum(1 for r in task.runways if len(r.allowed_operations) > 1)
    mix_score = min(1.0, mixed / 2.0)
    pens = [float(r.weather_penalty) for r in task.runways]
    stress = max(0.0, min(1.0, (sum(pens) / len(pens) - 1.0) / 0.72))
    caps = [60.0 / max(1, int(r.hourly_capacity)) for r in task.runways]
    cap_n = max(0.0, min(1.0, (sum(caps) / len(caps) - 2.0) / 3.5))
    tight_scores: List[float] = []
    for f in task.flights:
        w = max(1, int(f.latest_minute) - int(f.earliest_minute))
        tight_scores.append(1.0 - min(1.0, w / 48.0))
    tight_n = sum(tight_scores) / max(1, len(tight_scores))
    raw = 0.24 * nfl + 0.16 * rw_score + 0.18 * mix_score + 0.20 * stress + 0.12 * cap_n + 0.22 * tight_n
    return max(0.0, min(1.0, float(raw)))


def apply_meaningful_structural_variation(
    template: TaskDefinition,
    d: float,
    rng: random.Random,
) -> TaskDefinition:
    """Capacity / weather / budgets / windows — solver-gated; avoids timestamp-only edits."""
    d = max(0.0, min(1.0, float(d)))
    s = d * d * (3.0 - 2.0 * d)
    orig_rw = {r.runway_id: r for r in template.runways}
    t = template.model_copy(deep=True)
    for r in t.runways:
        o = orig_rw.get(r.runway_id)
        if o is None:
            continue
        r.weather_penalty = min(
            1.78,
            max(1.0, float(o.weather_penalty) * (1.0 + 0.42 * s * rng.uniform(0.88, 1.12))),
        )
        r.hourly_capacity = max(12, int(round(float(o.hourly_capacity) * (1.0 - 0.20 * s * rng.uniform(0.9, 1.1)))))
    t.delay_budget = max(60, int(round(float(template.delay_budget) * (1.0 + 0.18 * s * rng.uniform(0.95, 1.05)))))
    t.fuel_budget = max(200.0, float(template.fuel_budget) * (1.0 + 0.10 * s))
    for f in t.flights:
        span = int(f.latest_minute) - int(f.earliest_minute)
        shrink = int(round(1 + s * 2 * rng.random()))
        if span > shrink + 8 and rng.random() < 0.35 + 0.4 * s:
            f.latest_minute = int(f.latest_minute) - shrink
    shift = int(round((rng.random() - 0.5) * 5 * s))
    shift = max(-2, min(2, shift))
    if shift != 0:
        for f in t.flights:
            f.scheduled_minute = int(f.scheduled_minute) + shift
            f.earliest_minute = max(0, int(f.earliest_minute) + shift)
            f.latest_minute = max(int(f.earliest_minute), int(f.latest_minute) + shift)
    if not rule_based_plan_succeeds(t):
        return template
    return t


def apply_structural_variation(
    task: TaskDefinition,
    d: float,
    anchor: float,
    rng: random.Random,
) -> TaskDefinition:
    """Small integer time shifts scaled by |d-anchor|, solver-gated (structural only)."""
    t = task.model_copy(deep=True)
    span = max(1e-6, abs(float(d) - float(anchor)))
    delta = int(round(4.0 * span * (rng.random() - 0.5) * 2.0))
    delta = max(-3, min(3, delta))
    if delta == 0 and rng.random() < 0.2 * float(d):
        delta = rng.choice([-1, 1])
    if delta == 0:
        return t
    for f in t.flights:
        f.scheduled_minute = int(f.scheduled_minute) + delta
        f.earliest_minute = max(0, int(f.earliest_minute) + delta)
        f.latest_minute = max(int(f.earliest_minute), int(f.latest_minute) + delta)
    if not rule_based_plan_succeeds(t):
        return task
    return t


@dataclass
class ContinuousCurriculumState:
    """Performance-driven continuous curriculum (Beta sampling + mu update)."""

    mu: float = 0.45
    c: float = 10.0
    k_mu: float = 0.035
    target_success: float = 0.42
    uniform_mix: float = 0.14
    global_batches: int = 0
    use_adaptive_sampling: bool = False
    reward_by_bin: Dict[int, Deque[float]] = field(default_factory=dict)
    # Sweet-spot emphasis: upweight sampling near d where bin success in (0.25, 0.6)
    sweet_lo: float = 0.25
    sweet_hi: float = 0.60
    sweet_boost: float = 0.22
    # Safeguards
    reward_std_ema: float = 0.15
    min_c: float = 4.0
    max_c: float = 28.0
    # Binned success tracking (continuous d still drives sampling; bins are statistics only)
    window: int = 256
    success_by_bin: Dict[int, Deque[float]] = field(default_factory=dict)
    d_samples_log: Deque[float] = field(default_factory=lambda: deque(maxlen=4096))
    reward_corr_pairs: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=2048))

    def __post_init__(self) -> None:
        self.mu = max(0.05, min(0.95, float(self.mu)))
        self.c = max(self.min_c, min(self.max_c, float(self.c)))

    def sample_d_adaptive(self, rng: random.Random) -> float:
        """Beta(μc,(1−μ)c) + sweet-spot nudge only (outer 70/30 uniform mixture is separate)."""
        a, b = beta_ab(self.mu, self.c)
        base = rng.betavariate(max(0.05, a), max(0.05, b))
        if rng.random() < self.sweet_boost:
            base = 0.25 + rng.random() * 0.35
        return max(0.0, min(1.0, float(base)))

    def sample_d(self, rng: random.Random) -> float:
        """Legacy path: inner uniform + Beta + sweet (static dataset builds)."""
        if rng.random() < self.uniform_mix:
            base = rng.random()
        else:
            a, b = beta_ab(self.mu, self.c)
            base = rng.betavariate(max(0.05, a), max(0.05, b))
        if rng.random() < self.sweet_boost:
            base = 0.25 + rng.random() * 0.35
        return max(0.0, min(1.0, float(base)))

    def record_batch(
        self,
        difficulties: List[float],
        rewards: List[float],
        controller_mask: Optional[List[bool]] = None,
    ) -> None:
        """Update mu from batch success proxy; refresh bin stats; safeguards on reward std."""
        if not difficulties or not rewards:
            return
        n = min(len(difficulties), len(rewards))
        ds = [max(0.0, min(1.0, float(difficulties[i]))) for i in range(n)]
        rs = [float(rewards[i]) for i in range(n)]
        mask = controller_mask if controller_mask and len(controller_mask) >= n else [True] * n
        successes = []
        pair_ds: List[float] = []
        pair_rs: List[float] = []
        for i in range(n):
            if not mask[i]:
                continue
            ok = 1.0 if rs[i] > 0.02 else 0.0
            successes.append(ok)
            b = bucket_index(ds[i])
            if b not in self.success_by_bin:
                self.success_by_bin[b] = deque(maxlen=self.window)
            self.success_by_bin[b].append(ok)
            if b not in self.reward_by_bin:
                self.reward_by_bin[b] = deque(maxlen=self.window)
            self.reward_by_bin[b].append(float(rs[i]))
            pair_ds.append(ds[i])
            pair_rs.append(rs[i])
            self.d_samples_log.append(ds[i])
            self.reward_corr_pairs.append((ds[i], rs[i]))
        if not successes:
            return
        actual = sum(successes) / len(successes)
        self.mu = max(0.05, min(0.95, self.mu + self.k_mu * (self.target_success - actual)))
        self.global_batches += 1
        self.use_adaptive_sampling = True

        batch_std = _pop_std(rs)
        self.reward_std_ema = 0.9 * self.reward_std_ema + 0.1 * max(batch_std, 1e-6)
        if self.reward_std_ema < 0.04:
            self.c = min(self.max_c, self.c * 1.08)
        if self.reward_std_ema > 0.2:
            self.c = max(self.min_c, self.c * 0.97)

        if actual < 0.08:
            self.mu = max(0.05, self.mu - 0.06)

        # Pull μ toward quartile centers where recent success is in the learning band (continuous nudge).
        for b, q in self.success_by_bin.items():
            if len(q) < 12:
                continue
            sr = sum(q) / len(q)
            if 0.25 <= sr <= 0.60 and 0 <= b < len(DIFFICULTY_BUCKETS):
                lo, hi = DIFFICULTY_BUCKETS[b]
                center = 0.5 * (lo + hi)
                self.mu = max(0.05, min(0.95, self.mu + 0.015 * (center - self.mu)))

    def bin_success_snapshot(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, (lo, hi) in enumerate(DIFFICULTY_BUCKETS):
            q = self.success_by_bin.get(i)
            if q and len(q) > 0:
                out[f"{lo:.2f}-{hi:.2f}"] = sum(q) / len(q)
            else:
                out[f"{lo:.2f}-{hi:.2f}"] = float("nan")
        return out

    def bin_reward_mean_snapshot(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, (lo, hi) in enumerate(DIFFICULTY_BUCKETS):
            q = self.reward_by_bin.get(i)
            if q and len(q) > 0:
                out[f"{lo:.2f}-{hi:.2f}"] = sum(q) / len(q)
            else:
                out[f"{lo:.2f}-{hi:.2f}"] = float("nan")
        return out

    def reward_difficulty_correlation(self) -> float:
        pairs = list(self.reward_corr_pairs)
        if len(pairs) < 8:
            return float("nan")
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx < 1e-12 or vy < 1e-12:
            return float("nan")
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        return cov / math.sqrt(vx * vy)

    def to_json(self) -> str:
        payload = {
            "mu": self.mu,
            "c": self.c,
            "k_mu": self.k_mu,
            "target_success": self.target_success,
            "uniform_mix": self.uniform_mix,
            "global_batches": self.global_batches,
            "use_adaptive_sampling": self.use_adaptive_sampling,
            "reward_std_ema": self.reward_std_ema,
            "success_by_bin": {str(k): list(v) for k, v in self.success_by_bin.items()},
            "reward_by_bin": {str(k): list(v) for k, v in self.reward_by_bin.items()},
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json_file(cls, path: Path) -> "ContinuousCurriculumState":
        data = json.loads(path.read_text())
        st = cls(
            mu=float(data.get("mu", 0.45)),
            c=float(data.get("c", 10.0)),
            k_mu=float(data.get("k_mu", 0.035)),
            target_success=float(data.get("target_success", 0.42)),
            uniform_mix=float(data.get("uniform_mix", 0.14)),
            global_batches=int(data.get("global_batches", 0)),
            use_adaptive_sampling=bool(data.get("use_adaptive_sampling", False)),
            reward_std_ema=float(data.get("reward_std_ema", 0.15)),
        )
        for k, vals in data.get("success_by_bin", {}).items():
            st.success_by_bin[int(k)] = deque(vals, maxlen=st.window)
        for k, vals in data.get("reward_by_bin", {}).items():
            st.reward_by_bin[int(k)] = deque(vals, maxlen=st.window)
        return st


def _pop_std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def load_or_create_continuous_state(path: Optional[Path]) -> ContinuousCurriculumState:
    if path is not None and path.is_file():
        return ContinuousCurriculumState.from_json_file(path)
    return ContinuousCurriculumState()


def save_continuous_state(path: Path, state: ContinuousCurriculumState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json(), encoding="utf-8")
