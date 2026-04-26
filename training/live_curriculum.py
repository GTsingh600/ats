"""Live grounded GRPO curriculum: same-run resampling, continuous d only, no static epoch.

- Difficulty is always a scalar ``d ∈ [0, 1]``; quartiles are logging only.
- Sampling: with probability ``uniform_mixture`` (default 0.3) draw ``d ~ Uniform(0,1)``;
  otherwise draw from the adaptive Beta(μc,(1−μ)c) + sweet-spot nudge (no discrete levels).
- Template choice: softmax over top-k templates by inverse distance to anchors (not argmin).
- Structural edits: runway capacity, weather penalty, delay budget, window widths — all
  solver-gated; rejects scenarios whose difficulty proxy diverges from ``d``.
- ``IterableDataset`` yields rows indefinitely; training uses ``max_steps`` to terminate.

Each episode emits ``ROSTER_PACK_SIZE`` rows: AMAN, DMAN, GENERATOR, SUPERVISOR,
and two ADAPT rows (domain transfer + grounded shift) so ADAPT is ≥25% of samples
and every aligned batch can include all roles.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from models import RunwaySpec, TaskDefinition

from training.continuous_curriculum import (
    ContinuousCurriculumState,
    bucket_index,
    difficulty_proxy_task,
    load_or_create_continuous_state,
    save_continuous_state,
    softmax_pick_grounded_task,
    structural_signature_tuple,
    apply_meaningful_structural_variation,
)
from training.curriculum_grounded import grounded_task_proxy_score
from training.dataset import (
    SUPERVISOR_PROFILES,
    _build_solver_merged_plan_json,
    _estimate_controller_score,
    _make_aman_sample,
    _make_dman_sample,
    _make_generator_sample,
    _make_supervisor_sample,
    conflict_proxy_from_solver,
)
from training.adapt_curriculum import build_dual_adapt_samples
from training.roster_integrity import ROSTER_PACK_SIZE, get_roster_pack_size
from domains import get_all_domain_tasks
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.generator import ChallengeGenerator
from multi_agent.supervisor import SupervisorAgent


class CurriculumManager:
    """Thread-safe curriculum + scenario generation for live training."""

    def __init__(
        self,
        *,
        seed: int,
        output_dir: Path,
        continuous_state_path: Path,
        diversity_min_unique: float = 0.42,
        proxy_tolerance: float = 0.24,
        uniform_mixture: float = 0.30,
        softmax_top_k: int = 4,
        softmax_tau: float = 7.0,
        resample_attempts: int = 14,
        save_every_batches: int = 25,
    ) -> None:
        self._lock = threading.Lock()
        self._rng_master = random.Random(seed)
        self.output_dir = Path(output_dir)
        self.continuous_path = Path(continuous_state_path)
        self.state = load_or_create_continuous_state(
            self.continuous_path if self.continuous_path.is_file() else None
        )
        self.uniform_mixture = float(uniform_mixture)
        self.softmax_top_k = int(softmax_top_k)
        self.softmax_tau = float(softmax_tau)
        self.proxy_tolerance = float(proxy_tolerance)
        self.resample_attempts = int(resample_attempts)
        self.diversity_min_unique = float(diversity_min_unique)
        self.save_every_batches = int(save_every_batches)
        self.env = MultiAgentATCEnvironment(seed=seed)
        self.supervisor = SupervisorAgent()
        self.generator = ChallengeGenerator(seed=seed)
        try:
            self._domain_tasks = get_all_domain_tasks()
        except Exception:
            self._domain_tasks = {}
        self._episode_seq = 0
        self._dist_log_path = self.output_dir / "curriculum_effective_distribution.jsonl"
        self._last_generator_entropy_bits: float = 0.0
        self._last_structural_divergence: float = 0.0

    def _fork_rng(self) -> random.Random:
        with self._lock:
            return random.Random(self._rng_master.randint(1, 10**9))

    def sample_d(self, rng: random.Random) -> float:
        """0.3 Uniform + 0.7 adaptive Beta mixture (hard anti-collapse)."""
        if rng.random() < self.uniform_mixture:
            return rng.random()
        return self.state.sample_d_adaptive(rng)

    def pick_template(self, d: float, rng: random.Random) -> TaskDefinition:
        return softmax_pick_grounded_task(
            d, rng, k=self.softmax_top_k, temperature=self.softmax_tau
        )

    def materialize_task(self, template: TaskDefinition, d: float, rng: random.Random) -> TaskDefinition:
        return apply_meaningful_structural_variation(template, d, rng)

    def generate_episode_pack(
        self,
        ep_id: int,
        rng: random.Random,
        recent_keys: Optional[Set[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], float, str, float]:
        """Return ``ROSTER_PACK_SIZE`` samples: AMAN, DMAN, GENERATOR, SUPERVISOR, ADAPT×2."""
        recent_keys = recent_keys or set()
        best_pack: Optional[List[Dict[str, Any]]] = None
        best_d = 0.0
        best_tid = ""
        best_proxy = 0.0
        best_key = ""

        for attempt in range(self.resample_attempts):
            d = self.sample_d(rng)
            template = self.pick_template(d, rng)
            task = self.materialize_task(template, d, rng)
            proxy = difficulty_proxy_task(task)
            if abs(proxy - d) > self.proxy_tolerance:
                continue
            sig = structural_signature_tuple(task)
            sig_t = structural_signature_tuple(template)
            if sig == sig_t and attempt < self.resample_attempts - 1:
                continue
            key = f"{task.task_id}|{sig}"
            if recent_keys and key in recent_keys and attempt < self.resample_attempts - 3:
                continue
            ctrl = float(_estimate_controller_score(task))
            self.generator.update(ctrl)

            profile = self.supervisor.sample_profile(ep_id)
            sup_desc = SUPERVISOR_PROFILES[profile]["description"]
            aman_obs, dman_obs = self.env.reset(
                episode_id=ep_id,
                supervisor_profile=profile,
                mutated_task=task,
                randomize=False,
            )
            atfm_json = json.dumps(self.env._state.atfm_deadlines)
            ps = float(grounded_task_proxy_score(task))
            bidx = bucket_index(d)
            div = 0.0 if sig == sig_t else 1.0
            self._last_structural_divergence = div
            self._last_generator_entropy_bits = float(div + min(1.0, abs(proxy - d)))
            meta = {
                "grounded_curriculum": True,
                "live_curriculum": True,
                "continuous_difficulty": float(d),
                "difficulty_proxy": float(proxy),
                "difficulty_bucket_index": bidx,
                "grounded_template_id": template.task_id,
                "structural_signature": json.dumps(sig),
                "rule_proxy_score": ps,
                "training_band": f"log_quartile_{bidx}",
                "generator_from_grounded": True,
                "generator_structural_divergence": div,
                "generator_entropy_hint_bits": self._last_generator_entropy_bits,
            }
            gen_row = _make_generator_sample(
                ep_id,
                template,
                profile,
                self.generator.difficulty_level,
                self.generator.ema_score,
                float(d),
                self.generator.difficulty_distribution,
            )
            sup_row = _make_supervisor_sample(
                ep_id,
                task,
                profile,
                sup_desc,
                float(d),
                merged_plan_json=_build_solver_merged_plan_json(task),
            )
            if not self._domain_tasks:
                raise RuntimeError(
                    "Live curriculum requires non-empty domain tasks for ADAPT "
                    "(see domains/ registry)."
                )
            adapt_rows, trig = build_dual_adapt_samples(
                ep_id,
                mutated_grounded_task=task,
                profile=profile,
                d=float(d),
                rng=rng,
                domain_tasks=self._domain_tasks,
                conflict_proxy=conflict_proxy_from_solver(task),
            )
            meta["adapt_bundle_triggers"] = trig
            rows = [
                _make_aman_sample(
                    ep_id, aman_obs, atfm_json, "[]", sup_desc, profile, "bid", float(d)
                ),
                _make_dman_sample(
                    ep_id, dman_obs, atfm_json, "[]", sup_desc, profile, "bid", float(d)
                ),
                gen_row,
                sup_row,
            ]
            rows.extend(adapt_rows)
            for r in rows:
                r.update(meta)
            best_pack = rows
            best_d = d
            best_tid = template.task_id
            best_proxy = proxy
            best_key = key
            break

        if best_pack is None:
            # Fallback: easiest valid template
            d = 0.12
            template = self.pick_template(d, rng)
            task = self.materialize_task(template, d, rng)
            profile = self.supervisor.sample_profile(ep_id)
            sup_desc = SUPERVISOR_PROFILES[profile]["description"]
            aman_obs, dman_obs = self.env.reset(
                episode_id=ep_id,
                supervisor_profile=profile,
                mutated_task=task,
                randomize=False,
            )
            atfm_json = json.dumps(self.env._state.atfm_deadlines)
            proxy = difficulty_proxy_task(task)
            meta = {
                "grounded_curriculum": True,
                "live_curriculum": True,
                "continuous_difficulty": float(d),
                "difficulty_proxy": float(proxy),
                "difficulty_bucket_index": bucket_index(d),
                "grounded_template_id": template.task_id,
                "structural_signature": json.dumps(structural_signature_tuple(task)),
                "rule_proxy_score": float(grounded_task_proxy_score(task)),
                "training_band": "log_fallback",
            }
            ctrl = float(_estimate_controller_score(task))
            self.generator.update(ctrl)
            gen_row = _make_generator_sample(
                ep_id,
                template,
                profile,
                self.generator.difficulty_level,
                self.generator.ema_score,
                float(d),
                self.generator.difficulty_distribution,
            )
            sup_row = _make_supervisor_sample(
                ep_id,
                task,
                profile,
                sup_desc,
                float(d),
                merged_plan_json=_build_solver_merged_plan_json(task),
            )
            if not self._domain_tasks:
                raise RuntimeError("domains/ registry empty — cannot build ADAPT rows.")
            adapt_rows, trig = build_dual_adapt_samples(
                ep_id,
                mutated_grounded_task=task,
                profile=profile,
                d=float(d),
                rng=rng,
                domain_tasks=self._domain_tasks,
                conflict_proxy=conflict_proxy_from_solver(task),
            )
            meta["adapt_bundle_triggers"] = trig
            meta["generator_from_grounded"] = True
            best_pack = [
                _make_aman_sample(ep_id, aman_obs, atfm_json, "[]", sup_desc, profile, "bid", float(d)),
                _make_dman_sample(ep_id, dman_obs, atfm_json, "[]", sup_desc, profile, "bid", float(d)),
                gen_row,
                sup_row,
            ]
            best_pack.extend(adapt_rows)
            for r in best_pack:
                r.update(meta)
            best_d = d
            best_tid = template.task_id
            best_proxy = proxy
            best_key = "fallback"

        return best_pack, best_d, best_tid, best_proxy

    def next_episode_id(self) -> int:
        with self._lock:
            e = self._episode_seq
            self._episode_seq += 1
            return e

    def on_reward_batch(
        self,
        difficulties: List[float],
        rewards: List[float],
        roles: List[str],
    ) -> None:
        from multi_agent.models import AgentRole

        mask = [
            r in (AgentRole.AMAN.value, AgentRole.DMAN.value) for r in roles
        ]
        with self._lock:
            self.state.record_batch(difficulties, rewards, mask)
            if self.state.global_batches % self.save_every_batches == 0:
                save_continuous_state(self.continuous_path, self.state)
                self._log_distribution_locked()

    def save(self) -> None:
        with self._lock:
            save_continuous_state(self.continuous_path, self.state)

    def _log_distribution_locked(self) -> None:
        row = {
            "batches": self.state.global_batches,
            "mu": round(self.state.mu, 5),
            "c": round(self.state.c, 5),
            "reward_std_ema": round(self.state.reward_std_ema, 5),
            "bin_success": self.state.bin_success_snapshot(),
            "bin_reward_mean": self.state.bin_reward_mean_snapshot(),
            "mixture_uniform_weight": self.uniform_mixture,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self._dist_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def iter_live_grounded_rows(
    manager: CurriculumManager,
    seed: int,
    *,
    batch_window: int = 48,
) -> Iterator[Dict[str, Any]]:
    """Infinite generator of training rows; use ``max_steps`` in GRPO to bound training."""
    rng = random.Random(int(seed) + 7919)
    recent: Set[str] = set()

    while True:
        ep = manager.next_episode_id()
        pack, _d, _tid, _proxy = manager.generate_episode_pack(ep, rng, recent)
        keys_in_pack = {r.get("structural_signature", "") + "|" + str(r.get("task_id")) for r in pack}
        uniq = len(set(keys_in_pack)) / max(1, len(keys_in_pack))
        if uniq < manager.diversity_min_unique:
            pack, _, _, _ = manager.generate_episode_pack(ep + 1337, rng, set())
        for row in pack:
            k = f"{row.get('task_id')}|{row.get('structural_signature')}"
            recent.add(k)
            if len(recent) > batch_window * 6:
                recent.clear()
            yield row


def live_max_steps(n_episodes: int, batch_size: int, grad_accum: int, passes: float = 2.5) -> int:
    """Approximate steps to cover ``passes`` virtual epochs over roster rows per episode (pack size from mode)."""
    rows_per_ep = max(1, int(get_roster_pack_size()))
    total_rows = max(1, int(n_episodes * rows_per_ep * passes))
    eff_bs = max(1, int(batch_size) * max(1, int(grad_accum)))
    return max(64, (total_rows + eff_bs - 1) // eff_bs)
