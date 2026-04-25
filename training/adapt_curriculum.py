"""ADAPT curriculum hooks for grounded + live GRPO.

ADAPT is invoked for:
  - cross-domain transfer (ICU / other registered domains)
  - structural re-interpretation of the *current* grounded scenario (distribution shift)
  - high continuous difficulty ``d`` (edge / stress)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from models import TaskDefinition
from multi_agent.adapt import build_adapt_observation
from multi_agent.models import SupervisorProfileName

from training.dataset import _make_adapt_sample


def adapt_trigger_reasons(
    d: float,
    *,
    conflict_proxy: float,
    rng: random.Random,
) -> List[str]:
    """Return one or more symbolic triggers for logging."""
    reasons: List[str] = []
    if d >= 0.72:
        reasons.append("high_d")
    if conflict_proxy >= 0.35:
        reasons.append("conflict_proxy")
    if rng.random() < 0.18:
        reasons.append("stochastic_perturbation")
    if not reasons:
        reasons.append("scheduled_coverage")
    return reasons


def pick_domain_transfer_task(
    domain_tasks: Dict[str, TaskDefinition],
    rng: random.Random,
) -> Tuple[str, TaskDefinition]:
    if not domain_tasks:
        raise RuntimeError("ADAPT curriculum: no domain tasks (domains/ registry empty)")
    tid = rng.choice(list(domain_tasks.keys()))
    return tid, domain_tasks[tid]


def build_dual_adapt_samples(
    ep_id: int,
    *,
    mutated_grounded_task: TaskDefinition,
    profile: SupervisorProfileName,
    d: float,
    rng: random.Random,
    domain_tasks: Dict[str, TaskDefinition],
    conflict_proxy: float,
) -> Tuple[List[Dict[str, Any]], str]:
    """Two ADAPT rows: domain transfer + grounded structural shift (≥25% of a 6-pack)."""
    triggers = adapt_trigger_reasons(d, conflict_proxy=conflict_proxy, rng=rng)
    _, dom_task = pick_domain_transfer_task(domain_tasks, rng)
    obs_dom = build_adapt_observation(dom_task, profile)
    row_dom = _make_adapt_sample(ep_id, obs_dom, dom_task, difficulty_scalar=float(d))
    row_dom["adapt_curriculum"] = True
    row_dom["adapt_trigger"] = "|".join(triggers)
    row_dom["adapt_track"] = "domain_transfer"

    obs_g = build_adapt_observation(mutated_grounded_task, profile)
    row_g = _make_adapt_sample(ep_id, obs_g, mutated_grounded_task, difficulty_scalar=float(d))
    row_g["adapt_curriculum"] = True
    row_g["adapt_trigger"] = "|".join(triggers + ["grounded_structural_shift"])
    row_g["adapt_track"] = "grounded_shift"

    return [row_dom, row_g], "|".join(triggers)
