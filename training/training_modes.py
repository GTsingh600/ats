"""GRPO training modes (dataset roster + episode composition).

Modes are selected via ``--training_mode`` / ``ATC_TRAINING_MODE``:

- ``full`` — default: AMAN, DMAN, GENERATOR, SUPERVISOR, ADAPT (6-row packs).
- ``hyper_minimal`` — AMAN + DMAN only (2-row packs, faster, ATC catalog + generator mutations).
- ``adapt_multidomain`` — AMAN + DMAN + ADAPT on **domain tasks only** (3-row packs), ≥10 domain
  worlds via ``domains`` registry (ICU + synthetic pack).  Dataset uses ``difficulty_scalar=0`` and
  ``randomize=False`` on env reset; ADAPT reward skips adaptive curriculum blend and biases
  improvement vs heuristic so GRPO sees less tie-flat variance.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Tuple

from multi_agent.models import AgentRole


class TrainingMode(str, Enum):
    FULL = "full"
    HYPER_MINIMAL = "hyper_minimal"
    ADAPT_MULTIDOMAIN = "adapt_multidomain"


def resolve_training_mode(explicit: str | None = None) -> TrainingMode:
    raw = (explicit or os.environ.get("ATC_TRAINING_MODE", "full")).strip().lower()
    for m in TrainingMode:
        if m.value == raw:
            return m
    return TrainingMode.FULL


def roster_config(mode: TrainingMode | None = None) -> Tuple[int, Tuple[str, ...]]:
    """Return ``(pack_size, required_roles)`` for strict roster checks."""
    m = mode or resolve_training_mode()
    if m == TrainingMode.HYPER_MINIMAL:
        return 2, (AgentRole.AMAN.value, AgentRole.DMAN.value)
    if m == TrainingMode.ADAPT_MULTIDOMAIN:
        return 3, (AgentRole.AMAN.value, AgentRole.DMAN.value, AgentRole.ADAPT.value)
    return 6, (
        AgentRole.AMAN.value,
        AgentRole.DMAN.value,
        AgentRole.GENERATOR.value,
        AgentRole.SUPERVISOR.value,
        AgentRole.ADAPT.value,
    )
