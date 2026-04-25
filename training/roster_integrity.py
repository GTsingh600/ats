"""Full multi-agent roster guarantees for GRPO training.

Training rows are emitted in fixed packs so that, with batch_size a multiple of
``ROSTER_PACK_SIZE`` and ``shuffle_train_dataset=False``, every optimizer batch
contains at least one sample per role (AMAN, DMAN, GENERATOR, SUPERVISOR, ADAPT).
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional

from multi_agent.models import AgentRole

ROSTER_PACK_SIZE = 6  # AMAN, DMAN, GENERATOR, SUPERVISOR, ADAPT×2 (≥25% ADAPT rows)

REQUIRED_ROLES = (
    AgentRole.AMAN.value,
    AgentRole.DMAN.value,
    AgentRole.GENERATOR.value,
    AgentRole.SUPERVISOR.value,
    AgentRole.ADAPT.value,
)


def strict_roster_enabled(
    *,
    grounded_live: bool = False,
    use_grounded_curriculum: bool = False,
) -> bool:
    v = os.environ.get("ATC_RELAX_ROSTER", "").strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return False
    if grounded_live or use_grounded_curriculum:
        return True
    return os.environ.get("ATC_STRICT_ROSTER_ALL", "").strip().lower() in {"1", "true", "yes", "on"}


def align_batch_size_to_roster(batch_size: int, pack: int = ROSTER_PACK_SIZE) -> int:
    """Largest ``<= batch_size`` that is divisible by ``pack`` (at least ``pack``)."""
    bs = int(max(pack, batch_size))
    return max(pack, (bs // pack) * pack)


def count_roles_in_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        role = r.get("agent_role")
        if isinstance(role, str):
            c[role] += 1
    return dict(c)


def assert_dataset_has_full_roster(
    rows: List[Mapping[str, Any]],
    *,
    context: str = "dataset",
) -> None:
    """Pre-flight: every ``ROSTER_PACK_SIZE``-row window must contain all required roles."""
    if len(rows) < ROSTER_PACK_SIZE:
        raise RuntimeError(
            f"{context}: need at least {ROSTER_PACK_SIZE} rows for full roster, got {len(rows)}"
        )
    for i in range(0, len(rows) - ROSTER_PACK_SIZE + 1, ROSTER_PACK_SIZE):
        window = rows[i : i + ROSTER_PACK_SIZE]
        roles = {r.get("agent_role") for r in window}
        missing = [x for x in REQUIRED_ROLES if x not in roles]
        if missing:
            raise RuntimeError(
                f"{context}: roster pack starting at index {i} missing roles {missing}. "
                f"Got roles={sorted(roles)}"
            )


def assert_batch_role_counts(
    roles: List[str],
    *,
    batch_index: int,
    min_each: int = 1,
) -> Dict[str, int]:
    """Hard assert: no role may be absent from this reward batch."""
    if not roles:
        raise RuntimeError(f"batch {batch_index}: empty agent_role list")
    counts = Counter(str(r) for r in roles)
    out = {k: counts.get(k, 0) for k in REQUIRED_ROLES}
    zeros = [k for k, v in out.items() if v < min_each]
    if zeros:
        raise RuntimeError(
            f"batch {batch_index}: incomplete multi-agent roster counts={dict(counts)} "
            f"(required each>={min_each} for {REQUIRED_ROLES}, missing {zeros})"
        )
    return out


def format_role_distribution(counts: Mapping[str, int]) -> str:
    parts = [f"{k}={counts.get(k, 0)}" for k in REQUIRED_ROLES]
    return " ".join(parts)
