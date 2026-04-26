"""Full multi-agent roster guarantees for GRPO training.

Training rows are emitted in fixed packs so that, with batch_size a multiple of
the active ``pack_size`` and ``shuffle_train_dataset=False``, every optimizer batch
contains the required roles (depends on :mod:`training.training_modes`).
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from multi_agent.models import AgentRole

ROSTER_PACK_SIZE_DEFAULT = 6  # AMAN, DMAN, GENERATOR, SUPERVISOR, ADAPT×2
REQUIRED_ROLES_DEFAULT = (
    AgentRole.AMAN.value,
    AgentRole.DMAN.value,
    AgentRole.GENERATOR.value,
    AgentRole.SUPERVISOR.value,
    AgentRole.ADAPT.value,
)

# Back-compat names (``full`` mode); prefer :func:`get_roster_pack_size` / :func:`get_required_roles`.
ROSTER_PACK_SIZE = ROSTER_PACK_SIZE_DEFAULT
REQUIRED_ROLES = REQUIRED_ROLES_DEFAULT


def get_roster_pack_size() -> int:
    try:
        from training.training_modes import roster_config

        return roster_config()[0]
    except Exception:
        return ROSTER_PACK_SIZE_DEFAULT


def get_required_roles() -> Tuple[str, ...]:
    try:
        from training.training_modes import roster_config

        return roster_config()[1]
    except Exception:
        return REQUIRED_ROLES_DEFAULT


def strict_roster_enabled(
    *,
    grounded_live: bool = False,
    use_grounded_curriculum: bool = False,
) -> bool:
    v = os.environ.get("ATC_RELAX_ROSTER", "").strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return False
    try:
        from training.training_modes import TrainingMode, resolve_training_mode

        if resolve_training_mode() != TrainingMode.FULL:
            return True
    except Exception:
        pass
    if grounded_live or use_grounded_curriculum:
        return True
    return os.environ.get("ATC_STRICT_ROSTER_ALL", "").strip().lower() in {"1", "true", "yes", "on"}


def align_batch_size_to_roster(batch_size: int, pack: Optional[int] = None) -> int:
    """Largest ``<= batch_size`` that is divisible by ``pack`` (at least ``pack``)."""
    p = int(pack if pack is not None else get_roster_pack_size())
    bs = int(max(p, batch_size))
    return max(p, (bs // p) * p)


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
    """Pre-flight: every pack-sized window must contain all required roles."""
    pack = get_roster_pack_size()
    req = get_required_roles()
    if len(rows) < pack:
        raise RuntimeError(f"{context}: need at least {pack} rows for roster, got {len(rows)}")
    for i in range(0, len(rows) - pack + 1, pack):
        window = rows[i : i + pack]
        roles = {r.get("agent_role") for r in window}
        missing = [x for x in req if x not in roles]
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
    """Hard assert: no required role may be absent from this reward batch."""
    req = get_required_roles()
    if not roles:
        raise RuntimeError(f"batch {batch_index}: empty agent_role list")
    counts = Counter(str(r) for r in roles)
    out = {k: counts.get(k, 0) for k in req}
    zeros = [k for k, v in out.items() if v < min_each]
    if zeros:
        raise RuntimeError(
            f"batch {batch_index}: incomplete roster counts={dict(counts)} "
            f"(required each>={min_each} for {req}, missing {zeros})"
        )
    return out


def format_role_distribution(counts: Mapping[str, int]) -> str:
    req = get_required_roles()
    parts = [f"{k}={counts.get(k, 0)}" for k in req]
    return " ".join(parts)
