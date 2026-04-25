"""Domain task catalogs for ADAPT agent transfer learning.

Each domain module provides a catalog of TaskDefinition objects that
represent scheduling problems from non-ATC domains. ADAPT receives these
tasks and infers ATC parameter mappings purely from structural signals —
no domain-specific knowledge is encoded here.

Adding a new domain:
  1. Create domains/<name>.py exporting <name>_task_catalog() -> Dict[str, TaskDefinition]
  2. Add an import below and register it in ALL_DOMAINS.
"""

from __future__ import annotations

from typing import Any, Dict

from models import TaskDefinition

# Registered domain modules — add new domains here
_DOMAIN_REGISTRY: list[str] = ["icu"]


def get_all_domain_tasks() -> Dict[str, TaskDefinition]:
    """Aggregate tasks from every registered domain.

    Returns a flat dict {task_id: TaskDefinition} across all domains.
    Safe to call even if some domain modules are missing (they are skipped
    with a warning).
    """
    all_tasks: Dict[str, TaskDefinition] = {}
    for name in _DOMAIN_REGISTRY:
        try:
            mod = __import__(f"domains.{name}", fromlist=[f"{name}_task_catalog"])
            catalog_fn = getattr(mod, f"{name}_task_catalog")
            catalog: Dict[str, TaskDefinition] = catalog_fn()
            all_tasks.update(catalog)
        except Exception as exc:
            import warnings
            warnings.warn(f"[domains] Could not load domain '{name}': {exc}")
    return all_tasks


def get_domain_description(domain_name: str) -> str:
    """Return a human-readable description for a registered domain, or ''."""
    try:
        mod = __import__(f"domains.{domain_name}", fromlist=["_DOMAIN_DESCRIPTION"])
        for attr in (f"{domain_name.upper()}_DOMAIN_DESCRIPTION", "_DOMAIN_DESCRIPTION"):
            if hasattr(mod, attr):
                return getattr(mod, attr)
    except Exception:
        pass
    return ""
