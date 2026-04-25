"""Grounded curriculum: rule solver, Level-0 gate, dataset path."""

from training.curriculum_grounded import (
    rule_based_plan_succeeds,
    solve_grounded_rule_based,
    validate_level0_gate,
)
from training.dataset import build_episode_dataset
from tasks_grounded import GROUNDED_CURRICULUM_TASKS


def test_rule_solver_all_grounded_tasks():
    for t in GROUNDED_CURRICULUM_TASKS:
        slots = solve_grounded_rule_based(t)
        assert slots is not None and len(slots) == len(t.flights)
        assert rule_based_plan_succeeds(t)


def test_level0_gate():
    ok, detail = validate_level0_gate(0.8)
    assert ok
    assert "gc_l0" in detail[0]


def test_build_episode_dataset_grounded_smoke():
    rows = build_episode_dataset(
        n_episodes=16,
        seed=0,
        use_grounded_curriculum=True,
    )
    assert rows
    gc_rows = [r for r in rows if r.get("grounded_curriculum")]
    assert len(gc_rows) == 16 * 3  # AMAN + DMAN + SUP per episode, no GEN/ADAPT
    assert all(r["task_id"].startswith("gc_") for r in gc_rows)
    assert all("continuous_difficulty" in r for r in gc_rows)
    assert set(r["training_band"] for r in gc_rows) <= {f"bucket_{i}" for i in range(4)}
