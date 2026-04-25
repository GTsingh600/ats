"""Continuous difficulty curriculum."""

import random

from tasks_grounded import GROUNDED_CURRICULUM_TASKS
from training.continuous_curriculum import (
    balanced_d_sequence,
    bucket_index,
    scenario_features,
    softmax_pick_grounded_task,
    validate_bucket_balance,
    ContinuousCurriculumState,
)
from training.dataset import build_episode_dataset


def test_balanced_d_sequence_buckets():
    rng = random.Random(0)
    seq = balanced_d_sequence(40, rng)
    assert len(seq) == 40
    counts = [0, 0, 0, 0]
    for d in seq:
        counts[bucket_index(d)] += 1
    assert counts == [10, 10, 10, 10]


def test_softmax_prefers_close_templates():
    rng = random.Random(0)
    hits = {t.task_id: 0 for t in GROUNDED_CURRICULUM_TASKS}
    for _ in range(400):
        t = softmax_pick_grounded_task(0.05, rng, k=4, temperature=5.0)
        hits[t.task_id] += 1
    assert hits.get("gc_l0_isolated_rwys", 0) > hits.get("gc_l5_capacity_stress", 0)


def test_scenario_features_smooth():
    f0 = scenario_features(0.0)
    f1 = scenario_features(1.0)
    assert f0["aircraft_count"] < f1["aircraft_count"]


def test_continuous_state_mu_update():
    st = ContinuousCurriculumState(mu=0.5, k_mu=0.1, target_success=0.5)
    st.record_batch([0.2, 0.8], [0.9, 0.9], [True, True])
    assert st.global_batches == 1


def test_build_grounded_dataset_bucket_balance():
    rows = build_episode_dataset(
        n_episodes=16,
        seed=1,
        use_grounded_curriculum=True,
    )
    gc = [r for r in rows if r.get("grounded_curriculum")]
    by_ep: dict = {}
    for r in gc:
        e = r["episode_id"]
        by_ep.setdefault(e, set()).add(r.get("continuous_difficulty"))
    assert len(by_ep) == 16
    counts = [0, 0, 0, 0]
    for r in gc:
        if r.get("agent_role") == "AMAN":
            counts[int(r["difficulty_bucket_index"])] += 1
    assert counts == [4, 4, 4, 4]


def test_live_curriculum_yields_rows(tmp_path):
    from training.live_curriculum import CurriculumManager, iter_live_grounded_rows

    mgr = CurriculumManager(
        seed=0,
        output_dir=tmp_path,
        continuous_state_path=tmp_path / "cc.json",
    )
    it = iter_live_grounded_rows(mgr, seed=1)
    rows = [next(it) for _ in range(9)]
    assert len(rows) == 9
    assert all(r.get("live_curriculum") for r in rows)
    assert all("continuous_difficulty" in r for r in rows)


def test_validate_bucket_balance_rejects_tiny_n():
    try:
        validate_bucket_balance(4)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
