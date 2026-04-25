"""Roster alignment and full-pack assertions."""

import pytest

from training.roster_integrity import (
    ROSTER_PACK_SIZE,
    align_batch_size_to_roster,
    assert_batch_role_counts,
    assert_dataset_has_full_roster,
    strict_roster_enabled,
)
from multi_agent.models import AgentRole


def test_align_batch_size_to_roster():
    assert align_batch_size_to_roster(8) == 6
    assert align_batch_size_to_roster(12) == 12
    assert align_batch_size_to_roster(5) == 6


def test_assert_batch_role_counts_ok():
    roles = (
        [AgentRole.AMAN.value] * 2
        + [AgentRole.DMAN.value] * 2
        + [AgentRole.GENERATOR.value] * 2
        + [AgentRole.SUPERVISOR.value] * 1
        + [AgentRole.ADAPT.value] * 1
    )
    assert len(roles) == 8
    assert_batch_role_counts([str(x) for x in roles], batch_index=0, min_each=1)


def test_assert_batch_role_counts_missing():
    roles = [AgentRole.AMAN.value] * 8
    with pytest.raises(RuntimeError):
        assert_batch_role_counts([str(x) for x in roles], batch_index=3)


def test_assert_dataset_pack():
    pack = []
    for _ in range(2):
        pack.extend(
            [
                {"agent_role": "AMAN"},
                {"agent_role": "DMAN"},
                {"agent_role": "GENERATOR"},
                {"agent_role": "SUPERVISOR"},
                {"agent_role": "ADAPT"},
                {"agent_role": "ADAPT"},
            ]
        )
    assert_dataset_has_full_roster(pack, context="test")


def test_strict_flag_grounded():
    assert strict_roster_enabled(grounded_live=True, use_grounded_curriculum=False) is True
    assert strict_roster_enabled(grounded_live=False, use_grounded_curriculum=True) is True
