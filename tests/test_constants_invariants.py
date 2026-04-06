"""Constants and precision invariants used by scoring and reporting."""

from __future__ import annotations

from constants import (
    AIRLINE_DELAY_PRECISION,
    FUEL_PRECISION,
    METRIC_PRECISION,
    SCORE_WEIGHTS,
    SEPARATION_BY_WAKE,
)


def test_score_weights_sum_to_one() -> None:
    assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9


def test_precision_values_are_non_negative() -> None:
    assert METRIC_PRECISION >= 0
    assert FUEL_PRECISION >= 0
    assert AIRLINE_DELAY_PRECISION >= 0


def test_wake_separation_matrix_has_all_class_pairs() -> None:
    wake_classes = {"H", "M", "L"}
    for lead in wake_classes:
        for trail in wake_classes:
            assert (lead, trail) in SEPARATION_BY_WAKE
            assert SEPARATION_BY_WAKE[(lead, trail)] >= 0
