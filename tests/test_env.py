"""Basic smoke tests for the ATC OpenEnv benchmark."""

from server.atc_environment import ATCOptimizationEnvironment
from models import ATCOptimizationAction
from planner import build_heuristic_plan


def test_reset_exposes_tasks() -> None:
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id="delhi_monsoon_recovery_easy")
    assert obs.task_id == "delhi_monsoon_recovery_easy"
    assert len(obs.flights) >= 3
    assert obs.steps_remaining > 0


def test_step_returns_bounded_score() -> None:
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id="mumbai_bank_balance_medium")
    proposal = build_heuristic_plan(obs)
    result = env.step(
        ATCOptimizationAction(
            proposal=proposal,
            rationale="Heuristic baseline plan for testing.",
            commit=True,
        )
    )
    assert 0.0 <= result.current_metrics.overall_score <= 1.0
    assert result.done is True
