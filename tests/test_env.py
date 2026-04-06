"""Basic smoke tests for the ATC OpenEnv benchmark."""

from engine import simulate_plan
from graders import grade_task
from server.atc_environment import ATCOptimizationEnvironment
from models import ATCOptimizationAction
from planner import build_heuristic_plan, build_refined_plan
from tasks import task_catalog


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


def test_refined_plan_is_not_worse_than_seed_plan() -> None:
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id="bengaluru_irrops_hard")
    seed_plan = build_heuristic_plan(obs)
    refined_plan = build_refined_plan(obs, seed_plan=seed_plan)
    task = task_catalog()[obs.task_id]
    seed_outcome = simulate_plan(task, seed_plan)
    refined_outcome = simulate_plan(task, refined_plan)
    assert refined_outcome.metrics.overall_score >= seed_outcome.metrics.overall_score


def test_composite_grader_is_deterministic() -> None:
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id="mumbai_bank_balance_medium")
    proposal = build_heuristic_plan(obs)
    task = task_catalog()[obs.task_id]
    outcome = simulate_plan(task, proposal)
    first = {
        grade.grader_name: grade.score
        for grade in grade_task(task, outcome, proposal, "Deterministic plan.")
    }
    second = {
        grade.grader_name: grade.score
        for grade in grade_task(task, outcome, proposal, "Deterministic plan.")
    }
    assert first["composite_task_grader"] == second["composite_task_grader"]


def test_duplicate_assignments_are_penalized() -> None:
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id="delhi_monsoon_recovery_easy")
    proposal = build_heuristic_plan(obs)
    duplicated = proposal + [proposal[0]]
    result = env.step(
        ATCOptimizationAction(
            proposal=duplicated,
            rationale="Intentional duplicate for penalty validation.",
            commit=True,
        )
    )
    assert result.current_metrics.invalid_assignments >= 1
