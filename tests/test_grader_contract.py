"""Grader contract tests for boundedness and deterministic scoring policy."""

from __future__ import annotations

from engine import simulate_plan
from graders import CompositeTaskGrader, SupervisorHeuristicGrader, grade_task
from models import ATCOptimizationAction
from planner import build_heuristic_plan
from server.atc_environment import ATCOptimizationEnvironment
from tasks import task_catalog


def _task_and_outcome(task_id: str = "mumbai_bank_balance_medium"):
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id=task_id)
    proposal = build_heuristic_plan(obs)
    task = task_catalog()[task_id]
    outcome = simulate_plan(task, proposal)
    return task, outcome, proposal


def test_supervisor_heuristic_score_is_bounded() -> None:
    task, outcome, proposal = _task_and_outcome("bengaluru_irrops_hard")
    grader = SupervisorHeuristicGrader()
    grade = grader.grade(task, outcome, proposal, rationale="short")
    assert 0.0 <= grade.score <= 1.0


def test_composite_uses_documented_deterministic_formula() -> None:
    task, outcome, proposal = _task_and_outcome()
    composite = CompositeTaskGrader().grade(task, outcome, proposal, rationale="Deterministic plan")
    all_grades = {g.grader_name: g for g in grade_task(task, outcome, proposal, "Deterministic plan")}

    expected = max(
        0.0,
        min(
            1.0,
            0.65 * outcome.metrics.overall_score
            + 0.20 * all_grades["supervisor_heuristic"].score
            + 0.15 * all_grades["deterministic_audit"].score,
        ),
    )
    assert abs(composite.score - round(expected, 4)) < 1e-9


def test_step_rejects_duplicate_assignment_with_bounded_score() -> None:
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id="delhi_monsoon_recovery_easy")
    proposal = build_heuristic_plan(obs)
    duplicated = proposal + [proposal[0]]
    result = env.step(
        ATCOptimizationAction(
            proposal=duplicated,
            rationale="Intentional duplicate for robustness grading.",
            commit=True,
        )
    )
    assert result.current_metrics.invalid_assignments >= 1
    assert 0.0 <= result.current_metrics.overall_score <= 1.0
