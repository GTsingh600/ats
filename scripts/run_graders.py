"""Enumerate tasks, run the graders, and assert score ranges."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import simulate_plan
from graders import grade_task
from models import ATCOptimizationObservation
from planner import build_heuristic_plan, build_refined_plan
from tasks import ordered_tasks, render_task_briefing


def main() -> int:
    rows = []
    for task in ordered_tasks():
        observation = ATCOptimizationObservation(
            task_id=task.task_id,
            task_title=task.title,
            difficulty=task.difficulty,
            airport=task.airport,
            briefing=render_task_briefing(task),
            objective=task.objective,
            grading_focus=task.grading_focus,
            flights=task.flights,
            runways=task.runways,
            steps_remaining=task.max_steps,
        )
        seed_plan = build_heuristic_plan(observation)
        proposal = build_refined_plan(observation, seed_plan=seed_plan)
        outcome = simulate_plan(task, proposal)
        grades = grade_task(task, outcome, proposal, "Deterministic two-step baseline.")
        for grade in grades:
            assert 0.0 <= grade.score <= 1.0, f"{grade.grader_name} out of range for {task.task_id}"
        rows.append(
            {
                "task_id": task.task_id,
                "difficulty": task.difficulty.value,
                "seed_operational_score": simulate_plan(task, seed_plan).metrics.overall_score,
                "operational_score": outcome.metrics.overall_score,
                "grader_scores": {grade.grader_name: grade.score for grade in grades},
            }
        )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
