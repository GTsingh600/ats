"""Core OpenEnv environment implementation for ATC optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..engine import empty_metrics, simulate_plan
    from ..graders import grade_task
    from ..models import (
        ATCOptimizationAction,
        ATCOptimizationObservation,
        ATCOptimizationState,
        PlanSnapshot,
        TaskDefinition,
        TaskGrade,
        TaskMetrics,
    )
    from ..tasks import ordered_tasks, render_task_briefing, task_catalog
except ImportError:
    from engine import empty_metrics, simulate_plan
    from graders import grade_task
    from models import (
        ATCOptimizationAction,
        ATCOptimizationObservation,
        ATCOptimizationState,
        PlanSnapshot,
        TaskDefinition,
        TaskGrade,
        TaskMetrics,
    )
    from tasks import ordered_tasks, render_task_briefing, task_catalog


class ATCOptimizationEnvironment(
    Environment[ATCOptimizationAction, ATCOptimizationObservation, ATCOptimizationState]
):
    """A realistic runway/slot recovery environment for ATC planning."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._catalog: Dict[str, TaskDefinition] = task_catalog()
        self._ordered_tasks: List[TaskDefinition] = list(ordered_tasks())
        self._state = ATCOptimizationState(active_task_ids=list(self._catalog))
        self._task: Optional[TaskDefinition] = None
        self._briefing = ""
        self._best_score = 0.0
        self._previous_score = 0.0  # used for potential-based reward shaping

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **_: object,
    ) -> ATCOptimizationObservation:
        """Reset into a chosen task or deterministically cycle by seed."""

        if task_id and task_id in self._catalog:
            task = self._catalog[task_id]
        elif difficulty:
            filtered = [item for item in self._ordered_tasks if item.difficulty.value == difficulty]
            task = filtered[0] if filtered else self._ordered_tasks[0]
        else:
            index = (seed or 0) % len(self._ordered_tasks)
            task = self._ordered_tasks[index]

        self._task = task
        self._briefing = render_task_briefing(task)
        self._best_score = 0.0
        self._previous_score = 0.0
        self._state = ATCOptimizationState(
            episode_id=episode_id or f"{task.task_id}-episode",
            step_count=0,
            task_id=task.task_id,
            task_title=task.title,
            difficulty=task.difficulty,
            airport=task.airport,
            seed=seed or 0,
            max_steps=task.max_steps,
            current_metrics=empty_metrics(),
            best_metrics=empty_metrics(),
            current_plan=[],
            best_plan=[],
            history=[],
            grader_history=[],
            final_summary="Scenario initialized.",
            active_task_ids=list(self._catalog),
        )
        return self._build_observation(
            reward=None,
            done=False,
            diagnostics=[
                "Submit a full slot plan covering every flight.",
                "Keep spacing safe under the reduced runway capacity.",
            ],
            recommendations=[
                "Prioritize emergency, medical, and connection-sensitive traffic.",
                "Use both runways when beneficial, but respect each flight's allowed runway set.",
            ],
            grader_feedback=["No grading yet; waiting for the first proposal."],
        )

    def step(
        self,
        action: ATCOptimizationAction,
        **_: object,
    ) -> ATCOptimizationObservation:
        """Score a candidate slot plan and emit dense reward feedback."""

        if self._task is None:
            return self.reset()

        outcome = simulate_plan(self._task, action.proposal)
        grades = grade_task(self._task, outcome, action.proposal, action.rationale)
        composite = next(
            (grade for grade in grades if grade.grader_name == "composite_task_grader"),
            grades[-1],
        )

        previous_best = self._best_score
        current_score = composite.score
        # Potential-based reward shaping (Ng et al. 1999): F(s,s') = Φ(s') - Φ(s)
        # Using previous *step* score as the state potential, not best-ever,
        # ensuring the reward signal is policy-gradient-safe and Markovian.
        reward = round(current_score - self._previous_score, 4)
        reward = max(-1.0, min(1.0, reward))
        done = (
            action.commit
            or self._state.step_count + 1 >= self._task.max_steps
            or current_score >= 0.98
        )

        current_metrics = outcome.metrics.model_copy(deep=True)
        current_metrics.agent_judgment = composite.score
        current_metrics.overall_score = composite.score

        best_metrics = self._state.best_metrics
        best_plan = self._state.best_plan
        if current_score >= self._best_score:
            self._best_score = current_score
            best_metrics = current_metrics
            best_plan = list(action.proposal)

        self._state.step_count += 1
        self._state.current_metrics = current_metrics
        self._state.best_metrics = best_metrics
        self._state.current_plan = list(action.proposal)
        self._state.best_plan = best_plan
        self._state.grader_history = list(grades)
        self._state.history.append(
            PlanSnapshot(
                step=self._state.step_count,
                score=current_score,
                reward=reward,
                commit=action.commit,
            )
        )
        self._state.final_summary = self._build_summary(current_metrics, grades, done)
        self._previous_score = current_score  # advance state potential for next step

        grader_feedback = [
            f"{grade.grader_name}: {grade.score:.3f} - {grade.rationale}" for grade in grades
        ]
        return self._build_observation(
            reward=reward,
            done=done,
            diagnostics=outcome.diagnostics,
            recommendations=outcome.recommendations,
            grader_feedback=grader_feedback,
        )

    @property
    def state(self) -> ATCOptimizationState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        readme_path = Path(__file__).resolve().parent.parent / "README.md"
        readme_content = (
            readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
        )
        return EnvironmentMetadata(
            name="ATC Optimization OpenEnv",
            description="Realistic air traffic control disruption recovery benchmark.",
            version="1.0.0",
            author="ATC Optimization OpenEnv Contributors",
            readme_content=readme_content,
        )

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
        diagnostics: List[str],
        recommendations: List[str],
        grader_feedback: List[str],
    ) -> ATCOptimizationObservation:
        task = self._task
        assert task is not None
        return ATCOptimizationObservation(
            task_id=task.task_id,
            task_title=task.title,
            difficulty=task.difficulty,
            airport=task.airport,
            briefing=self._briefing,
            objective=task.objective,
            grading_focus=task.grading_focus,
            flights=task.flights,
            runways=task.runways,
            current_metrics=self._state.current_metrics,
            best_metrics=self._state.best_metrics,
            current_plan=self._state.current_plan,
            diagnostics=diagnostics,
            recommendations=recommendations,
            grader_feedback=grader_feedback,
            steps_remaining=max(0, task.max_steps - self._state.step_count),
            reward=reward,
            done=done,
        )

    def _build_summary(self, metrics: TaskMetrics, grades: List[TaskGrade], done: bool) -> str:
        lead_grade = next(
            (grade for grade in grades if grade.grader_name == "composite_task_grader"),
            grades[-1],
        )
        status = "completed" if done else "in progress"
        return (
            f"Plan {status}. Composite score={lead_grade.score:.3f}, "
            f"delay={metrics.total_delay_minutes} min, "
            f"conflicts={metrics.conflict_count}, "
            f"priority_violations={metrics.priority_violations}."
        )
