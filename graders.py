"""Task graders for benchmark scoring."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from openai import OpenAI

try:
    from .engine import SimulationOutcome
    from .models import SlotAssignment, TaskDefinition, TaskGrade
except ImportError:
    from engine import SimulationOutcome
    from models import SlotAssignment, TaskDefinition, TaskGrade


class BaseTaskGrader(ABC):
    """Base class for task graders."""

    grader_name: str

    @abstractmethod
    def grade(
        self,
        task: TaskDefinition,
        outcome: SimulationOutcome,
        proposal: Iterable[SlotAssignment],
        rationale: str = "",
    ) -> TaskGrade:
        """Return a score in the inclusive range [0.0, 1.0]."""


class SupervisorHeuristicGrader(BaseTaskGrader):
    """Deterministic controller-in-the-loop style grader."""

    grader_name = "supervisor_heuristic"

    def grade(
        self,
        task: TaskDefinition,
        outcome: SimulationOutcome,
        proposal: Iterable[SlotAssignment],
        rationale: str = "",
    ) -> TaskGrade:
        proposal_count = len(list(proposal))
        rationale_bonus = 0.08 if len(rationale.split()) >= 12 else 0.03 if rationale else 0.0
        diagnostic_penalty = min(0.12, 0.01 * len(outcome.diagnostics))
        missing_penalty = min(
            0.35,
            0.05 * outcome.metrics.missing_assignments
            + 0.04 * outcome.metrics.invalid_assignments,
        )
        score = max(
            0.0,
            min(
                1.0,
                0.30 * outcome.metrics.schedule_completeness
                + 0.25 * outcome.metrics.conflict_free_ratio
                + 0.18 * outcome.metrics.priority_handling
                + 0.10 * outcome.metrics.fairness
                + 0.08 * outcome.metrics.delay_efficiency
                + 0.04 * outcome.metrics.fuel_efficiency
                + 0.05 * min(1.0, proposal_count / max(1, len(task.flights)))
                + rationale_bonus
                - diagnostic_penalty,
            ),
        )
        score = max(0.0, min(1.0, score - missing_penalty))
        rationale_text = (
            "The supervisor accepted the proposal as operationally credible."
            if score >= 0.8
            else "The supervisor found the plan partially acceptable but still risky."
            if score >= 0.5
            else "The supervisor rejected the plan because safety or prioritization remained weak."
        )
        return TaskGrade(
            grader_name=self.grader_name,
            score=round(score, 4),
            rationale=rationale_text,
            sub_scores={
                "conflict_free_ratio": outcome.metrics.conflict_free_ratio,
                "priority_handling": outcome.metrics.priority_handling,
                "fairness": outcome.metrics.fairness,
                "delay_efficiency": outcome.metrics.delay_efficiency,
            },
        )


class LLMSupervisorGrader(BaseTaskGrader):
    """Optional LLM-backed supervisor used when credentials are available."""

    grader_name = "llm_supervisor"

    def __init__(self, model_name: Optional[str] = None):
        self.api_base_url = os.getenv("API_BASE_URL", "")
        self.api_key = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", ""))
        self.model_name = model_name or os.getenv("MODEL_NAME", "")

    def _enabled(self) -> bool:
        return bool(self.api_base_url and self.api_key and self.model_name)

    def grade(
        self,
        task: TaskDefinition,
        outcome: SimulationOutcome,
        proposal: Iterable[SlotAssignment],
        rationale: str = "",
    ) -> TaskGrade:
        if not self._enabled():
            return TaskGrade(
                grader_name=self.grader_name,
                score=round(outcome.metrics.overall_score, 4),
                rationale="LLM grader disabled; using deterministic fallback identical to operational score.",
                sub_scores={"fallback": outcome.metrics.overall_score},
            )

        client = OpenAI(base_url=self.api_base_url, api_key=self.api_key)
        proposal_summary = [
            {
                "flight_id": assignment.flight_id,
                "runway": assignment.runway,
                "assigned_minute": assignment.assigned_minute,
            }
            for assignment in proposal
        ]
        prompt = (
            "You are a senior ATC supervisor grading a runway recovery plan.\n"
            "Return strict JSON with keys score and rationale.\n"
            "Score must be a float between 0.0 and 1.0.\n\n"
            f"Task: {task.title}\n"
            f"Objective: {task.objective}\n"
            f"Operational metrics: {outcome.metrics.model_dump_json()}\n"
            f"Diagnostics: {json.dumps(outcome.diagnostics)}\n"
            f"Proposal: {json.dumps(proposal_summary)}\n"
            f"Agent rationale: {rationale}\n"
        )
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                max_tokens=180,
                messages=[
                    {
                        "role": "system",
                        "content": "Grade ATC recovery plans conservatively and output strict JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw_text = (response.choices[0].message.content or "").strip()
            data = json.loads(raw_text)
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            rationale_text = str(data.get("rationale", "LLM grader returned no rationale."))
        except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as exc:
            score = outcome.metrics.overall_score
            rationale_text = f"LLM grading failed, reverted to deterministic score: {exc}"

        return TaskGrade(
            grader_name=self.grader_name,
            score=round(score, 4),
            rationale=rationale_text,
            sub_scores={"operational_score": outcome.metrics.overall_score},
        )


class CompositeTaskGrader(BaseTaskGrader):
    """Blends deterministic and optional LLM judgment into one score."""

    grader_name = "composite_task_grader"

    def __init__(self) -> None:
        self.heuristic = SupervisorHeuristicGrader()
        self.llm = LLMSupervisorGrader()

    def grade(
        self,
        task: TaskDefinition,
        outcome: SimulationOutcome,
        proposal: Iterable[SlotAssignment],
        rationale: str = "",
    ) -> TaskGrade:
        proposal_list = list(proposal)
        heuristic = self.heuristic.grade(task, outcome, proposal_list, rationale)
        llm = self.llm.grade(task, outcome, proposal_list, rationale)
        final_score = max(
            0.0,
            min(
                1.0,
                0.65 * outcome.metrics.overall_score
                + 0.20 * heuristic.score
                + 0.15 * llm.score,
            ),
        )
        rationale_text = (
            f"Heuristic supervisor: {heuristic.rationale} "
            f"LLM supervisor: {llm.rationale}"
        )
        return TaskGrade(
            grader_name=self.grader_name,
            score=round(final_score, 4),
            rationale=rationale_text,
            sub_scores={
                "heuristic": heuristic.score,
                "llm": llm.score,
            },
        )


def grade_task(
    task: TaskDefinition,
    outcome: SimulationOutcome,
    proposal: Iterable[SlotAssignment],
    rationale: str = "",
) -> List[TaskGrade]:
    """Run all task graders and return their scores."""

    graders: List[BaseTaskGrader] = [
        SupervisorHeuristicGrader(),
        LLMSupervisorGrader(),
        CompositeTaskGrader(),
    ]
    proposal_list = list(proposal)
    return [grader.grade(task, outcome, proposal_list, rationale) for grader in graders]
