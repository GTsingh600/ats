"""Task graders for benchmark scoring.

Architecture (3-layer gated design):
  Layer 1 — SafetyGateEvaluator:    Hard constraint gates. Any violation caps the ceiling score.
  Layer 2 — PriorityRubricGrader:   Grades quality of emergency/medical/connection prioritization.
  Layer 3 — EfficiencyRubricGrader: Grades operational efficiency (delay, fuel, fairness, connection impact).
  GatedCompositeGrader:             Official score = min(gate_ceiling, 0.30*priority + 0.70*efficiency).

  LLMSupervisorGrader runs independently as auxiliary information only and is NOT part of the
  official composite score.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, OpenAIError, RateLimitError

try:
    from .engine import SimulationOutcome
    from .models import PriorityClass, SlotAssignment, TaskDefinition, TaskGrade
except ImportError:
    from engine import SimulationOutcome
    from models import PriorityClass, SlotAssignment, TaskDefinition, TaskGrade


STRICT_SCORE_EPSILON = 1e-4


def _strict_score(value: float) -> float:
    """Normalize to the strict open interval (0, 1) with stable output precision."""
    clipped = max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, float(value)))
    return round(clipped, 4)


# ── Layer 1: Hard Safety Gates ────────────────────────────────────────────────

class SafetyGateEvaluator:
    """Computes the hard constraint ceiling that caps the final composite score.

    Violations of safety-critical rules (separation conflicts, incomplete schedule,
    emergency flight delayed) set a score ceiling regardless of efficiency metrics.
    This mirrors real-world ATC: a conflicting plan is REJECTED, not averaged.
    """

    def evaluate(
        self, outcome: SimulationOutcome, task: TaskDefinition
    ) -> Tuple[float, List[str]]:
        """Return (ceiling, list_of_gate_violation_descriptions)."""
        ceiling = 1.0
        violations: List[str] = []

        # Gate 1: Separation conflicts — hardest veto (each extra conflict makes it worse)
        if outcome.metrics.conflict_count > 0:
            # First conflict drops to 0.40; each additional conflict reduces further, minimum 0.10
            conflict_ceiling = max(0.10, 0.40 - 0.05 * (outcome.metrics.conflict_count - 1))
            ceiling = min(ceiling, conflict_ceiling)
            violations.append(
                f"{outcome.metrics.conflict_count} separation conflict(s) detected — plan is operationally unsafe"
            )

        # Gate 2: Incomplete schedule
        if outcome.metrics.missing_assignments > 0:
            missing = outcome.metrics.missing_assignments
            # Each missing flight reduces ceiling further, minimum 0.20
            completeness_ceiling = max(0.20, 0.50 - 0.04 * missing)
            ceiling = min(ceiling, completeness_ceiling)
            violations.append(
                f"{missing} flight(s) unassigned — schedule is incomplete"
            )

        # Gate 3: Emergency flight delay violation — life-safety ceiling
        if outcome.metrics.emergency_violations > 0:
            ceiling = min(ceiling, 0.35)
            violations.append(
                "EMERGENCY flight delay tolerance exceeded — life-safety risk"
            )

        return ceiling, violations


# ── Layer 2: Priority Rubric ──────────────────────────────────────────────────

class PriorityRubricGrader:
    """Grades the quality of priority-aware sequencing decisions.

    Emergency protection is binary (0 or 1). Medical and connection handling
    are graded proportionally. Connection scoring uses the connection_risk field
    via the pre-computed connection_impact_score in metrics.
    """

    def grade(self, task: TaskDefinition, outcome: SimulationOutcome) -> float:
        metrics = outcome.metrics

        # Emergency: binary — any violation collapses this to 0.0
        emergency_count = sum(
            1 for f in task.flights if f.priority == PriorityClass.EMERGENCY
        )
        if emergency_count > 0:
            emergency_score = 0.0 if metrics.emergency_violations > 0 else 1.0
        else:
            emergency_score = 1.0  # No emergency flights = no penalty

        # Medical: partial credit proportional to violations
        medical_count = sum(
            1 for f in task.flights if f.priority == PriorityClass.MEDICAL
        )
        if medical_count > 0:
            medical_score = max(0.0, 1.0 - metrics.medical_violations / medical_count)
        else:
            medical_score = 1.0

        # Connection: uses the risk-weighted connection_impact_score from engine
        # (0.0 = all high-risk connections were heavily delayed; 1.0 = all protected)
        connection_score = metrics.connection_impact_score

        return 0.50 * emergency_score + 0.30 * medical_score + 0.20 * connection_score


# ── Layer 3: Efficiency Rubric ────────────────────────────────────────────────

class EfficiencyRubricGrader:
    """Grades the operational efficiency dimensions of the plan.

    Weights are domain-derived:
      delay        0.35 — total system delay is the primary efficiency metric (ICAO)
      fuel         0.25 — fuel burn directly impacts cost and environment
      fairness     0.20 — equitable delay distribution across airlines (IATA WSG)
      connection   0.20 — risk-weighted connection impact (previously unused field)
    """

    WEIGHTS: Dict[str, float] = {
        "delay": 0.35,
        "fuel": 0.25,
        "fairness": 0.20,
        "connection_impact": 0.20,
    }

    def grade(self, outcome: SimulationOutcome) -> float:
        m = outcome.metrics
        return (
            self.WEIGHTS["delay"] * m.delay_efficiency
            + self.WEIGHTS["fuel"] * m.fuel_efficiency
            + self.WEIGHTS["fairness"] * m.fairness
            + self.WEIGHTS["connection_impact"] * m.connection_impact_score
        )


# ── Composite Grader (Official Score) ────────────────────────────────────────

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
        """Return a score in the strict range (0.0, 1.0)."""


class GatedCompositeGrader(BaseTaskGrader):
    """Official deterministic benchmark score.

    Formula:
        raw   = 0.30 * priority_score + 0.70 * efficiency_score
        score = min(gate_ceiling, raw)

    Safety gates (Layer 1) apply a hard ceiling: a plan with conflicts or missing
    flights can never compensate with excellent efficiency scores. This prevents
    reward hacking and aligns with FAA GDP evaluation practice.
    """

    grader_name = "composite_task_grader"

    def __init__(self) -> None:
        self._safety_gate = SafetyGateEvaluator()
        self._priority = PriorityRubricGrader()
        self._efficiency = EfficiencyRubricGrader()

    def grade(
        self,
        task: TaskDefinition,
        outcome: SimulationOutcome,
        proposal: Iterable[SlotAssignment],
        rationale: str = "",
    ) -> TaskGrade:
        ceiling, gate_violations = self._safety_gate.evaluate(outcome, task)
        priority_score = self._priority.grade(task, outcome)
        efficiency_score = self._efficiency.grade(outcome)

        raw = 0.30 * priority_score + 0.70 * efficiency_score
        final = min(ceiling, max(0.0, raw))

        if gate_violations:
            status = f"GATED at {ceiling:.3f} — {'; '.join(gate_violations)}."
        elif final >= 0.85:
            status = "Plan accepted: safe, complete, and operationally balanced."
        elif final >= 0.55:
            status = "Plan accepted with operational weaknesses remaining."
        else:
            status = "Plan scored low on efficiency or priority dimensions."

        return TaskGrade(
            grader_name=self.grader_name,
            score=_strict_score(final),
            rationale=status,
            sub_scores={
                "gate_ceiling": round(ceiling, 4),
                "priority_score": round(priority_score, 4),
                "efficiency_score": round(efficiency_score, 4),
            },
        )


# ── Auxiliary LLM Grader (non-official) ─────────────────────────────────────

class LLMSupervisorGrader(BaseTaskGrader):
    """Optional LLM-backed supervisor used when credentials are available.

    This grader is AUXILIARY ONLY and does NOT contribute to the official
    composite score. It is preserved as a side-channel for analysis.
    """

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
                score=_strict_score(outcome.metrics.overall_score),
                rationale="LLM grader disabled; using deterministic fallback identical to operational score.",
                sub_scores={"fallback": _strict_score(outcome.metrics.overall_score)},
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
            "Score must be a float strictly between 0.0 and 1.0.\n\n"
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
                        "content": (
                            "Grade ATC recovery plans conservatively and output strict JSON only. "
                            "Score must be strictly greater than 0 and strictly less than 1."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw_text = (response.choices[0].message.content or "").strip()
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or start > end:
                raise ValueError("LLM grader response contains no valid JSON object")
            data = json.loads(raw_text[start: end + 1])
            if not isinstance(data, dict):
                raise ValueError("LLM grader payload must be a JSON object")
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            rationale_text = str(data.get("rationale", "LLM grader returned no rationale."))
        except (
            APIConnectionError,
            APITimeoutError,
            APIError,
            RateLimitError,
            OpenAIError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
            AttributeError,
            TypeError,
        ) as exc:
            score = outcome.metrics.overall_score
            rationale_text = f"LLM grading failed, reverted to deterministic score: {exc}"

        return TaskGrade(
            grader_name=self.grader_name,
            score=_strict_score(score),
            rationale=rationale_text,
            sub_scores={"operational_score": _strict_score(outcome.metrics.overall_score)},
        )


# ── Public Entry Point ────────────────────────────────────────────────────────

def grade_task(
    task: TaskDefinition,
    outcome: SimulationOutcome,
    proposal: Iterable[SlotAssignment],
    rationale: str = "",
) -> List[TaskGrade]:
    """Run all task graders and return their scores.

    Each sub-grader is instantiated once and called once. The composite grader
    does NOT internally re-run sub-graders (no duplicate computation).

    Order: [composite, llm_supervisor]
      - composite_task_grader: official benchmark score (deterministic, gated)
      - llm_supervisor:        auxiliary LLM signal (non-official)
    """
    proposal_list = proposal if isinstance(proposal, list) else list(proposal)
    composite = GatedCompositeGrader().grade(task, outcome, proposal_list, rationale)
    llm = LLMSupervisorGrader().grade(task, outcome, proposal_list, rationale)
    return [composite, llm]
