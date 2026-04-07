"""Typed data models for the ATC optimization environment."""

from __future__ import annotations

import json
from enum import Enum
from typing import Dict, List, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator


class OperationType(str, Enum):
    """Flight operation category."""

    ARRIVAL = "arrival"
    DEPARTURE = "departure"


class WakeClass(str, Enum):
    """Wake turbulence class used for separation logic."""

    LIGHT = "L"
    MEDIUM = "M"
    HEAVY = "H"


class PriorityClass(str, Enum):
    """Operational priority categories for dispatching."""

    NORMAL = "normal"
    CONNECTION = "connection"
    MEDICAL = "medical"
    EMERGENCY = "emergency"


class Difficulty(str, Enum):
    """Task difficulty level."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


DEFAULT_ALLOWED_OPERATIONS = (
    OperationType.ARRIVAL,
    OperationType.DEPARTURE,
)


def default_allowed_operations() -> List[OperationType]:
    return list(DEFAULT_ALLOWED_OPERATIONS)


class RunwaySpec(BaseModel):
    """Runway configuration available to the controller."""

    runway_id: str = Field(..., description="Runway identifier")
    allowed_operations: List[OperationType] = Field(
        default_factory=default_allowed_operations,
        description="Operations permitted on the runway",
    )
    hourly_capacity: int = Field(
        ..., ge=4, le=60, description="Maximum operations per hour"
    )
    weather_penalty: float = Field(
        default=1.0,
        ge=1.0,
        le=2.0,
        description="Multiplier applied to capacity spacing during disruption",
    )
    notes: str = Field(default="", description="Operational notes for the runway")


class FlightRecord(BaseModel):
    """Flight requiring a runway/slot decision."""

    flight_id: str = Field(..., description="Flight callsign or identifier")
    airline: str = Field(..., description="Airline code")
    operation: OperationType = Field(..., description="Arrival or departure")
    wake_class: WakeClass = Field(..., description="Wake turbulence category")
    scheduled_minute: int = Field(..., ge=0, description="Original planned slot")
    earliest_minute: int = Field(..., ge=0, description="Earliest feasible slot")
    latest_minute: int = Field(..., ge=0, description="Latest feasible slot")
    allowed_runways: List[str] = Field(..., description="Runways the flight may use")
    passengers: int = Field(..., ge=1, description="Passengers impacted")
    fuel_burn_per_minute: float = Field(
        ..., ge=1.0, description="Estimated fuel burn cost for airborne/ground delay"
    )
    priority: PriorityClass = Field(
        default=PriorityClass.NORMAL, description="Operational priority"
    )
    connection_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Severity of onward connection disruption",
    )
    notes: str = Field(default="", description="Operational nuance for the flight")


class SlotAssignment(BaseModel):
    """Controller proposal for a single flight."""

    flight_id: str = Field(..., description="Flight being assigned")
    runway: str = Field(..., description="Runway chosen for the flight")
    assigned_minute: int = Field(..., ge=0, description="Assigned slot minute")
    hold_minutes: int = Field(
        default=0,
        ge=0,
        description="Explicit holding/queueing amount declared by the agent",
    )


class TaskMetrics(BaseModel):
    """Normalized metrics produced by the simulator/grader."""

    schedule_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    conflict_free_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    priority_handling: float = Field(default=0.0, ge=0.0, le=1.0)
    delay_efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    fairness: float = Field(default=0.0, ge=0.0, le=1.0)
    fuel_efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    connection_impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
    agent_judgment: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_delay_minutes: int = Field(default=0, ge=0)
    max_delay_minutes: int = Field(default=0, ge=0)
    estimated_fuel_burn: float = Field(default=0.0, ge=0.0)
    conflict_count: int = Field(default=0, ge=0)
    capacity_violations: int = Field(default=0, ge=0)
    priority_violations: int = Field(default=0, ge=0)
    emergency_violations: int = Field(default=0, ge=0)
    medical_violations: int = Field(default=0, ge=0)
    connection_violations: int = Field(default=0, ge=0)
    missing_assignments: int = Field(default=0, ge=0)
    invalid_assignments: int = Field(default=0, ge=0)
    per_airline_average_delay: Dict[str, float] = Field(default_factory=dict)


class TaskDefinition(BaseModel):
    """Scenario definition shipped with the benchmark."""

    task_id: str
    title: str
    difficulty: Difficulty
    airport: str
    description: str
    objective: str
    grading_focus: List[str]
    planning_horizon_minutes: int = Field(..., ge=30)
    max_steps: int = Field(default=4, ge=1, le=8)
    delay_budget: int = Field(..., ge=30)
    fuel_budget: float = Field(..., ge=50.0)
    fairness_tolerance: float = Field(..., ge=1.0)
    runways: List[RunwaySpec]
    flights: List[FlightRecord]


class TaskGrade(BaseModel):
    """Outcome returned by a task grader."""

    grader_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    sub_scores: Dict[str, float] = Field(default_factory=dict)


class ATCOptimizationAction(Action):
    """Agent proposal for re-sequencing the disrupted traffic bank."""

    proposal: List[SlotAssignment] = Field(
        ...,
        description="Full candidate slot plan for all or some flights in the scenario",
    )
    rationale: str = Field(
        default="",
        description="Short explanation describing prioritization decisions",
    )
    commit: bool = Field(
        default=True,
        description="Whether to finish the episode after evaluating this plan",
    )

    @field_validator("proposal", mode="before")
    @classmethod
    def parse_proposal_json(cls, value):
        """Allow playground users to paste proposal JSON into a text box."""

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "proposal must be a JSON list of slot assignments"
                ) from exc
            return parsed
        return value


class ATCOptimizationObservation(Observation):
    """Observation returned after reset/step."""

    task_id: str = Field(default="", description="Task identifier")
    task_title: str = Field(default="", description="Human-readable task name")
    difficulty: Difficulty = Field(default=Difficulty.EASY)
    airport: str = Field(default="", description="Airport code")
    briefing: str = Field(default="", description="Text briefing for the agent")
    objective: str = Field(default="", description="Task goal summary")
    grading_focus: List[str] = Field(default_factory=list)
    flights: List[FlightRecord] = Field(default_factory=list)
    runways: List[RunwaySpec] = Field(default_factory=list)
    current_metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    best_metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    current_plan: List[SlotAssignment] = Field(default_factory=list)
    diagnostics: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    grader_feedback: List[str] = Field(default_factory=list)
    steps_remaining: int = Field(default=0, ge=0)


class PlanSnapshot(BaseModel):
    """Compact history item used inside state."""

    step: int = Field(..., ge=0)
    score: float = Field(..., ge=0.0, le=1.0)
    reward: float = Field(..., ge=-1.0, le=1.0)
    commit: bool = Field(default=True)


class ATCOptimizationState(State):
    """Serializable internal state for inspection through `/state`."""

    task_id: str = Field(default="", description="Current task id")
    task_title: str = Field(default="", description="Current task title")
    difficulty: Difficulty = Field(default=Difficulty.EASY)
    airport: str = Field(default="", description="Airport code")
    seed: int = Field(default=0, ge=0)
    max_steps: int = Field(default=0, ge=0)
    current_metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    best_metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    current_plan: List[SlotAssignment] = Field(default_factory=list)
    best_plan: List[SlotAssignment] = Field(default_factory=list)
    history: List[PlanSnapshot] = Field(default_factory=list)
    grader_history: List[TaskGrade] = Field(default_factory=list)
    final_summary: str = Field(default="", description="Latest summary for the task")
    active_task_ids: List[str] = Field(default_factory=list)
    mode: Literal["planning"] = Field(default="planning")
