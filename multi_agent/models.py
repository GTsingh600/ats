"""Multi-agent data models for AMAN/DMAN coordination environment.

Architecture:
  AMAN  — Arrival Manager:  owns arrival sequencing, partial view (arrivals only)
  DMAN  — Departure Manager: owns departure sequencing, partial view (departures only)
  GENERATOR — Adversarial self-play: mutates tasks to break coordination
  SUPERVISOR — Snorkel AI expert: changing preferences each episode

Negotiation protocol (3 rounds):
  Round 0 BID:       each agent submits independent plan for own flights
  Round 1 NEGOTIATE: conflict list broadcast; agents revise + send messages
  Round 2 FINAL:     merged plan graded; per-agent rewards computed
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from ..models import FlightRecord, PriorityClass, RunwaySpec, SlotAssignment, TaskMetrics
except ImportError:
    from models import FlightRecord, PriorityClass, RunwaySpec, SlotAssignment, TaskMetrics


# ── Agent roles ──────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    AMAN = "AMAN"
    DMAN = "DMAN"
    GENERATOR = "GENERATOR"
    SUPERVISOR = "SUPERVISOR"


# ── Negotiation messaging ─────────────────────────────────────────────────────

class MessageType(str, Enum):
    RUNWAY_CLAIM    = "runway_claim"       # "I need runway X at minute Y"
    EMERGENCY_BCAST = "emergency_broadcast" # "I have EMERGENCY — yield now"
    YIELD           = "yield"              # "I yield slot Y on runway X"
    ACKNOWLEDGE     = "acknowledge"        # "Received, adjusting"
    REQUEST_GAP     = "request_gap"        # "Leave gap at T+N on runway X"
    CONFLICT_FLAG   = "conflict_flag"      # environment injects this


class SlotProposal(BaseModel):
    """An alternative slot a negotiating agent is willing to accept."""

    flight_id: str
    runway_id: str
    minute: int
    delay_cost: float = Field(
        default=0.0,
        description="Self-reported cost of accepting this alternative (minutes of delay)",
    )


class NegotiationMessage(BaseModel):
    from_role:       AgentRole
    message_type:    MessageType
    flight_id:       str
    requested_minute: int
    runway_id:       str
    priority:        PriorityClass
    reason:          str
    is_emergency:    bool = False
    # Richer negotiation fields — enable more expressive coordination
    content: str = Field(
        default="",
        description="Free-form natural language explanation beyond the reason field",
    )
    proposed_alternatives: List[SlotProposal] = Field(
        default_factory=list,
        description="Alternative slots this agent is willing to accept instead",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's certainty that this request/yield is correct",
    )
    theory_of_mind_claim: str = Field(
        default="",
        description="Agent's explicit prediction about the other agent's constraint or intent",
    )


# ── Agent actions ─────────────────────────────────────────────────────────────

class AMANAction(BaseModel):
    """Arrival Manager's slot proposal — arrivals only."""

    arrival_slots:       List[SlotAssignment] = Field(default_factory=list)
    rationale:           str = Field(default="")
    emergency_yields:    List[str] = Field(default_factory=list,
                             description="Flight IDs yielded for incoming emergencies")
    outgoing_messages:   List[NegotiationMessage] = Field(default_factory=list)
    commit:              bool = False


class DMANAction(BaseModel):
    """Departure Manager's slot proposal — departures only."""

    departure_slots:     List[SlotAssignment] = Field(default_factory=list)
    rationale:           str = Field(default="")
    atfm_compliance:     Dict[str, int] = Field(default_factory=dict,
                             description="flight_id -> ATFM deadline minute respected")
    emergency_broadcasts: List[str] = Field(default_factory=list,
                             description="Fuel/medical emergency departure IDs broadcast to AMAN")
    outgoing_messages:   List[NegotiationMessage] = Field(default_factory=list)
    commit:              bool = False


# ── Generator actions ─────────────────────────────────────────────────────────

class MutationType(str, Enum):
    TIGHTEN_WINDOW          = "tighten_window"          # narrow earliest/latest window
    INJECT_EMERGENCY        = "inject_emergency"         # add EMERGENCY/MEDICAL flight
    INCREASE_WEATHER_PENALTY = "increase_weather_penalty" # degrade runway capacity
    ADD_ATFM_DEADLINE       = "add_atfm_deadline"        # hard network slot constraint
    CLOSE_RUNWAY_WINDOW     = "close_runway_window"      # runway unavailable T1→T2
    ADD_CONFLICTING_FLIGHT  = "add_conflicting_flight"   # inject flight that forces wake conflict


class GeneratorMutation(BaseModel):
    mutation_type:    MutationType
    target_flight_id: Optional[str] = None
    target_runway_id: Optional[str] = None
    params:           Dict[str, Any] = Field(default_factory=dict)
    rationale:        str = ""


class GeneratorAction(BaseModel):
    """Self-play generator produces a batch of mutations per episode."""

    mutations: List[GeneratorMutation] = Field(default_factory=list,
                   description="Ordered list of mutations to apply to base task")
    strategy:  str = Field(default="",
                   description="High-level explanation of how this breaks coordination")


# ── Supervisor / Snorkel AI ───────────────────────────────────────────────────

class SupervisorProfileName(str, Enum):
    SAFETY_STRICT    = "safety_strict"
    THROUGHPUT_MAX   = "throughput_max"
    FUEL_PRIORITY    = "fuel_priority"
    EMERGENCY_FOCUS  = "emergency_focus"
    AIRLINE_FAIRNESS = "airline_fairness"


SUPERVISOR_PROFILES: Dict[str, Dict[str, Any]] = {
    SupervisorProfileName.SAFETY_STRICT: {
        "description": (
            "Zero tolerance for separation conflicts today. Delay is acceptable — "
            "a conflict is not. Prioritize safety margins above all throughput goals."
        ),
        "conflict_weight": 3.0,
        "delay_weight":    0.3,
        "fuel_weight":     0.3,
        "fairness_weight": 0.5,
        "priority_weight": 1.5,
    },
    SupervisorProfileName.THROUGHPUT_MAX: {
        "description": (
            "Pack the runway. I need maximum aircraft-per-hour today. "
            "Minor spacing risks are acceptable; just keep the flow moving."
        ),
        "conflict_weight": 0.8,
        "delay_weight":    1.8,
        "fuel_weight":     0.5,
        "fairness_weight": 0.5,
        "priority_weight": 1.0,
    },
    SupervisorProfileName.FUEL_PRIORITY: {
        "description": (
            "Fuel costs are critical. Every airborne hold is money burning. "
            "Minimize all unnecessary holding — arrivals and departures both."
        ),
        "conflict_weight": 1.2,
        "delay_weight":    0.6,
        "fuel_weight":     2.5,
        "fairness_weight": 0.5,
        "priority_weight": 1.0,
    },
    SupervisorProfileName.EMERGENCY_FOCUS: {
        "description": (
            "Medical and emergency flights are the only thing that matters today. "
            "Everything else waits if needed. No emergency flight may be delayed."
        ),
        "conflict_weight": 1.5,
        "delay_weight":    0.4,
        "fuel_weight":     0.4,
        "fairness_weight": 0.3,
        "priority_weight": 3.5,
    },
    SupervisorProfileName.AIRLINE_FAIRNESS: {
        "description": (
            "All airlines must be treated equally today. I don't care if IndiGo "
            "is bigger — fairness across carriers is the metric I'm watching."
        ),
        "conflict_weight": 1.2,
        "delay_weight":    0.8,
        "fuel_weight":     0.8,
        "fairness_weight": 2.5,
        "priority_weight": 1.0,
    },
}


# ── Round / episode state ─────────────────────────────────────────────────────

class RoundType(str, Enum):
    BID       = "bid"        # Round 0: independent proposals
    NEGOTIATE = "negotiate"  # Round 1: conflict resolution
    FINAL     = "final"      # Round 2: commit and grade


class MultiAgentObservation(BaseModel):
    """Role-specific observation injected into the LLM system prompt."""

    task_id:                  str
    airport:                  str
    briefing:                 str
    role:                     AgentRole
    my_flights:               List[FlightRecord]   # filtered to agent's operation type
    all_runways:              List[RunwaySpec]
    supervisor_profile_name:  SupervisorProfileName
    supervisor_description:   str
    atfm_deadlines:           Dict[str, int] = Field(default_factory=dict,
                                  description="flight_id -> hard deadline minute from ATFM")
    incoming_messages:        List[NegotiationMessage] = Field(default_factory=list)
    conflict_log:             List[str] = Field(default_factory=list)
    current_metrics:          Optional[TaskMetrics] = None
    steps_remaining:          int = 0
    round_type:               RoundType = RoundType.BID
    round_number:             int = 0
    reward:                   Optional[float] = None
    done:                     bool = False

    def to_prompt_text(self) -> str:
        """Render observation as structured text for LLM input."""
        prof = SUPERVISOR_PROFILES[self.supervisor_profile_name]
        weight_str = (
            f"conflict={prof['conflict_weight']:.1f}x  "
            f"delay={prof['delay_weight']:.1f}x  "
            f"fuel={prof['fuel_weight']:.1f}x  "
            f"fairness={prof['fairness_weight']:.1f}x  "
            f"priority={prof['priority_weight']:.1f}x"
        )
        lines = [
            f"=== {self.role.value} OBSERVATION — {self.airport} ===",
            f"Task: {self.task_id}",
            f"Round: {self.round_type.value.upper()} ({self.round_number}/2)",
            f"Steps remaining: {self.steps_remaining}",
            "",
            f"SUPERVISOR TODAY: {self.supervisor_description}",
            f"SUPERVISOR WEIGHTS: {weight_str}",
            "",
        ]

        if self.atfm_deadlines:
            lines.append("ATFM HARD DEADLINES (network slots — cannot miss):")
            for fid, minute in self.atfm_deadlines.items():
                lines.append(f"  {fid}: depart/land by minute {minute}")
            lines.append("")

        lines.append(f"YOUR FLIGHTS ({len(self.my_flights)} total):")
        for f in self.my_flights:
            lines.append(
                f"  {f.flight_id} | {f.airline} | {f.operation.value.upper()} | "
                f"wake={f.wake_class.value} | priority={f.priority.value} | "
                f"window=[{f.earliest_minute},{f.latest_minute}] | "
                f"scheduled={f.scheduled_minute} | pax={f.passengers} | "
                f"burn={f.fuel_burn_per_minute}/min | runways={f.allowed_runways}"
                + (f" | NOTE: {f.notes}" if f.notes else "")
            )

        lines.append("")
        lines.append("RUNWAYS:")
        for rwy in self.all_runways:
            ops = "/".join(o.value for o in rwy.allowed_operations)
            lines.append(
                f"  {rwy.runway_id}: {ops} | capacity={rwy.hourly_capacity}/hr | "
                f"weather_penalty={rwy.weather_penalty}x"
                + (f" | {rwy.notes}" if rwy.notes else "")
            )

        if self.incoming_messages:
            lines.append("")
            lines.append("MESSAGES FROM OTHER AGENT:")
            for msg in self.incoming_messages:
                tag = " [EMERGENCY]" if msg.is_emergency else ""
                conf_str = f" conf={msg.confidence:.1f}" if msg.confidence < 1.0 else ""
                lines.append(
                    f"  [{msg.from_role.value} → {msg.message_type.value}]{tag}{conf_str} "
                    f"{msg.flight_id} @ runway {msg.runway_id} T+{msg.requested_minute} "
                    f"({msg.priority.value}): {msg.reason}"
                )
                if msg.content:
                    lines.append(f"    Detail: {msg.content}")
                if msg.theory_of_mind_claim:
                    lines.append(f"    Prediction: {msg.theory_of_mind_claim}")
                if msg.proposed_alternatives:
                    alts = ", ".join(
                        f"{a.flight_id}@{a.runway_id}+{a.minute}min(cost={a.delay_cost:.0f})"
                        for a in msg.proposed_alternatives
                    )
                    lines.append(f"    Alternatives offered: {alts}")

        if self.conflict_log:
            lines.append("")
            lines.append("CONFLICTS DETECTED (must resolve):")
            for c in self.conflict_log:
                lines.append(f"  ⚠ {c}")

        if self.current_metrics:
            lines.append("")
            lines.append(
                f"LAST SCORE: {self.current_metrics.overall_score:.3f} | "
                f"conflicts={self.current_metrics.conflict_count} | "
                f"delay={self.current_metrics.total_delay_minutes}min"
            )

        return "\n".join(lines)


# ── Per-role metrics ──────────────────────────────────────────────────────────

class PerRoleMetrics(BaseModel):
    """Metrics decomposed by AMAN / DMAN responsibility."""

    # AMAN
    arrival_count:             int   = 0
    arrival_delay_total:       int   = 0
    arrival_delay_mean:        float = 0.0
    arrivals_missing:          int   = 0
    emergency_arrivals_ok:     int   = 0
    emergency_arrivals_missed: int   = 0

    # DMAN
    departure_count:           int   = 0
    departure_delay_total:     int   = 0
    departure_delay_mean:      float = 0.0
    departures_missing:        int   = 0
    atfm_violations:           int   = 0
    atfm_compliant:            int   = 0
    emergency_departures_ok:   int   = 0
    emergency_departures_missed: int = 0

    # Cross-agent
    cross_lane_conflicts:      int   = 0   # conflicts between AMAN and DMAN slots
    intra_lane_conflicts:      int   = 0   # conflicts within one agent's slots
    negotiation_rounds_used:   int   = 0
    coordination_score:        float = 0.0  # 0→1 quality of negotiation


# ── Episode result ────────────────────────────────────────────────────────────

class MultiAgentEpisodeResult(BaseModel):
    """Full episode result used by training loop."""

    task_id:             str
    supervisor_profile:  SupervisorProfileName
    composite_score:     float
    aman_reward:         float
    dman_reward:         float
    generator_reward:    float
    supervisor_score:    float
    per_role:            PerRoleMetrics
    negotiation_rounds:  int
    scenario_solvable:   bool = True
