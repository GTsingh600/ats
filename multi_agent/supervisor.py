"""Snorkel AI Simulated Expert Supervisor.

Each episode the supervisor draws a preference profile (safety / throughput /
fuel / emergency / fairness). The LLM agents must read the stated preference
and optimise toward it — training theory-of-mind and instruction-following
simultaneously.

Supervisor reward is independent of the composite grader and is used only
for the supervisor-alignment component of per-agent rewards.
"""

from __future__ import annotations

from typing import Dict, List

try:
    from ..engine import simulate_plan, SimulationOutcome
    from ..models import OperationType, PriorityClass, TaskDefinition
    from .models import (
        SUPERVISOR_PROFILES,
        PerRoleMetrics,
        SupervisorProfileName,
    )
except ImportError:
    from engine import simulate_plan, SimulationOutcome
    from models import OperationType, PriorityClass, TaskDefinition
    from multi_agent.models import (
        SUPERVISOR_PROFILES,
        PerRoleMetrics,
        SupervisorProfileName,
    )


_PROFILE_CYCLE: List[SupervisorProfileName] = list(SupervisorProfileName)


class SupervisorAgent:
    """Simulates a real ATC supervisor with episodically changing preferences.

    Key behaviours:
      - Deterministic profile rotation (reproducible training runs)
      - Profile-weighted scoring replaces fixed SCORE_WEIGHTS for this episode
      - Natural language description injected into every agent's system prompt
      - Supervisor score returned as auxiliary signal (not part of composite)
    """

    def sample_profile(self, episode_id: int) -> SupervisorProfileName:
        """Deterministic round-robin through profiles for reproducibility."""
        return _PROFILE_CYCLE[episode_id % len(_PROFILE_CYCLE)]

    def get_description(self, profile_name: SupervisorProfileName) -> str:
        return SUPERVISOR_PROFILES[profile_name]["description"]

    def score_plan(
        self,
        outcome: SimulationOutcome,
        task: TaskDefinition,
        profile_name: SupervisorProfileName,
    ) -> float:
        """Score final plan against supervisor's current preference profile.

        Uses profile weight overrides applied to the raw simulation metrics.
        Returns normalised score in [0, 1].
        """
        profile = SUPERVISOR_PROFILES[profile_name]
        metrics = outcome.metrics

        w_conflict  = profile.get("conflict_weight",  1.0)
        w_delay     = profile.get("delay_weight",     1.0)
        w_fuel      = profile.get("fuel_weight",      1.0)
        w_fairness  = profile.get("fairness_weight",  1.0)
        w_priority  = profile.get("priority_weight",  1.0)

        total_w = w_conflict + w_delay + w_fuel + w_fairness + w_priority

        raw = (
            w_conflict * metrics.conflict_free_ratio
            + w_delay  * metrics.delay_efficiency
            + w_fuel   * metrics.fuel_efficiency
            + w_fairness * metrics.fairness
            + w_priority * metrics.priority_handling
        ) / total_w

        # Emergency focus profile: hard penalty if any emergency missed
        if profile_name == SupervisorProfileName.EMERGENCY_FOCUS:
            if metrics.emergency_violations > 0 or metrics.medical_violations > 0:
                raw *= 0.3  # severe penalty — supervisor is watching emergency flights only

        # Safety strict: any conflict = cap at 0.4
        if profile_name == SupervisorProfileName.SAFETY_STRICT:
            if metrics.conflict_count > 0:
                raw = min(raw, 0.4)

        return round(max(0.0, min(1.0, raw)), 4)

    def build_system_suffix(self, profile_name: SupervisorProfileName) -> str:
        """Extra system prompt text reminding agent of supervisor preference."""
        desc = self.get_description(profile_name)
        return (
            f"\n\nSUPERVISOR PREFERENCE THIS EPISODE:\n"
            f"  {desc}\n"
            f"Your plan will be scored against this preference. "
            f"Explicitly state in your rationale how you are satisfying it."
        )
