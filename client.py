"""OpenEnv client for the ATC optimization environment."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        ATCOptimizationAction,
        ATCOptimizationObservation,
        ATCOptimizationState,
    )
except ImportError:
    from models import (
        ATCOptimizationAction,
        ATCOptimizationObservation,
        ATCOptimizationState,
    )


class ATCOptimizationEnv(
    EnvClient[ATCOptimizationAction, ATCOptimizationObservation, ATCOptimizationState]
):
    """Persistent client for the ATC optimization environment."""

    def _step_payload(self, action: ATCOptimizationAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[ATCOptimizationObservation]:
        obs_data = payload.get("observation", {})
        observation = ATCOptimizationObservation.model_validate(
            {
                **obs_data,
                "reward": payload.get("reward"),
                "done": payload.get("done", False),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ATCOptimizationState:
        return ATCOptimizationState.model_validate(payload)
