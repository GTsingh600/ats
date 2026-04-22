"""OpenEnv-compliant client for the ATC multi-agent environment.

Usage (async):
    async with ATCEnvClient(base_url="http://localhost:8000") as env:
        result = await env.reset(episode_id="0", task_id="bengaluru_irrops_hard")
        obs = result.observation  # ATCObservation

        action = ATCAction(
            aman_completion=aman_llm_output,
            dman_completion=dman_llm_output,
            round_type="bid",
        )
        result = await env.step(action)
        if result.done:
            print(f"Episode done. Reward: {result.reward:.3f}")

Usage (sync, handy for scripts):
    env = ATCEnvClient(base_url="http://localhost:8000").sync()
    result = env.reset()
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from openenv.core import EnvClient, StepResult
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False

from .models import ATCAction, ATCObservation, ATCState


if _OPENENV_AVAILABLE:
    class ATCEnvClient(EnvClient[ATCAction, ATCObservation, ATCState]):
        """Thin OpenEnv client wrapping the ATC multi-agent server."""

        def _step_payload(self, action: ATCAction) -> Dict[str, Any]:
            return action.model_dump()

        def _parse_result(self, payload: Dict[str, Any]) -> "StepResult[ATCObservation]":
            obs = ATCObservation(**payload["observation"])
            return StepResult(
                observation=obs,
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: Dict[str, Any]) -> ATCState:
            return ATCState(**payload)

else:
    class ATCEnvClient:  # type: ignore[no-redef]
        """Stub when openenv is not installed — raises on use."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "openenv package is required for ATCEnvClient. "
                "Install with: pip install openenv"
            )
