"""FastAPI application entrypoint for the ATC optimization environment."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required to run this environment. Install dependencies first."
    ) from exc

try:
    from ..models import ATCOptimizationAction, ATCOptimizationObservation
    from . import ui_runner
    from .atc_environment import ATCOptimizationEnvironment
except ImportError:
    from models import ATCOptimizationAction, ATCOptimizationObservation
    from server import ui_runner
    from server.atc_environment import ATCOptimizationEnvironment

try:
    from ..multi_agent.environment import MultiAgentATCEnvironment
    from ..multi_agent.generator import ChallengeGenerator
    from ..multi_agent.supervisor import SupervisorAgent
    from ..multi_agent.inference import run_episode
    from ..multi_agent.models import (
        AMANAction, DMANAction, MultiAgentEpisodeResult, SupervisorProfileName,
    )
except ImportError:
    from multi_agent.environment import MultiAgentATCEnvironment
    from multi_agent.generator import ChallengeGenerator
    from multi_agent.supervisor import SupervisorAgent
    from multi_agent.inference import run_episode
    from multi_agent.models import (
        AMANAction, DMANAction, MultiAgentEpisodeResult, SupervisorProfileName,
    )

# Shared multi-agent singletons (one env per server process, serialised via lock)
_ma_env       = MultiAgentATCEnvironment(seed=0)
_ma_generator = ChallengeGenerator(seed=0)
_ma_supervisor = SupervisorAgent()
_MA_LOCK = asyncio.Lock()


UI_TEMPLATE_PATH = Path(__file__).with_name("ui_console.html")
PLANE_IMAGE_PATH = Path(__file__).with_name("ui_aircraft.png")


app = create_app(
    ATCOptimizationEnvironment,
    ATCOptimizationAction,
    ATCOptimizationObservation,
    env_name="atc_env",
    max_concurrent_envs=8,
)


def _render_task_strips() -> str:
    return "\n".join(
        (
            f'<article class="strip" data-task-id="{task["task_id"]}" data-state="ready">'
            f'<div class="strip-top"><span class="strip-title">{task["title"]}</span>'
            f'<span class="strip-difficulty {task["difficulty"].lower()}">{task["difficulty"]}</span></div>'
            f'<div class="strip-summary">{task["summary"]}</div>'
            f'<div class="strip-meta"><span>Baseline {task["random_baseline"]:.2f}</span>'
            '<span class="strip-score">Awaiting run</span></div>'
            "</article>"
        )
        for task in ui_runner.UI_TASKS
    )


def _render_index_html() -> str:
    template = UI_TEMPLATE_PATH.read_text(encoding="utf-8")
    model_options = "\n".join(
        f'<option value="{model}">{model}</option>'
        for model in ui_runner.MODEL_OPTIONS
    )
    plane_image_bytes = PLANE_IMAGE_PATH.read_bytes()
    plane_image_src = (
        "data:image/png;base64," + base64.b64encode(plane_image_bytes).decode("ascii")
    )
    return (
        template.replace("__MODEL_OPTIONS__", model_options)
        .replace("__PLANE_IMAGE_SRC__", plane_image_src)
        .replace("__TASK_DATA__", json.dumps(ui_runner.UI_TASKS))
        .replace("__TASK_STRIPS__", _render_task_strips())
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(_render_index_html())


@app.post("/ui/run-inference", include_in_schema=False)
async def run_inference_ui(
    payload: ui_runner.InferenceRunRequest,
) -> Dict[str, Any]:
    """Run user-triggered inference from the Space UI."""

    async with ui_runner.INFERENCE_LOCK:
        try:
            return await asyncio.to_thread(ui_runner.run_requested_inference, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HTTPException(
                status_code=500,
                detail=f"Inference run failed: {exc}",
            ) from exc


# ── Multi-agent REST endpoints ─────────────────────────────────────────────────

class MAResetRequest(BaseModel):
    task_id: str
    episode_id: int = 0
    supervisor_profile: Optional[str] = None


class MABidRequest(BaseModel):
    aman_action: Dict[str, Any]
    dman_action: Dict[str, Any]


class MAEpisodeRequest(BaseModel):
    task_id: str
    episode_id: int = 0
    use_generator: bool = True
    use_heuristic: bool = True


@app.post("/multi_agent/reset", tags=["multi-agent"])
async def ma_reset(req: MAResetRequest) -> Dict[str, Any]:
    """Reset multi-agent environment, return AMAN and DMAN observations."""
    async with _MA_LOCK:
        try:
            profile = (
                SupervisorProfileName(req.supervisor_profile)
                if req.supervisor_profile
                else None
            )
            aman_obs, dman_obs = await asyncio.to_thread(
                _ma_env.reset,
                task_id=req.task_id,
                episode_id=req.episode_id,
                supervisor_profile=profile,
            )
            return {
                "aman_observation": json.loads(aman_obs.model_dump_json()),
                "dman_observation": json.loads(dman_obs.model_dump_json()),
                "atfm_deadlines": _ma_env._state.atfm_deadlines,
                "round_number": _ma_env._state.round_number,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/multi_agent/step/bid", tags=["multi-agent"])
async def ma_step_bid(req: MABidRequest) -> Dict[str, Any]:
    """Submit BID-round actions from both agents."""
    async with _MA_LOCK:
        try:
            aman_action = AMANAction.model_validate(req.aman_action)
            dman_action  = DMANAction.model_validate(req.dman_action)
            aman_obs2, dman_obs2, reward, done = await asyncio.to_thread(
                _ma_env.step_bid, aman_action, dman_action
            )
            return {
                "aman_observation": json.loads(aman_obs2.model_dump_json()),
                "dman_observation": json.loads(dman_obs2.model_dump_json()),
                "partial_reward": reward,
                "done": done,
                "round_number": _ma_env._state.round_number,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/multi_agent/finalize", tags=["multi-agent"])
async def ma_finalize() -> Dict[str, Any]:
    """Finalize episode, return full scored result."""
    async with _MA_LOCK:
        try:
            result: MultiAgentEpisodeResult = await asyncio.to_thread(_ma_env.finalize)
            return json.loads(result.model_dump_json())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/multi_agent/episode", tags=["multi-agent"])
async def ma_run_episode(req: MAEpisodeRequest) -> Dict[str, Any]:
    """Run a full multi-agent episode with heuristic or LLM agents."""
    async with _MA_LOCK:
        try:
            result = await asyncio.to_thread(
                run_episode,
                task_id=req.task_id,
                client=None,
                env=_ma_env,
                generator=_ma_generator if req.use_generator else None,
                supervisor=_ma_supervisor,
                episode_id=req.episode_id,
                use_generator=req.use_generator,
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/multi_agent/profiles", tags=["multi-agent"])
async def ma_profiles() -> List[str]:
    """List available supervisor preference profiles."""
    return [p.value for p in SupervisorProfileName]


@app.get("/multi_agent/status", tags=["multi-agent"])
async def ma_status() -> Dict[str, Any]:
    """Current multi-agent environment state summary."""
    state = _ma_env._state
    if state is None:
        return {"initialized": False}
    return {
        "initialized": True,
        "task_id": state.task.task_id if state.task else None,
        "round_number": state.round_number,
        "aman_slots_count": len(state.aman_slots),
        "dman_slots_count": len(state.dman_slots),
        "generator_difficulty": _ma_generator.difficulty_level,
        "supervisor_profile": (
            state.supervisor_profile.value if state.supervisor_profile else None
        ),
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server directly."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
