"""FastAPI application entrypoint for the ATC optimization environment."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

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


UI_TEMPLATE_PATH = Path(__file__).with_name("ui_console.html")


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
    return (
        template.replace("__MODEL_OPTIONS__", model_options)
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


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server directly."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
