"""Helpers for the browser-driven inference console."""

from __future__ import annotations

import asyncio
import io
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from .. import inference as inference_runner
    from .atc_environment import ATCOptimizationEnvironment
except ImportError:
    import inference as inference_runner

    from server.atc_environment import ATCOptimizationEnvironment


UI_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "delhi_monsoon_recovery_easy",
        "title": "Delhi Monsoon Recovery",
        "difficulty": "Easy",
        "summary": "10 flights | 2 runways | weather disruption",
        "random_baseline": 0.21,
    },
    {
        "task_id": "mumbai_bank_balance_medium",
        "title": "Mumbai Hub Bank Balance",
        "difficulty": "Medium",
        "summary": "13 flights | 2 runways | airline equity",
        "random_baseline": 0.18,
    },
    {
        "task_id": "bengaluru_irrops_hard",
        "title": "Bengaluru IRROPS Recovery",
        "difficulty": "Hard",
        "summary": "17 flights | 2 runways | emergency priority",
        "random_baseline": 0.12,
    },
    {
        "task_id": "hyderabad_cargo_crunch_medium_hard",
        "title": "Hyderabad Cargo Crunch",
        "difficulty": "Hard",
        "summary": "7 flights | 1 runway | capacity constraint",
        "random_baseline": 0.15,
    },
]

MODEL_OPTIONS = [
    "heuristic-baseline",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "meta-llama/Llama-3.1-8B-Instruct",
]

AVERAGE_RANDOM_BASELINE = round(
    sum(float(task["random_baseline"]) for task in UI_TASKS) / len(UI_TASKS), 4
)
INFERENCE_LOCK = asyncio.Lock()


class InferenceRunRequest(BaseModel):
    """Payload accepted by the console inference endpoint."""

    hf_token: str = Field(default="")
    model_name: str = Field(default="heuristic-baseline")
    api_base_url: str = Field(default_factory=lambda: inference_runner.API_BASE_URL)


def _parse_log_fields(log_line: str) -> Dict[str, str]:
    """Parse key=value tokens from the structured inference logs."""

    fields: Dict[str, str] = {}
    for token in log_line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def _task_outcome(score: float) -> str:
    if score >= 0.9:
        return "Clean landing"
    if score >= 0.75:
        return "Controlled approach"
    if score >= 0.55:
        return "Holding pattern"
    if score >= 0.35:
        return "Turbulence warning"
    return "Loss of control"


def _scene_for_run(avg_score: float, min_score: float, had_stderr: bool) -> Dict[str, str]:
    if had_stderr or min_score < 0.35 or avg_score < 0.45:
        return {
            "scene": "crash",
            "scene_title": "Loss of control",
            "scene_caption": (
                "The run broke down badly enough to trigger the crash case. "
                "Inspect the console log and task strips before trusting the output."
            ),
        }
    if min_score < 0.55 or avg_score < 0.62:
        return {
            "scene": "warning",
            "scene_title": "Turbulence warning",
            "scene_caption": (
                "The model stayed airborne but made enough mistakes that the aircraft "
                "is flying unstable and needs immediate correction."
            ),
        }
    if avg_score < 0.8:
        return {
            "scene": "holding",
            "scene_title": "Holding pattern",
            "scene_caption": (
                "The plan is workable but hesitant. Expect extra vectoring, longer holds, "
                "and lower confidence on the harder tasks."
            ),
        }
    if avg_score < 0.92:
        return {
            "scene": "approach",
            "scene_title": "Controlled approach",
            "scene_caption": (
                "The model is lining up well. This is a stable approach with a few rough "
                "edges, not yet the best possible landing."
            ),
        }
    return {
        "scene": "landing",
        "scene_title": "Smooth landing",
        "scene_caption": (
            "The run stayed composed from start to finish. The aircraft touches down cleanly "
            "with strong task scores and low operational drama."
        ),
    }


def _run_single_task(task: Dict[str, Any], client: Optional[Any]) -> Dict[str, Any]:
    """Execute one task using the same planner logic as inference.py."""

    task_id = str(task["task_id"])
    env = ATCOptimizationEnvironment()
    rewards: List[float] = []
    steps_used = 0
    score = float(inference_runner.SCORE_EPSILON)
    success = False
    runtime_model = (
        inference_runner.MODEL_NAME if client is not None else "heuristic-baseline"
    )

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        inference_runner.log_start(
            task=task_id,
            env=inference_runner.BENCHMARK,
            model=runtime_model,
        )
        try:
            observation = env.reset(task_id=task_id)
            max_steps = inference_runner._step_budget(observation.steps_remaining)

            for step_index in range(max_steps):
                if observation.done:
                    break
                step_num = step_index + 1
                action = inference_runner.get_model_action(
                    client=client,
                    observation=observation,
                    task_id=task_id,
                    step=step_num,
                )
                observation = env.step(action)
                reward = float(observation.reward or 0.0)
                rewards.append(reward)
                steps_used = step_num
                inference_runner.log_step(
                    step=step_num,
                    action=(
                        f"submit_plan(count={len(action.proposal)},"
                        f"commit={inference_runner._bool_token(action.commit)})"
                    ),
                    reward=reward,
                    done=observation.done,
                    error=None,
                )
                if observation.done:
                    break

            score = max(
                inference_runner.SCORE_EPSILON,
                min(
                    1.0 - inference_runner.SCORE_EPSILON,
                    observation.current_metrics.overall_score,
                ),
            )
            success = score >= inference_runner.SUCCESS_SCORE_THRESHOLD
        except Exception as exc:  # pragma: no cover - defensive guard
            inference_runner._safe_stderr(f"Task execution failed for {task_id}: {exc}")
        finally:
            inference_runner.log_end(
                task=task_id,
                success=success,
                steps=steps_used,
                score=score,
                rewards=rewards,
            )

    stdout_lines = [
        line.strip() for line in stdout_buffer.getvalue().splitlines() if line.strip()
    ]
    stderr_lines = [
        line.strip() for line in stderr_buffer.getvalue().splitlines() if line.strip()
    ]
    end_fields = _parse_log_fields(stdout_lines[-1]) if stdout_lines else {}

    return {
        "task_id": task_id,
        "title": task["title"],
        "difficulty": task["difficulty"],
        "summary": task["summary"],
        "random_baseline": float(task["random_baseline"]),
        "score": round(float(score), 4),
        "improvement": round(float(score) - float(task["random_baseline"]), 4),
        "steps_used": int(end_fields.get("steps", steps_used or 0)),
        "success": bool(success),
        "outcome": _task_outcome(float(score)),
        "logs": stdout_lines,
        "stderr": stderr_lines,
    }


def run_requested_inference(payload: InferenceRunRequest) -> Dict[str, Any]:
    """Run the selected model end to end for the browser UI."""

    model_name = payload.model_name.strip() or "heuristic-baseline"
    hf_token = payload.hf_token.strip()
    api_base_url = payload.api_base_url.strip().rstrip("/") or inference_runner.API_BASE_URL

    if model_name != "heuristic-baseline" and not hf_token:
        raise ValueError("HF token is required for hosted model inference.")

    previous_model = inference_runner.MODEL_NAME
    previous_token = inference_runner.HF_TOKEN
    previous_api_base = inference_runner.API_BASE_URL
    client: Optional[Any] = None

    started_at = time.perf_counter()
    try:
        inference_runner.MODEL_NAME = model_name
        inference_runner.HF_TOKEN = hf_token
        inference_runner.API_BASE_URL = api_base_url

        if model_name != "heuristic-baseline" and hf_token:
            client = inference_runner.OpenAI(
                base_url=api_base_url,
                api_key=hf_token,
                timeout=180.0,
            )

        task_results = [_run_single_task(task, client) for task in UI_TASKS]
    finally:
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:
                pass
        inference_runner.MODEL_NAME = previous_model
        inference_runner.HF_TOKEN = previous_token
        inference_runner.API_BASE_URL = previous_api_base

    total_runtime_seconds = round(time.perf_counter() - started_at, 2)
    average_agent_score = round(
        sum(float(task["score"]) for task in task_results) / len(task_results), 4
    )
    min_score = min(float(task["score"]) for task in task_results)
    had_stderr = any(task["stderr"] for task in task_results)
    scene = _scene_for_run(average_agent_score, min_score, had_stderr)

    return {
        "model": model_name,
        "api_base_url": api_base_url,
        "total_runtime_seconds": total_runtime_seconds,
        "average_agent_score": average_agent_score,
        "average_random_baseline": AVERAGE_RANDOM_BASELINE,
        "average_improvement": round(
            average_agent_score - AVERAGE_RANDOM_BASELINE, 4
        ),
        "task_count": len(task_results),
        "tasks": task_results,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        **scene,
    }
