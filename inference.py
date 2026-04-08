"""Baseline inference runner with strict structured logs."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, OpenAIError, RateLimitError

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client import ATCOptimizationEnv
from models import ATCOptimizationAction
from planner import build_heuristic_plan, build_refined_plan


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", os.getenv("IMAGE_NAME", ""))
BENCHMARK = "atc_optimization_openenv"
TASK_IDS = [
    "delhi_monsoon_recovery_easy",
    "mumbai_bank_balance_medium",
    "bengaluru_irrops_hard",
]
SUCCESS_SCORE_THRESHOLD = 0.65
SCORE_EPSILON = 1e-4
MAX_STEPS_CAP = int(os.getenv("MAX_STEPS_CAP", "4"))
MAX_TOKENS = 1400
TEMPERATURE = 0


def _bool_token(value: bool) -> str:
    return "true" if value else "false"


def _step_budget(steps_remaining: int) -> int:
    return min(MAX_STEPS_CAP, max(1, int(steps_remaining)))


def _safe_print(msg: str) -> None:
    """Print to stdout, silently ignoring broken pipe errors."""
    try:
        print(msg, flush=True)
    except (BrokenPipeError, OSError):
        pass


def log_start(task: str, env: str, model: str) -> None:
    _safe_print(f"[START] task={task} env={env} model={model}")


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_text = "null" if error is None else error.replace("\n", " ").strip()
    _safe_print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={_bool_token(done)} error={error_text}"
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{item:.2f}" for item in rewards)
    _safe_print(
        f"[END] success={_bool_token(success)} steps={steps} score={score:.2f} rewards={rewards_text}"
    )


async def wait_for_server(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Optional[str] = None
    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.time() < deadline:
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(0.5)
    suffix = f" (last_error={last_error})" if last_error else ""
    raise RuntimeError(f"Timed out waiting for server health at {base_url}{suffix}")


async def prepare_base_url() -> Tuple[str, Optional[subprocess.Popen]]:
    for env_name in ("ENV_BASE_URL", "SPACE_URL", "PING_URL"):
        if os.getenv(env_name):
            return os.environ[env_name].rstrip("/"), None

    port = int(os.getenv("LOCAL_PORT", "8000"))
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        await wait_for_server(base_url)
        return base_url, process
    except RuntimeError:
        process.terminate()
        raise


def build_seed_plan(observation, step: int):
    if not observation.flights:
        raise ValueError("observation contains no flights")
    if observation.steps_remaining < 0:
        raise ValueError("observation.steps_remaining cannot be negative")
    if step <= 1 or not observation.current_plan:
        return build_heuristic_plan(observation)
    return build_refined_plan(observation, seed_plan=list(observation.current_plan))


def _extract_json_object(response_text: str) -> dict:
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start == -1 or end == -1:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"response is not JSON: {response_text[:120]}") from exc
    else:
        payload = json.loads(response_text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError(f"response payload is not an object: {type(payload)}")
    return payload


def get_model_action(client: Optional[OpenAI], observation, task_id: str, step: int) -> ATCOptimizationAction:
    seed_plan = build_seed_plan(observation, step)
    seed_json = json.dumps([item.model_dump() for item in seed_plan], ensure_ascii=True)
    should_commit = step >= _step_budget(observation.steps_remaining)

    if client is None or not API_BASE_URL or not HF_TOKEN or MODEL_NAME == "heuristic-baseline":
        return ATCOptimizationAction(
            proposal=seed_plan,
            rationale=(
                "Deterministic multi-step baseline used because no model endpoint is configured. "
                f"Planning step {step} {'refines the prior plan' if step > 1 else 'establishes a safe initial schedule'}."
            ),
            commit=should_commit,
        )

    prompt = (
        "You are optimizing a disrupted ATC runway schedule.\n"
        "Return strict JSON only with keys rationale and proposal.\n"
        "proposal must be an array of objects with flight_id, runway, assigned_minute, hold_minutes.\n"
        "Do not omit any flight. Keep assignments conflict-free and priority aware.\n\n"
        f"Task: {task_id}\n"
        f"Planning step: {step}\n"
        f"Briefing:\n{observation.briefing}\n\n"
        f"Current metrics: {observation.current_metrics.model_dump_json()}\n"
        f"Diagnostics: {json.dumps(observation.diagnostics)}\n"
        f"Recommendations: {json.dumps(observation.recommendations)}\n"
        f"Seed candidate plan:\n{seed_json}\n"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative ATC planner. Output strict JSON only. "
                        "When this is not the final step, improve the plan but keep it open for one more evaluation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("empty model response")
        payload = _extract_json_object(text)
        if "proposal" not in payload:
            raise ValueError("response missing 'proposal' field")
        if not isinstance(payload["proposal"], list):
            raise ValueError("response field 'proposal' must be a list")

        action = ATCOptimizationAction.model_validate(
            {
                "proposal": payload["proposal"],
                "rationale": payload.get("rationale", "Model-generated ATC plan."),
                "commit": should_commit,
            }
        )
        if len(action.proposal) < len(observation.flights):
            raise ValueError("model proposal omitted flights")
        return action
    except (
        APIConnectionError,
        APITimeoutError,
        APIError,
        RateLimitError,
        OpenAIError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Model request failed for {task_id}: {exc}", file=sys.stderr, flush=True)
        return ATCOptimizationAction(
            proposal=seed_plan,
            rationale="Fell back to deterministic planner after model failure.",
            commit=should_commit,
        )


async def run_task(client: Optional[OpenAI], base_url: str, task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = SCORE_EPSILON
    success = False
    runtime_model = MODEL_NAME if client is not None else "heuristic-baseline"
    log_start(task=task_id, env=BENCHMARK, model=runtime_model)
    env: Optional[ATCOptimizationEnv] = None

    try:
        env = ATCOptimizationEnv(base_url=base_url)
        await env.__aenter__()
        result = await env.reset(task_id=task_id)
        max_steps = _step_budget(result.observation.steps_remaining)

        for step_index in range(max_steps):
            if result.done:
                break
            step_num = step_index + 1
            action = get_model_action(client, result.observation, task_id, step_num)
            result = await env.step(action)
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step_num
            action_log = f"submit_plan(count={len(action.proposal)},commit={_bool_token(action.commit)})"
            log_step(step=step_num, action=action_log, reward=reward, done=result.done, error=None)
            if result.done:
                break

        score = max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, result.observation.current_metrics.overall_score))
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"Task execution failed for {task_id}: {exc}", file=sys.stderr, flush=True)
        return SCORE_EPSILON
    finally:
        if env is not None:
            await env.__aexit__(None, None, None)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    process: Optional[subprocess.Popen] = None
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if API_BASE_URL and HF_TOKEN else None
        base_url, process = await prepare_base_url()
        for task_id in TASK_IDS:
            await run_task(client, base_url, task_id)
    except Exception as exc:
        try:
            print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        except (BrokenPipeError, OSError):
            pass
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    # On Linux, reset SIGPIPE to default so broken-pipe produces a clean exit
    # instead of a Python exception.  On Windows this signal doesn't exist.
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    try:
        asyncio.run(main())
    except (BrokenPipeError, OSError, KeyboardInterrupt):
        pass
    finally:
        # Silently close stdout/stderr to avoid secondary BrokenPipeError
        # during interpreter shutdown.
        try:
            sys.stdout.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            sys.stderr.close()
        except (BrokenPipeError, OSError):
            pass
