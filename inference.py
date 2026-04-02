"""Baseline inference runner with strict structured logs."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client import ATCOptimizationEnv
from models import ATCOptimizationAction
from planner import build_heuristic_plan


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or ""
)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", os.getenv("IMAGE_NAME", ""))
BENCHMARK = "atc_optimization_openenv"
TASK_IDS = [
    "delhi_monsoon_recovery_easy",
    "mumbai_bank_balance_medium",
    "bengaluru_irrops_hard",
]
SUCCESS_SCORE_THRESHOLD = 0.65
MAX_STEPS = 1
MAX_TOKENS = 1400
TEMPERATURE = 0


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_text = "null" if error is None else error.replace("\n", " ").strip()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{item:.2f}" for item in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_text}",
        flush=True,
    )


async def wait_for_server(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.time() < deadline:
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for server health at {base_url}")


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
    except Exception:
        process.terminate()
        raise


def get_model_action(client: Optional[OpenAI], observation, task_id: str) -> ATCOptimizationAction:
    heuristic_plan = build_heuristic_plan(observation)
    heuristic_json = json.dumps(
        [item.model_dump() for item in heuristic_plan],
        ensure_ascii=True,
    )
    if client is None or not API_BASE_URL or not API_KEY or MODEL_NAME == "heuristic-baseline":
        return ATCOptimizationAction(
            proposal=heuristic_plan,
            rationale="Deterministic heuristic baseline used because no model endpoint is configured.",
            commit=True,
        )

    prompt = (
        "You are optimizing a disrupted ATC runway schedule.\n"
        "Return strict JSON only with keys rationale and proposal.\n"
        "proposal must be an array of objects with flight_id, runway, assigned_minute, hold_minutes.\n"
        "Do not omit any flight. Keep assignments conflict-free and priority aware.\n\n"
        f"Task: {task_id}\n"
        f"Briefing:\n{observation.briefing}\n\n"
        f"Heuristic candidate plan:\n{heuristic_json}\n"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a conservative ATC planner. Output strict JSON only.",
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
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"response is not JSON: {text[:120]}")
        payload = json.loads(text[start : end + 1])
        action = ATCOptimizationAction.model_validate(
            {
                "proposal": payload["proposal"],
                "rationale": payload.get("rationale", "Model-generated ATC plan."),
                "commit": True,
            }
        )
        if len(action.proposal) < len(observation.flights):
            raise ValueError("model proposal omitted flights")
        return action
    except Exception as exc:
        print(f"Model request failed for {task_id}: {exc}", file=sys.stderr, flush=True)
        return ATCOptimizationAction(
            proposal=heuristic_plan,
            rationale="Fell back to the deterministic heuristic planner after model failure.",
            commit=True,
        )


async def run_task(client: Optional[OpenAI], base_url: str, task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    runtime_model = MODEL_NAME if client is not None else "heuristic-baseline"
    log_start(task=task_id, env=BENCHMARK, model=runtime_model)

    async with ATCOptimizationEnv(base_url=base_url) as env:
        result = await env.reset(task_id=task_id)
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
            action = get_model_action(client, result.observation, task_id)
            result = await env.step(action)
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            action_log = (
                f"submit_plan(count={len(action.proposal)},commit={str(action.commit).lower()})"
            )
            log_step(
                step=step,
                action=action_log,
                reward=reward,
                done=result.done,
                error=None,
            )
            if result.done:
                break

        score = max(0.0, min(1.0, result.observation.current_metrics.overall_score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, rewards=rewards)
    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_BASE_URL and API_KEY else None
    base_url, process = await prepare_base_url()
    try:
        for task_id in TASK_IDS:
            await run_task(client, base_url, task_id)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    asyncio.run(main())
