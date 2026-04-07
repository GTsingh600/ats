"""
Multi-model benchmark runner for ATC Optimization OpenEnv.

Usage:
    python scripts/benchmark_models.py

Set environment variables:
    HF_TOKEN   - Your HuggingFace token
    MODELS     - (optional) Comma-separated model list to override defaults

Results are saved to:
    scripts/benchmark_results/results_<timestamp>.json
    scripts/benchmark_results/summary_<timestamp>.txt
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client import ATCOptimizationEnv
from models import ATCOptimizationAction
from planner import build_heuristic_plan, build_refined_plan

try:
    from openai import OpenAI, APIConnectionError, APIError, APITimeoutError, OpenAIError, RateLimitError
except ImportError:
    print("openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Models to benchmark — override with MODELS env var (comma-separated)
DEFAULT_MODELS: List[str] = [
    # Deterministic heuristic fallback (no API needed — use as baseline)
    "heuristic-baseline",
    # Small / fast models  (confirmed chat-compatible on HF router)
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    # Mid-range models
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    # Larger models (require HF Pro tier)
    # "meta-llama/Meta-Llama-3.1-70B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
]

TASK_IDS = [
    "delhi_monsoon_recovery_easy",
    "mumbai_bank_balance_medium",
    "bengaluru_irrops_hard",
    "hyderabad_cargo_crunch_medium_hard",
]

MAX_TOKENS = 1400
TEMPERATURE = 0
SUCCESS_THRESHOLD = 0.65
MAX_STEPS_CAP = 4

OUTPUT_DIR = ROOT / "scripts" / "benchmark_results"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _step_budget(steps_remaining: int) -> int:
    return min(MAX_STEPS_CAP, max(1, int(steps_remaining)))


async def _wait_for_server(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Optional[str] = None
    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    return
                last_error = f"HTTP {resp.status_code}"
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_error = str(exc)
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Server not ready at {base_url} ({last_error})")


async def _prepare_server() -> tuple[str, Optional[subprocess.Popen]]:
    for var in ("ENV_BASE_URL", "SPACE_URL", "PING_URL"):
        if os.getenv(var):
            return os.environ[var].rstrip("/"), None
    port = int(os.getenv("LOCAL_PORT", "8000"))
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        await _wait_for_server(base_url)
        return base_url, proc
    except RuntimeError:
        proc.terminate()
        raise


def _build_seed_plan(observation, step: int):
    if step <= 1 or not observation.current_plan:
        return build_heuristic_plan(observation)
    return build_refined_plan(observation, seed_plan=list(observation.current_plan))


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start > end:
        raise ValueError(f"No JSON object found in response: {text[:120]}")
    return json.loads(text[start: end + 1])


def _get_action(
    client: Optional[OpenAI],
    model_name: str,
    observation,
    task_id: str,
    step: int,
) -> ATCOptimizationAction:
    seed_plan = _build_seed_plan(observation, step)
    should_commit = step >= _step_budget(observation.steps_remaining)

    if client is None or model_name == "heuristic-baseline":
        return ATCOptimizationAction(
            proposal=seed_plan,
            rationale=(
                f"Deterministic heuristic plan for step {step}. "
                "Used as baseline — no model endpoint involved."
            ),
            commit=should_commit,
        )

    seed_json = json.dumps([item.model_dump() for item in seed_plan], ensure_ascii=True)
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
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative ATC planner. Output strict JSON only. "
                        "Improve on the seed plan but keep it safe and complete."
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
        if "proposal" not in payload or not isinstance(payload["proposal"], list):
            raise ValueError("response missing valid 'proposal' list")
        action = ATCOptimizationAction.model_validate(
            {
                "proposal": payload["proposal"],
                "rationale": payload.get("rationale", "Model-generated ATC plan."),
                "commit": should_commit,
            }
        )
        if len(action.proposal) < len(observation.flights):
            raise ValueError(f"model omitted {len(observation.flights) - len(action.proposal)} flights")
        return action
    except (
        APIConnectionError, APITimeoutError, APIError,
        RateLimitError, OpenAIError,
        json.JSONDecodeError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"  [WARN] Model request failed ({exc}); using heuristic fallback.", file=sys.stderr)
        return ATCOptimizationAction(
            proposal=seed_plan,
            rationale=f"Fallback heuristic after model failure: {exc}",
            commit=should_commit,
        )


async def _run_task(
    client: Optional[OpenAI],
    model_name: str,
    base_url: str,
    task_id: str,
) -> dict:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    t_start = time.time()
    env: Optional[ATCOptimizationEnv] = None
    error_msg = None

    try:
        env = ATCOptimizationEnv(base_url=base_url)
        await env.__aenter__()
        result = await env.reset(task_id=task_id)
        max_steps = _step_budget(result.observation.steps_remaining)

        for step_index in range(max_steps):
            if result.done:
                break
            step_num = step_index + 1
            action = _get_action(client, model_name, result.observation, task_id, step_num)
            result = await env.step(action)
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step_num
            if result.done:
                break

        score = max(0.0, min(1.0, result.observation.current_metrics.overall_score))
    except Exception as exc:
        error_msg = str(exc)
        print(f"  [ERROR] Task {task_id} failed: {exc}", file=sys.stderr)
    finally:
        if env is not None:
            await env.__aexit__(None, None, None)

    elapsed = round(time.time() - t_start, 2)
    return {
        "task_id": task_id,
        "score": round(score, 4),
        "success": score >= SUCCESS_THRESHOLD,
        "steps": steps_taken,
        "rewards": [round(r, 4) for r in rewards],
        "total_reward": round(sum(rewards), 4),
        "elapsed_s": elapsed,
        "error": error_msg,
    }


async def _run_model(model_name: str, base_url: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    client: Optional[OpenAI] = None
    if model_name != "heuristic-baseline" and API_BASE_URL and HF_TOKEN:
        # 180s timeout — large models (72B) can take >60s; default keepalive drops at 60s
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=180.0)

    task_results = []
    for task_id in TASK_IDS:
        print(f"  Running task: {task_id} ...", end=" ", flush=True)
        result = await _run_task(client, model_name, base_url, task_id)
        # Retry once on WebSocket/connection errors (score=0 with an error message)
        if result["score"] == 0.0 and result["error"] is not None:
            print(f"  [RETRY] Connection dropped, retrying {task_id}...", flush=True)
            result = await _run_task(client, model_name, base_url, task_id)
        task_results.append(result)
        status = "✓" if result["success"] else "✗"
        print(f"{status} score={result['score']:.4f}  ({result['elapsed_s']}s)")

    avg_score = round(sum(r["score"] for r in task_results) / len(task_results), 4)
    return {
        "model": model_name,
        "avg_score": avg_score,
        "tasks": task_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _print_summary(all_results: List[dict]) -> None:
    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<44} {'Avg':>6}  {'Easy':>6}  {'Med':>6}  {'Blr':>6}  {'Hyd':>6}"
    print(header)
    print("-" * len(header))
    sorted_results = sorted(all_results, key=lambda r: r["avg_score"], reverse=True)
    for entry in sorted_results:
        scores = {t["task_id"]: t["score"] for t in entry["tasks"]}
        easy   = scores.get("delhi_monsoon_recovery_easy",       0.0)
        medium = scores.get("mumbai_bank_balance_medium",         0.0)
        blr    = scores.get("bengaluru_irrops_hard",             0.0)
        hyd    = scores.get("hyderabad_cargo_crunch_medium_hard", 0.0)
        model_short = entry["model"][-44:] if len(entry["model"]) > 44 else entry["model"]
        print(f"{model_short:<44} {entry['avg_score']:>6.4f}  {easy:>6.4f}  {medium:>6.4f}  {blr:>6.4f}  {hyd:>6.4f}")
    print(f"{'='*60}\n")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Allow MODELS env override: MODELS="Qwen/Qwen2.5-7B-Instruct,heuristic-baseline"
    models_env = os.getenv("MODELS", "")
    models = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else DEFAULT_MODELS

    if not HF_TOKEN and any(m != "heuristic-baseline" for m in models):
        print("[WARN] HF_TOKEN is not set. Only heuristic-baseline will work.", file=sys.stderr)

    print(f"Starting benchmark: {len(models)} model(s) × {len(TASK_IDS)} task(s)")
    print(f"API base: {API_BASE_URL}")
    print(f"Models: {models}")

    base_url, server_proc = await _prepare_server()
    print(f"Environment server: {base_url}")

    all_results: List[dict] = []
    try:
        for model_name in models:
            result = await _run_model(model_name, base_url)
            all_results.append(result)
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    _print_summary(all_results)

    # Save JSON results
    json_path = OUTPUT_DIR / f"results_{timestamp}.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Results saved: {json_path}")

    # Save human-readable summary
    summary_lines = []
    summary_lines.append(f"ATC Benchmark Results — {timestamp}\n")
    summary_lines.append(f"{'Model':<44} {'Avg':>6}  {'Easy':>6}  {'Med':>6}  {'Blr':>6}  {'Hyd':>6}")
    summary_lines.append("-" * 76)
    for entry in sorted(all_results, key=lambda r: r["avg_score"], reverse=True):
        scores = {t["task_id"]: t["score"] for t in entry["tasks"]}
        easy   = scores.get("delhi_monsoon_recovery_easy",       0.0)
        medium = scores.get("mumbai_bank_balance_medium",         0.0)
        blr    = scores.get("bengaluru_irrops_hard",             0.0)
        hyd    = scores.get("hyderabad_cargo_crunch_medium_hard", 0.0)
        m = entry["model"][-44:]
        summary_lines.append(f"{m:<44} {entry['avg_score']:>6.4f}  {easy:>6.4f}  {medium:>6.4f}  {blr:>6.4f}  {hyd:>6.4f}")
    txt_path = OUTPUT_DIR / f"summary_{timestamp}.txt"
    txt_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Summary saved: {txt_path}")


if __name__ == "__main__":
    asyncio.run(main())
