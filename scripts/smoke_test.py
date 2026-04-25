#!/usr/bin/env python3
"""
ATC OpenEnv — Smoke Test
=========================
Validates structure + minimal behavior. No GPU required. ~30s to run.

Usage:
    python scripts/smoke_test.py [--save-dir DIR]

Exit: 0 = all pass, 1 = any failure
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Terminal colour helpers ───────────────────────────────────────────────────
_TTY = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s

OK   = _c("32;1", "✓")
ERR  = _c("31;1", "✗")
SKP  = _c("33;1", "–")
BOLD = lambda s: _c("1", s)
DIM  = lambda s: _c("2", s)

# ── Result accumulator: (name, status, detail)  status = "pass"|"fail"|"skip" ─
_results: List[Tuple[str, str, str]] = []

class _Skip(Exception):
    pass

def run_check(name: str, fn, *args, **kwargs) -> str:
    try:
        detail = fn(*args, **kwargs) or ""
        _results.append((name, "pass", str(detail)))
        print(f"  {OK}  {name}{DIM(f'  {detail}') if detail else ''}")
        return "pass"
    except _Skip as exc:
        detail = str(exc)[:110]
        _results.append((name, "skip", detail))
        print(f"  {SKP}  {name}{DIM(f'  {detail}') if detail else ''}")
        return "skip"
    except Exception as exc:
        detail = str(exc).split("\n")[0][:110]
        _results.append((name, "fail", detail))
        print(f"  {ERR}  {name}")
        print(f"       {_c('31', detail)}")
        if os.getenv("SMOKE_VERBOSE"):
            traceback.print_exc()
        return "fail"

def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise _Skip("matplotlib not installed (will pass on server)")


# ══════════════════════════════════════════════════════════════════════════════
# 1. FILE STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FILES = [
    "openenv.yaml",
    "pyproject.toml",
    "server/app.py",
    "server/atc_environment.py",
    "models.py",
    "engine.py",
    "graders.py",
    "tasks.py",
    "planner.py",
    "constants.py",
    "multi_agent/__init__.py",
    "multi_agent/environment.py",
    "multi_agent/models.py",
    "multi_agent/supervisor.py",
    "multi_agent/generator.py",
    "training/train_grpo.py",
    "training/reward_functions.py",
    "training/dataset.py",
    "training/eval.py",
    "training/plot_rewards.py",
    "training/visualization.py",
]

def _check_files() -> str:
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {', '.join(missing)}")
    return f"{len(REQUIRED_FILES)} files"


def _check_openenv_yaml() -> str:
    try:
        import yaml
    except ImportError:
        raise ImportError("pip install pyyaml")
    data = yaml.safe_load((ROOT / "openenv.yaml").read_text())
    for field in ("spec_version", "name", "action_space", "observation_space", "state_space"):
        if field not in data:
            raise KeyError(f"Missing required field: {field}")
    if not data.get("tags"):
        raise ValueError("No tags defined")
    reserved = {"bash", "python", "exec", "run", "shell", "reset", "step"}
    tag_conflicts = reserved & {t.lower() for t in data["tags"]}
    if tag_conflicts:
        raise ValueError(f"Reserved tag names: {tag_conflicts}")
    return f"v{data['spec_version']}  name={data['name']}  tags={len(data['tags'])}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

def _check_core_imports() -> str:
    import models       # noqa: F401
    import engine       # noqa: F401
    import graders      # noqa: F401
    import tasks        # noqa: F401
    import planner      # noqa: F401
    import constants    # noqa: F401
    return "OK"

def _check_server_imports() -> str:
    import server.app              # noqa: F401
    import server.atc_environment  # noqa: F401
    return "OK"

def _check_multiagent_imports() -> str:
    import multi_agent.environment  # noqa: F401
    import multi_agent.models       # noqa: F401
    import multi_agent.supervisor   # noqa: F401
    import multi_agent.generator    # noqa: F401
    return "OK"

def _check_training_imports() -> str:
    import training.dataset           # noqa: F401
    import training.reward_functions  # noqa: F401
    try:
        import training.plot_rewards   # noqa: F401
        import training.visualization  # noqa: F401
    except ImportError as exc:
        if "matplotlib" in str(exc) or "pandas" in str(exc):
            raise _Skip(f"plot libs absent ({exc}) — will pass on server")
        raise
    return "OK"


# ══════════════════════════════════════════════════════════════════════════════
# 3. ENVIRONMENT BEHAVIOR
# ══════════════════════════════════════════════════════════════════════════════

def _check_env_reset_all_tasks() -> str:
    from server.atc_environment import ATCOptimizationEnvironment
    from tasks import task_catalog
    env = ATCOptimizationEnvironment()
    catalog = task_catalog()
    for task_id in catalog:
        obs = env.reset(task_id=task_id)
        assert obs.task_id == task_id, f"task_id mismatch: {obs.task_id}"
        assert len(obs.flights) >= 3, f"{task_id}: only {len(obs.flights)} flights"
        assert obs.steps_remaining > 0
    return f"{len(catalog)} tasks OK"

def _check_env_step() -> str:
    from server.atc_environment import ATCOptimizationEnvironment
    from models import ATCOptimizationAction
    from planner import build_heuristic_plan
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id="delhi_monsoon_recovery_easy")
    plan = build_heuristic_plan(obs)
    result = env.step(ATCOptimizationAction(proposal=plan, rationale="smoke", commit=True))
    score = result.current_metrics.overall_score
    assert 0.0 <= score <= 1.0, f"score out of [0,1]: {score}"
    return f"score={score:.3f}"

def _check_graders_deterministic() -> str:
    from server.atc_environment import ATCOptimizationEnvironment
    from planner import build_heuristic_plan
    from engine import simulate_plan
    from graders import grade_task
    from tasks import task_catalog
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id="mumbai_bank_balance_medium")
    plan = build_heuristic_plan(obs)
    task = task_catalog()["mumbai_bank_balance_medium"]
    outcome = simulate_plan(task, plan)
    a = {g.grader_name: round(g.score, 6) for g in grade_task(task, outcome, plan, "t")}
    b = {g.grader_name: round(g.score, 6) for g in grade_task(task, outcome, plan, "t")}
    assert a == b, f"Non-deterministic grader output"
    composite = a.get("composite_task_grader", "?")
    return f"composite={composite:.3f}" if isinstance(composite, float) else f"composite={composite}"

def _check_refined_not_worse() -> str:
    from server.atc_environment import ATCOptimizationEnvironment
    from planner import build_heuristic_plan, build_refined_plan
    from engine import simulate_plan
    from tasks import task_catalog
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id="bengaluru_irrops_hard")
    task = task_catalog()["bengaluru_irrops_hard"]
    seed_plan = build_heuristic_plan(obs)
    refined = build_refined_plan(obs, seed_plan=seed_plan)
    seed_s = simulate_plan(task, seed_plan).metrics.overall_score
    ref_s  = simulate_plan(task, refined).metrics.overall_score
    assert ref_s >= seed_s - 1e-6, f"refined ({ref_s:.3f}) < seed ({seed_s:.3f})"
    return f"seed={seed_s:.3f}  refined={ref_s:.3f}"

def _check_multiagent_reset() -> str:
    from multi_agent.environment import MultiAgentATCEnvironment
    env = MultiAgentATCEnvironment()
    obs = env.reset(task_id="delhi_monsoon_recovery_easy")
    assert obs is not None
    return "OK"


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_reward_curves(n: int = 150) -> dict:
    rng = random.Random(42)
    def _curve(start: float, end: float, noise: float) -> list:
        v, out = start, []
        for i in range(n):
            t = i / (n - 1)
            target = start + (end - start) * (1 - (1 - t) ** 2.2)
            v = 0.72 * v + 0.28 * target + rng.gauss(0, noise)
            v = max(-1.0, min(1.0, v))
            out.append(round(v, 4))
        return out
    return {
        "AMAN":       _curve(-0.20,  0.65, 0.11),
        "DMAN":       _curve(-0.15,  0.61, 0.11),
        "GENERATOR":  _curve( 0.10,  0.38, 0.09),
        "SUPERVISOR": _curve( 0.22,  0.73, 0.07),
        "composite":  _curve(-0.10,  0.63, 0.09),
    }

def _synthetic_eval_results() -> dict:
    return {
        "base": {
            "mean_composite": 0.30, "mean_aman": 0.27, "mean_dman": 0.25,
            "mean_coord": 0.19, "success_rate": 0.12,
        },
        "trained": {
            "mean_composite": 0.64, "mean_aman": 0.66, "mean_dman": 0.62,
            "mean_coord": 0.59, "success_rate": 0.57,
        },
    }

def _check_plot_training_curves(save_dir: Path) -> str:
    _require_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    from training.plot_rewards import plot_training_curves
    plot_training_curves(_synthetic_reward_curves(), save_dir=str(save_dir), show=False)
    path = save_dir / "training_curves.png"
    assert path.exists() and path.stat().st_size > 10_000, "PNG too small or missing"
    return str(path.relative_to(ROOT))

def _check_plot_eval_comparison(save_dir: Path) -> str:
    _require_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    from training.plot_rewards import plot_eval_comparison
    plot_eval_comparison(_synthetic_eval_results(), save_dir=str(save_dir), show=False)
    path = save_dir / "eval_comparison.png"
    assert path.exists() and path.stat().st_size > 5_000, "PNG too small or missing"
    return str(path.relative_to(ROOT))

def _check_plot_dashboard(save_dir: Path) -> str:
    _require_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from training.visualization import create_training_panel, refresh_training_panel

    rng = random.Random(7)
    curves = _synthetic_reward_curves()
    n = len(curves["AMAN"])
    roles = ("AMAN", "DMAN", "GENERATOR", "SUPERVISOR")

    query_logs = [
        {
            "step": step,
            "role": role,
            "reward": curves[role][step],
            "weighted_loss": max(0, 0.85 - 0.005 * step + rng.gauss(0, 0.05)),
            "parse_ok": rng.random() > 0.08,
            "prompt_chars": rng.randint(320, 500),
            "completion_chars": rng.randint(90, 280),
        }
        for step in range(n)
        for role in roles
    ]

    tasks_list = ["delhi_easy", "mumbai_med", "bengaluru_hard", "hyderabad_hard"]
    profiles   = ["safety_strict", "throughput_max", "fuel_economy", "emergency_priority"]
    n_eps = 40
    episode_logs = [
        {
            "episode": ep,
            "task_id": tasks_list[ep % 4],
            "supervisor_profile": profiles[ep % 4],
            "composite_score":  max(0, 0.25 + 0.40 * (ep / n_eps) + rng.gauss(0, 0.06)),
            "coord_score":      max(0, 0.20 + 0.38 * (ep / n_eps) + rng.gauss(0, 0.07)),
            "conflicts":        max(0, int(4 - 3.5 * (ep / n_eps) + rng.gauss(0, 0.4))),
            "aman_reward":      max(0, 0.22 + 0.43 * (ep / n_eps) + rng.gauss(0, 0.07)),
            "dman_reward":      max(0, 0.20 + 0.40 * (ep / n_eps) + rng.gauss(0, 0.07)),
            "generator_reward": max(0, 0.15 + 0.20 * (ep / n_eps) + rng.gauss(0, 0.05)),
            "supervisor_score": max(0, 0.25 + 0.47 * (ep / n_eps) + rng.gauss(0, 0.06)),
        }
        for ep in range(n_eps)
    ]

    fig, axes = create_training_panel()
    save_path = save_dir / "training_panel.png"
    refresh_training_panel(query_logs, episode_logs, fig, axes, save_path=save_path, force=True)
    plt.close("all")
    assert save_path.exists() and save_path.stat().st_size > 20_000
    return str(save_path.relative_to(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="ATC OpenEnv smoke test")
    parser.add_argument("--save-dir", default="outputs/smoke_test_plots",
                        help="Where to save validation plots")
    args = parser.parse_args()

    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    width = 60
    print()
    print(BOLD("=" * width))
    print(BOLD("  ATC OpenEnv  ·  Smoke Test"))
    print(BOLD("=" * width))

    t0 = time.monotonic()

    print(BOLD("\n── File structure ──"))
    run_check("Required files",         _check_files)
    run_check("openenv.yaml valid",     _check_openenv_yaml)

    print(BOLD("\n── Imports ──"))
    run_check("Core modules",           _check_core_imports)
    run_check("Server modules",         _check_server_imports)
    run_check("Multi-agent modules",    _check_multiagent_imports)
    run_check("Training modules",       _check_training_imports)

    print(BOLD("\n── Environment behavior ──"))
    run_check("env.reset (all 4 tasks)",        _check_env_reset_all_tasks)
    run_check("env.step  (heuristic, easy)",    _check_env_step)
    run_check("Graders deterministic",          _check_graders_deterministic)
    run_check("Refined plan ≥ seed plan",       _check_refined_not_worse)
    run_check("MultiAgent env.reset",           _check_multiagent_reset)

    print(BOLD("\n── Plots (synthetic data) ──"))
    run_check("training_curves.png",    _check_plot_training_curves,   save_dir)
    run_check("eval_comparison.png",    _check_plot_eval_comparison,    save_dir)
    run_check("training_panel.png",     _check_plot_dashboard,          save_dir)

    elapsed = time.monotonic() - t0
    passed  = sum(1 for _, s, _ in _results if s == "pass")
    skipped = sum(1 for _, s, _ in _results if s == "skip")
    failed  = sum(1 for _, s, _ in _results if s == "fail")
    total   = len(_results)

    print()
    print(BOLD("=" * width))
    if failed == 0:
        skip_note = DIM(f"  {skipped} skipped (server-only)") if skipped else ""
        print(BOLD(_c("32", f"  PASSED  {passed}/{total - skipped}")) + DIM(f"   ({elapsed:.1f}s)") + skip_note)
        plot_files = list(save_dir.glob("*.png"))
        if plot_files:
            print(DIM(f"  Plots ({len(plot_files)}) → {save_dir.relative_to(ROOT)}/"))
    else:
        print(BOLD(_c("31", f"  FAILED  {failed}/{total} checks")) + DIM(f"   ({elapsed:.1f}s)"))
        print()
        print(_c("31;1", "  Failures:"))
        for name, status, detail in _results:
            if status == "fail":
                print(f"    {ERR}  {name}")
                if detail:
                    print(f"         {_c('31', detail)}")
    print(BOLD("=" * width))
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
