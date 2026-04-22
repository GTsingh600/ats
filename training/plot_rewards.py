"""Reward curve visualization for multi-agent ATC training.

Reads reward_curves.json produced by train_grpo.py and generates:
  1. Per-role reward curves (AMAN, DMAN, GENERATOR, SUPERVISOR)
  2. Coordination score progression
  3. Generator difficulty escalation overlay
  4. Before/after comparison bar chart (from eval.py output)

Usage:
  python training/plot_rewards.py --input outputs/atc-multiagent/reward_curves.json
  python training/plot_rewards.py --eval_results eval_output.json --save plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _smooth(values: List[float], window: int = 10) -> List[float]:
    """Exponential moving average smoothing."""
    if not values:
        return values
    smoothed = [values[0]]
    alpha = 2.0 / (window + 1)
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def plot_training_curves(
    reward_curves: Dict[str, List[float]],
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[ERROR] pip install matplotlib")
        sys.exit(1)

    roles   = ["AMAN", "DMAN", "GENERATOR", "SUPERVISOR"]
    colours = {"AMAN": "#2196F3", "DMAN": "#FF9800", "GENERATOR": "#F44336", "SUPERVISOR": "#4CAF50"}

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Multi-Agent ATC — GRPO Training Curves", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ── Plot 1: Per-role rewards ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for role in roles:
        data = reward_curves.get(role, [])
        if not data:
            continue
        xs   = list(range(len(data)))
        raw  = data
        smt  = _smooth(data, window=15)
        ax1.plot(xs, raw, alpha=0.2, color=colours[role], linewidth=0.8)
        ax1.plot(xs, smt, label=role, color=colours[role], linewidth=2)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Reward")
    ax1.set_title("Per-Role Reward Progression (shaded=raw, solid=EMA)")
    ax1.legend(loc="lower right")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: AMAN vs DMAN convergence ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    aman_data = _smooth(reward_curves.get("AMAN", []))
    dman_data = _smooth(reward_curves.get("DMAN", []))
    n = min(len(aman_data), len(dman_data))
    if n > 0:
        xs = list(range(n))
        ax2.plot(xs, aman_data[:n], label="AMAN", color=colours["AMAN"], linewidth=2)
        ax2.plot(xs, dman_data[:n], label="DMAN", color=colours["DMAN"], linewidth=2)
        # Shade cooperation region (both > 0.5)
        ax2.fill_between(
            xs,
            [min(a, d) for a, d in zip(aman_data[:n], dman_data[:n])],
            0,
            where=[a > 0.4 and d > 0.4 for a, d in zip(aman_data[:n], dman_data[:n])],
            alpha=0.15,
            color="green",
            label="Cooperation zone",
        )
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Reward")
    ax2.set_title("AMAN vs DMAN — Coordination Emergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.05)

    # ── Plot 3: Generator adversarial reward + composite ─────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    gen_data = _smooth(reward_curves.get("GENERATOR", []))
    comp_data = _smooth(reward_curves.get("composite", []))
    n = max(len(gen_data), len(comp_data))
    if gen_data:
        xs = list(range(len(gen_data)))
        ax3.plot(xs, gen_data, label="Generator reward", color=colours["GENERATOR"],
                 linewidth=2, linestyle="--")
    if comp_data:
        xs = list(range(len(comp_data)))
        ax3_r = ax3.twinx()
        ax3_r.plot(xs, comp_data, label="Composite score", color="#9C27B0", linewidth=2)
        ax3_r.set_ylabel("Composite Score", color="#9C27B0")
        ax3_r.set_ylim(0, 1.05)
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Generator Reward", color=colours["GENERATOR"])
    ax3.set_title("Self-Play Arms Race: Generator vs Controllers")
    ax3.grid(True, alpha=0.3)

    lines1, labels1 = ax3.get_legend_handles_labels()
    if comp_data:
        lines2, labels2 = ax3_r.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        ax3.legend(loc="upper left")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / "training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


def plot_eval_comparison(eval_results: Dict, save_dir: Optional[str] = None, show: bool = True) -> None:
    """Bar chart comparing base vs trained on key metrics."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[ERROR] pip install matplotlib numpy")
        sys.exit(1)

    base    = eval_results.get("base",    {})
    trained = eval_results.get("trained", {})
    if not base or not trained:
        print("[WARN] eval_results must have 'base' and 'trained' keys")
        return

    metrics = [
        ("Composite Score",    "mean_composite"),
        ("AMAN Reward",        "mean_aman"),
        ("DMAN Reward",        "mean_dman"),
        ("Coordination Score", "mean_coord"),
        ("Success Rate",       "success_rate"),
    ]

    labels  = [m[0] for m in metrics]
    base_v  = [base.get(m[1],    0.0) for m in metrics]
    train_v = [trained.get(m[1], 0.0) for m in metrics]

    x   = range(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar([i - w/2 for i in x], base_v,  w, label="Base (untrained)", color="#90A4AE", alpha=0.85)
    bars2 = ax.bar([i + w/2 for i in x], train_v, w, label="Trained (GRPO)",   color="#1565C0", alpha=0.85)

    for bar in bars2:
        ax.annotate(
            f"{bar.get_height():.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color="#1565C0",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Multi-Agent ATC: Before vs After GRPO Training", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.6, color="orange", linestyle="--", linewidth=1, label="Success threshold (0.60)")

    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / "eval_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-agent ATC reward curves")
    parser.add_argument("--input",       default=None,
                        help="reward_curves.json from train_grpo.py")
    parser.add_argument("--eval_results", default=None,
                        help="eval output JSON from eval.py")
    parser.add_argument("--save",        default=None,
                        help="Directory to save PNG files")
    parser.add_argument("--no_show",     action="store_true",
                        help="Don't display plots interactively")
    args = parser.parse_args()

    show = not args.no_show

    if args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"[ERROR] {path} not found")
            sys.exit(1)
        data = json.loads(path.read_text())
        plot_training_curves(data, save_dir=args.save, show=show)

    if args.eval_results:
        path = Path(args.eval_results)
        if not path.exists():
            print(f"[ERROR] {path} not found")
            sys.exit(1)
        data = json.loads(path.read_text())
        plot_eval_comparison(data, save_dir=args.save, show=show)

    if not args.input and not args.eval_results:
        print("Usage: python training/plot_rewards.py --input reward_curves.json")
        print("       python training/plot_rewards.py --eval_results eval_output.json")


if __name__ == "__main__":
    main()
