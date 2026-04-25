"""Reusable training visualization utilities.

Provides `create_training_panel` and `refresh_training_panel` that work in
notebooks (interactive display) and headless servers (saves PNG snapshots).

Keep implementations lightweight and robust for the DGX/A100 offline setup.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import matplotlib

# Use non-interactive backend when DISPLAY not available (server-safe)
if not os.getenv("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_training_panel(figsize: Tuple[int, int] = (19, 13)) -> Tuple[plt.Figure, Any]:
    """Create a 3x3 matplotlib panel for training visualisations.

    Returns (fig, axes) where axes is a 3x3 ndarray.
    """
    plt.ion()
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    return fig, axes


def _rolling_mean(values: Iterable[float], window: int = 5) -> np.ndarray:
    s = pd.Series(list(values), dtype="float64")
    return s.rolling(window=min(window, max(1, len(s))), min_periods=1).mean().to_numpy()


def _safe_legend(ax: Any, **kwargs: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def refresh_training_panel(
    query_logs: Iterable[dict],
    episode_logs: Iterable[dict],
    fig: plt.Figure,
    axes: Any,
    save_path: Optional[Path] = None,
    live_plot_every: int = 1,
    rolling_window: int = 5,
    force: bool = False,
) -> None:
    """Refresh the panel using `query_logs` and `episode_logs`.

    If `save_path` is provided the figure will be saved (headless-friendly).
    """
    _ql = list(query_logs)
    _el = list(episode_logs)
    qdf = pd.DataFrame(_ql) if _ql else pd.DataFrame()
    edf = pd.DataFrame(_el) if _el else pd.DataFrame()

    if qdf.empty and edf.empty:
        return

    # Clear axes
    for ax in axes.flat:
        ax.clear()

    # 1) Reward over global steps + rolling trend
    ax = axes[0, 0]
    if not qdf.empty:
        for role, grp in qdf.groupby("role"):
            grp = grp.sort_values("step")
            ax.plot(grp["step"], grp["reward"], alpha=0.30, marker=".", linewidth=1.0, label=f"{role} raw")
            ax.plot(grp["step"], _rolling_mean(grp["reward"], rolling_window), linewidth=2.0, label=f"{role} roll")
    ax.set_title("Reward by Global Step (raw + rolling)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True)
    _safe_legend(ax, fontsize=7, ncol=2)

    # 2) Weighted loss over steps
    ax = axes[0, 1]
    if not qdf.empty:
        for role, grp in qdf.groupby("role"):
            grp = grp.sort_values("step")
            y = pd.Series(grp.get("weighted_loss", [])).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            ax.plot(grp["step"], y, alpha=0.30, marker=".", linewidth=1.0, label=f"{role} raw")
            ax.plot(grp["step"], _rolling_mean(y, rolling_window), linewidth=2.0, label=f"{role} roll")
    ax.set_title("Weighted Loss by Step (raw + rolling)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Weighted Loss")
    ax.grid(True)
    _safe_legend(ax, fontsize=7, ncol=2)

    # 3) Parse success cumulative mean
    ax = axes[0, 2]
    if not qdf.empty and "parse_ok" in qdf.columns:
        for role, grp in qdf.groupby("role"):
            grp = grp.sort_values("step")
            cumulative = grp["parse_ok"].astype(float).expanding().mean()
            ax.plot(grp["step"], cumulative, marker="o", linewidth=1.8, label=role)
    ax.set_title("Parse Success Cumulative Mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("Parse Success")
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    _safe_legend(ax, fontsize=8)

    # 4) Episode macro metrics
    ax = axes[1, 0]
    if not edf.empty:
        edf = edf.sort_values("episode")
        ax.plot(edf["episode"], edf["composite_score"], marker="o", label="composite")
        ax.plot(edf["episode"], edf["coord_score"], marker="o", label="coord")
        ax.plot(edf["episode"], edf["conflicts"], marker="x", label="conflicts")
    ax.set_title("Episode Macro Metrics")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.grid(True)
    _safe_legend(ax, fontsize=8)

    # 5) Episode role rewards
    ax = axes[1, 1]
    if not edf.empty:
        ax.plot(edf["episode"], edf.get("aman_reward", []), marker="o", label="AMAN reward")
        ax.plot(edf["episode"], edf.get("dman_reward", []), marker="o", label="DMAN reward")
        ax.plot(edf["episode"], edf.get("generator_reward", []), marker="x", label="Generator reward")
        ax.plot(edf["episode"], edf.get("supervisor_score", []), marker="x", label="Supervisor score")
    ax.set_title("Per-Episode Role Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward / Score")
    ax.grid(True)
    _safe_legend(ax, fontsize=7, ncol=2)

    # 6) Reward distribution
    ax = axes[1, 2]
    if not qdf.empty:
        bins = np.linspace(-1.0, 1.0, 16)
        for role, grp in qdf.groupby("role"):
            ax.hist(grp["reward"], bins=bins, alpha=0.35, label=role)
    ax.set_title("Reward Distribution by Role")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.grid(True)
    _safe_legend(ax, fontsize=8)

    # 7) Prompt/completion size trends
    ax = axes[2, 0]
    if not qdf.empty:
        for role, grp in qdf.groupby("role"):
            grp = grp.sort_values("step")
            ax.plot(grp["step"], _rolling_mean(grp.get("prompt_chars", []), rolling_window), linewidth=2.0, label=f"{role} prompt")
            ax.plot(grp["step"], _rolling_mean(grp.get("completion_chars", []), rolling_window), linestyle="--", linewidth=1.8, label=f"{role} completion")
    ax.set_title("Prompt vs Completion Length (rolling)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Chars")
    ax.grid(True)
    _safe_legend(ax, fontsize=7, ncol=2)

    # 8) Mean composite by task
    ax = axes[2, 1]
    if not edf.empty:
        mean_by_task = edf.groupby("task_id")["composite_score"].mean().sort_values()
        ax.bar(mean_by_task.index.astype(str), mean_by_task.values, color="tab:green", alpha=0.8)
        ax.tick_params(axis="x", labelrotation=20)
    ax.set_title("Average Composite Score by Task")
    ax.set_xlabel("Task")
    ax.set_ylabel("Avg Composite")
    ax.grid(True, axis="y")

    # 9) Mean composite by supervisor profile
    ax = axes[2, 2]
    if not edf.empty:
        mean_by_sup = edf.groupby("supervisor_profile")["composite_score"].mean().sort_values()
        ax.bar(mean_by_sup.index.astype(str), mean_by_sup.values, color="tab:orange", alpha=0.8)
        ax.tick_params(axis="x", labelrotation=20)
    ax.set_title("Average Composite by Supervisor Profile")
    ax.set_xlabel("Supervisor Profile")
    ax.set_ylabel("Avg Composite")
    ax.grid(True, axis="y")

    fig.tight_layout()

    if save_path is not None:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        except Exception:
            # swallowing save errors keeps training robust
            pass

    # In interactive environments, show the figure
    try:
        from IPython.display import clear_output, display

        clear_output(wait=True)
        display(fig)
    except Exception:
        # Not running in a notebook
        pass


__all__ = ["create_training_panel", "refresh_training_panel"]
