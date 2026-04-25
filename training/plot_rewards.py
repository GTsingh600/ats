"""Reward curve visualization for multi-agent ATC training.

Reads reward_curves.json and training_diagnostics.json from train_grpo.py and generates:
  1. Per-role reward curves (AMAN, DMAN, GENERATOR, SUPERVISOR, ADAPT)
  2. Coordination / generator vs composite views
  3. Optional diagnostics: parse rates, batch health, difficulty vs reward, curriculum traces
  4. Before/after comparison bar chart (from eval metrics)

Usage:
  python training/plot_rewards.py --input outputs/atc-multiagent/reward_curves.json
  python training/plot_rewards.py --output_dir outputs/atc-multiagent --save plots/
  python training/plot_rewards.py --eval_results eval_output.json --save plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "figure.dpi": 150,
    })

    roles = ["AMAN", "DMAN", "GENERATOR", "SUPERVISOR", "ADAPT"]
    colours = {
        "AMAN": "#1976D2",
        "DMAN": "#F57C00",
        "GENERATOR": "#C62828",
        "SUPERVISOR": "#2E7D32",
        "ADAPT": "#6A1B9A",
    }

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Multi-Agent ATC — GRPO Training Curves", fontsize=15, fontweight="bold", y=0.98)
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


def _roll_mean(vals: List[float], window: int) -> List[float]:
    if not vals or window < 2:
        return [float(x) for x in vals]
    out: List[float] = []
    acc = 0.0
    buf: List[float] = []
    for v in vals:
        x = float(v)
        buf.append(x)
        acc += x
        if len(buf) > window:
            acc -= buf.pop(0)
        out.append(acc / len(buf))
    return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _batch_x(row: Dict[str, Any]) -> Optional[int]:
    if "batch" in row:
        return int(row["batch"])
    if "batches" in row:
        return int(row["batches"])
    return None


def _setup_matplotlib(show: bool) -> Any:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.dpi": 150,
        }
    )
    return plt


def plot_reward_histograms_figure(
    reward_log: Dict[str, List[float]],
    save_dir: str,
    *,
    show: bool = False,
) -> Optional[Path]:
    """Per-role + composite reward histograms (all samples)."""
    plt = _setup_matplotlib(show)
    roles = [r for r in ("AMAN", "DMAN", "GENERATOR", "SUPERVISOR", "ADAPT") if reward_log.get(r)]
    if not roles and not reward_log.get("composite"):
        plt.close("all")
        return None
    n = len(roles) + (1 if reward_log.get("composite") else 0)
    cols = 3
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    idx = 0
    colours = {
        "AMAN": "#1976D2",
        "DMAN": "#F57C00",
        "GENERATOR": "#C62828",
        "SUPERVISOR": "#2E7D32",
        "ADAPT": "#6A1B9A",
        "composite": "#9C27B0",
    }
    for role in roles + (["composite"] if reward_log.get("composite") else []):
        ax = axes_flat[idx]
        data = [float(x) for x in reward_log.get(role, []) if x == x]
        if data:
            ax.hist(data, bins=40, color=colours.get(role, "#546E7A"), alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{role} reward")
        ax.set_xlabel("reward")
        ax.set_ylabel("count")
        idx += 1
    for j in range(idx, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Reward distributions (full run)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "reward_histograms.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()
    return out


def plot_difficulty_histogram(
    difficulty_log: List[float],
    save_dir: str,
    *,
    show: bool = False,
) -> Optional[Path]:
    ds = [float(x) for x in difficulty_log if x == x]
    if len(ds) < 2:
        return None
    plt = _setup_matplotlib(show)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(ds, bins=36, color="#00838F", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("continuous difficulty d")
    ax.set_ylabel("samples")
    ax.set_title("Difficulty scalar — where the curriculum spent mass")
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "difficulty_histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()
    return out


def plot_parse_failure_summary(
    parse_debug_samples: List[Dict[str, Any]],
    save_dir: str,
    *,
    show: bool = False,
) -> Optional[Path]:
    if not parse_debug_samples:
        return None
    fails: Dict[str, int] = {}
    for row in parse_debug_samples:
        if int(row.get("parse_ok", 1)) == 0:
            r = str(row.get("role", "?"))
            fails[r] = fails.get(r, 0) + 1
    if not fails:
        return None
    plt = _setup_matplotlib(show)
    fig, ax = plt.subplots(figsize=(7, 4))
    roles = list(fails.keys())
    vals = [fails[k] for k in roles]
    ax.bar(roles, vals, color="#C62828", alpha=0.85)
    ax.set_ylabel("logged parse failures")
    ax.set_title("Parse failures by role (from parse_debug_samples)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "parse_failures_by_role.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()
    return out


def plot_batch_health_timeseries(
    batch_diagnostics: List[Dict[str, Any]],
    save_dir: str,
    *,
    show: bool = False,
) -> Optional[Path]:
    if not batch_diagnostics:
        return None
    plt = _setup_matplotlib(show)
    xs = [int(b.get("batch_index", i)) for i, b in enumerate(batch_diagnostics)]
    c_mean = [float(b.get("composite_mean", 0)) for b in batch_diagnostics]
    c_std = [float(b.get("composite_std", 0)) for b in batch_diagnostics]
    d_mean = []
    for b in batch_diagnostics:
        v = b.get("difficulty_mean")
        d_mean.append(float(v) if v is not None and v == v else float("nan"))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(xs, c_mean, color="#1565C0", linewidth=1.2, label="composite_mean")
    axes[0].fill_between(xs, c_mean, alpha=0.12, color="#1565C0")
    axes[0].set_ylabel("mean reward")
    axes[0].set_title("Per-batch composite mean / std (within-batch spread)")
    axes[0].legend(loc="upper right")

    axes[1].plot(xs, c_std, color="#6A1B9A", linewidth=1.0)
    axes[1].set_ylabel("composite std")

    if any(v == v for v in d_mean):
        axes[2].plot(xs, d_mean, color="#00838F", linewidth=1.0)
    axes[2].set_ylabel("mean d")
    axes[2].set_xlabel("batch index")
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "batch_health_timeseries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()
    return out


def plot_action_diversity_timeseries(
    batch_diagnostics: List[Dict[str, Any]],
    save_dir: str,
    *,
    show: bool = False,
) -> Optional[Path]:
    if not batch_diagnostics:
        return None
    plt = _setup_matplotlib(show)
    xs = [int(b.get("batch_index", i)) for i, b in enumerate(batch_diagnostics)]
    colours = {
        "AMAN": "#1976D2",
        "DMAN": "#F57C00",
        "GENERATOR": "#C62828",
        "SUPERVISOR": "#2E7D32",
        "ADAPT": "#6A1B9A",
    }
    fig, ax = plt.subplots(figsize=(12, 4.5))
    any_line = False
    for role in ("AMAN", "DMAN", "GENERATOR", "SUPERVISOR", "ADAPT"):
        ys: List[float] = []
        for b in batch_diagnostics:
            ad = b.get("action_diversity") or {}
            v = ad.get(role)
            ys.append(float(v) if v is not None and v == v else float("nan"))
        if any(y == y for y in ys):
            ax.plot(xs, ys, label=role, color=colours[role], linewidth=1.1, alpha=0.9)
            any_line = True
    if not any_line:
        plt.close()
        return None
    ax.set_xlabel("batch index")
    ax.set_ylabel("unique signatures / batch length")
    ax.set_title("Action diversity per role (per batch)")
    ax.legend(ncol=5, loc="upper right", fontsize=8)
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "action_diversity_timeseries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()
    return out


def plot_batch_parse_rates(
    batch_diagnostics: List[Dict[str, Any]],
    save_dir: str,
    *,
    show: bool = False,
) -> Optional[Path]:
    if not batch_diagnostics:
        return None
    plt = _setup_matplotlib(show)
    xs = [int(b.get("batch_index", i)) for i, b in enumerate(batch_diagnostics)]
    colours = {
        "AMAN": "#1976D2",
        "DMAN": "#F57C00",
        "GENERATOR": "#C62828",
        "SUPERVISOR": "#2E7D32",
        "ADAPT": "#6A1B9A",
    }
    fig, ax = plt.subplots(figsize=(12, 4.5))
    drew = False
    for role in ("AMAN", "DMAN", "GENERATOR", "SUPERVISOR", "ADAPT"):
        ys: List[float] = []
        for b in batch_diagnostics:
            pr = b.get("parse_rate") or {}
            v = pr.get(role)
            ys.append(float(v) if v is not None and v == v else float("nan"))
        if any(y == y for y in ys):
            ax.plot(xs, ys, label=role, color=colours[role], linewidth=1.0, alpha=0.85)
            drew = True
    if not drew:
        plt.close()
        return None
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("batch index")
    ax.set_ylabel("parse rate")
    ax.set_title("Batch-level parse success (mean within batch)")
    ax.legend(ncol=5, loc="lower right", fontsize=8)
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "batch_parse_rates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()
    return out


def plot_training_diagnostics_report(
    diag: Dict[str, Any],
    save_dir: str,
    *,
    show: bool = False,
) -> Optional[Path]:
    """Single multi-panel figure from training_diagnostics.json."""
    reward_log = diag.get("reward_log") or {}
    parse_log = diag.get("parse_log") or {}
    difficulty_log = diag.get("difficulty_log") or []
    composite = reward_log.get("composite") or []
    batch_diag = diag.get("batch_diagnostics") or []
    by_bin = diag.get("reward_by_difficulty_bin") or {}
    corr = diag.get("reward_difficulty_correlation")

    plt = _setup_matplotlib(show)
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Training diagnostics (parsed logs + batch summaries)", fontsize=13, fontweight="bold", y=0.98)

    # (0,0) rolling parse
    ax0 = fig.add_subplot(2, 3, 1)
    win = max(16, min(128, max(8, len(parse_log.get("AMAN", [])) // 40 or 8)))
    colours = {
        "AMAN": "#1976D2",
        "DMAN": "#F57C00",
        "GENERATOR": "#C62828",
        "SUPERVISOR": "#2E7D32",
        "ADAPT": "#6A1B9A",
    }
    for role, c in colours.items():
        pr = parse_log.get(role)
        if not pr:
            continue
        sm = _roll_mean([float(x) for x in pr], win)
        ax0.plot(range(len(sm)), sm, label=role, color=c, linewidth=1.2)
    ax0.set_ylim(-0.02, 1.05)
    ax0.set_xlabel("sample index")
    ax0.set_ylabel(f"parse success (rolling mean, w={win})")
    ax0.set_title("Parse quality over training")
    ax0.legend(loc="lower right", fontsize=7, ncol=2)

    # (0,1) SUP vs ADAPT rewards smoothed
    ax1 = fig.add_subplot(2, 3, 2)
    for role in ("SUPERVISOR", "ADAPT"):
        data = reward_log.get(role, [])
        if len(data) < 2:
            continue
        sm = _smooth([float(x) for x in data], window=21)
        ax1.plot(range(len(sm)), sm, label=role, color=colours[role], linewidth=1.4)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_xlabel("sample index")
    ax1.set_ylabel("reward (EMA)")
    ax1.set_title("Supervisor vs ADAPT (smoothed)")
    ax1.legend()

    # (0,2) difficulty vs composite scatter (subsample)
    ax2 = fig.add_subplot(2, 3, 3)
    n = min(len(difficulty_log), len(composite))
    if n > 2:
        import random

        rng = random.Random(0)
        cap = min(n, 12000)
        idxs = sorted(rng.sample(range(n), cap)) if n > cap else list(range(n))
        dx = [float(difficulty_log[i]) for i in idxs]
        cy = [float(composite[i]) for i in idxs]
        ax2.scatter(dx, cy, s=4, alpha=0.18, c="#37474F", edgecolors="none")
        ax2.set_xlabel("difficulty d")
        ax2.set_ylabel("per-sample composite reward")
        ax2.set_title("Difficulty vs reward (subsampled)")
    else:
        ax2.text(0.5, 0.5, "insufficient paired d/reward", ha="center", va="center")

    # (1,0) batch composite std
    ax3 = fig.add_subplot(2, 3, 4)
    if batch_diag:
        bx = [int(b.get("batch_index", i)) for i, b in enumerate(batch_diag)]
        ax3.plot(bx, [float(b.get("composite_std", 0)) for b in batch_diag], color="#6A1B9A", lw=1)
        ax3.set_xlabel("batch")
        ax3.set_ylabel("composite std")
        ax3.set_title("Within-batch reward spread")

    # (1,1) reward by difficulty bin
    ax4 = fig.add_subplot(2, 3, 5)
    if by_bin:
        keys = sorted(by_bin.keys())
        means = []
        errs = []
        for k in keys:
            cell = by_bin[k]
            m = cell.get("mean_reward")
            s = cell.get("std_reward")
            if m is None or m != m:
                means.append(0.0)
                errs.append(0.0)
            else:
                means.append(float(m))
                errs.append(float(s) if s is not None and s == s else 0.0)
        x = range(len(keys))
        ax4.bar(x, means, yerr=errs, capsize=3, color="#1565C0", alpha=0.85, ecolor="#37474F")
        ax4.set_xticks(list(x))
        ax4.set_xticklabels(keys, rotation=25, ha="right", fontsize=7)
        ax4.set_ylabel("mean reward")
        ax4.set_title("Mean composite reward by difficulty bin")
        ax4.set_ylim(bottom=min(-0.15, min(means + [0]) - 0.05))

    # (1,2) text summary
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.axis("off")
    lines = [
        f"ρ(difficulty, composite) = {corr}" if corr == corr else "ρ(difficulty, composite) = n/a",
        f"composite samples: {len(composite)}",
        f"difficulty samples: {len(difficulty_log)}",
        f"batches logged: {len(batch_diag)}",
    ]
    for role in ("AMAN", "DMAN", "GENERATOR", "SUPERVISOR", "ADAPT"):
        rs = reward_log.get(role, [])
        if rs:
            lines.append(f"{role}: n={len(rs)}  mean={sum(rs)/len(rs):.4f}")
    ax5.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "training_diagnostics_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()
    return out


def plot_curriculum_jsonl_figure(
    rows: List[Dict[str, Any]],
    title: str,
    save_path: Path,
    *,
    show: bool = False,
) -> Optional[Path]:
    if not rows:
        return None
    plt = _setup_matplotlib(show)
    xs: List[int] = []
    for i, r in enumerate(rows):
        x = _batch_x(r)
        xs.append(int(x) if x is not None else i)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    mus = [float(r.get("mu", 0)) for r in rows]
    cs = [float(r.get("c", 0)) for r in rows]
    ax = axes[0, 0]
    ax.plot(xs, mus, color="#1565C0", label="μ")
    ax.set_ylabel("μ")
    ax2 = ax.twinx()
    ax2.plot(xs, cs, color="#C62828", alpha=0.85, label="c")
    ax2.set_ylabel("c", color="#C62828")
    ax.set_title("Curriculum Beta parameters")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)

    rho = [float(r["rho_rd"]) for r in rows if r.get("rho_rd") is not None and float(r["rho_rd"]) == float(r["rho_rd"])]
    rho_x = [xs[i] for i, r in enumerate(rows) if r.get("rho_rd") is not None and float(r["rho_rd"]) == float(r["rho_rd"])]
    axb = axes[0, 1]
    if rho:
        axb.plot(rho_x, rho, color="#2E7D32", marker=".", markersize=3, linestyle="-")
        axb.set_ylabel("ρ")
        axb.set_title("Reward–difficulty correlation (logged)")
    else:
        rse = [float(r.get("reward_std_ema", float("nan"))) for r in rows]
        if any(v == v for v in rse):
            axb.plot(xs, rse, color="#F57C00", linewidth=1.2)
            axb.set_ylabel("reward_std_ema")
            axb.set_title("Reward std EMA (live curriculum log)")
        else:
            axb.text(0.5, 0.5, "no ρ or reward_std_ema", ha="center", va="center")

    # bin success trajectories
    axc = axes[1, 0]
    bin_keys: List[str] = []
    for r in rows:
        bs = r.get("bin_success")
        if isinstance(bs, dict) and bs:
            bin_keys = sorted(bs.keys())
            break
    if bin_keys:
        for bk in bin_keys:
            ys: List[float] = []
            for r in rows:
                bs = r.get("bin_success") or {}
                v = bs.get(bk)
                ys.append(float(v) if v is not None and v == v else float("nan"))
            axc.plot(xs, ys, marker="", linewidth=1.0, label=bk)
        axc.set_ylim(-0.05, 1.05)
        axc.legend(loc="best", fontsize=6, ncol=2)
    axc.set_xlabel("batch")
    axc.set_ylabel("bin success rate")
    axc.set_title("Success rate by difficulty bucket (windowed)")

    axd = axes[1, 1]
    brm_keys: List[str] = []
    for r in rows:
        br = r.get("bin_reward_mean")
        if isinstance(br, dict) and br:
            brm_keys = sorted(br.keys())
            break
    if brm_keys:
        for bk in brm_keys:
            ys = []
            for r in rows:
                br = r.get("bin_reward_mean") or {}
                v = br.get(bk)
                ys.append(float(v) if v is not None and v == v else float("nan"))
            axd.plot(xs, ys, linewidth=1.0, label=bk)
        axd.legend(loc="best", fontsize=6, ncol=2)
    axd.set_xlabel("batch")
    axd.set_ylabel("mean reward")
    axd.set_title("Mean reward by difficulty bucket")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close()
    return save_path


def plot_continuous_state_snapshot(
    state_path: Path,
    save_dir: str,
    *,
    show: bool = False,
) -> Optional[Path]:
    if not state_path.is_file():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sbb = data.get("success_by_bin") or {}
    if not sbb:
        return None
    plt = _setup_matplotlib(show)
    labels: List[str] = []
    means: List[float] = []
    def _sort_bin_key(k: str) -> Tuple[int, str]:
        try:
            return (int(k), "")
        except ValueError:
            return (10_000, str(k))

    for k in sorted(sbb.keys(), key=_sort_bin_key):
        vals = sbb[k]
        if not vals:
            continue
        fv = [float(v) for v in vals if v == v]
        if not fv:
            continue
        labels.append(f"bin {k}")
        means.append(sum(fv) / len(fv))
    if not means:
        plt.close()
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, means, color="#00838F", alpha=0.88)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean success (1=ok) in deque")
    ax.set_title(f"continuous_curriculum_state.json — snapshot @ batches={data.get('global_batches', '?')}")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = Path(save_dir) / "continuous_state_bin_success.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()
    return out


def plot_grounded_dataset_figure(gsum: Dict[str, Any], save_dir: str, *, show: bool = False) -> Optional[Path]:
    plt = _setup_matplotlib(show)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    mode = str(gsum.get("mode", ""))
    fig.suptitle(f"Grounded dataset summary ({mode})", fontsize=12, fontweight="bold")

    tc = gsum.get("grounded_template_counts") or {}
    if isinstance(tc, dict) and tc:
        items = sorted(tc.items(), key=lambda kv: kv[1], reverse=True)[:18]
        names = [k[:28] + ("…" if len(k) > 28 else "") for k, _ in items]
        vals = [v for _, v in items]
        axes[0].barh(names[::-1], vals[::-1], color="#37474F")
        axes[0].set_xlabel("count")
        axes[0].set_title("Top grounded templates")
    else:
        axes[0].axis("off")
        axes[0].text(
            0.1,
            0.5,
            f"materialized_rows={gsum.get('materialized_rows')}\nmax_steps={gsum.get('max_steps')}",
            fontsize=10,
            va="center",
        )

    bc = gsum.get("difficulty_bucket_counts") or {}
    if isinstance(bc, dict) and bc:
        ks = sorted(bc.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
        vs = [int(bc[k]) for k in ks]
        axes[1].bar([str(k) for k in ks], vs, color="#1565C0", alpha=0.88)
        axes[1].set_xlabel("bucket index")
        axes[1].set_ylabel("count")
        axes[1].set_title("Difficulty bucket counts (static list)")
    else:
        axes[1].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.9))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "grounded_dataset_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()
    return out


def plot_all_training_artifacts(
    output_dir: str | Path,
    save_dir: str | Path,
    *,
    show: bool = False,
) -> List[Path]:
    """Generate every plot we can from a train_grpo output directory."""
    out = Path(output_dir)
    dest = Path(save_dir)
    dest.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    curves_path = out / "reward_curves.json"
    if curves_path.is_file():
        data = json.loads(curves_path.read_text(encoding="utf-8"))
        plot_training_curves(data, save_dir=str(dest), show=show)
        saved.append(dest / "training_curves.png")

    diag_path = out / "training_diagnostics.json"
    if diag_path.is_file():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        p = plot_training_diagnostics_report(diag, str(dest), show=show)
        if p:
            saved.append(p)
        rl = diag.get("reward_log") or {}
        p2 = plot_reward_histograms_figure(rl, str(dest), show=show)
        if p2:
            saved.append(p2)
        dl = diag.get("difficulty_log") or []
        p3 = plot_difficulty_histogram(dl, str(dest), show=show)
        if p3:
            saved.append(p3)
        p4 = plot_parse_failure_summary(diag.get("parse_debug_samples") or [], str(dest), show=show)
        if p4:
            saved.append(p4)
        bd = diag.get("batch_diagnostics") or []
        for fn in (
            plot_batch_health_timeseries,
            plot_action_diversity_timeseries,
            plot_batch_parse_rates,
        ):
            pr = fn(bd, str(dest), show=show)
            if pr:
                saved.append(pr)

    cc_log = out / "continuous_curriculum_log.jsonl"
    rows_cc = _read_jsonl(cc_log)
    if rows_cc:
        p = plot_curriculum_jsonl_figure(
            rows_cc,
            "Continuous curriculum (continuous_curriculum_log.jsonl)",
            dest / "curriculum_continuous_log.png",
            show=show,
        )
        if p:
            saved.append(p)

    eff = out / "curriculum_effective_distribution.jsonl"
    rows_e = _read_jsonl(eff)
    if rows_e:
        p = plot_curriculum_jsonl_figure(
            rows_e,
            "Live curriculum effective distribution",
            dest / "curriculum_effective_distribution.png",
            show=show,
        )
        if p:
            saved.append(p)

    gpath = out / "grounded_dataset_summary.json"
    if gpath.is_file():
        p = plot_grounded_dataset_figure(json.loads(gpath.read_text(encoding="utf-8")), str(dest), show=show)
        if p:
            saved.append(p)

    st_path = out / "continuous_curriculum_state.json"
    p = plot_continuous_state_snapshot(st_path, str(dest), show=show)
    if p:
        saved.append(p)

    base_p = out / "base_model_metrics.json"
    trained_p = out / "trained_model_metrics.json"
    if base_p.is_file() and trained_p.is_file():
        eval_data = {
            "base": json.loads(base_p.read_text(encoding="utf-8")),
            "trained": json.loads(trained_p.read_text(encoding="utf-8")),
        }
        plot_eval_comparison(eval_data, save_dir=str(dest), show=show)
        saved.append(dest / "eval_comparison.png")

    return saved


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

    # Support both short keys (smoke-test synthetic) and long keys (train_grpo.py output)
    def _get(d: dict, *keys: str) -> float:
        for k in keys:
            if k in d:
                return d[k]
        return 0.0

    metrics = [
        ("Composite Score",    "mean_composite",    "mean_composite"),
        ("AMAN Reward",        "mean_aman",         "mean_aman_reward"),
        ("DMAN Reward",        "mean_dman",         "mean_dman_reward"),
        ("Coordination Score", "mean_coord",        "mean_coordination"),
        ("Success Rate",       "success_rate",      "success_rate"),
    ]

    labels  = [m[0] for m in metrics]
    base_v  = [_get(base,    m[1], m[2]) for m in metrics]
    train_v = [_get(trained, m[1], m[2]) for m in metrics]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150,
    })

    x   = list(range(len(labels)))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar([i - w/2 for i in x], base_v,  w, label="Base (untrained)", color="#90A4AE", alpha=0.85)
    bars2 = ax.bar([i + w/2 for i in x], train_v, w, label="Trained (GRPO)", color="#1565C0", alpha=0.90)

    for i, (bar, bv, tv) in enumerate(zip(bars2, base_v, train_v)):
        ax.annotate(
            f"{tv:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, tv),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1565C0",
        )
        if bv > 0:
            pct = (tv - bv) / bv * 100
            ax.annotate(
                f"+{pct:.0f}%" if pct >= 0 else f"{pct:.0f}%",
                xy=(i, max(tv, bv) + 0.08),
                ha="center", va="bottom", fontsize=7.5,
                color="#2E7D32" if pct >= 0 else "#C62828",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.25)
    ax.set_title("Multi-Agent ATC: Before vs After GRPO Training",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(0.6, color="#FF8F00", linestyle="--", linewidth=1.2,
               label="Success threshold (0.60)", zorder=0)

    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / "eval_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


def _eval_metric_get(d: dict, *keys: str) -> float:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


EVAL_METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("Composite Score", "mean_composite", "mean_composite"),
    ("AMAN Reward", "mean_aman", "mean_aman_reward"),
    ("DMAN Reward", "mean_dman", "mean_dman_reward"),
    ("Coordination Score", "mean_coord", "mean_coordination"),
    ("Success Rate", "success_rate", "success_rate"),
]


def plot_ablation_eval_bars(
    baseline: Dict[str, Any],
    trained_runs: List[Tuple[str, Dict[str, Any]]],
    save_dir: str | Path,
    *,
    title: str = "Ablation: same baseline vs GRPO runs",
    show: bool = False,
) -> Optional[Path]:
    """Grouped bars: one shared baseline + one trained dict per run label."""
    if not baseline or not trained_runs:
        return None
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[ERROR] pip install matplotlib numpy")
        return None

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    labels = [m[0] for m in EVAL_METRIC_SPECS]
    base_v = [_eval_metric_get(baseline, m[1], m[2]) for m in EVAL_METRIC_SPECS]
    run_vals: List[List[float]] = []
    for _, tr in trained_runs:
        run_vals.append([_eval_metric_get(tr, m[1], m[2]) for m in EVAL_METRIC_SPECS])

    x = np.arange(len(labels))
    n_bars = 1 + len(trained_runs)
    width = min(0.22, 0.9 / (n_bars + 0.5))
    offsets = np.linspace(-(n_bars - 1) * width / 2, (n_bars - 1) * width / 2, n_bars)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x + offsets[0], base_v, width, label="Baseline (pre-GRPO eval)", color="#90A4AE", alpha=0.9)
    colors = ["#1565C0", "#2E7D32", "#6A1B9A", "#C62828", "#00838F"]
    for j, ((name, _), vals) in enumerate(zip(trained_runs, run_vals)):
        ax.bar(x + offsets[j + 1], vals, width, label=name, color=colors[j % len(colors)], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.2)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(0.6, color="#FF8F00", linestyle="--", linewidth=1.0, alpha=0.8)
    plt.tight_layout()
    dest = Path(save_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "ablation_eval_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()
    return out


def plot_ablation_composite_overlay(
    curves: List[Tuple[str, Dict[str, List[float]]]],
    save_dir: str | Path,
    *,
    show: bool = False,
) -> Optional[Path]:
    """Overlay smoothed composite reward for several ``reward_curves.json`` payloads."""
    if len(curves) < 2:
        return None  # need at least two curves to compare
    plt = _setup_matplotlib(show)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1565C0", "#2E7D32", "#6A1B9A", "#C62828", "#00838F"]
    drew = False
    for j, (name, data) in enumerate(curves):
        comp = data.get("composite") or []
        if len(comp) < 2:
            continue
        sm = _smooth([float(x) for x in comp], window=15)
        xs = list(range(len(sm)))
        ax.plot(xs, sm, label=name, color=colors[j % len(colors)], linewidth=2)
        drew = True
    if not drew:
        plt.close()
        return None
    ax.set_xlabel("Training step (per-sample composite index)")
    ax.set_ylabel("Composite (EMA)")
    ax.set_title("Ablation: composite training curves overlaid", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    dest = Path(save_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "ablation_composite_overlay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-agent ATC reward curves")
    parser.add_argument("--input", default=None, help="reward_curves.json from train_grpo.py")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="train_grpo output directory (reward_curves, diagnostics, curriculum logs, eval metrics)",
    )
    parser.add_argument("--eval_results", default=None, help="eval output JSON from eval.py")
    parser.add_argument("--save", default=None, help="Directory to save PNG files")
    parser.add_argument("--no_show", action="store_true", help="Don't display plots interactively")
    args = parser.parse_args()

    show = not args.no_show

    if args.output_dir:
        out = Path(args.output_dir)
        if not out.is_dir():
            print(f"[ERROR] {out} is not a directory")
            sys.exit(1)
        save = args.save or str(out / "plots")
        plot_all_training_artifacts(out, save, show=show)

    if args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"[ERROR] {path} not found")
            sys.exit(1)
        data = json.loads(path.read_text(encoding="utf-8"))
        plot_training_curves(data, save_dir=args.save, show=show)

    if args.eval_results:
        path = Path(args.eval_results)
        if not path.exists():
            print(f"[ERROR] {path} not found")
            sys.exit(1)
        data = json.loads(path.read_text(encoding="utf-8"))
        plot_eval_comparison(data, save_dir=args.save, show=show)

    if not args.input and not args.eval_results and not args.output_dir:
        print("Usage: python training/plot_rewards.py --output_dir outputs/<run>/")
        print("       python training/plot_rewards.py --input reward_curves.json [--save plots/]")
        print("       python training/plot_rewards.py --eval_results eval_output.json [--save plots/]")


if __name__ == "__main__":
    main()
