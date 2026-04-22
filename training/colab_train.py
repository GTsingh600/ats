# -*- coding: utf-8 -*-
"""ATC Multi-Agent GRPO Training — Colab Notebook
Run each cell top-to-bottom on a T4 GPU runtime.
Runtime → Change runtime type → T4 GPU
"""

# ══════════════════════════════════════════════════════════════════════════════
# Cell 1 — Mount Drive + Clone Repo
# ══════════════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount("/content/drive")

import subprocess, sys, os

BRANCH     = "multiagent-readme-sync"   # change to "main" once merged
REPO_URL   = "https://github.com/GTsingh600/ats.git"
REPO_DIR   = "/content/ATC"
OUTPUT_DIR = "/content/drive/MyDrive/atc-multiagent"

subprocess.run(["rm", "-rf", REPO_DIR], check=True)
subprocess.run(
    ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, REPO_DIR],
    check=True,
)
os.chdir(REPO_DIR)
print(f"Repo ready: {REPO_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 2 — Install Dependencies (single clean call)
# ══════════════════════════════════════════════════════════════════════════════

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "trl"], check=False)

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
        # Core training stack
        "unsloth[colab-new]",
        "trl==0.15.2",
        "transformers==4.51.3",
        "accelerate>=0.32.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.43.0",
        # Data / utilities
        "datasets>=2.20.0",
        "numpy>=1.26.0",
        "matplotlib>=3.9.0",
        # Environment / API
        "openenv-core[core]>=0.2.3",
        "openai>=1.30.0",
        "fastapi>=0.111.0",
        "pydantic>=2.7.0",
        "uvicorn>=0.29.0",
    ],
    check=True,
)

# Disable wandb noise
os.environ["WANDB_MODE"]             = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Install complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 3 — Smoke-test Imports
# ══════════════════════════════════════════════════════════════════════════════

import torch, trl, transformers
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

print(f"Python      : {sys.version.split()[0]}")
print(f"Torch       : {torch.__version__}")
print(f"TRL         : {trl.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA        : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Repo imports
sys.path.insert(0, REPO_DIR)
from training.dataset import build_episode_dataset
data = build_episode_dataset(n_episodes=2, seed=42)
print(f"\nDataset smoke-test: {len(data)} samples, roles: {sorted({x['agent_role'] for x in data})}")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 4 — Train
#
# --n_generations 2   → safe for T4 (14.5 GB); use 4 if you have A100
# --episodes 50       → ~2 hr on T4; use 200 for full training
# --no_eval           → add this flag to skip before/after model inference
#                        (saves ~15 min per eval run on T4)
# ══════════════════════════════════════════════════════════════════════════════

os.chdir(REPO_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

subprocess.run(
    [
        sys.executable, "training/train_grpo.py",
        "--model",          "Qwen/Qwen2.5-7B-Instruct",
        "--episodes",       "50",
        "--lora_rank",      "16",
        "--n_generations",  "2",      # 2 = T4 safe; 4 = better gradients (A100)
        "--seed",           "42",
        "--output_dir",     OUTPUT_DIR,
        # "--no_eval",       # uncomment to skip the ~15-min before/after model eval
    ],
    check=True,
    cwd=REPO_DIR,
)


# ══════════════════════════════════════════════════════════════════════════════
# Cell 5 — Plot Reward Curves
# ══════════════════════════════════════════════════════════════════════════════

PLOTS_DIR = f"{OUTPUT_DIR}/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

subprocess.run(
    [
        sys.executable, "training/plot_rewards.py",
        "--input", f"{OUTPUT_DIR}/reward_curves.json",
        "--save",  PLOTS_DIR,
        "--no_show",
    ],
    check=False,          # non-fatal if plot file not found
    cwd=REPO_DIR,
)

# Display inline if plots were produced
from pathlib import Path
from IPython.display import display, Image

for png in sorted(Path(PLOTS_DIR).glob("*.png")):
    print(png.name)
    display(Image(str(png)))


# ══════════════════════════════════════════════════════════════════════════════
# Cell 6 — Standalone Eval (optional, run after training)
#
# Compares heuristic baseline vs trained checkpoint across N episodes.
# Already runs automatically inside training (Cell 4) via run_eval=True.
# Use this cell only if you want more episodes or a separate eval run.
# ══════════════════════════════════════════════════════════════════════════════

EVAL_OUT = f"{OUTPUT_DIR}/eval_results.json"

subprocess.run(
    [
        sys.executable, "training/eval.py",
        "--base",     "heuristic-baseline",
        "--trained",  OUTPUT_DIR,
        "--episodes", "5",
        "--output",   EVAL_OUT,
    ],
    check=False,
    cwd=REPO_DIR,
)

# Pretty-print results
import json
if Path(EVAL_OUT).exists():
    results = json.loads(Path(EVAL_OUT).read_text())
    print("\n=== EVAL RESULTS ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 7 — Quick Sanity: Run Heuristic Baseline
#
# No model needed. Verifies the multi-agent environment works end-to-end.
# ══════════════════════════════════════════════════════════════════════════════

subprocess.run(
    [
        sys.executable, "multi_agent/inference.py",
        "--all_tasks",
        "--episodes", "1",
        "--no_generator",
    ],
    check=False,
    cwd=REPO_DIR,
)
