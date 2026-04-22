# -*- coding: utf-8 -*-
"""ATC Multi-Agent GRPO Training — Colab Notebook
Runtime → Change runtime type → T4 GPU
Run cells top-to-bottom.
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
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print(f"Repo ready: {REPO_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 2 — Install Dependencies
# ══════════════════════════════════════════════════════════════════════════════

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "trl"], check=False)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
        "unsloth[colab-new]",
        "trl==0.15.2",
        "transformers==4.51.3",
        "accelerate>=0.32.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.43.0",
        "datasets>=2.20.0",
        "numpy>=1.26.0",
        "matplotlib>=3.9.0",
        "openenv-core[core]>=0.2.3",
        "openai>=1.30.0",
        "fastapi>=0.111.0",
        "pydantic>=2.7.0",
        "uvicorn>=0.29.0",
    ],
    check=True,
)

os.environ["WANDB_MODE"]             = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Install complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 3 — Smoke-test Imports
# (unsloth MUST be imported before torch/trl/transformers)
# ══════════════════════════════════════════════════════════════════════════════

import unsloth                          # ← first, enables Unsloth kernel patches
import torch
import trl
import transformers
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

print(f"Python      : {sys.version.split()[0]}")
print(f"Torch       : {torch.__version__}")
print(f"TRL         : {trl.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA        : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"VRAM        : {props.total_memory / 1e9:.1f} GB")

# Repo imports
from training.dataset import build_episode_dataset
data = build_episode_dataset(n_episodes=2, seed=42)
print(f"\nDataset smoke: {len(data)} samples | roles: {sorted({x['agent_role'] for x in data})}")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 4 — Train  (in-process — full tracebacks visible)
#
# T4 safe settings:  N_GENERATIONS=2, BATCH_SIZE=2, GRAD_ACCUM=4
#   → effective batch = 8, group size = 2
# A100 / better VRAM: set N_GENERATIONS=4 for more stable advantage estimates
#
# run_eval=True  → runs base-model eval BEFORE training and trained-model eval
#                  AFTER training, prints before/after comparison table.
#                  Each eval pass ≈ 15 min on T4 (3 model inference episodes).
#                  Set run_eval=False to skip and save ~30 min total.
# ══════════════════════════════════════════════════════════════════════════════

import training.train_grpo as _grpo

# Override memory-critical constants for T4 before training starts
_grpo.N_GENERATIONS = 2   # 2 = T4 safe; 4 = better gradient quality (A100)
_grpo.BATCH_SIZE    = 2   # must stay divisible by N_GENERATIONS
_grpo.GRAD_ACCUM    = 4   # effective batch = BATCH_SIZE * GRAD_ACCUM = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

_grpo.train(
    model_name  = "Qwen/Qwen2.5-7B-Instruct",
    output_dir  = OUTPUT_DIR,
    n_episodes  = 50,       # ~2 hr on T4; use 200 for full training
    lora_rank   = 16,
    seed        = 42,
    run_eval    = True,     # set False to skip before/after model inference
)


# ══════════════════════════════════════════════════════════════════════════════
# Cell 5 — Plot Reward Curves
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path
from IPython.display import display, Image

PLOTS_DIR = f"{OUTPUT_DIR}/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

subprocess.run(
    [
        sys.executable, "training/plot_rewards.py",
        "--input",   f"{OUTPUT_DIR}/reward_curves.json",
        "--save",    PLOTS_DIR,
        "--no_show",
    ],
    check=False,
    cwd=REPO_DIR,
)

for png in sorted(Path(PLOTS_DIR).glob("*.png")):
    print(png.name)
    display(Image(str(png)))


# ══════════════════════════════════════════════════════════════════════════════
# Cell 6 — Standalone Eval  (optional — already runs inside Cell 4)
#
# Use this only if you want more episodes or re-run eval separately.
# ══════════════════════════════════════════════════════════════════════════════

import json

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

if Path(EVAL_OUT).exists():
    results = json.loads(Path(EVAL_OUT).read_text())
    print("\n=== EVAL RESULTS ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 7 — Heuristic Sanity Check  (no model needed)
# Verifies multi-agent environment works end-to-end.
# ══════════════════════════════════════════════════════════════════════════════

from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.generator import ChallengeGenerator
from multi_agent.supervisor import SupervisorAgent
from multi_agent.inference import run_episode

env = MultiAgentATCEnvironment(seed=0)
gen = ChallengeGenerator(seed=0)
sup = SupervisorAgent()

result = run_episode(
    task_id      = "bengaluru_irrops_hard",
    client       = None,          # heuristic mode — no LLM
    env          = env,
    generator    = gen,
    supervisor   = sup,
    episode_id   = 0,
    use_generator= False,
)
print(f"\nHeuristic sanity: composite={result['composite']:.3f} "
      f"aman={result['aman_reward']:.3f} dman={result['dman_reward']:.3f} "
      f"conflicts={result['conflicts']}")
