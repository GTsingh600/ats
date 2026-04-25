#!/bin/bash
#SBATCH --job-name=atc_grpo
#SBATCH --nodes=1
#SBATCH --nodelist=dgxanode03
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/neeraj/atc/logs/%j.out
#SBATCH --error=/home/neeraj/atc/logs/%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "========== JOB START =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "================================"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 2000))

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

ulimit -n 65536 || true

SLURM_BIN=/cm/shared/apps/slurm/current/bin

"${SLURM_BIN}/srun" --mpi=none \
     --container-image /home/hpc_apps/nvidia+pytorch+25.04-py3.sqsh \
     --container-mounts /home/neeraj:/hostd \
     --container-workdir /hostd/atc/repo \
     bash -lc "
        set -euo pipefail

        echo '===== INSIDE CONTAINER ====='
        echo 'PWD:' \$(pwd)
        nvidia-smi || true
        python --version
        python -c \"import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())\"

        # Load W&B key if available
        for env_file in /hostd/atc/repo/.env /hostd/atc/repo/.env.wandb; do
            if [ -f \"\$env_file\" ]; then
                set -a
                . \"\$env_file\"
                set +a
                echo \"Loaded: \$env_file\"
            fi
        done

        # ===== CRITICAL FIX: Delete stale Unsloth compilation cache =====
        echo 'Clearing stale Unsloth compilation cache...'
        find /hostd/atc/repo -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
        # ===== PINNED PACKAGE INSTALLATION =====
        echo '===== INSTALLING PACKAGES ====='
        
        python -m pip uninstall -y vllm vllm-flash-attn 2>/dev/null || true

        python -m pip install --user --upgrade --no-input --no-deps \
            trl==0.16.0 \
            unsloth==2026.4.7 \
            unsloth-zoo==2026.4.9 \
            accelerate==1.13.0 \
            peft==0.19.1 \
            bitsandbytes==0.49.2 \
            xformers==0.0.31 \
            datasets==2.20.0

        python -m pip install --user --upgrade --no-input --no-deps \
            multiprocess==0.70.17 \
            pyarrow-hotfix==0.7 \
            xxhash==3.6.0 \
            huggingface-hub==0.36.2 \
            hf_transfer==0.1.9 \
            tyro==0.9.17

        python -m pip install --user --upgrade --no-input 'wandb>=0.18,<0.20'

        # ===== CRITICAL: Disable torch.compile to avoid Dynamo issues =====
        export TORCH_COMPILE_DISABLE=1
        echo 'TORCH_COMPILE_DISABLE=1 (Dynamo disabled to avoid GRPO compilation error)'

        # ===== VERIFY INSTALLATIONS =====
        echo '===== VERIFYING ====='
        python -c \"
import unsloth
import torch, transformers, trl
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'TRL: {trl.__version__}')
print(f'Unsloth: {unsloth.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
print('All imports OK')
\"

        # ===== TRAINING =====
        echo '===== START TRAINING ====='
        
        export PYTHONPATH=/hostd/atc/repo:\${PYTHONPATH:-}
        export WORLD_SIZE=1
        export RANK=0
        export LOCAL_RANK=0
        export LOCAL_WORLD_SIZE=1
        
        mkdir -p /hostd/atc/outputs /hostd/atc/logs

        # W&B setup
        if [ -n \"\${WANDB_API_KEY:-}\" ]; then
            # Normalize common .env mistakes: assignment prefix and whitespace.
            _wandb_key_raw=\"\${WANDB_API_KEY//$'\\r'/}\"
            _wandb_key_raw=\"\${_wandb_key_raw##*=}\"
            _wandb_key_raw=\"\${_wandb_key_raw//[[:space:]]/}\"

            if [ -n \"\$_wandb_key_raw\" ]; then
                export WANDB_API_KEY=\"\$_wandb_key_raw\"
                export WANDB_MODE=online
                export WANDB_PROJECT="\${WANDB_PROJECT:-atc-multiagent-grpo}"
                echo \"W&B key metadata: raw_len=\${#WANDB_API_KEY} normalized_len=\${#_wandb_key_raw}\"
                echo 'W&B online mode (attempting login)'
                python - <<'PY' || {
import os
import wandb

wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
PY
                    echo 'W&B login failed, switching to offline'
                    export WANDB_MODE=offline
                }
            else
                echo 'WANDB_API_KEY was set but empty after normalization; switching to offline'
                unset WANDB_API_KEY
                export WANDB_MODE=offline
            fi
        else
            export WANDB_MODE=offline
            echo 'W&B offline (no key)'
        fi

        # ===== SMOKE GUN (fast sanity run) =====
        # Runs the full training stack quickly before the real run.
        if [ \"\${SKIP_SMOKE:-0}\" != \"1\" ]; then
            export SMOKE_MODEL="\${SMOKE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
            export SMOKE_OUTPUT_DIR="\${SMOKE_OUTPUT_DIR:-/hostd/atc/outputs/atc-smoke}"
            mkdir -p \"\${SMOKE_OUTPUT_DIR}\"
            echo \"===== SMOKE GUN START (model=\${SMOKE_MODEL}, episodes=1) =====\"
            python training/train_grpo.py \
                --model \"\${SMOKE_MODEL}\" \
                --output_dir \"\${SMOKE_OUTPUT_DIR}\" \
                --episodes 1 \
                --n_generations 2 \
                --seed 42 \
                --no_eval
            echo '===== SMOKE GUN COMPLETE ====='
        else
            echo 'Skipping smoke run because SKIP_SMOKE=1'
        fi
        
        python training/train_grpo.py \
            --model Qwen/Qwen2.5-7B-Instruct \
            --output_dir /hostd/atc/outputs/atc-multiagent \
            --episodes 2 \
            --n_generations 4 \
            --seed 42

        echo '===== TRAINING COMPLETE ====='

        # ===== PLOT GENERATION =====
        echo '===== GENERATING PLOTS ====='
        PLOTS_DIR=/hostd/atc/outputs/plots
        OUT_DIR=/hostd/atc/outputs/atc-multiagent
        mkdir -p \"\${PLOTS_DIR}\"

        # 1. Per-role reward curves
        CURVES=\${OUT_DIR}/reward_curves.json
        if [ -f \"\${CURVES}\" ]; then
            python training/plot_rewards.py \
                --input \"\${CURVES}\" \
                --save \"\${PLOTS_DIR}\" \
                --no_show
            echo \"  training_curves.png → \${PLOTS_DIR}\"
        else
            echo '  [WARN] reward_curves.json not found, skipping training_curves plot'
        fi

        # 2. Before/after comparison (base_model_metrics + trained_model_metrics)
        BASE_M=\${OUT_DIR}/base_model_metrics.json
        TRAINED_M=\${OUT_DIR}/trained_model_metrics.json
        if [ -f \"\${BASE_M}\" ] && [ -f \"\${TRAINED_M}\" ]; then
            python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/hostd/atc/repo")
base    = json.loads(Path("/hostd/atc/outputs/atc-multiagent/base_model_metrics.json").read_text())
trained = json.loads(Path("/hostd/atc/outputs/atc-multiagent/trained_model_metrics.json").read_text())
from training.plot_rewards import plot_eval_comparison
plot_eval_comparison({"base": base, "trained": trained},
                     save_dir="/hostd/atc/outputs/plots", show=False)
print("  eval_comparison.png → /hostd/atc/outputs/plots")
PY
        else
            echo '  [WARN] base/trained metrics not found, skipping eval_comparison plot'
        fi

        echo \"===== PLOTS SAVED TO \${PLOTS_DIR} =====\"
     "

echo "End time: $(date)"
echo "=========== JOB END ==========="