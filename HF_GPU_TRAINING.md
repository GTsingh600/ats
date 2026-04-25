# Running ATC GRPO Training on Hugging Face GPUs

Three ways, ordered by ease. Pick based on what hardware you need.

---

## Option A — HF Spaces JupyterLab (recommended for training)

Best for: interactive debugging, watching live reward output, iterating fast.

### Step 1 — Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Space name**: `atc-grpo-training`
   - **SDK**: `Docker`  ← not Gradio, not Static
   - **Hardware**: `T4 (free tier)` for 1.5B model, `A10G` for 7B
   - **Visibility**: Private (avoids others using your GPU quota)
3. Click **Create Space**

### Step 2 — Upload a Dockerfile for JupyterLab

In the Space's **Files** tab, create `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y git curl python3.11 python3-pip && \
    ln -s /usr/bin/python3.11 /usr/bin/python

RUN pip install jupyterlab

EXPOSE 7860

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=7860", \
     "--no-browser", "--allow-root", "--NotebookApp.token=''"]
```

Push and wait for the Space to build (~3 min). Then open the JupyterLab URL.

### Step 3 — Inside JupyterLab terminal

```bash
# Clone repo
git clone --branch yashh --depth 1 https://github.com/GTsingh600/ATC.git
cd ATC

# Install deps (correct order — see install notes in notebook)
pip install unsloth==2026.4.7 unsloth-zoo==2026.4.9
pip install trl==0.16.0 accelerate==1.13.0 peft==0.19.1 \
            bitsandbytes==0.49.2 datasets==2.20.0 --no-deps
pip install huggingface-hub==0.36.2 hf_transfer==0.1.9 --no-deps
pip install wandb openenv-core fastapi pydantic uvicorn matplotlib numpy

# Run training (streams live reward output)
PYTHONUNBUFFERED=1 python training/train_grpo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir ./outputs \
    --episodes 150 \
    --n_generations 4 \
    --seed 42
```

Or open `training/atc_colab_kaggle.ipynb` in JupyterLab and run cell-by-cell.

### Step 4 — Save checkpoint to HF Hub

```bash
# After training completes
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./outputs",
    repo_id="YOUR_HF_USERNAME/atc-grpo-adapter",
    repo_type="model",
)
PY
```

---

## Option B — `hf_train.py` script (push-and-run, no UI needed)

Use when: you want to trigger training from your local machine and have results
pushed to HF Hub automatically. Uses a Space as a compute node.

**File**: `training/hf_train.py` (already created alongside this guide)

```bash
# From local machine — pushes code and triggers training on HF Space GPU
python training/hf_train.py \
    --hf_token   hf_xxxx \
    --hf_user    YOUR_USERNAME \
    --model      Qwen/Qwen2.5-1.5B-Instruct \
    --episodes   150
```

What it does:
1. Creates (or updates) a Space called `atc-grpo-runner` on your account
2. Pushes the repo code to the Space
3. The Space runs `train_grpo.py` on startup via `Dockerfile CMD`
4. Reward curves + adapter weights are pushed to `YOUR_USERNAME/atc-grpo-adapter`

---

## Option C — ZeroGPU (free A100, time-limited)

Use when: just want a quick smoke test. **Not suitable for full training.**  
ZeroGPU gives A100 access but caps each GPU call at ~60 seconds.

In a Space with `@spaces.GPU` decorator, you can run at most a few steps.
Good for verifying the env works, not for 150 episodes.

See `spaces/app.py` in this repo — it uses ZeroGPU to demo single episodes.

---

## Hardware quick-reference

| Hardware | VRAM | Cost | Max model | Episodes/hr |
|----------|------|------|-----------|-------------|
| T4 (free) | 16 GB | $0 | 3B (4-bit) | ~100 with 1.5B |
| T4 small | 16 GB | $0.60/hr | 3B (4-bit) | ~100 with 1.5B |
| A10G | 24 GB | $3.15/hr | 7B (4-bit) | ~200 with 7B |
| A100 | 40 GB | $4.13/hr | 13B (4-bit) | ~400 with 7B |

For the competition: **T4 free tier + 1.5B model** is enough to show real learning curves.

---

## Model + output recommendations per hardware

```python
# T4 free / T4 small
MODEL_NAME    = "Qwen/Qwen2.5-1.5B-Instruct"
EPISODES      = 150
N_GENERATIONS = 4
LORA_RANK     = 16

# A10G
MODEL_NAME    = "Qwen/Qwen2.5-7B-Instruct"
EPISODES      = 200
N_GENERATIONS = 4
LORA_RANK     = 16

# A100
MODEL_NAME    = "Qwen/Qwen2.5-7B-Instruct"
EPISODES      = 300
N_GENERATIONS = 6    # bigger group = better advantage estimate
LORA_RANK     = 32
```

---

## Common errors on HF Spaces

| Error | Fix |
|-------|-----|
| `retry` ImportError | Install unsloth **without** `--no-deps` first |
| CUDA OOM | Reduce `N_GENERATIONS` to `2`; use smaller model |
| Space sleeps after 15 min | Set Space to **not sleep** in Settings, or use persistent runner |
| Output lost on restart | Push to HF Hub after each checkpoint (see Step 4) |
| `bitsandbytes` CUDA error | Match CUDA version in Dockerfile to installed PyTorch CUDA build |

---

## Env vars to set in Space Settings → Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `WANDB_API_KEY` | your key | Live reward curves on W&B |
| `WANDB_PROJECT` | `atc-multiagent-grpo` | Project name |
| `HF_TOKEN` | your token | Push checkpoints to Hub |
| `PYTHONUNBUFFERED` | `1` | Live stdout streaming |
| `TORCH_COMPILE_DISABLE` | `1` | Avoid Dynamo crash with GRPO |
