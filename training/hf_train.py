"""
HF Spaces GPU training launcher.

Creates (or updates) a private HF Space that runs train_grpo.py on GPU,
then pushes the trained adapter + reward curves back to HF Hub.

Usage:
    python training/hf_train.py \
        --hf_token   hf_xxxx \
        --hf_user    your_username \
        --model      Qwen/Qwen2.5-1.5B-Instruct \
        --episodes   150 \
        --hardware   t4-small

The Space repo will be:  https://huggingface.co/spaces/<hf_user>/atc-grpo-runner
The model repo will be:  https://huggingface.co/<hf_user>/atc-grpo-adapter
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ── Dockerfile written into the Space ─────────────────────────────────────────

def _dockerfile(
    repo_url: str,
    repo_branch: str,
    model: str,
    episodes: int,
    n_generations: int,
    lora_rank: int,
    seed: int,
    hf_output_repo: str,
) -> str:
    # Secrets (GITHUB_TOKEN, HF_TOKEN) are NOT available at Docker build time —
    # they only exist at container runtime. So we:
    #   Build time  → install OS packages + pip deps (cached across restarts)
    #   Runtime CMD → git clone (uses GITHUB_TOKEN secret) then train
    return textwrap.dedent(f"""\
        FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

        ENV DEBIAN_FRONTEND=noninteractive
        ENV PYTHONUNBUFFERED=1
        ENV TORCH_COMPILE_DISABLE=1
        ENV TOKENIZERS_PARALLELISM=false

        RUN apt-get update && apt-get install -y \\
            git curl python3.11 python3-pip \\
            && ln -sf /usr/bin/python3.11 /usr/bin/python \\
            && ln -sf /usr/bin/python3.11 /usr/bin/python3 \\
            && rm -rf /var/lib/apt/lists/*

        # ── Pre-install Python deps at build time (cached) ───────────────────
        # unsloth must be installed WITHOUT --no-deps so it pins the right
        # transformers version (avoids the retry ImportError at runtime).
        RUN pip install --no-cache-dir unsloth==2026.4.7 unsloth-zoo==2026.4.9
        RUN pip install --no-cache-dir --no-deps \\
            trl==0.16.0 accelerate==1.13.0 peft==0.19.1 \\
            bitsandbytes==0.49.2 datasets==2.20.0
        RUN pip install --no-cache-dir --no-deps \\
            huggingface-hub==0.36.2 hf_transfer==0.1.9 tyro==0.9.17
        RUN pip install --no-cache-dir \\
            wandb openenv-core fastapi pydantic uvicorn matplotlib numpy

        WORKDIR /app

        # ── Runtime entrypoint ────────────────────────────────────────────────
        # GITHUB_TOKEN and HF_TOKEN are injected as Space secrets at runtime.
        # Public repo: GITHUB_TOKEN not needed (leave blank in Space settings).
        # Private repo: set GITHUB_TOKEN in Space Settings -> Variables & Secrets.
        CMD bash -c ' \\
            set -e; \\
            echo "=== Cloning repo ==="; \\
            if [ -n "${{GITHUB_TOKEN}}" ]; then \\
                CLONE_URL="https://${{GITHUB_TOKEN}}@github.com/{repo_url.replace("https://github.com/", "")}"; \\
            else \\
                CLONE_URL="{repo_url}"; \\
            fi; \\
            git clone --branch {repo_branch} --depth 1 "$CLONE_URL" /app/repo; \\
            cd /app/repo; \\
            echo "=== Starting training ==="; \\
            python training/train_grpo.py \\
                --model        "{model}" \\
                --output_dir   /app/outputs \\
                --episodes     {episodes} \\
                --n_generations {n_generations} \\
                --lora_rank    {lora_rank} \\
                --seed         {seed}; \\
            echo "=== Pushing outputs ==="; \\
            python /app/repo/training/hf_push_outputs.py \\
                --output_dir /app/outputs \\
                --repo_id    "{hf_output_repo}"; \\
        '
    """)


# ── Script that runs inside the Space to push results ─────────────────────────

PUSH_SCRIPT = textwrap.dedent("""\
    \"\"\"Push training outputs to HF Hub model repo.\"\"\"
    import argparse, os
    from pathlib import Path
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--repo_id',    required=True)
    args = parser.parse_args()

    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    api   = HfApi(token=token)

    api.create_repo(repo_id=args.repo_id, repo_type='model', exist_ok=True, private=True)
    api.upload_folder(
        folder_path=args.output_dir,
        repo_id=args.repo_id,
        repo_type='model',
        ignore_patterns=['*.bin.tmp', '*.safetensors.tmp'],
    )
    print(f'Pushed outputs to https://huggingface.co/{args.repo_id}')
""")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], **kw) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)


def _hf_api(token: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] pip install huggingface_hub")
        sys.exit(1)
    return HfApi(token=token)


def _ensure_repo(api, repo_id: str, repo_type: str) -> str:
    kwargs = dict(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=True)
    if repo_type == "space":
        kwargs["space_sdk"] = "docker"
    url = api.create_repo(**kwargs)
    print(f"Repo ready: {url}")
    return url


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Launch ATC GRPO on HF Spaces GPU")
    parser.add_argument("--hf_token",      required=True,  help="HuggingFace write token")
    parser.add_argument("--hf_user",       required=True,  help="HF username")
    parser.add_argument("--repo_url",      default="https://github.com/GTsingh600/ats.git")
    parser.add_argument("--repo_branch",   default="main")
    parser.add_argument("--model",         default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--episodes",      type=int, default=150)
    parser.add_argument("--n_generations", type=int, default=4)
    parser.add_argument("--lora_rank",     type=int, default=16)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--hardware",      default="t4-small",
                        help=("HF Spaces hardware tier. Common values: "
                              "t4-small, t4-medium, a10g-small, a10g-large, "
                              "a100-large, l40s-small, l40s-large. "
                              "HF validates the value — if wrong, the Space will fail to start."))
    parser.add_argument("--space_name",    default="atc-grpo-runner")
    parser.add_argument("--adapter_name",  default="atc-grpo-adapter")
    parser.add_argument("--wandb_key",     default="", help="Optional W&B API key")
    parser.add_argument("--github_token",  default="",
                        help="GitHub PAT for private repo clone. "
                             "Leave empty if repo is public.")
    args = parser.parse_args()

    space_repo_id   = f"{args.hf_user}/{args.space_name}"
    adapter_repo_id = f"{args.hf_user}/{args.adapter_name}"

    print(f"\n{'='*60}")
    print(f"  ATC GRPO — HuggingFace Spaces launcher")
    print(f"  Space:    https://huggingface.co/spaces/{space_repo_id}")
    print(f"  Adapter:  https://huggingface.co/{adapter_repo_id}")
    print(f"  Hardware: {args.hardware}")
    print(f"  Model:    {args.model}")
    print(f"  Episodes: {args.episodes}")
    print(f"{'='*60}\n")

    api = _hf_api(args.hf_token)

    # ── 1. Create Space repo ──────────────────────────────────────────────────
    print("[1/4] Creating Space repo...")
    _ensure_repo(api, space_repo_id, "space")

    # ── 2. Write Space files to a temp dir and push ───────────────────────────
    print("[2/4] Uploading Dockerfile and helper script to Space...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Dockerfile
        (tmp / "Dockerfile").write_text(
            _dockerfile(
                repo_url=args.repo_url,
                repo_branch=args.repo_branch,
                model=args.model,
                episodes=args.episodes,
                n_generations=args.n_generations,
                lora_rank=args.lora_rank,
                seed=args.seed,
                hf_output_repo=adapter_repo_id,
            ),
            encoding="utf-8",
        )

        # Push script (runs inside container after training)
        (tmp / "hf_push_outputs.py").write_text(PUSH_SCRIPT, encoding="utf-8")

        # README for Space (required by HF)
        (tmp / "README.md").write_text(textwrap.dedent(f"""\
            ---
            title: ATC GRPO Runner
            sdk: docker
            hardware: {args.hardware}
            ---
            # ATC GRPO Training Runner

            Runs `train_grpo.py` on GPU and pushes the adapter to
            [{adapter_repo_id}](https://huggingface.co/{adapter_repo_id}).

            Model: `{args.model}` · Episodes: `{args.episodes}`
        """), encoding="utf-8")

        api.upload_folder(
            folder_path=str(tmp),
            repo_id=space_repo_id,
            repo_type="space",
        )

    # ── 3. Set hardware and secrets ───────────────────────────────────────────
    print(f"[3/4] Setting hardware to {args.hardware}...")
    try:
        api.request_space_hardware(repo_id=space_repo_id, hardware=args.hardware)
        print(f"  Hardware set to {args.hardware}")
    except Exception as e:
        print(f"  [WARN] Could not set hardware automatically: {e}")
        print(f"  Set it manually: https://huggingface.co/spaces/{space_repo_id}/settings")

    # Inject secrets — available to the container at runtime (not at build time)
    try:
        secrets_added = ["HF_TOKEN"]
        api.add_space_secret(repo_id=space_repo_id, key="HF_TOKEN", value=args.hf_token)
        if args.github_token:
            api.add_space_secret(repo_id=space_repo_id, key="GITHUB_TOKEN", value=args.github_token)
            secrets_added.append("GITHUB_TOKEN")
        if args.wandb_key:
            api.add_space_secret(repo_id=space_repo_id, key="WANDB_API_KEY", value=args.wandb_key)
            secrets_added.append("WANDB_API_KEY")
        print(f"  Secrets added: {', '.join(secrets_added)}")
    except Exception as e:
        print(f"  [WARN] Could not add secrets automatically: {e}")
        print(f"  Add HF_TOKEN (and GITHUB_TOKEN if repo is private) in Space settings.")

    # ── 4. Done ───────────────────────────────────────────────────────────────
    print("[4/4] Done.\n")
    print("Next steps:")
    print(f"  1. Watch training logs: https://huggingface.co/spaces/{space_repo_id}/logs")
    print(f"  2. Adapter saved to:    https://huggingface.co/{adapter_repo_id}")
    print(f"  3. Download adapter:")
    print(f"       huggingface-cli download {adapter_repo_id} --local-dir ./hf-adapter")
    print()
    print("Hardware upgrade (if needed):")
    print(f"  Go to https://huggingface.co/spaces/{space_repo_id}/settings → Hardware")
    print()
    print("Restart training after code changes:")
    print(f"  python training/hf_train.py --hf_token ... --hf_user {args.hf_user}")


if __name__ == "__main__":
    main()
