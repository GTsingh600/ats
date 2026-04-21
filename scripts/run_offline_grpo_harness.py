"""Offline ATC training harness for rigid server environments.

Purpose
-------
This is a *stability-first* runtime harness for DGX/A100 environments where:
- internet access may be blocked,
- model files are already cached,
- Unsloth/vLLM are unavailable,
- only standard torch/transformers/peft stack is allowed.

This script intentionally reuses existing project logic for:
- prompt construction (`training.dataset.build_episode_dataset`)
- reward behavior (`training.reward_functions.*`)

It does NOT replace the full GRPO trainer. It is a controlled bridge to validate
loader/runtime/training-loop stability under offline constraints.

Single-command entrypoint example
---------------------------------
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python scripts/run_offline_grpo_harness.py \
  --model Qwen/Qwen2.5-7B-Instruct --episodes 5 --max_steps 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Force offline behavior before importing HF/transformers modules.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("REWARD_FAILURE_MODE", "strict")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.dataset import build_episode_dataset
from training.reward_functions import (
    aman_reward_fn,
    dman_reward_fn,
    generator_reward_fn,
    supervisor_reward_fn,
)

ROLE_TO_REWARD_FN = {
    "AMAN": aman_reward_fn,
    "DMAN": dman_reward_fn,
    "GENERATOR": generator_reward_fn,
    "SUPERVISOR": supervisor_reward_fn,
}


def _resolve_model_path(model_ref: str) -> str:
    """Resolve a model reference to a local path without any network dependency."""
    p = Path(model_ref)
    if p.exists():
        return str(p)

    try:
        return snapshot_download(
            repo_id=model_ref,
            local_files_only=True,
            resume_download=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve local model files for '{model_ref}'. "
            "Provide a local model directory path or ensure cache is present."
        ) from exc


def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _build_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _compute_reward(sample: Dict[str, Any], completion: str) -> float:
    role = sample["agent_role"]
    fn = ROLE_TO_REWARD_FN[role]

    if role == "AMAN":
        return fn(
            [completion],
            task_id=[sample["task_id"]],
            supervisor_profile=[sample["supervisor_profile"]],
            dman_slots_json=[sample.get("dman_slots_json", "[]")],
            atfm_deadlines_json=[sample.get("atfm_deadlines_json", "{}")],
        )[0]

    if role == "DMAN":
        return fn(
            [completion],
            task_id=[sample["task_id"]],
            supervisor_profile=[sample["supervisor_profile"]],
            aman_slots_json=[sample.get("aman_slots_json", "[]")],
            atfm_deadlines_json=[sample.get("atfm_deadlines_json", "{}")],
        )[0]

    if role == "GENERATOR":
        return fn(
            [completion],
            task_id=[sample["task_id"]],
            controller_scores=[float(sample.get("controller_scores", 0.5))],
        )[0]

    # SUPERVISOR
    return fn(
        [completion],
        task_id=[sample["task_id"]],
        supervisor_profile=[sample["supervisor_profile"]],
        merged_plan_json=[sample.get("merged_plan_json", "[]")],
    )[0]


def run(args: argparse.Namespace) -> None:
    model_path = _resolve_model_path(args.model)
    dtype = _select_dtype()

    print(f"[INFO] Offline mode: TRANSFORMERS_OFFLINE={os.getenv('TRANSFORMERS_OFFLINE')}")
    print(f"[INFO] Model path: {model_path}")
    print(f"[INFO] dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    samples = build_episode_dataset(
        n_episodes=args.episodes,
        seed=args.seed,
        include_generator=True,
        include_supervisor=True,
    )
    if not samples:
        raise RuntimeError("Dataset builder returned zero samples")

    max_steps = min(args.max_steps, len(samples))
    print(f"[INFO] Samples available={len(samples)} | running_steps={max_steps}")

    model.train()
    for step in range(max_steps):
        sample = samples[step]
        prompt_text = _build_prompt(tokenizer, sample["prompt"])

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generation pass (eval, deterministic for stability)
        model.eval()
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        reward = float(_compute_reward(sample, completion))
        reward = max(-1.0, min(1.0, reward))

        # Loss pass (train)
        model.train()
        outputs = model(**inputs, labels=inputs["input_ids"])
        ce_loss = outputs.loss

        # Stable scalar weight: higher reward => lower effective loss.
        weight = max(0.1, 1.0 - reward)
        final_loss = ce_loss * weight

        if torch.isnan(final_loss) or torch.isinf(final_loss):
            print(f"[WARN] step={step} unstable loss detected; skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        role = sample["agent_role"]
        print(
            f"step={step:03d} role={role:<10} reward={reward:+.4f} "
            f"ce_loss={ce_loss.item():.4f} weighted_loss={final_loss.item():.4f}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[DONE] Saved adapter/tokenizer to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ATC runtime harness (no unsloth/vllm)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF model id or local model path")
    parser.add_argument("--episodes", type=int, default=5, help="Episode count for dataset generation")
    parser.add_argument("--max_steps", type=int, default=20, help="Max optimization steps to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", default="./outputs/offline-grpo-harness")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
