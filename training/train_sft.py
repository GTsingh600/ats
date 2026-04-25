#!/usr/bin/env python3
"""Supervised JSON-formatting stage before GRPO (grounded AMAN/DMAN/SUPERVISOR).

Builds gold completions from ``solve_grounded_rule_based`` + simulator supervisor
score, then runs TRL ``SFTTrainer`` on chat-formatted ``text`` rows.

Typical pipeline::

    python training/train_sft.py --model Qwen/Qwen2.5-3B-Instruct --output_dir outputs/atc-sft
    python training/train_grpo.py --adapter_in outputs/atc-sft ... --grounded_curriculum
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# Match kube-sre-gym / long GRPO runs: https://raw.githubusercontent.com/sid-rp/kube-sre-gym/main/train.py
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


def main() -> None:
    import inspect
    import time
    import torch

    from training.sft_data import build_grounded_json_sft_rows, materialize_text_rows
    from training.train_grpo import (
        LORA_ALPHA,
        LORA_TARGETS,
        MAX_SEQ_LEN,
        _configure_runtime_warnings,
        _load_model_with_fallback,
        _prefer_local_model_path,
    )

    p = argparse.ArgumentParser(description="ATC grounded JSON SFT (pre-GRPO)")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir", default=str(ROOT / "outputs" / "atc-sft-json"))
    p.add_argument("--n_episodes", type=int, default=120, help="Virtual episodes → ~3 SFT rows each")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument(
        "--continuous_curriculum_path",
        default=None,
        help="Optional path to continuous_curriculum_state.json (same as GRPO static build).",
    )
    p.add_argument("--curriculum_state", default=None, help="Optional grounded_curriculum_state.json")
    args = p.parse_args()

    _configure_runtime_warnings()

    try:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
    except Exception as exc:
        print(f"[ERROR] unsloth import failed: {exc}")
        sys.exit(1)

    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        print(f"[ERROR] datasets/trl import failed: {exc}")
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str((out / ".hf_datasets_cache").resolve()))

    print(f"[SFT] Building gold rows (n_episodes={args.n_episodes}, seed={args.seed})...")
    rows = build_grounded_json_sft_rows(
        args.n_episodes,
        args.seed,
        continuous_curriculum_path=args.continuous_curriculum_path,
        curriculum_state_path=args.curriculum_state,
    )
    if not rows:
        print("[ERROR] No SFT rows (solver failed for all episodes?). Increase n_episodes or fix tasks.")
        sys.exit(1)
    print(f"[SFT] Built {len(rows)} supervised rows")

    print(f"[SFT] Loading model {args.model!r}...")
    model_source = _prefer_local_model_path(args.model)
    model, tokenizer = _load_model_with_fallback(
        FastLanguageModel,
        model_source,
        max_seq_length=MAX_SEQ_LEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    text_rows = materialize_text_rows(rows, tokenizer)
    dataset = Dataset.from_list(text_rows)

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(args.lora_rank),
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(args.seed),
    )

    sft_kw: dict = {
        "output_dir": str(out),
        "max_steps": int(args.max_steps),
        "per_device_train_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.grad_accum),
        "learning_rate": float(args.lr),
        "logging_steps": int(args.logging_steps),
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": not torch.cuda.is_bf16_supported(),
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "warmup_ratio": 0.03,
        "report_to": "none",
        "dataset_text_field": "text",
        "max_seq_length": int(min(4096, MAX_SEQ_LEN)),
        "save_strategy": "steps",
        "save_steps": max(50, int(args.max_steps) // 4),
        "save_total_limit": 2,
    }
    sig = inspect.signature(SFTConfig.__init__).parameters
    sft_kw = {k: v for k, v in sft_kw.items() if k in sig}

    trainer_kw: dict = {
        "model": model,
        "train_dataset": dataset,
        "args": SFTConfig(**sft_kw),
    }
    tr_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in tr_sig:
        trainer_kw["processing_class"] = tokenizer
    else:
        trainer_kw["tokenizer"] = tokenizer
    if "max_seq_length" in tr_sig:
        trainer_kw["max_seq_length"] = int(min(4096, MAX_SEQ_LEN))

    trainer = SFTTrainer(**trainer_kw)
    if hasattr(trainer.args, "dataloader_num_workers"):
        trainer.args.dataloader_num_workers = 0

    print("[SFT] Training...")
    t0 = time.monotonic()
    trainer.train()
    print(f"[SFT] Done in {(time.monotonic() - t0) / 60:.1f} min")

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "sft_meta.json").write_text(
        '{"role":"json_format_sft","source":"training/sft_data.py"}\n',
        encoding="utf-8",
    )
    print(f"[SFT] Saved adapter + tokenizer → {out}")
    print(f"[SFT] Next: python training/train_grpo.py --adapter_in {out} --grounded_curriculum ...")


if __name__ == "__main__":
    main()
