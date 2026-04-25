#!/usr/bin/env python3
"""Compare SFT-only vs GRPO-only vs SFT+GRPO on the same eval harness (no training).

Requires GPU + Unsloth for model runs. Each mode logs composite and per-role
rewards from ``train_grpo._run_model_episodes`` (AMAN/DMAN-focused eval).

Usage:
  PYTHONPATH=. python training/run_ablation_eval.py \\
    --episodes 5 --output_json outputs/ablation.json \\
    --sft_adapter /path/to/sft \\
    --grpo_adapter /path/to/grpo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(mode: str, *, episodes: int, seed: int, sft: Optional[Path], grpo: Optional[Path]) -> Dict[str, Any]:
    from training.train_grpo import _load_model_with_fallback, _require_training_deps, _run_model_episodes

    torch, FastLanguageModel, _, _ = _require_training_deps()
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

    model_name = os.environ.get("ATC_ABLATION_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    model, tokenizer = _load_model_with_fallback(
        FastLanguageModel,
        model_name,
        max_seq_length=int(os.environ.get("ATC_MAX_SEQ_LEN", "4096")),
    )
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    adapter: Optional[Path] = None
    tag = mode
    if mode == "sft_only":
        adapter = sft
        tag = "SFT only"
    elif mode == "grpo_only":
        adapter = None
        tag = "GRPO only (fresh LoRA)"
    elif mode == "sft_grpo":
        adapter = grpo
        tag = "SFT + GRPO"
    else:
        raise ValueError(mode)

    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json under {adapter}")

    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)

    model.eval()
    metrics = _run_model_episodes(
        model,
        tokenizer,
        n_episodes=episodes,
        tag=tag,
        eval_task_ids=None,
    )
    return {"mode": mode, "metrics": metrics}


def main() -> None:
    p = argparse.ArgumentParser(description="SFT vs GRPO ablation eval (no training)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sft_adapter", type=Path, required=True)
    p.add_argument("--grpo_adapter", type=Path, required=True)
    p.add_argument("--output_json", type=Path, default=Path("outputs/ablation_eval.json"))
    args = p.parse_args()

    out: Dict[str, Any] = {"seed": args.seed, "runs": []}
    for mode in ("sft_only", "grpo_only", "sft_grpo"):
        print(f"\n===== ABLATION: {mode} =====")
        out["runs"].append(
            _run(
                mode,
                episodes=max(1, int(args.episodes)),
                seed=int(args.seed),
                sft=args.sft_adapter,
                grpo=args.grpo_adapter,
            )
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
