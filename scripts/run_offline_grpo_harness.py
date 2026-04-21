"""Offline multi-agent ATC trainer (final server runtime path).

This script is designed for rigid DGX/A100 execution:
- fully offline (cached model files only)
- no Unsloth
- no vLLM
- no live Hugging Face metadata/model lookup

It uses the *actual* ATC multi-agent environment and graders by running real
episodes through `MultiAgentATCEnvironment.finalize()` (which computes
composite/coordination outputs via project grading logic).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.generator import ChallengeGenerator
from multi_agent.inference import _build_aman_heuristic, _build_dman_heuristic
from multi_agent.models import AgentRole, SUPERVISOR_PROFILES
from multi_agent.supervisor import SupervisorAgent
from tasks import ordered_tasks, task_catalog
from training.dataset import AMAN_SYSTEM, DMAN_SYSTEM, parse_aman_action, parse_dman_action


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


def _chat_prompt(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate_completion(
    model,
    tokenizer,
    system: str,
    user: str,
    max_length: int,
    max_new_tokens: int,
) -> str:
    prompt = _chat_prompt(tokenizer, system, user)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def _policy_update(
    model,
    tokenizer,
    optimizer,
    system: str,
    user: str,
    completion: str,
    reward: float,
    max_length: int,
) -> Tuple[float, float]:
    # Train on prompt+completion sequence; reward scales gradient magnitude.
    text = _chat_prompt(tokenizer, system, user) + completion
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    batch = {k: v.to(model.device) for k, v in batch.items()}

    outputs = model(**batch, labels=batch["input_ids"])
    ce_loss = outputs.loss
    weight = max(0.1, 1.0 - reward)
    final_loss = ce_loss * weight

    if torch.isnan(final_loss) or torch.isinf(final_loss):
        optimizer.zero_grad(set_to_none=True)
        return float(ce_loss.item()), float("nan")

    final_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(ce_loss.item()), float(final_loss.item())


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
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Avoid generation warnings for ignored sampling params in model config.
    try:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    except Exception:
        pass

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
    catalog = task_catalog()
    tasks = [t.task_id for t in ordered_tasks()]
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in catalog]
        if not tasks:
            raise RuntimeError("No valid task IDs provided in --tasks")

    env = MultiAgentATCEnvironment(seed=args.seed)
    generator = ChallengeGenerator(seed=args.seed)
    supervisor = SupervisorAgent()
    rng = random.Random(args.seed)

    print(f"[INFO] Tasks={tasks}")
    print(f"[INFO] Episodes={args.episodes} (actual grader + actual tasks)")

    global_step = 0
    model.train()
    for ep in range(args.episodes):
        task_id = tasks[ep % len(tasks)]
        base_task = catalog[task_id]
        profile = supervisor.sample_profile(ep)
        sup_desc = SUPERVISOR_PROFILES[profile]["description"]

        if args.use_generator:
            mutated_task, _ = generator.mutate(base_task)
        else:
            mutated_task = base_task

        aman_obs, dman_obs = env.reset(
            task_id=task_id,
            episode_id=ep,
            supervisor_profile=profile,
            mutated_task=mutated_task,
        )
        atfm = env._state.atfm_deadlines

        aman_system = AMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}"
        dman_system = DMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}"

        aman_completion = _generate_completion(
            model,
            tokenizer,
            aman_system,
            aman_obs.to_prompt_text(),
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
        aman_action = parse_aman_action(aman_completion) or _build_aman_heuristic(aman_obs)

        dman_completion = _generate_completion(
            model,
            tokenizer,
            dman_system,
            dman_obs.to_prompt_text(),
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
        dman_action = parse_dman_action(dman_completion) or _build_dman_heuristic(dman_obs, atfm)

        aman_obs2, dman_obs2, _, done = env.step_bid(aman_action, dman_action)

        # Optional one negotiation pass (actual environment path)
        if not done and args.negotiate_rounds > 0:
            aman_completion_2 = _generate_completion(
                model,
                tokenizer,
                aman_system,
                aman_obs2.to_prompt_text(),
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
            dman_completion_2 = _generate_completion(
                model,
                tokenizer,
                dman_system,
                dman_obs2.to_prompt_text(),
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
            aman_action_2 = parse_aman_action(aman_completion_2) or _build_aman_heuristic(aman_obs2)
            dman_action_2 = parse_dman_action(dman_completion_2) or _build_dman_heuristic(dman_obs2, atfm)
            env.step_negotiate(aman_action_2, dman_action_2)

        result = env.finalize()
        if args.use_generator:
            generator.update(result.composite_score)

        # Train only on AMAN/DMAN prompt+completion using actual graded rewards.
        aman_ce, aman_w = _policy_update(
            model,
            tokenizer,
            optimizer,
            aman_system,
            aman_obs.to_prompt_text(),
            aman_completion,
            reward=max(-1.0, min(1.0, float(result.aman_reward))),
            max_length=args.max_length,
        )
        print(
            f"step={global_step:03d} role={'AMAN':<10} reward={result.aman_reward:+.4f} "
            f"ce_loss={aman_ce:.4f} weighted_loss={aman_w:.4f}"
        )
        global_step += 1

        dman_ce, dman_w = _policy_update(
            model,
            tokenizer,
            optimizer,
            dman_system,
            dman_obs.to_prompt_text(),
            dman_completion,
            reward=max(-1.0, min(1.0, float(result.dman_reward))),
            max_length=args.max_length,
        )
        print(
            f"step={global_step:03d} role={'DMAN':<10} reward={result.dman_reward:+.4f} "
            f"ce_loss={dman_ce:.4f} weighted_loss={dman_w:.4f}"
        )
        global_step += 1

        print(
            f"[EP {ep:03d}] task={task_id} composite={result.composite_score:.4f} "
            f"coord={result.per_role.coordination_score:.4f} conflicts={result.per_role.cross_lane_conflicts} "
            f"aman={result.aman_reward:.4f} dman={result.dman_reward:.4f} "
            f"generator={result.generator_reward:.4f} supervisor={result.supervisor_score:.4f}"
        )

        if args.max_steps > 0 and global_step >= args.max_steps:
            print(f"[INFO] Reached --max_steps={args.max_steps}. Stopping early.")
            break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[DONE] Saved adapter/tokenizer to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ATC runtime harness (no unsloth/vllm)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF model id or local model path")
    parser.add_argument("--episodes", type=int, default=5, help="Episode count over real ATC tasks")
    parser.add_argument("--max_steps", type=int, default=0, help="Optional optimizer-step cap (0 = no cap)")
    parser.add_argument("--tasks", default="", help="Comma-separated task IDs (default: all ordered tasks)")
    parser.add_argument("--use_generator", action="store_true", help="Enable adversarial generator mutation")
    parser.add_argument("--negotiate_rounds", type=int, default=1, help="Negotiation rounds to run (0 or 1)")
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
