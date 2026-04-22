"""Multi-Agent ATC Training with DAPO/GRPO + Unsloth.

Architecture overview:
  - Single LLM (Qwen2.5-7B-Instruct, 4-bit LoRA) plays 4 roles via system prompts
  - DAPO (Dynamic Asymmetric Policy Optimization, ByteDance 2025) when supported by
    the installed TRL version; falls back gracefully to vanilla GRPO otherwise.
    DAPO fixes entropy collapse via asymmetric clipping and zero-variance group
    filtering — typically 15–25% more stable training vs vanilla GRPO.
  - Group-relative advantage: A_i = (r_i - mean(r_group)) / (std(r_group) + ε)
  - Four independent reward functions (AMAN, DMAN, GENERATOR, SUPERVISOR)
  - Potential-based reward shaping within each role (policy-gradient safe)
  - Adaptive curriculum: generator escalates difficulty as agents improve

DAPO vs GRPO key differences (when TRL supports loss_type="dapo"):
  - Asymmetric clipping: epsilon_low=0.2, epsilon_high=0.28 (harder clip on
    negative advantages prevents large policy regression from bad rollouts)
  - filter_groups=True: drops groups where all rollouts have equal reward
    (zero-variance groups give no training signal and waste compute)
  - Lower KL coefficient: more exploration room for multi-agent coordination

Training loop:
  Episode → Generator mutates task → AMAN bids → DMAN bids →
  Negotiate (if conflicts) → Grade → Per-agent DAPO/GRPO update

Colab T4 resource profile:
  Model:        Qwen2.5-7B-Instruct, 4-bit QLoRA
  LoRA rank:    16 (q_proj, v_proj, k_proj, o_proj)
  Batch size:   1, gradient accumulation 4 → effective batch 4
  Generations:  4 per prompt (GRPO group size)
  Max tokens:   512 per completion
  Training:     ~200 episodes ≈ 800 samples ≈ 2 hr on T4

Usage:
  python training/train_grpo.py [--episodes 200] [--model Qwen/Qwen2.5-7B-Instruct]

Colab one-liner:
  !python training/train_grpo.py --episodes 100 --output_dir /content/atc-multiagent
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Lazy imports (allow importing module without GPU) ─────────────────────────
def _require_training_deps():
    if sys.version_info >= (3, 14):
        print(
            "[ERROR] Python 3.14 is not currently supported by the GRPO stack "
            "(trl/transformers/tokenizers/mergekit wheels)."
        )
        print("Use Python 3.11 or 3.12 for training runs.")
        print("Recommended: run training on your remote A100 Jupyter environment.")
        sys.exit(1)

    try:
        import torch
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"[ERROR] Training dependencies missing: {e}")
        print("Install core deps: pip install trl torch transformers")
        print("If using this script as-is, also install: pip install unsloth")
        print("Or with uv: uv sync --extra training --extra training-unsloth")
        sys.exit(1)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("[ERROR] 'unsloth' is not installed.")
        print("This training script currently requires Unsloth for 4-bit QLoRA loading.")
        print("Install: pip install unsloth")
        print("Or with uv: uv sync --extra training --extra training-unsloth")
        sys.exit(1)

    return torch, FastLanguageModel, GRPOConfig, GRPOTrainer


from training.dataset import (
    build_episode_dataset,
    parse_aman_action,
    parse_dman_action,
)
from training.reward_functions import (
    aman_reward_fn,
    dman_reward_fn,
    generator_reward_fn,
    supervisor_reward_fn,
)
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.generator import ChallengeGenerator
from multi_agent.models import AgentRole, SupervisorProfileName
from multi_agent.supervisor import SupervisorAgent
from tasks import task_catalog, ordered_tasks
from engine import simulate_plan
from graders import grade_task


# ── Default hyperparameters (tuned for Colab T4) ─────────────────────────────

DEFAULT_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT = "./outputs/atc-multiagent"
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_TARGETS   = ["q_proj", "v_proj", "k_proj", "o_proj"]
MAX_SEQ_LEN    = 4096
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.7
N_GENERATIONS  = 2    # safer default for modern GRPOTrainer on Colab GPUs
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
LR             = 5e-5
KL_COEFF       = 0.01  # KL penalty — lowered for DAPO (more exploration room)
WARMUP_RATIO   = 0.05

# DAPO-specific (used only when the installed TRL version supports them)
DAPO_EPSILON_HIGH  = 0.28   # asymmetric upper clip (vs standard 0.2 lower clip)
DAPO_FILTER_GROUPS = True   # drop zero-variance groups (no training signal)


# ── Role-dispatch reward table ────────────────────────────────────────────────

REWARD_FN_DISPATCH = {
    AgentRole.AMAN.value:       aman_reward_fn,
    AgentRole.DMAN.value:       dman_reward_fn,
    AgentRole.GENERATOR.value:  generator_reward_fn,
    AgentRole.SUPERVISOR.value: supervisor_reward_fn,
}


def _reward_failure_mode() -> str:
    mode = os.getenv("REWARD_FAILURE_MODE", "strict").strip().lower()
    if mode not in {"strict", "penalize"}:
        return "strict"
    return mode


def _config_supports(param: str, config_cls) -> bool:
    try:
        return param in inspect.signature(config_cls.__init__).parameters
    except Exception:
        return False


def _trainer_supports(param: str, trainer_cls) -> bool:
    try:
        return param in inspect.signature(trainer_cls.__init__).parameters
    except Exception:
        return False


def _resolve_num_generations(batch_size: int, requested: int) -> int:
    requested = max(1, requested)
    batch_size = max(1, batch_size)
    if batch_size % requested == 0:
        return requested
    for candidate in range(min(requested, batch_size), 0, -1):
        if batch_size % candidate == 0:
            return candidate
    return 1


def _select_sample_value(value: Any, index: int) -> Any:
    if isinstance(value, list):
        if not value:
            return None
        if index < len(value):
            return value[index]
        return value[-1]
    return value


def combined_reward_fn(completions: List[str], **kwargs) -> List[float]:
    """Unified reward dispatcher — routes to per-role reward function.

    TRL GRPOTrainer calls this with a batch of completions.
    kwargs contains per-sample metadata from the dataset.
    """
    roles = kwargs.get("agent_role", [AgentRole.AMAN.value] * len(completions))
    if not isinstance(roles, list):
        roles = [roles] * len(completions)
    elif len(roles) < len(completions):
        roles = roles + [roles[-1] if roles else AgentRole.AMAN.value] * (len(completions) - len(roles))
    rewards: List[float] = []
    failure_mode = _reward_failure_mode()

    for i, (completion, role) in enumerate(zip(completions, roles)):
        fn = REWARD_FN_DISPATCH.get(role, aman_reward_fn)
        # Build single-item kwargs for this sample
        sample_kwargs = {k: [_select_sample_value(v, i)] for k, v in kwargs.items()}
        try:
            r = fn([completion], **sample_kwargs)
            if not r:
                raise RuntimeError(f"reward function returned empty list for role={role}")
            rewards.append(r[0])
        except Exception as exc:
            message = f"reward_fn({role}) failed at sample_index={i}: {exc}"
            if failure_mode == "strict":
                raise RuntimeError(message) from exc
            print(f"[WARN] {message}")
            rewards.append(-1.0)

    return rewards


# ── Training entry point ──────────────────────────────────────────────────────

def train(
    model_name:    str  = DEFAULT_MODEL,
    output_dir:    str  = DEFAULT_OUTPUT,
    n_episodes:    int  = 200,
    lora_rank:     int  = LORA_RANK,
    seed:          int  = 42,
    push_to_hub:   bool = False,
    hub_model_id:  Optional[str] = None,
) -> None:
    torch, FastLanguageModel, GRPOConfig, GRPOTrainer = _require_training_deps()
    num_generations = _resolve_num_generations(BATCH_SIZE, N_GENERATIONS)
    if num_generations != N_GENERATIONS:
        print(
            f"[WARN] Adjusting num_generations from {N_GENERATIONS} to {num_generations} "
            f"to satisfy current GRPO batch-size constraints."
        )

    print(f"\n{'='*60}")
    print(f"  ATC Multi-Agent GRPO Training")
    print(f"  Model:    {model_name}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Output:   {output_dir}")
    print(f"  Device:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    # ── 1. Load model with Unsloth 4-bit QLoRA ────────────────────────────────
    print("[1/5] Loading model with Unsloth 4-bit QLoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves ~30% VRAM
        random_state=seed,
    )
    print(f"    LoRA rank={lora_rank}, targets={LORA_TARGETS}")
    print(f"    Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── 2. Build training dataset ─────────────────────────────────────────────
    print(f"\n[2/5] Building {n_episodes}-episode multi-agent dataset...")
    t0 = time.time()
    dataset_raw = build_episode_dataset(
        n_episodes=n_episodes,
        seed=seed,
        include_generator=True,
        include_supervisor=True,
    )
    print(f"    Dataset: {len(dataset_raw)} training samples ({time.time()-t0:.1f}s)")

    # Role breakdown
    role_counts: Dict[str, int] = {}
    for s in dataset_raw:
        r = s.get("agent_role", "unknown")
        role_counts[r] = role_counts.get(r, 0) + 1
    for role, count in role_counts.items():
        print(f"    {role}: {count} samples")

    # Convert to HF Dataset
    try:
        from datasets import Dataset
        dataset = Dataset.from_list(dataset_raw)
    except ImportError:
        print("[ERROR] pip install datasets")
        sys.exit(1)

    # ── 3. GRPO config ────────────────────────────────────────────────────────
    print(f"\n[3/5] Configuring GRPO (group_size={num_generations})...")
    grpo_kwargs = {
        "num_generations": num_generations,
        "temperature": TEMPERATURE,
        "learning_rate": LR,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "num_train_epochs": 1,
        "warmup_ratio": WARMUP_RATIO,
        "logging_steps": 10,
        "output_dir": output_dir,
        "report_to": "wandb" if _wandb_available() else "none",
        "run_name": f"atc-multiagent-grpo-{int(time.time())}",
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
    }
    if _config_supports("max_completion_length", GRPOConfig):
        grpo_kwargs["max_completion_length"] = MAX_NEW_TOKENS
    elif _config_supports("max_new_tokens", GRPOConfig):
        grpo_kwargs["max_new_tokens"] = MAX_NEW_TOKENS

    if _config_supports("beta", GRPOConfig):
        grpo_kwargs["beta"] = KL_COEFF
    elif _config_supports("kl_coeff", GRPOConfig):
        grpo_kwargs["kl_coeff"] = KL_COEFF

    if _config_supports("use_vllm", GRPOConfig):
        grpo_kwargs["use_vllm"] = False

    # ── DAPO upgrade (graceful fallback to vanilla GRPO if TRL is older) ──────
    # DAPO (ByteDance 2025): asymmetric clipping + zero-variance group filtering
    # prevents entropy collapse that vanilla GRPO suffers in multi-agent settings.
    _dapo_active = False
    if _config_supports("loss_type", GRPOConfig):
        grpo_kwargs["loss_type"] = "dapo"
        _dapo_active = True
    if _config_supports("epsilon_high", GRPOConfig):
        grpo_kwargs["epsilon_high"] = DAPO_EPSILON_HIGH
    if _config_supports("filter_groups", GRPOConfig):
        grpo_kwargs["filter_groups"] = DAPO_FILTER_GROUPS

    print(f"    Loss type: {'DAPO (asymmetric clip + group filter)' if _dapo_active else 'GRPO (vanilla — upgrade TRL for DAPO)'}")

    grpo_config = GRPOConfig(**grpo_kwargs)

    # ── 4. Build per-role reward logging callbacks ────────────────────────────
    print("\n[4/5] Setting up reward logging callbacks...")

    reward_log: Dict[str, List[float]] = {
        "AMAN": [], "DMAN": [], "GENERATOR": [], "SUPERVISOR": [], "composite": []
    }

    class RewardLogger:
        __name__ = "combined_reward_fn"

        def __call__(self, *args, **kwargs):
            # TRL <0.17: reward_func(completions=..., **kwargs)
            # TRL >=0.17: reward_func(prompts, completions, **kwargs)
            if len(args) >= 2:
                completions = args[1]
            elif args:
                completions = args[0]
            else:
                completions = kwargs.pop("completions", [])
            kwargs.pop("prompts", None)
            rewards = combined_reward_fn(completions, **kwargs)
            roles = kwargs.get("agent_role", [])
            if not isinstance(roles, list):
                roles = [roles] * len(rewards)
            elif len(roles) < len(rewards):
                roles = roles + [roles[-1] if roles else AgentRole.AMAN.value] * (len(rewards) - len(roles))
            for r, role in zip(rewards, roles):
                if role in reward_log:
                    reward_log[role].append(r)
                reward_log["composite"].append(r)
            return rewards

    reward_logger = RewardLogger()

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("\n[5/5] Starting GRPO training...")
    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "reward_funcs": [reward_logger],
        "train_dataset": dataset,
    }
    if _trainer_supports("args", GRPOTrainer):
        trainer_kwargs["args"] = grpo_config
    else:
        trainer_kwargs["config"] = grpo_config

    trainer = GRPOTrainer(**trainer_kwargs)

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save reward curves for demo
    curves_path = Path(output_dir) / "reward_curves.json"
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    with open(curves_path, "w") as f:
        json.dump(reward_log, f, indent=2)
    print(f"Reward curves saved to {curves_path}")

    # Print final stats
    _print_final_stats(reward_log)

    if push_to_hub and hub_model_id:
        print(f"\nPushing to Hub: {hub_model_id}")
        trainer.push_to_hub(hub_model_id)

    return trainer


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
    model_name_or_path: str,
    n_episodes:         int = 20,
    seed:               int = 99,
) -> Dict[str, Any]:
    """Run evaluation loop and return per-role reward statistics.

    Used to produce the before/after comparison for hackathon demo.
    """
    torch, FastLanguageModel, _, _ = _require_training_deps()
    from transformers import pipeline

    print(f"\nEvaluating {model_name_or_path} on {n_episodes} episodes...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name_or_path,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    env       = MultiAgentATCEnvironment(seed=seed)
    generator = ChallengeGenerator(seed=seed)
    supervisor = SupervisorAgent()
    catalog   = task_catalog()
    task_list = list(ordered_tasks())

    import random
    rng = random.Random(seed)

    results = {
        "aman_rewards":      [],
        "dman_rewards":      [],
        "composite_scores":  [],
        "conflict_counts":   [],
        "emergency_handled": [],
        "coordination_scores": [],
        "negotiation_rounds":  [],
        "generator_difficulty": [],
    }

    for ep in range(n_episodes):
        base_task = rng.choice(task_list)
        profile   = supervisor.sample_profile(ep)
        mutated_task, is_solvable = generator.mutate(base_task)

        aman_obs, dman_obs = env.reset(
            episode_id=ep,
            supervisor_profile=profile,
            mutated_task=mutated_task,
        )

        from training.dataset import AMAN_SYSTEM, DMAN_SYSTEM, SUPERVISOR_PROFILES
        sup_desc = SUPERVISOR_PROFILES[profile]["description"]

        # Generate AMAN action
        aman_prompt = _format_chat(
            system=AMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}",
            user=aman_obs.to_prompt_text(),
            tokenizer=tokenizer,
        )
        aman_completion = _generate(model, tokenizer, aman_prompt)
        aman_action = parse_aman_action(aman_completion)

        # Generate DMAN action
        dman_prompt = _format_chat(
            system=DMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}",
            user=dman_obs.to_prompt_text(),
            tokenizer=tokenizer,
        )
        dman_completion = _generate(model, tokenizer, dman_prompt)
        dman_action = parse_dman_action(dman_completion)

        if not aman_action or not dman_action:
            continue

        aman_obs, dman_obs, _, done = env.step_bid(aman_action, dman_action)
        if not done:
            env.step_negotiate(aman_action, dman_action)

        result = env.finalize()
        generator.update(result.composite_score)

        results["aman_rewards"].append(result.aman_reward)
        results["dman_rewards"].append(result.dman_reward)
        results["composite_scores"].append(result.composite_score)
        results["conflict_counts"].append(result.per_role.cross_lane_conflicts)
        results["coordination_scores"].append(result.per_role.coordination_score)
        results["negotiation_rounds"].append(result.negotiation_rounds)
        results["generator_difficulty"].append(generator.difficulty_level)

        emg_handled = (
            result.per_role.emergency_arrivals_ok
            + result.per_role.emergency_departures_ok
        )
        results["emergency_handled"].append(emg_handled)

        print(
            f"  ep{ep:3d} | composite={result.composite_score:.3f} | "
            f"AMAN={result.aman_reward:.3f} | DMAN={result.dman_reward:.3f} | "
            f"coord={result.per_role.coordination_score:.3f} | "
            f"gen_lvl={generator.difficulty_level}"
        )

    # Summary statistics
    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    summary = {
        "mean_composite":     round(_mean(results["composite_scores"]),  3),
        "mean_aman_reward":   round(_mean(results["aman_rewards"]),       3),
        "mean_dman_reward":   round(_mean(results["dman_rewards"]),       3),
        "mean_coordination":  round(_mean(results["coordination_scores"]), 3),
        "mean_conflicts":     round(_mean(results["conflict_counts"]),    2),
        "mean_emg_handled":   round(_mean(results["emergency_handled"]),  2),
        "final_gen_difficulty": results["generator_difficulty"][-1] if results["generator_difficulty"] else 1,
    }
    print("\n=== EVALUATION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return {**results, "summary": summary}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _format_chat(system: str, user: str, tokenizer) -> str:
    messages = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _wandb_available() -> bool:
    try:
        import wandb
        return True
    except ImportError:
        return False


def _print_final_stats(reward_log: Dict[str, List[float]]) -> None:
    print("\n=== TRAINING REWARD SUMMARY ===")
    for role, rewards in reward_log.items():
        if not rewards:
            continue
        n = len(rewards)
        first_q  = rewards[:max(1, n//4)]
        last_q   = rewards[max(0, 3*n//4):]
        mean_all = sum(rewards) / n
        mean_first = sum(first_q) / len(first_q)
        mean_last  = sum(last_q)  / len(last_q)
        trend = "↑" if mean_last > mean_first + 0.05 else ("↓" if mean_last < mean_first - 0.05 else "→")
        print(
            f"  {role:12s}: mean={mean_all:.3f} | "
            f"first_q={mean_first:.3f} → last_q={mean_last:.3f} {trend}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent ATC GRPO Training")
    parser.add_argument("--model",      default=DEFAULT_MODEL,  help="HF model ID or local path")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--episodes",   type=int, default=200,  help="Number of training episodes")
    parser.add_argument("--lora_rank",  type=int, default=LORA_RANK)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--eval_only",  action="store_true",    help="Run eval instead of training")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.model, n_episodes=20, seed=args.seed)
    else:
        train(
            model_name=args.model,
            output_dir=args.output_dir,
            n_episodes=args.episodes,
            lora_rank=args.lora_rank,
            seed=args.seed,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        )


if __name__ == "__main__":
    main()
