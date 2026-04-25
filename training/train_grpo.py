"""Multi-Agent ATC GRPO Training with Unsloth.

Architecture:
  - Single LLM (Qwen2.5-7B-Instruct, 4-bit QLoRA) plays 4 roles via system prompts
  - GRPO: group-relative advantage  A_i = (r_i - mean(group)) / (std(group) + eps)
  - Five reward heads (AMAN, DMAN, GENERATOR, SUPERVISOR, ADAPT)
  - Potential-based reward shaping per role (policy-gradient safe, Ng et al. 1999)
  - Adaptive curriculum: ChallengeGenerator escalates difficulty as agents improve
  - Per-role reward curves saved to reward_curves.json for demo

Training loop:
  Episode -> Generator mutates task -> AMAN bids -> DMAN bids ->
  Negotiate (if conflicts) -> Grade -> Per-agent GRPO update

Colab T4 resource profile:
  Model:        Qwen2.5-7B-Instruct, 4-bit QLoRA
  LoRA rank:    16 (q_proj, v_proj, k_proj, o_proj)
  Batch size:   2, gradient accumulation 4 -> effective batch 8
  Generations:  4 per prompt (GRPO group size — minimum for stable advantage estimate)
  Max tokens:   512 per completion
  Training:     ~200 episodes ≈ 800 samples ≈ 2 hr on T4

Usage:
  python training/train_grpo.py [--episodes 200] [--model Qwen/Qwen2.5-7B-Instruct]

Colab one-liner:
  !python training/train_grpo.py --episodes 100 --output_dir /content/atc-multiagent
"""

from __future__ import annotations

import argparse
import math
import inspect
import json
import logging
import os
import random
import re
import sys
import time
import warnings
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# OpenEnv hackathon winners / strong baselines — same defaults as sid-rp/kube-sre-gym train.py:
#   https://raw.githubusercontent.com/sid-rp/kube-sre-gym/main/train.py
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

def _require_training_deps():
    if sys.version_info >= (3, 14):
        print("[ERROR] Python 3.14 not supported. Use 3.11 or 3.12.")
        sys.exit(1)
    try:
        import torch
    except ImportError as e:
        print(f"[ERROR] Training deps missing: {e}")
        print("Install: pip install torch")
        sys.exit(1)
    try:
        # Unsloth should be imported before TRL/Transformers/PEFT.
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
    except Exception as e:
        print(f"[ERROR] unsloth import failed: {e}")
        print("Install: pip install unsloth unsloth-zoo")
        sys.exit(1)
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"[ERROR] Training deps missing: {e}")
        print("Install: pip install trl transformers")
        sys.exit(1)
    return torch, FastLanguageModel, GRPOConfig, GRPOTrainer


from training.continuous_curriculum import (
    load_or_create_continuous_state,
    save_continuous_state,
)
from training.roster_integrity import (
    ROSTER_PACK_SIZE,
    align_batch_size_to_roster,
    assert_batch_role_counts,
    assert_dataset_has_full_roster,
    format_role_distribution,
    strict_roster_enabled,
)
from training.dataset import (
    build_episode_dataset,
    parse_aman_action,
    parse_dman_action,
    parse_generator_action,
)
from training.reward_functions import (
    adapt_reward_fn,
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
from multi_agent.adapt import parse_adapt_action


# ── Hyperparameters ───────────────────────────────────────────────────────────

DEFAULT_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT = "./outputs/atc-multiagent"
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_TARGETS   = ["q_proj", "v_proj", "k_proj", "o_proj"]
MAX_SEQ_LEN    = 4096
MAX_NEW_TOKENS = 384
TEMPERATURE    = 0.7
# Role-aware generation budgets. GRPO supports one global max token cap, so we
# use the max here and enforce per-role compactness in rewards/logging.
ROLE_MAX_NEW_TOKENS = {
    AgentRole.AMAN.value: 384,
    AgentRole.DMAN.value: 384,
    AgentRole.GENERATOR.value: 224,
    AgentRole.SUPERVISOR.value: 160,
    AgentRole.ADAPT.value: 224,
}
# 4 generations per prompt: minimum group size for a stable GRPO advantage estimate.
# With N=2 the group std is near-zero, making the normalised advantage meaningless.
N_GENERATIONS  = 4
BATCH_SIZE     = 4
GRAD_ACCUM     = 2           # effective batch = 8
LR             = 5e-5
# In trl==0.16.0 + unsloth==2026.4.7 with PEFT, non-zero KL can fail when
# ref_per_token_logps is absent in the fast path (ref=None crash).
KL_COEFF       = 0.0
WARMUP_STEPS_FRACTION = 0.03
LOGGING_STEPS  = 1
SAVE_STEPS     = 50
SAVE_TOTAL_LIMIT = 3         # keep only 3 checkpoints on disk
REWARD_CLIP_ABS = 0.95


def _configure_runtime_warnings() -> None:
    """Hide repetitive upstream warnings that don't affect correctness."""
    warnings.filterwarnings(
        "ignore",
        message=r"Both `max_new_tokens` \(=.*\) and `max_length`\(=.*\) seem to have been set\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Passing `generation_config` together with generation-related arguments=.* is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*generation_config.*generation-related arguments.*deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`use_return_dict` is deprecated! Use `return_dict` instead!",
    )
    # Some transformers builds emit this through logging, not warnings.
    class _SuppressMaxLenWarning(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            suppressed = (
                "Both `max_new_tokens`" in msg
                or "generation_config" in msg and "generation-related arguments" in msg
                or "`use_return_dict` is deprecated" in msg
            )
            return not suppressed

    for name in (
        "transformers.generation.utils",
        "transformers.generation.configuration_utils",
        "transformers.modeling_utils",
    ):
        logging.getLogger(name).addFilter(_SuppressMaxLenWarning())


def _auto_tune_for_gpu(torch_module) -> Dict[str, int]:
    """Return tuned batch/accum/token settings based on detected VRAM."""
    tuned = {
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    if not torch_module.cuda.is_available():
        return tuned
    try:
        vram_gb = torch_module.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return tuned

    # 80GB-class GPUs: increase throughput while keeping rollout quality.
    if vram_gb >= 70:
        tuned["batch_size"] = max(tuned["batch_size"], 8)
        tuned["grad_accum"] = 1
        tuned["max_new_tokens"] = min(tuned["max_new_tokens"], 384)
    elif vram_gb >= 40:
        tuned["batch_size"] = max(tuned["batch_size"], 6)
        tuned["grad_accum"] = min(tuned["grad_accum"], 2)
        tuned["max_new_tokens"] = min(tuned["max_new_tokens"], 384)
    return tuned


def _prefer_local_model_path(model_name: str) -> str:
    """Use local HF cache path when available to avoid network flakiness."""
    if os.path.isdir(model_name):
        return model_name
    try:
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(repo_id=model_name, local_files_only=True)
        print(f"[INFO] Using local model snapshot cache: {local_path}")
        return local_path
    except Exception:
        return model_name


def _is_transient_network_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    needles = (
        "temporary failure in name resolution",
        "name resolution",
        "connecterror",
        "connection error",
        "failed to establish a new connection",
    )
    return any(n in msg for n in needles)


def _load_model_with_fallback(
    FastLanguageModel,
    model_source: str,
    *,
    max_seq_length: int,
):
    """Load model/tokenizer, retrying in offline mode on DNS/network failures."""
    kwargs = {
        "model_name": model_source,
        "max_seq_length": max_seq_length,
        "load_in_4bit": True,
        "dtype": None,
    }
    try:
        return FastLanguageModel.from_pretrained(**kwargs)
    except Exception as exc:
        if not _is_transient_network_error(exc):
            raise
        print("[WARN] Network/DNS issue while loading tokenizer/model. Retrying from local cache...")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        kwargs["local_files_only"] = True
        return FastLanguageModel.from_pretrained(**kwargs)


# ── Role-dispatch table ───────────────────────────────────────────────────────

REWARD_FN_DISPATCH = {
    AgentRole.AMAN.value:       aman_reward_fn,
    AgentRole.DMAN.value:       dman_reward_fn,
    AgentRole.GENERATOR.value:  generator_reward_fn,
    AgentRole.SUPERVISOR.value: supervisor_reward_fn,
    AgentRole.ADAPT.value:      adapt_reward_fn,
}


def _reward_failure_mode() -> str:
    mode = os.getenv("REWARD_FAILURE_MODE", "strict").strip().lower()
    return mode if mode in {"strict", "penalize"} else "strict"


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


def _maybe_patch_trainer_sampler(trainer) -> None:
    """Handle TRL/Transformers sampler signature drift across versions."""
    try:
        sampler = getattr(type(trainer), "_get_train_sampler", None)
        if sampler is None:
            return
        # Old TRL versions expose _get_train_sampler(self) while newer
        # Transformers call sampler_fn(dataset). Patch only that old form.
        if len(inspect.signature(sampler).parameters) == 1:
            from types import MethodType

            original = trainer._get_train_sampler

            def _compat_get_train_sampler(self, train_dataset=None):
                return original()

            trainer._get_train_sampler = MethodType(_compat_get_train_sampler, trainer)
            print("[WARN] Applied sampler compatibility shim for this TRL/Transformers pair.")
    except Exception as exc:
        print(f"[WARN] Could not apply sampler compatibility shim: {exc}")


def _maybe_force_sequential_train_sampler(trainer, *, strict_roster: bool) -> None:
    """Unsloth/TRL may ignore ``shuffle_train_dataset=False``; force roster pack order."""
    if not strict_roster:
        return
    try:
        from types import MethodType

        from torch.utils.data import SequentialSampler

        def _sequential_train_sampler(self, train_dataset=None):
            ds = train_dataset if train_dataset is not None else self.train_dataset
            return SequentialSampler(ds)

        trainer._get_train_sampler = MethodType(_sequential_train_sampler, trainer)
        print(
            "[STRICT_ROSTER] Forced SequentialSampler (ignores shuffle; keeps 6-role packs contiguous)."
        )
    except Exception as exc:
        print(f"[WARN] Could not force SequentialSampler: {exc}")


def _as_agent_role_list(role_val: Any) -> List[str]:
    """Batch ``agent_role`` column → Python ``str`` list (handles tensor/ndarray)."""
    if role_val is None:
        return []
    if isinstance(role_val, (list, tuple)):
        return [str(x) for x in role_val]
    try:
        import numpy as np

        if isinstance(role_val, np.ndarray):
            return [str(x) for x in role_val.tolist()]
    except Exception:
        pass
    try:
        import torch

        if isinstance(role_val, torch.Tensor):
            return [str(x) for x in role_val.detach().cpu().tolist()]
    except Exception:
        pass
    return [str(role_val)]


def _coerce_scalar_list(values: Any) -> List[Any]:
    """Turn ``difficulty_scalar``-like batch fields into a flat list."""
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return list(values)
    try:
        import numpy as np

        if isinstance(values, np.ndarray):
            return values.tolist()
    except Exception:
        pass
    try:
        import torch

        if isinstance(values, torch.Tensor):
            return values.detach().cpu().tolist()
    except Exception:
        pass
    return [values]


def _expand_kw_to_completions(n_completions: int, seq: List[Any], *, name: str) -> List[Any]:
    """Repeat each entry ``G`` times when TRL stacks ``G`` completions per dataset row (GRPO)."""
    if n_completions <= 0:
        return []
    if not seq:
        return [None] * n_completions
    base = len(seq)
    if base == n_completions:
        return list(seq)
    if n_completions % base == 0:
        g = n_completions // base
        out: List[Any] = []
        for x in seq:
            out.extend([x] * g)
        return out
    print(
        f"[WARN] {name}: len={base} does not divide n_completions={n_completions}; "
        "padding with last value (roster assert may still fail if the base batch is wrong)."
    )
    return list(seq) + [seq[-1]] * (n_completions - base)


def _maybe_patch_unsloth_grad_accum(trainer) -> None:
    """Provide missing attribute expected by some Unsloth GRPO trainer builds."""
    if hasattr(trainer, "current_gradient_accumulation_steps"):
        return
    steps = 1
    try:
        args = getattr(trainer, "args", None)
        if args is not None:
            steps = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    except Exception:
        steps = 1
    trainer.current_gradient_accumulation_steps = steps
    print(
        "[WARN] Applied Unsloth GRPO compatibility shim: "
        f"current_gradient_accumulation_steps={steps}"
    )


def _maybe_patch_unsloth_loss_type(trainer) -> None:
    """Ensure loss_type exists for Unsloth/TRL compatibility.

    Some compiled trainer paths branch on hasattr(self.args, "loss_type").
    If missing, they may take an older unpack path that mismatches newer
    return signatures and crashes with "too many values to unpack".
    """
    try:
        args = getattr(trainer, "args", None)
        if args is None:
            return
        if getattr(args, "loss_type", None) is None:
            setattr(args, "loss_type", "grpo")
            print("[WARN] Applied Unsloth GRPO compatibility shim: loss_type='grpo'")
    except Exception as exc:
        print(f"[WARN] Could not apply loss_type compatibility shim: {exc}")


def _maybe_patch_unsloth_runtime_attrs(trainer) -> None:
    """Backfill runtime attrs expected by some Unsloth compiled trainer builds."""
    try:
        args = getattr(trainer, "args", None)

        if not hasattr(trainer, "importance_sampling_level"):
            level = "token"
            if args is not None:
                level = getattr(args, "importance_sampling_level", level) or level
            setattr(trainer, "importance_sampling_level", level)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"importance_sampling_level={level!r}"
            )

        if not hasattr(trainer, "epsilon_low"):
            epsilon_low = 0.2
            if args is not None:
                epsilon_low = float(getattr(args, "epsilon", epsilon_low) or epsilon_low)
            setattr(trainer, "epsilon_low", epsilon_low)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"epsilon_low={epsilon_low}"
            )

        if not hasattr(trainer, "epsilon_high"):
            epsilon_high = getattr(trainer, "epsilon_low", 0.2)
            if args is not None:
                epsilon_high = float(
                    getattr(args, "epsilon_high", epsilon_high) or epsilon_high
                )
            setattr(trainer, "epsilon_high", epsilon_high)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"epsilon_high={epsilon_high}"
            )

        if not hasattr(trainer, "vllm_importance_sampling_cap"):
            cap = 2.0
            if args is not None:
                cap = float(
                    getattr(args, "vllm_importance_sampling_cap", cap) or cap
                )
            setattr(trainer, "vllm_importance_sampling_cap", cap)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"vllm_importance_sampling_cap={cap}"
            )

        if not hasattr(trainer, "loss_type"):
            loss_type = "grpo"
            if args is not None:
                loss_type = getattr(args, "loss_type", loss_type) or loss_type
            setattr(trainer, "loss_type", loss_type)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"loss_type={loss_type!r}"
            )
    except Exception as exc:
        print(f"[WARN] Could not apply runtime attr compatibility shim: {exc}")


def _maybe_patch_unsloth_args_attrs(trainer) -> None:
    """Backfill args fields expected by some Unsloth compiled trainer paths."""
    try:
        args = getattr(trainer, "args", None)
        if args is None:
            return

        if not hasattr(args, "delta") or getattr(args, "delta", None) is None:
            # None can crash compiled GRPO paths with NoneType - Tensor arithmetic.
            setattr(args, "delta", 0.0)
            print("[WARN] Applied Unsloth GRPO compatibility shim: args.delta=0.0")

        if not hasattr(args, "temperature"):
            setattr(args, "temperature", TEMPERATURE)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"args.temperature={TEMPERATURE}"
            )

        if not hasattr(args, "max_completion_length"):
            setattr(args, "max_completion_length", MAX_NEW_TOKENS)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"args.max_completion_length={MAX_NEW_TOKENS}"
            )
    except Exception as exc:
        print(f"[WARN] Could not apply args attr compatibility shim: {exc}")


def _maybe_patch_nanmin_symbols() -> None:
    """Provide nanmin/nanmax symbols expected by some generated trainer code."""
    try:
        import builtins
        import torch as _torch

        def _compat_nanmin(x, *args, **kwargs):
            if hasattr(_torch, "nanmin"):
                return _torch.nanmin(x, *args, **kwargs)
            x2 = _torch.nan_to_num(x, nan=float("inf"))
            if args or kwargs:
                return _torch.amin(x2, *args, **kwargs)
            return _torch.min(x2)

        def _compat_nanmax(x, *args, **kwargs):
            if hasattr(_torch, "nanmax"):
                return _torch.nanmax(x, *args, **kwargs)
            x2 = _torch.nan_to_num(x, nan=float("-inf"))
            if args or kwargs:
                return _torch.amax(x2, *args, **kwargs)
            return _torch.max(x2)

        if not hasattr(builtins, "nanmin"):
            builtins.nanmin = _compat_nanmin
            print("[WARN] Applied compatibility shim: builtins.nanmin")
        if not hasattr(builtins, "nanmax"):
            builtins.nanmax = _compat_nanmax
            print("[WARN] Applied compatibility shim: builtins.nanmax")
    except Exception as exc:
        print(f"[WARN] Could not apply nanmin/nanmax compatibility shim: {exc}")


def _preflight_unsloth_grpo_args(trainer) -> None:
    """Fail-safe normalization for known unstable compiled GRPO argument states."""
    args = getattr(trainer, "args", None)
    if args is None:
        return

    # Ensure arithmetic operands are numeric, never None.
    if getattr(args, "delta", None) is None:
        setattr(args, "delta", 0.0)
        print("[SAFETY] Normalized args.delta=None -> 0.0")

    # If KL is not explicitly forced, keep all KL knobs at zero to avoid
    # ref_per_token_logps=None paths in this Unsloth/TRL combination.
    force_kl = os.getenv("ATC_FORCE_KL", "").strip().lower() in {"1", "true", "yes"}
    if not force_kl:
        for name in ("beta", "kl_coeff"):
            if hasattr(args, name) and getattr(args, name) not in (0, 0.0, None):
                setattr(args, name, 0.0)
                print(f"[SAFETY] Normalized args.{name} -> 0.0")
        for name in ("beta", "kl_coeff"):
            if hasattr(trainer, name) and getattr(trainer, name) not in (0, 0.0, None):
                setattr(trainer, name, 0.0)
                print(f"[SAFETY] Normalized trainer.{name} -> 0.0")

    print(
        "[INFO] GRPO preflight: "
        f"delta={getattr(args, 'delta', 'n/a')}, "
        f"beta={getattr(args, 'beta', 'n/a')}, "
        f"kl_coeff={getattr(args, 'kl_coeff', 'n/a')}, "
        f"ATC_FORCE_KL={os.getenv('ATC_FORCE_KL', '')!r}"
    )


def _resolve_num_generations(batch_size: int, requested: int) -> int:
    requested = max(1, requested)
    for candidate in range(min(requested, batch_size), 0, -1):
        if batch_size % candidate == 0:
            return candidate
    return 1


def _effective_kl_coeff() -> float:
    raw = os.getenv("ATC_KL_COEFF", str(KL_COEFF)).strip()
    try:
        value = float(raw)
    except ValueError:
        print(f"[WARN] Invalid ATC_KL_COEFF={raw!r}; using default {KL_COEFF}.")
        return KL_COEFF

    if value < 0.0:
        print(f"[WARN] Negative KL coeff {value} is invalid; clamping to 0.0.")
        value = 0.0

    if value <= 0.0:
        return 0.0

    # Unsloth+TRL fast path can crash with KL enabled unless explicitly forced.
    force_kl = os.getenv("ATC_FORCE_KL", "").strip().lower() in {"1", "true", "yes"}
    if not force_kl:
        print(
            "[WARN] KL>0 requested but disabled for this Unsloth/TRL stack to avoid "
            "ref_per_token_logps=None crashes. Set ATC_FORCE_KL=1 to override."
        )
        return 0.0
    print(f"[INFO] KL penalty enabled (forced): {value:.4f}")
    return value


def _select_sample_value(value: Any, index: int) -> Any:
    if isinstance(value, list):
        if not value:
            return None
        return value[index] if index < len(value) else value[-1]
    return value


def _shannon_entropy(items: List[str]) -> float:
    if not items:
        return 0.0
    counts: Dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    total = float(len(items))
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p + 1e-12, 2)
    return h


def _difficulty_reward_profile(difficulties: List[float], rewards: List[float]) -> Dict[str, Any]:
    pairs = [(d, r) for d, r in zip(difficulties, rewards) if d == d and r == r]
    if not pairs:
        return {}
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    out: Dict[str, Any] = {}
    for lo, hi in bins:
        vals = [r for d, r in pairs if lo <= d < hi]
        key = f"{lo:.2f}-{min(1.0, hi):.2f}"
        if not vals:
            out[key] = {"count": 0, "mean_reward": None, "std_reward": None}
            continue
        mean_val = sum(vals) / len(vals)
        std_val = (
            math.sqrt(sum((v - mean_val) ** 2 for v in vals) / len(vals))
            if len(vals) > 1
            else 0.0
        )
        out[key] = {
            "count": len(vals),
            "mean_reward": round(mean_val, 4),
            "std_reward": round(std_val, 4),
        }
    return out


# ── Unified reward dispatcher ─────────────────────────────────────────────────

def combined_reward_fn(completions: List[str], **kwargs) -> List[float]:
    """Route each completion to its role-specific reward function.

    TRL GRPOTrainer calls this with a batch of completions.
    kwargs contains per-sample metadata from the dataset.
    """
    roles = _as_agent_role_list(kwargs.get("agent_role"))
    if not roles:
        roles = [AgentRole.AMAN.value] * len(completions)
    roles = _expand_kw_to_completions(len(completions), roles, name="agent_role")

    rewards: List[float] = []
    failure_mode = _reward_failure_mode()

    for i, (completion, role) in enumerate(zip(completions, roles)):
        fn = REWARD_FN_DISPATCH.get(role, aman_reward_fn)
        sample_kwargs = {k: [_select_sample_value(v, i)] for k, v in kwargs.items()}
        try:
            r = fn([completion], **sample_kwargs)
            if not r:
                raise RuntimeError(f"empty reward list for role={role}")
            rewards.append(max(-REWARD_CLIP_ABS, min(REWARD_CLIP_ABS, float(r[0]))))
        except Exception as exc:
            msg = f"reward_fn({role}) failed at index={i}: {exc}"
            if failure_mode == "strict":
                raise RuntimeError(msg) from exc
            print(f"[WARN] {msg}")
            rewards.append(-REWARD_CLIP_ABS)

    return rewards


# ── Training entry point ──────────────────────────────────────────────────────

def train(
    model_name:   str  = DEFAULT_MODEL,
    output_dir:   str  = DEFAULT_OUTPUT,
    n_episodes:   int  = 200,
    lora_rank:    int  = LORA_RANK,
    seed:         int  = 42,
    push_to_hub:  bool = False,
    hub_model_id: Optional[str] = None,
    run_eval:     bool = True,
    eval_episodes: int = 3,
    use_grounded_curriculum: bool = False,
    curriculum_state_path: Optional[str] = None,
    adapter_in: Optional[str] = None,
    max_steps: Optional[int] = None,
    mid_eval_every: int = 0,
    mid_eval_episodes: Optional[int] = None,
) -> None:
    torch, FastLanguageModel, GRPOConfig, GRPOTrainer = _require_training_deps()
    _configure_runtime_warnings()
    from transformers import TrainerCallback

    tuned = _auto_tune_for_gpu(torch)
    batch_size = tuned["batch_size"]
    grad_accum = tuned["grad_accum"]
    max_new_tokens = max(tuned["max_new_tokens"], max(ROLE_MAX_NEW_TOKENS.values()))

    grounded_live_early = (
        use_grounded_curriculum
        and os.environ.get("ATC_STATIC_GROUNDED_DATASET", "").lower() not in {"1", "true", "yes"}
    )
    if strict_roster_enabled(
        grounded_live=grounded_live_early,
        use_grounded_curriculum=use_grounded_curriculum,
    ):
        nb = align_batch_size_to_roster(batch_size, ROSTER_PACK_SIZE)
        if nb != batch_size:
            print(
                f"[STRICT_ROSTER] batch_size {batch_size} -> {nb} "
                f"(multiple of {ROSTER_PACK_SIZE} for full multi-agent batches)"
            )
        batch_size = nb
        tuned["batch_size"] = batch_size

    num_generations = _resolve_num_generations(batch_size, N_GENERATIONS)
    if num_generations != N_GENERATIONS:
        print(
            f"[WARN] Adjusted num_generations {N_GENERATIONS} -> {num_generations} "
            f"to satisfy GRPO batch-size divisibility constraint."
        )

    print(f"\n{'='*60}")
    print(f"  ATC Multi-Agent GRPO Training")
    print(f"  Model:        {model_name}")
    print(f"  Episodes:     {n_episodes}")
    if run_eval:
        print(f"  Eval episodes:{max(1, int(eval_episodes))}")
    print(f"  Generations:  {num_generations} per prompt")
    print(f"  Output:       {output_dir}")
    device_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device:       {device_str}")
    print(
        f"  Tune:         batch={batch_size}, accum={grad_accum}, "
        f"max_new_tokens={max_new_tokens}, temp={TEMPERATURE}, logging_steps={LOGGING_STEPS}"
    )
    print(
        "  Role tokens:  "
        f"AMAN={ROLE_MAX_NEW_TOKENS[AgentRole.AMAN.value]}, "
        f"DMAN={ROLE_MAX_NEW_TOKENS[AgentRole.DMAN.value]}, "
        f"GEN={ROLE_MAX_NEW_TOKENS[AgentRole.GENERATOR.value]}, "
        f"SUP={ROLE_MAX_NEW_TOKENS[AgentRole.SUPERVISOR.value]}, "
        f"ADAPT={ROLE_MAX_NEW_TOKENS[AgentRole.ADAPT.value]}"
    )
    print(f"{'='*60}\n")

    # Hold out unseen scenarios for validation/debugging.
    all_task_ids = [t.task_id for t in ordered_tasks()]
    holdout_rng = random.Random(seed + 173)
    non_grounded_ids = [tid for tid in all_task_ids if not tid.startswith("gc_")]
    if use_grounded_curriculum:
        holdout_task_ids = set()
        print("[INFO] Grounded curriculum: no holdout split for gc_* tasks (all used for training).")
    else:
        hc = max(1, int(round(0.2 * len(non_grounded_ids)))) if non_grounded_ids else 0
        holdout_task_ids = (
            set(holdout_rng.sample(non_grounded_ids, hc)) if hc else set()
        )
    if holdout_task_ids:
        print(f"[INFO] Holdout task set ({len(holdout_task_ids)}): {sorted(holdout_task_ids)}")

    # ── 1. Capture pre-training baseline metrics ──────────────────────────────
    if run_eval:
        print("[0/5] Capturing pre-training baseline metrics...")
        baseline = _quick_heuristic_eval(n_episodes=min(10, n_episodes))
        _save_json(baseline, Path(output_dir) / "baseline_metrics.json")
        print(f"    Baseline composite: {baseline['mean_composite']:.3f}")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print("[1/5] Loading model with Unsloth 4-bit QLoRA...")
    model_source = _prefer_local_model_path(model_name)
    model, tokenizer = _load_model_with_fallback(
        FastLanguageModel,
        model_source,
        max_seq_length=MAX_SEQ_LEN,
    )
    # Prevent generate() ambiguity warnings from inherited max_length defaults.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    _adapter_path = Path(adapter_in) if adapter_in else None
    if _adapter_path and (_adapter_path / "adapter_config.json").is_file():
        from peft import PeftModel

        print(f"    Loading existing LoRA (SFT or prior GRPO): {_adapter_path}")
        model = PeftModel.from_pretrained(model, str(_adapter_path), is_trainable=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Continued adapter trainable params: {trainable:,}")
    else:
        if adapter_in:
            print(f"[WARN] --adapter_in {adapter_in!r} missing adapter_config.json; training fresh LoRA.")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGETS,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    LoRA rank={lora_rank}, trainable params: {trainable:,}")

    print(f"[VRAM] {_vram_snapshot(torch)}")

    # ── 2b. Base model eval (before any gradient steps) ──────────────────────
    base_model_metrics: Optional[Dict[str, Any]] = None
    eval_task_ids = sorted(holdout_task_ids) if holdout_task_ids else None
    if run_eval:
        print("\n[1.5/5] Measuring base model score (untrained LoRA)...")
        model.eval()
        base_model_metrics = _run_model_episodes(
            model,
            tokenizer,
            n_episodes=max(1, int(eval_episodes)),
            tag="BASE MODEL (no fine-tune)",
            eval_task_ids=eval_task_ids,
        )
        model.train()
        _save_json(base_model_metrics, Path(output_dir) / "base_model_metrics.json")
        print(f"    Base model composite: {base_model_metrics['mean_composite']:.3f}"
              f"  (AMAN {base_model_metrics['mean_aman_reward']:.3f}"
              f" / DMAN {base_model_metrics['mean_dman_reward']:.3f})")

    # ── 3. Build training dataset (static list OR live iterable for grounded) ─
    print(f"\n[2/5] Building training data...")
    t0 = time.time()
    continuous_path = Path(output_dir) / "continuous_curriculum_state.json"
    curriculum_manager = None
    grounded_live = (
        use_grounded_curriculum
        and os.environ.get("ATC_STATIC_GROUNDED_DATASET", "").lower() not in {"1", "true", "yes"}
    )
    from training.live_curriculum import live_max_steps as _live_max_steps

    try:
        _live_passes = float(os.environ.get("ATC_LIVE_PASSES", "2.5").strip() or "2.5")
    except ValueError:
        _live_passes = 2.5
    _live_passes = max(1.0, min(5.0, _live_passes))
    computed_max_steps = _live_max_steps(n_episodes, batch_size, grad_accum, passes=_live_passes)
    if _live_passes != 2.5:
        print(f"[INFO] ATC_LIVE_PASSES={_live_passes} (live_max_steps / materialized row budget)")
    if max_steps is not None:
        cap = max(1, int(max_steps))
        derived = int(computed_max_steps)
        computed_max_steps = min(derived, cap)
        if computed_max_steps != derived:
            print(f"[INFO] max_steps cap: curriculum-derived {derived} -> {computed_max_steps}")

    if grounded_live:
        if curriculum_state_path:
            _src = Path(curriculum_state_path)
            if _src.is_file():
                continuous_path.parent.mkdir(parents=True, exist_ok=True)
                continuous_path.write_bytes(_src.read_bytes())
                print(f"    Warm-start live curriculum from {_src}")

        from training.live_curriculum import CurriculumManager

        curriculum_manager = CurriculumManager(
            seed=seed,
            output_dir=Path(output_dir),
            continuous_state_path=continuous_path,
        )
        # HF IterableDataset + TRL/Unsloth still collate nested ``prompt`` message dicts in a way
        # that leaves string leaves in the batch dict; Accelerate's find_batch_size then crashes
        # (TypeError: ... got str). Materialize a finite prefix and use ``Dataset.from_list`` —
        # identical to the static grounded path, which is known-good with GRPOTrainer.
        from training.live_curriculum import iter_live_grounded_rows

        _max_rows = min(
            600_000,
            int(computed_max_steps) * int(max(1, batch_size)) * int(max(1, grad_accum))
            * int(max(4, num_generations))
            + 512,
        )
        _buf: List[Dict[str, Any]] = []
        _row_it = iter_live_grounded_rows(curriculum_manager, int(seed))
        for _ in range(_max_rows):
            _buf.append(next(_row_it))
        if strict_roster_enabled(
            grounded_live=True,
            use_grounded_curriculum=True,
        ):
            _trim = len(_buf) % ROSTER_PACK_SIZE
            if _trim:
                del _buf[-_trim:]
                print(f"[STRICT_ROSTER] trimmed {_trim} tail rows for pack alignment (n={len(_buf)})")
            assert_dataset_has_full_roster(_buf, context="live_materialized")
        try:
            from datasets import Dataset as HFDataset
        except ImportError:
            print("[ERROR] pip install datasets (required for live grounded GRPO)")
            sys.exit(1)
        dataset = HFDataset.from_list(_buf)
        dataset_train: List[Dict[str, Any]] = []
        dataset_val: List[Dict[str, Any]] = []
        print(
            f"    Live grounded (materialized n={len(_buf)}, max_steps={computed_max_steps}, "
            f"virtual_episodes≈{n_episodes}) — init {time.time()-t0:.1f}s"
        )
        _save_json(
            {
                "mode": "live_materialized",
                "materialized_rows": len(_buf),
                "max_steps": computed_max_steps,
                "virtual_episodes": n_episodes,
            },
            Path(output_dir) / "grounded_dataset_summary.json",
        )
    else:
        dataset_raw = build_episode_dataset(
            n_episodes=n_episodes,
            seed=seed,
            include_generator=True,
            include_supervisor=True,
            include_adapt=True,
            domain_episode_ratio=0.30,
            use_grounded_curriculum=use_grounded_curriculum,
            curriculum_state_path=curriculum_state_path,
            continuous_curriculum_path=str(continuous_path),
        )
        print(f"    Dataset (pre-split): {len(dataset_raw)} samples ({time.time()-t0:.1f}s)")

        dataset_train = [
            s for s in dataset_raw
            if s.get("task_id") == "domain_transfer" or s.get("task_id") not in holdout_task_ids
        ]
        dataset_val = [
            s for s in dataset_raw
            if s.get("task_id") in holdout_task_ids
        ]
        print(f"    Train samples: {len(dataset_train)} | Holdout samples: {len(dataset_val)}")

        if use_grounded_curriculum:
            bucket_counts: Dict[int, int] = {}
            template_counts: Dict[str, int] = {}
            d_vals: List[float] = []
            for s in dataset_train:
                if not s.get("grounded_curriculum"):
                    continue
                bidx = int(s.get("difficulty_bucket_index", -1))
                if bidx >= 0:
                    bucket_counts[bidx] = bucket_counts.get(bidx, 0) + 1
                tid = s.get("grounded_template_id")
                if isinstance(tid, str):
                    template_counts[tid] = template_counts.get(tid, 0) + 1
                cd = s.get("continuous_difficulty")
                if isinstance(cd, (int, float)):
                    d_vals.append(float(cd))
            grounded_summary = {
                "mode": "static_list",
                "difficulty_bucket_counts": {str(k): v for k, v in sorted(bucket_counts.items())},
                "grounded_template_counts": dict(sorted(template_counts.items())),
                "continuous_d_mean": round(sum(d_vals) / len(d_vals), 4) if d_vals else None,
                "continuous_d_min": round(min(d_vals), 4) if d_vals else None,
                "continuous_d_max": round(max(d_vals), 4) if d_vals else None,
            }
            _save_json(grounded_summary, Path(output_dir) / "grounded_dataset_summary.json")
            print(f"    Grounded dataset summary: {grounded_summary}")

        role_counts: Dict[str, int] = {}
        for s in dataset_train:
            r = s.get("agent_role", "unknown")
            role_counts[r] = role_counts.get(r, 0) + 1
        for role, count in sorted(role_counts.items()):
            print(f"    {role}: {count} samples")

        mut_types: List[str] = []
        for s in dataset_train:
            if s.get("agent_role") != AgentRole.GENERATOR.value:
                continue
            text = " ".join(m.get("content", "") for m in s.get("prompt", []))
            for m in re.findall(
                r"(tighten_window|inject_emergency|increase_weather_penalty|add_atfm_deadline|close_runway_window|add_conflicting_flight)",
                text,
            ):
                mut_types.append(m)
        print(f"    Generator mutation entropy: {_shannon_entropy(mut_types):.3f} bits")

        try:
            from datasets import Dataset

            dataset = Dataset.from_list(dataset_train)
        except ImportError:
            print("[ERROR] pip install datasets")
            sys.exit(1)

        if use_grounded_curriculum and strict_roster_enabled(
            grounded_live=False,
            use_grounded_curriculum=True,
        ):
            _trim_s = len(dataset_train) % ROSTER_PACK_SIZE
            if _trim_s:
                dataset_train = dataset_train[: len(dataset_train) - _trim_s]
                dataset = Dataset.from_list(dataset_train)
                print(f"[STRICT_ROSTER] trimmed {_trim_s} static rows (n={len(dataset_train)})")
            assert_dataset_has_full_roster(dataset_train, context="static_grounded")

    # ── 4. GRPO config ────────────────────────────────────────────────────────
    kl_coeff = _effective_kl_coeff()
    if curriculum_manager is not None:
        est_steps = max(1, computed_max_steps // max(1, batch_size))
    else:
        est_steps = max(1, len(dataset_train) // max(1, batch_size))
    warmup_steps = max(1, int(round(est_steps * WARMUP_STEPS_FRACTION)))
    print(
        f"\n[3/5] Configuring GRPO (group_size={num_generations}, lr={LR}, kl={kl_coeff}, "
        f"warmup_steps={warmup_steps})..."
    )
    save_steps_eff = max(1, _env_int("ATC_SAVE_STEPS", SAVE_STEPS))
    save_total_eff = max(1, _env_int("ATC_SAVE_TOTAL_LIMIT", SAVE_TOTAL_LIMIT))
    if curriculum_manager is not None:
        save_steps_eff = min(save_steps_eff, max(1, int(computed_max_steps)))

    grpo_kwargs: Dict[str, Any] = {
        "num_generations":              num_generations,
        "temperature":                  TEMPERATURE,
        "learning_rate":                LR,
        "per_device_train_batch_size":  batch_size,
        "gradient_accumulation_steps":  grad_accum,
        "lr_scheduler_type":            "cosine",
        "logging_steps":                LOGGING_STEPS,
        "save_steps":                   save_steps_eff,
        "save_total_limit":             save_total_eff,
        "output_dir":                   output_dir,
        "run_name":                     f"atc-multiagent-grpo-{int(time.time())}",
        "bf16":                         torch.cuda.is_bf16_supported(),
        "fp16":                         not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing":       True,
        "optim":                        "paged_adamw_8bit",
    }
    if curriculum_manager is not None:
        grpo_kwargs["max_steps"] = int(computed_max_steps)
        # Live manager is not picklable; keep all iteration in the main process.
        if _config_supports("dataloader_num_workers", GRPOConfig):
            grpo_kwargs["dataloader_num_workers"] = 0
    else:
        grpo_kwargs["num_train_epochs"] = 1

    if _wandb_available():
        grpo_kwargs["report_to"] = "wandb"
    else:
        grpo_kwargs["report_to"] = "none"

    # Compatibility shims for different TRL versions
    if _config_supports("warmup_steps", GRPOConfig):
        grpo_kwargs["warmup_steps"] = warmup_steps
    elif _config_supports("warmup_ratio", GRPOConfig):
        # Backward-compatible fallback only when warmup_steps is unavailable.
        grpo_kwargs["warmup_ratio"] = WARMUP_STEPS_FRACTION

    if _config_supports("max_completion_length", GRPOConfig):
        grpo_kwargs["max_completion_length"] = max_new_tokens
    elif _config_supports("max_new_tokens", GRPOConfig):
        grpo_kwargs["max_new_tokens"] = max_new_tokens

    if kl_coeff > 0.0:
        if _config_supports("beta", GRPOConfig):
            grpo_kwargs["beta"] = kl_coeff
        elif _config_supports("kl_coeff", GRPOConfig):
            grpo_kwargs["kl_coeff"] = kl_coeff

    if _config_supports("use_vllm", GRPOConfig):
        grpo_kwargs["use_vllm"] = False

    if strict_roster_enabled(
        grounded_live=grounded_live,
        use_grounded_curriculum=use_grounded_curriculum,
    ) and _config_supports("shuffle_train_dataset", GRPOConfig):
        grpo_kwargs["shuffle_train_dataset"] = False
        print("[STRICT_ROSTER] shuffle_train_dataset=False (preserve roster packs)")

    grpo_config = GRPOConfig(**grpo_kwargs)
    print(f"    checkpoint save: save_steps={save_steps_eff} save_total_limit={save_total_eff}")

    # ── 5. Per-role reward logger ─────────────────────────────────────────────
    print("\n[4/5] Setting up per-role reward logging...")

    strict_roster = strict_roster_enabled(
        grounded_live=grounded_live,
        use_grounded_curriculum=use_grounded_curriculum,
    )
    sup_reward_ring: deque[float] = deque(maxlen=512)

    # Separate lists so we can show per-role curves in the demo
    reward_log: Dict[str, List[float]] = {
        "AMAN": [], "DMAN": [], "GENERATOR": [], "SUPERVISOR": [], "ADAPT": [], "composite": []
    }
    parse_log: Dict[str, List[int]] = {
        "AMAN": [], "DMAN": [], "GENERATOR": [], "SUPERVISOR": [], "ADAPT": []
    }
    batch_diagnostics: List[Dict[str, Any]] = []
    parse_debug_samples: List[Dict[str, Any]] = []
    action_signature_log: Dict[str, List[str]] = {
        "AMAN": [], "DMAN": [], "GENERATOR": [], "SUPERVISOR": [], "ADAPT": []
    }
    difficulty_log: List[float] = []
    cc_runtime = None
    if curriculum_manager is None:
        cc_runtime = load_or_create_continuous_state(
            continuous_path if continuous_path.is_file() else None
        )
    cc_log_path = Path(output_dir) / "continuous_curriculum_log.jsonl"

    def _safe_mean(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        return sum(vals) / len(vals)

    def _safe_std(vals: List[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = _safe_mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    def _safe_corr(xs: List[float], ys: List[float]) -> float:
        n = min(len(xs), len(ys))
        if n < 3:
            return float("nan")
        x = xs[-n:]
        y = ys[-n:]
        mx = _safe_mean(x)
        my = _safe_mean(y)
        cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
        sx = math.sqrt(sum((a - mx) ** 2 for a in x))
        sy = math.sqrt(sum((b - my) ** 2 for b in y))
        if sx <= 1e-12 or sy <= 1e-12:
            return float("nan")
        return cov / (sx * sy)

    def _action_signature(role: str, completion: Any) -> str:
        try:
            if role == AgentRole.AMAN.value:
                action = parse_aman_action(completion)
                if action is None:
                    return "parse_fail"
                runways = sorted({s.runway for s in action.arrival_slots})
                return f"slots={len(action.arrival_slots)}|rwy={','.join(runways[:3])}|commit={int(action.commit)}"
            if role == AgentRole.DMAN.value:
                action = parse_dman_action(completion)
                if action is None:
                    return "parse_fail"
                runways = sorted({s.runway for s in action.departure_slots})
                return f"slots={len(action.departure_slots)}|rwy={','.join(runways[:3])}|commit={int(action.commit)}"
            if role == AgentRole.GENERATOR.value:
                action = parse_generator_action(completion)
                if action is None:
                    return "parse_fail"
                muts = sorted(m.mutation_type.value for m in action.mutations)
                return f"mut={','.join(muts[:3])}|n={len(muts)}"
            if role == AgentRole.ADAPT.value:
                action = parse_adapt_action(completion)
                if action is None:
                    return "parse_fail"
                return f"wake={len(action.entity_wake_map)}|prio={len(action.entity_priority_map)}"
            if role == AgentRole.SUPERVISOR.value:
                text = str(completion)
                ok = bool(re.search(r'"score"\s*:\s*-?\d+(?:\.\d+)?', text))
                return "score_json" if ok else "parse_fail"
        except Exception:
            return "parse_fail"
        return "unknown"

    class RewardLogger:
        __name__ = "combined_reward_fn"

        def __call__(self, *args, **kwargs):
            # TRL <0.17: reward_func(completions=..., **kwargs)
            # TRL >=0.17: reward_func(prompts, completions, **kwargs)
            if "completions" in kwargs:
                completions = kwargs.pop("completions")
            elif len(args) >= 2:
                completions = args[1]
            elif args:
                completions = args[0]
            else:
                completions = []
            kwargs.pop("prompts", None)
            kwargs.pop("prompt_ids", None)

            # Some TRL versions pass conversational turns. Extract the assistant text.
            if completions and isinstance(completions[0], list):
                flattened = []
                for c in completions:
                    if c and isinstance(c[-1], dict) and "content" in c[-1]:
                        flattened.append(c[-1]["content"])
                    else:
                        flattened.append(str(c))
                completions = flattened

            # GRPO stacks ``num_generations`` completions per dataset row; TRL keeps one
            # ``agent_role`` (and difficulty) per row unless we expand here. Padding with
            # the last role used to break strict roster asserts (e.g. 3×DMAN + 3×SUP).
            kwargs = dict(kwargs)
            _n = len(completions)
            _base_roles = _as_agent_role_list(kwargs.get("agent_role"))
            if not _base_roles:
                print("[WARN] batch missing agent_role metadata; rewards may be mis-routed.")
                _base_roles = [AgentRole.AMAN.value]
            kwargs["agent_role"] = _expand_kw_to_completions(_n, _base_roles, name="agent_role")
            if "difficulty_scalar" in kwargs:
                kwargs["difficulty_scalar"] = _expand_kw_to_completions(
                    _n, _coerce_scalar_list(kwargs.get("difficulty_scalar")), name="difficulty_scalar"
                )

            rewards = combined_reward_fn(completions, **kwargs)

            roles = kwargs["agent_role"]

            if strict_roster and len(roles) == len(rewards) and rewards:
                assert_batch_role_counts(
                    [str(x) for x in roles],
                    batch_index=len(batch_diagnostics),
                )
                print(
                    f"[ROSTER] batch={len(batch_diagnostics)} "
                    f"{format_role_distribution(dict(Counter(str(x) for x in roles)))}"
                )
            for role_i, r_i in zip(roles, rewards):
                if role_i == AgentRole.SUPERVISOR.value:
                    sup_reward_ring.append(float(r_i))
            if strict_roster and len(sup_reward_ring) >= 48:
                tail = list(sup_reward_ring)[-48:]
                m = sum(tail) / len(tail)
                v = sum((x - m) ** 2 for x in tail) / len(tail)
                if v < 1e-14:
                    raise RuntimeError(
                        "Supervisor reward variance collapsed (strict roster integrity). "
                        "Inspect supervisor_reward_fn and merged_plan_json inputs."
                    )

            def _is_parse_ok(role: str, completion: Any) -> int:
                if role == AgentRole.AMAN.value:
                    return 1 if parse_aman_action(completion) is not None else 0
                if role == AgentRole.DMAN.value:
                    return 1 if parse_dman_action(completion) is not None else 0
                if role == AgentRole.GENERATOR.value:
                    return 1 if parse_generator_action(completion) is not None else 0
                if role == AgentRole.SUPERVISOR.value:
                    text = str(completion)
                    if re.search(r'"score"\s*:\s*-?\d+(?:\.\d+)?', text):
                        return 1
                    return 0
                if role == AgentRole.ADAPT.value:
                    return 1 if parse_adapt_action(completion) is not None else 0
                return 0

            role_reward_batch: Dict[str, List[float]] = {k: [] for k in reward_log if k != "composite"}
            role_parse_batch: Dict[str, List[int]] = {k: [] for k in parse_log}
            role_sig_batch: Dict[str, List[str]] = {k: [] for k in parse_log}
            batch_difficulty: List[float] = []
            raw_difficulty = kwargs.get("difficulty_scalar", [])
            if not isinstance(raw_difficulty, list):
                raw_difficulty = [raw_difficulty] * len(rewards)

            for idx, (completion, r, role) in enumerate(zip(completions, rewards, roles)):
                if role in reward_log:
                    reward_log[role].append(r)
                    role_reward_batch[role].append(r)
                if role in parse_log:
                    parse_ok = _is_parse_ok(role, completion)
                    parse_log[role].append(parse_ok)
                    role_parse_batch[role].append(parse_ok)
                    sig = _action_signature(role, completion)
                    action_signature_log[role].append(sig)
                    role_sig_batch[role].append(sig)
                    if len(parse_debug_samples) < 240 and (
                        parse_ok == 0 or len(parse_debug_samples) < 40
                    ):
                        parse_debug_samples.append(
                            {
                                "role": role,
                                "parse_ok": int(parse_ok),
                                "reward": round(float(r), 4),
                                "signature": sig,
                                "completion_excerpt": str(completion)[:700],
                            }
                        )
                raw_d = _select_sample_value(raw_difficulty, idx)
                try:
                    d = max(0.0, min(1.0, float(raw_d)))
                    difficulty_log.append(d)
                    batch_difficulty.append(d)
                except Exception:
                    pass
                reward_log["composite"].append(r)

            batch_summary: Dict[str, Any] = {
                "batch_index": len(batch_diagnostics),
                "composite_mean": round(_safe_mean(rewards), 4) if rewards else 0.0,
                "composite_std": round(_safe_std(rewards), 4) if rewards else 0.0,
                "difficulty_mean": round(_safe_mean(batch_difficulty), 4) if batch_difficulty else None,
                "parse_rate": {},
                "action_diversity": {},
                "reward_distribution": {},
            }
            for role_key in ("AMAN", "DMAN", "GENERATOR", "SUPERVISOR", "ADAPT"):
                prs = role_parse_batch.get(role_key, [])
                rs = role_reward_batch.get(role_key, [])
                sigs = [s for s in role_sig_batch.get(role_key, []) if s != "parse_fail"]
                if prs:
                    batch_summary["parse_rate"][role_key] = round(_safe_mean(prs), 4)
                if rs:
                    batch_summary["reward_distribution"][role_key] = {
                        "mean": round(_safe_mean(rs), 4),
                        "std": round(_safe_std(rs), 4),
                        "min": round(min(rs), 4),
                        "max": round(max(rs), 4),
                    }
                if sigs:
                    batch_summary["action_diversity"][role_key] = round(
                        len(set(sigs)) / max(1, len(sigs)),
                        4,
                    )
            batch_diagnostics.append(batch_summary)

            if rewards and (curriculum_manager is not None or (use_grounded_curriculum and cc_runtime is not None)):
                ds_batch: List[float] = []
                rs_batch: List[float] = []
                mask_batch: List[bool] = []
                for idx in range(len(rewards)):
                    role_i = _select_sample_value(roles, idx)
                    mask_batch.append(role_i in (AgentRole.AMAN.value, AgentRole.DMAN.value))
                    try:
                        raw_di = _select_sample_value(raw_difficulty, idx)
                        ds_batch.append(max(0.0, min(1.0, float(raw_di))))
                    except Exception:
                        ds_batch.append(0.0)
                    rs_batch.append(float(rewards[idx]))
                if curriculum_manager is not None:
                    curriculum_manager.on_reward_batch(ds_batch, rs_batch, list(roles))
                    st = curriculum_manager.state
                else:
                    cc_runtime.record_batch(ds_batch, rs_batch, mask_batch)
                    st = cc_runtime
                if st.global_batches % 25 == 0:
                    save_continuous_state(continuous_path, st)
                    _rho = st.reward_difficulty_correlation()
                    row = {
                        "batch": st.global_batches,
                        "mu": round(st.mu, 4),
                        "c": round(st.c, 4),
                        "bin_success": st.bin_success_snapshot(),
                        "bin_reward_mean": st.bin_reward_mean_snapshot(),
                        "rho_rd": None if (_rho != _rho) else round(_rho, 4),
                    }
                    cc_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with cc_log_path.open("a", encoding="utf-8") as lf:
                        lf.write(json.dumps(row) + "\n")

            # Reward-hacking detection: warn when composite rises but per-role variance
            # collapses (all roles getting same score = likely gaming)
            if len(reward_log["composite"]) % 50 == 0 and len(reward_log["composite"]) > 50:
                _check_reward_hacking(reward_log)

            return rewards

    reward_logger = RewardLogger()

    mid_eval_every_eff = int(mid_eval_every)
    if mid_eval_every_eff <= 0:
        mid_eval_every_eff = _env_int("ATC_MID_EVAL_EVERY", 0)
    if mid_eval_every_eff > 0 and mid_eval_every_eff < 50:
        print(
            f"[WARN] mid_eval_every={mid_eval_every_eff} < 50 burns time on noisy probes; clamping to 50."
        )
        mid_eval_every_eff = 50
    mid_eval_eps_final = 0
    if mid_eval_every_eff > 0:
        raw_me_eps = (
            int(mid_eval_episodes)
            if mid_eval_episodes is not None
            else _env_int("ATC_MID_EVAL_EPISODES", 0)
        )
        if raw_me_eps <= 0:
            raw_me_eps = max(50, int(eval_episodes))
        mid_eval_eps_final = max(50, raw_me_eps)
        print(
            f"[MID_EVAL] periodic eval enabled: every {mid_eval_every_eff} optimizer steps, "
            f"n_episodes={mid_eval_eps_final} (same _run_model_episodes as post-train; "
            f"<50 episodes is too noisy to be worth the GPU time)"
        )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("\n[5/5] Starting GRPO training...")
    trainer_kwargs: Dict[str, Any] = {
        "model":            model,
        "processing_class": tokenizer,
        "reward_funcs":     [reward_logger],
        "train_dataset":    dataset,
    }
    if _trainer_supports("args", GRPOTrainer):
        trainer_kwargs["args"] = grpo_config
    else:
        trainer_kwargs["config"] = grpo_config

    trainer = GRPOTrainer(**trainer_kwargs)
    if curriculum_manager is not None and hasattr(trainer.args, "dataloader_num_workers"):
        trainer.args.dataloader_num_workers = 0

    class LiveMetricsCallback(TrainerCallback):
        """Stream concise live metrics into notebook/stdout while training."""
        def __init__(self):
            self._zero_std_streak = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if not logs:
                return
            step = int(logs.get("step", getattr(state, "global_step", 0)) or 0)
            max_steps = int(getattr(state, "max_steps", 0) or 0)
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            rstd = logs.get("reward_std")

            def _avg_last(key: str, window: int = 64) -> float:
                vals = reward_log.get(key, [])
                if not vals:
                    return float("nan")
                tail = vals[-min(window, len(vals)) :]
                return sum(tail) / max(1, len(tail))

            def _fmt(v: float) -> str:
                return "n/a" if v != v else f"{v:.3f}"
            
            def _parse_rate(role: str, window: int = 64) -> float:
                vals = parse_log.get(role, [])
                if not vals:
                    return float("nan")
                tail = vals[-min(window, len(vals)) :]
                return sum(tail) / max(1, len(tail))

            def _action_diversity(role: str, window: int = 64) -> float:
                vals = action_signature_log.get(role, [])
                if not vals:
                    return float("nan")
                tail = [v for v in vals[-min(window, len(vals)) :] if v != "parse_fail"]
                if not tail:
                    return 0.0
                return len(set(tail)) / max(1, len(tail))

            def _dist(role: str, window: int = 64) -> str:
                vals = reward_log.get(role, [])
                if not vals:
                    return "n/a"
                tail = vals[-min(window, len(vals)) :]
                return f"{min(tail):.2f}/{_safe_mean(tail):.2f}/{max(tail):.2f}"

            comp = _avg_last("composite")
            aman = _avg_last("AMAN")
            dman = _avg_last("DMAN")
            gen = _avg_last("GENERATOR")
            sup = _avg_last("SUPERVISOR")
            p_aman = _parse_rate("AMAN")
            p_dman = _parse_rate("DMAN")
            div_aman = _action_diversity("AMAN")
            div_dman = _action_diversity("DMAN")
            corr_rd = _safe_corr(difficulty_log, reward_log.get("composite", []))
            try:
                rstd_val = float(rstd) if rstd is not None else float("nan")
            except Exception:
                rstd_val = float("nan")
            if rstd_val == rstd_val and rstd_val <= 1e-9:
                self._zero_std_streak += 1
            elif rstd_val == rstd_val:
                self._zero_std_streak = 0
            print(
                "[LIVE] "
                f"step={step}/{max_steps} "
                f"loss={loss if loss is not None else 'n/a'} "
                f"lr={lr if lr is not None else 'n/a'} "
                f"comp64={_fmt(comp)} AMAN={_fmt(aman)} DMAN={_fmt(dman)} "
                f"GEN={_fmt(gen)} SUP={_fmt(sup)} "
                f"parse64[A={_fmt(p_aman)} D={_fmt(p_dman)}] "
                f"div64[A={_fmt(div_aman)} D={_fmt(div_dman)}] "
                f"r64[A={_dist('AMAN')} D={_dist('DMAN')}] "
                f"corr(reward,d)={_fmt(corr_rd)} "
                f"samples[A={len(reward_log['AMAN'])} D={len(reward_log['DMAN'])} "
                f"G={len(reward_log['GENERATOR'])} S={len(reward_log['SUPERVISOR'])} "
                f"AD={len(reward_log['ADAPT'])}] "
                f"rstd={_fmt(rstd_val)} zstd_streak={self._zero_std_streak}"
            )

    if hasattr(trainer, "add_callback"):
        live_cb = LiveMetricsCallback()
        # Fail fast before trainer.train() if callback API is incompatible.
        missing = [name for name in ("on_train_begin", "on_log", "on_train_end") if not hasattr(live_cb, name)]
        if missing:
            raise RuntimeError(
                "LiveMetricsCallback is incompatible with Trainer API; "
                f"missing methods: {missing}"
            )
        trainer.add_callback(live_cb)

        class GrpoCheckpointArtifactsCallback(TrainerCallback):
            """Snapshot plots/tables/JSON on each HF checkpoint save; optional heavy mid-eval on GRPO."""

            def __init__(self) -> None:
                self.trainer: Any = None
                self.mid_eval_every = mid_eval_every_eff
                self.mid_eval_episodes = mid_eval_eps_final

            def on_save(self, args, state, control, **kwargs):
                step = int(getattr(state, "global_step", 0) or 0)
                _dump_grpo_checkpoint_bundle(
                    Path(output_dir),
                    step,
                    reward_log,
                    parse_log,
                    difficulty_log,
                    batch_diagnostics,
                    parse_debug_samples,
                    torch,
                )
                print(
                    f"[CHECKPOINT_ARTIFACTS] step={step} -> "
                    f"{output_dir}/checkpoint_artifacts/step_{step:06d}/"
                )

            def on_step_end(self, args, state, control, **kwargs):
                if self.mid_eval_every <= 0 or not run_eval or self.mid_eval_episodes <= 0:
                    return
                step = int(getattr(state, "global_step", 0) or 0)
                if step < 1 or step % self.mid_eval_every != 0:
                    return
                tr = self.trainer
                if tr is None:
                    return
                m = tr.model
                was_training = m.training
                m.eval()
                try:
                    with torch.inference_mode():
                        unwrapped = tr.accelerator.unwrap_model(m)
                        metrics = _run_model_episodes(
                            unwrapped,
                            tokenizer,
                            n_episodes=int(self.mid_eval_episodes),
                            tag=f"MID_EVAL_step_{step}",
                            eval_task_ids=eval_task_ids,
                        )
                    out_p = Path(output_dir) / "checkpoint_artifacts" / f"mid_eval_step_{step:06d}.json"
                    _save_json(metrics, out_p)
                    vram_p = Path(output_dir) / "checkpoint_artifacts" / "vram_log.jsonl"
                    vram_p.parent.mkdir(parents=True, exist_ok=True)
                    with vram_p.open("a", encoding="utf-8") as vf:
                        vf.write(json.dumps({"step": step, **_vram_snapshot(torch)}) + "\n")
                    print(
                        f"[MID_EVAL] step={step} n={self.mid_eval_episodes} "
                        f"mean_composite={metrics.get('mean_composite')}"
                    )
                except Exception as exc:
                    print(f"[WARN] mid_eval step={step}: {exc}")
                finally:
                    if was_training:
                        m.train()

        _art_cb = GrpoCheckpointArtifactsCallback()
        trainer.add_callback(_art_cb)
        _art_cb.trainer = trainer

    # ── CRITICAL: Apply ALL compatibility patches BEFORE any training ──
    _maybe_patch_trainer_sampler(trainer)
    _maybe_force_sequential_train_sampler(trainer, strict_roster=strict_roster)
    _maybe_patch_unsloth_grad_accum(trainer)
    _maybe_patch_unsloth_loss_type(trainer)
    _maybe_patch_unsloth_runtime_attrs(trainer)
    _maybe_patch_unsloth_args_attrs(trainer)
    _maybe_patch_nanmin_symbols()
    _preflight_unsloth_grpo_args(trainer)
    
    # ── ADD THIS: Extra safety check for any remaining missing attrs ──
    # Some Unsloth compiled paths access these directly on the trainer
    _safety_attrs = {
        "importance_sampling_level": "token",
        "epsilon_low": 0.2,
        "epsilon_high": 0.2,
        "vllm_importance_sampling_cap": 2.0,
        "current_gradient_accumulation_steps": getattr(trainer, "current_gradient_accumulation_steps", 1),
    }
    for attr, default in _safety_attrs.items():
        if not hasattr(trainer, attr):
            setattr(trainer, attr, default)
            print(f"[SAFETY] Added missing trainer.{attr} = {default!r}")
    
    # ── ADD THIS: Delete stale compiled cache so Unsloth rebuilds with patched attrs ──
    import shutil
    compiled_cache_dirs = [
        Path.cwd() / "unsloth_compiled_cache",
        Path.home() / ".cache" / "unsloth" / "compiled_cache",
    ]
    for cache_dir in compiled_cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"[INFO] Deleted stale compiled cache: {cache_dir}")
    
    # Now train with fresh cache
    trainer.train()

    if curriculum_manager is not None:
        curriculum_manager.save()
    elif use_grounded_curriculum and cc_runtime is not None:
        save_continuous_state(continuous_path, cc_runtime)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    curves_path = Path(output_dir) / "reward_curves.json"
    _save_json(reward_log, curves_path)
    print(f"Reward curves -> {curves_path}")
    diagnostics_path = Path(output_dir) / "training_diagnostics.json"
    _save_json(
        {
            "reward_log": reward_log,
            "parse_log": parse_log,
            "action_signatures": action_signature_log,
            "difficulty_log": difficulty_log,
            "batch_diagnostics": batch_diagnostics,
            "parse_debug_samples": parse_debug_samples,
            "reward_difficulty_correlation": _safe_corr(difficulty_log, reward_log["composite"]),
            "reward_by_difficulty_bin": _difficulty_reward_profile(difficulty_log, reward_log["composite"]),
        },
        diagnostics_path,
    )
    print(f"Diagnostics -> {diagnostics_path}")

    _print_final_stats(reward_log)

    # ── Post-training eval ────────────────────────────────────────────────────
    if run_eval:
        print("\n[Post] Measuring trained model score...")
        FastLanguageModel.for_inference(model)  # fuse LoRA weights for faster generation
        trained_model_metrics = _run_model_episodes(
            model,
            tokenizer,
            n_episodes=max(1, int(eval_episodes)),
            tag="TRAINED MODEL",
            eval_task_ids=eval_task_ids,
        )
        _save_json(trained_model_metrics, Path(output_dir) / "trained_model_metrics.json")

        if base_model_metrics is not None:
            _print_improvement(base_model_metrics, trained_model_metrics)
        else:
            # Fallback: compare heuristic baseline vs trained model
            _print_improvement(
                {**baseline, "tag": "HEURISTIC BASELINE"},
                {**trained_model_metrics, "tag": "TRAINED MODEL"},
            )

    if push_to_hub and hub_model_id:
        print(f"\nPushing to Hub: {hub_model_id}")
        trainer.push_to_hub(hub_model_id)

    return trainer


# ── Quick heuristic eval (no LLM needed — uses planner baseline) ──────────────

def _quick_heuristic_eval(n_episodes: int = 6) -> Dict[str, Any]:
    """Run heuristic-only multi-agent episodes (client=None → deterministic planner).

    Uses run_episode so metrics are AMAN/DMAN rewards from the real multi-agent
    environment — not single-agent grades. Same format as _run_model_episodes so
    _print_improvement can compare them directly.
    """
    from multi_agent.inference import run_episode as _run_ep

    env = MultiAgentATCEnvironment(seed=99)
    sup = SupervisorAgent()

    # Fixed task list — no generator mutations for a stable repeatable baseline
    eval_tasks = ["delhi_monsoon_recovery_easy", "bengaluru_irrops_hard"]

    composites, aman_rews, dman_rews, conflict_list, emg_list = [], [], [], [], []

    for ep in range(n_episodes):
        task_id = eval_tasks[ep % len(eval_tasks)]
        try:
            r = _run_ep(
                task_id      = task_id,
                client       = None,   # heuristic mode — no LLM
                env          = env,
                generator    = None,
                supervisor   = sup,
                episode_id   = ep,
                use_generator= False,
            )
            composites.append(float(r.get("composite", 0)))
            aman_rews.append(float(r.get("aman_reward", 0)))
            dman_rews.append(float(r.get("dman_reward", 0)))
            conflict_list.append(int(r.get("conflicts", 0)))
            emg_list.append(int(r.get("emg_arr_ok", 0)) + int(r.get("emg_dep_ok", 0)))
        except Exception as exc:
            print(f"  [WARN] Heuristic eval ep {ep} failed: {exc}")

    def _mean(lst: list) -> float:
        return round(sum(lst) / max(1, len(lst)), 3) if lst else 0.0

    return {
        "tag":              "HEURISTIC BASELINE",
        "n_episodes":       n_episodes,
        "mean_composite":   _mean(composites),
        "mean_aman_reward": _mean(aman_rews),
        "mean_dman_reward": _mean(dman_rews),
        "mean_conflicts":   _mean(conflict_list),
        "mean_emg_handled": _mean(emg_list),
        "scores":           [round(s, 3) for s in composites],
    }


def _print_improvement(
    before: Dict[str, Any], after: Dict[str, Any]
) -> None:
    tag_b = before.get("tag", "BEFORE")
    tag_a = after.get("tag", "AFTER")
    rows = [
        ("mean_composite",   "Composite score"),
        ("mean_aman_reward", "AMAN reward"),
        ("mean_dman_reward", "DMAN reward"),
        ("mean_conflicts",   "Avg conflicts"),
        ("mean_emg_handled", "Emg handled"),
    ]
    width = 56
    print(f"\n{'='*width}")
    print(f"  BEFORE vs AFTER TRAINING")
    print(f"  {tag_b!r:24s}  →  {tag_a!r}")
    print(f"{'='*width}")
    for key, label in rows:
        bv = before.get(key, 0.0)
        av = after.get(key, 0.0)
        delta = av - bv
        arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
        sign = "+" if delta >= 0 else ""
        print(f"  {label:20s}: {bv:6.3f}  →  {av:6.3f}  ({sign}{delta:.3f} {arrow})")
    print(f"{'='*width}")


# ── Local model client for in-process inference eval ──────────────────────────

class _LocalModelClient:
    """Duck-type OpenAI client wrapping a locally loaded Unsloth/PEFT model."""

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        if hasattr(self._model, "generation_config") and self._model.generation_config is not None:
            self._model.generation_config.max_length = None

    def _create(self, *, model=None, messages, temperature=0.3, max_tokens=MAX_NEW_TOKENS, **kw):
        import torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        use_cuda = torch.cuda.is_available() and str(self._model.device).startswith("cuda")
        cast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=cast_dtype):
                    out = self._model.generate(
                        **inputs,
                        max_new_tokens=min(int(max_tokens), MAX_NEW_TOKENS),
                        temperature=max(float(temperature), 0.01),
                        do_sample=float(temperature) > 0.01,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
            else:
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=min(int(max_tokens), MAX_NEW_TOKENS),
                    temperature=max(float(temperature), 0.01),
                    do_sample=float(temperature) > 0.01,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        text = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        class _Msg:
            content = text
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]

        return _Resp()

    @property
    def chat(self):
        _self = self
        class _Comp:
            def create(self, **kw):
                return _self._create(**kw)
        class _Chat:
            completions = _Comp()
        return _Chat()


def _run_model_episodes(
    model,
    tokenizer,
    n_episodes: int = 3,
    tag: str = "MODEL",
    use_generator: bool = False,
    eval_task_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run multi-agent episodes using an in-process model client.

    use_generator=False keeps tasks fixed so base and trained models
    see identical scenarios — essential for a fair comparison.
    """
    from multi_agent.inference import run_episode as _run_ep

    client = _LocalModelClient(model, tokenizer)
    env = MultiAgentATCEnvironment(seed=77)
    gen = ChallengeGenerator(seed=77)
    sup = SupervisorAgent()

    # Two representative tasks: one easy, one hard
    eval_tasks = eval_task_ids or ["delhi_monsoon_recovery_easy", "bengaluru_irrops_hard"]

    composites, aman_rews, dman_rews, conflict_list, emg_list = [], [], [], [], []

    for ep in range(n_episodes):
        task_id = eval_tasks[ep % len(eval_tasks)]
        try:
            r = _run_ep(
                task_id=task_id,
                client=client,
                env=env,
                generator=gen if use_generator else None,
                supervisor=sup,
                episode_id=ep,
                use_generator=use_generator,
                model_name="local",
            )
            composites.append(float(r.get("composite", 0)))
            aman_rews.append(float(r.get("aman_reward", 0)))
            dman_rews.append(float(r.get("dman_reward", 0)))
            conflict_list.append(int(r.get("conflicts", 0)))
            emg_list.append(
                int(r.get("emg_arr_ok", 0)) + int(r.get("emg_dep_ok", 0))
            )
        except Exception as exc:
            print(f"  [WARN] model eval episode {ep} failed: {exc}")

    def _m(lst: list) -> float:
        return round(sum(lst) / max(1, len(lst)), 3) if lst else 0.0

    return {
        "tag":              tag,
        "n_episodes":       n_episodes,
        "mean_composite":   _m(composites),
        "mean_aman_reward": _m(aman_rews),
        "mean_dman_reward": _m(dman_rews),
        "mean_conflicts":   _m(conflict_list),
        "mean_emg_handled": _m(emg_list),
        "scores":           [round(s, 3) for s in composites],
    }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _wandb_available() -> bool:
    try:
        import wandb
        return True
    except ImportError:
        return False


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _vram_snapshot(torch_mod: Any) -> Dict[str, Any]:
    snap: Dict[str, Any] = {"cuda_available": bool(torch_mod.cuda.is_available())}
    if not snap["cuda_available"]:
        return snap
    try:
        free_b, total_b = torch_mod.cuda.mem_get_info()
        snap["free_bytes"] = int(free_b)
        snap["total_bytes"] = int(total_b)
        snap["free_gib"] = round(free_b / (1024**3), 3)
        snap["total_gib"] = round(total_b / (1024**3), 3)
    except Exception as exc:
        snap["error"] = str(exc)
    return snap


def _list_std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _dump_grpo_checkpoint_bundle(
    out_root: Path,
    step: int,
    reward_log: Dict[str, List[float]],
    parse_log: Dict[str, List[int]],
    difficulty_log: List[float],
    batch_diagnostics: List[Dict[str, Any]],
    parse_debug_samples: List[Dict[str, Any]],
    torch_mod: Any,
) -> None:
    """JSON + curve plot + summary table under checkpoint_artifacts/ (GRPO step 2 only)."""
    art = out_root / "checkpoint_artifacts" / f"step_{step:06d}"
    art.mkdir(parents=True, exist_ok=True)
    window = 256
    tail_stats: Dict[str, Any] = {"reward_mean_std": {}, "parse_rate_tail": {}}
    for role, xs in reward_log.items():
        if not xs:
            continue
        t = xs[-min(window, len(xs)) :]
        tail_stats["reward_mean_std"][role] = {
            "mean": round(sum(t) / len(t), 4),
            "std": round(_list_std(t), 4),
            "n": len(t),
        }
    for role, xs in parse_log.items():
        if not xs:
            continue
        t = xs[-min(window, len(xs)) :]
        tail_stats["parse_rate_tail"][role] = round(sum(t) / len(t), 4)

    summary = {
        "global_step": step,
        "vram": _vram_snapshot(torch_mod),
        "tail_stats": tail_stats,
        "n_batch_diagnostics": len(batch_diagnostics),
    }
    _save_json(summary, art / "metrics_summary.json")

    curves = {k: list(v) for k, v in reward_log.items()}
    _save_json(curves, art / "reward_curves_snapshot.json")
    _save_json({k: list(v) for k, v in parse_log.items()}, art / "parse_log_snapshot.json")
    _save_json(list(difficulty_log), art / "difficulty_log_snapshot.json")
    _save_json(batch_diagnostics[-500:], art / "batch_diagnostics_tail.json")
    _save_json(parse_debug_samples[-120:], art / "parse_debug_samples_tail.json")

    idx_path = out_root / "checkpoint_artifacts" / "index.jsonl"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    with idx_path.open("a", encoding="utf-8") as lf:
        lf.write(json.dumps({"step": step, "artifact_dir": art.name, **summary["vram"]}) + "\n")

    try:
        from training.plot_rewards import plot_training_curves

        plot_training_curves(curves, save_dir=str(art), show=False)
    except Exception as exc:
        print(f"[WARN] checkpoint training_curves plot: {exc}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        roles = ["AMAN", "DMAN", "GENERATOR", "SUPERVISOR", "ADAPT"]
        rows_tbl: List[List[str]] = []
        for r in roles:
            ms = tail_stats["reward_mean_std"].get(r, {})
            pr = tail_stats["parse_rate_tail"].get(r, "")
            rows_tbl.append(
                [
                    r,
                    f"{ms.get('mean', '')}" if ms else "",
                    f"{ms.get('std', '')}" if ms else "",
                    f"{pr}" if pr != "" else "",
                ]
            )
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        tbl = ax.table(
            cellText=rows_tbl,
            colLabels=["role", "reward_mean_tail", "reward_std_tail", "parse_rate_tail"],
            loc="center",
            cellLoc="center",
        )
        tbl.scale(1.2, 1.8)
        ax.set_title(f"GRPO checkpoint summary (step {step})", fontsize=12)
        fig.tight_layout()
        fig.savefig(art / "metrics_table.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] checkpoint metrics_table.png: {exc}")


def _check_reward_hacking(reward_log: Dict[str, List[float]]) -> None:
    """Warn when mean composite rises but role rewards collapse (gaming signal)."""
    comp = reward_log["composite"]
    if len(comp) < 100:
        return
    recent_50  = comp[-50:]
    earlier_50 = comp[-100:-50]
    mean_recent  = sum(recent_50)  / 50
    mean_earlier = sum(earlier_50) / 50
    if mean_recent > mean_earlier + 0.1:
        # Check if any role's recent std collapsed (< 0.05 = suspiciously uniform)
        for role in ("AMAN", "DMAN"):
            rs = reward_log.get(role, [])
            if len(rs) >= 20:
                recent = rs[-20:]
                mean_r = sum(recent) / len(recent)
                std_r = (sum((x - mean_r) ** 2 for x in recent) / len(recent)) ** 0.5
                if std_r < 0.05:
                    print(
                        f"[WARN] Possible reward hacking: {role} std={std_r:.4f} "
                        f"while composite reward is rising. Sample outputs and inspect."
                    )


def _print_final_stats(reward_log: Dict[str, List[float]]) -> None:
    print("\n=== TRAINING REWARD SUMMARY ===")
    for role, rewards in reward_log.items():
        if not rewards:
            continue
        n = len(rewards)
        first_q = rewards[:max(1, n // 4)]
        last_q  = rewards[max(0, 3 * n // 4):]
        mean_all   = sum(rewards) / n
        mean_first = sum(first_q) / len(first_q)
        mean_last  = sum(last_q)  / len(last_q)
        trend = "↑" if mean_last > mean_first + 0.05 else (
            "↓" if mean_last < mean_first - 0.05 else "→"
        )
        print(
            f"  {role:12s}: mean={mean_all:.3f} | "
            f"first_q={mean_first:.3f} -> last_q={mean_last:.3f} {trend}"
        )


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(model_name_or_path: str, n_episodes: int = 20, seed: int = 99) -> Dict[str, Any]:
    """Run trained model on evaluation episodes."""
    torch, FastLanguageModel, _, _ = _require_training_deps()
    _configure_runtime_warnings()

    print(f"\nEvaluating {model_name_or_path} on {n_episodes} episodes...")
    model_source = _prefer_local_model_path(model_name_or_path)
    model, tokenizer = _load_model_with_fallback(
        FastLanguageModel,
        model_source,
        max_seq_length=MAX_SEQ_LEN,
    )
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    FastLanguageModel.for_inference(model)

    import random
    from training.dataset import AMAN_SYSTEM, DMAN_SYSTEM, SUPERVISOR_PROFILES

    env        = MultiAgentATCEnvironment(seed=seed)
    generator  = ChallengeGenerator(seed=seed)
    supervisor = SupervisorAgent()
    task_list  = list(ordered_tasks())
    rng        = random.Random(seed)

    results: Dict[str, List] = {
        "aman_rewards": [], "dman_rewards": [], "composite_scores": [],
        "conflict_counts": [], "coordination_scores": [],
        "generator_difficulty": [],
    }

    for ep in range(n_episodes):
        base_task = rng.choice(task_list)
        profile   = supervisor.sample_profile(ep)
        mutated, is_solvable = generator.mutate(base_task)

        aman_obs, dman_obs = env.reset(episode_id=ep, mutated_task=mutated)
        sup_desc = SUPERVISOR_PROFILES[profile]["description"]

        def _chat(system, user):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        def _gen(prompt):
            import torch as _torch
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with _torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE, do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        aman_comp = _gen(_chat(AMAN_SYSTEM + f"\n\nSUPERVISOR: {sup_desc}", aman_obs.to_prompt_text()))
        dman_comp = _gen(_chat(DMAN_SYSTEM + f"\n\nSUPERVISOR: {sup_desc}", dman_obs.to_prompt_text()))

        aman_action = parse_aman_action(aman_comp)
        dman_action = parse_dman_action(dman_comp)
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
        results["generator_difficulty"].append(generator.difficulty_level)

        print(
            f"  ep{ep:3d} | composite={result.composite_score:.3f} | "
            f"AMAN={result.aman_reward:.3f} | DMAN={result.dman_reward:.3f} | "
            f"coord={result.per_role.coordination_score:.3f} | "
            f"gen_lvl={generator.difficulty_level}"
        )

    def _mean(lst):
        return round(sum(lst) / max(1, len(lst)), 3)

    summary = {
        "mean_composite":    _mean(results["composite_scores"]),
        "mean_aman_reward":  _mean(results["aman_rewards"]),
        "mean_dman_reward":  _mean(results["dman_rewards"]),
        "mean_coordination": _mean(results["coordination_scores"]),
        "mean_conflicts":    _mean(results["conflict_counts"]),
        "final_gen_difficulty": results["generator_difficulty"][-1] if results["generator_difficulty"] else 1,
    }
    print("\n=== EVALUATION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return {**results, "summary": summary}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ATC Multi-Agent GRPO Training")
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--output_dir",     default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes",       type=int, default=200)
    parser.add_argument("--lora_rank",      type=int, default=LORA_RANK)
    parser.add_argument("--n_generations",  type=int, default=None,
                        help="GRPO group size (default: N_GENERATIONS constant). "
                             "Use 2 on T4 Colab, 4 for best gradient quality.")
    parser.add_argument("--batch_size",     type=int, default=None)
    parser.add_argument("--grad_accum",     type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature",    type=float, default=None)
    parser.add_argument("--logging_steps",  type=int, default=None)
    parser.add_argument("--eval_episodes",  type=int, default=3,
                        help="Episodes for pre/post training eval runs.")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--no_eval",        action="store_true", help="Skip before/after eval")
    parser.add_argument("--eval_only",      action="store_true")
    parser.add_argument("--push_to_hub",    action="store_true")
    parser.add_argument("--hub_model_id",   default=None)
    parser.add_argument(
        "--grounded_curriculum",
        action="store_true",
        help="Train on deterministic grounded gc_* tasks only (no generator, no ADAPT, no env randomize).",
    )
    parser.add_argument(
        "--curriculum_state",
        default=None,
        help="Optional path to continuous_curriculum_state.json from a prior run (copied into output_dir for live; passed to static builder otherwise).",
    )
    parser.add_argument(
        "--static_grounded_dataset",
        action="store_true",
        help="With --grounded_curriculum, use a fixed pre-built dataset (disables live iterable resampling).",
    )
    parser.add_argument(
        "--adapter_in",
        default=None,
        help="Directory with a prior LoRA adapter (e.g. outputs/atc-sft-json from train_sft.py).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Cap GRPO optimizer steps (dev/smoke). Default: derived from --episodes and batch layout.",
    )
    parser.add_argument(
        "--mid_eval_every",
        type=int,
        default=0,
        help="GRPO only: run full _run_model_episodes mid-train every N global steps (min 50; "
        "also set ATC_MID_EVAL_EVERY). Uses ≥50 episodes unless --mid_eval_episodes set.",
    )
    parser.add_argument(
        "--mid_eval_episodes",
        type=int,
        default=0,
        help="GRPO only: mid-eval episode count (default 0 = max(50, --eval_episodes)). "
        "Values <50 are raised to 50.",
    )
    args = parser.parse_args()

    # Allow CLI override of group size (useful for Colab memory tuning)
    if args.n_generations is not None:
        global N_GENERATIONS, BATCH_SIZE
        N_GENERATIONS = args.n_generations
        # Adjust batch size to stay divisible
        if BATCH_SIZE % N_GENERATIONS != 0:
            BATCH_SIZE = N_GENERATIONS

    global GRAD_ACCUM, MAX_NEW_TOKENS, LOGGING_STEPS, TEMPERATURE
    if args.batch_size is not None:
        BATCH_SIZE = max(1, args.batch_size)
    if args.grad_accum is not None:
        GRAD_ACCUM = max(1, args.grad_accum)
    if args.max_new_tokens is not None:
        MAX_NEW_TOKENS = max(32, args.max_new_tokens)
    if args.temperature is not None:
        TEMPERATURE = max(0.1, min(1.5, args.temperature))
    if args.logging_steps is not None:
        LOGGING_STEPS = max(1, args.logging_steps)

    if args.eval_only:
        evaluate(args.model, n_episodes=max(1, int(args.eval_episodes)), seed=args.seed)
    else:
        if args.static_grounded_dataset:
            os.environ["ATC_STATIC_GROUNDED_DATASET"] = "1"
        me_every = int(args.mid_eval_every)
        if me_every == 0:
            me_every = _env_int("ATC_MID_EVAL_EVERY", 0)
        me_eps = int(args.mid_eval_episodes) if args.mid_eval_episodes else None
        train(
            model_name=args.model,
            output_dir=args.output_dir,
            n_episodes=args.episodes,
            lora_rank=args.lora_rank,
            seed=args.seed,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            run_eval=not args.no_eval,
            eval_episodes=max(1, int(args.eval_episodes)),
            use_grounded_curriculum=bool(args.grounded_curriculum),
            curriculum_state_path=args.curriculum_state,
            adapter_in=args.adapter_in,
            max_steps=args.max_steps,
            mid_eval_every=me_every,
            mid_eval_episodes=me_eps,
        )


if __name__ == "__main__":
    main()
