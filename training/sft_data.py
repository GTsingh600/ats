"""Gold JSON rows for supervised fine-tuning before GRPO.

Each row is a single ``text`` field: full chat transcript (system + user + assistant)
where the assistant message is **deterministic** JSON that passes the same parsers
used in ``training/dataset.py`` / reward functions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine import simulate_plan
from models import OperationType, SlotAssignment, TaskDefinition

from training.curriculum_grounded import (
    GroundedCurriculumState,
    grounded_task_proxy_score,
    solve_grounded_rule_based,
)
from training.continuous_curriculum import (
    apply_meaningful_structural_variation,
    balanced_d_sequence,
    bucket_index,
    difficulty_proxy_task,
    load_or_create_continuous_state,
    scenario_features,
    softmax_pick_grounded_task,
    validate_bucket_balance,
)
from training.dataset import (
    SUPERVISOR_PROFILES,
    _build_solver_merged_plan_json,
    _make_aman_sample,
    _make_dman_sample,
    _make_supervisor_sample,
)
from tasks_grounded import GROUNDED_LEVEL_BY_TASK_ID
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.models import AgentRole
from multi_agent.supervisor import SupervisorAgent
def _slots_for_operation(
    task: TaskDefinition, slots: List[SlotAssignment], op: OperationType
) -> List[Dict[str, Any]]:
    by_fid = {f.flight_id: f.operation for f in task.flights}
    rows: List[Dict[str, Any]] = []
    for s in slots:
        if by_fid.get(s.flight_id) != op:
            continue
        rows.append(
            {
                "flight_id": str(s.flight_id),
                "runway": str(s.runway),
                "assigned_minute": int(s.assigned_minute),
                "hold_minutes": int(s.hold_minutes),
            }
        )
    return rows


def _gold_aman_json(task: TaskDefinition, slots: List[SlotAssignment]) -> str:
    payload = {
        "arrival_slots": _slots_for_operation(task, slots, OperationType.ARRIVAL),
        "rationale": "Reference feasible arrival plan (solver).",
        "emergency_yields": [],
        "outgoing_messages": [],
        "commit": True,
    }
    return json.dumps(payload, ensure_ascii=False)


def _gold_dman_json(task: TaskDefinition, slots: List[SlotAssignment]) -> str:
    payload = {
        "departure_slots": _slots_for_operation(task, slots, OperationType.DEPARTURE),
        "rationale": "Reference feasible departure plan (solver).",
        "atfm_compliance": {},
        "emergency_broadcasts": [],
        "outgoing_messages": [],
        "commit": True,
    }
    return json.dumps(payload, ensure_ascii=False)


def _gold_supervisor_json(task: TaskDefinition, profile, slots: List[SlotAssignment], sup: SupervisorAgent) -> str:
    outcome = simulate_plan(task, slots)
    score = float(sup.score_plan(outcome, task, profile))
    payload = {
        "score": round(max(0.0, min(1.0, score)), 4),
        "alignment": "Reference evaluation from simulator metrics under the active profile.",
        "key_violations": [],
    }
    return json.dumps(payload, ensure_ascii=False)


def _messages_plus_assistant(prompt: List[Dict[str, str]], gold: str) -> List[Dict[str, str]]:
    out = [dict(x) for x in prompt]
    out.append({"role": "assistant", "content": gold})
    return out


def build_grounded_json_sft_rows(
    n_episodes: int,
    seed: int,
    *,
    continuous_curriculum_path: Optional[str] = None,
    curriculum_state_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build SFT rows (grounded curriculum only) with gold JSON completions.

    Skips episodes where ``solve_grounded_rule_based`` fails (rare malformed templates).
    """
    rng = random.Random(int(seed))
    supervisor = SupervisorAgent()
    env = MultiAgentATCEnvironment(seed=seed)

    grounded_state: Optional[GroundedCurriculumState] = None
    if curriculum_state_path:
        p = Path(curriculum_state_path)
        if p.is_file():
            grounded_state = GroundedCurriculumState.from_json_file(p)
    if grounded_state is None:
        grounded_state = GroundedCurriculumState()

    cc_path = Path(continuous_curriculum_path) if continuous_curriculum_path else None
    cc_state = load_or_create_continuous_state(cc_path)
    if not cc_state.use_adaptive_sampling:
        if n_episodes > 0:
            validate_bucket_balance(n_episodes)
        d_sequence = balanced_d_sequence(n_episodes, rng)
    else:
        d_sequence = [cc_state.sample_d(rng) for _ in range(n_episodes)]

    rows: List[Dict[str, Any]] = []

    for ep_id in range(n_episodes):
        d = float(d_sequence[ep_id])
        base_task = softmax_pick_grounded_task(d, rng, k=4, temperature=7.0)
        mutated_task = apply_meaningful_structural_variation(base_task, d, rng)
        difficulty_scalar = d
        grounded_meta = {
            "grounded_curriculum": True,
            "continuous_difficulty": d,
            "difficulty_proxy": float(difficulty_proxy_task(mutated_task)),
            "difficulty_bucket_index": bucket_index(d),
            "scenario_features": scenario_features(d),
            "grounded_level": int(GROUNDED_LEVEL_BY_TASK_ID.get(base_task.task_id, 0)),
            "grounded_template_id": base_task.task_id,
            "training_band": f"bucket_{bucket_index(d)}",
            "curriculum_active_level": grounded_state.active_level,
            "rule_proxy_score": float(grounded_task_proxy_score(mutated_task)),
        }

        profile = supervisor.sample_profile(ep_id)
        sup_desc = SUPERVISOR_PROFILES[profile]["description"]

        aman_obs, dman_obs = env.reset(
            episode_id=ep_id,
            supervisor_profile=profile,
            mutated_task=mutated_task,
            randomize=False,
        )
        atfm_json = json.dumps(env._state.atfm_deadlines)

        slots = solve_grounded_rule_based(mutated_task)
        if not slots:
            continue

        aman_s = _make_aman_sample(
            ep_id, aman_obs, atfm_json, "[]", sup_desc, profile, "bid", difficulty_scalar
        )
        aman_s.update(grounded_meta)
        dman_s = _make_dman_sample(
            ep_id, dman_obs, atfm_json, "[]", sup_desc, profile, "bid", difficulty_scalar
        )
        dman_s.update(grounded_meta)

        merged = _build_solver_merged_plan_json(mutated_task)
        sup_s = _make_supervisor_sample(
            ep_id, mutated_task, profile, sup_desc, difficulty_scalar, merged_plan_json=merged
        )
        sup_s.update(grounded_meta)

        gold_a = _gold_aman_json(mutated_task, slots)
        gold_d = _gold_dman_json(mutated_task, slots)
        gold_s = _gold_supervisor_json(mutated_task, profile, slots, supervisor)

        for sample, gold, role in (
            (aman_s, gold_a, AgentRole.AMAN.value),
            (dman_s, gold_d, AgentRole.DMAN.value),
            (sup_s, gold_s, AgentRole.SUPERVISOR.value),
        ):
            msgs = _messages_plus_assistant(sample["prompt"], gold)
            rows.append(
                {
                    "messages": msgs,
                    "agent_role": role,
                    "task_id": sample.get("task_id", ""),
                    "episode_id": ep_id,
                }
            )

    return rows


def materialize_text_rows(rows: List[Dict[str, Any]], tokenizer) -> List[Dict[str, str]]:
    """Add ``text`` via chat template (Qwen / compatible tokenizers)."""
    out: List[Dict[str, str]] = []
    for r in rows:
        msgs = r["messages"]
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        out.append({"text": text})
    return out
