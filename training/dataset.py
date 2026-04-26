"""Episode dataset builder for multi-agent GRPO training.

Each training sample = one agent turn in one episode.
Format required by TRL GRPOTrainer:
    {"prompt": [{"role": "system", "content": ...}, {"role": "user", "content": ...}],
     "task_id": ..., "agent_role": ..., ...metadata...}

System prompts encode:
  - Role identity + operational rules
  - Output JSON schema (strict)
  - Supervisor preference for this episode
  - Negotiation protocol rules

Parsing utilities decode LLM JSON completions back to typed actions.
"""

from __future__ import annotations

import json
import re
import sys, os
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import simulate_plan
from models import OperationType, SlotAssignment, TaskDefinition
from tasks import ordered_tasks
from tasks_grounded import GROUNDED_LEVEL_BY_TASK_ID
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
from training.curriculum_grounded import (
    GroundedCurriculumState,
    grounded_task_proxy_score,
    solve_grounded_rule_based,
)
def conflict_proxy_from_solver(task: TaskDefinition) -> float:
    """Normalised conflict pressure [0,1] from solver-merged plan (for ADAPT triggers)."""
    try:
        raw = _build_solver_merged_plan_json(task)
        slots = [SlotAssignment(**s) for s in json.loads(raw)]
        oc = simulate_plan(task, slots)
        return min(1.0, oc.metrics.conflict_count / max(1, len(task.flights)))
    except Exception:
        return 0.0


from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.generator import ChallengeGenerator
from multi_agent.models import (
    AMANAction,
    ADAPTObservation,
    DMANAction,
    GeneratorAction,
    GeneratorMutation,
    MutationType,
    NegotiationMessage,
    MessageType,
    AgentRole,
    SupervisorProfileName,
    SUPERVISOR_PROFILES,
)
from multi_agent.supervisor import SupervisorAgent


# ── System prompts ────────────────────────────────────────────────────────────

AMAN_SYSTEM = """You are AMAN (Arrival Manager) at a busy Indian airport.
You ONLY control ARRIVAL flights. Do NOT assign departure flights.

CORE RULES (non-negotiable):
1. EMERGENCY and MEDICAL arrivals land FIRST — delay them ≤5 min maximum.
2. Respect wake turbulence separation: H→H≥4min, H→M≥5min, H→L≥6min, M→M≥3min.
3. Every arrival must stay within its [earliest, latest] window.
4. Only assign each flight to runways listed in its allowed_runways.
5. If DMAN broadcasts an EMERGENCY departure, yield your next runway slot to them.
6. Pre-empt gaps: if you know a DMAN emergency is at T+N, leave runway clear ±3 min.

NEGOTIATION PROTOCOL:
- Round BID: submit your best independent plan.
- Round NEGOTIATE: if conflicts reported, revise plan and send yield/acknowledge messages.
- Use outgoing_messages to communicate runway claims and yields to DMAN.

OUTPUT FORMAT (strict JSON, no markdown):
{
  "arrival_slots": [
    {"flight_id": "...", "runway": "...", "assigned_minute": N, "hold_minutes": N}
  ],
  "rationale": "explain your sequencing decisions and how you satisfy supervisor preference",
  "emergency_yields": ["flight_id_you_yielded_for"],
  "outgoing_messages": [
    {
      "from_role": "AMAN",
      "message_type": "runway_claim|yield|acknowledge|request_gap|emergency_broadcast",
      "flight_id": "...",
      "requested_minute": N,
      "runway_id": "...",
      "priority": "normal|connection|medical|emergency",
      "reason": "...",
      "is_emergency": false
    }
  ],
  "commit": false
}"""


DMAN_SYSTEM = """You are DMAN (Departure Manager) at a busy Indian airport.
You ONLY control DEPARTURE flights. Do NOT assign arrival flights.

CORE RULES (non-negotiable):
1. ATFM network slot deadlines are HARD — missing them cascades to 3+ airports.
2. MEDICAL and EMERGENCY departures jump to the front of the departure queue.
3. Every departure must stay within its [earliest, latest] window.
4. Only assign each flight to runways listed in its allowed_runways.
5. If AMAN broadcasts an EMERGENCY arrival, clear the runway immediately.
6. Broadcast your own fuel/medical emergencies to AMAN in outgoing_messages.

PRIORITY RULE (air vs ground):
If BOTH a medical ARRIVAL and a medical DEPARTURE need the same slot:
→ The ARRIVAL wins (airborne aircraft cannot divert fuel-free; ground can hold).

NEGOTIATION PROTOCOL:
- Round BID: submit your best independent plan.
- Round NEGOTIATE: revise after conflict report; send messages to AMAN.

OUTPUT FORMAT (strict JSON, no markdown):
{
  "departure_slots": [
    {"flight_id": "...", "runway": "...", "assigned_minute": N, "hold_minutes": N}
  ],
  "rationale": "explain sequencing and ATFM compliance and supervisor preference",
  "atfm_compliance": {"flight_id": deadline_minute_you_respected},
  "emergency_broadcasts": ["flight_id_of_your_emergency_departures"],
  "outgoing_messages": [
    {
      "from_role": "DMAN",
      "message_type": "runway_claim|yield|acknowledge|request_gap|emergency_broadcast",
      "flight_id": "...",
      "requested_minute": N,
      "runway_id": "...",
      "priority": "normal|connection|medical|emergency",
      "reason": "...",
      "is_emergency": false
    }
  ],
  "commit": false
}"""


GENERATOR_SYSTEM = """You are the Scenario Generator for multi-agent ATC training.
Your goal: mutate the scenario to make AMAN and DMAN fail to coordinate.
You are rewarded when they score LOW. You are penalised if the scenario is UNSOLVABLE.

MUTATION TYPES:
- tighten_window: squeeze a flight's time window (make it harder to sequence)
- inject_emergency: add a new EMERGENCY/MEDICAL arrival to disrupt sequencing
- increase_weather_penalty: degrade runway capacity
- add_atfm_deadline: add a hard network slot constraint to a departure
- close_runway_window: make a runway unavailable during peak period
- add_conflicting_flight: inject a Heavy arrival before a Light to create wake trap

STRATEGY TIPS:
- Simultaneous medical arrival + fuel emergency departure on same runway = maximum conflict
- Injecting emergency during peak hour breaks AMAN's sequence
- ATFM deadlines during weather degradation stress DMAN

OUTPUT FORMAT (strict JSON, no markdown):
{
  "mutations": [
    {
      "mutation_type": "tighten_window|inject_emergency|increase_weather_penalty|add_atfm_deadline|close_runway_window|add_conflicting_flight",
      "target_flight_id": "flight_id or null",
      "target_runway_id": "runway_id or null",
      "params": {"key": "value"},
      "rationale": "why this breaks coordination"
    }
  ],
  "strategy": "overall explanation of how these mutations disrupt AMAN/DMAN coordination"
}"""


SUPERVISOR_SYSTEM_TEMPLATE = """You are an ATC Supervisor evaluating a completed runway plan.
Your preference this shift: {preference}

Score the plan 0.0-1.0 based on how well it satisfies YOUR preference (not generic quality).
Be specific about what satisfies or violates your stated priority.

OUTPUT FORMAT (strict JSON, no markdown):
{{
  "score": 0.0,
  "alignment": "explain how well the plan matches your stated preference",
  "key_violations": ["list specific violations of your preference"]
}}"""


ADAPT_SYSTEM = """You are ADAPT (STRUCTURAL Domain Meta-Agent).
You are given a scheduling task from an UNKNOWN domain (e.g. Hospital ICU, Port Logistics).
You do NOT know the domain's terminology. You must ignore labels like "TRAUMA" or "BERTH" and focus on:
1. time_pressure: How narrow is the execution window?
2. connection_risk: Is this entity part of a sequence (risk of cascade)?
3. Resource Intensity: How much runway/resource time does it need?

Your job: Map these abstract entities into ATC-specific parameters (Wake Class and Priority)
so that the existing AMAN/DMAN models can solve the task with zero retraining.

MAPPING GUIDE:
- Wake Class (H, M, L): Structural separation. Map high-intensity/high-risk to 'H', low to 'L'.
- Priority (emergency, medical, connection, normal): Sequence urgency. Map highest time pressure to 'emergency'.

OUTPUT FORMAT (strict JSON, no markdown):
{
  "entity_wake_map": {"ENTITY_TYPE_A": "H|M|L", "ENTITY_TYPE_B": "..."},
  "entity_priority_map": {"ENTITY_TYPE_A": "emergency|medical|connection|normal", ...},
  "rationale": "Explain using NUMERICAL structural signals (time pressure, risk) why you chose these mappings."
}"""


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_episode_dataset(
    n_episodes: int = 200,
    seed: int = 42,
    include_generator: bool = True,
    include_supervisor: bool = True,
    include_adapt: bool = True,
    domain_episode_ratio: float = 0.30,
    use_grounded_curriculum: bool = False,
    curriculum_state_path: Optional[str] = None,
    continuous_curriculum_path: Optional[str] = None,
    grounded_balanced_buckets: bool = True,
    require_bucket_balance: bool = True,
    training_mode: str = "full",
) -> List[Dict[str, Any]]:
    """Build full multi-agent training dataset.

    ``training_mode`` (see :mod:`training.training_modes`):

    - ``full`` — default multi-agent roster.
    - ``hyper_minimal`` — AMAN+DMAN rows only (faster); ATC ``ordered_tasks`` + generator.
    - ``adapt_multidomain`` — each episode is a domain task only (AMAN+DMAN+ADAPT);
      incompatible with ``use_grounded_curriculum=True``.

    Returns list of training samples, one per agent turn per episode.
    Each episode has: 1 AMAN bid + 1 DMAN bid + optionally 1 negotiation round.
    If include_generator: also 1 generator turn per episode.
    If include_supervisor: also 1 supervisor turn per episode.

    When ``use_grounded_curriculum`` is True, episodes use deterministic canonical
    templates (no ChallengeGenerator mutations, no env parametric randomization).
    Difficulty is a continuous scalar ``d`` in ``[0, 1]`` (``continuous_difficulty`` /
    ``difficulty_scalar``). Initial builds use equal mass per quartile bucket via
    ``balanced_d_sequence`` unless ``continuous_curriculum_state.json`` requests
    adaptive Beta sampling. Templates are chosen with ``nearest_grounded_task(d)``;
    optional timing offsets are solver-gated. Discrete ``difficulty_bucket_index``
    and ``grounded_level`` appear only for logging.
    """
    import random

    from training.training_modes import TrainingMode, resolve_training_mode

    mode = resolve_training_mode(training_mode)
    if mode == TrainingMode.ADAPT_MULTIDOMAIN and use_grounded_curriculum:
        raise ValueError(
            "training_mode=adapt_multidomain is incompatible with use_grounded_curriculum=True"
        )
    if mode == TrainingMode.HYPER_MINIMAL:
        include_generator = False
        include_supervisor = False
        include_adapt = False

    rng = random.Random(seed)
    task_list = list(ordered_tasks())
    supervisor = SupervisorAgent()
    env = MultiAgentATCEnvironment(seed=seed)
    generator = ChallengeGenerator(seed=seed)

    grounded_state: Optional[GroundedCurriculumState] = None
    cc_state = None
    d_sequence: List[float] = []
    if use_grounded_curriculum:
        if curriculum_state_path:
            from pathlib import Path

            p = Path(curriculum_state_path)
            if p.is_file():
                grounded_state = GroundedCurriculumState.from_json_file(p)
        if grounded_state is None:
            grounded_state = GroundedCurriculumState()

        from pathlib import Path as _Path

        cc_path = _Path(continuous_curriculum_path) if continuous_curriculum_path else None
        cc_state = load_or_create_continuous_state(cc_path)
        if grounded_balanced_buckets and not cc_state.use_adaptive_sampling:
            if require_bucket_balance:
                validate_bucket_balance(n_episodes)
            d_sequence = balanced_d_sequence(n_episodes, rng)
        else:
            d_sequence = [cc_state.sample_d(rng) for _ in range(n_episodes)]

    if mode == TrainingMode.ADAPT_MULTIDOMAIN:
        from domains import get_all_domain_tasks
        from multi_agent.adapt import (
            _build_adapt_heuristic,
            apply_adapt_mapping,
            build_adapt_observation,
        )

        domain_tasks_md = get_all_domain_tasks()
        if not domain_tasks_md:
            raise RuntimeError("adapt_multidomain: no domain tasks registered")
        task_ids_md = list(domain_tasks_md.keys())
        samples_md: List[Dict[str, Any]] = []
        for ep_id in range(n_episodes):
            tid = rng.choice(task_ids_md)
            domain_task = domain_tasks_md[tid]
            profile = supervisor.sample_profile(ep_id)
            sup_desc = SUPERVISOR_PROFILES[profile]["description"]
            adapt_obs = build_adapt_observation(domain_task, profile)
            h_action = _build_adapt_heuristic(adapt_obs, domain_task)
            mapped = apply_adapt_mapping(domain_task, h_action)
            aman_obs, dman_obs = env.reset(
                episode_id=ep_id,
                supervisor_profile=profile,
                mutated_task=mapped,
                randomize=True,
            )
            atfm_json = json.dumps(env._state.atfm_deadlines)
            difficulty_scalar = float(rng.uniform(0.25, 0.95))
            tm_meta = {
                "grounded_curriculum": False,
                "training_mode": mode.value,
                "domain_source_task_id": tid,
            }
            aman_s = _make_aman_sample(
                ep_id=ep_id,
                obs=aman_obs,
                atfm_json=atfm_json,
                dman_slots_json="[]",
                sup_desc=sup_desc,
                profile=profile,
                round_name="bid",
                difficulty_scalar=difficulty_scalar,
            )
            aman_s.update(tm_meta)
            dman_s = _make_dman_sample(
                ep_id=ep_id,
                obs=dman_obs,
                atfm_json=atfm_json,
                aman_slots_json="[]",
                sup_desc=sup_desc,
                profile=profile,
                round_name="bid",
                difficulty_scalar=difficulty_scalar,
            )
            dman_s.update(tm_meta)
            adapt_row = _make_adapt_sample(
                ep_id,
                adapt_obs,
                domain_task,
                difficulty_scalar,
                domain_source_task_id=tid,
            )
            adapt_row.update(tm_meta)
            samples_md.extend([aman_s, dman_s, adapt_row])
            controller_score = _estimate_controller_score(mapped)
            generator.update(controller_score)
            generator.record(tid, [], controller_score)
        return samples_md

    samples: List[Dict[str, Any]] = []

    adapt_episode_ids: set[int] = set()
    domain_tasks: Dict[str, TaskDefinition] = {}
    if include_adapt:
        from domains import get_all_domain_tasks

        domain_tasks = get_all_domain_tasks()
        if not use_grounded_curriculum:
            ratio = max(0.25, min(0.30, max(0.0, float(domain_episode_ratio))))
            target_adapt = max(0, min(n_episodes, int(round(n_episodes * ratio))))
            if target_adapt > 0:
                ep_ids = list(range(n_episodes))
                rng.shuffle(ep_ids)
                adapt_episode_ids = set(ep_ids[:target_adapt])

    for ep_id in range(n_episodes):
        if use_grounded_curriculum:
            assert grounded_state is not None
            assert cc_state is not None and d_sequence
            d = float(d_sequence[ep_id])
            base_task = softmax_pick_grounded_task(d, rng, k=4, temperature=7.0)
            mutated_task = apply_meaningful_structural_variation(base_task, d, rng)
            mutation_types = []
            task_level = int(GROUNDED_LEVEL_BY_TASK_ID.get(base_task.task_id, 0))
            difficulty_scalar = d
            difficulty_level = 0
            proxy_score = grounded_task_proxy_score(mutated_task)
            d_proxy = float(difficulty_proxy_task(mutated_task))
            feats = scenario_features(d)
            bidx = bucket_index(d)
            band = f"bucket_{bidx}"
            grounded_meta = {
                "grounded_curriculum": True,
                "continuous_difficulty": d,
                "difficulty_proxy": d_proxy,
                "difficulty_bucket_index": bidx,
                "scenario_features": feats,
                "grounded_level": task_level,
                "grounded_template_id": base_task.task_id,
                "training_band": band,
                "curriculum_active_level": grounded_state.active_level,
                "rule_proxy_score": float(proxy_score),
            }
        else:
            base_task = rng.choice(task_list)
            grounded_meta = {"grounded_curriculum": False}
            # Apply generator mutation (rule-based for dataset generation)
            mutated_task, is_solvable = generator.mutate(base_task)
            mutation_types = generator.last_mutation_types
            difficulty_scalar = generator.difficulty_scalar
            difficulty_level = generator.difficulty_level

        profile = supervisor.sample_profile(ep_id)
        sup_desc = SUPERVISOR_PROFILES[profile]["description"]

        aman_obs, dman_obs = env.reset(
            episode_id=ep_id,
            supervisor_profile=profile,
            mutated_task=mutated_task,
            randomize=not use_grounded_curriculum,
        )

        atfm_json = json.dumps(env._state.atfm_deadlines)

        # AMAN BID sample
        aman_s = _make_aman_sample(
            ep_id=ep_id,
            obs=aman_obs,
            atfm_json=atfm_json,
            dman_slots_json="[]",  # no DMAN info yet at bid round
            sup_desc=sup_desc,
            profile=profile,
            round_name="bid",
            difficulty_scalar=difficulty_scalar,
        )
        aman_s.update(grounded_meta)
        samples.append(aman_s)

        # DMAN BID sample
        dman_s = _make_dman_sample(
            ep_id=ep_id,
            obs=dman_obs,
            atfm_json=atfm_json,
            aman_slots_json="[]",  # no AMAN info yet at bid round
            sup_desc=sup_desc,
            profile=profile,
            round_name="bid",
            difficulty_scalar=difficulty_scalar,
        )
        dman_s.update(grounded_meta)
        samples.append(dman_s)

        # Generator sample
        if include_generator:
            gen_s = _make_generator_sample(
                ep_id=ep_id,
                task=base_task,
                profile=profile,
                difficulty_level=generator.difficulty_level,
                ema_score=generator.ema_score,
                difficulty_scalar=difficulty_scalar,
                difficulty_distribution=generator.difficulty_distribution,
            )
            gen_s.update(grounded_meta)
            if use_grounded_curriculum:
                gen_s["generator_from_grounded"] = True
            samples.append(gen_s)

        # Supervisor sample (uses a dummy merged plan for dataset; real plan used at inference)
        if include_supervisor:
            merged_json = (
                _build_solver_merged_plan_json(mutated_task)
                if use_grounded_curriculum
                else _build_reference_merged_plan_json(mutated_task)
            )
            sup_s = _make_supervisor_sample(
                ep_id=ep_id,
                task=mutated_task,
                profile=profile,
                sup_desc=sup_desc,
                difficulty_scalar=difficulty_scalar,
                merged_plan_json=merged_json,
            )
            sup_s.update(grounded_meta)
            samples.append(sup_s)

        if include_adapt and domain_tasks:
            from training.adapt_curriculum import build_dual_adapt_samples

            if use_grounded_curriculum:
                adapt_rows, trig = build_dual_adapt_samples(
                    ep_id,
                    mutated_grounded_task=mutated_task,
                    profile=profile,
                    d=float(difficulty_scalar),
                    rng=rng,
                    domain_tasks=domain_tasks,
                    conflict_proxy=conflict_proxy_from_solver(mutated_task),
                )
                for ar in adapt_rows:
                    ar.update(grounded_meta)
                    ar["adapt_bundle_triggers"] = trig
                    samples.append(ar)
            elif ep_id in adapt_episode_ids:
                from multi_agent.adapt import build_adapt_observation

                tid = rng.choice(list(domain_tasks.keys()))
                dtask = domain_tasks[tid]
                obs = build_adapt_observation(dtask, profile)
                samples.append(_make_adapt_sample(ep_id, obs, dtask, difficulty_scalar=difficulty_scalar))

        controller_score = _estimate_controller_score(mutated_task)
        generator.update(controller_score)
        if not use_grounded_curriculum:
            generator.record(base_task.task_id, mutation_types, controller_score)
        elif include_generator:
            generator.record(base_task.task_id, mutation_types if mutation_types else [], controller_score)

    return samples


# ── Sample builders ───────────────────────────────────────────────────────────

def _make_aman_sample(
    ep_id: int,
    obs,
    atfm_json: str,
    dman_slots_json: str,
    sup_desc: str,
    profile: SupervisorProfileName,
    round_name: str,
    difficulty_scalar: float,
) -> Dict[str, Any]:
    system = AMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}"
    user = obs.to_prompt_text()
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "task_id":            obs.task_id,
        "agent_role":         AgentRole.AMAN.value,
        "episode_id":         ep_id,
        "round":              round_name,
        "supervisor_profile": profile.value,
        "atfm_deadlines_json": atfm_json,
        "dman_slots_json":    dman_slots_json,
        "difficulty_scalar":  float(difficulty_scalar),
    }


def _make_dman_sample(
    ep_id: int,
    obs,
    atfm_json: str,
    aman_slots_json: str,
    sup_desc: str,
    profile: SupervisorProfileName,
    round_name: str,
    difficulty_scalar: float,
) -> Dict[str, Any]:
    system = DMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}"
    user = obs.to_prompt_text()
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "task_id":            obs.task_id,
        "agent_role":         AgentRole.DMAN.value,
        "episode_id":         ep_id,
        "round":              round_name,
        "supervisor_profile": profile.value,
        "atfm_deadlines_json": atfm_json,
        "aman_slots_json":    aman_slots_json,
        "difficulty_scalar":  float(difficulty_scalar),
    }


def _make_generator_sample(
    ep_id: int,
    task,
    profile: SupervisorProfileName,
    difficulty_level: int,
    ema_score: float,
    difficulty_scalar: float,
    difficulty_distribution: Dict[str, float],
) -> Dict[str, Any]:
    user_content = (
        f"Current agent performance (EMA): {ema_score:.2f}\n"
        f"Current scalar difficulty d: {difficulty_scalar:.3f}\n"
        f"Difficulty distribution Beta(alpha={difficulty_distribution.get('alpha', 1.0):.2f}, "
        f"beta={difficulty_distribution.get('beta', 1.0):.2f})\n"
        f"Target difficulty level: {difficulty_level}/6\n\n"
        f"Base task: {task.task_id} ({task.difficulty.value})\n"
        f"Flights: {len(task.flights)} | Runways: {len(task.runways)}\n"
        f"Airport: {task.airport}\n\n"
        f"Design mutations that will make AMAN and DMAN fail to coordinate "
        f"at difficulty level {difficulty_level}. Remember: solvable but hard."
    )
    return {
        "prompt": [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        "task_id":            task.task_id,
        "agent_role":         AgentRole.GENERATOR.value,
        "episode_id":         ep_id,
        "round":              "generate",
        "supervisor_profile": profile.value,
        "controller_scores":  ema_score,
        "difficulty_scalar":  float(difficulty_scalar),
    }


def _make_supervisor_sample(
    ep_id: int,
    task,
    profile: SupervisorProfileName,
    sup_desc: str,
    difficulty_scalar: float,
    merged_plan_json: Optional[str] = None,
) -> Dict[str, Any]:
    if merged_plan_json is None:
        merged_plan_json = _build_reference_merged_plan_json(task)
    system = SUPERVISOR_SYSTEM_TEMPLATE.format(preference=sup_desc)
    user_content = (
        f"Task: {task.task_id}\nAirport: {task.airport}\n"
        f"Flights: {len(task.flights)} | Runways: {len(task.runways)}\n\n"
        f"A merged AMAN+DMAN plan was submitted. Evaluate it against your preference."
    )
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        "task_id":            task.task_id,
        "agent_role":         AgentRole.SUPERVISOR.value,
        "episode_id":         ep_id,
        "round":              "evaluate",
        "supervisor_profile": profile.value,
        "merged_plan_json":   merged_plan_json,
        "difficulty_scalar":  float(difficulty_scalar),
    }


def _build_solver_merged_plan_json(task: TaskDefinition) -> str:
    slots = solve_grounded_rule_based(task)
    if not slots:
        return _build_reference_merged_plan_json(task)
    payload = [
        {
            "flight_id": str(s.flight_id),
            "runway": str(s.runway),
            "assigned_minute": int(s.assigned_minute),
            "hold_minutes": int(s.hold_minutes),
        }
        for s in slots
    ]
    return json.dumps(payload)


def _build_reference_merged_plan_json(task) -> str:
    """Build a deterministic full-plan baseline for supervisor training."""
    slots: List[Dict[str, Any]] = []
    for flight in task.flights:
        if not flight.allowed_runways:
            continue
        assigned_minute = max(
            int(flight.earliest_minute),
            min(int(flight.latest_minute), int(flight.scheduled_minute)),
        )
        hold_minutes = max(0, abs(assigned_minute - int(flight.scheduled_minute)))
        slots.append(
            {
                "flight_id": str(flight.flight_id),
                "runway": str(flight.allowed_runways[0]),
                "assigned_minute": int(assigned_minute),
                "hold_minutes": int(hold_minutes),
            }
        )
    return json.dumps(slots)


def _make_adapt_sample(
    ep_id: int,
    obs: ADAPTObservation,
    domain_task: TaskDefinition,
    difficulty_scalar: float,
    domain_source_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "prompt": [
            {"role": "system", "content": ADAPT_SYSTEM},
            {"role": "user", "content": obs.to_prompt_text()},
        ],
        "task_id": "domain_transfer",
        "agent_role": AgentRole.ADAPT.value,
        "episode_id": int(ep_id),
        "round": "adapt",
        "domain_task_json": domain_task.model_dump_json(),
        "supervisor_profile": obs.supervisor_profile_name.value,
        "difficulty_scalar": float(difficulty_scalar),
    }
    if domain_source_task_id is not None:
        row["domain_source_task_id"] = domain_source_task_id
    return row


def _estimate_controller_score(task: TaskDefinition) -> float:
    """Deterministic proxy controller performance for curriculum updates."""
    slots = []
    for flight in task.flights:
        if not flight.allowed_runways:
            continue
        assigned_minute = max(
            int(flight.earliest_minute),
            min(int(flight.latest_minute), int(flight.scheduled_minute)),
        )
        slots.append(
            SlotAssignment(
                flight_id=str(flight.flight_id),
                runway=str(flight.allowed_runways[0]),
                assigned_minute=int(assigned_minute),
                hold_minutes=max(0, abs(assigned_minute - int(flight.scheduled_minute))),
            )
        )
    try:
        outcome = simulate_plan(task, slots)
        return max(0.0, min(1.0, float(outcome.normalized_score)))
    except Exception:
        return 0.5


# ── Action parsers (completion → typed action) ────────────────────────────────

def _coerce_completion_text(completion: Any) -> str:
    """Normalise chat-style completions from TRL into plain text."""
    if completion is None:
        return ""
    if isinstance(completion, bytes):
        return completion.decode("utf-8", errors="ignore")
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        for key in ("content", "text", "completion", "generated_text"):
            if key in completion:
                return _coerce_completion_text(completion[key])
        try:
            return json.dumps(completion)
        except Exception:
            return str(completion)
    if isinstance(completion, list):
        parts = [_coerce_completion_text(item) for item in completion]
        return "\n".join(part for part in parts if part)
    return str(completion)


def _extract_json(text: Any) -> Optional[str]:
    """Extract first JSON object from an LLM completion.

    Handles the most common LLM output quirks:
      - markdown fences (```json, ```JSON, ```)
      - Python literals: True/False/None → true/false/null
      - single-quote dicts  → double-quote JSON (ast fallback)
    """
    text = _coerce_completion_text(text)
    # Strip all markdown code fences regardless of language tag or case
    text = re.sub(r"```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"```", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    raw = match.group(0)
    # Normalise Python literals so json.loads can parse them
    # Use word-boundary replacements to avoid mangling string values
    raw = re.sub(r"\bTrue\b",  "true",  raw)
    raw = re.sub(r"\bFalse\b", "false", raw)
    raw = re.sub(r"\bNone\b",  "null",  raw)
    return raw


def _loads_lenient(raw: str) -> Optional[dict]:
    """json.loads with ast.literal_eval fallback for single-quote dicts."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            import ast
            obj = ast.literal_eval(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _safe_slot(s: dict, op: str) -> Optional[SlotAssignment]:
    """Build a SlotAssignment tolerating wrong field types from LLM output."""
    try:
        return SlotAssignment(
            flight_id=str(s.get("flight_id", "")),
            runway=str(s.get("runway", "")),
            assigned_minute=int(float(s.get("assigned_minute", 0))),
            hold_minutes=int(float(s.get("hold_minutes", 0))),
        )
    except Exception:
        return None


def parse_aman_action(completion: Any) -> Optional[AMANAction]:
    raw = _extract_json(completion)
    if not raw:
        return None
    data = _loads_lenient(raw)
    if not isinstance(data, dict):
        return None
    try:
        # Per-slot try/except: one bad slot skips that slot, not the whole action
        slots = [s for s in (_safe_slot(x, "arrival") for x in data.get("arrival_slots", [])) if s]
        msgs = []
        for m in data.get("outgoing_messages", []):
            try:
                msgs.append(NegotiationMessage(
                    from_role=AgentRole.AMAN,
                    message_type=MessageType(m.get("message_type", "runway_claim")),
                    flight_id=str(m.get("flight_id", "")),
                    requested_minute=int(float(m.get("requested_minute", 0))),
                    runway_id=str(m.get("runway_id", "")),
                    priority=str(m.get("priority", "normal")),
                    reason=str(m.get("reason", "")),
                    is_emergency=bool(m.get("is_emergency", False)),
                ))
            except Exception:
                continue
        return AMANAction(
            arrival_slots=slots,
            rationale=str(data.get("rationale", "")),
            emergency_yields=list(data.get("emergency_yields", [])),
            outgoing_messages=msgs,
            commit=bool(data.get("commit", False)),
        )
    except Exception:
        return None


def parse_dman_action(completion: Any) -> Optional[DMANAction]:
    raw = _extract_json(completion)
    if not raw:
        return None
    data = _loads_lenient(raw)
    if not isinstance(data, dict):
        return None
    try:
        slots = [s for s in (_safe_slot(x, "departure") for x in data.get("departure_slots", [])) if s]
        msgs = []
        for m in data.get("outgoing_messages", []):
            try:
                msgs.append(NegotiationMessage(
                    from_role=AgentRole.DMAN,
                    message_type=MessageType(m.get("message_type", "runway_claim")),
                    flight_id=str(m.get("flight_id", "")),
                    requested_minute=int(float(m.get("requested_minute", 0))),
                    runway_id=str(m.get("runway_id", "")),
                    priority=str(m.get("priority", "normal")),
                    reason=str(m.get("reason", "")),
                    is_emergency=bool(m.get("is_emergency", False)),
                ))
            except Exception:
                continue
        return DMANAction(
            departure_slots=slots,
            rationale=str(data.get("rationale", "")),
            atfm_compliance=dict(data.get("atfm_compliance", {})),
            emergency_broadcasts=list(data.get("emergency_broadcasts", [])),
            outgoing_messages=msgs,
            commit=bool(data.get("commit", False)),
        )
    except Exception:
        return None


def parse_generator_action(completion: Any) -> Optional[GeneratorAction]:
    raw = _extract_json(completion)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        mutations = []
        for m in data.get("mutations", []):
            try:
                mutations.append(GeneratorMutation(
                    mutation_type=MutationType(m.get("mutation_type", "tighten_window")),
                    target_flight_id=m.get("target_flight_id"),
                    target_runway_id=m.get("target_runway_id"),
                    params=m.get("params", {}),
                    rationale=m.get("rationale", ""),
                ))
            except Exception:
                continue
        return GeneratorAction(
            mutations=mutations,
            strategy=data.get("strategy", ""),
        )
    except Exception:
        return None
