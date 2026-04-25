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

from models import OperationType, SlotAssignment, TaskDefinition
from tasks import task_catalog, ordered_tasks
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
) -> List[Dict[str, Any]]:
    """Build full multi-agent training dataset.

    Returns list of training samples, one per agent turn per episode.
    Each episode has: 1 AMAN bid + 1 DMAN bid + optionally 1 negotiation round.
    If include_generator: also 1 generator turn per episode.
    If include_supervisor: also 1 supervisor turn per episode.
    """
    import random
    rng = random.Random(seed)
    catalog = task_catalog()
    task_list = list(ordered_tasks())
    supervisor = SupervisorAgent()
    env = MultiAgentATCEnvironment(seed=seed)
    generator = ChallengeGenerator(seed=seed)

    samples: List[Dict[str, Any]] = []

    for ep_id in range(n_episodes):
        # ── ADAPT domain sample (stochastic) ────────────────────────────────
        if include_adapt and rng.random() < domain_episode_ratio:
            from domains import get_all_domain_tasks
            from multi_agent.adapt import build_adapt_observation

            domain_tasks = get_all_domain_tasks()
            if domain_tasks:
                tid = rng.choice(list(domain_tasks.keys()))
                dtask = domain_tasks[tid]
                profile = supervisor.sample_profile(ep_id)
                obs = build_adapt_observation(dtask, profile)
                samples.append(_make_adapt_sample(ep_id, obs, dtask))
                continue

        base_task = rng.choice(task_list)
        profile = supervisor.sample_profile(ep_id)
        sup_desc = SUPERVISOR_PROFILES[profile]["description"]

        # Apply generator mutation (rule-based for dataset generation)
        mutated_task, is_solvable = generator.mutate(base_task)

        aman_obs, dman_obs = env.reset(
            episode_id=ep_id,
            supervisor_profile=profile,
            mutated_task=mutated_task,
        )

        atfm_json = json.dumps(env._state.atfm_deadlines)

        # AMAN BID sample
        samples.append(_make_aman_sample(
            ep_id=ep_id,
            obs=aman_obs,
            atfm_json=atfm_json,
            dman_slots_json="[]",  # no DMAN info yet at bid round
            sup_desc=sup_desc,
            profile=profile,
            round_name="bid",
        ))

        # DMAN BID sample
        samples.append(_make_dman_sample(
            ep_id=ep_id,
            obs=dman_obs,
            atfm_json=atfm_json,
            aman_slots_json="[]",  # no AMAN info yet at bid round
            sup_desc=sup_desc,
            profile=profile,
            round_name="bid",
        ))

        # Generator sample
        if include_generator:
            samples.append(_make_generator_sample(
                ep_id=ep_id,
                task=base_task,
                profile=profile,
                difficulty_level=generator.difficulty_level,
                ema_score=generator.ema_score,
            ))

        # Supervisor sample (uses a dummy merged plan for dataset; real plan used at inference)
        if include_supervisor:
            samples.append(_make_supervisor_sample(
                ep_id=ep_id,
                task=mutated_task,
                profile=profile,
                sup_desc=sup_desc,
            ))

        # Simulate a mid-score episode to update generator curriculum
        generator.update(rng.uniform(0.25, 0.75))

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
    }


def _make_dman_sample(
    ep_id: int,
    obs,
    atfm_json: str,
    aman_slots_json: str,
    sup_desc: str,
    profile: SupervisorProfileName,
    round_name: str,
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
    }


def _make_generator_sample(
    ep_id: int,
    task,
    profile: SupervisorProfileName,
    difficulty_level: int,
    ema_score: float,
) -> Dict[str, Any]:
    user_content = (
        f"Current agent performance (EMA): {ema_score:.2f}\n"
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
    }


def _make_supervisor_sample(
    ep_id: int,
    task,
    profile: SupervisorProfileName,
    sup_desc: str,
) -> Dict[str, Any]:
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
    }


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


def _make_adapt_sample(ep_id: int, obs: ADAPTObservation, domain_task: TaskDefinition) -> Dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": ADAPT_SYSTEM},
            {"role": "user", "content": obs.to_prompt_text()},
        ],
        "task_id": "domain_transfer",
        "agent_role": AgentRole.ADAPT.value,
        "round": "adapt",
        "domain_task_json": domain_task.model_dump_json(),
        "supervisor_profile": obs.supervisor_profile_name.value,
    }


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
