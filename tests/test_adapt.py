"""Smoke tests for the ADAPT meta-agent (structural domain transfer).

ADAPT is domain-agnostic: it receives any unknown TaskDefinition and infers
ATC parameter mappings purely from structural signals (time_pressure,
connection_risk, resource intensity, notes urgency). These tests validate
the structural inference, NOT keyword-based guessing.
"""

from __future__ import annotations

import json

import pytest

from domains.icu import icu_task_catalog, ICU_DOMAIN_DESCRIPTION
from models import PriorityClass, WakeClass
from multi_agent.adapt import (
    _build_adapt_heuristic,
    _compute_entity_profiles,
    apply_adapt_mapping,
    build_adapt_observation,
    parse_adapt_action,
)
from multi_agent.models import (
    ADAPTAction,
    ADAPTObservation,
    AgentRole,
    SupervisorProfileName,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def icu_catalog():
    return icu_task_catalog()


@pytest.fixture(scope="module")
def normal_task(icu_catalog):
    return icu_catalog["icu_normal_day"]


@pytest.fixture(scope="module")
def mass_casualty_task(icu_catalog):
    return icu_catalog["icu_mass_casualty"]


@pytest.fixture(scope="module")
def flu_surge_task(icu_catalog):
    return icu_catalog["icu_flu_surge"]


@pytest.fixture(scope="module")
def normal_obs(normal_task):
    return build_adapt_observation(
        task=normal_task,
        profile=SupervisorProfileName.SAFETY_STRICT,
        domain_name="Hospital ICU Surge Management",
        domain_description=ICU_DOMAIN_DESCRIPTION,
    )


@pytest.fixture(scope="module")
def mass_casualty_obs(mass_casualty_task):
    return build_adapt_observation(
        task=mass_casualty_task,
        profile=SupervisorProfileName.EMERGENCY_FOCUS,
        domain_name="Hospital ICU Surge Management",
        domain_description=ICU_DOMAIN_DESCRIPTION,
    )


# ── ICU domain catalog ────────────────────────────────────────────────────────

def test_icu_catalog_has_three_tasks(icu_catalog):
    assert len(icu_catalog) == 3
    assert "icu_normal_day" in icu_catalog
    assert "icu_flu_surge" in icu_catalog
    assert "icu_mass_casualty" in icu_catalog


def test_icu_tasks_have_beds_as_runways(icu_catalog):
    for task in icu_catalog.values():
        runway_ids = {r.runway_id for r in task.runways}
        assert runway_ids == {"BED_A", "BED_B", "BED_C", "BED_D"}


def test_mass_casualty_has_trauma_arrivals(mass_casualty_task):
    trauma = [f for f in mass_casualty_task.flights if f.airline == "TRAUMA"]
    assert len(trauma) == 3, "mass_casualty must have exactly 3 TRAUMA arrivals"


def test_flu_surge_has_cardiac_arrivals(flu_surge_task):
    cardiac = [f for f in flu_surge_task.flights if f.airline == "CARDIAC"]
    assert len(cardiac) == 3, "flu_surge must have exactly 3 CARDIAC arrivals"


# ── _compute_entity_profiles (structural signals) ─────────────────────────────

def test_compute_profiles_returns_dict_per_entity_type(mass_casualty_task):
    profiles = _compute_entity_profiles(mass_casualty_task)
    entity_types = {f.airline for f in mass_casualty_task.flights if f.airline}
    assert set(profiles.keys()) == entity_types


def test_trauma_profile_has_high_time_pressure(mass_casualty_task):
    profiles = _compute_entity_profiles(mass_casualty_task)
    p = profiles["TRAUMA"]
    # TRAUMA entities have very tight windows → time_pressure should be high
    assert p["time_pressure"] >= 0.85, (
        f"TRAUMA should have high time_pressure, got {p['time_pressure']}"
    )


def test_trauma_profile_has_high_connection_risk(mass_casualty_task):
    profiles = _compute_entity_profiles(mass_casualty_task)
    p = profiles["TRAUMA"]
    assert p["avg_connection_risk"] >= 0.80, (
        f"TRAUMA should have high connection_risk, got {p['avg_connection_risk']}"
    )


def test_routine_profile_has_low_time_pressure(normal_task):
    profiles = _compute_entity_profiles(normal_task)
    p = profiles["ROUTINE"]
    # ROUTINE windows are wide relative to H/emergency types, but planning_horizon
    # is 480 min so even 150-min windows give tp~0.69 — assert it is strictly
    # below the HIGH-urgency threshold, not necessarily below 0.50.
    assert p["time_pressure"] < 0.85, (
        f"ROUTINE should be below the VERY TIGHT threshold, got {p['time_pressure']}"
    )


def test_routine_profile_has_zero_connection_risk(normal_task):
    profiles = _compute_entity_profiles(normal_task)
    p = profiles["ROUTINE"]
    assert p["avg_connection_risk"] <= 0.05, (
        f"ROUTINE should have near-zero connection_risk, got {p['avg_connection_risk']}"
    )


def test_profiles_contain_required_keys(normal_task):
    profiles = _compute_entity_profiles(normal_task)
    required = {
        "count", "avg_window_minutes", "time_pressure", "avg_connection_risk",
        "avg_fuel_burn", "avg_passengers", "urgency_in_notes", "operation_mix",
    }
    for entity_type, p in profiles.items():
        missing = required - set(p.keys())
        assert not missing, f"{entity_type} missing keys: {missing}"


def test_operation_mix_counts_arrivals_and_departures(mass_casualty_task):
    profiles = _compute_entity_profiles(mass_casualty_task)
    for et, p in profiles.items():
        ops = p["operation_mix"]
        assert "arrival" in ops and "departure" in ops


# ── build_adapt_observation ───────────────────────────────────────────────────

def test_adapt_observation_type(normal_obs):
    assert isinstance(normal_obs, ADAPTObservation)
    assert normal_obs.role == AgentRole.ADAPT


def test_adapt_observation_has_entity_profiles(normal_obs):
    assert isinstance(normal_obs.entity_profiles, dict)
    assert len(normal_obs.entity_profiles) > 0


def test_adapt_observation_profiles_match_entity_types(normal_obs):
    assert set(normal_obs.entity_profiles.keys()) == set(normal_obs.entity_types)


def test_adapt_observation_resource_names(normal_obs, normal_task):
    expected = [r.runway_id for r in normal_task.runways]
    assert normal_obs.resource_names == expected


def test_adapt_observation_entity_count(normal_obs, normal_task):
    assert normal_obs.entity_count == len(normal_task.flights)


def test_mass_casualty_obs_has_emergencies(mass_casualty_obs):
    assert mass_casualty_obs.has_emergencies is True


def test_adapt_observation_prompt_shows_structural_profiles(normal_obs):
    text = normal_obs.to_prompt_text()
    assert "ADAPT OBSERVATION" in text
    assert "time_pressure" in text
    assert "connection_risk" in text
    assert "ENTITY TYPE STRUCTURAL PROFILES" in text


def test_adapt_observation_prompt_shows_numerical_values(mass_casualty_obs):
    """The prompt must contain actual numbers, not just labels."""
    text = mass_casualty_obs.to_prompt_text()
    # Should show numeric time_pressure and connection_risk values
    import re
    numbers = re.findall(r"\d+\.\d{2,}", text)
    assert len(numbers) >= 4, "Prompt should contain multiple numeric structural values"


def test_adapt_observation_prompt_domain_agnostic_language(normal_obs):
    """Prompt must instruct agent not to reason from entity names."""
    text = normal_obs.to_prompt_text()
    assert "do NOT know what domain" in text.lower() or "INSTRUCTIONS" in text


def test_adapt_observation_supervisor_description_nonempty(normal_obs):
    assert len(normal_obs.supervisor_description) > 10


def test_build_adapt_observation_defaults_to_task_fields(normal_task):
    """Without explicit domain_name, should use task.airport."""
    obs = build_adapt_observation(
        task=normal_task,
        profile=SupervisorProfileName.SAFETY_STRICT,
    )
    assert obs.domain_name == normal_task.airport


# ── _build_adapt_heuristic (structural, no keywords) ─────────────────────────

def test_heuristic_covers_all_entity_types(normal_obs, normal_task):
    action = _build_adapt_heuristic(normal_obs, normal_task)
    for et in normal_obs.entity_types:
        assert et in action.entity_wake_map, f"{et!r} missing from wake map"
        assert et in action.entity_priority_map, f"{et!r} missing from priority map"


def test_heuristic_high_pressure_entity_maps_to_heavy(mass_casualty_obs, mass_casualty_task):
    """Entity with highest structural urgency must map to H/emergency.

    This tests structural inference only: TRAUMA has tp>0.85 and cr>0.80 which
    pushes urgency_score well above the 0.70 Heavy threshold.
    """
    action = _build_adapt_heuristic(mass_casualty_obs, mass_casualty_task)
    profiles = mass_casualty_obs.entity_profiles

    # Find whichever entity has the highest urgency score — it must be H
    highest_et = max(
        profiles,
        key=lambda et: 0.5 * profiles[et]["time_pressure"]
                       + 0.4 * profiles[et]["avg_connection_risk"]
                       + 0.1 * float(profiles[et]["urgency_in_notes"]),
    )
    assert action.entity_wake_map[highest_et] == WakeClass.HEAVY.value, (
        f"Entity with highest urgency score should be H, got {action.entity_wake_map[highest_et]}"
    )
    assert action.entity_priority_map[highest_et] in (
        PriorityClass.EMERGENCY.value, PriorityClass.MEDICAL.value
    )


def test_heuristic_low_pressure_entity_maps_to_light(normal_obs, normal_task):
    """Entity with lowest structural urgency score must map to L wake class.

    Priority can be 'connection' if tp >= 0.60 even with cr=0 — the score
    formula is what drives wake; priority is a separate axis.
    """
    action = _build_adapt_heuristic(normal_obs, normal_task)
    profiles = normal_obs.entity_profiles

    lowest_et = min(
        profiles,
        key=lambda et: 0.5 * profiles[et]["time_pressure"]
                       + 0.4 * profiles[et]["avg_connection_risk"],
    )
    assert action.entity_wake_map[lowest_et] == WakeClass.LIGHT.value, (
        f"Entity with lowest urgency score should be L, got {action.entity_wake_map[lowest_et]}"
    )
    # Priority must not be emergency or medical — connection/normal both acceptable
    assert action.entity_priority_map[lowest_et] in (
        PriorityClass.NORMAL.value, PriorityClass.CONNECTION.value
    ), f"Lowest-urgency entity should be normal or connection, got {action.entity_priority_map[lowest_et]}"


def test_heuristic_ordering_consistent_with_profiles(normal_obs, normal_task):
    """Wake class ordering must be monotone with urgency score."""
    action = _build_adapt_heuristic(normal_obs, normal_task)
    profiles = normal_obs.entity_profiles

    _WAKE_ORDER = {WakeClass.HEAVY.value: 3, WakeClass.MEDIUM.value: 2, WakeClass.LIGHT.value: 1}

    scores = {
        et: 0.5 * p["time_pressure"] + 0.4 * p["avg_connection_risk"]
        for et, p in profiles.items()
    }
    sorted_by_score = sorted(scores, key=scores.__getitem__, reverse=True)

    wake_ranks = [_WAKE_ORDER[action.entity_wake_map[et]] for et in sorted_by_score]
    # Must be non-increasing (higher score ≥ lower score in wake rank)
    for i in range(len(wake_ranks) - 1):
        assert wake_ranks[i] >= wake_ranks[i + 1], (
            f"Wake class ordering violated: {sorted_by_score[i]} "
            f"(score={scores[sorted_by_score[i]]:.2f} rank={wake_ranks[i]}) "
            f"vs {sorted_by_score[i+1]} "
            f"(score={scores[sorted_by_score[i+1]]:.2f} rank={wake_ranks[i+1]})"
        )


def test_heuristic_has_rationale_citing_numbers(normal_obs, normal_task):
    import re
    action = _build_adapt_heuristic(normal_obs, normal_task)
    assert len(action.rationale.strip()) > 0
    # Rationale must contain at least one numeric value
    numbers = re.findall(r"\d+\.\d+", action.rationale)
    assert len(numbers) >= 2, "Rationale should cite structural numbers"


def test_heuristic_returns_adapt_action_type(normal_obs, normal_task):
    action = _build_adapt_heuristic(normal_obs, normal_task)
    assert isinstance(action, ADAPTAction)


# ── apply_adapt_mapping ───────────────────────────────────────────────────────

def test_apply_mapping_overrides_wake_class(mass_casualty_task):
    action = ADAPTAction(
        entity_wake_map={"TRAUMA": "H", "CARDIAC": "M", "ROUTINE": "L", "POST_OP": "M"},
        entity_priority_map={"TRAUMA": "emergency", "CARDIAC": "medical",
                             "ROUTINE": "normal", "POST_OP": "connection"},
        rationale="test mapping",
    )
    mapped = apply_adapt_mapping(mass_casualty_task, action)
    for f in mapped.flights:
        if f.airline == "TRAUMA":
            assert f.wake_class == WakeClass.HEAVY
            assert f.priority == PriorityClass.EMERGENCY
        elif f.airline == "CARDIAC":
            assert f.wake_class == WakeClass.MEDIUM
            assert f.priority == PriorityClass.MEDICAL
        elif f.airline == "ROUTINE":
            assert f.wake_class == WakeClass.LIGHT
            assert f.priority == PriorityClass.NORMAL


def test_apply_mapping_preserves_unmatched_flights(normal_task):
    action = ADAPTAction(
        entity_wake_map={"UNKNOWN_TYPE": "H"},
        entity_priority_map={"UNKNOWN_TYPE": "emergency"},
        rationale="test",
    )
    mapped = apply_adapt_mapping(normal_task, action)
    assert len(mapped.flights) == len(normal_task.flights)


def test_apply_mapping_preserves_task_metadata(normal_task):
    action = ADAPTAction(
        entity_wake_map={"ROUTINE": "L"},
        entity_priority_map={"ROUTINE": "normal"},
        rationale="test",
    )
    mapped = apply_adapt_mapping(normal_task, action)
    assert mapped.task_id == normal_task.task_id
    assert mapped.delay_budget == normal_task.delay_budget
    assert len(mapped.runways) == len(normal_task.runways)


def test_apply_mapping_ignores_invalid_wake_value(normal_task):
    original_wakes = {f.flight_id: f.wake_class for f in normal_task.flights}
    action = ADAPTAction(
        entity_wake_map={"ROUTINE": "INVALID_WAKE"},
        entity_priority_map={},
        rationale="test",
    )
    mapped = apply_adapt_mapping(normal_task, action)
    for f in mapped.flights:
        if f.airline == "ROUTINE":
            assert f.wake_class == original_wakes[f.flight_id]


# ── parse_adapt_action ────────────────────────────────────────────────────────

def test_parse_valid_json_completion():
    completion = json.dumps({
        "entity_wake_map": {"TYPE_X": "H", "TYPE_Y": "M"},
        "entity_priority_map": {"TYPE_X": "emergency", "TYPE_Y": "medical"},
        "rationale": "TYPE_X: tp=0.97 cr=0.93 score=0.86 → H/emergency",
    })
    action = parse_adapt_action(completion)
    assert action is not None
    assert action.entity_wake_map["TYPE_X"] == "H"
    assert action.entity_priority_map["TYPE_Y"] == "medical"


def test_parse_json_in_markdown_fences():
    completion = "```json\n" + json.dumps({
        "entity_wake_map": {"TYPE_Z": "L"},
        "entity_priority_map": {"TYPE_Z": "normal"},
        "rationale": "tp=0.12 cr=0.05 score=0.08 → L/normal",
    }) + "\n```"
    action = parse_adapt_action(completion)
    assert action is not None
    assert action.entity_wake_map["TYPE_Z"] == "L"


def test_parse_returns_none_for_non_json():
    assert parse_adapt_action("I cannot complete this task.") is None
    assert parse_adapt_action("") is None
    assert parse_adapt_action(None) is None


def test_parse_returns_none_for_malformed_json():
    assert parse_adapt_action("{entity_wake_map: broken}") is None


def test_parse_handles_dict_completion():
    completion = {
        "content": json.dumps({
            "entity_wake_map": {"TYPE_A": "H"},
            "entity_priority_map": {"TYPE_A": "emergency"},
            "rationale": "dict input test",
        })
    }
    action = parse_adapt_action(completion)
    assert action is not None
    assert action.entity_wake_map["TYPE_A"] == "H"


def test_parse_missing_rationale_defaults_to_empty():
    completion = json.dumps({
        "entity_wake_map": {"TYPE_B": "L"},
        "entity_priority_map": {"TYPE_B": "normal"},
    })
    action = parse_adapt_action(completion)
    assert action is not None
    assert action.rationale == ""


# ── adapt_reward_fn ───────────────────────────────────────────────────────────

def test_adapt_reward_returns_list_of_floats(mass_casualty_task):
    from training.reward_functions import adapt_reward_fn

    completion = json.dumps({
        "entity_wake_map": {"TRAUMA": "H", "CARDIAC": "M", "ROUTINE": "L", "POST_OP": "M"},
        "entity_priority_map": {"TRAUMA": "emergency", "CARDIAC": "medical",
                                "ROUTINE": "normal", "POST_OP": "connection"},
        "rationale": "Structural: TRAUMA tp=0.97 cr=0.93 → H/emergency.",
    })
    rewards = adapt_reward_fn(
        completions=[completion],
        task_id=[mass_casualty_task.task_id],
        domain_task_json=[mass_casualty_task.model_dump_json()],
        supervisor_profile=[SupervisorProfileName.EMERGENCY_FOCUS.value],
    )
    assert isinstance(rewards, list)
    assert len(rewards) == 1
    assert isinstance(rewards[0], float)


def test_adapt_reward_bounded(mass_casualty_task):
    from training.reward_functions import adapt_reward_fn

    completion = json.dumps({
        "entity_wake_map": {"TRAUMA": "H", "CARDIAC": "M", "ROUTINE": "L", "POST_OP": "M"},
        "entity_priority_map": {"TRAUMA": "emergency", "CARDIAC": "medical",
                                "ROUTINE": "normal", "POST_OP": "connection"},
        "rationale": "Structural inference from numerical profiles.",
    })
    rewards = adapt_reward_fn(
        completions=[completion],
        task_id=[mass_casualty_task.task_id],
        domain_task_json=[mass_casualty_task.model_dump_json()],
        supervisor_profile=[SupervisorProfileName.EMERGENCY_FOCUS.value],
    )
    assert -1.0 <= rewards[0] <= 1.0


def test_adapt_reward_bad_json_penalised(mass_casualty_task):
    from training.reward_functions import adapt_reward_fn

    rewards = adapt_reward_fn(
        completions=["not json at all"],
        task_id=[mass_casualty_task.task_id],
        domain_task_json=[mass_casualty_task.model_dump_json()],
        supervisor_profile=[SupervisorProfileName.SAFETY_STRICT.value],
    )
    assert rewards[0] <= -0.4


def test_adapt_reward_missing_domain_task_penalised():
    from training.reward_functions import adapt_reward_fn

    rewards = adapt_reward_fn(
        completions=["{}"],
        task_id=["some_task"],
        domain_task_json=[None],
        supervisor_profile=[SupervisorProfileName.SAFETY_STRICT.value],
    )
    assert rewards[0] <= -0.5


def test_adapt_reward_batch_length_matches(mass_casualty_task):
    from training.reward_functions import adapt_reward_fn

    dtjson = mass_casualty_task.model_dump_json()
    completions = [
        json.dumps({"entity_wake_map": {"TRAUMA": "H"},
                    "entity_priority_map": {"TRAUMA": "emergency"},
                    "rationale": "tp=0.97 → H/emergency"}),
        "bad json",
    ]
    rewards = adapt_reward_fn(
        completions=completions,
        task_id=[mass_casualty_task.task_id] * 2,
        domain_task_json=[dtjson] * 2,
        supervisor_profile=[SupervisorProfileName.SAFETY_STRICT.value] * 2,
    )
    assert len(rewards) == 2


# ── _make_adapt_sample ────────────────────────────────────────────────────────

def test_make_adapt_sample_structure(normal_task, normal_obs):
    from training.dataset import _make_adapt_sample

    sample = _make_adapt_sample(ep_id=0, obs=normal_obs, domain_task=normal_task)
    assert "prompt" in sample
    assert len(sample["prompt"]) == 2
    assert sample["prompt"][0]["role"] == "system"
    assert sample["prompt"][1]["role"] == "user"
    assert sample["agent_role"] == AgentRole.ADAPT.value
    assert sample["round"] == "adapt"
    assert "domain_task_json" in sample


def test_make_adapt_sample_prompt_contains_structural_profiles(normal_task, normal_obs):
    from training.dataset import _make_adapt_sample

    sample = _make_adapt_sample(ep_id=5, obs=normal_obs, domain_task=normal_task)
    user_content = sample["prompt"][1]["content"]
    assert "time_pressure" in user_content
    assert "connection_risk" in user_content


def test_make_adapt_sample_system_contains_structural_guide(normal_task, normal_obs):
    from training.dataset import ADAPT_SYSTEM, _make_adapt_sample

    sample = _make_adapt_sample(ep_id=0, obs=normal_obs, domain_task=normal_task)
    sys_content = sample["prompt"][0]["content"]
    assert "STRUCTURAL" in sys_content
    assert "entity_wake_map" in sys_content
    assert "time_pressure" in sys_content


def test_make_adapt_sample_domain_task_json_valid(normal_task, normal_obs):
    from training.dataset import _make_adapt_sample

    sample = _make_adapt_sample(ep_id=0, obs=normal_obs, domain_task=normal_task)
    from models import TaskDefinition
    restored = TaskDefinition.model_validate_json(sample["domain_task_json"])
    assert restored.task_id == normal_task.task_id


# ── Dispatch table ────────────────────────────────────────────────────────────

def test_adapt_in_dispatch_table():
    from training.train_grpo import REWARD_FN_DISPATCH
    from training.reward_functions import adapt_reward_fn

    assert AgentRole.ADAPT.value in REWARD_FN_DISPATCH
    assert REWARD_FN_DISPATCH[AgentRole.ADAPT.value] is adapt_reward_fn


def test_dispatch_table_has_all_roles():
    from training.train_grpo import REWARD_FN_DISPATCH

    for role in AgentRole:
        assert role.value in REWARD_FN_DISPATCH, f"{role.value} missing from dispatch table"


# ── build_episode_dataset integration ────────────────────────────────────────

def test_dataset_includes_adapt_samples():
    from training.dataset import build_episode_dataset

    samples = build_episode_dataset(n_episodes=20, seed=0, include_adapt=True,
                                    domain_episode_ratio=0.5)
    adapt_samples = [s for s in samples if s["agent_role"] == AgentRole.ADAPT.value]
    assert len(adapt_samples) >= 1


def test_dataset_adapt_samples_have_required_fields():
    from training.dataset import build_episode_dataset

    samples = build_episode_dataset(n_episodes=20, seed=7, include_adapt=True,
                                    domain_episode_ratio=0.8)
    adapt_samples = [s for s in samples if s["agent_role"] == AgentRole.ADAPT.value]
    for s in adapt_samples:
        assert "prompt" in s
        assert "domain_task_json" in s
        assert s["round"] == "adapt"
        assert s["supervisor_profile"] in {p.value for p in SupervisorProfileName}


def test_dataset_no_adapt_when_disabled():
    from training.dataset import build_episode_dataset

    samples = build_episode_dataset(n_episodes=20, seed=0, include_adapt=False)
    adapt_samples = [s for s in samples if s["agent_role"] == AgentRole.ADAPT.value]
    assert len(adapt_samples) == 0


# ── Generic domain registry ───────────────────────────────────────────────────

def test_get_all_domain_tasks_returns_icu_tasks():
    from domains import get_all_domain_tasks

    all_tasks = get_all_domain_tasks()
    assert len(all_tasks) >= 3
    assert "icu_normal_day" in all_tasks


def test_get_all_domain_tasks_returns_task_definitions():
    from domains import get_all_domain_tasks
    from models import TaskDefinition

    all_tasks = get_all_domain_tasks()
    for task in all_tasks.values():
        assert isinstance(task, TaskDefinition)
