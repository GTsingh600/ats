---
title: ATC Optimization OpenEnv
sdk: docker
app_port: 8000
license: mit
tags:
  - openenv
---

# ATC Optimization OpenEnv

This repository implements a realistic OpenEnv benchmark for air traffic control disruption recovery. The agent acts like a tactical tower/flow controller who must re-sequence arrivals and departures after weather, runway inspection, or irregular-operations events. The benchmark ships with typed OpenEnv models, `/reset`, `/step`, and `/state` endpoints, a strict `openenv.yaml`, three graded tasks from easy to hard, a reproducible baseline `inference.py`, and Docker/Hugging Face Spaces deployment assets.

## Why this is a real-world task

This is not a toy routing problem. Each episode models the kind of tactical optimization controllers and airline operations centers actually face:

- runway assignment under reduced capacity
- wake-turbulence spacing constraints
- emergency or medical prioritization
- connection-bank protection
- fairness across airlines during disruption
- fuel burn and passenger delay tradeoffs

The agent must produce a full slot plan, not just rank flights. Plans are graded on safety, completeness, priority handling, delay efficiency, fairness, and fuel efficiency.

## OpenEnv compliance

The environment is spec-compliant for the current OpenEnv validator:

- `openenv.yaml` is in the repo root
- typed Pydantic models are defined in [models.py](models.py)
- FastAPI entrypoint is [server/app.py](server/app.py)
- environment logic is in [server/atc_environment.py](server/atc_environment.py)
- package metadata and `server` script are defined in [pyproject.toml](pyproject.toml)
- `uv.lock` is generated for reproducible builds

## Tasks

Three benchmark tasks are included in [tasks.py](tasks.py):

1. `delhi_monsoon_recovery_easy`
   Weather-reduced departure recovery with one medical flight.
2. `mumbai_bank_balance_medium`
   Mixed arrival/departure bank balancing with fairness pressure.
3. `bengaluru_irrops_hard`
   Irregular operations scenario with an emergency arrival and heavy congestion.

Each task is scored by:

- the simulator in [engine.py](/workspace/engine.py)
- a deterministic supervisor grader
- an optional LLM supervisor grader
- a composite grader in [graders.py](/workspace/graders.py)

All grader outputs are clamped to `0.0-1.0`.

## Action space

The action model is [ATCOptimizationAction](models.py):

- `proposal`: list of typed `SlotAssignment` items
- `rationale`: short explanation of the sequencing logic
- `commit`: whether to end the episode after the current plan

Each `SlotAssignment` contains:

- `flight_id`
- `runway`
- `assigned_minute`
- `hold_minutes`

This makes the task easy for both programmatic agents and LLM planners: propose a full schedule in structured JSON, send it once, get dense operational feedback back.

## Observation space

The observation model is [ATCOptimizationObservation](models.py). It includes:

- scenario identity and difficulty
- full flight list and runway list
- text briefing for prompt-based agents
- current metrics and best metrics
- diagnostics and recommendations
- grader feedback
- `steps_remaining`

State inspection is available through [ATCOptimizationState](models.py) and exposed by `/state`.

## Reward design

Reward is dense and gives partial progress signals:

- every submitted plan is simulated and graded
- step reward is the improvement over the previous best plan
- the first submitted plan receives its full score as reward
- incomplete plans, invalid assignments, runway conflicts, and priority misses sharply reduce score

The simulator score uses:

- schedule completeness
- conflict-free ratio
- priority handling
- delay efficiency
- fairness
- fuel efficiency

To avoid gaming the benchmark with tiny partial plans, the operational score is multiplied by completeness and additionally penalized when conflicts remain.

## Baseline inference

[inference.py](inference.py) is the required root-level inference script. It:

- uses the OpenAI Python client for LLM calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- emits strict `[START]`, `[STEP]`, and `[END]` logs
- runs all three tasks
- falls back to a deterministic planner from [planner.py](/workspace/planner.py) if model output fails

The deterministic planner is useful for reproducibility and CI smoke tests. When a model endpoint is configured, the LLM gets the task briefing and a heuristic seed plan and is asked to return strict JSON.

### Reproducible baseline scores

With no LLM credentials configured, `inference.py` falls back to the deterministic heuristic planner and produces these baseline composite scores:

- `delhi_monsoon_recovery_easy`: `0.9102`
- `mumbai_bank_balance_medium`: `0.7813`
- `bengaluru_irrops_hard`: `0.8179`

The script emits only the required `[START]`, `[STEP]`, and `[END]` stdout records, with rewards formatted to two decimal places for validator compatibility.

## File guide

- [models.py](models.py): typed OpenEnv action, observation, state, task, and grading models
- [tasks.py](tasks.py): the easy, medium, and hard benchmark scenarios
- [engine.py](engine.py): ATC simulator, safety checks, and reward shaping
- [graders.py](graders.py): deterministic and optional LLM supervisor graders
- [planner.py](planner.py): deterministic baseline planner
- [client.py](client.py): typed OpenEnv client
- [server/atc_environment.py](server/atc_environment.py): environment implementation
- [server/app.py](server/app.py): FastAPI/OpenEnv server entrypoint
- [inference.py](inference.py): submission-time baseline runner
- [Dockerfile](Dockerfile): container build for HF Spaces and validator docker builds
- [scripts/run_graders.py](scripts/run_graders.py): enumerate tasks and verify grader score ranges
- [scripts/ping_env.py](scripts/ping_env.py): ping a local or deployed environment and test `/reset`
- [scripts/run_local_checklist.ps1](scripts/run_local_checklist.ps1): one-command local validation runner for Windows/PowerShell
- [scripts/pre_submission_validate.sh](scripts/pre_submission_validate.sh): convenience wrapper for the checklist

## Setup

### Local Python setup

```bash
pip install uv
uv sync
```

### Required environment variables

Before running `inference.py`, define:

```bash
export API_BASE_URL="https://your-llm-endpoint/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-secret-token"
```

`OPENAI_API_KEY` is also accepted as a credential fallback when `HF_TOKEN` is not set.

For local testing without an LLM endpoint, the script still runs by falling back to the deterministic planner.

## Running the environment

### Start the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Validate the OpenEnv package

```bash
python -m openenv.cli validate .
```

### Run the baseline inference

```bash
python inference.py
```

### Run graders across all tasks

```bash
python scripts/run_graders.py
```

### Run the full local checklist on Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_checklist.ps1
```

## Hugging Face Spaces deployment

This repo is ready for a Docker Space:

1. Create a new Hugging Face Space with SDK = `Docker`.
2. Push this repository contents to the Space.
3. Add `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` as Space secrets if you want the LLM-backed grader and inference improvements enabled.
4. After deploy, verify:

```bash
python scripts/ping_env.py https://<your-space>.hf.space
```

The Space should return `200` on `/health` and respond to `/reset`.

## Docker

Build locally:

```bash
docker build .
```

Run locally:

```bash
docker run --rm -p 8000:8000 atc-openenv
```

## Pre-submission checklist

These commands cover the expected checks:

```bash
python -m openenv.cli validate .
python scripts/run_graders.py
python inference.py
docker build .
```

If you also have a deployed Space URL:

```bash
bash scripts/pre_submission_validate.sh https://<your-space>.hf.space .
```

## Work to be done before submission

The environment is code-complete and locally validated, but the following submission checks still need to be completed on deployment infrastructure:

- verify `docker build .` on a machine with Docker Engine running
- deploy the repository to a public Hugging Face Docker Space
- confirm the live Space responds with `200` on `/health` and `/reset`
- run `scripts/pre_submission_validate.sh` against the deployed Space URL
- record the final public repository URL and Space URL in the submission form

These are deployment and publication tasks rather than environment logic gaps.

## Concepts behind the code

The benchmark separates concerns deliberately:

- data contracts live in typed Pydantic models
- scenario content lives in the task catalog
- operations logic lives in the simulator
- reward shaping and grading stay explicit and inspectable
- the baseline planner is deterministic so scores can be reproduced
- the LLM layer is optional and isolated to inference and the optional supervisor grader

That structure makes it easy to extend the benchmark with new airports, new disruption types, or richer planners without breaking the OpenEnv contract.
