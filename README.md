---
title: ATC Optimization OpenEnv
sdk: docker
app_port: 8000
license: mit
tags:
  - openenv
---

# ATC Optimization OpenEnv

A real-world OpenEnv benchmark for **air traffic disruption recovery**. The agent acts as a tactical ATC/flow controller and must build safe, complete runway slot plans under operational pressure (weather, inspections, emergency priorities, bank balancing).

## Judge Quick View

- Real-world domain: ATC disruption recovery (not a game)
- OpenEnv API: typed models + `/reset`, `/step`, `/state`
- Tasks: 3 deterministic graded tasks (`easy`, `medium`, `hard`)
- Graders: bounded `0.0-1.0`, deterministic composite score
- Baseline: root-level `inference.py`, structured `[START]/[STEP]/[END]` logs
- Infra: Dockerfile + HF Space deployment flow

## Requirement-to-Evidence Matrix

| Requirement | Evidence |
|---|---|
| Real-world utility | `tasks.py` models real ATC disruption scenarios with safety and fairness constraints |
| Full OpenEnv spec | `openenv.yaml`, typed Pydantic models in `models.py`, server endpoints in `server/app.py` |
| 3+ tasks with graders | `delhi_monsoon_recovery_easy`, `mumbai_bank_balance_medium`, `bengaluru_irrops_hard` in `tasks.py`; grading in `graders.py` |
| Meaningful reward shaping | Dense step-wise reward with partial progress in `engine.py` |
| Reproducible baseline | Root `inference.py` with deterministic fallback + strict stdout format |
| Docker + Space deployability | `Dockerfile`, `scripts/deploy_hf_space.py`, `scripts/validate-submission.sh` |
| Validation workflow | `python -m openenv.cli validate .`, `python scripts/run_graders.py`, `python inference.py` |

## Environment Design

### Action Space
`ATCOptimizationAction` in `models.py`

- `proposal`: list of `SlotAssignment`
- `rationale`: reasoning summary
- `commit`: finish episode flag

Each `SlotAssignment` includes:

- `flight_id`
- `runway`
- `assigned_minute`
- `hold_minutes`

### Observation Space
`ATCOptimizationObservation` in `models.py`

- task metadata and briefing
- flight and runway state
- current and best metrics
- diagnostics and recommendations
- grader feedback
- `steps_remaining`

### State Space
`ATCOptimizationState` exposed by `/state`

- includes `active_task_ids` so tasks are enumerable by validators/agents

## Tasks and Difficulty

Defined in `tasks.py`:

1. `delhi_monsoon_recovery_easy` (easy)
2. `mumbai_bank_balance_medium` (medium)
3. `bengaluru_irrops_hard` (hard)

All tasks are scored with deterministic and bounded outputs (`0.0-1.0`).

## Reward and Scoring

Operational score components in `engine.py`:

- completeness
- conflict-free ratio
- priority handling
- delay efficiency
- fairness
- fuel efficiency

Reward behavior:

- first valid submission receives full score
- next steps receive incremental improvement reward
- invalid/incomplete/conflicting plans are penalized

Official submission score is deterministic composite grading (`graders.py`).

## Baseline Inference (Submission Script)

Root file: `inference.py`

- uses OpenAI client interface
- reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- emits strict structured logs:
  - `[START]`
  - `[STEP]`
  - `[END]`
- deterministic fallback when model output is unavailable/invalid

### Baseline Scores (Deterministic Fallback)

- `delhi_monsoon_recovery_easy`: `0.98`
- `mumbai_bank_balance_medium`: `0.98`
- `bengaluru_irrops_hard`: `0.94`

## Repository Layout

- `models.py`: typed contracts
- `tasks.py`: scenario catalog
- `engine.py`: simulation + reward shaping
- `graders.py`: deterministic graders + optional LLM side-channel
- `planner.py`: deterministic baseline planner
- `server/atc_environment.py`: environment implementation
- `server/app.py`: FastAPI/OpenEnv entrypoint
- `inference.py`: required baseline script (root)
- `openenv.yaml`: environment metadata
- `Dockerfile`: container runtime
- `scripts/run_graders.py`: task grading check
- `scripts/ping_env.py`: deployment health/reset ping
- `scripts/validate-submission.sh`: judge-style validator
- `scripts/pre_submission_validate.sh`: convenience wrapper
- `scripts/deploy_hf_space.py`: HF API deployment helper

## Setup

```bash
pip install uv
uv sync
```

## Required Environment Variables

```bash
export API_BASE_URL="https://your-llm-endpoint/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-secret-token"
```

Note: use `python -m inference` (module) or `python inference.py` (script), not `python -m inference.py`.

## Local Runbook

Start server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run OpenEnv validation:

```bash
python -m openenv.cli validate .
```

Run all graders:

```bash
python scripts/run_graders.py
```

Run baseline inference:

```bash
python inference.py
```

Run tests:

```bash
uv run pytest -q
```

## Docker

Build:

```bash
docker build .
```

Run:

```bash
docker run --rm -p 8000:8000 atc-openenv
```

## Hugging Face Space Deployment

### Option A: Manual

1. Create HF Space with SDK = Docker
2. Push repository
3. Set secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
4. Ping deployment:

```bash
python scripts/ping_env.py https://<your-space>.hf.space
```

### Option B: HF API Helper

Token-only template is provided in `.env.hf_space.example`.

```bash
export HF_TOKEN="hf_xxx"
export HF_SPACE_ID="<owner>/<space-name>"
python scripts/deploy_hf_space.py --space-id "$HF_SPACE_ID" --repo-dir .
```

Then validate:

```bash
./scripts/validate-submission.sh "https://<owner>-<space-name>.hf.space" .
```

## Pre-Submission Checklist

Run all before final submission:

```bash
python -m openenv.cli validate .
uv run pytest -q
python scripts/run_graders.py
python inference.py
./scripts/validate-submission.sh https://<your-space>.hf.space .
```

Expected:

- OpenEnv validate: `[OK] : Ready for multi-mode deployment`
- Grader scores bounded `0.0-1.0`
- Inference logs strictly follow `[START]/[STEP]/[END]`
- Space responds to `/health` and `/reset`

## Notes for Judges

- Deterministic scoring is intentional for reproducibility and anti-gaming.
- Optional LLM signals are preserved as auxiliary analysis, not official score drivers.
- Release branch is hackathon-standard focused; Groq-only development artifacts are excluded.
