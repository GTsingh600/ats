# Frontend Plan (Post-Submission)

## Goal

Ship an interactive demo UI that lets judges and users:

- run ATC episodes quickly,
- compare baseline vs trained behavior,
- inspect per-role outputs and rewards,
- view curriculum and checkpoint progression.

## User Stories

1. As a judge, I can run one episode in <60s and see final score + key diagnostics.
2. As a reviewer, I can compare base vs trained on the same task side-by-side.
3. As a researcher, I can inspect AMAN/DMAN decisions per round and conflict resolution.
4. As an engineer, I can open historical checkpoints and replay reward/parse trends.

## UX Surfaces

1. **Quick Run Panel**
   - Task selector
   - Model selector (base / trained / checkpoint)
   - Start button + progress state
2. **Round Timeline**
   - BID, NEGOTIATE, FINAL cards
   - AMAN and DMAN JSON outputs with validation badges
3. **Scoreboard**
   - Composite, AMAN, DMAN, conflicts, emergency handling
   - Baseline vs trained delta badges
4. **Training Analytics**
   - Reward curves (all roles + composite)
   - Parse-rate curves
   - ADAPT/SUP dynamics
   - Checkpoint tail-mean progression

## Technical Stack

- **Frontend:** React + Vite + TypeScript + ECharts/Recharts
- **Backend/API:** existing FastAPI (`server.app`)
- **Data:** JSON artifacts from `outputs/*` and `checkpoint_artifacts/*`
- **Deployment:** Hugging Face Space (Docker) with static build served by FastAPI or Nginx

## API Additions

1. `GET /demo/tasks` - list tasks and metadata
2. `POST /demo/run_episode` - trigger episode run with chosen model
3. `GET /demo/artifacts` - list available run/checkpoint artifacts
4. `GET /demo/artifacts/{run}/{file}` - stream JSON/PNG artifacts

## Milestones

1. **M1 (1 day):** Quick Run + Scoreboard
2. **M2 (1 day):** Round Timeline with per-role outputs
3. **M3 (1 day):** Analytics pages from exported plots + JSON
4. **M4 (0.5 day):** polishing, tooltips, error handling, judge-mode script

## Non-Goals for Round 2 deadline

- Full multi-user auth
- Persistent DB
- Heavy real-time websockets

