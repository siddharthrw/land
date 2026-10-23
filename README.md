# Lunar Pilot

A cinematic Earth-to-Moon rocket mission simulator with an autonomous lunar-descent
pilot and a live mission-control HUD. See [plan.md](plan.md) for the full design and
[architecture.md](architecture.md) for how the system fits together.

**Status: P0 complete and verified end-to-end.** Launch → ascent → first-stage
separation → fairing separation → orbital insertion → translunar transfer → lunar
approach → powered descent → touchdown, all driven by the backend state machine, with
live telemetry, an event-sourced score, and a decision log from the real PD guidance
controller. P1 (PPO training) and further P2 cinematic polish are scaffolded but not
the primary deliverable yet — see "What's real vs. scripted" below.

## Quick start

Two servers, run in separate terminals.

**Backend** (Python 3.12):
```bash
cd backend
pip install -r requirements.txt
uvicorn api.server:app --port 8000
```
This installs `gymnasium`/`stable-baselines3` too (needed for P1 training), but the
live app and its tests only require `fastapi`, `uvicorn`, `numpy`, and `pydantic`.

**Frontend** (Node 18+):
```bash
cd frontend
npm install
npm run dev
```
Open the printed `http://localhost:5173` URL. It connects to the backend at
`ws://localhost:8000/ws` by default (override with `VITE_MISSION_WS_URL`).

Click **Launch** and watch the mission run. Switch camera angles from the dropdown
(`LAUNCHPAD`, `ROCKET FOLLOW`, `ORBITAL`, `MOON APPROACH`, `LANDING`, `FREE`,
`MISSION OVERVIEW`). During `LUNAR_DESCENT` you can switch the pilot to `Manual
override` and fly it yourself with the throttle/tilt sliders.

## Running the tests

```bash
cd backend
python -m pytest tests/ -q
```
14 tests cover physics invariants (no negative fuel/mass, no surface penetration, tilt
limits), the full happy-path mission reaching `MISSION_COMPLETE`, idempotent staging
and scoring, pause/restart correctness, and landing classification.

## What's real vs. scripted (shown live in the HUD's fidelity badge)

- **Scripted**: ascent trajectory, Earth-orbit coast, translunar transfer, lunar
  approach. These follow a deterministic, time-compressed flight path — not a 6-DOF
  physics solve. Staging *events* fire off real backend state (fuel/time thresholds),
  not off a visual timer.
- **Simulated**: lunar descent. Real 3-DOF physics (`backend/simulation/physics.py`)
  integrated at a fixed 50 Hz timestep, independent of render framerate, with a genuine
  feedback guidance controller (`backend/agents/baseline.py`) computing throttle and
  tilt from live telemetry every tick.
- **Learned**: the PPO controller, only available once you've trained one (see below).
  The app runs correctly on the baseline controller alone if you never train one.

## Training a PPO pilot (optional, P1)

```bash
cd backend
pip install -r requirements.txt   # gymnasium + stable-baselines3 if not already installed
python -m agents.train --timesteps 200000
python -m agents.evaluate --episodes 100
```
Training writes `models/ppo_lunar_descent.zip`. Restart the backend afterward — it
checks for this file at startup and enables the "Trained PPO" option in the pilot
selector only if a model was found. `evaluate.py` runs baseline vs. PPO over many
seeded episodes (not a single anecdotal run) and prints success rate, average landing
speed, fuel used, and distance-to-target for both.

## Project layout

```
backend/
  simulation/   physics.py, mission.py, staging.py, rewards.py, lunar_env.py
  agents/       baseline.py (PD controller), train.py, evaluate.py
  api/          server.py (FastAPI + WebSocket, fixed-timestep sim loop)
  tests/        pytest suite
frontend/
  src/
    components/ SpaceScene, Earth, Moon, Rocket, Stage, Exhaust, CameraRig, MissionHUD,
                MissionTimeline, DecisionLog, ScorePanel, Controls
    store/      missionStore.ts (Zustand)
    services/   missionSocket.ts (WebSocket client)
    types/      mission.ts (shared state/event/command contract)
models/   trained PPO checkpoints (empty until you train one)
logs/     reserved for mission replay logs (P1/P2)
```

## Known limitations (honest accounting)

- Ascent/transfer geometry is stylized (world units, not real km) for a readable,
  cinematic view of the whole Earth-Moon system — not a navigation-grade trajectory.
- Replay-from-log and the pilot-comparison UI panel described in `plan.md` section 8 are
  not wired into the frontend yet; `agents/evaluate.py` produces the comparison data
  from the command line today.
- Cinematic polish (richer particle exhaust, ground dust on touchdown, shadows) is
  partial — the core mechanics and staging/scoring correctness were prioritized first,
  per the plan's P0-before-P2 ordering.
