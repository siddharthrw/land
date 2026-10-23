# LUNAR PILOT — Build Plan

## 0. Product summary

A cinematic 3D rocket-launch-to-lunar-landing simulator in the browser, driven by a Python
physics/AI backend over WebSockets. The player watches an autonomous (or scripted) mission
run end to end — liftoff, staging, transfer, lunar descent, landing/crash — with live
telemetry, an event log, and a mission score.

**Honesty constraint carried through every phase:** the UI must always make clear which
parts of the mission are physically simulated, which are scripted/guided, and which are
controlled by a trained RL policy. We never claim more autonomy or fidelity than what is
actually implemented.

## 1. Mission phase state machine

Phases (happy path):

1. PRELAUNCH
2. LIFTOFF
3. ASCENT
4. FIRST_STAGE_SEPARATION
5. SECOND_STAGE_ASCENT
6. FAIRING_SEPARATION
7. ORBITAL_INSERTION
8. EARTH_ORBIT
9. TRANSLUNAR_TRANSFER
10. LUNAR_APPROACH
11. LUNAR_DESCENT
12. TOUCHDOWN
13. MISSION_COMPLETE

Failure states (can be entered from most of the above): LAUNCH_FAILURE, STAGING_FAILURE,
FUEL_EXHAUSTED, TRAJECTORY_FAILURE, CRASH, ABORTED.

Fidelity per phase (stated explicitly, and surfaced in the UI):

| Phase | Fidelity |
|---|---|
| PRELAUNCH → ORBITAL_INSERTION | Scripted/guided ascent profile with simplified single-axis thrust/gravity integration. Staging events are real state transitions, not just animation timers. |
| EARTH_ORBIT → LUNAR_APPROACH | Scripted transfer trajectory (precomputed/interpolated path), clearly labeled "scripted" in HUD. |
| LUNAR_DESCENT | Real physics (3-DOF: horizontal pos/vel, vertical pos/vel, tilt, fuel/mass) with a genuine feedback controller (PD baseline) or, optionally, a trained PPO policy. This is the one phase actually "played" by an AI. |
| TOUCHDOWN / outcome | Deterministic physical check against landing-safety thresholds (speed, tilt, position). |

## 2. Rocket model & staging

Modular Three.js groups, each independently transformable/detachable:
- First-stage booster (engines + tanks)
- Interstage adapter
- Second-stage engine section
- Payload fairing (two halves)
- Upper-stage / transfer vehicle
- Lunar descent module (the "active vehicle" during descent)

**Separation sequence (applies to first-stage, fairing, upper-stage):**
1. Backend confirms the staging condition (fuel/time/altitude threshold) in the mission
   state — this is the source of truth, not a frontend animation timer.
2. Backend emits a `STAGE_SEPARATION` event with: which stage, active-vehicle state
   post-separation, discarded-stage initial state (position/velocity/rotation).
3. Frontend receives the event, detaches the corresponding Three.js group from the active
   vehicle's parent, spawns it as an independent object with its own velocity/trail, and
   plays the visual separation impulse.
4. Score is awarded exactly once per event (idempotency key = event id from backend).
5. Discarded stages keep simulating simple ballistic motion client-side (cosmetic only —
   not fed back into mission scoring) so they drift away convincingly.

Never: detaching geometry without a backend event; awarding points on a timer; keeping a
detached mesh visually parented to the active vehicle.

## 3. Physics

Deterministic fixed-timestep simulation in the backend (e.g. 50 Hz), decoupled from
render framerate. The frontend interpolates between backend state snapshots for smooth
visuals even if WebSocket messages arrive at a lower/irregular rate.

Lunar descent equations of motion (3-DOF, theta measured from vertical):

```
ax = T * sin(theta) / m
ay = T * cos(theta) / m - g_moon
dm/dt = -T / (Isp * g0)
```

State: x, y, vx, vy, theta, fuel_mass, dry_mass.
Constraints enforced every step: fuel >= 0, mass >= dry_mass, throttle in [0, max],
tilt in [-tilt_limit, +tilt_limit], y >= lunar_surface_height(x) (no penetration).

Terminal conditions:
- TOUCHDOWN (success): |vy| and |vx| under safe thresholds, |theta| under safe threshold,
  within landing-site radius.
- CRASH: surface contact outside safe thresholds.
- FUEL_EXHAUSTED: fuel hits zero before touchdown.

Constants (documented assumptions, tunable in one config module):
g_moon = 1.62 m/s², Isp ≈ 311s (descent-engine-like), g0 = 9.81 m/s² (for the rocket
equation only, not local gravity), dry mass / propellant mass chosen for a believable
~2–3 minute descent window.

## 4. AI pilots

**Controller A — baseline guidance (always available, no training required):**
PD-style feedback controller: vertical channel drives throttle to track a descending
velocity target proportional to altitude (classic "velocity to be killed by surface"
guidance); horizontal channel drives tilt to null cross-range error and velocity. Hard
limits enforced on throttle and tilt. This is the default controller and the fallback if
no trained model is present.

**Controller B — PPO (optional, trained offline):**
- `backend/simulation/lunar_env.py`: Gymnasium `Env` wrapping the same physics used by the
  baseline (single source of truth — no separate "RL-only" physics).
  - Observation: altitude, horizontal error, vertical velocity, horizontal velocity, fuel
    fraction, tilt, angular velocity, target-relative position.
  - Action: continuous [throttle, tilt_command], bounded via `Box`.
  - Reward: shaped (progress toward safe landing, penalize high touchdown speed/tilt/fuel
    waste/oscillation) + large terminal bonus for safe touchdown, large penalty for
    crash/fuel-out. Randomized initial conditions (altitude, offset, velocity) per episode
    so the policy generalizes instead of memorizing one descent.
- `backend/agents/train.py`: PPO via Stable-Baselines3, small MLP policy, headless
  (no rendering), checkpointing to `models/`.
- `backend/agents/evaluate.py`: runs N episodes, reports success rate, avg touchdown
  speed, fuel used, landing error — used for the pilot-comparison view.
- If `models/` has no saved policy, the app transparently uses Controller A and says so in
  the UI. Training failure/absence must never be hidden or implied as "AI landed it."

## 5. Reward / scoring system

Two separate things, never conflated in the UI:
1. **RL training reward** — dense shaping signal used only to train PPO (not shown to the
   user as "the score").
2. **Mission event score** — discrete, event-sourced points shown in the HUD:

| Event | Points |
|---|---:|
| Successful liftoff | +50 |
| Stable ascent | +50 |
| First-stage separation | +100 |
| Second-stage ignition & stable operation | +50 |
| Fairing separation | +75 |
| Upper-stage separation | +75 |
| Orbital insertion | +150 |
| Transfer phase success | +100 |
| Lunar approach success | +100 |
| Controlled descent | up to +150 |
| Fuel efficiency | up to +100 |
| Landing near target | up to +100 |
| Safe touchdown | +300 |
| Engine shutdown | +50 |

Rules: each event scored once (idempotent by event id), evaluated against explicit
numeric conditions (never "a timer expired"), crash can never award the touchdown bonus,
duplicate backend messages must not double-score (dedupe on event id client- and
server-side).

## 6. Mission control UI

- Main viewport (Three.js canvas) + HUD overlay: phase, mission elapsed time, altitude,
  vertical/horizontal speed, throttle, fuel, attitude, score, active controller
  (baseline/PPO), sim speed, camera mode, and a "fidelity" badge per phase
  (simulated/scripted/learned).
- Event timeline (timestamped, only real backend-sourced events).
- Decision log (controller's stated reason + resulting score delta, generated from actual
  telemetry/controller output, not invented text).
- Controls: Launch / Pause / Resume / Restart / Autopilot toggle / manual override
  (throttle+tilt during descent only) / camera select / time warp / replay / compare
  pilots. No dead buttons — anything not implemented is hidden, not disabled-and-visible.

## 7. Camera system

LAUNCHPAD, ROCKET_FOLLOW, ORBITAL, MOON_APPROACH, LANDING, FREE, MISSION_OVERVIEW.
Camera is purely a rendering concern — switching cameras never touches simulation state.

## 8. Replay & comparison

Backend logs every tick (or every N ticks) during a mission: vehicle state, controller
action, fuel, events, score deltas, terminal outcome, to a session log file. Frontend can
replay a finished mission from this log (deterministic if seeded). Pilot comparison view
runs baseline vs. PPO over the same set of seeded descent scenarios (multiple episodes,
not a single cherry-picked run) and reports success rate, avg landing speed, fuel used,
distance-to-target, failure rate side by side.

## 9. Project structure

```
frontend/
  src/
    components/ (SpaceScene, Rocket, Stage, Earth, Moon, Exhaust,
                 MissionHUD, MissionTimeline, DecisionLog, ScorePanel)
    store/ (missionStore.ts — Zustand)
    services/ (missionSocket.ts — WebSocket client)
    App.tsx
backend/
  simulation/ (physics.py, mission.py, staging.py, rewards.py, lunar_env.py)
  agents/ (baseline.py, train.py, evaluate.py)
  api/ (server.py — FastAPI + WebSocket)
models/        # saved PPO checkpoints
logs/          # mission replay logs
plan.md
architecture.md
README.md
```

## 10. Priority order

**P0 (today's primary deliverable) — working end-to-end slice:**
- Scaffolding for both frontend and backend, runnable with documented commands.
- Full phase state machine with scripted ascent/transfer and a real staging sequence
  (first-stage separation minimum; fairing/upper-stage if time allows).
- Independently moving discarded stages.
- Real lunar-descent physics + baseline PD controller.
- Landing / crash detection and a mission result screen.
- HUD with live telemetry, event log, and event-sourced score.
- Restart works correctly (full state reset).

**P1 — AI integration:**
- Gymnasium env, PPO training script, model save/load, headless training, evaluation
  over multiple episodes, pilot comparison view wired to real evaluation data.

**P2 — cinematic polish:**
- Materials/detailing, exhaust & smoke particles, lighting/shadows, camera choreography,
  lunar dust on touchdown, Earth atmosphere/clouds/starfield, smooth camera transitions.

**P3 — stretch:**
- Fuller 3D dynamics, better orbital mechanics, terrain hazards, sensor noise/fault
  injection, more advanced guidance.

## 11. Acceptance checklist (must pass before calling P0 "done")

1. App starts with documented commands (frontend + backend).
2. Rocket launches from a scripted ignition sequence.
3. At least one genuine, backend-driven stage-separation event occurs.
4. Detached stage(s) persist and move independently after separation.
5. Active spacecraft state is continuous across separation (no teleport/velocity jump).
6. Fuel decreases only while engines are firing, consistent with throttle.
7. Physics tick rate is independent of rendering FPS.
8. Descent controller (baseline, and PPO if trained) produces real, varying control
   outputs, not a canned animation.
9. Mission reaches TOUCHDOWN or a failure state — never hangs.
10. Score reflects real events/outcomes, each counted exactly once.
11. Restart fully resets mission + score + stage state.
12. If no trained PPO model exists, the app still runs correctly on the baseline
    controller, and the UI says so.
13. No unhandled exceptions in frontend console or backend logs during a full mission run.
14. HUD visibly distinguishes scripted vs. simulated vs. learned phases.

Basic automated tests to include: physics integration (no NaN/negative fuel/mass,
surface non-penetration), staging transition correctness (idempotent scoring, valid
state handoff), reward/scoring unit tests (each event condition), landing-condition
classification (success vs. crash boundary cases).
