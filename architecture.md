# LUNAR PILOT — Architecture

## 1. High-level shape

Two processes, one source of truth:

```
┌─────────────────────────────┐        WebSocket         ┌───────────────────────────────┐
│          BACKEND             │  (mission state stream)  │           FRONTEND              │
│  Python / FastAPI            │ ───────────────────────► │  React + TS + Vite               │
│                               │                           │  Three.js via @react-three/fiber│
│  - Mission state machine     │ ◄─────────────────────── │                                   │
│  - Deterministic physics     │   (commands: launch,     │  - Pure renderer of backend state│
│    (fixed timestep)          │    pause, restart,       │  - Zustand store mirrors state   │
│  - Staging logic             │    manual override)      │  - HUD / timeline / decision log │
│  - Scoring / event log       │                           │  - Camera system (view-only)     │
│  - Baseline PD controller    │                           │                                   │
│  - PPO policy (optional)     │                           │                                   │
└───────────────┬───────────────┘                           └───────────────────────────────┘
                │
                ▼
      models/ (trained PPO checkpoints)
      logs/   (mission replay logs)
```

**The core design rule:** the backend is the single source of truth for everything that
matters to the mission (phase, physics, fuel, score, events). The frontend never computes
or decides mission-state facts — it only renders the state it is told about and sends
user intents (launch, pause, camera change, manual stick input) back. This is what makes
"visual events tied to real simulation state" actually true instead of aspirational.

## 2. Backend internals

```
backend/
  simulation/
    physics.py     — integrators: ascent (simplified), lunar descent (3-DOF), shared
                      constants/config, numerical safety (no NaN/negative fuel/mass,
                      surface non-penetration clamp)
    mission.py      — the MissionStateMachine: phase transitions, entry/exit conditions,
                      emits MissionEvent objects (id, type, timestamp, payload)
    staging.py      — staging-condition checks + the detach/discard state transforms
                      (gives each discarded stage its own position/velocity/rotation)
    rewards.py      — (a) event-sourced mission score table + idempotent award logic,
                      (b) RL shaping reward used only inside lunar_env.py
    lunar_env.py    — Gymnasium Env wrapping physics.py's descent integrator;
                      reset()/step()/observation_space/action_space
  agents/
    baseline.py     — PD/guidance controller, pure function: observation -> (throttle, tilt)
    train.py        — headless PPO training via Stable-Baselines3, writes to models/
    evaluate.py     — runs N seeded episodes per controller, returns comparison metrics
  api/
    server.py       — FastAPI app; WebSocket endpoint runs the fixed-timestep sim loop,
                      serializes MissionState to JSON each tick, handles incoming
                      commands (launch/pause/resume/restart/set_controller/manual_input)
```

**Simulation loop (inside `server.py`, per connected mission session):**

```
loop at fixed dt (e.g. 1/50s):
    if not paused:
        controller_output = active_controller(current_observation)   # baseline or PPO
        mission.step(dt, controller_output)   # advances phase, physics, staging, scoring
        new_events = mission.drain_events()   # events generated this tick, each with a uid
    broadcast({ state: mission.snapshot(), events: new_events })
    sleep until next tick boundary (so sim speed is decoupled from wall-clock jitter)
```

This loop's `dt` never depends on how fast the browser renders — render-rate independence
is enforced structurally, not just by convention, because the backend doesn't know or
care if a frontend is even connected.

## 3. Frontend internals

```
frontend/src/
  services/missionSocket.ts
      — owns the WebSocket connection; on message, pushes { state, events } into the
        Zustand store; exposes send() for commands (launch/pause/restart/camera/manual)
  store/missionStore.ts
      — Zustand store: latest backend snapshot + a local interpolation buffer (keeps the
        last 2 snapshots to interpolate vehicle position smoothly between ticks so render
        FPS can exceed/differ from backend tick rate without stutter or false precision)
      — event log (deduped by event uid) and derived score for the HUD
  components/
      SpaceScene.tsx    — top-level R3F <Canvas>, lighting, camera rig, starfield
      Earth.tsx / Moon.tsx — procedural sphere + materials, rotation
      Rocket.tsx        — composes Stage.tsx instances (booster/second-stage/fairing/
                          lander) positioned from store state; on a STAGE_SEPARATION
                          event, re-parents the relevant Stage into an independent
                          "debris" list in the scene rather than deleting/animating blindly
      Stage.tsx         — one reusable stage mesh/group (booster, fairing half, etc.)
      Exhaust.tsx       — particle system driven by current throttle value from the store
      MissionHUD.tsx / MissionTimeline.tsx / DecisionLog.tsx / ScorePanel.tsx
      — pure display components reading directly from missionStore
  App.tsx — wires socket connection + mounts scene/HUD
```

**Why Zustand sits between the socket and the components:** the WebSocket fires often
(50 Hz-ish); React components re-render from the store's selectors, so only the parts of
the UI that actually depend on changed fields re-render, instead of the whole tree
re-rendering on every packet.

## 4. Data contracts (the thing both sides agree on)

`MissionState` (backend → frontend, every tick):
```
{
  phase: PhaseEnum,
  simTime: number,
  vehicle: { position, velocity, attitude, fuelFraction, mass, throttle },
  stages: { boosterActive, fairingActive, upperStageActive, ... },
  debris: [{ id, position, velocity, rotation }],   // discarded stages, cosmetic
  activeController: "baseline" | "ppo" | "manual",
  score: number,
  camera: { mode },
  fidelity: { [phase]: "simulated" | "scripted" | "learned" }
}
```

`MissionEvent` (backend → frontend, emitted once, deduped by `id`):
```
{ id, type, timestamp, phase, scoreDelta, message }
```

Commands (frontend → backend):
```
{ type: "launch" | "pause" | "resume" | "restart" | "set_camera" | "set_controller"
       | "manual_input", payload }
```

Keeping this contract explicit (and mirrored as TypeScript types + Python dataclasses/
pydantic models) is what prevents the frontend from silently drifting out of sync with
what the backend actually simulates.

## 5. How a mission actually flows through the system (walkthrough)

1. User clicks **Launch** → frontend sends `{type:"launch"}` → backend transitions
   PRELAUNCH → LIFTOFF, starts the sim loop.
2. Each tick, backend advances the scripted ascent profile, streams `MissionState`.
   Frontend interpolates vehicle position/orientation and updates exhaust intensity from
   `throttle`.
3. Backend detects the first-stage staging condition → transitions to
   FIRST_STAGE_SEPARATION → runs `staging.py` to compute discarded-booster state →
   emits a `STAGE_SEPARATION` event + updated `debris` list → awards score once.
4. Frontend receives the event, detaches the booster `Stage` from the active rocket group,
   adds it to the debris render list using the state the backend provided (no re-deriving
   physics on the frontend), plays the separation VFX.
5. ... ASCENT → ORBITAL_INSERTION → EARTH_ORBIT → TRANSLUNAR_TRANSFER (scripted path) →
   LUNAR_APPROACH, each a real phase transition with its own event + score.
6. LUNAR_DESCENT: backend switches to the real 3-DOF physics integrator; each tick the
   active controller (baseline PD, or PPO if selected and a model is loaded) computes
   `(throttle, tilt)` from the current observation; physics integrates; decision log
   entries are generated from the controller's actual inputs/outputs (never invented).
7. TOUCHDOWN or CRASH/FUEL_EXHAUSTED: backend classifies the outcome against explicit
   numeric thresholds, emits final event(s), locks the score, transitions to
   MISSION_COMPLETE (or the relevant failure state).
8. Full tick-by-tick log has been written to `logs/` throughout; **Replay** reloads that
   log and drives the same frontend renderer from recorded states instead of a live
   socket. **Compare pilots** calls `agents/evaluate.py` offline (baseline vs. PPO across
   many seeded episodes) and renders the resulting metrics table — never a single
   anecdotal run.

## 6. Why this architecture satisfies the hard requirements

- **Frame-rate-independent physics:** enforced by having physics live entirely in the
  backend's fixed-dt loop; the frontend only interpolates, never integrates.
- **No duplicate/fake scoring:** every score change originates from a backend
  `MissionEvent` with a unique id; both the backend scorer and the frontend event log
  dedupe on that id.
- **Stages never "fake" detach:** separation is a backend state transition first; the
  frontend reacts to the event payload rather than guessing when to play an animation.
- **Honest AI labeling:** `fidelity` is part of the state contract, not a UI afterthought,
  so the HUD can always truthfully show "scripted" vs "simulated" vs "learned" per phase.
- **Works without a trained model:** `activeController` defaults to `"baseline"`; PPO is
  only selected if `models/` contains a loadable checkpoint, checked at startup.
