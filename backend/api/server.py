"""FastAPI + WebSocket server. One mission session per WebSocket connection.
The simulation loop runs at a fixed timestep regardless of how often the
browser renders or how fast messages are consumed.

Run from backend/:  uvicorn api.server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agents.evaluate import evaluate
from simulation.mission import MissionEvent, MissionStateMachine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lunar_pilot")

TICK_HZ = 50
TICK_DT = 1.0 / TICK_HZ

app = FastAPI(title="Lunar Pilot Mission Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_lunar_descent.zip")


def try_load_ppo_model():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        logger.info("No trained PPO model found at %s — PPO controller unavailable.", DEFAULT_MODEL_PATH)
        return None
    try:
        from stable_baselines3 import PPO

        model = PPO.load(DEFAULT_MODEL_PATH)
        logger.info("Loaded trained PPO model from %s", DEFAULT_MODEL_PATH)
        return model
    except Exception as exc:  # pragma: no cover - defensive, keeps app usable
        logger.warning("Failed to load PPO model (%s) — falling back to baseline.", exc)
        return None


_ppo_model = try_load_ppo_model()


def event_to_dict(event: MissionEvent) -> dict:
    return dataclasses.asdict(event)


@app.get("/health")
async def health():
    return {"status": "ok", "ppoAvailable": _ppo_model is not None}


_compare_cache: dict[int, dict] = {}


@app.get("/api/compare")
async def compare_pilots(episodes: int = 60):
    """Baseline vs. PPO over many seeded descent episodes (never a single
    anecdotal run — see agents/evaluate.py). Runs off the event loop thread
    so it doesn't stall any mission WebSocket currently ticking, and caches
    by episode count since a run is deterministic (seeds 0..N-1)."""
    episodes = max(5, min(500, episodes))
    if episodes in _compare_cache:
        return _compare_cache[episodes]

    def run():
        baseline_metrics = evaluate("baseline", episodes)
        ppo_metrics = evaluate("ppo", episodes, model=_ppo_model) if _ppo_model is not None else None
        return {"baseline": baseline_metrics, "ppo": ppo_metrics, "episodes": episodes}

    result = await asyncio.to_thread(run)
    _compare_cache[episodes] = result
    return result


@app.websocket("/ws")
async def mission_socket(websocket: WebSocket):
    await websocket.accept()
    mission = MissionStateMachine()
    mission.set_ppo_model(_ppo_model)
    logger.info("Client connected — new mission session started.")

    async def receiver():
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                    mission.handle_command(msg.get("type"), msg.get("payload"))
                except (json.JSONDecodeError, AttributeError):
                    logger.warning("Ignoring malformed command: %r", raw)
        except WebSocketDisconnect:
            pass

    async def sim_loop():
        try:
            while True:
                # Run `sim_speed` physics steps per broadcast instead of
                # shortening the sleep — keeps the websocket message rate
                # constant (no network/render overload at high speed) while
                # sim_time advances faster than wall-clock time.
                for _ in range(mission.sim_speed):
                    mission.step(TICK_DT)
                events = mission.drain_events()
                payload = {
                    "state": mission.snapshot(),
                    "events": [event_to_dict(e) for e in events],
                    "ppoAvailable": _ppo_model is not None,
                }
                await websocket.send_text(json.dumps(payload))
                await asyncio.sleep(TICK_DT)
        except WebSocketDisconnect:
            pass

    receiver_task = asyncio.create_task(receiver())
    sim_task = asyncio.create_task(sim_loop())
    try:
        await asyncio.wait([receiver_task, sim_task], return_when=asyncio.FIRST_COMPLETED)
    finally:
        receiver_task.cancel()
        sim_task.cancel()
        logger.info("Client disconnected — mission session ended.")
