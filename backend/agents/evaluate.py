"""Evaluate and compare the baseline controller against a trained PPO policy
over many seeded descent episodes — never a single anecdotal run.

Run from backend/:  python -m agents.evaluate --episodes 100
"""
from __future__ import annotations

import argparse
import json
import os

from agents import baseline
from simulation import physics
from simulation.lunar_env import DT, MAX_STEPS, observation_from_state

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def _run_episode(seed: int, controller: str, model=None):
    import random

    rng = random.Random(seed)
    alt = rng.uniform(1100.0, 2600.0)
    offset = rng.uniform(-850.0, 850.0)
    state = physics.initial_descent_state(seed_offset_x=offset, seed_altitude=alt)

    for _ in range(MAX_STEPS):
        if controller == "ppo" and model is not None:
            import numpy as np

            obs = np.array(observation_from_state(state), dtype="float32")
            action, _ = model.predict(obs, deterministic=True)
            throttle_cmd = float((action[0] + 1.0) / 2.0)
            tilt_cmd = float(action[1]) * physics.TILT_LIMIT_RAD
        else:
            throttle_cmd, tilt_cmd, _ = baseline.compute_action(state)

        state = physics.step_descent(state, throttle_cmd, tilt_cmd, DT)
        outcome = physics.classify_outcome(state)
        if outcome != "FLYING":
            return outcome, state
    return "TIMEOUT", state


def evaluate(controller: str, episodes: int, model=None) -> dict:
    results = []
    for seed in range(episodes):
        outcome, state = _run_episode(seed, controller, model)
        results.append((outcome, state))

    successes = [r for r in results if r[0] == "TOUCHDOWN"]
    n = len(results)
    return {
        "controller": controller,
        "episodes": n,
        "successRate": len(successes) / n if n else 0.0,
        "avgLandingSpeed": (
            sum(abs(s.vy) + abs(s.vx) for _, s in successes) / len(successes) if successes else None
        ),
        "avgFuelUsedKg": (
            sum(physics.FUEL_MASS_INITIAL - s.fuel_mass for _, s in successes) / len(successes)
            if successes
            else None
        ),
        "avgDistanceToTarget": (
            sum(physics.distance_to_target(s) for _, s in successes) / len(successes) if successes else None
        ),
        "failureRate": 1.0 - (len(successes) / n if n else 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model", type=str, default="ppo_lunar_descent")
    args = parser.parse_args()

    baseline_metrics = evaluate("baseline", args.episodes)
    report = {"baseline": baseline_metrics}

    model_path = os.path.join(MODELS_DIR, f"{args.model}.zip")
    if os.path.exists(model_path):
        from stable_baselines3 import PPO

        model = PPO.load(model_path)
        report["ppo"] = evaluate("ppo", args.episodes, model=model)
    else:
        report["ppo"] = None
        print(f"No trained model found at {model_path} — skipping PPO comparison.")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
