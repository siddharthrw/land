"""Headless PPO training for the lunar descent environment.

Run from backend/:  python -m agents.train --timesteps 200000

Requires gymnasium + stable-baselines3 (see requirements.txt). Writes the
checkpoint to models/ppo_lunar_descent.zip. The live app works without ever
running this script — it just falls back to the baseline controller.
"""
from __future__ import annotations

import argparse
import os

from simulation.lunar_env import LunarDescentEnv

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--out", type=str, default="ppo_lunar_descent")
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to an existing checkpoint (.zip) to continue training from, instead of a fresh policy.",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="Entropy bonus coefficient — higher encourages more exploration (e.g. trying full throttle) "
        "instead of settling into an overly cautious policy.",
    )
    args = parser.parse_args()

    if LunarDescentEnv is None:
        raise SystemExit("gymnasium is not installed — run `pip install -r requirements.txt` first.")

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    env = make_vec_env(lambda: LunarDescentEnv(randomize=True), n_envs=8)
    # Reward-only normalization (never obs — observation_from_state() is
    # already hand-scaled to roughly [-5, 5], and the live app / evaluate.py
    # feed the policy raw observations with no normalization wrapper, so
    # normalizing obs here would silently break inference).
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if args.resume_from:
        model = PPO.load(args.resume_from, env=env)
        print(f"Resumed from {args.resume_from}")
    else:
        # A full descent can run ~3000 steps. PPO's default gamma=0.99 decays
        # a reward 3000 steps away to effectively zero (0.99**3000 ~= 1e-13),
        # so the terminal landing/crash signal would never reach early
        # decisions. gamma=0.999 keeps meaningful credit across the whole
        # descent. n_steps is raised so each rollout buffer captures a
        # larger slice of a descent rather than many tiny, disconnected
        # fragments.
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=2048,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=args.ent_coef,
        )
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume_from)

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, args.out)
    model.save(out_path)
    print(f"Saved trained policy to {out_path}.zip")


if __name__ == "__main__":
    main()
