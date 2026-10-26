"""DAgger (Dataset Aggregation): fixes plain imitation's blind spot.

Behavior cloning (agents/behavior_clone.py) only ever shows the network
baseline's "everything went right" trajectories — it never sees what to do
once it's already drifted slightly off that path, because the expert never
drifts. Over a long flight, tiny per-step imitation errors compound, and the
student ends up in situations its training data never covered at all.

DAgger fixes this directly: let the CURRENT student fly on its own (with its
own mistakes), and at every state it actually visits along that imperfect
flight, ask the expert "what would you have done, right here" — then add
that correction to the training set and retrain. Repeat a few rounds, each
one targeting whatever blind spot is left. The training data grows to cover
the student's actual mistakes, not just the expert's clean path.

Run from backend/:  python -m agents.dagger --init ppo_lunar_descent_fixed_imitation_only --iterations 5
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from agents import baseline
from agents.behavior_clone import generate_demos
from simulation import physics
from simulation.lunar_env import LunarDescentEnv

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def collect_student_rollout(model, n_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Lets the CURRENT policy fly itself (not baseline), and at every
    state it actually visits, records what baseline would have done there
    instead — these are the correction labels. Executing the student's own
    action (not the expert's) is the whole point: it's what puts the
    student in its own mistake-states instead of only ever seeing the
    expert's clean ones."""
    env = LunarDescentEnv(randomize=True)
    obs_list: list[list[float]] = []
    act_list: list[list[float]] = []
    t0 = time.time()

    for ep in range(n_episodes):
        if ep % 50 == 0 and ep > 0:
            rate = ep / (time.time() - t0)
            print(f"    rollout episode {ep}/{n_episodes}  ({rate:.1f} ep/s)", flush=True)
        obs, _ = env.reset(seed=None)
        done = False
        while not done:
            state = env.state
            exp_throttle, exp_rcs, _ = baseline.compute_action(state)
            expert_action = np.clip(
                [exp_throttle * 2.0 - 1.0, exp_rcs], -1.0, 1.0
            )
            obs_list.append(obs.tolist())
            act_list.append(expert_action.tolist())

            student_action, _ = model.predict(np.array(obs, dtype="float32"), deterministic=True)
            obs, _, terminated, truncated, _ = env.step(student_action)
            done = terminated or truncated

    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def policy_bc(model, obs: np.ndarray, actions: np.ndarray, epochs: int, batch_size: int = 256):
    import torch
    import torch.nn.functional as F

    policy = model.policy
    obs_t = torch.as_tensor(obs, device=policy.device)
    act_t = torch.as_tensor(actions, device=policy.device)
    n = len(obs_t)
    optimizer = policy.optimizer

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            obs_b, act_b = obs_t[idx], act_t[idx]
            features = policy.extract_features(obs_b, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            mean_actions = policy.action_net(latent_pi)
            loss = F.mse_loss(mean_actions, act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        print(f"    epoch {epoch + 1}/{epochs}  policy_mse={total_loss / batches:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, required=True, help="Checkpoint name (no .zip) to start from.")
    parser.add_argument("--initial-demo-episodes", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--episodes-per-iteration", type=int, default=300)
    parser.add_argument("--epochs-per-iteration", type=int, default=6)
    parser.add_argument("--max-dataset-size", type=int, default=3_000_000)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--out", type=str, default="ppo_lunar_descent_dagger")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from agents.evaluate import evaluate

    env = LunarDescentEnv(randomize=True)
    init_path = os.path.join(MODELS_DIR, f"{args.init}.zip")
    model = PPO.load(init_path, env=None)
    print(f"Loaded starting policy from {init_path}", flush=True)

    print(f"Building initial dataset from {args.initial_demo_episodes} expert episodes...", flush=True)
    all_obs, all_act, _ = generate_demos(args.initial_demo_episodes)
    print(f"Initial dataset: {len(all_obs)} pairs.", flush=True)

    baseline_metrics = evaluate("baseline", args.eval_episodes)
    print(f"Baseline reference: {baseline_metrics['successRate']:.1%} success", flush=True)
    start_metrics = evaluate("ppo", args.eval_episodes, model=model)
    print(f"Starting PPO success rate: {start_metrics['successRate']:.1%}", flush=True)

    for it in range(args.iterations):
        print(f"\n=== DAgger iteration {it + 1}/{args.iterations} ===", flush=True)
        print(f"  Rolling out the CURRENT student for {args.episodes_per_iteration} episodes, "
              f"labeling its own visited states with expert corrections...", flush=True)
        new_obs, new_act = collect_student_rollout(model, args.episodes_per_iteration)
        print(f"  Collected {len(new_obs)} correction pairs from the student's own mistakes.", flush=True)

        all_obs = np.concatenate([all_obs, new_obs])
        all_act = np.concatenate([all_act, new_act])
        if len(all_obs) > args.max_dataset_size:
            keep_idx = np.random.choice(len(all_obs), args.max_dataset_size, replace=False)
            all_obs, all_act = all_obs[keep_idx], all_act[keep_idx]
        print(f"  Aggregated dataset now {len(all_obs)} pairs. Retraining...", flush=True)

        policy_bc(model, all_obs, all_act, epochs=args.epochs_per_iteration)

        os.makedirs(MODELS_DIR, exist_ok=True)
        iter_path = os.path.join(MODELS_DIR, f"{args.out}_iter{it + 1}")
        model.save(iter_path)

        metrics = evaluate("ppo", args.eval_episodes, model=model)
        print(f"  Iteration {it + 1} success rate: {metrics['successRate']:.1%} "
              f"(started at {start_metrics['successRate']:.1%}, baseline is {baseline_metrics['successRate']:.1%})", flush=True)

    final_path = os.path.join(MODELS_DIR, args.out)
    model.save(final_path)
    print(f"\nSaved final DAgger policy to {final_path}.zip", flush=True)


if __name__ == "__main__":
    main()
