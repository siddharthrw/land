"""Gymnasium environment wrapping the same descent physics used by the
baseline controller and the live mission (simulation/physics.py) — there is
only one physics implementation, so a trained policy behaves identically in
training and in the live mission.
"""
from __future__ import annotations

import math
import random

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - gymnasium is optional for running the live app
    gym = None
    spaces = None

from simulation import physics, rewards
from simulation.physics import DescentState

DT = 1.0 / 20.0
# Baseline deliberately descends gently (never faster than 18 m/s), so a
# full descent from the top of the randomized altitude range (2600 m) can
# take ~175-200s even flown well. A verified physics sweep (see
# conversation history / README) confirmed episodes that looked
# "unrecoverable" at a lower cap were actually just cut off before a real
# outcome happened -- not real crashes. 4500 steps (225s) gives real margin.
MAX_STEPS = 4500


OBS_CLIP = 5.0


def observation_from_state(state: DescentState) -> list[float]:
    altitude = max(0.0, state.y)
    fuel_fraction = state.fuel_mass / physics.FUEL_MASS_INITIAL
    rcs_fuel_fraction = (
        state.rcs_fuel_mass / physics.RCS_FUEL_MASS_INITIAL if physics.RCS_FUEL_MASS_INITIAL > 0 else 0.0
    )
    raw = [
        altitude / physics.INITIAL_ALTITUDE,
        state.x / physics.INITIAL_HORIZONTAL_OFFSET,
        state.vy / 20.0,
        state.vx / 20.0,
        fuel_fraction,
        rcs_fuel_fraction,
        state.rcs_cmd,  # last applied RCS command, already in [-1, 1]
        state.x / physics.LANDING_RADIUS,
    ]
    return [max(-OBS_CLIP, min(OBS_CLIP, v)) for v in raw]


if gym is not None:

    class LunarDescentEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, randomize: bool = True):
            super().__init__()
            self.randomize = randomize
            self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(8,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.state: DescentState | None = None
            self.steps = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            rng = random.Random(seed)
            if self.randomize:
                # Matches the actual physically-recoverable envelope, found
                # by sweeping baseline across a wide grid of starting
                # positions/altitudes rather than guessed: below ~1000m
                # altitude the safe sideways range narrows (not enough time
                # left to correct a big drift against the ~28deg tilt
                # limit), and nothing lands past about +-1000m offset at any
                # altitude -- that's a hard physical wall, not a skill gap.
                # Training on a range that includes genuinely unrecoverable
                # starts would just teach it to imitate failure.
                alt = rng.uniform(1100.0, 2600.0)
                offset = rng.uniform(-850.0, 850.0)
            else:
                alt = physics.INITIAL_ALTITUDE
                offset = physics.INITIAL_HORIZONTAL_OFFSET
            self.state = physics.initial_descent_state(seed_offset_x=offset, seed_altitude=alt)
            self.steps = 0
            return np.array(observation_from_state(self.state), dtype=np.float32), {}

        def step(self, action):
            throttle_cmd = float((action[0] + 1.0) / 2.0)
            rcs_cmd = float(action[1])

            prev = self.state
            self.state = physics.step_descent(prev, throttle_cmd, rcs_cmd, DT)
            self.steps += 1

            outcome = physics.classify_outcome(self.state)
            terminated = outcome in ("TOUCHDOWN", "CRASH", "FUEL_EXHAUSTED")
            truncated = self.steps >= MAX_STEPS and not terminated

            reward = rewards.shaping_reward(prev, self.state, outcome if terminated else "FLYING")
            if truncated:
                # Running out the clock without landing must never be a safe
                # alternative to attempting a landing. Without this, a policy
                # can discover that hovering just above a "nearly safe" speed
                # indefinitely collects positive per-step shaping reward
                # while paying no terminal penalty at all — which is exactly
                # what happened here (it drove descent speed to just under
                # the crash threshold and then stopped descending further).
                reward -= 70.0
            obs = np.array(observation_from_state(self.state), dtype=np.float32)
            info = {"outcome": outcome}
            return obs, reward, terminated, truncated, info

else:  # pragma: no cover
    LunarDescentEnv = None
