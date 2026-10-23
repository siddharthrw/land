import numpy as np
import pytest

gym_checker = pytest.importorskip("gymnasium.utils.env_checker")

from simulation.lunar_env import LunarDescentEnv, observation_from_state
from simulation import physics


def test_env_passes_gymnasium_checker():
    env = LunarDescentEnv(randomize=True)
    gym_checker.check_env(env, skip_render_check=True)


def test_observation_is_always_within_bounds():
    extreme = physics.DescentState(
        t=0, x=10000, y=50000, vx=500, vy=-500, theta=physics.TILT_LIMIT_RAD,
        fuel_mass=physics.FUEL_MASS_INITIAL, throttle=1.0,
    )
    obs = observation_from_state(extreme)
    assert all(-5.0 <= v <= 5.0 for v in obs)


def test_reset_is_seed_reproducible():
    env = LunarDescentEnv(randomize=True)
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs1, obs2)


def test_episode_terminates_within_max_steps():
    from simulation.lunar_env import MAX_STEPS

    env = LunarDescentEnv(randomize=True)
    env.reset(seed=1)
    # Zero throttle (action[0] = -1) lets it free-fall into a crash well
    # before MAX_STEPS, exercising termination rather than truncation.
    for _ in range(MAX_STEPS):
        _, _, terminated, truncated, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            assert terminated and info["outcome"] == "CRASH"
            return
    pytest.fail("episode never terminated or truncated")
