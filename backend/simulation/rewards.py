"""Event-sourced mission score (shown to the user) and RL shaping reward
(used only to train PPO, never shown as "the score"). Keeping these separate
is intentional — see plan.md section 5.
"""
from __future__ import annotations

from agents import baseline
from simulation import physics
from simulation.physics import DescentState

FIXED_EVENT_POINTS: dict[str, int] = {
    "SAFE_TOUCHDOWN": 300,
    "ENGINE_SHUTDOWN": 50,
}

MAX_CONTROLLED_DESCENT_POINTS = 150
MAX_FUEL_EFFICIENCY_POINTS = 100
MAX_RCS_EFFICIENCY_POINTS = 50
MAX_LANDING_ACCURACY_POINTS = 100


def fixed_points(event_type: str) -> int:
    return FIXED_EVENT_POINTS.get(event_type, 0)


def controlled_descent_points(state: DescentState) -> int:
    """Scored at touchdown: smoother terminal velocities score higher. Tilt
    is no longer part of this -- with the main engine fixed vertical and
    RCS independent, theta is purely cosmetic now, not a safety signal."""
    vy_score = max(0.0, 1.0 - abs(state.vy) / physics.SAFE_VY)
    vx_score = max(0.0, 1.0 - abs(state.vx) / physics.SAFE_VX)
    quality = (vy_score + vx_score) / 2.0
    return round(MAX_CONTROLLED_DESCENT_POINTS * quality)


def fuel_efficiency_points(state: DescentState) -> int:
    remaining_fraction = state.fuel_mass / physics.FUEL_MASS_INITIAL
    # Reward ending with some margin, but not hoarding fuel by never landing.
    ideal = 0.15
    closeness = max(0.0, 1.0 - abs(remaining_fraction - ideal) / 0.5)
    return round(MAX_FUEL_EFFICIENCY_POINTS * closeness)


def rcs_efficiency_points(state: DescentState) -> int:
    """Separate from main-fuel efficiency -- rewards leaving RCS propellant
    in reserve, since it's the "save it for a rainy day" backup system."""
    if physics.RCS_FUEL_MASS_INITIAL <= 0:
        return 0
    remaining_fraction = state.rcs_fuel_mass / physics.RCS_FUEL_MASS_INITIAL
    return round(MAX_RCS_EFFICIENCY_POINTS * remaining_fraction)


def landing_accuracy_points(state: DescentState) -> int:
    dist = physics.distance_to_target(state)
    if dist >= physics.LANDING_RADIUS:
        return 0
    closeness = 1.0 - dist / physics.LANDING_RADIUS
    return round(MAX_LANDING_ACCURACY_POINTS * closeness)


import math

VELOCITY_SCALE = 45.0  # m/s, roughly the terminal speed reached from a long freefall


def _potential(state: DescentState) -> float:
    """Potential-based shaping term: higher (closer to 0) the closer the
    lander is to a safe, slow landing. Components are normalized to ~[0, 1]
    so none swamps the others. Critically this includes a velocity term —
    without it, a policy can freefall the entire descent (minimizing
    altitude fast) and only discover braking from the sparse terminal
    crash/touchdown signal, which is a much harder credit-assignment
    problem. Rewarding speed reduction continuously (the classic shaping
    used in Box2D LunarLander) gives a dense, immediate signal to start
    braking long before the ground is close.

    Vertical speed is weighted more heavily than horizontal — observed
    failures were consistently "centered but still falling too fast", not
    "drifted off target", so the gradient should push harder on braking
    the vertical descent rate specifically.
    """
    alt_term = state.y / physics.INITIAL_ALTITUDE
    horiz_term = abs(state.x) / physics.INITIAL_HORIZONTAL_OFFSET
    vy_term = abs(state.vy) / VELOCITY_SCALE
    vx_term = abs(state.vx) / VELOCITY_SCALE
    return -(alt_term + 0.25 * horiz_term + 0.45 * vy_term + 0.3 * vx_term)


def shaping_reward(prev: DescentState, state: DescentState, outcome: str) -> float:
    """Dense reward used only inside the Gymnasium env (lunar_env.py) to
    train PPO. Never surfaced to the user as a score.

    Potential-based progress shaping (gaining altitude costs exactly as much
    as losing it gains, scaled up so it isn't negligible next to the per-step
    penalties below) plus a small constant time penalty, so that drifting
    away from the surface forever is never a safer bet than attempting a
    landing — a prior version of this function let PPO discover exactly that
    degenerate "fly away and never terminate" policy.
    """
    reward = (_potential(state) - _potential(prev)) * 8.0
    # Dense "fly the known-good speed curve" signal: baseline already knows
    # exactly what vertical speed is safe to be at for any given altitude
    # (desired_descent_rate). Rather than only rewarding "did the final
    # outcome work out", reward matching that curve on every single tick —
    # this turns "learn to brake" from a one-shot discovery at the very end
    # of a long episode into a signal present from the first frame.
    ideal_vy = baseline.desired_descent_rate(max(0.0, state.y))
    reward -= abs(state.vy - ideal_vy) * 0.06
    # RCS is a real, limited resource now (not a free gimbal angle), so its
    # usage gets priced directly: a small continuous cost for firing it at
    # all (encourages efficiency), plus a bigger cost for whipping it back
    # and forth tick to tick (the same "flip-flopping" instability problem
    # the old tilt-change penalty solved, now applied to the thing that
    # actually costs propellant).
    reward -= abs(state.rcs_cmd) * 0.15
    reward -= abs(state.rcs_cmd - prev.rcs_cmd) * 4.0
    reward -= abs(state.vx - prev.vx) * 0.05
    reward -= 0.01

    if outcome == "TOUCHDOWN":
        speed_penalty = abs(state.vy) + abs(state.vx)
        reward += 150.0 - min(120.0, speed_penalty * 10.0)
    elif outcome == "CRASH":
        # Graded, not flat: a crash at 2.1 m/s (barely over the safe limit)
        # should barely hurt; a crash at 20 m/s should hurt a lot. A flat
        # -60 regardless of severity gives no gradient telling the policy
        # "brake harder" vs. "you were already close" — which is exactly
        # the failure mode observed (centered over the target, still
        # crashing from excess vertical speed).
        speed_excess = max(0.0, abs(state.vy) - physics.SAFE_VY) + max(0.0, abs(state.vx) - physics.SAFE_VX)
        severity = min(100.0, 25.0 + speed_excess * 7.0)
        reward -= severity
        # Partial credit for being near the target even in a hard landing,
        # so "crashed dead-center" scores better than "crashed off to the
        # side" instead of both being an identical flat penalty.
        dist = physics.distance_to_target(state)
        reward += max(0.0, 1.0 - dist / physics.LANDING_RADIUS) * 15.0
    elif outcome == "FUEL_EXHAUSTED":
        reward -= 60.0

    return reward
