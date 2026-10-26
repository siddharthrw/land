"""Controller A: a feedback guidance pilot for the lunar descent.

No training required — always available, used as the default controller and
as the fallback whenever no trained PPO checkpoint is present. Pure function
of the current descent state; produces (throttle, tilt_cmd, reason) so the
decision log can show a truthful, telemetry-derived explanation.
"""
from __future__ import annotations

import math

from simulation import physics
from simulation.physics import DescentState

# Gains, tuned by hand for the physics constants in physics.py.
VY_TARGET_GAIN = 0.65     # how aggressively descent speed scales with altitude
VY_KP = 0.9
VY_KD = 0.0
MIN_TIME_TO_GO = 2.5  # s, floor to keep the 1/T^2 term from blowing up near the ground
# Was capped well below the vehicle's real tilt capability
# (physics.TILT_LIMIT_RAD, 28deg) -- fine for normal drift, but it meant
# the controller was out of turning authority (saturated at this cap more
# than half the time) exactly when an abnormally large sideways speed
# needed more of it, not less.
TILT_FROM_LATERAL_LIMIT = math.radians(26.0)

HOVER_THROTTLE = (physics.DRY_MASS + physics.FUEL_MASS_INITIAL * 0.5) * physics.LUNAR_G / physics.MAX_THRUST


def desired_descent_rate(altitude: float) -> float:
    """A classic 'velocity to kill by the surface' guidance curve."""
    return -min(18.0, VY_TARGET_GAIN * math.sqrt(max(0.0, altitude)))


def estimate_time_to_go(state: DescentState) -> float:
    """How many seconds until touchdown, at the current vertical rate. If
    not meaningfully descending yet (e.g. still climbing right after an
    extreme manual input), fall back to the normal guidance curve's implied
    rate instead of letting a near-zero/positive vy blow up the estimate."""
    if state.vy < -0.5:
        return state.y / (-state.vy)
    ideal_vy = desired_descent_rate(state.y)
    return state.y / max(1.0, -ideal_vy)


def lateral_accel_command(state: DescentState, time_to_go: float) -> float:
    """The acceleration needed RIGHT NOW to drive both lateral position and
    lateral velocity to zero exactly when time_to_go runs out. This is a
    standard terminal-guidance law (the same shape used in real powered-
    descent guidance, e.g. Apollo's lunar module) — solving the two-point
    boundary problem "x(T)=0, v(T)=0" for a smoothly-varying acceleration
    gives a = -6x/T^2 - 4*vx/T. Unlike a plain position-error controller,
    this naturally weights "close the distance" against "kill the speed" by
    how much time is actually left, instead of always maximizing one and
    only noticing the other once it's nearly out of room.
    """
    T = max(MIN_TIME_TO_GO, time_to_go)
    return -6.0 * state.x / (T * T) - 4.0 * state.vx / T


def compute_action(state: DescentState) -> tuple[float, float, str]:
    altitude = max(0.0, state.y)
    vy_target = desired_descent_rate(altitude)
    vy_error = vy_target - state.vy

    mass = physics.total_mass(state)
    hover_throttle = mass * physics.LUNAR_G / physics.MAX_THRUST
    # vy_error = vy_target - vy. If vy_error is very negative, the target
    # descent rate is far below the actual rate (we're not falling fast
    # enough yet) -> cut throttle and let gravity do more work. If vy_error
    # is positive, we're falling faster than the target profile -> add
    # throttle to slow the descent.
    throttle = hover_throttle + VY_KP * vy_error * 0.12
    throttle = min(physics.MAX_THROTTLE, max(physics.MIN_THROTTLE, throttle))

    time_to_go = estimate_time_to_go(state)
    a_lateral = lateral_accel_command(state, time_to_go)
    thrust = max(1.0, throttle * physics.MAX_THRUST)
    # sin(theta) = required lateral accel * mass / available thrust — how
    # much the thrust vector needs to tilt to produce that acceleration.
    sin_theta = max(-1.0, min(1.0, a_lateral * mass / thrust))
    tilt_cmd = math.asin(sin_theta)
    tilt_cmd = min(TILT_FROM_LATERAL_LIMIT, max(-TILT_FROM_LATERAL_LIMIT, tilt_cmd))

    if altitude < 10.0:
        tilt_cmd *= max(0.0, altitude / 10.0)

    if state.fuel_mass <= 0.0:
        return 0.0, 0.0, "Fuel depleted — engine cannot fire."

    if altitude < 15.0 and abs(state.vy) < 3.0:
        reason = "Final approach: holding gentle descent rate for touchdown."
    elif abs(vy_error) > 3.0:
        reason = (
            f"Increasing thrust: descent speed {state.vy:.1f} m/s exceeds "
            f"target envelope {vy_target:.1f} m/s."
        )
    elif abs(state.x) > 15.0:
        reason = f"Reducing tilt to correct lateral drift of {state.x:.0f} m."
    elif state.fuel_mass < physics.FUEL_MASS_INITIAL * 0.15:
        reason = "Fuel reserve is low; switching to a conservative descent profile."
    else:
        reason = "Tracking nominal descent profile."

    return throttle, tilt_cmd, reason
