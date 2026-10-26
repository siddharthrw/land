"""Controller A: a feedback guidance pilot for the lunar descent.

No training required — always available, used as the default controller and
as the fallback whenever no trained PPO checkpoint is present. Pure function
of the current descent state; produces (throttle, rcs_cmd, reason) so the
decision log can show a truthful, telemetry-derived explanation.

Vertical (main engine) and lateral (RCS) are fully independent now -- no
more coupling through a gimbal angle -- so each gets its own clean control
law with no need to borrow authority from the other.
"""
from __future__ import annotations

import math

from simulation import physics
from simulation.physics import DescentState

# Gains, tuned by hand for the physics constants in physics.py.
VY_TARGET_GAIN = 0.65     # how aggressively descent speed scales with altitude
VY_KP = 0.9
MIN_TIME_TO_GO = 2.5  # s, floor to keep the 1/T^2 term from blowing up near the ground

HOVER_THROTTLE = (physics.DRY_MASS + physics.FUEL_MASS_INITIAL * 0.5) * physics.LUNAR_G / physics.MAX_THRUST


def desired_descent_rate(altitude: float) -> float:
    """A classic 'velocity to kill by the surface' guidance curve."""
    return -min(18.0, VY_TARGET_GAIN * math.sqrt(max(0.0, altitude)))


def estimate_time_to_go(state: DescentState) -> float:
    """How many seconds until touchdown, at the current vertical rate. If
    not meaningfully descending yet, fall back to the normal guidance
    curve's implied rate instead of letting a near-zero/positive vy blow up
    the estimate."""
    if state.vy < -0.5:
        return state.y / (-state.vy)
    ideal_vy = desired_descent_rate(state.y)
    return state.y / max(1.0, -ideal_vy)


def lateral_accel_command(state: DescentState, time_to_go: float) -> float:
    """The acceleration needed RIGHT NOW to drive both lateral position and
    lateral velocity to zero exactly when time_to_go runs out -- the same
    terminal-guidance law used in real powered-descent guidance (e.g.
    Apollo's lunar module). Solving the two-point boundary problem
    "x(T)=0, v(T)=0" for a smoothly-varying acceleration gives
    a = -6x/T^2 - 4*vx/T.
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
    # RCS is independent of the main engine now -- straight conversion from
    # required acceleration to thrust fraction, no gimbal geometry, no
    # borrowing throttle from the vertical law.
    rcs_cmd = a_lateral * mass / physics.RCS_MAX_THRUST
    rcs_cmd = min(1.0, max(-1.0, rcs_cmd))

    if state.fuel_mass <= 0.0 and state.rcs_fuel_mass <= 0.0:
        return 0.0, 0.0, "Main fuel and RCS both depleted — no propulsion remaining."
    if state.fuel_mass <= 0.0:
        return 0.0, rcs_cmd, "Main fuel depleted — vertical braking lost, using RCS to stay centered on the way down."
    if state.rcs_fuel_mass <= 0.0:
        rcs_cmd = 0.0

    if altitude < 15.0 and abs(state.vy) < 3.0:
        reason = "Final approach: holding gentle descent rate for touchdown."
    elif abs(vy_error) > 3.0:
        reason = (
            f"Increasing thrust: descent speed {state.vy:.1f} m/s exceeds "
            f"target envelope {vy_target:.1f} m/s."
        )
    elif abs(state.x) > 15.0:
        reason = f"RCS correcting lateral drift of {state.x:.0f} m."
    elif state.fuel_mass < physics.FUEL_MASS_INITIAL * 0.15:
        reason = "Fuel reserve is low; switching to a conservative descent profile."
    else:
        reason = "Tracking nominal descent profile."

    return throttle, rcs_cmd, reason
