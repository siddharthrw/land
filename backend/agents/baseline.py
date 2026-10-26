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
X_KP = 0.0065
VX_KP = 0.11
# A normal drop never produces more than roughly +-15 m/s of sideways speed
# on its own; much more than that only happens via manual flight or an
# artificial scenario. Past that point, the plain P-controller below (tuned
# for normal speeds) keeps commanding "push toward center" without
# accounting for momentum it hasn't planned for, and overshoots straight
# through the target, crashing on the far side instead of slowing to meet
# it. This boosts the damping term specifically when speed is abnormally
# high, so it prioritizes killing that excess speed before it keeps pushing
# — normal flights (excess_speed == 0) are completely unaffected.
VX_NORMAL_RANGE = 15.0
VX_DAMPING_BOOST_GAIN = 0.45
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

    lateral_error = -state.x
    excess_speed = max(0.0, abs(state.vx) - VX_NORMAL_RANGE)
    damping_boost = 1.0 + excess_speed * VX_DAMPING_BOOST_GAIN
    tilt_cmd = X_KP * lateral_error - VX_KP * damping_boost * state.vx
    tilt_cmd = min(TILT_FROM_LATERAL_LIMIT, max(-TILT_FROM_LATERAL_LIMIT, tilt_cmd))

    if altitude < 40.0:
        tilt_cmd *= max(0.0, altitude / 40.0)

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
