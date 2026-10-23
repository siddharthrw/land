"""Deterministic physics for the lunar descent phase.

Everything here runs at a fixed timestep driven by mission.py. Units are SI
(meters, seconds, kilograms) for the descent model, independent of the
cinematic render-world scale used for the scripted ascent/transfer phases.

Documented assumptions (tunable, kept in one place):
  - Lunar surface gravity: 1.62 m/s^2.
  - Descent engine Isp ~311s (LM descent-engine-like), used only in the
    Tsiolkovsky mass-flow term below.
  - Dry mass / propellant split chosen for a ~2-3 minute descent window
    from ~1800m altitude at a modest horizontal offset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

G0 = 9.81            # standard gravity, used only for the rocket equation
LUNAR_G = 1.62        # local gravitational acceleration on the Moon
ISP = 311.0           # seconds

DRY_MASS = 1800.0        # kg, lander structure
FUEL_MASS_INITIAL = 950.0  # kg, descent propellant
MAX_THRUST = 16000.0      # N
MIN_THROTTLE = 0.0
MAX_THROTTLE = 1.0
TILT_LIMIT_RAD = math.radians(28.0)
MAX_TILT_RATE = math.radians(45.0)  # rad/s, how fast commanded tilt can be tracked

SAFE_VY = 2.0         # m/s, max safe vertical speed at touchdown
SAFE_VX = 1.5         # m/s, max safe horizontal speed at touchdown
SAFE_TILT_RAD = math.radians(10.0)
LANDING_SITE_X = 0.0
LANDING_RADIUS = 60.0  # m, counts as "near target" for scoring

INITIAL_ALTITUDE = 1800.0
INITIAL_HORIZONTAL_OFFSET = 420.0


@dataclass(frozen=True)
class DescentState:
    t: float
    x: float          # horizontal distance from landing site, meters
    y: float          # altitude, meters (0 = surface)
    vx: float
    vy: float
    theta: float       # tilt from vertical, radians, positive = thrust tilts +x
    fuel_mass: float
    throttle: float    # last applied throttle, for telemetry/exhaust FX


def initial_descent_state(seed_offset_x: float = INITIAL_HORIZONTAL_OFFSET,
                           seed_altitude: float = INITIAL_ALTITUDE) -> DescentState:
    return DescentState(
        t=0.0,
        x=seed_offset_x,
        y=seed_altitude,
        vx=-6.0,
        vy=-2.0,
        theta=0.0,
        fuel_mass=FUEL_MASS_INITIAL,
        throttle=0.0,
    )


def surface_height(x: float) -> float:
    """Deterministic, mildly cratered flat landing plain. 0 near the site."""
    return max(0.0, 2.5 * math.sin(x / 140.0) * math.exp(-abs(x) / 500.0))


def total_mass(state: DescentState) -> float:
    return DRY_MASS + state.fuel_mass


def step_descent(state: DescentState, throttle_cmd: float, tilt_cmd: float, dt: float) -> DescentState:
    """Advance the descent physics by one fixed timestep.

    throttle_cmd in [0, 1], tilt_cmd in [-TILT_LIMIT_RAD, TILT_LIMIT_RAD].
    Enforces: no negative fuel/mass, tilt rate limit, no penetration of the
    lunar surface.
    """
    throttle = min(MAX_THROTTLE, max(MIN_THROTTLE, throttle_cmd))
    tilt_target = min(TILT_LIMIT_RAD, max(-TILT_LIMIT_RAD, tilt_cmd))

    if state.fuel_mass <= 0.0:
        throttle = 0.0

    # Track the commanded tilt at a bounded rate (actuator isn't instantaneous).
    tilt_delta = tilt_target - state.theta
    max_step = MAX_TILT_RATE * dt
    theta = state.theta + max(-max_step, min(max_step, tilt_delta))

    mass = total_mass(state)
    thrust = throttle * MAX_THRUST

    ax = thrust * math.sin(theta) / mass
    ay = thrust * math.cos(theta) / mass - LUNAR_G

    vx = state.vx + ax * dt
    vy = state.vy + ay * dt
    x = state.x + vx * dt
    y = state.y + vy * dt

    mass_flow = thrust / (ISP * G0)
    fuel_mass = max(0.0, state.fuel_mass - mass_flow * dt)

    # Clamp position to the surface on contact, but preserve the actual
    # impact velocity — classify_outcome() needs the real vx/vy to tell a
    # safe touchdown from a crash. (Zeroing velocity here used to make every
    # ground contact look like a perfect zero-speed landing regardless of
    # impact speed, which a policy could trivially exploit by never firing
    # the engine at all.) Once contact occurs, mission.py stops stepping
    # this state, so there's no sinking/instability risk from leaving the
    # velocity intact.
    ground = surface_height(x)
    if y <= ground:
        y = ground

    return DescentState(
        t=state.t + dt,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        theta=theta,
        fuel_mass=fuel_mass,
        throttle=throttle,
    )


def has_landed(state: DescentState) -> bool:
    return state.y <= surface_height(state.x) + 1e-6


def classify_outcome(state: DescentState) -> str:
    """Returns one of: FLYING, TOUCHDOWN, CRASH, FUEL_EXHAUSTED."""
    if not has_landed(state):
        if state.fuel_mass <= 0.0 and state.y > 5.0:
            return "FUEL_EXHAUSTED"
        return "FLYING"

    safe = (
        abs(state.vy) <= SAFE_VY
        and abs(state.vx) <= SAFE_VX
        and abs(state.theta) <= SAFE_TILT_RAD
    )
    return "TOUCHDOWN" if safe else "CRASH"


def distance_to_target(state: DescentState) -> float:
    return abs(state.x - LANDING_SITE_X)
