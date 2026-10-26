"""Deterministic physics for the lunar descent phase.

Everything here runs at a fixed timestep driven by mission.py. Units are SI
(meters, seconds, kilograms) for the descent model, independent of the
cinematic render-world scale used for the scripted ascent/transfer phases.

Two independent propulsion systems, like a real lander (e.g. the Apollo
Lunar Module's descent engine + RCS):
  - Main engine: fixed pointing straight up, variable throttle, does all
    the vertical braking. Draws from the main fuel tank.
  - RCS (Reaction Control System): weak, lateral-only thrusters for fine
    positioning, on their own small, separate propellant budget. Running
    out of RCS fuel doesn't affect the main engine, and vice versa --
    burning through main fuel still leaves RCS available to nudge toward
    the target even in freefall, just with no further vertical braking.

Documented assumptions (tunable, kept in one place):
  - Lunar surface gravity: 1.62 m/s^2.
  - Descent engine Isp ~311s (LM descent-engine-like), used only in the
    Tsiolkovsky mass-flow term below.
  - RCS Isp ~65s (cold-gas-thruster-like -- much less propellant-efficient
    than the main engine, which is realistic and also why it's reserved
    for fine corrections, not primary braking).
  - Dry mass / propellant split chosen for a ~2-3 minute descent window
    from ~1800m altitude at a modest horizontal offset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

G0 = 9.81            # standard gravity, used only for the rocket equation
LUNAR_G = 1.62        # local gravitational acceleration on the Moon
ISP = 311.0           # seconds, main engine
RCS_ISP = 150.0       # seconds, RCS thrusters (hydrazine-monoprop-like, not cold-gas)

DRY_MASS = 1800.0        # kg, lander structure
FUEL_MASS_INITIAL = 950.0   # kg, main descent propellant
RCS_FUEL_MASS_INITIAL = 60.0  # kg, RCS propellant -- small and separate
MAX_THRUST = 16000.0      # N, main engine
RCS_MAX_THRUST = 1700.0   # N, per-side RCS -- ~11% of the main engine, weak
                           # enough to clearly be a secondary system, but
                           # with enough real authority to reliably finish
                           # the job across the normal recoverable range
MIN_THROTTLE = 0.0
MAX_THROTTLE = 1.0
COSMETIC_TILT_MAX = math.radians(20.0)  # purely visual lean, no longer physical
MAX_TILT_RATE = math.radians(60.0)  # rad/s, how fast the cosmetic lean animates

SAFE_VY = 2.0         # m/s, max safe vertical speed at touchdown
SAFE_VX = 1.5         # m/s, max safe horizontal speed at touchdown
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
    theta: float       # COSMETIC lean only now -- does not affect physics,
                       # just gives the lander sprite something to bank
                       # into when RCS is firing hard.
    fuel_mass: float      # main engine propellant remaining
    rcs_fuel_mass: float  # RCS propellant remaining
    throttle: float    # last applied main-engine throttle, for telemetry/FX
    rcs_cmd: float      # last applied RCS command [-1, 1], for telemetry/FX


def initial_descent_state(seed_offset_x: float = INITIAL_HORIZONTAL_OFFSET,
                           seed_altitude: float = INITIAL_ALTITUDE,
                           seed_fuel_fraction: float = 1.0,
                           seed_rcs_fraction: float = 1.0) -> DescentState:
    return DescentState(
        t=0.0,
        x=seed_offset_x,
        y=seed_altitude,
        vx=-6.0,
        vy=-2.0,
        theta=0.0,
        fuel_mass=FUEL_MASS_INITIAL * max(0.0, min(1.0, seed_fuel_fraction)),
        rcs_fuel_mass=RCS_FUEL_MASS_INITIAL * max(0.0, min(1.0, seed_rcs_fraction)),
        throttle=0.0,
        rcs_cmd=0.0,
    )


def surface_height(x: float) -> float:
    """Deterministic, mildly cratered flat landing plain. 0 near the site."""
    return max(0.0, 2.5 * math.sin(x / 140.0) * math.exp(-abs(x) / 500.0))


def total_mass(state: DescentState) -> float:
    return DRY_MASS + state.fuel_mass + state.rcs_fuel_mass


def step_descent(state: DescentState, throttle_cmd: float, rcs_cmd: float, dt: float) -> DescentState:
    """Advance the descent physics by one fixed timestep.

    throttle_cmd in [0, 1] drives the main engine (vertical thrust only).
    rcs_cmd in [-1, 1] drives the RCS thrusters (lateral thrust only, its
    own fuel budget). Enforces: no negative fuel/mass on either tank, no
    penetration of the lunar surface.
    """
    throttle = min(MAX_THROTTLE, max(MIN_THROTTLE, throttle_cmd))
    if state.fuel_mass <= 0.0:
        throttle = 0.0

    rcs = min(1.0, max(-1.0, rcs_cmd))
    if state.rcs_fuel_mass <= 0.0:
        rcs = 0.0

    mass = total_mass(state)
    main_thrust = throttle * MAX_THRUST
    rcs_thrust = rcs * RCS_MAX_THRUST

    ax = rcs_thrust / mass
    ay = main_thrust / mass - LUNAR_G

    vx = state.vx + ax * dt
    vy = state.vy + ay * dt
    x = state.x + vx * dt
    y = state.y + vy * dt

    main_mass_flow = main_thrust / (ISP * G0)
    fuel_mass = max(0.0, state.fuel_mass - main_mass_flow * dt)

    rcs_mass_flow = abs(rcs_thrust) / (RCS_ISP * G0)
    rcs_fuel_mass = max(0.0, state.rcs_fuel_mass - rcs_mass_flow * dt)

    # Cosmetic lean only -- banks visually toward whichever way RCS is
    # firing, rate-limited so it animates smoothly instead of snapping.
    tilt_target = -rcs * COSMETIC_TILT_MAX
    tilt_delta = tilt_target - state.theta
    max_step = MAX_TILT_RATE * dt
    theta = state.theta + max(-max_step, min(max_step, tilt_delta))

    # Clamp position to the surface on contact, but preserve the actual
    # impact velocity — classify_outcome() needs the real vx/vy to tell a
    # safe touchdown from a crash.
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
        rcs_fuel_mass=rcs_fuel_mass,
        throttle=throttle,
        rcs_cmd=rcs,
    )


def has_landed(state: DescentState) -> bool:
    return state.y <= surface_height(state.x) + 1e-6


def classify_outcome(state: DescentState) -> str:
    """Returns one of: FLYING, TOUCHDOWN, CRASH, FUEL_EXHAUSTED.

    Tilt is no longer part of the safety check -- with the main engine
    fixed pointing up and RCS separate, there's no physical "tipped over"
    failure mode anymore (theta is purely cosmetic). Landing safety now
    depends only on impact speed, same as a real powered vertical landing.
    """
    if not has_landed(state):
        if state.fuel_mass <= 0.0 and state.y > 5.0:
            return "FUEL_EXHAUSTED"
        return "FLYING"

    safe = abs(state.vy) <= SAFE_VY and abs(state.vx) <= SAFE_VX
    return "TOUCHDOWN" if safe else "CRASH"


def distance_to_target(state: DescentState) -> float:
    return abs(state.x - LANDING_SITE_X)
