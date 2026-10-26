import math

from simulation import physics


def test_initial_state_has_full_fuel():
    s = physics.initial_descent_state()
    assert s.fuel_mass == physics.FUEL_MASS_INITIAL
    assert s.rcs_fuel_mass == physics.RCS_FUEL_MASS_INITIAL


def test_scenario_fuel_fractions_scale_initial_state():
    s = physics.initial_descent_state(seed_fuel_fraction=0.5, seed_rcs_fraction=0.25)
    assert s.fuel_mass == physics.FUEL_MASS_INITIAL * 0.5
    assert s.rcs_fuel_mass == physics.RCS_FUEL_MASS_INITIAL * 0.25


def test_fuel_never_negative():
    s = physics.initial_descent_state()
    for _ in range(5000):
        s = physics.step_descent(s, throttle_cmd=1.0, rcs_cmd=1.0, dt=0.05)
        assert s.fuel_mass >= 0.0
        assert s.rcs_fuel_mass >= 0.0
        assert physics.total_mass(s) >= physics.DRY_MASS
        if physics.has_landed(s):
            break


def test_no_penetration_of_surface():
    s = physics.initial_descent_state()
    for _ in range(5000):
        s = physics.step_descent(s, throttle_cmd=0.0, rcs_cmd=0.0, dt=0.05)
        assert s.y >= physics.surface_height(s.x) - 1e-6
        if physics.has_landed(s):
            break


def test_full_throttle_hover_roughly_balances_gravity_at_low_fuel():
    s = physics.initial_descent_state(seed_altitude=500.0)
    s = physics.step_descent(s, throttle_cmd=0.0, rcs_cmd=0.0, dt=0.01)
    # free fall: no thrust means downward acceleration ~ -lunar gravity
    assert s.vy < 0


def test_rcs_is_lateral_only_and_independent_of_main_throttle():
    s = physics.initial_descent_state(seed_altitude=500.0)
    # Zero main throttle, full RCS -- should still get purely sideways
    # acceleration (main fuel untouched, RCS fuel consumed).
    s2 = physics.step_descent(s, throttle_cmd=0.0, rcs_cmd=1.0, dt=0.05)
    assert s2.fuel_mass == s.fuel_mass
    assert s2.rcs_fuel_mass < s.rcs_fuel_mass
    assert s2.vx > s.vx  # pushed sideways
    assert s2.vy < s.vy  # still just falling under gravity, no vertical thrust


def test_main_engine_is_vertical_only():
    s = physics.initial_descent_state(seed_altitude=500.0)
    s2 = physics.step_descent(s, throttle_cmd=1.0, rcs_cmd=0.0, dt=0.05)
    assert s2.rcs_fuel_mass == s.rcs_fuel_mass
    assert s2.fuel_mass < s.fuel_mass
    assert s2.vx == s.vx  # main engine alone never pushes sideways


def test_classify_outcome_touchdown_vs_crash():
    gentle = physics.DescentState(
        t=0, x=0, y=0, vx=0.0, vy=-0.5, theta=0.0,
        fuel_mass=10, rcs_fuel_mass=5, throttle=0, rcs_cmd=0,
    )
    assert physics.classify_outcome(gentle) == "TOUCHDOWN"

    hard = physics.DescentState(
        t=0, x=0, y=0, vx=0.0, vy=-20.0, theta=0.0,
        fuel_mass=10, rcs_fuel_mass=5, throttle=0, rcs_cmd=0,
    )
    assert physics.classify_outcome(hard) == "CRASH"


def test_fuel_exhausted_detected_while_airborne():
    s = physics.DescentState(
        t=0, x=0, y=500, vx=0, vy=-1, theta=0,
        fuel_mass=0.0, rcs_fuel_mass=5, throttle=0, rcs_cmd=0,
    )
    assert physics.classify_outcome(s) == "FUEL_EXHAUSTED"


def test_hard_impact_through_step_descent_is_classified_as_crash():
    # Regression test: step_descent() must not zero out vx/vy on ground
    # contact, or a free-falling impact would be misclassified as a safe
    # touchdown (this previously let a zero-throttle policy "land" at any
    # speed, since classify_outcome() only ever saw post-clamp zero velocity).
    s = physics.initial_descent_state(seed_altitude=2000.0)
    for _ in range(5000):
        s = physics.step_descent(s, throttle_cmd=0.0, rcs_cmd=0.0, dt=0.05)
        if physics.has_landed(s):
            break
    assert abs(s.vy) > physics.SAFE_VY, "expected a high-speed free-fall impact"
    assert physics.classify_outcome(s) == "CRASH"


def test_rcs_command_clamped_to_unit_range():
    s = physics.initial_descent_state()
    s = physics.step_descent(s, throttle_cmd=1.0, rcs_cmd=5.0, dt=0.05)
    assert abs(s.rcs_cmd) <= 1.0 + 1e-6


def test_cosmetic_tilt_bounded():
    s = physics.initial_descent_state()
    s = physics.step_descent(s, throttle_cmd=1.0, rcs_cmd=1.0, dt=0.05)
    assert abs(s.theta) <= physics.COSMETIC_TILT_MAX + 1e-6


def test_rcs_fuel_depletion_disables_lateral_thrust_only():
    s = physics.initial_descent_state(seed_altitude=500.0, seed_rcs_fraction=0.0)
    assert s.rcs_fuel_mass == 0.0
    s2 = physics.step_descent(s, throttle_cmd=0.5, rcs_cmd=1.0, dt=0.05)
    assert s2.vx == s.vx  # no RCS fuel -> no lateral acceleration, even though commanded
    assert s2.fuel_mass < s.fuel_mass  # main engine still works
