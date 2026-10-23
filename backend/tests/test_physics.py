import math

from simulation import physics


def test_initial_state_has_full_fuel():
    s = physics.initial_descent_state()
    assert s.fuel_mass == physics.FUEL_MASS_INITIAL


def test_fuel_never_negative():
    s = physics.initial_descent_state()
    for _ in range(5000):
        s = physics.step_descent(s, throttle_cmd=1.0, tilt_cmd=0.0, dt=0.05)
        assert s.fuel_mass >= 0.0
        assert physics.total_mass(s) >= physics.DRY_MASS
        if physics.has_landed(s):
            break


def test_no_penetration_of_surface():
    s = physics.initial_descent_state()
    for _ in range(5000):
        s = physics.step_descent(s, throttle_cmd=0.0, tilt_cmd=0.0, dt=0.05)
        assert s.y >= physics.surface_height(s.x) - 1e-6
        if physics.has_landed(s):
            break


def test_full_throttle_hover_roughly_balances_gravity_at_low_fuel():
    s = physics.initial_descent_state(seed_altitude=500.0)
    s = physics.step_descent(s, throttle_cmd=0.0, tilt_cmd=0.0, dt=0.01)
    # free fall: no thrust means downward acceleration ~ -lunar gravity
    assert s.vy < 0


def test_classify_outcome_touchdown_vs_crash():
    gentle = physics.DescentState(t=0, x=0, y=0, vx=0.0, vy=-0.5, theta=0.0, fuel_mass=10, throttle=0)
    assert physics.classify_outcome(gentle) == "TOUCHDOWN"

    hard = physics.DescentState(t=0, x=0, y=0, vx=0.0, vy=-20.0, theta=0.0, fuel_mass=10, throttle=0)
    assert physics.classify_outcome(hard) == "CRASH"


def test_fuel_exhausted_detected_while_airborne():
    s = physics.DescentState(t=0, x=0, y=500, vx=0, vy=-1, theta=0, fuel_mass=0.0, throttle=0)
    assert physics.classify_outcome(s) == "FUEL_EXHAUSTED"


def test_hard_impact_through_step_descent_is_classified_as_crash():
    # Regression test: step_descent() must not zero out vx/vy on ground
    # contact, or a free-falling impact would be misclassified as a safe
    # touchdown (this previously let a zero-throttle policy "land" at any
    # speed, since classify_outcome() only ever saw post-clamp zero velocity).
    s = physics.initial_descent_state(seed_altitude=2000.0)
    for _ in range(5000):
        s = physics.step_descent(s, throttle_cmd=0.0, tilt_cmd=0.0, dt=0.05)
        if physics.has_landed(s):
            break
    assert abs(s.vy) > physics.SAFE_VY, "expected a high-speed free-fall impact"
    assert physics.classify_outcome(s) == "CRASH"


def test_tilt_limit_enforced():
    s = physics.initial_descent_state()
    s = physics.step_descent(s, throttle_cmd=1.0, tilt_cmd=math.radians(90), dt=0.05)
    assert abs(s.theta) <= physics.TILT_LIMIT_RAD + 1e-6
