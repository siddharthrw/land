from simulation import mission as m


def run_until(ms: m.MissionStateMachine, phase: str, dt=0.1, max_ticks=20000):
    for _ in range(max_ticks):
        if ms.phase == phase:
            return True
        ms.step(dt)
    return False


def test_launch_transitions_straight_into_descent():
    ms = m.MissionStateMachine()
    assert ms.phase == m.PRELAUNCH
    ms.handle_command("launch")
    assert ms.phase == m.LUNAR_DESCENT
    assert ms.descent_state is not None


def test_full_happy_path_reaches_mission_complete():
    ms = m.MissionStateMachine()
    ms.handle_command("launch")
    reached = run_until(ms, m.MISSION_COMPLETE, dt=0.1, max_ticks=50000)
    assert reached, f"mission stalled in phase {ms.phase}"
    assert ms.score > 0


def test_events_are_not_double_awarded():
    ms = m.MissionStateMachine()
    ms.handle_command("launch")
    seen_types = []
    for _ in range(3000):
        ms.step(0.1)
        for e in ms.drain_events():
            seen_types.append(e.type)
        if ms.phase in (m.TOUCHDOWN, m.CRASH, m.FUEL_EXHAUSTED):
            break
    assert seen_types.count("SAFE_TOUCHDOWN") <= 1


def test_restart_resets_score_and_phase():
    ms = m.MissionStateMachine()
    ms.handle_command("launch")
    for _ in range(50):
        ms.step(0.1)
    assert ms.phase != m.PRELAUNCH
    ms.handle_command("restart")
    assert ms.phase == m.PRELAUNCH
    assert ms.score == 0
    assert ms.descent_state is None


def test_restart_resets_fuel_and_throttle_telemetry():
    ms = m.MissionStateMachine()
    ms.handle_command("launch")
    for _ in range(400):
        ms.step(0.1)
    assert ms.fuel_fraction < 1.0
    ms.handle_command("restart")
    assert ms.fuel_fraction == 1.0
    assert ms.throttle == 0.0


def test_pause_freezes_sim_time():
    ms = m.MissionStateMachine()
    ms.handle_command("launch")
    ms.step(0.1)
    t_before = ms.sim_time
    ms.handle_command("pause")
    ms.step(0.1)
    ms.step(0.1)
    assert ms.sim_time == t_before
