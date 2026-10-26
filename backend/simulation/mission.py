"""The mission state machine — a pure landing game. The lander starts in
freefall over the lunar surface and must touch down between the two marker
poles. The frontend only ever renders what `snapshot()` returns and reacts
to events from `drain_events()`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import count

from agents import baseline
from simulation import physics, rewards
from simulation.physics import DescentState

# The bounds the *live* PPO checkpoint was actually trained within (not tied
# to lunar_env.py's training-range constants, which have since changed for
# newer experiments) -- a verified physics sweep confirmed PPO's reliability
# collapses once it's pushed outside what it's actually practiced. Rather
# than pretend a small learned network can safely handle any input, PPO only
# flies inside this box; baseline (a plain formula, reliable at any input)
# takes over instantly whenever the lander drifts outside it, and PPO
# resumes the moment it's back in range.
PPO_TRUSTED_MAX_ALTITUDE = 2200.0
PPO_TRUSTED_MAX_OFFSET = 550.0


def _in_ppo_comfort_zone(state: DescentState) -> bool:
    return state.y <= PPO_TRUSTED_MAX_ALTITUDE and abs(state.x) <= PPO_TRUSTED_MAX_OFFSET

# --- Phases -----------------------------------------------------------------

PRELAUNCH = "PRELAUNCH"
LUNAR_DESCENT = "LUNAR_DESCENT"
TOUCHDOWN = "TOUCHDOWN"
MISSION_COMPLETE = "MISSION_COMPLETE"

FUEL_EXHAUSTED = "FUEL_EXHAUSTED"
CRASH = "CRASH"
ABORTED = "ABORTED"

TERMINAL_PHASES = {MISSION_COMPLETE, FUEL_EXHAUSTED, CRASH, ABORTED}

FIDELITY_MAP = {
    PRELAUNCH: "scripted",
    LUNAR_DESCENT: "simulated",
    TOUCHDOWN: "simulated",
    MISSION_COMPLETE: "simulated",
}


@dataclass
class MissionEvent:
    id: str
    type: str
    message: str
    phase: str
    sim_time: float
    score_delta: int = 0


@dataclass
class MissionStateMachine:
    phase: str = PRELAUNCH
    sim_time: float = 0.0
    paused: bool = False
    score: int = 0
    active_controller: str = "baseline"
    manual_input: tuple[float, float] | None = None
    sim_speed: int = 1

    descent_state: DescentState | None = None
    decision_log: list = field(default_factory=list)
    throttle: float = 0.0
    fuel_fraction: float = 1.0

    _awarded: set = field(default_factory=set)
    _events: list = field(default_factory=list)
    _event_ids: "count" = field(default_factory=lambda: count(1))
    _phase_start: float = 0.0
    _ppo_model: object = None

    def set_ppo_model(self, model) -> None:
        self._ppo_model = model

    # -- commands -------------------------------------------------------

    def handle_command(self, cmd_type: str, payload=None) -> None:
        if cmd_type == "launch" and self.phase == PRELAUNCH:
            self.descent_state = physics.initial_descent_state()
            self.phase = LUNAR_DESCENT
            self._phase_start = self.sim_time
            self._emit("DESCENT_START", "Descent engines ignited — beginning final approach.", 0)
        elif cmd_type == "pause":
            self.paused = True
        elif cmd_type == "resume":
            self.paused = False
        elif cmd_type == "restart":
            self.reset()
        elif cmd_type == "abort" and self.phase not in TERMINAL_PHASES:
            self.phase = ABORTED
            self._emit("MISSION_ABORTED", "Mission aborted by operator.", 0)
        elif cmd_type == "set_controller":
            if payload == "ppo" and self._ppo_model is None:
                self._emit(
                    "CONTROLLER_UNAVAILABLE",
                    "No trained PPO model loaded; staying on baseline controller.",
                    0,
                )
            else:
                self.active_controller = payload
        elif cmd_type == "manual_input":
            self.manual_input = (payload.get("throttle", 0.0), payload.get("tilt", 0.0))
        elif cmd_type == "set_sim_speed":
            self.sim_speed = max(1, min(8, int(payload)))

    def reset(self) -> None:
        self.phase = PRELAUNCH
        self.sim_time = 0.0
        self.paused = False
        self.score = 0
        self.manual_input = None
        self.descent_state = None
        self.decision_log = []
        self.throttle = 0.0
        self.fuel_fraction = 1.0
        self._awarded = set()
        self._events = []
        self._phase_start = 0.0

    # -- event/score helpers ---------------------------------------------

    def _emit(self, event_type: str, message: str, score_delta: int) -> MissionEvent:
        event = MissionEvent(
            id=f"evt-{next(self._event_ids)}",
            type=event_type,
            message=message,
            phase=self.phase,
            sim_time=round(self.sim_time, 2),
            score_delta=score_delta,
        )
        self.score += score_delta
        self._events.append(event)
        return event

    def _award_once(self, key: str, event_type: str, message: str) -> None:
        if key in self._awarded:
            return
        self._awarded.add(key)
        self._emit(event_type, message, rewards.fixed_points(event_type))

    def drain_events(self):
        pending = self._events
        self._events = []
        return pending

    # -- main loop ---------------------------------------------------------

    def step(self, dt: float) -> None:
        if self.paused or self.phase in TERMINAL_PHASES or self.phase == PRELAUNCH:
            return
        self.sim_time += dt
        if self.phase == LUNAR_DESCENT:
            self._step_lunar_descent(dt)
        elif self.phase == TOUCHDOWN:
            self._step_touchdown(dt)

    def _goto(self, phase: str) -> None:
        self.phase = phase
        self._phase_start = self.sim_time

    # -- the only real physics: lunar descent --------------------------------

    def _step_lunar_descent(self, dt: float) -> None:
        state = self.descent_state
        if self.active_controller == "manual" and self.manual_input is not None:
            throttle_cmd, tilt_cmd = self.manual_input
            reason = "Manual control."
        elif self.active_controller == "ppo" and self._ppo_model is not None:
            if _in_ppo_comfort_zone(state):
                import numpy as np
                from simulation.lunar_env import observation_from_state
                obs = observation_from_state(state)
                action, _ = self._ppo_model.predict(np.array(obs, dtype="float32"), deterministic=True)
                throttle_cmd = float((action[0] + 1.0) / 2.0)
                tilt_cmd = float(action[1]) * physics.TILT_LIMIT_RAD
                reason = "PPO policy output."
            else:
                throttle_cmd, tilt_cmd, baseline_reason = baseline.compute_action(state)
                reason = f"[Safety fallback — outside PPO's trained range] {baseline_reason}"
        else:
            throttle_cmd, tilt_cmd, reason = baseline.compute_action(state)

        new_state = physics.step_descent(state, throttle_cmd, tilt_cmd, dt)
        self.descent_state = new_state
        self.throttle = new_state.throttle
        self.fuel_fraction = new_state.fuel_mass / physics.FUEL_MASS_INITIAL

        self.decision_log.append({
            "simTime": round(self.sim_time, 1),
            "controller": self.active_controller,
            "throttle": round(new_state.throttle, 2),
            "tilt": round(math.degrees(new_state.theta), 1),
            "reason": reason,
        })
        if len(self.decision_log) > 40:
            self.decision_log.pop(0)

        outcome = physics.classify_outcome(new_state)
        if outcome == "FLYING":
            return

        if outcome == "TOUCHDOWN":
            self._award_once("SAFE_TOUCHDOWN", "SAFE_TOUCHDOWN", "Touchdown confirmed — the Eagle has landed.")
            self._emit(
                "CONTROLLED_DESCENT_BONUS", "Controlled-descent quality bonus.",
                rewards.controlled_descent_points(new_state),
            )
            self._emit(
                "FUEL_EFFICIENCY_BONUS", "Fuel-efficiency bonus.",
                rewards.fuel_efficiency_points(new_state),
            )
            self._emit(
                "LANDING_ACCURACY_BONUS", f"Landed {physics.distance_to_target(new_state):.0f} m from the target site.",
                rewards.landing_accuracy_points(new_state),
            )
            self._goto(TOUCHDOWN)
        elif outcome == "CRASH":
            self._emit("CRASH", "Hard landing — structural failure on impact.", 0)
            self.phase = CRASH
        elif outcome == "FUEL_EXHAUSTED":
            self._emit("FUEL_EXHAUSTED", "Propellant exhausted before touchdown.", 0)
            self.phase = FUEL_EXHAUSTED

    def _step_touchdown(self, dt: float) -> None:
        self.throttle = 0.0
        if self.sim_time - self._phase_start >= 2.0:
            self._award_once("ENGINE_SHUTDOWN", "ENGINE_SHUTDOWN", "Descent engine shutdown. Mission complete.")
            self._goto(MISSION_COMPLETE)

    # -- serialization -------------------------------------------------------

    def snapshot(self) -> dict:
        state = self.descent_state
        return {
            "phase": self.phase,
            "simTime": round(self.sim_time, 2),
            "paused": self.paused,
            "simSpeed": self.sim_speed,
            "vehicle": {
                "attitude": [0.0, 0.0, state.theta if state else 0.0],
                "fuelFraction": round(self.fuel_fraction, 4),
                "mass": physics.total_mass(state) if state else None,
                "throttle": round(self.throttle, 3),
            },
            "activeController": self.active_controller,
            "score": self.score,
            "fidelity": FIDELITY_MAP,
            "decisionLog": self.decision_log[-8:],
            "descent": (
                {
                    "x": round(state.x, 1),
                    "y": round(state.y, 1),
                    "vx": round(state.vx, 2),
                    "vy": round(state.vy, 2),
                    "fuelKg": round(state.fuel_mass, 1),
                }
                if state
                else None
            ),
        }
