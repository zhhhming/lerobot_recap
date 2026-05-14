import logging
import time
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class TeleopState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    HOMING = "homing"
    FAULT = "fault"


class TeleopStateMachine:
    """Per-arm state machine. Implements:
      * Deadman activation on grip (no toggle).
      * Edge-triggered home on a dedicated button (Y for left, B for right).
      * Home-lockout: once home starts we ignore grip until the user releases it
        at least once. This prevents the arm from snapping away the instant home
        finishes if the user is still holding grip.
      * Fault state fed by SharedArmState.
    """

    def __init__(
        self,
        side: str,
        activation_threshold: float = 0.9,
        release_threshold: float = 0.5,
        home_speed_rad_s: float = 0.4,
        home_joints_rad: list[float] | None = None,
        ik_hz: int = 90,
    ) -> None:
        self.side = side
        self.activation_threshold = float(activation_threshold)
        self.release_threshold = float(release_threshold)
        self.home_speed_rad_s = float(home_speed_rad_s)
        self.home_joints_rad = np.asarray(
            home_joints_rad or [0.0] * 7, dtype=float
        )
        self.dt = 1.0 / max(1, ik_hz)

        self.state: TeleopState = TeleopState.IDLE
        self._home_btn_prev: bool = False
        self._grip_released_since_home: bool = True

        # Home trajectory state.
        self._home_q_from: np.ndarray | None = None
        self._home_step_vec: np.ndarray | None = None
        self._home_total_steps: int = 0
        self._home_step_idx: int = 0

    # ---------------- transitions ----------------

    def step(
        self, grip: float, home_btn: bool, q_now: np.ndarray, fault_reason: str | None
    ) -> TeleopState:
        if fault_reason is not None:
            if self.state != TeleopState.FAULT:
                logger.warning("[%s] entering FAULT: %s", self.side, fault_reason)
            self.state = TeleopState.FAULT
            self._home_btn_prev = home_btn
            return self.state

        # Home button rising edge takes priority.
        home_edge = home_btn and not self._home_btn_prev
        self._home_btn_prev = home_btn
        if home_edge:
            if self.state == TeleopState.HOMING:
                self._cancel_home()
            else:
                self._begin_home(q_now)
            return self.state

        if self.state == TeleopState.HOMING:
            return self.state

        # Not homing, not faulted: drive activation purely from grip (deadman).
        if self.state == TeleopState.IDLE:
            if self._grip_released_since_home and grip > self.activation_threshold:
                logger.info("[%s] ACTIVE", self.side)
                self.state = TeleopState.ACTIVE
        elif self.state == TeleopState.ACTIVE:
            if grip < self.release_threshold:
                logger.info("[%s] IDLE", self.side)
                self.state = TeleopState.IDLE

        if grip < self.release_threshold:
            self._grip_released_since_home = True

        return self.state

    def clear_fault(self) -> None:
        if self.state == TeleopState.FAULT:
            self.state = TeleopState.IDLE
            self._grip_released_since_home = False

    # ---------------- home trajectory ----------------

    def _begin_home(self, q_now: np.ndarray) -> None:
        q_now = np.asarray(q_now, dtype=float)
        delta = self.home_joints_rad - q_now
        max_delta = float(np.max(np.abs(delta)))
        steps = max(1, int(np.ceil(max_delta / (self.home_speed_rad_s * self.dt))))
        self._home_q_from = q_now.copy()
        self._home_step_vec = delta / steps
        self._home_total_steps = steps
        self._home_step_idx = 0
        self.state = TeleopState.HOMING
        self._grip_released_since_home = False
        logger.info("[%s] HOMING (%d steps, max_delta=%.3f rad)", self.side, steps, max_delta)

    def _cancel_home(self) -> None:
        self.state = TeleopState.IDLE
        self._grip_released_since_home = False
        logger.info("[%s] home cancelled -> IDLE", self.side)

    def next_home_waypoint(self) -> np.ndarray:
        """Advance one step along the home trajectory. Returns the next joint target."""
        self._home_step_idx += 1
        if self._home_step_idx >= self._home_total_steps:
            q = self.home_joints_rad.copy()
            self.state = TeleopState.IDLE
            logger.info("[%s] home complete -> IDLE", self.side)
            return q
        return self._home_q_from + self._home_step_vec * self._home_step_idx
