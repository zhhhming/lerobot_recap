import logging
import threading
import time
from functools import cached_property

import numpy as np

from lerobot.robots.nero_follower import NeroFollower
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_pico_nero_teleop import PicoNeroTeleopConfig
from .placo_kinematics import PlacoNeroKinematics
from .pose_mapper import PoseMapper, _pose_debug, _rotmat_log
from .teleop_state_machine import TeleopState, TeleopStateMachine
from .xr_client_wrapper import XrClient

logger = logging.getLogger(__name__)

JOINT_KEYS = [f"joint_{i}.pos" for i in range(1, 8)]
GRIPPER_KEY = "gripper.pos"
GRIPPER_MAX_WIDTH_M = 0.1


def _trigger_to_gripper_m(trigger: float, deadzone: float, scale: float) -> float:
    """Trigger 0..1 -> gripper width max..0 m (released = open, pressed = closed)."""
    t = float(max(0.0, min(1.0, trigger)))
    if t < deadzone:
        t = 0.0
    s = float(max(0.0, min(1.0, scale)))
    return (1.0 - t) * GRIPPER_MAX_WIDTH_M * s


class PicoNeroTeleop(Teleoperator):
    """Single-arm Pico teleop driving one Nero arm.

    Runtime deps (NeroFollower + XrClient) must be supplied via attach() before
    connect(). The bimanual wrapper does this for you; for standalone use call
    attach() explicitly after instantiating both the robot and this teleop.
    """

    config_class = PicoNeroTeleopConfig
    name = "pico_nero_teleop"

    def __init__(self, config: PicoNeroTeleopConfig):
        super().__init__(config)
        self.config = config

        self._arm: NeroFollower | None = None
        self._xr: XrClient | None = None
        self._owns_xr: bool = False

        self._kin: PlacoNeroKinematics | None = None
        self._mapper = PoseMapper(
            translation_scale=config.translation_scale,
            rotation_scale=config.rotation_scale,
            xr_yaw_quadrants=config.xr_yaw_quadrants,
        )
        self._sm = TeleopStateMachine(
            side=config.side,
            activation_threshold=config.activation_grip_threshold,
            release_threshold=config.release_grip_threshold,
            home_speed_rad_s=config.home_speed_rad_s,
            home_joints_rad=config.home_joints_rad,
            ik_hz=config.ik_hz,
        )

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._state_lock = threading.RLock()
        self._action_lock = threading.RLock()
        self._latest_action: dict[str, float] | None = None
        self._latest_debug: dict[str, object] = {
            "state": self._sm.state.value,
            "grip": 0.0,
            "trigger": 0.0,
            "home_btn": False,
            "fault": None,
            "refs_valid": False,
            "ik_fail_count": 0,
            "ctrl_xyz": None,
            "q_now": None,
            "q_cmd": None,
            "ik_status": "never_called",
            "ik_pos_err_m": None,
            "ik_ori_err_rad": None,
            "ik_iters": 0,
            "ref_ctrl_pose": None,
            "ctrl_pose": None,
            "ref_ee_pose": None,
            "desired_ee_pose": None,
            "target_ee_pose": None,
            "delta_pos_xr_m": None,
            "delta_pos_robot_m": None,
            "delta_rot_xr_deg": None,
            "delta_rot_robot_deg": None,
            "target_limited": False,
            "target_gap_pos_mm": None,
            "target_gap_ori_deg": None,
            "R_adjust": None,
            "note": "init",
        }

        self._ik_fail_count = 0
        self._last_q_target: np.ndarray | None = None
        self._last_raw_q_target: np.ndarray | None = None
        self._hold_cached_action_in_idle = False
        self._last_tick_start_s: float | None = None
        self._loop_hz: float | None = None
        self._last_tick_dt_ms: float | None = None

    # ---------------- wiring ----------------

    def attach(self, arm: NeroFollower, xr_client: XrClient | None = None) -> None:
        """Wire the Nero arm and (optionally) an external XrClient.

        When xr_client is None the teleop creates its own XrClient and owns it
        (will close() on disconnect); otherwise the caller is responsible for
        lifecycle.
        """
        self._arm = arm
        if xr_client is None:
            self._xr = XrClient()
            self._owns_xr = True
        else:
            self._xr = xr_client
            self._owns_xr = False

    # ---------------- Teleoperator API ----------------

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**{k: float for k in JOINT_KEYS}, GRIPPER_KEY: float}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if self._arm is None or self._xr is None:
            raise RuntimeError("PicoNeroTeleop.attach(arm, xr_client) must be called before connect().")

        self._kin = PlacoNeroKinematics(
            urdf_path=self.config.urdf_path,
            ee_link=self.config.ee_link,
            joint_names=self.config.joint_names,
            frame_position_weight=self.config.frame_position_weight,
            frame_orientation_weight=self.config.frame_orientation_weight,
            posture_weight=self.config.posture_weight,
            manipulability_weight=self.config.manipulability_weight,
            dt=self.config.solver_dt,
        )

        q_now_list = self._arm.driver.get_joint_angles()
        if q_now_list is not None:
            q_now = np.asarray(q_now_list, dtype=float)
            g_now = self._arm.driver.get_gripper_width()
            g_hold = 0.0 if g_now is None else float(g_now)
            self._last_q_target = q_now.copy()
            self._store_raw_action(q_now)
            self._publish_action(q_now, g_hold)
            with self._action_lock:
                self._latest_debug = {
                    **self._latest_debug,
                    "state": self._sm.state.value,
                    "q_now": [float(v) for v in q_now],
                    "q_cmd": [float(v) for v in q_now],
                    "note": "connect_seed_idle",
                }

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name=f"pico-nero-{self.config.side}", daemon=True
        )
        self._thread.start()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        with self._action_lock:
            if self._latest_action is None:
                return {**{k: 0.0 for k in JOINT_KEYS}, GRIPPER_KEY: 0.0}
            return dict(self._latest_action)

    def get_debug_snapshot(self) -> dict[str, object]:
        with self._action_lock:
            return dict(self._latest_debug)

    def prime_action(self, action: RobotAction, *, lock_reference: bool = True) -> bool:
        """Seed the teleop command cache from an externally-sent robot action.

        HIL uses this at policy->correction handoff so the first correction
        command is continuous with the last command sent to the robot, not with
        the measured joint state or an old teleop cache.
        """
        missing = [key for key in JOINT_KEYS if key not in action]
        if missing:
            logger.warning(
                "Cannot prime PicoNeroTeleop[%s]; missing keys: %s",
                self.config.side,
                missing,
            )
            return False

        q_cmd = np.asarray([float(action[key]) for key in JOINT_KEYS], dtype=float)
        g_cmd = float(action.get(GRIPPER_KEY, self._current_cached_gripper()))

        with self._state_lock:
            self._store_raw_action(q_cmd)
            self._last_q_target = q_cmd.copy()
            self._hold_cached_action_in_idle = True
            self._ik_fail_count = 0

            ctrl_pose = None
            if lock_reference:
                if self._xr is not None and self._kin is not None:
                    ctrl_pose = np.asarray(self._xr.get_pose_by_name(self._controller_name()), dtype=float)
                    self._mapper.lock_refs(ctrl_pose, self._kin.fk(q_cmd))
                    note = "prime_action_locked_refs"
                else:
                    self._mapper.reset()
                    note = "prime_action_no_refs"
            else:
                self._mapper.reset()
                note = "prime_action"

            self._publish_action(q_cmd, g_cmd)
            with self._action_lock:
                self._latest_debug = {
                    **self._latest_debug,
                    "q_cmd": [float(v) for v in q_cmd],
                    "ref_ctrl_pose": None if ctrl_pose is None else [float(v) for v in ctrl_pose[:7]],
                    "ref_ee_pose": None
                    if self._mapper.ref_ee_T is None
                    else _pose_debug(self._mapper.ref_ee_T),
                    "refs_valid": self._mapper.refs_valid,
                    "ik_fail_count": self._ik_fail_count,
                    "note": note,
                }

        return True

    def release_primed_action(self) -> None:
        with self._state_lock:
            self._hold_cached_action_in_idle = False
            self._mapper.reset()

    def send_feedback(self, feedback: dict) -> None:
        return

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._owns_xr and self._xr is not None:
            try:
                self._xr.close()
            except Exception as e:  # nosec B110
                logger.warning("XrClient.close() raised: %s", e)
            self._xr = None
            self._owns_xr = False

    # ---------------- main loop ----------------

    def _controller_name(self) -> str:
        return f"{self.config.side}_controller"

    def _grip_key(self) -> str:
        return f"{self.config.side}_grip"

    def _trigger_key(self) -> str:
        return f"{self.config.side}_trigger"

    def _publish_action(self, q: np.ndarray, g_m: float) -> None:
        action: dict[str, float] = {JOINT_KEYS[i]: float(q[i]) for i in range(7)}
        action[GRIPPER_KEY] = float(g_m)
        with self._action_lock:
            self._latest_action = action

    def _publish_gripper(self, q_fallback: np.ndarray, g_m: float) -> None:
        with self._action_lock:
            if self._latest_action is None:
                self._latest_action = {JOINT_KEYS[i]: float(q_fallback[i]) for i in range(7)}
            self._latest_action[GRIPPER_KEY] = float(g_m)

    def _current_cached_gripper(self) -> float:
        with self._action_lock:
            if self._latest_action is not None and GRIPPER_KEY in self._latest_action:
                return float(self._latest_action[GRIPPER_KEY])
        return 0.0

    def _store_raw_action(self, q_raw: np.ndarray) -> None:
        self._last_raw_q_target = np.asarray(q_raw, dtype=float).copy()

    def _get_cached_raw_action(self, q_fallback: np.ndarray) -> np.ndarray:
        q_raw = q_fallback if self._last_raw_q_target is None else self._last_raw_q_target
        return np.asarray(q_raw, dtype=float).copy()

    def _update_debug(
        self,
        *,
        grip: float,
        trigger: float,
        home_btn: bool,
        fault: str | None,
        ctrl_pose: np.ndarray,
        q_now: np.ndarray,
        q_cmd: np.ndarray | None,
        desired_target_T: np.ndarray | None,
        applied_target_T: np.ndarray | None,
        target_limited: bool,
        target_gap_pos_m: float | None,
        target_gap_ori_rad: float | None,
        note: str,
    ) -> None:
        mapper_debug = self._mapper.get_debug_snapshot(ctrl_pose)
        with self._action_lock:
            self._latest_debug = {
                "state": self._sm.state.value,
                "grip": float(grip),
                "trigger": float(trigger),
                "home_btn": bool(home_btn),
                "fault": fault,
                "refs_valid": bool(self._mapper.refs_valid),
                "ik_fail_count": int(self._ik_fail_count),
                "ctrl_xyz": [float(v) for v in ctrl_pose[:3]],
                "q_now": [float(v) for v in q_now],
                "q_cmd": None if q_cmd is None else [float(v) for v in q_cmd],
                "ik_status": self._kin.get_last_solve_debug()["status"] if self._kin is not None else "no_kin",
                "ik_pos_err_m": self._kin.get_last_solve_debug()["pos_err_m"] if self._kin is not None else None,
                "ik_ori_err_rad": self._kin.get_last_solve_debug()["ori_err_rad"] if self._kin is not None else None,
                "ik_iters": self._kin.get_last_solve_debug()["iterations"] if self._kin is not None else 0,
                "ref_ctrl_pose": mapper_debug.get("ref_ctrl_pose"),
                "ctrl_pose": [float(v) for v in ctrl_pose[:7]],
                "ref_ee_pose": mapper_debug.get("ref_ee_pose"),
                "desired_ee_pose": None if desired_target_T is None else _pose_debug(desired_target_T),
                "target_ee_pose": None if applied_target_T is None else _pose_debug(applied_target_T),
                "delta_pos_xr_m": mapper_debug.get("delta_pos_xr_m"),
                "delta_pos_robot_m": mapper_debug.get("delta_pos_robot_m"),
                "delta_rot_xr_deg": mapper_debug.get("delta_rot_xr_deg"),
                "delta_rot_robot_deg": mapper_debug.get("delta_rot_robot_deg"),
                "target_limited": bool(target_limited),
                "target_gap_pos_mm": None if target_gap_pos_m is None else float(target_gap_pos_m) * 1000.0,
                "target_gap_ori_deg": None if target_gap_ori_rad is None else float(np.degrees(target_gap_ori_rad)),
                "R_adjust": mapper_debug.get("R_adjust"),
                "loop_hz": self._loop_hz,
                "tick_dt_ms": self._last_tick_dt_ms,
                "note": note,
            }

    def _run_loop(self) -> None:
        dt = 1.0 / max(1, self.config.ik_hz)
        logger.info("PicoNeroTeleop[%s] loop at %dHz", self.config.side, self.config.ik_hz)

        while not self._stop_event.is_set():
            t0 = time.monotonic()
            if self._last_tick_start_s is not None:
                period_s = t0 - self._last_tick_start_s
                if period_s > 1e-6:
                    hz_now = 1.0 / period_s
                    self._loop_hz = hz_now if self._loop_hz is None else (0.2 * hz_now + 0.8 * self._loop_hz)
            self._last_tick_start_s = t0
            try:
                self._tick()
            except Exception as e:
                logger.exception("teleop[%s] tick error: %s", self.config.side, e)
                self._arm.shared_state.set_fault(str(e))

            self._last_tick_dt_ms = (time.monotonic() - t0) * 1000.0
            sleep_for = dt - (self._last_tick_dt_ms / 1000.0)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _tick(self) -> None:
        arm = self._arm
        xr = self._xr
        assert arm is not None and xr is not None and self._kin is not None

        q_now_list = arm.driver.get_joint_angles()
        if q_now_list is None:
            # SDK not yet publishing state; hold until it is.
            return
        q_now = np.asarray(q_now_list, dtype=float)

        grip = xr.get_key_value_by_name(self._grip_key())
        trigger = xr.get_key_value_by_name(self._trigger_key())
        home_btn = xr.get_button_state_by_name(self.config.home_button)
        ctrl_pose = np.asarray(xr.get_pose_by_name(self._controller_name()), dtype=float)

        with self._state_lock:
            self._tick_command_state(q_now, grip, trigger, home_btn, ctrl_pose)

    def _tick_command_state(
        self,
        q_now: np.ndarray,
        grip: float,
        trigger: float,
        home_btn: bool,
        ctrl_pose: np.ndarray,
    ) -> None:
        arm = self._arm
        assert arm is not None and self._kin is not None

        fault = arm.shared_state.get_fault()
        prev_state = self._sm.state
        state = self._sm.step(grip, home_btn, q_now, fault)

        # Gripper tracks the trigger independently from the arm teleop state.
        g_target_m = _trigger_to_gripper_m(
            trigger,
            self.config.trigger_deadzone,
            self.config.trigger_gripper_scale,
        )
        self._publish_gripper(q_now, g_target_m)

        if state == TeleopState.FAULT:
            # Hold the last joint target; caller must clear_fault.
            self._update_debug(
                grip=grip,
                trigger=trigger,
                home_btn=home_btn,
                fault=fault,
                ctrl_pose=ctrl_pose,
                q_now=q_now,
                q_cmd=None,
                desired_target_T=None,
                applied_target_T=None,
                target_limited=False,
                target_gap_pos_m=None,
                target_gap_ori_rad=None,
                note="fault_hold",
            )
            return

        if state == TeleopState.HOMING:
            q_cmd = self._sm.next_home_waypoint()
            self._store_raw_action(q_cmd)
            self._last_q_target = q_cmd.copy()
            self._mapper.reset()
            self._ik_fail_count = 0
            self._publish_action(q_cmd, g_target_m)
            self._update_debug(
                grip=grip,
                trigger=trigger,
                home_btn=home_btn,
                fault=fault,
                ctrl_pose=ctrl_pose,
                q_now=q_now,
                q_cmd=q_cmd,
                desired_target_T=None,
                applied_target_T=None,
                target_limited=False,
                target_gap_pos_m=None,
                target_gap_ori_rad=None,
                note="homing",
            )
            return

        if state == TeleopState.IDLE:
            self._ik_fail_count = 0
            self._mapper.reset()
            if self._hold_cached_action_in_idle:
                q_cmd = self._get_cached_raw_action(q_now)
                note = "idle_hold_cached_command"
            else:
                q_cmd = q_now.copy()
                note = "idle_hold_current"
            self._store_raw_action(q_cmd)
            self._last_q_target = q_cmd.copy()
            self._publish_action(q_cmd, g_target_m)
            self._update_debug(
                grip=grip,
                trigger=trigger,
                home_btn=home_btn,
                fault=fault,
                ctrl_pose=ctrl_pose,
                q_now=q_now,
                q_cmd=q_cmd,
                desired_target_T=None,
                applied_target_T=None,
                target_limited=False,
                target_gap_pos_m=None,
                target_gap_ori_rad=None,
                note=note,
            )
            return

        # ACTIVE
        T_now = self._kin.fk(q_now)
        q_seed = self._get_cached_raw_action(q_now) if self._hold_cached_action_in_idle else q_now
        if prev_state != TeleopState.ACTIVE or not self._mapper.refs_valid:
            ref_q = self._get_cached_raw_action(q_now) if self._hold_cached_action_in_idle else q_now
            self._mapper.lock_refs(ctrl_pose, self._kin.fk(ref_q))
            self._ik_fail_count = 0

        T_desired = self._mapper.compute_target(ctrl_pose)
        if T_desired is None:
            q_raw = self._get_cached_raw_action(q_now)
            T_target = None
            target_limited = False
            target_gap_pos_m = None
            target_gap_ori_rad = None
            debug_note = "active_no_target"
        else:
            T_target = T_desired
            target_limited = False
            target_gap_pos_m = float(np.linalg.norm(T_target[:3, 3] - T_now[:3, 3]))
            rv_gap = _rotmat_log(T_target[:3, :3] @ T_now[:3, :3].T)
            target_gap_ori_rad = float(np.linalg.norm(rv_gap))
            q_sol = self._kin.solve(
                T_target,
                q_seed=q_seed,
                q_posture_ref=np.asarray(self.config.home_joints_rad, dtype=float),
            )
            if q_sol is None:
                self._ik_fail_count += 1
                q_raw = self._get_cached_raw_action(q_now)
                debug_note = "active_ik_none"
            else:
                q_raw = q_sol
                self._store_raw_action(q_raw)
                debug_note = "active_ik_ok"
                self._ik_fail_count = 0

        q_cmd = np.asarray(q_raw, dtype=float)
        self._last_q_target = q_cmd.copy()
        self._publish_action(q_cmd, g_target_m)
        self._update_debug(
            grip=grip,
            trigger=trigger,
            home_btn=home_btn,
            fault=fault,
            ctrl_pose=ctrl_pose,
            q_now=q_now,
            q_cmd=q_cmd,
            desired_target_T=T_desired,
            applied_target_T=T_target,
            target_limited=target_limited,
            target_gap_pos_m=target_gap_pos_m,
            target_gap_ori_rad=target_gap_ori_rad,
            note=debug_note,
        )
