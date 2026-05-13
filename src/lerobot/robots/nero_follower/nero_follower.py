import logging
import math
import threading
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_nero_follower import NeroFollowerConfig
from .nero_arm_driver import NeroArmDriver
from .shared_state import SharedArmState

logger = logging.getLogger(__name__)

JOINT_KEYS = [f"joint_{i}.pos" for i in range(1, 8)]
GRIPPER_KEY = "gripper.pos"
EE_KEYS = ["ee_x", "ee_y", "ee_z", "ee_roll", "ee_pitch", "ee_yaw"]


class NeroFollower(Robot):
    """Single Nero arm + AGX gripper exposed as a lerobot Robot.

    The exec thread runs at config.control_hz and forwards the latest
    SharedArmState target to pyAgxArm. send_action() updates that target so
    teleop, policy, replay, and deployment all use the same action entrypoint.
    """

    config_class = NeroFollowerConfig
    name = "nero_follower"

    def __init__(self, config: NeroFollowerConfig):
        super().__init__(config)
        self.config = config

        self.driver = NeroArmDriver(
            channel=config.can_channel,
            firmware_version=config.firmware_version,
            can_interface=config.can_interface,
            control_mode=config.control_mode,
            use_move_js=config.use_move_js,
            speed_percent=config.speed_percent,
            enable_joint_limits=config.enable_joint_limits,
            gripper_force_n=config.gripper_force_n,
            mit_kp=config.mit_kp,
            mit_kd=config.mit_kd,
            mit_manual_t_ff=config.mit_manual_t_ff,
            mit_gravity_factor=config.mit_gravity_factor,
            mit_gravity_urdf_path=config.mit_gravity_urdf_path,
        )
        self.shared_state = SharedArmState()
        self.cameras = make_cameras_from_configs(config.cameras)

        self._exec_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_sent_q: list[float] | None = None
        self._last_sent_g: float | None = None

    # ---------------- feature dicts ----------------

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {**{k: float for k in JOINT_KEYS}, GRIPPER_KEY: float}

    @property
    def _ee_ft(self) -> dict[str, type]:
        return {k: float for k in EE_KEYS}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            name: (cam.height, cam.width, 3) for name, cam in self.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    # ---------------- lifecycle ----------------

    @property
    def is_connected(self) -> bool:
        cams_ok = all(cam.is_connected for cam in self.cameras.values())
        return self.driver.is_connected and cams_ok

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.driver.connect(wait_feedback_s=self.config.connect_timeout_s)
        for cam in self.cameras.values():
            cam.connect()

        # Seed the shared target with the current measurement so exec thread
        # has something coherent to send on frame 0.
        q0 = self.driver.get_joint_angles() or list(self.config.home_joints_rad)
        g0 = self.driver.get_gripper_width() or 0.0
        self.shared_state.write_target(q0, g0, source="init")

        self._stop_event.clear()
        self._exec_thread = threading.Thread(
            target=self._run_exec_loop, name=f"nero-exec-{self.config.can_channel}", daemon=True
        )
        self._exec_thread.start()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    # ---------------- observation / action ----------------

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict = {}

        q = self.driver.get_joint_angles() or [0.0] * 7
        for i, key in enumerate(JOINT_KEYS):
            obs[key] = float(q[i])

        g = self.driver.get_gripper_width()
        obs[GRIPPER_KEY] = float(g) if g is not None else 0.0

        for name, cam in self.cameras.items():
            obs[name] = cam.async_read()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        q_rad = []
        for key in JOINT_KEYS:
            if key not in action:
                raise KeyError(f"Missing action key {key!r}")
            q_rad.append(float(action[key]))

        if GRIPPER_KEY not in action:
            raise KeyError(f"Missing action key {GRIPPER_KEY!r}")
        gripper_m = float(max(0.0, min(0.1, action[GRIPPER_KEY])))

        self.shared_state.write_target(q_rad, gripper_m, source="send_action")
        return {**{JOINT_KEYS[i]: q_rad[i] for i in range(7)}, GRIPPER_KEY: gripper_m}

    @check_if_not_connected
    def disconnect(self) -> None:
        self._stop_event.set()
        if self._exec_thread is not None:
            self._exec_thread.join(timeout=1.0)
            self._exec_thread = None
        for cam in self.cameras.values():
            cam.disconnect()
        self.driver.disconnect(disable=self.config.disable_on_disconnect)

    # ---------------- exec loop ----------------

    def _run_exec_loop(self) -> None:
        dt = 1.0 / max(1, self.config.control_hz)
        logger.info("Nero[%s] exec loop started at %dHz", self.config.can_channel, self.config.control_hz)
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                cmd = self.shared_state.read_target()
                if cmd is not None and all(math.isfinite(v) for v in cmd.q_rad):
                    send_q = self.config.control_mode in {"js", "mit"} or self._last_sent_q != cmd.q_rad
                    if send_q:
                        self.driver.move_joints(cmd.q_rad, gripper_m=cmd.gripper_m)
                        self._last_sent_q = list(cmd.q_rad)
                    if self._last_sent_g != cmd.gripper_m:
                        self.driver.move_gripper(cmd.gripper_m)
                        self._last_sent_g = cmd.gripper_m
            except Exception as e:
                logger.exception("Nero[%s] exec error: %s", self.config.can_channel, e)
                self.shared_state.set_fault(str(e))

            sleep_for = dt - (time.monotonic() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)
