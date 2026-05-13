import logging
from functools import cached_property
from typing import Literal

from lerobot.robots.nero_follower import NeroFollower, NeroFollowerConfig, SharedArmState
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_bi_nero_follower import BiNeroFollowerConfig

logger = logging.getLogger(__name__)


class BiNeroFollower(Robot):
    """Bimanual Nero follower. Wraps two NeroFollower instances with left_/right_ prefixes."""

    config_class = BiNeroFollowerConfig
    name = "bi_nero_follower"

    def __init__(self, config: BiNeroFollowerConfig):
        super().__init__(config)
        self.config = config

        if config.cameras:
            left_cameras = config.cameras
            right_cameras = {}
        else:
            left_cameras = config.left_arm_config.cameras
            right_cameras = config.right_arm_config.cameras

        left_cfg = NeroFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            can_channel=config.left_arm_config.can_channel,
            firmware_version=config.left_arm_config.firmware_version,
            can_interface=config.left_arm_config.can_interface,
            control_mode=config.left_arm_config.control_mode,
            use_move_js=config.left_arm_config.use_move_js,
            speed_percent=config.left_arm_config.speed_percent,
            enable_joint_limits=config.left_arm_config.enable_joint_limits,
            control_hz=config.left_arm_config.control_hz,
            gripper_force_n=config.left_arm_config.gripper_force_n,
            home_joints_rad=config.left_arm_config.home_joints_rad,
            disable_on_disconnect=config.left_arm_config.disable_on_disconnect,
            connect_timeout_s=config.left_arm_config.connect_timeout_s,
            cameras=left_cameras,
            mit_kp=config.left_arm_config.mit_kp,
            mit_kd=config.left_arm_config.mit_kd,
            mit_manual_t_ff=config.left_arm_config.mit_manual_t_ff,
            mit_gravity_factor=config.left_arm_config.mit_gravity_factor,
            mit_gravity_urdf_path=config.left_arm_config.mit_gravity_urdf_path,
        )
        right_cfg = NeroFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            can_channel=config.right_arm_config.can_channel,
            firmware_version=config.right_arm_config.firmware_version,
            can_interface=config.right_arm_config.can_interface,
            control_mode=config.right_arm_config.control_mode,
            use_move_js=config.right_arm_config.use_move_js,
            speed_percent=config.right_arm_config.speed_percent,
            enable_joint_limits=config.right_arm_config.enable_joint_limits,
            control_hz=config.right_arm_config.control_hz,
            gripper_force_n=config.right_arm_config.gripper_force_n,
            home_joints_rad=config.right_arm_config.home_joints_rad,
            disable_on_disconnect=config.right_arm_config.disable_on_disconnect,
            connect_timeout_s=config.right_arm_config.connect_timeout_s,
            cameras=right_cameras,
            mit_kp=config.right_arm_config.mit_kp,
            mit_kd=config.right_arm_config.mit_kd,
            mit_manual_t_ff=config.right_arm_config.mit_manual_t_ff,
            mit_gravity_factor=config.right_arm_config.mit_gravity_factor,
            mit_gravity_urdf_path=config.right_arm_config.mit_gravity_urdf_path,
        )

        self.left_arm = NeroFollower(left_cfg)
        self.right_arm = NeroFollower(right_cfg)
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    # ---------------- features ----------------

    def _prefixed(self, side: str, d: dict) -> dict:
        return {f"{side}_{k}": v for k, v in d.items()}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **self._prefixed("left", self.left_arm._motors_ft),
            **self._prefixed("right", self.right_arm._motors_ft),
        }

    @property
    def _ee_ft(self) -> dict[str, type]:
        return {
            **self._prefixed("left", self.left_arm._ee_ft),
            **self._prefixed("right", self.right_arm._ee_ft),
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {**self.left_arm._cameras_ft, **self.right_arm._cameras_ft}

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    # ---------------- lifecycle ----------------

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    # ---------------- observation / action ----------------

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict = {}
        left_cam_keys = set(self.left_arm.cameras.keys())
        right_cam_keys = set(self.right_arm.cameras.keys())

        for key, value in self.left_arm.get_observation().items():
            obs[key if key in left_cam_keys else f"left_{key}"] = value
        for key, value in self.right_arm.get_observation().items():
            obs[key if key in right_cam_keys else f"right_{key}"] = value
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)
        return {
            **{f"left_{key}": value for key, value in sent_left.items()},
            **{f"right_{key}": value for key, value in sent_right.items()},
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    # ---------------- hand-off API for the teleop ----------------

    def get_shared_state(self, side: Literal["left", "right"]) -> SharedArmState:
        return self.left_arm.shared_state if side == "left" else self.right_arm.shared_state

    def get_arm(self, side: Literal["left", "right"]) -> NeroFollower:
        return self.left_arm if side == "left" else self.right_arm
