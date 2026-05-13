import logging
from functools import cached_property

from lerobot.robots.bi_nero_follower import BiNeroFollower
from lerobot.teleoperators.pico_nero_teleop import (
    PicoNeroTeleop,
    PicoNeroTeleopConfig,
)
from lerobot.teleoperators.pico_nero_teleop.xr_client_wrapper import XrClient
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_bi_pico_nero_teleop import BiPicoNeroTeleopConfig

logger = logging.getLogger(__name__)


class BiPicoNeroTeleop(Teleoperator):
    """Bimanual Pico teleop driving a BiNeroFollower.

    Owns a single XrClient (xrobotoolkit_sdk.init() must happen exactly once
    per process) and shares it with both per-arm teleops. Call attach(robot)
    with the matching BiNeroFollower before connect().
    """

    config_class = BiPicoNeroTeleopConfig
    name = "bi_pico_nero_teleop"

    def __init__(self, config: BiPicoNeroTeleopConfig):
        super().__init__(config)
        self.config = config

        left_cfg = PicoNeroTeleopConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            side="left",
            urdf_path=config.left_teleop_config.urdf_path,
            ee_link=config.left_teleop_config.ee_link,
            joint_names=config.left_teleop_config.joint_names,
            ik_hz=config.left_teleop_config.ik_hz,
            solver_dt=config.left_teleop_config.solver_dt,
            translation_scale=config.left_teleop_config.translation_scale,
            rotation_scale=config.left_teleop_config.rotation_scale,
            max_target_step_m=config.left_teleop_config.max_target_step_m,
            max_target_step_rad=config.left_teleop_config.max_target_step_rad,
            xr_yaw_quadrants=config.left_teleop_config.xr_yaw_quadrants,
            activation_grip_threshold=config.left_teleop_config.activation_grip_threshold,
            release_grip_threshold=config.left_teleop_config.release_grip_threshold,
            trigger_deadzone=config.left_teleop_config.trigger_deadzone,
            trigger_gripper_scale=config.left_teleop_config.trigger_gripper_scale,
            home_button=config.left_teleop_config.home_button,
            home_joints_rad=config.left_teleop_config.home_joints_rad,
            home_speed_rad_s=config.left_teleop_config.home_speed_rad_s,
            frame_position_weight=config.left_teleop_config.frame_position_weight,
            frame_orientation_weight=config.left_teleop_config.frame_orientation_weight,
            posture_weight=config.left_teleop_config.posture_weight,
            manipulability_weight=config.left_teleop_config.manipulability_weight,
            smoother_alpha=config.left_teleop_config.smoother_alpha,
            max_qdot=config.left_teleop_config.max_qdot,
            reacquire_frames=config.left_teleop_config.reacquire_frames,
        )
        right_cfg = PicoNeroTeleopConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            side="right",
            urdf_path=config.right_teleop_config.urdf_path,
            ee_link=config.right_teleop_config.ee_link,
            joint_names=config.right_teleop_config.joint_names,
            ik_hz=config.right_teleop_config.ik_hz,
            solver_dt=config.right_teleop_config.solver_dt,
            translation_scale=config.right_teleop_config.translation_scale,
            rotation_scale=config.right_teleop_config.rotation_scale,
            max_target_step_m=config.right_teleop_config.max_target_step_m,
            max_target_step_rad=config.right_teleop_config.max_target_step_rad,
            xr_yaw_quadrants=config.right_teleop_config.xr_yaw_quadrants,
            activation_grip_threshold=config.right_teleop_config.activation_grip_threshold,
            release_grip_threshold=config.right_teleop_config.release_grip_threshold,
            trigger_deadzone=config.right_teleop_config.trigger_deadzone,
            trigger_gripper_scale=config.right_teleop_config.trigger_gripper_scale,
            home_button=config.right_teleop_config.home_button,
            home_joints_rad=config.right_teleop_config.home_joints_rad,
            home_speed_rad_s=config.right_teleop_config.home_speed_rad_s,
            frame_position_weight=config.right_teleop_config.frame_position_weight,
            frame_orientation_weight=config.right_teleop_config.frame_orientation_weight,
            posture_weight=config.right_teleop_config.posture_weight,
            manipulability_weight=config.right_teleop_config.manipulability_weight,
            smoother_alpha=config.right_teleop_config.smoother_alpha,
            max_qdot=config.right_teleop_config.max_qdot,
            reacquire_frames=config.right_teleop_config.reacquire_frames,
        )

        self.left = PicoNeroTeleop(left_cfg)
        self.right = PicoNeroTeleop(right_cfg)

        self._xr: XrClient | None = None
        self._attached: bool = False

    # ---------------- wiring ----------------

    def attach(self, robot: BiNeroFollower) -> None:
        self._xr = XrClient()
        self.left.attach(robot.left_arm, xr_client=self._xr)
        self.right.attach(robot.right_arm, xr_client=self._xr)
        self._attached = True

    # ---------------- Teleoperator API ----------------

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left.action_features.items()},
            **{f"right_{k}": v for k, v in self.right.action_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left.is_connected and self.right.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if not self._attached:
            raise RuntimeError("BiPicoNeroTeleop.attach(robot) must be called before connect().")
        self.left.connect(calibrate)
        self.right.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left.is_calibrated and self.right.is_calibrated

    def calibrate(self) -> None:
        self.left.calibrate()
        self.right.calibrate()

    def configure(self) -> None:
        self.left.configure()
        self.right.configure()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action: dict[str, float] = {}
        action.update({f"left_{k}": v for k, v in self.left.get_action().items()})
        action.update({f"right_{k}": v for k, v in self.right.get_action().items()})
        return action

    def get_debug_snapshot(self) -> dict[str, dict[str, object]]:
        return {
            "left": self.left.get_debug_snapshot(),
            "right": self.right.get_debug_snapshot(),
        }

    def prime_action(self, action: RobotAction, *, lock_reference: bool = True) -> bool:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }
        left_ok = self.left.prime_action(left_action, lock_reference=lock_reference)
        right_ok = self.right.prime_action(right_action, lock_reference=lock_reference)
        return left_ok and right_ok

    def release_primed_action(self) -> None:
        self.left.release_primed_action()
        self.right.release_primed_action()

    def send_feedback(self, feedback: dict) -> None:
        return

    def disconnect(self) -> None:
        # Stop both per-arm threads first; neither of them owns the XrClient,
        # so they won't try to close it.
        if self.left.is_connected:
            self.left.disconnect()
        if self.right.is_connected:
            self.right.disconnect()
        if self._xr is not None:
            try:
                self._xr.close()
            except Exception as e:  # nosec B110
                logger.warning("XrClient.close() raised: %s", e)
            self._xr = None
