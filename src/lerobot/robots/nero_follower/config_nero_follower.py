from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class NeroFollowerConfigBase:
    """Base configuration for a single Nero arm + AGX gripper."""

    # CAN channel name, e.g. "left" or "right" (already brought up by can_muti_activate.sh).
    can_channel: str

    # pyAgxArm firmware version string: "auto", "default" (<=1.10), or "v111" (>=1.11).
    firmware_version: str = "auto"

    # socketcan / agx_cando / slcan. On our Linux host, always "socketcan".
    can_interface: str = "socketcan"

    # Joint control mode:
    # - "j": move_j
    # - "js": move_js passthrough
    # - "mit": per-joint MIT with gravity compensation
    control_mode: Literal["j", "js", "mit"] = "mit"

    # Backward-compat shim. Prefer control_mode.
    use_move_js: bool = False

    # 0..100, feeds robot.set_speed_percent() for move_j / move_p / move_l.
    speed_percent: int = 60

    enable_joint_limits: bool = True

    # Rate of the background exec thread that forwards the latest target to pyAgxArm.
    control_hz: int = 90

    # Gripper force passed to move_gripper_m (N, range [0, 3]).
    gripper_force_n: float = 1.0

    # Home / zero joint configuration (rad). Used when the corresponding home
    # button is pressed on the teleoperator side and as the safety fallback.
    home_joints_rad: list[float] = field(
        default_factory=lambda: [0.0, 0.35, 0.0, 1.75, 0.0, 0.0, -0.6]
    )

    # Whether to call robot.disable() on disconnect.
    # Note: disable() makes raised joints DROP. Keep False unless arm is in a safe pose.
    disable_on_disconnect: bool = False

    # Seconds to wait for first joint-angle feedback before giving up on connect().
    connect_timeout_s: float = 5.0

    # Camera configurations attached to this arm (e.g. wrist cam).
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # MIT+gravity parameters. Used only when control_mode == "mit".
    mit_kp: list[float] = field(default_factory=lambda: [35.0, 35.0, 35.0, 40.0, 25.0, 25.0, 25.0])
    mit_kd: list[float] = field(default_factory=lambda: [0.8, 1.1, 0.8, 0.8, 0.6, 0.6, 0.6])
    mit_manual_t_ff: list[float] = field(default_factory=lambda: [0.0] * 7)
    mit_gravity_factor: float = 1.0
    mit_gravity_urdf_path: Path = Path(
        "/home/zenbot-robot/repos/lerobot/src/lerobot/assets/nero/urdf/nero_with_gripper_description.urdf"
    )


@RobotConfig.register_subclass("nero_follower")
@dataclass(kw_only=True)
class NeroFollowerConfig(RobotConfig, NeroFollowerConfigBase):
    # Base class requires can_channel; declare a default so kw_only works.
    can_channel: str = "left"
