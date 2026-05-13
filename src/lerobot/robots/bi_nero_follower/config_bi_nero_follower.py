from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig
from lerobot.robots.nero_follower import NeroFollowerConfigBase

from ..config import RobotConfig


def nero_left_cameras_config() -> dict[str, CameraConfig]:
    return {
        "left_wrist": OrbbecCameraConfig(
            serial_number_or_name="CP2AB530007Z",
            width=640,
            height=480,
            fps=30,
        ),
        "third_person": OrbbecCameraConfig(
            serial_number_or_name="CP2R553000EP",
            width=640,
            height=480,
            fps=30,
        ),
    }


def nero_right_cameras_config() -> dict[str, CameraConfig]:
    return {
        "right_wrist": OrbbecCameraConfig(
            serial_number_or_name="CP2R553000NZ",
            width=640,
            height=480,
            fps=30,
        ),
    }


@RobotConfig.register_subclass("bi_nero_follower")
@dataclass(kw_only=True)
class BiNeroFollowerConfig(RobotConfig):
    id: str | None = "bi_nero_follower"

    left_arm_config: NeroFollowerConfigBase = field(
        default_factory=lambda: NeroFollowerConfigBase(
            can_channel="left",
            cameras=nero_left_cameras_config(),
        )
    )
    right_arm_config: NeroFollowerConfigBase = field(
        default_factory=lambda: NeroFollowerConfigBase(
            can_channel="right",
            cameras=nero_right_cameras_config(),
        )
    )

    # Top-level cameras shared across both arms (per lerobot bimanual convention).
    # When non-empty, they're assigned to the left arm and per-arm cameras are ignored.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
