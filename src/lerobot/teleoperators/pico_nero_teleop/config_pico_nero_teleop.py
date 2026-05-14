from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..config import TeleoperatorConfig

DEFAULT_URDF = Path(
    "/home/zenbot-robot/repos/lerobot/src/lerobot/assets/nero/urdf/nero_with_gripper_flange_description.urdf"
)


@dataclass
class PicoNeroTeleopConfigBase:
    """Base config for a single-arm Pico + Nero teleoperator."""

    side: Literal["left", "right"] = "left"

    # Placo IK setup
    urdf_path: Path = DEFAULT_URDF
    ee_link: str = "end_effector_sphere"
    joint_names: list[str] = field(
        default_factory=lambda: [f"joint{i}" for i in range(1, 8)]
    )

    # Main teleop-thread rate (XR poll + IK action-cache updates).
    ik_hz: int = 90
    # Internal placo integration step. Keep this independent from ik_hz so the
    # per-iteration joint step size and convergence test are not coupled to the
    # outer teleop loop frequency.
    solver_dt: float = 0.1

    # Controller -> EE delta mapping.
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    # Per-tick cartesian target clamp applied before IK. This keeps large
    # controller jumps from turning into a single oversized IK request.
    max_target_step_m: float = 0.03
    max_target_step_rad: float = 0.2
    # Number of extra 90-degree rotations about the robot Z axis to apply after
    # the fixed raw-XR -> robot-base axis remapping.
    xr_yaw_quadrants: int = 0

    # Deadman activation on the grip button.
    activation_grip_threshold: float = 0.9
    release_grip_threshold: float = 0.5
    trigger_deadzone: float = 0.02
    # Scale trigger -> gripper opening range. 1.0 keeps the calibrated full
    # open/close span; 0.5 limits trigger-released opening to half span.
    trigger_gripper_scale: float = 0.5

    # Home / zero behavior. home_button: "Y" (left arm by convention) or "B" (right).
    home_button: str = "Y"
    home_joints_rad: list[float] = field(
        default_factory=lambda: [0.0, 0.35, 0.0, 1.75, 0.0, 0.0, -0.6]
    )
    home_speed_rad_s: float = 0.4

    # Placo task weights.
    frame_position_weight: float = 1.0
    frame_orientation_weight: float = 1.0
    posture_weight: float = 1e-6
    manipulability_weight: float = 1e-6

    # Deprecated: smoothing now belongs in the external control loop so the
    # sent action can be recorded and reused by policy/replay paths.
    smoother_alpha: float = 1.0  # 1.0 disables EMA
    max_qdot: list[float] | None = None  # rad/s per-joint; None disables clamp

    # IK-failure handling: how many consecutive no-solution frames before we
    # re-anchor reference frames on the next successful solve, to prevent a
    # large jump after the controller re-enters the reachable workspace.
    reacquire_frames: int = 10


@TeleoperatorConfig.register_subclass("pico_nero_teleop")
@dataclass(kw_only=True)
class PicoNeroTeleopConfig(TeleoperatorConfig, PicoNeroTeleopConfigBase):
    pass
