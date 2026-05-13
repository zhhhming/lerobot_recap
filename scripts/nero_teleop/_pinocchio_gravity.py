"""Pinocchio-based gravity compensation helpers for Nero + gripper."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from _pinocchio_env import isolate_python_env_for_pinocchio


ENV_INFO = isolate_python_env_for_pinocchio()


def _preload_pinocchio_deps() -> None:
    """Force the current process to use the conda/cmeel eigenpy runtime."""
    cmeel_lib = Path(str(ENV_INFO["cmeel_lib"]))
    libeigenpy = cmeel_lib / "libeigenpy.so"
    if not libeigenpy.exists():
        raise FileNotFoundError(f"Required eigenpy runtime not found: {libeigenpy}")
    ctypes.CDLL(str(libeigenpy), mode=ctypes.RTLD_GLOBAL)


_preload_pinocchio_deps()

import pinocchio as pin  # noqa: E402


DEFAULT_URDF = Path(
    "/home/zenbot-robot/repos/lerobot/scripts/nero_teleop/generated/nero_with_gripper_description.urdf"
)


def build_q_full(q_arm_rad: list[float], gripper_m: float) -> np.ndarray:
    if len(q_arm_rad) != 7:
        raise ValueError(f"Expected 7 arm joints, got {len(q_arm_rad)}")
    g = float(max(0.0, min(0.1, gripper_m)))
    g_half = g / 2.0
    return np.asarray([*q_arm_rad, g_half, -g_half], dtype=float)


class NeroPinocchioGravity:
    def __init__(self, urdf_path: str | Path = DEFAULT_URDF):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(
                f"URDF not found: {self.urdf_path}. Run build_nero_urdf.sh first."
            )
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

    def compute_full(self, q_arm_rad: list[float], gripper_m: float) -> np.ndarray:
        q_full = build_q_full(q_arm_rad, gripper_m)
        return pin.computeGeneralizedGravity(self.model, self.data, q_full)

    def compute_arm(self, q_arm_rad: list[float], gripper_m: float) -> np.ndarray:
        tau_full = self.compute_full(q_arm_rad, gripper_m)
        return np.asarray(tau_full[:7], dtype=float)
