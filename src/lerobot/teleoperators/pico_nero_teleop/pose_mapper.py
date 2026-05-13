import numpy as np


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n <= 0.0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def _rotmat_log(R: np.ndarray) -> np.ndarray:
    """SO(3) log map -> axis-angle 3-vector (axis * angle)."""
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        # Near 180 deg: use a numerically safer fallback.
        d = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.linalg.norm(d) < 1e-6:
            diag = np.maximum(np.diag(R), 0.0)
            axis = np.sqrt(0.5 * (diag + 1.0))
            return axis * theta
        return (theta / (2.0 * np.sin(theta))) * d
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def _rotvec_to_rotmat(rv: np.ndarray) -> np.ndarray:
    """Rodrigues: axis-angle 3-vector -> 3x3 rotation matrix."""
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _pose_debug(T: np.ndarray) -> dict[str, list[float]]:
    return {
        "xyz": [float(v) for v in T[:3, 3]],
        "rotvec_rad": [float(v) for v in _rotmat_log(T[:3, :3])],
        "rotvec_deg": [float(v) for v in np.degrees(_rotmat_log(T[:3, :3]))],
    }


def rotz_quadrant(n: int) -> np.ndarray:
    """n * 90-deg rotation about Z. n is taken mod 4."""
    n = int(n) % 4
    c = [1.0, 0.0, -1.0, 0.0][n]
    s = [0.0, 1.0, 0.0, -1.0][n]
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# Controller raw axes -> robot base axes, inferred from live probes:
#   raw x- -> robot y-  (right), i.e. raw x+ -> robot y+ (left)
#   raw y+ -> robot z+  (up)
#   raw z- -> robot x-  (backward), i.e. raw z+ -> robot x+ (forward)
RAW_TO_ROBOT = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


class PoseMapper:
    """Turns a Pico controller pose into a delta added onto the reference EE pose.

    Mirrors the reference gen3 implementation:
      * When `reset_refs` is called (entering ACTIVE or after an IK re-acquire),
        the current controller pose and current EE pose are latched.
      * On each step, (pos_ctrl - pos_ref) and (quat_ctrl * quat_ref^-1) are
        projected first through a fixed raw-XR -> robot-base axis remapping,
        then through a configurable 90-deg-quadrant rotation about the robot's
        current Z axis, and scaled before being applied onto the latched
        reference EE frame.
    """

    def __init__(
        self,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        xr_yaw_quadrants: int = 0,
    ) -> None:
        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)
        self.R_raw_to_robot = RAW_TO_ROBOT.copy()
        self.R_adjust = rotz_quadrant(xr_yaw_quadrants) @ self.R_raw_to_robot

        self._ref_ctrl_pos: np.ndarray | None = None
        self._ref_ctrl_quat: np.ndarray | None = None
        self._ref_ctrl_R: np.ndarray | None = None
        self._ref_ee_T: np.ndarray | None = None

    @property
    def refs_valid(self) -> bool:
        return self._ref_ee_T is not None

    @property
    def ref_ee_T(self) -> np.ndarray | None:
        return self._ref_ee_T

    def set_yaw_quadrants(self, n: int) -> None:
        """Rebuild the mapped-XR -> robot adjustment from a new quadrant count."""
        self.R_adjust = rotz_quadrant(n) @ self.R_raw_to_robot

    def lock_refs(self, controller_pose_xyzw: np.ndarray, ee_T: np.ndarray) -> None:
        """Latch the reference frames. Call on activation and on IK re-acquire."""
        self._ref_ctrl_pos = np.asarray(controller_pose_xyzw[:3], dtype=float).copy()
        self._ref_ctrl_quat = np.asarray(controller_pose_xyzw[3:7], dtype=float).copy()
        self._ref_ctrl_R = _quat_xyzw_to_rotmat(self._ref_ctrl_quat)
        self._ref_ee_T = np.asarray(ee_T, dtype=float).copy()

    def reset(self) -> None:
        self._ref_ctrl_pos = None
        self._ref_ctrl_quat = None
        self._ref_ctrl_R = None
        self._ref_ee_T = None

    def _compute_target_details(self, controller_pose_xyzw: np.ndarray) -> dict[str, object] | None:
        if not self.refs_valid:
            return None

        ctrl_pos = np.asarray(controller_pose_xyzw[:3], dtype=float)
        ctrl_quat = np.asarray(controller_pose_xyzw[3:7], dtype=float)
        ctrl_R = _quat_xyzw_to_rotmat(ctrl_quat)

        delta_pos_xr = (ctrl_pos - self._ref_ctrl_pos) * self.translation_scale
        delta_pos = self.R_adjust @ delta_pos_xr

        R_delta_xr = ctrl_R @ self._ref_ctrl_R.T
        rv_xr = _rotmat_log(R_delta_xr) * self.rotation_scale
        rv = self.R_adjust @ rv_xr
        R_delta = _rotvec_to_rotmat(rv)

        T_target = np.eye(4)
        T_target[:3, :3] = R_delta @ self._ref_ee_T[:3, :3]
        T_target[:3, 3] = self._ref_ee_T[:3, 3] + delta_pos

        return {
            "ref_ctrl_pose": [float(v) for v in np.concatenate([self._ref_ctrl_pos, self._ref_ctrl_quat])],
            "ctrl_pose": [float(v) for v in np.concatenate([ctrl_pos, ctrl_quat])],
            "ref_ee_pose": _pose_debug(self._ref_ee_T),
            "target_ee_pose": _pose_debug(T_target),
            "delta_pos_xr_m": [float(v) for v in delta_pos_xr],
            "delta_pos_robot_m": [float(v) for v in delta_pos],
            "delta_rot_xr_rad": [float(v) for v in rv_xr],
            "delta_rot_xr_deg": [float(v) for v in np.degrees(rv_xr)],
            "delta_rot_robot_rad": [float(v) for v in rv],
            "delta_rot_robot_deg": [float(v) for v in np.degrees(rv)],
            "target_T": T_target.tolist(),
        }

    def compute_target(self, controller_pose_xyzw: np.ndarray) -> np.ndarray | None:
        """Return the 4x4 target EE pose in robot base frame, or None if refs are unset."""
        details = self._compute_target_details(controller_pose_xyzw)
        return None if details is None else np.asarray(details["target_T"], dtype=float)

    def get_debug_snapshot(self, controller_pose_xyzw: np.ndarray | None = None) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "R_adjust": self.R_adjust.tolist(),
            "ref_ctrl_pose": None,
            "ctrl_pose": None,
            "ref_ee_pose": None,
            "target_ee_pose": None,
            "delta_pos_xr_m": None,
            "delta_pos_robot_m": None,
            "delta_rot_xr_rad": None,
            "delta_rot_xr_deg": None,
            "delta_rot_robot_rad": None,
            "delta_rot_robot_deg": None,
        }
        if not self.refs_valid:
            return snapshot

        snapshot["ref_ee_pose"] = _pose_debug(self._ref_ee_T)
        if controller_pose_xyzw is None:
            return snapshot

        details = self._compute_target_details(np.asarray(controller_pose_xyzw, dtype=float))
        if details is None:
            return snapshot
        snapshot.update(details)
        return snapshot
