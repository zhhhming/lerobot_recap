import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """[x, y, z, w] unit quaternion -> 3x3 rotation matrix."""
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


def _rotmat_to_euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """Extract (roll, pitch, yaw) with intrinsic XYZ convention. Matches Nero's get_flange_pose."""
    sy = -R[2, 0]
    if abs(sy) < 1.0 - 1e-9:
        pitch = float(np.arcsin(sy))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = float(np.pi / 2 if sy > 0 else -np.pi / 2)
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = 0.0
    return roll, pitch, yaw


class PlacoNeroKinematics:
    """FK / IK for one Nero arm using placo, on the end_effector_sphere link.

    IK seeds from the current arm joints (passed in on every solve) so the
    solution stays near the current configuration. A joints-task regularizes
    toward the seed, and a manipulability task keeps the arm away from
    singularities without hard-constraining posture.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        ee_link: str = "end_effector_sphere",
        joint_names: list[str] | None = None,
        frame_position_weight: float = 1.0,
        frame_orientation_weight: float = 1.0,
        posture_weight: float = 1e-6,
        manipulability_weight: float = 1e-6,
        dt: float = 0.1
    ) -> None:
        import placo

        self._placo = placo
        self.urdf_path = Path(urdf_path)
        self.ee_link = ee_link
        self.joint_names = list(joint_names) if joint_names else [f"joint{i}" for i in range(1, 8)]

        # ignore_collisions skips the self-collision warning on neutral pose.
        self.robot = placo.RobotWrapper(str(self.urdf_path), int(placo.Flags.ignore_collisions))
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.dt = float(dt)
        # URDF declares a fixed world->base_link joint but pinocchio still adds a
        # free-flyer root; lock it so IK only moves the 7 arm joints.
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        # Use placo purely as an IK solver here; hardware-side motion shaping is
        # handled elsewhere, so do not cap solver progress with joint velocity limits.
        self.solver.enable_velocity_limits(False)
        self.robot.update_kinematics()

        self.frame_task = self.solver.add_frame_task(self.ee_link, np.eye(4))
        self.frame_task.configure(
            "ee_frame", "soft", float(frame_position_weight), float(frame_orientation_weight)
        )

        self.posture_task = self.solver.add_joints_task()
        self.posture_task.configure("posture", "soft", float(posture_weight))

        self.manip_task = self.solver.add_manipulability_task(
            self.ee_link, "both", float(manipulability_weight)
        )
        self.manip_task.configure("manipulability", "soft", float(manipulability_weight))

        self._last_solve_debug: dict[str, float | int | str | None] = {
            "status": "never_called",
            "iterations": 0,
            "pos_err_m": None,
            "ori_err_rad": None,
            "best_pos_err_m": None,
            "best_ori_err_rad": None,
            "q_step_max_rad": None,
        }

    # ---------------- helpers ----------------

    def _write_joints(self, q: np.ndarray) -> None:
        for name, val in zip(self.joint_names, q, strict=True):
            self.robot.set_joint(name, float(val))
        self.robot.update_kinematics()

    def _read_joints(self) -> np.ndarray:
        return np.array([self.robot.get_joint(n) for n in self.joint_names], dtype=float)

    # ---------------- public API ----------------

    def fk(self, q: np.ndarray | list[float]) -> np.ndarray:
        """Forward kinematics for the EE link. Returns a 4x4 homogeneous matrix."""
        q = np.asarray(q, dtype=float).reshape(-1)
        self._write_joints(q)
        return np.asarray(self.robot.get_T_world_frame(self.ee_link), dtype=float)

    def fk_flange_euler(self, q: np.ndarray | list[float], frame: str = "gripper_flange") -> list[float]:
        """Return [x,y,z,roll,pitch,yaw] for a given frame (defaults to flange)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        self._write_joints(q)
        T = np.asarray(self.robot.get_T_world_frame(frame), dtype=float)
        r, p, y = _rotmat_to_euler_xyz(T[:3, :3])
        return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), r, p, y]

    def get_last_solve_debug(self) -> dict[str, float | int | str | None]:
        return dict(self._last_solve_debug)

    def solve(
        self,
        T_target: np.ndarray,
        q_seed: np.ndarray | list[float],
        q_posture_ref: np.ndarray | list[float] | None = None,
        max_iters: int = 200,
        pos_tol: float = 1.5e-2,
        ori_tol: float = 1.5e-1,
        q_step_tol: float = 1e-4,
    ) -> np.ndarray | None:
        """Return a joint solution near q_seed that places the EE at T_target, or None.

        The solver is iterative; we step it up to max_iters times and treat the
        joint vector as numerically converged once the max absolute joint change
        between consecutive iterations falls below q_step_tol. After convergence,
        we accept only if the final frame error is within (pos_tol, ori_tol);
        otherwise we report no-solution.
        """
        q_seed = np.asarray(q_seed, dtype=float).reshape(-1)
        if q_posture_ref is None:
            q_posture_ref = q_seed
        else:
            q_posture_ref = np.asarray(q_posture_ref, dtype=float).reshape(-1)

        self._write_joints(q_seed)
        self.frame_task.T_world_frame = np.asarray(T_target, dtype=float)
        self.posture_task.set_joints({n: float(v) for n, v in zip(self.joint_names, q_posture_ref, strict=True)})

        best_pos_err = float("inf")
        best_ori_err = float("inf")
        last_pos_err = None
        last_ori_err = None
        last_q = self._read_joints()
        last_q_step_max = None

        for iteration in range(1, max_iters + 1):
            try:
                self.solver.solve(True)
                self.robot.update_kinematics()
            except Exception as e:  # noqa: BLE001
                logger.debug("placo solve raised: %s", e)
                self._last_solve_debug = {
                    "status": f"exception:{type(e).__name__}",
                    "iterations": iteration,
                    "pos_err_m": last_pos_err,
                    "ori_err_rad": last_ori_err,
                    "best_pos_err_m": None if best_pos_err == float("inf") else best_pos_err,
                    "best_ori_err_rad": None if best_ori_err == float("inf") else best_ori_err,
                    "q_step_max_rad": last_q_step_max,
                }
                return None

            q_now = self._read_joints()
            q_step_max = float(np.max(np.abs(q_now - last_q)))
            last_q_step_max = q_step_max

            T_now = np.asarray(self.robot.get_T_world_frame(self.ee_link), dtype=float)
            pos_err = float(np.linalg.norm(T_now[:3, 3] - T_target[:3, 3]))
            R_err = T_now[:3, :3].T @ T_target[:3, :3]
            cos_theta = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
            ori_err = float(np.arccos(cos_theta))
            last_pos_err = pos_err
            last_ori_err = ori_err
            if (pos_err < best_pos_err) or (np.isclose(pos_err, best_pos_err) and ori_err < best_ori_err):
                best_pos_err = pos_err
                best_ori_err = ori_err

            if q_step_max < q_step_tol:
                status = "ok" if (pos_err < pos_tol and ori_err < ori_tol) else "converged_large_error"
                self._last_solve_debug = {
                    "status": status,
                    "iterations": iteration,
                    "pos_err_m": pos_err,
                    "ori_err_rad": ori_err,
                    "best_pos_err_m": best_pos_err,
                    "best_ori_err_rad": best_ori_err,
                    "q_step_max_rad": q_step_max,
                }
                return q_now if status == "ok" else None

            last_q = q_now

        self._last_solve_debug = {
            "status": "max_iters_no_converge",
            "iterations": max_iters,
            "pos_err_m": last_pos_err,
            "ori_err_rad": last_ori_err,
            "best_pos_err_m": None if best_pos_err == float("inf") else best_pos_err,
            "best_ori_err_rad": None if best_ori_err == float("inf") else best_ori_err,
            "q_step_max_rad": last_q_step_max,
        }
        return None
