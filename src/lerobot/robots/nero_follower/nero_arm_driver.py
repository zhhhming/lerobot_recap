import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _firmware_version_key(raw: str) -> tuple[int, ...]:
    text = str(raw).strip()
    if not text:
        return (0,)
    parts = []
    for token in text.split("."):
        try:
            parts.append(int(token))
        except ValueError:
            digits = "".join(ch for ch in token if ch.isdigit())
            parts.append(int(digits) if digits else 0)
    return tuple(parts) if parts else (0,)


class NeroArmDriver:
    """Thin wrapper around pyAgxArm for one Nero arm + AGX gripper.

    No lerobot coupling, so this can be exercised from a standalone script.
    All motion / read APIs are one-to-one with the SDK; we only add:
      * a unified `move_joints` that dispatches move_j / move_js based on config
      * safe enable/disable on connect/disconnect
    """

    def __init__(
        self,
        channel: str,
        firmware_version: str = "auto",
        can_interface: str = "socketcan",
        control_mode: str = "j",
        use_move_js: bool = False,
        speed_percent: int = 60,
        enable_joint_limits: bool = True,
        gripper_force_n: float = 1.0,
        mit_kp: list[float] | None = None,
        mit_kd: list[float] | None = None,
        mit_manual_t_ff: list[float] | None = None,
        mit_gravity_factor: float = 1.0,
        mit_gravity_urdf_path: str | Path | None = None,
    ) -> None:
        self.channel = channel
        self.firmware_version = firmware_version
        self.can_interface = can_interface
        requested_mode = str(control_mode or "").lower()
        if requested_mode == "mit":
            self.control_mode = "mit"
        elif use_move_js:
            self.control_mode = "js"
        else:
            self.control_mode = requested_mode or "j"
        if self.control_mode not in {"j", "js", "mit"}:
            raise ValueError(f"Unsupported Nero control_mode: {control_mode!r}")
        self.use_move_js = self.control_mode == "js"
        self.speed_percent = int(max(0, min(100, speed_percent)))
        self.enable_joint_limits = enable_joint_limits
        self.gripper_force_n = float(gripper_force_n)
        self.mit_kp = [float(v) for v in (mit_kp or [35.0, 35.0, 35.0, 40.0, 25.0, 25.0, 25.0])]
        self.mit_kd = [float(v) for v in (mit_kd or [0.8, 1.1, 0.8, 0.8, 0.6, 0.6, 0.6])]
        self.mit_manual_t_ff = [float(v) for v in (mit_manual_t_ff or [0.0] * 7)]
        self.mit_gravity_factor = float(mit_gravity_factor)
        self.mit_gravity_urdf_path = None if mit_gravity_urdf_path is None else Path(mit_gravity_urdf_path)

        self._robot = None
        self._effector = None
        self._connected = False
        self._resolved_firmware = "default"
        self._last_gripper_cmd = 0.0
        self._last_mit_t_ff = [0.0] * 7
        self._mit_entered = False
        self._gravity = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, wait_feedback_s: float = 5.0) -> None:
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

        requested_fw = str(self.firmware_version).lower()
        reported_fw = None
        if requested_fw in {"auto", "", "detect"}:
            probe_cfg = create_agx_arm_config(
                robot=ArmModel.NERO,
                channel=self.channel,
                interface=self.can_interface,
            )
            probe_robot = AgxArmFactory.create_arm(probe_cfg)
            probe_robot.connect()
            deadline = time.monotonic() + max(8.0, wait_feedback_s)
            try:
                while time.monotonic() < deadline:
                    fw_msg = probe_robot.get_firmware()
                    if fw_msg is not None:
                        reported_fw = str(fw_msg.get("software_version", "")).strip()
                        if reported_fw:
                            break
                    try:
                        probe_robot.enable()
                        probe_robot.set_normal_mode()
                    except Exception:
                        pass
                    time.sleep(0.05)
            finally:
                probe_robot.disconnect()
            if not reported_fw:
                raise RuntimeError(f"Nero[{self.channel}] firmware auto-detect timed out")
            fw = NeroFW.V111 if _firmware_version_key(reported_fw) >= (1, 11) else NeroFW.DEFAULT
        else:
            fw = NeroFW.V111 if requested_fw in {"v111", "1.11"} else NeroFW.DEFAULT
            reported_fw = str(self.firmware_version)

        cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=fw,
            channel=self.channel,
            interface=self.can_interface,
            enable_joint_limits=self.enable_joint_limits,
        )
        self._robot = AgxArmFactory.create_arm(cfg)
        # Gripper must be initialized before connect() so the read thread picks up its frames.
        self._effector = self._robot.init_effector(self._robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._robot.connect()

        start = time.monotonic()
        while not self._robot.enable():
            # On Nero, normal mode is what enables CAN state push once the arm is enabled.
            self._robot.set_normal_mode()
            if time.monotonic() - start > wait_feedback_s:
                raise RuntimeError(f"Nero[{self.channel}] enable() timed out")
            time.sleep(0.02)

        # One more call after enable() succeeds keeps the arm in single-arm normal mode.
        self._robot.set_normal_mode()

        start = time.monotonic()
        while time.monotonic() - start < wait_feedback_s:
            if (
                self._robot.get_joint_angles() is not None
                and self._robot.get_arm_status() is not None
            ):
                break
            time.sleep(0.02)
        else:
            raise RuntimeError(
                f"Nero[{self.channel}] did not return state feedback within {wait_feedback_s}s"
            )

        self._robot.set_speed_percent(self.speed_percent)
        self._resolved_firmware = "v111" if fw == NeroFW.V111 else "default"
        if self.control_mode == "mit":
            from .pinocchio_gravity import DEFAULT_URDF, NeroPinocchioGravity

            urdf_path = self.mit_gravity_urdf_path or DEFAULT_URDF
            self._gravity = NeroPinocchioGravity(urdf_path)
        self._connected = True
        logger.info(
            "Nero[%s] connected (firmware_request=%s, firmware_resolved=%s, firmware_reported=%s, control_mode=%s)",
            self.channel,
            self.firmware_version,
            self._resolved_firmware,
            reported_fw,
            self.control_mode,
        )

    def disconnect(self, disable: bool = False) -> None:
        if self._robot is None:
            return
        try:
            if disable:
                # disable() drops raised joints; only call when arm is in a safe pose.
                self._robot.disable()
        except Exception as e:  # nosec B110
            logger.warning("Nero[%s] disable() failed: %s", self.channel, e)
        try:
            self._robot.disconnect()
        except Exception as e:  # nosec B110
            logger.warning("Nero[%s] disconnect() failed: %s", self.channel, e)
        self._connected = False

    # ---------------- state (non-blocking reads from SDK cache) ----------------

    def get_joint_angles(self) -> list[float] | None:
        msg = self._robot.get_joint_angles()
        return list(msg.msg) if msg is not None else None

    def get_flange_pose(self) -> list[float] | None:
        msg = self._robot.get_flange_pose()
        return list(msg.msg) if msg is not None else None

    def get_gripper_width(self) -> float | None:
        if self._effector is None:
            return None
        gs = self._effector.get_gripper_status()
        return float(gs.msg.value) if gs is not None else None

    def get_arm_status(self):
        return self._robot.get_arm_status()

    # ---------------- motion ----------------

    def _mit_t_ff_limits(self) -> list[float]:
        if self._resolved_firmware == "v111":
            return [16.0] * 7
        return [24.0, 24.0, 18.0, 18.0, 8.0, 8.0, 8.0]

    def _ensure_mit_mode(self) -> None:
        if self._mit_entered:
            return
        self._robot.set_auto_set_motion_mode_enabled(False)
        self._robot.set_motion_mode(self._robot.OPTIONS.MOTION_MODE.MIT)
        time.sleep(0.02)
        self._mit_entered = True

    def move_joints(self, q_rad: list[float], gripper_m: float | None = None) -> None:
        if self.control_mode == "js":
            self._robot.move_js(list(q_rad))
            return
        if self.control_mode == "j":
            self._robot.move_j(list(q_rad))
            return

        if self._gravity is None:
            raise RuntimeError("MIT+gravity requested but Pinocchio gravity model is not initialized")

        self._ensure_mit_mode()
        q_meas = self.get_joint_angles()
        q_use = list(q_rad if q_meas is None else q_meas)
        g_use = self.get_gripper_width()
        if g_use is None:
            g_use = self._last_gripper_cmd if gripper_m is None else float(gripper_m)
        tau_g = self._gravity.compute_arm(q_use, float(g_use)).tolist()
        t_ff_limits = self._mit_t_ff_limits()
        t_ff_total = []
        for i in range(7):
            tau = self.mit_gravity_factor * tau_g[i] + self.mit_manual_t_ff[i]
            tau = max(-t_ff_limits[i], min(t_ff_limits[i], tau))
            t_ff_total.append(tau)
        self._last_mit_t_ff = list(t_ff_total)
        for joint_index in range(1, 8):
            self._robot.move_mit(
                joint_index=joint_index,
                p_des=float(q_rad[joint_index - 1]),
                v_des=0.0,
                kp=float(self.mit_kp[joint_index - 1]),
                kd=float(self.mit_kd[joint_index - 1]),
                t_ff=float(t_ff_total[joint_index - 1]),
            )

    def move_gripper(self, width_m: float, force_n: float | None = None) -> None:
        f = self.gripper_force_n if force_n is None else float(force_n)
        # Gripper travel tested from 0.0 (closed) to 0.1 (open); clamp to be safe.
        w = float(max(0.0, min(0.1, width_m)))
        self._last_gripper_cmd = w
        self._effector.move_gripper_m(value=w, force=f)

    def emergency_stop(self) -> None:
        try:
            self._robot.electronic_emergency_stop()
        except Exception as e:
            logger.error("Nero[%s] e-stop failed: %s", self.channel, e)
