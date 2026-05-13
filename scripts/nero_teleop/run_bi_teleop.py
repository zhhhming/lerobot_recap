"""End-to-end bimanual teleop runner (no dataset recording).

Boots both arms, the Pico teleop, and runs a fast action loop until Ctrl-C.
Use as a quick bring-up check before wiring up lerobot_record.

Usage:
    python scripts/nero_teleop/run_bi_teleop.py
    python scripts/nero_teleop/run_bi_teleop.py --xr-yaw-quadrants 1
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from lerobot.robots.bi_nero_follower import BiNeroFollower, BiNeroFollowerConfig
from lerobot.robots.nero_follower import NeroFollowerConfigBase
from lerobot.teleoperators.bi_pico_nero_teleop import (
    BiPicoNeroTeleop,
    BiPicoNeroTeleopConfig,
)
from lerobot.teleoperators.pico_nero_teleop import PicoNeroTeleopConfigBase
from lerobot.utils.teleop_rerun import TeleopRerunMonitor

MIT_KP = [35.0, 35.0, 35.0, 40.0, 25.0, 25.0, 25.0]
MIT_KD = [0.8, 0.6, 0.8, 0.8, 0.6, 0.6, 0.6]
MIT_URDF = Path("/home/zenbot-robot/repos/lerobot/src/lerobot/assets/nero/urdf/nero_with_gripper_description.urdf")


class ActionSmoother:
    """EMA smoother for joint targets; gripper commands pass through directly."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self._prev: dict[str, float] | None = None

    def reset(self) -> None:
        self._prev = None

    def step(self, action: dict[str, float]) -> dict[str, float]:
        if self._prev is None:
            self._prev = dict(action)
            return dict(action)

        out: dict[str, float] = {}
        for key, value in action.items():
            v = float(value)
            if "_joint_" in key:
                prev = float(self._prev.get(key, v))
                out[key] = self.alpha * v + (1.0 - self.alpha) * prev
            else:
                out[key] = v

        self._prev = dict(out)
        return out


def _fmt_opt_vec(v: object, scale: float = 1.0, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    vals = [float(x) * scale for x in v]
    return "[" + ", ".join(f"{x:+.{digits}f}" for x in vals) + "]"


def _fmt_pose_xyz_rot(pose: object) -> str:
    if not isinstance(pose, dict):
        return "n/a"
    xyz = _fmt_opt_vec(pose.get("xyz"))
    rv = _fmt_opt_vec(pose.get("rotvec_deg"), digits=2)
    return f"xyz={xyz} rv_deg={rv}"


def _fmt_opt_scalar(v: object, digits: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{digits}f}"


def _trace_arm(label: str, dbg: dict[str, object]) -> str:
    return (
        f"{label}["
        f"ref_ee={_fmt_pose_xyz_rot(dbg.get('ref_ee_pose'))} "
        f"ref_ctrl={_fmt_opt_vec(dbg.get('ref_ctrl_pose'))} "
        f"ctrl={_fmt_opt_vec(dbg.get('ctrl_pose'))} "
        f"dpos_xr_mm={_fmt_opt_vec(dbg.get('delta_pos_xr_m'), scale=1000.0, digits=1)} "
        f"dpos_robot_mm={_fmt_opt_vec(dbg.get('delta_pos_robot_m'), scale=1000.0, digits=1)} "
        f"drot_xr_deg={_fmt_opt_vec(dbg.get('delta_rot_xr_deg'), digits=2)} "
        f"drot_robot_deg={_fmt_opt_vec(dbg.get('delta_rot_robot_deg'), digits=2)} "
        f"gap_mm={_fmt_opt_scalar(dbg.get('target_gap_pos_mm'), digits=1)} "
        f"gap_deg={_fmt_opt_scalar(dbg.get('target_gap_ori_deg'), digits=1)} "
        f"limited={dbg.get('target_limited')} "
        f"desired_ee={_fmt_pose_xyz_rot(dbg.get('desired_ee_pose'))} "
        f"target_ee={_fmt_pose_xyz_rot(dbg.get('target_ee_pose'))}"
        f"]"
    )


def build_configs(args: argparse.Namespace) -> tuple[BiNeroFollowerConfig, BiPicoNeroTeleopConfig]:
    robot_cfg = BiNeroFollowerConfig(
        left_arm_config=NeroFollowerConfigBase(
            can_channel="left",
            firmware_version=args.firmware,
            control_mode=args.move_mode,
            use_move_js=(args.move_mode == "js"),
            speed_percent=args.speed,
            control_hz=int(round(args.control_hz)),
            gripper_force_n=1.0,
            mit_kp=list(MIT_KP),
            mit_kd=list(MIT_KD),
            mit_gravity_factor=args.mit_gravity_factor,
            mit_gravity_urdf_path=MIT_URDF,
        ),
        right_arm_config=NeroFollowerConfigBase(
            can_channel="right",
            firmware_version=args.firmware,
            control_mode=args.move_mode,
            use_move_js=(args.move_mode == "js"),
            speed_percent=args.speed,
            control_hz=int(round(args.control_hz)),
            gripper_force_n=1.0,
            mit_kp=list(MIT_KP),
            mit_kd=list(MIT_KD),
            mit_gravity_factor=args.mit_gravity_factor,
            mit_gravity_urdf_path=MIT_URDF,
        ),
    )
    teleop_cfg = BiPicoNeroTeleopConfig(
        left_teleop_config=PicoNeroTeleopConfigBase(
            side="left",
            home_button="Y",
            ik_hz=int(round(args.control_hz)),
            solver_dt=args.solver_dt,
            rotation_scale=args.rotation_scale,
            xr_yaw_quadrants=args.xr_yaw_quadrants,
            trigger_gripper_scale=args.trigger_gripper_scale,
        ),
        right_teleop_config=PicoNeroTeleopConfigBase(
            side="right",
            home_button="B",
            ik_hz=int(round(args.control_hz)),
            solver_dt=args.solver_dt,
            rotation_scale=args.rotation_scale,
            xr_yaw_quadrants=args.xr_yaw_quadrants,
            trigger_gripper_scale=args.trigger_gripper_scale,
        ),
    )
    return robot_cfg, teleop_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--firmware", default="auto", choices=["auto", "default", "v111"])
    ap.add_argument("--speed", type=int, default=60)
    ap.add_argument("--move-mode", choices=["j", "js", "mit"], default="mit")
    ap.add_argument("--solver-dt", type=float, default=0.1)
    ap.add_argument("--rotation-scale", type=float, default=1.0)
    ap.add_argument("--xr-yaw-quadrants", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument(
        "--trigger-gripper-scale",
        type=float,
        default=1.0,
        help="Scale trigger-to-gripper opening range: 1.0 full span, 0.5 half span.",
    )
    ap.add_argument("--smoother-alpha", type=float, default=0.2)
    ap.add_argument("--mit-gravity-factor", type=float, default=1.0)
    ap.add_argument("--control-hz", type=float, default=90.0)
    ap.add_argument("--obs-stride", type=int, default=3)
    ap.add_argument("--viz", action="store_true", help="Enable optional Rerun teleop monitor.")
    ap.add_argument("--viz-ip", type=str, default=None, help="Optional Rerun server IP.")
    ap.add_argument("--viz-port", type=int, default=None, help="Optional Rerun server port.")
    ap.add_argument("--viz-session", type=str, default="nero_bi_teleop", help="Rerun session name.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    robot_cfg, teleop_cfg = build_configs(args)

    robot = BiNeroFollower(robot_cfg)
    teleop = BiPicoNeroTeleop(teleop_cfg)
    viz = TeleopRerunMonitor(
        enabled=args.viz,
        session_name=args.viz_session,
        ip=args.viz_ip,
        port=args.viz_port,
    )
    smoother = ActionSmoother(alpha=args.smoother_alpha)

    stop = {"flag": False}
    last_loop_t: float | None = None
    loop_i = 0

    def _sigint(_sig, _frm):  # noqa: ANN001
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sigint)

    robot.connect()
    teleop.attach(robot)
    teleop.connect()
    print(
        f"Teleop running (move_mode={args.move_mode}, speed={args.speed}, "
        f"rotation_scale={args.rotation_scale}, solver_dt={args.solver_dt}, "
        f"control_hz={args.control_hz}, obs_stride={args.obs_stride}, "
        f"smoother_alpha={args.smoother_alpha}, trigger_gripper_scale={args.trigger_gripper_scale}, "
        f"mit_gravity_factor={args.mit_gravity_factor}). "
        "Hold grip to activate; press Y (left) / B (right) to home. Ctrl-C to stop."
    )

    dt = 1.0 / max(0.1, args.control_hz)
    obs_stride = max(1, args.obs_stride)
    try:
        while not stop["flag"]:
            t0 = time.monotonic()
            raw_act = teleop.get_action()
            act = smoother.step(raw_act)
            robot.send_action(act)
            dbg = teleop.get_debug_snapshot()
            loop_hz = None if last_loop_t is None else 1.0 / max(1e-6, t0 - last_loop_t)
            last_loop_t = t0
            if loop_i % obs_stride == 0:
                obs = robot.get_observation()
                viz.log_frame(obs=obs, act=act, dbg=dbg, main_loop_hz=loop_hz)
                print(
                    "LEFT q=" + " ".join(f"{obs[f'left_joint_{i}.pos']:+.2f}" for i in range(1, 8)),
                    f" g={obs['left_gripper.pos']:.3f}  |  ",
                    "RIGHT q=" + " ".join(f"{obs[f'right_joint_{i}.pos']:+.2f}" for i in range(1, 8)),
                    f" g={obs['right_gripper.pos']:.3f}",
                    flush=True,
                )
                print(
                    "DEBUG"
                    f"  loop_hz={0.0 if loop_hz is None else loop_hz:.1f}"
                    f"  L[state={dbg['left']['state']} grip={dbg['left']['grip']:.2f} trig={dbg['left']['trigger']:.2f}"
                    f" note={dbg['left']['note']} ik_fail={dbg['left']['ik_fail_count']} refs={dbg['left']['refs_valid']}"
                    f" ik={dbg['left']['ik_status']} pos_mm={0.0 if dbg['left']['ik_pos_err_m'] is None else dbg['left']['ik_pos_err_m'] * 1000:.1f}"
                    f" ori_deg={0.0 if dbg['left']['ik_ori_err_rad'] is None else dbg['left']['ik_ori_err_rad'] * 180.0 / 3.141592653589793:.1f}]"
                    f"  R[state={dbg['right']['state']} grip={dbg['right']['grip']:.2f} trig={dbg['right']['trigger']:.2f}"
                    f" note={dbg['right']['note']} ik_fail={dbg['right']['ik_fail_count']} refs={dbg['right']['refs_valid']}"
                    f" ik={dbg['right']['ik_status']} pos_mm={0.0 if dbg['right']['ik_pos_err_m'] is None else dbg['right']['ik_pos_err_m'] * 1000:.1f}"
                    f" ori_deg={0.0 if dbg['right']['ik_ori_err_rad'] is None else dbg['right']['ik_ori_err_rad'] * 180.0 / 3.141592653589793:.1f}]",
                    flush=True,
                )
                if dbg["left"]["refs_valid"] or dbg["right"]["refs_valid"]:
                    print(
                        "TRACE"
                        f"  {_trace_arm('L', dbg['left'])}"
                        f"  {_trace_arm('R', dbg['right'])}",
                        flush=True,
                    )
            loop_i += 1
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    finally:
        print("Shutting down ...")
        viz.close()
        teleop.disconnect()
        robot.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
