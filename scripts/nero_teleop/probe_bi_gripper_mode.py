"""Probe bimanual Nero gripper feedback mode/value.

This is a diagnostics-only runner for checking whether AGX gripper feedback is
reported in width or angle mode while using the same Nero robot stack as teleop.

Examples:
    conda run -n lerobot-main python scripts/nero_teleop/probe_bi_gripper_mode.py
    conda run -n lerobot-main python scripts/nero_teleop/probe_bi_gripper_mode.py --teleop --csv /tmp/gripper_modes.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

from lerobot.robots.bi_nero_follower import BiNeroFollower
from lerobot.teleoperators.bi_pico_nero_teleop import BiPicoNeroTeleop

from run_bi_teleop import ActionSmoother, build_configs


def _read_gripper_status(arm) -> dict[str, Any]:
    effector = arm.driver._effector
    status = None if effector is None else effector.get_gripper_status()
    ctrl = None if effector is None else effector.get_gripper_ctrl_states()
    target = arm.shared_state.read_target()

    out: dict[str, Any] = {
        "mode": None,
        "value": None,
        "force": None,
        "hz": None,
        "ctrl_value": None,
        "ctrl_force": None,
        "ctrl_status_code": None,
        "target_gripper_m": None if target is None else target.gripper_m,
        "target_source": None if target is None else target.source,
        "last_sent_g": arm._last_sent_g,
        "last_gripper_cmd": arm.driver._last_gripper_cmd,
    }
    if status is not None:
        out.update(
            mode=status.msg.mode,
            value=float(status.msg.value),
            force=float(status.msg.force),
            hz=float(status.hz),
        )
    if ctrl is not None:
        out.update(
            ctrl_value=float(ctrl.msg.value),
            ctrl_force=float(ctrl.msg.force),
            ctrl_status_code=int(ctrl.msg.status_code),
        )
    return out


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_side(label: str, data: dict[str, Any], obs_value: float | None, act_value: float | None) -> str:
    return (
        f"{label}["
        f"mode={_fmt(data['mode'])} "
        f"value={_fmt(data['value'])} "
        f"obs={_fmt(obs_value)} "
        f"act={_fmt(act_value)} "
        f"target={_fmt(data['target_gripper_m'])}/{_fmt(data['target_source'])} "
        f"last_sent={_fmt(data['last_sent_g'])} "
        f"last_cmd={_fmt(data['last_gripper_cmd'])} "
        f"force={_fmt(data['force'], digits=3)} "
        f"hz={_fmt(data['hz'], digits=1)} "
        f"ctrl={_fmt(data['ctrl_value'])}/{_fmt(data['ctrl_status_code'])}"
        f"]"
    )


def _write_csv_row(writer: csv.DictWriter, t_rel: float, left: dict[str, Any], right: dict[str, Any], obs, act):
    row = {"t": t_rel}
    for side, data in (("left", left), ("right", right)):
        for key, value in data.items():
            row[f"{side}_{key}"] = value
    row["obs_left_gripper_pos"] = None if obs is None else obs.get("left_gripper.pos")
    row["obs_right_gripper_pos"] = None if obs is None else obs.get("right_gripper.pos")
    row["act_left_gripper_pos"] = None if act is None else act.get("left_gripper.pos")
    row["act_right_gripper_pos"] = None if act is None else act.get("right_gripper.pos")
    writer.writerow(row)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teleop", action="store_true", help="Run the bimanual Pico teleop action loop.")
    ap.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means until Ctrl-C.")
    ap.add_argument("--period", type=float, default=0.2, help="Print/sample period in seconds.")
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV path for sampled status rows.")

    ap.add_argument("--firmware", default="auto", choices=["auto", "default", "v111"])
    ap.add_argument("--speed", type=int, default=60)
    ap.add_argument("--move-mode", choices=["j", "js", "mit"], default="mit")
    ap.add_argument("--solver-dt", type=float, default=0.1)
    ap.add_argument("--rotation-scale", type=float, default=1.0)
    ap.add_argument("--xr-yaw-quadrants", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--trigger-gripper-scale", type=float, default=1.0)
    ap.add_argument("--smoother-alpha", type=float, default=0.2)
    ap.add_argument("--mit-gravity-factor", type=float, default=1.0)
    ap.add_argument("--control-hz", type=float, default=90.0)
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    robot_cfg, teleop_cfg = build_configs(args)
    robot = BiNeroFollower(robot_cfg)
    teleop = BiPicoNeroTeleop(teleop_cfg) if args.teleop else None
    smoother = ActionSmoother(alpha=args.smoother_alpha)

    stop = {"flag": False}

    def _sigint(_sig, _frm):  # noqa: ANN001
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sigint)

    csv_file = None
    writer = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.csv.open("w", newline="")
        fieldnames = ["t"]
        status_keys = [
            "mode",
            "value",
            "force",
            "hz",
            "ctrl_value",
            "ctrl_force",
            "ctrl_status_code",
            "target_gripper_m",
            "target_source",
            "last_sent_g",
            "last_gripper_cmd",
        ]
        for side in ("left", "right"):
            fieldnames.extend(f"{side}_{key}" for key in status_keys)
        fieldnames.extend(
            [
                "obs_left_gripper_pos",
                "obs_right_gripper_pos",
                "act_left_gripper_pos",
                "act_right_gripper_pos",
            ]
        )
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    robot.connect()
    if teleop is not None:
        teleop.attach(robot)
        teleop.connect()

    print(
        f"Probe running teleop={args.teleop} move_mode={args.move_mode} "
        f"control_hz={args.control_hz} period={args.period}. Ctrl-C to stop.",
        flush=True,
    )

    t_start = time.monotonic()
    next_sample = t_start
    dt = 1.0 / max(1.0, args.control_hz)
    try:
        while not stop["flag"]:
            now = time.monotonic()
            if args.duration > 0 and now - t_start >= args.duration:
                break

            act = None
            if teleop is not None:
                raw_act = teleop.get_action()
                act = smoother.step(raw_act)
                robot.send_action(act)

            if now >= next_sample:
                obs = robot.get_observation()
                left = _read_gripper_status(robot.left_arm)
                right = _read_gripper_status(robot.right_arm)
                t_rel = now - t_start
                print(
                    f"{t_rel:8.3f}s "
                    f"{_format_side('L', left, obs.get('left_gripper.pos'), None if act is None else act.get('left_gripper.pos'))} "
                    f"{_format_side('R', right, obs.get('right_gripper.pos'), None if act is None else act.get('right_gripper.pos'))}",
                    flush=True,
                )
                if writer is not None:
                    _write_csv_row(writer, t_rel, left, right, obs, act)
                    csv_file.flush()
                next_sample += max(0.01, args.period)

            time.sleep(max(0.0, dt - (time.monotonic() - now)))
    finally:
        print("Shutting down ...", flush=True)
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    sys.exit(main())
