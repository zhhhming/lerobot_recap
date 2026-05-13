import logging
import os
import time
from typing import Any

import numpy as np

from lerobot.types import RobotAction, RobotObservation

logger = logging.getLogger(__name__)


def _state_to_code(state: object) -> int:
    mapping = {
        "idle": 0,
        "active": 1,
        "homing": 2,
        "fault": 3,
    }
    return mapping.get(str(state).lower(), -1)


def _ik_status_to_code(status: object) -> int:
    s = str(status).lower()
    if s == "ok":
        return 0
    if s in {"never_called", "no_kin"}:
        return 1
    if s == "converged_large_error":
        return 2
    if s == "max_iters_no_converge":
        return 3
    if s.startswith("exception"):
        return 4
    return -1


def _text_badge(text: object) -> str:
    return f"**`{str(text).upper()}`**"


def _safe_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_vec3(v: object) -> list[float] | None:
    if v is None:
        return None
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _fmt_or_na(value: object, *, scale: float = 1.0, digits: int = 2) -> str:
    val = _safe_float(value)
    if val is None:
        return "n/a"
    return f"{val * scale:.{digits}f}"


def _fmt_vec_or_na(values: object, *, digits: int = 4) -> str:
    if values is None:
        return "n/a"
    vals = [round(float(v), digits) for v in values]
    return "`[" + ", ".join(f"{v:.{digits}f}" for v in vals) + "]`"


class TeleopRerunMonitor:
    """Optional Rerun monitor for Nero teleop.

    This is intentionally thin and side-effect free for the control path:
      * if `enabled=False`, every method becomes a no-op
      * if rerun is unavailable in the current environment, it auto-disables
      * the caller only needs `log_frame(...)` inside its existing loop
    """

    def __init__(
        self,
        *,
        enabled: bool,
        session_name: str = "nero_teleop",
        ip: str | None = None,
        port: int | None = None,
        use_blueprint: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.session_name = session_name
        self.ip = ip
        self.port = port
        self.use_blueprint = bool(use_blueprint)

        self._rr: Any | None = None
        self._rrb: Any | None = None
        self._active = False
        self._t0 = time.monotonic()
        self._step = 0
        self._last_status_line: dict[str, str | None] = {"left": None, "right": None}

        if self.enabled:
            self._try_init()

    @property
    def is_active(self) -> bool:
        return self._active

    def _try_init(self) -> None:
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError:
            logger.warning("rerun is not available in this environment; teleop visualization disabled")
            self.enabled = False
            return

        batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
        rr.init(self.session_name)
        memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
        if self.ip and self.port:
            rr.connect_grpc(url=f"rerun+http://{self.ip}:{self.port}/proxy")
        else:
            rr.spawn(memory_limit=memory_limit)

        self._rr = rr
        self._rrb = rrb
        self._active = True

        self._log_static_series_styles()
        if self.use_blueprint:
            self._send_default_blueprint()

    def _log_static_series_styles(self) -> None:
        assert self._rr is not None
        joint_colors = [
            [86, 180, 233],
            [230, 159, 0],
            [0, 158, 115],
            [240, 228, 66],
            [0, 114, 178],
            [213, 94, 0],
            [204, 121, 167],
        ]
        pos_colors = {
            "x": [255, 99, 71],
            "y": [60, 179, 113],
            "z": [65, 105, 225],
        }
        quat_colors = {
            "qx": [255, 99, 71],
            "qy": [60, 179, 113],
            "qz": [65, 105, 225],
            "qw": [220, 220, 220],
        }
        for side in ("left", "right"):
            for joint_idx, color in enumerate(joint_colors, start=1):
                self._rr.log(
                    f"robot/joints/{side}/joint_{joint_idx}_rad",
                    self._rr.SeriesPoints(
                        colors=color,
                        markers="circle",
                        marker_sizes=4.0,
                        names=f"joint_{joint_idx}_rad",
                    ),
                    static=True,
                )
            for axis, color in pos_colors.items():
                self._rr.log(
                    f"controller/{side}/pos/{axis}_m",
                    self._rr.SeriesPoints(
                        colors=color,
                        markers="circle",
                        marker_sizes=4.0,
                        names=f"{axis}_m",
                    ),
                    static=True,
                )
            for axis, color in quat_colors.items():
                self._rr.log(
                    f"controller/{side}/orientation/{axis}",
                    self._rr.SeriesPoints(
                        colors=color,
                        markers="circle",
                        marker_sizes=4.0,
                        names=axis,
                    ),
                    static=True,
                )

    def _send_default_blueprint(self) -> None:
        assert self._rr is not None and self._rrb is not None
        layout = self._rrb.Vertical(
            self._rrb.TextDocumentView(origin="status", name="Status"),
            self._rrb.Vertical(
                self._rrb.Horizontal(
                    self._rrb.TimeSeriesView(origin="robot/joints/left", name="Left Joints"),
                    self._rrb.TimeSeriesView(origin="robot/joints/right", name="Right Joints"),
                    column_shares=[0.5, 0.5],
                ),
                self._rrb.TimeSeriesView(origin="ik_status", name="IK Status"),
                row_shares=[0.72, 0.28],
            ),
            self._rrb.Horizontal(
                self._rrb.Vertical(
                    self._rrb.TimeSeriesView(origin="controller/left/pos", name="Left Controller Pos"),
                    self._rrb.TimeSeriesView(
                        origin="controller/left/orientation",
                        name="Left Controller Orientation",
                    ),
                    row_shares=[0.5, 0.5],
                ),
                self._rrb.Vertical(
                    self._rrb.TimeSeriesView(origin="controller/right/pos", name="Right Controller Pos"),
                    self._rrb.TimeSeriesView(
                        origin="controller/right/orientation",
                        name="Right Controller Orientation",
                    ),
                    row_shares=[0.5, 0.5],
                ),
                column_shares=[0.5, 0.5],
            ),
            row_shares=[0.34, 0.28, 0.38],
            name="Nero Teleop",
        )
        self._rr.send_blueprint(layout, make_active=True, make_default=True)

    def close(self) -> None:
        if self._rr is not None:
            try:
                self._rr.rerun_shutdown()
            except Exception as e:  # noqa: BLE001
                logger.debug("rerun shutdown raised: %s", e)
        self._active = False

    def _set_time(self) -> float:
        assert self._rr is not None
        t_rel = time.monotonic() - self._t0
        self._rr.set_time("teleop_step", sequence=self._step)
        self._rr.set_time("teleop_time", duration=t_rel)
        self._step += 1
        return t_rel

    def _log_scalar(self, path: str, value: object, *, scale: float = 1.0) -> None:
        assert self._rr is not None
        val = _safe_float(value)
        if val is None:
            return
        self._rr.log(path, self._rr.Scalars(val * scale))

    def _log_joint_series(
        self,
        *,
        side: str,
        values: dict[str, float],
    ) -> None:
        for idx in range(1, 8):
            key = f"{side}_joint_{idx}.pos"
            if key in values:
                self._log_scalar(f"robot/joints/{side}/joint_{idx}_rad", values[key])

    def _log_debug_side(self, side: str, dbg: dict[str, object]) -> None:
        assert self._rr is not None
        self._log_scalar(f"ik_status/{side}", _ik_status_to_code(dbg.get("ik_status")))

        ctrl_pose = dbg.get("ctrl_pose")
        if ctrl_pose is not None:
            ctrl_arr = np.asarray(ctrl_pose, dtype=float).reshape(-1)
            if ctrl_arr.size >= 7:
                for axis, value in zip(("x", "y", "z"), ctrl_arr[:3], strict=True):
                    self._log_scalar(f"controller/{side}/pos/{axis}_m", value)
                for axis, value in zip(("qx", "qy", "qz", "qw"), ctrl_arr[3:7], strict=True):
                    self._log_scalar(f"controller/{side}/orientation/{axis}", value)

        state = str(dbg.get("state"))
        note = str(dbg.get("note"))
        ik_status = str(dbg.get("ik_status"))
        status_line = f"{state} | {note} | ik={ik_status}"
        if self._last_status_line[side] != status_line:
            self._rr.log(f"status/{side}/events", self._rr.TextLog(status_line))
            self._last_status_line[side] = status_line

    def _build_summary(
        self,
        *,
        obs: RobotObservation | None,
        act: RobotAction | None,
        dbg: dict[str, dict[str, object]],
        main_loop_hz: float | None,
    ) -> str:
        lines = [
            "# Nero Teleop Monitor",
            "",
            f"main_loop_hz: **{'n/a' if main_loop_hz is None else f'{main_loop_hz:.1f}'}**",
            "",
            "IK status code: `ok=0`, `never/no_kin=1`, `large_error=2`, `no_converge=3`, `exception=4`",
            "",
            "| Field | LEFT | RIGHT |",
            "|---|---:|---:|",
        ]
        left = dbg.get("left", {})
        right = dbg.get("right", {})
        left_g_cur = None if obs is None else obs.get("left_gripper.pos")
        right_g_cur = None if obs is None else obs.get("right_gripper.pos")
        left_g_tgt = None if act is None else act.get("left_gripper.pos")
        right_g_tgt = None if act is None else act.get("right_gripper.pos")
        rows = [
            (
                "state",
                _text_badge(left.get("state")),
                _text_badge(right.get("state")),
            ),
            ("note", left.get("note"), right.get("note")),
            (
                "ik_status",
                _text_badge(left.get("ik_status")),
                _text_badge(right.get("ik_status")),
            ),
            ("ik_fail_count", left.get("ik_fail_count"), right.get("ik_fail_count")),
            (
                "ik_pos_err_mm",
                _fmt_or_na(left.get("ik_pos_err_m"), scale=1000.0, digits=2),
                _fmt_or_na(right.get("ik_pos_err_m"), scale=1000.0, digits=2),
            ),
            (
                "ik_ori_err_deg",
                _fmt_or_na(left.get("ik_ori_err_rad"), scale=180.0 / np.pi, digits=2),
                _fmt_or_na(right.get("ik_ori_err_rad"), scale=180.0 / np.pi, digits=2),
            ),
            ("control_hz", _fmt_or_na(left.get("loop_hz"), digits=1), _fmt_or_na(right.get("loop_hz"), digits=1)),
            (
                "tick_ms",
                _fmt_or_na(left.get("tick_dt_ms"), digits=2),
                _fmt_or_na(right.get("tick_dt_ms"), digits=2),
            ),
            ("gripper_current_m", _fmt_or_na(left_g_cur, digits=4), _fmt_or_na(right_g_cur, digits=4)),
            ("gripper_target_m", _fmt_or_na(left_g_tgt, digits=4), _fmt_or_na(right_g_tgt, digits=4)),
            ("q_now_rad", _fmt_vec_or_na(left.get("q_now")), _fmt_vec_or_na(right.get("q_now"))),
            ("q_cmd_rad", _fmt_vec_or_na(left.get("q_cmd")), _fmt_vec_or_na(right.get("q_cmd"))),
        ]
        for key, left_val, right_val in rows:
            lines.append(f"| {key} | {left_val} | {right_val} |")
        return "\n".join(lines)

    def log_frame(
        self,
        *,
        obs: RobotObservation | None,
        act: RobotAction | None,
        dbg: dict[str, dict[str, object]],
        main_loop_hz: float | None = None,
    ) -> None:
        if not self._active or self._rr is None:
            return

        self._set_time()

        if obs is not None:
            self._log_joint_series(side="left", values=obs)
            self._log_joint_series(side="right", values=obs)

        for side in ("left", "right"):
            if side in dbg:
                self._log_debug_side(side, dbg[side])

        summary = self._build_summary(obs=obs, act=act, dbg=dbg, main_loop_hz=main_loop_hz)
        self._rr.log("status/summary", self._rr.TextDocument(summary, media_type="text/markdown"))
