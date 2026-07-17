#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import threading
import time
from io import StringIO

import pytest

from lerobot.utils.terminal_status import (
    TerminalStatusDisplay,
    _format_consumed_event,
    _format_deploy_status_lines,
)


ESC = "\x1b["
STATUS_LINES = (
    "[STATE]    running      [EVENT] right/start @ 14:32:10",
    "[LATENCY]  total=104.2ms delay=4f queue=31 phase=between_inferences",
    "[TIMING]   build=0.3 prep=1.2 preprocess=2.4 predict=98.7 post=0.6 merge=0.2ms",
    "[SUBTASK]  Subtask: Pick up the fork.; Progress: 0.4",
    "[MEMORY]   Subtask: Pick up the fork.; Progress: 0.2",
    "[TIME]     idx=4 running raw=37.2s input=37.2s cap=48.9s subtask=Stir the beaten eggs.",
)


class FakeStream(StringIO):
    def __init__(self, *, is_tty: bool) -> None:
        super().__init__()
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ConcurrentDetectingStream(FakeStream):
    def __init__(self) -> None:
        super().__init__(is_tty=True)
        self._write_guard = threading.Lock()
        self.concurrent_writes = 0

    def write(self, text: str) -> int:
        if not self._write_guard.acquire(blocking=False):
            self.concurrent_writes += 1
            with self._write_guard:
                return super().write(text)
        try:
            time.sleep(0.0001)
            return super().write(text)
        finally:
            self._write_guard.release()


def _make_logger(stream: FakeStream, formatter: logging.Formatter | None = None):
    logger = logging.Logger(f"terminal-status-test-{id(stream)}", level=logging.DEBUG)
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter or logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler


def _make_display(
    stream: FakeStream,
    *,
    mode: str = "live",
    clock=None,
    formatter: logging.Formatter | None = None,
):
    logger, original_handler = _make_logger(stream, formatter)
    display = TerminalStatusDisplay(
        logger=logger,
        mode=mode,
        refresh_hz=4.0,
        environ={"TERM": "xterm-256color"},
        clock=clock,
    )
    display.start()
    return display, logger, original_handler


def test_live_updates_reuse_the_same_six_terminal_rows():
    stream = FakeStream(is_tty=True)
    display, _, _ = _make_display(stream)

    assert display.update(STATUS_LINES, force=True)
    first_render = stream.getvalue()
    assert first_render.count("\n") == 5
    labels = ("[STATE]", "[LATENCY]", "[TIMING]", "[SUBTASK]", "[MEMORY]", "[TIME]")
    assert all(label in first_render for label in labels)

    before = len(first_render)
    assert display.update(tuple(line.replace("running", "paused") for line in STATUS_LINES), force=True)
    redraw = stream.getvalue()[before:]
    assert redraw.count("\x1b[1A") == 5
    assert redraw.count("\n") == 5
    assert "[STATE]    paused" in redraw

    display.stop()


def test_regular_and_multiline_logs_clear_then_redraw_footer():
    stream = FakeStream(is_tty=True)
    display, logger, _ = _make_display(stream)
    display.update(STATUS_LINES, force=True)
    before = len(stream.getvalue())

    logger.info("Camera connected\nHoming complete")

    output = stream.getvalue()[before:]
    assert output.count("\x1b[1A") == 5
    assert "INFO Camera connected\nHoming complete\n" in output
    assert output.rfind("[TIME]") > output.find("Homing complete")
    display.stop()


def test_logger_exception_keeps_multiline_traceback_with_footer():
    class MessageOnlyFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return f"{record.levelname} {record.getMessage()}"

    stream = FakeStream(is_tty=True)
    display, logger, _ = _make_display(stream, formatter=MessageOnlyFormatter())
    display.update(STATUS_LINES, force=True)

    try:
        raise RuntimeError("dashboard boom")
    except RuntimeError:
        logger.exception("Fatal engine error")

    output = stream.getvalue()
    assert "Traceback (most recent call last):" in output
    assert "RuntimeError: dashboard boom" in output
    assert output.rfind("[TIME]") > output.find("RuntimeError: dashboard boom")
    display.stop()


@pytest.mark.parametrize(
    ("is_tty", "term", "expected"),
    [(True, "xterm-256color", "live"), (False, "xterm-256color", "plain"), (True, "dumb", "plain")],
)
def test_auto_mode_is_tty_and_term_aware(is_tty, term, expected):
    stream = FakeStream(is_tty=is_tty)
    logger, _ = _make_logger(stream)
    display = TerminalStatusDisplay(
        logger=logger,
        mode="auto",
        refresh_hz=4.0,
        environ={"TERM": term},
    )
    display.start()
    assert display.resolved_mode == expected
    display.update(STATUS_LINES, force=True)
    if expected == "plain":
        assert ESC not in stream.getvalue()
    display.stop()


def test_plain_mode_is_one_hz_compact_and_has_no_ansi():
    clock = FakeClock()
    stream = FakeStream(is_tty=False)
    display, _, _ = _make_display(stream, mode="plain", clock=clock)

    assert display.update(STATUS_LINES)
    assert not display.update(STATUS_LINES)
    clock.advance(0.99)
    assert not display.update(STATUS_LINES)
    clock.advance(0.01)
    assert display.update(STATUS_LINES)

    output = stream.getvalue()
    assert output.count("[STATUS]") == 2
    assert all(label in output for label in ("[STATE]", "[LATENCY]", "[SUBTASK]", "[MEMORY]", "[TIME]"))
    assert ESC not in output
    display.stop()


def test_live_output_is_width_limited_without_losing_labels(monkeypatch):
    stream = FakeStream(is_tty=True)
    display, _, _ = _make_display(stream)
    monkeypatch.setattr(display, "_terminal_width", lambda: 42)

    display.update(STATUS_LINES, force=True)

    visible_lines = [part.rsplit("\x1b[2K", 1)[-1] for part in stream.getvalue().splitlines()]
    assert all(len(line) <= 42 for line in visible_lines)
    assert "[TIME]" in visible_lines[-1]
    display.stop()


def test_logging_and_updates_share_one_write_lock_without_deadlock():
    stream = ConcurrentDetectingStream()
    display, logger, _ = _make_display(stream)
    display.update(STATUS_LINES, force=True)

    def log_worker() -> None:
        for index in range(25):
            logger.warning("camera-log-%d", index)

    def update_worker() -> None:
        for index in range(25):
            lines = (*STATUS_LINES[:-2], f"[MEMORY]   memory-{index}", STATUS_LINES[-1])
            display.update(lines, force=True)

    threads = [threading.Thread(target=log_worker) for _ in range(2)] + [
        threading.Thread(target=update_worker) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert all(not thread.is_alive() for thread in threads)
    assert stream.concurrent_writes == 0
    assert "camera-log-24" in stream.getvalue()
    display.stop()


def test_stop_restores_original_handler_cursor_and_newline():
    stream = FakeStream(is_tty=True)
    display, logger, original_handler = _make_display(stream)
    display.update(STATUS_LINES, force=True)
    assert original_handler not in logger.handlers

    display.stop()

    assert original_handler in logger.handlers
    assert display.handler not in logger.handlers
    assert "\x1b[?25h\n" in stream.getvalue()


def test_live_ansi_never_reaches_file_handler(tmp_path):
    stream = FakeStream(is_tty=True)
    logger, _ = _make_logger(stream)
    log_path = tmp_path / "deploy.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    display = TerminalStatusDisplay(
        logger=logger,
        mode="live",
        refresh_hz=4.0,
        environ={"TERM": "xterm"},
    )
    display.start()

    display.update(STATUS_LINES, force=True)
    logger.warning("slow-loop warning")
    display.stop()
    file_handler.close()

    file_output = log_path.read_text()
    assert "slow-loop warning" in file_output
    assert ESC not in file_output


def test_consumed_event_uses_latest_queue_value_and_consumption_time(monkeypatch):
    monkeypatch.setattr(time, "strftime", lambda *_args, **_kwargs: "14:32:10")
    assert _format_consumed_event("right") == "right/start @ 14:32:10"
    assert _format_consumed_event("space") == "space/pause @ 14:32:10"
    assert _format_consumed_event("h") == "h/home @ 14:32:10"
    assert _format_consumed_event("esc") == "esc/exit @ 14:32:10"


def test_deploy_status_lines_use_committed_rtc_snapshot_values():
    lines = _format_deploy_status_lines(
        state="running",
        event_text="right/start @ 14:32:10",
        rtc_debug={
            "last_latency_s": 0.1042,
            "last_real_delay": 4,
            "queue_size": 31,
            "current_phase": "between_inferences",
            "last_timing_ms": {
                "build_frame": 0.3,
                "prepare_obs": 1.2,
                "preprocess": 2.4,
                "predict": 98.7,
                "postprocess": 0.6,
                "merge": 0.2,
            },
            "last_subtask_output_text": "Subtask: Pick up the fork.; Progress: 0.4",
            "last_memory_input_text": "Subtask: Pick up the fork.; Progress: 0.2",
            "subtask_time_enabled": True,
            "subtask_time_valid": True,
            "subtask_time_current_index": 4,
            "subtask_time_current_name": "Stir the beaten eggs.",
            "subtask_time_raw_elapsed_seconds": 37.2,
            "subtask_time_effective_seconds": 37.2,
            "subtask_time_cap_seconds": 48.9,
            "subtask_time_running": True,
            "subtask_time_paused": False,
            "subtask_time_last_input_seconds": 37.2,
        },
    )

    assert lines == STATUS_LINES


def test_deploy_status_lines_show_none_after_reset():
    lines = _format_deploy_status_lines(state="paused", event_text="space/pause @ 14:32:10", rtc_debug={})
    assert lines[3] == "[SUBTASK]  <none>"
    assert lines[4] == "[MEMORY]   <none>"
    assert lines[5] == "[TIME]     disabled"


@pytest.mark.parametrize(
    ("rtc_debug", "expected"),
    [
        (
            {"subtask_time_enabled": True, "subtask_time_valid": False},
            "[TIME]     waiting-for-first-subtask",
        ),
        (
            {
                "subtask_time_enabled": True,
                "subtask_time_valid": True,
                "subtask_time_current_index": 2,
                "subtask_time_current_name": "Third.",
                "subtask_time_raw_elapsed_seconds": 8.0,
                "subtask_time_effective_seconds": 8.0,
                "subtask_time_cap_seconds": 12.0,
                "subtask_time_running": False,
                "subtask_time_paused": True,
                "subtask_time_last_input_seconds": 7.5,
            },
            "[TIME]     idx=2 paused raw=8.0s input=7.5s cap=12.0s subtask=Third.",
        ),
        (
            {
                "subtask_time_enabled": True,
                "subtask_time_valid": True,
                "subtask_time_current_index": 2,
                "subtask_time_current_name": "Third.",
                "subtask_time_raw_elapsed_seconds": 20.0,
                "subtask_time_effective_seconds": 12.0,
                "subtask_time_cap_seconds": 12.0,
                "subtask_time_running": True,
                "subtask_time_paused": False,
                "subtask_time_last_input_seconds": 12.0,
            },
            "[TIME]     idx=2 running raw=20.0s input=12.0s cap=12.0s subtask=Third.",
        ),
    ],
)
def test_time_status_waiting_paused_and_capped_views(rtc_debug, expected):
    lines = _format_deploy_status_lines(state="paused", event_text="<none>", rtc_debug=rtc_debug)
    assert lines[-1] == expected


@pytest.mark.parametrize(("mode", "refresh_hz"), [("curses", 4.0), ("auto", 0.0), ("live", -1.0)])
def test_terminal_status_rejects_invalid_config(mode, refresh_hz):
    stream = FakeStream(is_tty=True)
    logger, _ = _make_logger(stream)
    with pytest.raises(ValueError):
        TerminalStatusDisplay(logger=logger, mode=mode, refresh_hz=refresh_hz)
