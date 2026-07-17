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

"""Thread-safe terminal footer used by interactive policy deployment."""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Literal


StatusDisplayMode = Literal["auto", "live", "plain"]

_CLEAR_LINE = "\x1b[2K"
_CURSOR_UP = "\x1b[1A"
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_DEFAULT_TERMINAL_WIDTH = 120
_PLAIN_REFRESH_HZ = 1.0


def _format_consumed_event(event: str) -> str:
    labels = {
        "right": "right/start",
        "space": "space/pause",
        "h": "h/home",
        "esc": "esc/exit",
    }
    return f"{labels.get(event, event)} @ {time.strftime('%H:%M:%S')}"


def _format_deploy_status_lines(
    *,
    state: str,
    event_text: str,
    rtc_debug: Mapping[str, object],
) -> tuple[str, ...]:
    latency_s = rtc_debug.get("last_latency_s")
    latency_text = "-" if latency_s is None else f"{float(latency_s) * 1e3:.1f}ms"
    delay = rtc_debug.get("last_real_delay")
    delay_text = "-" if delay is None else f"{int(delay)}f"
    queue = rtc_debug.get("queue_size")
    queue_text = "-" if queue is None else str(queue)
    phase = rtc_debug.get("current_phase") or "-"

    timing = rtc_debug.get("last_timing_ms")
    timing = timing if isinstance(timing, Mapping) else {}

    def timing_value(key: str) -> str:
        value = timing.get(key)
        return "-" if value is None else f"{float(value):.1f}"

    subtask = str(rtc_debug.get("last_subtask_output_text") or "<none>")
    memory = str(rtc_debug.get("last_memory_input_text") or "<none>")
    if not rtc_debug.get("subtask_time_enabled", False):
        subtask_time = "[TIME]     disabled"
    elif not rtc_debug.get("subtask_time_valid", False):
        subtask_time = "[TIME]     waiting-for-first-subtask"
    else:
        index = rtc_debug.get("subtask_time_current_index")
        name = str(rtc_debug.get("subtask_time_current_name") or "<unknown>")
        raw = float(rtc_debug.get("subtask_time_raw_elapsed_seconds") or 0.0)
        cap_value = rtc_debug.get("subtask_time_cap_seconds")
        cap_text = "-" if cap_value is None else f"{float(cap_value):.1f}s"
        input_value = rtc_debug.get("subtask_time_last_input_seconds")
        input_text = "-" if input_value is None else f"{float(input_value):.1f}s"
        if rtc_debug.get("subtask_time_paused", False):
            timer_state = "paused"
        elif rtc_debug.get("subtask_time_running", False):
            timer_state = "running"
        else:
            timer_state = "stopped"
        subtask_time = (
            f"[TIME]     idx={index} {timer_state} raw={raw:.1f}s input={input_text} "
            f"cap={cap_text} subtask={name}"
        )
    return (
        f"[STATE]    {state:<13}[EVENT] {event_text}",
        f"[LATENCY]  total={latency_text} delay={delay_text} queue={queue_text} phase={phase}",
        "[TIMING]   "
        f"build={timing_value('build_frame')} prep={timing_value('prepare_obs')} "
        f"preprocess={timing_value('preprocess')} predict={timing_value('predict')} "
        f"post={timing_value('postprocess')} merge={timing_value('merge')}ms",
        f"[SUBTASK]  {subtask}",
        f"[MEMORY]   {memory}",
        subtask_time,
    )


class _StatusConsoleHandler(logging.Handler):
    def __init__(self, display: TerminalStatusDisplay, original: logging.StreamHandler) -> None:
        super().__init__(level=original.level)
        self._display = display
        self.stream = original.stream
        self.terminator = getattr(original, "terminator", "\n")
        self.setFormatter(original.formatter)
        for log_filter in original.filters:
            self.addFilter(log_filter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._display._emit_record(record)
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        self._display._flush_stream()


class TerminalStatusDisplay:
    """Temporarily wrap one console handler with a fixed live footer or plain status output."""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        mode: StatusDisplayMode | str = "auto",
        refresh_hz: float = 4.0,
        environ: Mapping[str, str] | None = None,
        clock=None,
    ) -> None:
        if mode not in ("auto", "live", "plain"):
            raise ValueError("status display mode must be one of: auto, live, plain")
        if not math.isfinite(refresh_hz) or refresh_hz <= 0:
            raise ValueError("status display refresh_hz must be > 0")

        self._logger = logger
        self._requested_mode = mode
        self._refresh_hz = float(refresh_hz)
        self._environ = os.environ if environ is None else environ
        self._clock = time.monotonic if clock is None else clock
        self._lock = RLock()
        self._original_handler = self._find_console_handler(logger)
        self._handler = _StatusConsoleHandler(self, self._original_handler)
        self._stream = self._handler.stream
        self._resolved_mode = self._resolve_mode()
        self._started = False
        self._footer_visible = False
        self._lines: tuple[str, ...] = ()
        self._last_update_s: float | None = None

    @staticmethod
    def _find_console_handler(logger: logging.Logger) -> logging.StreamHandler:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                return handler
        raise RuntimeError("TerminalStatusDisplay requires an existing console StreamHandler")

    @property
    def resolved_mode(self) -> Literal["live", "plain"]:
        return self._resolved_mode

    @property
    def handler(self) -> logging.Handler:
        return self._handler

    @property
    def refresh_interval_s(self) -> float:
        hz = _PLAIN_REFRESH_HZ if self._resolved_mode == "plain" else self._refresh_hz
        return 1.0 / hz

    def _resolve_mode(self) -> Literal["live", "plain"]:
        if self._requested_mode != "auto":
            return self._requested_mode
        is_tty = bool(getattr(self._stream, "isatty", lambda: False)())
        term = self._environ.get("TERM", "").strip().lower()
        return "live" if is_tty and term not in ("", "dumb") else "plain"

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            index = self._logger.handlers.index(self._original_handler)
            self._logger.removeHandler(self._original_handler)
            self._logger.addHandler(self._handler)
            self._logger.handlers.remove(self._handler)
            self._logger.handlers.insert(index, self._handler)
            self._started = True

    def refresh_due(self, *, force: bool = False) -> bool:
        with self._lock:
            if force or self._last_update_s is None:
                return True
            return self._clock() - self._last_update_s >= self.refresh_interval_s

    def update(self, lines: Sequence[str], *, force: bool = False) -> bool:
        with self._lock:
            now = self._clock()
            if not force and self._last_update_s is not None:
                if now - self._last_update_s < self.refresh_interval_s:
                    return False
            width = self._terminal_width()
            next_lines = tuple(self._truncate_line(line, width) for line in lines)
            if not next_lines:
                raise ValueError("terminal status requires at least one line")
            self._last_update_s = now

            if self._resolved_mode == "plain":
                self._lines = next_lines
                self._stream.write("[STATUS] " + " | ".join(self._lines) + self._handler.terminator)
            else:
                if not self._footer_visible:
                    self._stream.write(_HIDE_CURSOR)
                else:
                    self._clear_footer_locked()
                self._lines = next_lines
                self._draw_footer_locked()
            self._flush_stream_locked()
            return True

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            if self._resolved_mode == "live":
                self._stream.write(_SHOW_CURSOR + "\n")
                self._flush_stream_locked()
            index = self._logger.handlers.index(self._handler)
            self._logger.removeHandler(self._handler)
            self._logger.addHandler(self._original_handler)
            self._logger.handlers.remove(self._original_handler)
            self._logger.handlers.insert(index, self._original_handler)
            self._started = False
            self._footer_visible = False

    def __enter__(self) -> TerminalStatusDisplay:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def _emit_record(self, record: logging.LogRecord) -> None:
        message = self._handler.format(record)
        formatter = self._handler.formatter
        if record.exc_info is not None and formatter is not None:
            traceback_text = formatter.formatException(record.exc_info)
            if traceback_text and traceback_text not in message:
                message += "\n" + traceback_text
        if record.stack_info and record.stack_info not in message:
            message += "\n" + record.stack_info

        with self._lock:
            should_redraw = self._resolved_mode == "live" and self._footer_visible
            if should_redraw:
                self._clear_footer_locked()
            self._stream.write(message + self._handler.terminator)
            if should_redraw:
                self._draw_footer_locked()
            self._flush_stream_locked()

    def _terminal_width(self) -> int:
        try:
            return max(1, os.get_terminal_size(self._stream.fileno()).columns)
        except (AttributeError, OSError, ValueError):
            return max(1, shutil.get_terminal_size(fallback=(_DEFAULT_TERMINAL_WIDTH, 24)).columns)

    @staticmethod
    def _truncate_line(line: str, width: int) -> str:
        normalized = " ".join(str(line).splitlines())
        if len(normalized) <= width:
            return normalized
        if width == 1:
            return "…"
        return normalized[: width - 1] + "…"

    def _clear_footer_locked(self) -> None:
        if not self._footer_visible:
            return
        self._stream.write("\r" + _CLEAR_LINE)
        for _ in range(len(self._lines) - 1):
            self._stream.write(_CURSOR_UP + "\r" + _CLEAR_LINE)
        self._footer_visible = False

    def _draw_footer_locked(self) -> None:
        for index, line in enumerate(self._lines):
            self._stream.write("\r" + _CLEAR_LINE + line)
            if index < len(self._lines) - 1:
                self._stream.write("\n")
        self._footer_visible = True

    def _flush_stream(self) -> None:
        with self._lock:
            self._flush_stream_locked()

    def _flush_stream_locked(self) -> None:
        if hasattr(self._stream, "flush"):
            self._stream.flush()
