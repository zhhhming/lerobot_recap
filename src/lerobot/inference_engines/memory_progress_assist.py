#!/usr/bin/env python

"""Deployment-only memory progress assistance for the nero egg task."""

from __future__ import annotations

import math
import numbers
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from lerobot.datasets.subtask_timing import normalize_subtask_name


_SUBTASK_OUTPUT_PATTERN = re.compile(
    r"\A\s*subtask\s*:\s*(?P<name>.*?)\s*;\s*progress\s*:\s*(?P<progress>.*?)\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
_NERO_EGG_RULES = {
    normalize_subtask_name("Stir the beaten eggs."): (6, 8, 6.0),
    normalize_subtask_name("Start frying the eggs."): (7, 9, 6.0),
}


@dataclass(frozen=True)
class MemoryProgressAssistResult:
    """One committed model output and the memory text derived from it."""

    text: str
    subtask_name: str | None
    raw_progress: float | None
    effective_progress: float | None
    reason: str
    adjusted: bool
    forced: bool


def _parse_quantized_subtask_output(output: str) -> tuple[str, str, int] | None:
    match = _SUBTASK_OUTPUT_PATTERN.fullmatch(output)
    if match is None:
        return None

    name = " ".join(match.group("name").strip().split())
    normalized_name = normalize_subtask_name(name)
    if not name or not normalized_name:
        return None

    try:
        progress = Decimal(match.group("progress").strip())
    except InvalidOperation:
        return None
    if not progress.is_finite() or progress < 0 or progress > 1:
        return None

    progress_tenths = progress * 10
    if progress_tenths != progress_tenths.to_integral_value():
        return None
    return name, normalized_name, int(progress_tenths)


class NeroEggMemoryProgressAssist:
    """Advance selected quantized memory progress values after a real-time stall."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(clock):
            raise TypeError(f"clock must be callable, got {type(clock).__name__}")
        self._clock = clock
        self._last_clock_seconds: float | None = None
        self._current_subtask: str | None = None
        self._effective_tenth: int | None = None
        self._plateau_started_at: float | None = None
        self._last_result = self._pass_through("", reason="initialized")

    @property
    def last_result(self) -> MemoryProgressAssistResult:
        return self._last_result

    def _read_clock(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"monotonic clock must return a real scalar, got {value!r}")
        now = float(value)
        if not math.isfinite(now) or now < 0:
            raise ValueError(f"monotonic clock must be finite and non-negative, got {value!r}")
        if self._last_clock_seconds is not None and now < self._last_clock_seconds:
            raise RuntimeError(
                f"Monotonic clock moved backwards: {now} < {self._last_clock_seconds}"
            )
        self._last_clock_seconds = now
        return now

    @staticmethod
    def _pass_through(output: str, *, reason: str) -> MemoryProgressAssistResult:
        return MemoryProgressAssistResult(
            text=output,
            subtask_name=None,
            raw_progress=None,
            effective_progress=None,
            reason=reason,
            adjusted=False,
            forced=False,
        )

    def _clear_progress_state(self) -> None:
        self._current_subtask = None
        self._effective_tenth = None
        self._plateau_started_at = None

    def reset(self) -> None:
        self._clear_progress_state()
        self._last_result = self._pass_through("", reason="reset")

    def apply(self, output: str) -> MemoryProgressAssistResult:
        """Commit one successful model output and return the next memory text."""
        if not isinstance(output, str):
            raise TypeError(f"memory output must be a string, got {type(output).__name__}")

        parsed = _parse_quantized_subtask_output(output)
        if parsed is None:
            self._clear_progress_state()
            self._last_result = self._pass_through(output, reason="unparseable")
            return self._last_result

        name, normalized_name, raw_tenth = parsed
        rule = _NERO_EGG_RULES.get(normalized_name)
        if rule is None:
            self._clear_progress_state()
            self._last_result = self._pass_through(output, reason="unrelated_subtask")
            return self._last_result
        assist_start_tenth, terminal_tenth, stall_seconds = rule

        now = self._read_clock()
        reason = "current"
        forced = False
        previous_effective = self._effective_tenth

        if normalized_name != self._current_subtask or previous_effective is None:
            self._current_subtask = normalized_name
            effective_tenth = raw_tenth
            reason = "started"
            self._plateau_started_at = None
        elif raw_tenth > previous_effective:
            effective_tenth = raw_tenth
            reason = "natural_progress"
            self._plateau_started_at = None
        else:
            effective_tenth = previous_effective
            if raw_tenth < previous_effective:
                reason = "clamped_regression"

        assistible = assist_start_tenth <= effective_tenth < terminal_tenth
        if assistible:
            if self._plateau_started_at is None:
                self._plateau_started_at = now
                if reason == "current":
                    reason = "waiting"
            elif now - self._plateau_started_at >= stall_seconds:
                effective_tenth += 1
                forced = True
                reason = "forced_progress"
                self._plateau_started_at = now if effective_tenth < terminal_tenth else None
        else:
            self._plateau_started_at = None

        self._effective_tenth = effective_tenth
        adjusted = effective_tenth != raw_tenth
        text = (
            f"Subtask: {name}; Progress: {effective_tenth / 10:.1f}"
            if adjusted
            else output
        )
        self._last_result = MemoryProgressAssistResult(
            text=text,
            subtask_name=name,
            raw_progress=raw_tenth / 10,
            effective_progress=effective_tenth / 10,
            reason=reason,
            adjusted=adjusted,
            forced=forced,
        )
        return self._last_result
