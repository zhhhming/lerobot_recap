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

"""Strict ordered-subtask state and monotonic elapsed-time tracking."""

from __future__ import annotations

import math
import numbers
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from lerobot.datasets.subtask_timing import (
    SubtaskSequenceContract,
    normalize_subtask_name,
)


_SUBTASK_OUTPUT_PATTERN = re.compile(
    r"\A\s*subtask\s*:\s*(?P<name>.*?)\s*;\s*progress\s*:\s*(?P<progress>.*)\Z",
    flags=re.IGNORECASE | re.DOTALL,
)
_SUBTASK_FIELD_PATTERN = re.compile(r"(?:\A|;)\s*subtask\s*:", flags=re.IGNORECASE)


@dataclass(frozen=True)
class SubtaskTimeTrackerSnapshot:
    """Immutable state for RTC injection, diagnostics, and status rendering."""

    current_index: int | None
    current_name: str | None
    raw_elapsed_seconds: float
    effective_elapsed_seconds: float
    cap_seconds: float | None
    time_valid: bool
    running: bool
    paused: bool
    last_transition_reason: str
    last_rejected_output: str
    last_rejection_reason: str


def parse_subtask_output_name(output: str) -> str | None:
    """Extract a clear subtask name from the model's canonical AR output."""
    if not isinstance(output, str):
        raise TypeError(f"subtask output must be a string, got {type(output).__name__}")
    match = _SUBTASK_OUTPUT_PATTERN.fullmatch(output)
    if match is None:
        return None
    if _SUBTASK_FIELD_PATTERN.search(match.group("progress")) is not None:
        return None
    name = " ".join(match.group("name").strip().split())
    if not name or not normalize_subtask_name(name):
        return None
    return name


def _finite_real(value: object, *, name: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar, got {value!r}")
    result = float(value)
    valid_bound = result > 0 if positive else result >= 0
    if not math.isfinite(result) or not valid_bound:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}, got {value!r}")
    return result


def _validate_contract(contract: SubtaskSequenceContract) -> dict[str, int]:
    if not isinstance(contract, SubtaskSequenceContract):
        raise TypeError(
            "contract must be a SubtaskSequenceContract, "
            f"got {type(contract).__name__}"
        )
    _finite_real(contract.fps, name="contract fps", positive=True)
    if not contract.ordered_subtasks:
        raise ValueError("Subtask sequence contract must contain at least one subtask")

    index_by_name: dict[str, int] = {}
    for index, stats in enumerate(contract.ordered_subtasks):
        canonical_normalized = normalize_subtask_name(stats.canonical_name)
        stored_normalized = stats.normalized_name
        if not stored_normalized or stored_normalized != canonical_normalized:
            raise ValueError(
                "Subtask sequence contract normalized_name mismatch at index "
                f"{index}: canonical={stats.canonical_name!r}, "
                f"normalized_name={stored_normalized!r}, expected={canonical_normalized!r}"
            )
        if stored_normalized in index_by_name:
            previous = index_by_name[stored_normalized]
            raise ValueError(
                "Subtask sequence contract normalization collision for "
                f"{stored_normalized!r} at indices {previous} and {index}"
            )

        maximum = _finite_real(
            stats.max_elapsed_seconds,
            name=f"max_elapsed_seconds at index {index}",
            positive=False,
        )
        cap = _finite_real(
            stats.deployment_cap_seconds,
            name=f"deployment_cap_seconds at index {index}",
            positive=False,
        )
        if cap < maximum:
            raise ValueError(
                f"deployment_cap_seconds at index {index} must be >= max_elapsed_seconds; "
                f"got {cap} < {maximum}"
            )
        index_by_name[stored_normalized] = index
    return index_by_name


class SubtaskTimeTracker:
    """Track a strict forward-only subtask sequence and active elapsed time.

    The tracker deliberately does not own a lock. The RTC engine will call it
    while holding its semantic state lock so history and time can be committed
    as one transaction.
    """

    def __init__(
        self,
        contract: SubtaskSequenceContract,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(clock):
            raise TypeError(f"clock must be callable, got {type(clock).__name__}")
        self._contract = contract
        self._index_by_name = _validate_contract(contract)
        self._clock = clock
        self._last_clock_seconds: float | None = None

        self._current_index: int | None = None
        self._accumulated_active_seconds = 0.0
        self._running_since_monotonic: float | None = None
        self._paused = False
        self._last_transition_reason = "initialized"
        self._last_rejected_output = ""
        self._last_rejection_reason = ""

    @property
    def contract(self) -> SubtaskSequenceContract:
        return self._contract

    def _read_clock(self) -> float:
        return self._observe_clock(self._clock())

    def _observe_clock(self, value: object) -> float:
        """Validate and record a caller-sampled monotonic timestamp."""
        now = _finite_real(value, name="monotonic clock", positive=False)
        if self._last_clock_seconds is not None and now < self._last_clock_seconds:
            raise RuntimeError(
                "Monotonic clock moved backwards: "
                f"{now} < {self._last_clock_seconds}"
            )
        self._last_clock_seconds = now
        return now

    def _clear_rejection(self) -> None:
        self._last_rejected_output = ""
        self._last_rejection_reason = ""

    def _reject(self, output: str, *, transition: str, reason: str) -> None:
        self._last_transition_reason = transition
        self._last_rejected_output = output
        self._last_rejection_reason = reason

    def _raw_elapsed(self, now: float | None = None) -> float:
        if self._current_index is None:
            return 0.0
        raw = self._accumulated_active_seconds
        if self._running_since_monotonic is not None:
            if now is None:
                now = self._read_clock()
            raw += now - self._running_since_monotonic
        if raw < 0:
            raise RuntimeError(f"Subtask elapsed time became negative: {raw}")
        return raw

    def _make_snapshot(self, *, now: float | None = None) -> SubtaskTimeTrackerSnapshot:
        if self._current_index is None:
            return SubtaskTimeTrackerSnapshot(
                current_index=None,
                current_name=None,
                raw_elapsed_seconds=0.0,
                effective_elapsed_seconds=0.0,
                cap_seconds=None,
                time_valid=False,
                running=False,
                paused=self._paused,
                last_transition_reason=self._last_transition_reason,
                last_rejected_output=self._last_rejected_output,
                last_rejection_reason=self._last_rejection_reason,
            )

        stats = self._contract.ordered_subtasks[self._current_index]
        raw = self._raw_elapsed(now)
        return SubtaskTimeTrackerSnapshot(
            current_index=self._current_index,
            current_name=stats.canonical_name,
            raw_elapsed_seconds=raw,
            effective_elapsed_seconds=min(raw, stats.deployment_cap_seconds),
            cap_seconds=stats.deployment_cap_seconds,
            time_valid=True,
            running=self._running_since_monotonic is not None and not self._paused,
            paused=self._paused,
            last_transition_reason=self._last_transition_reason,
            last_rejected_output=self._last_rejected_output,
            last_rejection_reason=self._last_rejection_reason,
        )

    def snapshot(self, *, at_monotonic: float | None = None) -> SubtaskTimeTrackerSnapshot:
        """Return an immutable view at an optional caller-sampled clock value."""
        now = self._observe_clock(at_monotonic) if at_monotonic is not None else None
        return self._make_snapshot(now=now)

    def commit_subtask_output(self, output: str) -> SubtaskTimeTrackerSnapshot:
        """Commit a successful inference output under the strict sequence rules."""
        name = parse_subtask_output_name(output)
        if name is None:
            self._reject(output, transition="rejected_parse", reason="parse_failure")
            return self._make_snapshot()

        normalized = normalize_subtask_name(name)
        candidate_index = self._index_by_name.get(normalized)
        if candidate_index is None:
            self._reject(output, transition="rejected_unknown", reason="unknown_subtask")
            return self._make_snapshot()

        if self._current_index is None:
            if candidate_index != 0:
                self._reject(
                    output,
                    transition="rejected_initial_not_first",
                    reason="initial_subtask_must_be_index_zero",
                )
                return self._make_snapshot()
            now = None if self._paused else self._read_clock()
            self._current_index = 0
            self._accumulated_active_seconds = 0.0
            self._running_since_monotonic = now
            self._last_transition_reason = "started"
            self._clear_rejection()
            return self._make_snapshot(now=now)

        if candidate_index == self._current_index:
            self._last_transition_reason = "current"
            self._clear_rejection()
            return self._make_snapshot()

        if candidate_index == self._current_index + 1:
            now = None if self._paused else self._read_clock()
            self._current_index = candidate_index
            self._accumulated_active_seconds = 0.0
            self._running_since_monotonic = now
            self._last_transition_reason = "advanced"
            self._clear_rejection()
            return self._make_snapshot(now=now)

        if candidate_index < self._current_index:
            self._reject(output, transition="rejected_old", reason="old_subtask")
        else:
            self._reject(output, transition="rejected_skip", reason="skipped_future_subtask")
        return self._make_snapshot()

    def pause(self) -> SubtaskTimeTrackerSnapshot:
        """Freeze active elapsed time while preserving the confirmed subtask."""
        if self._paused:
            return self._make_snapshot()
        now = None
        if self._running_since_monotonic is not None:
            now = self._read_clock()
            self._accumulated_active_seconds = self._raw_elapsed(now)
            self._running_since_monotonic = None
        self._paused = True
        self._last_transition_reason = "paused"
        return self._make_snapshot(now=now)

    def resume(self) -> SubtaskTimeTrackerSnapshot:
        """Resume active time from the frozen value without counting pause wall time."""
        if not self._paused:
            return self._make_snapshot()
        now = None
        if self._current_index is not None:
            now = self._read_clock()
            self._running_since_monotonic = now
        self._paused = False
        self._last_transition_reason = "resumed"
        return self._make_snapshot(now=now)

    def full_reset(self) -> SubtaskTimeTrackerSnapshot:
        """Clear semantic time while preserving the caller-controlled pause state."""
        self._current_index = None
        self._accumulated_active_seconds = 0.0
        self._running_since_monotonic = None
        self._last_transition_reason = "full_reset"
        self._clear_rejection()
        return self._make_snapshot()
