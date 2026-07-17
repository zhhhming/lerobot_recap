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

"""Lightweight subtask-segment timing lookup for map-style datasets."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import torch


SUBTASK_TIMING_REQUIRED_FEATURES = frozenset(
    {"episode_index", "frame_index", "index", "subtask"}
)
SUBTASK_TIMING_COLUMNS = ("episode_index", "frame_index", "index", "subtask")
DEFAULT_SUBTASK_TIME_DEPLOYMENT_MARGIN_SECONDS = 5.0
_TRAILING_PERIODS = (".", "。", "．", "｡")


@dataclass(frozen=True)
class SubtaskSegmentStats:
    canonical_name: str
    normalized_name: str
    max_elapsed_seconds: float
    deployment_cap_seconds: float


@dataclass(frozen=True)
class SubtaskSequenceContract:
    fps: float
    ordered_subtasks: tuple[SubtaskSegmentStats, ...]


@dataclass(frozen=True)
class SubtaskTimingScan:
    elapsed_seconds: torch.Tensor
    valid: torch.Tensor
    segment_indices: torch.Tensor
    sequence_contract: SubtaskSequenceContract

    @property
    def lookup_nbytes(self) -> int:
        return sum(
            tensor.nelement() * tensor.element_size()
            for tensor in (self.elapsed_seconds, self.valid, self.segment_indices)
        )


def normalize_subtask_name(value: str) -> str:
    """Normalize labels for shared dataset-segment and deployment matching rules."""
    if not isinstance(value, str):
        raise TypeError(f"subtask name must be a string, got {value!r}")
    normalized = " ".join(value.strip().split()).casefold()
    if normalized.endswith(_TRAILING_PERIODS):
        normalized = normalized[:-1].rstrip()
    return normalized


def validate_subtask_timing_features(features: dict[str, Any]) -> None:
    """Validate the raw columns required to derive subtask elapsed time."""
    missing = sorted(SUBTASK_TIMING_REQUIRED_FEATURES.difference(features))
    if missing:
        raise ValueError(
            "Subtask time conditioning dataset features are missing: "
            f"{', '.join(missing)}. Rebuild or select a LeRobotDataset with per-frame "
            "subtask, episode_index, frame_index, and index annotations."
        )


def _finite_nonnegative_real(value: Any, *, name: str, positive: bool) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(f"{name} must be a real scalar, got tensor shape {tuple(value.shape)}")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar, got {value!r}")
    result = float(value)
    lower_bound_ok = result > 0 if positive else result >= 0
    if not math.isfinite(result) or not lower_bound_ok:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}, got {value!r}")
    return result


def _integer_scalar(value: Any, *, name: str, row: int) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name} must be an integer scalar at row {row}, "
                f"got tensor shape {tuple(value.shape)}"
            )
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer scalar at row {row}, got {value!r}")
    return int(value)


def _extract_columns(rows: Any) -> dict[str, list[Any]]:
    try:
        frame_count = len(rows)
    except TypeError as exc:
        raise TypeError("subtask timing rows must be a sized column view or row sequence") from exc
    if frame_count == 0:
        raise ValueError("Subtask timing dataset contains no frames")

    columns: dict[str, list[Any]] = {}
    supports_column_access = True
    for name in SUBTASK_TIMING_COLUMNS:
        try:
            values = rows[name]
        except (KeyError, TypeError, IndexError):
            supports_column_access = False
            break
        columns[name] = list(values)

    if not supports_column_access:
        columns = {name: [] for name in SUBTASK_TIMING_COLUMNS}
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Subtask timing row {row_index} must be a mapping, got {type(row).__name__}"
                )
            missing = [name for name in SUBTASK_TIMING_COLUMNS if name not in row]
            if missing:
                raise ValueError(
                    f"Subtask timing rows are missing columns at row {row_index}: "
                    f"{', '.join(missing)}"
                )
            for name in SUBTASK_TIMING_COLUMNS:
                columns[name].append(row[name])

    bad_lengths = {name: len(values) for name, values in columns.items() if len(values) != frame_count}
    if bad_lengths:
        raise ValueError(
            f"Subtask timing columns have inconsistent lengths; expected {frame_count}, got {bad_lengths}"
        )
    return columns


def scan_subtask_timing(
    rows: Any,
    *,
    fps: float,
    deployment_margin_seconds: float = DEFAULT_SUBTASK_TIME_DEPLOYMENT_MARGIN_SECONDS,
    dataset_name: str = "dataset",
) -> SubtaskTimingScan:
    """Scan lightweight columns once and build O(1) per-frame timing lookup tensors."""
    fps_value = _finite_nonnegative_real(fps, name="fps", positive=True)
    margin = _finite_nonnegative_real(
        deployment_margin_seconds,
        name="deployment margin seconds",
        positive=False,
    )
    columns = _extract_columns(rows)
    frame_count = len(columns["index"])
    elapsed_seconds = torch.empty(frame_count, dtype=torch.float32)
    valid = torch.ones(frame_count, dtype=torch.bool)
    segment_indices = torch.empty(frame_count, dtype=torch.int64)

    episode_sequences: dict[int, list[str]] = {}
    episode_canonical_sequences: dict[int, list[str]] = {}
    durations: dict[str, list[float]] = {}
    canonical_names: dict[str, str] = {}
    seen_episodes: set[int] = set()

    current_episode: int | None = None
    current_normalized: str | None = None
    segment_start_frame = 0
    segment_index = -1
    previous_frame: int | None = None
    previous_absolute_index: int | None = None

    def finish_segment() -> None:
        if current_normalized is None or previous_frame is None:
            return
        durations.setdefault(current_normalized, []).append(
            (previous_frame - segment_start_frame) / fps_value
        )

    for row in range(frame_count):
        episode = _integer_scalar(columns["episode_index"][row], name="episode_index", row=row)
        frame = _integer_scalar(columns["frame_index"][row], name="frame_index", row=row)
        absolute_index = _integer_scalar(columns["index"][row], name="index", row=row)
        label = columns["subtask"][row]
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"Subtask label must be a non-empty string in {dataset_name} at row {row}: {label!r}"
            )
        canonical = " ".join(label.strip().split())
        normalized = normalize_subtask_name(canonical)
        if not normalized:
            raise ValueError(
                f"Subtask label normalizes to an empty string in {dataset_name} at row {row}: {label!r}"
            )

        episode_changed = current_episode is None or episode != current_episode
        if episode_changed:
            if current_episode is not None:
                finish_segment()
            if episode in seen_episodes:
                raise ValueError(
                    f"Dataset {dataset_name} episode {episode} reappears after another episode at row {row}"
                )
            seen_episodes.add(episode)
            if frame != 0:
                raise ValueError(
                    f"Dataset {dataset_name} frame_index must start at 0 for episode {episode}; "
                    f"got {frame} at row {row}"
                )
            current_episode = episode
            current_normalized = normalized
            segment_start_frame = frame
            segment_index = 0
            episode_sequences[episode] = [normalized]
            episode_canonical_sequences[episode] = [canonical]
            canonical_names.setdefault(normalized, canonical)
        else:
            assert previous_frame is not None
            assert previous_absolute_index is not None
            expected_frame = previous_frame + 1
            if frame != expected_frame:
                raise ValueError(
                    f"Dataset {dataset_name} frame_index mismatch in episode {episode} at row {row}: "
                    f"expected {expected_frame}, got {frame}"
                )
            expected_index = previous_absolute_index + 1
            if absolute_index != expected_index:
                raise ValueError(
                    f"Dataset {dataset_name} index mismatch in episode {episode} at row {row}: "
                    f"expected {expected_index}, got {absolute_index}"
                )
            if normalized != current_normalized:
                finish_segment()
                current_normalized = normalized
                segment_start_frame = frame
                segment_index += 1
                episode_sequences[episode].append(normalized)
                episode_canonical_sequences[episode].append(canonical)
                canonical_names.setdefault(normalized, canonical)

        elapsed_seconds[row] = (frame - segment_start_frame) / fps_value
        segment_indices[row] = segment_index
        previous_frame = frame
        previous_absolute_index = absolute_index

    finish_segment()

    first_episode = next(iter(episode_sequences))
    expected_sequence = episode_sequences[first_episode]
    expected_canonical = episode_canonical_sequences[first_episode]
    if len(set(expected_sequence)) != len(expected_sequence):
        raise ValueError(
            f"Dataset {dataset_name} episode {first_episode} repeats a normalized subtask: "
            f"{expected_canonical!r}"
        )
    for episode, actual in episode_sequences.items():
        if len(set(actual)) != len(actual):
            raise ValueError(
                f"Dataset {dataset_name} episode {episode} repeats a normalized subtask: "
                f"{episode_canonical_sequences[episode]!r}"
            )
        if actual != expected_sequence:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(expected_sequence, actual, strict=False))
                    if pair[0] != pair[1]
                ),
                min(len(expected_sequence), len(actual)),
            )
            raise ValueError(
                f"Dataset {dataset_name} has inconsistent subtask sequence in episode {episode} "
                f"at position {mismatch}; expected={expected_canonical!r}, "
                f"actual={episode_canonical_sequences[episode]!r}"
            )

    ordered_stats = tuple(
        SubtaskSegmentStats(
            canonical_name=canonical_names[normalized],
            normalized_name=normalized,
            max_elapsed_seconds=max(durations[normalized]),
            deployment_cap_seconds=max(durations[normalized]) + margin,
        )
        for normalized in expected_sequence
    )
    return SubtaskTimingScan(
        elapsed_seconds=elapsed_seconds,
        valid=valid,
        segment_indices=segment_indices,
        sequence_contract=SubtaskSequenceContract(
            fps=fps_value,
            ordered_subtasks=ordered_stats,
        ),
    )


class SubtaskTimingDataset(torch.utils.data.Dataset):
    """Append true elapsed time derived from a precomputed lightweight column scan."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        deployment_margin_seconds: float = DEFAULT_SUBTASK_TIME_DEPLOYMENT_MARGIN_SECONDS,
    ) -> None:
        super().__init__()
        if not callable(getattr(dataset, "select_columns", None)) or not callable(
            getattr(dataset, "get_raw_item", None)
        ):
            raise ValueError(
                "SubtaskTimingDataset requires a non-streaming map-style dataset with "
                "select_columns() and get_raw_item()."
            )
        meta = getattr(dataset, "meta", None)
        features = getattr(meta, "features", None)
        if not isinstance(features, dict):
            raise ValueError("SubtaskTimingDataset requires dataset.meta.features for validation")
        validate_subtask_timing_features(features)

        column_view = dataset.select_columns(list(SUBTASK_TIMING_COLUMNS))
        dataset_name = f"{getattr(dataset, 'repo_id', 'dataset')} ({getattr(dataset, 'root', '')})"
        scan = scan_subtask_timing(
            column_view,
            fps=getattr(meta, "fps", None),
            deployment_margin_seconds=deployment_margin_seconds,
            dataset_name=dataset_name,
        )
        if len(dataset) != scan.elapsed_seconds.numel():
            raise ValueError(
                "Subtask timing column view length does not match the wrapped dataset: "
                f"{scan.elapsed_seconds.numel()} != {len(dataset)}"
            )

        self.dataset = dataset
        self._timing_scan = scan

    @property
    def meta(self):
        return self.dataset.meta

    @property
    def episodes(self):
        return self.dataset.episodes

    @property
    def num_frames(self):
        return self.dataset.num_frames

    @property
    def num_episodes(self):
        return self.dataset.num_episodes

    @property
    def features(self):
        return self.dataset.features

    @property
    def fps(self):
        return self.dataset.fps

    @property
    def repo_id(self):
        return self.dataset.repo_id

    @property
    def root(self):
        return self.dataset.root

    @property
    def sequence_contract(self) -> SubtaskSequenceContract:
        return self._timing_scan.sequence_contract

    @property
    def lookup_nbytes(self) -> int:
        return self._timing_scan.lookup_nbytes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        item.update(
            {
                "subtask_elapsed_seconds": self._timing_scan.elapsed_seconds[index].clone(),
                "subtask_time_valid": bool(self._timing_scan.valid[index].item()),
                "subtask_segment_index": self._timing_scan.segment_indices[index].clone(),
            }
        )
        return item

    def get_raw_item(self, index: int) -> dict[str, Any]:
        return self.dataset.get_raw_item(index)

    def select_columns(self, column_names: str | list[str]):
        return self.dataset.select_columns(column_names)
