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

"""Dynamic same-episode history for PI0/PI0.5 memory conditioning."""

import math
import numbers
from typing import Any

import torch

MEMORY_REQUIRED_FEATURES = frozenset(
    {"subtask", "subtask_progress", "frame_index", "episode_index", "index"}
)
DEFAULT_MEMORY_LOOKBACK_MIN_FRAMES = 1
DEFAULT_MEMORY_LOOKBACK_MAX_FRAMES = 12


def validate_memory_history_features(features: dict[str, Any]) -> None:
    """Validate the raw columns required for dynamic memory history."""
    missing = sorted(MEMORY_REQUIRED_FEATURES.difference(features))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Memory conditioning requires dataset features "
            "subtask, subtask_progress, frame_index, episode_index, and index; "
            f"missing: {missing_text}. Rebuild the LeRobotDataset with per-frame subtask annotations."
        )


def _int_scalar(value: Any, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"{name} must be a scalar, got tensor shape {tuple(value.shape)}")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise RuntimeError(f"{name} must be an integer scalar, got {value!r}")
    return int(value)


def _finite_float_scalar(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class MemoryHistoryDataset(torch.utils.data.Dataset):
    """Add a randomly sampled previous subtask/progress to a map-style dataset.

    The current item is loaded through the wrapped dataset's normal ``__getitem__``
    path. History uses ``get_raw_item`` so that images and temporal windows are not
    decoded a second time.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        lookback_min_frames: int = DEFAULT_MEMORY_LOOKBACK_MIN_FRAMES,
        lookback_max_frames: int = DEFAULT_MEMORY_LOOKBACK_MAX_FRAMES,
    ) -> None:
        super().__init__()
        if (
            isinstance(lookback_min_frames, bool)
            or isinstance(lookback_max_frames, bool)
            or not isinstance(lookback_min_frames, numbers.Integral)
            or not isinstance(lookback_max_frames, numbers.Integral)
            or lookback_min_frames < 1
            or lookback_min_frames > lookback_max_frames
        ):
            raise ValueError(
                "memory lookback must satisfy 1 <= lookback_min_frames <= "
                f"lookback_max_frames, got {lookback_min_frames} and {lookback_max_frames}"
            )
        if not callable(getattr(dataset, "get_raw_item", None)):
            raise ValueError(
                "MemoryHistoryDataset requires a non-streaming map-style dataset with get_raw_item()."
            )

        meta = getattr(dataset, "meta", None)
        features = getattr(meta, "features", None)
        if not isinstance(features, dict):
            raise ValueError("MemoryHistoryDataset requires dataset.meta.features for early validation.")
        validate_memory_history_features(features)

        self.dataset = dataset
        self.lookback_min_frames = int(lookback_min_frames)
        self.lookback_max_frames = int(lookback_max_frames)

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

    def __len__(self) -> int:
        return len(self.dataset)

    def get_raw_item(self, index: int) -> dict[str, Any]:
        """Delegate raw access without adding current or history synthetic fields."""
        return self.dataset.get_raw_item(index)

    def select_columns(self, column_names: str | list[str]):
        """Delegate lightweight column selection through composed wrappers."""
        return self.dataset.select_columns(column_names)

    @staticmethod
    def _no_memory(item: dict[str, Any], offset: int) -> dict[str, Any]:
        item.update(
            {
                "memory_subtask": "",
                "memory_subtask_progress": torch.tensor(0.0, dtype=torch.float32),
                "memory_valid": False,
                "memory_frame_offset": offset,
            }
        )
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        offset = int(
            torch.randint(
                self.lookback_min_frames,
                self.lookback_max_frames + 1,
                (1,),
            ).item()
        )

        current_frame_index = _int_scalar(item.get("frame_index"), name="current frame_index")
        if current_frame_index - offset < 0:
            return self._no_memory(item, offset)

        history_index = int(index) - offset
        history = self.dataset.get_raw_item(history_index)
        current_episode_index = _int_scalar(
            item.get("episode_index"), name="current episode_index"
        )
        history_episode_index = _int_scalar(
            history.get("episode_index"), name="history episode_index"
        )
        if history_episode_index != current_episode_index:
            raise RuntimeError(
                "Memory history crossed an episode boundary: "
                f"current episode {current_episode_index}, history episode {history_episode_index}."
            )

        history_frame_index = _int_scalar(
            history.get("frame_index"), name="history frame_index"
        )
        expected_history_frame = current_frame_index - offset
        if history_frame_index != expected_history_frame:
            raise RuntimeError(
                "Memory history frame mismatch: "
                f"expected frame_index {expected_history_frame}, got {history_frame_index}."
            )

        current_absolute_index = _int_scalar(item.get("index"), name="current index")
        history_absolute_index = _int_scalar(history.get("index"), name="history index")
        if history_absolute_index != current_absolute_index - offset:
            raise RuntimeError(
                "Memory history absolute index mismatch: "
                f"expected {current_absolute_index - offset}, got {history_absolute_index}."
            )

        subtask = history.get("subtask")
        progress = _finite_float_scalar(history.get("subtask_progress"))
        if not isinstance(subtask, str) or not subtask.strip() or progress is None:
            return self._no_memory(item, offset)

        item.update(
            {
                "memory_subtask": subtask,
                "memory_subtask_progress": torch.tensor(progress, dtype=torch.float32),
                "memory_valid": True,
                "memory_frame_offset": offset,
            }
        )
        return item
