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

"""Train-only dropout and diagnostics for PI0/PI0.5 memory conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

MEMORY_CONDITION_KEPT = "memory_condition_kept"


def _as_bool_mask(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        value = value.unsqueeze(0)
    elif value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [B] or [B, 1], got {tuple(value.shape)}")
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must contain boolean values")
    return value


def _as_string_list(value: Any, *, name: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(f"{name} must contain one string per sample")
    if not values or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{name} must contain one string per sample")
    return values


def _as_offsets(value: Any, *, name: str, batch_size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        value = value.unsqueeze(0)
    elif value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(f"{name} must contain one offset per sample with shape [B] or [B, 1]")
    if value.dtype == torch.bool or value.is_floating_point() or (value <= 0).any():
        raise ValueError(f"{name} must contain positive integer offsets")
    return value.to(dtype=torch.int64)


def _memory_eligibility(
    batch: dict[str, Any],
    *,
    valid_key: str,
    source_key: str,
) -> torch.Tensor:
    if valid_key not in batch:
        raise ValueError(f"Memory training requires validity field {valid_key!r}")
    if source_key not in batch:
        raise ValueError(f"Memory training requires source field {source_key!r}")

    valid = _as_bool_mask(batch[valid_key], name=valid_key)
    sources = _as_string_list(batch[source_key], name=source_key)
    if len(sources) != valid.shape[0]:
        raise ValueError(
            f"{source_key} batch size ({len(sources)}) does not match "
            f"{valid_key} batch size ({valid.shape[0]})"
        )
    non_empty = torch.tensor(
        [bool(source.strip()) for source in sources],
        dtype=torch.bool,
        device=valid.device,
    )
    return valid & non_empty


def sample_memory_condition_mask(
    batch: dict[str, Any],
    *,
    dropout_prob: float,
    valid_key: str = "memory_valid",
    source_key: str = "memory_subtask",
    condition_kept_key: str = MEMORY_CONDITION_KEPT,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Add the train-only memory-condition keep mask without mutating ``batch``."""

    if not 0.0 <= dropout_prob <= 1.0:
        raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}")
    eligible = _memory_eligibility(batch, valid_key=valid_key, source_key=source_key)

    if dropout_prob == 0.0:
        kept = eligible
    elif dropout_prob == 1.0:
        kept = torch.zeros_like(eligible)
    else:
        sampled = torch.rand(
            eligible.shape[0],
            generator=generator,
            device=eligible.device,
        )
        kept = eligible & (sampled >= dropout_prob)

    result = dict(batch)
    result[condition_kept_key] = kept
    return result


@dataclass
class MemoryTrainingMetrics:
    """Accumulate sample-weighted memory diagnostics over a logging window."""

    total_samples: int = 0
    valid_samples: int = 0
    kept_samples: int = 0
    lookback_sum: int = 0
    lookback_min: int | None = None
    lookback_max: int | None = None

    def update(
        self,
        batch: dict[str, Any],
        *,
        valid_key: str = "memory_valid",
        source_key: str = "memory_subtask",
        condition_kept_key: str = MEMORY_CONDITION_KEPT,
        offset_key: str = "memory_frame_offset",
    ) -> None:
        eligible = _memory_eligibility(batch, valid_key=valid_key, source_key=source_key)
        if condition_kept_key not in batch:
            raise ValueError(f"Memory metrics require keep-mask field {condition_kept_key!r}")
        kept = _as_bool_mask(batch[condition_kept_key], name=condition_kept_key)
        if kept.shape != eligible.shape:
            raise ValueError(
                f"{condition_kept_key} shape {tuple(kept.shape)} does not match "
                f"memory eligibility shape {tuple(eligible.shape)}"
            )
        kept = kept.to(device=eligible.device)
        if (kept & ~eligible).any():
            raise ValueError(f"{condition_kept_key} cannot keep ineligible memory samples")
        if offset_key not in batch:
            raise ValueError(f"Memory metrics require lookback field {offset_key!r}")
        offsets = _as_offsets(batch[offset_key], name=offset_key, batch_size=eligible.shape[0])

        self.total_samples += eligible.shape[0]
        self.valid_samples += int(eligible.sum().item())
        self.kept_samples += int(kept.sum().item())
        self.lookback_sum += int(offsets.sum().item())
        current_min = int(offsets.min().item())
        current_max = int(offsets.max().item())
        self.lookback_min = current_min if self.lookback_min is None else min(self.lookback_min, current_min)
        self.lookback_max = current_max if self.lookback_max is None else max(self.lookback_max, current_max)

    def to_dict(self) -> dict[str, float]:
        if self.total_samples == 0:
            return {}
        dropped = self.valid_samples - self.kept_samples
        dropout_fraction = dropped / self.valid_samples if self.valid_samples else 0.0
        return {
            "memory/history_valid_fraction": self.valid_samples / self.total_samples,
            "memory/condition_kept_fraction": self.kept_samples / self.total_samples,
            "memory/dropout_fraction_among_valid": dropout_fraction,
            "memory/lookback_frames_mean": self.lookback_sum / self.total_samples,
            "memory/lookback_frames_min_seen": float(self.lookback_min),
            "memory/lookback_frames_max_seen": float(self.lookback_max),
        }

    def reset(self) -> None:
        self.total_samples = 0
        self.valid_samples = 0
        self.kept_samples = 0
        self.lookback_sum = 0
        self.lookback_min = None
        self.lookback_max = None


def compute_memory_training_metrics(batch: dict[str, Any]) -> dict[str, float]:
    """Return the canonical diagnostics for one already-masked training batch."""

    metrics = MemoryTrainingMetrics()
    metrics.update(batch)
    return metrics.to_dict()
