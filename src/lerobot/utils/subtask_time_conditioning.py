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

"""Train-only noise, dropout, and diagnostics for subtask elapsed time."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import torch

SUBTASK_ELAPSED_SECONDS = "subtask_elapsed_seconds"
SUBTASK_TIME_VALID = "subtask_time_valid"
SUBTASK_TIME_SECONDS = "subtask_time_seconds"
SUBTASK_TIME_CONDITION_KEPT = "subtask_time_condition_kept"


def _validate_nonnegative_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite non-negative real number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return converted


def _validate_dropout_prob(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"dropout_prob must be a finite real number in [0, 1], got {value!r}")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"dropout_prob must be finite and in [0, 1], got {value!r}")
    return converted


def _as_vector(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"{name} must contain numeric values") from error
    if value.ndim == 0:
        value = value.unsqueeze(0)
    elif value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [B] or [B, 1], got {tuple(value.shape)}")
    return value


def _as_elapsed_seconds(value: Any, *, name: str) -> torch.Tensor:
    value = _as_vector(value, name=name)
    if value.dtype == torch.bool or value.is_complex():
        raise ValueError(f"{name} must contain numeric elapsed seconds, not {value.dtype}")
    value = value.to(dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if (value < 0).any():
        raise ValueError(f"{name} must contain non-negative values")
    return value


def _as_bool_mask(value: Any, *, name: str) -> torch.Tensor:
    value = _as_vector(value, name=name)
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must contain boolean values")
    return value


def _read_time_fields(
    batch: dict[str, Any],
    *,
    elapsed_key: str,
    valid_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if elapsed_key not in batch:
        raise ValueError(f"Subtask-time training requires elapsed field {elapsed_key!r}")
    if valid_key not in batch:
        raise ValueError(f"Subtask-time training requires validity field {valid_key!r}")
    elapsed = _as_elapsed_seconds(batch[elapsed_key], name=elapsed_key)
    valid = _as_bool_mask(batch[valid_key], name=valid_key)
    if valid.shape[0] != elapsed.shape[0]:
        raise ValueError(
            f"{valid_key} batch size ({valid.shape[0]}) does not match "
            f"{elapsed_key} batch size ({elapsed.shape[0]})"
        )
    return elapsed, valid.to(device=elapsed.device)


def sample_subtask_time_condition(
    batch: dict[str, Any],
    noise_ratio: float,
    noise_max_seconds: float,
    dropout_prob: float,
    generator: torch.Generator | None = None,
    *,
    elapsed_key: str = SUBTASK_ELAPSED_SECONDS,
    valid_key: str = SUBTASK_TIME_VALID,
    seconds_key: str = SUBTASK_TIME_SECONDS,
    condition_kept_key: str = SUBTASK_TIME_CONDITION_KEPT,
) -> dict[str, Any]:
    """Add noisy elapsed seconds and an independent keep mask without mutating ``batch``.

    Every enabled call consumes exactly two full-batch uniform draws: noise first,
    then dropout. Invalid samples participate in both draws to keep the RNG stream
    independent of validity patterns, but their keep mask is always false.
    """

    noise_ratio = _validate_nonnegative_finite(noise_ratio, name="noise_ratio")
    noise_max_seconds = _validate_nonnegative_finite(
        noise_max_seconds, name="noise_max_seconds"
    )
    dropout_prob = _validate_dropout_prob(dropout_prob)
    elapsed, valid = _read_time_fields(batch, elapsed_key=elapsed_key, valid_key=valid_key)

    noise_uniform = torch.rand(
        elapsed.shape[0],
        dtype=torch.float32,
        device=elapsed.device,
        generator=generator,
    )
    dropout_uniform = torch.rand(
        elapsed.shape[0],
        dtype=torch.float32,
        device=elapsed.device,
        generator=generator,
    )
    amplitude = torch.clamp(elapsed * noise_ratio, max=noise_max_seconds)
    noise = (noise_uniform * 2.0 - 1.0) * amplitude
    noisy_seconds = torch.clamp_min(elapsed + noise, 0.0)
    kept = valid & (dropout_uniform >= dropout_prob)

    result = dict(batch)
    result[seconds_key] = noisy_seconds
    result[condition_kept_key] = kept
    return result


@dataclass
class SubtaskTimeTrainingMetrics:
    """Accumulate sample-weighted subtask-time diagnostics over a logging window."""

    total_samples: int = 0
    valid_samples: int = 0
    kept_samples: int = 0
    true_seconds_sum: float = 0.0
    true_seconds_max: float | None = None
    noisy_seconds_sum: float = 0.0
    noise_abs_sum: float = 0.0
    noise_abs_max: float | None = None
    clamped_to_zero_samples: int = 0

    def update(
        self,
        batch: dict[str, Any],
        *,
        elapsed_key: str = SUBTASK_ELAPSED_SECONDS,
        valid_key: str = SUBTASK_TIME_VALID,
        seconds_key: str = SUBTASK_TIME_SECONDS,
        condition_kept_key: str = SUBTASK_TIME_CONDITION_KEPT,
    ) -> None:
        elapsed, valid = _read_time_fields(batch, elapsed_key=elapsed_key, valid_key=valid_key)
        if seconds_key not in batch:
            raise ValueError(f"Subtask-time metrics require noisy-seconds field {seconds_key!r}")
        if condition_kept_key not in batch:
            raise ValueError(f"Subtask-time metrics require keep-mask field {condition_kept_key!r}")

        noisy = _as_elapsed_seconds(batch[seconds_key], name=seconds_key).to(device=elapsed.device)
        kept = _as_bool_mask(batch[condition_kept_key], name=condition_kept_key).to(
            device=elapsed.device
        )
        if noisy.shape != elapsed.shape:
            raise ValueError(
                f"{seconds_key} batch size ({noisy.shape[0]}) does not match "
                f"{elapsed_key} batch size ({elapsed.shape[0]})"
            )
        if kept.shape != valid.shape:
            raise ValueError(
                f"{condition_kept_key} batch size ({kept.shape[0]}) does not match "
                f"{valid_key} batch size ({valid.shape[0]})"
            )
        if (kept & ~valid).any():
            raise ValueError(f"{condition_kept_key} cannot keep invalid subtask-time samples")

        batch_size = elapsed.shape[0]
        self.total_samples += batch_size
        current_valid = int(valid.sum().item())
        self.valid_samples += current_valid
        self.kept_samples += int(kept.sum().item())
        if current_valid == 0:
            return

        valid_true = elapsed[valid].to(dtype=torch.float64)
        valid_noisy = noisy[valid].to(dtype=torch.float64)
        noise_abs = (valid_noisy - valid_true).abs()
        self.true_seconds_sum += float(valid_true.sum().item())
        self.noisy_seconds_sum += float(valid_noisy.sum().item())
        self.noise_abs_sum += float(noise_abs.sum().item())
        current_true_max = float(valid_true.max().item())
        current_noise_max = float(noise_abs.max().item())
        self.true_seconds_max = (
            current_true_max
            if self.true_seconds_max is None
            else max(self.true_seconds_max, current_true_max)
        )
        self.noise_abs_max = (
            current_noise_max
            if self.noise_abs_max is None
            else max(self.noise_abs_max, current_noise_max)
        )
        self.clamped_to_zero_samples += int(((valid_true > 0) & (valid_noisy == 0)).sum().item())

    def to_dict(self) -> dict[str, float]:
        if self.total_samples == 0:
            return {}
        if self.valid_samples:
            dropout_fraction = (self.valid_samples - self.kept_samples) / self.valid_samples
            true_mean = self.true_seconds_sum / self.valid_samples
            noisy_mean = self.noisy_seconds_sum / self.valid_samples
            noise_abs_mean = self.noise_abs_sum / self.valid_samples
            clamped_fraction = self.clamped_to_zero_samples / self.valid_samples
        else:
            dropout_fraction = 0.0
            true_mean = 0.0
            noisy_mean = 0.0
            noise_abs_mean = 0.0
            clamped_fraction = 0.0
        return {
            "subtask_time/valid_fraction": self.valid_samples / self.total_samples,
            "subtask_time/condition_kept_fraction": self.kept_samples / self.total_samples,
            "subtask_time/dropout_fraction_among_valid": dropout_fraction,
            "subtask_time/true_seconds_mean": true_mean,
            "subtask_time/true_seconds_max_seen": self.true_seconds_max or 0.0,
            "subtask_time/noisy_seconds_mean": noisy_mean,
            "subtask_time/noise_abs_mean": noise_abs_mean,
            "subtask_time/noise_abs_max_seen": self.noise_abs_max or 0.0,
            "subtask_time/clamped_to_zero_fraction": clamped_fraction,
        }

    def reset(self) -> None:
        self.total_samples = 0
        self.valid_samples = 0
        self.kept_samples = 0
        self.true_seconds_sum = 0.0
        self.true_seconds_max = None
        self.noisy_seconds_sum = 0.0
        self.noise_abs_sum = 0.0
        self.noise_abs_max = None
        self.clamped_to_zero_samples = 0


def compute_subtask_time_training_metrics(batch: dict[str, Any]) -> dict[str, float]:
    """Return the canonical diagnostics for one already-sampled training batch."""

    metrics = SubtaskTimeTrainingMetrics()
    metrics.update(batch)
    return metrics.to_dict()
