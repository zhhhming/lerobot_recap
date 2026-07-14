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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


ADVANTAGE_CONDITION_KEPT = "advantage_condition_kept"
_CONDITION_LABELS = {"positive", "negative"}


def _as_label_list(value: Any, *, name: str) -> list[str]:
    if isinstance(value, str):
        labels = [value]
    elif isinstance(value, np.ndarray):
        labels = value.tolist()
    elif isinstance(value, (list, tuple)):
        labels = list(value)
    else:
        raise ValueError(f"{name} must contain one string label per sample")
    if not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"{name} must contain one string label per sample")
    return [label.strip() for label in labels]


def _as_per_sample_tensor(value: Any, *, name: str, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        value = value.unsqueeze(0)
    elif value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [B] or [B, 1], got {tuple(value.shape)}")
    return value.to(dtype=dtype)


def sample_advantage_condition_mask(
    batch: dict[str, Any],
    *,
    label_key: str,
    dropout_prob: float,
    ignore_label: str = "ignore",
    condition_kept_key: str = ADVANTAGE_CONDITION_KEPT,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Add the train-only classifier-free advantage-condition keep mask.

    Only positive and negative labels participate in Bernoulli sampling. Ignore
    samples are always unconditioned. The input batch is not mutated.
    """

    if not 0.0 <= dropout_prob <= 1.0:
        raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}")
    if label_key not in batch:
        raise ValueError(
            f"Training with advantage conditioning requires label field {label_key!r}"
        )

    labels = _as_label_list(batch[label_key], name=label_key)
    allowed = _CONDITION_LABELS | {ignore_label}
    invalid = sorted({label for label in labels if label not in allowed})
    if invalid:
        raise ValueError(
            f"Unsupported labels in {label_key}: {invalid}; expected positive, negative, or "
            f"{ignore_label!r}"
        )

    eligible = torch.tensor([label in _CONDITION_LABELS for label in labels], dtype=torch.bool)
    if dropout_prob == 0.0:
        kept = eligible
    elif dropout_prob == 1.0:
        kept = torch.zeros_like(eligible)
    else:
        kept = eligible & (torch.rand(len(labels), generator=generator) >= dropout_prob)

    result = dict(batch)
    result[condition_kept_key] = kept
    return result


@dataclass
class AdvantageWeights:
    """Read offline group-relative weights and apply train-time label semantics."""

    loss_weight_key: str = "advantage_loss_weight_global"
    label_key: str = "advantage_label_global"
    ignore_label: str = "ignore"
    disable_weight_when_condition_dropped: bool = True
    condition_kept_key: str = ADVANTAGE_CONDITION_KEPT

    def __post_init__(self) -> None:
        for name in ("loss_weight_key", "label_key", "ignore_label", "condition_kept_key"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.ignore_label in _CONDITION_LABELS:
            raise ValueError("ignore_label cannot be positive or negative")

    def compute_batch_weights(
        self, batch: dict[str, Any], *, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
        if self.label_key not in batch:
            raise ValueError(f"Batch is missing advantage label field {self.label_key!r}")
        if self.loss_weight_key not in batch:
            raise ValueError(f"Batch is missing advantage weight field {self.loss_weight_key!r}")

        labels = _as_label_list(batch[self.label_key], name=self.label_key)
        weights = _as_per_sample_tensor(
            batch[self.loss_weight_key], name=self.loss_weight_key, dtype=torch.float32
        )
        if weights.shape[0] != len(labels):
            raise ValueError(
                f"{self.loss_weight_key} batch size ({weights.shape[0]}) does not match "
                f"{self.label_key} batch size ({len(labels)})"
            )
        if not torch.isfinite(weights).all():
            raise ValueError(f"{self.loss_weight_key} must contain only finite values")
        if (weights < 0).any():
            raise ValueError(f"{self.loss_weight_key} cannot contain negative values")

        allowed = _CONDITION_LABELS | {self.ignore_label}
        invalid = sorted({label for label in labels if label not in allowed})
        if invalid:
            raise ValueError(
                f"Unsupported labels in {self.label_key}: {invalid}; expected positive, negative, "
                f"or {self.ignore_label!r}"
            )

        negative = torch.tensor([label == "negative" for label in labels], device=weights.device)
        ignore = torch.tensor([label == self.ignore_label for label in labels], device=weights.device)
        if negative.any() and not torch.allclose(
            weights[negative], torch.ones_like(weights[negative]), rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"Negative samples in {self.loss_weight_key} must have weight 1.0")

        effective = weights.clone()
        effective[ignore] = 0.0

        dropped = torch.zeros_like(ignore)
        if self.condition_kept_key in batch:
            kept = _as_per_sample_tensor(
                batch[self.condition_kept_key], name=self.condition_kept_key, dtype=torch.bool
            )
            if kept.shape[0] != len(labels):
                raise ValueError(
                    f"{self.condition_kept_key} batch size ({kept.shape[0]}) does not match "
                    f"{self.label_key} batch size ({len(labels)})"
                )
            dropped = ~kept & ~ignore
            if self.disable_weight_when_condition_dropped:
                effective[dropped] = 1.0

        if device is not None:
            effective = effective.to(device=device)

        num_positive = sum(label == "positive" for label in labels)
        num_negative = sum(label == "negative" for label in labels)
        num_ignore = sum(label == self.ignore_label for label in labels)
        stats: dict[str, float | int | bool] = {
            "mean_weight": float(effective.mean().item()),
            "weight_sum": float(effective.sum().item()),
            "num_positive": num_positive,
            "num_negative": num_negative,
            "num_ignore": num_ignore,
            "num_condition_dropped": int(dropped.sum().item()),
            "all_ignore": num_ignore == len(labels),
        }
        return effective, stats


def distributed_weighted_mean(
    per_sample_loss: torch.Tensor,
    weights: torch.Tensor,
    *,
    accelerator: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a DDP-correct weighted mean and the detached global weight sum.

    DDP averages gradients across ranks. Scaling each local numerator by the
    world size and dividing by the globally reduced denominator therefore gives
    the same gradient as a weighted mean over the global batch.
    """

    if per_sample_loss.ndim != 1:
        raise ValueError(f"per_sample_loss must have shape [B], got {tuple(per_sample_loss.shape)}")
    if weights.ndim != 1 or weights.shape != per_sample_loss.shape:
        raise ValueError(
            f"weights must have shape {tuple(per_sample_loss.shape)}, got {tuple(weights.shape)}"
        )
    weights = weights.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and non-negative")

    local_weight_sum = weights.sum()
    if accelerator is None:
        global_weight_sum = local_weight_sum.detach()
        world_size = 1
    else:
        global_weight_sum = accelerator.reduce(local_weight_sum.detach(), reduction="sum")
        world_size = accelerator.num_processes

    if global_weight_sum.item() <= 0:
        return per_sample_loss.sum() * 0.0, global_weight_sum
    loss = world_size * (per_sample_loss * weights).sum() / global_weight_sum
    return loss, global_weight_sum
