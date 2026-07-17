#!/usr/bin/env python

"""Deterministic elapsed-time text conditioning for policy main prompts."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .pipeline import ComplementaryDataProcessorStep, ProcessorStepRegistry


def _as_list(value: Any, *, name: str) -> tuple[list[Any], torch.device | None]:
    device = value.device if isinstance(value, torch.Tensor) else None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return [value.item()], device
        if value.ndim == 1:
            return value.detach().cpu().tolist(), device
        if value.ndim == 2 and value.shape[-1] == 1:
            return value.detach().cpu().squeeze(-1).tolist(), device
        raise ValueError(f"{name} must have shape [B] or [B, 1], got {tuple(value.shape)}")
    if isinstance(value, (list, tuple)):
        return list(value), device
    return [value], device


def _require_batch_size(values: list[Any], *, name: str, batch_size: int) -> None:
    if len(values) != batch_size:
        raise ValueError(
            f"{name} batch size ({len(values)}) does not match task batch size ({batch_size})"
        )


def _normalize_seconds(seconds: Any) -> float:
    if isinstance(seconds, torch.Tensor):
        if seconds.numel() != 1:
            raise ValueError(
                "subtask elapsed time tensor must contain exactly one element, "
                f"got shape {tuple(seconds.shape)}"
            )
        seconds = seconds.detach().cpu().item()
    if isinstance(seconds, bool) or not isinstance(seconds, numbers.Real):
        raise ValueError("subtask elapsed time must be a real numeric scalar")
    value = float(seconds)
    if not math.isfinite(value):
        raise ValueError(f"subtask elapsed time must be finite, got {seconds!r}")
    if value < 0:
        raise ValueError(f"subtask elapsed time must be non-negative, got {seconds!r}")
    return 0.0 if value == 0.0 else value


def _format_seconds(seconds: Any) -> str:
    return f"{_normalize_seconds(seconds):.1f}s"


def format_subtask_elapsed_time(seconds: Any) -> str:
    """Format one elapsed-time value using the canonical prompt contract."""

    return f"Subtask elapsed time: {_format_seconds(seconds)}"


@dataclass
@ProcessorStepRegistry.register(name="subtask_time_condition_processor")
class SubtaskTimeConditionProcessorStep(ComplementaryDataProcessorStep):
    """Append an already sampled elapsed-time condition to the main prompt."""

    task_key: str = "task"
    seconds_key: str = "subtask_time_seconds"
    valid_key: str = "subtask_time_valid"
    condition_kept_key: str = "subtask_time_condition_kept"

    def __post_init__(self) -> None:
        for name in (
            "task_key",
            "seconds_key",
            "valid_key",
            "condition_kept_key",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        tasks_value = complementary_data.get(self.task_key)
        if tasks_value is None:
            raise ValueError(f"{self.task_key!r} is required for subtask-time conditioning")
        tasks, _ = _as_list(tasks_value, name=self.task_key)
        if not tasks or not all(isinstance(task, str) for task in tasks):
            raise ValueError(f"{self.task_key!r} must contain one or more strings")
        batch_size = len(tasks)

        source_keys = (self.seconds_key, self.valid_key, self.condition_kept_key)
        present_keys = [key for key in source_keys if key in complementary_data]
        result = dict(complementary_data)
        if not present_keys:
            result[self.condition_kept_key] = torch.zeros(batch_size, dtype=torch.bool)
            return result
        missing_keys = [key for key in source_keys if key not in complementary_data]
        if missing_keys:
            raise ValueError(
                "Subtask-time source is missing required fields: " + ", ".join(missing_keys)
            )

        seconds, _ = _as_list(complementary_data[self.seconds_key], name=self.seconds_key)
        valid, valid_device = _as_list(complementary_data[self.valid_key], name=self.valid_key)
        requested_keep, keep_device = _as_list(
            complementary_data[self.condition_kept_key], name=self.condition_kept_key
        )
        for values, name in (
            (seconds, self.seconds_key),
            (valid, self.valid_key),
            (requested_keep, self.condition_kept_key),
        ):
            _require_batch_size(values, name=name, batch_size=batch_size)
        if not all(isinstance(value, bool) for value in valid):
            raise ValueError(f"{self.valid_key} must contain boolean values")
        if not all(isinstance(value, bool) for value in requested_keep):
            raise ValueError(f"{self.condition_kept_key} must contain boolean values")
        normalized_seconds = [_normalize_seconds(value) for value in seconds]

        formatted_tasks: list[str] = []
        effective_keep: list[bool] = []
        for task, elapsed, is_valid, keep in zip(
            tasks, normalized_seconds, valid, requested_keep, strict=True
        ):
            should_append = is_valid and keep
            effective_keep.append(should_append)
            if should_append:
                condition = format_subtask_elapsed_time(elapsed)
                base_task = task.rstrip()
                formatted_tasks.append(f"{base_task}\n{condition}" if base_task else condition)
            else:
                formatted_tasks.append(task)

        if isinstance(tasks_value, str):
            result[self.task_key] = formatted_tasks[0]
        elif isinstance(tasks_value, tuple):
            result[self.task_key] = tuple(formatted_tasks)
        else:
            result[self.task_key] = formatted_tasks
        result[self.condition_kept_key] = torch.tensor(
            effective_keep,
            dtype=torch.bool,
            device=keep_device or valid_device,
        )
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "seconds_key": self.seconds_key,
            "valid_key": self.valid_key,
            "condition_kept_key": self.condition_kept_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
