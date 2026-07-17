#!/usr/bin/env python

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .pipeline import ComplementaryDataProcessorStep, ProcessorStepRegistry
from .subtask_processor import format_subtask_output


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
            f"{name} length ({len(values)}) does not match task batch size ({batch_size})"
        )


def _normalize_prediction_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must contain strings")
    return " ".join(value.split())


@dataclass
@ProcessorStepRegistry.register(name="memory_condition_processor")
class MemoryConditionProcessorStep(ComplementaryDataProcessorStep):
    """Deterministically append a previous subtask/progress output to the main prompt."""

    task_key: str = "task"
    memory_text_key: str = "memory_text"
    memory_subtask_key: str = "memory_subtask"
    memory_progress_key: str = "memory_subtask_progress"
    memory_valid_key: str = "memory_valid"
    condition_kept_key: str = "memory_condition_kept"
    memory_prefix: str = "Memory: "

    def __post_init__(self) -> None:
        for name in (
            "task_key",
            "memory_text_key",
            "memory_subtask_key",
            "memory_progress_key",
            "memory_valid_key",
            "condition_kept_key",
            "memory_prefix",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        tasks_value = complementary_data.get(self.task_key)
        if tasks_value is None:
            raise ValueError(f"{self.task_key!r} is required for memory conditioning")
        tasks, _ = _as_list(tasks_value, name=self.task_key)
        if not tasks or not all(isinstance(task, str) for task in tasks):
            raise ValueError(f"{self.task_key!r} must contain one or more strings")
        batch_size = len(tasks)

        has_prediction = self.memory_text_key in complementary_data
        has_training = any(
            key in complementary_data for key in (self.memory_subtask_key, self.memory_progress_key)
        )
        if has_prediction and has_training:
            raise ValueError("Provide exactly one memory source: memory_text or training subtask/progress")

        result = dict(complementary_data)
        if not has_prediction and not has_training:
            result[self.condition_kept_key] = torch.zeros(batch_size, dtype=torch.bool)
            return result

        if self.memory_valid_key not in complementary_data:
            raise ValueError(f"{self.memory_valid_key} is required when a memory source is present")
        valid, valid_device = _as_list(
            complementary_data[self.memory_valid_key], name=self.memory_valid_key
        )
        _require_batch_size(valid, name=self.memory_valid_key, batch_size=batch_size)
        if not all(isinstance(value, bool) for value in valid):
            raise ValueError(f"{self.memory_valid_key} must contain boolean values")

        keep_device = valid_device
        if self.condition_kept_key in complementary_data:
            requested_keep, condition_device = _as_list(
                complementary_data[self.condition_kept_key], name=self.condition_kept_key
            )
            _require_batch_size(requested_keep, name=self.condition_kept_key, batch_size=batch_size)
            if not all(isinstance(value, bool) for value in requested_keep):
                raise ValueError(f"{self.condition_kept_key} must contain boolean values")
            keep_device = condition_device or keep_device
        else:
            requested_keep = [True] * batch_size

        if has_prediction:
            source_values, _ = _as_list(
                complementary_data[self.memory_text_key], name=self.memory_text_key
            )
            _require_batch_size(source_values, name=self.memory_text_key, batch_size=batch_size)
            memory_texts = [
                _normalize_prediction_text(value, name=self.memory_text_key) for value in source_values
            ]
        else:
            missing = [
                key
                for key in (self.memory_subtask_key, self.memory_progress_key)
                if key not in complementary_data
            ]
            if missing:
                raise ValueError(f"Training memory source is missing required fields: {', '.join(missing)}")
            subtasks, _ = _as_list(
                complementary_data[self.memory_subtask_key], name=self.memory_subtask_key
            )
            progresses, _ = _as_list(
                complementary_data[self.memory_progress_key], name=self.memory_progress_key
            )
            _require_batch_size(subtasks, name=self.memory_subtask_key, batch_size=batch_size)
            _require_batch_size(progresses, name=self.memory_progress_key, batch_size=batch_size)
            memory_texts = [
                format_subtask_output(subtask, progress)
                for subtask, progress in zip(subtasks, progresses, strict=True)
            ]

        formatted_tasks: list[str] = []
        effective_keep: list[bool] = []
        for task, memory_text, is_valid, keep in zip(
            tasks, memory_texts, valid, requested_keep, strict=True
        ):
            should_append = is_valid and keep and bool(memory_text)
            effective_keep.append(should_append)
            if should_append:
                condition = f"{self.memory_prefix}{memory_text}"
                base_task = task.rstrip()
                formatted_tasks.append(f"{base_task}\n{condition}" if base_task else condition)
            else:
                formatted_tasks.append(task)

        result[self.task_key] = formatted_tasks[0] if isinstance(tasks_value, str) else formatted_tasks
        result[self.condition_kept_key] = torch.tensor(
            effective_keep, dtype=torch.bool, device=keep_device
        )
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "memory_text_key": self.memory_text_key,
            "memory_subtask_key": self.memory_subtask_key,
            "memory_progress_key": self.memory_progress_key,
            "memory_valid_key": self.memory_valid_key,
            "condition_kept_key": self.condition_kept_key,
            "memory_prefix": self.memory_prefix,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
