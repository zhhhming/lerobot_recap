#!/usr/bin/env python

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .pipeline import ComplementaryDataProcessorStep, ProcessorStepRegistry


_CONDITION_LABELS = {"positive", "negative"}
_NON_CONDITION_LABELS = {"", "ignore"}
_INFERENCE_LABELS = _CONDITION_LABELS | {"none"}


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


@dataclass
@ProcessorStepRegistry.register(name="advantage_condition_processor")
class AdvantageConditionProcessorStep(ComplementaryDataProcessorStep):
    """Deterministically append an advantage label to the main task prompt.

    Random classifier-free dropout intentionally lives outside this processor. During
    training, callers provide ``advantage_condition_kept``. When both the dataset label
    and keep mask are absent, the processor uses the configured inference label.
    """

    label_key: str = "advantage_label_global"
    condition_format: str = "Advantage: {label}"
    inference_label: str = "positive"
    task_key: str = "task"
    condition_kept_key: str = "advantage_condition_kept"

    def __post_init__(self) -> None:
        if not self.label_key:
            raise ValueError("label_key must be non-empty")
        if not self.task_key:
            raise ValueError("task_key must be non-empty")
        if not self.condition_kept_key:
            raise ValueError("condition_kept_key must be non-empty")
        if self.inference_label not in _INFERENCE_LABELS:
            raise ValueError(
                "inference_label must be one of positive, negative, or none, "
                f"got {self.inference_label!r}"
            )
        if "{label}" not in self.condition_format:
            raise ValueError("condition_format must include the {label} placeholder")
        try:
            self.condition_format.format(label="positive")
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError("condition_format must be formattable with {label}") from exc

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        tasks_value = complementary_data.get(self.task_key)
        if tasks_value is None:
            raise ValueError(f"{self.task_key!r} is required for advantage conditioning")

        tasks, _ = _as_list(tasks_value, name=self.task_key)
        if not tasks or not all(isinstance(task, str) for task in tasks):
            raise ValueError(f"{self.task_key!r} must contain one or more strings")
        batch_size = len(tasks)

        label_present = self.label_key in complementary_data
        mask_present = self.condition_kept_key in complementary_data

        if label_present:
            labels, _ = _as_list(complementary_data[self.label_key], name=self.label_key)
            if len(labels) != batch_size:
                raise ValueError(
                    f"{self.label_key} length ({len(labels)}) does not match task batch size "
                    f"({batch_size})"
                )
        elif mask_present:
            # A keep mask marks the training path. Missing training labels must not
            # silently fall back to the deployment-time positive condition.
            labels = [""] * batch_size
        else:
            fallback = "" if self.inference_label == "none" else self.inference_label
            labels = [fallback] * batch_size

        mask_device = None
        if mask_present:
            requested_keep, mask_device = _as_list(
                complementary_data[self.condition_kept_key], name=self.condition_kept_key
            )
            if len(requested_keep) != batch_size:
                raise ValueError(
                    f"{self.condition_kept_key} length ({len(requested_keep)}) does not match "
                    f"task batch size ({batch_size})"
                )
            if not all(isinstance(value, bool) for value in requested_keep):
                raise ValueError(f"{self.condition_kept_key} must contain boolean values")
        else:
            requested_keep = [True] * batch_size

        formatted_tasks: list[str] = []
        effective_keep: list[bool] = []
        for task, label_value, keep in zip(tasks, labels, requested_keep, strict=True):
            if not isinstance(label_value, str):
                raise ValueError(f"{self.label_key} must contain string labels")
            label = label_value.strip()
            if label not in _CONDITION_LABELS | _NON_CONDITION_LABELS:
                raise ValueError(
                    f"Unsupported advantage label {label!r}; expected positive, negative, ignore, or empty"
                )

            should_append = keep and label in _CONDITION_LABELS
            effective_keep.append(should_append)
            if should_append:
                condition = self.condition_format.format(label=label).strip()
                base_task = task.rstrip()
                formatted_tasks.append(f"{base_task}\n{condition}" if base_task else condition)
            else:
                formatted_tasks.append(task)

        result = dict(complementary_data)
        result[self.task_key] = formatted_tasks[0] if isinstance(tasks_value, str) else formatted_tasks
        result[self.condition_kept_key] = torch.tensor(
            effective_keep, dtype=torch.bool, device=mask_device
        )
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "label_key": self.label_key,
            "condition_format": self.condition_format,
            "inference_label": self.inference_label,
            "task_key": self.task_key,
            "condition_kept_key": self.condition_kept_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
