#!/usr/bin/env python

import logging
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .pipeline import ComplementaryDataProcessorStep, ProcessorStepRegistry

logger = logging.getLogger(__name__)


def _to_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return [value.item()]
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass
@ProcessorStepRegistry.register(name="subtask_text_processor")
class SubtaskTextProcessorStep(ComplementaryDataProcessorStep):
    """Build the AR subtask text segment consumed by the tokenizer."""

    subtask_key: str = "subtask"
    progress_key: str = "subtask_progress"

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        if self.subtask_key not in complementary_data:
            return complementary_data

        subtasks = _to_list(complementary_data.get(self.subtask_key))
        progress_value = complementary_data.get(self.progress_key)
        if progress_value is None:
            if any(isinstance(s, str) and s.strip() for s in subtasks):
                logger.warning(
                    "subtask_progress is missing for non-empty subtask labels; using progress=1.0"
                )
            progresses = [1.0] * len(subtasks)
        else:
            progresses = _to_list(progress_value)
            if len(progresses) == 1 and len(subtasks) > 1:
                progresses = progresses * len(subtasks)
            elif len(progresses) != len(subtasks):
                raise ValueError(
                    f"subtask_progress length ({len(progresses)}) does not match subtask length "
                    f"({len(subtasks)})"
                )

        formatted: list[str] = []
        for subtask, progress in zip(subtasks, progresses, strict=True):
            if not isinstance(subtask, str) or not subtask.strip():
                formatted.append("")
                continue

            progress_float = round(float(progress), 1)
            progress_float = min(1.0, max(0.0, progress_float))
            formatted.append(f"Subtask: {subtask.strip()}; Progress: {progress_float:.1f}\n")

        new_complementary_data = dict(complementary_data)
        if isinstance(complementary_data.get(self.subtask_key), str):
            new_complementary_data[self.subtask_key] = formatted[0]
        else:
            new_complementary_data[self.subtask_key] = formatted
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
