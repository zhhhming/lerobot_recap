#!/usr/bin/env python

"""Synchronous inference engine."""

from __future__ import annotations

from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine


class SyncInferenceEngine(InferenceEngine):
    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def reset(self) -> None:
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if obs_frame is None:
            return None

        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation,
                self._device,
                self._task,
                self._robot_type,
            )
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)

        action_tensor = action.squeeze(0).cpu()
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])
