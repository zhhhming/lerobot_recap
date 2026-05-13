#!/usr/bin/env python

"""Small inference backend interface for policy rollout.

The caller can use the same control-loop code for synchronous policies and
RTC-style asynchronous chunk generation.
"""

from __future__ import annotations

import abc

import torch


class InferenceEngine(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None:
        """Prepare the backend."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Release backend resources."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear policy, processor, and queue state."""

    @abc.abstractmethod
    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Return the next action tensor, or None if no action is ready."""

    def notify_observation(self, obs: dict) -> None:  # noqa: B027
        """Publish the latest processed observation. Default: no-op."""

    def pause(self) -> None:  # noqa: B027
        """Pause background inference. Default: no-op."""

    def resume(self) -> None:  # noqa: B027
        """Resume background inference. Default: no-op."""

    @property
    def ready(self) -> bool:
        return True

    @property
    def failed(self) -> bool:
        return False
