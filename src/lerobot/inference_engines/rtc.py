#!/usr/bin/env python

"""Real-Time Chunking inference engine."""

from __future__ import annotations

import logging
import math
import time
import traceback
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionQueue, LatencyTracker, reanchor_relative_rtc_prefix
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline, RelativeActionsProcessorStep

from .base import InferenceEngine
from .robot_wrapper import ThreadSafeRobot

logger = logging.getLogger(__name__)

_RTC_IDLE_SLEEP_S = 0.01
_RTC_ERROR_RETRY_DELAY_S = 0.5
_RTC_MAX_CONSECUTIVE_ERRORS = 10
_RTC_JOIN_TIMEOUT_S = 3.0


class RTCInferenceEngine(InferenceEngine):
    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        rtc_config: RTCConfig,
        hw_features: dict,
        task: str,
        fps: float,
        device: str | None,
        rtc_queue_threshold: int = 40,
        shutdown_event: Event | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._rtc_config = rtc_config
        self._hw_features = hw_features
        self._task = task
        self._fps = fps
        self._device = device or "cpu"
        self._rtc_queue_threshold = rtc_queue_threshold
        self._global_shutdown_event = shutdown_event

        self._action_queue: ActionQueue | None = None
        self._obs_holder: dict[str, Any] = {}
        self._obs_lock = Lock()
        self._policy_active = Event()
        self._shutdown_event = Event()
        self._rtc_error = Event()
        self._rtc_thread: Thread | None = None

        self._relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        self._normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )

    @property
    def failed(self) -> bool:
        return self._rtc_error.is_set()

    def start(self) -> None:
        self._action_queue = ActionQueue(self._rtc_config)
        self._obs_holder = {"obs": None}
        self._shutdown_event.clear()
        self._rtc_thread = Thread(target=self._rtc_loop, daemon=True, name="RTCInference")
        self._rtc_thread.start()

    def stop(self) -> None:
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._rtc_thread is not None and self._rtc_thread.is_alive():
            self._rtc_thread.join(timeout=_RTC_JOIN_TIMEOUT_S)
            if self._rtc_thread.is_alive():
                logger.warning("RTC thread did not join within %.1fs", _RTC_JOIN_TIMEOUT_S)
        self._rtc_thread = None

    def pause(self) -> None:
        self._policy_active.clear()

    def resume(self) -> None:
        self._policy_active.set()

    def reset(self) -> None:
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        if self._action_queue is not None:
            self._action_queue.clear()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        _ = obs_frame
        if self._action_queue is None:
            return None
        return self._action_queue.get()

    def notify_observation(self, obs: dict) -> None:
        with self._obs_lock:
            self._obs_holder["obs"] = obs

    def _rtc_loop(self) -> None:
        try:
            latency_tracker = LatencyTracker()
            time_per_chunk = 1.0 / self._fps
            policy_device = torch.device(self._device)
            consecutive_errors = 0

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                queue = self._action_queue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                if queue is None or obs is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                if queue.qsize() > self._rtc_queue_threshold:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                try:
                    current_time = time.perf_counter()
                    idx_before = queue.get_action_index()
                    prev_actions = None

                    latency = latency_tracker.max()
                    delay = math.ceil(latency / time_per_chunk) if latency else 0

                    obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
                    obs_batch = prepare_observation_for_inference(
                        obs_batch,
                        policy_device,
                        self._task,
                        self._robot.robot_type,
                    )
                    obs_batch["task"] = [self._task]

                    preprocessed = self._preprocessor(obs_batch)

                    if self._relative_step is not None:
                        prev_abs = queue.get_processed_left_over()
                        raw_state = self._relative_step._last_state
                        if prev_abs is not None and prev_abs.numel() > 0 and raw_state is not None:
                            prev_actions = reanchor_relative_rtc_prefix(
                                prev_actions_absolute=prev_abs,
                                current_state=raw_state,
                                relative_step=self._relative_step,
                                normalizer_step=self._normalizer_step,
                                policy_device=policy_device,
                            )
                    else:
                        prev_actions = queue.get_left_over()

                    actions = self._policy.predict_action_chunk(
                        preprocessed,
                        inference_delay=delay,
                        prev_chunk_left_over=prev_actions,
                    )

                    original = actions.squeeze(0).clone()
                    processed = self._postprocessor(actions).squeeze(0)
                    new_latency = time.perf_counter() - current_time
                    real_delay = max(0, queue.get_action_index() - idx_before)

                    latency_tracker.add(new_latency)
                    queue.merge(original, processed, real_delay, idx_before)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        "RTC inference error (%d/%d): %s",
                        consecutive_errors,
                        _RTC_MAX_CONSECUTIVE_ERRORS,
                        e,
                    )
                    logger.debug(traceback.format_exc())
                    if consecutive_errors >= _RTC_MAX_CONSECUTIVE_ERRORS:
                        raise
                    time.sleep(_RTC_ERROR_RETRY_DELAY_S)
        except Exception as e:
            logger.error("Fatal error in RTC thread: %s", e)
            logger.error(traceback.format_exc())
            self._rtc_error.set()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
