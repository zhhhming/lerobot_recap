#!/usr/bin/env python

"""Real-Time Chunking inference engine."""

from __future__ import annotations

import logging
import math
import time
import traceback
from collections.abc import Callable
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.datasets.subtask_timing import SubtaskSequenceContract
from lerobot.inference_engines.memory_progress_assist import NeroEggMemoryProgressAssist
from lerobot.inference_engines.subtask_time_tracker import (
    SubtaskTimeTracker,
    SubtaskTimeTrackerSnapshot,
)
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
        subtask_sequence_contract: SubtaskSequenceContract | None = None,
        subtask_time_enabled: bool = False,
        subtask_time_clock: Callable[[], float] = time.monotonic,
        memory_progress_assist: NeroEggMemoryProgressAssist | None = None,
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
        self._state_lock = Lock()
        self._inference_lock = Lock()
        self._policy_active = Event()
        self._shutdown_event = Event()
        self._rtc_error = Event()
        self._rtc_thread: Thread | None = None
        self._reset_version = 0
        self._last_latency_s: float | None = None
        self._last_real_delay: int | None = None
        self._last_queue_size: int = 0
        self._last_timing_ms: dict[str, float] = {}
        self._inference_count = 0
        self._phase_info: tuple[str, float] = ("init", time.perf_counter())

        policy_config = getattr(policy, "config", None)
        self._memory_conditioning_enabled = bool(
            getattr(policy_config, "use_memory_conditioning", False)
        )
        self._memory_updates_enabled = self._memory_conditioning_enabled and bool(
            getattr(policy_config, "subtask_generate_at_inference", True)
        )
        self._memory_text_for_next_inference = ""
        self._last_memory_input_text = ""
        self._last_subtask_output_text = ""
        self._memory_source_inference_id: int | None = None
        if memory_progress_assist is not None and not self._memory_updates_enabled:
            raise ValueError(
                "memory_progress_assist requires deployment memory updates to be enabled"
            )
        self._memory_progress_assist = memory_progress_assist

        self._subtask_time_enabled = bool(subtask_time_enabled)
        if self._subtask_time_enabled and subtask_sequence_contract is None:
            raise ValueError(
                "subtask_sequence_contract is required when subtask_time_enabled=True"
            )
        self._subtask_time_clock = subtask_time_clock
        self._subtask_time_tracker = (
            SubtaskTimeTracker(subtask_sequence_contract, clock=subtask_time_clock)
            if self._subtask_time_enabled and subtask_sequence_contract is not None
            else None
        )
        self._subtask_time_updates_enabled = self._subtask_time_enabled and bool(
            getattr(policy_config, "subtask_generate_at_inference", True)
        )
        self._last_subtask_time_input_seconds: float | None = None
        self._subtask_time_capped_indices: set[int] = set()
        if self._subtask_time_enabled and not self._subtask_time_updates_enabled:
            logger.warning(
                "RTC subtask elapsed-time conditioning is enabled but "
                "subtask_generate_at_inference=False; elapsed time will remain invalid."
            )

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
        """Stop inference without changing semantic or runtime state.

        This low-level compatibility API is intentionally distinct from
        :meth:`soft_pause`, which is the safe deploy-session operation.
        """
        self._policy_active.clear()

    def resume(self) -> None:
        with self._state_lock:
            if self._subtask_time_tracker is not None:
                self._subtask_time_tracker.resume()
            self._policy_active.set()

    def soft_pause(self) -> None:
        """Safely pause deployment while freezing confirmed subtask time."""
        self._policy_active.clear()
        with self._state_lock:
            self._reset_version += 1
            self._clear_memory_state_locked()
            if self._subtask_time_tracker is not None:
                self._subtask_time_tracker.pause()
            if self._action_queue is not None:
                self._action_queue.clear()
            with self._obs_lock:
                self._obs_holder["obs"] = None
        self._reset_policy_runtime()
        logger.info("RTC soft pause: runtime cleared and subtask elapsed time frozen.")

    def full_reset(self) -> None:
        """Stop inference and clear all runtime and semantic session state."""
        self._policy_active.clear()
        self._reset_runtime_state()
        logger.info("RTC full reset: runtime and subtask elapsed-time state cleared.")

    def reset(self) -> None:
        """Compatibility reset that preserves the caller-controlled active flag."""
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        with self._state_lock:
            self._reset_version += 1
            self._clear_memory_state_locked()
            self._clear_subtask_time_state_locked()
            if self._action_queue is not None:
                self._action_queue.clear()
            with self._obs_lock:
                self._obs_holder["obs"] = None
        self._reset_policy_runtime()

    def _reset_policy_runtime(self) -> None:
        with self._inference_lock:
            self._policy.reset()
            self._preprocessor.reset()
            self._postprocessor.reset()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        _ = obs_frame
        if self._action_queue is None:
            return None
        return self._action_queue.get()

    def notify_observation(self, obs: dict) -> None:
        with self._obs_lock:
            self._obs_holder["obs"] = obs

    def debug_snapshot(self) -> dict[str, Any]:
        queue = self._action_queue
        with self._state_lock:
            phase_name, phase_start = self._phase_info
            phase_duration_ms = (time.perf_counter() - phase_start) * 1e3
            subtask_time = (
                self._subtask_time_tracker.snapshot()
                if self._subtask_time_tracker is not None
                else None
            )
            return {
                "queue_size": queue.qsize() if queue is not None else 0,
                "last_latency_s": self._last_latency_s,
                "last_real_delay": self._last_real_delay,
                "last_queue_size": self._last_queue_size,
                "last_timing_ms": dict(self._last_timing_ms),
                "inference_count": self._inference_count,
                "active": self._policy_active.is_set(),
                "failed": self.failed,
                "current_phase": phase_name,
                "current_phase_duration_ms": round(phase_duration_ms, 1),
                "memory_text_for_next_inference": self._memory_text_for_next_inference,
                "last_memory_input_text": self._last_memory_input_text,
                "last_subtask_output_text": self._last_subtask_output_text,
                "memory_source_inference_id": self._memory_source_inference_id,
                **self._memory_progress_assist_debug_fields(),
                **self._subtask_time_debug_fields(subtask_time),
            }

    def _clear_memory_state_locked(self) -> None:
        """Clear semantic inference state while the caller holds ``_state_lock``."""
        self._memory_text_for_next_inference = ""
        self._last_memory_input_text = ""
        self._last_subtask_output_text = ""
        self._memory_source_inference_id = None
        if self._memory_progress_assist is not None:
            self._memory_progress_assist.reset()

    def _memory_progress_assist_debug_fields(self) -> dict[str, Any]:
        if self._memory_progress_assist is None:
            return {
                "memory_progress_assist_enabled": False,
                "memory_progress_assist_subtask": None,
                "memory_progress_assist_raw_progress": None,
                "memory_progress_assist_effective_progress": None,
                "memory_progress_assist_reason": "disabled",
                "memory_progress_assist_adjusted": False,
                "memory_progress_assist_forced": False,
            }
        result = self._memory_progress_assist.last_result
        return {
            "memory_progress_assist_enabled": True,
            "memory_progress_assist_subtask": result.subtask_name,
            "memory_progress_assist_raw_progress": result.raw_progress,
            "memory_progress_assist_effective_progress": result.effective_progress,
            "memory_progress_assist_reason": result.reason,
            "memory_progress_assist_adjusted": result.adjusted,
            "memory_progress_assist_forced": result.forced,
        }

    def _clear_subtask_time_state_locked(self) -> None:
        """Clear elapsed-time semantic state while holding ``_state_lock``."""
        if self._subtask_time_tracker is not None:
            self._subtask_time_tracker.full_reset()
        self._last_subtask_time_input_seconds = None
        self._subtask_time_capped_indices.clear()

    def _subtask_time_debug_fields(
        self, snapshot: SubtaskTimeTrackerSnapshot | None
    ) -> dict[str, Any]:
        if snapshot is None:
            return {
                "subtask_time_enabled": False,
                "subtask_time_current_index": None,
                "subtask_time_current_name": None,
                "subtask_time_raw_elapsed_seconds": 0.0,
                "subtask_time_effective_seconds": 0.0,
                "subtask_time_cap_seconds": None,
                "subtask_time_valid": False,
                "subtask_time_running": False,
                "subtask_time_paused": False,
                "subtask_time_last_transition": "disabled",
                "subtask_time_last_rejected_output": "",
                "subtask_time_last_rejection_reason": "",
                "subtask_time_last_input_seconds": None,
            }
        return {
            "subtask_time_enabled": True,
            "subtask_time_current_index": snapshot.current_index,
            "subtask_time_current_name": snapshot.current_name,
            "subtask_time_raw_elapsed_seconds": snapshot.raw_elapsed_seconds,
            "subtask_time_effective_seconds": snapshot.effective_elapsed_seconds,
            "subtask_time_cap_seconds": snapshot.cap_seconds,
            "subtask_time_valid": snapshot.time_valid,
            "subtask_time_running": snapshot.running,
            "subtask_time_paused": snapshot.paused,
            "subtask_time_last_transition": snapshot.last_transition_reason,
            "subtask_time_last_rejected_output": snapshot.last_rejected_output,
            "subtask_time_last_rejection_reason": snapshot.last_rejection_reason,
            "subtask_time_last_input_seconds": self._last_subtask_time_input_seconds,
        }

    def _subtask_output_candidate(self) -> str:
        output = getattr(self._policy, "last_subtask_text", "")
        if output is None:
            return ""
        if not isinstance(output, str):
            raise ValueError(
                "RTC subtask output must be a string from deployment batch size 1; "
                f"got {type(output).__name__}"
            )
        return " ".join(output.split())

    def _rtc_loop(self) -> None:
        try:
            latency_tracker = LatencyTracker()
            time_per_chunk = 1.0 / self._fps
            policy_device = torch.device(self._device)
            consecutive_errors = 0

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    self._phase_info = ("idle_paused", time.perf_counter())
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                with self._state_lock:
                    reset_version = self._reset_version
                    memory_input_text = (
                        self._memory_text_for_next_inference
                        if self._memory_updates_enabled
                        else ""
                    )

                queue = self._action_queue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                if queue is None or obs is None:
                    self._phase_info = ("idle_no_obs", time.perf_counter())
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                if queue.qsize() > self._rtc_queue_threshold:
                    self._phase_info = ("idle_queue_full", time.perf_counter())
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                subtask_time_snapshot: SubtaskTimeTrackerSnapshot | None = None
                with self._state_lock:
                    if (
                        self._shutdown_event.is_set()
                        or not self._policy_active.is_set()
                        or reset_version != self._reset_version
                    ):
                        self._phase_info = ("idle_reset", time.perf_counter())
                        continue
                    if self._subtask_time_tracker is not None:
                        inference_start_monotonic = self._subtask_time_clock()
                        subtask_time_snapshot = self._subtask_time_tracker.snapshot(
                            at_monotonic=inference_start_monotonic
                        )
                        if (
                            subtask_time_snapshot.time_valid
                            and subtask_time_snapshot.current_index is not None
                            and subtask_time_snapshot.cap_seconds is not None
                            and subtask_time_snapshot.raw_elapsed_seconds
                            >= subtask_time_snapshot.cap_seconds
                            and subtask_time_snapshot.current_index
                            not in self._subtask_time_capped_indices
                        ):
                            self._subtask_time_capped_indices.add(
                                subtask_time_snapshot.current_index
                            )
                            logger.warning(
                                "Subtask elapsed time reached deployment cap: index=%d "
                                "subtask=%s raw=%.1fs cap=%.1fs",
                                subtask_time_snapshot.current_index,
                                subtask_time_snapshot.current_name,
                                subtask_time_snapshot.raw_elapsed_seconds,
                                subtask_time_snapshot.cap_seconds,
                            )

                subtask_time_input_seconds = (
                    subtask_time_snapshot.effective_elapsed_seconds
                    if subtask_time_snapshot is not None and subtask_time_snapshot.time_valid
                    else None
                )

                try:
                    current_time = time.perf_counter()
                    idx_before = queue.get_action_index()
                    prev_actions = None

                    latency = latency_tracker.max()
                    delay = math.ceil(latency / time_per_chunk) if latency else 0
                    timings: dict[str, float] = {}

                    with self._inference_lock:
                        stage_start = time.perf_counter()
                        self._phase_info = ("build_frame", stage_start)
                        obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
                        timings["build_frame"] = time.perf_counter() - stage_start

                        stage_start = time.perf_counter()
                        self._phase_info = ("prepare_obs", stage_start)
                        obs_batch = prepare_observation_for_inference(
                            obs_batch,
                            policy_device,
                            self._task,
                            self._robot.robot_type,
                        )
                        obs_batch["task"] = [self._task]
                        if self._memory_conditioning_enabled:
                            memory_valid = bool(memory_input_text)
                            obs_batch["memory_text"] = [memory_input_text]
                            obs_batch["memory_valid"] = [memory_valid]
                            obs_batch["memory_condition_kept"] = [memory_valid]
                        if self._subtask_time_enabled:
                            time_valid = subtask_time_input_seconds is not None
                            obs_batch["subtask_time_seconds"] = [
                                subtask_time_input_seconds if time_valid else 0.0
                            ]
                            obs_batch["subtask_time_valid"] = [time_valid]
                            obs_batch["subtask_time_condition_kept"] = [time_valid]
                        timings["prepare_obs"] = time.perf_counter() - stage_start

                        stage_start = time.perf_counter()
                        self._phase_info = ("preprocess", stage_start)
                        preprocessed = self._preprocessor(obs_batch)
                        timings["preprocess"] = time.perf_counter() - stage_start

                        stage_start = time.perf_counter()
                        self._phase_info = ("leftover", stage_start)
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
                        timings["leftover"] = time.perf_counter() - stage_start

                        stage_start = time.perf_counter()
                        self._phase_info = ("predict", stage_start)
                        actions = self._policy.predict_action_chunk(
                            preprocessed,
                            inference_delay=delay,
                            prev_chunk_left_over=prev_actions,
                        )
                        if actions.ndim == 0 or actions.shape[0] != 1:
                            batch_size = 0 if actions.ndim == 0 else actions.shape[0]
                            raise ValueError(
                                "RTC inference supports deployment batch size 1; "
                                f"got batch size {batch_size}"
                            )
                        subtask_output_candidate = self._subtask_output_candidate()
                        timings["predict"] = time.perf_counter() - stage_start

                        stage_start = time.perf_counter()
                        self._phase_info = ("postprocess", stage_start)
                        original = actions.squeeze(0).clone()
                        processed = self._postprocessor(actions).squeeze(0)
                        timings["postprocess"] = time.perf_counter() - stage_start
                    new_latency = time.perf_counter() - current_time
                    real_delay = max(0, queue.get_action_index() - idx_before)

                    with self._state_lock:
                        if (
                            self._shutdown_event.is_set()
                            or not self._policy_active.is_set()
                            or reset_version != self._reset_version
                        ):
                            self._phase_info = ("idle_reset", time.perf_counter())
                            continue
                        latency_tracker.add(new_latency)
                        stage_start = time.perf_counter()
                        self._phase_info = ("merge", stage_start)
                        queue.merge(original, processed, real_delay, idx_before)
                        if (
                            self._subtask_time_tracker is not None
                            and self._subtask_time_updates_enabled
                        ):
                            committed_time = self._subtask_time_tracker.commit_subtask_output(
                                subtask_output_candidate
                            )
                            if committed_time.last_transition_reason in ("started", "advanced"):
                                logger.info(
                                    "Subtask elapsed-time tracker %s: index=%d subtask=%s",
                                    committed_time.last_transition_reason,
                                    committed_time.current_index,
                                    committed_time.current_name,
                                )
                            elif committed_time.last_transition_reason.startswith("rejected_"):
                                logger.debug(
                                    "Subtask elapsed-time tracker rejected output: transition=%s "
                                    "reason=%s output=%r",
                                    committed_time.last_transition_reason,
                                    committed_time.last_rejection_reason,
                                    committed_time.last_rejected_output,
                                )
                        timings["merge"] = time.perf_counter() - stage_start
                        timings["total"] = new_latency
                        self._last_latency_s = new_latency
                        self._last_real_delay = real_delay
                        self._last_queue_size = queue.qsize()
                        self._last_timing_ms = {key: value * 1e3 for key, value in timings.items()}
                        committed_inference_id = self._inference_count + 1
                        self._last_memory_input_text = memory_input_text
                        self._last_subtask_output_text = subtask_output_candidate
                        next_memory = (
                            subtask_output_candidate if self._memory_updates_enabled else ""
                        )
                        if self._memory_progress_assist is not None:
                            assist_result = self._memory_progress_assist.apply(next_memory)
                            next_memory = assist_result.text
                            if assist_result.forced:
                                logger.warning(
                                    "Nero egg memory progress assist forced progress: "
                                    "subtask=%s raw=%.1f effective=%.1f",
                                    assist_result.subtask_name,
                                    assist_result.raw_progress,
                                    assist_result.effective_progress,
                                )
                            elif assist_result.adjusted:
                                logger.debug(
                                    "Nero egg memory progress assist preserved progress: "
                                    "subtask=%s raw=%.1f effective=%.1f reason=%s",
                                    assist_result.subtask_name,
                                    assist_result.raw_progress,
                                    assist_result.effective_progress,
                                    assist_result.reason,
                                )
                        self._memory_text_for_next_inference = next_memory
                        self._memory_source_inference_id = (
                            committed_inference_id if next_memory else None
                        )
                        self._last_subtask_time_input_seconds = subtask_time_input_seconds
                        self._inference_count = committed_inference_id
                    self._phase_info = ("between_inferences", time.perf_counter())
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
