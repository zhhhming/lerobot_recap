#!/usr/bin/env python

"""Transactional elapsed-time tests for the asynchronous RTC inference engine."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import lerobot.inference_engines.rtc as rtc_module
from lerobot.datasets.subtask_timing import (
    SubtaskSegmentStats,
    SubtaskSequenceContract,
    normalize_subtask_name,
)
from lerobot.inference_engines.memory_progress_assist import NeroEggMemoryProgressAssist
from lerobot.inference_engines.rtc import RTCInferenceEngine
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor.memory_processor import MemoryConditionProcessorStep
from lerobot.processor.subtask_time_processor import SubtaskTimeConditionProcessorStep


def _copy_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


class _FakeClock:
    def __init__(self, trace: list[str] | None = None) -> None:
        self.now = 0.0
        self.calls = 0
        self.trace = trace

    def __call__(self) -> float:
        self.calls += 1
        if self.trace is not None:
            self.trace.append("clock")
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _contract() -> SubtaskSequenceContract:
    names = ("First.", "Second.", "Third.")
    return SubtaskSequenceContract(
        fps=30.0,
        ordered_subtasks=tuple(
            SubtaskSegmentStats(
                canonical_name=name,
                normalized_name=normalize_subtask_name(name),
                max_elapsed_seconds=5.0 + index,
                deployment_cap_seconds=10.0 + index,
            )
            for index, name in enumerate(names)
        ),
    )


class _FakePreprocessor:
    def __init__(self, *, use_memory: bool, use_time: bool, trace: list[str]) -> None:
        self.steps = []
        if use_memory:
            self.steps.append(MemoryConditionProcessorStep())
        if use_time:
            self.steps.append(SubtaskTimeConditionProcessorStep())
        self.trace = trace
        self.reset_calls = 0

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.trace.append("preprocess")
        result = dict(batch)
        for step in self.steps:
            result = step.complementary_data(result)
        return result

    def reset(self) -> None:
        self.reset_calls += 1


class _FakePostprocessor:
    def __init__(self) -> None:
        self.call_count = 0
        self.fail_calls: set[int] = set()
        self.failure_event = threading.Event()
        self.reset_calls = 0

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        if self.call_count in self.fail_calls:
            self.failure_event.set()
            raise RuntimeError("postprocess failure")
        return actions

    def reset(self) -> None:
        self.reset_calls += 1


class _FakePolicy:
    def __init__(
        self,
        *,
        outputs: list[str | list[str]],
        use_memory: bool,
        use_time: bool,
        generate_subtask: bool,
        trace: list[str],
    ) -> None:
        self.config = SimpleNamespace(
            type="pi0",
            use_memory_conditioning=use_memory,
            use_subtask_time_conditioning=use_time,
            subtask_generate_at_inference=generate_subtask,
        )
        self.outputs = outputs
        self.trace = trace
        self.inputs: list[dict[str, Any]] = []
        self.last_subtask_text: str | list[str] = ""
        self.call_count = 0
        self.fail_calls: set[int] = set()
        self.failure_event = threading.Event()
        self.block_calls: set[int] = set()
        self.block_entered = threading.Event()
        self.block_release = threading.Event()
        self.reset_calls = 0

    def predict_action_chunk(self, batch: dict[str, Any], **_: Any) -> torch.Tensor:
        self.call_count += 1
        self.trace.append("predict")
        self.inputs.append(_copy_value(batch))
        output = self.outputs[min(self.call_count - 1, len(self.outputs) - 1)]
        self.last_subtask_text = output

        if self.call_count in self.block_calls:
            self.block_entered.set()
            if not self.block_release.wait(timeout=3.0):
                raise TimeoutError("test did not release blocked prediction")
        if self.call_count in self.fail_calls:
            self.failure_event.set()
            raise RuntimeError("predict failure")

        batch_size = len(output) if isinstance(output, list) else 1
        return torch.zeros(batch_size, 1, 2)

    def reset(self) -> None:
        self.reset_calls += 1
        self.last_subtask_text = ""


@pytest.fixture(autouse=True)
def _patch_observation_path(monkeypatch):
    def build(_features, obs, prefix):
        obs["trace"].append("build")
        return {f"{prefix}.state": obs["state"], "trace": obs["trace"]}

    def prepare(batch, _device, task, robot_type):
        batch["trace"].append("prepare")
        return {**batch, "task": task, "robot_type": robot_type}

    monkeypatch.setattr(rtc_module, "build_dataset_frame", build)
    monkeypatch.setattr(rtc_module, "prepare_observation_for_inference", prepare)
    monkeypatch.setattr(rtc_module, "_RTC_ERROR_RETRY_DELAY_S", 0.01)


def _make_engine(
    *,
    outputs: list[str | list[str]],
    use_time: bool = True,
    use_memory: bool = True,
    generate_subtask: bool = True,
    clock: _FakeClock | None = None,
    trace: list[str] | None = None,
    contract: SubtaskSequenceContract | None = None,
    memory_progress_assist: NeroEggMemoryProgressAssist | None = None,
) -> tuple[RTCInferenceEngine, _FakePolicy, _FakePreprocessor, _FakePostprocessor, _FakeClock]:
    trace = [] if trace is None else trace
    clock = _FakeClock(trace) if clock is None else clock
    policy = _FakePolicy(
        outputs=outputs,
        use_memory=use_memory,
        use_time=use_time,
        generate_subtask=generate_subtask,
        trace=trace,
    )
    preprocessor = _FakePreprocessor(use_memory=use_memory, use_time=use_time, trace=trace)
    postprocessor = _FakePostprocessor()
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=SimpleNamespace(robot_type="fake_robot"),
        rtc_config=RTCConfig(enabled=True),
        hw_features={},
        task="main task",
        fps=30,
        device="cpu",
        rtc_queue_threshold=0,
        subtask_sequence_contract=(contract or _contract()) if use_time else None,
        subtask_time_enabled=use_time,
        subtask_time_clock=clock,
        memory_progress_assist=memory_progress_assist,
    )
    return engine, policy, preprocessor, postprocessor, clock


def _start_engine(engine: RTCInferenceEngine, trace: list[str]) -> None:
    engine.start()
    engine.notify_observation({"state": torch.zeros(2), "trace": trace})
    engine.resume()


def _wait_until(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("condition was not reached before timeout")


def _queue_size(engine: RTCInferenceEngine) -> int:
    queue = engine._action_queue
    return 0 if queue is None else queue.qsize()


def _consume_committed_action(engine: RTCInferenceEngine) -> None:
    _wait_until(lambda: _queue_size(engine) == 1)
    assert engine.get_action(None) is not None


@pytest.mark.parametrize("policy_type", ["pi0", "pi05"])
def test_first_commit_enables_time_for_the_following_inference(policy_type: str) -> None:
    trace: list[str] = []
    output = "Subtask: First.; Progress: 0.2"
    engine, policy, _, _, clock = _make_engine(outputs=[output, output], trace=trace)
    policy.config.type = policy_type

    try:
        _start_engine(engine, trace)
        _wait_until(lambda: policy.call_count == 1)
        _wait_until(lambda: _queue_size(engine) == 1)

        first_input = policy.inputs[0]
        assert first_input["subtask_time_seconds"] == [0.0]
        assert first_input["subtask_time_valid"] == [False]
        assert not first_input["subtask_time_condition_kept"].item()
        assert "Subtask elapsed time:" not in first_input["task"][0]

        first_snapshot = engine.debug_snapshot()
        assert first_snapshot["subtask_time_current_index"] == 0
        assert first_snapshot["subtask_time_last_input_seconds"] is None

        clock.advance(1.25)
        _consume_committed_action(engine)
        _wait_until(lambda: policy.call_count == 2)

        second_input = policy.inputs[1]
        assert second_input["subtask_time_seconds"] == [pytest.approx(1.25)]
        assert second_input["subtask_time_valid"] == [True]
        assert second_input["subtask_time_condition_kept"].item()
        assert second_input["task"][0].endswith("Subtask elapsed time: 1.2s")
    finally:
        engine.stop()


def test_inference_start_samples_once_and_latency_and_queue_wait_count() -> None:
    trace: list[str] = []
    output = "Subtask: First.; Progress: 0.2"
    engine, policy, _, _, clock = _make_engine(outputs=[output, output], trace=trace)

    try:
        _start_engine(engine, trace)
        _wait_until(lambda: _queue_size(engine) == 1)

        clock.advance(4.25)
        queue_wait_snapshot = engine.debug_snapshot()
        assert policy.call_count == 1
        assert queue_wait_snapshot["subtask_time_raw_elapsed_seconds"] == pytest.approx(4.25)

        policy.block_calls.add(2)
        trace.clear()
        _consume_committed_action(engine)
        _wait_until(policy.block_entered.is_set)

        assert trace[:5] == ["clock", "build", "prepare", "preprocess", "predict"]
        assert trace[:5].count("clock") == 1
        assert policy.inputs[1]["subtask_time_seconds"] == [pytest.approx(4.25)]

        clock.advance(2.0)
        policy.block_release.set()
        _wait_until(lambda: _queue_size(engine) == 1)
        latency_snapshot = engine.debug_snapshot()
        assert latency_snapshot["subtask_time_raw_elapsed_seconds"] == pytest.approx(6.25)
        assert latency_snapshot["subtask_time_last_input_seconds"] == pytest.approx(4.25)
    finally:
        policy.block_release.set()
        engine.stop()


@pytest.mark.parametrize("failure_stage", ["predict", "postprocess", "merge"])
def test_failed_transaction_does_not_advance_time_tracker(failure_stage: str) -> None:
    output_a = "Subtask: First.; Progress: 0.2"
    output_b = "Subtask: Second.; Progress: 0.1"
    engine, policy, _, postprocessor, _ = _make_engine(outputs=[output_a, output_b])

    try:
        _start_engine(engine, policy.trace)
        _wait_until(lambda: _queue_size(engine) == 1)

        failure_event = threading.Event()
        if failure_stage == "predict":
            policy.fail_calls.add(2)
            failure_event = policy.failure_event
        elif failure_stage == "postprocess":
            postprocessor.fail_calls.add(2)
            failure_event = postprocessor.failure_event
        else:
            queue = engine._action_queue
            assert queue is not None

            def fail_merge(*_args, **_kwargs):
                failure_event.set()
                raise RuntimeError("merge failure")

            queue.merge = fail_merge

        _consume_committed_action(engine)
        _wait_until(failure_event.is_set)
        engine.pause()

        snapshot = engine.debug_snapshot()
        assert snapshot["inference_count"] == 1
        assert snapshot["subtask_time_current_index"] == 0
        assert snapshot["subtask_time_current_name"] == "First."
        assert snapshot["last_subtask_output_text"] == output_a
    finally:
        engine.stop()


def test_reset_version_race_discards_candidate_and_clears_tracker() -> None:
    output_a = "Subtask: First.; Progress: 0.2"
    output_b = "Subtask: Second.; Progress: 0.1"
    engine, policy, preprocessor, postprocessor, _ = _make_engine(outputs=[output_a, output_b])

    try:
        _start_engine(engine, policy.trace)
        _wait_until(lambda: _queue_size(engine) == 1)
        policy.block_calls.add(2)
        _consume_committed_action(engine)
        _wait_until(policy.block_entered.is_set)

        engine.pause()
        reset_version = engine._reset_version
        reset_thread = threading.Thread(target=engine.reset)
        reset_thread.start()
        _wait_until(lambda: engine._reset_version == reset_version + 1)
        policy.block_release.set()
        reset_thread.join(timeout=3.0)
        assert not reset_thread.is_alive()

        snapshot = engine.debug_snapshot()
        assert snapshot["inference_count"] == 1
        assert snapshot["subtask_time_current_index"] is None
        assert snapshot["subtask_time_effective_seconds"] == 0.0
        assert snapshot["subtask_time_last_input_seconds"] is None
        assert snapshot["last_subtask_output_text"] == ""
        assert policy.reset_calls == 1
        assert preprocessor.reset_calls == 1
        assert postprocessor.reset_calls == 1
    finally:
        policy.block_release.set()
        engine.stop()


def test_history_and_time_debug_snapshot_commit_atomically() -> None:
    outputs = [
        "Subtask: First.; Progress: 0.2",
        "Subtask: Second.; Progress: 0.1",
        "Subtask: Third.; Progress: 0.1",
    ]
    engine, policy, _, _, clock = _make_engine(outputs=outputs)
    stop_reader = threading.Event()
    snapshots: list[tuple[int, str, str, int | None, float | None]] = []

    def read_snapshots() -> None:
        while not stop_reader.is_set():
            snapshot = engine.debug_snapshot()
            snapshots.append(
                (
                    snapshot["inference_count"],
                    snapshot["last_subtask_output_text"],
                    snapshot["memory_text_for_next_inference"],
                    snapshot["subtask_time_current_index"],
                    snapshot["subtask_time_last_input_seconds"],
                )
            )

    reader = threading.Thread(target=read_snapshots)
    try:
        _start_engine(engine, policy.trace)
        reader.start()
        for expected_count in range(1, len(outputs) + 1):
            _wait_until(lambda: engine.debug_snapshot()["inference_count"] == expected_count)
            if expected_count < len(outputs):
                clock.advance(1.0)
                _consume_committed_action(engine)

        stop_reader.set()
        reader.join(timeout=3.0)
        assert not reader.is_alive()

        allowed = {
            (0, "", "", None, None),
            (1, outputs[0], outputs[0], 0, None),
            (2, outputs[1], outputs[1], 1, 1.0),
            (3, outputs[2], outputs[2], 2, 1.0),
        }
        assert snapshots
        assert set(snapshots) <= allowed
    finally:
        stop_reader.set()
        reader.join(timeout=3.0)
        engine.stop()


def test_time_disabled_injects_no_fields_and_maintains_no_tracker() -> None:
    output = "Subtask: First.; Progress: 0.2"
    engine, policy, _, _, _ = _make_engine(outputs=[output], use_time=False)

    try:
        _start_engine(engine, policy.trace)
        _wait_until(lambda: _queue_size(engine) == 1)
        assert not any(key.startswith("subtask_time_") for key in policy.inputs[0])

        snapshot = engine.debug_snapshot()
        assert snapshot["subtask_time_enabled"] is False
        assert snapshot["subtask_time_current_index"] is None
        assert snapshot["subtask_time_last_input_seconds"] is None
        assert engine._subtask_time_tracker is None
    finally:
        engine.stop()


def test_nero_egg_memory_assist_and_subtask_time_run_independently_together() -> None:
    trace: list[str] = []
    clock = _FakeClock(trace)
    frying = "Start frying the eggs."
    output_point_seven = f"Subtask: {frying}; Progress: 0.7"
    output_point_eight = f"Subtask: {frying}; Progress: 0.8"
    frying_contract = SubtaskSequenceContract(
        fps=30.0,
        ordered_subtasks=(
            SubtaskSegmentStats(
                canonical_name=frying,
                normalized_name=normalize_subtask_name(frying),
                max_elapsed_seconds=10.0,
                deployment_cap_seconds=15.0,
            ),
        ),
    )
    assist = NeroEggMemoryProgressAssist(clock=clock)
    engine, policy, _, _, _ = _make_engine(
        outputs=[output_point_seven, output_point_seven, output_point_seven],
        clock=clock,
        trace=trace,
        contract=frying_contract,
        memory_progress_assist=assist,
    )

    try:
        _start_engine(engine, trace)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 1)
        clock.advance(6.0)
        _consume_committed_action(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 2)

        snapshot = engine.debug_snapshot()
        assert snapshot["subtask_time_current_name"] == frying
        assert snapshot["subtask_time_last_input_seconds"] == pytest.approx(6.0)
        assert snapshot["memory_progress_assist_forced"] is True
        assert snapshot["memory_text_for_next_inference"] == output_point_eight

        _consume_committed_action(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 3)
        assert policy.inputs[2]["memory_text"] == [output_point_eight]
        assert policy.inputs[2]["subtask_time_valid"] == [True]
    finally:
        engine.stop()


def test_generation_disabled_keeps_time_invalid_and_warns(caplog) -> None:
    output = "Subtask: First.; Progress: 0.2"
    with caplog.at_level("WARNING"):
        engine, policy, _, _, _ = _make_engine(
            outputs=[output, output],
            generate_subtask=False,
        )
    assert "subtask_generate_at_inference=False" in caplog.text

    try:
        _start_engine(engine, policy.trace)
        _wait_until(lambda: _queue_size(engine) == 1)
        _consume_committed_action(engine)
        _wait_until(lambda: policy.call_count == 2)

        assert all(item["subtask_time_valid"] == [False] for item in policy.inputs)
        snapshot = engine.debug_snapshot()
        assert snapshot["subtask_time_current_index"] is None
        assert snapshot["subtask_time_last_input_seconds"] is None
    finally:
        engine.stop()


def test_multi_sample_output_does_not_commit_time(caplog) -> None:
    engine, policy, _, _, _ = _make_engine(
        outputs=[["Subtask: First.; Progress: 0.1", "Subtask: Second.; Progress: 0.1"]]
    )

    try:
        _start_engine(engine, policy.trace)
        _wait_until(lambda: policy.call_count >= 1)
        _wait_until(lambda: "batch size 1" in caplog.text)
        engine.pause()
        snapshot = engine.debug_snapshot()
        assert snapshot["inference_count"] == 0
        assert snapshot["subtask_time_current_index"] is None
    finally:
        engine.stop()


def test_soft_pause_freezes_time_and_clears_runtime_without_clearing_tracker() -> None:
    output = "Subtask: First.; Progress: 0.2"
    engine, policy, preprocessor, postprocessor, clock = _make_engine(outputs=[output, output])

    try:
        _start_engine(engine, policy.trace)
        _wait_until(lambda: _queue_size(engine) == 1)
        clock.advance(3.5)
        engine.notify_observation({"state": torch.ones(2), "trace": policy.trace})

        engine.soft_pause()
        paused = engine.debug_snapshot()
        assert paused["subtask_time_current_index"] == 0
        assert paused["subtask_time_paused"] is True
        assert paused["subtask_time_running"] is False
        assert paused["subtask_time_raw_elapsed_seconds"] == pytest.approx(3.5)
        assert paused["queue_size"] == 0
        assert engine._obs_holder["obs"] is None
        assert paused["last_subtask_output_text"] == ""
        assert paused["memory_text_for_next_inference"] == ""
        assert policy.reset_calls == 1
        assert preprocessor.reset_calls == 1
        assert postprocessor.reset_calls == 1

        clock.advance(90.0)
        assert engine.debug_snapshot()["subtask_time_raw_elapsed_seconds"] == pytest.approx(3.5)

        engine.resume()
        clock.advance(0.8)
        resumed = engine.debug_snapshot()
        assert resumed["subtask_time_running"] is True
        assert resumed["subtask_time_paused"] is False
        assert resumed["subtask_time_raw_elapsed_seconds"] == pytest.approx(4.3)
    finally:
        engine.stop()


def test_soft_pause_during_inference_discards_candidate_and_preserves_committed_time() -> None:
    output_a = "Subtask: First.; Progress: 0.2"
    output_b = "Subtask: Second.; Progress: 0.1"
    engine, policy, _, _, clock = _make_engine(outputs=[output_a, output_b])

    try:
        _start_engine(engine, policy.trace)
        _wait_until(lambda: _queue_size(engine) == 1)
        policy.block_calls.add(2)
        _consume_committed_action(engine)
        _wait_until(policy.block_entered.is_set)
        clock.advance(2.0)

        pause_thread = threading.Thread(target=engine.soft_pause)
        pause_thread.start()
        _wait_until(lambda: engine._reset_version == 1)
        policy.block_release.set()
        pause_thread.join(timeout=3.0)
        assert not pause_thread.is_alive()

        snapshot = engine.debug_snapshot()
        assert snapshot["inference_count"] == 1
        assert snapshot["subtask_time_current_index"] == 0
        assert snapshot["subtask_time_paused"] is True
        assert snapshot["subtask_time_raw_elapsed_seconds"] == pytest.approx(2.0)
        assert snapshot["queue_size"] == 0
    finally:
        policy.block_release.set()
        engine.stop()


def test_full_reset_after_soft_pause_clears_semantic_time_and_is_idempotent() -> None:
    output = "Subtask: First.; Progress: 0.2"
    engine, policy, _, _, clock = _make_engine(outputs=[output])

    try:
        _start_engine(engine, policy.trace)
        _wait_until(lambda: _queue_size(engine) == 1)
        clock.advance(1.5)
        engine.soft_pause()
        engine.soft_pause()
        assert engine.debug_snapshot()["subtask_time_raw_elapsed_seconds"] == pytest.approx(1.5)

        engine.full_reset()
        engine.full_reset()
        reset = engine.debug_snapshot()
        assert reset["active"] is False
        assert reset["subtask_time_current_index"] is None
        assert reset["subtask_time_valid"] is False
        assert reset["subtask_time_effective_seconds"] == 0.0
        assert reset["subtask_time_last_input_seconds"] is None

        engine.resume()
        engine.resume()
        resumed = engine.debug_snapshot()
        assert resumed["active"] is True
        assert resumed["subtask_time_current_index"] is None
        assert resumed["subtask_time_paused"] is False
    finally:
        engine.stop()


def test_deployment_cap_warning_is_emitted_once_per_subtask(caplog) -> None:
    output = "Subtask: First.; Progress: 0.2"
    engine, policy, _, _, clock = _make_engine(outputs=[output, output, output])

    try:
        with caplog.at_level("WARNING"):
            _start_engine(engine, policy.trace)
            _wait_until(lambda: _queue_size(engine) == 1)
            clock.advance(12.0)
            _consume_committed_action(engine)
            _wait_until(lambda: policy.call_count == 2)
            _wait_until(lambda: _queue_size(engine) == 1)
            _consume_committed_action(engine)
            _wait_until(lambda: policy.call_count == 3)
            engine.pause()

        assert caplog.text.count("reached deployment cap") == 1
    finally:
        engine.stop()
