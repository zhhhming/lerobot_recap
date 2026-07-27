#!/usr/bin/env python

"""Transactional memory tests for the asynchronous RTC inference engine."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import lerobot.inference_engines.rtc as rtc_module
from lerobot.inference_engines.memory_progress_assist import NeroEggMemoryProgressAssist
from lerobot.inference_engines.rtc import RTCInferenceEngine
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor.memory_processor import MemoryConditionProcessorStep


def _copy_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


class _FakePreprocessor:
    def __init__(self, *, use_memory: bool) -> None:
        self.steps = [MemoryConditionProcessorStep()] if use_memory else []
        self.reset_calls = 0

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
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
        policy_type: str,
        outputs: list[str | list[str]],
        use_memory: bool = True,
        generate_subtask: bool = True,
    ) -> None:
        self.config = SimpleNamespace(
            type=policy_type,
            use_memory_conditioning=use_memory,
            subtask_generate_at_inference=generate_subtask,
        )
        self.outputs = outputs
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
    monkeypatch.setattr(
        rtc_module,
        "build_dataset_frame",
        lambda _features, obs, prefix: {f"{prefix}.state": obs["state"]},
    )
    monkeypatch.setattr(
        rtc_module,
        "prepare_observation_for_inference",
        lambda batch, _device, task, robot_type: {
            **batch,
            "task": task,
            "robot_type": robot_type,
        },
    )
    monkeypatch.setattr(rtc_module, "_RTC_ERROR_RETRY_DELAY_S", 0.01)


def _make_engine(
    *,
    policy_type: str = "pi0",
    outputs: list[str | list[str]],
    use_memory: bool = True,
    generate_subtask: bool = True,
    memory_progress_assist: NeroEggMemoryProgressAssist | None = None,
) -> tuple[RTCInferenceEngine, _FakePolicy, _FakePreprocessor, _FakePostprocessor]:
    policy = _FakePolicy(
        policy_type=policy_type,
        outputs=outputs,
        use_memory=use_memory,
        generate_subtask=generate_subtask,
    )
    preprocessor = _FakePreprocessor(use_memory=use_memory)
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
        memory_progress_assist=memory_progress_assist,
    )
    return engine, policy, preprocessor, postprocessor


def _start_engine(engine: RTCInferenceEngine) -> None:
    engine.start()
    engine.notify_observation({"state": torch.zeros(2)})
    engine.resume()


def _wait_until(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("condition was not reached before timeout")


def _consume_committed_action(engine: RTCInferenceEngine) -> None:
    _wait_until(lambda: engine.debug_snapshot()["queue_size"] == 1)
    assert engine.get_action(None) is not None


@pytest.mark.parametrize("policy_type", ["pi0", "pi05"])
def test_successful_outputs_feed_the_next_inference(policy_type: str) -> None:
    output_a = "Subtask: pick up fork.; Progress: 0.2"
    output_b = "Subtask: pick up fork.; Progress: 0.4"
    output_c = "Subtask: move fork.; Progress: 0.1"
    engine, policy, _, _ = _make_engine(
        policy_type=policy_type,
        outputs=[f"  {output_a}\n", output_b, output_c],
    )

    try:
        _start_engine(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 1)

        assert "Memory:" not in policy.inputs[0]["task"][0]
        assert policy.inputs[0]["memory_text"] == [""]
        assert policy.inputs[0]["memory_valid"] == [False]
        first = engine.debug_snapshot()
        assert first["last_memory_input_text"] == ""
        assert first["last_subtask_output_text"] == output_a
        assert first["memory_text_for_next_inference"] == output_a
        assert first["memory_source_inference_id"] == 1

        _consume_committed_action(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 2)
        assert policy.inputs[1]["task"][0] == f"main task\nMemory: {output_a}"
        assert policy.inputs[1]["memory_text"] == [output_a]

        _consume_committed_action(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 3)
        assert policy.inputs[2]["task"][0] == f"main task\nMemory: {output_b}"
        third = engine.debug_snapshot()
        assert third["last_memory_input_text"] == output_b
        assert third["last_subtask_output_text"] == output_c
    finally:
        engine.stop()


@pytest.mark.parametrize("failure_stage", ["predict", "postprocess", "merge"])
def test_failed_transaction_does_not_commit_candidate(failure_stage: str) -> None:
    output_a = "Subtask: A; Progress: 0.2"
    output_b = "Subtask: B; Progress: 0.3"
    engine, policy, _, postprocessor = _make_engine(outputs=[output_a, output_b])

    try:
        _start_engine(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 1)

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
        assert snapshot["last_subtask_output_text"] == output_a
        assert snapshot["memory_text_for_next_inference"] == output_a
        assert snapshot["memory_source_inference_id"] == 1
    finally:
        engine.stop()


def test_reset_during_inference_discards_candidate_and_clears_memory() -> None:
    output_a = "Subtask: A; Progress: 0.2"
    output_b = "Subtask: B; Progress: 0.3"
    output_c = "Subtask: C; Progress: 0.1"
    engine, policy, preprocessor, postprocessor = _make_engine(outputs=[output_a, output_b, output_c])

    try:
        _start_engine(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 1)
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

        reset_snapshot = engine.debug_snapshot()
        assert reset_snapshot["inference_count"] == 1
        assert reset_snapshot["last_memory_input_text"] == ""
        assert reset_snapshot["last_subtask_output_text"] == ""
        assert reset_snapshot["memory_text_for_next_inference"] == ""
        assert reset_snapshot["memory_source_inference_id"] is None
        assert policy.reset_calls == 1
        assert preprocessor.reset_calls == 1
        assert postprocessor.reset_calls == 1

        engine.notify_observation({"state": torch.zeros(2)})
        engine.resume()
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 2)
        assert "Memory:" not in policy.inputs[2]["task"][0]
        assert policy.inputs[2]["memory_valid"] == [False]
        assert engine.debug_snapshot()["last_subtask_output_text"] == output_c
    finally:
        policy.block_release.set()
        engine.stop()


def test_empty_output_clears_memory_for_the_following_inference() -> None:
    output_a = "Subtask: A; Progress: 0.2"
    output_c = "Subtask: C; Progress: 0.1"
    engine, policy, _, _ = _make_engine(outputs=[output_a, " \n\t ", output_c])

    try:
        _start_engine(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 1)
        _consume_committed_action(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 2)

        second = engine.debug_snapshot()
        assert second["last_memory_input_text"] == output_a
        assert second["last_subtask_output_text"] == ""
        assert second["memory_text_for_next_inference"] == ""
        assert second["memory_source_inference_id"] is None

        _consume_committed_action(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 3)
        assert "Memory:" not in policy.inputs[2]["task"][0]
        assert policy.inputs[2]["memory_valid"] == [False]
    finally:
        engine.stop()


def test_memory_disabled_does_not_inject_fields_or_update_next_memory() -> None:
    output = "Subtask: A; Progress: 0.2"
    engine, policy, _, _ = _make_engine(outputs=[output], use_memory=False)

    try:
        _start_engine(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 1)
        policy_input = policy.inputs[0]
        assert not any(key.startswith("memory_") for key in policy_input)

        snapshot = engine.debug_snapshot()
        assert snapshot["last_subtask_output_text"] == output
        assert snapshot["memory_text_for_next_inference"] == ""
        assert snapshot["memory_source_inference_id"] is None
    finally:
        engine.stop()


def test_subtask_generation_disabled_remains_a_no_memory_ablation() -> None:
    output = "Subtask: should not become memory; Progress: 0.2"
    engine, policy, _, _ = _make_engine(outputs=[output, output], generate_subtask=False)

    try:
        _start_engine(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 1)
        assert engine.debug_snapshot()["memory_text_for_next_inference"] == ""

        _consume_committed_action(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 2)
        assert "Memory:" not in policy.inputs[1]["task"][0]
        assert policy.inputs[1]["memory_valid"] == [False]
        assert engine.debug_snapshot()["memory_text_for_next_inference"] == ""
    finally:
        engine.stop()


def test_multi_sample_subtask_output_is_rejected(caplog) -> None:
    engine, policy, _, _ = _make_engine(outputs=[["Subtask: A", "Subtask: B"]])

    try:
        _start_engine(engine)
        _wait_until(lambda: policy.call_count >= 1)
        _wait_until(lambda: "batch size 1" in caplog.text)
        engine.pause()
        assert engine.debug_snapshot()["inference_count"] == 0
        assert engine.debug_snapshot()["memory_text_for_next_inference"] == ""
    finally:
        engine.stop()


def test_debug_snapshot_never_observes_torn_memory_commit() -> None:
    outputs = [f"Subtask: step {index}; Progress: 0.1" for index in range(1, 21)]
    engine, _, _, _ = _make_engine(outputs=outputs)
    stop_reader = threading.Event()
    snapshots: list[tuple[int, str, str, str, int | None]] = []

    def read_snapshots() -> None:
        while not stop_reader.is_set():
            snapshot = engine.debug_snapshot()
            snapshots.append(
                (
                    snapshot["inference_count"],
                    snapshot["last_memory_input_text"],
                    snapshot["last_subtask_output_text"],
                    snapshot["memory_text_for_next_inference"],
                    snapshot["memory_source_inference_id"],
                )
            )

    reader = threading.Thread(target=read_snapshots)
    try:
        _start_engine(engine)
        reader.start()
        for expected_count in range(1, len(outputs) + 1):
            _wait_until(lambda: engine.debug_snapshot()["inference_count"] == expected_count)
            if expected_count < len(outputs):
                _consume_committed_action(engine)

        stop_reader.set()
        reader.join(timeout=3.0)
        assert not reader.is_alive()

        allowed = {(0, "", "", "", None)}
        for index, output in enumerate(outputs, start=1):
            previous = "" if index == 1 else outputs[index - 2]
            allowed.add((index, previous, output, output, index))
        assert snapshots
        assert set(snapshots) <= allowed
    finally:
        stop_reader.set()
        reader.join(timeout=3.0)
        engine.stop()


def test_nero_egg_assist_changes_only_next_memory_after_successful_commit() -> None:
    now = 0.0

    def clock() -> float:
        return now

    frying_point_seven = "Subtask: Start frying the eggs.; Progress: 0.7"
    frying_point_eight = "Subtask: Start frying the eggs.; Progress: 0.8"
    assist = NeroEggMemoryProgressAssist(clock=clock)
    engine, policy, _, _ = _make_engine(
        outputs=[frying_point_seven, frying_point_seven, frying_point_seven],
        memory_progress_assist=assist,
    )

    try:
        _start_engine(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 1)
        assert engine.debug_snapshot()["memory_text_for_next_inference"] == frying_point_seven

        now = 6.0
        _consume_committed_action(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 2)
        forced = engine.debug_snapshot()
        assert forced["last_subtask_output_text"] == frying_point_seven
        assert forced["memory_text_for_next_inference"] == frying_point_eight
        assert forced["memory_progress_assist_raw_progress"] == 0.7
        assert forced["memory_progress_assist_effective_progress"] == 0.8
        assert forced["memory_progress_assist_forced"] is True

        _consume_committed_action(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 3)
        assert policy.inputs[2]["memory_text"] == [frying_point_eight]
        assert policy.inputs[2]["task"][0] == f"main task\nMemory: {frying_point_eight}"
    finally:
        engine.stop()


def test_nero_egg_assist_is_cleared_by_soft_pause() -> None:
    now = 0.0

    def clock() -> float:
        return now

    output = "Subtask: Start frying the eggs.; Progress: 0.7"
    assist = NeroEggMemoryProgressAssist(clock=clock)
    engine, _, _, _ = _make_engine(outputs=[output, output], memory_progress_assist=assist)

    try:
        _start_engine(engine)
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 1)
        now = 10.0
        engine.soft_pause()
        paused = engine.debug_snapshot()
        assert paused["memory_text_for_next_inference"] == ""
        assert paused["memory_progress_assist_reason"] == "reset"

        engine.notify_observation({"state": torch.zeros(2)})
        engine.resume()
        _wait_until(lambda: engine.debug_snapshot()["inference_count"] == 2)
        resumed = engine.debug_snapshot()
        assert resumed["memory_text_for_next_inference"] == output
        assert resumed["memory_progress_assist_forced"] is False
    finally:
        engine.stop()


def test_assist_rejects_memory_disabled_or_non_generating_policy() -> None:
    assist = NeroEggMemoryProgressAssist()

    with pytest.raises(ValueError, match="requires deployment memory updates"):
        _make_engine(outputs=[""], use_memory=False, memory_progress_assist=assist)
    with pytest.raises(ValueError, match="requires deployment memory updates"):
        _make_engine(outputs=[""], generate_subtask=False, memory_progress_assist=assist)
