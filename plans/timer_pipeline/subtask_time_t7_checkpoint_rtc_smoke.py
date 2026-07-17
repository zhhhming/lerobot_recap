#!/usr/bin/env python

"""Validate one real T7 checkpoint and production RTC with fake BiNero input.

The policy and tokenizer are real.  RTC actions come from the real model and a
real dataset observation.  A test-only output controller records the model's
natural subtask text, then supplies deterministic canonical current/next/old
text so the irreversible tracker transaction can be validated reproducibly.
No robot connection is made.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import platform
import threading
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.memory_history import MemoryHistoryDataset
from lerobot.datasets.subtask_timing import SubtaskTimingDataset
from lerobot.inference_engines.rtc import RTCInferenceEngine
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import (
    MemoryConditionProcessorStep,
    SubtaskTimeConditionProcessorStep,
    TokenizerProcessorStep,
)
from lerobot.scripts.lerobot_policy_deploy import _PolicyDeployRuntime
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
from lerobot.utils.memory_conditioning import sample_memory_condition_mask
from lerobot.utils.subtask_time_conditioning import sample_subtask_time_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="ming326/nero_egg_subtask")
    parser.add_argument("--expected-policy", choices=("pi0", "pi05"), required=True)
    parser.add_argument("--sample-index", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


class FakeClock:
    def __init__(self) -> None:
        self._value = 1000.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds


class RuntimePart:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class OutputController:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.original_predict = policy.predict_action_chunk
        self.outputs: list[str] = []
        self.natural_outputs: list[str] = []
        self.committed_outputs: list[str] = []

        def controlled_predict(instance, batch, *args, **kwargs):
            actions = self.original_predict(batch, *args, **kwargs)
            natural = getattr(instance, "last_subtask_text", "")
            self.natural_outputs.append(natural if isinstance(natural, str) else repr(natural))
            if not self.outputs:
                raise RuntimeError("T7 scripted subtask output queue is empty")
            scripted = self.outputs.pop(0)
            instance.last_subtask_text = scripted
            self.committed_outputs.append(scripted)
            return actions

        policy.predict_action_chunk = MethodType(controlled_predict, policy)

    def set_outputs(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.natural_outputs = []
        self.committed_outputs = []


def wait_for(engine: RTCInferenceEngine, predicate, *, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last = engine.debug_snapshot()
    while time.monotonic() < deadline:
        last = engine.debug_snapshot()
        if last["failed"]:
            raise RuntimeError(f"RTC engine failed: {last}")
        if predicate(last):
            return last
        time.sleep(0.02)
    raise TimeoutError(f"RTC condition was not reached; last snapshot: {last}")


def raw_bi_nero_observation(item: dict, features: dict) -> dict[str, object]:
    state = item["observation.state"].detach().cpu().numpy()
    state_names = features["observation.state"]["names"]
    observation: dict[str, object] = {
        name: float(value) for name, value in zip(state_names, state, strict=True)
    }
    for key, feature in features.items():
        if feature["dtype"] not in {"image", "video"} or not key.startswith(
            "observation.images."
        ):
            continue
        image = item[key].detach().cpu()
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected CHW image for {key}, got {tuple(image.shape)}")
        array = image.permute(1, 2, 0).numpy()
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
        observation[key.removeprefix("observation.images.")] = array
    return observation


def consume_action(engine: RTCInferenceEngine) -> None:
    action = engine.get_action(None)
    if action is None:
        raise AssertionError("Expected one finite fake-RTC action, got None")
    finite = (
        bool(torch.isfinite(action).all().item())
        if isinstance(action, torch.Tensor)
        else bool(np.isfinite(action).all())
    )
    if not finite:
        raise AssertionError(f"Expected one finite fake-RTC action, got {action!r}")


def real_model_condition_matrix(
    *,
    policy,
    preprocessor,
    tokenizer: TokenizerProcessorStep,
    dataset_repo_id: str,
    dataset_root: Path,
    sample_index: int,
) -> dict[str, dict[str, object]]:
    config = policy.config
    delta_timestamps = {
        "action": [index / 30.0 for index in config.action_delta_indices]
    }
    base = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        episodes=[0],
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    dataset = MemoryHistoryDataset(SubtaskTimingDataset(base))
    torch.manual_seed(11)
    raw_batch = torch.utils.data.default_collate([dataset[sample_index]])
    if not raw_batch["memory_valid"].item() or not raw_batch["subtask_time_valid"].item():
        raise AssertionError("T7 matrix sample must have valid memory and elapsed time")

    cases = {
        "both_clean": (0.0, 0.0, 0.0),
        "history_only": (0.0, 1.0, 0.0),
        "time_only_noisy": (1.0, 0.0, 0.4),
        "neither_noisy": (1.0, 1.0, 0.4),
        "default_time_dropout_noise": (0.0, 0.2, 0.4),
    }
    original_subtask_dropout = config.subtask_dropout_prob
    core = getattr(policy, "model", None)
    original_checkpointing = getattr(core, "gradient_checkpointing_enabled", None)
    if original_checkpointing is not None:
        core.gradient_checkpointing_enabled = False
    config.subtask_dropout_prob = 0.0
    results: dict[str, dict[str, object]] = {}
    try:
        policy.train()
        for index, (name, (memory_dropout, time_dropout, noise_ratio)) in enumerate(cases.items()):
            generator = torch.Generator().manual_seed(1200 + index)
            batch = sample_memory_condition_mask(
                raw_batch,
                dropout_prob=memory_dropout,
                generator=generator,
            )
            batch = sample_subtask_time_condition(
                batch,
                noise_ratio=noise_ratio,
                noise_max_seconds=5.0,
                dropout_prob=time_dropout,
                generator=generator,
            )
            true_seconds = float(batch["subtask_elapsed_seconds"].item())
            noisy_seconds = float(batch["subtask_time_seconds"].item())
            if abs(noisy_seconds - true_seconds) > min(0.4 * true_seconds, 5.0) + 1e-5:
                raise AssertionError(f"Noise bound exceeded in {name}")

            preprocessor.reset()
            policy.reset()
            processed = preprocessor(batch)
            prompt = tokenizer.input_tokenizer.batch_decode(
                processed[OBS_LANGUAGE_TOKENS], skip_special_tokens=True
            )[0]
            with torch.no_grad():
                loss, info = policy(processed)
            values = {
                "loss": float(loss.item()),
                "fm_loss": float(info["fm_loss"]),
                "ce_loss": float(info["ce_loss"]),
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise AssertionError(f"Non-finite real-model result for {name}: {values}")
            if values["ce_loss"] <= 0.0:
                raise AssertionError(f"Current-subtask CE was removed for {name}: {values}")
            time_kept = bool(batch["subtask_time_condition_kept"].item())
            memory_kept = bool(batch["memory_condition_kept"].item())
            if ("Subtask elapsed time:" in prompt) != time_kept:
                raise AssertionError(f"Time prompt mismatch for {name}: {prompt!r}")
            if ("Memory:" in prompt) != memory_kept:
                raise AssertionError(f"Memory prompt mismatch for {name}: {prompt!r}")
            results[name] = {
                **values,
                "memory_dropout": memory_dropout,
                "time_dropout": time_dropout,
                "noise_ratio": noise_ratio,
                "memory_kept": memory_kept,
                "time_kept": time_kept,
                "true_seconds": true_seconds,
                "input_seconds": noisy_seconds,
                "canonical_time_present": "Subtask elapsed time:" in prompt,
            }
    finally:
        config.subtask_dropout_prob = original_subtask_dropout
        if original_checkpointing is not None:
            core.gradient_checkpointing_enabled = original_checkpointing
        preprocessor.reset()
        policy.reset()
        policy.eval()
    return results


def rtc_mode(
    *,
    policy,
    preprocessor,
    postprocessor,
    controller: OutputController,
    contract,
    raw_observation: dict,
    hw_features: dict,
    task: str,
    fps: float,
    history_enabled: bool,
    time_enabled: bool,
    timeout_s: float,
) -> dict[str, object]:
    first_name = contract.ordered_subtasks[0].canonical_name
    second_name = contract.ordered_subtasks[1].canonical_name
    first = f"Subtask: {first_name}; Progress: 0.2"
    second = f"Subtask: {second_name}; Progress: 0.1"
    outputs = [first, second]
    if time_enabled:
        outputs = [first, second, first, second, first]
    controller.set_outputs(outputs)
    policy.config.use_memory_conditioning = history_enabled
    clock = FakeClock()
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=SimpleNamespace(robot_type="bi_nero_follower"),
        rtc_config=RTCConfig(enabled=True),
        hw_features=hw_features,
        task=task,
        fps=fps,
        device="cuda",
        rtc_queue_threshold=policy.config.chunk_size - 1,
        subtask_sequence_contract=contract if time_enabled else None,
        subtask_time_enabled=time_enabled,
        subtask_time_clock=clock,
    )
    interpolator = RuntimePart()
    smoother = RuntimePart()
    runtime = _PolicyDeployRuntime(engine, interpolator, smoother)
    snapshots: dict[str, dict] = {}
    try:
        engine.start()
        runtime.start_or_resume()
        engine.notify_observation(raw_observation)
        snapshots["initial"] = wait_for(
            engine,
            lambda value: value["inference_count"] == 1,
            timeout_s=timeout_s,
        )
        consume_action(engine)
        snapshots["second"] = wait_for(
            engine,
            lambda value: value["inference_count"] == 2,
            timeout_s=timeout_s,
        )

        if history_enabled:
            if snapshots["initial"]["last_memory_input_text"]:
                raise AssertionError("First RTC inference unexpectedly used history")
            if snapshots["second"]["last_memory_input_text"] != first:
                raise AssertionError("Second RTC inference did not receive committed history")
        else:
            if snapshots["second"]["last_memory_input_text"]:
                raise AssertionError("History-off RTC injected memory")

        if not time_enabled:
            if snapshots["initial"]["subtask_time_enabled"]:
                raise AssertionError("Time-off RTC maintained a tracker")
            return {
                "history_enabled": history_enabled,
                "time_enabled": False,
                "inference_count": snapshots["second"]["inference_count"],
                "natural_outputs": list(controller.natural_outputs),
                "committed_outputs": list(controller.committed_outputs),
                "reset_counts": [interpolator.reset_count, smoother.reset_count],
            }

        if snapshots["initial"]["subtask_time_current_index"] != 0:
            raise AssertionError(f"Initial output did not start tracker: {snapshots['initial']}")
        if snapshots["initial"]["subtask_time_last_input_seconds"] is not None:
            raise AssertionError("First inference unexpectedly had elapsed-time input")
        if snapshots["second"]["subtask_time_current_index"] != 1:
            raise AssertionError(f"Next output did not advance tracker: {snapshots['second']}")

        clock.advance(2.0)
        consume_action(engine)
        snapshots["old"] = wait_for(
            engine,
            lambda value: value["inference_count"] == 3,
            timeout_s=timeout_s,
        )
        if snapshots["old"]["subtask_time_current_index"] != 1:
            raise AssertionError("Old output moved tracker backwards")
        if snapshots["old"]["subtask_time_last_transition"] != "rejected_old":
            raise AssertionError(f"Old output was not explicitly rejected: {snapshots['old']}")

        clock.advance(3.0)
        runtime.soft_pause()
        snapshots["paused"] = engine.debug_snapshot()
        frozen = snapshots["paused"]["subtask_time_raw_elapsed_seconds"]
        if not snapshots["paused"]["subtask_time_paused"]:
            raise AssertionError("Soft pause did not freeze tracker")
        clock.advance(90.0)
        if engine.debug_snapshot()["subtask_time_raw_elapsed_seconds"] != frozen:
            raise AssertionError("Paused wall-clock time leaked into active elapsed")

        runtime.start_or_resume()
        clock.advance(0.8)
        engine.notify_observation(raw_observation)
        snapshots["resumed"] = wait_for(
            engine,
            lambda value: value["inference_count"] == 4,
            timeout_s=timeout_s,
        )
        resumed_input = snapshots["resumed"]["subtask_time_last_input_seconds"]
        if resumed_input is None or not math.isclose(resumed_input, frozen + 0.8, abs_tol=1e-5):
            raise AssertionError(
                f"Resume elapsed mismatch: input={resumed_input}, expected={frozen + 0.8}"
            )

        runtime.full_reset()
        snapshots["home_reset"] = engine.debug_snapshot()
        if snapshots["home_reset"]["subtask_time_current_index"] is not None:
            raise AssertionError("Home/full reset retained tracker state")
        runtime.start_or_resume()
        engine.notify_observation(raw_observation)
        snapshots["fresh"] = wait_for(
            engine,
            lambda value: value["inference_count"] == 5,
            timeout_s=timeout_s,
        )
        if snapshots["fresh"]["subtask_time_current_index"] != 0:
            raise AssertionError("Fresh session did not restart from first subtask")
        if snapshots["fresh"]["subtask_time_last_input_seconds"] is not None:
            raise AssertionError("Fresh session unexpectedly reused elapsed time")
        return {
            "history_enabled": history_enabled,
            "time_enabled": True,
            "inference_count": snapshots["fresh"]["inference_count"],
            "old_rejection": snapshots["old"]["subtask_time_last_transition"],
            "paused_elapsed_seconds": frozen,
            "resumed_input_seconds": resumed_input,
            "home_reset": True,
            "natural_outputs": list(controller.natural_outputs),
            "committed_outputs": list(controller.committed_outputs),
            "reset_counts": [interpolator.reset_count, smoother.reset_count],
        }
    finally:
        engine.stop()
        preprocessor.reset()
        postprocessor.reset()
        policy.reset()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    if not checkpoint.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError(f"checkpoint={checkpoint}, dataset={dataset_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the T7 real-checkpoint smoke")

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if config.type != args.expected_policy:
        raise AssertionError(f"Expected {args.expected_policy}, got {config.type}")
    if not config.predict_subtask or not config.subtask_generate_at_inference:
        raise AssertionError("T7 checkpoint must predict and generate subtask text")
    if not config.use_memory_conditioning or not config.use_subtask_time_conditioning:
        raise AssertionError("T7 checkpoint must persist both memory and elapsed-time conditioning")
    config.device = "cuda"
    config.dtype = "bfloat16"
    config.compile_model = False

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    memory_steps = [
        step for step in preprocessor.steps if isinstance(step, MemoryConditionProcessorStep)
    ]
    time_steps = [
        step for step in preprocessor.steps if isinstance(step, SubtaskTimeConditionProcessorStep)
    ]
    if len(memory_steps) != 1 or len(time_steps) != 1:
        raise AssertionError(
            f"Expected one memory and one time processor, got {len(memory_steps)} and {len(time_steps)}"
        )
    tokenizer = next(
        step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep)
    )
    expected_length = 128 if config.type == "pi0" else 200
    if tokenizer.max_length != expected_length or tokenizer.truncation_side != "left":
        raise AssertionError(
            f"Unexpected tokenizer contract: length={tokenizer.max_length}, "
            f"truncation={tokenizer.truncation_side}"
        )

    policy_cls = get_policy_class(config.type)
    load_stdout = io.StringIO()
    with contextlib.redirect_stdout(load_stdout):
        policy = policy_cls.from_pretrained(checkpoint, config=config, strict=True)
    load_text = load_stdout.getvalue()
    if "All keys loaded successfully!" not in load_text:
        raise AssertionError(f"Strict checkpoint reload did not report clean keys:\n{load_text}")
    policy.to("cuda")
    policy.eval()
    torch.cuda.reset_peak_memory_stats()

    condition_matrix = real_model_condition_matrix(
        policy=policy,
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=dataset_root,
        sample_index=args.sample_index,
    )
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()

    base_dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        download_videos=False,
    )
    timed_dataset = SubtaskTimingDataset(base_dataset)
    item = base_dataset[args.sample_index]
    raw_observation = raw_bi_nero_observation(item, base_dataset.meta.features)
    hw_features = {
        key: value
        for key, value in base_dataset.meta.features.items()
        if key.startswith("observation.") or key == "action"
    }
    controller = OutputController(policy)
    rtc_results = {}
    for history_enabled, time_enabled in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        key = f"history_{'on' if history_enabled else 'off'}_time_{'on' if time_enabled else 'off'}"
        rtc_results[key] = rtc_mode(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            controller=controller,
            contract=timed_dataset.sequence_contract,
            raw_observation=raw_observation,
            hw_features=hw_features,
            task=item["task"],
            fps=base_dataset.fps,
            history_enabled=history_enabled,
            time_enabled=time_enabled,
            timeout_s=args.timeout_s,
        )

    report = {
        "policy": config.type,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        "strict_reload_clean": True,
        "tokenizer_max_length": tokenizer.max_length,
        "processor_steps": [type(step).__name__ for step in preprocessor.steps],
        "condition_matrix": condition_matrix,
        "rtc_matrix": rtc_results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
