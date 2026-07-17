#!/usr/bin/env python

"""Load one real PI0/PI0.5 memory checkpoint and run dataset-backed fake RTC.

This script never connects to robot hardware.  It converts one decoded dataset
sample back into the same raw state/camera representation produced by a
BiNero robot, then drives the production RTCInferenceEngine for two committed
inferences and one reset.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.memory_history import MemoryHistoryDataset
from lerobot.inference_engines.rtc import RTCInferenceEngine
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import MemoryConditionProcessorStep, TokenizerProcessorStep
from lerobot.utils.memory_conditioning import sample_memory_condition_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="ming326/nero_egg_subtask")
    parser.add_argument("--expected-policy", choices=("pi0", "pi05"), required=True)
    parser.add_argument("--memory-mode", choices=("on", "off"), default="on")
    parser.add_argument("--sample-index", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


def _wait_for(engine: RTCInferenceEngine, predicate, *, timeout_s: float) -> dict:
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


def _raw_bi_nero_observation(item: dict, features: dict) -> dict[str, object]:
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


def _real_model_dropout_matrix(
    *,
    policy,
    preprocessor,
    dataset_repo_id: str,
    dataset_root: Path,
    sample_index: int,
) -> dict[str, dict[str, float]]:
    config = policy.config
    delta_timestamps = {
        "action": [index / 30.0 for index in config.action_delta_indices]
    }
    training_dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        episodes=[0],
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    memory_dataset = MemoryHistoryDataset(training_dataset)
    raw_batch = torch.utils.data.default_collate([memory_dataset[sample_index]])
    if not raw_batch["memory_valid"].item():
        raise AssertionError(f"M8 dropout sample unexpectedly has no history: {raw_batch}")

    cases = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.2, 0.2),
    )
    original_subtask_dropout = config.subtask_dropout_prob
    core = getattr(policy, "model", None)
    original_checkpointing = getattr(core, "gradient_checkpointing_enabled", None)
    if original_checkpointing is not None:
        core.gradient_checkpointing_enabled = False
    results: dict[str, dict[str, float]] = {}
    try:
        policy.train()
        for memory_dropout, subtask_dropout in cases:
            generator = torch.Generator().manual_seed(1000)
            batch = sample_memory_condition_mask(
                raw_batch,
                dropout_prob=memory_dropout,
                generator=generator,
            )
            config.subtask_dropout_prob = subtask_dropout
            preprocessor.reset()
            policy.reset()
            processed = preprocessor(batch)
            with torch.no_grad():
                loss, info = policy(processed)
            values = {
                "loss": float(loss.item()),
                "fm_loss": float(info["fm_loss"]),
                "ce_loss": float(info["ce_loss"]),
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise AssertionError(
                    "Non-finite real-model dropout result for "
                    f"memory={memory_dropout}, subtask={subtask_dropout}: {values}"
                )
            key = f"memory_{memory_dropout:g}_subtask_{subtask_dropout:g}"
            results[key] = values
    finally:
        config.subtask_dropout_prob = original_subtask_dropout
        if original_checkpointing is not None:
            core.gradient_checkpointing_enabled = original_checkpointing
        preprocessor.reset()
        policy.reset()
        policy.eval()
    return results


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real checkpoint RTC smoke")

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if config.type != args.expected_policy:
        raise AssertionError(f"Expected {args.expected_policy}, got {config.type}")
    memory_enabled = args.memory_mode == "on"
    if memory_enabled:
        if not config.predict_subtask or not config.use_memory_conditioning:
            raise AssertionError("Checkpoint must persist predict_subtask and memory conditioning")
        if not config.subtask_generate_at_inference:
            raise AssertionError("Checkpoint must generate subtask text during fake RTC")
    elif config.use_memory_conditioning:
        raise AssertionError("Memory-off baseline checkpoint unexpectedly enables memory")
    config.device = "cuda"
    config.dtype = "bfloat16"
    config.compile_model = False

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    memory_steps = [step for step in preprocessor.steps if isinstance(step, MemoryConditionProcessorStep)]
    expected_memory_steps = 1 if memory_enabled else 0
    if len(memory_steps) != expected_memory_steps:
        raise AssertionError(
            f"Expected {expected_memory_steps} Memory processor steps, found {len(memory_steps)}"
        )
    tokenizer_step = next(
        step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep)
    )
    expected_length = 128 if config.type == "pi0" and memory_enabled else (
        48 if config.type == "pi0" else 200
    )
    if tokenizer_step.max_length != expected_length:
        raise AssertionError(
            f"Unexpected tokenizer length {tokenizer_step.max_length}, expected {expected_length}"
        )

    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(checkpoint, config=config)
    policy.to("cuda")
    policy.eval()
    dropout_matrix = (
        _real_model_dropout_matrix(
            policy=policy,
            preprocessor=preprocessor,
            dataset_repo_id=args.dataset_repo_id,
            dataset_root=dataset_root,
            sample_index=args.sample_index,
        )
        if memory_enabled
        else {}
    )
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        download_videos=False,
    )
    item = dataset[args.sample_index]
    raw_observation = _raw_bi_nero_observation(item, dataset.meta.features)
    hw_features = {
        key: value
        for key, value in dataset.meta.features.items()
        if key.startswith("observation.") or key == "action"
    }
    task = item["task"]
    robot_wrapper = SimpleNamespace(robot_type="bi_nero_follower")
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot_wrapper,
        rtc_config=RTCConfig(enabled=True),
        hw_features=hw_features,
        task=task,
        fps=dataset.fps,
        device="cuda",
        rtc_queue_threshold=config.chunk_size - 1,
    )

    report: dict[str, object] = {
        "policy": config.type,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "memory_mode": args.memory_mode,
        "tokenizer_max_length": tokenizer_step.max_length,
        "dropout_matrix": dropout_matrix,
    }
    try:
        engine.start()
        engine.notify_observation(raw_observation)
        engine.resume()
        first = _wait_for(
            engine,
            lambda snapshot: snapshot["inference_count"] == 1,
            timeout_s=args.timeout_s,
        )
        if first["last_memory_input_text"] != "":
            raise AssertionError(f"First RTC inference unexpectedly used memory: {first}")
        output_a = first["last_subtask_output_text"]
        if not math.isfinite(float(first["last_latency_s"])):
            raise AssertionError(f"First RTC latency is not finite: {first}")
        action = engine.get_action(None)
        if action is None or action.shape != (config.output_features["action"].shape[0],):
            raise AssertionError(f"Unexpected fake RTC action: {None if action is None else action.shape}")

        second = None
        output_b = ""
        if memory_enabled:
            if not isinstance(output_a, str) or not output_a.strip():
                raise AssertionError(f"First RTC inference produced empty subtask text: {first}")
            second = _wait_for(
                engine,
                lambda snapshot: snapshot["inference_count"] == 2,
                timeout_s=args.timeout_s,
            )
            if second["last_memory_input_text"] != output_a:
                raise AssertionError(
                    "Second RTC inference did not use the complete first subtask output: "
                    f"{second['last_memory_input_text']!r} != {output_a!r}"
                )
            output_b = second["last_subtask_output_text"]
            if not isinstance(output_b, str) or not output_b.strip():
                raise AssertionError(f"Second RTC inference produced empty subtask text: {second}")
        else:
            if first["memory_text_for_next_inference"] != "":
                raise AssertionError(f"Memory-off baseline updated memory: {first}")
            if first["memory_source_inference_id"] is not None:
                raise AssertionError(f"Memory-off baseline set a memory source: {first}")

        engine.pause()
        engine.reset()
        reset = engine.debug_snapshot()
        for key in (
            "last_memory_input_text",
            "last_subtask_output_text",
            "memory_text_for_next_inference",
        ):
            if reset[key] != "":
                raise AssertionError(f"Reset did not clear {key}: {reset}")
        if reset["memory_source_inference_id"] is not None:
            raise AssertionError(f"Reset kept a memory source id: {reset}")

        report.update(
            {
                "first_output": output_a,
                "second_memory_input": "" if second is None else second["last_memory_input_text"],
                "second_output": output_b,
                "first_latency_ms": round(float(first["last_latency_s"]) * 1000.0, 1),
                "second_latency_ms": None
                if second is None
                else round(float(second["last_latency_s"]) * 1000.0, 1),
                "reset_cleared": True,
            }
        )
    finally:
        engine.stop()

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
