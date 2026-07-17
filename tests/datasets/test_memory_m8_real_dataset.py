#!/usr/bin/env python

"""Milestone 8 checks against an explicitly supplied real LeRobotDataset.

These tests stay skipped in the normal unit suite.  The milestone validation
script supplies the local dataset and PI0/PI0.5 base checkpoint paths.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import MemoryConditionProcessorStep, TokenizerProcessorStep
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
from lerobot.utils.memory_conditioning import sample_memory_condition_mask


DATASET_ROOT_ENV = "LEROBOT_M8_DATASET_ROOT"
DATASET_REPO_ENV = "LEROBOT_M8_DATASET_REPO_ID"
PI0_BASE_ENV = "LEROBOT_M8_PI0_BASE"
PI05_BASE_ENV = "LEROBOT_M8_PI05_BASE"


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is required for the Milestone 8 real-data suite")
    path = Path(value).resolve()
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


def _repo_id() -> str:
    return os.environ.get(DATASET_REPO_ENV, "ming326/nero_egg_subtask")


def _memory_config(checkpoint_env: str):
    config = PreTrainedConfig.from_pretrained(_required_path(checkpoint_env))
    return dataclasses.replace(
        config,
        device="cpu",
        compile_model=False,
        gradient_checkpointing=False,
        predict_subtask=True,
        subtask_generate_at_inference=True,
        subtask_max_decode_tokens=min(48, config.subtask_max_tokens),
        use_memory_conditioning=True,
        memory_tokenizer_max_length=128,
    )


def _make_real_memory_dataset(checkpoint_env: str, *, episodes: list[int] | None = None):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=_repo_id(),
            root=str(_required_path(DATASET_ROOT_ENV)),
            episodes=episodes,
            streaming=False,
        ),
        policy=_memory_config(checkpoint_env),
    )
    return make_dataset(cfg), cfg.policy


def test_real_dataset_schema_values_and_episode_continuity():
    root = _required_path(DATASET_ROOT_ENV)
    info = json.loads((root / "meta/info.json").read_text())

    assert info["codebase_version"] == "v3.0"
    assert info["robot_type"] == "bi_nero_follower"
    assert info["fps"] == 30
    assert info["total_episodes"] == 61
    assert info["total_frames"] == 350010
    assert info["features"]["subtask"]["dtype"] == "string"
    assert info["features"]["subtask_progress"]["dtype"] == "float32"

    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    assert files
    table = pq.read_table(
        files,
        columns=["index", "episode_index", "frame_index", "subtask", "subtask_progress"],
    )
    index = table["index"].to_numpy()
    episode = table["episode_index"].to_numpy()
    frame = table["frame_index"].to_numpy()
    progress = table["subtask_progress"].to_numpy()
    subtasks = np.asarray(table["subtask"].to_pylist(), dtype=object)
    episode_start = np.r_[True, episode[1:] != episode[:-1]]

    assert all(table[name].null_count == 0 for name in table.column_names)
    assert np.array_equal(index, np.arange(info["total_frames"]))
    assert len(np.unique(episode)) == info["total_episodes"]
    assert np.all(frame[episode_start] == 0)
    assert np.all(np.diff(frame)[~episode_start[1:]] == 1)
    assert len(np.unique(subtasks)) == 12
    assert all(isinstance(value, str) and value.strip() for value in subtasks)
    assert np.isfinite(progress).all()
    assert np.all((0.0 <= progress) & (progress <= 1.0))


@pytest.mark.parametrize("checkpoint_env", [PI0_BASE_ENV, PI05_BASE_ENV])
def test_real_factory_history_dataloader_and_prompt(checkpoint_env: str):
    dataset, policy_config = _make_real_memory_dataset(checkpoint_env, episodes=[0, 1])
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 5612 + 6402

    # Episode starts always consume an offset but must never leak the preceding episode.
    for index in (0, 5612):
        item = dataset[index]
        assert 1 <= item["memory_frame_offset"] <= 12
        assert item["memory_valid"] is False
        assert item["memory_subtask"] == ""

    # A non-boundary sample has valid same-episode history and real decoded images.
    item = dataset[20]
    offset = item["memory_frame_offset"]
    history = dataset.dataset.get_raw_item(20 - offset)
    assert item["memory_valid"] is True
    assert item["memory_subtask"] == history["subtask"]
    assert item["memory_subtask_progress"].item() == pytest.approx(
        history["subtask_progress"].item()
    )
    assert item["episode_index"].item() == history["episode_index"].item()
    image_keys = sorted(key for key in item if key.startswith("observation.images."))
    assert image_keys == [
        "observation.images.left_wrist",
        "observation.images.right_wrist",
        "observation.images.third_person",
    ]
    assert all(tuple(item[key].shape) == (3, 480, 640) for key in image_keys)

    offsets = [dataset[20]["memory_frame_offset"] for _ in range(24)]
    assert all(1 <= value <= 12 for value in offsets)
    assert len(set(offsets)) >= 4

    subset = torch.utils.data.Subset(dataset, [20, 21, 22, 23])
    batches = {}
    for workers in (0, 2):
        loader = torch.utils.data.DataLoader(subset, batch_size=2, num_workers=workers)
        batches[workers] = next(iter(loader))
        assert batches[workers]["memory_frame_offset"].shape == (2,)
        assert batches[workers]["memory_valid"].dtype == torch.bool

    batch = sample_memory_condition_mask(batches[0], dropout_prob=0.0)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_config,
        dataset_stats=dataset.meta.stats,
    )
    assert any(isinstance(step, MemoryConditionProcessorStep) for step in preprocessor.steps)
    tokenizer_step = next(
        step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep)
    )
    expected_length = 128 if policy_config.type == "pi0" else 200
    assert tokenizer_step.max_length == expected_length
    assert tokenizer_step.truncation_side == "left"

    processed = preprocessor(batch)
    prompts = tokenizer_step.input_tokenizer.batch_decode(
        processed[OBS_LANGUAGE_TOKENS], skip_special_tokens=True
    )
    assert len(prompts) == 2
    assert all("Memory:" in prompt for prompt in prompts)
    assert all("Subtask:" in prompt and "Progress:" in prompt for prompt in prompts)
    if policy_config.type == "pi05":
        assert all("State:" in prompt for prompt in prompts)

