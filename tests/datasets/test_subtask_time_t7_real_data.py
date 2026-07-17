#!/usr/bin/env python

"""Milestone T7 real-data and real-tokenizer elapsed-time validation.

The suite is opt-in so normal unit-test runs do not depend on local datasets or
multi-gigabyte checkpoints.  The T7 validation script provides all paths and
runs fully offline.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import (
    MemoryConditionProcessorStep,
    SubtaskTimeConditionProcessorStep,
    TokenizerProcessorStep,
)
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
from lerobot.utils.memory_conditioning import sample_memory_condition_mask
from lerobot.utils.subtask_time_conditioning import sample_subtask_time_condition


EGG_ROOT_ENV = "LEROBOT_T7_EGG_ROOT"
EGG_REPO_ENV = "LEROBOT_T7_EGG_REPO_ID"
PI0_BASE_ENV = "LEROBOT_T7_PI0_BASE"
PI05_BASE_ENV = "LEROBOT_T7_PI05_BASE"


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is required for the T7 real-data suite")
    path = Path(value).resolve()
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


def _repo_id() -> str:
    return os.environ.get(EGG_REPO_ENV, "ming326/nero_egg_subtask")


def _policy_config(checkpoint_env: str, *, memory: bool, time: bool):
    config = PreTrainedConfig.from_pretrained(_required_path(checkpoint_env))
    return dataclasses.replace(
        config,
        device="cpu",
        compile_model=False,
        gradient_checkpointing=False,
        predict_subtask=True,
        subtask_generate_at_inference=True,
        subtask_max_decode_tokens=min(48, config.subtask_max_tokens),
        use_memory_conditioning=memory,
        memory_tokenizer_max_length=128,
        use_subtask_time_conditioning=time,
        subtask_time_tokenizer_max_length=128,
    )


def _real_batch(checkpoint_env: str, *, memory: bool, time: bool):
    policy_config = _policy_config(checkpoint_env, memory=memory, time=time)
    train_config = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=_repo_id(),
            root=str(_required_path(EGG_ROOT_ENV)),
            episodes=[0],
            streaming=False,
        ),
        policy=policy_config,
    )
    dataset = make_dataset(train_config)
    torch.manual_seed(7)
    batch = torch.utils.data.default_collate([dataset[20]])
    if memory:
        assert batch["memory_valid"].item()
        batch = sample_memory_condition_mask(batch, dropout_prob=0.0)
    if time:
        batch = sample_subtask_time_condition(
            batch,
            noise_ratio=0.0,
            noise_max_seconds=5.0,
            dropout_prob=0.0,
        )
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_config,
        dataset_stats=dataset.meta.stats,
    )
    return dataset, policy_config, batch, preprocessor


@pytest.mark.parametrize("checkpoint_env", [PI0_BASE_ENV, PI05_BASE_ENV])
@pytest.mark.parametrize("memory", [False, True])
def test_true_tokenizer_decodes_canonical_elapsed_time(checkpoint_env: str, memory: bool):
    dataset, policy_config, batch, preprocessor = _real_batch(
        checkpoint_env,
        memory=memory,
        time=True,
    )
    true_seconds = float(batch["subtask_elapsed_seconds"].item())
    assert true_seconds > 0.0
    assert batch["subtask_time_condition_kept"].item()

    processed = preprocessor(batch)
    tokenizer = next(
        step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep)
    )
    prompt = tokenizer.input_tokenizer.batch_decode(
        processed[OBS_LANGUAGE_TOKENS], skip_special_tokens=True
    )[0]

    assert f"Subtask elapsed time: {true_seconds:.1f}s" in prompt
    assert sum(isinstance(step, SubtaskTimeConditionProcessorStep) for step in preprocessor.steps) == 1
    assert sum(isinstance(step, MemoryConditionProcessorStep) for step in preprocessor.steps) == int(memory)
    assert tokenizer.max_length == (128 if policy_config.type == "pi0" else 200)
    assert tokenizer.truncation_side == "left"
    if memory:
        assert "Memory:" in prompt
    else:
        assert "Memory:" not in prompt
    if policy_config.type == "pi05":
        assert "State:" in prompt
    assert dataset.num_frames == 5612


@pytest.mark.parametrize("checkpoint_env", [PI0_BASE_ENV, PI05_BASE_ENV])
def test_real_time_disabled_pipeline_keeps_checkpoint_baseline(checkpoint_env: str):
    _, policy_config, batch, preprocessor = _real_batch(
        checkpoint_env,
        memory=False,
        time=False,
    )
    processed = preprocessor(batch)
    tokenizer = next(
        step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep)
    )
    prompt = tokenizer.input_tokenizer.batch_decode(
        processed[OBS_LANGUAGE_TOKENS], skip_special_tokens=True
    )[0]

    assert "Subtask elapsed time:" not in prompt
    assert "Memory:" not in prompt
    assert not any(isinstance(step, SubtaskTimeConditionProcessorStep) for step in preprocessor.steps)
    assert not any(key.startswith("subtask_time_") for key in processed)
    assert tokenizer.max_length == (48 if policy_config.type == "pi0" else 200)
    assert tokenizer.truncation_side is None
