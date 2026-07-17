#!/usr/bin/env python

"""Milestone T3 contracts for elapsed-time prompt conditioning."""

from __future__ import annotations

import math

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.processor_pi0 import Pi0NewLineProcessor, make_pi0_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import (
    Pi05PrepareStateTokenizerProcessorStep,
    make_pi05_pre_post_processors,
)
from lerobot.processor import (
    AdvantageConditionProcessorStep,
    DataProcessorPipeline,
    MemoryConditionProcessorStep,
    ProcessorStepRegistry,
    SubtaskTextProcessorStep,
    TokenizerProcessorStep,
)
from lerobot.processor.converters import create_transition
from lerobot.processor.subtask_time_processor import (
    SubtaskTimeConditionProcessorStep,
    format_subtask_elapsed_time,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class CharacterTokenizer:
    eos_token_id = 3
    pad_token_id = 0

    def __init__(self):
        self.truncation_side = "right"

    def encode(self, text, add_special_tokens=False):
        token_ids = [ord(character) for character in text]
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids

    def __call__(
        self,
        text,
        *,
        max_length,
        truncation,
        padding,
        padding_side,
        return_tensors,
        **kwargs,
    ):
        del kwargs, return_tensors
        texts = [text] if isinstance(text, str) else text
        encoded = []
        for item in texts:
            token_ids = self.encode(item, add_special_tokens=True)
            if truncation and len(token_ids) > max_length:
                token_ids = (
                    token_ids[-max_length:]
                    if self.truncation_side == "left"
                    else token_ids[:max_length]
                )
            if padding == "max_length" and len(token_ids) < max_length:
                pad = [self.pad_token_id] * (max_length - len(token_ids))
                token_ids = pad + token_ids if padding_side == "left" else token_ids + pad
            encoded.append(token_ids)
        input_ids = torch.tensor(encoded, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id).long(),
        }


def _visible_text(tokens: torch.Tensor) -> str:
    return "".join(chr(token) for token in tokens.tolist() if token not in {0, 3})


def _config(config_cls, *, time=False, memory=False, advantage=False):
    config = config_cls(
        predict_subtask=True,
        use_subtask_time_conditioning=time,
        use_memory_conditioning=memory,
        use_advantage_conditioning=advantage,
        device="cpu",
        push_to_hub=False,
    )
    config.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    return config


def _batch(batch_size=2, *, memory=False, advantage=False):
    batch = {
        OBS_STATE: torch.zeros(batch_size, 2),
        ACTION: torch.zeros(batch_size, 2),
        "task": ["prepare the egg carefully"] * batch_size,
        "subtask": ["Stir the beaten eggs."] * batch_size,
        "subtask_progress": torch.full((batch_size,), 0.4),
        "subtask_time_seconds": torch.tensor([95.76] * batch_size),
        "subtask_time_valid": torch.ones(batch_size, dtype=torch.bool),
        "subtask_time_condition_kept": torch.ones(batch_size, dtype=torch.bool),
    }
    if memory:
        batch.update(
            {
                "memory_subtask": ["Pick up the fork."] * batch_size,
                "memory_subtask_progress": torch.full((batch_size,), 0.8),
                "memory_valid": torch.ones(batch_size, dtype=torch.bool),
                "memory_condition_kept": torch.ones(batch_size, dtype=torch.bool),
            }
        )
    if advantage:
        batch.update(
            {
                "advantage_label_global": ["positive"] * batch_size,
                "advantage_condition_kept": torch.ones(batch_size, dtype=torch.bool),
            }
        )
    return batch


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "Subtask elapsed time: 0.0s"),
        (-0.0, "Subtask elapsed time: 0.0s"),
        (1.24, "Subtask elapsed time: 1.2s"),
        (torch.tensor([40.55]), "Subtask elapsed time: 40.5s"),
        (95.766667, "Subtask elapsed time: 95.8s"),
        (1e20, "Subtask elapsed time: 100000000000000000000.0s"),
    ],
)
def test_formatter_emits_canonical_fixed_point_text(value, expected):
    assert format_subtask_elapsed_time(value) == expected


@pytest.mark.parametrize(
    "invalid",
    [True, "1.2", -0.1, math.nan, math.inf, -math.inf, torch.tensor([1.0, 2.0])],
)
def test_formatter_rejects_invalid_values(invalid):
    with pytest.raises((TypeError, ValueError)):
        format_subtask_elapsed_time(invalid)


def test_processor_appends_only_effectively_kept_values_and_preserves_noops():
    transition = create_transition(
        complementary_data={
            "task": ["task-a  ", "task-b\n", "task-c\t"],
            "subtask_time_seconds": torch.tensor([[1.24], [2.0], [3.0]]),
            "subtask_time_valid": torch.tensor([[True], [False], [True]]),
            "subtask_time_condition_kept": torch.tensor([[True], [True], [False]]),
        }
    )
    result = SubtaskTimeConditionProcessorStep()(transition)[TransitionKey.COMPLEMENTARY_DATA]

    assert result["task"] == [
        "task-a\nSubtask elapsed time: 1.2s",
        "task-b\n",
        "task-c\t",
    ]
    assert result["subtask_time_condition_kept"].tolist() == [True, False, False]
    assert transition[TransitionKey.COMPLEMENTARY_DATA]["task"][0] == "task-a  "


def test_processor_accepts_scalar_list_tuple_and_tensor_contracts():
    cases = [
        {"task": "a", "seconds": 1.2, "valid": True, "keep": True},
        {"task": ["a"], "seconds": [1.2], "valid": [True], "keep": [True]},
        {"task": ("a",), "seconds": (1.2,), "valid": (True,), "keep": (True,)},
        {
            "task": ["a"],
            "seconds": torch.tensor([1.2]),
            "valid": torch.tensor([True]),
            "keep": torch.tensor([True]),
        },
    ]
    for case in cases:
        result = SubtaskTimeConditionProcessorStep()(
            create_transition(
                complementary_data={
                    "task": case["task"],
                    "subtask_time_seconds": case["seconds"],
                    "subtask_time_valid": case["valid"],
                    "subtask_time_condition_kept": case["keep"],
                }
            )
        )[TransitionKey.COMPLEMENTARY_DATA]
        tasks = [result["task"]] if isinstance(result["task"], str) else list(result["task"])
        assert tasks == ["a\nSubtask elapsed time: 1.2s"]


def test_processor_without_time_source_is_byte_for_byte_noop():
    original = [" keep spaces  ", "keep newline\n"]
    result = SubtaskTimeConditionProcessorStep()(
        create_transition(complementary_data={"task": list(original)})
    )[TransitionKey.COMPLEMENTARY_DATA]

    assert result["task"] == original
    assert result["subtask_time_condition_kept"].tolist() == [False, False]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "task"),
        ({"task": ["a"], "subtask_time_seconds": [1.0]}, "required"),
        (
            {
                "task": ["a", "b"],
                "subtask_time_seconds": [1.0],
                "subtask_time_valid": [True],
                "subtask_time_condition_kept": [True],
            },
            "batch size",
        ),
        (
            {
                "task": ["a"],
                "subtask_time_seconds": [1.0],
                "subtask_time_valid": [1],
                "subtask_time_condition_kept": [True],
            },
            "boolean",
        ),
        (
            {
                "task": ["a"],
                "subtask_time_seconds": [True],
                "subtask_time_valid": [True],
                "subtask_time_condition_kept": [True],
            },
            "real numeric",
        ),
        (
            {
                "task": ["a"],
                "subtask_time_seconds": torch.ones(1, 1, 1),
                "subtask_time_valid": [True],
                "subtask_time_condition_kept": [True],
            },
            "shape",
        ),
    ],
)
def test_processor_rejects_partial_or_invalid_batch_contract(data, message):
    with pytest.raises(ValueError, match=message):
        SubtaskTimeConditionProcessorStep()(create_transition(complementary_data=data))


def test_processor_registry_config_and_save_reload_preserve_output(tmp_path):
    processor = SubtaskTimeConditionProcessorStep(seconds_key="elapsed")
    pipeline = DataProcessorPipeline(steps=[processor], name="subtask_time_test")
    pipeline.save_pretrained(tmp_path, config_filename="subtask_time.json")
    loaded = DataProcessorPipeline.from_pretrained(
        tmp_path, config_filename="subtask_time.json"
    )
    data = {
        "task": "task",
        "elapsed": 2.25,
        "subtask_time_valid": True,
        "subtask_time_condition_kept": True,
    }

    assert (
        ProcessorStepRegistry.get("subtask_time_condition_processor")
        is SubtaskTimeConditionProcessorStep
    )
    assert processor.get_config()["seconds_key"] == "elapsed"
    assert isinstance(loaded.steps[0], SubtaskTimeConditionProcessorStep)
    assert (
        loaded.process_complementary_data(data)["task"]
        == "task\nSubtask elapsed time: 2.2s"
    )


@pytest.mark.parametrize("config_cls", [PI0Config, PI05Config])
def test_policy_config_defaults_validation_and_generation_warning(config_cls):
    config = config_cls()
    assert config.use_subtask_time_conditioning is False
    assert config.subtask_time_tokenizer_max_length == 128

    with pytest.raises(ValueError, match="predict_subtask"):
        config_cls(use_subtask_time_conditioning=True, predict_subtask=False)
    with pytest.raises(ValueError, match="subtask_time_tokenizer_max_length"):
        config_cls(
            use_subtask_time_conditioning=True,
            predict_subtask=True,
            subtask_time_tokenizer_max_length=0,
        )
    with pytest.warns(UserWarning, match="elapsed-time"):
        config_cls(
            use_subtask_time_conditioning=True,
            predict_subtask=True,
            subtask_generate_at_inference=False,
        )


@pytest.mark.parametrize(
    ("config_cls", "factory", "time", "memory", "expected_length"),
    [
        (PI0Config, make_pi0_pre_post_processors, False, False, 48),
        (PI0Config, make_pi0_pre_post_processors, True, False, 128),
        (PI0Config, make_pi0_pre_post_processors, False, True, 128),
        (PI0Config, make_pi0_pre_post_processors, True, True, 128),
        (PI05Config, make_pi05_pre_post_processors, False, False, 200),
        (PI05Config, make_pi05_pre_post_processors, True, False, 200),
        (PI05Config, make_pi05_pre_post_processors, False, True, 200),
        (PI05Config, make_pi05_pre_post_processors, True, True, 200),
    ],
)
def test_pipeline_four_configurations_budget_and_truncation(
    monkeypatch, config_cls, factory, time, memory, expected_length
):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    preprocessor, _ = factory(_config(config_cls, time=time, memory=memory))
    tokenizer = next(step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep))

    assert any(isinstance(step, SubtaskTimeConditionProcessorStep) for step in preprocessor.steps) is time
    assert any(isinstance(step, MemoryConditionProcessorStep) for step in preprocessor.steps) is memory
    assert tokenizer.max_length == expected_length
    assert tokenizer.truncation_side == ("left" if time or memory else None)


@pytest.mark.parametrize(
    ("config_cls", "factory", "prompt_step_cls"),
    [
        (PI0Config, make_pi0_pre_post_processors, Pi0NewLineProcessor),
        (PI05Config, make_pi05_pre_post_processors, Pi05PrepareStateTokenizerProcessorStep),
    ],
)
def test_pipeline_order_long_prompt_and_current_subtask_isolation(
    monkeypatch, config_cls, factory, prompt_step_cls
):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    preprocessor, _ = factory(
        _config(config_cls, time=True, memory=True, advantage=True), dataset_stats={}
    )
    steps = preprocessor.steps
    indices = {
        cls: next(i for i, step in enumerate(steps) if isinstance(step, cls))
        for cls in (
            AdvantageConditionProcessorStep,
            MemoryConditionProcessorStep,
            SubtaskTimeConditionProcessorStep,
            prompt_step_cls,
            SubtaskTextProcessorStep,
            TokenizerProcessorStep,
        )
    }
    assert indices[AdvantageConditionProcessorStep] < indices[MemoryConditionProcessorStep]
    assert indices[MemoryConditionProcessorStep] < indices[SubtaskTimeConditionProcessorStep]
    assert indices[SubtaskTimeConditionProcessorStep] < indices[prompt_step_cls]
    assert indices[prompt_step_cls] < indices[SubtaskTextProcessorStep]
    assert indices[SubtaskTextProcessorStep] < indices[TokenizerProcessorStep]

    batch = _batch(memory=True, advantage=True)
    batch["subtask_time_condition_kept"] = torch.tensor([True, False])
    processed = preprocessor(batch)
    kept_prompt = _visible_text(processed[OBS_LANGUAGE_TOKENS][0])
    dropped_prompt = _visible_text(processed[OBS_LANGUAGE_TOKENS][1])
    assert "Memory: Subtask: Pick up the fork.; Progress: 0.8" in kept_prompt
    assert "Subtask elapsed time: 95.8s" in kept_prompt
    assert "Subtask elapsed time:" not in dropped_prompt
    if config_cls is PI05Config:
        assert "State: 128 128;" in kept_prompt
    assert torch.equal(
        processed[OBS_LANGUAGE_SUBTASK_TOKENS][0],
        processed[OBS_LANGUAGE_SUBTASK_TOKENS][1],
    )
