#!/usr/bin/env python

"""Memory prompt, policy config, and PI0/PI0.5 pipeline contracts."""

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
from lerobot.processor.subtask_processor import format_subtask_output
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class CharacterTokenizer:
    """Offline tokenizer that leaves prompt characters observable in assertions."""

    eos_token_id = 3
    pad_token_id = 0

    def __init__(self):
        self.truncation_side = "right"

    def encode(self, text, add_special_tokens=False):
        tokens = [ord(character) for character in text]
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

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
                if self.truncation_side == "left":
                    token_ids = token_ids[-max_length:]
                else:
                    token_ids = token_ids[:max_length]
            if padding == "max_length" and len(token_ids) < max_length:
                pad = [self.pad_token_id] * (max_length - len(token_ids))
                token_ids = pad + token_ids if padding_side == "left" else token_ids + pad
            encoded.append(token_ids)
        input_ids = torch.tensor(encoded, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id).to(dtype=torch.long),
        }


def _config(config_cls, *, memory=True, advantage=False):
    config = config_cls(
        predict_subtask=True,
        use_memory_conditioning=memory,
        use_advantage_conditioning=advantage,
        device="cpu",
    )
    config.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    return config


def _batch(batch_size=1, *, long_task=False, advantage=False):
    tasks = [("old task " * 80 if long_task else "do task") for _ in range(batch_size)]
    batch = {
        OBS_STATE: torch.zeros(batch_size, 2),
        ACTION: torch.zeros(batch_size, 2),
        "task": tasks,
        "subtask": ["current"] * batch_size,
        "subtask_progress": torch.full((batch_size,), 0.4),
        "memory_text": ["  Subtask:   previous;\n Progress: 0.7  "] * batch_size,
        "memory_valid": torch.ones(batch_size, dtype=torch.bool),
        "memory_condition_kept": torch.ones(batch_size, dtype=torch.bool),
    }
    if advantage:
        batch["advantage_label_global"] = ["positive"] * batch_size
        batch["advantage_condition_kept"] = torch.ones(batch_size, dtype=torch.bool)
    return batch


def _visible_text(tokens: torch.Tensor) -> str:
    hidden = {0, CharacterTokenizer.eos_token_id}
    return "".join(chr(token) for token in tokens.tolist() if token not in hidden)


@pytest.mark.parametrize("config_cls", [PI0Config, PI05Config])
def test_memory_policy_config_safe_defaults(config_cls):
    config = config_cls()

    assert config.use_memory_conditioning is False
    assert config.memory_tokenizer_max_length == 128


@pytest.mark.parametrize("config_cls", [PI0Config, PI05Config])
def test_memory_policy_config_requires_subtask_ar_and_positive_token_budget(config_cls):
    with pytest.raises(ValueError, match="predict_subtask"):
        config_cls(use_memory_conditioning=True, predict_subtask=False)
    with pytest.raises(ValueError, match="memory_tokenizer_max_length"):
        config_cls(
            use_memory_conditioning=True,
            predict_subtask=True,
            memory_tokenizer_max_length=0,
        )


@pytest.mark.parametrize("config_cls", [PI0Config, PI05Config])
def test_memory_policy_config_warns_when_inference_generation_is_disabled(config_cls):
    with pytest.warns(UserWarning, match="cannot update deployment memory"):
        config_cls(
            use_memory_conditioning=True,
            predict_subtask=True,
            subtask_generate_at_inference=False,
        )


def test_shared_subtask_formatter_preserves_current_target_contract():
    assert format_subtask_output(" pick ", 0.46) == "Subtask: pick; Progress: 0.5"
    assert format_subtask_output("place", 1.4) == "Subtask: place; Progress: 1.0"
    assert format_subtask_output("", 0.5) == ""

    result = SubtaskTextProcessorStep()(
        create_transition(
            complementary_data={"subtask": [" pick "], "subtask_progress": torch.tensor([0.46])}
        )
    )
    assert result[TransitionKey.COMPLEMENTARY_DATA]["subtask"] == [
        "Subtask: pick; Progress: 0.5\n"
    ]


def test_training_memory_fields_generate_canonical_text_for_supported_shapes():
    processor = MemoryConditionProcessorStep()
    result = processor(
        create_transition(
            complementary_data={
                "task": ["task-a", "task-b"],
                "memory_subtask": [" pick ", "place"],
                "memory_subtask_progress": torch.tensor([[0.46], [1.4]]),
                "memory_valid": torch.tensor([[True], [True]]),
                "memory_condition_kept": [True, True],
            }
        )
    )[TransitionKey.COMPLEMENTARY_DATA]

    assert result["task"] == [
        "task-a\nMemory: Subtask: pick; Progress: 0.5",
        "task-b\nMemory: Subtask: place; Progress: 1.0",
    ]
    assert result["memory_condition_kept"].dtype == torch.bool
    assert result["memory_condition_kept"].tolist() == [True, True]


def test_inference_memory_text_keeps_complete_prediction_and_normalizes_whitespace():
    processor = MemoryConditionProcessorStep()
    result = processor(
        create_transition(
            complementary_data={
                "task": "task",
                "memory_text": "  Subtask:  open drawer;\n Progress: 0.6  ",
                "memory_valid": True,
            }
        )
    )[TransitionKey.COMPLEMENTARY_DATA]

    assert result["task"] == "task\nMemory: Subtask: open drawer; Progress: 0.6"
    assert result["memory_condition_kept"].tolist() == [True]


def test_training_and_inference_sources_produce_the_same_canonical_memory_block():
    processor = MemoryConditionProcessorStep()
    training = processor(
        create_transition(
            complementary_data={
                "task": "task",
                "memory_subtask": "open drawer",
                "memory_subtask_progress": 0.6,
                "memory_valid": True,
            }
        )
    )[TransitionKey.COMPLEMENTARY_DATA]
    inference = processor(
        create_transition(
            complementary_data={
                "task": "task",
                "memory_text": "Subtask: open drawer; Progress: 0.6",
                "memory_valid": True,
            }
        )
    )[TransitionKey.COMPLEMENTARY_DATA]

    assert training["task"] == inference["task"]


def test_invalid_empty_and_dropped_memory_leave_task_byte_for_byte_unchanged():
    processor = MemoryConditionProcessorStep()
    original_tasks = ["task-a\n", "task-b", " task-c "]
    transition = create_transition(
        complementary_data={
            "task": original_tasks,
            "memory_text": ["history-a", "", "history-c"],
            "memory_valid": torch.tensor([False, True, True]),
            "memory_condition_kept": torch.tensor([True, True, False]),
        }
    )

    result = processor(transition)[TransitionKey.COMPLEMENTARY_DATA]

    assert result["task"] == original_tasks
    assert result["memory_condition_kept"].tolist() == [False, False, False]


def test_missing_all_memory_fields_is_silent_no_memory():
    original = {"task": ["task-a\n", " task-b "]}
    result = MemoryConditionProcessorStep()(create_transition(complementary_data=original))[
        TransitionKey.COMPLEMENTARY_DATA
    ]

    assert result["task"] == original["task"]
    assert result["memory_condition_kept"].tolist() == [False, False]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            {"task": ["a", "b"], "memory_text": ["history"], "memory_valid": [True, True]},
            "does not match task batch size",
        ),
        (
            {"task": ["a"], "memory_text": ["history"], "memory_valid": torch.tensor([1])},
            "boolean",
        ),
        (
            {
                "task": ["a"],
                "memory_subtask": ["history"],
                "memory_subtask_progress": torch.tensor([float("nan")]),
                "memory_valid": [True],
            },
            "finite",
        ),
        (
            {
                "task": ["a"],
                "memory_subtask": ["history"],
                "memory_subtask_progress": ["0.5"],
                "memory_valid": [True],
            },
            "numeric",
        ),
        (
            {
                "task": ["a"],
                "memory_text": ["history"],
                "memory_subtask": ["history"],
                "memory_subtask_progress": [0.5],
                "memory_valid": [True],
            },
            "exactly one memory source",
        ),
    ],
)
def test_memory_processor_rejects_invalid_batch_contract(data, message):
    with pytest.raises(ValueError, match=message):
        MemoryConditionProcessorStep()(create_transition(complementary_data=data))


def test_memory_processor_registry_save_reload_preserves_output(tmp_path):
    processor = MemoryConditionProcessorStep(memory_prefix="Previous memory: ")
    pipeline = DataProcessorPipeline(steps=[processor], name="memory_test")
    pipeline.save_pretrained(tmp_path, config_filename="memory.json")
    loaded = DataProcessorPipeline.from_pretrained(tmp_path, config_filename="memory.json")
    transition = create_transition(
        complementary_data={"task": "task", "memory_text": "history", "memory_valid": True}
    )
    complementary_data = transition[TransitionKey.COMPLEMENTARY_DATA]

    assert ProcessorStepRegistry.get("memory_condition_processor") is MemoryConditionProcessorStep
    assert isinstance(loaded.steps[0], MemoryConditionProcessorStep)
    loaded_result = loaded.process_complementary_data(complementary_data)
    original_result = pipeline.process_complementary_data(complementary_data)
    assert loaded_result["task"] == original_result["task"]
    assert torch.equal(
        loaded_result["memory_condition_kept"], original_result["memory_condition_kept"]
    )


@pytest.mark.parametrize(
    ("config_cls", "factory", "prompt_step_cls", "expected_length"),
    [
        (PI0Config, make_pi0_pre_post_processors, Pi0NewLineProcessor, 128),
        (PI05Config, make_pi05_pre_post_processors, Pi05PrepareStateTokenizerProcessorStep, 200),
    ],
)
def test_memory_pipeline_order_budget_left_truncation_and_subtask_isolation(
    monkeypatch, config_cls, factory, prompt_step_cls, expected_length
):
    tokenizer = CharacterTokenizer()
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    preprocessor, _ = factory(_config(config_cls, advantage=True))
    steps = preprocessor.steps

    advantage_index = next(
        i for i, step in enumerate(steps) if isinstance(step, AdvantageConditionProcessorStep)
    )
    memory_index = next(
        i for i, step in enumerate(steps) if isinstance(step, MemoryConditionProcessorStep)
    )
    prompt_index = next(i for i, step in enumerate(steps) if isinstance(step, prompt_step_cls))
    subtask_index = next(
        i for i, step in enumerate(steps) if isinstance(step, SubtaskTextProcessorStep)
    )
    tokenizer_index = next(
        i for i, step in enumerate(steps) if isinstance(step, TokenizerProcessorStep)
    )
    assert advantage_index < memory_index < prompt_index < subtask_index < tokenizer_index

    tokenizer_step = steps[tokenizer_index]
    assert tokenizer_step.max_length == expected_length
    assert tokenizer_step.truncation_side == "left"

    batch = _batch(2, long_task=True, advantage=True)
    batch["memory_condition_kept"] = torch.tensor([True, False])
    processed = preprocessor(batch)
    kept_prompt = _visible_text(processed[OBS_LANGUAGE_TOKENS][0])
    assert "Memory: Subtask: previous; Progress: 0.7" in kept_prompt
    assert "Advantage: positive" in kept_prompt
    if config_cls is PI05Config:
        assert "State: 128 128;" in kept_prompt
    assert "Memory:" not in _visible_text(processed[OBS_LANGUAGE_TOKENS][1])
    assert torch.equal(
        processed[OBS_LANGUAGE_SUBTASK_TOKENS][0],
        processed[OBS_LANGUAGE_SUBTASK_TOKENS][1],
    )


@pytest.mark.parametrize(
    ("config_cls", "factory", "expected_length"),
    [
        (PI0Config, make_pi0_pre_post_processors, 48),
        (PI05Config, make_pi05_pre_post_processors, 200),
    ],
)
def test_memory_disabled_pipeline_has_no_step_and_keeps_original_budget(
    monkeypatch, config_cls, factory, expected_length
):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    preprocessor, _ = factory(_config(config_cls, memory=False))
    tokenizer_step = next(
        step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep)
    )

    assert not any(isinstance(step, MemoryConditionProcessorStep) for step in preprocessor.steps)
    assert tokenizer_step.max_length == expected_length
    assert tokenizer_step.truncation_side is None


@pytest.mark.parametrize(
    ("config_cls", "factory"),
    [
        (PI0Config, make_pi0_pre_post_processors),
        (PI05Config, make_pi05_pre_post_processors),
    ],
)
def test_memory_alone_enables_left_truncation(monkeypatch, config_cls, factory):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    preprocessor, _ = factory(_config(config_cls, memory=True, advantage=False))
    tokenizer_step = next(
        step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep)
    )

    assert tokenizer_step.truncation_side == "left"
