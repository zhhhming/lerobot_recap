#!/usr/bin/env python

import pytest
import torch

from lerobot.configs.types import PipelineFeatureType
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import (
    Pi05PrepareStateTokenizerProcessorStep,
    make_pi05_pre_post_processors,
)
from lerobot.processor import SubtaskTextProcessorStep, TokenizerProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
)


class MockTokenizer:
    eos_token_id = 99

    def __call__(
        self,
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        padding_side="right",
        return_tensors="pt",
        **kwargs,
    ):
        texts = [text] if isinstance(text, str) else text
        input_ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_length, dtype=torch.long)
        for i, item in enumerate(texts):
            token_ids = self.encode(item, add_special_tokens=True)
            token_ids = token_ids[:max_length]
            input_ids[i, : len(token_ids)] = torch.tensor(token_ids)
            attention_mask[i, : len(token_ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text, add_special_tokens=False):
        if not text:
            return []
        token_ids = [ord(ch) % 50 + 1 for ch in text]
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids


def test_subtask_text_processor_formats_batch_and_empty_values():
    processor = SubtaskTextProcessorStep()
    transition = create_transition(
        complementary_data={
            "task": ["task a", "task b", "task c"],
            "subtask": [" pick ", "", "place"],
            "subtask_progress": torch.tensor([0.46, 0.9, 1.4]),
        }
    )

    result = processor(transition)

    assert result[TransitionKey.COMPLEMENTARY_DATA]["subtask"] == [
        "Subtask: pick; Progress: 0.5\n",
        "",
        "Subtask: place; Progress: 1.0\n",
    ]


def test_subtask_text_processor_missing_subtask_is_noop():
    processor = SubtaskTextProcessorStep()
    transition = create_transition(complementary_data={"task": ["task only"]})

    result = processor(transition)

    assert "subtask" not in result[TransitionKey.COMPLEMENTARY_DATA]


def test_subtask_text_processor_missing_progress_defaults_to_one(caplog):
    processor = SubtaskTextProcessorStep()
    transition = create_transition(complementary_data={"subtask": "pick"})

    result = processor(transition)

    assert result[TransitionKey.COMPLEMENTARY_DATA]["subtask"] == "Subtask: pick; Progress: 1.0\n"
    assert "subtask_progress is missing" in caplog.text


def test_tokenizer_ignores_subtask_by_default():
    processor = TokenizerProcessorStep(tokenizer=MockTokenizer(), max_length=8)
    transition = create_transition(
        observation={},
        complementary_data={
            "task": ["pick"],
            "subtask": ["Subtask: pick; Progress: 0.5\n"],
        }
    )

    result = processor(transition)

    observation = result[TransitionKey.OBSERVATION]
    assert OBS_LANGUAGE_SUBTASK_TOKENS not in observation
    assert OBS_LANGUAGE_SUBTASK_ATTENTION_MASK not in observation


def test_tokenizer_subtask_branch_appends_eos_and_pads_empty_text():
    processor = TokenizerProcessorStep(
        tokenizer=MockTokenizer(),
        max_length=8,
        tokenize_subtask=True,
        subtask_max_length=6,
    )
    transition = create_transition(
        observation={},
        complementary_data={
            "task": ["pick", "place"],
            "subtask": ["go", ""],
        }
    )

    result = processor(transition)

    observation = result[TransitionKey.OBSERVATION]
    tokens = observation[OBS_LANGUAGE_SUBTASK_TOKENS]
    masks = observation[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK]
    assert tokens.shape == (2, 6)
    assert masks.shape == (2, 6)
    assert tokens[0, 2].item() == MockTokenizer.eos_token_id
    assert masks[0].tolist() == [True, True, True, False, False, False]
    assert tokens[1].tolist() == [0, 0, 0, 0, 0, 0]
    assert masks[1].tolist() == [False, False, False, False, False, False]


def test_tokenizer_config_and_features_include_subtask_fields_when_enabled():
    processor = TokenizerProcessorStep(
        tokenizer=MockTokenizer(),
        max_length=8,
        tokenize_subtask=True,
        subtask_max_length=6,
    )

    config = processor.get_config()
    features = processor.transform_features({PipelineFeatureType.OBSERVATION: {}})

    assert config["tokenize_subtask"] is True
    assert config["subtask_max_length"] == 6
    assert features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_SUBTASK_TOKENS].shape == (6,)
    assert features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_SUBTASK_ATTENTION_MASK].shape == (6,)


def test_pi05_prepare_state_prompt_suffix_switch():
    transition = create_transition(
        observation={"observation.state": torch.zeros(1, 2)},
        complementary_data={"task": ["pick_cube"]},
    )

    default_step = Pi05PrepareStateTokenizerProcessorStep()
    subtask_step = Pi05PrepareStateTokenizerProcessorStep(omit_action_suffix=True)

    default_prompt = default_step(transition)[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    subtask_prompt = subtask_step(transition)[TransitionKey.COMPLEMENTARY_DATA]["task"][0]

    assert default_prompt.endswith("\nAction: ")
    assert subtask_prompt.endswith(";\n")
    assert "Action: " not in subtask_prompt


@pytest.mark.parametrize("config_cls", [PI0Config, PI05Config])
def test_subtask_config_validation_rejects_train_expert_only(config_cls):
    with pytest.raises(ValueError, match="requires train_expert_only=False"):
        config_cls(predict_subtask=True, train_expert_only=True)


@pytest.mark.parametrize("config_cls", [PI0Config, PI05Config])
def test_subtask_config_warns_when_decode_limit_exceeds_train_limit(config_cls):
    with pytest.warns(UserWarning, match="subtask_max_decode_tokens"):
        config_cls(predict_subtask=True, subtask_max_tokens=8, subtask_max_decode_tokens=12)


def test_pi0_and_pi05_pipeline_wiring(monkeypatch):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: MockTokenizer(),
    )

    pi0_pre, _ = make_pi0_pre_post_processors(PI0Config(predict_subtask=True, subtask_max_tokens=7))
    pi05_pre, _ = make_pi05_pre_post_processors(PI05Config(predict_subtask=True, subtask_max_tokens=9))
    pi05_default_pre, _ = make_pi05_pre_post_processors(PI05Config())

    pi0_step_names = [type(step).__name__ for step in pi0_pre.steps]
    pi05_step_names = [type(step).__name__ for step in pi05_pre.steps]
    assert "SubtaskTextProcessorStep" in pi0_step_names
    assert "SubtaskTextProcessorStep" in pi05_step_names

    pi0_tokenizer = next(step for step in pi0_pre.steps if isinstance(step, TokenizerProcessorStep))
    pi05_tokenizer = next(step for step in pi05_pre.steps if isinstance(step, TokenizerProcessorStep))
    assert pi0_tokenizer.tokenize_subtask is True
    assert pi0_tokenizer.subtask_max_length == 7
    assert pi05_tokenizer.tokenize_subtask is True
    assert pi05_tokenizer.subtask_max_length == 9

    pi05_prepare = next(
        step for step in pi05_pre.steps if isinstance(step, Pi05PrepareStateTokenizerProcessorStep)
    )
    pi05_default_prepare = next(
        step for step in pi05_default_pre.steps if isinstance(step, Pi05PrepareStateTokenizerProcessorStep)
    )
    assert pi05_prepare.omit_action_suffix is True
    assert pi05_default_prepare.omit_action_suffix is False
    assert "SubtaskTextProcessorStep" not in [type(step).__name__ for step in pi05_default_pre.steps]
