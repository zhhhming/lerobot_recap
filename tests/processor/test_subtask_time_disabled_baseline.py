#!/usr/bin/env python

"""Golden prompt/tensor regressions while subtask-time conditioning is disabled."""

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.processor import TokenizerProcessorStep
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class GoldenCharacterTokenizer:
    """Deterministic character tokenizer that exposes prompt bytes in tensors."""

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


def _config(config_cls):
    config = config_cls(
        predict_subtask=True,
        use_advantage_conditioning=False,
        use_memory_conditioning=False,
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


def _batch():
    return {
        OBS_STATE: torch.zeros(1, 2),
        ACTION: torch.zeros(1, 2),
        "task": ["pick cube"],
        "subtask": ["grasp"],
        "subtask_progress": torch.tensor([0.4]),
    }


def _expected_tokens(prompt: str, max_length: int) -> torch.Tensor:
    token_ids = [ord(character) for character in prompt] + [GoldenCharacterTokenizer.eos_token_id]
    return torch.tensor(
        [token_ids + [GoldenCharacterTokenizer.pad_token_id] * (max_length - len(token_ids))],
        dtype=torch.long,
    )


def _assert_time_disabled_output(processed, preprocessor, expected_prompt, main_length):
    expected_main = _expected_tokens(expected_prompt, main_length)
    expected_subtask = _expected_tokens("Subtask: grasp; Progress: 0.4\n", 48)

    assert processed["task"] == [expected_prompt]
    assert torch.equal(processed[OBS_LANGUAGE_TOKENS], expected_main)
    assert torch.equal(processed[OBS_LANGUAGE_ATTENTION_MASK], expected_main.ne(0))
    assert torch.equal(processed[OBS_LANGUAGE_SUBTASK_TOKENS], expected_subtask)
    assert torch.equal(processed[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK], expected_subtask.ne(0))
    assert not any("SubtaskTime" in type(step).__name__ for step in preprocessor.steps)
    assert not any(key.startswith("subtask_time_") for key in processed)


def test_pi0_time_disabled_prompt_and_token_tensors_are_unchanged(monkeypatch):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: GoldenCharacterTokenizer(),
    )
    preprocessor, _ = make_pi0_pre_post_processors(_config(PI0Config))

    processed = preprocessor(_batch())

    _assert_time_disabled_output(processed, preprocessor, "pick cube\n", 48)
    tokenizer = next(step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep))
    assert tokenizer.max_length == 48
    assert tokenizer.truncation_side is None


def test_pi05_time_disabled_prompt_and_token_tensors_are_unchanged(monkeypatch):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: GoldenCharacterTokenizer(),
    )
    preprocessor, _ = make_pi05_pre_post_processors(_config(PI05Config))

    processed = preprocessor(_batch())

    _assert_time_disabled_output(
        processed,
        preprocessor,
        "Task: pick cube, State: 128 128;\n",
        200,
    )
    tokenizer = next(step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep))
    assert tokenizer.max_length == 200
    assert tokenizer.truncation_side is None
