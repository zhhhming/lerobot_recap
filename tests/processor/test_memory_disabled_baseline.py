#!/usr/bin/env python

"""Golden regression tests for the pre-memory PI0/PI0.5 prompt pipelines."""

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
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class GoldenCharacterTokenizer:
    """Small deterministic tokenizer that makes prompt bytes visible in assertions."""

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
                padding_tokens = [self.pad_token_id] * (max_length - len(token_ids))
                token_ids = (
                    padding_tokens + token_ids if padding_side == "left" else token_ids + padding_tokens
                )
            encoded.append(token_ids)

        input_ids = torch.tensor(encoded, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id).to(dtype=torch.long),
        }


def _config(config_cls):
    config = config_cls(
        predict_subtask=False,
        use_advantage_conditioning=False,
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
    }


def _expected_tokens(prompt: str, max_length: int) -> torch.Tensor:
    token_ids = [ord(character) for character in prompt] + [GoldenCharacterTokenizer.eos_token_id]
    return torch.tensor(
        [token_ids + [GoldenCharacterTokenizer.pad_token_id] * (max_length - len(token_ids))],
        dtype=torch.long,
    )


def test_pi0_memory_disabled_prompt_and_token_tensor_golden(monkeypatch):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: GoldenCharacterTokenizer(),
    )
    preprocessor, _ = make_pi0_pre_post_processors(_config(PI0Config))

    processed = preprocessor(_batch())

    expected_prompt = "pick cube\n"
    assert processed["task"] == [expected_prompt]
    assert torch.equal(processed[OBS_LANGUAGE_TOKENS], _expected_tokens(expected_prompt, 48))
    assert torch.equal(
        processed[OBS_LANGUAGE_ATTENTION_MASK],
        _expected_tokens(expected_prompt, 48).ne(GoldenCharacterTokenizer.pad_token_id),
    )

    tokenizer_step = next(step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep))
    assert tokenizer_step.max_length == 48
    assert tokenizer_step.truncation_side is None
    assert not any(type(step).__name__.startswith("Memory") for step in preprocessor.steps)
    assert not any(key.startswith("memory_") for key in processed)


def test_pi05_memory_disabled_prompt_and_token_tensor_golden(monkeypatch):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: GoldenCharacterTokenizer(),
    )
    preprocessor, _ = make_pi05_pre_post_processors(_config(PI05Config))

    processed = preprocessor(_batch())

    expected_prompt = "Task: pick cube, State: 128 128;\nAction: "
    assert processed["task"] == [expected_prompt]
    assert torch.equal(processed[OBS_LANGUAGE_TOKENS], _expected_tokens(expected_prompt, 200))
    assert torch.equal(
        processed[OBS_LANGUAGE_ATTENTION_MASK],
        _expected_tokens(expected_prompt, 200).ne(GoldenCharacterTokenizer.pad_token_id),
    )

    tokenizer_step = next(step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep))
    assert tokenizer_step.max_length == 200
    assert tokenizer_step.truncation_side is None
    assert not any(type(step).__name__.startswith("Memory") for step in preprocessor.steps)
    assert not any(key.startswith("memory_") for key in processed)
