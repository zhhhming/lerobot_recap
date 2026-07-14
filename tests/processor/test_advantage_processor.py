#!/usr/bin/env python

import re

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
    ProcessorStepRegistry,
    SubtaskTextProcessorStep,
    TokenizerProcessorStep,
)
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class WordTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.truncation_side = "right"
        self.vocab = {"<pad>": self.pad_token_id, "<eos>": self.eos_token_id}

    def _tokens(self, text):
        return re.findall(r"[A-Za-z]+|[-+]?\d+|[^\w\s]", text)

    def _id(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def encode(self, text, add_special_tokens=False):
        ids = [self._id(token) for token in self._tokens(text)]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

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
            ids = self.encode(item, add_special_tokens=True)
            if truncation and len(ids) > max_length:
                ids = ids[-max_length:] if self.truncation_side == "left" else ids[:max_length]
            if padding == "max_length" and len(ids) < max_length:
                pad = [self.pad_token_id] * (max_length - len(ids))
                ids = pad + ids if padding_side == "left" else ids + pad
            encoded.append(ids)
        input_ids = torch.tensor(encoded, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id).to(dtype=torch.long),
        }


def _complementary(result):
    return result[TransitionKey.COMPLEMENTARY_DATA]


def test_advantage_processor_applies_batch_mask_and_normalizes_effective_mask():
    processor = AdvantageConditionProcessorStep()
    transition = create_transition(
        complementary_data={
            "task": ["pick", "place", "finish", "wait"],
            "advantage_label_global": ["positive", "negative", "ignore", ""],
            "advantage_condition_kept": torch.tensor([[True], [False], [True], [True]]),
        }
    )

    result = _complementary(processor(transition))

    assert result["task"] == ["pick\nAdvantage: positive", "place", "finish", "wait"]
    assert torch.equal(
        result["advantage_condition_kept"], torch.tensor([True, False, False, False])
    )


def test_advantage_processor_all_true_and_all_false_masks_model_dropout_extremes():
    processor = AdvantageConditionProcessorStep()
    base = {
        "task": ["pick", "place"],
        "advantage_label_global": ["positive", "negative"],
    }

    kept = _complementary(
        processor(
            create_transition(
                complementary_data={**base, "advantage_condition_kept": [True, True]}
            )
        )
    )
    dropped = _complementary(
        processor(
            create_transition(
                complementary_data={**base, "advantage_condition_kept": [False, False]}
            )
        )
    )

    assert kept["task"] == ["pick\nAdvantage: positive", "place\nAdvantage: negative"]
    assert dropped["task"] == ["pick", "place"]
    assert kept["advantage_condition_kept"].tolist() == [True, True]
    assert dropped["advantage_condition_kept"].tolist() == [False, False]


@pytest.mark.parametrize(
    ("inference_label", "expected_task", "expected_kept"),
    [
        ("positive", "deploy\nAdvantage: positive", True),
        ("negative", "deploy\nAdvantage: negative", True),
        ("none", "deploy", False),
    ],
)
def test_advantage_processor_inference_fallback_without_dataset_controls(
    inference_label, expected_task, expected_kept
):
    processor = AdvantageConditionProcessorStep(inference_label=inference_label)
    result = _complementary(
        processor(create_transition(complementary_data={"task": "deploy"}))
    )

    assert result["task"] == expected_task
    assert result["advantage_condition_kept"].tolist() == [expected_kept]


def test_training_mask_without_label_does_not_use_inference_fallback():
    processor = AdvantageConditionProcessorStep(inference_label="positive")
    result = _complementary(
        processor(
            create_transition(
                complementary_data={
                    "task": ["train a", "train b"],
                    "advantage_condition_kept": torch.tensor([True, False]),
                }
            )
        )
    )

    assert result["task"] == ["train a", "train b"]
    assert result["advantage_condition_kept"].tolist() == [False, False]


def test_custom_label_key_and_format():
    processor = AdvantageConditionProcessorStep(
        label_key="advantage_label_subtask",
        condition_format="Quality={label}",
    )
    result = _complementary(
        processor(
            create_transition(
                complementary_data={
                    "task": "strike",
                    "advantage_label_subtask": "positive",
                }
            )
        )
    )

    assert result["task"] == "strike\nQuality=positive"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            {"task": ["a", "b"], "advantage_label_global": ["positive"]},
            "does not match task batch size",
        ),
        (
            {"task": ["a"], "advantage_label_global": ["maybe"]},
            "Unsupported advantage label",
        ),
        (
            {
                "task": ["a"],
                "advantage_label_global": ["positive"],
                "advantage_condition_kept": torch.tensor([1]),
            },
            "must contain boolean",
        ),
    ],
)
def test_advantage_processor_rejects_invalid_batch_contract(data, message):
    with pytest.raises(ValueError, match=message):
        AdvantageConditionProcessorStep()(create_transition(complementary_data=data))


def test_advantage_processor_registry_config_and_pipeline_reload(tmp_path):
    processor = AdvantageConditionProcessorStep(
        label_key="advantage_label_subtask",
        condition_format="Adv: {label}",
        inference_label="negative",
    )
    pipeline = DataProcessorPipeline(steps=[processor], name="advantage_test")
    pipeline.save_pretrained(tmp_path, config_filename="advantage.json")
    loaded = DataProcessorPipeline.from_pretrained(tmp_path, config_filename="advantage.json")

    assert ProcessorStepRegistry.get("advantage_condition_processor") is AdvantageConditionProcessorStep
    assert isinstance(loaded.steps[0], AdvantageConditionProcessorStep)
    assert loaded.steps[0].get_config() == processor.get_config()


@pytest.mark.parametrize("config_cls", [PI0Config, PI05Config])
def test_advantage_policy_config_defaults_and_validation(config_cls):
    config = config_cls()
    assert config.use_advantage_conditioning is False
    assert config.advantage_label_key == "advantage_label_global"
    assert config.advantage_loss_weight_key == "advantage_loss_weight_global"
    assert config.inference_advantage_label == "positive"

    with pytest.raises(ValueError, match="inference_advantage_label"):
        config_cls(inference_advantage_label="ignore")
    with pytest.raises(ValueError, match=r"include the \{label\} placeholder"):
        config_cls(advantage_condition_format="Advantage")


def _policy_config(config_cls, *, enabled=True, predict_subtask=True, max_length=None):
    kwargs = {
        "use_advantage_conditioning": enabled,
        "predict_subtask": predict_subtask,
        "device": "cpu",
    }
    if max_length is not None:
        kwargs["tokenizer_max_length"] = max_length
    config = config_cls(**kwargs)
    config.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    return config


def _make_batch(task):
    batch_size = len(task)
    return {
        OBS_STATE: torch.zeros(batch_size, 2),
        ACTION: torch.zeros(batch_size, 2),
        "task": task,
        "subtask": ["same subtask"] * batch_size,
        "subtask_progress": torch.full((batch_size,), 0.5),
        "advantage_label_global": ["positive", "negative"][:batch_size],
        "advantage_condition_kept": torch.tensor([True, False][:batch_size]),
    }


@pytest.mark.parametrize(
    ("config_cls", "factory", "prompt_step_cls"),
    [
        (PI0Config, make_pi0_pre_post_processors, Pi0NewLineProcessor),
        (PI05Config, make_pi05_pre_post_processors, Pi05PrepareStateTokenizerProcessorStep),
    ],
)
def test_pi0_pi05_wiring_prompt_order_and_subtask_isolation(
    monkeypatch, config_cls, factory, prompt_step_cls
):
    tokenizer = WordTokenizer()
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    config = _policy_config(config_cls)
    preprocessor, _ = factory(config)

    advantage_index = next(
        i for i, step in enumerate(preprocessor.steps) if isinstance(step, AdvantageConditionProcessorStep)
    )
    prompt_index = next(i for i, step in enumerate(preprocessor.steps) if isinstance(step, prompt_step_cls))
    subtask_index = next(
        i for i, step in enumerate(preprocessor.steps) if isinstance(step, SubtaskTextProcessorStep)
    )
    tokenizer_index = next(
        i for i, step in enumerate(preprocessor.steps) if isinstance(step, TokenizerProcessorStep)
    )
    assert advantage_index < prompt_index < subtask_index < tokenizer_index

    processed = preprocessor(_make_batch(["pick cube", "place cube"]))
    assert "Advantage: positive" in processed["task"][0]
    assert "Advantage:" not in processed["task"][1]
    assert processed["advantage_condition_kept"].tolist() == [True, False]
    assert torch.equal(
        processed[OBS_LANGUAGE_SUBTASK_TOKENS][0],
        processed[OBS_LANGUAGE_SUBTASK_TOKENS][1],
    )
    if config_cls is PI05Config:
        assert processed["task"][0].startswith("Task: pick cube Advantage: positive, State:")


@pytest.mark.parametrize(
    ("config_cls", "factory"),
    [
        (PI0Config, make_pi0_pre_post_processors),
        (PI05Config, make_pi05_pre_post_processors),
    ],
)
def test_disabled_pipeline_is_unmodified_and_conditioning_preserves_long_prompt_suffix(
    monkeypatch, config_cls, factory
):
    disabled_tokenizer = WordTokenizer()
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: disabled_tokenizer,
    )
    disabled, _ = factory(_policy_config(config_cls, enabled=False, predict_subtask=False))
    assert not any(isinstance(step, AdvantageConditionProcessorStep) for step in disabled.steps)
    disabled_tokenizer_step = next(
        step for step in disabled.steps if isinstance(step, TokenizerProcessorStep)
    )
    assert disabled_tokenizer_step.truncation_side is None

    enabled_tokenizer = WordTokenizer()
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: enabled_tokenizer,
    )
    config = _policy_config(config_cls, enabled=True, predict_subtask=False, max_length=40)
    enabled, _ = factory(config)
    long_task = " ".join(["word"] * 500)
    batch = _make_batch([long_task])
    processed = enabled(batch)

    tokenizer_step = next(step for step in enabled.steps if isinstance(step, TokenizerProcessorStep))
    assert tokenizer_step.truncation_side == "left"
    advantage_id = enabled_tokenizer.vocab["Advantage"]
    positive_id = enabled_tokenizer.vocab["positive"]
    main_tokens = processed[OBS_LANGUAGE_TOKENS][0].tolist()
    assert advantage_id in main_tokens
    assert positive_id in main_tokens
