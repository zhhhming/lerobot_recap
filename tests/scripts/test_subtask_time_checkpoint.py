#!/usr/bin/env python

"""Checkpoint and processor-structure contracts for elapsed-time conditioning."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.processor import SubtaskTimeConditionProcessorStep, TokenizerProcessorStep
from lerobot.scripts.lerobot_train import make_train_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE, PRETRAINED_MODEL_DIR
from lerobot.utils.train_utils import save_checkpoint


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


class _CheckpointPolicy:
    def __init__(self, config):
        self.config = config

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(path)


def _policy_config(config_cls, *, time=True):
    config = config_cls(
        predict_subtask=True,
        use_subtask_time_conditioning=time,
        use_memory_conditioning=False,
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


def _train_config(tmp_path, *, policy):
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/subtask-time", streaming=False),
        policy=policy,
        output_dir=tmp_path / "output",
        use_policy_training_preset=False,
        optimizer=policy.get_optimizer_preset(),
        scheduler=policy.get_scheduler_preset(),
    )


@pytest.mark.parametrize("config_cls", [PI0Config, PI05Config])
def test_old_config_without_time_fields_loads_safe_defaults(tmp_path, config_cls):
    config = _policy_config(config_cls, time=False)
    config.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    serialized = json.loads(config_path.read_text())
    serialized.pop("use_subtask_time_conditioning")
    serialized.pop("subtask_time_tokenizer_max_length")
    config_path.write_text(json.dumps(serialized))

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert loaded.use_subtask_time_conditioning is False
    assert loaded.subtask_time_tokenizer_max_length == 128


@pytest.mark.parametrize(
    ("config_cls", "factory", "expected_length"),
    [
        (PI0Config, make_pi0_pre_post_processors, 128),
        (PI05Config, make_pi05_pre_post_processors, 200),
    ],
)
def test_old_checkpoint_cli_time_override_rebuilds_real_train_processor(
    caplog, monkeypatch, tmp_path, config_cls, factory, expected_length
):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    old_dir = tmp_path / "old"
    old_config = _policy_config(config_cls, time=False)
    old_pre, old_post = factory(old_config, dataset_stats={})
    old_config.save_pretrained(old_dir)
    old_pre.save_pretrained(old_dir)
    old_post.save_pretrained(old_dir)

    current_config = PreTrainedConfig.from_pretrained(
        old_dir,
        cli_overrides=["--use_subtask_time_conditioning=true", "--predict_subtask=true"],
    )
    current_config.pretrained_path = old_dir
    cfg = _train_config(tmp_path, policy=current_config)
    policy = SimpleNamespace(config=current_config)
    dataset = SimpleNamespace(meta=SimpleNamespace(stats={}))

    with caplog.at_level("INFO"):
        preprocessor, _ = make_train_pre_post_processors(
            cfg=cfg,
            policy=policy,
            dataset=dataset,
            device=torch.device("cpu"),
        )

    assert current_config.pretrained_path == old_dir
    assert "use_subtask_time_conditioning" in caplog.text
    assert any(
        isinstance(step, SubtaskTimeConditionProcessorStep) for step in preprocessor.steps
    )
    tokenizer = next(step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep))
    assert tokenizer.max_length == expected_length


@pytest.mark.parametrize("config_cls", [PI0Config, PI05Config])
def test_checkpoint_save_reload_resume_and_deploy_load_preserve_time_processor(
    monkeypatch, tmp_path, config_cls
):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    policy_config = _policy_config(config_cls)
    cfg = _train_config(tmp_path, policy=policy_config)
    policy = _CheckpointPolicy(policy_config)
    dataset = SimpleNamespace(meta=SimpleNamespace(stats={}))
    preprocessor, postprocessor = make_train_pre_post_processors(
        cfg=cfg,
        policy=policy,
        dataset=dataset,
        device=torch.device("cpu"),
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=2,
        cfg=cfg,
        policy=policy,
        optimizer=optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    saved_policy_config = json.loads((pretrained_dir / "config.json").read_text())
    assert saved_policy_config["use_subtask_time_conditioning"] is True
    assert saved_policy_config["subtask_time_tokenizer_max_length"] == 128
    processor_config = json.loads((pretrained_dir / "policy_preprocessor.json").read_text())
    registry_names = [step["registry_name"] for step in processor_config["steps"]]
    assert registry_names.count("subtask_time_condition_processor") == 1
    tokenizer_config = next(
        step["config"]
        for step in processor_config["steps"]
        if step["registry_name"] == "tokenizer_processor"
    )
    assert tokenizer_config["max_length"] == (128 if config_cls is PI0Config else 200)

    resume_cfg = _train_config(tmp_path / "resume", policy=policy_config)
    resume_cfg.resume = True
    resume_cfg.policy.pretrained_path = pretrained_dir
    resumed_preprocessor, _ = make_train_pre_post_processors(
        cfg=resume_cfg,
        policy=policy,
        dataset=dataset,
        device=torch.device("cpu"),
    )
    assert any(
        isinstance(step, SubtaskTimeConditionProcessorStep)
        for step in resumed_preprocessor.steps
    )

    deploy_preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=pretrained_dir,
    )
    assert any(
        isinstance(step, SubtaskTimeConditionProcessorStep)
        for step in deploy_preprocessor.steps
    )
