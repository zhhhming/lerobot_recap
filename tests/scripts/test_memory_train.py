#!/usr/bin/env python

"""Milestone 3 integration tests for memory-aware offline training."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.processor import MemoryConditionProcessorStep, TokenizerProcessorStep
from lerobot.scripts.lerobot_train import make_train_pre_post_processors, update_policy
from lerobot.utils.advantage_weights import sample_advantage_condition_mask
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE, PRETRAINED_MODEL_DIR
from lerobot.utils.memory_conditioning import MemoryTrainingMetrics, sample_memory_condition_mask
from lerobot.utils.train_utils import load_training_state, save_checkpoint


class CharacterTokenizer:
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


class _BatchDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {
            "memory_valid": index != 0,
            "memory_subtask": "history" if index != 0 else "",
            "memory_subtask_progress": torch.tensor(0.5),
            "memory_frame_offset": index + 1,
        }


class _Accelerator:
    num_processes = 1

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, policy, keep_fp32_wrapper=True):
        return policy


class _TinyPolicy(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = config

    def forward(self, batch):
        assert OBS_LANGUAGE_TOKENS in batch
        assert "memory_condition_kept" in batch
        loss = self.scale.square() + batch[OBS_LANGUAGE_TOKENS].float().sum() * 0.0
        return loss, {"loss": loss.detach().item()}


class _CheckpointPolicy:
    def __init__(self, config):
        self.config = config

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(path)


def _policy_config(config_cls, *, memory=True):
    config = config_cls(
        predict_subtask=True,
        use_memory_conditioning=memory,
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


def _train_config(tmp_path, config_cls=PI0Config, **kwargs):
    policy = kwargs.pop("policy", _policy_config(config_cls))
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/memory", streaming=False),
        policy=policy,
        output_dir=tmp_path / "output",
        use_policy_training_preset=False,
        optimizer=policy.get_optimizer_preset(),
        scheduler=policy.get_scheduler_preset(),
        **kwargs,
    )


def _raw_batch(batch_size):
    return {
        OBS_STATE: torch.zeros(batch_size, 2),
        ACTION: torch.zeros(batch_size, 2),
        "task": ["do task"] * batch_size,
        "subtask": ["current"] * batch_size,
        "subtask_progress": torch.full((batch_size,), 0.4),
        "memory_subtask": ["previous"] * batch_size,
        "memory_subtask_progress": torch.full((batch_size,), 0.6),
        "memory_valid": torch.ones(batch_size, dtype=torch.bool),
        "memory_frame_offset": torch.arange(1, batch_size + 1),
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"memory_lookback_min_frames": 0}, "memory lookback"),
        ({"memory_lookback_min_frames": 5, "memory_lookback_max_frames": 4}, "memory lookback"),
        ({"memory_dropout_prob": -0.1}, "memory_dropout_prob"),
        ({"memory_dropout_prob": 1.1}, "memory_dropout_prob"),
    ],
)
def test_memory_train_config_rejects_invalid_values(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _train_config(tmp_path, **kwargs).validate()


def test_memory_train_config_rejects_streaming_and_has_safe_defaults(tmp_path):
    config = _train_config(tmp_path)
    assert config.memory_lookback_min_frames == 1
    assert config.memory_lookback_max_frames == 12
    assert config.memory_dropout_prob == 0.2

    config.dataset.streaming = True
    with pytest.raises(ValueError, match="non-streaming"):
        config.validate()


@pytest.mark.parametrize("advantage_dropout", [0.0, 1.0])
@pytest.mark.parametrize("memory_dropout", [0.0, 1.0])
def test_advantage_and_memory_dropout_form_all_four_independent_combinations(
    advantage_dropout, memory_dropout
):
    batch = {
        "advantage_label_global": ["positive", "negative"],
        "memory_valid": torch.ones(2, dtype=torch.bool),
        "memory_subtask": ["a", "b"],
    }
    with_advantage = sample_advantage_condition_mask(
        batch,
        label_key="advantage_label_global",
        dropout_prob=advantage_dropout,
    )
    result = sample_memory_condition_mask(with_advantage, dropout_prob=memory_dropout)

    assert result["advantage_condition_kept"].tolist() == [
        advantage_dropout == 0.0,
        advantage_dropout == 0.0,
    ]
    assert result["memory_condition_kept"].tolist() == [
        memory_dropout == 0.0,
        memory_dropout == 0.0,
    ]
    assert result["advantage_label_global"] == batch["advantage_label_global"]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_memory_mask_handles_batch_one_last_small_batch_and_workers(num_workers):
    loader = torch.utils.data.DataLoader(
        _BatchDataset(), batch_size=2, num_workers=num_workers, drop_last=False
    )
    sizes = []
    for batch in loader:
        result = sample_memory_condition_mask(batch, dropout_prob=0.0)
        sizes.append(result["memory_condition_kept"].shape[0])
    assert sizes == [2, 1]


@pytest.mark.parametrize(
    ("config_cls", "factory", "expected_length"),
    [
        (PI0Config, make_pi0_pre_post_processors, 128),
        (PI05Config, make_pi05_pre_post_processors, 200),
    ],
)
def test_old_checkpoint_cli_memory_override_rebuilds_real_train_processor(
    caplog, monkeypatch, tmp_path, config_cls, factory, expected_length
):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    old_dir = tmp_path / "old"
    old_config = _policy_config(config_cls, memory=False)
    old_pre, old_post = factory(old_config, dataset_stats={})
    old_config.save_pretrained(old_dir)
    old_pre.save_pretrained(old_dir)
    old_post.save_pretrained(old_dir)

    current_config = PreTrainedConfig.from_pretrained(
        old_dir,
        cli_overrides=["--use_memory_conditioning=true", "--predict_subtask=true"],
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
    assert "structural processor config change" in caplog.text
    assert any(isinstance(step, MemoryConditionProcessorStep) for step in preprocessor.steps)
    tokenizer = next(step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep))
    assert tokenizer.max_length == expected_length


@pytest.mark.parametrize("config_cls", [PI0Config, PI05Config])
def test_checkpoint_save_reload_and_resume_preserve_memory_processor(
    monkeypatch, tmp_path, config_cls
):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    cfg = _train_config(tmp_path, config_cls=config_cls)
    policy = _CheckpointPolicy(cfg.policy)
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
    processor_config = json.loads((pretrained_dir / "policy_preprocessor.json").read_text())
    registry_names = [step["registry_name"] for step in processor_config["steps"]]
    assert "memory_condition_processor" in registry_names
    tokenizer_config = next(
        step["config"]
        for step in processor_config["steps"]
        if step["registry_name"] == "tokenizer_processor"
    )
    expected_length = 128 if config_cls is PI0Config else 200
    assert tokenizer_config["max_length"] == expected_length

    resume_cfg = _train_config(tmp_path / "resume", policy=cfg.policy)
    resume_cfg.resume = True
    resume_cfg.policy.pretrained_path = pretrained_dir
    resumed_preprocessor, _ = make_train_pre_post_processors(
        cfg=resume_cfg,
        policy=policy,
        dataset=dataset,
        device=torch.device("cpu"),
    )
    assert any(
        isinstance(step, MemoryConditionProcessorStep) for step in resumed_preprocessor.steps
    )
    loaded_step, _, _ = load_training_state(checkpoint_dir, optimizer, None)
    assert loaded_step == 2


@pytest.mark.parametrize(
    ("config_cls", "factory"),
    [
        (PI0Config, make_pi0_pre_post_processors),
        (PI05Config, make_pi05_pre_post_processors),
    ],
)
def test_each_policy_completes_two_small_memory_updates(monkeypatch, config_cls, factory):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    config = _policy_config(config_cls)
    preprocessor, _ = factory(config, dataset_stats={})
    policy = _TinyPolicy(config)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    metrics = SimpleNamespace(loss=None, grad_norm=None, lr=None, update_s=None)
    accumulator = MemoryTrainingMetrics()

    for batch_size in (2, 1):
        batch = sample_memory_condition_mask(_raw_batch(batch_size), dropout_prob=0.0)
        accumulator.update(batch)
        processed = preprocessor(batch)
        metrics, _ = update_policy(
            metrics,
            policy,
            processed,
            optimizer,
            grad_clip_norm=0.0,
            accelerator=_Accelerator(),
        )

    assert policy.scale.item() < 1.0
    assert accumulator.to_dict()["memory/condition_kept_fraction"] == 1.0
