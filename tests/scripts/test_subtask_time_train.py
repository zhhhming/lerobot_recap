#!/usr/bin/env python

"""Milestone T2 integration tests for subtask-time-aware offline training."""

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import apply_subtask_attention_dropout
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.lerobot_train import prepare_training_batch_conditions, update_policy
from lerobot.utils.subtask_time_conditioning import SubtaskTimeTrainingMetrics


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


class _TinyTimePolicy(torch.nn.Module):
    def __init__(self, policy_type):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(type=policy_type)

    def forward(self, batch):
        assert batch["subtask_time_seconds"].shape == (2,)
        assert batch["subtask_time_condition_kept"].dtype == torch.bool
        assert torch.isfinite(batch["subtask_time_seconds"]).all()
        loss = self.scale.square() + batch["subtask_time_seconds"].sum() * 0.0
        return loss, {"loss": loss.detach().item()}


def _policy_config(config_cls=PI0Config, *, enabled=True, predict_subtask=True):
    policy = config_cls(
        predict_subtask=predict_subtask,
        device="cpu",
        push_to_hub=False,
    )
    policy.use_subtask_time_conditioning = enabled
    return policy


def _train_config(tmp_path, *, policy=None, streaming=False, **kwargs):
    policy = _policy_config() if policy is None else policy
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/subtask-time", streaming=streaming),
        policy=policy,
        output_dir=tmp_path / "output",
        use_policy_training_preset=False,
        optimizer=policy.get_optimizer_preset(),
        scheduler=policy.get_scheduler_preset(),
        **kwargs,
    )


def _condition_cfg(*, advantage=True, memory=True, time=True):
    return SimpleNamespace(
        policy=SimpleNamespace(
            use_advantage_conditioning=advantage,
            use_memory_conditioning=memory,
            use_subtask_time_conditioning=time,
        ),
        advantage_label_key="advantage_label_global",
        advantage_condition_dropout_prob=0.0,
        advantage_ignore_label="ignore",
        memory_dropout_prob=0.0,
        subtask_time_noise_ratio=0.0,
        subtask_time_noise_max_seconds=5.0,
        subtask_time_dropout_prob=0.0,
    )


def _raw_batch():
    return {
        "advantage_label_global": ["positive", "negative"],
        "memory_valid": torch.tensor([True, True]),
        "memory_subtask": ["previous a", "previous b"],
        "memory_frame_offset": torch.tensor([1, 2]),
        "subtask_elapsed_seconds": torch.tensor([0.0, 40.5]),
        "subtask_time_valid": torch.tensor([True, True]),
        "subtask": ["current a", "current b"],
        "subtask_progress": torch.tensor([0.2, 0.6]),
    }


def test_train_config_has_safe_defaults_and_accepts_zero_noise_ablation(tmp_path):
    config = _train_config(tmp_path)
    assert config.subtask_time_noise_ratio == 0.4
    assert config.subtask_time_noise_max_seconds == 5.0
    assert config.subtask_time_dropout_prob == 0.2
    serialized = config.to_dict()
    assert serialized["subtask_time_noise_ratio"] == 0.4
    assert serialized["subtask_time_noise_max_seconds"] == 5.0
    assert serialized["subtask_time_dropout_prob"] == 0.2

    ablation = _train_config(
        tmp_path / "ablation",
        subtask_time_noise_ratio=0.0,
        subtask_time_noise_max_seconds=0.0,
        subtask_time_dropout_prob=0.0,
    )
    ablation.validate()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"subtask_time_noise_ratio": -0.1}, "subtask_time_noise_ratio"),
        ({"subtask_time_noise_ratio": math.inf}, "subtask_time_noise_ratio"),
        ({"subtask_time_noise_max_seconds": -0.1}, "subtask_time_noise_max_seconds"),
        ({"subtask_time_noise_max_seconds": math.nan}, "subtask_time_noise_max_seconds"),
        ({"subtask_time_dropout_prob": -0.1}, "subtask_time_dropout_prob"),
        ({"subtask_time_dropout_prob": 1.1}, "subtask_time_dropout_prob"),
        ({"subtask_time_dropout_prob": math.nan}, "subtask_time_dropout_prob"),
    ],
)
def test_train_config_rejects_invalid_time_values(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _train_config(tmp_path, **kwargs).validate()


def test_train_config_rejects_streaming_missing_subtask_prediction_and_other_policy(tmp_path):
    with pytest.raises(ValueError, match="non-streaming"):
        _train_config(tmp_path / "streaming", streaming=True).validate()

    with pytest.raises(ValueError, match="predict_subtask"):
        _train_config(
            tmp_path / "no-subtask",
            policy=_policy_config(predict_subtask=False),
        ).validate()

    act = ACTConfig(device="cpu", push_to_hub=False)
    act.use_subtask_time_conditioning = True
    act.predict_subtask = True
    with pytest.raises(ValueError, match="pi0.*pi05"):
        _train_config(tmp_path / "act", policy=act).validate()


def test_training_condition_helpers_run_in_fixed_order_without_overwriting(monkeypatch):
    import lerobot.scripts.lerobot_train as train_module

    calls = []

    def advantage(batch, **kwargs):
        calls.append("advantage")
        return {**batch, "advantage_condition_kept": torch.tensor([True, False])}

    def memory(batch, **kwargs):
        calls.append("memory")
        assert "advantage_condition_kept" in batch
        return {**batch, "memory_condition_kept": torch.tensor([False, True])}

    def subtask_time(batch, **kwargs):
        calls.append("time")
        assert "memory_condition_kept" in batch
        return {
            **batch,
            "subtask_time_seconds": torch.tensor([0.0, 1.0]),
            "subtask_time_condition_kept": torch.tensor([True, True]),
        }

    monkeypatch.setattr(train_module, "sample_advantage_condition_mask", advantage)
    monkeypatch.setattr(train_module, "sample_memory_condition_mask", memory)
    monkeypatch.setattr(train_module, "sample_subtask_time_condition", subtask_time)
    result = prepare_training_batch_conditions(_raw_batch(), _condition_cfg())

    assert calls == ["advantage", "memory", "time"]
    assert result["advantage_condition_kept"].tolist() == [True, False]
    assert result["memory_condition_kept"].tolist() == [False, True]
    assert result["subtask_time_condition_kept"].tolist() == [True, True]
    assert result["subtask"] == ["current a", "current b"]


def test_time_disabled_does_not_call_helper_or_consume_rng(monkeypatch):
    import lerobot.scripts.lerobot_train as train_module

    def unexpected(*args, **kwargs):
        raise AssertionError("time-disabled training must not call the time helper")

    monkeypatch.setattr(train_module, "sample_subtask_time_condition", unexpected)
    generator_state = torch.random.get_rng_state().clone()
    batch = _raw_batch()
    result = prepare_training_batch_conditions(
        batch,
        _condition_cfg(advantage=False, memory=False, time=False),
    )
    assert result is batch
    assert torch.equal(torch.random.get_rng_state(), generator_state)
    assert "subtask_time_seconds" not in result


@pytest.mark.parametrize("advantage_dropout", [0.0, 1.0])
@pytest.mark.parametrize("memory_dropout", [0.0, 1.0])
@pytest.mark.parametrize("time_dropout", [0.0, 1.0])
def test_advantage_memory_and_time_form_all_eight_independent_combinations(
    advantage_dropout, memory_dropout, time_dropout
):
    cfg = _condition_cfg()
    cfg.advantage_condition_dropout_prob = advantage_dropout
    cfg.memory_dropout_prob = memory_dropout
    cfg.subtask_time_dropout_prob = time_dropout
    result = prepare_training_batch_conditions(_raw_batch(), cfg)

    expected_advantage = advantage_dropout == 0.0
    expected_memory = memory_dropout == 0.0
    expected_time = time_dropout == 0.0
    assert result["advantage_condition_kept"].tolist() == [
        expected_advantage,
        expected_advantage,
    ]
    assert result["memory_condition_kept"].tolist() == [expected_memory, expected_memory]
    assert result["subtask_time_condition_kept"].tolist() == [expected_time, expected_time]
    assert result["subtask"] == ["current a", "current b"]


@pytest.mark.parametrize("time_dropout", [0.0, 1.0])
@pytest.mark.parametrize("subtask_dropout", [0.0, 1.0])
def test_time_and_current_subtask_attention_dropout_are_independent(
    time_dropout, subtask_dropout
):
    cfg = _condition_cfg(advantage=False, memory=False)
    cfg.subtask_time_dropout_prob = time_dropout
    result = prepare_training_batch_conditions(_raw_batch(), cfg)

    attention = torch.ones(2, 6, 6, dtype=torch.bool)
    dropped = apply_subtask_attention_dropout(
        attention,
        subtask_start=2,
        subtask_end=4,
        suffix_len=2,
        dropout_prob=subtask_dropout,
        training=True,
    )
    assert result["subtask_time_condition_kept"].tolist() == [
        time_dropout == 0.0,
        time_dropout == 0.0,
    ]
    assert dropped[:, -2:, 2:4].eq(subtask_dropout == 0.0).all()
    assert result["subtask"] == ["current a", "current b"]


@pytest.mark.parametrize("policy_type", ["pi0", "pi05"])
def test_cpu_single_step_with_time_condition_is_finite(policy_type):
    cfg = _condition_cfg(advantage=False, memory=False)
    metrics = SubtaskTimeTrainingMetrics()
    batch = prepare_training_batch_conditions(
        _raw_batch(),
        cfg,
        subtask_time_metrics=metrics,
    )
    policy = _TinyTimePolicy(policy_type)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    train_metrics = SimpleNamespace(loss=None, grad_norm=None, lr=None, update_s=None)

    train_metrics, output = update_policy(
        train_metrics,
        policy,
        batch,
        optimizer,
        grad_clip_norm=1.0,
        accelerator=_Accelerator(),
    )

    assert math.isfinite(train_metrics.loss)
    assert math.isfinite(train_metrics.grad_norm)
    assert math.isfinite(output["loss"])
    assert policy.scale.item() < 1.0
    assert metrics.to_dict()["subtask_time/condition_kept_fraction"] == 1.0
