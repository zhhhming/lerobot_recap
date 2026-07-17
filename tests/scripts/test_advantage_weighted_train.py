from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.advantage_weights import (
    AdvantageWeights,
    sample_advantage_condition_mask,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.memory_conditioning import sample_memory_condition_mask


LABEL_KEY = "advantage_label_global"
WEIGHT_KEY = "advantage_loss_weight_global"


class _CharacterTokenizer:
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

    def reduce(self, tensor, reduction):
        assert reduction == "sum"
        return tensor


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fm_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.ce_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(subtask_ce_loss_weight=0.25)
        self.last_forward = None

    def forward(self, batch, reduction="mean", *, return_loss_components=False):
        self.last_forward = (reduction, return_loss_components)
        fm = self.fm_scale * torch.tensor([1.0, 3.0, 5.0])
        ce = self.ce_scale * torch.tensor([2.0, 4.0, 6.0])
        output = {"fm_loss": fm.mean().item(), "ce_loss": ce.mean().item()}
        if return_loss_components:
            return fm, ce, output
        total = fm + self.config.subtask_ce_loss_weight * ce
        if reduction == "none":
            return total, output
        return total.mean(), output


def _metrics():
    return SimpleNamespace(loss=None, grad_norm=None, lr=None, update_s=None)


def _advantage_provider():
    return AdvantageWeights(loss_weight_key=WEIGHT_KEY, label_key=LABEL_KEY)


def test_update_policy_weights_only_fm_and_keeps_ce_mean():
    policy = _Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    batch = {
        LABEL_KEY: ["positive", "negative", "positive"],
        WEIGHT_KEY: torch.tensor([2.0, 1.0, 0.5]),
        "advantage_condition_kept": torch.tensor([True, True, True]),
    }

    metrics, output = update_policy(
        _metrics(),
        policy,
        batch,
        optimizer,
        grad_clip_norm=0.0,
        accelerator=_Accelerator(),
        advantage_weights_provider=_advantage_provider(),
    )

    expected_fm_gradient = (2.0 * 1.0 + 1.0 * 3.0 + 0.5 * 5.0) / 3.5
    expected_ce_gradient = 0.25 * (2.0 + 4.0 + 6.0) / 3.0
    assert policy.fm_scale.item() == pytest.approx(1.0 - 0.1 * expected_fm_gradient)
    assert policy.ce_scale.item() == pytest.approx(1.0 - 0.1 * expected_ce_gradient)
    assert policy.last_forward == ("none", True)
    assert metrics.loss == pytest.approx(output["loss"])
    assert output["advantage_weight_sum"] == pytest.approx(3.5)
    assert output["advantage_num_positive"] == 2
    assert output["advantage_num_negative"] == 1


@pytest.mark.parametrize("memory_kept", [False, True], ids=["memory_drop", "memory_keep"])
def test_update_policy_all_ignore_skips_fm_but_keeps_subtask_ce(memory_kept):
    policy = _Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    batch = {
        LABEL_KEY: ["ignore", "ignore", "ignore"],
        WEIGHT_KEY: torch.zeros(3),
        "advantage_condition_kept": torch.tensor([False, False, False]),
        "memory_condition_kept": torch.full((3,), memory_kept, dtype=torch.bool),
    }

    metrics, output = update_policy(
        _metrics(),
        policy,
        batch,
        optimizer,
        grad_clip_norm=0.0,
        accelerator=_Accelerator(),
        advantage_weights_provider=_advantage_provider(),
    )

    assert policy.fm_scale.item() == pytest.approx(1.0)
    assert policy.ce_scale.item() == pytest.approx(0.9)
    assert metrics.loss == pytest.approx(1.0)
    assert output["advantage_weighted_fm_loss"] == 0.0
    assert output["advantage_all_ignore_batch"] is True


class _RABCProvider:
    def compute_batch_weights(self, batch):
        return torch.tensor([1.0, 1.0, 1.0]), {
            "raw_mean_weight": 1.0,
            "num_zero_weight": 0,
            "num_full_weight": 3,
        }


def test_update_policy_keeps_legacy_rabc_path_and_rejects_combination():
    policy = _Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    metrics, output = update_policy(
        _metrics(),
        policy,
        {},
        optimizer,
        grad_clip_norm=0.0,
        accelerator=_Accelerator(),
        rabc_weights_provider=_RABCProvider(),
    )
    assert policy.last_forward == ("none", False)
    assert metrics.loss == pytest.approx(4.0, abs=1e-5)
    assert output["rabc_mean_weight"] == 1.0

    policy = _Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="cannot be active"):
        update_policy(
            _metrics(),
            policy,
            {LABEL_KEY: ["positive"] * 3, WEIGHT_KEY: torch.ones(3)},
            optimizer,
            grad_clip_norm=0.0,
            accelerator=_Accelerator(),
            rabc_weights_provider=_RABCProvider(),
            advantage_weights_provider=_advantage_provider(),
        )


def _train_config(tmp_path, **kwargs):
    policy = kwargs.pop("policy", PI0Config())
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/advantage"),
        policy=policy,
        output_dir=tmp_path / "output",
        **kwargs,
    )


def test_train_config_rejects_rabc_combination_and_invalid_dropout(tmp_path):
    memory_policy = PI0Config(
        predict_subtask=True,
        use_memory_conditioning=True,
    )
    with pytest.raises(ValueError, match="cannot be enabled together"):
        _train_config(
            tmp_path,
            policy=memory_policy,
            use_rabc=True,
            rabc_progress_path="progress.parquet",
            use_advantage_weighting=True,
        ).validate()

    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        _train_config(tmp_path, advantage_condition_dropout_prob=1.1).validate()


def test_train_config_requires_policy_and_training_keys_to_match(tmp_path):
    conditioning_policy = PI0Config(
        use_advantage_conditioning=True,
        advantage_label_key="advantage_label_subtask",
    )
    with pytest.raises(ValueError, match="label keys must match"):
        _train_config(tmp_path, policy=conditioning_policy).validate()

    weighting_policy = PI0Config(
        advantage_loss_weight_key="advantage_loss_weight_subtask",
    )
    with pytest.raises(ValueError, match="loss weight keys must match"):
        _train_config(
            tmp_path,
            policy=weighting_policy,
            use_advantage_weighting=True,
        ).validate()


def test_train_config_accepts_matching_advantage_configuration(tmp_path):
    policy = PI0Config(
        use_advantage_conditioning=True,
        advantage_label_key="advantage_label_subtask",
        advantage_loss_weight_key="advantage_loss_weight_subtask",
        push_to_hub=False,
    )
    config = _train_config(
        tmp_path,
        policy=policy,
        use_advantage_weighting=True,
        advantage_label_key="advantage_label_subtask",
        advantage_loss_weight_key="advantage_loss_weight_subtask",
        advantage_condition_dropout_prob=0.25,
    )
    config.validate()

    assert config.optimizer is not None
    assert config.scheduler is not None


def _memory_advantage_policy_config(config_cls, *, label_key, weight_key):
    config = config_cls(
        predict_subtask=True,
        use_memory_conditioning=True,
        use_advantage_conditioning=True,
        advantage_label_key=label_key,
        advantage_loss_weight_key=weight_key,
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


def _memory_advantage_batch(*, label_key, weight_key):
    return {
        OBS_STATE: torch.zeros(3, 2),
        ACTION: torch.zeros(3, 2),
        "task": ["perform task"] * 3,
        "subtask": ["current positive", "current negative", "current ignore"],
        "subtask_progress": torch.tensor([0.2, 0.4, 0.6]),
        "memory_subtask": ["history positive", "history negative", "history ignore"],
        "memory_subtask_progress": torch.tensor([0.1, 0.3, 0.5]),
        "memory_valid": torch.ones(3, dtype=torch.bool),
        "memory_frame_offset": torch.tensor([1, 6, 12]),
        label_key: ["positive", "negative", "ignore"],
        weight_key: torch.tensor([2.0, 1.0, 17.0]),
    }


@pytest.mark.parametrize(
    ("config_cls", "processor_factory"),
    [
        pytest.param(PI0Config, make_pi0_pre_post_processors, id="pi0"),
        pytest.param(PI05Config, make_pi05_pre_post_processors, id="pi05"),
    ],
)
@pytest.mark.parametrize(
    ("label_key", "weight_key"),
    [
        pytest.param(
            "advantage_label_global",
            "advantage_loss_weight_global",
            id="global",
        ),
        pytest.param(
            "advantage_label_subtask",
            "advantage_loss_weight_subtask",
            id="subtask",
        ),
    ],
)
@pytest.mark.parametrize("memory_dropout", [0.0, 1.0], ids=["memory_keep", "memory_drop"])
@pytest.mark.parametrize(
    "advantage_dropout",
    [0.0, 1.0],
    ids=["advantage_keep", "advantage_drop"],
)
def test_pi_memory_and_advantage_integration_keeps_current_frame_loss_semantics(
    monkeypatch,
    config_cls,
    processor_factory,
    label_key,
    weight_key,
    memory_dropout,
    advantage_dropout,
):
    """Cover PI0/PI0.5 x global/subtask x both independent condition dropouts."""

    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _CharacterTokenizer(),
    )
    config = _memory_advantage_policy_config(
        config_cls,
        label_key=label_key,
        weight_key=weight_key,
    )
    preprocessor, _ = processor_factory(config, dataset_stats={})
    batch = _memory_advantage_batch(label_key=label_key, weight_key=weight_key)
    batch = sample_advantage_condition_mask(
        batch,
        label_key=label_key,
        dropout_prob=advantage_dropout,
    )
    batch = sample_memory_condition_mask(batch, dropout_prob=memory_dropout)
    processed = preprocessor(batch)

    expected_advantage_kept = [advantage_dropout == 0.0] * 2 + [False]
    expected_memory_kept = [memory_dropout == 0.0] * 3
    assert processed["advantage_condition_kept"].tolist() == expected_advantage_kept
    assert processed["memory_condition_kept"].tolist() == expected_memory_kept
    for index, task in enumerate(processed["task"]):
        assert ("Advantage:" in task) is expected_advantage_kept[index]
        assert ("Memory:" in task) is expected_memory_kept[index]

    provider = AdvantageWeights(loss_weight_key=weight_key, label_key=label_key)
    effective_weights, _ = provider.compute_batch_weights(processed)
    expected_weights = (
        torch.tensor([2.0, 1.0, 0.0])
        if advantage_dropout == 0.0
        else torch.tensor([1.0, 1.0, 0.0])
    )
    assert torch.equal(effective_weights, expected_weights)

    policy = _Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    metrics, output = update_policy(
        _metrics(),
        policy,
        processed,
        optimizer,
        grad_clip_norm=0.0,
        accelerator=_Accelerator(),
        advantage_weights_provider=provider,
    )

    expected_fm_gradient = (
        (2.0 * 1.0 + 1.0 * 3.0) / 3.0
        if advantage_dropout == 0.0
        else (1.0 + 3.0) / 2.0
    )
    expected_ce_gradient = 0.25 * (2.0 + 4.0 + 6.0) / 3.0
    assert policy.fm_scale.item() == pytest.approx(1.0 - 0.1 * expected_fm_gradient)
    assert policy.ce_scale.item() == pytest.approx(1.0 - 0.1 * expected_ce_gradient)
    assert output["advantage_weight_sum"] == pytest.approx(expected_weights.sum().item())
    assert output["advantage_num_positive"] == 1
    assert output["advantage_num_negative"] == 1
    assert output["advantage_num_ignore"] == 1
    assert metrics.loss == pytest.approx(output["loss"])
