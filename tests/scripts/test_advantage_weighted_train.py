from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.advantage_weights import AdvantageWeights


LABEL_KEY = "advantage_label_global"
WEIGHT_KEY = "advantage_loss_weight_global"


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


def test_update_policy_all_ignore_skips_fm_but_keeps_subtask_ce():
    policy = _Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    batch = {
        LABEL_KEY: ["ignore", "ignore", "ignore"],
        WEIGHT_KEY: torch.zeros(3),
        "advantage_condition_kept": torch.tensor([False, False, False]),
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
    with pytest.raises(ValueError, match="cannot be enabled together"):
        _train_config(
            tmp_path,
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
