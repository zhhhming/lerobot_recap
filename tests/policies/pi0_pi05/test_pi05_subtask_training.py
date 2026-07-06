#!/usr/bin/env python

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    apply_subtask_attention_dropout,
    compute_subtask_ce_loss_per_sample,
    make_att_2d_masks,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)


def test_subtask_attention_mask_layout():
    # Layout: prefix P=2, subtask S=3, action A=2. The last subtask token is padding.
    pad_masks = torch.tensor([[True, True, True, True, False, True, True]])
    att_masks = torch.tensor([[0, 0, 1, 1, 1, 1, 0]], dtype=torch.bool)

    mask = make_att_2d_masks(pad_masks, att_masks)[0]

    # Prefix is bidirectional inside prefix and cannot see subtask/action.
    assert mask[0, :2].all()
    assert not mask[0, 2:].any()
    assert mask[1, :2].all()
    assert not mask[1, 2:].any()

    # Subtask is causal and can see prefix.
    assert mask[2, :3].all()
    assert not mask[2, 3:].any()
    assert mask[3, :4].all()
    assert not mask[3, 4:].any()

    # Subtask padding row/column is fully masked.
    assert not mask[4, :].any()
    assert not mask[:, 4].any()

    # Action block sees all valid prefix/subtask/action tokens.
    expected_valid = torch.tensor([True, True, True, True, False, True, True])
    assert torch.equal(mask[5], expected_valid)
    assert torch.equal(mask[6], expected_valid)


def test_compute_subtask_ce_loss_per_sample_masks_empty_rows():
    lm_head = torch.nn.Linear(5, 5, bias=False)
    lm_head.weight.data.copy_(torch.eye(5))

    hidden = torch.tensor(
        [
            [
                [3.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            torch.zeros(4, 5),
        ]
    )
    tokens = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 0, 0, 0],
        ]
    )
    masks = torch.tensor(
        [
            [True, True, True, False],
            [False, False, False, False],
        ]
    )

    ce = compute_subtask_ce_loss_per_sample(hidden, tokens, masks, lm_head)

    expected = torch.nn.functional.cross_entropy(
        hidden[0, :2],
        torch.tensor([1, 2]),
        reduction="mean",
    )
    assert torch.allclose(ce[0], expected)
    assert ce[1].item() == 0.0


def test_apply_subtask_attention_dropout_only_masks_suffix_rows():
    att_2d_masks = torch.ones(2, 7, 7, dtype=torch.bool)

    dropped = apply_subtask_attention_dropout(
        att_2d_masks,
        subtask_start=2,
        subtask_end=5,
        suffix_len=2,
        dropout_prob=1.0,
        training=True,
    )

    assert not dropped[:, -2:, 2:5].any()
    assert dropped[:, :5, 2:5].all()
    assert dropped[:, -2:, :2].all()
    assert dropped[:, -2:, 5:].all()

    unchanged = apply_subtask_attention_dropout(
        att_2d_masks,
        subtask_start=2,
        subtask_end=5,
        suffix_len=2,
        dropout_prob=1.0,
        training=False,
    )
    assert torch.equal(unchanged, att_2d_masks)


class _FakeModel:
    def __init__(self, losses, ce_loss_per_sample):
        self.losses = losses
        self.ce_loss_per_sample = ce_loss_per_sample
        self.last_kwargs = None

    def forward(self, images, img_masks, tokens, masks, actions, **kwargs):
        self.last_kwargs = kwargs
        return self.losses, self.ce_loss_per_sample


class _FakePI05Policy:
    def __init__(self, model, *, predict_subtask=True, weight=0.25, action_dim=2):
        self.model = model
        self.config = SimpleNamespace(
            predict_subtask=predict_subtask,
            subtask_ce_loss_weight=weight,
            output_features={ACTION: SimpleNamespace(shape=(action_dim,))},
        )

    def _preprocess_images(self, batch):
        return [], []

    def prepare_action(self, batch):
        return batch[ACTION]


def _make_policy_batch():
    return {
        OBS_LANGUAGE_TOKENS: torch.ones(2, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 4, dtype=torch.bool),
        OBS_LANGUAGE_SUBTASK_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_SUBTASK_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        ACTION: torch.zeros(2, 2, 3),
    }


def test_pi05_policy_forward_combines_fm_and_ce_loss_mean():
    losses = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    ce_loss_per_sample = torch.tensor([0.5, 1.5])
    fake_model = _FakeModel(losses, ce_loss_per_sample)
    policy = _FakePI05Policy(fake_model, predict_subtask=True, weight=0.25, action_dim=2)

    loss, loss_dict = PI05Policy.forward(policy, _make_policy_batch(), reduction="mean")

    fm_losses = losses[:, :, :2]
    expected_fm = fm_losses.mean()
    expected_ce = ce_loss_per_sample.mean()
    assert torch.allclose(loss, expected_fm + 0.25 * expected_ce)
    assert loss_dict["fm_loss"] == pytest.approx(expected_fm.item())
    assert loss_dict["ce_loss"] == pytest.approx(expected_ce.item())
    assert loss_dict["loss"] == pytest.approx(loss.item())
    assert fake_model.last_kwargs["subtask_tokens"].shape == (2, 3)
    assert fake_model.last_kwargs["subtask_masks"].shape == (2, 3)


def test_pi05_policy_forward_combines_fm_and_ce_loss_none():
    losses = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    ce_loss_per_sample = torch.tensor([0.5, 1.5])
    policy = _FakePI05Policy(_FakeModel(losses, ce_loss_per_sample), predict_subtask=True, weight=0.5)

    loss, loss_dict = PI05Policy.forward(policy, _make_policy_batch(), reduction="none")

    expected = losses[:, :, :2].mean(dim=(1, 2)) + 0.5 * ce_loss_per_sample
    assert torch.allclose(loss, expected)
    assert loss_dict["loss"] == pytest.approx(expected.mean().item())


def test_pi05_policy_forward_without_subtask_keeps_fm_only_behavior():
    losses = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    fake_model = _FakeModel(losses, None)
    policy = _FakePI05Policy(fake_model, predict_subtask=False, weight=10.0, action_dim=2)
    batch = _make_policy_batch()
    batch.pop(OBS_LANGUAGE_SUBTASK_TOKENS)
    batch.pop(OBS_LANGUAGE_SUBTASK_ATTENTION_MASK)

    loss, loss_dict = PI05Policy.forward(policy, batch, reduction="mean")

    expected_fm = losses[:, :, :2].mean()
    assert torch.allclose(loss, expected_fm)
    assert loss_dict["ce_loss"] == 0.0
    assert fake_model.last_kwargs["subtask_tokens"] is None
    assert fake_model.last_kwargs["subtask_masks"] is None
