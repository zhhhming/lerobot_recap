from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from lerobot.value_function.configuration import DEFAULT_VALUE_IMAGE_KEYS, ValueFunctionConfig
from lerobot.value_function.modeling_pi0_value import (
    PaliGemmaValueBackbone,
    PI0ValueFunctionModel,
    decode_value_expectation,
    gather_subtask_head,
    select_paired_subtask_head,
    two_hot_targets,
)
from lerobot.value_function.schema import (
    VALUE_GLOBAL_ELAPSED_NORM_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT,
    VALUE_SUBTASK_ELAPSED_NORM_GT,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_ID_PRED_SMOOTH,
    VALUE_SUBTASK_REMAINING_NORM_GT,
    VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD,
    VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD,
)


class TinyBackbone(nn.Module):
    output_dim = 12

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, self.output_dim)

    def forward(self, batch):
        first = batch[DEFAULT_VALUE_IMAGE_KEYS[0]]
        pooled = first.float().reshape(first.shape[0], -1).mean(dim=1, keepdim=True)
        return self.projection(pooled)


def make_batch(batch_size: int = 3, state_dim: int = 16) -> dict[str, torch.Tensor]:
    batch = {
        key: torch.rand(batch_size, 3, 8, 8)
        for key in DEFAULT_VALUE_IMAGE_KEYS
    }
    batch.update(
        {
            "observation.state": torch.randn(batch_size, state_dim),
            VALUE_GLOBAL_REMAINING_NORM_GT: torch.linspace(0.0, 1.0, batch_size),
            VALUE_GLOBAL_ELAPSED_NORM_GT: torch.linspace(1.0, 0.0, batch_size),
            VALUE_SUBTASK_ID_GT: torch.arange(batch_size) % 3,
            VALUE_SUBTASK_ID_PRED_SMOOTH: (torch.arange(batch_size) + 1) % 3,
            VALUE_SUBTASK_REMAINING_NORM_GT: torch.linspace(0.2, 0.8, batch_size),
            VALUE_SUBTASK_ELAPSED_NORM_GT: torch.linspace(0.8, 0.2, batch_size),
        }
    )
    return batch


@pytest.mark.parametrize("num_bins", [64, 128, 256, 512])
def test_global_output_shape_supports_configurable_bins(num_bins: int):
    config = ValueFunctionConfig(mode="global", num_bins=num_bins, dropout=0.0)
    model = PI0ValueFunctionModel(config, backbone=TinyBackbone())

    outputs = model(make_batch())

    assert outputs["global_remaining_logits"].shape == (3, num_bins)
    assert outputs["global_remaining_value"].shape == (3,)
    assert set(outputs) == {"features", "global_remaining_logits", "global_remaining_value"}


@pytest.mark.parametrize("num_subtasks", [3, 7])
def test_subtask_output_shape_is_dynamic(num_subtasks: int):
    config = ValueFunctionConfig(
        mode="subtask",
        num_subtasks=num_subtasks,
        subtask_num_bins=64,
        use_elapsed_aux=True,
        dropout=0.0,
    )
    model = PI0ValueFunctionModel(config, backbone=TinyBackbone())

    outputs = model(make_batch())

    assert outputs["subtask_logits"].shape == (3, num_subtasks)
    assert outputs["subtask_remaining_logits"].shape == (3, num_subtasks, 64)
    assert outputs["subtask_elapsed_logits"].shape == (3, num_subtasks, 64)


def test_both_mode_has_global_subtask_and_elapsed_heads():
    config = ValueFunctionConfig(
        mode="both",
        num_subtasks=3,
        global_num_bins=32,
        subtask_num_bins=16,
        use_elapsed_aux=True,
        elapsed_loss_weight=0.25,
        dropout=0.0,
    )
    model = PI0ValueFunctionModel(config, backbone=TinyBackbone())

    outputs = model(make_batch())
    losses = model.compute_loss(outputs, make_batch())

    assert outputs["global_remaining_logits"].shape == (3, 32)
    assert outputs["global_elapsed_logits"].shape == (3, 32)
    assert outputs["subtask_remaining_logits"].shape == (3, 3, 16)
    assert outputs["subtask_elapsed_logits"].shape == (3, 3, 16)
    assert set(losses) == {
        "global_remaining_loss",
        "global_elapsed_loss",
        "subtask_ce_loss",
        "subtask_remaining_loss",
        "subtask_elapsed_loss",
        "loss",
    }
    expected = (
        losses["global_remaining_loss"]
        + losses["subtask_remaining_loss"]
        + 0.2 * losses["subtask_ce_loss"]
        + 0.25 * (losses["global_elapsed_loss"] + losses["subtask_elapsed_loss"])
    )
    torch.testing.assert_close(losses["loss"], expected)


def test_two_hot_targets_interpolate_and_decode_inside_support():
    values = torch.tensor([0.0, 0.125, 0.5, 1.0])
    targets = two_hot_targets(values, num_bins=5)

    torch.testing.assert_close(targets.sum(dim=-1), torch.ones(4))
    torch.testing.assert_close(targets[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(targets[1], torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(targets[-1], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))

    decoded = decode_value_expectation(torch.randn(10, 5))
    assert torch.all(decoded >= 0.0)
    assert torch.all(decoded <= 1.0)


def test_gather_loss_has_no_gradient_for_unselected_subtask_heads():
    logits = torch.randn(2, 3, 8, requires_grad=True)
    ids = torch.tensor([0, 2])
    selected = gather_subtask_head(logits, ids)
    selected.sum().backward()

    assert torch.count_nonzero(logits.grad[0, 0]) == 8
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0
    assert torch.count_nonzero(logits.grad[1, :2]) == 0
    assert torch.count_nonzero(logits.grad[1, 2]) == 8


def test_paired_head_selection_owns_id_source_and_output_name():
    logits = torch.arange(2 * 3 * 4).reshape(2, 3, 4).float()
    batch = {
        VALUE_SUBTASK_ID_GT: torch.tensor([0, 1]),
        VALUE_SUBTASK_ID_PRED_SMOOTH: torch.tensor([2, 0]),
    }

    gt_selected, gt_key = select_paired_subtask_head(logits, batch, "gt_conditioned")
    smooth_selected, smooth_key = select_paired_subtask_head(logits, batch, "pred_smooth")

    torch.testing.assert_close(gt_selected, torch.stack([logits[0, 0], logits[1, 1]]))
    torch.testing.assert_close(smooth_selected, torch.stack([logits[0, 2], logits[1, 0]]))
    assert gt_key == VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD
    assert smooth_key == VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD
    with pytest.raises(KeyError, match=VALUE_SUBTASK_ID_PRED_SMOOTH):
        select_paired_subtask_head(logits, {VALUE_SUBTASK_ID_GT: torch.tensor([0, 1])}, "pred_smooth")
    with pytest.raises(ValueError, match="Unsupported subtask inference path"):
        select_paired_subtask_head(logits, batch, "mixed")


def test_use_state_false_does_not_require_state():
    config = ValueFunctionConfig(mode="global", use_state=False, dropout=0.0)
    model = PI0ValueFunctionModel(config, backbone=TinyBackbone())
    batch = make_batch()
    del batch["observation.state"]

    assert model(batch)["global_remaining_logits"].shape == (3, 256)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_value_attention_mask_matches_language_model_dtype(dtype):
    pad_masks = torch.tensor([[True, True, False]])

    mask = PaliGemmaValueBackbone._attention_mask_4d(pad_masks, dtype=dtype)

    assert mask.shape == (1, 1, 3, 3)
    assert mask.dtype == dtype
    assert mask.device == pad_masks.device
    assert torch.isfinite(mask).all()
    assert mask.max() == 0
    assert mask.min() < -1e30


def test_loss_rejects_fractional_and_out_of_range_subtask_ids():
    config = ValueFunctionConfig(mode="subtask", num_subtasks=3, dropout=0.0)
    model = PI0ValueFunctionModel(config, backbone=TinyBackbone())
    batch = make_batch()
    outputs = model(batch)

    batch[VALUE_SUBTASK_ID_GT] = torch.tensor([0.0, 1.5, 2.0])
    with pytest.raises(ValueError, match="must be integers"):
        model.compute_loss(outputs, batch)
    batch[VALUE_SUBTASK_ID_GT] = torch.tensor([0, 1, 3])
    with pytest.raises(ValueError, match="outside the configured head range"):
        model.compute_loss(outputs, batch)


def test_state_stats_and_config_round_trip():
    torch.manual_seed(7)
    config = ValueFunctionConfig(mode="subtask", num_subtasks=3, dropout=0.0)
    model = PI0ValueFunctionModel(config, backbone=TinyBackbone())
    mean = torch.arange(16).float()
    std = torch.arange(1, 17).float()
    model.set_state_normalization_stats(mean, std)
    model.eval()
    batch = make_batch()
    expected = model(batch)["subtask_remaining_logits"]
    payload = model.checkpoint_payload()

    restored_config = ValueFunctionConfig.from_dict(copy.deepcopy(payload["model_config"]))
    restored = PI0ValueFunctionModel(restored_config, backbone=TinyBackbone())
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    restored.eval()

    assert restored.config == config
    torch.testing.assert_close(restored.state_mean, mean)
    torch.testing.assert_close(restored.state_std, std)
    torch.testing.assert_close(restored(batch)["subtask_remaining_logits"], expected)
    assert payload["state_normalization"]["key"] == "observation.state"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"num_bins": 1}, "num_bins"),
        ({"mode": "subtask", "num_subtasks": None}, "num_subtasks"),
        ({"backbone_type": "pi0", "num_vlm_layers": 0}, "num_vlm_layers"),
        ({"use_task_text": True}, "do not accept task text"),
        ({"use_state": False, "state_dim": 0}, "state_dim"),
    ],
)
def test_config_rejects_invalid_contracts(updates, message):
    with pytest.raises(ValueError, match=message):
        ValueFunctionConfig(**updates)
