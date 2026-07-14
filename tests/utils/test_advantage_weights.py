import pytest
import torch

from lerobot.utils.advantage_weights import (
    AdvantageWeights,
    distributed_weighted_mean,
    sample_advantage_condition_mask,
)


LABEL_KEY = "advantage_label_global"
WEIGHT_KEY = "advantage_loss_weight_global"


def test_condition_mask_dropout_extremes_and_ignore_priority():
    batch = {LABEL_KEY: ["positive", "negative", "ignore"]}

    kept = sample_advantage_condition_mask(batch, label_key=LABEL_KEY, dropout_prob=0.0)
    dropped = sample_advantage_condition_mask(batch, label_key=LABEL_KEY, dropout_prob=1.0)

    assert torch.equal(kept["advantage_condition_kept"], torch.tensor([True, True, False]))
    assert not dropped["advantage_condition_kept"].any()
    assert "advantage_condition_kept" not in batch


def test_condition_mask_is_reproducible_with_generator():
    batch = {LABEL_KEY: ["positive", "negative"] * 20}
    first = sample_advantage_condition_mask(
        batch,
        label_key=LABEL_KEY,
        dropout_prob=0.4,
        generator=torch.Generator().manual_seed(7),
    )
    second = sample_advantage_condition_mask(
        batch,
        label_key=LABEL_KEY,
        dropout_prob=0.4,
        generator=torch.Generator().manual_seed(7),
    )
    third = sample_advantage_condition_mask(
        batch,
        label_key=LABEL_KEY,
        dropout_prob=0.4,
        generator=torch.Generator().manual_seed(8),
    )

    assert torch.equal(
        first["advantage_condition_kept"], second["advantage_condition_kept"]
    )
    assert not torch.equal(
        first["advantage_condition_kept"], third["advantage_condition_kept"]
    )


def test_effective_weights_apply_dropout_and_ignore_semantics():
    provider = AdvantageWeights(loss_weight_key=WEIGHT_KEY, label_key=LABEL_KEY)
    weights, stats = provider.compute_batch_weights(
        {
            LABEL_KEY: ["positive", "negative", "positive", "ignore"],
            WEIGHT_KEY: torch.tensor([[2.0], [1.0], [0.4], [17.0]]),
            "advantage_condition_kept": torch.tensor([[True], [False], [False], [False]]),
        }
    )

    assert torch.allclose(weights, torch.tensor([2.0, 1.0, 1.0, 0.0]))
    assert stats == {
        "mean_weight": 1.0,
        "weight_sum": 4.0,
        "num_positive": 2,
        "num_negative": 1,
        "num_ignore": 1,
        "num_condition_dropped": 2,
        "all_ignore": False,
    }


def test_effective_weights_can_retain_weight_when_condition_dropped():
    provider = AdvantageWeights(
        loss_weight_key=WEIGHT_KEY,
        label_key=LABEL_KEY,
        disable_weight_when_condition_dropped=False,
    )
    weights, _ = provider.compute_batch_weights(
        {
            LABEL_KEY: ["positive", "negative"],
            WEIGHT_KEY: [0.4, 1.0],
            "advantage_condition_kept": [False, False],
        }
    )
    assert torch.allclose(weights, torch.tensor([0.4, 1.0]))


def test_effective_weights_without_condition_mask_keep_offline_weighting():
    provider = AdvantageWeights(loss_weight_key=WEIGHT_KEY, label_key=LABEL_KEY)
    weights, stats = provider.compute_batch_weights(
        {
            LABEL_KEY: ["positive", "negative", "positive"],
            WEIGHT_KEY: [2.0, 1.0, 0.4],
        }
    )
    assert torch.allclose(weights, torch.tensor([2.0, 1.0, 0.4]))
    assert stats["num_condition_dropped"] == 0


@pytest.mark.parametrize(
    ("batch", "match"),
    [
        ({WEIGHT_KEY: [1.0]}, "missing advantage label"),
        ({LABEL_KEY: ["positive"]}, "missing advantage weight"),
        (
            {LABEL_KEY: ["negative"], WEIGHT_KEY: [0.5]},
            "Negative samples.*must have weight 1.0",
        ),
        ({LABEL_KEY: ["bad"], WEIGHT_KEY: [1.0]}, "Unsupported labels"),
        ({LABEL_KEY: ["positive"], WEIGHT_KEY: [float("nan")]}, "finite"),
        ({LABEL_KEY: ["positive"], WEIGHT_KEY: [-1.0]}, "negative"),
    ],
)
def test_effective_weights_reject_invalid_batches(batch, match):
    provider = AdvantageWeights(loss_weight_key=WEIGHT_KEY, label_key=LABEL_KEY)
    with pytest.raises(ValueError, match=match):
        provider.compute_batch_weights(batch)


def test_weighted_mean_uses_actual_positive_weights_and_handles_all_ignore():
    losses = torch.tensor([1.0, 3.0, 10.0], requires_grad=True)
    weighted, weight_sum = distributed_weighted_mean(
        losses, torch.tensor([2.0, 0.5, 1.0])
    )
    assert weighted.item() == pytest.approx((2.0 + 1.5 + 10.0) / 3.5)
    assert weight_sum.item() == pytest.approx(3.5)

    ignored, ignored_sum = distributed_weighted_mean(losses, torch.zeros(3))
    assert ignored.item() == 0.0
    assert ignored_sum.item() == 0.0
    ignored.backward()
    assert torch.equal(losses.grad, torch.zeros(3))


class _FakeDistributedAccelerator:
    num_processes = 2

    def __init__(self, global_weight_sum):
        self.global_weight_sum = torch.tensor(global_weight_sum)

    def reduce(self, tensor, reduction):
        assert reduction == "sum"
        return self.global_weight_sum.to(tensor)


def test_distributed_weighted_mean_has_global_batch_gradient():
    rank0 = torch.tensor([1.0, 2.0], requires_grad=True)
    rank1 = torch.tensor([4.0], requires_grad=True)
    accelerator = _FakeDistributedAccelerator(global_weight_sum=4.0)

    loss0, _ = distributed_weighted_mean(
        rank0, torch.tensor([2.0, 1.0]), accelerator=accelerator
    )
    loss1, _ = distributed_weighted_mean(
        rank1, torch.tensor([1.0]), accelerator=accelerator
    )
    loss0.backward()
    loss1.backward()

    # DDP averages these rank-local gradients. Their average equals the
    # derivative of the global weighted mean: [2/4, 1/4] and [1/4].
    assert torch.allclose(rank0.grad / 2, torch.tensor([0.5, 0.25]))
    assert torch.allclose(rank1.grad / 2, torch.tensor([0.25]))


def test_condition_mask_rejects_missing_or_invalid_labels():
    with pytest.raises(ValueError, match="requires label field"):
        sample_advantage_condition_mask({}, label_key=LABEL_KEY, dropout_prob=0.1)
    with pytest.raises(ValueError, match="Unsupported labels"):
        sample_advantage_condition_mask(
            {LABEL_KEY: ["positive", "bad"]}, label_key=LABEL_KEY, dropout_prob=0.1
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sample_advantage_condition_mask(
            {LABEL_KEY: ["positive"]}, label_key=LABEL_KEY, dropout_prob=1.1
        )


def test_provider_configuration_validation():
    with pytest.raises(ValueError, match="ignore_label"):
        AdvantageWeights(ignore_label="positive")
    with pytest.raises(ValueError, match="loss_weight_key"):
        AdvantageWeights(loss_weight_key="")
