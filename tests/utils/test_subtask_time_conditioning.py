#!/usr/bin/env python

"""Executable contracts for train-only subtask-time noise, dropout, and metrics."""

import math

import pytest
import torch

from lerobot.utils.subtask_time_conditioning import (
    SubtaskTimeTrainingMetrics,
    compute_subtask_time_training_metrics,
    sample_subtask_time_condition,
)


def _batch(elapsed, valid=None):
    elapsed_tensor = torch.tensor(elapsed)
    batch_size = elapsed_tensor.shape[0]
    if valid is None:
        valid = [True] * batch_size
    return {
        "subtask_elapsed_seconds": elapsed_tensor,
        "subtask_time_valid": torch.tensor(valid, dtype=torch.bool),
    }


def test_noise_is_bounded_reproducible_nonnegative_and_non_mutating():
    batch = _batch([0.0, 1.0, 12.5, 40.5, 95.8])
    original = {key: value.clone() for key, value in batch.items()}
    first = sample_subtask_time_condition(
        batch,
        noise_ratio=0.4,
        noise_max_seconds=5.0,
        dropout_prob=0.2,
        generator=torch.Generator().manual_seed(1234),
    )
    second = sample_subtask_time_condition(
        batch,
        noise_ratio=0.4,
        noise_max_seconds=5.0,
        dropout_prob=0.2,
        generator=torch.Generator().manual_seed(1234),
    )

    assert torch.equal(first["subtask_time_seconds"], second["subtask_time_seconds"])
    assert torch.equal(
        first["subtask_time_condition_kept"], second["subtask_time_condition_kept"]
    )
    noisy = first["subtask_time_seconds"]
    assert noisy.dtype == torch.float32
    assert noisy.shape == (5,)
    assert noisy[0].item() == 0.0
    assert abs(noisy[1].item() - 1.0) <= 0.4
    assert abs(noisy[2].item() - 12.5) <= 5.0
    assert abs(noisy[3].item() - 40.5) <= 5.0
    assert abs(noisy[4].item() - 95.8) <= 5.0
    assert torch.isfinite(noisy).all()
    assert torch.all(noisy >= 0)
    assert all(torch.equal(batch[key], value) for key, value in original.items())
    assert "subtask_time_seconds" not in batch
    assert "subtask_time_condition_kept" not in batch


def test_noise_and_dropout_use_two_full_batch_draws_in_fixed_order():
    batch = _batch([[0.0], [1.0], [4.0]], valid=[False, True, True])
    actual_generator = torch.Generator().manual_seed(91)
    expected_generator = torch.Generator().manual_seed(91)

    result = sample_subtask_time_condition(
        batch,
        noise_ratio=0.0,
        noise_max_seconds=0.0,
        dropout_prob=0.0,
        generator=actual_generator,
    )
    expected_noise_draw = torch.rand(3, generator=expected_generator)
    expected_dropout_draw = torch.rand(3, generator=expected_generator)

    assert result["subtask_time_condition_kept"].tolist() == [False, True, True]
    assert expected_noise_draw.shape == expected_dropout_draw.shape == (3,)
    assert torch.equal(
        torch.rand(4, generator=actual_generator),
        torch.rand(4, generator=expected_generator),
    )


@pytest.mark.parametrize(
    ("dropout_prob", "expected"),
    [
        (0.0, [False, True, True]),
        (1.0, [False, False, False]),
    ],
)
def test_dropout_boundaries_and_invalid_samples(dropout_prob, expected):
    result = sample_subtask_time_condition(
        _batch([[0.0], [1.0], [12.5]], valid=[False, True, True]),
        noise_ratio=0.0,
        noise_max_seconds=5.0,
        dropout_prob=dropout_prob,
    )

    assert result["subtask_time_seconds"].shape == (3,)
    assert result["subtask_time_condition_kept"].dtype == torch.bool
    assert result["subtask_time_condition_kept"].tolist() == expected


def test_large_relative_noise_clamps_to_zero_after_sampling():
    seed = next(
        seed
        for seed in range(1000)
        if torch.rand(1, generator=torch.Generator().manual_seed(seed)).item() < 0.25
    )
    result = sample_subtask_time_condition(
        _batch([1.0]),
        noise_ratio=2.0,
        noise_max_seconds=10.0,
        dropout_prob=0.0,
        generator=torch.Generator().manual_seed(seed),
    )
    assert result["subtask_time_seconds"].item() == 0.0


def test_sampling_does_not_round_processor_input():
    result = sample_subtask_time_condition(
        _batch([1.234]),
        noise_ratio=0.0,
        noise_max_seconds=5.0,
        dropout_prob=0.0,
        generator=torch.Generator().manual_seed(7),
    )
    assert result["subtask_time_seconds"].item() == pytest.approx(1.234)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"noise_ratio": -0.1}, "noise_ratio"),
        ({"noise_ratio": math.inf}, "noise_ratio"),
        ({"noise_max_seconds": -0.1}, "noise_max_seconds"),
        ({"noise_max_seconds": math.nan}, "noise_max_seconds"),
        ({"dropout_prob": -0.1}, "dropout_prob"),
        ({"dropout_prob": 1.1}, "dropout_prob"),
        ({"dropout_prob": math.nan}, "dropout_prob"),
    ],
)
def test_sampling_rejects_invalid_parameters(kwargs, message):
    parameters = {
        "noise_ratio": 0.4,
        "noise_max_seconds": 5.0,
        "dropout_prob": 0.2,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        sample_subtask_time_condition(_batch([1.0]), **parameters)


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        (
            {
                "subtask_time_valid": torch.tensor([True]),
            },
            "subtask_elapsed_seconds",
        ),
        (
            {
                "subtask_elapsed_seconds": torch.tensor([1.0]),
            },
            "subtask_time_valid",
        ),
        (_batch([float("nan")]), "finite"),
        (_batch([float("inf")]), "finite"),
        (_batch([-0.1]), "non-negative"),
        (
            {
                "subtask_elapsed_seconds": torch.tensor([[1.0, 2.0]]),
                "subtask_time_valid": torch.tensor([True]),
            },
            "shape",
        ),
        (
            {
                "subtask_elapsed_seconds": torch.tensor([1.0, 2.0]),
                "subtask_time_valid": torch.tensor([True]),
            },
            "batch size",
        ),
        (
            {
                "subtask_elapsed_seconds": torch.tensor([1.0]),
                "subtask_time_valid": torch.tensor([1]),
            },
            "boolean",
        ),
        (
            {
                "subtask_elapsed_seconds": torch.tensor([True]),
                "subtask_time_valid": torch.tensor([True]),
            },
            "numeric",
        ),
    ],
)
def test_sampling_rejects_invalid_batch_fields(batch, message):
    with pytest.raises(ValueError, match=message):
        sample_subtask_time_condition(
            batch,
            noise_ratio=0.4,
            noise_max_seconds=5.0,
            dropout_prob=0.2,
        )


def test_metrics_match_manual_values_and_accumulate_true_extrema():
    first = {
        "subtask_elapsed_seconds": torch.tensor([0.0, 2.0, 4.0, 8.0]),
        "subtask_time_valid": torch.tensor([False, True, True, True]),
        "subtask_time_seconds": torch.tensor([0.0, 1.0, 5.0, 0.0]),
        "subtask_time_condition_kept": torch.tensor([False, True, False, True]),
    }
    second = {
        "subtask_elapsed_seconds": torch.tensor([[10.0]]),
        "subtask_time_valid": torch.tensor([[True]]),
        "subtask_time_seconds": torch.tensor([[12.0]]),
        "subtask_time_condition_kept": torch.tensor([[True]]),
    }

    one_batch = compute_subtask_time_training_metrics(first)
    assert one_batch == pytest.approx(
        {
            "subtask_time/valid_fraction": 0.75,
            "subtask_time/condition_kept_fraction": 0.5,
            "subtask_time/dropout_fraction_among_valid": 1 / 3,
            "subtask_time/true_seconds_mean": 14 / 3,
            "subtask_time/true_seconds_max_seen": 8.0,
            "subtask_time/noisy_seconds_mean": 2.0,
            "subtask_time/noise_abs_mean": 10 / 3,
            "subtask_time/noise_abs_max_seen": 8.0,
            "subtask_time/clamped_to_zero_fraction": 1 / 3,
        }
    )

    accumulator = SubtaskTimeTrainingMetrics()
    accumulator.update(first)
    accumulator.update(second)
    assert accumulator.to_dict() == pytest.approx(
        {
            "subtask_time/valid_fraction": 0.8,
            "subtask_time/condition_kept_fraction": 0.6,
            "subtask_time/dropout_fraction_among_valid": 0.25,
            "subtask_time/true_seconds_mean": 6.0,
            "subtask_time/true_seconds_max_seen": 10.0,
            "subtask_time/noisy_seconds_mean": 4.5,
            "subtask_time/noise_abs_mean": 3.0,
            "subtask_time/noise_abs_max_seen": 8.0,
            "subtask_time/clamped_to_zero_fraction": 0.25,
        }
    )
    accumulator.reset()
    assert accumulator.to_dict() == {}


def test_metrics_reject_kept_invalid_and_nonfinite_noisy_values():
    kept_invalid = {
        "subtask_elapsed_seconds": torch.tensor([1.0]),
        "subtask_time_valid": torch.tensor([False]),
        "subtask_time_seconds": torch.tensor([1.0]),
        "subtask_time_condition_kept": torch.tensor([True]),
    }
    with pytest.raises(ValueError, match="invalid"):
        SubtaskTimeTrainingMetrics().update(kept_invalid)

    nonfinite = dict(kept_invalid)
    nonfinite["subtask_time_valid"] = torch.tensor([True])
    nonfinite["subtask_time_condition_kept"] = torch.tensor([False])
    nonfinite["subtask_time_seconds"] = torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="finite"):
        SubtaskTimeTrainingMetrics().update(nonfinite)
