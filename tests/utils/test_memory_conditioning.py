#!/usr/bin/env python

"""Executable contracts for train-only memory conditioning helpers."""

import torch

from lerobot.utils.memory_conditioning import (
    MemoryTrainingMetrics,
    compute_memory_training_metrics,
    sample_memory_condition_mask,
)


def test_natural_invalid_and_dropout_share_the_same_false_keep_semantics():
    batch = {
        "memory_valid": torch.tensor([False, True, True]),
        "memory_subtask": ["history-a", "history-b", "history-c"],
    }
    original = {**batch, "memory_valid": batch["memory_valid"].clone()}

    kept = sample_memory_condition_mask(batch, dropout_prob=0.0)
    dropped = sample_memory_condition_mask(batch, dropout_prob=1.0)

    assert kept["memory_condition_kept"].tolist() == [False, True, True]
    assert dropped["memory_condition_kept"].tolist() == [False, False, False]
    assert torch.equal(batch["memory_valid"], original["memory_valid"])
    assert "memory_condition_kept" not in batch


def test_memory_keep_sampling_is_reproducible_and_uses_torch_rng():
    batch = {
        "memory_valid": torch.ones(32, dtype=torch.bool),
        "memory_subtask": ["history"] * 32,
    }
    first_generator = torch.Generator().manual_seed(1234)
    second_generator = torch.Generator().manual_seed(1234)

    first = sample_memory_condition_mask(batch, dropout_prob=0.2, generator=first_generator)
    second = sample_memory_condition_mask(batch, dropout_prob=0.2, generator=second_generator)

    assert torch.equal(first["memory_condition_kept"], second["memory_condition_kept"])
    assert first["memory_condition_kept"].dtype == torch.bool
    assert first["memory_condition_kept"].shape == (32,)


def test_empty_memory_is_ineligible_even_when_valid_flag_is_true():
    result = sample_memory_condition_mask(
        {
            "memory_valid": torch.tensor([True, True, True]),
            "memory_subtask": ["history", "  ", ""],
        },
        dropout_prob=0.0,
    )

    assert result["memory_condition_kept"].tolist() == [True, False, False]


def test_memory_metrics_match_manual_batch_and_accumulate_true_extrema():
    first = {
        "memory_valid": torch.tensor([True, True, False, True]),
        "memory_subtask": ["a", "b", "", "c"],
        "memory_condition_kept": torch.tensor([True, False, False, True]),
        "memory_frame_offset": torch.tensor([1, 4, 9, 12]),
    }
    second = {
        "memory_valid": torch.tensor([True]),
        "memory_subtask": ["d"],
        "memory_condition_kept": torch.tensor([False]),
        "memory_frame_offset": torch.tensor([7]),
    }

    one_batch = compute_memory_training_metrics(first)
    assert one_batch == {
        "memory/history_valid_fraction": 0.75,
        "memory/condition_kept_fraction": 0.5,
        "memory/dropout_fraction_among_valid": 1 / 3,
        "memory/lookback_frames_mean": 6.5,
        "memory/lookback_frames_min_seen": 1.0,
        "memory/lookback_frames_max_seen": 12.0,
    }

    accumulator = MemoryTrainingMetrics()
    accumulator.update(first)
    accumulator.update(second)
    combined = accumulator.to_dict()
    assert combined["memory/history_valid_fraction"] == 0.8
    assert combined["memory/condition_kept_fraction"] == 0.4
    assert combined["memory/dropout_fraction_among_valid"] == 0.5
    assert combined["memory/lookback_frames_mean"] == 6.6
    assert combined["memory/lookback_frames_min_seen"] == 1.0
    assert combined["memory/lookback_frames_max_seen"] == 12.0

    accumulator.reset()
    assert accumulator.to_dict() == {}
