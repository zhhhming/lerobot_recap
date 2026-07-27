#!/usr/bin/env python

"""Fake-clock contracts for the nero egg deployment memory progress assist."""

from __future__ import annotations

import pytest

from lerobot.inference_engines.memory_progress_assist import NeroEggMemoryProgressAssist


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _output(subtask: str, progress: float) -> str:
    return f"Subtask: {subtask}; Progress: {progress:.1f}"


def test_stirring_only_advances_point_six_to_point_seven_after_six_seconds():
    clock = FakeClock()
    assist = NeroEggMemoryProgressAssist(clock=clock)
    stirring = "Stir the beaten eggs."

    assert assist.apply(_output(stirring, 0.5)).text == _output(stirring, 0.5)
    clock.advance(20.0)
    assert assist.apply(_output(stirring, 0.5)).text == _output(stirring, 0.5)

    assert assist.apply(_output(stirring, 0.6)).text == _output(stirring, 0.6)
    clock.advance(5.999)
    assert assist.apply(_output(stirring, 0.6)).text == _output(stirring, 0.6)

    clock.advance(0.001)
    forced = assist.apply(_output(stirring, 0.6))
    assert forced.text == _output(stirring, 0.7)
    assert forced.forced is True

    clock.advance(20.0)
    terminal = assist.apply(_output(stirring, 0.6))
    assert terminal.text == _output(stirring, 0.7)
    assert terminal.forced is False
    assert terminal.reason == "clamped_regression"

    completed = assist.apply(_output(stirring, 0.8))
    assert completed.text == _output(stirring, 0.8)
    assert completed.reason == "natural_progress"


def test_frying_only_advances_point_seven_to_point_eight_after_six_seconds():
    clock = FakeClock()
    assist = NeroEggMemoryProgressAssist(clock=clock)
    frying = "Start frying the eggs."

    assist.apply(_output(frying, 0.6))
    clock.advance(20.0)
    assert assist.apply(_output(frying, 0.6)).text == _output(frying, 0.6)

    assist.apply(_output(frying, 0.7))
    clock.advance(5.999)
    assert assist.apply(_output(frying, 0.7)).text == _output(frying, 0.7)

    clock.advance(0.001)
    forced = assist.apply(_output(frying, 0.7))
    assert forced.text == _output(frying, 0.8)
    assert forced.forced is True

    clock.advance(20.0)
    terminal = assist.apply(_output(frying, 0.7))
    assert terminal.text == _output(frying, 0.8)
    assert terminal.forced is False
    assert terminal.reason == "clamped_regression"


@pytest.mark.parametrize(
    ("subtask", "start_progress", "natural_progress"),
    [
        ("Stir the beaten eggs.", 0.6, 0.7),
        ("Start frying the eggs.", 0.7, 0.8),
    ],
)
def test_natural_progress_before_six_seconds_is_not_forced(
    subtask: str,
    start_progress: float,
    natural_progress: float,
):
    clock = FakeClock()
    assist = NeroEggMemoryProgressAssist(clock=clock)

    assist.apply(_output(subtask, start_progress))
    clock.advance(5.9)
    natural = assist.apply(_output(subtask, natural_progress))
    assert natural.text == _output(subtask, natural_progress)
    assert natural.reason == "natural_progress"
    assert natural.forced is False


@pytest.mark.parametrize(
    "output",
    [
        "Subtask: Pick up the pan and the spatula.; Progress: 0.5",
        "not a subtask output",
        "Subtask: Start frying the eggs.; Progress: 0.55",
        "",
    ],
)
def test_unrelated_or_unparseable_outputs_pass_through(output: str):
    assist = NeroEggMemoryProgressAssist(clock=FakeClock())

    result = assist.apply(output)

    assert result.text == output
    assert result.adjusted is False
    assert result.forced is False


def test_name_matching_ignores_case_whitespace_and_trailing_period():
    clock = FakeClock()
    assist = NeroEggMemoryProgressAssist(clock=clock)
    first = "Subtask:  START   FRYING THE EGGS ; Progress: 0.7"

    assert assist.apply(first).text == first
    clock.advance(6.0)
    result = assist.apply(first)

    assert result.text == "Subtask: START FRYING THE EGGS; Progress: 0.8"


def test_unrelated_output_and_reset_both_discard_an_old_plateau():
    clock = FakeClock()
    assist = NeroEggMemoryProgressAssist(clock=clock)
    stirring = "Stir the beaten eggs."

    assist.apply(_output(stirring, 0.6))
    clock.advance(5.0)
    assist.apply(_output("Pick up the pan and the spatula.", 0.5))
    clock.advance(2.0)
    assert assist.apply(_output(stirring, 0.6)).text == _output(stirring, 0.6)

    clock.advance(5.0)
    assist.reset()
    clock.advance(2.0)
    assert assist.apply(_output(stirring, 0.6)).text == _output(stirring, 0.6)


def test_clock_must_not_move_backwards():
    clock = FakeClock()
    assist = NeroEggMemoryProgressAssist(clock=clock)
    frying = "Start frying the eggs."
    assist.apply(_output(frying, 0.7))
    clock.now = -1.0

    with pytest.raises(ValueError, match="finite and non-negative"):
        assist.apply(_output(frying, 0.7))
