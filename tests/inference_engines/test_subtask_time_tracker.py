#!/usr/bin/env python

"""Strict fake-clock contracts for the deployment subtask-time tracker."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _contract():
    from lerobot.datasets.subtask_timing import SubtaskSegmentStats, SubtaskSequenceContract

    return SubtaskSequenceContract(
        fps=30.0,
        ordered_subtasks=(
            SubtaskSegmentStats("First.", "first", 1.0, 6.0),
            SubtaskSegmentStats("Second.", "second", 2.0, 7.0),
            SubtaskSegmentStats("Third.", "third", 3.0, 8.0),
        ),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Subtask: First.; Progress: 0.1", "First."),
        ("  subTASK :  FIRST 。 ; proGRESS : anything  ", "FIRST 。"),
        ("Subtask:\n Pick   up the fork. ;\nProgress: 0.8", "Pick up the fork."),
    ],
)
def test_parser_accepts_only_clear_subtask_progress_output(text, expected):
    from lerobot.inference_engines.subtask_time_tracker import parse_subtask_output_name

    assert parse_subtask_output_name(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Subtask: First.",
        "First.; Progress: 0.1",
        "prefix Subtask: First.; Progress: 0.1",
        "Subtask: ; Progress: 0.1",
        "Subtask: First.; Other: 0.1",
        "Subtask: First.; Progress: 0.1; Subtask: Second.; Progress: 0.2",
    ],
)
def test_parser_rejects_missing_empty_or_ambiguous_formats(text):
    from lerobot.inference_engines.subtask_time_tracker import parse_subtask_output_name

    assert parse_subtask_output_name(text) is None


def test_parser_rejects_non_string_input():
    from lerobot.inference_engines.subtask_time_tracker import parse_subtask_output_name

    with pytest.raises(TypeError, match="string"):
        parse_subtask_output_name(None)


def test_tracker_accepts_only_initial_current_or_immediate_next():
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    tracker = SubtaskTimeTracker(_contract(), clock=_FakeClock())
    tracker.commit_subtask_output("Subtask: Second.; Progress: 0.1")
    initial_rejection = tracker.snapshot()
    assert initial_rejection.current_index is None
    assert initial_rejection.last_transition_reason == "rejected_initial_not_first"

    tracker.commit_subtask_output("Subtask:  FIRST ; Progress: 0.2")
    assert tracker.snapshot().current_index == 0
    tracker.commit_subtask_output("Subtask: First.; Progress: 0.3")
    assert tracker.snapshot().current_index == 0
    tracker.commit_subtask_output("Subtask: Third.; Progress: 0.4")
    skipped = tracker.snapshot()
    assert skipped.current_index == 0
    assert skipped.last_transition_reason == "rejected_skip"

    tracker.commit_subtask_output("Subtask: second; Progress: 0.5")
    assert tracker.snapshot().current_index == 1
    tracker.commit_subtask_output("Subtask: First.; Progress: 0.6")
    old = tracker.snapshot()
    assert old.current_index == 1
    assert old.last_transition_reason == "rejected_old"

    tracker.commit_subtask_output("unparseable output")
    unparseable = tracker.snapshot()
    assert unparseable.current_index == 1
    assert unparseable.last_transition_reason == "rejected_parse"
    assert unparseable.last_rejected_output == "unparseable output"


def test_tracker_uses_exact_normalized_match_without_fuzzy_matching():
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    tracker = SubtaskTimeTracker(_contract(), clock=_FakeClock())
    tracker.commit_subtask_output("Subtask: first task; Progress: 0.1")
    snapshot = tracker.snapshot()
    assert snapshot.current_index is None
    assert snapshot.last_transition_reason == "rejected_unknown"
    assert snapshot.last_rejection_reason == "unknown_subtask"


def test_current_output_does_not_restart_timer_and_next_resets_it():
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    clock = _FakeClock()
    tracker = SubtaskTimeTracker(_contract(), clock=clock)
    tracker.commit_subtask_output("Subtask: First.; Progress: 0.1")
    clock.advance(2.0)
    tracker.commit_subtask_output("Subtask: first; Progress: 0.4")
    clock.advance(1.0)
    assert tracker.snapshot().raw_elapsed_seconds == pytest.approx(3.0)

    tracker.commit_subtask_output("Subtask: SECOND.; Progress: 0.1")
    advanced = tracker.snapshot()
    assert advanced.current_index == 1
    assert advanced.current_name == "Second."
    assert advanced.raw_elapsed_seconds == 0.0
    assert advanced.last_transition_reason == "advanced"

    clock.advance(0.75)
    assert tracker.snapshot().raw_elapsed_seconds == pytest.approx(0.75)


def test_tracker_never_wraps_after_the_last_subtask():
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    tracker = SubtaskTimeTracker(_contract(), clock=_FakeClock())
    for name in ("First", "Second", "Third"):
        tracker.commit_subtask_output(f"Subtask: {name}; Progress: 0.1")
    tracker.commit_subtask_output("Subtask: First.; Progress: 0.2")

    snapshot = tracker.snapshot()
    assert snapshot.current_index == 2
    assert snapshot.last_transition_reason == "rejected_old"


def test_tracker_fake_clock_pause_cap_and_full_reset_contract():
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    clock = _FakeClock()
    tracker = SubtaskTimeTracker(_contract(), clock=clock)
    tracker.commit_subtask_output("Subtask: First.; Progress: 0.1")
    clock.advance(2.0)
    running = tracker.snapshot()
    assert running.raw_elapsed_seconds == pytest.approx(2.0)
    assert running.effective_elapsed_seconds == pytest.approx(2.0)
    assert running.cap_seconds == pytest.approx(6.0)
    assert running.running is True

    tracker.pause()
    clock.advance(90.0)
    paused = tracker.snapshot()
    assert paused.raw_elapsed_seconds == pytest.approx(2.0)
    assert paused.paused is True
    assert paused.running is False

    tracker.resume()
    clock.advance(0.8)
    assert tracker.snapshot().raw_elapsed_seconds == pytest.approx(2.8)

    clock.advance(10.0)
    capped = tracker.snapshot()
    assert capped.raw_elapsed_seconds == pytest.approx(12.8)
    assert capped.effective_elapsed_seconds == pytest.approx(6.0)

    tracker.full_reset()
    reset = tracker.snapshot()
    assert reset.current_index is None
    assert reset.current_name is None
    assert reset.time_valid is False
    assert reset.raw_elapsed_seconds == 0.0
    assert reset.effective_elapsed_seconds == 0.0
    assert reset.cap_seconds is None
    assert reset.last_transition_reason == "full_reset"


def test_pause_resume_are_idempotent_and_pause_before_start_is_supported():
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    clock = _FakeClock()
    tracker = SubtaskTimeTracker(_contract(), clock=clock)
    tracker.pause()
    tracker.pause()
    tracker.commit_subtask_output("Subtask: First.; Progress: 0.1")
    clock.advance(10.0)
    assert tracker.snapshot().raw_elapsed_seconds == 0.0

    tracker.resume()
    tracker.resume()
    clock.advance(1.25)
    assert tracker.snapshot().raw_elapsed_seconds == pytest.approx(1.25)


def test_full_reset_preserves_session_pause_state_but_clears_semantic_time():
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    tracker = SubtaskTimeTracker(_contract(), clock=_FakeClock())
    tracker.commit_subtask_output("Subtask: First.; Progress: 0.1")
    tracker.pause()
    tracker.full_reset()

    snapshot = tracker.snapshot()
    assert snapshot.paused is True
    assert snapshot.current_index is None
    assert snapshot.time_valid is False


def test_backward_clock_raises_without_corrupting_elapsed_state():
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    clock = _FakeClock()
    tracker = SubtaskTimeTracker(_contract(), clock=clock)
    tracker.commit_subtask_output("Subtask: First.; Progress: 0.1")
    clock.advance(3.0)
    assert tracker.snapshot().raw_elapsed_seconds == pytest.approx(3.0)

    clock.now = 2.0
    with pytest.raises(RuntimeError, match="moved backwards"):
        tracker.snapshot()

    clock.now = 4.0
    assert tracker.snapshot().raw_elapsed_seconds == pytest.approx(4.0)


@pytest.mark.parametrize("bad_clock_value", [True, "1", float("nan"), float("inf")])
def test_tracker_rejects_invalid_clock_values(bad_clock_value):
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    tracker = SubtaskTimeTracker(_contract(), clock=lambda: bad_clock_value)
    with pytest.raises((TypeError, ValueError), match="clock"):
        tracker.commit_subtask_output("Subtask: First.; Progress: 0.1")


def test_snapshot_is_frozen_and_cannot_mutate_tracker_state():
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    tracker = SubtaskTimeTracker(_contract(), clock=_FakeClock())
    tracker.commit_subtask_output("Subtask: First.; Progress: 0.1")
    snapshot = tracker.snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.current_index = 2
    assert tracker.snapshot().current_index == 0


def test_contract_rejects_normalization_collision():
    from lerobot.datasets.subtask_timing import SubtaskSegmentStats
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    contract = _contract()
    collision = replace(
        contract,
        ordered_subtasks=contract.ordered_subtasks
        + (SubtaskSegmentStats(" FIRST 。", "first", 4.0, 9.0),),
    )
    with pytest.raises(ValueError, match="normalization collision.*first"):
        SubtaskTimeTracker(collision)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda contract: replace(contract, fps=0.0), "fps"),
        (lambda contract: replace(contract, ordered_subtasks=()), "at least one"),
        (
            lambda contract: replace(
                contract,
                ordered_subtasks=(replace(contract.ordered_subtasks[0], normalized_name="wrong"),),
            ),
            "normalized_name",
        ),
        (
            lambda contract: replace(
                contract,
                ordered_subtasks=(replace(contract.ordered_subtasks[0], max_elapsed_seconds=-1.0),),
            ),
            "max_elapsed_seconds",
        ),
        (
            lambda contract: replace(
                contract,
                ordered_subtasks=(replace(contract.ordered_subtasks[0], deployment_cap_seconds=0.5),),
            ),
            "deployment_cap_seconds",
        ),
    ],
)
def test_tracker_revalidates_manual_sequence_contracts(mutate, message):
    from lerobot.inference_engines.subtask_time_tracker import SubtaskTimeTracker

    with pytest.raises((TypeError, ValueError), match=message):
        SubtaskTimeTracker(mutate(_contract()))
