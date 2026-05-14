#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from lerobot.scripts.lerobot_hil_record import (
    HILRecordConfig,
    HILRecordDatasetConfig,
    KeyboardEvents,
    _TerminalKeyboardListener,
    hil_record,
)
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


class _NoopListener:
    def stop(self) -> None:
        pass


def _make_teleop_hil_config(root, *, force: bool = False) -> HILRecordConfig:
    return HILRecordConfig(
        robot=MockRobotConfig(),
        dataset=HILRecordDatasetConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=root,
            push_to_hub=False,
            force=force,
        ),
        mode="teleop",
        teleop=MockTeleopConfig(),
        play_sounds=False,
    )


@pytest.mark.parametrize(
    ("text", "event"),
    [
        ("\x1b[C", "right"),
        ("\x1bOC", "right"),
        ("\x1b[1;5C", "right"),
        ("\x1b[D", "left"),
        ("\x1bOD", "left"),
        ("\x1b[1;5D", "left"),
        ("\r", "enter"),
        ("\n", "enter"),
        (" ", "space"),
        ("\x1b", "esc"),
        ("q", "q"),
        ("e", "e"),
    ],
)
def test_terminal_keyboard_listener_parses_control_keys(text, event):
    assert _TerminalKeyboardListener.parse_key(text) == event


def test_terminal_keyboard_listener_ignores_unknown_keys():
    assert _TerminalKeyboardListener.parse_key("x") is None


def test_terminal_keyboard_listener_does_not_start_without_tty():
    class NonTty:
        def isatty(self):
            return False

    assert not _TerminalKeyboardListener(KeyboardEvents(), stdin=NonTty()).start()


def test_hil_record_removes_new_dataset_when_no_episodes_are_saved(tmp_path, monkeypatch):
    root = tmp_path / "empty_hil_record"

    def stop_immediately(events):
        events.push("esc")
        return _NoopListener()

    monkeypatch.setattr("lerobot.scripts.lerobot_hil_record._init_keyboard_listener", stop_immediately)

    dataset = hil_record(_make_teleop_hil_config(root))

    assert dataset.num_episodes == 0
    assert not root.exists()


def test_hil_record_force_refuses_non_dataset_directory(tmp_path):
    root = tmp_path / "not_a_dataset"
    root.mkdir()
    (root / "keep.txt").write_text("not a LeRobot dataset")

    with pytest.raises(FileExistsError, match="Refusing to remove"):
        hil_record(_make_teleop_hil_config(root, force=True))

    assert root.exists()
    assert (root / "keep.txt").read_text() == "not a LeRobot dataset"


def test_hil_record_force_removes_existing_lerobot_dataset_directory(tmp_path, monkeypatch):
    root = tmp_path / "existing_hil_record"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    (root / "old_marker.txt").write_text("old")

    def stop_immediately(events):
        events.push("esc")
        return _NoopListener()

    monkeypatch.setattr("lerobot.scripts.lerobot_hil_record._init_keyboard_listener", stop_immediately)

    dataset = hil_record(_make_teleop_hil_config(root, force=True))

    assert dataset.num_episodes == 0
    assert not root.exists()


def test_hil_record_force_cannot_be_used_with_resume(tmp_path):
    with pytest.raises(ValueError, match="--dataset.force cannot be used with --resume"):
        HILRecordConfig(
            robot=MockRobotConfig(),
            dataset=HILRecordDatasetConfig(
                repo_id=DUMMY_REPO_ID,
                single_task="Dummy task",
                root=tmp_path / "record",
                push_to_hub=False,
                force=True,
            ),
            mode="teleop",
            teleop=MockTeleopConfig(),
            play_sounds=False,
            resume=True,
        )
