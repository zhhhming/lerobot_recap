#!/usr/bin/env python

"""Deploy configuration, dataset, and session tests for subtask elapsed time."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

import lerobot.scripts.lerobot_policy_deploy as deploy_module
from lerobot.datasets.subtask_timing import (
    SubtaskSegmentStats,
    SubtaskSequenceContract,
    normalize_subtask_name,
)
from lerobot.processor.subtask_time_processor import SubtaskTimeConditionProcessorStep


def _runtime_features() -> dict:
    return {
        "action": {"dtype": "float32", "shape": (2,), "names": ["joint.pos", "gripper.pos"]},
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["joint.pos", "gripper.pos"],
        },
        "observation.images.main": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
            "info": {"video.codec": "av1"},
        },
    }


def _dataset_meta(*, runtime_features: dict | None = None) -> SimpleNamespace:
    features = deepcopy(runtime_features or _runtime_features())
    features.update(deepcopy(deploy_module.DEFAULT_FEATURES))
    features.update(
        {
            "subtask": {"dtype": "string", "shape": (1,), "names": None},
            "subtask_progress": {"dtype": "float32", "shape": (1,), "names": None},
        }
    )
    return SimpleNamespace(robot_type="test_robot", fps=30, features=features)


def test_metadata_compatibility_ignores_only_dataset_annotation_features_without_mutation():
    runtime_features = _runtime_features()
    dataset_meta = _dataset_meta(runtime_features=runtime_features)
    original_features = deepcopy(dataset_meta.features)

    deploy_module._check_metadata_compatibility(dataset_meta, "test_robot", 30, runtime_features)

    assert dataset_meta.features == original_features


@pytest.mark.parametrize(
    ("mutation", "error_field"),
    [
        (lambda meta: setattr(meta, "robot_type", "other_robot"), "robot_type"),
        (lambda meta: setattr(meta, "fps", 20), "fps"),
        (lambda meta: meta.features["action"].update(shape=(3,)), "features"),
        (lambda meta: meta.features["action"].update(dtype="float64"), "features"),
        (
            lambda meta: meta.features["observation.state"].update(
                names=["other_joint.pos", "gripper.pos"]
            ),
            "features",
        ),
        (lambda meta: meta.features.pop("observation.images.main"), "features"),
        (
            lambda meta: meta.features.update(
                {
                    "observation.images.extra": {
                        "dtype": "video",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channels"],
                    }
                }
            ),
            "features",
        ),
        (
            lambda meta: meta.features.update(
                {"operator_note": {"dtype": "string", "shape": (1,), "names": None}}
            ),
            "features",
        ),
    ],
)
def test_metadata_compatibility_keeps_robot_fps_and_hardware_checks_strict(mutation, error_field):
    runtime_features = _runtime_features()
    dataset_meta = _dataset_meta(runtime_features=runtime_features)
    mutation(dataset_meta)

    with pytest.raises(ValueError, match=error_field):
        deploy_module._check_metadata_compatibility(dataset_meta, "test_robot", 30, runtime_features)


def test_metadata_compatibility_failure_happens_before_robot_connect(monkeypatch):
    calls: list[str] = []

    class FakeRobot:
        robot_type = "test_robot"
        is_connected = False

        def connect(self):
            calls.append("connect")
            self.is_connected = True

        def disconnect(self):
            calls.append("disconnect")
            self.is_connected = False

    class FakeStatusDisplay:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            calls.append("status_start")

        def stop(self):
            calls.append("status_stop")

    cfg = SimpleNamespace(
        hf_hub_offline=False,
        policy=SimpleNamespace(
            use_subtask_time_conditioning=False,
            pretrained_path="/tmp/checkpoint",
            device="cpu",
        ),
        use_subtask_time_conditioning=None,
        subtask_time_deployment_margin_seconds=5.0,
        dataset=SimpleNamespace(
            repo_id="owner/repo",
            root="/tmp/dataset",
            revision=None,
            fps=30,
            rename_map={},
        ),
        robot=SimpleNamespace(),
        status_display="plain",
        status_refresh_hz=1.0,
        play_sounds=False,
    )
    robot = FakeRobot()
    dataset_meta = _dataset_meta()

    monkeypatch.setattr(deploy_module, "asdict", lambda _cfg: {})
    monkeypatch.setattr(deploy_module, "init_logging", lambda: None)
    monkeypatch.setattr(deploy_module, "log_say", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(deploy_module, "_load_subtask_time_sequence_contract", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(deploy_module, "LeRobotDatasetMetadata", lambda *_args, **_kwargs: dataset_meta)
    monkeypatch.setattr(deploy_module, "make_robot_from_config", lambda _cfg: robot)
    monkeypatch.setattr(deploy_module, "make_default_processors", lambda: (None, None, None))
    monkeypatch.setattr(
        deploy_module, "_build_dataset_features", lambda *_args, **_kwargs: _runtime_features()
    )
    monkeypatch.setattr(deploy_module, "TerminalStatusDisplay", FakeStatusDisplay)
    monkeypatch.setattr(deploy_module, "_load_policy_from_model_dir", lambda _cfg: SimpleNamespace())
    monkeypatch.setattr(
        deploy_module,
        "make_pre_post_processors",
        lambda **_kwargs: (SimpleNamespace(steps=[]), SimpleNamespace()),
    )
    monkeypatch.setattr(
        deploy_module,
        "_validate_subtask_time_processor_presence",
        lambda *_args, **_kwargs: None,
    )

    def fail_compatibility(*_args, **_kwargs):
        calls.append("compatibility_check")
        raise ValueError("incompatible metadata")

    monkeypatch.setattr(deploy_module, "_check_metadata_compatibility", fail_compatibility)

    with pytest.raises(ValueError, match="incompatible metadata"):
        deploy_module.policy_deploy.__wrapped__(cfg)

    assert calls == ["compatibility_check"]


@pytest.mark.parametrize(
    ("checkpoint_enabled", "override", "expected"),
    [
        (False, None, False),
        (True, None, True),
        (False, False, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_deploy_time_flag_resolution(checkpoint_enabled, override, expected):
    assert (
        deploy_module._resolve_subtask_time_enabled(
            checkpoint_enabled=checkpoint_enabled,
            deploy_override=override,
        )
        is expected
    )


def test_deploy_time_forced_on_rejects_old_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        deploy_module._resolve_subtask_time_enabled(
            checkpoint_enabled=False,
            deploy_override=True,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_deployment_margin_must_be_finite_and_nonnegative(value):
    if value == 0.0:
        assert deploy_module._validate_subtask_time_deployment_margin(value) == 0.0
    else:
        with pytest.raises(ValueError, match="margin"):
            deploy_module._validate_subtask_time_deployment_margin(value)


def test_checkpoint_processor_presence_contract():
    empty = SimpleNamespace(steps=[])
    one = SimpleNamespace(steps=[SubtaskTimeConditionProcessorStep()])
    duplicate = SimpleNamespace(
        steps=[SubtaskTimeConditionProcessorStep(), SubtaskTimeConditionProcessorStep()]
    )

    deploy_module._validate_subtask_time_processor_presence(
        one,
        checkpoint_enabled=True,
        effective_enabled=True,
    )
    deploy_module._validate_subtask_time_processor_presence(
        one,
        checkpoint_enabled=True,
        effective_enabled=False,
    )
    with pytest.raises(ValueError, match="missing"):
        deploy_module._validate_subtask_time_processor_presence(
            empty,
            checkpoint_enabled=True,
            effective_enabled=False,
        )
    with pytest.raises(ValueError, match="exactly one"):
        deploy_module._validate_subtask_time_processor_presence(
            duplicate,
            checkpoint_enabled=True,
            effective_enabled=True,
        )


def test_time_off_does_not_require_or_load_deploy_dataset(monkeypatch):
    monkeypatch.setattr(
        deploy_module,
        "LeRobotDataset",
        lambda *_args, **_kwargs: pytest.fail("time-off deployment must not load the dataset"),
    )
    dataset_cfg = SimpleNamespace(repo_id=None, root=None, revision=None)
    assert (
        deploy_module._load_subtask_time_sequence_contract(
            dataset_cfg,
            effective_enabled=False,
            deployment_margin_seconds=5.0,
        )
        is None
    )


def test_time_on_requires_repo_id():
    dataset_cfg = SimpleNamespace(repo_id=None, root=None, revision=None)
    with pytest.raises(ValueError, match="dataset.repo_id"):
        deploy_module._load_subtask_time_sequence_contract(
            dataset_cfg,
            effective_enabled=True,
            deployment_margin_seconds=5.0,
        )


def test_time_on_scans_lightweight_columns_and_applies_margin(monkeypatch):
    rows = [
        {"episode_index": 0, "frame_index": 0, "index": 0, "subtask": "First."},
        {"episode_index": 0, "frame_index": 1, "index": 1, "subtask": "First."},
        {"episode_index": 0, "frame_index": 2, "index": 2, "subtask": "Second."},
    ]

    class FakeDataset:
        def __init__(self, repo_id, root=None, revision=None, download_videos=True):
            assert repo_id == "owner/repo"
            assert root == "/tmp/fake"
            assert revision == "revision"
            assert download_videos is False
            self.repo_id = repo_id
            self.root = root
            self.meta = SimpleNamespace(
                fps=10.0,
                features={name: {} for name in ("episode_index", "frame_index", "index", "subtask")},
            )
            self.selected = None

        def select_columns(self, columns):
            self.selected = tuple(columns)
            return rows

    monkeypatch.setattr(deploy_module, "LeRobotDataset", FakeDataset)
    contract = deploy_module._load_subtask_time_sequence_contract(
        SimpleNamespace(repo_id="owner/repo", root="/tmp/fake", revision="revision"),
        effective_enabled=True,
        deployment_margin_seconds=2.5,
    )

    assert contract is not None
    assert contract.fps == 10.0
    assert [item.canonical_name for item in contract.ordered_subtasks] == ["First.", "Second."]
    assert contract.ordered_subtasks[0].max_elapsed_seconds == pytest.approx(0.1)
    assert contract.ordered_subtasks[0].deployment_cap_seconds == pytest.approx(2.6)


def test_build_engine_forwards_effective_time_contract(monkeypatch):
    captured = {}

    class FakeRTCInferenceEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(deploy_module, "RTCInferenceEngine", FakeRTCInferenceEngine)
    stats = SubtaskSegmentStats(
        canonical_name="First.",
        normalized_name=normalize_subtask_name("First."),
        max_elapsed_seconds=1.0,
        deployment_cap_seconds=6.0,
    )
    contract = SubtaskSequenceContract(fps=30.0, ordered_subtasks=(stats,))
    cfg = SimpleNamespace(
        rtc=SimpleNamespace(enabled=False),
        dataset=SimpleNamespace(task="task"),
        policy=SimpleNamespace(device="cpu"),
        rtc_queue_threshold=7,
    )
    policy = SimpleNamespace(config=SimpleNamespace())

    result = deploy_module._build_engine(
        cfg,
        policy,
        preprocessor="pre",
        postprocessor="post",
        robot_wrapper="robot",
        dataset_features={"feature": {}},
        dataset_fps=30,
        shutdown_event="shutdown",
        subtask_sequence_contract=contract,
        subtask_time_enabled=True,
    )

    assert isinstance(result, FakeRTCInferenceEngine)
    assert cfg.rtc.enabled is True
    assert captured["subtask_sequence_contract"] is contract
    assert captured["subtask_time_enabled"] is True
    assert captured["rtc_queue_threshold"] == 7


@pytest.mark.parametrize(
    ("text", "event"),
    [("\x1b[C", "right"), (" ", "space"), ("h", "h"), ("\x1b", "esc")],
)
def test_deploy_keyboard_contract(text, event):
    assert deploy_module._TerminalKeyboardListener.parse_key(text) == event


def test_nero_action_clamp_and_homing_keep_gripper_within_limit():
    action = deploy_module._clamp_policy_action(
        {"left_joint.pos": 0.5, "left_gripper.pos": 0.2},
        gripper_max_width_m=0.05,
    )
    assert action == {"left_joint.pos": 0.5, "left_gripper.pos": 0.05}

    action_keys = [*(f"left_joint_{index}.pos" for index in range(7)), "left_gripper.pos"]
    obs = {key: 1.0 for key in action_keys}
    waypoints = deploy_module._start_homing(
        obs,
        action_keys,
        {"left_": [0.0] * 7},
        gripper_max_width_m=0.05,
        home_speed_rad_s=1.0,
        control_interval=0.5,
    )
    assert len(waypoints) == 2
    assert waypoints[-1]["left_gripper.pos"] == pytest.approx(0.05)
    assert all(waypoints[-1][f"left_joint_{index}.pos"] == pytest.approx(0.0) for index in range(7))


class _FakeRuntimePart:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def reset(self) -> None:
        self.calls.append("reset")


class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def soft_pause(self) -> None:
        self.calls.append("soft_pause")

    def full_reset(self) -> None:
        self.calls.append("full_reset")

    def resume(self) -> None:
        self.calls.append("resume")


def test_deploy_runtime_initial_start_soft_pause_resume_and_home_reset():
    engine = _FakeEngine()
    interpolator = _FakeRuntimePart()
    smoother = _FakeRuntimePart()
    runtime = deploy_module._PolicyDeployRuntime(engine, interpolator, smoother)

    assert runtime.paused_session_resumable is False
    runtime.start_or_resume()
    assert engine.calls == ["full_reset", "resume"]

    runtime.soft_pause()
    assert runtime.paused_session_resumable is True
    assert engine.calls[-1] == "soft_pause"
    assert interpolator.calls == ["reset", "reset"]
    assert smoother.calls == ["reset", "reset"]

    runtime.start_or_resume()
    assert engine.calls[-1] == "resume"
    assert engine.calls.count("full_reset") == 1

    runtime.full_reset()
    assert runtime.paused_session_resumable is False
    assert engine.calls[-1] == "full_reset"

    runtime.start_or_resume()
    assert engine.calls[-2:] == ["full_reset", "resume"]
