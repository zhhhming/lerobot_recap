#!/usr/bin/env python

"""Deploy activation contracts for the targeted nero egg memory progress assist."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import lerobot.scripts.lerobot_policy_deploy as deploy_module
from lerobot.inference_engines.memory_progress_assist import NeroEggMemoryProgressAssist


def _dataset(repo_id: str | None) -> SimpleNamespace:
    return SimpleNamespace(repo_id=repo_id)


def _policy(
    *,
    memory: bool = True,
    generate_subtask: bool = True,
    subtask_time: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        use_memory_conditioning=memory,
        subtask_generate_at_inference=generate_subtask,
        use_subtask_time_conditioning=subtask_time,
    )


@pytest.mark.parametrize("subtask_time", [False, True])
def test_exact_nero_egg_repo_auto_enables_assist_independently_of_subtask_time(subtask_time):
    result = deploy_module._make_nero_egg_memory_progress_assist(
        _dataset("ming326/nero_egg_subtask"),
        _policy(subtask_time=subtask_time),
        enabled=True,
    )

    assert isinstance(result, NeroEggMemoryProgressAssist)


def test_repo_matching_is_whitespace_and_case_insensitive():
    result = deploy_module._make_nero_egg_memory_progress_assist(
        _dataset("  MING326/NERO_EGG_SUBTASK  "),
        _policy(),
        enabled=True,
    )

    assert isinstance(result, NeroEggMemoryProgressAssist)


@pytest.mark.parametrize(
    ("repo_id", "enabled"),
    [
        ("owner/other", True),
        (None, True),
        ("ming326/nero_egg_subtask", False),
    ],
)
def test_other_repos_or_explicit_disable_do_not_enable_assist(repo_id, enabled):
    assert (
        deploy_module._make_nero_egg_memory_progress_assist(
            _dataset(repo_id),
            _policy(),
            enabled=enabled,
        )
        is None
    )


@pytest.mark.parametrize(
    "policy",
    [
        _policy(memory=False),
        _policy(generate_subtask=False),
    ],
)
def test_checkpoint_without_memory_updates_disables_assist(policy, caplog):
    result = deploy_module._make_nero_egg_memory_progress_assist(
        _dataset("ming326/nero_egg_subtask"),
        policy,
        enabled=True,
    )

    assert result is None
    assert "memory updates are disabled" in caplog.text


def test_build_engine_forwards_memory_progress_assist(monkeypatch):
    captured = {}

    class FakeRTCInferenceEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(deploy_module, "RTCInferenceEngine", FakeRTCInferenceEngine)
    cfg = SimpleNamespace(
        rtc=SimpleNamespace(enabled=False),
        dataset=SimpleNamespace(task="task"),
        policy=SimpleNamespace(device="cpu"),
        rtc_queue_threshold=7,
    )
    policy = SimpleNamespace(config=SimpleNamespace())
    assist = NeroEggMemoryProgressAssist()

    result = deploy_module._build_engine(
        cfg,
        policy,
        preprocessor="pre",
        postprocessor="post",
        robot_wrapper="robot",
        dataset_features={"feature": {}},
        dataset_fps=30,
        shutdown_event="shutdown",
        memory_progress_assist=assist,
    )

    assert isinstance(result, FakeRTCInferenceEngine)
    assert captured["memory_progress_assist"] is assist
