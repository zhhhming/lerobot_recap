#!/usr/bin/env python

"""Contracts for subtask segment scanning and the training timing wrapper."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch


TIMING_COLUMNS = ("episode_index", "frame_index", "index", "subtask")


def _rows():
    labels = ["A.", "A.", "A.", "B", "B", "A.", "A.", "B"]
    episodes = [0, 0, 0, 0, 0, 1, 1, 1]
    frames = [0, 1, 2, 3, 4, 0, 1, 2]
    return [
        {
            "index": index,
            "episode_index": episode,
            "frame_index": frame,
            "subtask": label,
            "subtask_progress": torch.tensor(0.5, dtype=torch.float32),
        }
        for index, (episode, frame, label) in enumerate(
            zip(episodes, frames, labels, strict=True)
        )
    ]


class _ColumnView:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


class _FakeDataset:
    def __init__(self, rows=None, *, features=None, fps=10, episodes=None):
        self.rows = _rows() if rows is None else rows
        feature_names = self.rows[0] if features is None and self.rows else features
        feature_names = () if feature_names is None else feature_names
        self.meta = SimpleNamespace(
            fps=fps,
            features={name: {} for name in feature_names},
            camera_keys=[],
            stats={},
        )
        self.fps = fps
        self.episodes = episodes
        self.repo_id = "fake/timing"
        self.root = "/fake/timing"
        self.item_calls = []
        self.raw_item_calls = []
        self.select_calls = []

    def __len__(self):
        return len(self.rows)

    @property
    def num_frames(self):
        return len(self.rows)

    @property
    def num_episodes(self):
        return len({row["episode_index"] for row in self.rows})

    @property
    def features(self):
        return self.meta.features

    def __getitem__(self, index):
        self.item_calls.append(index)
        return dict(self.rows[index])

    def get_raw_item(self, index):
        self.raw_item_calls.append(index)
        return dict(self.rows[index])

    def select_columns(self, names):
        self.select_calls.append(tuple(names))
        return _ColumnView([{name: row[name] for name in names} for row in self.rows])


def _timing_rows(labels, *, episode=0, start_index=0):
    return [
        {
            "index": start_index + frame,
            "episode_index": episode,
            "frame_index": frame,
            "subtask": label,
        }
        for frame, label in enumerate(labels)
    ]


def test_scanner_freezes_elapsed_segment_and_cap_contract():
    from lerobot.datasets.subtask_timing import scan_subtask_timing

    scan = scan_subtask_timing(_rows(), fps=10, deployment_margin_seconds=5.0)

    assert scan.elapsed_seconds.tolist() == pytest.approx(
        [0.0, 0.1, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0]
    )
    assert scan.elapsed_seconds.dtype == torch.float32
    assert scan.valid.dtype == torch.bool
    assert scan.valid.tolist() == [True] * 8
    assert scan.segment_indices.dtype == torch.int64
    assert scan.segment_indices.tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    contract = scan.sequence_contract
    assert contract.fps == 10
    assert [item.canonical_name for item in contract.ordered_subtasks] == ["A.", "B"]
    assert [item.normalized_name for item in contract.ordered_subtasks] == ["a", "b"]
    assert contract.ordered_subtasks[0].max_elapsed_seconds == pytest.approx(0.2)
    assert contract.ordered_subtasks[0].deployment_cap_seconds == pytest.approx(5.2)


def test_frame_33_is_1p1_seconds_at_30_fps_and_switch_resets_to_zero():
    from lerobot.datasets.subtask_timing import scan_subtask_timing

    rows = _timing_rows(["A"] * 34 + ["B"])
    scan = scan_subtask_timing(rows, fps=30)

    assert scan.elapsed_seconds[0].item() == 0.0
    assert scan.elapsed_seconds[33].item() == pytest.approx(1.1)
    assert scan.elapsed_seconds[34].item() == 0.0
    assert scan.segment_indices[34].item() == 1


def test_episode_boundary_resets_time_and_selected_absolute_indices_are_supported():
    from lerobot.datasets.subtask_timing import scan_subtask_timing

    rows = _timing_rows(["A", "A", "B"], episode=2, start_index=100)
    rows += _timing_rows(["A", "A", "B"], episode=7, start_index=500)
    scan = scan_subtask_timing(rows, fps=20, dataset_name="selected-view")

    assert scan.elapsed_seconds.tolist() == pytest.approx([0.0, 0.05, 0.0, 0.0, 0.05, 0.0])
    assert scan.segment_indices.tolist() == [0, 0, 1, 0, 0, 1]
    assert [item.normalized_name for item in scan.sequence_contract.ordered_subtasks] == ["a", "b"]


def test_scanner_uses_normalized_labels_for_segments_and_preserves_canonical_name():
    from lerobot.datasets.subtask_timing import normalize_subtask_name, scan_subtask_timing

    rows = _timing_rows(["  Pick   Up. ", "pick up", "B。"])
    scan = scan_subtask_timing(rows, fps=10)

    assert normalize_subtask_name("  Pick   Up. ") == "pick up"
    assert scan.elapsed_seconds.tolist() == pytest.approx([0.0, 0.1, 0.0])
    assert scan.segment_indices.tolist() == [0, 0, 1]
    first = scan.sequence_contract.ordered_subtasks[0]
    assert first.canonical_name == "Pick Up."
    assert first.normalized_name == "pick up"


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True, "30"])
def test_scanner_rejects_invalid_fps(fps):
    from lerobot.datasets.subtask_timing import scan_subtask_timing

    with pytest.raises((TypeError, ValueError), match="fps"):
        scan_subtask_timing(_rows(), fps=fps)


@pytest.mark.parametrize("margin", [-1, float("nan"), float("inf"), True, "5"])
def test_scanner_rejects_invalid_deployment_margin(margin):
    from lerobot.datasets.subtask_timing import scan_subtask_timing

    with pytest.raises((TypeError, ValueError), match="margin"):
        scan_subtask_timing(_rows(), fps=10, deployment_margin_seconds=margin)


@pytest.mark.parametrize("label", ["", "   ", None, 3])
def test_scanner_rejects_empty_or_non_string_labels(label):
    from lerobot.datasets.subtask_timing import scan_subtask_timing

    rows = _rows()
    rows[1] = {**rows[1], "subtask": label}
    with pytest.raises(ValueError, match="non-empty string.*row 1"):
        scan_subtask_timing(rows, fps=10, dataset_name="bad-label")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_index", 2, "frame_index"),
        ("index", 9, "index"),
        ("episode_index", 0.5, "episode_index"),
    ],
)
def test_scanner_rejects_noncontiguous_or_invalid_indices(field, value, message):
    from lerobot.datasets.subtask_timing import scan_subtask_timing

    rows = _rows()
    rows[1] = {**rows[1], field: value}
    with pytest.raises(ValueError, match=message):
        scan_subtask_timing(rows, fps=10, dataset_name="bad-index")


def test_scanner_rejects_reappearing_episode_and_empty_data():
    from lerobot.datasets.subtask_timing import scan_subtask_timing

    rows = _timing_rows(["A"], episode=0)
    rows += _timing_rows(["A"], episode=1, start_index=10)
    rows += _timing_rows(["A"], episode=0, start_index=20)
    with pytest.raises(ValueError, match="episode.*reappears"):
        scan_subtask_timing(rows, fps=10, dataset_name="interleaved")
    with pytest.raises(ValueError, match="no frames"):
        scan_subtask_timing([], fps=10)


def test_scanner_rejects_repeated_or_inconsistent_episode_sequences_with_diagnostics():
    from lerobot.datasets.subtask_timing import scan_subtask_timing

    repeated = _timing_rows(["A", "B", "A"], episode=0)
    with pytest.raises(ValueError, match=r"repeats.*\['A', 'B', 'A'\]"):
        scan_subtask_timing(repeated, fps=10, dataset_name="repeat-dataset")

    inconsistent = _timing_rows(["A", "B"], episode=0)
    inconsistent += _timing_rows(["A", "C"], episode=1, start_index=20)
    with pytest.raises(ValueError, match="inconsistent.*episode 1.*expected=.*actual="):
        scan_subtask_timing(inconsistent, fps=10, dataset_name="different-sequence")


def test_scanner_rejects_missing_columns_before_scanning():
    from lerobot.datasets.subtask_timing import scan_subtask_timing

    rows = [{"index": 0, "episode_index": 0, "frame_index": 0}]
    with pytest.raises(ValueError, match="missing.*subtask"):
        scan_subtask_timing(rows, fps=10)


def test_wrapper_adds_true_time_without_mutating_or_rescanning_on_getitem():
    from lerobot.datasets.subtask_timing import SubtaskTimingDataset

    base = _FakeDataset()
    wrapped = SubtaskTimingDataset(base)
    assert base.select_calls == [TIMING_COLUMNS]

    item = wrapped[2]
    switched = wrapped[3]

    assert item["subtask_elapsed_seconds"].dtype == torch.float32
    assert item["subtask_elapsed_seconds"].item() == pytest.approx(0.2)
    assert item["subtask_time_valid"] is True
    assert item["subtask_segment_index"].dtype == torch.int64
    assert item["subtask_segment_index"].item() == 0
    assert switched["subtask_elapsed_seconds"].item() == pytest.approx(0.0)
    assert switched["subtask_segment_index"].item() == 1
    assert base.item_calls == [2, 3]
    assert len(base.select_calls) == 1
    assert "subtask_elapsed_seconds" not in base.rows[2]
    assert wrapped.get_raw_item(1) == base.rows[1]
    assert base.raw_item_calls == [1]
    assert wrapped.select_columns(["index"])["index"] == list(range(8))
    assert base.select_calls[-1] == ("index",)


def test_wrapper_proxies_training_attributes_and_protects_lookup_storage():
    from lerobot.datasets.subtask_timing import SubtaskTimingDataset

    base = _FakeDataset(episodes=[0, 1])
    wrapped = SubtaskTimingDataset(base)
    assert wrapped.dataset is base
    assert wrapped.meta is base.meta
    assert wrapped.episodes is base.episodes
    assert wrapped.num_frames == base.num_frames
    assert wrapped.num_episodes == base.num_episodes
    assert wrapped.features is base.features
    assert wrapped.fps == base.fps
    assert wrapped.repo_id == base.repo_id
    assert wrapped.root == base.root
    assert len(wrapped) == len(base)
    assert wrapped.sequence_contract.fps == 10
    assert wrapped.lookup_nbytes == len(base) * (4 + 1 + 8)

    item = wrapped[1]
    item["subtask_elapsed_seconds"].fill_(999)
    item["subtask_segment_index"].fill_(999)
    again = wrapped[1]
    assert again["subtask_elapsed_seconds"].item() == pytest.approx(0.1)
    assert again["subtask_segment_index"].item() == 0


def test_wrapper_lookup_indices_align_with_a_selected_episode_view():
    from lerobot.datasets.subtask_timing import SubtaskTimingDataset

    rows = _timing_rows(["A", "A", "B"], episode=2, start_index=100)
    rows += _timing_rows(["A", "A", "B"], episode=7, start_index=500)
    base = _FakeDataset(rows, fps=20, episodes=[2, 7])
    wrapped = SubtaskTimingDataset(base)

    item = wrapped[4]
    assert item["episode_index"] == 7
    assert item["frame_index"] == 1
    assert item["subtask_elapsed_seconds"].item() == pytest.approx(0.05)
    assert item["subtask_segment_index"].item() == 0


def _collect_loader_batches(num_workers):
    from lerobot.datasets.subtask_timing import SubtaskTimingDataset

    wrapped = SubtaskTimingDataset(_FakeDataset())
    kwargs = {"multiprocessing_context": "spawn"} if num_workers else {}
    loader = torch.utils.data.DataLoader(
        wrapped,
        batch_size=3,
        num_workers=num_workers,
        shuffle=False,
        **kwargs,
    )
    return list(loader)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_wrapper_dataloader_workers_collate_stable_timing_fields(num_workers):
    batches = _collect_loader_batches(num_workers)
    elapsed = torch.cat([batch["subtask_elapsed_seconds"] for batch in batches])
    valid = torch.cat([batch["subtask_time_valid"] for batch in batches])
    segments = torch.cat([batch["subtask_segment_index"] for batch in batches])

    assert elapsed.dtype == torch.float32
    assert elapsed.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0])
    assert valid.dtype == torch.bool
    assert valid.tolist() == [True] * 8
    assert segments.dtype == torch.int64


def _collect_composed_loader_batches(num_workers):
    from lerobot.datasets.memory_history import MemoryHistoryDataset
    from lerobot.datasets.subtask_timing import SubtaskTimingDataset

    wrapped = MemoryHistoryDataset(
        SubtaskTimingDataset(_FakeDataset()),
        lookback_min_frames=1,
        lookback_max_frames=1,
    )
    kwargs = {"multiprocessing_context": "spawn"} if num_workers else {}
    loader = torch.utils.data.DataLoader(
        wrapped,
        batch_size=3,
        num_workers=num_workers,
        shuffle=False,
        **kwargs,
    )
    return list(loader)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_composed_time_memory_wrappers_work_with_dataloader_workers(num_workers):
    batches = _collect_composed_loader_batches(num_workers)
    elapsed = torch.cat([batch["subtask_elapsed_seconds"] for batch in batches])
    offsets = torch.cat([batch["memory_frame_offset"] for batch in batches])

    assert elapsed.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0])
    assert offsets.tolist() == [1] * 8
    assert all("memory_valid" in batch for batch in batches)


def _factory_cfg(
    *,
    time_enabled,
    memory_enabled,
    policy_type="pi0",
    predict_subtask=True,
    streaming=False,
):
    return SimpleNamespace(
        policy=SimpleNamespace(
            type=policy_type,
            use_subtask_time_conditioning=time_enabled,
            use_memory_conditioning=memory_enabled,
            predict_subtask=predict_subtask,
            observation_delta_indices=None,
            action_delta_indices=None,
            reward_delta_indices=None,
        ),
        dataset=SimpleNamespace(
            repo_id="fake/repo",
            root=None,
            revision=None,
            episodes=[0, 1],
            streaming=streaming,
            image_transforms=SimpleNamespace(enable=False),
            video_backend="pyav",
            use_imagenet_stats=False,
        ),
        memory_lookback_min_frames=1,
        memory_lookback_max_frames=2,
        tolerance_s=1e-4,
        num_workers=2,
    )


def _patch_factory_dataset(monkeypatch, *, features=None):
    from lerobot.datasets import factory

    dataset = _FakeDataset(features=features, episodes=[0, 1])
    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", lambda *args, **kwargs: dataset.meta)
    monkeypatch.setattr(factory, "LeRobotDataset", lambda *args, **kwargs: dataset)
    return factory, dataset


@pytest.mark.parametrize("policy_type", ["pi0", "pi05"])
@pytest.mark.parametrize(
    ("time_enabled", "memory_enabled"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_factory_supports_independent_time_memory_wrapper_matrix(
    monkeypatch, policy_type, time_enabled, memory_enabled
):
    from lerobot.datasets.memory_history import MemoryHistoryDataset
    from lerobot.datasets.subtask_timing import SubtaskTimingDataset

    factory, base = _patch_factory_dataset(monkeypatch)
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([1]))
    cfg = _factory_cfg(
        time_enabled=time_enabled,
        memory_enabled=memory_enabled,
        policy_type=policy_type,
    )

    result = factory.make_dataset(cfg)

    if not time_enabled and not memory_enabled:
        assert result is base
    elif time_enabled and not memory_enabled:
        assert isinstance(result, SubtaskTimingDataset)
        assert result.dataset is base
    elif not time_enabled and memory_enabled:
        assert isinstance(result, MemoryHistoryDataset)
        assert result.dataset is base
    else:
        assert isinstance(result, MemoryHistoryDataset)
        assert isinstance(result.dataset, SubtaskTimingDataset)
        assert result.dataset.dataset is base

    item = result[2]
    assert any(key.startswith("subtask_time_") for key in item) is time_enabled
    assert ("subtask_elapsed_seconds" in item) is time_enabled
    assert any(key.startswith("memory_") for key in item) is memory_enabled
    assert len(base.item_calls) == 1
    assert len(base.select_calls) == (1 if time_enabled else 0)


def test_time_only_does_not_consume_history_rng_and_disabled_does_not_scan(monkeypatch):
    factory, base = _patch_factory_dataset(monkeypatch)

    def unexpected_randint(*args, **kwargs):
        raise AssertionError("time-only must not consume history RNG")

    monkeypatch.setattr(torch, "randint", unexpected_randint)
    timed = factory.make_dataset(_factory_cfg(time_enabled=True, memory_enabled=False))
    timed[2]
    assert base.select_calls == [TIMING_COLUMNS]

    factory, disabled_base = _patch_factory_dataset(monkeypatch)
    disabled = factory.make_dataset(_factory_cfg(time_enabled=False, memory_enabled=False))
    assert disabled is disabled_base
    assert disabled_base.select_calls == []


def test_composed_wrapper_proxies_raw_access_without_synthetic_history(monkeypatch):
    factory, base = _patch_factory_dataset(monkeypatch)
    result = factory.make_dataset(_factory_cfg(time_enabled=True, memory_enabled=True))

    raw = result.get_raw_item(2)
    selected = result.select_columns(["subtask"])

    assert raw == base.rows[2]
    assert not any(key.startswith("memory_") for key in raw)
    assert not any(key.startswith("subtask_time_") for key in raw)
    assert selected["subtask"] == [row["subtask"] for row in base.rows]


def test_factory_rejects_time_streaming_unsupported_policy_or_missing_subtask(monkeypatch):
    from lerobot.datasets import factory

    def unexpected_metadata(*args, **kwargs):
        raise AssertionError("unsupported configs must fail before metadata construction")

    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", unexpected_metadata)
    with pytest.raises(ValueError, match="time.*streaming=false"):
        factory.make_dataset(
            _factory_cfg(time_enabled=True, memory_enabled=False, streaming=True)
        )
    with pytest.raises(ValueError, match="pi0.*pi05"):
        factory.make_dataset(
            _factory_cfg(time_enabled=True, memory_enabled=False, policy_type="act")
        )
    with pytest.raises(ValueError, match="predict_subtask"):
        factory.make_dataset(
            _factory_cfg(
                time_enabled=True,
                memory_enabled=False,
                predict_subtask=False,
            )
        )

    factory, _ = _patch_factory_dataset(
        monkeypatch,
        features={"index", "episode_index", "frame_index", "subtask_progress"},
    )

    def unexpected_dataset(*args, **kwargs):
        raise AssertionError("missing timing features must fail before dataset construction")

    monkeypatch.setattr(factory, "LeRobotDataset", unexpected_dataset)
    with pytest.raises(ValueError, match="missing.*subtask"):
        factory.make_dataset(_factory_cfg(time_enabled=True, memory_enabled=False))


def test_lookup_storage_formula_stays_linear_and_compact():
    from lerobot.datasets.subtask_timing import scan_subtask_timing

    frame_count = 10_000
    scan = scan_subtask_timing(_timing_rows(["A"] * frame_count), fps=30)
    lookup_nbytes = sum(
        tensor.nelement() * tensor.element_size()
        for tensor in (scan.elapsed_seconds, scan.valid, scan.segment_indices)
    )

    assert lookup_nbytes == frame_count * (4 + 1 + 8)
    assert math.isclose(scan.elapsed_seconds[-1].item(), (frame_count - 1) / 30, rel_tol=1e-6)
