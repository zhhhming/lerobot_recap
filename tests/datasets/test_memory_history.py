#!/usr/bin/env python

"""Contracts for the dynamic history dataset used by memory conditioning."""

from types import SimpleNamespace

import pytest
import torch


class _FakeDataset:
    def __init__(self, rows, *, features=None):
        self.rows = rows
        feature_names = features if features is not None else rows[0]
        self.meta = SimpleNamespace(features={key: {} for key in feature_names})
        self.episodes = None
        self.item_calls = []
        self.raw_item_calls = []
        self.fps = 30
        self.repo_id = "fake/repo"
        self.root = "/fake/root"

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


def _rows(length, *, episode_index=0, start_index=0):
    return [
        {
            "index": start_index + frame_index,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "subtask": f"subtask-{episode_index}-{frame_index}",
            "subtask_progress": torch.tensor(
                frame_index / max(1, length - 1), dtype=torch.float32
            ),
        }
        for frame_index in range(length)
    ]


def _fixed_randint(value, calls=None):
    calls = [] if calls is None else calls

    def randint(low, high, size, **kwargs):
        calls.append((low, high, size, kwargs))
        return torch.tensor([value], dtype=torch.int64)

    return randint


@pytest.mark.parametrize(("frame_index", "offset"), [(0, 1), (4, 9)])
def test_history_samples_offset_before_episode_start_check(monkeypatch, frame_index, offset):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    randint_calls = []
    monkeypatch.setattr(torch, "randint", _fixed_randint(offset, randint_calls))
    dataset = _FakeDataset(_rows(13))
    wrapped = MemoryHistoryDataset(dataset, lookback_min_frames=1, lookback_max_frames=12)

    item = wrapped[frame_index]

    assert randint_calls and randint_calls[0][:3] == (1, 13, (1,))
    assert item["memory_frame_offset"] == offset
    assert item["memory_valid"] is False
    assert item["memory_subtask"] == ""
    assert item["memory_subtask_progress"].dtype == torch.float32
    assert item["memory_subtask_progress"].item() == 0.0
    assert dataset.raw_item_calls == []


def test_history_offset_twelve_reads_frame_zero_without_video_decode(monkeypatch):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    monkeypatch.setattr(torch, "randint", _fixed_randint(12))
    dataset = _FakeDataset(_rows(13))
    wrapped = MemoryHistoryDataset(dataset, lookback_min_frames=1, lookback_max_frames=12)

    item = wrapped[12]

    assert item["memory_frame_offset"] == 12
    assert item["memory_valid"] is True
    assert item["memory_subtask"] == "subtask-0-0"
    assert item["memory_subtask_progress"].dtype == torch.float32
    assert item["memory_subtask_progress"].item() == pytest.approx(0.0)
    assert dataset.item_calls == [12]
    assert dataset.raw_item_calls == [0]


def test_first_frame_of_later_episode_never_leaks_previous_episode(monkeypatch):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    monkeypatch.setattr(torch, "randint", _fixed_randint(1))
    rows = _rows(3, episode_index=0) + _rows(3, episode_index=1, start_index=20)
    dataset = _FakeDataset(rows)
    wrapped = MemoryHistoryDataset(dataset, lookback_min_frames=1, lookback_max_frames=12)

    item = wrapped[3]

    assert item["episode_index"] == 1
    assert item["frame_index"] == 0
    assert item["memory_frame_offset"] == 1
    assert item["memory_valid"] is False
    assert dataset.raw_item_calls == []


def test_selected_episode_relative_indices_stay_within_episode(monkeypatch):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    monkeypatch.setattr(torch, "randint", _fixed_randint(2))
    rows = _rows(4, episode_index=2, start_index=100) + _rows(
        5, episode_index=7, start_index=500
    )
    dataset = _FakeDataset(rows)
    wrapped = MemoryHistoryDataset(dataset, lookback_min_frames=1, lookback_max_frames=12)

    item = wrapped[7]

    assert item["episode_index"] == 7
    assert item["frame_index"] == 3
    assert item["memory_subtask"] == "subtask-7-1"
    assert dataset.raw_item_calls == [5]


@pytest.mark.parametrize(
    "invalid_progress",
    [float("nan"), float("inf"), torch.tensor([0.1, 0.2]), "not-a-number"],
)
def test_invalid_history_progress_produces_no_memory(monkeypatch, invalid_progress):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    monkeypatch.setattr(torch, "randint", _fixed_randint(1))
    rows = _rows(3)
    rows[1]["subtask_progress"] = invalid_progress
    dataset = _FakeDataset(rows)
    item = MemoryHistoryDataset(dataset)[2]

    assert item["memory_valid"] is False
    assert item["memory_subtask"] == ""
    assert item["memory_subtask_progress"].item() == 0.0
    assert dataset.raw_item_calls == [1]


@pytest.mark.parametrize("subtask", ["", "   ", None, 123])
def test_empty_or_non_string_history_subtask_produces_no_memory(monkeypatch, subtask):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    monkeypatch.setattr(torch, "randint", _fixed_randint(1))
    rows = _rows(3)
    rows[1]["subtask"] = subtask
    item = MemoryHistoryDataset(_FakeDataset(rows))[2]

    assert item["memory_valid"] is False
    assert item["memory_subtask"] == ""


def test_wrapper_preserves_current_item_and_proxies_training_attributes(monkeypatch):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    monkeypatch.setattr(torch, "randint", _fixed_randint(1))
    dataset = _FakeDataset(_rows(4))
    wrapped = MemoryHistoryDataset(dataset)

    item = wrapped[3]

    assert item["subtask"] == "subtask-0-3"
    assert item["subtask_progress"].item() == pytest.approx(1.0)
    assert wrapped.dataset is dataset
    assert wrapped.meta is dataset.meta
    assert wrapped.episodes is dataset.episodes
    assert wrapped.num_frames == dataset.num_frames
    assert wrapped.num_episodes == dataset.num_episodes
    assert wrapped.features is dataset.features
    assert wrapped.fps == dataset.fps
    assert wrapped.repo_id == dataset.repo_id
    assert wrapped.root == dataset.root
    assert len(wrapped) == len(dataset)


@pytest.mark.parametrize(
    ("label_key", "weight_key"),
    [
        ("advantage_label_global", "advantage_loss_weight_global"),
        ("advantage_label_subtask", "advantage_loss_weight_subtask"),
    ],
)
def test_history_reads_only_subtask_progress_and_preserves_current_advantage(
    monkeypatch, label_key, weight_key
):
    """A conflicting history label/weight must never become the current training weight."""

    from lerobot.datasets.memory_history import MemoryHistoryDataset

    monkeypatch.setattr(torch, "randint", _fixed_randint(1))
    rows = _rows(3)
    rows[1][label_key] = "negative"
    rows[1][weight_key] = torch.tensor(1.0)
    rows[2][label_key] = "positive"
    rows[2][weight_key] = torch.tensor(2.0)
    dataset = _FakeDataset(rows)

    item = MemoryHistoryDataset(dataset)[2]

    assert item["memory_subtask"] == rows[1]["subtask"]
    assert item["memory_subtask_progress"].item() == pytest.approx(
        rows[1]["subtask_progress"].item()
    )
    assert item[label_key] == "positive"
    assert item[weight_key].item() == pytest.approx(2.0)
    assert f"memory_{label_key}" not in item
    assert f"memory_{weight_key}" not in item


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(0, 12), (4, 3), (1.5, 12), (True, 12)],
)
def test_wrapper_rejects_invalid_lookback_range(minimum, maximum):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    with pytest.raises(ValueError, match="lookback"):
        MemoryHistoryDataset(
            _FakeDataset(_rows(3)),
            lookback_min_frames=minimum,
            lookback_max_frames=maximum,
        )


def test_wrapper_rejects_missing_required_features():
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    dataset = _FakeDataset(_rows(3), features={"index", "episode_index", "frame_index"})

    with pytest.raises(ValueError, match="subtask.*subtask_progress"):
        MemoryHistoryDataset(dataset)


def _collect_worker_seeded_offsets(seed):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    # DataLoader seeds the default PyTorch RNG independently in every worker.
    # Calling torch.randint without a private generator therefore picks up the
    # worker seed while remaining reproducible for the same worker setup.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        wrapped = MemoryHistoryDataset(_FakeDataset(_rows(48)))
        return [wrapped[index]["memory_frame_offset"] for index in range(20, 48)]


def test_worker_rng_is_bounded_varied_and_reproducible():
    worker_zero_first = _collect_worker_seeded_offsets(1234)
    worker_zero_second = _collect_worker_seeded_offsets(1234)
    worker_one = _collect_worker_seeded_offsets(1235)

    assert worker_zero_first == worker_zero_second
    assert worker_zero_first != worker_one
    assert all(1 <= offset <= 12 for offset in worker_zero_first + worker_one)
    assert len(set(worker_zero_first)) > 1


def _collect_dataloader_offsets(seed):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    loader = torch.utils.data.DataLoader(
        MemoryHistoryDataset(_FakeDataset(_rows(32))),
        batch_size=1,
        num_workers=2,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
        multiprocessing_context="spawn",
    )
    return [int(batch["memory_frame_offset"].item()) for batch in loader]


def test_two_dataloader_workers_have_reproducible_distinct_rng_sequences():
    first = _collect_dataloader_offsets(2026)
    second = _collect_dataloader_offsets(2026)

    assert first == second
    assert all(1 <= offset <= 12 for offset in first)
    assert first[::2] != first[1::2]


def _factory_cfg(*, enabled, policy_type="pi0", streaming=False, minimum=1, maximum=12):
    return SimpleNamespace(
        policy=SimpleNamespace(
            type=policy_type,
            use_memory_conditioning=enabled,
            observation_delta_indices=None,
            action_delta_indices=None,
            reward_delta_indices=None,
        ),
        dataset=SimpleNamespace(
            repo_id="fake/repo",
            root=None,
            revision=None,
            episodes=None,
            streaming=streaming,
            image_transforms=SimpleNamespace(enable=False),
            video_backend="pyav",
            use_imagenet_stats=False,
        ),
        memory_lookback_min_frames=minimum,
        memory_lookback_max_frames=maximum,
        tolerance_s=1e-4,
        num_workers=2,
    )


def _patch_factory_dataset(monkeypatch, *, features=None):
    from lerobot.datasets import factory

    rows = _rows(13)
    dataset = _FakeDataset(rows, features=features)
    metadata = SimpleNamespace(
        features=dataset.meta.features,
        camera_keys=[],
        stats={},
        fps=30,
    )
    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(factory, "LeRobotDataset", lambda *args, **kwargs: dataset)
    return factory, dataset


@pytest.mark.parametrize("policy_type", ["pi0", "pi05"])
def test_factory_wraps_only_enabled_pi_memory_dataset(monkeypatch, policy_type):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    factory, dataset = _patch_factory_dataset(monkeypatch)
    cfg = _factory_cfg(enabled=True, policy_type=policy_type, minimum=2, maximum=8)

    result = factory.make_dataset(cfg)

    assert isinstance(result, MemoryHistoryDataset)
    assert result.dataset is dataset
    assert result.lookback_min_frames == 2
    assert result.lookback_max_frames == 8


def test_factory_memory_disabled_returns_original_dataset_unchanged(monkeypatch):
    factory, dataset = _patch_factory_dataset(monkeypatch)

    result = factory.make_dataset(_factory_cfg(enabled=False))
    item = result[12]

    assert result is dataset
    assert not any(key.startswith("memory_") for key in item)
    assert dataset.item_calls == [12]
    assert dataset.raw_item_calls == []


def test_factory_uses_safe_lookback_defaults_before_train_config_fields_exist(monkeypatch):
    from lerobot.datasets.memory_history import MemoryHistoryDataset

    factory, _ = _patch_factory_dataset(monkeypatch)
    cfg = _factory_cfg(enabled=True)
    del cfg.memory_lookback_min_frames
    del cfg.memory_lookback_max_frames

    result = factory.make_dataset(cfg)

    assert isinstance(result, MemoryHistoryDataset)
    assert result.lookback_min_frames == 1
    assert result.lookback_max_frames == 12


def test_factory_rejects_streaming_before_constructing_metadata(monkeypatch):
    from lerobot.datasets import factory

    def unexpected_metadata(*args, **kwargs):
        raise AssertionError("metadata must not be constructed for unsupported streaming memory")

    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", unexpected_metadata)

    with pytest.raises(ValueError, match="streaming=false"):
        factory.make_dataset(_factory_cfg(enabled=True, streaming=True))


def test_factory_rejects_missing_features_before_constructing_dataset(monkeypatch):
    factory, _ = _patch_factory_dataset(
        monkeypatch, features={"index", "episode_index", "frame_index"}
    )

    def unexpected_dataset(*args, **kwargs):
        raise AssertionError("dataset must not be constructed when required fields are missing")

    monkeypatch.setattr(factory, "LeRobotDataset", unexpected_dataset)

    with pytest.raises(ValueError, match="subtask.*subtask_progress"):
        factory.make_dataset(_factory_cfg(enabled=True))


def test_factory_rejects_memory_for_unsupported_policy_before_loading_data():
    from lerobot.datasets import factory

    with pytest.raises(ValueError, match="pi0.*pi05"):
        factory.make_dataset(_factory_cfg(enabled=True, policy_type="act"))
