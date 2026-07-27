from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from lerobot.value_function.dataset import (
    RawValueFrameDataset,
    ValueAugmentationConfig,
    compute_state_statistics,
    parse_val_episode_keys,
    split_episode_indices,
)
from lerobot.value_function.schema import EXTRAS_FILENAME, RAW_FORMAT_VERSION
from lerobot.value_function.targets import ValueTargetConfig, prepare_value_targets


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_training_run(
    tmp_path: Path,
    name: str = "raw_run",
    *,
    fps: int = 30,
    image_keys: tuple[str, ...] = ("observation.images.third_person",),
    image_format: str = "png",
) -> Path:
    root = tmp_path / name
    root.mkdir()
    features = {
        "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "observation.state": {
            "dtype": "float32",
            "shape": [2],
            "names": ["s0", "s1"],
        },
    }
    for key in image_keys:
        features[key] = {"dtype": "image", "shape": [16, 16, 3], "names": None}
    run_meta = {
        "version": RAW_FORMAT_VERSION,
        "fps": fps,
        "task": "test task",
        "robot_type": "test_robot",
        "features": features,
    }
    if image_format == "jpeg":
        run_meta["image_encoding"] = {
            "format": "jpeg",
            "extension": ".jpg",
            "quality": 95,
            "subsampling": 0,
        }
    _write_json(root / "run_meta.json", run_meta)
    _write_json(
        root / "annotation_config.json",
        {"subtasks": [{"name": "pick"}, {"name": "place"}]},
    )
    labels = ["pick", "pick", "place", "place"]
    progress = [0.0, 1.0, 0.0, 1.0]
    for episode_index in range(2):
        episode = root / f"ep_{episode_index:06d}"
        episode.mkdir()
        _write_json(
            episode / "info.json",
            {"length": len(labels), "task": "test task", "success": True},
        )
        states = [
            [float(episode_index * 10 + frame), float(episode_index * 10 + frame + 1)]
            for frame in range(len(labels))
        ]
        pq.write_table(
            pa.Table.from_arrays(
                [
                    pa.array(range(len(labels)), type=pa.int64()),
                    pa.array(states, type=pa.list_(pa.float32(), 2)),
                    pa.array([[0.0, 0.0]] * len(labels), type=pa.list_(pa.float32(), 2)),
                ],
                names=["frame_index", "observation.state", "action"],
            ),
            episode / "frames.parquet",
        )
        pq.write_table(
            pa.Table.from_arrays(
                [pa.array(labels), pa.array(progress, type=pa.float32())],
                names=["subtask", "subtask_progress"],
            ),
            episode / EXTRAS_FILENAME,
        )
        for key in image_keys:
            camera = episode / key.split(".")[-1]
            camera.mkdir()
            for frame in range(len(labels)):
                pixels = np.full((16, 16, 3), episode_index * 30 + frame * 10, dtype=np.uint8)
                extension = ".jpg" if image_format == "jpeg" else ".png"
                save_kwargs = {"quality": 95, "subsampling": 0} if image_format == "jpeg" else {}
                Image.fromarray(pixels).save(camera / f"{frame:06d}{extension}", **save_kwargs)
    prepare_value_targets(
        ValueTargetConfig(
            root=root,
            mode="both",
            num_bins=8,
            global_scale="max",
            subtask_scale="max",
            elapsed_aux=True,
        )
    )
    return root


def test_raw_value_dataset_loads_images_state_targets_and_metadata(tmp_path):
    root = _make_training_run(tmp_path)
    dataset = RawValueFrameDataset(
        [root],
        mode="both",
        image_keys=("observation.images.third_person",),
        use_elapsed_aux=True,
        augmentation=ValueAugmentationConfig(enabled=False),
    )

    sample = dataset[0]

    assert len(dataset) == 8
    assert sample["observation.images.third_person"].shape == (3, 16, 16)
    assert sample["observation.images.third_person"].dtype == torch.float32
    assert sample["observation.state"].tolist() == [0.0, 1.0]
    assert sample["value_global_remaining_norm_gt"].item() == 1.0
    assert sample["value_subtask_id_gt"].item() == 0
    assert sample["value_root_index"].item() == 0
    assert sample["value_episode_index"].item() == 0
    assert dataset.subtask_order == ("pick", "place")


def test_raw_value_dataset_loads_jpeg_backed_run(tmp_path):
    root = _make_training_run(tmp_path, image_format="jpeg")
    dataset = RawValueFrameDataset(
        [root],
        mode="global",
        image_keys=("observation.images.third_person",),
        augmentation=ValueAugmentationConfig(enabled=False),
    )

    sample = dataset[0]

    assert sample["observation.images.third_person"].shape == (3, 16, 16)
    assert dataset.contracts[0].image_encoding.extension == ".jpg"


def test_episode_split_has_no_frame_leakage_and_state_stats_use_train_only(tmp_path):
    root = _make_training_run(tmp_path)
    dataset = RawValueFrameDataset(
        [root],
        mode="subtask",
        image_keys=("observation.images.third_person",),
        augmentation=ValueAugmentationConfig(enabled=False),
    )
    train, val, train_episodes, val_episodes = split_episode_indices(dataset, val_episodes={(0, 1)})
    mean, std = compute_state_statistics(dataset, train)

    assert train_episodes == [(0, 0)]
    assert val_episodes == [(0, 1)]
    assert set(train).isdisjoint(val)
    torch.testing.assert_close(mean, torch.tensor([1.5, 2.5]))
    torch.testing.assert_close(std, torch.tensor([1.118034, 1.118034]))


def test_multi_root_contract_accepts_equal_runs_and_rejects_fps_mismatch(tmp_path):
    first = _make_training_run(tmp_path, "first")
    second = _make_training_run(tmp_path, "second")
    dataset = RawValueFrameDataset(
        [first, second],
        mode="global",
        image_keys=("observation.images.third_person",),
        augmentation=ValueAugmentationConfig(enabled=False),
    )
    assert len(dataset) == 16
    assert parse_val_episode_keys(["1:0"], num_roots=2) == {(1, 0)}

    incompatible = _make_training_run(tmp_path, "incompatible", fps=20)
    with pytest.raises(ValueError, match="Incompatible fps"):
        RawValueFrameDataset(
            [first, incompatible],
            mode="global",
            image_keys=("observation.images.third_person",),
            augmentation=ValueAugmentationConfig(enabled=False),
        )


def test_joint_augmentation_keeps_identical_camera_views_identical(tmp_path):
    keys = ("observation.images.left_wrist", "observation.images.third_person")
    root = _make_training_run(tmp_path, image_keys=keys)
    # The fixture writes identical pixels to both cameras.
    dataset = RawValueFrameDataset(
        [root],
        mode="global",
        image_keys=keys,
        augmentation=ValueAugmentationConfig(enabled=True, noise_std=0.0),
    )
    torch.manual_seed(7)
    sample = dataset.get_item(1, augment=True)
    torch.testing.assert_close(sample[keys[0]], sample[keys[1]])


def test_missing_or_stale_target_metadata_is_rejected(tmp_path):
    root = _make_training_run(tmp_path)
    (root / "value_function_meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="value_function_meta"):
        RawValueFrameDataset(
            [root],
            mode="global",
            image_keys=("observation.images.third_person",),
        )


def test_inactive_leftover_target_columns_are_not_accepted(tmp_path):
    root = _make_training_run(tmp_path)
    # A global-only rerun leaves old subtask columns in extras by design, but the active
    # target-stage manifest must prevent training from consuming them.
    prepare_value_targets(ValueTargetConfig(root=root, mode="global", num_bins=8, global_scale="max"))

    with pytest.raises(ValueError, match="current target stage does not provide"):
        RawValueFrameDataset(
            [root],
            mode="subtask",
            image_keys=("observation.images.third_person",),
            augmentation=ValueAugmentationConfig(enabled=False),
        )
