from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from lerobot.value_function.configuration import ValueFunctionConfig
from lerobot.value_function.dataset import RawValueFrameDataset, ValueAugmentationConfig
from lerobot.value_function.modeling_pi0_value import PI0ValueFunctionModel
from lerobot.value_function.targets import ValueTargetConfig, prepare_value_targets


RUN_REAL_SMOKE = os.environ.get("LEROBOT_RUN_REAL_VALUE_TRAIN_DATA_SMOKE") == "1"


def _shadow_one_episode(source: Path, destination: Path) -> Path:
    destination.mkdir()
    for filename in ("run_meta.json", "annotation_config.json"):
        shutil.copy2(source / filename, destination / filename)
    source_episode = source / "ep_000000"
    target_episode = destination / "ep_000000"
    target_episode.mkdir()
    for filename in ("info.json", "frames.parquet", "extras.parquet"):
        shutil.copy2(source_episode / filename, target_episode / filename)
    for camera in ("left_wrist", "right_wrist", "third_person"):
        (target_episode / camera).symlink_to(source_episode / camera, target_is_directory=True)
    return destination


@pytest.mark.skipif(
    not RUN_REAL_SMOKE,
    reason="Set LEROBOT_RUN_REAL_VALUE_TRAIN_DATA_SMOKE=1 for the real raw/checkpoint smoke",
)
def test_real_raw_dataset_builds_and_feeds_pi0_value_forward(tmp_path):
    raw_value = os.environ.get("LEROBOT_RAW_RUN")
    checkpoint_value = os.environ.get("LEROBOT_PI0_CHECKPOINT")
    if not raw_value or not checkpoint_value:
        pytest.fail("LEROBOT_RAW_RUN and LEROBOT_PI0_CHECKPOINT are required")
    raw_root = Path(raw_value).expanduser()
    checkpoint = Path(checkpoint_value).expanduser()
    if not raw_root.is_dir() or not checkpoint.is_file():
        pytest.fail(f"Missing real smoke input: raw={raw_root}, checkpoint={checkpoint}")

    shadow = _shadow_one_episode(raw_root, tmp_path / "shadow_raw")
    summary = prepare_value_targets(
        ValueTargetConfig(
            root=shadow,
            mode="both",
            num_bins=8,
            global_scale="max",
            subtask_scale="max",
        )
    )
    dataset = RawValueFrameDataset(
        [shadow],
        mode="subtask",
        image_keys=("observation.images.third_person",),
        use_state=True,
        augmentation=ValueAugmentationConfig(enabled=False),
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    config = ValueFunctionConfig(
        mode="subtask",
        backbone_type="pi0",
        pretrained_path=str(checkpoint),
        image_keys=("observation.images.third_person",),
        num_vlm_layers=1,
        num_bins=8,
        num_subtasks=len(summary["subtask_order"]),
        use_state=True,
        state_dim=dataset.state_dim,
        state_hidden_dim=8,
        fusion_hidden_dim=32,
        head_hidden_dim=16,
        freeze_backbone=True,
        dropout=0.0,
    )
    model = PI0ValueFunctionModel(config).eval()
    with torch.inference_mode():
        outputs = model(batch)

    assert outputs["subtask_logits"].shape == (1, len(summary["subtask_order"]))
    assert outputs["subtask_remaining_logits"].shape == (
        1,
        len(summary["subtask_order"]),
        8,
    )
    assert torch.isfinite(outputs["subtask_remaining_logits"]).all()
