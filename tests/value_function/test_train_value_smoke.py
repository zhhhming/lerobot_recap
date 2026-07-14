from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from torch import nn

from lerobot.value_function.configuration import ValueFunctionConfig
from lerobot.value_function.dataset import RawValueFrameDataset, ValueAugmentationConfig
from lerobot.value_function.modeling_pi0_value import PI0ValueFunctionModel
from lerobot.value_function.schema import EXTRAS_FILENAME, RAW_FORMAT_VERSION
from lerobot.value_function.targets import ValueTargetConfig, prepare_value_targets
from lerobot.value_function.training import (
    MetricAccumulator,
    ValueTrainingConfig,
    load_value_function_checkpoint,
    train_value_function,
)


class TinyBackbone(nn.Module):
    output_dim = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, self.output_dim)

    def forward(self, batch):
        image = batch["observation.images.third_person"]
        return self.projection(image.mean(dim=(-2, -1)))


def _model_factory(config: ValueFunctionConfig) -> PI0ValueFunctionModel:
    return PI0ValueFunctionModel(config, backbone=TinyBackbone())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_training_run(tmp_path: Path) -> Path:
    root = tmp_path / "training_run"
    root.mkdir()
    _write_json(
        root / "run_meta.json",
        {
            "version": RAW_FORMAT_VERSION,
            "fps": 30,
            "task": "train value",
            "robot_type": "test_robot",
            "features": {
                "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["s0", "s1"],
                },
                "observation.images.third_person": {
                    "dtype": "image",
                    "shape": [12, 12, 3],
                    "names": None,
                },
            },
        },
    )
    _write_json(
        root / "annotation_config.json",
        {"subtasks": [{"name": "pick"}, {"name": "place"}]},
    )
    labels = ["pick", "pick", "place", "place"]
    for episode_index in range(3):
        episode = root / f"ep_{episode_index:06d}"
        episode.mkdir()
        _write_json(
            episode / "info.json",
            {"length": 4, "task": "train value", "success": True},
        )
        state = [[episode_index + frame, episode_index + frame + 0.5] for frame in range(4)]
        pq.write_table(
            pa.Table.from_arrays(
                [
                    pa.array(range(4), type=pa.int64()),
                    pa.array(state, type=pa.list_(pa.float32(), 2)),
                    pa.array([[0.0, 0.0]] * 4, type=pa.list_(pa.float32(), 2)),
                ],
                names=["frame_index", "observation.state", "action"],
            ),
            episode / "frames.parquet",
        )
        pq.write_table(
            pa.Table.from_arrays(
                [
                    pa.array(labels),
                    pa.array([0.0, 1.0, 0.0, 1.0], type=pa.float32()),
                ],
                names=["subtask", "subtask_progress"],
            ),
            episode / EXTRAS_FILENAME,
        )
        camera = episode / "third_person"
        camera.mkdir()
        for frame in range(4):
            pixels = np.full((12, 12, 3), 20 * episode_index + 5 * frame, dtype=np.uint8)
            Image.fromarray(pixels).save(camera / f"{frame:06d}.png")
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


def _training_config(root: Path, output: Path) -> ValueTrainingConfig:
    return ValueTrainingConfig(
        roots=(str(root),),
        output_dir=str(output),
        model=ValueFunctionConfig(
            mode="both",
            backbone_type="vision_only",
            image_keys=("observation.images.third_person",),
            image_resolution=(12, 12),
            num_bins=8,
            num_subtasks=1,
            use_elapsed_aux=True,
            elapsed_loss_weight=0.25,
            state_dim=2,
            state_hidden_dim=4,
            fusion_hidden_dim=8,
            head_hidden_dim=8,
            dropout=0.0,
        ),
        val_episodes=("2",),
        epochs=3,
        max_steps=2,
        batch_size=2,
        learning_rate=1e-3,
        device="cpu",
        augmentation=ValueAugmentationConfig(enabled=False),
    )


def test_two_step_training_writes_complete_artifacts_and_metrics(tmp_path):
    root = _make_training_run(tmp_path)
    output = tmp_path / "output"

    summary = train_value_function(_training_config(root, output), model_factory=_model_factory)

    assert summary["steps"] == 2
    assert summary["train_frames"] == 8
    assert summary["val_frames"] == 4
    for filename in (
        "checkpoint.pt",
        "config.json",
        "value_function_meta.json",
        "train_metrics.jsonl",
    ):
        assert (output / filename).is_file()
    metrics_lines = (output / "train_metrics.jsonl").read_text().splitlines()
    assert len(metrics_lines) == 1
    metrics = json.loads(metrics_lines[0])
    assert metrics["train"]["samples"] == 4
    assert metrics["val"]["samples"] == 4
    assert "global" in metrics["val"]["frame_mae"]
    assert "subtask" in metrics["val"]["frame_mae"]
    assert "subtask_accuracy" in metrics["val"]
    assert "monotonic_violation_rate" in metrics["val"]
    assert "subtask:pick" in metrics["val"]["clip_rate"]


def test_checkpoint_reload_preserves_config_state_stats_and_output_shapes(tmp_path):
    root = _make_training_run(tmp_path)
    output = tmp_path / "output"
    train_value_function(_training_config(root, output), model_factory=_model_factory)

    loaded, payload = load_value_function_checkpoint(
        output / "checkpoint.pt", model_factory=_model_factory
    )
    dataset = RawValueFrameDataset(
        [root],
        mode="both",
        image_keys=("observation.images.third_person",),
        use_elapsed_aux=True,
        augmentation=ValueAugmentationConfig(enabled=False),
    )
    sample = dataset[0]
    batch = {
        key: value.unsqueeze(0)
        for key, value in sample.items()
        if isinstance(value, torch.Tensor)
    }
    loaded.eval()
    with torch.inference_mode():
        outputs = loaded(batch)

    assert payload["step"] == 2
    assert payload["model_config"]["num_subtasks"] == 2
    assert outputs["global_remaining_logits"].shape == (1, 8)
    assert outputs["subtask_remaining_logits"].shape == (1, 2, 8)
    torch.testing.assert_close(loaded.state_mean.cpu(), torch.tensor([2.0, 2.5]))


def test_training_rejects_model_bins_that_do_not_match_target_metadata(tmp_path):
    root = _make_training_run(tmp_path)
    config = _training_config(root, tmp_path / "output")
    config.model.num_bins = 16

    with pytest.raises(ValueError, match="does not match prepared target metadata"):
        train_value_function(config, model_factory=_model_factory)


def test_frame_mae_uses_unclipped_frame_gt_for_p95_tail_samples():
    accumulator = MetricAccumulator("global", use_elapsed_aux=False, progress_bins=10)
    accumulator.update(
        {"features": torch.zeros(1, 2), "global_remaining_value": torch.tensor([1.0])},
        {"loss": torch.tensor(0.0)},
        {
            "value_root_index": torch.tensor([0]),
            "value_episode_index": torch.tensor([0]),
            "value_frame_index": torch.tensor([0]),
            "value_subtask_progress": torch.tensor([0.0]),
            "value_global_scale_frames": torch.tensor([5.0]),
            "value_global_remaining_norm_gt": torch.tensor([1.0]),
            "value_global_remaining_frames_gt": torch.tensor([6.0]),
            "value_global_remaining_norm_gt_is_clipped": torch.tensor([True]),
        },
    )

    metrics = accumulator.finalize()
    assert metrics["normalized_mae"]["global"] == 0.0
    assert metrics["frame_mae"]["global"] == 1.0
