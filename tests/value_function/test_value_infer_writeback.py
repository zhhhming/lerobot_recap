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

from lerobot.scripts.lerobot_build_dataset import (
    BuildDatasetConfig,
    _load_extras_schema,
    build_dataset,
)
from lerobot.value_function.advantage import AdvantageConfig, compute_advantage
from lerobot.value_function.configuration import ValueFunctionConfig
from lerobot.value_function.dataset import (
    RawValueFrameDataset,
    ValueAugmentationConfig,
    training_data_contract,
)
from lerobot.value_function.inference import (
    ValueInferenceConfig,
    infer_value_function,
    monotonic_viterbi,
)
from lerobot.value_function.raw_io import read_value_function_metadata
from lerobot.value_function.schema import (
    EXTRAS_FILENAME,
    PREDICTION_SOURCE_MODEL,
    RAW_FORMAT_VERSION,
    TARGET_STAGE,
    VALUE_GLOBAL_REMAINING_FRAMES_PRED,
    VALUE_GLOBAL_REMAINING_NORM_PRED,
    VALUE_INFERENCE_STAGE_PREFIX,
    VALUE_SUBTASK_CONFIDENCE,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_ID_PRED,
    VALUE_SUBTASK_ID_PRED_SMOOTH,
    VALUE_SUBTASK_NAME_PRED_SMOOTH,
    VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD,
    VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD,
    VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD,
    VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD,
)
from lerobot.value_function.targets import ValueTargetConfig, prepare_value_targets


IMAGE_KEY = "observation.images.third_person"


class DeterministicInferenceModel(nn.Module):
    def __init__(self, config: ValueFunctionConfig, *, fail_after: int | None = None):
        super().__init__()
        self.config = config
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.fail_after = fail_after
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("deliberate inference failure")
        frame = batch[self.config.state_key][:, 0].float()
        parity = frame.long().remainder(2)
        logits = torch.stack(
            [torch.where(parity == 0, 5.0, 0.0), torch.where(parity == 1, 5.0, 0.0)],
            dim=-1,
        )
        heads = torch.stack([0.1 * (frame + 1), 0.5 + 0.1 * frame], dim=-1)
        return {
            "global_remaining_value": 0.1 * (frame + 1) + self.anchor * 0.0,
            "subtask_logits": logits + self.anchor * 0.0,
            "subtask_remaining_value": heads + self.anchor * 0.0,
        }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_raw_run(tmp_path: Path) -> Path:
    root = tmp_path / "raw_run"
    root.mkdir()
    _write_json(
        root / "run_meta.json",
        {
            "version": RAW_FORMAT_VERSION,
            "fps": 30,
            "task": "infer value",
            "robot_type": "test_robot",
            "features": {
                "action": {"dtype": "float32", "shape": [1], "names": ["a"]},
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["s0", "s1"],
                },
                IMAGE_KEY: {"dtype": "image", "shape": [12, 12, 3], "names": None},
            },
        },
    )
    _write_json(
        root / "annotation_config.json",
        {"subtasks": [{"name": "pick"}, {"name": "place"}]},
    )
    labels = ["pick", "pick", "place", "place"]
    for episode_index in range(2):
        episode = root / f"ep_{episode_index:06d}"
        episode.mkdir()
        _write_json(
            episode / "info.json",
            {"length": 4, "task": "infer value", "success": True},
        )
        pq.write_table(
            pa.Table.from_arrays(
                [
                    pa.array(range(4), type=pa.int64()),
                    pa.array(
                        [[float(frame), float(episode_index)] for frame in range(4)],
                        type=pa.list_(pa.float32(), 2),
                    ),
                    pa.array([[0.0]] * 4, type=pa.list_(pa.float32(), 1)),
                ],
                names=["frame_index", "observation.state", "action"],
            ),
            episode / "frames.parquet",
        )
        pq.write_table(
            pa.Table.from_arrays(
                [
                    pa.array(labels, type=pa.string()),
                    pa.array([0.0, 1.0, 0.0, 1.0], type=pa.float32()),
                    pa.array([f"keep-{episode_index}"] * 4, type=pa.string()),
                ],
                names=["subtask", "subtask_progress", "existing_column"],
            ),
            episode / EXTRAS_FILENAME,
        )
        camera = episode / "third_person"
        camera.mkdir()
        for frame in range(4):
            Image.fromarray(np.full((12, 12, 3), frame * 20, dtype=np.uint8)).save(
                camera / f"{frame:06d}.png"
            )
    prepare_value_targets(
        ValueTargetConfig(
            root=root,
            mode="both",
            num_bins=8,
            global_scale="manual",
            global_scale_frames=10.0,
            subtask_scale="manual",
            subtask_scale_frames={"pick": 2.0, "place": 4.0},
        )
    )
    return root


def _model_config() -> ValueFunctionConfig:
    return ValueFunctionConfig(
        mode="both",
        backbone_type="vision_only",
        image_keys=(IMAGE_KEY,),
        image_resolution=(12, 12),
        num_bins=8,
        num_subtasks=2,
        use_state=True,
        state_dim=2,
        state_hidden_dim=4,
        fusion_hidden_dim=8,
        head_hidden_dim=8,
        dropout=0.0,
    )


def _factory(config: ValueFunctionConfig) -> DeterministicInferenceModel:
    return DeterministicInferenceModel(config)


def _write_checkpoint(
    root: Path,
    path: Path,
    *,
    data_patch: dict | None = None,
) -> Path:
    config = _model_config()
    dataset = RawValueFrameDataset(
        [root],
        mode="both",
        image_keys=(IMAGE_KEY,),
        augmentation=ValueAugmentationConfig(enabled=False),
    )
    contract = training_data_contract(dataset)
    contract.update(data_patch or {})
    model = DeterministicInferenceModel(config)
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "data_contract": contract,
            "step": 12,
            "epoch": 3,
        },
        path,
    )
    return path


def _extras(root: Path, episode: int = 0) -> pa.Table:
    return pq.read_table(root / f"ep_{episode:06d}" / EXTRAS_FILENAME)


def test_monotonic_viterbi_enforces_complete_canonical_path():
    probabilities = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.80, 0.10, 0.10],
            [0.05, 0.10, 0.85],
        ],
        dtype=np.float64,
    )
    path = monotonic_viterbi(np.log(probabilities))

    assert path[0] == 0
    assert path[-1] == 2
    assert np.all(np.diff(path) >= 0)
    assert np.all(np.diff(path) <= 1)
    assert list(dict.fromkeys(path.tolist())) == [0, 1, 2]


def test_monotonic_viterbi_rejects_impossible_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="without skips"):
        monotonic_viterbi(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        monotonic_viterbi(np.asarray([[0.0, np.nan], [0.0, 0.0]]))


def test_both_mode_writes_atomic_paired_predictions_and_provenance(tmp_path):
    root = _make_raw_run(tmp_path)
    checkpoint = _write_checkpoint(root, tmp_path / "checkpoint.pt")

    summary = infer_value_function(
        ValueInferenceConfig(
            root=root,
            checkpoint=checkpoint,
            mode="both",
            batch_size=3,
            device="cpu",
        ),
        model_factory=_factory,
    )

    assert summary["prediction_source"] == PREDICTION_SOURCE_MODEL
    assert summary["synthetic"] is False
    assert summary["episodes"] == 2
    assert summary["frames"] == 8
    assert set(summary["stages"]) == {"global", "subtask"}

    expected_schema = _extras(root, 0).schema
    for episode in range(2):
        table = _extras(root, episode)
        assert table.schema == expected_schema
        assert table.column("existing_column").to_pylist() == [f"keep-{episode}"] * 4
        assert table.column(VALUE_GLOBAL_REMAINING_NORM_PRED).to_pylist() == pytest.approx(
            [0.1, 0.2, 0.3, 0.4]
        )
        assert table.column(VALUE_GLOBAL_REMAINING_FRAMES_PRED).to_pylist() == pytest.approx(
            [1.0, 2.0, 3.0, 4.0]
        )
        assert table.column(VALUE_SUBTASK_ID_PRED).to_pylist() == [0, 1, 0, 1]
        confidence = table.column(VALUE_SUBTASK_CONFIDENCE).to_pylist()
        assert all(0.99 < value <= 1.0 for value in confidence)
        smooth = table.column(VALUE_SUBTASK_ID_PRED_SMOOTH).to_pylist()
        assert smooth[0] == 0 and smooth[-1] == 1
        assert all(current >= previous for previous, current in zip(smooth, smooth[1:]))
        assert table.column(VALUE_SUBTASK_NAME_PRED_SMOOTH).to_pylist() == [
            ("pick" if index == 0 else "place") for index in smooth
        ]

        assert table.column(VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD).to_pylist() == (
            pytest.approx([0.1, 0.2, 0.7, 0.8])
        )
        assert table.column(VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD).to_pylist() == (
            pytest.approx([0.2, 0.4, 2.8, 3.2])
        )
        all_heads = np.asarray(
            [[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]], dtype=np.float32
        )
        smooth_norm = table.column(
            VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD
        ).to_pylist()
        smooth_frames = table.column(
            VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD
        ).to_pylist()
        assert smooth_norm == pytest.approx(
            [all_heads[frame, subtask] for frame, subtask in enumerate(smooth)]
        )
        assert smooth_frames == pytest.approx(
            [
                all_heads[frame, subtask] * (2.0 if subtask == 0 else 4.0)
                for frame, subtask in enumerate(smooth)
            ]
        )

    metadata = read_value_function_metadata(root)
    for mode in ("global", "subtask"):
        stage = metadata["stages"][f"{VALUE_INFERENCE_STAGE_PREFIX}.{mode}"]
        assert stage["prediction_source"] == PREDICTION_SOURCE_MODEL
        assert stage["synthetic"] is False
        assert stage["dependencies"][TARGET_STAGE]
        assert stage["config"]["checkpoint_sha256"] == summary["checkpoint_sha256"]
        assert stage["output_fingerprint"]
    subtask_outputs = set(
        metadata["stages"][f"{VALUE_INFERENCE_STAGE_PREFIX}.subtask"]["output_columns"]
    )
    assert VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD in subtask_outputs
    assert VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD in subtask_outputs


@pytest.mark.parametrize(
    ("path", "expected", "unexpected"),
    [
        (
            "gt_conditioned",
            VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD,
            VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD,
        ),
        (
            "pred_smooth",
            VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD,
            VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD,
        ),
    ],
)
def test_single_subtask_path_has_an_unambiguous_stage_manifest(
    tmp_path, path, expected, unexpected
):
    root = _make_raw_run(tmp_path)
    checkpoint = _write_checkpoint(root, tmp_path / "checkpoint.pt")
    infer_value_function(
        ValueInferenceConfig(
            root=root,
            checkpoint=checkpoint,
            mode="subtask",
            subtask_inference_path=path,
            device="cpu",
        ),
        model_factory=_factory,
    )

    stage = read_value_function_metadata(root)["stages"][
        f"{VALUE_INFERENCE_STAGE_PREFIX}.subtask"
    ]
    assert expected in stage["output_columns"]
    assert unexpected not in stage["output_columns"]
    assert (VALUE_SUBTASK_ID_GT in stage["input_columns"]) is (path == "gt_conditioned")


def test_checkpoint_contract_failure_and_model_failure_leave_extras_unchanged(tmp_path):
    root = _make_raw_run(tmp_path)
    before = {
        episode: (root / f"ep_{episode:06d}" / EXTRAS_FILENAME).read_bytes()
        for episode in range(2)
    }
    incompatible = _write_checkpoint(
        root,
        tmp_path / "incompatible.pt",
        data_patch={"global_scale_frames": 99.0},
    )
    with pytest.raises(ValueError, match="global scale mismatch"):
        infer_value_function(
            ValueInferenceConfig(root=root, checkpoint=incompatible, mode="global", device="cpu"),
            model_factory=_factory,
        )
    assert all(
        (root / f"ep_{episode:06d}" / EXTRAS_FILENAME).read_bytes() == contents
        for episode, contents in before.items()
    )

    checkpoint = _write_checkpoint(root, tmp_path / "failure.pt")

    def failing_factory(config):
        return DeterministicInferenceModel(config, fail_after=1)

    with pytest.raises(RuntimeError, match="deliberate inference failure"):
        infer_value_function(
            ValueInferenceConfig(
                root=root, checkpoint=checkpoint, mode="both", batch_size=2, device="cpu"
            ),
            model_factory=failing_factory,
        )
    assert all(
        (root / f"ep_{episode:06d}" / EXTRAS_FILENAME).read_bytes() == contents
        for episode, contents in before.items()
    )


@pytest.mark.parametrize(
    ("mode", "data_patch", "message"),
    [
        (
            "global",
            {
                "image_features": {
                    IMAGE_KEY: {"dtype": "image", "shape": [99, 99, 3], "names": None}
                }
            },
            "image schema mismatch",
        ),
        (
            "global",
            {
                "state_feature": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": ["s0", "s1", "s2"],
                }
            },
            "state schema mismatch",
        ),
        ("subtask", {"subtask_order": ["place", "pick"]}, "subtask order mismatch"),
        (
            "subtask",
            {"subtask_scale_frames": {"pick": 3.0, "place": 4.0}},
            "subtask scales mismatch",
        ),
    ],
)
def test_checkpoint_rejects_image_state_and_subtask_contract_mismatches(
    tmp_path, mode, data_patch, message
):
    root = _make_raw_run(tmp_path)
    checkpoint = _write_checkpoint(
        root, tmp_path / "incompatible.pt", data_patch=data_patch
    )

    with pytest.raises(ValueError, match=message):
        infer_value_function(
            ValueInferenceConfig(root=root, checkpoint=checkpoint, mode=mode, device="cpu"),
            model_factory=_factory,
        )
    assert VALUE_GLOBAL_REMAINING_NORM_PRED not in _extras(root).column_names
    assert VALUE_SUBTASK_ID_PRED not in _extras(root).column_names


def test_checkpoint_rejects_changed_target_stage_on_its_training_root(tmp_path):
    root = _make_raw_run(tmp_path)
    checkpoint = _write_checkpoint(root, tmp_path / "checkpoint.pt")
    prepare_value_targets(
        ValueTargetConfig(
            root=root,
            mode="both",
            num_bins=8,
            global_scale="manual",
            global_scale_frames=10.0,
            subtask_scale="manual",
            subtask_scale_frames={"pick": 2.0, "place": 4.0},
        )
    )

    with pytest.raises(ValueError, match="target-stage fingerprint"):
        infer_value_function(
            ValueInferenceConfig(root=root, checkpoint=checkpoint, mode="both", device="cpu"),
            model_factory=_factory,
        )


def test_prediction_columns_build_and_feed_both_advantage_paths(tmp_path):
    root = _make_raw_run(tmp_path)
    checkpoint = _write_checkpoint(root, tmp_path / "checkpoint.pt")
    infer_value_function(
        ValueInferenceConfig(root=root, checkpoint=checkpoint, mode="both", device="cpu"),
        model_factory=_factory,
    )

    features, columns = _load_extras_schema(
        [root / "ep_000000", root / "ep_000001"]
    )
    assert VALUE_GLOBAL_REMAINING_FRAMES_PRED in columns
    assert VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD in columns
    assert features[VALUE_SUBTASK_ID_PRED_SMOOTH]["dtype"] == "int32"
    assert features[VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD]["dtype"] == "float32"
    assert (
        build_dataset(
            BuildDatasetConfig(
                runs=[str(root)],
                output_repo_id="test/value-inference",
                video=False,
                push_to_hub=False,
                dry_run=True,
            )
        )
        is None
    )

    gt_summary = compute_advantage(
        AdvantageConfig(
            root=root,
            value_mode="subtask",
            value_source="model_pred",
            subtask_inference_path="gt_conditioned",
            chunk_size=1,
        )
    )
    assert gt_summary["value_column"] == VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD
    # Rerunning inference makes the downstream gt-conditioned advantage stale and restores
    # the model prediction stage before exercising the alternative paired path.
    infer_value_function(
        ValueInferenceConfig(root=root, checkpoint=checkpoint, mode="both", device="cpu"),
        model_factory=_factory,
    )
    smooth_summary = compute_advantage(
        AdvantageConfig(
            root=root,
            value_mode="subtask",
            value_source="model_pred",
            subtask_inference_path="pred_smooth",
            chunk_size=1,
        )
    )
    assert (
        smooth_summary["value_column"]
        == VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD
    )
