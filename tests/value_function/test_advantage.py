import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.scripts.lerobot_build_dataset import (
    BuildDatasetConfig,
    _load_extras_schema,
    build_dataset,
)
from lerobot.scripts.lerobot_compute_advantage import main as compute_advantage_main
from lerobot.value_function.advantage import (
    AdvantageConfig,
    compute_advantage,
    compute_global_advantage_columns,
    compute_subtask_advantage_columns,
)
from lerobot.value_function.mock_predictions import (
    MockPredictionConfig,
    generate_mock_predictions,
)
from lerobot.value_function.raw_io import (
    StalePipelineArtifactError,
    fingerprint_raw_run_columns,
    merge_episode_extras,
    merge_raw_run_extras,
    update_stage_metadata,
)
from lerobot.value_function.schema import (
    ADVANTAGE_GLOBAL_CHUNK,
    ADVANTAGE_GLOBAL_IS_VALID,
    ADVANTAGE_GLOBAL_VALID_HORIZON,
    ADVANTAGE_STAGE_PREFIX,
    ADVANTAGE_SUBTASK_BOUNDARY_PROGRESS,
    ADVANTAGE_SUBTASK_CHUNK,
    ADVANTAGE_SUBTASK_IS_VALID,
    ADVANTAGE_SUBTASK_NUM_CROSSINGS,
    ADVANTAGE_SUBTASK_VALID_HORIZON,
    ADVANTAGE_SUBTASK_WITHIN_SUBTASK_HORIZON,
    EXTRAS_FILENAME,
    MOCK_PREDICTIONS_STAGE,
    PREDICTION_SOURCE_MODEL,
    RAW_FORMAT_VERSION,
    VALUE_FUNCTION_META_FILENAME,
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_ID_PRED_SMOOTH,
    VALUE_SUBTASK_REMAINING_FRAMES_GT,
    VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD,
    VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD,
    VALUE_INFERENCE_STAGE_PREFIX,
)
from lerobot.value_function.targets import ValueTargetConfig, prepare_value_targets


def _write_json(path, payload):
    path.write_text(json.dumps(payload))


def _make_target_run(tmp_path, labels=None, *, name="raw_run"):
    labels = labels or ["pick"] * 3 + ["place"] * 3
    root = tmp_path / name
    root.mkdir()
    _write_json(
        root / "run_meta.json",
        {
            "version": RAW_FORMAT_VERSION,
            "fps": 30,
            "task": "test task",
            "robot_type": "test_robot",
            "features": {"action": {"dtype": "float32", "shape": [1], "names": ["a"]}},
        },
    )
    order = list(dict.fromkeys(labels))
    _write_json(
        root / "annotation_config.json",
        {"feature_name": "subtask", "subtasks": [{"name": name} for name in order]},
    )
    episode = root / "ep_000000"
    episode.mkdir()
    _write_json(episode / "info.json", {"length": len(labels), "task": "test task"})
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(list(range(len(labels))), type=pa.int64()),
                pa.array([[0.0]] * len(labels), type=pa.list_(pa.float32(), 1)),
            ],
            names=["frame_index", "action"],
        ),
        episode / "frames.parquet",
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(labels, type=pa.string()),
                pa.array([0.5] * len(labels), type=pa.float32()),
            ],
            names=["subtask", "subtask_progress"],
        ),
        episode / EXTRAS_FILENAME,
    )
    prepare_value_targets(
        ValueTargetConfig(root=root, mode="both", global_scale="max", subtask_scale="max")
    )
    return root


def _extras(root):
    return pq.read_table(root / "ep_000000" / EXTRAS_FILENAME)


def _install_model_subtask_stage(
    root,
    *,
    gt_head_values=None,
    smooth_head_values=None,
    smooth_ids=None,
    prediction_source=PREDICTION_SOURCE_MODEL,
    synthetic=False,
):
    length = _extras(root).num_rows
    episode_columns = {0: {}}
    if gt_head_values is not None:
        episode_columns[0][VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD] = pa.array(
            gt_head_values, type=pa.float32()
        )
    if smooth_head_values is not None:
        episode_columns[0][VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD] = pa.array(
            smooth_head_values, type=pa.float32()
        )
    if smooth_ids is not None:
        episode_columns[0][VALUE_SUBTASK_ID_PRED_SMOOTH] = pa.array(
            smooth_ids, type=pa.int32()
        )
    assert all(len(column) == length for column in episode_columns[0].values())
    merge_raw_run_extras(root, episode_columns)
    output_columns = sorted(episode_columns[0])
    output_fingerprint = fingerprint_raw_run_columns(root, output_columns)
    update_stage_metadata(
        root,
        f"{VALUE_INFERENCE_STAGE_PREFIX}.subtask",
        config={"checkpoint": "test", "mode": "subtask"},
        input_columns=[],
        input_fingerprint=fingerprint_raw_run_columns(root, []),
        output_columns=output_columns,
        output_fingerprint=output_fingerprint,
        prediction_source=prediction_source,
        synthetic=synthetic,
    )


def test_global_linear_episode_has_zero_advantage():
    columns = compute_global_advantage_columns(
        np.asarray([5, 4, 3, 2, 1, 0], dtype=np.float32), chunk_size=2
    )

    assert columns[ADVANTAGE_GLOBAL_CHUNK].tolist() == pytest.approx([0.0] * 6)
    assert columns[ADVANTAGE_GLOBAL_VALID_HORIZON].tolist() == [2, 2, 2, 2, 1, 0]
    assert columns[ADVANTAGE_GLOBAL_IS_VALID].tolist()[-1] is False


def test_global_stuck_segment_has_negative_advantage():
    columns = compute_global_advantage_columns(
        np.asarray([5, 5, 5, 4], dtype=np.float32), chunk_size=2
    )

    assert columns[ADVANTAGE_GLOBAL_CHUNK][0] == pytest.approx(-2.0)


def test_global_tail_padding_uses_valid_horizon():
    columns = compute_global_advantage_columns(
        np.asarray([3, 2, 1, 0], dtype=np.float32), chunk_size=50
    )

    assert columns[ADVANTAGE_GLOBAL_VALID_HORIZON].tolist() == [3, 2, 1, 0]
    assert columns[ADVANTAGE_GLOBAL_CHUNK].tolist() == pytest.approx([0.0] * 4)


def test_subtask_ideal_chunk_without_boundary_is_zero():
    columns = compute_subtask_advantage_columns(
        np.asarray([3, 2, 1, 0], dtype=np.float32),
        np.asarray([0, 0, 0, 0], dtype=np.int32),
        chunk_size=3,
    )

    assert columns[ADVANTAGE_SUBTASK_CHUNK][0] == pytest.approx(0.0)
    assert columns[ADVANTAGE_SUBTASK_NUM_CROSSINGS][0] == 0


def test_subtask_ideal_chunk_crossing_one_boundary_is_zero():
    columns = compute_subtask_advantage_columns(
        np.asarray([2, 1, 0, 2, 1, 0], dtype=np.float32),
        np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        chunk_size=4,
    )

    assert columns[ADVANTAGE_SUBTASK_CHUNK][0] == pytest.approx(0.0)
    assert columns[ADVANTAGE_SUBTASK_NUM_CROSSINGS][0] == 1
    assert columns[ADVANTAGE_SUBTASK_WITHIN_SUBTASK_HORIZON][0] == 3
    assert columns[ADVANTAGE_SUBTASK_BOUNDARY_PROGRESS][0] == pytest.approx(1.0)


def test_subtask_ideal_chunk_crossing_two_boundaries_is_zero():
    columns = compute_subtask_advantage_columns(
        np.asarray([1, 0, 1, 0, 1, 0], dtype=np.float32),
        np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        chunk_size=5,
    )

    assert columns[ADVANTAGE_SUBTASK_CHUNK][0] == pytest.approx(0.0)
    assert columns[ADVANTAGE_SUBTASK_NUM_CROSSINGS][0] == 2
    assert columns[ADVANTAGE_SUBTASK_WITHIN_SUBTASK_HORIZON][0] == 3
    assert columns[ADVANTAGE_SUBTASK_BOUNDARY_PROGRESS][0] == pytest.approx(2.0)


def test_boundary_transition_zero_reduces_advantage_by_crossing_count():
    columns = compute_subtask_advantage_columns(
        np.asarray([1, 0, 1, 0, 1, 0], dtype=np.float32),
        np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        chunk_size=5,
        boundary_transition_value=0.0,
    )

    assert columns[ADVANTAGE_SUBTASK_CHUNK][0] == pytest.approx(-2.0)


def test_subtask_stuck_segment_is_negative():
    columns = compute_subtask_advantage_columns(
        np.asarray([3, 3, 3, 2], dtype=np.float32),
        np.asarray([0, 0, 0, 0], dtype=np.int32),
        chunk_size=2,
    )

    assert columns[ADVANTAGE_SUBTASK_CHUNK][0] == pytest.approx(-2.0)


def test_frame_unit_advantage_is_independent_of_normalized_scale():
    values = np.asarray([2, 1, 0, 2, 1, 0], dtype=np.float32)
    ids = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    first = compute_subtask_advantage_columns(values, ids, chunk_size=4)
    # Normalized values/scales are deliberately not inputs to the frame-unit formula.
    second = compute_subtask_advantage_columns(values.copy(), ids.copy(), chunk_size=4)

    np.testing.assert_array_equal(
        first[ADVANTAGE_SUBTASK_CHUNK], second[ADVANTAGE_SUBTASK_CHUNK]
    )


def test_gt_subtask_write_has_new_debug_columns_and_formula_metadata(tmp_path):
    root = _make_target_run(tmp_path)

    summary = compute_advantage(
        AdvantageConfig(root=root, value_mode="subtask", chunk_size=4)
    )
    table = _extras(root)
    metadata = json.loads((root / VALUE_FUNCTION_META_FILENAME).read_text())

    assert table.column(ADVANTAGE_SUBTASK_CHUNK).to_pylist()[0] == pytest.approx(0.0)
    assert ADVANTAGE_SUBTASK_NUM_CROSSINGS in table.column_names
    assert summary["advantage_formula_version"] == "subtask_boundary_transition_v2"
    assert summary["experiment_eligible"] is False
    assert summary["synthetic"] is False
    assert metadata["advantage"]["subtask"]["boundary_transition_value"] == 1.0
    assert metadata["stages"][f"{ADVANTAGE_STAGE_PREFIX}.subtask"]["output_fingerprint"]


def test_deprecated_boundary_bonus_zero_maps_to_transition_one(tmp_path):
    root = _make_target_run(tmp_path)

    with pytest.warns(FutureWarning, match="boundary_bonus is deprecated"):
        summary = compute_advantage(
            AdvantageConfig(
                root=root,
                value_mode="subtask",
                chunk_size=4,
                boundary_bonus=0.0,
                dry_run=True,
            )
        )

    assert summary["boundary_transition_value"] == 1.0


@pytest.mark.parametrize("source", ["gt", "mock_pred"])
def test_gt_and_mock_reject_pred_smooth_path(tmp_path, source):
    root = _make_target_run(tmp_path)
    if source == "mock_pred":
        generate_mock_predictions(MockPredictionConfig(root=root, mode="subtask"))

    with pytest.raises(ValueError, match="only supports.*gt_conditioned"):
        compute_advantage(
            AdvantageConfig(
                root=root,
                value_mode="subtask",
                value_source=source,
                subtask_inference_path="pred_smooth",
                dry_run=True,
            )
        )


def test_model_gt_conditioned_paired_path(tmp_path):
    root = _make_target_run(tmp_path)
    gt_values = _extras(root).column(VALUE_SUBTASK_REMAINING_FRAMES_GT).to_pylist()
    _install_model_subtask_stage(root, gt_head_values=gt_values)

    summary = compute_advantage(
        AdvantageConfig(
            root=root,
            value_mode="subtask",
            value_source="model_pred",
            subtask_inference_path="gt_conditioned",
            chunk_size=4,
            dry_run=True,
        )
    )

    assert summary["value_column"] == VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD
    assert summary["boundary_column"] == VALUE_SUBTASK_ID_GT
    assert summary["experiment_eligible"] is True


def test_model_pred_smooth_paired_path(tmp_path):
    root = _make_target_run(tmp_path)
    gt_values = _extras(root).column(VALUE_SUBTASK_REMAINING_FRAMES_GT).to_pylist()
    smooth_ids = _extras(root).column(VALUE_SUBTASK_ID_GT).to_pylist()
    _install_model_subtask_stage(
        root, smooth_head_values=gt_values, smooth_ids=smooth_ids
    )

    summary = compute_advantage(
        AdvantageConfig(
            root=root,
            value_mode="subtask",
            value_source="model_pred",
            subtask_inference_path="pred_smooth",
            dry_run=True,
        )
    )

    assert summary["value_column"] == VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD
    assert summary["boundary_column"] == VALUE_SUBTASK_ID_PRED_SMOOTH


@pytest.mark.parametrize(
    ("installed", "requested"),
    [("gt", "pred_smooth"), ("smooth", "gt_conditioned")],
)
def test_model_head_boundary_mismatch_is_rejected(tmp_path, installed, requested):
    root = _make_target_run(tmp_path)
    values = _extras(root).column(VALUE_SUBTASK_REMAINING_FRAMES_GT).to_pylist()
    ids = _extras(root).column(VALUE_SUBTASK_ID_GT).to_pylist()
    if installed == "gt":
        _install_model_subtask_stage(root, gt_head_values=values)
    else:
        _install_model_subtask_stage(root, smooth_head_values=values, smooth_ids=ids)

    with pytest.raises(ValueError, match="missing required paired columns"):
        compute_advantage(
            AdvantageConfig(
                root=root,
                value_mode="subtask",
                value_source="model_pred",
                subtask_inference_path=requested,
                dry_run=True,
            )
        )


def test_nonfinite_value_and_unknown_id_are_counted_invalid(tmp_path):
    root = _make_target_run(tmp_path)
    values = [2.0, np.nan, 0.0, 1.0, 0.0, 0.0]
    ids = [0, 0, -1, 1, 1, 1]
    _install_model_subtask_stage(root, smooth_head_values=values, smooth_ids=ids)

    summary = compute_advantage(
        AdvantageConfig(
            root=root,
            value_mode="subtask",
            value_source="model_pred",
            subtask_inference_path="pred_smooth",
            chunk_size=1,
            dry_run=True,
        )
    )

    assert summary["invalid_reason_counts"]["nonfinite_value"] >= 1
    assert summary["invalid_reason_counts"]["unknown_subtask_id"] >= 1


@pytest.mark.parametrize(
    "ids",
    [
        [0, 0, 1, 1, 0, 0],
        [0, 0, 2, 2, 1, 1],
    ],
)
def test_invalid_predicted_subtask_sequence_is_rejected(tmp_path, ids):
    root = _make_target_run(tmp_path)
    values = _extras(root).column(VALUE_SUBTASK_REMAINING_FRAMES_GT).to_pylist()
    _install_model_subtask_stage(root, smooth_head_values=values, smooth_ids=ids)

    with pytest.raises(ValueError, match="Subtask sequence invalid"):
        compute_advantage(
            AdvantageConfig(
                root=root,
                value_mode="subtask",
                value_source="model_pred",
                subtask_inference_path="pred_smooth",
                dry_run=True,
            )
        )


def test_stale_mock_source_is_rejected(tmp_path):
    root = _make_target_run(tmp_path)
    generate_mock_predictions(MockPredictionConfig(root=root, mode="both"))
    prepare_value_targets(ValueTargetConfig(root=root, mode="both", num_bins=128))
    metadata = json.loads((root / VALUE_FUNCTION_META_FILENAME).read_text())
    assert metadata["stages"][MOCK_PREDICTIONS_STAGE]["stale"] is True

    with pytest.raises(StalePipelineArtifactError, match="stale"):
        compute_advantage(
            AdvantageConfig(
                root=root,
                value_mode="global",
                value_source="mock_pred",
                dry_run=True,
            )
        )


def test_externally_modified_gt_source_is_rejected(tmp_path):
    root = _make_target_run(tmp_path)
    merge_episode_extras(
        root / "ep_000000",
        {
            VALUE_GLOBAL_REMAINING_FRAMES_GT: pa.array(
                [99, 98, 97, 96, 95, 94], type=pa.int32()
            )
        },
    )

    with pytest.raises(StalePipelineArtifactError, match="outputs changed"):
        compute_advantage(
            AdvantageConfig(root=root, value_mode="global", value_source="gt", dry_run=True)
        )


def test_mock_advantage_is_synthetic_and_reproducible(tmp_path):
    root = _make_target_run(tmp_path)
    generate_mock_predictions(
        MockPredictionConfig(root=root, mode="both", seed=5, noise_std_frames=2.0)
    )

    first = compute_advantage(
        AdvantageConfig(
            root=root,
            value_mode="subtask",
            value_source="mock_pred",
            chunk_size=2,
        )
    )
    first_values = _extras(root).column(ADVANTAGE_SUBTASK_CHUNK).to_pylist()
    second = compute_advantage(
        AdvantageConfig(
            root=root,
            value_mode="subtask",
            value_source="mock_pred",
            chunk_size=2,
        )
    )
    second_values = _extras(root).column(ADVANTAGE_SUBTASK_CHUNK).to_pylist()

    assert first["synthetic"] is True
    assert first["experiment_eligible"] is False
    assert first["invalid_reason_counts"] == second["invalid_reason_counts"]
    assert first_values == second_values
    assert any(abs(value) > 1e-6 for value in first_values[:-1])


def test_missing_model_inference_stage_is_rejected(tmp_path):
    root = _make_target_run(tmp_path)

    with pytest.raises(ValueError, match="Missing pipeline stage metadata"):
        compute_advantage(
            AdvantageConfig(
                root=root,
                value_mode="subtask",
                value_source="model_pred",
                dry_run=True,
            )
        )


def test_model_inference_prediction_source_mismatch_is_rejected(tmp_path):
    root = _make_target_run(tmp_path)
    values = _extras(root).column(VALUE_SUBTASK_REMAINING_FRAMES_GT).to_pylist()
    _install_model_subtask_stage(
        root,
        gt_head_values=values,
        prediction_source="mock_pred",
        synthetic=True,
    )

    with pytest.raises(ValueError, match="prediction_source mismatch"):
        compute_advantage(
            AdvantageConfig(
                root=root,
                value_mode="subtask",
                value_source="model_pred",
                dry_run=True,
            )
        )


def test_cli_dry_run_and_deprecated_alias(tmp_path):
    root = _make_target_run(tmp_path)

    with pytest.warns(FutureWarning, match="boundary_bonus is deprecated"):
        summary = compute_advantage_main(
            [
                "--root",
                str(root),
                "--value_mode",
                "subtask",
                "--value_source",
                "gt",
                "--subtask_inference_path",
                "gt_conditioned",
                "--boundary_bonus",
                "0",
                "--dry_run",
            ]
        )

    assert summary["boundary_transition_value"] == 1.0
    assert ADVANTAGE_SUBTASK_CHUNK not in _extras(root).column_names


def test_cli_rejects_new_and_deprecated_boundary_flags_together(tmp_path):
    root = _make_target_run(tmp_path)

    with pytest.raises(SystemExit):
        compute_advantage_main(
            [
                "--root",
                str(root),
                "--value_mode",
                "subtask",
                "--boundary_transition_value",
                "1",
                "--boundary_bonus",
                "0",
                "--dry_run",
            ]
        )


def test_metadata_and_build_dataset_recognize_advantage_columns(tmp_path):
    root = _make_target_run(tmp_path)
    compute_advantage(AdvantageConfig(root=root, value_mode="subtask", chunk_size=2))

    features, columns = _load_extras_schema([root / "ep_000000"])
    result = build_dataset(
        BuildDatasetConfig(
            runs=[str(root)],
            output_repo_id="test/advantage",
            video=False,
            push_to_hub=False,
            dry_run=True,
        )
    )

    assert result is None
    assert ADVANTAGE_SUBTASK_CHUNK in columns
    assert ADVANTAGE_SUBTASK_NUM_CROSSINGS in columns
    assert features[ADVANTAGE_SUBTASK_CHUNK]["dtype"] == "float32"
    assert features[ADVANTAGE_SUBTASK_NUM_CROSSINGS]["dtype"] == "int32"
