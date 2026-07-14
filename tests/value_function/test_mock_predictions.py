import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.scripts.lerobot_build_dataset import (
    BuildDatasetConfig,
    _load_extras_schema,
    build_dataset,
)
from lerobot.scripts.lerobot_value_mock_predictions import main as mock_predictions_main
from lerobot.value_function.mock_predictions import (
    MockPredictionConfig,
    generate_mock_predictions,
)
from lerobot.value_function.raw_io import StalePipelineArtifactError, merge_episode_extras
from lerobot.value_function.schema import (
    EXTRAS_FILENAME,
    MOCK_PREDICTIONS_STAGE,
    RAW_FORMAT_VERSION,
    VALUE_FUNCTION_META_FILENAME,
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED,
    VALUE_GLOBAL_REMAINING_FRAMES_PRED,
    VALUE_SUBTASK_REMAINING_FRAMES_GT,
    VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED,
)
from lerobot.value_function.targets import ValueTargetConfig, prepare_value_targets


def _write_json(path, payload):
    path.write_text(json.dumps(payload))


def _make_raw_run(tmp_path, name="raw_run"):
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
    labels_by_episode = [
        ["pick"] * 5 + ["place"],
        ["pick"] * 3 + ["place"] * 2,
    ]
    _write_json(
        root / "annotation_config.json",
        {
            "feature_name": "subtask",
            "subtasks": [{"name": "place"}, {"name": "pick"}],
        },
    )
    for index, labels in enumerate(labels_by_episode):
        episode = root / f"ep_{index:06d}"
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


def _column(root, name, episode_index=0):
    table = pq.read_table(root / f"ep_{episode_index:06d}" / EXTRAS_FILENAME)
    return table.column(name).to_pylist()


def test_same_seed_is_bitwise_reproducible_and_different_seed_changes_values(tmp_path):
    root = _make_raw_run(tmp_path)
    config = MockPredictionConfig(root=root, mode="both", seed=42, noise_std_frames=3.0)

    generate_mock_predictions(config)
    first_global = _column(root, VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED)
    first_subtask = _column(root, VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED)
    generate_mock_predictions(config)

    assert _column(root, VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED) == first_global
    assert _column(root, VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED) == first_subtask

    generate_mock_predictions(
        MockPredictionConfig(root=root, mode="both", seed=43, noise_std_frames=3.0)
    )
    assert _column(root, VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED) != first_global
    assert _column(root, VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED) != first_subtask


def test_zero_noise_recovers_gt_exactly(tmp_path):
    root = _make_raw_run(tmp_path)

    generate_mock_predictions(
        MockPredictionConfig(root=root, mode="both", noise_std_frames=0.0)
    )

    assert _column(root, VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED) == _column(
        root, VALUE_GLOBAL_REMAINING_FRAMES_GT
    )
    assert _column(root, VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED) == _column(
        root, VALUE_SUBTASK_REMAINING_FRAMES_GT
    )


def test_global_random_stream_is_independent_of_mode(tmp_path):
    global_root = _make_raw_run(tmp_path, "global_run")
    both_root = _make_raw_run(tmp_path, "both_run")

    generate_mock_predictions(
        MockPredictionConfig(root=global_root, mode="global", seed=7, noise_std_frames=2.0)
    )
    generate_mock_predictions(
        MockPredictionConfig(root=both_root, mode="both", seed=7, noise_std_frames=2.0)
    )

    assert _column(global_root, VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED) == _column(
        both_root, VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED
    )


def test_subtask_smoothing_does_not_cross_boundaries(tmp_path):
    root = _make_raw_run(tmp_path)

    generate_mock_predictions(
        MockPredictionConfig(
            root=root,
            mode="subtask",
            noise_std_frames=0.0,
            temporal_smoothing_sigma_frames=2.0,
        )
    )

    # Episode 0 has a one-frame place segment with GT value 0. It stays exactly 0 rather
    # than being blended with the preceding pick values.
    assert _column(root, VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED)[-1] == 0.0


def test_mock_does_not_overwrite_model_prediction_columns(tmp_path):
    root = _make_raw_run(tmp_path)
    expected = [123.0] * 6
    for episode_index, length in ((0, 6), (1, 5)):
        merge_episode_extras(
            root / f"ep_{episode_index:06d}",
            {
                VALUE_GLOBAL_REMAINING_FRAMES_PRED: pa.array(
                    [123.0] * length, type=pa.float32()
                )
            },
        )

    generate_mock_predictions(MockPredictionConfig(root=root, mode="global"))

    assert _column(root, VALUE_GLOBAL_REMAINING_FRAMES_PRED) == expected


def test_mock_metadata_is_explicitly_synthetic(tmp_path):
    root = _make_raw_run(tmp_path)

    summary = generate_mock_predictions(
        MockPredictionConfig(
            root=root,
            mode="both",
            seed=9,
            noise_std_frames=1.5,
            temporal_smoothing_sigma_frames=0.75,
        )
    )
    metadata = json.loads((root / VALUE_FUNCTION_META_FILENAME).read_text())

    assert summary["warning"] == "SYNTHETIC / NOT FOR EXPERIMENT"
    assert metadata["mock_predictions"]["generator"] == "synthetic_gt_gaussian_noise"
    assert metadata["mock_predictions"]["experiment_eligible"] is False
    assert metadata["mock_predictions"]["source_gt_fingerprint"]
    assert metadata["stages"][MOCK_PREDICTIONS_STAGE]["prediction_source"] == "mock_pred"
    assert metadata["stages"][MOCK_PREDICTIONS_STAGE]["synthetic"] is True


def test_mock_rejects_inactive_target_mode(tmp_path):
    root = _make_raw_run(tmp_path)
    prepare_value_targets(ValueTargetConfig(root=root, mode="global"))

    with pytest.raises(ValueError, match="requires active target columns"):
        generate_mock_predictions(MockPredictionConfig(root=root, mode="subtask"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": -1}, "seed must be >= 0"),
        ({"noise_std_frames": -0.1}, "noise_std_frames must be >= 0"),
        (
            {"temporal_smoothing_sigma_frames": -0.1},
            "temporal_smoothing_sigma_frames must be >= 0",
        ),
    ],
)
def test_mock_rejects_invalid_numeric_config(tmp_path, kwargs, message):
    root = _make_raw_run(tmp_path)

    with pytest.raises(ValueError, match=message):
        generate_mock_predictions(MockPredictionConfig(root=root, **kwargs))


def test_mock_rejects_externally_modified_gt_output(tmp_path):
    root = _make_raw_run(tmp_path)
    episode = root / "ep_000000"
    merge_episode_extras(
        episode,
        {
            VALUE_GLOBAL_REMAINING_FRAMES_GT: pa.array(
                [99, 98, 97, 96, 95, 94], type=pa.int32()
            )
        },
    )

    with pytest.raises(StalePipelineArtifactError, match="outputs changed"):
        generate_mock_predictions(MockPredictionConfig(root=root, mode="global"))


def test_target_rerun_marks_mock_stage_stale(tmp_path):
    root = _make_raw_run(tmp_path)
    generate_mock_predictions(MockPredictionConfig(root=root, mode="both"))

    prepare_value_targets(ValueTargetConfig(root=root, mode="both", num_bins=128))
    metadata = json.loads((root / VALUE_FUNCTION_META_FILENAME).read_text())

    assert metadata["stages"][MOCK_PREDICTIONS_STAGE]["stale"] is True


def test_mock_dry_run_and_cli_do_not_write(tmp_path):
    root = _make_raw_run(tmp_path)

    summary = mock_predictions_main(
        [
            "--root",
            str(root),
            "--mode",
            "both",
            "--seed",
            "11",
            "--noise_std_frames",
            "2.0",
            "--dry_run",
        ]
    )
    metadata = json.loads((root / VALUE_FUNCTION_META_FILENAME).read_text())

    assert summary["dry_run"] is True
    assert VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED not in pq.read_table(
        root / "ep_000000" / EXTRAS_FILENAME
    ).column_names
    assert MOCK_PREDICTIONS_STAGE not in metadata["stages"]


def test_build_dataset_dry_run_recognizes_mock_columns(tmp_path):
    root = _make_raw_run(tmp_path)
    generate_mock_predictions(MockPredictionConfig(root=root, mode="both"))

    features, columns = _load_extras_schema(
        [root / "ep_000000", root / "ep_000001"]
    )
    result = build_dataset(
        BuildDatasetConfig(
            runs=[str(root)],
            output_repo_id="test/mock_predictions",
            video=False,
            push_to_hub=False,
            dry_run=True,
        )
    )

    assert result is None
    assert VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED in columns
    assert VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED in columns
    assert features[VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED]["dtype"] == "float32"
