import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.scripts.lerobot_build_dataset import BuildDatasetConfig, _load_extras_schema, build_dataset
from lerobot.scripts.lerobot_compute_advantage_weights import main as weights_main
from lerobot.value_function.advantage_labeling import advantage_columns, label_column
from lerobot.value_function.advantage_weights import (
    AdvantageWeightConfig,
    compute_advantage_weights,
    compute_group_relative_weights,
    load_advantage_weight_chunks,
    weight_output_columns,
)
from lerobot.value_function.raw_io import (
    StalePipelineArtifactError,
    fingerprint_raw_run_columns,
    update_stage_metadata,
)
from lerobot.value_function.schema import (
    ADVANTAGE_GLOBAL_CHUNK,
    ADVANTAGE_GLOBAL_IS_VALID,
    ADVANTAGE_GLOBAL_VALID_HORIZON,
    ADVANTAGE_GROUP_ID_GLOBAL,
    ADVANTAGE_GROUP_ID_SUBTASK,
    ADVANTAGE_LABEL_GLOBAL,
    ADVANTAGE_LABEL_SUBTASK,
    ADVANTAGE_LOSS_WEIGHT_GLOBAL,
    ADVANTAGE_LOSS_WEIGHT_SUBTASK,
    ADVANTAGE_SUBTASK_CHUNK,
    ADVANTAGE_SUBTASK_IS_VALID,
    ADVANTAGE_SUBTASK_VALID_HORIZON,
    MOCK_PREDICTIONS_STAGE,
    RAW_FORMAT_VERSION,
    TARGET_STAGE,
    VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED,
    VALUE_GLOBAL_REMAINING_NORM_GT,
    VALUE_GLOBAL_REMAINING_NORM_PRED,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_ID_PRED_SMOOTH,
    VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED,
    VALUE_SUBTASK_REMAINING_NORM_GT,
    VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD,
    VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD,
)


def _write_run(tmp_path, *, synthetic=False, frame_count=8, pred_smooth=False):
    root = tmp_path / "raw_run"
    root.mkdir()
    (root / "run_meta.json").write_text(
        json.dumps(
            {
                "version": RAW_FORMAT_VERSION,
                "fps": 30,
                "task": "weight test",
                "robot_type": "test_robot",
                "features": {
                    "action": {"dtype": "float32", "shape": [1], "names": ["a"]},
                    "observation.images.third_person": {
                        "dtype": "image",
                        "shape": [8, 8, 3],
                        "names": None,
                    },
                },
            }
        )
    )
    episode = root / "ep_000000"
    episode.mkdir()
    (episode / "info.json").write_text(json.dumps({"length": frame_count, "task": "weight test"}))
    pq.write_table(
        pa.table(
            {
                "frame_index": pa.array(range(frame_count), type=pa.int64()),
                "action": pa.array([[0.0]] * frame_count, type=pa.list_(pa.float32(), 1)),
            }
        ),
        episode / "frames.parquet",
    )
    advantages = [float(frame_count - index) for index in range(frame_count)]
    horizons = [2] * (frame_count - 1) + [0]
    validity = [True] * (frame_count - 1) + [False]
    labels = ["positive"] * 4 + ["negative"] * 3 + ["ignore"] * (frame_count - 7)
    columns = {
        ADVANTAGE_GLOBAL_CHUNK: pa.array(advantages, type=pa.float32()),
        ADVANTAGE_GLOBAL_VALID_HORIZON: pa.array(horizons, type=pa.int32()),
        ADVANTAGE_GLOBAL_IS_VALID: pa.array(validity, type=pa.bool_()),
        ADVANTAGE_LABEL_GLOBAL: pa.array(labels, type=pa.string()),
        ADVANTAGE_SUBTASK_CHUNK: pa.array(advantages, type=pa.float32()),
        ADVANTAGE_SUBTASK_VALID_HORIZON: pa.array(horizons, type=pa.int32()),
        ADVANTAGE_SUBTASK_IS_VALID: pa.array(validity, type=pa.bool_()),
        ADVANTAGE_LABEL_SUBTASK: pa.array(labels, type=pa.string()),
        VALUE_SUBTASK_ID_GT: pa.array([0] * frame_count, type=pa.int32()),
        "subtask_progress": pa.array([0.25] * frame_count, type=pa.float32()),
    }
    if synthetic:
        columns.update(
            {
                VALUE_GLOBAL_REMAINING_NORM_GT: pa.array([0.25] * frame_count, type=pa.float32()),
                VALUE_SUBTASK_REMAINING_NORM_GT: pa.array([0.25] * frame_count, type=pa.float32()),
                VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED: pa.array(
                    [2.5] * frame_count, type=pa.float32()
                ),
                VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED: pa.array(
                    [2.5] * frame_count, type=pa.float32()
                ),
            }
        )
    else:
        columns.update(
            {
                VALUE_GLOBAL_REMAINING_NORM_PRED: pa.array([0.25] * frame_count, type=pa.float32()),
                VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD: pa.array(
                    [0.25] * frame_count, type=pa.float32()
                ),
            }
        )
        if pred_smooth:
            columns[VALUE_SUBTASK_ID_PRED_SMOOTH] = pa.array(
                [0] * frame_count, type=pa.int32()
            )
            columns[VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD] = pa.array(
                [0.25] * frame_count, type=pa.float32()
            )
    pq.write_table(pa.table(columns), episode / "extras.parquet")
    _record_stages(root, synthetic=synthetic, pred_smooth=pred_smooth)
    return root


def _record(root, name, output_columns, *, source, synthetic, dependencies=(), patch=None):
    update_stage_metadata(
        root,
        name,
        config={"stage": name},
        input_columns=[],
        input_fingerprint=fingerprint_raw_run_columns(root, []),
        output_columns=output_columns,
        output_fingerprint=fingerprint_raw_run_columns(root, output_columns),
        prediction_source=source,
        synthetic=synthetic,
        dependencies=dependencies,
        metadata_patch=patch,
    )


def _record_stages(root, *, synthetic, pred_smooth=False):
    source = "mock_pred" if synthetic else "model_pred"
    if synthetic:
        target_columns = [
            VALUE_GLOBAL_REMAINING_NORM_GT,
            VALUE_SUBTASK_REMAINING_NORM_GT,
            VALUE_SUBTASK_ID_GT,
        ]
        _record(
            root,
            TARGET_STAGE,
            target_columns,
            source="gt",
            synthetic=False,
            patch={
                "global_scale": {"frames": 10.0},
                "subtask_order": ["pick"],
                "subtask_scale": {"frames_by_subtask": {"pick": 10.0}},
            },
        )
        _record(
            root,
            MOCK_PREDICTIONS_STAGE,
            [VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED, VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED],
            source=source,
            synthetic=True,
            dependencies=[TARGET_STAGE],
        )
        value_dependencies = [MOCK_PREDICTIONS_STAGE]
    else:
        _record(
            root,
            "value_inference.global",
            [VALUE_GLOBAL_REMAINING_NORM_PRED],
            source=source,
            synthetic=False,
        )
        _record(
            root,
            "value_inference.subtask",
            (
                [
                    VALUE_SUBTASK_ID_PRED_SMOOTH,
                    VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD,
                ]
                if pred_smooth
                else [VALUE_SUBTASK_ID_GT, VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD]
            ),
            source=source,
            synthetic=False,
        )
        value_dependencies = []

    for mode in ("global", "subtask"):
        advantage_stage = f"advantage.{mode}"
        labeling_stage = f"advantage_labeling.{mode}"
        dependencies = value_dependencies or [f"value_inference.{mode}"]
        _record(
            root,
            advantage_stage,
            list(advantage_columns(mode)),
            source=source,
            synthetic=synthetic,
            dependencies=dependencies,
            patch={
                "advantage": {
                    mode: {
                        "value_source": source,
                        "prediction_source": source,
                        "subtask_inference_path": (
                            "pred_smooth"
                            if mode == "subtask" and pred_smooth
                            else "gt_conditioned" if mode == "subtask" else None
                        ),
                        "experiment_eligible": not synthetic,
                        "synthetic": synthetic,
                    }
                }
            },
        )
        _record(
            root,
            labeling_stage,
            [label_column(mode)],
            source=source,
            synthetic=synthetic,
            dependencies=[advantage_stage],
            patch={
                "advantage_labeling": {
                    mode: {
                        "eligibility": {
                            "prediction_source": source,
                            "synthetic": synthetic,
                            "experiment_eligible": not synthetic,
                        }
                    }
                }
            },
        )


def _extras(root):
    return pq.read_table(root / "ep_000000" / "extras.parquet")


def test_weight_formula_monotonic_ratio_and_label_defaults():
    advantages = [4.0, 3.0, 2.0, 1.0, 0.0, -1.0]
    labels = ["positive"] * 4 + ["negative", "ignore"]
    weights, ranks, groups = compute_group_relative_weights(["g"] * 6, advantages, labels)

    assert weights[:4].tolist() == sorted(weights[:4].tolist(), reverse=True)
    assert weights[0] == pytest.approx(2.0)
    assert weights[3] < 1.0
    assert weights[4:].tolist() == [1.0, 0.0]
    assert ranks == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0, None, None]
    raw = [0.1 + 1.9 / (1 + math.exp(-(0.8 - rank) / 0.08)) for rank in ranks[:4]]
    assert weights[1] / weights[2] == pytest.approx(raw[1] / raw[2])
    assert groups["g"]["rank_weighted"] is True


def test_ties_share_rank_and_small_groups_fallback_to_one():
    weights, ranks, groups = compute_group_relative_weights(
        ["large"] * 4 + ["small"] * 2,
        [3.0, 2.0, 2.0, 1.0, 4.0, 0.0],
        ["positive"] * 6,
        min_group_size=4,
    )
    assert ranks[1] == ranks[2]
    assert weights[1] == weights[2]
    assert weights[4:].tolist() == [1.0, 1.0]
    assert groups["small"]["rank_weighted"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tau": 0.0}, "tau"),
        ({"q": 2.0}, "q"),
        ({"w_min": 2.0, "w_max": 1.0}, "w_min"),
        ({"min_group_size": 0}, "min_group_size"),
    ],
)
def test_invalid_weight_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        compute_group_relative_weights(["g"], [1.0], ["positive"], **kwargs)


def test_global_and_subtask_export_columns_metadata_and_viz_load(tmp_path):
    root = _write_run(tmp_path)
    global_summary = compute_advantage_weights(AdvantageWeightConfig(root=root, value_mode="global"))
    subtask_summary = compute_advantage_weights(
        AdvantageWeightConfig(root=root, value_mode="subtask")
    )
    extras = _extras(root)

    assert global_summary["group_source"] == "value"
    assert subtask_summary["group_source"] == "progress"
    assert extras.column(ADVANTAGE_GROUP_ID_GLOBAL).type == pa.string()
    assert extras.column(ADVANTAGE_LOSS_WEIGHT_GLOBAL).type == pa.float32()
    assert extras.column(ADVANTAGE_GROUP_ID_SUBTASK).to_pylist() == [
        "subtask:0000:bin:+00002"
    ] * 8
    assert extras.column(ADVANTAGE_LOSS_WEIGHT_SUBTASK).to_pylist()[:4] == pytest.approx(
        extras.column(ADVANTAGE_LOSS_WEIGHT_GLOBAL).to_pylist()[:4]
    )
    chunks, groups = load_advantage_weight_chunks(root, "global")
    assert len(chunks) == 8
    assert chunks[0]["positive_rank"] == 1.0
    assert groups[0]["positive_count"] == 4
    metadata = json.loads((root / "value_function_meta.json").read_text())
    assert metadata["stages"]["advantage_weights.global"]["dependencies"][
        "advantage_labeling.global"
    ]


def test_dry_run_cli_and_synthetic_gate(tmp_path, capsys):
    root = _write_run(tmp_path, synthetic=True)
    with pytest.raises(ValueError, match="Formal weight export requires model_pred"):
        compute_advantage_weights(AdvantageWeightConfig(root=root, value_mode="global"))
    summary = weights_main(
        [
            "--root",
            str(root),
            "--value_mode",
            "global",
            "--allow_synthetic",
            "--dry_run",
        ]
    )
    assert summary["group_scalar_column"] == VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED
    assert summary["eligibility"]["experiment_eligible"] is False
    assert ADVANTAGE_GROUP_ID_GLOBAL not in _extras(root).column_names
    assert "NOT FOR EXPERIMENT" in capsys.readouterr().out


def test_pred_smooth_auto_grouping_uses_paired_predicted_value(tmp_path):
    root = _write_run(tmp_path, pred_smooth=True)
    summary = compute_advantage_weights(
        AdvantageWeightConfig(root=root, value_mode="subtask")
    )
    assert summary["group_source"] == "value"
    assert summary["group_subtask_id_column"] == VALUE_SUBTASK_ID_PRED_SMOOTH
    assert summary["group_scalar_column"] == VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD
    with pytest.raises(ValueError, match="cannot pair GT progress"):
        compute_advantage_weights(
            AdvantageWeightConfig(root=root, value_mode="subtask", group_source="progress")
        )


def test_ignore_not_ranked_and_stale_label_rejected(tmp_path):
    root = _write_run(tmp_path)
    table = _extras(root)
    labels = table.column(ADVANTAGE_LABEL_GLOBAL).to_pylist()
    labels[0] = "ignore"
    changed = table.set_column(
        table.schema.get_field_index(ADVANTAGE_LABEL_GLOBAL),
        ADVANTAGE_LABEL_GLOBAL,
        pa.array(labels, type=pa.string()),
    )
    pq.write_table(changed, root / "ep_000000" / "extras.parquet")

    with pytest.raises(StalePipelineArtifactError, match="outputs changed"):
        compute_advantage_weights(AdvantageWeightConfig(root=root, value_mode="global"))


def test_build_dataset_dry_run_recognizes_weight_columns(tmp_path):
    root = _write_run(tmp_path)
    for mode in ("global", "subtask"):
        compute_advantage_weights(AdvantageWeightConfig(root=root, value_mode=mode))

    features, columns = _load_extras_schema([root / "ep_000000"])
    result = build_dataset(
        BuildDatasetConfig(
            runs=[str(root)],
            output_repo_id="test/advantage_weights",
            video=False,
            push_to_hub=False,
            dry_run=True,
        )
    )
    assert result is None
    assert set(weight_output_columns("global")) | set(weight_output_columns("subtask")) <= columns
    assert features[ADVANTAGE_GROUP_ID_GLOBAL]["dtype"] == "string"
    assert features[ADVANTAGE_LOSS_WEIGHT_SUBTASK]["dtype"] == "float32"
