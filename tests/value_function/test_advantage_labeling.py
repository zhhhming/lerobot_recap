import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.scripts.lerobot_advantage_labeler import main as labeler_main
from lerobot.scripts.lerobot_build_dataset import (
    BuildDatasetConfig,
    _load_extras_schema,
    build_dataset,
)
from lerobot.value_function.advantage_labeling import (
    AdvantageLabelingConfig,
    advantage_columns,
    chunk_key,
    export_advantage_labels,
    export_change_summary,
    load_advantage_chunks,
    load_saved_overrides,
    parse_chunk_key,
    preview_advantage_labels,
    sorted_advantage_chunks,
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
    ADVANTAGE_LABEL_GLOBAL,
    ADVANTAGE_LABEL_SUBTASK,
    ADVANTAGE_SUBTASK_CHUNK,
    ADVANTAGE_SUBTASK_IS_VALID,
    ADVANTAGE_SUBTASK_VALID_HORIZON,
    EXTRAS_FILENAME,
    RAW_FORMAT_VERSION,
    VALUE_FUNCTION_META_FILENAME,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload))


def _write_frames(ep_dir, length):
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(list(range(length)), type=pa.int64()),
                pa.array([[0.0]] * length, type=pa.list_(pa.float32(), 1)),
            ],
            names=["frame_index", "action"],
        ),
        ep_dir / "frames.parquet",
    )


def _record_advantage_stage(root, mode, *, synthetic=False):
    columns = list(advantage_columns(mode))
    prediction_source = "mock_pred" if synthetic else "model_pred"
    update_stage_metadata(
        root,
        f"advantage.{mode}",
        config={"value_mode": mode, "value_source": prediction_source},
        input_columns=[],
        input_fingerprint=fingerprint_raw_run_columns(root, []),
        output_columns=columns,
        output_fingerprint=fingerprint_raw_run_columns(root, columns),
        prediction_source=prediction_source,
        synthetic=synthetic,
        metadata_patch={
            "advantage": {
                mode: {
                    "prediction_source": prediction_source,
                    "synthetic": synthetic,
                    "experiment_eligible": not synthetic,
                    "columns_written": columns,
                }
            }
        },
    )


def _make_raw_run(tmp_path, *, synthetic=False):
    root = tmp_path / "raw_run"
    root.mkdir()
    _write_json(
        root / "run_meta.json",
        {
            "version": RAW_FORMAT_VERSION,
            "fps": 30,
            "task": "test task",
            "robot_type": "test_robot",
            "features": {
                "action": {"dtype": "float32", "shape": [1], "names": ["a"]},
                "observation.images.third_person": {
                    "dtype": "image",
                    "shape": [64, 64, 3],
                    "names": None,
                },
            },
        },
    )
    payloads = [
        {
            "advantages": [3.0, 1.0, -1.0, 0.0],
            "valid_horizon": [2, 2, 1, 0],
            "is_valid": [True, True, True, False],
        },
        {
            "advantages": [2.0, -2.0, 0.0],
            "valid_horizon": [2, 1, 0],
            "is_valid": [True, True, False],
        },
    ]
    for idx, payload in enumerate(payloads):
        length = len(payload["advantages"])
        ep_dir = root / f"ep_{idx:06d}"
        ep_dir.mkdir()
        _write_json(ep_dir / "info.json", {"length": length, "task": "test task"})
        _write_frames(ep_dir, length)
        pq.write_table(
            pa.Table.from_arrays(
                [
                    pa.array(["pick"] * length, type=pa.string()),
                    pa.array([0.5] * length, type=pa.float32()),
                    pa.array(payload["advantages"], type=pa.float32()),
                    pa.array(payload["valid_horizon"], type=pa.int32()),
                    pa.array(payload["is_valid"], type=pa.bool_()),
                    pa.array([-value for value in payload["advantages"]], type=pa.float32()),
                    pa.array(payload["valid_horizon"], type=pa.int32()),
                    pa.array(payload["is_valid"], type=pa.bool_()),
                ],
                names=[
                    "subtask",
                    "subtask_progress",
                    ADVANTAGE_GLOBAL_CHUNK,
                    ADVANTAGE_GLOBAL_VALID_HORIZON,
                    ADVANTAGE_GLOBAL_IS_VALID,
                    ADVANTAGE_SUBTASK_CHUNK,
                    ADVANTAGE_SUBTASK_VALID_HORIZON,
                    ADVANTAGE_SUBTASK_IS_VALID,
                ],
            ),
            ep_dir / EXTRAS_FILENAME,
        )
    _record_advantage_stage(root, "global", synthetic=synthetic)
    _record_advantage_stage(root, "subtask", synthetic=synthetic)
    return root


def _extras(root, episode_index=0):
    return pq.read_table(root / f"ep_{episode_index:06d}" / EXTRAS_FILENAME)


def test_chunk_key_round_trip():
    key = chunk_key(12, 345)
    assert key == "ep_000012:frame_000345"
    assert parse_chunk_key(key) == (12, 345)
    with pytest.raises(ValueError, match="Invalid chunk key"):
        parse_chunk_key("bad-key")


def test_load_and_sort_advantage_chunks(tmp_path):
    root = _make_raw_run(tmp_path)
    chunks = load_advantage_chunks(root, value_mode="global")

    assert len(chunks) == 7
    assert sorted_advantage_chunks(chunks)[0]["key"] == "ep_000000:frame_000000"
    assert sorted_advantage_chunks(chunks, "asc")[0]["key"] == "ep_000001:frame_000001"
    assert sorted_advantage_chunks(chunks)[-1]["is_valid"] is False
    assert chunks[0]["stored_label"] == ""


def test_sort_direction_never_changes_positive_semantics(tmp_path):
    chunks = load_advantage_chunks(_make_raw_run(tmp_path), value_mode="global")
    desc = preview_advantage_labels(chunks, top_percent=0.4, sort_order="desc")
    asc = preview_advantage_labels(chunks, top_percent=0.4, sort_order="asc")

    assert desc["labels"] == asc["labels"]
    assert desc["labels"]["ep_000000:frame_000000"] == "positive"
    assert desc["sorted_chunks"][0]["advantage"] == 3.0
    assert asc["sorted_chunks"][0]["advantage"] == -2.0
    assert desc["threshold"]["positive_direction"] == "high"


def test_tie_policy_and_explicit_preview_fields():
    chunks = [
        {
            "key": chunk_key(0, i),
            "episode_index": 0,
            "frame_index": i,
            "advantage": value,
            "is_valid": True,
            "stored_label": "",
        }
        for i, value in enumerate([3.0, 2.0, 2.0, 2.0, 1.0])
    ]
    exact = preview_advantage_labels(chunks, top_percent=0.4, tie_policy="exact_count")
    inclusive = preview_advantage_labels(chunks, top_percent=0.4, tie_policy="include_all")

    assert exact["counts"]["positive"] == 2
    assert inclusive["counts"]["positive"] == 4
    assert exact["threshold"] == {
        "positive_direction": "high",
        "tie_policy": "exact_count",
        "threshold_value": 2.0,
        "tie_count": 3,
        "selected_at_tie_count": 1,
        "target_positive_count": 2,
        "actual_threshold_positive_count": 2,
        "valid_count": 5,
        "invalid_count": 0,
    }
    item = exact["sorted_chunks"][0]
    assert {"stored_label", "threshold_label", "manual_override_label", "preview_label", "label_source"} <= set(item)


def test_invalid_and_unknown_overrides_are_rejected(tmp_path):
    chunks = load_advantage_chunks(_make_raw_run(tmp_path), value_mode="global")
    with pytest.raises(ValueError, match="Invalid override label"):
        preview_advantage_labels(chunks, overrides={chunks[0]["key"]: "maybe"})
    with pytest.raises(ValueError, match="do not belong"):
        preview_advantage_labels(chunks, overrides={chunk_key(99, 0): "positive"})
    with pytest.raises(ValueError, match="Invalid chunk key"):
        preview_advantage_labels(chunks, overrides={"bad": "positive"})


def test_preview_counts_include_every_chunk_and_manual_source(tmp_path):
    chunks = load_advantage_chunks(_make_raw_run(tmp_path), value_mode="global")
    preview = preview_advantage_labels(
        chunks,
        top_percent=40,
        overrides={"ep_000000:frame_000003": "positive"},
    )

    assert preview["top_percent"] == pytest.approx(0.4)
    assert sum(preview["counts"].values()) == len(chunks)
    manual = next(
        item for item in preview["sorted_chunks"] if item["key"] == "ep_000000:frame_000003"
    )
    assert manual["threshold_label"] == "ignore"
    assert manual["manual_override_label"] == "positive"
    assert manual["preview_label"] == "positive"
    assert manual["label_source"] == "manual_override"


def test_export_preview_change_summary_and_atomic_persistence(tmp_path):
    root = _make_raw_run(tmp_path)
    config = AdvantageLabelingConfig(
        root=root,
        value_mode="global",
        top_percent=0.4,
        overrides={"ep_000001:frame_000001": "positive"},
    )
    dry_summary = export_advantage_labels(
        AdvantageLabelingConfig(**{**config.__dict__, "dry_run": True})
    )

    assert dry_summary["change_summary"]["unset_to_labeled"] == 7
    assert ADVANTAGE_LABEL_GLOBAL not in _extras(root, 0).column_names
    summary = export_advantage_labels(config)
    assert summary["counts"] == {"ignore": 2, "negative": 2, "positive": 3}
    assert _extras(root, 0).column("subtask").to_pylist() == ["pick"] * 4
    assert _extras(root, 0).column(ADVANTAGE_LABEL_GLOBAL).to_pylist() == [
        "positive",
        "negative",
        "negative",
        "ignore",
    ]
    assert _extras(root, 1).column(ADVANTAGE_LABEL_GLOBAL).to_pylist() == [
        "positive",
        "positive",
        "ignore",
    ]

    chunks = load_advantage_chunks(root, "global")
    second = export_change_summary(chunks, summary["change_summary"] and {
        item["key"]: item["stored_label"] for item in chunks
    })
    assert second["unchanged"] == 7
    metadata = json.loads((root / VALUE_FUNCTION_META_FILENAME).read_text())
    assert metadata["advantage_labeling"]["global"]["overrides"] == config.overrides
    assert ADVANTAGE_LABEL_GLOBAL in metadata["columns_written"]


def test_overrides_persist_and_are_isolated_by_mode(tmp_path):
    root = _make_raw_run(tmp_path)
    global_override = {chunk_key(0, 1): "positive"}
    subtask_override = {chunk_key(1, 1): "ignore"}
    export_advantage_labels(
        AdvantageLabelingConfig(root=root, value_mode="global", overrides=global_override)
    )
    export_advantage_labels(
        AdvantageLabelingConfig(root=root, value_mode="subtask", overrides=subtask_override)
    )

    assert load_saved_overrides(root, "global") == global_override
    assert load_saved_overrides(root, "subtask") == subtask_override
    restarted = export_advantage_labels(
        AdvantageLabelingConfig(root=root, value_mode="global", dry_run=True)
    )
    assert restarted["overrides"] == global_override
    assert ADVANTAGE_LABEL_GLOBAL in _extras(root, 0).column_names
    assert ADVANTAGE_LABEL_SUBTASK in _extras(root, 0).column_names


def test_synthetic_export_gate_and_explicit_test_escape_hatch(tmp_path):
    root = _make_raw_run(tmp_path, synthetic=True)
    with pytest.raises(ValueError, match="Formal label export requires model_pred"):
        export_advantage_labels(AdvantageLabelingConfig(root=root, dry_run=True))

    summary = export_advantage_labels(
        AdvantageLabelingConfig(root=root, allow_synthetic=True, dry_run=True)
    )
    assert summary["eligibility"]["experiment_eligible"] is False
    assert "NOT FOR EXPERIMENT" in summary["eligibility"]["warning"]


def test_stale_advantage_output_is_rejected(tmp_path):
    root = _make_raw_run(tmp_path)
    table = _extras(root, 0)
    changed = table.set_column(
        table.schema.get_field_index(ADVANTAGE_GLOBAL_CHUNK),
        ADVANTAGE_GLOBAL_CHUNK,
        pa.array([99.0, 1.0, -1.0, 0.0], type=pa.float32()),
    )
    pq.write_table(changed, root / "ep_000000" / EXTRAS_FILENAME)

    with pytest.raises(StalePipelineArtifactError, match="outputs changed"):
        load_advantage_chunks(root, "global")


def test_headless_cli_dry_run_and_export(tmp_path, capsys):
    root = _make_raw_run(tmp_path)
    dry = labeler_main(["--root", str(root), "--export", "--dry_run", "--top_percent", "40"])
    assert dry["dry_run"] is True
    assert ADVANTAGE_LABEL_GLOBAL not in _extras(root, 0).column_names
    exported = labeler_main(["--root", str(root), "--export", "--top_percent", "40"])
    assert exported["dry_run"] is False
    assert ADVANTAGE_LABEL_GLOBAL in _extras(root, 0).column_names
    assert '"positive_direction": "high"' in capsys.readouterr().out


def test_build_dataset_dry_run_recognizes_label_columns(tmp_path):
    root = _make_raw_run(tmp_path)
    export_advantage_labels(AdvantageLabelingConfig(root=root, value_mode="global"))
    export_advantage_labels(AdvantageLabelingConfig(root=root, value_mode="subtask"))

    features, columns = _load_extras_schema([root / "ep_000000", root / "ep_000001"])
    result = build_dataset(
        BuildDatasetConfig(
            runs=[str(root)],
            output_repo_id="test/advantage_labels",
            video=False,
            push_to_hub=False,
            dry_run=True,
        )
    )
    assert result is None
    assert {ADVANTAGE_LABEL_GLOBAL, ADVANTAGE_LABEL_SUBTASK} <= set(columns)
    assert features[ADVANTAGE_LABEL_GLOBAL]["dtype"] == "string"
