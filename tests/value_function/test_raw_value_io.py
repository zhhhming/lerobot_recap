import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import lerobot.value_function.raw_io as raw_io
from lerobot.scripts.lerobot_build_dataset import (
    BuildDatasetConfig,
    _load_extras_schema,
    build_dataset,
)
from lerobot.value_function.raw_io import (
    StalePipelineArtifactError,
    assert_stage_dependencies_current,
    discover_episodes,
    fingerprint_payload,
    fingerprint_raw_run_columns,
    frame_image_paths,
    get_image_keys,
    merge_value_function_metadata,
    merge_episode_extras,
    merge_raw_run_extras,
    normalize_stage_config,
    read_value_function_metadata,
    update_stage_metadata,
    write_value_function_metadata,
)
from lerobot.value_function.schema import (
    ADVANTAGE_LABELING_STAGE_PREFIX,
    ADVANTAGE_STAGE_PREFIX,
    EXTRAS_FILENAME,
    PIPELINE_SCHEMA_VERSION,
    RAW_FORMAT_VERSION,
    TARGET_STAGE,
    VALUE_FUNCTION_META_FILENAME,
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_NAME_GT,
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


def _write_extras(ep_dir, length, *, note_prefix="keep"):
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(["pick"] * length, type=pa.string()),
                pa.array([0.5] * length, type=pa.float32()),
                pa.array([f"{note_prefix}-{i}" for i in range(length)], type=pa.string()),
            ],
            names=["subtask", "subtask_progress", "operator_note"],
        ),
        ep_dir / EXTRAS_FILENAME,
    )


def _make_raw_run(tmp_path, lengths=(4,), *, with_extras=True):
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
    for idx, length in enumerate(lengths):
        ep_dir = root / f"ep_{idx:06d}"
        ep_dir.mkdir()
        _write_json(ep_dir / "info.json", {"length": length, "task": "test task"})
        _write_frames(ep_dir, length)
        if with_extras:
            _write_extras(ep_dir, length, note_prefix=f"ep{idx}")
    return root


def test_discover_episodes_and_image_paths(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(2, 3), with_extras=False)

    episodes = discover_episodes(root)
    image_keys = get_image_keys(json.loads((root / "run_meta.json").read_text()))
    paths = frame_image_paths(episodes[0].path, 1, image_keys)

    assert [(ep.index, ep.frame_count) for ep in episodes] == [(0, 2), (1, 3)]
    assert image_keys == ["observation.images.third_person"]
    assert paths["observation.images.third_person"] == (
        episodes[0].path / "third_person" / "000001.png"
    )


def test_merge_episode_extras_preserves_existing_columns(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(4,), with_extras=True)
    ep_dir = root / "ep_000000"

    table = merge_episode_extras(
        ep_dir,
        {
            VALUE_GLOBAL_REMAINING_FRAMES_GT: pa.array([3, 2, 1, 0], type=pa.int32()),
            VALUE_GLOBAL_REMAINING_NORM_GT: pa.array([1.0, 0.66, 0.33, 0.0], type=pa.float32()),
        },
    )

    assert table.column_names == [
        "subtask",
        "subtask_progress",
        "operator_note",
        VALUE_GLOBAL_REMAINING_FRAMES_GT,
        VALUE_GLOBAL_REMAINING_NORM_GT,
    ]
    assert table.column("subtask").to_pylist() == ["pick"] * 4
    assert table.column("operator_note").to_pylist() == ["ep0-0", "ep0-1", "ep0-2", "ep0-3"]
    assert table.column(VALUE_GLOBAL_REMAINING_FRAMES_GT).to_pylist() == [3, 2, 1, 0]


def test_merge_episode_extras_replaces_existing_column_in_place(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(3,), with_extras=True)
    ep_dir = root / "ep_000000"

    table = merge_episode_extras(
        ep_dir,
        {
            "subtask_progress": pa.array([0.0, 0.5, 1.0], type=pa.float32()),
            VALUE_SUBTASK_ID_GT: pa.array([0, 0, 0], type=pa.int32()),
        },
    )

    assert table.column_names == [
        "subtask",
        "subtask_progress",
        "operator_note",
        VALUE_SUBTASK_ID_GT,
    ]
    assert table.column("subtask_progress").to_pylist() == pytest.approx([0.0, 0.5, 1.0])


def test_merge_episode_extras_rejects_length_mismatch(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(3,), with_extras=True)

    with pytest.raises(ValueError, match="length .* does not match"):
        merge_episode_extras(
            root / "ep_000000",
            {VALUE_GLOBAL_REMAINING_FRAMES_GT: pa.array([2, 1], type=pa.int32())},
        )


def test_merge_episode_extras_write_failure_preserves_original(tmp_path, monkeypatch):
    root = _make_raw_run(tmp_path, lengths=(3,), with_extras=True)
    extras_path = root / "ep_000000" / EXTRAS_FILENAME
    original_bytes = extras_path.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("injected parquet write failure")

    monkeypatch.setattr(raw_io.pq, "write_table", fail_write)
    with pytest.raises(OSError, match="injected parquet write failure"):
        merge_episode_extras(
            extras_path.parent,
            {VALUE_SUBTASK_ID_GT: pa.array([0, 0, 0], type=pa.int32())},
        )

    assert extras_path.read_bytes() == original_bytes
    assert list(extras_path.parent.glob(f".{EXTRAS_FILENAME}.*")) == []


def test_merge_episode_extras_replace_failure_preserves_original(tmp_path, monkeypatch):
    root = _make_raw_run(tmp_path, lengths=(3,), with_extras=True)
    extras_path = root / "ep_000000" / EXTRAS_FILENAME
    original_bytes = extras_path.read_bytes()
    real_replace = raw_io.os.replace

    def fail_replace(source, destination):
        if destination == extras_path:
            raise OSError("injected replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(raw_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        merge_episode_extras(
            extras_path.parent,
            {VALUE_SUBTASK_ID_GT: pa.array([0, 0, 0], type=pa.int32())},
        )

    assert extras_path.read_bytes() == original_bytes
    assert list(extras_path.parent.glob(f".{EXTRAS_FILENAME}.*")) == []


def test_merge_raw_run_extras_enforces_consistent_schema(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(2, 3), with_extras=True)

    written = merge_raw_run_extras(
        root,
        {
            0: {
                VALUE_SUBTASK_ID_GT: pa.array([0, 0], type=pa.int32()),
                VALUE_SUBTASK_NAME_GT: pa.array(["pick", "pick"], type=pa.string()),
            },
            1: {
                VALUE_SUBTASK_ID_GT: pa.array([0, 1, 1], type=pa.int32()),
                VALUE_SUBTASK_NAME_GT: pa.array(["pick", "place", "place"], type=pa.string()),
            },
        },
    )

    schemas = [pq.read_schema(path) for path in written.values()]
    assert len(set(str(schema) for schema in schemas)) == 1
    assert schemas[0].names == [
        "subtask",
        "subtask_progress",
        "operator_note",
        VALUE_SUBTASK_ID_GT,
        VALUE_SUBTASK_NAME_GT,
    ]


def test_merge_raw_run_extras_requires_all_episode_columns(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(2, 2), with_extras=True)

    with pytest.raises(ValueError, match="missing=\\[1\\]"):
        merge_raw_run_extras(root, {0: {VALUE_SUBTASK_ID_GT: pa.array([0, 0], type=pa.int32())}})


def test_merge_raw_run_extras_rolls_back_all_episodes_on_commit_failure(tmp_path, monkeypatch):
    root = _make_raw_run(tmp_path, lengths=(2, 3), with_extras=True)
    paths = [root / f"ep_{index:06d}" / EXTRAS_FILENAME for index in range(2)]
    original_bytes = [path.read_bytes() for path in paths]
    real_replace = raw_io.os.replace
    destination_replaces = 0
    failed = False

    def fail_second_destination(source, destination):
        nonlocal destination_replaces, failed
        if Path(destination).name == EXTRAS_FILENAME:
            destination_replaces += 1
            if destination_replaces == 2 and not failed:
                failed = True
                raise OSError("injected second commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(raw_io.os, "replace", fail_second_destination)
    with pytest.raises(OSError, match="injected second commit failure"):
        merge_raw_run_extras(
            root,
            {
                0: {VALUE_SUBTASK_ID_GT: pa.array([0, 0], type=pa.int32())},
                1: {VALUE_SUBTASK_ID_GT: pa.array([0, 0, 0], type=pa.int32())},
            },
        )

    assert [path.read_bytes() for path in paths] == original_bytes
    assert all(VALUE_SUBTASK_ID_GT not in pq.read_table(path).column_names for path in paths)
    assert not list(root.glob(f"ep_*/.{EXTRAS_FILENAME}.*"))


def test_merge_raw_run_extras_stages_all_files_before_commit(tmp_path, monkeypatch):
    root = _make_raw_run(tmp_path, lengths=(2, 2), with_extras=True)
    paths = [root / f"ep_{index:06d}" / EXTRAS_FILENAME for index in range(2)]
    original_bytes = [path.read_bytes() for path in paths]
    real_write_table = raw_io.pq.write_table
    writes = 0

    def fail_second_staged_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected second staging failure")
        return real_write_table(*args, **kwargs)

    monkeypatch.setattr(raw_io.pq, "write_table", fail_second_staged_write)
    with pytest.raises(OSError, match="injected second staging failure"):
        merge_raw_run_extras(
            root,
            {
                0: {VALUE_SUBTASK_ID_GT: pa.array([0, 0], type=pa.int32())},
                1: {VALUE_SUBTASK_ID_GT: pa.array([0, 0], type=pa.int32())},
            },
        )

    assert [path.read_bytes() for path in paths] == original_bytes
    assert not list(root.glob(f"ep_*/.{EXTRAS_FILENAME}.*"))


def test_merge_raw_run_extras_rolls_back_new_files_on_commit_failure(tmp_path, monkeypatch):
    root = _make_raw_run(tmp_path, lengths=(2, 2), with_extras=False)
    real_replace = raw_io.os.replace
    destination_replaces = 0
    failed = False

    def fail_second_destination(source, destination):
        nonlocal destination_replaces, failed
        if Path(destination).name == EXTRAS_FILENAME:
            destination_replaces += 1
            if destination_replaces == 2 and not failed:
                failed = True
                raise OSError("injected second commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(raw_io.os, "replace", fail_second_destination)
    with pytest.raises(OSError, match="injected second commit failure"):
        merge_raw_run_extras(
            root,
            {
                0: {VALUE_SUBTASK_ID_GT: pa.array([0, 0], type=pa.int32())},
                1: {VALUE_SUBTASK_ID_GT: pa.array([0, 0], type=pa.int32())},
            },
        )

    assert all(
        not (root / f"ep_{index:06d}" / EXTRAS_FILENAME).exists() for index in range(2)
    )
    assert not list(root.glob(f"ep_*/.{EXTRAS_FILENAME}.*"))


def test_value_function_metadata_round_trip(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(1,), with_extras=False)

    path = write_value_function_metadata(
        root,
        {
            "value_mode": "both",
            "num_bins": 256,
            "global_scale": {"strategy": "p95", "frames": 100.0},
            "subtask_names": ["pick", "place"],
            "subtask_scale": {"pick": 10.0, "place": 12.0},
            "checkpoint_path": None,
            "image_keys": ["observation.images.third_person"],
        },
    )
    payload = read_value_function_metadata(root)

    assert path == root / VALUE_FUNCTION_META_FILENAME
    assert payload["value_mode"] == "both"
    assert payload["num_bins"] == 256
    assert payload["created_at"]


def test_metadata_replace_failure_preserves_original(tmp_path, monkeypatch):
    root = _make_raw_run(tmp_path, lengths=(1,), with_extras=False)
    meta_path = write_value_function_metadata(root, {"keep": {"value": 1}})
    original_bytes = meta_path.read_bytes()
    real_replace = raw_io.os.replace

    def fail_metadata_replace(source, destination):
        if destination == meta_path:
            raise OSError("injected metadata replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(raw_io.os, "replace", fail_metadata_replace)
    with pytest.raises(OSError, match="injected metadata replace failure"):
        merge_value_function_metadata(root, {"new": True})

    assert meta_path.read_bytes() == original_bytes
    assert read_value_function_metadata(root) == {
        "keep": {"value": 1},
        "created_at": json.loads(original_bytes)["created_at"],
    }
    assert list(root.glob(f".{VALUE_FUNCTION_META_FILENAME}.*")) == []


def test_metadata_deep_merge_preserves_other_stages_and_timestamps(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(1,), with_extras=False)
    write_value_function_metadata(
        root,
        {
            "targets": {"created_at": "target-time", "config": {"mode": "both"}},
            "advantage": {"global": {"created_at": "advantage-time", "chunk_size": 4}},
            "user_note": "keep",
        },
    )

    merge_value_function_metadata(
        root,
        {"targets": {"created_at": "new-target-time", "config": {"num_bins": 256}}},
    )
    metadata = read_value_function_metadata(root)

    assert metadata["targets"] == {
        "created_at": "new-target-time",
        "config": {"mode": "both", "num_bins": 256},
    }
    assert metadata["advantage"]["global"]["created_at"] == "advantage-time"
    assert metadata["user_note"] == "keep"


def test_stage_config_normalization_and_fingerprint_are_stable(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(1,), with_extras=False)
    left = {"path": root, "nested": {"b": 2, "a": (1, True)}}
    right = {"nested": {"a": [1, True], "b": 2}, "path": root.resolve()}

    assert normalize_stage_config(left) == normalize_stage_config(right)
    assert fingerprint_payload(left) == fingerprint_payload(right)
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        normalize_stage_config({1: "invalid"})


def test_input_fingerprint_tracks_only_selected_columns(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(3,), with_extras=True)
    ep_dir = root / "ep_000000"
    original = fingerprint_raw_run_columns(root, ["subtask"])

    merge_episode_extras(
        ep_dir,
        {"operator_note": pa.array(["changed-0", "changed-1", "changed-2"], type=pa.string())},
    )
    assert fingerprint_raw_run_columns(root, ["subtask"]) == original

    merge_episode_extras(
        ep_dir,
        {"subtask": pa.array(["pick", "place", "place"], type=pa.string())},
    )
    assert fingerprint_raw_run_columns(root, ["subtask"]) != original


def test_stage_rerun_marks_transitive_dependents_stale(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(1,), with_extras=False)
    update_stage_metadata(
        root,
        TARGET_STAGE,
        config={"mode": "both"},
        input_columns=[],
        input_fingerprint="target-input-v1",
        output_columns=["value"],
        prediction_source="gt",
        synthetic=False,
    )
    advantage_stage = f"{ADVANTAGE_STAGE_PREFIX}.global"
    update_stage_metadata(
        root,
        advantage_stage,
        config={"chunk_size": 2},
        input_columns=["value"],
        input_fingerprint="advantage-input",
        output_columns=["advantage"],
        prediction_source="gt",
        synthetic=True,
        dependencies=[TARGET_STAGE],
    )
    labeling_stage = f"{ADVANTAGE_LABELING_STAGE_PREFIX}.global"
    update_stage_metadata(
        root,
        labeling_stage,
        config={"top_percent": 0.8},
        input_columns=["advantage"],
        input_fingerprint="label-input",
        output_columns=["label"],
        prediction_source="gt",
        synthetic=True,
        dependencies=[advantage_stage],
    )
    before = read_value_function_metadata(root)
    advantage_created_at = before["stages"][advantage_stage]["created_at"]

    update_stage_metadata(
        root,
        TARGET_STAGE,
        config={"mode": "both", "num_bins": 512},
        input_columns=[],
        input_fingerprint="target-input-v2",
        output_columns=["value"],
        prediction_source="gt",
        synthetic=False,
    )
    metadata = read_value_function_metadata(root)

    assert metadata["pipeline_schema_version"] == PIPELINE_SCHEMA_VERSION
    assert metadata["stages"][advantage_stage]["stale"] is True
    assert metadata["stages"][labeling_stage]["stale"] is True
    assert metadata["stages"][advantage_stage]["created_at"] == advantage_created_at
    with pytest.raises(StalePipelineArtifactError, match="stale"):
        assert_stage_dependencies_current(root, advantage_stage)


def test_stage_check_detects_externally_changed_input_columns(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(2,), with_extras=True)
    input_fingerprint = fingerprint_raw_run_columns(root, ["subtask"])
    update_stage_metadata(
        root,
        TARGET_STAGE,
        config={"mode": "subtask"},
        input_columns=["subtask"],
        input_fingerprint=input_fingerprint,
        output_columns=["value"],
        prediction_source="gt",
        synthetic=False,
    )
    assert_stage_dependencies_current(root, TARGET_STAGE)

    merge_episode_extras(
        root / "ep_000000",
        {"subtask": pa.array(["pick", "place"], type=pa.string())},
    )
    with pytest.raises(StalePipelineArtifactError, match="inputs changed"):
        assert_stage_dependencies_current(root, TARGET_STAGE)


def test_build_dataset_dry_run_recognizes_new_extras_columns(tmp_path):
    root = _make_raw_run(tmp_path, lengths=(3,), with_extras=True)
    merge_episode_extras(
        root / "ep_000000",
        {
            VALUE_GLOBAL_REMAINING_FRAMES_GT: pa.array([2, 1, 0], type=pa.int32()),
            VALUE_GLOBAL_REMAINING_NORM_GT: pa.array([1.0, 0.5, 0.0], type=pa.float32()),
        },
    )

    features, columns = _load_extras_schema([root / "ep_000000"])
    result = build_dataset(
        BuildDatasetConfig(
            runs=[str(root)],
            output_repo_id="test/value_raw_io",
            video=False,
            push_to_hub=False,
            dry_run=True,
        )
    )

    assert result is None
    assert VALUE_GLOBAL_REMAINING_FRAMES_GT in columns
    assert features[VALUE_GLOBAL_REMAINING_FRAMES_GT]["dtype"] == "int32"
    assert features[VALUE_GLOBAL_REMAINING_NORM_GT]["dtype"] == "float32"
