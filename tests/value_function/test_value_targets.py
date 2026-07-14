import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.scripts.lerobot_build_dataset import (
    BuildDatasetConfig,
    _load_extras_schema,
    build_dataset,
)
from lerobot.scripts.lerobot_value_prepare_targets import main as prepare_targets_main
from lerobot.value_function.advantage import AdvantageConfig, compute_advantage
from lerobot.value_function.schema import (
    ADVANTAGE_STAGE_PREFIX,
    EXTRAS_FILENAME,
    RAW_FORMAT_VERSION,
    TARGET_STAGE,
    VALUE_FUNCTION_META_FILENAME,
    VALUE_GLOBAL_ELAPSED_FRAMES_GT,
    VALUE_GLOBAL_ELAPSED_NORM_GT,
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT_IS_CLIPPED,
    VALUE_SUBTASK_ELAPSED_FRAMES_GT,
    VALUE_SUBTASK_ELAPSED_NORM_GT,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_NAME_GT,
    VALUE_SUBTASK_REMAINING_FRAMES_GT,
    VALUE_SUBTASK_REMAINING_NORM_GT,
    VALUE_SUBTASK_REMAINING_NORM_GT_IS_CLIPPED,
)
from lerobot.value_function.targets import ValueTargetConfig, prepare_value_targets


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


def _make_raw_run(
    tmp_path,
    labels_by_episode,
    *,
    annotation_subtasks=None,
    with_extras=True,
    episode_info_overrides=None,
):
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
    if annotation_subtasks is not None:
        _write_json(
            root / "annotation_config.json",
            {
                "feature_name": "subtask",
                "subtasks": [{"name": name, "color": "#000000"} for name in annotation_subtasks],
            },
        )

    for idx, labels in enumerate(labels_by_episode):
        ep_dir = root / f"ep_{idx:06d}"
        ep_dir.mkdir()
        info = {"length": len(labels), "task": "test task"}
        info.update((episode_info_overrides or {}).get(idx, {}))
        _write_json(ep_dir / "info.json", info)
        _write_frames(ep_dir, len(labels))
        if with_extras:
            pq.write_table(
                pa.Table.from_arrays(
                    [
                        pa.array(labels, type=pa.string()),
                        pa.array([0.1] * len(labels), type=pa.float32()),
                        pa.array([f"note-{idx}-{i}" for i in range(len(labels))], type=pa.string()),
                    ],
                    names=["subtask", "subtask_progress", "operator_note"],
                ),
                ep_dir / EXTRAS_FILENAME,
            )
    return root


def _extras(root, episode_index=0):
    return pq.read_table(root / f"ep_{episode_index:06d}" / EXTRAS_FILENAME)


def test_global_targets_for_length_ten_episode(tmp_path):
    root = _make_raw_run(tmp_path, [["pick"] * 10], annotation_subtasks=["pick"], with_extras=False)

    prepare_value_targets(
        ValueTargetConfig(root=root, mode="global", global_scale="max", elapsed_aux=True)
    )

    table = _extras(root)
    assert table.column(VALUE_GLOBAL_REMAINING_FRAMES_GT).to_pylist() == list(range(9, -1, -1))
    assert table.column(VALUE_GLOBAL_REMAINING_NORM_GT).to_pylist() == pytest.approx(
        [v / 9.0 for v in range(9, -1, -1)]
    )
    assert table.column(VALUE_GLOBAL_ELAPSED_FRAMES_GT).to_pylist() == list(range(10))
    assert table.column(VALUE_GLOBAL_ELAPSED_NORM_GT).to_pylist() == pytest.approx(
        [min(v / 9.0, 1.0) for v in range(10)]
    )


def test_subtask_targets_for_contiguous_segments(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick", "pick", "pick", "pick", "place", "place"]],
        annotation_subtasks=["pick", "place"],
    )

    prepare_value_targets(ValueTargetConfig(root=root, mode="subtask", subtask_scale="max"))

    table = _extras(root)
    assert table.column("subtask").to_pylist() == ["pick", "pick", "pick", "pick", "place", "place"]
    assert table.column("operator_note").to_pylist() == [f"note-0-{i}" for i in range(6)]
    assert table.column(VALUE_SUBTASK_ID_GT).to_pylist() == [0, 0, 0, 0, 1, 1]
    assert table.column(VALUE_SUBTASK_NAME_GT).to_pylist() == [
        "pick",
        "pick",
        "pick",
        "pick",
        "place",
        "place",
    ]
    assert table.column(VALUE_SUBTASK_REMAINING_FRAMES_GT).to_pylist() == pytest.approx(
        [3, 2, 1, 0, 1, 0]
    )
    assert table.column(VALUE_SUBTASK_REMAINING_NORM_GT).to_pylist() == pytest.approx(
        [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0, 1.0, 0.0]
    )


def test_subtask_elapsed_aux_targets(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick", "pick", "pick", "place", "place"]],
        annotation_subtasks=["pick", "place"],
    )

    prepare_value_targets(
        ValueTargetConfig(root=root, mode="subtask", subtask_scale="max", elapsed_aux=True)
    )

    table = _extras(root)
    assert table.column(VALUE_SUBTASK_ELAPSED_FRAMES_GT).to_pylist() == pytest.approx(
        [0, 1, 2, 0, 1]
    )
    assert table.column(VALUE_SUBTASK_ELAPSED_NORM_GT).to_pylist() == pytest.approx(
        [0.0, 0.5, 1.0, 0.0, 1.0]
    )


def test_subtask_order_can_be_inferred_without_annotation_config(tmp_path):
    root = _make_raw_run(tmp_path, [["open", "open", "close"]], annotation_subtasks=None)

    summary = prepare_value_targets(ValueTargetConfig(root=root, mode="subtask"))

    assert summary["subtask_names"] == ["open", "close"]
    assert _extras(root).column(VALUE_SUBTASK_ID_GT).to_pylist() == [0, 0, 1]


def test_annotation_config_names_do_not_define_task_order(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick", "pick", "ready", "ready", "place"]],
        annotation_subtasks=["ready", "place", "pick"],
    )

    summary = prepare_value_targets(ValueTargetConfig(root=root, mode="subtask"))

    assert summary["subtask_names"] == ["pick", "ready", "place"]
    assert _extras(root).column(VALUE_SUBTASK_ID_GT).to_pylist() == [0, 0, 1, 1, 2]


def test_subtask_order_regression_raises(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick", "place", "pick"]],
        annotation_subtasks=["pick", "place"],
    )

    with pytest.raises(ValueError, match="Subtask order regressed"):
        prepare_value_targets(ValueTargetConfig(root=root, mode="subtask"))


def test_unlabeled_frames_error_by_default(tmp_path):
    root = _make_raw_run(tmp_path, [["pick", "", "pick"]], annotation_subtasks=["pick"])

    with pytest.raises(ValueError, match="unlabeled subtask frames"):
        prepare_value_targets(ValueTargetConfig(root=root, mode="subtask"))


def test_unlabeled_frames_can_be_skipped(tmp_path):
    root = _make_raw_run(tmp_path, [["pick", "", "pick"]], annotation_subtasks=["pick"])

    prepare_value_targets(
        ValueTargetConfig(
            root=root,
            mode="subtask",
            allow_unlabeled="skip",
            subtask_scale="max",
            require_single_segment_per_subtask=False,
        )
    )

    table = _extras(root)
    remaining = table.column(VALUE_SUBTASK_REMAINING_FRAMES_GT).to_pylist()
    norm = table.column(VALUE_SUBTASK_REMAINING_NORM_GT).to_pylist()
    assert table.column(VALUE_SUBTASK_ID_GT).to_pylist() == [0, -1, 0]
    assert table.column(VALUE_SUBTASK_NAME_GT).to_pylist() == ["pick", "", "pick"]
    assert math.isnan(remaining[1])
    assert math.isnan(norm[1])


def test_manual_scale_clips_norm_and_records_metadata(tmp_path):
    root = _make_raw_run(tmp_path, [["pick"] * 7], annotation_subtasks=["pick"])

    summary = prepare_value_targets(
        ValueTargetConfig(
            root=root,
            mode="both",
            global_scale="manual",
            global_scale_frames=5.0,
            subtask_scale="manual",
            subtask_scale_frames={"pick": 5.0},
        )
    )

    table = _extras(root)
    assert table.column(VALUE_GLOBAL_REMAINING_NORM_GT).to_pylist() == pytest.approx(
        [1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    )
    assert table.column(VALUE_GLOBAL_REMAINING_NORM_GT_IS_CLIPPED).to_pylist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert table.column(VALUE_SUBTASK_REMAINING_NORM_GT_IS_CLIPPED).to_pylist()[0] is True
    assert summary["clip_summary"]["global_clip_rate"] == pytest.approx(1.0 / 7.0)

    meta = json.loads((root / VALUE_FUNCTION_META_FILENAME).read_text())
    assert meta["global_scale"] == {"strategy": "manual", "frames": 5.0}
    assert meta["subtask_scale"]["frames_by_subtask"] == {"pick": 5.0}


def test_mode_global_only_writes_global_columns(tmp_path):
    root = _make_raw_run(tmp_path, [["pick", "pick", "pick"]], with_extras=False)

    prepare_value_targets(ValueTargetConfig(root=root, mode="global"))

    names = _extras(root).column_names
    assert VALUE_GLOBAL_REMAINING_FRAMES_GT in names
    assert VALUE_SUBTASK_ID_GT not in names


def test_cli_dry_run_does_not_write_targets(tmp_path):
    root = _make_raw_run(tmp_path, [["pick", "pick"]], annotation_subtasks=["pick"])

    summary = prepare_targets_main(["--root", str(root), "--mode", "both", "--dry_run"])

    assert summary["dry_run"] is True
    assert VALUE_GLOBAL_REMAINING_FRAMES_GT not in _extras(root).column_names
    assert not (root / VALUE_FUNCTION_META_FILENAME).exists()


def test_cli_accepts_explicit_order_and_strict_flags(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["place", "pick"]],
        annotation_subtasks=["pick", "place"],
    )

    summary = prepare_targets_main(
        [
            "--root",
            str(root),
            "--mode",
            "subtask",
            "--subtask_order_json",
            '["place", "pick"]',
            "--require_all_subtasks",
            "true",
            "--require_single_segment_per_subtask",
            "true",
            "--require_success_only",
            "true",
            "--dry_run",
        ]
    )

    assert summary["subtask_order"] == ["place", "pick"]
    assert summary["strict_validation"]["require_all_subtasks"] is True


def test_build_dataset_dry_run_recognizes_value_target_columns(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick", "pick", "place"]],
        annotation_subtasks=["pick", "place"],
    )
    prepare_value_targets(ValueTargetConfig(root=root, mode="both"))

    features, columns = _load_extras_schema([root / "ep_000000"])
    result = build_dataset(
        BuildDatasetConfig(
            runs=[str(root)],
            output_repo_id="test/value_targets",
            video=False,
            push_to_hub=False,
            dry_run=True,
        )
    )

    assert result is None
    assert VALUE_GLOBAL_REMAINING_FRAMES_GT in columns
    assert VALUE_SUBTASK_REMAINING_NORM_GT in columns
    assert features[VALUE_GLOBAL_REMAINING_FRAMES_GT]["dtype"] == "int32"
    assert features[VALUE_SUBTASK_REMAINING_NORM_GT]["dtype"] == "float32"


def test_rerunning_targets_preserves_and_marks_existing_advantage_stale(tmp_path):
    root = _make_raw_run(tmp_path, [["pick"] * 5], annotation_subtasks=["pick"])
    prepare_value_targets(ValueTargetConfig(root=root, mode="global", global_scale="max"))
    compute_advantage(AdvantageConfig(root=root, value_mode="global", chunk_size=2))

    first_metadata = json.loads((root / VALUE_FUNCTION_META_FILENAME).read_text())
    advantage_stage = f"{ADVANTAGE_STAGE_PREFIX}.global"
    assert first_metadata["stages"][advantage_stage]["stale"] is False

    prepare_value_targets(ValueTargetConfig(root=root, mode="global", global_scale="p95"))
    metadata = json.loads((root / VALUE_FUNCTION_META_FILENAME).read_text())

    assert metadata["advantage"]["global"]["chunk_size"] == 2
    assert metadata["stages"][TARGET_STAGE]["stale"] is False
    assert metadata["stages"][advantage_stage]["stale"] is True


def test_strict_contract_rejects_missing_subtask_with_boundaries(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick", "pick", "place"], ["pick", "pick"]],
        annotation_subtasks=["pick", "place"],
    )

    with pytest.raises(ValueError, match=r"episode 1.*segments=.*missing=\['place'\]"):
        prepare_value_targets(ValueTargetConfig(root=root, mode="subtask"))


def test_strict_contract_rejects_repeated_subtask_segments(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick", "pick", "place", "pick"]],
        annotation_subtasks=["pick", "place"],
    )

    with pytest.raises(ValueError, match=r"episode 0.*repeated=.*pick.*\(0, 1\).*\(3, 3\)"):
        prepare_value_targets(ValueTargetConfig(root=root, mode="subtask"))


def test_strict_contract_rejects_swapped_subtask_order(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick", "place"], ["place", "pick"]],
        annotation_subtasks=["pick", "place"],
    )

    with pytest.raises(ValueError, match=r"episode 1.*observed=\['place', 'pick'\]"):
        prepare_value_targets(ValueTargetConfig(root=root, mode="subtask"))


def test_strict_contract_rejects_extra_unknown_subtask(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick", "place", "inspect"]],
        annotation_subtasks=["pick", "place"],
    )

    with pytest.raises(ValueError, match=r"not present.*inspect"):
        prepare_value_targets(ValueTargetConfig(root=root, mode="subtask"))


def test_first_complete_episode_defines_order_then_missing_episode_fails(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick", "pick"], ["place", "pick"]],
        annotation_subtasks=["pick", "place"],
    )

    with pytest.raises(
        ValueError, match=r"episode 0: canonical=\['place', 'pick'\].*missing=\['place'\]"
    ):
        prepare_value_targets(ValueTargetConfig(root=root, mode="subtask"))


def test_explicit_subtask_order_has_priority(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["place", "place", "pick"]],
        annotation_subtasks=["pick", "place"],
    )

    summary = prepare_value_targets(
        ValueTargetConfig(root=root, mode="subtask", subtask_order=["place", "pick"])
    )

    assert summary["subtask_order"] == ["place", "pick"]
    assert _extras(root).column(VALUE_SUBTASK_ID_GT).to_pylist() == [0, 0, 1]


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"num_bins": 0},
        {"num_bins": 1},
        {"num_bins": -1},
        {"global_num_bins": 0},
        {"global_num_bins": 1},
        {"subtask_num_bins": -2},
    ],
)
def test_all_bin_counts_must_be_at_least_two(tmp_path, config_kwargs):
    root = _make_raw_run(tmp_path, [["pick"]], annotation_subtasks=["pick"])

    with pytest.raises(ValueError, match="All bin counts must be >= 2"):
        prepare_value_targets(ValueTargetConfig(root=root, mode="both", **config_kwargs))


@pytest.mark.parametrize(
    ("scales", "expected"),
    [
        ({"pick": 2.0}, r"missing=\['place'\], extra=\[\]"),
        ({"pick": 2.0, "place": 1.0, "unused": 4.0}, r"missing=\[\], extra=\['unused'\]"),
    ],
)
def test_manual_subtask_scale_keys_must_match_canonical_names(tmp_path, scales, expected):
    root = _make_raw_run(
        tmp_path,
        [["pick", "pick", "place"]],
        annotation_subtasks=["pick", "place"],
    )

    with pytest.raises(ValueError, match=expected):
        prepare_value_targets(
            ValueTargetConfig(
                root=root,
                mode="subtask",
                subtask_scale="manual",
                subtask_scale_frames=scales,
            )
        )


def test_global_clip_rate_is_frame_weighted(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick"] * 3, ["pick"] * 7],
        annotation_subtasks=["pick"],
    )

    summary = prepare_value_targets(
        ValueTargetConfig(
            root=root,
            mode="global",
            global_scale="manual",
            global_scale_frames=2.0,
        )
    )

    assert summary["clip_summary"]["global"] == {
        "clipped_frames": 4,
        "eligible_frames": 10,
        "clip_rate": pytest.approx(0.4),
    }


def test_subtask_clip_rate_is_frame_weighted(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick", "pick", "place"], ["pick"] * 6 + ["place"]],
        annotation_subtasks=["pick", "place"],
    )

    summary = prepare_value_targets(
        ValueTargetConfig(
            root=root,
            mode="subtask",
            subtask_scale="manual",
            subtask_scale_frames={"pick": 2.0, "place": 1.0},
        )
    )

    assert summary["clip_summary"]["subtask_by_name"]["pick"] == {
        "clipped_frames": 3,
        "eligible_frames": 8,
        "clip_rate": pytest.approx(3.0 / 8.0),
    }


def test_success_only_rejects_explicit_failure(tmp_path):
    root = _make_raw_run(
        tmp_path,
        [["pick"]],
        annotation_subtasks=["pick"],
        episode_info_overrides={0: {"success": False}},
    )

    with pytest.raises(ValueError, match="not explicitly successful"):
        prepare_value_targets(ValueTargetConfig(root=root, mode="global"))


def test_success_only_declared_when_outcome_fields_are_absent(tmp_path):
    root = _make_raw_run(tmp_path, [["pick"]], annotation_subtasks=["pick"])

    summary = prepare_value_targets(ValueTargetConfig(root=root, mode="global"))
    metadata = json.loads((root / VALUE_FUNCTION_META_FILENAME).read_text())

    assert summary["success_validation"] == "declared_no_outcome_field"
    assert metadata["all_episodes_successful"] is True
    assert metadata["success_validation"] == "declared_no_outcome_field"


def test_failure_episode_mode_is_not_supported(tmp_path):
    root = _make_raw_run(tmp_path, [["pick"]], annotation_subtasks=["pick"])

    with pytest.raises(ValueError, match="failure/timeout/abort targets are not designed"):
        prepare_value_targets(
            ValueTargetConfig(root=root, mode="global", require_success_only=False)
        )


def test_rerun_mode_marks_preserved_old_target_columns_inactive(tmp_path):
    root = _make_raw_run(tmp_path, [["pick"] * 3], annotation_subtasks=["pick"])
    prepare_value_targets(ValueTargetConfig(root=root, mode="both"))

    summary = prepare_value_targets(ValueTargetConfig(root=root, mode="global"))

    assert VALUE_GLOBAL_REMAINING_FRAMES_GT in summary["target_columns"]["active"]
    assert VALUE_SUBTASK_REMAINING_FRAMES_GT in summary["target_columns"]["inactive_present"]
    assert VALUE_SUBTASK_REMAINING_FRAMES_GT in _extras(root).column_names
