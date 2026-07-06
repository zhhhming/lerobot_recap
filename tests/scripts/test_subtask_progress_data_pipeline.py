#!/usr/bin/env python

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.processor.converters import batch_to_transition
from lerobot.scripts.lerobot_annotate_subtask import (
    ANNOTATIONS_FILENAME,
    CONFIG_FILENAME,
    RUN_META_FILENAME,
    RawRun,
    export_extras,
)
from lerobot.scripts.lerobot_build_dataset import _load_extras_row_as_dict, _load_extras_schema
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _make_raw_run(tmp_path, labels, *, existing_extras: dict[str, list] | None = None) -> RawRun:
    root = tmp_path / "raw_run"
    ep_dir = root / "ep_000000"
    ep_dir.mkdir(parents=True)
    length = len(labels)

    _write_json(
        root / RUN_META_FILENAME,
        {
            "fps": 30,
            "task": "test task",
            "robot_type": "test_robot",
            "features": {},
        },
    )
    _write_json(root / CONFIG_FILENAME, {"feature_name": "subtask", "default_value": ""})
    _write_json(root / ANNOTATIONS_FILENAME, {"0": {"labels": labels}})
    _write_json(ep_dir / "info.json", {"length": length})

    pq.write_table(
        pa.Table.from_arrays(
            [pa.array([[0.0]] * length, type=pa.list_(pa.float32(), 1))],
            names=["action"],
        ),
        ep_dir / "frames.parquet",
    )
    if existing_extras:
        pq.write_table(
            pa.Table.from_pydict(existing_extras),
            ep_dir / "extras.parquet",
        )

    return RawRun(root)


def test_export_extras_adds_subtask_progress_and_preserves_other_columns(tmp_path):
    run = _make_raw_run(
        tmp_path,
        ["pick", "pick", "", "place", "place", "place"],
        existing_extras={
            "subtask": ["old"] * 6,
            "subtask_progress": [9.0] * 6,
            "operator_note": ["keep"] * 6,
        },
    )

    summary = export_extras(run)

    assert summary["episodes"][0]["length"] == 6
    table = pq.read_table(run.episode_dir(0) / "extras.parquet")
    assert table.column_names == ["subtask", "subtask_progress", "operator_note"]
    assert table.column("subtask").to_pylist() == ["pick", "pick", "", "place", "place", "place"]
    assert table.column("subtask_progress").to_pylist() == pytest.approx(
        [0.5, 1.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    )
    assert table.column("operator_note").to_pylist() == ["keep"] * 6
    assert table.schema.field("subtask_progress").type.equals(pa.float32())


def test_export_extras_resets_progress_for_empty_and_changed_labels(tmp_path):
    run = _make_raw_run(tmp_path, [None, "pick", "place", "place", "pick", ""])

    export_extras(run)

    table = pq.read_table(run.episode_dir(0) / "extras.parquet")
    assert table.column("subtask").to_pylist() == ["", "pick", "place", "place", "pick", ""]
    assert table.column("subtask_progress").to_pylist() == pytest.approx(
        [0.0, 1.0, 0.5, 1.0, 1.0, 0.0]
    )


def test_load_extras_schema_maps_float32_progress_dtype(tmp_path):
    run = _make_raw_run(tmp_path, ["pick", "pick"])
    export_extras(run)

    features, columns = _load_extras_schema([run.episode_dir(0)])

    assert columns == {"subtask", "subtask_progress"}
    assert features["subtask"] == {"dtype": "string", "shape": (1,), "names": None}
    assert features["subtask_progress"] == {"dtype": "float32", "shape": (1,), "names": None}


def test_load_extras_row_wraps_scalar_progress_as_numpy_array():
    row = {"subtask": "pick", "subtask_progress": 0.5}
    features = {
        "subtask": {"dtype": "string", "shape": (1,), "names": None},
        "subtask_progress": {"dtype": "float32", "shape": (1,), "names": None},
    }

    converted = _load_extras_row_as_dict(row, features)

    assert converted["subtask"] == "pick"
    assert isinstance(converted["subtask_progress"], np.ndarray)
    assert converted["subtask_progress"].dtype == np.float32
    assert converted["subtask_progress"].shape == (1,)
    assert converted["subtask_progress"].item() == pytest.approx(0.5)


def test_batch_to_transition_routes_subtask_progress():
    batch = {
        ACTION: torch.zeros(1, 2),
        "task": ["test task"],
        "subtask": ["pick"],
        "subtask_progress": torch.tensor([0.5]),
    }

    transition = batch_to_transition(batch)

    complementary_data = transition[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary_data["subtask"] == ["pick"]
    assert torch.equal(complementary_data["subtask_progress"], torch.tensor([0.5]))
