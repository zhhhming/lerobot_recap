import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from lerobot.scripts.lerobot_value_viz import ValueRun, _compressed_intervals, _sample_indices
from lerobot.value_function.schema import (
    EXTRAS_FILENAME,
    RAW_FORMAT_VERSION,
    VALUE_GLOBAL_ELAPSED_FRAMES_GT,
    VALUE_GLOBAL_ELAPSED_NORM_GT,
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_GLOBAL_REMAINING_FRAMES_PRED,
    VALUE_GLOBAL_REMAINING_NORM_GT,
    VALUE_GLOBAL_REMAINING_NORM_PRED,
    VALUE_SUBTASK_ELAPSED_FRAMES_GT,
    VALUE_SUBTASK_ELAPSED_NORM_GT,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_ID_PRED_SMOOTH,
    VALUE_SUBTASK_NAME_GT,
    VALUE_SUBTASK_NAME_PRED_SMOOTH,
    VALUE_SUBTASK_REMAINING_FRAMES_GT,
    VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD,
    VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD,
    VALUE_SUBTASK_REMAINING_NORM_GT,
    VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD,
    VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD,
)


def _columns(frame_count: int = 8) -> dict[str, pa.Array]:
    ids_gt = [0, 0, 0, 1, 1, 1, 1, 1][:frame_count]
    ids_smooth = [0, 0, 1, 1, 1, 1, 1, 1][:frame_count]
    remaining = list(reversed(range(frame_count)))
    subtask_remaining = [2, 1, 0, 4, 3, 2, 1, 0][:frame_count]
    return {
        VALUE_GLOBAL_REMAINING_FRAMES_GT: pa.array(remaining, type=pa.int32()),
        VALUE_GLOBAL_REMAINING_NORM_GT: pa.array(
            [value / max(frame_count - 1, 1) for value in remaining], type=pa.float32()
        ),
        VALUE_GLOBAL_REMAINING_FRAMES_PRED: pa.array(
            [value + 0.25 for value in remaining], type=pa.float32()
        ),
        VALUE_GLOBAL_REMAINING_NORM_PRED: pa.array(
            [(value + 0.25) / max(frame_count - 1, 1) for value in remaining], type=pa.float32()
        ),
        VALUE_GLOBAL_ELAPSED_FRAMES_GT: pa.array(range(frame_count), type=pa.int32()),
        VALUE_GLOBAL_ELAPSED_NORM_GT: pa.array(
            [value / max(frame_count - 1, 1) for value in range(frame_count)], type=pa.float32()
        ),
        VALUE_SUBTASK_ID_GT: pa.array(ids_gt, type=pa.int32()),
        VALUE_SUBTASK_NAME_GT: pa.array(
            ["pick" if value == 0 else "place" for value in ids_gt], type=pa.string()
        ),
        VALUE_SUBTASK_ID_PRED_SMOOTH: pa.array(ids_smooth, type=pa.int32()),
        VALUE_SUBTASK_NAME_PRED_SMOOTH: pa.array(
            ["pick" if value == 0 else "place" for value in ids_smooth], type=pa.string()
        ),
        VALUE_SUBTASK_REMAINING_FRAMES_GT: pa.array(subtask_remaining, type=pa.float32()),
        VALUE_SUBTASK_REMAINING_NORM_GT: pa.array(
            [value / 4 for value in subtask_remaining], type=pa.float32()
        ),
        VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD: pa.array(
            [value + 0.1 for value in subtask_remaining], type=pa.float32()
        ),
        VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD: pa.array(
            [(value + 0.1) / 4 for value in subtask_remaining], type=pa.float32()
        ),
        VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD: pa.array(
            [value + 0.2 for value in subtask_remaining], type=pa.float32()
        ),
        VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD: pa.array(
            [(value + 0.2) / 4 for value in subtask_remaining], type=pa.float32()
        ),
        VALUE_SUBTASK_ELAPSED_FRAMES_GT: pa.array([0, 1, 2, 0, 1, 2, 3, 4][:frame_count], type=pa.float32()),
        VALUE_SUBTASK_ELAPSED_NORM_GT: pa.array(
            [0, 0.25, 0.5, 0, 0.25, 0.5, 0.75, 1][:frame_count], type=pa.float32()
        ),
    }


def write_value_viz_run(
    tmp_path: Path,
    *,
    complete: bool = True,
    frame_count: int = 8,
    image_format: str = "png",
) -> Path:
    root = tmp_path / (f"complete_run_{image_format}" if complete else f"missing_run_{image_format}")
    root.mkdir()
    features = {
        "action": {"dtype": "float32", "shape": [1], "names": ["a"]},
        **{
            f"observation.images.{name}": {
                "dtype": "video",
                "shape": [8, 8, 3],
                "names": ["height", "width", "channels"],
            }
            for name in ("left_wrist", "third_person", "right_wrist")
        },
    }
    run_meta = {
        "version": RAW_FORMAT_VERSION,
        "fps": 20,
        "task": "value viz test",
        "robot_type": "bi_nero_follower",
        "features": features,
    }
    if image_format == "jpeg":
        run_meta["image_encoding"] = {
            "format": "jpeg",
            "extension": ".jpg",
            "quality": 95,
            "subsampling": 0,
        }
    (root / "run_meta.json").write_text(json.dumps(run_meta))
    for episode_index in range(2):
        episode = root / f"ep_{episode_index:06d}"
        episode.mkdir()
        (episode / "info.json").write_text(json.dumps({"length": frame_count}))
        pq.write_table(
            pa.table(
                {
                    "frame_index": pa.array(range(frame_count), type=pa.int64()),
                    "action": pa.array([[0.0]] * frame_count, type=pa.list_(pa.float32(), 1)),
                }
            ),
            episode / "frames.parquet",
        )
        columns = (
            _columns(frame_count)
            if complete
            else {
                VALUE_SUBTASK_ID_GT: pa.array([0] * frame_count, type=pa.int32()),
                VALUE_SUBTASK_NAME_GT: pa.array(["pick"] * frame_count, type=pa.string()),
            }
        )
        pq.write_table(pa.table(columns), episode / EXTRAS_FILENAME)
        for camera in ("left_wrist", "third_person", "right_wrist"):
            directory = episode / camera
            directory.mkdir()
            for frame in range(frame_count):
                if image_format == "jpeg":
                    Image.new("RGB", (8, 8), color=(frame, frame, frame)).save(
                        directory / f"{frame:06d}.jpg",
                        quality=95,
                        subsampling=0,
                    )
                else:
                    (directory / f"{frame:06d}.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    if complete:
        (root / "value_function_meta.json").write_text(
            json.dumps(
                {
                    "stages": {
                        "value_inference.global": {
                            "created_at": "2026-07-14T00:00:00Z",
                            "config": {"checkpoint": "/tmp/global.pt"},
                            "prediction_source": "model_pred",
                            "synthetic": False,
                            "stale": False,
                        },
                        "value_inference.subtask": {
                            "created_at": "2026-07-14T00:00:00Z",
                            "config": {"checkpoint": "/tmp/subtask.pt"},
                            "prediction_source": "model_pred",
                            "synthetic": False,
                            "stale": False,
                        },
                    }
                }
            )
        )
    return root


def test_series_boundaries_exact_frame_and_read_only(tmp_path):
    root = write_value_viz_run(tmp_path)
    extras = root / "ep_000000" / EXTRAS_FILENAME
    before = extras.read_bytes()
    run = ValueRun(root, chunk_size=4)

    meta = run.metadata()
    assert len(meta["episodes"]) == 2
    assert [camera["subdir"] for camera in meta["cameras"]] == [
        "left_wrist",
        "third_person",
        "right_wrist",
    ]
    assert all(series["available"] for series in meta["series"])
    assert meta["provenance"]["global"]["status"] == "current"

    gt = run.curves(0, unit="norm", boundary="gt", max_points=5)
    assert gt["sampled_points"] == 5
    assert gt["subtask_intervals"] == [
        {"start": 0, "end": 2, "id": 0, "name": "pick"},
        {"start": 3, "end": 7, "id": 1, "name": "place"},
    ]
    pred = run.curves(0, unit="frames", boundary="pred_smooth", max_points=8)
    assert pred["subtask_intervals"][0]["end"] == 1
    assert pred["subtask_intervals"][1]["start"] == 2

    frame = run.frame(0, 1, boundary="gt")
    assert frame["values"]["global_remaining_pred"]["frames"] == pytest.approx(6.25)
    assert frame["chunk_end"] == 5
    assert [(item["start"], item["end"]) for item in frame["chunk_segments"]] == [
        (1, 2),
        (3, 5),
    ]
    assert run.image_path(0, "third_person", 1).name == "000001.png"
    assert extras.read_bytes() == before


def test_missing_columns_are_unavailable_not_errors(tmp_path):
    root = write_value_viz_run(tmp_path, complete=False)
    run = ValueRun(root)
    meta = run.metadata()
    assert all(not series["available"] for series in meta["series"])
    assert meta["provenance"]["global"]["status"] == "missing"
    curves = run.curves(0, unit="norm", boundary="gt")
    assert all(not curve["available"] and curve["points"] == [] for curve in curves["curves"])
    assert curves["subtask_intervals"] == [{"start": 0, "end": 7, "id": 0, "name": "pick"}]
    assert all(
        units == {"norm": None, "frames": None} for units in run.frame(0, 0, boundary="gt")["values"].values()
    )


def test_jpeg_backed_value_run_resolves_jpg_images(tmp_path):
    run = ValueRun(write_value_viz_run(tmp_path, image_format="jpeg"))

    assert run.image_encoding.mime_type == "image/jpeg"
    assert run.image_path(0, "third_person", 1).name == "000001.jpg"


def test_sampling_is_bounded_and_preserves_endpoints():
    indices = _sample_indices(50_000, 2_000)
    assert len(indices) == 2_000
    assert indices[0] == 0
    assert indices[-1] == 49_999
    assert indices == sorted(set(indices))
    assert _compressed_intervals([0, 0, 1], ["a", "a", "b"])[-1]["end"] == 2


def test_invalid_requests_are_explicit(tmp_path):
    run = ValueRun(write_value_viz_run(tmp_path))
    with pytest.raises(ValueError, match="Unknown value unit"):
        run.curves(0, unit="seconds", boundary="gt")
    with pytest.raises(ValueError, match="Unknown boundary"):
        run.curves(0, unit="norm", boundary="mixed")
    with pytest.raises(FileNotFoundError, match="Unknown camera"):
        run.image_path(0, "../third_person", 0)
    with pytest.raises(FileNotFoundError, match="outside episode"):
        run.frame(0, 100, boundary="gt")


def test_frontend_has_frame_chart_value_and_keyboard_sync():
    source = (Path(__file__).parents[2] / "src/lerobot/scripts/value_viz/app.js").read_text()
    set_frame = source.split("async function setFrame", 1)[1].split("function updateImage", 1)[0]
    assert "updateImage()" in set_frame
    assert "/frame/${clamped}" in set_frame
    assert "renderCurrentValues()" in set_frame
    assert "renderChart()" in set_frame
    assert 'event.key !== "ArrowLeft"' in source
    assert "chunk_segments" in source
    assert "unavailable" in source
