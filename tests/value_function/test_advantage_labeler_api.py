import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.scripts.lerobot_advantage_labeler import ChunkCache, Handler, RawRun
from lerobot.value_function.advantage_labeling import advantage_columns
from lerobot.value_function.raw_io import fingerprint_raw_run_columns, update_stage_metadata
from lerobot.value_function.schema import (
    ADVANTAGE_GLOBAL_CHUNK,
    ADVANTAGE_GLOBAL_IS_VALID,
    ADVANTAGE_GLOBAL_VALID_HORIZON,
    ADVANTAGE_LABEL_GLOBAL,
    EXTRAS_FILENAME,
    RAW_FORMAT_VERSION,
)


def _write_run(tmp_path, frame_count=7):
    root = tmp_path / "raw_run"
    root.mkdir()
    (root / "run_meta.json").write_text(
        json.dumps(
            {
                "version": RAW_FORMAT_VERSION,
                "fps": 30,
                "task": "api test",
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
    pq.write_table(
        pa.table(
            {
                ADVANTAGE_GLOBAL_CHUNK: pa.array(
                    [float(frame_count - index) for index in range(frame_count)],
                    type=pa.float32(),
                ),
                ADVANTAGE_GLOBAL_VALID_HORIZON: pa.array(
                    [2] * (frame_count - 1) + [0], type=pa.int32()
                ),
                ADVANTAGE_GLOBAL_IS_VALID: pa.array(
                    [True] * (frame_count - 1) + [False], type=pa.bool_()
                ),
            }
        ),
        episode / EXTRAS_FILENAME,
    )
    columns = list(advantage_columns("global"))
    update_stage_metadata(
        root,
        "advantage.global",
        config={"value_mode": "global", "value_source": "model_pred"},
        input_columns=[],
        input_fingerprint=fingerprint_raw_run_columns(root, []),
        output_columns=columns,
        output_fingerprint=fingerprint_raw_run_columns(root, columns),
        prediction_source="model_pred",
        synthetic=False,
        metadata_patch={
            "advantage": {
                "global": {
                    "prediction_source": "model_pred",
                    "synthetic": False,
                    "experiment_eligible": True,
                }
            }
        },
    )
    return root


@contextmanager
def _serve(root):
    Handler.run = RawRun(root)
    Handler.cache = ChunkCache(root)
    Handler.value_mode = "global"
    Handler.top_percent = 0.8
    Handler.sort_order = "desc"
    Handler.tie_policy = "exact_count"
    Handler.allow_synthetic = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Handler.cache
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=10) as response:
            body = response.read()
            content_type = response.headers.get_content_type()
            return response.status, json.loads(body) if content_type == "application/json" else body
    except HTTPError as exc:
        body = exc.read()
        content_type = exc.headers.get_content_type()
        return exc.code, json.loads(body) if content_type == "application/json" else body


def test_static_meta_paginated_preview_and_safe_404(tmp_path):
    root = _write_run(tmp_path)
    with _serve(root) as (base, cache):
        status, index = _request(base, "/")
        assert status == 200
        assert b"Advantage Labeler" in index
        status, meta = _request(base, "/api/meta")
        assert status == 200
        assert meta["eligibility_by_mode"]["global"]["experiment_eligible"] is True
        assert meta["overrides_by_mode"] == {"global": {}, "subtask": {}}

        status, page = _request(base, "/api/chunks?page=2&page_size=3")
        assert status == 200
        assert (page["page"], page["page_size"], page["total"], len(page["items"])) == (
            2,
            3,
            7,
            3,
        )
        assert sum(page["counts"].values()) == 7
        assert cache.load_count == 1
        _request(base, "/api/chunks?page=1&page_size=3")
        assert cache.load_count == 1

        assert _request(base, "/api/chunks?page_size=501")[0] == 400
        assert _request(base, "/api/episode/0/img/not-a-camera/0")[0] == 404
        assert _request(base, "/missing")[0] == 404


def test_preview_validation_and_confirmed_export(tmp_path):
    root = _write_run(tmp_path)
    body = {
        "value_mode": "global",
        "top_percent": 0.5,
        "sort_order": "asc",
        "tie_policy": "exact_count",
        "overrides": {"ep_000000:frame_000001": "ignore"},
        "page": 1,
        "page_size": 3,
    }
    with _serve(root) as (base, cache):
        status, preview = _request(base, "/api/preview", body)
        assert status == 200
        assert preview["items"][0]["advantage"] < preview["items"][1]["advantage"]
        assert preview["threshold"]["positive_direction"] == "high"
        assert _request(
            base,
            "/api/preview",
            {**body, "overrides": {"ep_999999:frame_000000": "positive"}},
        )[0] == 400

        status, dry = _request(base, "/api/export-preview", body)
        assert status == 200
        assert dry["summary"]["dry_run"] is True
        assert dry["summary"]["change_summary"]["total"] == 7
        assert ADVANTAGE_LABEL_GLOBAL not in pq.read_table(
            root / "ep_000000" / EXTRAS_FILENAME
        ).column_names
        assert _request(base, "/api/export", body)[0] == 400

        status, exported = _request(base, "/api/export", {**body, "confirm": True})
        assert status == 200
        assert exported["summary"]["overrides"] == body["overrides"]
        labels = pq.read_table(root / "ep_000000" / EXTRAS_FILENAME).column(
            ADVANTAGE_LABEL_GLOBAL
        ).to_pylist()
        assert len(labels) == 7
        assert labels[1] == "ignore"
        assert cache.load_count == 1
        _request(base, "/api/chunks?page_size=3")
        assert cache.load_count == 2


def test_fifty_thousand_chunks_return_only_one_page(tmp_path):
    root = _write_run(tmp_path, frame_count=50_000)
    with _serve(root) as (base, _cache):
        status, page = _request(base, "/api/chunks?page=250&page_size=100")

    assert status == 200
    assert page["total"] == 50_000
    assert page["total_pages"] == 500
    assert len(page["items"]) == 100
    assert sum(page["counts"].values()) == 50_000


def test_frontend_uses_pagination_debounce_and_local_frame_updates():
    source = (
        __import__("pathlib").Path(__file__).parents[2]
        / "src/lerobot/scripts/advantage_labeler/app.js"
    ).read_text()
    select_chunk = source.split("function selectChunk", 1)[1].split(
        "function adjustOffset", 1
    )[0]

    assert "page_size: state.pageSize" in source
    assert "debounce(() => refresh({ resetPage: true }), 150)" in source
    assert "renderList" not in select_chunk
    assert "frameImage.src" in select_chunk
