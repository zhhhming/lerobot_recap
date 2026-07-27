import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

from PIL import Image

from lerobot.scripts.lerobot_compute_advantage_weights import Handler, RawRun, WeightCache
from lerobot.value_function.advantage_weights import AdvantageWeightConfig, compute_advantage_weights
from tests.value_function.test_advantage_weights import _write_run


@contextmanager
def _serve(root):
    Handler.run = RawRun(root)
    Handler.cache = WeightCache(root)
    Handler.value_mode = "global"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Handler.cache
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(base, path):
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(Request(base + path), timeout=10) as response:
            body = response.read()
            content_type = response.headers.get_content_type()
            return response.status, json.loads(body) if content_type == "application/json" else body
    except HTTPError as exc:
        body = exc.read()
        content_type = exc.headers.get_content_type()
        return exc.code, json.loads(body) if content_type == "application/json" else body


def _weighted_run(tmp_path, *, frame_count=8):
    root = _write_run(tmp_path, frame_count=frame_count)
    for mode in ("global", "subtask"):
        compute_advantage_weights(AdvantageWeightConfig(root=root, value_mode=mode))
    return root


def test_static_meta_groups_chunks_cache_and_clear_image_404(tmp_path):
    root = _weighted_run(tmp_path)
    with _serve(root) as (base, cache):
        status, index = _request(base, "/")
        assert status == 200
        assert b"Advantage Weight Inspector" in index
        status, meta = _request(base, "/api/meta")
        assert status == 200
        assert meta["modes"]["global"]["group_source"] == "value"

        status, groups = _request(base, "/api/groups?page=1&page_size=10")
        assert status == 200
        assert groups["total"] == 1
        group_id = groups["items"][0]["group_id"]
        assert cache.load_count == 1

        status, chunks = _request(
            base, f"/api/chunks?group_id={quote(group_id)}&page=1&page_size=3"
        )
        assert status == 200
        assert chunks["total"] == 8
        assert len(chunks["items"]) == 3
        assert chunks["items"][0]["positive_rank"] == 1.0
        assert cache.load_count == 1
        assert _request(base, "/api/chunks?group_id=missing")[0] == 400
        status, missing = _request(base, "/api/episode/0/img/third_person/0")
        assert status == 404
        assert "Image not found" in missing["error"]


def test_jpeg_image_endpoint(tmp_path):
    root = _weighted_run(tmp_path)
    metadata_path = root / "run_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["image_encoding"] = {
        "format": "jpeg",
        "extension": ".jpg",
        "quality": 95,
        "subsampling": 0,
    }
    metadata_path.write_text(json.dumps(metadata))
    camera = root / "ep_000000" / "third_person"
    camera.mkdir()
    Image.new("RGB", (8, 8), color=(20, 30, 40)).save(camera / "000000.jpg", quality=95)

    with _serve(root) as (base, _cache):
        status, image = _request(base, "/api/episode/0/img/third_person/0")

    assert status == 200
    assert image.startswith(b"\xff\xd8")


def test_fifty_thousand_chunks_are_paginated(tmp_path):
    root = _weighted_run(tmp_path, frame_count=50_000)
    with _serve(root) as (base, _cache):
        _, groups = _request(base, "/api/groups")
        group_id = groups["items"][0]["group_id"]
        status, chunks = _request(
            base, f"/api/chunks?group_id={quote(group_id)}&page=200&page_size=100"
        )
    assert status == 200
    assert chunks["total"] == 50_000
    assert len(chunks["items"]) == 100


def test_frontend_pages_groups_and_chunks_and_updates_image_locally():
    source = (
        __import__("pathlib").Path(__file__).parents[2]
        / "src/lerobot/scripts/advantage_weight_viz/app.js"
    ).read_text()
    select_chunk = source.split("function selectChunk", 1)[1].split(
        "function adjustOffset", 1
    )[0]
    assert "page_size=100" in source
    assert "page_size=150" in source
    assert "groupStrip.replaceChildren" in source
    assert "chunkList.replaceChildren" not in select_chunk
    assert "frameImage.src" in select_chunk
