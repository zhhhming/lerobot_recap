import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

from lerobot.scripts.lerobot_value_viz import Handler, ValueRun
from tests.value_function.test_value_viz_data import write_value_viz_run


@contextmanager
def _serve(root):
    Handler.run = ValueRun(root, chunk_size=4)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(base, path):
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(base + path, timeout=10) as response:
            body = response.read()
            return (
                response.status,
                (json.loads(body) if response.headers.get_content_type() == "application/json" else body),
            )
    except HTTPError as exc:
        body = exc.read()
        return (
            exc.code,
            json.loads(body) if exc.headers.get_content_type() == "application/json" else body,
        )


def test_static_meta_curves_frame_image_and_safe_errors(tmp_path):
    root = write_value_viz_run(tmp_path)
    extras = root / "ep_000000" / "extras.parquet"
    before = extras.read_bytes()
    with _serve(root) as base:
        status, index = _request(base, "/")
        assert status == 200
        assert b"Value Curves" in index
        assert _request(base, "/app.js")[0] == 200
        assert _request(base, "/style.css")[0] == 200

        status, meta = _request(base, "/api/meta")
        assert status == 200
        assert meta["episodes"][0] == {"index": 0, "name": "ep_000000", "length": 8}
        assert meta["provenance"]["subtask"]["status"] == "current"

        status, curves = _request(base, "/api/episode/0/curves?unit=norm&boundary=pred_smooth&max_points=5")
        assert status == 200
        assert curves["sampled_points"] == 5
        assert curves["subtask_intervals"][0]["end"] == 1

        status, frame = _request(base, "/api/episode/0/frame/1?boundary=gt")
        assert status == 200
        assert frame["frame"] == 1
        assert len(frame["chunk_segments"]) == 2

        status, image = _request(base, "/api/episode/0/img/third_person/1")
        assert status == 200
        assert image.startswith(b"\x89PNG")

        assert _request(base, "/api/episode/0/curves?max_points=1")[0] == 400
        assert _request(base, "/api/episode/0/curves?unit=seconds")[0] == 400
        assert _request(base, "/api/episode/0/frame/99")[0] == 404
        assert _request(base, "/api/episode/0/img/not-a-camera/0")[0] == 404
        assert _request(base, "/missing")[0] == 404
    assert extras.read_bytes() == before


def test_jpeg_image_endpoint(tmp_path):
    root = write_value_viz_run(tmp_path, image_format="jpeg")
    with _serve(root) as base:
        status, image = _request(base, "/api/episode/0/img/third_person/1")

    assert status == 200
    assert image.startswith(b"\xff\xd8")
