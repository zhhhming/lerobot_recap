#!/usr/bin/env python
"""Local web-based subtask annotator for `lerobot-raw-record` runs.

Reads a raw run directory (the on-disk format produced by
`lerobot_raw_record.py`) and serves a small annotation UI. You define a set of
reusable, color-coded subtasks for the run and paint them onto each episode's
timeline (either by filling the region between two keyframes, or per-frame).

Annotations are stored as a human-readable sidecar `annotations.json` at the
run root. When you click "Export", a per-episode `extras.parquet` is written
into every episode directory with one string column (default name: "subtask").
`lerobot_build_dataset.py` already knows how to merge `extras.parquet` columns
into the final LeRobotDataset, so the annotated subtask becomes a per-frame
feature usable for training.

Usage:

    lerobot-annotate-subtask --root ~/.cache/huggingface/lerobot/raw/user/my_raw_data
    # then open http://127.0.0.1:8000

Only the Python standard library is used for the server; pyarrow is used for
reading/writing parquet. Both are already available in the lerobot env.
"""

import argparse
import json
import logging
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("subtask_annotator")

RUN_META_FILENAME = "run_meta.json"
CONFIG_FILENAME = "annotation_config.json"
ANNOTATIONS_FILENAME = "annotations.json"
STATIC_DIR = Path(__file__).resolve().parent / "subtask_annotator"

EP_RE = re.compile(r"^ep_(\d+)$")

# Raised when the browser cancels an in-flight request (e.g. rapid scrubbing
# replaces image <img>.src before the previous PNG finished sending). Harmless.
CONNECTION_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

# A pleasant default palette offered to the UI when creating new subtasks.
DEFAULT_PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
    "#76b7b2", "#edc948", "#ff9da7", "#9c755f", "#bab0ac",
]


class RawRun:
    """Thin reader over a raw run directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        meta_path = self.root / RUN_META_FILENAME
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"No {RUN_META_FILENAME} found in {self.root}. "
                f"Point --root at a directory produced by lerobot-raw-record."
            )
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.fps = int(self.meta.get("fps", 30))
        self.task = self.meta.get("task", "")
        self.robot_type = self.meta.get("robot_type", "")
        self.features = self.meta.get("features", {})

        self.image_keys = [
            k for k, v in self.features.items() if v.get("dtype") in ("image", "video")
        ]
        # camera subdir name == last dotted component, matching the recorder.
        self.cam_subdir = {k: k.split(".")[-1] for k in self.image_keys}

        self.action_names = list(self.features.get("action", {}).get("names") or [])
        self.state_names = list(
            self.features.get("observation.state", {}).get("names") or []
        )
        self._frame_cache: dict[int, dict] = {}

    # -- episodes -----------------------------------------------------------
    def episode_dir(self, idx: int) -> Path:
        return self.root / f"ep_{idx:06d}"

    def list_episodes(self) -> list[dict]:
        eps = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            m = EP_RE.match(d.name)
            if not m or not (d / "info.json").is_file():
                continue
            idx = int(m.group(1))
            length = None
            try:
                with open(d / "info.json") as f:
                    length = json.load(f).get("length")
            except Exception:
                pass
            eps.append({"index": idx, "length": length, "dir": d.name})
        eps.sort(key=lambda e: e["index"])
        return eps

    def load_frames(self, idx: int) -> dict:
        if idx in self._frame_cache:
            return self._frame_cache[idx]
        ep_dir = self.episode_dir(idx)
        fpath = ep_dir / "frames.parquet"
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing {fpath}")
        table = pq.read_table(fpath)
        cols = table.column_names
        rows = table.num_rows
        action = table.column("action").to_pylist() if "action" in cols else [None] * rows
        state = (
            table.column("observation.state").to_pylist()
            if "observation.state" in cols
            else [None] * rows
        )
        wall = table.column("wall_time_s").to_pylist() if "wall_time_s" in cols else None
        source = table.column("source").to_pylist() if "source" in cols else None
        data = {
            "length": rows,
            "action": action,
            "state": state,
            "wall_time_s": wall,
            "source": source,
        }
        self._frame_cache[idx] = data
        return data


# --------------------------------------------------------------------------
# Config + annotation persistence
# --------------------------------------------------------------------------
def load_config(run: RawRun) -> dict:
    path = run.root / CONFIG_FILENAME
    if path.is_file():
        with open(path) as f:
            cfg = json.load(f)
    else:
        cfg = {}
    cfg.setdefault("feature_name", "subtask")
    cfg.setdefault("default_value", "")
    cfg.setdefault("subtasks", [])  # [{name, color}]
    cfg.setdefault("palette", DEFAULT_PALETTE)
    return cfg


def save_config(run: RawRun, cfg: dict) -> None:
    with open(run.root / CONFIG_FILENAME, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def load_annotations(run: RawRun) -> dict:
    path = run.root / ANNOTATIONS_FILENAME
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {}


def save_annotations(run: RawRun, ann: dict) -> None:
    with open(run.root / ANNOTATIONS_FILENAME, "w") as f:
        json.dump(ann, f, indent=2, ensure_ascii=False)


def default_episode_annotation(length: int) -> dict:
    last = max(length - 1, 0)
    return {"keyframes": sorted({0, last}), "labels": [None] * length}


# --------------------------------------------------------------------------
# Export to extras.parquet
# --------------------------------------------------------------------------
def export_extras(run: RawRun) -> dict:
    cfg = load_config(run)
    ann = load_annotations(run)
    feature_name = cfg["feature_name"]
    default_value = cfg.get("default_value", "") or ""

    summary = {"episodes": [], "feature_name": feature_name, "total_unlabeled": 0}
    for ep in run.list_episodes():
        idx = ep["index"]
        frames = run.load_frames(idx)
        length = frames["length"]
        ep_ann = ann.get(str(idx)) or default_episode_annotation(length)
        labels = list(ep_ann.get("labels") or [])
        # Normalize length.
        if len(labels) < length:
            labels += [None] * (length - len(labels))
        labels = labels[:length]

        unlabeled = sum(1 for v in labels if v in (None, ""))
        values = [(v if v not in (None,) else default_value) for v in labels]

        ep_dir = run.episode_dir(idx)
        extras_path = ep_dir / "extras.parquet"

        # Merge with any pre-existing extras columns, replacing our own column.
        other_cols: dict[str, list] = {}
        if extras_path.is_file():
            try:
                existing = pq.read_table(extras_path)
                for name in existing.column_names:
                    if name == feature_name:
                        continue
                    col = existing.column(name).to_pylist()
                    if len(col) == length:
                        other_cols[name] = col
            except Exception:
                logger.warning("Could not read existing %s; overwriting.", extras_path)

        arrays = [pa.array(values, type=pa.string())]
        names = [feature_name]
        for name, col in other_cols.items():
            arrays.append(pa.array(col))
            names.append(name)
        table = pa.Table.from_arrays(arrays, names=names)
        pq.write_table(table, extras_path)

        summary["total_unlabeled"] += unlabeled
        summary["episodes"].append(
            {"index": idx, "length": length, "unlabeled": unlabeled, "path": str(extras_path)}
        )
    return summary


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    run: RawRun = None  # set on the server instance / class

    def log_message(self, fmt, *args):  # quieter logging
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers ------------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status=200, cache=True):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=86400")
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _safe_error(self, status, msg):
        """Send an error response, swallowing failures on an already-closed socket."""
        try:
            self._send_json({"error": msg}, status=status)
        except CONNECTION_ERRORS:
            pass

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_static(self, name: str):
        path = (STATIC_DIR / name).resolve()
        if not str(path).startswith(str(STATIC_DIR)) or not path.is_file():
            self._send_bytes(b"Not found", "text/plain", status=404, cache=False)
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        self._send_bytes(path.read_bytes(), ctype, cache=False)

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/" or path == "/index.html":
                self._send_static("index.html")
            elif path in ("/app.js", "/style.css"):
                self._send_static(path.lstrip("/"))
            elif path == "/api/meta":
                self._api_meta()
            elif path == "/api/config":
                self._send_json(load_config(self.run))
            elif path == "/api/annotations":
                self._send_json(load_annotations(self.run))
            elif (m := re.match(r"^/api/episode/(\d+)$", path)):
                self._api_episode(int(m.group(1)))
            elif (m := re.match(r"^/api/episode/(\d+)/img/([^/]+)/(\d+)$", path)):
                self._api_image(int(m.group(1)), m.group(2), int(m.group(3)))
            else:
                self._send_bytes(b"Not found", "text/plain", status=404, cache=False)
        except CONNECTION_ERRORS:
            return  # client canceled the request (e.g. fast scrubbing) — ignore
        except FileNotFoundError as e:
            self._safe_error(404, str(e))
        except Exception as e:
            logger.exception("GET %s failed", path)
            self._safe_error(500, str(e))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/config":
                cfg = self._read_json_body()
                save_config(self.run, cfg)
                self._send_json({"ok": True})
            elif (m := re.match(r"^/api/episode/(\d+)/annotation$", path)):
                self._api_save_annotation(int(m.group(1)))
            elif path == "/api/export":
                summary = export_extras(self.run)
                self._send_json({"ok": True, "summary": summary})
            else:
                self._send_bytes(b"Not found", "text/plain", status=404, cache=False)
        except CONNECTION_ERRORS:
            return  # client canceled the request — ignore
        except Exception as e:
            logger.exception("POST %s failed", path)
            self._safe_error(500, str(e))

    # -- api impls ----------------------------------------------------------
    def _api_meta(self):
        run = self.run
        self._send_json(
            {
                "root": str(run.root),
                "fps": run.fps,
                "task": run.task,
                "robot_type": run.robot_type,
                "cameras": [
                    {"key": k, "subdir": run.cam_subdir[k], "label": run.cam_subdir[k]}
                    for k in run.image_keys
                ],
                "action_names": run.action_names,
                "state_names": run.state_names,
                "episodes": run.list_episodes(),
                "config": load_config(run),
            }
        )

    def _api_episode(self, idx: int):
        run = self.run
        frames = run.load_frames(idx)
        ann = load_annotations(run)
        ep_ann = ann.get(str(idx))
        if ep_ann is None:
            ep_ann = default_episode_annotation(frames["length"])
        else:
            # Make sure labels length matches.
            labels = list(ep_ann.get("labels") or [])
            length = frames["length"]
            if len(labels) < length:
                labels += [None] * (length - len(labels))
            ep_ann = {
                "keyframes": ep_ann.get("keyframes") or [0, max(length - 1, 0)],
                "labels": labels[:length],
            }
        self._send_json(
            {
                "index": idx,
                "length": frames["length"],
                "wall_time_s": frames["wall_time_s"],
                "source": frames["source"],
                "action": frames["action"],
                "state": frames["state"],
                "annotation": ep_ann,
            }
        )

    def _api_image(self, idx: int, cam: str, frame: int):
        run = self.run
        ep_dir = run.episode_dir(idx)
        # cam may be the subdir name directly.
        path = ep_dir / cam / f"{frame:06d}.png"
        if not path.is_file():
            self._send_bytes(b"Not found", "text/plain", status=404, cache=False)
            return
        self._send_bytes(path.read_bytes(), "image/png", cache=True)

    def _api_save_annotation(self, idx: int):
        body = self._read_json_body()
        ann = load_annotations(self.run)
        ann[str(idx)] = {
            "keyframes": body.get("keyframes", []),
            "labels": body.get("labels", []),
        }
        save_annotations(self.run, ann)
        self._send_json({"ok": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, help="Raw run directory (contains run_meta.json)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = RawRun(Path(args.root))
    Handler.run = run

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    logger.info("Serving subtask annotator for %s", run.root)
    logger.info("Episodes: %d | Task: %s", len(run.list_episodes()), run.task)
    logger.info("Open %s", url)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
