#!/usr/bin/env python

"""Serve a read-only local UI for raw-run value curves and frame images."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import threading
import webbrowser
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.raw_media import raw_frame_image_path, raw_image_encoding_from_meta
from lerobot.value_function.raw_io import (
    camera_subdir_name,
    discover_episodes,
    get_image_keys,
    read_run_meta,
    read_value_function_metadata,
)
from lerobot.value_function.schema import (
    EXTRAS_FILENAME,
    VALUE_GLOBAL_ELAPSED_FRAMES_GT,
    VALUE_GLOBAL_ELAPSED_NORM_GT,
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_GLOBAL_REMAINING_FRAMES_PRED,
    VALUE_GLOBAL_REMAINING_NORM_GT,
    VALUE_GLOBAL_REMAINING_NORM_PRED,
    VALUE_INFERENCE_STAGE_PREFIX,
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

logger = logging.getLogger("value_viz")
STATIC_DIR = Path(__file__).resolve().parent / "value_viz"
CONNECTION_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_MAX_POINTS = 2_000
MAX_MAX_POINTS = 10_000


SERIES: tuple[dict[str, str], ...] = (
    {
        "id": "global_remaining_gt",
        "label": "Global remaining GT",
        "mode": "global",
        "kind": "remaining",
        "norm": VALUE_GLOBAL_REMAINING_NORM_GT,
        "frames": VALUE_GLOBAL_REMAINING_FRAMES_GT,
    },
    {
        "id": "global_remaining_pred",
        "label": "Global remaining prediction",
        "mode": "global",
        "kind": "remaining",
        "norm": VALUE_GLOBAL_REMAINING_NORM_PRED,
        "frames": VALUE_GLOBAL_REMAINING_FRAMES_PRED,
    },
    {
        "id": "global_elapsed_gt",
        "label": "Global elapsed GT",
        "mode": "global",
        "kind": "elapsed",
        "norm": VALUE_GLOBAL_ELAPSED_NORM_GT,
        "frames": VALUE_GLOBAL_ELAPSED_FRAMES_GT,
    },
    {
        "id": "subtask_remaining_gt",
        "label": "Subtask remaining GT",
        "mode": "subtask",
        "kind": "remaining",
        "norm": VALUE_SUBTASK_REMAINING_NORM_GT,
        "frames": VALUE_SUBTASK_REMAINING_FRAMES_GT,
    },
    {
        "id": "subtask_remaining_gt_head",
        "label": "Subtask prediction (GT-conditioned head)",
        "mode": "subtask",
        "kind": "remaining",
        "norm": VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD,
        "frames": VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD,
    },
    {
        "id": "subtask_remaining_smooth_head",
        "label": "Subtask prediction (smoothed head)",
        "mode": "subtask",
        "kind": "remaining",
        "norm": VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD,
        "frames": VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD,
    },
    {
        "id": "subtask_elapsed_gt",
        "label": "Subtask elapsed GT",
        "mode": "subtask",
        "kind": "elapsed",
        "norm": VALUE_SUBTASK_ELAPSED_NORM_GT,
        "frames": VALUE_SUBTASK_ELAPSED_FRAMES_GT,
    },
)

BOUNDARIES: tuple[dict[str, str], ...] = (
    {
        "id": "gt",
        "label": "GT subtask",
        "id_column": VALUE_SUBTASK_ID_GT,
        "name_column": VALUE_SUBTASK_NAME_GT,
    },
    {
        "id": "pred_smooth",
        "label": "Predicted smoothed subtask",
        "id_column": VALUE_SUBTASK_ID_PRED_SMOOTH,
        "name_column": VALUE_SUBTASK_NAME_PRED_SMOOTH,
    },
)


def _positive_int(value: str | int | None, *, default: int, maximum: int) -> int:
    parsed = int(default if value is None else value)
    if parsed < 2 or parsed > maximum:
        raise ValueError(f"Expected an integer in [2, {maximum}], got {parsed}")
    return parsed


def _safe_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(value) if isinstance(value, int) else number
    return None


def _sample_indices(length: int, max_points: int) -> list[int]:
    """Return deterministic, bounded indices while always preserving both endpoints."""

    if length <= 0:
        return []
    if length <= max_points:
        return list(range(length))
    return [round(index * (length - 1) / (max_points - 1)) for index in range(max_points)]


def _compressed_intervals(ids: Sequence[Any], names: Sequence[Any] | None = None) -> list[dict]:
    if not ids:
        return []
    if names is not None and len(names) != len(ids):
        raise ValueError("Subtask ID/name columns have different lengths")
    intervals: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(ids) + 1):
        if index < len(ids) and ids[index] == ids[start]:
            continue
        raw_id = ids[start]
        raw_name = names[start] if names is not None else None
        intervals.append(
            {
                "start": start,
                "end": index - 1,
                "id": int(raw_id) if raw_id is not None else None,
                "name": str(raw_name) if raw_name not in (None, "") else f"Subtask {raw_id}",
            }
        )
        start = index
    return intervals


def _chunk_segments(
    intervals: Sequence[Mapping[str, Any]], *, frame: int, chunk_size: int, frame_count: int
) -> list[dict]:
    if chunk_size <= 0 or frame_count <= 0:
        return []
    chunk_end = min(frame + chunk_size, frame_count - 1)
    segments = []
    for interval in intervals:
        start = max(frame, int(interval["start"]))
        end = min(chunk_end, int(interval["end"]))
        if start <= end:
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "id": interval["id"],
                    "name": interval["name"],
                }
            )
    return segments


class ValueRun:
    """Validated read-only view of one raw run."""

    def __init__(self, root: str | Path, *, chunk_size: int = 50) -> None:
        self.root = Path(root).expanduser().resolve()
        self.meta = read_run_meta(self.root)
        self.image_encoding = raw_image_encoding_from_meta(self.meta)
        self.episodes = discover_episodes(self.root)
        self.episodes_by_index = {episode.index: episode for episode in self.episodes}
        self.fps = int(self.meta.get("fps", 30))
        self.task = str(self.meta.get("task", ""))
        self.robot_type = str(self.meta.get("robot_type", ""))
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        image_keys = get_image_keys(self.meta)
        self.cameras = [
            {"key": key, "subdir": camera_subdir_name(key), "label": camera_subdir_name(key)}
            for key in image_keys
        ]
        self.camera_subdirs = {camera["subdir"] for camera in self.cameras}
        self._cache: dict[int, tuple[tuple[int, int], pa.Table]] = {}
        self._schema_names = self._validate_extras_schemas()

    def _validate_extras_schemas(self) -> set[str]:
        schemas: list[tuple[Path, pa.Schema]] = []
        for episode in self.episodes:
            path = episode.path / EXTRAS_FILENAME
            if not path.is_file():
                continue
            schema = pq.read_schema(path)
            schemas.append((path, schema))
        if not schemas:
            return set()
        first_path, first_schema = schemas[0]
        for path, schema in schemas[1:]:
            if schema != first_schema:
                raise ValueError(f"extras.parquet schema differs between {first_path} and {path}")
        if len(schemas) != len(self.episodes):
            raise ValueError("Some episodes have extras.parquet and others do not")
        return set(first_schema.names)

    def episode(self, index: int):
        try:
            return self.episodes_by_index[index]
        except KeyError as exc:
            raise FileNotFoundError(f"Episode {index} does not exist") from exc

    def table(self, index: int) -> pa.Table:
        episode = self.episode(index)
        path = episode.path / EXTRAS_FILENAME
        if not path.is_file():
            return pa.table({})
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        cached = self._cache.get(index)
        if cached is None or cached[0] != signature:
            table = pq.read_table(path)
            if table.num_rows != episode.frame_count:
                raise ValueError(
                    f"{path} length ({table.num_rows}) does not match frame count ({episode.frame_count})"
                )
            self._cache[index] = (signature, table)
        return self._cache[index][1]

    def _provenance(self) -> dict[str, Any]:
        try:
            metadata = read_value_function_metadata(self.root)
        except FileNotFoundError:
            metadata = {}
        stages = metadata.get("stages") if isinstance(metadata, Mapping) else {}
        stages = stages if isinstance(stages, Mapping) else {}
        result = {}
        for mode in ("global", "subtask"):
            name = f"{VALUE_INFERENCE_STAGE_PREFIX}.{mode}"
            record = stages.get(name)
            if not isinstance(record, Mapping):
                result[mode] = {
                    "stage": name,
                    "status": "missing",
                    "warning": "No model inference provenance is recorded.",
                }
                continue
            warning = None
            status = "current"
            if record.get("stale"):
                status = "stale"
                warning = str(record.get("stale_reason") or "Inference artifact is stale.")
            elif record.get("synthetic"):
                status = "synthetic"
                warning = "Synthetic prediction: interface smoke only, not for experiments."
            result[mode] = {
                "stage": name,
                "status": status,
                "warning": warning,
                "prediction_source": record.get("prediction_source"),
                "synthetic": bool(record.get("synthetic")),
                "created_at": record.get("created_at"),
                "checkpoint": (record.get("config") or {}).get("checkpoint"),
            }
        return result

    def metadata(self) -> dict[str, Any]:
        available_series = []
        for definition in SERIES:
            entry = {key: definition[key] for key in ("id", "label", "mode", "kind")}
            entry["columns"] = {
                unit: definition[unit] if definition[unit] in self._schema_names else None
                for unit in ("norm", "frames")
            }
            entry["available"] = any(entry["columns"].values())
            available_series.append(entry)
        boundaries = []
        for definition in BOUNDARIES:
            available = definition["id_column"] in self._schema_names
            boundaries.append(
                {
                    "id": definition["id"],
                    "label": definition["label"],
                    "available": available,
                    "id_column": definition["id_column"] if available else None,
                    "name_column": (
                        definition["name_column"] if definition["name_column"] in self._schema_names else None
                    ),
                }
            )
        return {
            "root": str(self.root),
            "fps": self.fps,
            "task": self.task,
            "robot_type": self.robot_type,
            "chunk_size": self.chunk_size,
            "cameras": self.cameras,
            "episodes": [
                {"index": ep.index, "name": ep.path.name, "length": ep.frame_count} for ep in self.episodes
            ],
            "series": available_series,
            "boundaries": boundaries,
            "provenance": self._provenance(),
        }

    def _boundary_intervals(self, table: pa.Table, boundary: str) -> list[dict]:
        definition = next((item for item in BOUNDARIES if item["id"] == boundary), None)
        if definition is None:
            raise ValueError(f"Unknown boundary source: {boundary}")
        if definition["id_column"] not in table.column_names:
            return []
        ids = table.column(definition["id_column"]).to_pylist()
        names = (
            table.column(definition["name_column"]).to_pylist()
            if definition["name_column"] in table.column_names
            else None
        )
        return _compressed_intervals(ids, names)

    def curves(
        self, index: int, *, unit: str, boundary: str, max_points: int = DEFAULT_MAX_POINTS
    ) -> dict[str, Any]:
        if unit not in {"norm", "frames"}:
            raise ValueError(f"Unknown value unit: {unit}")
        episode = self.episode(index)
        table = self.table(index)
        indices = _sample_indices(episode.frame_count, max_points)
        curves = []
        for definition in SERIES:
            column = definition[unit]
            available = column in table.column_names
            values = table.column(column).to_pylist() if available else []
            curves.append(
                {
                    "id": definition["id"],
                    "label": definition["label"],
                    "mode": definition["mode"],
                    "kind": definition["kind"],
                    "column": column if available else None,
                    "available": available,
                    "points": (
                        [[frame, _safe_number(values[frame])] for frame in indices] if available else []
                    ),
                }
            )
        intervals = self._boundary_intervals(table, boundary)
        return {
            "episode_index": episode.index,
            "frame_count": episode.frame_count,
            "unit": unit,
            "sampled_points": len(indices),
            "curves": curves,
            "boundary": boundary,
            "subtask_intervals": intervals,
        }

    def frame(self, index: int, frame: int, *, boundary: str) -> dict[str, Any]:
        episode = self.episode(index)
        if frame < 0 or frame >= episode.frame_count:
            raise FileNotFoundError(
                f"Frame {frame} is outside episode {index} [0, {episode.frame_count - 1}]"
            )
        table = self.table(index)
        values = {}
        for definition in SERIES:
            values[definition["id"]] = {
                unit: (
                    _safe_number(table.column(definition[unit])[frame].as_py())
                    if definition[unit] in table.column_names
                    else None
                )
                for unit in ("norm", "frames")
            }
        intervals = self._boundary_intervals(table, boundary)
        current_interval = next((item for item in intervals if item["start"] <= frame <= item["end"]), None)
        return {
            "episode_index": index,
            "frame": frame,
            "time_seconds": frame / self.fps,
            "values": values,
            "boundary": boundary,
            "subtask": current_interval,
            "chunk_start": frame,
            "chunk_end": min(frame + self.chunk_size, episode.frame_count - 1),
            "chunk_segments": _chunk_segments(
                intervals,
                frame=frame,
                chunk_size=self.chunk_size,
                frame_count=episode.frame_count,
            ),
        }

    def image_path(self, index: int, camera: str, frame: int) -> Path:
        episode = self.episode(index)
        if camera not in self.camera_subdirs:
            raise FileNotFoundError(f"Unknown camera: {camera}")
        if frame < 0 or frame >= episode.frame_count:
            raise FileNotFoundError(f"Frame {frame} does not exist in episode {index}")
        path = raw_frame_image_path(episode.path, camera, frame, self.image_encoding)
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")
        return path


class Handler(BaseHTTPRequestHandler):
    run: ValueRun = None  # type: ignore[assignment]

    def log_message(self, fmt, *args):
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send_bytes(self, body: bytes, content_type: str, *, status: int = 200, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, *, status: int = 200):
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _safe_error(self, status: int, message: str):
        try:
            self._send_json({"error": message}, status=status)
        except CONNECTION_ERRORS:
            pass

    def _send_static(self, name: str):
        path = (STATIC_DIR / name).resolve()
        if path.parent != STATIC_DIR.resolve() or not path.is_file():
            self._send_bytes(b"Not found", "text/plain", status=404)
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        self._send_bytes(path.read_bytes(), content_type)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path in {"/", "/index.html"}:
                self._send_static("index.html")
            elif path in {"/app.js", "/style.css"}:
                self._send_static(path.lstrip("/"))
            elif path == "/api/meta":
                self._send_json(self.run.metadata())
            elif match := re.fullmatch(r"/api/episode/(\d+)/curves", path):
                params = parse_qs(parsed.query)
                self._send_json(
                    self.run.curves(
                        int(match.group(1)),
                        unit=params.get("unit", ["norm"])[0],
                        boundary=params.get("boundary", ["gt"])[0],
                        max_points=_positive_int(
                            params.get("max_points", [None])[0],
                            default=DEFAULT_MAX_POINTS,
                            maximum=MAX_MAX_POINTS,
                        ),
                    )
                )
            elif match := re.fullmatch(r"/api/episode/(\d+)/frame/(\d+)", path):
                params = parse_qs(parsed.query)
                self._send_json(
                    self.run.frame(
                        int(match.group(1)),
                        int(match.group(2)),
                        boundary=params.get("boundary", ["gt"])[0],
                    )
                )
            elif match := re.fullmatch(r"/api/episode/(\d+)/img/([^/]+)/(\d+)", path):
                image_path = self.run.image_path(int(match.group(1)), match.group(2), int(match.group(3)))
                self._send_bytes(
                    image_path.read_bytes(),
                    self.run.image_encoding.mime_type,
                    cache=True,
                )
            else:
                self._send_bytes(b"Not found", "text/plain", status=404)
        except CONNECTION_ERRORS:
            return
        except FileNotFoundError as exc:
            self._safe_error(404, str(exc))
        except (TypeError, ValueError) as exc:
            self._safe_error(400, str(exc))
        except Exception as exc:
            logger.exception("GET %s failed", path)
            self._safe_error(500, str(exc))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Raw run directory.")
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = ValueRun(args.root, chunk_size=args.chunk_size)
    Handler.run = run
    if args.host not in LOCAL_HOSTS:
        logger.warning("No authentication; raw images are exposed on host %s", args.host)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{server.server_port}"
    logger.info("Serving value visualization for %s", run.root)
    logger.info("Open %s", url)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
