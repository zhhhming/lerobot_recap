#!/usr/bin/env python

"""Headless export and paginated local UI for advantage labels."""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from lerobot.datasets.raw_media import raw_frame_image_path, raw_image_encoding_from_meta
from lerobot.value_function.advantage_labeling import (
    AdvantageLabelingConfig,
    advantage_export_eligibility,
    export_advantage_labels,
    load_advantage_chunks,
    load_saved_overrides,
    preview_advantage_labels,
)
from lerobot.value_function.raw_io import discover_episodes, get_image_keys, read_run_meta

logger = logging.getLogger("advantage_labeler")
STATIC_DIR = Path(__file__).resolve().parent / "advantage_labeler"
CONNECTION_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class RawRun:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.meta = read_run_meta(self.root)
        self.fps = int(self.meta.get("fps", 30))
        self.task = self.meta.get("task", "")
        self.robot_type = self.meta.get("robot_type", "")
        self.image_encoding = raw_image_encoding_from_meta(self.meta)
        self.image_keys = get_image_keys(self.meta)
        self.cam_subdir = {key: key.split(".")[-1] for key in self.image_keys}

    def episode_dir(self, idx: int) -> Path:
        return self.root / f"ep_{idx:06d}"

    def list_episodes(self) -> list[dict]:
        return [
            {"index": ep.index, "length": ep.frame_count, "dir": ep.path.name}
            for ep in discover_episodes(self.root)
        ]


class ChunkCache:
    """Cache parquet-derived chunks until extras file stats change."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._entries: dict[str, tuple[tuple, list[dict]]] = {}
        self.load_count = 0

    def _signature(self) -> tuple:
        signature = []
        for episode in discover_episodes(self.root):
            path = episode.path / "extras.parquet"
            stat = path.stat()
            signature.append((episode.index, stat.st_size, stat.st_mtime_ns))
        return tuple(signature)

    def get(self, value_mode: str) -> list[dict]:
        signature = self._signature()
        cached = self._entries.get(value_mode)
        if cached is None or cached[0] != signature:
            chunks = load_advantage_chunks(self.root, value_mode=value_mode)  # type: ignore[arg-type]
            self._entries[value_mode] = (signature, chunks)
            self.load_count += 1
        return self._entries[value_mode][1]

    def invalidate(self, value_mode: str | None = None) -> None:
        if value_mode is None:
            self._entries.clear()
        else:
            self._entries.pop(value_mode, None)


def _parse_page(value, *, default: int, minimum: int, maximum: int | None = None) -> int:
    parsed = int(value if value is not None else default)
    if parsed < minimum or (maximum is not None and parsed > maximum):
        limit = f"..{maximum}" if maximum is not None else "+"
        raise ValueError(f"Pagination value must be in {minimum}{limit}, got {parsed}")
    return parsed


def paginated_preview(
    chunks: list[dict],
    *,
    top_percent: float,
    sort_order: str,
    tie_policy: str,
    overrides: dict[str, str],
    page: int = 1,
    page_size: int = 200,
    episode_index: int | None = None,
    label_filter: str | None = None,
) -> dict:
    preview = preview_advantage_labels(
        chunks,
        top_percent=top_percent,
        sort_order=sort_order,  # type: ignore[arg-type]
        tie_policy=tie_policy,  # type: ignore[arg-type]
        overrides=overrides,
    )
    items = preview["sorted_chunks"]
    if episode_index is not None:
        items = [item for item in items if item["episode_index"] == episode_index]
    if label_filter:
        items = [item for item in items if item["preview_label"] == label_filter]
    total = len(items)
    total_pages = max(1, (total + page_size - 1) // page_size)
    threshold_value = preview["threshold"]["threshold_value"]
    threshold_indices = (
        [
            index
            for index, item in enumerate(items)
            if item["is_valid"] and item["advantage"] == threshold_value
        ]
        if threshold_value is not None
        else []
    )
    threshold_page = (
        threshold_indices[len(threshold_indices) // 2] // page_size + 1
        if threshold_indices
        else None
    )
    if page > total_pages:
        page = total_pages
    start = (page - 1) * page_size
    return {
        "items": items[start : start + page_size],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "threshold_page": threshold_page,
        "counts": preview["counts"],
        "threshold": preview["threshold"],
        "top_percent": preview["top_percent"],
        "sort_order": sort_order,
        "tie_policy": tie_policy,
        "override_count": len(preview["overrides"]),
    }


class Handler(BaseHTTPRequestHandler):
    run: RawRun = None  # type: ignore[assignment]
    cache: ChunkCache = None  # type: ignore[assignment]
    value_mode = "global"
    top_percent = 0.8
    sort_order = "desc"
    tie_policy = "exact_count"
    allow_synthetic = False

    def log_message(self, fmt, *args):
        logger.debug("%s - %s", self.address_string(), fmt % args)

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
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _safe_error(self, status, msg):
        try:
            self._send_json({"error": msg}, status=status)
        except CONNECTION_ERRORS:
            pass

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

    def _send_static(self, name: str):
        path = (STATIC_DIR / name).resolve()
        if path.parent != STATIC_DIR.resolve() or not path.is_file():
            self._send_bytes(b"Not found", "text/plain", status=404, cache=False)
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        self._send_bytes(path.read_bytes(), content_type, cache=False)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path in ("/", "/index.html"):
                self._send_static("index.html")
            elif path in ("/app.js", "/style.css"):
                self._send_static(path.lstrip("/"))
            elif path == "/api/meta":
                self._api_meta()
            elif path == "/api/chunks":
                self._api_chunks(parsed.query)
            elif match := re.match(r"^/api/episode/(\d+)/img/([^/]+)/(\d+)$", path):
                self._api_image(int(match.group(1)), match.group(2), int(match.group(3)))
            else:
                self._send_bytes(b"Not found", "text/plain", status=404, cache=False)
        except CONNECTION_ERRORS:
            return
        except FileNotFoundError as exc:
            self._safe_error(404, str(exc))
        except ValueError as exc:
            self._safe_error(400, str(exc))
        except Exception as exc:
            logger.exception("GET %s failed", path)
            self._safe_error(500, str(exc))

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/preview":
                self._api_preview()
            elif path == "/api/export-preview":
                self._api_export(preview_only=True)
            elif path == "/api/export":
                self._api_export(preview_only=False)
            else:
                self._send_bytes(b"Not found", "text/plain", status=404, cache=False)
        except CONNECTION_ERRORS:
            return
        except ValueError as exc:
            self._safe_error(400, str(exc))
        except Exception as exc:
            logger.exception("POST %s failed", path)
            self._safe_error(500, str(exc))

    def _api_meta(self):
        eligibility_by_mode = {}
        overrides_by_mode = {}
        for mode in ("global", "subtask"):
            try:
                eligibility_by_mode[mode] = advantage_export_eligibility(self.run.root, mode)  # type: ignore[arg-type]
                overrides_by_mode[mode] = load_saved_overrides(self.run.root, mode)  # type: ignore[arg-type]
            except (ValueError, FileNotFoundError):
                eligibility_by_mode[mode] = None
                overrides_by_mode[mode] = {}
        self._send_json(
            {
                "root": str(self.run.root),
                "fps": self.run.fps,
                "task": self.run.task,
                "robot_type": self.run.robot_type,
                "value_mode": self.value_mode,
                "top_percent": self.top_percent,
                "sort_order": self.sort_order,
                "tie_policy": self.tie_policy,
                "allow_synthetic": self.allow_synthetic,
                "eligibility_by_mode": eligibility_by_mode,
                "overrides_by_mode": overrides_by_mode,
                "cameras": [
                    {"key": key, "subdir": subdir, "label": subdir}
                    for key, subdir in self.run.cam_subdir.items()
                ],
                "episodes": self.run.list_episodes(),
            }
        )

    def _request_preview(self, body: dict) -> dict:
        mode = body.get("value_mode", self.value_mode)
        overrides = (
            body["overrides"]
            if "overrides" in body
            else load_saved_overrides(self.run.root, mode)
        )
        page = _parse_page(body.get("page"), default=1, minimum=1)
        page_size = _parse_page(body.get("page_size"), default=200, minimum=1, maximum=500)
        return paginated_preview(
            self.cache.get(mode),
            top_percent=float(body.get("top_percent", self.top_percent)),
            sort_order=body.get("sort_order", self.sort_order),
            tie_policy=body.get("tie_policy", self.tie_policy),
            overrides=overrides,
            page=page,
            page_size=page_size,
            episode_index=(
                int(body["episode_index"])
                if body.get("episode_index") not in (None, "")
                else None
            ),
            label_filter=body.get("label_filter") or None,
        )

    def _api_chunks(self, query: str):
        params = parse_qs(query)
        body = {key: values[0] for key, values in params.items()}
        self._send_json(self._request_preview(body))

    def _api_preview(self):
        self._send_json(self._request_preview(self._read_json_body()))

    def _api_export(self, *, preview_only: bool):
        body = self._read_json_body()
        if not preview_only and body.get("confirm") is not True:
            raise ValueError("Export requires confirm=true after reviewing the change summary")
        mode = body.get("value_mode", self.value_mode)
        summary = export_advantage_labels(
            AdvantageLabelingConfig(
                root=self.run.root,
                value_mode=mode,
                top_percent=float(body.get("top_percent", self.top_percent)),
                sort_order=body.get("sort_order", self.sort_order),
                tie_policy=body.get("tie_policy", self.tie_policy),
                overrides=body.get("overrides") if "overrides" in body else None,
                allow_synthetic=self.allow_synthetic,
                dry_run=preview_only,
            )
        )
        if not preview_only:
            self.cache.invalidate(mode)
        self._send_json({"ok": True, "summary": summary})

    def _api_image(self, idx: int, cam: str, frame: int):
        if cam not in set(self.run.cam_subdir.values()):
            self._send_bytes(b"Not found", "text/plain", status=404, cache=False)
            return
        path = raw_frame_image_path(
            self.run.episode_dir(idx),
            cam,
            frame,
            self.run.image_encoding,
        )
        if not path.is_file():
            self._send_bytes(b"Not found", "text/plain", status=404, cache=False)
            return
        self._send_bytes(path.read_bytes(), self.run.image_encoding.mime_type, cache=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Raw run directory.")
    parser.add_argument("--value_mode", choices=("global", "subtask"), default="global")
    parser.add_argument("--top_percent", type=float, default=0.8)
    parser.add_argument("--sort_order", choices=("desc", "asc"), default="desc")
    parser.add_argument(
        "--tie_policy", choices=("exact_count", "include_all"), default="exact_count"
    )
    parser.add_argument("--allow_synthetic", action="store_true")
    parser.add_argument("--export", action="store_true", dest="export_labels")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dry_run and not args.export_labels:
        parser.error("--dry_run requires --export")
    if args.export_labels:
        summary = export_advantage_labels(
            AdvantageLabelingConfig(
                root=args.root,
                value_mode=args.value_mode,
                top_percent=args.top_percent,
                sort_order=args.sort_order,
                tie_policy=args.tie_policy,
                allow_synthetic=args.allow_synthetic,
                dry_run=args.dry_run,
            )
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    run = RawRun(args.root)
    Handler.run = run
    Handler.cache = ChunkCache(run.root)
    Handler.value_mode = args.value_mode
    Handler.top_percent = args.top_percent
    Handler.sort_order = args.sort_order
    Handler.tie_policy = args.tie_policy
    Handler.allow_synthetic = args.allow_synthetic
    if args.host not in LOCAL_HOSTS:
        logger.warning("No authentication; write endpoints are exposed on host %s", args.host)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    logger.info("Serving advantage labeler for %s", run.root)
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
