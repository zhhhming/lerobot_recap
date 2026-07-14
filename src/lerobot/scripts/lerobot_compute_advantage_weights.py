#!/usr/bin/env python

"""Compute group-relative advantage weights or inspect persisted weights in a local UI."""

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

from lerobot.value_function.advantage_weights import (
    AdvantageWeightConfig,
    compute_advantage_weights,
    load_advantage_weight_chunks,
)
from lerobot.value_function.raw_io import (
    discover_episodes,
    get_image_keys,
    read_run_meta,
    read_value_function_metadata,
)

logger = logging.getLogger("advantage_weights")
STATIC_DIR = Path(__file__).resolve().parent / "advantage_weight_viz"
CONNECTION_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class RawRun:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.meta = read_run_meta(self.root)
        self.fps = int(self.meta.get("fps", 30))
        self.task = self.meta.get("task", "")
        self.image_keys = get_image_keys(self.meta)
        self.cam_subdir = {key: key.split(".")[-1] for key in self.image_keys}

    def episode_dir(self, index: int) -> Path:
        return self.root / f"ep_{index:06d}"

    def list_episodes(self) -> list[dict]:
        return [
            {"index": episode.index, "length": episode.frame_count}
            for episode in discover_episodes(self.root)
        ]


class WeightCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._entries: dict[str, tuple[tuple, tuple[list[dict], list[dict]]]] = {}
        self.load_count = 0

    def _signature(self) -> tuple:
        signature = []
        for episode in discover_episodes(self.root):
            stat = (episode.path / "extras.parquet").stat()
            signature.append((episode.index, stat.st_size, stat.st_mtime_ns))
        return tuple(signature)

    def get(self, value_mode: str) -> tuple[list[dict], list[dict]]:
        signature = self._signature()
        cached = self._entries.get(value_mode)
        if cached is None or cached[0] != signature:
            payload = load_advantage_weight_chunks(self.root, value_mode=value_mode)  # type: ignore[arg-type]
            self._entries[value_mode] = (signature, payload)
            self.load_count += 1
        return self._entries[value_mode][1]


def _parse_page(value, *, default: int, minimum: int, maximum: int | None = None) -> int:
    parsed = int(value if value is not None else default)
    if parsed < minimum or (maximum is not None and parsed > maximum):
        limit = f"..{maximum}" if maximum is not None else "+"
        raise ValueError(f"Pagination value must be in {minimum}{limit}, got {parsed}")
    return parsed


def paginated_groups(groups: list[dict], *, page: int = 1, page_size: int = 100) -> dict:
    total = len(groups)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    return {
        "items": groups[start : start + page_size],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
    }


def paginated_group_chunks(
    chunks: list[dict],
    *,
    group_id: str,
    page: int = 1,
    page_size: int = 150,
    sort_order: str = "desc",
) -> dict:
    if sort_order not in ("desc", "asc"):
        raise ValueError("sort_order must be 'desc' or 'asc'")
    items = [chunk for chunk in chunks if chunk["group_id"] == group_id]
    direction = -1 if sort_order == "desc" else 1
    items.sort(
        key=lambda chunk: (
            direction * chunk["advantage"],
            chunk["episode_index"],
            chunk["frame_index"],
        )
    )
    total = len(items)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    return {
        "items": items[start : start + page_size],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "group_id": group_id,
        "sort_order": sort_order,
    }


class Handler(BaseHTTPRequestHandler):
    run: RawRun = None  # type: ignore[assignment]
    cache: WeightCache = None  # type: ignore[assignment]
    value_mode = "global"

    def log_message(self, fmt, *args):
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode()
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

    def _safe_error(self, status: int, message: str):
        try:
            self._send_json({"error": message}, status=status)
        except CONNECTION_ERRORS:
            pass

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
            elif path == "/api/groups":
                self._api_groups(parsed.query)
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

    def _mode(self, params: dict[str, list[str]]) -> str:
        mode = params.get("value_mode", [self.value_mode])[0]
        if mode not in ("global", "subtask"):
            raise ValueError(f"Unsupported value_mode: {mode!r}")
        return mode

    def _api_meta(self):
        metadata = read_value_function_metadata(self.run.root)
        modes = {}
        for mode in ("global", "subtask"):
            modes[mode] = (metadata.get("advantage_weights") or {}).get(mode)
        self._send_json(
            {
                "root": str(self.run.root),
                "task": self.run.task,
                "fps": self.run.fps,
                "value_mode": self.value_mode,
                "modes": modes,
                "cameras": [
                    {"key": key, "subdir": subdir, "label": subdir}
                    for key, subdir in self.run.cam_subdir.items()
                ],
                "episodes": self.run.list_episodes(),
            }
        )

    def _api_groups(self, query: str):
        params = parse_qs(query)
        mode = self._mode(params)
        page = _parse_page(params.get("page", [None])[0], default=1, minimum=1)
        page_size = _parse_page(
            params.get("page_size", [None])[0], default=100, minimum=1, maximum=500
        )
        _chunks, groups = self.cache.get(mode)
        self._send_json(paginated_groups(groups, page=page, page_size=page_size))

    def _api_chunks(self, query: str):
        params = parse_qs(query)
        mode = self._mode(params)
        group_id = params.get("group_id", [""])[0]
        if not group_id:
            raise ValueError("group_id is required")
        page = _parse_page(params.get("page", [None])[0], default=1, minimum=1)
        page_size = _parse_page(
            params.get("page_size", [None])[0], default=150, minimum=1, maximum=500
        )
        chunks, groups = self.cache.get(mode)
        if group_id not in {group["group_id"] for group in groups}:
            raise ValueError(f"Unknown group_id: {group_id!r}")
        self._send_json(
            paginated_group_chunks(
                chunks,
                group_id=group_id,
                page=page,
                page_size=page_size,
                sort_order=params.get("sort_order", ["desc"])[0],
            )
        )

    def _api_image(self, episode: int, camera: str, frame: int):
        if camera not in set(self.run.cam_subdir.values()):
            raise FileNotFoundError(f"Unknown camera: {camera}")
        path = self.run.episode_dir(episode) / camera / f"{frame:06d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        self._send_bytes(path.read_bytes(), "image/png", cache=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--value_mode", choices=("global", "subtask"), default="global")
    parser.add_argument("--group_source", choices=("auto", "progress", "value"), default="auto")
    parser.add_argument("--group_bin_width", type=float)
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--w_min", type=float, default=0.1)
    parser.add_argument("--w_max", type=float, default=2.0)
    parser.add_argument("--positive_group_max_weight", type=float, default=2.0)
    parser.add_argument("--min_group_size", type=int, default=4)
    parser.add_argument("--negative_weight", type=float, default=1.0)
    parser.add_argument("--allow_synthetic", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.serve:
        if args.dry_run:
            parser.error("--serve and --dry_run are mutually exclusive")
        run = RawRun(args.root)
        Handler.run = run
        Handler.cache = WeightCache(run.root)
        Handler.value_mode = args.value_mode
        if args.host not in LOCAL_HOSTS:
            logger.warning("No authentication; UI is exposed on host %s", args.host)
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        url = f"http://{args.host}:{args.port}"
        logger.info("Serving advantage weight visualization for %s", run.root)
        logger.info("Open %s", url)
        if not args.no_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down.")
            server.shutdown()
        return None

    summary = compute_advantage_weights(
        AdvantageWeightConfig(
            root=args.root,
            value_mode=args.value_mode,
            group_source=args.group_source,
            group_bin_width=args.group_bin_width,
            q=args.q,
            tau=args.tau,
            w_min=args.w_min,
            w_max=args.w_max,
            positive_group_max_weight=args.positive_group_max_weight,
            min_group_size=args.min_group_size,
            negative_weight=args.negative_weight,
            allow_synthetic=args.allow_synthetic,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
