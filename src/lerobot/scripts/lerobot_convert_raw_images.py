#!/usr/bin/env python

"""Convert a PNG-backed editable raw run to a separate JPEG-backed raw run.

The source is never modified. A partially converted output can be continued
with ``--resume``; ``run_meta.json`` is written last so incomplete outputs are
not mistaken for valid raw runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import PIL.Image
import pyarrow.parquet as pq

from lerobot.datasets.raw_media import (
    RAW_FORMAT_VERSION,
    camera_subdir_name,
    make_raw_image_encoding,
    raw_frame_image_path,
    raw_image_encoding_from_meta,
    validate_raw_format_version,
)

logger = logging.getLogger(__name__)

RUN_META_FILENAME = "run_meta.json"
MARKER_FILENAME = ".raw_image_conversion.json"
EPISODE_RE = re.compile(r"^ep_(\d+)$")


@dataclass(frozen=True)
class ConvertRawImagesConfig:
    root: Path
    output_root: Path
    quality: int = 95
    subsampling: int = 0
    workers: int = 8
    resume: bool = False

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        make_raw_image_encoding(
            "jpeg",
            jpeg_quality=self.quality,
            jpeg_subsampling=self.subsampling,
        )


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    try:
        with open(temp, "w") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _episode_dirs(root: Path) -> list[Path]:
    episodes = [
        path
        for path in root.iterdir()
        if path.is_dir() and EPISODE_RE.fullmatch(path.name) and (path / "info.json").is_file()
    ]
    episodes.sort(key=lambda path: int(path.name.split("_", 1)[1]))
    if not episodes:
        raise ValueError(f"No raw episodes found in {root}")
    return episodes


def _frame_count(episode: Path) -> int:
    frames_path = episode / "frames.parquet"
    if not frames_path.is_file():
        raise FileNotFoundError(f"Missing {frames_path}")
    return pq.read_metadata(frames_path).num_rows


def _copy_non_image_content(
    source: Path,
    destination: Path,
    episodes: list[Path],
    camera_dirs: set[str],
) -> None:
    episode_names = {episode.name for episode in episodes}
    for item in source.iterdir():
        if item.name in {RUN_META_FILENAME, MARKER_FILENAME}:
            continue
        target = destination / item.name
        if item.is_dir() and item.name in episode_names:
            target.mkdir(parents=True, exist_ok=True)
            for child in item.iterdir():
                if child.is_dir() and child.name in camera_dirs:
                    continue
                child_target = target / child.name
                if child.is_dir():
                    shutil.copytree(child, child_target, dirs_exist_ok=True)
                else:
                    shutil.copy2(child, child_target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _image_size(path: Path) -> tuple[int, int]:
    with PIL.Image.open(path) as image:
        image.load()
        return image.size


def _convert_one_image(source: Path, destination: Path, quality: int, subsampling: int) -> bool:
    source_size = _image_size(source)
    if destination.is_file():
        if _image_size(destination) != source_size:
            raise ValueError(f"Existing converted image has the wrong size: {destination}")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp")
    try:
        with PIL.Image.open(source) as image:
            image.convert("RGB").save(
                temp,
                format="JPEG",
                quality=quality,
                subsampling=subsampling,
                optimize=False,
            )
        if _image_size(temp) != source_size:
            raise ValueError(f"Converted image has the wrong size: {temp}")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return True


def _prepare_output(output_root: Path, *, resume: bool, marker: dict[str, Any]) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise NotADirectoryError(output_root)
        marker_path = output_root / MARKER_FILENAME
        if not resume:
            raise FileExistsError(
                f"Output already exists: {output_root}. Pass --resume only for an incomplete "
                "conversion created by this command."
            )
        if not marker_path.is_file() or _read_json(marker_path) != marker:
            raise ValueError(f"Conversion marker does not match requested conversion: {marker_path}")
        if (output_root / RUN_META_FILENAME).exists():
            raise FileExistsError(f"Output conversion is already complete: {output_root}")
        return

    output_root.mkdir(parents=True)
    _write_json_atomic(output_root / MARKER_FILENAME, marker)


def convert_raw_images(cfg: ConvertRawImagesConfig) -> Path:
    source = cfg.root.expanduser().resolve()
    output = cfg.output_root.expanduser().resolve()
    if source == output or _is_within(output, source) or _is_within(source, output):
        raise ValueError("Source and output roots must be separate, non-nested directories")

    meta_path = source / RUN_META_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = _read_json(meta_path)
    validate_raw_format_version(meta, source)
    source_encoding = raw_image_encoding_from_meta(meta)
    if source_encoding.format != "png":
        raise ValueError(f"Source raw run is not PNG-backed: {source_encoding.format!r}")

    image_keys = [
        key
        for key, feature in (meta.get("features") or {}).items()
        if feature.get("dtype") in ("image", "video")
    ]
    if not image_keys:
        raise ValueError(f"Raw run has no image features: {source}")
    camera_dirs = {camera_subdir_name(key) for key in image_keys}
    episodes = _episode_dirs(source)
    target_encoding = make_raw_image_encoding(
        "jpeg",
        jpeg_quality=cfg.quality,
        jpeg_subsampling=cfg.subsampling,
    )
    marker = {
        "source": str(source),
        "quality": cfg.quality,
        "subsampling": cfg.subsampling,
    }
    _prepare_output(output, resume=cfg.resume, marker=marker)
    _copy_non_image_content(source, output, episodes, camera_dirs)

    converted = skipped = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        for episode_number, episode in enumerate(episodes, start=1):
            frame_count = _frame_count(episode)
            output_episode = output / episode.name
            for camera in sorted(camera_dirs):
                source_camera = episode / camera
                if not source_camera.is_dir():
                    raise FileNotFoundError(f"Missing camera directory {source_camera}")
                pairs = [
                    (
                        raw_frame_image_path(episode, camera, frame, source_encoding),
                        raw_frame_image_path(output_episode, camera, frame, target_encoding),
                    )
                    for frame in range(frame_count)
                ]
                results = executor.map(
                    lambda pair: _convert_one_image(pair[0], pair[1], cfg.quality, cfg.subsampling),
                    pairs,
                )
                for was_converted in results:
                    converted += int(was_converted)
                    skipped += int(not was_converted)
            logger.info(
                "Converted episode %s (%d/%d, %d frames)",
                episode.name,
                episode_number,
                len(episodes),
                frame_count,
            )

    output_meta = dict(meta)
    output_meta["version"] = RAW_FORMAT_VERSION
    output_meta["image_encoding"] = target_encoding.to_metadata()
    output_meta["image_conversion"] = {
        "source": str(source),
        "converted_at": datetime.now(UTC).isoformat(timespec="microseconds"),
    }
    _write_json_atomic(output / RUN_META_FILENAME, output_meta)
    (output / MARKER_FILENAME).unlink()
    logger.info(
        "Completed raw image conversion at %s (converted=%d, resumed=%d)",
        output,
        converted,
        skipped,
    )
    return output


def _parse_args() -> ConvertRawImagesConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Source PNG raw run")
    parser.add_argument("--output-root", type=Path, required=True, help="Separate JPEG raw run")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--subsampling", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return ConvertRawImagesConfig(
        root=args.root,
        output_root=args.output_root,
        quality=args.quality,
        subsampling=args.subsampling,
        workers=args.workers,
        resume=args.resume,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    convert_raw_images(_parse_args())


if __name__ == "__main__":
    main()
