#!/usr/bin/env python

"""Build a LeRobotDataset from one or more raw run directories produced by
`lerobot-raw-record`.

Usage:

    lerobot-build-dataset \
        --runs /path/to/raw/run_a /path/to/raw/run_b \
        --output_repo_id user/my_dataset \
        --video true \
        --vcodec libsvtav1 \
        --exclude_features "observation.images.cam_extra,advantage" \
        --push_to_hub false

The builder is the only place that touches the LeRobotDataset write API, so
the raw recorder stays simple and individual episodes can be deleted by
`rm -rf` before this script runs.
"""

import fnmatch
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import INFO_PATH
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


RAW_FORMAT_VERSION = 1
RUN_META_FILENAME = "run_meta.json"


@dataclass
class BuildDatasetConfig:
    runs: list[str]
    output_repo_id: str
    output_root: str | None = None
    video: bool = True
    vcodec: str = "libsvtav1"
    streaming_encoding: bool = False
    encoder_queue_maxsize: int = 30
    encoder_threads: int | None = None
    video_encoding_batch_size: int = 1
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    include_features: str = "*"
    exclude_features: str = ""
    fps: int | None = None
    task_override: str | None = None
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    force: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not self.runs:
            raise ValueError("--runs must list at least one raw run directory.")
        if not self.output_repo_id:
            raise ValueError("--output_repo_id is required.")


def _parse_patterns(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _deserialize_features(features: dict) -> dict:
    out = {}
    for k, v in features.items():
        ft = dict(v)
        shape = ft.get("shape")
        if isinstance(shape, list):
            ft["shape"] = tuple(shape)
        out[k] = ft
    return out


def _load_run_meta(run_dir: Path) -> dict:
    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {RUN_META_FILENAME} in '{run_dir}'.")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("version") != RAW_FORMAT_VERSION:
        raise ValueError(
            f"Raw format version mismatch in '{run_dir}': expected {RAW_FORMAT_VERSION}, "
            f"got {meta.get('version')}."
        )
    meta["features"] = _deserialize_features(meta["features"])
    return meta


def _discover_episodes(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory not found: {run_dir}")
    eps = sorted(
        d
        for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("ep_") and (d / "info.json").is_file()
    )
    return eps


def _merge_features(meta_features: list[dict]) -> dict:
    """Ensure every run has the same feature schema; return the unified schema."""
    if not meta_features:
        raise ValueError("No runs provided.")
    base = meta_features[0]
    for i, other in enumerate(meta_features[1:], start=1):
        if set(base) != set(other):
            base_only = sorted(set(base) - set(other))
            other_only = sorted(set(other) - set(base))
            raise ValueError(
                f"Feature keys differ between run 0 and run {i}: "
                f"only in run 0={base_only}, only in run {i}={other_only}"
            )
        for k in base:
            if base[k]["dtype"] != other[k]["dtype"] or tuple(base[k]["shape"]) != tuple(
                other[k]["shape"]
            ):
                raise ValueError(
                    f"Feature '{k}' mismatch between runs 0 and {i}: "
                    f"{base[k]} vs {other[k]}"
                )
    return base


def _filter_features(
    features: dict,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> dict:
    out = {}
    for k, v in features.items():
        if not _matches_any(k, include_patterns):
            continue
        if exclude_patterns and _matches_any(k, exclude_patterns):
            continue
        out[k] = v
    return out


def _force_image_dtype(features: dict, use_video: bool) -> dict:
    """Rewrite camera feature dtype to either 'video' or 'image'."""
    out = {}
    for k, v in features.items():
        ft = dict(v)
        if ft["dtype"] in ("image", "video"):
            ft["dtype"] = "video" if use_video else "image"
        out[k] = ft
    return out


def _load_extras_schema(ep_dirs: list[Path]) -> tuple[dict, set[str]]:
    """Read extras.parquet schema if present. Returns (features dict, columns).

    Enforces that either all episodes have extras or none do, and that all
    extras schemas match.
    """
    extras_paths = [(ep, ep / "extras.parquet") for ep in ep_dirs]
    has_extras = [p.is_file() for _, p in extras_paths]
    if not any(has_extras):
        return {}, set()
    if not all(has_extras):
        missing = [str(ep) for (ep, p), present in zip(extras_paths, has_extras) if not present]
        raise ValueError(
            "Some episodes have extras.parquet and others do not. Build aborted.\n"
            f"Missing in: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    first_schema = pq.read_schema(extras_paths[0][1])
    for ep, p in extras_paths[1:]:
        sch = pq.read_schema(p)
        if sch.names != first_schema.names:
            raise ValueError(
                f"extras.parquet schema differs in {ep}: columns {sch.names} vs {first_schema.names}"
            )

    extras_features: dict = {}
    for name in first_schema.names:
        # Map common pyarrow types into LeRobot features.
        ft = first_schema.field(name)
        pa_type = ft.type
        if pa_type.equals(pa.string()):
            extras_features[name] = {"dtype": "string", "shape": (1,), "names": None}
        elif pa.types.is_list(pa_type):
            elem = pa_type.value_type
            extras_features[name] = {
                "dtype": str(elem),
                "shape": (None,),
                "names": None,
            }
        elif (
            pa.types.is_floating(pa_type)
            or pa.types.is_integer(pa_type)
            or pa.types.is_boolean(pa_type)
        ):
            extras_features[name] = {"dtype": str(pa_type), "shape": (1,), "names": None}
        else:
            raise ValueError(f"Unsupported extras column type for '{name}': {pa_type}")
    return extras_features, set(first_schema.names)


def _resolve_output_root(repo_id: str, root: str | None) -> Path:
    return Path(root) if root is not None else HF_LEROBOT_HOME / repo_id


def _is_safe_to_remove(root: Path) -> bool:
    if root.is_symlink() or not root.is_dir():
        return False
    try:
        next(root.iterdir())
    except StopIteration:
        return True
    return (root / INFO_PATH).is_file()


def _safe_rmtree(root: Path, reason: str) -> None:
    if not root.exists():
        return
    if not _is_safe_to_remove(root):
        raise FileExistsError(
            f"Refusing to remove '{root}'. {reason} only removes empty directories or "
            f"existing LeRobot dataset directories containing '{INFO_PATH}'."
        )
    logger.warning("Removing existing dataset directory '%s' (%s).", root, reason)
    shutil.rmtree(root)


def _load_extras_row_as_dict(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, list):
            out[k] = np.asarray(v, dtype=np.float32)
        else:
            out[k] = v
    return out


def _row_to_frame(
    row: dict,
    features: dict,
    image_keys: list[str],
    scalar_keys: list[str],
    ep_dir: Path,
    cam_subdir: dict[str, str],
) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    for k in scalar_keys:
        v = row.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            frame[k] = np.asarray(v, dtype=np.float32)
        else:
            frame[k] = v
    for k in image_keys:
        subdir = cam_subdir[k]
        path = ep_dir / subdir / f"{row['frame_index']:06d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frame image {path}")
        with PIL.Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"))
        frame[k] = arr
    return frame


@parser.wrap()
def build_dataset(cfg: BuildDatasetConfig) -> LeRobotDataset | None:
    init_logging()
    logger.info(pformat(asdict(cfg)))

    include_patterns = _parse_patterns(cfg.include_features) or ["*"]
    exclude_patterns = _parse_patterns(cfg.exclude_features)

    run_dirs = [Path(p).expanduser().resolve() for p in cfg.runs]
    metas = [_load_run_meta(rd) for rd in run_dirs]

    fps_values = {m["fps"] for m in metas}
    if cfg.fps is None and len(fps_values) > 1:
        raise ValueError(f"Runs have different fps values: {fps_values}. Pass --fps to override.")
    fps = cfg.fps if cfg.fps is not None else fps_values.pop()

    robot_types = {m["robot_type"] for m in metas}
    if len(robot_types) != 1:
        raise ValueError(f"Runs have different robot_type values: {robot_types}.")
    robot_type = robot_types.pop()

    merged_raw_features = _merge_features([m["features"] for m in metas])

    all_eps: list[tuple[Path, Path]] = []
    for run_dir in run_dirs:
        for ep in _discover_episodes(run_dir):
            all_eps.append((run_dir, ep))
    if not all_eps:
        raise ValueError("No episodes found across the provided runs.")
    logger.info("Discovered %d episodes across %d runs.", len(all_eps), len(run_dirs))

    extras_features, extras_columns = _load_extras_schema([ep for _, ep in all_eps])

    candidate_features = dict(merged_raw_features)
    candidate_features.update(extras_features)
    filtered = _filter_features(candidate_features, include_patterns, exclude_patterns)
    if not filtered:
        raise ValueError("After filtering, no features remain.")

    final_features = _force_image_dtype(filtered, use_video=cfg.video)

    image_keys = [k for k, v in final_features.items() if v["dtype"] in ("image", "video")]
    scalar_keys = [
        k
        for k, v in final_features.items()
        if v["dtype"] not in ("image", "video") and k not in extras_columns
    ]
    extras_keys = [k for k in extras_columns if k in final_features]

    cam_subdir = {k: k.split(".")[-1] for k in image_keys}

    logger.info("Final feature schema (%d keys):\n%s", len(final_features), pformat(final_features))
    if cfg.dry_run:
        logger.info("--dry_run set; skipping LeRobotDataset creation.")
        return None

    output_root = _resolve_output_root(cfg.output_repo_id, cfg.output_root)
    if cfg.force:
        _safe_rmtree(output_root, reason="--force was set")

    num_cameras = len(image_keys)
    dataset = LeRobotDataset.create(
        cfg.output_repo_id,
        fps,
        root=cfg.output_root,
        robot_type=robot_type,
        features=final_features,
        use_videos=cfg.video,
        image_writer_processes=cfg.num_image_writer_processes,
        image_writer_threads=cfg.num_image_writer_threads_per_camera * max(num_cameras, 1),
        batch_encoding_size=cfg.video_encoding_batch_size,
        vcodec=cfg.vcodec,
        streaming_encoding=cfg.streaming_encoding,
        encoder_queue_maxsize=cfg.encoder_queue_maxsize,
        encoder_threads=cfg.encoder_threads,
    )

    try:
        with VideoEncodingManager(dataset):
            for run_dir, ep_dir in all_eps:
                _build_one_episode(
                    dataset=dataset,
                    run_dir=run_dir,
                    ep_dir=ep_dir,
                    final_features=final_features,
                    image_keys=image_keys,
                    scalar_keys=scalar_keys,
                    extras_keys=extras_keys,
                    cam_subdir=cam_subdir,
                    task_override=cfg.task_override,
                )
        dataset.finalize()
        logger.info("Finalized dataset at %s (%d episodes).", dataset.root, dataset.num_episodes)
    except Exception:
        logger.exception("Build failed.")
        raise

    if cfg.push_to_hub and dataset.num_episodes > 0:
        try:
            dataset.push_to_hub(tags=cfg.tags, private=cfg.private)
        except Exception:
            logger.exception("Failed to push dataset to the Hugging Face Hub.")

    return dataset


def _build_one_episode(
    dataset: LeRobotDataset,
    run_dir: Path,
    ep_dir: Path,
    final_features: dict,
    image_keys: list[str],
    scalar_keys: list[str],
    extras_keys: list[str],
    cam_subdir: dict[str, str],
    task_override: str | None,
) -> None:
    info_path = ep_dir / "info.json"
    frames_path = ep_dir / "frames.parquet"
    extras_path = ep_dir / "extras.parquet"

    with open(info_path) as f:
        info = json.load(f)

    frames_table = pq.read_table(frames_path)
    frames_rows = frames_table.to_pylist()

    extras_rows: list[dict] = []
    if extras_keys and extras_path.is_file():
        extras_table = pq.read_table(extras_path)
        extras_rows = extras_table.to_pylist()
        if len(extras_rows) != len(frames_rows):
            raise ValueError(
                f"extras.parquet length ({len(extras_rows)}) does not match frames.parquet "
                f"length ({len(frames_rows)}) in {ep_dir}"
            )

    task = task_override if task_override is not None else info.get("task")
    if task is None:
        raise ValueError(f"No task found for episode {ep_dir} and no --task_override given.")

    logger.info("Building episode %s (%d frames)...", ep_dir.name, len(frames_rows))
    for i, row in enumerate(frames_rows):
        frame = _row_to_frame(
            row=row,
            features=final_features,
            image_keys=image_keys,
            scalar_keys=scalar_keys,
            ep_dir=ep_dir,
            cam_subdir=cam_subdir,
        )
        if extras_rows:
            extras_dict = _load_extras_row_as_dict(extras_rows[i])
            for k in extras_keys:
                if k in extras_dict:
                    frame[k] = extras_dict[k]
        frame["task"] = task
        dataset.add_frame(frame)
    dataset.save_episode()


def main():
    register_third_party_plugins()
    build_dataset()


if __name__ == "__main__":
    main()
