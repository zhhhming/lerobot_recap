"""Ground-truth value target generation for raw LeRobot runs."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa

from lerobot.value_function.raw_io import (
    discover_episodes,
    fingerprint_raw_run_columns,
    get_image_keys,
    merge_raw_run_extras,
    read_extras_table,
    read_run_meta,
    update_stage_metadata,
)
from lerobot.value_function.schema import (
    ANNOTATION_CONFIG_FILENAME,
    PREDICTION_SOURCE_GT,
    SUBTASK_COLUMN,
    TARGET_STAGE,
    TARGET_COLUMNS,
    VALUE_GLOBAL_ELAPSED_FRAMES_GT,
    VALUE_GLOBAL_ELAPSED_NORM_GT,
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT_IS_CLIPPED,
    VALUE_SUBTASK_ELAPSED_FRAMES_GT,
    VALUE_SUBTASK_ELAPSED_NORM_GT,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_NAME_GT,
    VALUE_SUBTASK_REMAINING_FRAMES_GT,
    VALUE_SUBTASK_REMAINING_NORM_GT,
    VALUE_SUBTASK_REMAINING_NORM_GT_IS_CLIPPED,
)

ValueMode = Literal["global", "subtask", "both"]
ScaleStrategy = Literal["max", "p95", "manual"]
AllowUnlabeled = Literal["error", "skip", "default_subtask"]


@dataclass(frozen=True)
class ValueTargetConfig:
    root: Path | str
    mode: ValueMode = "both"
    num_bins: int = 256
    global_num_bins: int | None = None
    subtask_num_bins: int | None = None
    global_scale: ScaleStrategy = "p95"
    subtask_scale: ScaleStrategy = "p95"
    global_scale_frames: float | None = None
    subtask_scale_frames: Mapping[str, float] | None = None
    elapsed_aux: bool = False
    allow_unlabeled: AllowUnlabeled = "error"
    default_subtask: str | None = None
    subtask_column: str = SUBTASK_COLUMN
    subtask_order: Sequence[str] | None = None
    require_all_subtasks: bool = True
    require_single_segment_per_subtask: bool = True
    require_success_only: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class EpisodeLabels:
    index: int
    labels: list[str | None]


@dataclass(frozen=True)
class SubtaskSegment:
    episode_index: int
    name: str
    start: int
    end: int

    @property
    def max_remaining(self) -> int:
        return max(self.end - self.start, 0)


def _needs_global(mode: ValueMode) -> bool:
    return mode in ("global", "both")


def _needs_subtask(mode: ValueMode) -> bool:
    return mode in ("subtask", "both")


def _is_unlabeled(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _as_label(value: object, *, episode_index: int, frame_index: int) -> str | None:
    if _is_unlabeled(value):
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"Subtask label must be a string or empty in episode {episode_index}, "
            f"frame {frame_index}; got {type(value).__name__}"
        )
    return value.strip()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 1.0
    scale = float(np.percentile(np.asarray(values, dtype=np.float32), percentile))
    return max(scale, 1.0)


def _scale_from_values(
    values: list[float],
    strategy: ScaleStrategy,
    manual_value: float | None,
    *,
    name: str,
) -> float:
    if strategy == "manual":
        if manual_value is None:
            raise ValueError(f"manual scale for {name} requires an explicit scale value")
        scale = float(manual_value)
        if scale <= 0:
            raise ValueError(f"manual scale for {name} must be > 0, got {scale}")
        return scale
    if strategy == "max":
        return max(max(values) if values else 0.0, 1.0)
    if strategy == "p95":
        return _percentile(values, 95.0)
    raise ValueError(f"Unsupported scale strategy: {strategy}")


def _clip_norm(values: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    raw = values.astype(np.float32) / np.float32(scale)
    clipped_mask = raw > 1.0
    return np.clip(raw, 0.0, 1.0).astype(np.float32), clipped_mask.astype(bool)


def _clip_stats(mask: np.ndarray) -> dict[str, int | float]:
    eligible = int(mask.size)
    clipped = int(mask.sum())
    return {
        "clipped_frames": clipped,
        "eligible_frames": eligible,
        "clip_rate": float(clipped / eligible) if eligible else 0.0,
    }


def _load_annotation_subtasks(root: Path) -> list[str]:
    path = root / ANNOTATION_CONFIG_FILENAME
    if not path.is_file():
        return []
    with open(path) as f:
        cfg = json.load(f)
    names: list[str] = []
    for item in cfg.get("subtasks") or []:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, Mapping):
            name = str(item.get("name", "")).strip()
        else:
            continue
        if name and name not in names:
            names.append(name)
    return names


def _load_episode_labels(config: ValueTargetConfig) -> list[EpisodeLabels]:
    episodes = discover_episodes(config.root)
    labels_by_episode: list[EpisodeLabels] = []
    for episode in episodes:
        extras = read_extras_table(episode.path)
        if extras is None:
            raise FileNotFoundError(
                f"Missing extras.parquet in {episode.path}; subtask targets require annotations"
            )
        if extras.num_rows != episode.frame_count:
            raise ValueError(
                f"extras.parquet length ({extras.num_rows}) does not match frames.parquet "
                f"length ({episode.frame_count}) in {episode.path}"
            )
        if config.subtask_column not in extras.column_names:
            raise ValueError(
                f"Missing subtask column '{config.subtask_column}' in "
                f"{episode.path / 'extras.parquet'}"
            )
        raw_labels = extras.column(config.subtask_column).to_pylist()
        labels = [
            _as_label(value, episode_index=episode.index, frame_index=i)
            for i, value in enumerate(raw_labels)
        ]
        labels_by_episode.append(EpisodeLabels(index=episode.index, labels=labels))
    return labels_by_episode


def _normalize_explicit_order(order: Sequence[str] | None) -> list[str] | None:
    if order is None:
        return None
    names = [str(name).strip() for name in order]
    if not names or any(not name for name in names):
        raise ValueError("subtask_order must contain non-empty names")
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"subtask_order contains duplicate names: {duplicates}")
    return names


def _normalize_unlabeled_labels(
    labels_by_episode: list[EpisodeLabels],
    config: ValueTargetConfig,
) -> list[EpisodeLabels]:
    if config.allow_unlabeled == "error":
        for episode in labels_by_episode:
            unlabeled = [i for i, label in enumerate(episode.labels) if label is None]
            if unlabeled:
                raise ValueError(
                    f"Episode {episode.index} has unlabeled subtask frames; first indices: "
                    f"{unlabeled[:10]}"
                )
        return labels_by_episode

    if config.allow_unlabeled == "skip":
        return labels_by_episode

    if config.allow_unlabeled == "default_subtask":
        if not config.default_subtask:
            raise ValueError("allow_unlabeled=default_subtask requires default_subtask")
        return [
            EpisodeLabels(
                index=episode.index,
                labels=[
                    label if label is not None else config.default_subtask
                    for label in episode.labels
                ],
            )
            for episode in labels_by_episode
        ]

    raise ValueError(f"Unsupported allow_unlabeled mode: {config.allow_unlabeled}")


def _episode_segments(episode: EpisodeLabels) -> list[SubtaskSegment]:
    segments: list[SubtaskSegment] = []
    labels = episode.labels
    i = 0
    while i < len(labels):
        label = labels[i]
        if label is None:
            i += 1
            continue
        j = i + 1
        while j < len(labels) and labels[j] == label:
            j += 1
        segments.append(SubtaskSegment(episode.index, label, i, j - 1))
        i = j
    return segments


def _format_segments(segments: Sequence[SubtaskSegment]) -> str:
    return " -> ".join(f"{segment.name!r}[{segment.start}:{segment.end}]" for segment in segments)


def _resolve_and_validate_subtask_structure(
    root: Path,
    labels_by_episode: list[EpisodeLabels],
    config: ValueTargetConfig,
) -> tuple[list[str], list[SubtaskSegment]]:
    segments_by_episode = {
        episode.index: _episode_segments(episode) for episode in labels_by_episode
    }
    observed_names = {
        segment.name for segments in segments_by_episode.values() for segment in segments
    }
    if not observed_names:
        raise ValueError("No non-empty subtask labels found in extras.parquet")

    annotation_names = _load_annotation_subtasks(root)
    allowed_names = set(annotation_names)
    if config.allow_unlabeled == "default_subtask" and config.default_subtask:
        allowed_names.add(config.default_subtask)
    unknown = sorted(observed_names - allowed_names) if annotation_names else []
    if unknown:
        raise ValueError(
            f"Subtask labels are not present in {ANNOTATION_CONFIG_FILENAME}: {unknown}"
        )

    explicit_order = _normalize_explicit_order(config.subtask_order)
    expected_set = set(annotation_names) | (
        {config.default_subtask}
        if config.allow_unlabeled == "default_subtask" and config.default_subtask
        else set()
    )
    if not expected_set:
        expected_set = set(observed_names)

    if explicit_order is not None:
        missing_order_names = sorted(expected_set - set(explicit_order))
        extra_order_names = sorted(set(explicit_order) - expected_set)
        if missing_order_names or extra_order_names:
            raise ValueError(
                "Explicit subtask_order must exactly match the canonical subtask set: "
                f"missing={missing_order_names}, extra={extra_order_names}"
            )
        canonical_order = explicit_order
    else:
        canonical_order = []
        for episode in labels_by_episode:
            sequence = [segment.name for segment in segments_by_episode[episode.index]]
            if set(sequence) == expected_set:
                canonical_order = list(dict.fromkeys(sequence))
                break
        if not canonical_order:
            if not config.require_all_subtasks:
                canonical_order = [
                    segment.name for segment in segments_by_episode[labels_by_episode[0].index]
                ]
            else:
                raise ValueError(
                    "No episode contains every canonical subtask exactly once; "
                    f"expected={sorted(expected_set)}"
                )

    canonical_set = set(canonical_order)
    all_segments: list[SubtaskSegment] = []
    for episode in labels_by_episode:
        episode_segments = segments_by_episode[episode.index]
        all_segments.extend(episode_segments)
        sequence = [segment.name for segment in episode_segments]
        counts = Counter(sequence)
        missing = sorted(canonical_set - set(sequence))
        extra = sorted(set(sequence) - canonical_set)
        repeated = {
            name: [
                (segment.start, segment.end)
                for segment in episode_segments
                if segment.name == name
            ]
            for name, count in counts.items()
            if count > 1
        }
        canonical_index = {name: index for index, name in enumerate(canonical_order)}
        observed_ids = [canonical_index[name] for name in sequence if name in canonical_index]
        order_regressed = any(
            current < previous
            for previous, current in zip(observed_ids, observed_ids[1:], strict=False)
        )
        invalid = bool(extra)
        invalid |= config.require_all_subtasks and bool(missing)
        invalid |= config.require_single_segment_per_subtask and bool(repeated)
        invalid |= order_regressed
        if invalid:
            raise ValueError(
                f"Subtask structure invalid (Subtask order regressed or mismatched) in episode "
                f"{episode.index}: canonical={canonical_order}, observed={sequence}, "
                f"segments={_format_segments(episode_segments)}, missing={missing}, "
                f"extra={extra}, repeated={repeated}"
            )
    return canonical_order, all_segments


def _validate_success_only(config: ValueTargetConfig, root: Path) -> str:
    if not config.require_success_only:
        raise ValueError(
            "require_success_only=false is unsupported: "
            "failure/timeout/abort targets are not designed"
        )
    saw_outcome_field = False
    success_values = {"success", "successful", "completed", "complete", "done"}
    failure_values = {"failure", "failed", "timeout", "timed_out", "aborted", "abort"}
    for episode in discover_episodes(root):
        info_path = episode.path / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        for key in ("success", "successful"):
            if key in info:
                saw_outcome_field = True
                if info[key] is not True:
                    raise ValueError(
                        f"Episode {episode.index} is not explicitly successful: {key}={info[key]!r}"
                    )
        for key in ("outcome", "termination_reason"):
            if key not in info:
                continue
            saw_outcome_field = True
            value = str(info[key]).strip().lower()
            if value in failure_values or (key == "outcome" and value not in success_values):
                raise ValueError(
                    f"Episode {episode.index} violates success-only contract: {key}={info[key]!r}"
                )
    return "validated_outcome_fields" if saw_outcome_field else "declared_no_outcome_field"


def _validate_bin_counts(config: ValueTargetConfig) -> tuple[int, int]:
    values = {
        "num_bins": config.num_bins,
        "global_num_bins": (
            config.num_bins if config.global_num_bins is None else config.global_num_bins
        ),
        "subtask_num_bins": (
            config.num_bins if config.subtask_num_bins is None else config.subtask_num_bins
        ),
    }
    invalid = {name: value for name, value in values.items() if value < 2}
    if invalid:
        raise ValueError(f"All bin counts must be >= 2, got {invalid}")
    return values["global_num_bins"], values["subtask_num_bins"]


def _compute_global_scale(config: ValueTargetConfig) -> float:
    episodes = discover_episodes(config.root)
    max_remaining = [float(max(episode.frame_count - 1, 0)) for episode in episodes]
    return _scale_from_values(
        max_remaining,
        config.global_scale,
        config.global_scale_frames,
        name="global",
    )


def _compute_subtask_scales(
    config: ValueTargetConfig,
    subtask_names: list[str],
    segments: list[SubtaskSegment],
) -> dict[str, float]:
    values_by_name = {name: [] for name in subtask_names}
    for segment in segments:
        values_by_name.setdefault(segment.name, []).append(float(segment.max_remaining))

    scales: dict[str, float] = {}
    manual = dict(config.subtask_scale_frames or {})
    if config.subtask_scale == "manual":
        expected = set(subtask_names)
        missing = sorted(expected - set(manual))
        extra = sorted(set(manual) - expected)
        if missing or extra:
            raise ValueError(
                "Manual subtask scale keys must exactly match canonical subtask names: "
                f"missing={missing}, extra={extra}"
            )
    for name in subtask_names:
        scales[name] = _scale_from_values(
            values_by_name.get(name, []),
            config.subtask_scale,
            manual.get(name),
            name=f"subtask {name!r}",
        )
    return scales


def _global_columns(length: int, scale: float, *, elapsed_aux: bool) -> tuple[dict, dict]:
    remaining = np.arange(length - 1, -1, -1, dtype=np.int32)
    remaining_norm, remaining_clipped = _clip_norm(remaining.astype(np.float32), scale)
    columns = {
        VALUE_GLOBAL_REMAINING_FRAMES_GT: pa.array(remaining.tolist(), type=pa.int32()),
        VALUE_GLOBAL_REMAINING_NORM_GT: pa.array(remaining_norm.tolist(), type=pa.float32()),
        VALUE_GLOBAL_REMAINING_NORM_GT_IS_CLIPPED: pa.array(
            remaining_clipped.tolist(), type=pa.bool_()
        ),
    }
    summary = _clip_stats(remaining_clipped)
    if elapsed_aux:
        elapsed = np.arange(length, dtype=np.int32)
        elapsed_norm, _ = _clip_norm(elapsed.astype(np.float32), scale)
        columns[VALUE_GLOBAL_ELAPSED_FRAMES_GT] = pa.array(elapsed.tolist(), type=pa.int32())
        columns[VALUE_GLOBAL_ELAPSED_NORM_GT] = pa.array(elapsed_norm.tolist(), type=pa.float32())
    return columns, summary


def _subtask_columns(
    episode_labels: EpisodeLabels,
    subtask_names: list[str],
    subtask_scales: Mapping[str, float],
    *,
    elapsed_aux: bool,
) -> tuple[dict, dict]:
    subtask_to_id = {name: i for i, name in enumerate(subtask_names)}
    length = len(episode_labels.labels)
    ids = np.full(length, -1, dtype=np.int32)
    names = [""] * length
    remaining = np.full(length, np.nan, dtype=np.float32)
    elapsed = np.full(length, np.nan, dtype=np.float32)

    i = 0
    labels = episode_labels.labels
    while i < length:
        label = labels[i]
        if label is None:
            i += 1
            continue
        j = i + 1
        while j < length and labels[j] == label:
            j += 1
        subtask_id = subtask_to_id[label]
        end = j - 1
        for frame_index in range(i, j):
            ids[frame_index] = subtask_id
            names[frame_index] = label
            remaining[frame_index] = float(end - frame_index)
            elapsed[frame_index] = float(frame_index - i)
        i = j

    remaining_norm = np.full(length, np.nan, dtype=np.float32)
    remaining_clipped = np.zeros(length, dtype=bool)
    elapsed_norm = np.full(length, np.nan, dtype=np.float32)
    clip_stats_by_subtask = {}
    for name, subtask_id in subtask_to_id.items():
        mask = ids == subtask_id
        if not mask.any():
            continue
        norm, clipped = _clip_norm(remaining[mask], subtask_scales[name])
        remaining_norm[mask] = norm
        remaining_clipped[mask] = clipped
        clip_stats_by_subtask[name] = _clip_stats(clipped)
        if elapsed_aux:
            elapsed_norm[mask] = _clip_norm(elapsed[mask], subtask_scales[name])[0]

    columns = {
        VALUE_SUBTASK_ID_GT: pa.array(ids.tolist(), type=pa.int32()),
        VALUE_SUBTASK_NAME_GT: pa.array(names, type=pa.string()),
        VALUE_SUBTASK_REMAINING_FRAMES_GT: pa.array(remaining.tolist(), type=pa.float32()),
        VALUE_SUBTASK_REMAINING_NORM_GT: pa.array(remaining_norm.tolist(), type=pa.float32()),
        VALUE_SUBTASK_REMAINING_NORM_GT_IS_CLIPPED: pa.array(
            remaining_clipped.tolist(), type=pa.bool_()
        ),
    }
    if elapsed_aux:
        columns[VALUE_SUBTASK_ELAPSED_FRAMES_GT] = pa.array(elapsed.tolist(), type=pa.float32())
        columns[VALUE_SUBTASK_ELAPSED_NORM_GT] = pa.array(elapsed_norm.tolist(), type=pa.float32())
    return columns, {"clip_stats_by_subtask": clip_stats_by_subtask}


def prepare_value_targets(config: ValueTargetConfig) -> dict:
    root = Path(config.root).expanduser().resolve()
    episodes = discover_episodes(root)
    run_meta = read_run_meta(root)
    global_num_bins, subtask_num_bins = _validate_bin_counts(config)
    success_validation = _validate_success_only(config, root)

    labels_by_episode: list[EpisodeLabels] = []
    subtask_names: list[str] = []
    segments: list[SubtaskSegment] = []
    subtask_scales: dict[str, float] = {}
    if _needs_subtask(config.mode):
        loaded_labels = _load_episode_labels(config)
        labels_by_episode = _normalize_unlabeled_labels(loaded_labels, config)
        subtask_names, segments = _resolve_and_validate_subtask_structure(
            root, labels_by_episode, config
        )
        subtask_scales = _compute_subtask_scales(config, subtask_names, segments)

    global_scale = None
    if _needs_global(config.mode):
        global_scale = _compute_global_scale(config)

    labels_by_index = {episode.index: episode for episode in labels_by_episode}
    episode_columns: dict[int, dict] = {}
    episode_summaries = []
    global_clip_counts = {"clipped_frames": 0, "eligible_frames": 0}
    subtask_clip_counts = {
        name: {"clipped_frames": 0, "eligible_frames": 0} for name in subtask_names
    }
    for episode in episodes:
        columns: dict = {}
        ep_summary = {"index": episode.index, "length": episode.frame_count}
        if global_scale is not None:
            global_cols, global_summary = _global_columns(
                episode.frame_count, global_scale, elapsed_aux=config.elapsed_aux
            )
            columns.update(global_cols)
            global_clip_counts["clipped_frames"] += global_summary["clipped_frames"]
            global_clip_counts["eligible_frames"] += global_summary["eligible_frames"]
            ep_summary["global_clip_rate"] = global_summary["clip_rate"]
        if _needs_subtask(config.mode):
            subtask_cols, subtask_summary = _subtask_columns(
                labels_by_index[episode.index],
                subtask_names,
                subtask_scales,
                elapsed_aux=config.elapsed_aux,
            )
            columns.update(subtask_cols)
            for name, stats in subtask_summary["clip_stats_by_subtask"].items():
                subtask_clip_counts[name]["clipped_frames"] += stats["clipped_frames"]
                subtask_clip_counts[name]["eligible_frames"] += stats["eligible_frames"]
            ep_summary["subtask_clip_stats_by_subtask"] = subtask_summary[
                "clip_stats_by_subtask"
            ]
        episode_columns[episode.index] = columns
        episode_summaries.append(ep_summary)

    columns_written = sorted({name for columns in episode_columns.values() for name in columns})
    clip_summary = {}
    if _needs_global(config.mode):
        eligible = global_clip_counts["eligible_frames"]
        clip_summary["global"] = {
            **global_clip_counts,
            "clip_rate": (
                global_clip_counts["clipped_frames"] / eligible if eligible else 0.0
            ),
        }
        clip_summary["global_clip_rate"] = clip_summary["global"]["clip_rate"]
    if _needs_subtask(config.mode):
        clip_summary["subtask_by_name"] = {
            name: {
                **counts,
                "clip_rate": (
                    counts["clipped_frames"] / counts["eligible_frames"]
                    if counts["eligible_frames"]
                    else 0.0
                ),
            }
            for name, counts in subtask_clip_counts.items()
        }
        clip_summary["subtask_clip_rate_by_subtask"] = {
            name: stats["clip_rate"]
            for name, stats in clip_summary["subtask_by_name"].items()
        }

    existing_target_columns: set[str] = set()
    for episode in episodes:
        extras = read_extras_table(episode.path)
        if extras is not None:
            existing_target_columns.update(set(extras.column_names) & set(TARGET_COLUMNS))
    target_columns_manifest = {
        "active": columns_written,
        "inactive_present": sorted(existing_target_columns - set(columns_written)),
    }

    summary = {
        "root": str(root),
        "mode": config.mode,
        "dry_run": config.dry_run,
        "num_bins": config.num_bins,
        "global_num_bins": global_num_bins,
        "subtask_num_bins": subtask_num_bins,
        "global_scale": (
            {"strategy": config.global_scale, "frames": global_scale}
            if global_scale is not None
            else None
        ),
        "subtask_scale": {
            "strategy": config.subtask_scale,
            "frames_by_subtask": subtask_scales,
        }
        if subtask_scales
        else None,
        "subtask_names": subtask_names,
        "subtask_segments": [
            {
                "episode_index": segment.episode_index,
                "name": segment.name,
                "start": segment.start,
                "end": segment.end,
            }
            for segment in segments
        ],
        "episodes": episode_summaries,
        "total_frames": sum(episode.frame_count for episode in episodes),
        "elapsed_aux": config.elapsed_aux,
        "allow_unlabeled": config.allow_unlabeled,
        "all_episodes_successful": True,
        "success_validation": success_validation,
        "subtask_order": subtask_names,
        "strict_validation": {
            "require_all_subtasks": config.require_all_subtasks,
            "require_single_segment_per_subtask": config.require_single_segment_per_subtask,
            "require_success_only": config.require_success_only,
        },
        "image_keys": get_image_keys(run_meta),
        "columns_written": columns_written,
        "target_columns": target_columns_manifest,
        "clip_summary": clip_summary,
    }

    if not config.dry_run:
        input_columns = [config.subtask_column] if _needs_subtask(config.mode) else []
        input_fingerprint = fingerprint_raw_run_columns(root, input_columns)
        merge_raw_run_extras(root, episode_columns)
        output_fingerprint = fingerprint_raw_run_columns(root, columns_written)
        update_stage_metadata(
            root,
            TARGET_STAGE,
            config=config,
            input_columns=input_columns,
            input_fingerprint=input_fingerprint,
            output_columns=columns_written,
            output_fingerprint=output_fingerprint,
            prediction_source=PREDICTION_SOURCE_GT,
            synthetic=False,
            metadata_patch={
                "all_episodes_successful": True,
                "success_validation": success_validation,
                "subtask_order": subtask_names,
                "strict_validation": summary["strict_validation"],
                "value_mode": config.mode,
                "num_bins": config.num_bins,
                "global_num_bins": summary["global_num_bins"],
                "subtask_num_bins": summary["subtask_num_bins"],
                "global_scale": summary["global_scale"],
                "subtask_names": subtask_names,
                "subtask_scale": summary["subtask_scale"],
                "elapsed_aux": config.elapsed_aux,
                "allow_unlabeled": config.allow_unlabeled,
                "image_keys": summary["image_keys"],
                "columns_written": columns_written,
                "target_columns": target_columns_manifest,
                "clip_summary": clip_summary,
            },
        )

    return summary
