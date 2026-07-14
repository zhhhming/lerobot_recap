"""Group-relative loss weights for advantage-labeled action chunks."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pyarrow as pa

from lerobot.value_function.advantage_labeling import (
    advantage_columns,
    advantage_export_eligibility,
    label_column,
)
from lerobot.value_function.raw_io import (
    assert_stage_dependencies_current,
    discover_episodes,
    fingerprint_raw_run_columns,
    merge_raw_run_extras,
    read_extras_table,
    read_value_function_metadata,
    update_stage_metadata,
)
from lerobot.value_function.schema import (
    ADVANTAGE_GROUP_ID_GLOBAL,
    ADVANTAGE_GROUP_ID_SUBTASK,
    ADVANTAGE_LABELING_STAGE_PREFIX,
    ADVANTAGE_LOSS_WEIGHT_GLOBAL,
    ADVANTAGE_LOSS_WEIGHT_SUBTASK,
    ADVANTAGE_WEIGHTS_STAGE_PREFIX,
    MOCK_PREDICTIONS_STAGE,
    TARGET_STAGE,
    VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED,
    VALUE_GLOBAL_REMAINING_NORM_GT,
    VALUE_GLOBAL_REMAINING_NORM_PRED,
    VALUE_INFERENCE_STAGE_PREFIX,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_ID_PRED_SMOOTH,
    VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED,
    VALUE_SUBTASK_REMAINING_NORM_GT,
    VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD,
    VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD,
)

ValueMode = Literal["global", "subtask"]
GroupSource = Literal["auto", "progress", "value"]

VALID_LABELS = {"positive", "negative", "ignore"}
SUBTASK_PROGRESS_COLUMN = "subtask_progress"


@dataclass(frozen=True)
class AdvantageWeightConfig:
    root: Path | str
    value_mode: ValueMode = "global"
    group_source: GroupSource = "auto"
    group_bin_width: float | None = None
    q: float = 0.8
    tau: float = 0.08
    w_min: float = 0.1
    w_max: float = 2.0
    positive_group_max_weight: float = 2.0
    min_group_size: int = 4
    negative_weight: float = 1.0
    allow_synthetic: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class GroupingSpec:
    source: Literal["progress", "value"]
    scalar_column: str
    subtask_id_column: str | None
    scale_by_subtask: tuple[float, ...] | None
    scalar_is_frame_value: bool
    input_columns: tuple[str, ...]
    dependency_stages: tuple[str, ...]


def group_id_column(value_mode: ValueMode) -> str:
    return (
        ADVANTAGE_GROUP_ID_GLOBAL
        if value_mode == "global"
        else ADVANTAGE_GROUP_ID_SUBTASK
    )


def loss_weight_column(value_mode: ValueMode) -> str:
    return (
        ADVANTAGE_LOSS_WEIGHT_GLOBAL
        if value_mode == "global"
        else ADVANTAGE_LOSS_WEIGHT_SUBTASK
    )


def default_group_bin_width(value_mode: ValueMode) -> float:
    return 0.05 if value_mode == "global" else 0.1


def weight_output_columns(value_mode: ValueMode) -> tuple[str, str]:
    return group_id_column(value_mode), loss_weight_column(value_mode)


def _validate_config(config: AdvantageWeightConfig) -> float:
    if config.value_mode not in ("global", "subtask"):
        raise ValueError(f"Unsupported value_mode: {config.value_mode!r}")
    if config.group_source not in ("auto", "progress", "value"):
        raise ValueError(f"Unsupported group_source: {config.group_source!r}")
    width = (
        default_group_bin_width(config.value_mode)
        if config.group_bin_width is None
        else float(config.group_bin_width)
    )
    finite = {
        "group_bin_width": width,
        "q": config.q,
        "tau": config.tau,
        "w_min": config.w_min,
        "w_max": config.w_max,
        "positive_group_max_weight": config.positive_group_max_weight,
        "negative_weight": config.negative_weight,
    }
    for name, value in finite.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value}")
    if width <= 0:
        raise ValueError("group_bin_width must be > 0")
    if not 0 <= config.q <= 1:
        raise ValueError("q must be in [0, 1]")
    if config.tau <= 0:
        raise ValueError("tau must be > 0")
    if config.w_min < 0 or config.w_max < config.w_min:
        raise ValueError("weights must satisfy 0 <= w_min <= w_max")
    if config.w_max <= 0 or config.positive_group_max_weight <= 0:
        raise ValueError("w_max and positive_group_max_weight must be > 0")
    if config.min_group_size < 1:
        raise ValueError("min_group_size must be >= 1")
    if config.negative_weight < 0:
        raise ValueError("negative_weight must be >= 0")
    return width


def _require_active_columns(
    root: Path, metadata: dict, stage_name: str, columns: Sequence[str]
) -> None:
    assert_stage_dependencies_current(root, stage_name)
    stage = (metadata.get("stages") or {}).get(stage_name)
    if not isinstance(stage, dict):
        raise ValueError(f"Missing pipeline stage metadata for {stage_name!r}")
    missing = sorted(set(columns) - set(stage.get("output_columns") or []))
    if missing:
        raise ValueError(f"Stage {stage_name!r} is missing active columns: {missing}")


def _subtask_scales(metadata: dict) -> tuple[float, ...]:
    order = list(metadata.get("subtask_order") or [])
    scales = ((metadata.get("subtask_scale") or {}).get("frames_by_subtask") or {})
    if not order or set(scales) != set(order):
        raise ValueError("Mock subtask value grouping requires canonical subtask scales")
    values = tuple(float(scales[name]) for name in order)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Subtask scales must be finite and > 0")
    return values


def _resolve_grouping_spec(root: Path, config: AdvantageWeightConfig, metadata: dict) -> GroupingSpec:
    detail = (metadata.get("advantage") or {}).get(config.value_mode)
    if not isinstance(detail, dict):
        raise ValueError(f"Missing advantage metadata for mode={config.value_mode!r}")
    value_source = detail.get("value_source") or detail.get("prediction_source")
    inference_path = detail.get("subtask_inference_path")
    requested = config.group_source
    if config.value_mode == "global":
        if requested == "progress":
            raise ValueError("global grouping supports group_source='auto' or 'value'")
        if value_source == "model_pred":
            column = VALUE_GLOBAL_REMAINING_NORM_PRED
            stage = f"{VALUE_INFERENCE_STAGE_PREFIX}.global"
            _require_active_columns(root, metadata, stage, [column])
            return GroupingSpec("value", column, None, None, False, (column,), (stage,))
        if value_source == "gt":
            column = VALUE_GLOBAL_REMAINING_NORM_GT
            _require_active_columns(root, metadata, TARGET_STAGE, [column])
            return GroupingSpec("value", column, None, None, False, (column,), (TARGET_STAGE,))
        if value_source == "mock_pred":
            column = VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED
            _require_active_columns(root, metadata, MOCK_PREDICTIONS_STAGE, [column])
            scale = float(((metadata.get("global_scale") or {}).get("frames")))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Mock global value grouping requires global_scale.frames > 0")
            return GroupingSpec(
                "value", column, None, (scale,), True, (column,), (MOCK_PREDICTIONS_STAGE,)
            )
        raise ValueError(f"Unsupported advantage value_source: {value_source!r}")

    resolved_source: Literal["progress", "value"]
    if requested == "auto":
        resolved_source = "value" if inference_path == "pred_smooth" else "progress"
    else:
        resolved_source = requested
    boundary_column = (
        VALUE_SUBTASK_ID_PRED_SMOOTH
        if inference_path == "pred_smooth"
        else VALUE_SUBTASK_ID_GT
    )
    if resolved_source == "progress":
        if inference_path == "pred_smooth":
            raise ValueError(
                "group_source='progress' cannot pair GT progress with pred_smooth boundaries; "
                "use group_source='value'"
            )
        return GroupingSpec(
            "progress",
            SUBTASK_PROGRESS_COLUMN,
            boundary_column,
            None,
            False,
            (boundary_column, SUBTASK_PROGRESS_COLUMN),
            (),
        )

    if value_source == "model_pred":
        column = (
            VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD
            if inference_path == "pred_smooth"
            else VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD
        )
        stage = f"{VALUE_INFERENCE_STAGE_PREFIX}.subtask"
        _require_active_columns(root, metadata, stage, [boundary_column, column])
        return GroupingSpec(
            "value", column, boundary_column, None, False, (boundary_column, column), (stage,)
        )
    if value_source == "gt":
        column = VALUE_SUBTASK_REMAINING_NORM_GT
        _require_active_columns(root, metadata, TARGET_STAGE, [boundary_column, column])
        return GroupingSpec(
            "value", column, boundary_column, None, False, (boundary_column, column), (TARGET_STAGE,)
        )
    if value_source == "mock_pred":
        column = VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED
        _require_active_columns(root, metadata, MOCK_PREDICTIONS_STAGE, [column])
        return GroupingSpec(
            "value",
            column,
            boundary_column,
            _subtask_scales(metadata),
            True,
            (boundary_column, column),
            (MOCK_PREDICTIONS_STAGE,),
        )
    raise ValueError(f"Unsupported advantage value_source: {value_source!r}")


def _group_ids(
    scalars: np.ndarray,
    *,
    bin_width: float,
    subtask_ids: np.ndarray | None,
) -> list[str]:
    if not np.isfinite(scalars).all():
        indices = np.flatnonzero(~np.isfinite(scalars))[:10].tolist()
        raise ValueError(f"Grouping values must be finite; invalid frame indices: {indices}")
    bins = np.floor(scalars.astype(np.float64) / bin_width).astype(np.int64)
    if subtask_ids is None:
        return [f"global:bin:{int(bin_index):+06d}" for bin_index in bins]
    if np.any(subtask_ids < 0):
        indices = np.flatnonzero(subtask_ids < 0)[:10].tolist()
        raise ValueError(f"Subtask ids must be non-negative; invalid frame indices: {indices}")
    return [
        f"subtask:{int(subtask_id):04d}:bin:{int(bin_index):+06d}"
        for subtask_id, bin_index in zip(subtask_ids, bins, strict=True)
    ]


def _average_tie_ranks(indices: list[int], advantages: np.ndarray) -> dict[int, float]:
    ordered = sorted(indices, key=lambda index: (-float(advantages[index]), index))
    ranks: dict[int, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        value = float(advantages[ordered[start]])
        while end < len(ordered) and float(advantages[ordered[end]]) == value:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def compute_group_relative_weights(
    group_ids: Sequence[str],
    advantages: Sequence[float],
    labels: Sequence[str],
    *,
    q: float = 0.8,
    tau: float = 0.08,
    w_min: float = 0.1,
    w_max: float = 2.0,
    positive_group_max_weight: float = 2.0,
    min_group_size: int = 4,
    negative_weight: float = 1.0,
) -> tuple[np.ndarray, list[float | None], dict[str, dict]]:
    if not (len(group_ids) == len(advantages) == len(labels)):
        raise ValueError("group_ids, advantages, and labels must have equal length")
    validation = AdvantageWeightConfig(
        root=".",
        q=q,
        tau=tau,
        w_min=w_min,
        w_max=w_max,
        positive_group_max_weight=positive_group_max_weight,
        min_group_size=min_group_size,
        negative_weight=negative_weight,
    )
    _validate_config(validation)
    normalized_labels = [str(label).strip().lower() for label in labels]
    invalid_labels = sorted(set(normalized_labels) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(f"Unsupported advantage labels: {invalid_labels}")
    advantage_array = np.asarray(advantages, dtype=np.float64)
    if not np.isfinite(advantage_array).all():
        raise ValueError("Advantages must be finite")

    weights = np.asarray(
        [0.0 if label == "ignore" else negative_weight for label in normalized_labels],
        dtype=np.float64,
    )
    normalized_ranks: list[float | None] = [None] * len(group_ids)
    positives: dict[str, list[int]] = defaultdict(list)
    members: dict[str, list[int]] = defaultdict(list)
    for index, (group_id, label) in enumerate(zip(group_ids, normalized_labels, strict=True)):
        members[str(group_id)].append(index)
        if label == "positive":
            positives[str(group_id)].append(index)

    summaries: dict[str, dict] = {}
    for group_id in sorted(members):
        positive_indices = positives.get(group_id, [])
        weighted = len(positive_indices) >= min_group_size
        if positive_indices:
            ranks = _average_tie_ranks(positive_indices, advantage_array)
            denominator = max(len(positive_indices) - 1, 1)
            raw: dict[int, float] = {}
            for index, rank in ranks.items():
                u = rank / denominator
                normalized_ranks[index] = u
                raw[index] = w_min + (w_max - w_min) * _sigmoid((q - u) / tau)
            if weighted:
                group_max = max(raw.values())
                for index in positive_indices:
                    weights[index] = (
                        raw[index] / group_max * positive_group_max_weight
                    )
            else:
                weights[positive_indices] = 1.0
        group_labels = Counter(normalized_labels[index] for index in members[group_id])
        group_weights = weights[members[group_id]]
        summaries[group_id] = {
            "size": len(members[group_id]),
            "positive_count": len(positive_indices),
            "negative_count": group_labels["negative"],
            "ignore_count": group_labels["ignore"],
            "rank_weighted": weighted,
            "min_weight": float(group_weights.min()),
            "max_weight": float(group_weights.max()),
        }
    return weights.astype(np.float32), normalized_ranks, summaries


def _episode_grouping_values(
    extras: pa.Table,
    spec: GroupingSpec,
) -> tuple[np.ndarray, np.ndarray | None]:
    missing = [name for name in spec.input_columns if name not in extras.column_names]
    if missing:
        raise ValueError(f"Missing grouping columns in extras.parquet: {missing}")
    scalars = np.asarray(extras.column(spec.scalar_column).to_pylist(), dtype=np.float64)
    ids = None
    if spec.subtask_id_column is not None:
        ids = np.asarray(extras.column(spec.subtask_id_column).to_pylist(), dtype=np.int64)
    if spec.scalar_is_frame_value:
        if spec.scale_by_subtask is None:
            raise AssertionError("Frame-unit grouping requires scale metadata")
        if ids is None:
            scalars = scalars / spec.scale_by_subtask[0]
        else:
            scale_array = np.asarray(spec.scale_by_subtask, dtype=np.float64)
            if np.any(ids >= len(scale_array)):
                raise ValueError("Subtask id exceeds canonical scale table")
            scalars = scalars / scale_array[ids]
    return scalars, ids


def compute_advantage_weights(config: AdvantageWeightConfig) -> dict:
    bin_width = _validate_config(config)
    root = Path(config.root).expanduser().resolve()
    labeling_stage = f"{ADVANTAGE_LABELING_STAGE_PREFIX}.{config.value_mode}"
    assert_stage_dependencies_current(root, labeling_stage)
    eligibility = advantage_export_eligibility(root, config.value_mode)
    if not eligibility["experiment_eligible"] and not config.allow_synthetic:
        raise ValueError(
            f"Formal weight export requires model_pred labels; got "
            f"{eligibility['prediction_source']!r}. Use allow_synthetic only for tests."
        )
    metadata = read_value_function_metadata(root)
    spec = _resolve_grouping_spec(root, config, metadata)
    advantage_column = advantage_columns(config.value_mode)[0]
    label_name = label_column(config.value_mode)
    input_columns = list(dict.fromkeys([label_name, advantage_column, *spec.input_columns]))

    flat_groups: list[str] = []
    flat_advantages: list[float] = []
    flat_labels: list[str] = []
    episode_ranges: dict[int, tuple[int, int]] = {}
    for episode in discover_episodes(root):
        extras = read_extras_table(episode.path)
        if extras is None:
            raise FileNotFoundError(f"Missing extras.parquet in {episode.path}")
        missing = [name for name in (label_name, advantage_column) if name not in extras.column_names]
        if missing:
            raise ValueError(f"Missing weight input columns in {episode.path}: {missing}")
        scalars, subtask_ids = _episode_grouping_values(extras, spec)
        groups = _group_ids(scalars, bin_width=bin_width, subtask_ids=subtask_ids)
        advantages = [float(value) for value in extras.column(advantage_column).to_pylist()]
        labels = [str(value) for value in extras.column(label_name).to_pylist()]
        start = len(flat_groups)
        flat_groups.extend(groups)
        flat_advantages.extend(advantages)
        flat_labels.extend(labels)
        episode_ranges[episode.index] = (start, len(flat_groups))

    weights, _ranks, group_summaries = compute_group_relative_weights(
        flat_groups,
        flat_advantages,
        flat_labels,
        q=config.q,
        tau=config.tau,
        w_min=config.w_min,
        w_max=config.w_max,
        positive_group_max_weight=config.positive_group_max_weight,
        min_group_size=config.min_group_size,
        negative_weight=config.negative_weight,
    )
    group_column, weight_column = weight_output_columns(config.value_mode)
    episode_columns: dict[int, dict[str, pa.Array]] = {}
    for episode_index, (start, end) in episode_ranges.items():
        episode_columns[episode_index] = {
            group_column: pa.array(flat_groups[start:end], type=pa.string()),
            weight_column: pa.array(weights[start:end].tolist(), type=pa.float32()),
        }
    label_counts = Counter(flat_labels)
    weighted_groups = sum(group["rank_weighted"] for group in group_summaries.values())
    summary = {
        "root": str(root),
        "value_mode": config.value_mode,
        "group_source": spec.source,
        "group_scalar_column": spec.scalar_column,
        "group_subtask_id_column": spec.subtask_id_column,
        "group_bin_width": bin_width,
        "q": config.q,
        "tau": config.tau,
        "w_min": config.w_min,
        "w_max": config.w_max,
        "positive_group_max_weight": config.positive_group_max_weight,
        "min_group_size": config.min_group_size,
        "negative_weight": config.negative_weight,
        "ignore_weight": 0.0,
        "dry_run": config.dry_run,
        "eligibility": eligibility,
        "total_chunks": len(flat_groups),
        "label_counts": dict(sorted(label_counts.items())),
        "group_count": len(group_summaries),
        "rank_weighted_group_count": weighted_groups,
        "small_group_count": len(group_summaries) - weighted_groups,
        "weight_min": float(weights.min()) if len(weights) else None,
        "weight_max": float(weights.max()) if len(weights) else None,
        "columns_written": [group_column, weight_column],
    }
    if not config.dry_run:
        input_fingerprint = fingerprint_raw_run_columns(root, input_columns)
        merge_raw_run_extras(root, episode_columns)
        output_fingerprint = fingerprint_raw_run_columns(root, summary["columns_written"])
        clean_metadata = read_value_function_metadata(root)
        weights_meta = dict(clean_metadata.get("advantage_weights") or {})
        weights_meta[config.value_mode] = summary
        existing_columns = set(clean_metadata.get("columns_written") or [])
        dependencies = list(
            dict.fromkeys([labeling_stage, *spec.dependency_stages])
        )
        update_stage_metadata(
            root,
            f"{ADVANTAGE_WEIGHTS_STAGE_PREFIX}.{config.value_mode}",
            config=config,
            input_columns=input_columns,
            input_fingerprint=input_fingerprint,
            output_columns=summary["columns_written"],
            output_fingerprint=output_fingerprint,
            prediction_source=eligibility["prediction_source"],
            synthetic=bool(eligibility["synthetic"]),
            dependencies=dependencies,
            metadata_patch={
                "advantage_weights": weights_meta,
                "columns_written": sorted(existing_columns | set(summary["columns_written"])),
            },
        )
    return summary


def load_advantage_weight_chunks(
    root: Path | str, value_mode: ValueMode = "global"
) -> tuple[list[dict], list[dict]]:
    """Load persisted weights and reconstruct positive ranks for visualization."""

    root = Path(root).expanduser().resolve()
    stage_name = f"{ADVANTAGE_WEIGHTS_STAGE_PREFIX}.{value_mode}"
    assert_stage_dependencies_current(root, stage_name)
    metadata = read_value_function_metadata(root)
    stage = (metadata.get("stages") or {}).get(stage_name)
    if not isinstance(stage, dict):
        raise ValueError(f"Missing current advantage weights metadata for mode={value_mode!r}")
    group_column, weight_column = weight_output_columns(value_mode)
    advantage_column, horizon_column, valid_column = advantage_columns(value_mode)
    label_name = label_column(value_mode)
    required = [
        group_column,
        weight_column,
        advantage_column,
        horizon_column,
        valid_column,
        label_name,
    ]
    missing_active = sorted(
        {group_column, weight_column} - set(stage.get("output_columns") or [])
    )
    if missing_active:
        raise ValueError(f"Weights stage {stage_name!r} is missing active columns: {missing_active}")

    chunks: list[dict] = []
    for episode in discover_episodes(root):
        extras = read_extras_table(episode.path)
        if extras is None:
            raise FileNotFoundError(f"Missing extras.parquet in {episode.path}")
        missing = [name for name in required if name not in extras.column_names]
        if missing:
            raise ValueError(f"Missing weight visualization columns in {episode.path}: {missing}")
        columns = {name: extras.column(name).to_pylist() for name in required}
        for frame_index in range(episode.frame_count):
            chunks.append(
                {
                    "key": f"ep_{episode.index:06d}:frame_{frame_index:06d}",
                    "episode_index": episode.index,
                    "episode_name": f"ep_{episode.index:06d}",
                    "frame_index": frame_index,
                    "group_id": str(columns[group_column][frame_index]),
                    "advantage": float(columns[advantage_column][frame_index]),
                    "valid_horizon": int(columns[horizon_column][frame_index]),
                    "is_valid": bool(columns[valid_column][frame_index]),
                    "label": str(columns[label_name][frame_index]),
                    "weight": float(columns[weight_column][frame_index]),
                    "positive_rank": None,
                    "positive_rank_u": None,
                    "positive_group_size": 0,
                }
            )

    advantages = np.asarray([chunk["advantage"] for chunk in chunks], dtype=np.float64)
    group_members: dict[str, list[int]] = defaultdict(list)
    positive_members: dict[str, list[int]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        group_members[chunk["group_id"]].append(index)
        if chunk["label"] == "positive":
            positive_members[chunk["group_id"]].append(index)

    groups: list[dict] = []
    for group_id in sorted(group_members):
        members = group_members[group_id]
        positives = positive_members.get(group_id, [])
        ranks = _average_tie_ranks(positives, advantages)
        denominator = max(len(positives) - 1, 1)
        for index, rank in ranks.items():
            chunks[index]["positive_rank"] = rank + 1.0
            chunks[index]["positive_rank_u"] = rank / denominator
            chunks[index]["positive_group_size"] = len(positives)
        labels = Counter(chunks[index]["label"] for index in members)
        group_weights = [chunks[index]["weight"] for index in members]
        groups.append(
            {
                "group_id": group_id,
                "size": len(members),
                "positive_count": labels["positive"],
                "negative_count": labels["negative"],
                "ignore_count": labels["ignore"],
                "min_weight": min(group_weights),
                "max_weight": max(group_weights),
            }
        )
    return chunks, groups
