"""Action-chunk advantage computation for raw LeRobot runs."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa

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
    ADVANTAGE_GLOBAL_CHUNK,
    ADVANTAGE_GLOBAL_END_VALUE,
    ADVANTAGE_GLOBAL_IS_VALID,
    ADVANTAGE_GLOBAL_START_VALUE,
    ADVANTAGE_GLOBAL_VALID_HORIZON,
    ADVANTAGE_STAGE_PREFIX,
    ADVANTAGE_SUBTASK_BOUNDARY_PROGRESS,
    ADVANTAGE_SUBTASK_CHUNK,
    ADVANTAGE_SUBTASK_END_VALUE,
    ADVANTAGE_SUBTASK_IS_VALID,
    ADVANTAGE_SUBTASK_NUM_CROSSINGS,
    ADVANTAGE_SUBTASK_START_VALUE,
    ADVANTAGE_SUBTASK_VALID_HORIZON,
    ADVANTAGE_SUBTASK_WITHIN_SUBTASK_HORIZON,
    MOCK_PREDICTIONS_STAGE,
    PREDICTION_SOURCE_GT,
    PREDICTION_SOURCE_MOCK,
    PREDICTION_SOURCE_MODEL,
    TARGET_STAGE,
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED,
    VALUE_GLOBAL_REMAINING_FRAMES_PRED,
    VALUE_INFERENCE_STAGE_PREFIX,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_ID_PRED_SMOOTH,
    VALUE_SUBTASK_REMAINING_FRAMES_GT,
    VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED,
    VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD,
    VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD,
)

ValueMode = Literal["global", "subtask"]
ValueSource = Literal["gt", "mock_pred", "model_pred"]
SubtaskInferencePath = Literal["gt_conditioned", "pred_smooth"]

GLOBAL_FORMULA_VERSION = "global_centered_v1"
SUBTASK_FORMULA_VERSION = "subtask_boundary_transition_v2"


@dataclass(frozen=True)
class AdvantageConfig:
    root: Path | str
    value_mode: ValueMode = "global"
    value_source: ValueSource = "gt"
    chunk_size: int = 50
    subtask_inference_path: SubtaskInferencePath = "gt_conditioned"
    boundary_transition_value: float = 1.0
    boundary_bonus: float | None = None
    dry_run: bool = False


def _resolved_boundary_transition_value(config: AdvantageConfig) -> float:
    if config.boundary_bonus is not None:
        if config.boundary_transition_value != 1.0:
            raise ValueError(
                "boundary_bonus and boundary_transition_value cannot both be set"
            )
        warnings.warn(
            "boundary_bonus is deprecated; use boundary_transition_value. "
            "The compatibility mapping is boundary_transition_value=boundary_bonus+1.",
            FutureWarning,
            stacklevel=3,
        )
        return float(config.boundary_bonus) + 1.0
    return float(config.boundary_transition_value)


def _resolve_source_columns(config: AdvantageConfig) -> tuple[str, str | None]:
    if config.value_mode == "global":
        values = {
            "gt": VALUE_GLOBAL_REMAINING_FRAMES_GT,
            "mock_pred": VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED,
            "model_pred": VALUE_GLOBAL_REMAINING_FRAMES_PRED,
        }
        return values[config.value_source], None

    if config.value_source in ("gt", "mock_pred"):
        if config.subtask_inference_path != "gt_conditioned":
            raise ValueError(
                f"value_source={config.value_source!r} only supports "
                "subtask_inference_path='gt_conditioned'"
            )
        value_column = (
            VALUE_SUBTASK_REMAINING_FRAMES_GT
            if config.value_source == "gt"
            else VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED
        )
        return value_column, VALUE_SUBTASK_ID_GT

    if config.subtask_inference_path == "gt_conditioned":
        return VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD, VALUE_SUBTASK_ID_GT
    return (
        VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD,
        VALUE_SUBTASK_ID_PRED_SMOOTH,
    )


def _source_stage(config: AdvantageConfig) -> str:
    if config.value_source == "gt":
        return TARGET_STAGE
    if config.value_source == "mock_pred":
        return MOCK_PREDICTIONS_STAGE
    return f"{VALUE_INFERENCE_STAGE_PREFIX}.{config.value_mode}"


def _expected_prediction_source(config: AdvantageConfig) -> str:
    return {
        "gt": PREDICTION_SOURCE_GT,
        "mock_pred": PREDICTION_SOURCE_MOCK,
        "model_pred": PREDICTION_SOURCE_MODEL,
    }[config.value_source]


def _validate_source_provenance(
    root: Path,
    config: AdvantageConfig,
    required_columns: list[str],
) -> list[str]:
    stage_name = _source_stage(config)
    assert_stage_dependencies_current(root, stage_name)
    metadata = read_value_function_metadata(root)
    record = (metadata.get("stages") or {}).get(stage_name)
    if not isinstance(record, dict):
        raise ValueError(f"Missing source stage metadata for {stage_name!r}")
    expected_source = _expected_prediction_source(config)
    if record.get("prediction_source") != expected_source:
        raise ValueError(
            f"Source stage {stage_name!r} prediction_source mismatch: "
            f"expected {expected_source!r}, got {record.get('prediction_source')!r}"
        )
    active_columns = set(record.get("output_columns") or [])
    value_column = required_columns[0]
    source_required = {value_column}
    if len(required_columns) > 1 and required_columns[1] == VALUE_SUBTASK_ID_PRED_SMOOTH:
        source_required.add(required_columns[1])
    missing = sorted(source_required - active_columns)
    if missing:
        opposite = {
            VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD,
            VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD,
            VALUE_SUBTASK_ID_GT,
            VALUE_SUBTASK_ID_PRED_SMOOTH,
        } & active_columns
        detail = f"; available paired-path columns={sorted(opposite)}" if opposite else ""
        raise ValueError(
            f"Source stage {stage_name!r} is missing required paired columns {missing}{detail}"
        )
    if config.value_source == "mock_pred" and record.get("synthetic") is not True:
        raise ValueError("mock_pred source stage must be marked synthetic=true")
    if config.value_source == "model_pred" and record.get("synthetic") is True:
        raise ValueError("model_pred source stage cannot be synthetic")
    dependencies = [stage_name]
    if len(required_columns) > 1 and required_columns[1] == VALUE_SUBTASK_ID_GT:
        assert_stage_dependencies_current(root, TARGET_STAGE)
        target_record = (metadata.get("stages") or {}).get(TARGET_STAGE) or {}
        if VALUE_SUBTASK_ID_GT not in set(target_record.get("output_columns") or []):
            raise ValueError("GT-conditioned path requires active value_subtask_id_gt target")
        if TARGET_STAGE not in dependencies:
            dependencies.append(TARGET_STAGE)
    return dependencies


def _require_columns(episode_path: Path, names: list[str], available: set[str]) -> None:
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Missing required extras columns in {episode_path}: {missing}")


def _as_float_array(table: pa.Table, column_name: str) -> np.ndarray:
    return np.asarray(table.column(column_name).to_pylist(), dtype=np.float32)


def _as_int_array(table: pa.Table, column_name: str) -> np.ndarray:
    try:
        return np.asarray(table.column(column_name).to_pylist(), dtype=np.int32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Subtask boundary column {column_name!r} must contain integers") from exc


def _valid_horizon(frame_index: int, episode_len: int, chunk_size: int) -> int:
    return min(chunk_size, max(episode_len - 1 - frame_index, 0))


def _compute_global_advantage(
    values: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], Counter[str]]:
    length = int(values.shape[0])
    advantage = np.zeros(length, dtype=np.float32)
    valid_horizon = np.zeros(length, dtype=np.int32)
    is_valid = np.zeros(length, dtype=bool)
    start_value = np.full(length, np.nan, dtype=np.float32)
    end_value = np.full(length, np.nan, dtype=np.float32)
    reasons: Counter[str] = Counter()

    for frame_index in range(length):
        horizon = _valid_horizon(frame_index, length, chunk_size)
        end_index = frame_index + horizon
        valid_horizon[frame_index] = horizon
        if horizon <= 0:
            reasons["zero_horizon"] += 1
            continue
        window = values[frame_index : end_index + 1]
        if not np.isfinite(window).all():
            reasons["nonfinite_value"] += 1
            continue
        start_value[frame_index] = values[frame_index]
        end_value[frame_index] = values[end_index]
        is_valid[frame_index] = True
        advantage[frame_index] = values[frame_index] - values[end_index] - horizon

    return (
        {
            ADVANTAGE_GLOBAL_CHUNK: advantage,
            ADVANTAGE_GLOBAL_VALID_HORIZON: valid_horizon,
            ADVANTAGE_GLOBAL_IS_VALID: is_valid,
            ADVANTAGE_GLOBAL_START_VALUE: start_value,
            ADVANTAGE_GLOBAL_END_VALUE: end_value,
        },
        reasons,
    )


def compute_global_advantage_columns(
    values: np.ndarray,
    *,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    return _compute_global_advantage(values, chunk_size=chunk_size)[0]


def _compute_subtask_advantage(
    values: np.ndarray,
    subtask_ids: np.ndarray,
    *,
    chunk_size: int,
    boundary_transition_value: float,
    num_subtasks: int | None = None,
) -> tuple[dict[str, np.ndarray], Counter[str]]:
    if values.shape != subtask_ids.shape:
        raise ValueError(
            f"values and subtask_ids must have the same shape, got {values.shape} and "
            f"{subtask_ids.shape}"
        )
    length = int(values.shape[0])
    advantage = np.zeros(length, dtype=np.float32)
    valid_horizon = np.zeros(length, dtype=np.int32)
    is_valid = np.zeros(length, dtype=bool)
    start_value = np.full(length, np.nan, dtype=np.float32)
    end_value = np.full(length, np.nan, dtype=np.float32)
    num_crossings = np.zeros(length, dtype=np.int32)
    within_horizon = np.zeros(length, dtype=np.int32)
    boundary_progress = np.zeros(length, dtype=np.float32)
    reasons: Counter[str] = Counter()

    for frame_index in range(length):
        horizon = _valid_horizon(frame_index, length, chunk_size)
        valid_horizon[frame_index] = horizon
        if horizon <= 0:
            reasons["zero_horizon"] += 1
            continue
        end_index = frame_index + horizon
        ids = subtask_ids[frame_index : end_index + 1]
        if (ids < 0).any() or (num_subtasks is not None and (ids >= num_subtasks).any()):
            reasons["unknown_subtask_id"] += 1
            continue
        value_window = values[frame_index : end_index + 1]
        if not np.isfinite(value_window).all():
            reasons["nonfinite_value"] += 1
            continue

        crossings = int(np.count_nonzero(ids[:-1] != ids[1:]))
        within = horizon - crossings
        progress = 0.0
        start_sum = 0.0
        end_sum = 0.0
        pos = frame_index
        while pos <= end_index:
            segment_end = pos
            while (
                segment_end < end_index
                and subtask_ids[segment_end + 1] == subtask_ids[pos]
            ):
                segment_end += 1
            if segment_end > pos:
                segment_start_value = float(values[pos])
                segment_end_value = float(values[segment_end])
                progress += segment_start_value - segment_end_value
                start_sum += segment_start_value
                end_sum += segment_end_value
            pos = segment_end + 1

        boundary = float(boundary_transition_value * crossings)
        num_crossings[frame_index] = crossings
        within_horizon[frame_index] = within
        boundary_progress[frame_index] = boundary
        start_value[frame_index] = start_sum
        end_value[frame_index] = end_sum
        advantage[frame_index] = progress + boundary - horizon
        is_valid[frame_index] = True

    return (
        {
            ADVANTAGE_SUBTASK_CHUNK: advantage,
            ADVANTAGE_SUBTASK_VALID_HORIZON: valid_horizon,
            ADVANTAGE_SUBTASK_IS_VALID: is_valid,
            ADVANTAGE_SUBTASK_START_VALUE: start_value,
            ADVANTAGE_SUBTASK_END_VALUE: end_value,
            ADVANTAGE_SUBTASK_NUM_CROSSINGS: num_crossings,
            ADVANTAGE_SUBTASK_WITHIN_SUBTASK_HORIZON: within_horizon,
            ADVANTAGE_SUBTASK_BOUNDARY_PROGRESS: boundary_progress,
        },
        reasons,
    )


def compute_subtask_advantage_columns(
    values: np.ndarray,
    subtask_ids: np.ndarray,
    *,
    chunk_size: int,
    boundary_transition_value: float = 1.0,
    num_subtasks: int | None = None,
) -> dict[str, np.ndarray]:
    return _compute_subtask_advantage(
        values,
        subtask_ids,
        chunk_size=chunk_size,
        boundary_transition_value=boundary_transition_value,
        num_subtasks=num_subtasks,
    )[0]


def _validate_subtask_sequence(
    subtask_ids: np.ndarray,
    canonical_order: list[str],
    *,
    episode_index: int,
) -> None:
    num_subtasks = len(canonical_order)
    positive_extra = sorted({int(value) for value in subtask_ids if value >= num_subtasks})
    if positive_extra:
        raise ValueError(
            f"Subtask sequence invalid in episode {episode_index}: extra ids={positive_extra}, "
            f"canonical_order={canonical_order}"
        )
    known = [int(value) for value in subtask_ids if value >= 0]
    compressed: list[int] = []
    for value in known:
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    expected = list(range(num_subtasks))
    if compressed != expected:
        raise ValueError(
            f"Subtask sequence invalid in episode {episode_index}: expected ids={expected}, "
            f"observed ids={compressed}, canonical_order={canonical_order}"
        )


def _columns_to_arrow(columns: dict[str, np.ndarray]) -> dict[str, pa.Array]:
    arrow_columns: dict[str, pa.Array] = {}
    for name, values in columns.items():
        if values.dtype == np.bool_:
            arrow_columns[name] = pa.array(values.tolist(), type=pa.bool_())
        elif np.issubdtype(values.dtype, np.integer):
            arrow_columns[name] = pa.array(values.tolist(), type=pa.int32())
        else:
            arrow_columns[name] = pa.array(values.astype(np.float32).tolist(), type=pa.float32())
    return arrow_columns


def _metadata_patch(config: AdvantageConfig, summary: dict) -> dict:
    try:
        metadata = read_value_function_metadata(config.root)
    except FileNotFoundError:
        metadata = {}
    advantage_meta = dict(metadata.get("advantage") or {})
    advantage_meta[config.value_mode] = {
        "value_source": config.value_source,
        "value_column": summary["value_column"],
        "chunk_size": config.chunk_size,
        "subtask_inference_path": summary["subtask_inference_path"],
        "boundary_column": summary["boundary_column"],
        "boundary_transition_value": summary["boundary_transition_value"],
        "advantage_formula_version": summary["advantage_formula_version"],
        "prediction_source": summary["prediction_source"],
        "synthetic": summary["synthetic"],
        "experiment_eligible": summary["experiment_eligible"],
        "columns_written": summary["columns_written"],
        "summary": {
            "valid_chunks": summary["valid_chunks"],
            "invalid_chunks": summary["invalid_chunks"],
            "total_chunks": summary["total_chunks"],
            "invalid_reason_counts": summary["invalid_reason_counts"],
        },
    }
    existing_columns = set(metadata.get("columns_written") or [])
    return {
        "advantage": advantage_meta,
        "columns_written": sorted(existing_columns | set(summary["columns_written"])),
    }


def compute_advantage(config: AdvantageConfig) -> dict:
    if config.value_mode not in ("global", "subtask"):
        raise ValueError(f"Unsupported value_mode: {config.value_mode}")
    if config.value_source not in ("gt", "mock_pred", "model_pred"):
        raise ValueError(f"Unsupported value_source: {config.value_source}")
    if config.chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {config.chunk_size}")
    boundary_transition_value = _resolved_boundary_transition_value(config)
    if not np.isfinite(boundary_transition_value):
        raise ValueError("boundary_transition_value must be finite")
    if config.value_mode == "global" and (
        boundary_transition_value != 1.0 or config.boundary_bonus is not None
    ):
        raise ValueError("boundary transition settings only apply to value_mode='subtask'")

    root = Path(config.root).expanduser().resolve()
    value_column, boundary_column = _resolve_source_columns(config)
    required_columns = [value_column] + ([boundary_column] if boundary_column else [])
    source_stages = _validate_source_provenance(root, config, required_columns)
    metadata = read_value_function_metadata(root)
    canonical_order = list(metadata.get("subtask_order") or [])
    if config.value_mode == "subtask" and not canonical_order:
        raise ValueError("Subtask advantage requires canonical subtask_order metadata")

    episode_columns: dict[int, dict[str, pa.Array]] = {}
    episode_summaries = []
    reason_totals: Counter[str] = Counter()
    valid_total = 0
    total = 0
    for episode in discover_episodes(root):
        extras = read_extras_table(episode.path)
        if extras is None:
            raise FileNotFoundError(f"Missing extras.parquet in {episode.path}")
        if extras.num_rows != episode.frame_count:
            raise ValueError(
                f"extras.parquet length ({extras.num_rows}) does not match frames.parquet "
                f"length ({episode.frame_count}) in {episode.path}"
            )
        _require_columns(episode.path, required_columns, set(extras.column_names))
        values = _as_float_array(extras, value_column)
        if config.value_mode == "global":
            columns, reasons = _compute_global_advantage(
                values, chunk_size=config.chunk_size
            )
            is_valid_name = ADVANTAGE_GLOBAL_IS_VALID
        else:
            assert boundary_column is not None
            subtask_ids = _as_int_array(extras, boundary_column)
            _validate_subtask_sequence(
                subtask_ids, canonical_order, episode_index=episode.index
            )
            columns, reasons = _compute_subtask_advantage(
                values,
                subtask_ids,
                chunk_size=config.chunk_size,
                boundary_transition_value=boundary_transition_value,
                num_subtasks=len(canonical_order),
            )
            is_valid_name = ADVANTAGE_SUBTASK_IS_VALID
        valid_count = int(columns[is_valid_name].sum())
        reason_totals.update(reasons)
        episode_summaries.append(
            {
                "index": episode.index,
                "length": episode.frame_count,
                "valid_chunks": valid_count,
                "invalid_chunks": episode.frame_count - valid_count,
                "invalid_reason_counts": dict(sorted(reasons.items())),
            }
        )
        valid_total += valid_count
        total += episode.frame_count
        episode_columns[episode.index] = _columns_to_arrow(columns)

    columns_written = sorted(
        {name for columns in episode_columns.values() for name in columns}
    )
    synthetic = config.value_source == "mock_pred"
    experiment_eligible = config.value_source == "model_pred"
    summary = {
        "root": str(root),
        "value_mode": config.value_mode,
        "value_source": config.value_source,
        "value_column": value_column,
        "chunk_size": config.chunk_size,
        "subtask_inference_path": (
            config.subtask_inference_path if config.value_mode == "subtask" else None
        ),
        "boundary_column": boundary_column,
        "boundary_transition_value": (
            boundary_transition_value if config.value_mode == "subtask" else None
        ),
        "advantage_formula_version": (
            SUBTASK_FORMULA_VERSION
            if config.value_mode == "subtask"
            else GLOBAL_FORMULA_VERSION
        ),
        "prediction_source": _expected_prediction_source(config),
        "synthetic": synthetic,
        "experiment_eligible": experiment_eligible,
        "dry_run": config.dry_run,
        "episodes": episode_summaries,
        "total_chunks": total,
        "valid_chunks": valid_total,
        "invalid_chunks": total - valid_total,
        "invalid_reason_counts": dict(sorted(reason_totals.items())),
        "columns_written": columns_written,
    }

    if not config.dry_run:
        input_fingerprint = fingerprint_raw_run_columns(root, required_columns)
        merge_raw_run_extras(root, episode_columns)
        output_fingerprint = fingerprint_raw_run_columns(root, columns_written)
        update_stage_metadata(
            root,
            f"{ADVANTAGE_STAGE_PREFIX}.{config.value_mode}",
            config=config,
            input_columns=required_columns,
            input_fingerprint=input_fingerprint,
            output_columns=columns_written,
            output_fingerprint=output_fingerprint,
            prediction_source=_expected_prediction_source(config),
            synthetic=synthetic,
            dependencies=source_stages,
            metadata_patch=_metadata_patch(config, summary),
        )
    return summary
