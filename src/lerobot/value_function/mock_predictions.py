"""Reproducible synthetic value predictions for pipeline smoke tests only."""

from __future__ import annotations

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
    MOCK_PREDICTIONS_STAGE,
    PREDICTION_SOURCE_MOCK,
    TARGET_STAGE,
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_REMAINING_FRAMES_GT,
    VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED,
)

ValueMode = Literal["global", "subtask", "both"]
SYNTHETIC_GENERATOR = "synthetic_gt_gaussian_noise"


@dataclass(frozen=True)
class MockPredictionConfig:
    root: Path | str
    mode: ValueMode = "both"
    seed: int = 42
    noise_std_frames: float = 3.0
    temporal_smoothing_sigma_frames: float = 0.0
    dry_run: bool = False


def _needs_global(mode: ValueMode) -> bool:
    return mode in ("global", "both")


def _needs_subtask(mode: ValueMode) -> bool:
    return mode in ("subtask", "both")


def _gaussian_smooth_span(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or values.size <= 1:
        return values.copy()
    radius = max(1, int(np.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    kernel /= kernel.sum()

    finite = np.isfinite(values)
    numeric = np.where(finite, values, 0.0).astype(np.float64)
    padded_numeric = np.pad(numeric, (radius, radius), mode="edge")
    padded_finite = np.pad(finite.astype(np.float64), (radius, radius), mode="edge")
    numerator = np.convolve(padded_numeric, kernel, mode="valid")
    denominator = np.convolve(padded_finite, kernel, mode="valid")
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=smoothed, where=denominator > 0)
    smoothed[~finite] = np.nan
    return smoothed.astype(np.float32)


def _contiguous_subtask_spans(subtask_ids: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while start < subtask_ids.size:
        subtask_id = int(subtask_ids[start])
        end = start + 1
        while end < subtask_ids.size and int(subtask_ids[end]) == subtask_id:
            end += 1
        if subtask_id >= 0:
            spans.append((start, end))
        start = end
    return spans


def _add_noise(
    values: np.ndarray,
    *,
    seed: int,
    episode_index: int,
    stream: int,
    noise_std_frames: float,
) -> np.ndarray:
    result = values.astype(np.float32).copy()
    finite = np.isfinite(result)
    if noise_std_frames == 0 or not finite.any():
        return result
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode_index, stream]))
    noise = rng.normal(0.0, noise_std_frames, size=int(finite.sum())).astype(np.float32)
    result[finite] += noise
    return result


def _smooth_global(values: np.ndarray, sigma: float) -> np.ndarray:
    return _gaussian_smooth_span(values, sigma)


def _smooth_subtask(values: np.ndarray, subtask_ids: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.copy()
    result = values.copy()
    for start, end in _contiguous_subtask_spans(subtask_ids):
        result[start:end] = _gaussian_smooth_span(values[start:end], sigma)
    return result


def _require_active_target_columns(metadata: dict, required: list[str]) -> None:
    target_stage = (metadata.get("stages") or {}).get(TARGET_STAGE)
    if not isinstance(target_stage, dict):
        raise ValueError(
            "Mock prediction requires target stage metadata; run target preparation first"
        )
    active = set(target_stage.get("output_columns") or [])
    missing = sorted(set(required) - active)
    if missing:
        raise ValueError(
            f"Mock prediction requires active target columns {missing}; rerun target preparation "
            "with a compatible mode"
        )


def generate_mock_predictions(config: MockPredictionConfig) -> dict:
    if config.mode not in ("global", "subtask", "both"):
        raise ValueError(f"Unsupported mock prediction mode: {config.mode}")
    if config.seed < 0:
        raise ValueError("seed must be >= 0")
    if config.noise_std_frames < 0:
        raise ValueError("noise_std_frames must be >= 0")
    if config.temporal_smoothing_sigma_frames < 0:
        raise ValueError("temporal_smoothing_sigma_frames must be >= 0")

    root = Path(config.root).expanduser().resolve()
    assert_stage_dependencies_current(root, TARGET_STAGE)
    metadata = read_value_function_metadata(root)
    input_columns: list[str] = []
    if _needs_global(config.mode):
        input_columns.append(VALUE_GLOBAL_REMAINING_FRAMES_GT)
    if _needs_subtask(config.mode):
        input_columns.extend([VALUE_SUBTASK_REMAINING_FRAMES_GT, VALUE_SUBTASK_ID_GT])
    _require_active_target_columns(metadata, input_columns)
    input_fingerprint = fingerprint_raw_run_columns(root, input_columns)

    episode_columns: dict[int, dict[str, pa.Array]] = {}
    for episode in discover_episodes(root):
        extras = read_extras_table(episode.path)
        if extras is None:
            raise FileNotFoundError(f"Missing extras.parquet in {episode.path}")
        columns: dict[str, pa.Array] = {}
        if _needs_global(config.mode):
            gt = np.asarray(
                extras.column(VALUE_GLOBAL_REMAINING_FRAMES_GT).to_pylist(), dtype=np.float32
            )
            noisy = _add_noise(
                gt,
                seed=config.seed,
                episode_index=episode.index,
                stream=0,
                noise_std_frames=config.noise_std_frames,
            )
            mock = _smooth_global(noisy, config.temporal_smoothing_sigma_frames)
            columns[VALUE_GLOBAL_REMAINING_FRAMES_MOCK_PRED] = pa.array(
                mock.tolist(), type=pa.float32()
            )
        if _needs_subtask(config.mode):
            gt = np.asarray(
                extras.column(VALUE_SUBTASK_REMAINING_FRAMES_GT).to_pylist(), dtype=np.float32
            )
            subtask_ids = np.asarray(
                extras.column(VALUE_SUBTASK_ID_GT).to_pylist(), dtype=np.int32
            )
            noisy = _add_noise(
                gt,
                seed=config.seed,
                episode_index=episode.index,
                stream=1,
                noise_std_frames=config.noise_std_frames,
            )
            mock = _smooth_subtask(
                noisy, subtask_ids, config.temporal_smoothing_sigma_frames
            )
            columns[VALUE_SUBTASK_REMAINING_FRAMES_MOCK_PRED] = pa.array(
                mock.tolist(), type=pa.float32()
            )
        episode_columns[episode.index] = columns

    columns_written = sorted(
        {name for columns in episode_columns.values() for name in columns}
    )
    summary = {
        "root": str(root),
        "mode": config.mode,
        "dry_run": config.dry_run,
        "prediction_source": PREDICTION_SOURCE_MOCK,
        "generator": SYNTHETIC_GENERATOR,
        "synthetic": True,
        "experiment_eligible": False,
        "warning": "SYNTHETIC / NOT FOR EXPERIMENT",
        "seed": config.seed,
        "noise_std_frames": config.noise_std_frames,
        "temporal_smoothing_sigma_frames": config.temporal_smoothing_sigma_frames,
        "source_gt_columns": input_columns,
        "source_gt_fingerprint": input_fingerprint,
        "columns_written": columns_written,
        "episodes": len(episode_columns),
    }
    if not config.dry_run:
        merge_raw_run_extras(root, episode_columns)
        output_fingerprint = fingerprint_raw_run_columns(root, columns_written)
        update_stage_metadata(
            root,
            MOCK_PREDICTIONS_STAGE,
            config=config,
            input_columns=input_columns,
            input_fingerprint=input_fingerprint,
            output_columns=columns_written,
            output_fingerprint=output_fingerprint,
            prediction_source=PREDICTION_SOURCE_MOCK,
            synthetic=True,
            dependencies=[TARGET_STAGE],
            metadata_patch={"mock_predictions": summary},
        )
    return summary
