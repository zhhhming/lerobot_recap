"""Advantage ranking, label preview, override persistence, and export."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    ADVANTAGE_GLOBAL_IS_VALID,
    ADVANTAGE_GLOBAL_VALID_HORIZON,
    ADVANTAGE_LABEL_GLOBAL,
    ADVANTAGE_LABEL_SUBTASK,
    ADVANTAGE_LABELING_STAGE_PREFIX,
    ADVANTAGE_STAGE_PREFIX,
    ADVANTAGE_SUBTASK_CHUNK,
    ADVANTAGE_SUBTASK_IS_VALID,
    ADVANTAGE_SUBTASK_VALID_HORIZON,
)

ValueMode = Literal["global", "subtask"]
AdvantageLabel = Literal["positive", "negative", "ignore"]
SortOrder = Literal["desc", "asc"]
TiePolicy = Literal["exact_count", "include_all"]

VALID_LABELS = {"positive", "negative", "ignore"}
VALID_TIE_POLICIES = {"exact_count", "include_all"}


@dataclass(frozen=True)
class AdvantageLabelingConfig:
    root: Path | str
    value_mode: ValueMode = "global"
    top_percent: float = 0.8
    sort_order: SortOrder = "desc"
    tie_policy: TiePolicy = "exact_count"
    overrides: dict[str, str] | None = None
    allow_synthetic: bool = False
    dry_run: bool = False


def label_column(value_mode: ValueMode) -> str:
    return ADVANTAGE_LABEL_GLOBAL if value_mode == "global" else ADVANTAGE_LABEL_SUBTASK


def advantage_columns(value_mode: ValueMode) -> tuple[str, str, str]:
    if value_mode == "global":
        return (
            ADVANTAGE_GLOBAL_CHUNK,
            ADVANTAGE_GLOBAL_VALID_HORIZON,
            ADVANTAGE_GLOBAL_IS_VALID,
        )
    return (
        ADVANTAGE_SUBTASK_CHUNK,
        ADVANTAGE_SUBTASK_VALID_HORIZON,
        ADVANTAGE_SUBTASK_IS_VALID,
    )


def chunk_key(episode_index: int, frame_index: int) -> str:
    return f"ep_{episode_index:06d}:frame_{frame_index:06d}"


def parse_chunk_key(key: str) -> tuple[int, int]:
    try:
        ep_part, frame_part = key.split(":")
        if not ep_part.startswith("ep_") or not frame_part.startswith("frame_"):
            raise ValueError
        return int(ep_part.removeprefix("ep_")), int(frame_part.removeprefix("frame_"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid chunk key {key!r}; expected ep_000000:frame_000000") from exc


def _normalize_top_percent(top_percent: float) -> float:
    top_percent = float(top_percent)
    if top_percent > 1.0:
        top_percent /= 100.0
    if not math.isfinite(top_percent) or not 0.0 <= top_percent <= 1.0:
        raise ValueError(f"top_percent must be in [0, 1] or [0, 100], got {top_percent}")
    return top_percent


def _advantage_metadata(root: str | Path, value_mode: ValueMode) -> tuple[dict, dict]:
    metadata = read_value_function_metadata(root)
    stage_name = f"{ADVANTAGE_STAGE_PREFIX}.{value_mode}"
    assert_stage_dependencies_current(root, stage_name)
    stage = (metadata.get("stages") or {}).get(stage_name)
    detail = (metadata.get("advantage") or {}).get(value_mode)
    if not isinstance(stage, dict) or not isinstance(detail, dict):
        raise ValueError(f"Missing current advantage metadata for mode={value_mode!r}")
    required = set(advantage_columns(value_mode))
    missing = sorted(required - set(stage.get("output_columns") or []))
    if missing:
        raise ValueError(f"Advantage stage {stage_name!r} is missing active columns: {missing}")
    return stage, detail


def advantage_export_eligibility(root: str | Path, value_mode: ValueMode) -> dict:
    stage, detail = _advantage_metadata(root, value_mode)
    prediction_source = detail.get("prediction_source") or stage.get("prediction_source")
    experiment_eligible = bool(detail.get("experiment_eligible", False))
    return {
        "prediction_source": prediction_source,
        "synthetic": bool(detail.get("synthetic", stage.get("synthetic", False))),
        "experiment_eligible": experiment_eligible,
        "warning": (
            None
            if experiment_eligible
            else "SYNTHETIC / GT SANITY ONLY — NOT FOR EXPERIMENT"
        ),
    }


def load_saved_overrides(root: str | Path, value_mode: ValueMode) -> dict[str, str]:
    try:
        metadata = read_value_function_metadata(root)
    except FileNotFoundError:
        return {}
    mode_meta = (metadata.get("advantage_labeling") or {}).get(value_mode) or {}
    overrides = mode_meta.get("overrides") or {}
    return {str(key): str(value) for key, value in overrides.items()}


def load_advantage_chunks(root: str | Path, value_mode: ValueMode = "global") -> list[dict]:
    root = Path(root).expanduser().resolve()
    _advantage_metadata(root, value_mode)
    advantage_col, valid_horizon_col, is_valid_col = advantage_columns(value_mode)
    out_label = label_column(value_mode)
    chunks: list[dict] = []
    for episode in discover_episodes(root):
        extras = read_extras_table(episode.path)
        if extras is None:
            raise FileNotFoundError(f"Missing extras.parquet in {episode.path}")
        required = [advantage_col, valid_horizon_col, is_valid_col]
        missing = [name for name in required if name not in extras.column_names]
        if missing:
            raise ValueError(f"Missing required extras columns in {episode.path}: {missing}")
        advantages = extras.column(advantage_col).to_pylist()
        horizons = extras.column(valid_horizon_col).to_pylist()
        validity = extras.column(is_valid_col).to_pylist()
        stored = (
            extras.column(out_label).to_pylist()
            if out_label in extras.column_names
            else [""] * episode.frame_count
        )
        for frame_index in range(episode.frame_count):
            chunks.append(
                {
                    "key": chunk_key(episode.index, frame_index),
                    "episode_index": episode.index,
                    "episode_name": f"ep_{episode.index:06d}",
                    "frame_index": frame_index,
                    "advantage": float(advantages[frame_index]),
                    "valid_horizon": int(horizons[frame_index]),
                    "is_valid": bool(validity[frame_index]),
                    "stored_label": stored[frame_index] or "",
                    "label": stored[frame_index] or "",
                }
            )
    return chunks


def sorted_advantage_chunks(chunks: list[dict], sort_order: SortOrder = "desc") -> list[dict]:
    if sort_order not in ("desc", "asc"):
        raise ValueError(f"sort_order must be 'desc' or 'asc', got {sort_order!r}")
    direction = -1.0 if sort_order == "desc" else 1.0
    return sorted(
        chunks,
        key=lambda chunk: (
            not chunk["is_valid"],
            direction * chunk["advantage"],
            chunk["episode_index"],
            chunk["frame_index"],
        ),
    )


def _normalize_overrides(
    overrides: dict[str, str] | None,
    known_keys: set[str],
) -> dict[str, AdvantageLabel]:
    normalized: dict[str, AdvantageLabel] = {}
    unknown: list[str] = []
    for key, value in (overrides or {}).items():
        parse_chunk_key(key)
        if key not in known_keys:
            unknown.append(key)
            continue
        label = str(value).strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid override label for {key}: {value!r}")
        normalized[key] = label  # type: ignore[assignment]
    if unknown:
        raise ValueError(f"Override keys do not belong to the current run/mode: {sorted(unknown)}")
    return normalized


def preview_advantage_labels(
    chunks: list[dict],
    *,
    top_percent: float = 0.8,
    sort_order: SortOrder = "desc",
    tie_policy: TiePolicy = "exact_count",
    overrides: dict[str, str] | None = None,
) -> dict:
    top_percent = _normalize_top_percent(top_percent)
    if tie_policy not in VALID_TIE_POLICIES:
        raise ValueError(f"Unsupported tie_policy: {tie_policy!r}")
    known_keys = {chunk["key"] for chunk in chunks}
    if len(known_keys) != len(chunks):
        raise ValueError("Duplicate chunk keys in advantage input")
    normalized_overrides = _normalize_overrides(overrides, known_keys)

    ranked_valid = sorted(
        (chunk for chunk in chunks if chunk["is_valid"]),
        key=lambda chunk: (-chunk["advantage"], chunk["episode_index"], chunk["frame_index"]),
    )
    target_count = int(len(ranked_valid) * top_percent)
    if top_percent > 0.0 and ranked_valid and target_count == 0:
        target_count = 1
    threshold_value = (
        float(ranked_valid[target_count - 1]["advantage"]) if target_count > 0 else None
    )
    positive_keys = {chunk["key"] for chunk in ranked_valid[:target_count]}
    tie_count = 0
    selected_at_tie = 0
    if threshold_value is not None:
        tied = [chunk for chunk in ranked_valid if chunk["advantage"] == threshold_value]
        tie_count = len(tied)
        if tie_policy == "include_all":
            positive_keys.update(chunk["key"] for chunk in tied)
        selected_at_tie = sum(chunk["key"] in positive_keys for chunk in tied)

    threshold_labels: dict[str, AdvantageLabel] = {}
    labels: dict[str, AdvantageLabel] = {}
    sources: dict[str, str] = {}
    for chunk in chunks:
        key = chunk["key"]
        if not chunk["is_valid"]:
            threshold_labels[key] = "ignore"
            sources[key] = "invalid"
        elif key in positive_keys:
            threshold_labels[key] = "positive"
            sources[key] = "threshold"
        else:
            threshold_labels[key] = "negative"
            sources[key] = "threshold"
        labels[key] = threshold_labels[key]
    for key, label in normalized_overrides.items():
        labels[key] = label
        sources[key] = "manual_override"

    counts = {label: 0 for label in sorted(VALID_LABELS)}
    for label in labels.values():
        counts[label] += 1
    display_chunks = sorted_advantage_chunks(chunks, sort_order=sort_order)
    enriched = [
        {
            **chunk,
            "threshold_label": threshold_labels[chunk["key"]],
            "manual_override_label": normalized_overrides.get(chunk["key"], ""),
            "preview_label": labels[chunk["key"]],
            "label_source": sources[chunk["key"]],
        }
        for chunk in display_chunks
    ]
    threshold = {
        "positive_direction": "high",
        "tie_policy": tie_policy,
        "threshold_value": threshold_value,
        "tie_count": tie_count,
        "selected_at_tie_count": selected_at_tie,
        "target_positive_count": target_count,
        "actual_threshold_positive_count": len(positive_keys),
        "valid_count": len(ranked_valid),
        "invalid_count": len(chunks) - len(ranked_valid),
    }
    return {
        "labels": labels,
        "counts": counts,
        "sorted_chunks": enriched,
        "top_percent": top_percent,
        "sort_order": sort_order,
        "tie_policy": tie_policy,
        "threshold": threshold,
        "overrides": normalized_overrides,
    }


def export_change_summary(chunks: list[dict], labels: dict[str, str]) -> dict:
    transitions: Counter[str] = Counter()
    unchanged = changed = unset = 0
    for chunk in chunks:
        old = chunk.get("stored_label") or ""
        new = labels[chunk["key"]]
        if not old:
            unset += 1
        elif old == new:
            unchanged += 1
        else:
            changed += 1
        transitions[f"{old or '<unset>'}->{new}"] += 1
    return {
        "unchanged": unchanged,
        "changed": changed,
        "unset_to_labeled": unset,
        "total": len(chunks),
        "transitions": dict(sorted(transitions.items())),
    }


def export_advantage_labels(config: AdvantageLabelingConfig) -> dict:
    root = Path(config.root).expanduser().resolve()
    eligibility = advantage_export_eligibility(root, config.value_mode)
    if not eligibility["experiment_eligible"] and not config.allow_synthetic:
        raise ValueError(
            f"Formal label export requires model_pred advantage; got "
            f"{eligibility['prediction_source']!r}. Use allow_synthetic only for tests."
        )
    chunks = load_advantage_chunks(root, value_mode=config.value_mode)
    overrides = (
        load_saved_overrides(root, config.value_mode)
        if config.overrides is None
        else config.overrides
    )
    preview = preview_advantage_labels(
        chunks,
        top_percent=config.top_percent,
        sort_order=config.sort_order,
        tie_policy=config.tie_policy,
        overrides=overrides,
    )
    labels = preview["labels"]
    out_column = label_column(config.value_mode)
    episode_columns: dict[int, dict[str, pa.Array]] = {}
    episode_summaries = []
    for episode in discover_episodes(root):
        values = [labels[chunk_key(episode.index, i)] for i in range(episode.frame_count)]
        episode_columns[episode.index] = {out_column: pa.array(values, type=pa.string())}
        episode_summaries.append(
            {
                "index": episode.index,
                "length": episode.frame_count,
                "counts": {label: values.count(label) for label in sorted(VALID_LABELS)},
            }
        )
    change_summary = export_change_summary(chunks, labels)
    summary = {
        "root": str(root),
        "value_mode": config.value_mode,
        "top_percent": preview["top_percent"],
        "sort_order": config.sort_order,
        "positive_direction": "high",
        "tie_policy": config.tie_policy,
        "threshold": preview["threshold"],
        "dry_run": config.dry_run,
        "counts": preview["counts"],
        "overrides": preview["overrides"],
        "override_count": len(preview["overrides"]),
        "change_summary": change_summary,
        "eligibility": eligibility,
        "episodes": episode_summaries,
        "columns_written": [out_column],
    }
    if not config.dry_run:
        advantage_stage = f"{ADVANTAGE_STAGE_PREFIX}.{config.value_mode}"
        input_columns = list(advantage_columns(config.value_mode))
        input_fingerprint = fingerprint_raw_run_columns(root, input_columns)
        merge_raw_run_extras(root, episode_columns)
        output_fingerprint = fingerprint_raw_run_columns(root, [out_column])
        metadata = read_value_function_metadata(root)
        labeling_meta = dict(metadata.get("advantage_labeling") or {})
        labeling_meta[config.value_mode] = {
            key: summary[key]
            for key in (
                "top_percent",
                "sort_order",
                "positive_direction",
                "tie_policy",
                "threshold",
                "counts",
                "overrides",
                "override_count",
                "change_summary",
                "eligibility",
                "columns_written",
            )
        }
        existing_columns = set(metadata.get("columns_written") or [])
        update_stage_metadata(
            root,
            f"{ADVANTAGE_LABELING_STAGE_PREFIX}.{config.value_mode}",
            config=config,
            input_columns=input_columns,
            input_fingerprint=input_fingerprint,
            output_columns=[out_column],
            output_fingerprint=output_fingerprint,
            prediction_source=eligibility["prediction_source"],
            synthetic=bool(eligibility["synthetic"]),
            dependencies=[advantage_stage],
            metadata_patch={
                "advantage_labeling": labeling_meta,
                "columns_written": sorted(existing_columns | {out_column}),
            },
        )
    return summary
