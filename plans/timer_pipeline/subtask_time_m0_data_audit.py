#!/usr/bin/env python

"""Audit a labeled LeRobotDataset using only lightweight parquet columns."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass(frozen=True)
class SegmentSummary:
    canonical_name: str
    normalized_name: str
    segment_count: int
    min_end_elapsed_seconds: float
    max_end_elapsed_seconds: float
    deployment_cap_seconds: float


def normalize_subtask_name(value: str) -> str:
    normalized = " ".join(value.strip().split()).casefold()
    if normalized.endswith((".", "。", "．", "｡")):
        normalized = normalized[:-1].rstrip()
    return normalized


def _load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Dataset metadata does not exist: {info_path}")
    return json.loads(info_path.read_text())


def audit_dataset(
    root: Path,
    repo_id: str,
    deployment_margin_seconds: float,
) -> dict[str, Any]:
    info = _load_info(root)
    fps = float(info.get("fps", 0.0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Dataset fps must be finite and positive, got {fps!r}")
    if not math.isfinite(deployment_margin_seconds) or deployment_margin_seconds < 0:
        raise ValueError("deployment margin must be finite and non-negative")

    required = {"index", "episode_index", "frame_index", "subtask"}
    features = info.get("features", {})
    missing = sorted(required - features.keys())
    if missing:
        raise ValueError(f"Dataset is missing required features: {missing}")

    parquet_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found below {root / 'data'}")
    table = pq.read_table(
        parquet_files,
        columns=["index", "episode_index", "frame_index", "subtask"],
    )
    if any(table[column].null_count for column in table.column_names):
        nulls = {column: table[column].null_count for column in table.column_names}
        raise ValueError(f"Timing columns contain nulls: {nulls}")

    indices = table["index"].to_pylist()
    episodes = table["episode_index"].to_pylist()
    frames = table["frame_index"].to_pylist()
    labels = table["subtask"].to_pylist()
    if not labels:
        raise ValueError("Dataset contains no frames")

    expected_index = indices[0]
    if expected_index != 0:
        raise ValueError(f"Global index must start at 0, got {expected_index}")

    sequences: dict[int, list[str]] = {}
    normalized_sequences: dict[int, list[str]] = {}
    durations: dict[str, list[float]] = {}
    canonical_names: dict[str, str] = {}
    current_episode = None
    segment_start = 0

    for row, (index, episode, frame, label) in enumerate(
        zip(indices, episodes, frames, labels, strict=True)
    ):
        if index != row:
            raise ValueError(f"Global index mismatch at row {row}: expected {row}, got {index}")
        episode_changed = row == 0 or episode != episodes[row - 1]
        expected_frame = 0 if episode_changed else frames[row - 1] + 1
        if frame != expected_frame:
            raise ValueError(
                f"Frame index mismatch in episode {episode} at row {row}: "
                f"expected {expected_frame}, got {frame}"
            )
        if episode_changed and current_episode is not None and episode <= current_episode:
            raise ValueError(f"Episode indices must increase, got {current_episode} then {episode}")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Subtask label must be a non-empty string at row {row}: {label!r}")
        if episode_changed:
            current_episode = episode
            sequences[episode] = []
            normalized_sequences[episode] = []

        next_is_boundary = (
            row + 1 == len(labels)
            or episodes[row + 1] != episode
            or labels[row + 1] != label
        )
        if next_is_boundary:
            canonical = " ".join(label.strip().split())
            normalized = normalize_subtask_name(canonical)
            if not normalized:
                raise ValueError(f"Subtask normalizes to an empty value: {canonical!r}")
            prior = canonical_names.setdefault(normalized, canonical)
            if normalize_subtask_name(prior) != normalized:
                raise AssertionError("Internal normalization bookkeeping mismatch")
            sequences[episode].append(canonical)
            normalized_sequences[episode].append(normalized)
            durations.setdefault(normalized, []).append((frame - frames[segment_start]) / fps)
            segment_start = row + 1

    first_episode = min(sequences)
    expected_sequence = normalized_sequences[first_episode]
    if len(set(expected_sequence)) != len(expected_sequence):
        raise ValueError(
            f"Episode {first_episode} repeats a normalized subtask: {sequences[first_episode]}"
        )
    for episode in sorted(sequences):
        actual = normalized_sequences[episode]
        if actual != expected_sequence:
            mismatch = next(
                (index for index, pair in enumerate(zip(expected_sequence, actual)) if pair[0] != pair[1]),
                min(len(expected_sequence), len(actual)),
            )
            raise ValueError(
                f"Dataset {repo_id} episode {episode} sequence differs at position {mismatch}; "
                f"expected={sequences[first_episode]!r}, actual={sequences[episode]!r}"
            )

    summaries = []
    for normalized in expected_sequence:
        values = durations[normalized]
        summaries.append(
            SegmentSummary(
                canonical_name=canonical_names[normalized],
                normalized_name=normalized,
                segment_count=len(values),
                min_end_elapsed_seconds=round(min(values), 6),
                max_end_elapsed_seconds=round(max(values), 6),
                deployment_cap_seconds=round(max(values) + deployment_margin_seconds, 6),
            )
        )

    return {
        "repo_id": repo_id,
        "root": str(root),
        "fps": fps,
        "episodes": len(sequences),
        "frames": len(labels),
        "ordered_subtask_count": len(expected_sequence),
        "sequence_consistent": True,
        "global_index_contiguous": True,
        "episode_frame_index_contiguous": True,
        "deployment_margin_seconds": deployment_margin_seconds,
        "ordered_subtasks": [asdict(summary) for summary in summaries],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--deployment-margin-seconds", type=float, default=5.0)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-subtasks", type=int)
    return parser.parse_args()


def _require_equal(name: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"Expected {name}={expected}, got {actual}")


def main() -> None:
    args = parse_args()
    result = audit_dataset(
        args.root.resolve(),
        args.repo_id,
        args.deployment_margin_seconds,
    )
    _require_equal("episodes", result["episodes"], args.expected_episodes)
    _require_equal("frames", result["frames"], args.expected_frames)
    _require_equal("subtasks", result["ordered_subtask_count"], args.expected_subtasks)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
