#!/usr/bin/env python

"""Validate the production T1 timing lookup against a local labeled dataset."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.subtask_timing import SubtaskTimingDataset, normalize_subtask_name


def _parse_expected_stat(value: str) -> tuple[str, float, float]:
    try:
        name, maximum, cap = value.rsplit(":", 2)
        return normalize_subtask_name(name), float(maximum), float(cap)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected stat must use NAME:MAX_ELAPSED:DEPLOYMENT_CAP"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-subtasks", type=int, required=True)
    parser.add_argument("--max-lookup-mib", type=float, default=8.0)
    parser.add_argument(
        "--expected-stat",
        action="append",
        type=_parse_expected_stat,
        default=[],
        metavar="NAME:MAX:CAP",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")
    if not math.isfinite(args.max_lookup_mib) or args.max_lookup_mib <= 0:
        raise ValueError("max lookup MiB must be finite and positive")

    started = time.perf_counter()
    dataset = LeRobotDataset(args.repo_id, root=root)
    wrapped = SubtaskTimingDataset(dataset)
    construction_seconds = time.perf_counter() - started
    contract = wrapped.sequence_contract
    stats_by_name = {item.normalized_name: item for item in contract.ordered_subtasks}
    lookup_mib = wrapped.lookup_nbytes / (1024 * 1024)

    if wrapped.num_episodes != args.expected_episodes:
        raise ValueError(
            f"Expected {args.expected_episodes} episodes, got {wrapped.num_episodes}"
        )
    if len(wrapped) != args.expected_frames:
        raise ValueError(f"Expected {args.expected_frames} frames, got {len(wrapped)}")
    if len(contract.ordered_subtasks) != args.expected_subtasks:
        raise ValueError(
            f"Expected {args.expected_subtasks} subtasks, got {len(contract.ordered_subtasks)}"
        )
    if lookup_mib > args.max_lookup_mib:
        raise ValueError(
            f"Timing lookup is too large: {lookup_mib:.3f} MiB > {args.max_lookup_mib:.3f} MiB"
        )

    for normalized, expected_maximum, expected_cap in args.expected_stat:
        if normalized not in stats_by_name:
            raise ValueError(f"Expected subtask is absent from sequence contract: {normalized!r}")
        actual = stats_by_name[normalized]
        if not math.isclose(actual.max_elapsed_seconds, expected_maximum, abs_tol=1e-5):
            raise ValueError(
                f"Unexpected max elapsed for {normalized!r}: "
                f"{actual.max_elapsed_seconds} != {expected_maximum}"
            )
        if not math.isclose(actual.deployment_cap_seconds, expected_cap, abs_tol=1e-5):
            raise ValueError(
                f"Unexpected deployment cap for {normalized!r}: "
                f"{actual.deployment_cap_seconds} != {expected_cap}"
            )

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "root": str(root),
                "fps": contract.fps,
                "episodes": wrapped.num_episodes,
                "frames": len(wrapped),
                "subtasks": len(contract.ordered_subtasks),
                "construction_seconds": round(construction_seconds, 6),
                "lookup_bytes": wrapped.lookup_nbytes,
                "lookup_mib": round(lookup_mib, 6),
                "ordered_subtasks": [
                    {
                        "canonical_name": item.canonical_name,
                        "normalized_name": item.normalized_name,
                        "max_elapsed_seconds": round(item.max_elapsed_seconds, 6),
                        "deployment_cap_seconds": round(item.deployment_cap_seconds, 6),
                    }
                    for item in contract.ordered_subtasks
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
