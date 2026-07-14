#!/usr/bin/env python

"""Compute action-chunk advantage columns for raw LeRobot runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.value_function.advantage import AdvantageConfig, compute_advantage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Raw run root directory.")
    parser.add_argument("--value_mode", choices=("global", "subtask"), default="global")
    parser.add_argument(
        "--value_source", choices=("gt", "mock_pred", "model_pred"), default="gt"
    )
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument(
        "--subtask_inference_path",
        choices=("gt_conditioned", "pred_smooth"),
        default="gt_conditioned",
        help="Paired value-head and boundary path used for subtask advantage.",
    )
    parser.add_argument(
        "--boundary_transition_value",
        type=float,
        default=None,
        help="Progress assigned to each crossed subtask boundary (default: 1.0).",
    )
    parser.add_argument(
        "--boundary_bonus",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.boundary_transition_value is not None and args.boundary_bonus is not None:
        parser.error(
            "--boundary_transition_value and deprecated --boundary_bonus cannot be combined"
        )
    summary = compute_advantage(
        AdvantageConfig(
            root=args.root,
            value_mode=args.value_mode,
            value_source=args.value_source,
            chunk_size=args.chunk_size,
            subtask_inference_path=args.subtask_inference_path,
            boundary_transition_value=(
                1.0
                if args.boundary_transition_value is None
                else args.boundary_transition_value
            ),
            boundary_bonus=args.boundary_bonus,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
