#!/usr/bin/env python

"""Generate synthetic GT-plus-noise value predictions for smoke tests only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.value_function.mock_predictions import (
    MockPredictionConfig,
    generate_mock_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Raw run root directory.")
    parser.add_argument("--mode", choices=("global", "subtask", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_std_frames", type=float, default=3.0)
    parser.add_argument("--temporal_smoothing_sigma_frames", type=float, default=0.0)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    summary = generate_mock_predictions(
        MockPredictionConfig(
            root=args.root,
            mode=args.mode,
            seed=args.seed,
            noise_std_frames=args.noise_std_frames,
            temporal_smoothing_sigma_frames=args.temporal_smoothing_sigma_frames,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
