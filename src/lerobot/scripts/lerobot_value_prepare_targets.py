#!/usr/bin/env python

"""Generate value-function ground-truth targets for raw LeRobot runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.value_function.targets import ValueTargetConfig, prepare_value_targets


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def _parse_subtask_scale_json(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("--subtask_scale_frames_json must be a JSON object")
    parsed: dict[str, float] = {}
    for key, raw_scale in payload.items():
        scale = float(raw_scale)
        if scale <= 0:
            raise argparse.ArgumentTypeError(
                f"Manual subtask scale for {key!r} must be > 0, got {scale}"
            )
        parsed[str(key)] = scale
    return parsed


def _parse_subtask_order_json(value: str | None) -> list[str] | None:
    if not value:
        return None
    payload = json.loads(value)
    if not isinstance(payload, list) or not all(isinstance(name, str) for name in payload):
        raise argparse.ArgumentTypeError("--subtask_order_json must be a JSON list of strings")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Raw run root directory.")
    parser.add_argument("--mode", choices=("global", "subtask", "both"), default="both")
    parser.add_argument("--num_bins", type=int, default=256)
    parser.add_argument("--global_num_bins", type=int, default=None)
    parser.add_argument("--subtask_num_bins", type=int, default=None)
    parser.add_argument("--global_scale", choices=("max", "p95", "manual"), default="p95")
    parser.add_argument("--subtask_scale", choices=("max", "p95", "manual"), default="p95")
    parser.add_argument("--global_scale_frames", type=float, default=None)
    parser.add_argument(
        "--subtask_scale_frames_json",
        type=_parse_subtask_scale_json,
        default=None,
        help='Manual subtask scales, for example: \'{"pick": 30, "place": 25}\'.',
    )
    parser.add_argument("--elapsed_aux", type=_parse_bool, default=False)
    parser.add_argument(
        "--allow_unlabeled",
        choices=("error", "skip", "default_subtask"),
        default="error",
    )
    parser.add_argument("--default_subtask", default=None)
    parser.add_argument("--subtask_column", default="subtask")
    parser.add_argument(
        "--subtask_order_json",
        type=_parse_subtask_order_json,
        default=None,
        help='Canonical execution order, for example: \'["pick", "place"]\'.',
    )
    parser.add_argument("--require_all_subtasks", type=_parse_bool, default=True)
    parser.add_argument(
        "--require_single_segment_per_subtask", type=_parse_bool, default=True
    )
    parser.add_argument("--require_success_only", type=_parse_bool, default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    summary = prepare_value_targets(
        ValueTargetConfig(
            root=args.root,
            mode=args.mode,
            num_bins=args.num_bins,
            global_num_bins=args.global_num_bins,
            subtask_num_bins=args.subtask_num_bins,
            global_scale=args.global_scale,
            subtask_scale=args.subtask_scale,
            global_scale_frames=args.global_scale_frames,
            subtask_scale_frames=args.subtask_scale_frames_json,
            elapsed_aux=args.elapsed_aux,
            allow_unlabeled=args.allow_unlabeled,
            default_subtask=args.default_subtask,
            subtask_column=args.subtask_column,
            subtask_order=args.subtask_order_json,
            require_all_subtasks=args.require_all_subtasks,
            require_single_segment_per_subtask=args.require_single_segment_per_subtask,
            require_success_only=args.require_success_only,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
