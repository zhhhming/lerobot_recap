#!/usr/bin/env python

"""Run a trained value function over a complete raw run and write predictions back."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.value_function.inference import ValueInferenceConfig, infer_value_function


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("global", "subtask", "both"), default="both")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--image_keys",
        nargs="+",
        default=None,
        help="Must exactly match checkpoint camera keys and order; defaults to the checkpoint.",
    )
    parser.add_argument(
        "--subtask_inference_path",
        choices=("gt_conditioned", "pred_smooth", "both"),
        default="both",
    )
    parser.add_argument("--transition_penalty", type=float, default=0.0)
    parser.add_argument("--allow_subtask_skip", type=_parse_bool, default=False)
    parser.add_argument("--progress", type=_parse_bool, default=True)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    summary = infer_value_function(
        ValueInferenceConfig(
            root=args.root,
            checkpoint=args.checkpoint,
            mode=args.mode,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            image_keys=tuple(args.image_keys) if args.image_keys is not None else None,
            subtask_inference_path=args.subtask_inference_path,
            transition_penalty=args.transition_penalty,
            allow_subtask_skip=args.allow_subtask_skip,
            progress=args.progress,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
