#!/usr/bin/env python

"""Fail T7 validation if a real training log lacks finite loss/grad-norm updates."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LOSS = re.compile(rf"\bloss:\s*({NUMBER})")
GRAD = re.compile(rf"\bgrdn:\s*({NUMBER})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.log.read_text(errors="replace")
    losses = [float(value) for value in LOSS.findall(text)]
    gradients = [float(value) for value in GRAD.findall(text)]
    if len(losses) < args.expected_updates or len(gradients) < args.expected_updates:
        raise RuntimeError(
            f"{args.label}: expected at least {args.expected_updates} logged updates, "
            f"found loss={len(losses)} grad_norm={len(gradients)}"
        )
    losses = losses[-args.expected_updates :]
    gradients = gradients[-args.expected_updates :]
    if not all(math.isfinite(value) for value in [*losses, *gradients]):
        raise RuntimeError(
            f"{args.label}: non-finite training values: loss={losses}, grad_norm={gradients}"
        )
    rendered = json.dumps(
        {
            "label": args.label,
            "updates": args.expected_updates,
            "loss": losses,
            "grad_norm": gradients,
        },
        sort_keys=True,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
