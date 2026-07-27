#!/usr/bin/env python

"""Train an offline PI0/PI0.5 value function directly from raw run roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.value_function.configuration import ValueFunctionConfig
from lerobot.value_function.dataset import ValueAugmentationConfig
from lerobot.value_function.training import ValueTrainingConfig, train_value_function


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", dest="roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("global", "subtask", "both"), default="global")
    parser.add_argument("--backbone_type", choices=("pi0", "pi05", "vision_only"), default="pi0")
    parser.add_argument("--paligemma_variant", choices=("gemma_300m", "gemma_2b"), default="gemma_2b")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--local_files_only", type=_parse_bool, default=False)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--image_keys",
        nargs="+",
        default=[
            "observation.images.left_wrist",
            "observation.images.right_wrist",
            "observation.images.third_person",
        ],
    )
    parser.add_argument("--image_resolution", type=int, default=224)
    parser.add_argument("--num_vlm_layers", type=int, default=3)
    parser.add_argument("--num_bins", type=int, default=256)
    parser.add_argument("--global_num_bins", type=int, default=None)
    parser.add_argument("--subtask_num_bins", type=int, default=None)
    parser.add_argument("--use_elapsed_aux", type=_parse_bool, default=False)
    parser.add_argument("--use_state", type=_parse_bool, default=True)
    parser.add_argument("--state_key", default="observation.state")
    parser.add_argument("--state_hidden_dim", type=int, default=256)
    parser.add_argument("--fusion_hidden_dim", type=int, default=512)
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--head_depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_vision_encoder", type=_parse_bool, default=True)
    parser.add_argument("--freeze_backbone", type=_parse_bool, default=True)
    parser.add_argument("--num_unfrozen_backbone_layers", type=int, default=0)
    parser.add_argument("--remaining_loss_weight", type=float, default=1.0)
    parser.add_argument("--subtask_ce_loss_weight", type=float, default=0.2)
    parser.add_argument("--elapsed_loss_weight", type=float, default=0.0)

    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument(
        "--val_episodes",
        nargs="*",
        default=[],
        help="Episode ids for one root, or ROOT_INDEX:EPISODE_INDEX for multiple roots.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--progress", type=_parse_bool, default=True)
    parser.add_argument("--save_every_epoch", type=_parse_bool, default=True)
    parser.add_argument("--progress_bins", type=int, default=10)

    parser.add_argument("--augmentation", type=_parse_bool, default=True)
    parser.add_argument("--crop_scale_min", type=float, default=0.9)
    parser.add_argument("--crop_ratio_min", type=float, default=0.95)
    parser.add_argument("--crop_ratio_max", type=float, default=1.05)
    parser.add_argument("--rotation_degrees", type=float, default=3.0)
    parser.add_argument("--brightness", type=float, default=0.1)
    parser.add_argument("--contrast", type=float, default=0.1)
    parser.add_argument("--saturation", type=float, default=0.1)
    parser.add_argument("--gaussian_blur_prob", type=float, default=0.0)
    parser.add_argument("--grayscale_prob", type=float, default=0.0)
    parser.add_argument("--noise_std", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    model = ValueFunctionConfig(
        mode=args.mode,
        backbone_type=args.backbone_type,
        paligemma_variant=args.paligemma_variant,
        precision=args.precision,
        pretrained_path=args.pretrained_path,
        local_files_only=args.local_files_only,
        revision=args.revision,
        image_keys=tuple(args.image_keys),
        image_resolution=(args.image_resolution, args.image_resolution),
        num_vlm_layers=args.num_vlm_layers,
        num_bins=args.num_bins,
        global_num_bins=args.global_num_bins,
        subtask_num_bins=args.subtask_num_bins,
        num_subtasks=1 if args.mode in {"subtask", "both"} else None,
        use_elapsed_aux=args.use_elapsed_aux,
        use_state=args.use_state,
        state_key=args.state_key,
        state_hidden_dim=args.state_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        head_depth=args.head_depth,
        dropout=args.dropout,
        freeze_vision_encoder=args.freeze_vision_encoder,
        freeze_backbone=args.freeze_backbone,
        num_unfrozen_backbone_layers=args.num_unfrozen_backbone_layers,
        remaining_loss_weight=args.remaining_loss_weight,
        subtask_ce_loss_weight=args.subtask_ce_loss_weight,
        elapsed_loss_weight=args.elapsed_loss_weight,
    )
    augmentation = ValueAugmentationConfig(
        enabled=args.augmentation,
        crop_scale_min=args.crop_scale_min,
        crop_ratio_min=args.crop_ratio_min,
        crop_ratio_max=args.crop_ratio_max,
        rotation_degrees=args.rotation_degrees,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        gaussian_blur_prob=args.gaussian_blur_prob,
        grayscale_prob=args.grayscale_prob,
        noise_std=args.noise_std,
    )
    config = ValueTrainingConfig(
        roots=tuple(str(root) for root in args.roots),
        output_dir=str(args.output_dir),
        model=model,
        val_fraction=args.val_fraction,
        val_episodes=tuple(args.val_episodes),
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        log_every_steps=args.log_every_steps,
        progress=args.progress,
        save_every_epoch=args.save_every_epoch,
        progress_bins=args.progress_bins,
        augmentation=augmentation,
    )
    summary = train_value_function(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
