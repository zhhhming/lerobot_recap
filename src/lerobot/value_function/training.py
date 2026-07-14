"""Independent training loop for offline PI0/PI0.5 value functions."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lerobot.value_function.configuration import ValueFunctionConfig
from lerobot.value_function.dataset import (
    RawValueFrameDataset,
    ValueAugmentationConfig,
    ValueFrameSubset,
    compute_state_statistics,
    parse_val_episode_keys,
    split_episode_indices,
    training_data_contract,
)
from lerobot.value_function.modeling_pi0_value import PI0ValueFunctionModel
from lerobot.value_function.raw_io import iso_utc_now, normalize_stage_config
from lerobot.value_function.schema import (
    VALUE_GLOBAL_ELAPSED_NORM_GT,
    VALUE_GLOBAL_REMAINING_FRAMES_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT,
    VALUE_GLOBAL_REMAINING_NORM_GT_IS_CLIPPED,
    VALUE_SUBTASK_ELAPSED_NORM_GT,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_REMAINING_FRAMES_GT,
    VALUE_SUBTASK_REMAINING_NORM_GT,
    VALUE_SUBTASK_REMAINING_NORM_GT_IS_CLIPPED,
)

ModelFactory = Callable[[ValueFunctionConfig], nn.Module]


@dataclass
class ValueTrainingConfig:
    roots: tuple[str, ...]
    output_dir: str
    model: ValueFunctionConfig = field(default_factory=ValueFunctionConfig)
    val_fraction: float = 0.1
    val_episodes: tuple[str, ...] = ()
    epochs: int = 10
    max_steps: int | None = None
    batch_size: int = 8
    num_workers: int = 0
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    log_every_steps: int = 50
    progress_bins: int = 10
    augmentation: ValueAugmentationConfig = field(default_factory=ValueAugmentationConfig)

    def __post_init__(self) -> None:
        self.roots = tuple(str(Path(root).expanduser().resolve()) for root in self.roots)
        self.val_episodes = tuple(self.val_episodes)
        if not self.roots:
            raise ValueError("At least one raw run root is required")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if self.max_grad_norm < 0 or self.progress_bins < 1:
            raise ValueError("max_grad_norm must be non-negative and progress_bins positive")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(dict(payload), temp_name)
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _resolved_model_config(
    requested: ValueFunctionConfig,
    dataset: RawValueFrameDataset,
) -> ValueFunctionConfig:
    payload = requested.to_dict()
    if requested.use_state:
        payload["state_dim"] = dataset.state_dim
    if requested.mode in {"subtask", "both"}:
        payload["num_subtasks"] = len(dataset.subtask_order)
    resolved = ValueFunctionConfig.from_dict(payload)
    first = dataset.contracts[0]
    if resolved.mode in {"global", "both"}:
        if resolved.resolved_global_num_bins != first.global_num_bins:
            raise ValueError(
                "Model global_num_bins does not match prepared target metadata: "
                f"{resolved.resolved_global_num_bins} != {first.global_num_bins}"
            )
    if resolved.mode in {"subtask", "both"}:
        if resolved.resolved_subtask_num_bins != first.subtask_num_bins:
            raise ValueError(
                "Model subtask_num_bins does not match prepared target metadata: "
                f"{resolved.resolved_subtask_num_bins} != {first.subtask_num_bins}"
            )
    return resolved


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _selected_subtask_value(values: Tensor, ids: Tensor) -> Tensor:
    return values.gather(1, ids.long().unsqueeze(1)).squeeze(1)


class MetricAccumulator:
    def __init__(
        self,
        mode: str,
        *,
        use_elapsed_aux: bool,
        progress_bins: int,
        subtask_order: Sequence[str] = (),
    ):
        self.mode = mode
        self.use_elapsed_aux = use_elapsed_aux
        self.progress_bins = progress_bins
        self.subtask_order = tuple(subtask_order)
        self.count = 0
        self.loss_sums: defaultdict[str, float] = defaultdict(float)
        self.norm_abs_sum: defaultdict[str, float] = defaultdict(float)
        self.frame_abs_sum: defaultdict[str, float] = defaultdict(float)
        self.elapsed_abs_sum: defaultdict[str, float] = defaultdict(float)
        self.subtask_correct = 0
        self.clip_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.records: list[dict[str, Any]] = []

    def update(
        self,
        outputs: dict[str, Tensor],
        losses: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> None:
        batch_size = int(outputs["features"].shape[0])
        self.count += batch_size
        for name, value in losses.items():
            self.loss_sums[name] += float(value.detach().item()) * batch_size

        root_ids = batch["value_root_index"].detach().cpu().tolist()
        episode_ids = batch["value_episode_index"].detach().cpu().tolist()
        frame_ids = batch["value_frame_index"].detach().cpu().tolist()
        progress = batch["value_subtask_progress"].detach().cpu().tolist()
        subtask_ids = (
            batch[VALUE_SUBTASK_ID_GT].long() if VALUE_SUBTASK_ID_GT in batch else None
        )

        predictions: dict[str, Tensor] = {}
        if self.mode in {"global", "both"}:
            pred = outputs["global_remaining_value"]
            target = batch[VALUE_GLOBAL_REMAINING_NORM_GT]
            scale = batch["value_global_scale_frames"]
            self.norm_abs_sum["global"] += float((pred - target).abs().sum().item())
            self.frame_abs_sum["global"] += float(
                (pred * scale - batch[VALUE_GLOBAL_REMAINING_FRAMES_GT]).abs().sum().item()
            )
            predictions["global"] = pred
            clipped = batch[VALUE_GLOBAL_REMAINING_NORM_GT_IS_CLIPPED].bool()
            self.clip_counts["global"][0] += int(clipped.sum().item())
            self.clip_counts["global"][1] += batch_size
            if self.use_elapsed_aux:
                elapsed = outputs["global_elapsed_value"]
                elapsed_target = batch[VALUE_GLOBAL_ELAPSED_NORM_GT]
                self.elapsed_abs_sum["global"] += float((elapsed - elapsed_target).abs().sum().item())

        if self.mode in {"subtask", "both"}:
            pred = _selected_subtask_value(outputs["subtask_remaining_value"], subtask_ids)
            target = batch[VALUE_SUBTASK_REMAINING_NORM_GT]
            scale = batch["value_subtask_scale_frames"]
            self.norm_abs_sum["subtask"] += float((pred - target).abs().sum().item())
            self.frame_abs_sum["subtask"] += float(
                (pred * scale - batch[VALUE_SUBTASK_REMAINING_FRAMES_GT]).abs().sum().item()
            )
            predictions["subtask"] = pred
            self.subtask_correct += int(
                (outputs["subtask_logits"].argmax(dim=-1) == subtask_ids).sum().item()
            )
            clipped = batch[VALUE_SUBTASK_REMAINING_NORM_GT_IS_CLIPPED].bool()
            for subtask_id in subtask_ids.unique().tolist():
                mask = subtask_ids == subtask_id
                name = (
                    self.subtask_order[subtask_id]
                    if 0 <= subtask_id < len(self.subtask_order)
                    else str(subtask_id)
                )
                key = f"subtask:{name}"
                self.clip_counts[key][0] += int(clipped[mask].sum().item())
                self.clip_counts[key][1] += int(mask.sum().item())
            if self.use_elapsed_aux:
                elapsed = _selected_subtask_value(outputs["subtask_elapsed_value"], subtask_ids)
                elapsed_target = batch[VALUE_SUBTASK_ELAPSED_NORM_GT]
                self.elapsed_abs_sum["subtask"] += float(
                    (elapsed - elapsed_target).abs().sum().item()
                )

        cpu_predictions = {name: value.detach().cpu().tolist() for name, value in predictions.items()}
        cpu_subtasks = subtask_ids.detach().cpu().tolist() if subtask_ids is not None else [-1] * batch_size
        for index in range(batch_size):
            record = {
                "root": int(root_ids[index]),
                "episode": int(episode_ids[index]),
                "frame": int(frame_ids[index]),
                "subtask_id": int(cpu_subtasks[index]),
                "progress": float(progress[index]),
            }
            for name, values in cpu_predictions.items():
                record[f"{name}_prediction"] = float(values[index])
            self.records.append(record)

    def _monotonic_violation_rate(self, value_key: str) -> float | None:
        groups: defaultdict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            subtask = record["subtask_id"] if value_key == "subtask" else -1
            groups[(record["root"], record["episode"], subtask)].append(record)
        violations = 0
        comparisons = 0
        for records in groups.values():
            records.sort(key=lambda item: item["frame"])
            for previous, current in zip(records, records[1:], strict=False):
                if current["frame"] != previous["frame"] + 1:
                    continue
                comparisons += 1
                prediction_key = f"{value_key}_prediction"
                if current[prediction_key] > previous[prediction_key] + 1e-6:
                    violations += 1
        return violations / comparisons if comparisons else None

    def _group_prediction_variance(self) -> float | None:
        prediction_groups: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
        for record in self.records:
            if "subtask_prediction" not in record:
                continue
            progress = record["progress"]
            if not math.isfinite(progress):
                continue
            progress_bin = min(max(int(progress * self.progress_bins), 0), self.progress_bins - 1)
            prediction_groups[(record["subtask_id"], progress_bin)].append(
                record["subtask_prediction"]
            )
        weighted_sum = 0.0
        weighted_count = 0
        for values in prediction_groups.values():
            if len(values) < 2:
                continue
            weighted_sum += float(np.var(values)) * len(values)
            weighted_count += len(values)
        return weighted_sum / weighted_count if weighted_count else None

    def finalize(self) -> dict[str, Any]:
        if not self.count:
            return {"samples": 0}
        metrics: dict[str, Any] = {
            "samples": self.count,
            "losses": {name: value / self.count for name, value in self.loss_sums.items()},
            "normalized_mae": {
                name: value / self.count for name, value in self.norm_abs_sum.items()
            },
            "frame_mae": {name: value / self.count for name, value in self.frame_abs_sum.items()},
            "clip_rate": {
                name: clipped / eligible if eligible else None
                for name, (clipped, eligible) in self.clip_counts.items()
            },
            "monotonic_violation_rate": {
                name: self._monotonic_violation_rate(name) for name in self.norm_abs_sum
            },
        }
        if self.mode in {"subtask", "both"}:
            metrics["subtask_accuracy"] = self.subtask_correct / self.count
            metrics["subtask_progress_bin_prediction_variance"] = self._group_prediction_variance()
        if self.use_elapsed_aux:
            metrics["elapsed_normalized_mae"] = {
                name: value / self.count for name, value in self.elapsed_abs_sum.items()
            }
        return metrics


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _run_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    max_grad_norm: float,
    mode: str,
    use_elapsed_aux: bool,
    progress_bins: int,
    subtask_order: Sequence[str],
    remaining_steps: int | None,
) -> tuple[dict[str, Any], int]:
    training = optimizer is not None
    model.train(training)
    accumulator = MetricAccumulator(
        mode,
        use_elapsed_aux=use_elapsed_aux,
        progress_bins=progress_bins,
        subtask_order=subtask_order,
    )
    steps = 0
    for batch in loader:
        if remaining_steps is not None and steps >= remaining_steps:
            break
        batch = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(batch)
            losses = model.compute_loss(outputs, batch)
            if training:
                losses["loss"].backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        max_grad_norm,
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        accumulator.update(outputs, losses, batch)
        steps += 1
    return accumulator.finalize(), steps


def _model_checkpoint_payload(
    model: nn.Module,
    *,
    config: ValueTrainingConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_contract: dict[str, Any],
    state_mean: Tensor | None,
    state_std: Tensor | None,
    train_episodes: Sequence[tuple[int, int]],
    val_episodes: Sequence[tuple[int, int]],
    epoch: int,
    step: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(model.checkpoint_payload())
    payload.update(
        {
            "training_config": normalize_stage_config(config),
            "data_contract": data_contract,
            "split": {
                "train_episodes": [list(key) for key in train_episodes],
                "val_episodes": [list(key) for key in val_episodes],
            },
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "state_normalization": {
                "key": config.model.state_key if config.model.use_state else None,
                "mean": state_mean.tolist() if state_mean is not None else None,
                "std": state_std.tolist() if state_std is not None else None,
            },
        }
    )
    return payload


def load_value_function_checkpoint(
    checkpoint: str | Path,
    *,
    model_factory: ModelFactory | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(Path(checkpoint), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError(f"Invalid value-function checkpoint: {checkpoint}")
    config = ValueFunctionConfig.from_dict(payload["model_config"])
    factory = model_factory or (lambda cfg: PI0ValueFunctionModel(cfg, load_pretrained=False))
    model = factory(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def train_value_function(
    config: ValueTrainingConfig,
    *,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    _set_seed(config.seed)
    if config.model.elapsed_loss_weight > 0 and not config.model.use_elapsed_aux:
        raise ValueError("elapsed_loss_weight > 0 requires use_elapsed_aux=true")
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    dataset = RawValueFrameDataset(
        config.roots,
        mode=config.model.mode,
        image_keys=config.model.image_keys,
        state_key=config.model.state_key,
        use_state=config.model.use_state,
        use_elapsed_aux=config.model.use_elapsed_aux,
        augmentation=config.augmentation,
    )
    model_config = _resolved_model_config(config.model, dataset)
    config.model = model_config
    explicit_val = (
        parse_val_episode_keys(config.val_episodes, num_roots=len(config.roots))
        if config.val_episodes
        else None
    )
    train_indices, val_indices, train_episodes, val_episodes = split_episode_indices(
        dataset,
        val_fraction=config.val_fraction,
        val_episodes=explicit_val,
        seed=config.seed,
    )
    state_mean = state_std = None
    if model_config.use_state:
        state_mean, state_std = compute_state_statistics(dataset, train_indices)

    train_loader = DataLoader(
        ValueFrameSubset(dataset, train_indices, augment=config.augmentation.enabled),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=resolve_device(config.device).type == "cuda",
        drop_last=False,
        prefetch_factor=2 if config.num_workers else None,
    )
    val_loader = DataLoader(
        ValueFrameSubset(dataset, val_indices, augment=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=resolve_device(config.device).type == "cuda",
        drop_last=False,
        prefetch_factor=2 if config.num_workers else None,
    )
    factory = model_factory or (lambda cfg: PI0ValueFunctionModel(cfg))
    model = factory(model_config)
    if state_mean is not None:
        model.set_state_normalization_stats(state_mean, state_std)
    device = resolve_device(config.device)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Value model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    steps_per_epoch = max(math.ceil(len(train_indices) / config.batch_size), 1)
    total_steps = config.max_steps or steps_per_epoch * config.epochs
    scheduler = _build_scheduler(
        optimizer, total_steps=total_steps, warmup_ratio=config.warmup_ratio
    )
    contract = training_data_contract(dataset)
    config_payload = normalize_stage_config(config)
    _atomic_json(output_dir / "config.json", config_payload)
    _atomic_json(
        output_dir / "value_function_meta.json",
        {
            "created_at": iso_utc_now(),
            "stage": "value_training",
            "model_config": model_config.to_dict(),
            "data_contract": contract,
            "split": {
                "train_episodes": [list(key) for key in train_episodes],
                "val_episodes": [list(key) for key in val_episodes],
            },
            "state_normalization": {
                "key": model_config.state_key if model_config.use_state else None,
                "mean": state_mean.tolist() if state_mean is not None else None,
                "std": state_std.tolist() if state_std is not None else None,
            },
            "checkpoint": "checkpoint.pt",
            "metrics": "train_metrics.jsonl",
        },
    )

    global_step = 0
    final_record: dict[str, Any] = {}
    for epoch in range(1, config.epochs + 1):
        remaining = None if config.max_steps is None else config.max_steps - global_step
        if remaining is not None and remaining <= 0:
            break
        train_metrics, steps = _run_loader(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=config.max_grad_norm,
            mode=model_config.mode,
            use_elapsed_aux=model_config.use_elapsed_aux,
            progress_bins=config.progress_bins,
            subtask_order=dataset.subtask_order,
            remaining_steps=remaining,
        )
        global_step += steps
        with torch.inference_mode():
            val_metrics, _ = _run_loader(
                model,
                val_loader,
                device=device,
                optimizer=None,
                scheduler=None,
                max_grad_norm=0.0,
                mode=model_config.mode,
                use_elapsed_aux=model_config.use_elapsed_aux,
                progress_bins=config.progress_bins,
                subtask_order=dataset.subtask_order,
                remaining_steps=None,
            )
        final_record = {
            "created_at": iso_utc_now(),
            "epoch": epoch,
            "step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        _append_jsonl(metrics_path, final_record)
        checkpoint_payload = _model_checkpoint_payload(
            model,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            data_contract=contract,
            state_mean=state_mean,
            state_std=state_std,
            train_episodes=train_episodes,
            val_episodes=val_episodes,
            epoch=epoch,
            step=global_step,
            metrics=final_record,
        )
        _atomic_torch_save(output_dir / "checkpoint.pt", checkpoint_payload)
        if config.max_steps is not None and global_step >= config.max_steps:
            break

    if global_step == 0:
        raise RuntimeError("Value training completed without an optimizer step")
    return {
        "output_dir": str(output_dir),
        "checkpoint": str(output_dir / "checkpoint.pt"),
        "steps": global_step,
        "train_frames": len(train_indices),
        "val_frames": len(val_indices),
        "metrics": final_record,
    }
