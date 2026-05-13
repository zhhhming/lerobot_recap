#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a PI0/PaliGemma prefix-only RECAP value network on success episodes."""

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lerobot.configs import parser
from lerobot.configs.types import FeatureType
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.rl.algorithms.recap_value_network_pi0 import (
    RECAPPI0ValueNetwork,
    RECAPPI0ValueNetworkConfig,
)


@dataclass(frozen=True)
class FrameTarget:
    dataset_index: int
    frame_index: int
    abs_index: int
    episode_index: int
    target_value: float
    target_bin: int


@dataclass
class RECAPPI0ValueTrainingConfig:
    repo_id: str | None = None
    repo_ids: list[str] | None = None
    root: str | None = None
    roots: list[str] | None = None
    output_dir: str = "outputs/recap_pi0_value"
    episodes: list[int] | None = None
    revision: str | None = None

    epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    val_split_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"
    max_train_steps_per_epoch: int | None = None
    max_val_steps_per_epoch: int | None = None
    log_every_n_steps: int = 50

    num_value_bins: int = 101
    target_mode: Literal["soft_ce", "hard_ce"] = "soft_ce"
    normalize_by_task: bool = False

    tokenizer_max_length: int = 48
    image_size: int = 224
    paligemma_variant: str = "gemma_2b"
    model_precision: Literal["float32", "bfloat16"] = "float32"
    num_vlm_layers: int = 8
    freeze_vision_encoder: bool = True
    freeze_backbone: bool = True
    num_unfrozen_backbone_layers: int = 2
    value_head_hidden_dim: int = 512
    value_head_depth: int = 1
    dropout: float = 0.1
    pretrained_path: str | None = "lerobot/pi0_base"
    local_files_only: bool = False
    pretrained_revision: str | None = None


class RECAPPI0FrameDataset(Dataset):
    def __init__(self, datasets: list[LeRobotDataset], frame_targets: list[FrameTarget], num_value_bins: int):
        self.datasets = datasets
        self.frame_targets = frame_targets
        self.num_value_bins = num_value_bins

    def __len__(self) -> int:
        return len(self.frame_targets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        target = self.frame_targets[idx]
        frame = dict(self.datasets[target.dataset_index][target.frame_index])
        frame["dataset_index"] = torch.tensor(target.dataset_index, dtype=torch.long)
        frame["frame_index"] = torch.tensor(target.frame_index, dtype=torch.long)
        frame["abs_frame_index"] = torch.tensor(target.abs_index, dtype=torch.long)
        frame["target_value"] = torch.tensor(target.target_value, dtype=torch.float32)
        frame["target_bin"] = torch.tensor(target.target_bin, dtype=torch.long)
        frame["target_probs"] = torch.tensor(
            _target_probs(target.target_value, self.num_value_bins), dtype=torch.float32
        )
        return frame


_TRAINING_METADATA_KEYS = (
    "dataset_index",
    "episode_index",
    "frame_index",
    "abs_frame_index",
    "target_value",
    "target_bin",
    "target_probs",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


def _resolve_repo_ids(cfg: RECAPPI0ValueTrainingConfig) -> list[str]:
    if cfg.repo_ids:
        return cfg.repo_ids
    repo_ids = _split_csv(cfg.repo_id)
    if repo_ids:
        return repo_ids
    raise ValueError("Provide either repo_id or repo_ids.")


def _resolve_roots(cfg: RECAPPI0ValueTrainingConfig, repo_ids: list[str]) -> list[str | None]:
    if cfg.roots:
        if len(cfg.roots) != len(repo_ids):
            raise ValueError(f"roots has {len(cfg.roots)} entries but repo_ids has {len(repo_ids)}")
        return cfg.roots
    roots = _split_csv(cfg.root)
    if roots is not None:
        if len(roots) == 1 and len(repo_ids) > 1:
            return [str(Path(roots[0]).expanduser() / repo_id) for repo_id in repo_ids]
        if len(roots) != len(repo_ids):
            raise ValueError(f"root provides {len(roots)} entries but repo_ids has {len(repo_ids)}")
        return roots
    return [None] * len(repo_ids)


def _load_datasets(cfg: RECAPPI0ValueTrainingConfig) -> list[LeRobotDataset]:
    repo_ids = _resolve_repo_ids(cfg)
    roots = _resolve_roots(cfg, repo_ids)
    datasets: list[LeRobotDataset] = []
    for repo_id, root in zip(repo_ids, roots, strict=True):
        datasets.append(
            LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=cfg.episodes,
                revision=cfg.revision,
            )
        )
    return datasets


def _to_int(value: Any) -> int:
    if isinstance(value, Tensor):
        return int(value.item())
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def _to_float(value: Any) -> float:
    if isinstance(value, Tensor):
        return float(value.item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _target_probs(value: float, num_bins: int, v_min: float = -1.0, v_max: float = 0.0) -> list[float]:
    value = min(max(value, v_min), v_max)
    scaled = (value - v_min) / (v_max - v_min) * (num_bins - 1)
    lower = int(math.floor(scaled))
    upper = min(lower + 1, num_bins - 1)
    upper_weight = scaled - lower
    probs = [0.0] * num_bins
    probs[lower] += 1.0 - upper_weight
    probs[upper] += upper_weight
    return probs


def _target_bin(value: float, num_bins: int, v_min: float = -1.0, v_max: float = 0.0) -> int:
    value = min(max(value, v_min), v_max)
    scaled = (value - v_min) / (v_max - v_min) * (num_bins - 1)
    return int(round(scaled))


def _episode_task(dataset: LeRobotDataset, episode_index: int) -> str:
    ep_data = dataset.meta.episodes[episode_index]
    if "task_index" in ep_data:
        task_index = _to_int(ep_data["task_index"])
        return str(dataset.meta.tasks.iloc[task_index].name)
    if "tasks" in ep_data:
        tasks = ep_data["tasks"]
        if isinstance(tasks, list) and tasks:
            return str(tasks[0])
    return "__default_task__"


def _episode_indices(dataset: LeRobotDataset) -> list[int]:
    if dataset.episodes is not None:
        return list(dataset.episodes)
    return list(range(dataset.meta.total_episodes))


def _build_frame_targets(
    datasets: list[LeRobotDataset],
    num_value_bins: int,
    normalize_by_task: bool,
) -> list[FrameTarget]:
    episode_lengths: dict[tuple[int, int], int] = {}
    task_max_len: dict[str, int] = {}
    global_max_remaining = 1
    for dataset_index, dataset in enumerate(datasets):
        for episode_index in _episode_indices(dataset):
            ep = dataset.meta.episodes[episode_index]
            length = _to_int(ep["dataset_to_index"]) - _to_int(ep["dataset_from_index"])
            if length <= 0:
                continue
            episode_lengths[(dataset_index, episode_index)] = length
            max_remaining = max(length - 1, 1)
            global_max_remaining = max(global_max_remaining, max_remaining)
            task = _episode_task(dataset, episode_index)
            task_max_len[task] = max(task_max_len.get(task, 1), max_remaining)

    frame_targets: list[FrameTarget] = []
    for dataset_index, dataset in enumerate(datasets):
        abs_to_rel_idx = {
            _to_int(abs_idx): rel_idx
            for rel_idx, abs_idx in enumerate(dataset.hf_dataset["index"])
        }
        for episode_index in _episode_indices(dataset):
            ep = dataset.meta.episodes[episode_index]
            start = _to_int(ep["dataset_from_index"])
            end = _to_int(ep["dataset_to_index"])
            length = episode_lengths.get((dataset_index, episode_index))
            if length is None:
                continue
            denom = task_max_len[_episode_task(dataset, episode_index)] if normalize_by_task else global_max_remaining
            denom = max(float(denom), 1.0)
            for offset, abs_index in enumerate(range(start, end)):
                rel_idx = abs_to_rel_idx.get(abs_index)
                if rel_idx is None:
                    continue
                remaining = max(length - 1 - offset, 0)
                target_value = -float(remaining) / denom
                target_value = min(max(target_value, -1.0), 0.0)
                frame_targets.append(
                    FrameTarget(
                        dataset_index=dataset_index,
                        frame_index=rel_idx,
                        abs_index=abs_index,
                        episode_index=episode_index,
                        target_value=target_value,
                        target_bin=_target_bin(target_value, num_value_bins),
                    )
                )
    if not frame_targets:
        raise ValueError("No frame targets were built. Check dataset episodes and metadata.")
    return frame_targets


def _split_train_val_targets(
    frame_targets: list[FrameTarget],
    val_ratio: float,
    seed: int,
) -> tuple[list[FrameTarget], list[FrameTarget]]:
    by_episode: dict[tuple[int, int], list[FrameTarget]] = {}
    for target in frame_targets:
        by_episode.setdefault((target.dataset_index, target.episode_index), []).append(target)

    episode_keys = sorted(by_episode)
    rng = random.Random(seed)
    rng.shuffle(episode_keys)
    val_count = int(round(len(episode_keys) * val_ratio))
    if val_ratio > 0 and val_count == 0 and len(episode_keys) > 1:
        val_count = 1
    val_keys = set(episode_keys[:val_count])

    train_targets: list[FrameTarget] = []
    val_targets: list[FrameTarget] = []
    for key, targets in by_episode.items():
        if key in val_keys:
            val_targets.extend(targets)
        else:
            train_targets.extend(targets)
    if not train_targets:
        raise ValueError("Train split is empty. Reduce val_split_ratio.")
    return train_targets, val_targets


def _build_pi0_config(
    datasets: list[LeRobotDataset],
    cfg: RECAPPI0ValueTrainingConfig,
    device: torch.device,
) -> PI0Config:
    features = dataset_to_policy_features(datasets[0].meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    policy_cfg = PI0Config(
        input_features=input_features,
        output_features=output_features,
        paligemma_variant=cfg.paligemma_variant,
        dtype=cfg.model_precision,
        image_resolution=(cfg.image_size, cfg.image_size),
        tokenizer_max_length=cfg.tokenizer_max_length,
        device=str(device),
    )
    policy_cfg.validate_features()
    return policy_cfg


def _dataset_stats(datasets: list[LeRobotDataset]) -> dict[str, dict[str, Tensor]]:
    if len(datasets) == 1:
        return datasets[0].meta.stats
    return aggregate_stats([dataset.meta.stats for dataset in datasets])


def _preprocess_batch(batch: dict[str, Any], preprocessor) -> dict[str, Any]:
    preserved = {key: batch[key] for key in _TRAINING_METADATA_KEYS if key in batch}
    batch = preprocessor(batch)
    device = next((value.device for value in batch.values() if isinstance(value, Tensor)), torch.device("cpu"))
    for key, value in preserved.items():
        if isinstance(value, Tensor):
            value = value.to(device)
        batch[key] = value
    return batch


def _build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _loss(outputs: dict[str, Tensor], batch: dict[str, Tensor], target_mode: str) -> Tensor:
    logits = outputs["value_logits"]
    if target_mode == "hard_ce":
        return F.cross_entropy(logits, batch["target_bin"])
    if target_mode == "soft_ce":
        log_probs = F.log_softmax(logits, dim=-1)
        return -(batch["target_probs"] * log_probs).sum(dim=-1).mean()
    raise ValueError(f"Unknown target_mode: {target_mode}")


def _run_epoch(
    model: RECAPPI0ValueNetwork,
    loader: DataLoader,
    preprocessor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    cfg: RECAPPI0ValueTrainingConfig,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    phase = "train" if training else "val"
    total_loss = 0.0
    total_mae = 0.0
    total_acc = 0.0
    total_samples = 0
    max_steps = cfg.max_train_steps_per_epoch if training else cfg.max_val_steps_per_epoch
    grad_accum = max(1, cfg.gradient_accumulation_steps) if training else 1
    if training:
        optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = _preprocess_batch(batch, preprocessor)
        for key in ("target_value", "target_bin", "target_probs"):
            batch[key] = batch[key].to(device)

        with torch.set_grad_enabled(training):
            outputs = model(batch)
            loss = _loss(outputs, batch, cfg.target_mode)
            if training:
                (loss / grad_accum).backward()
                boundary = ((step + 1) % grad_accum == 0) or (step + 1 == len(loader))
                if boundary:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

        with torch.no_grad():
            batch_size = batch["target_value"].shape[0]
            pred_bins = outputs["value_logits"].argmax(dim=-1)
            acc = (pred_bins == batch["target_bin"]).float().mean()
            mae = torch.abs(outputs["expected_value"] - batch["target_value"]).mean()
            total_loss += float(loss.item()) * batch_size
            total_acc += float(acc.item()) * batch_size
            total_mae += float(mae.item()) * batch_size
            total_samples += batch_size

        if cfg.log_every_n_steps > 0 and ((step + 1) % cfg.log_every_n_steps == 0 or step == 0):
            elapsed = max(time.perf_counter() - start, 1e-9)
            logging.info(
                f"[{phase} epoch {epoch}/{total_epochs}] step={step + 1}/{len(loader)} "
                f"loss={total_loss / total_samples:.5f} "
                f"bin_acc={total_acc / total_samples:.4f} "
                f"value_mae={total_mae / total_samples:.5f} "
                f"samples/s={total_samples / elapsed:.2f}"
            )

    if total_samples == 0:
        return {"loss": float("nan"), "bin_acc": float("nan"), "value_mae": float("nan")}
    return {
        "loss": total_loss / total_samples,
        "bin_acc": total_acc / total_samples,
        "value_mae": total_mae / total_samples,
    }


@parser.wrap()
def run_recap_pi0_value_train(cfg: RECAPPI0ValueTrainingConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    _set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "train_config.json", asdict(cfg))

    device = _resolve_device(cfg.device)
    logging.info(f"Using device: {device}")

    datasets = _load_datasets(cfg)
    frame_targets = _build_frame_targets(
        datasets=datasets,
        num_value_bins=cfg.num_value_bins,
        normalize_by_task=cfg.normalize_by_task,
    )
    train_targets, val_targets = _split_train_val_targets(frame_targets, cfg.val_split_ratio, cfg.seed)
    logging.info(
        f"Built success-only targets: total_frames={len(frame_targets)}, "
        f"train={len(train_targets)}, val={len(val_targets)}, datasets={len(datasets)}"
    )

    policy_cfg = _build_pi0_config(datasets, cfg, device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        dataset_stats=_dataset_stats(datasets),
    )
    image_feature_keys = list(policy_cfg.image_features)

    train_dataset = RECAPPI0FrameDataset(datasets, train_targets, cfg.num_value_bins)
    val_dataset = RECAPPI0FrameDataset(datasets, val_targets, cfg.num_value_bins)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    model_config = RECAPPI0ValueNetworkConfig(
        paligemma_variant=cfg.paligemma_variant,
        precision=cfg.model_precision,
        image_resolution=(cfg.image_size, cfg.image_size),
        image_feature_keys=image_feature_keys,
        num_value_bins=cfg.num_value_bins,
        num_vlm_layers=cfg.num_vlm_layers,
        freeze_vision_encoder=cfg.freeze_vision_encoder,
        freeze_backbone=cfg.freeze_backbone,
        num_unfrozen_backbone_layers=cfg.num_unfrozen_backbone_layers,
        value_head_hidden_dim=cfg.value_head_hidden_dim,
        value_head_depth=cfg.value_head_depth,
        dropout=cfg.dropout,
        pretrained_path=cfg.pretrained_path,
        local_files_only=cfg.local_files_only,
        revision=cfg.pretrained_revision,
    )
    model = RECAPPI0ValueNetwork(model_config).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    logging.info(
        f"Trainable parameters: {sum(p.numel() for p in trainable_params):,} / "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    if cfg.max_train_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_steps_per_epoch)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / max(1, cfg.gradient_accumulation_steps))
    scheduler = _build_scheduler(optimizer, optimizer_steps_per_epoch * cfg.epochs, cfg.warmup_ratio)

    best_val_mae = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, preprocessor, device, optimizer, scheduler, cfg, epoch, cfg.epochs
        )
        val_metrics = (
            _run_epoch(model, val_loader, preprocessor, device, None, None, cfg, epoch, cfg.epochs)
            if len(val_dataset) > 0
            else {"loss": float("nan"), "bin_acc": float("nan"), "value_mae": float("nan")}
        )
        epoch_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_metrics)
        _save_json(output_dir / "metrics.json", {"history": history})
        logging.info(
            f"Epoch {epoch}/{cfg.epochs}: "
            f"train_loss={train_metrics['loss']:.5f} train_mae={train_metrics['value_mae']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_mae={val_metrics['value_mae']:.5f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model.config_dict(),
            "train_config": asdict(cfg),
            "epoch": epoch,
            "metrics": epoch_metrics,
        }
        torch.save(checkpoint, checkpoints_dir / f"epoch_{epoch:04d}.pt")
        val_mae = val_metrics["value_mae"]
        if math.isnan(val_mae) or val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(checkpoint, checkpoints_dir / "best.pt")
        torch.save(checkpoint, checkpoints_dir / "last.pt")


if __name__ == "__main__":
    run_recap_pi0_value_train()
