#!/usr/bin/env python

"""Train PI0 with RECAP advantage-positive language conditioning.

This script keeps RECAP annotations in a sidecar JSON file instead of modifying
LeRobot dataset parquet files.  It supports multiple local/Hub LeRobot datasets
for the narrow RECAP+PI0 workflow.
"""

import json
import logging
import math
import random
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset

from lerobot.configs import parser
from lerobot.configs.types import FeatureType
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.rl.algorithms import recap_train_value_network_pi0 as value_train
from lerobot.rl.algorithms.recap_value_network_pi0 import RECAPPI0ValueNetwork, RECAPPI0ValueNetworkConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass
class RECAPPI0TrainingConfig:
    repo_id: str | None = None
    repo_ids: list[str] | None = None
    root: str | None = None
    roots: list[str] | None = None
    output_dir: str = "outputs/recap_pi0"
    episodes: list[int] | None = None
    revision: str | None = None

    value_network_checkpoint: str | None = None
    advantage_labels_path: str | None = None
    recompute_advantage_labels: bool = False
    advantage_positive_percentile: float = 70.0
    positive_label: str = "advantage:positive"
    positive_label_position: Literal["prefix", "suffix"] = "prefix"
    positive_label_dropout: float = 0.3
    vn_batch_size: int = 8

    epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    learning_rate: float = 2.5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    val_split_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"
    max_train_steps_per_epoch: int | None = None
    max_val_steps_per_epoch: int | None = None
    log_every_n_steps: int = 50

    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    model_precision: Literal["float32", "bfloat16"] = "bfloat16"
    pretrained_path: str | None = "lerobot/pi0_base"
    pretrained_revision: str | None = None
    local_files_only: bool = False
    tokenizer_max_length: int = 64
    image_size: int = 224
    freeze_vision_encoder: bool = True
    freeze_backbone: bool = True
    num_unfrozen_backbone_layers: int = 3
    train_expert_only: bool = False
    gradient_checkpointing: bool = False

    image_cutout_prob: float = 0.0
    image_cutout_min_area: float = 0.02
    image_cutout_max_area: float = 0.15
    image_cutout_cameras: list[str] | None = None
    state_mask_prob: float = 0.0
    state_mask_dims: list[int] | None = None
    state_mask_num_dims: int = 1


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


def _resolve_repo_ids(cfg: RECAPPI0TrainingConfig) -> list[str]:
    if cfg.repo_ids:
        return cfg.repo_ids
    repo_ids = _split_csv(cfg.repo_id)
    if repo_ids:
        return repo_ids
    raise ValueError("Provide either repo_id or repo_ids.")


def _resolve_roots(cfg: RECAPPI0TrainingConfig, repo_ids: list[str]) -> list[str | None]:
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


def _load_datasets(
    cfg: RECAPPI0TrainingConfig,
    delta_timestamps: dict[str, list] | None = None,
) -> list[LeRobotDataset]:
    repo_ids = _resolve_repo_ids(cfg)
    roots = _resolve_roots(cfg, repo_ids)
    return [
        LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=cfg.episodes,
            revision=cfg.revision,
            delta_timestamps=delta_timestamps,
        )
        for repo_id, root in zip(repo_ids, roots, strict=True)
    ]


def _validate_dataset_compatibility(datasets: list[LeRobotDataset]) -> None:
    first = datasets[0]
    first_features = set(first.meta.features)
    for dataset in datasets[1:]:
        if dataset.fps != first.fps:
            raise ValueError(f"All datasets must share fps. Got {first.repo_id}={first.fps}, {dataset.repo_id}={dataset.fps}")
        if set(dataset.meta.features) != first_features:
            raise ValueError(
                "RECAP PI0 multi-dataset training expects identical feature keys. "
                f"{dataset.repo_id} differs from {first.repo_id}."
            )


def _dataset_stats(datasets: list[LeRobotDataset]) -> dict[str, dict[str, Tensor]]:
    if len(datasets) == 1:
        return datasets[0].meta.stats
    return aggregate_stats([dataset.meta.stats for dataset in datasets])


def _build_pi0_config(
    datasets: list[LeRobotDataset],
    cfg: RECAPPI0TrainingConfig,
    device: torch.device,
) -> PI0Config:
    features = dataset_to_policy_features(datasets[0].meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    policy_cfg = PI0Config(
        input_features=input_features,
        output_features=output_features,
        paligemma_variant=cfg.paligemma_variant,
        action_expert_variant=cfg.action_expert_variant,
        dtype=cfg.model_precision,
        image_resolution=(cfg.image_size, cfg.image_size),
        tokenizer_max_length=cfg.tokenizer_max_length,
        freeze_vision_encoder=cfg.freeze_vision_encoder,
        train_expert_only=cfg.train_expert_only,
        gradient_checkpointing=cfg.gradient_checkpointing,
        device=str(device),
    )
    policy_cfg.validate_features()
    return policy_cfg


def _load_value_network(checkpoint_path: str, device: torch.device) -> RECAPPI0ValueNetwork:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config_dict = dict(checkpoint["model_config"])
    model_config_dict["pretrained_path"] = None
    model_config = RECAPPI0ValueNetworkConfig(**model_config_dict)
    model = RECAPPI0ValueNetwork(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def _load_value_training_settings(checkpoint_path: str) -> tuple[int, bool]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config = checkpoint.get("train_config", {})
    model_config = checkpoint.get("model_config", {})
    num_value_bins = int(train_config.get("num_value_bins", model_config.get("num_value_bins", 101)))
    normalize_by_task = bool(train_config.get("normalize_by_task", False))
    del checkpoint
    return num_value_bins, normalize_by_task


@torch.no_grad()
def _compute_advantage_labels(
    datasets: list[LeRobotDataset],
    policy_cfg: PI0Config,
    cfg: RECAPPI0TrainingConfig,
    device: torch.device,
    output_path: Path,
) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int], float]]:
    if cfg.value_network_checkpoint is None:
        raise ValueError("value_network_checkpoint is required when advantage labels need to be computed.")

    num_value_bins, normalize_by_task = _load_value_training_settings(cfg.value_network_checkpoint)
    frame_targets = value_train._build_frame_targets(  # noqa: SLF001
        datasets=datasets,
        num_value_bins=num_value_bins,
        normalize_by_task=normalize_by_task,
    )
    frame_dataset = value_train.RECAPPI0FrameDataset(datasets, frame_targets, num_value_bins=num_value_bins)
    loader = DataLoader(
        frame_dataset,
        batch_size=cfg.vn_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    preprocessor, _ = make_pre_post_processors(policy_cfg=policy_cfg, dataset_stats=_dataset_stats(datasets))
    value_network = _load_value_network(cfg.value_network_checkpoint, device)

    entries: list[dict[str, Any]] = []
    advantages: list[float] = []
    for batch in loader:
        batch = value_train._preprocess_batch(batch, preprocessor)  # noqa: SLF001
        outputs = value_network(batch)
        expected_value = outputs["expected_value"].detach().cpu()
        target_value = batch["target_value"].detach().cpu()
        advantage = target_value - expected_value
        for i in range(advantage.shape[0]):
            adv = float(advantage[i].item())
            dataset_index = int(batch["dataset_index"][i].item())
            abs_index = int(batch["abs_frame_index"][i].item())
            episode_index = int(batch["episode_index"][i].item())
            entries.append(
                {
                    "dataset_index": dataset_index,
                    "repo_id": datasets[dataset_index].repo_id,
                    "index": abs_index,
                    "episode_index": episode_index,
                    "target_value": float(target_value[i].item()),
                    "value": float(expected_value[i].item()),
                    "advantage": adv,
                }
            )
            advantages.append(adv)

    if not entries:
        raise ValueError("No advantage entries were computed.")

    threshold = float(np.percentile(np.asarray(advantages, dtype=np.float64), cfg.advantage_positive_percentile))
    positive_count = 0
    positive_lookup: dict[tuple[int, int], bool] = {}
    advantage_lookup: dict[tuple[int, int], float] = {}
    for entry in entries:
        positive = bool(entry["advantage"] > threshold)
        entry["positive"] = positive
        positive_count += int(positive)
        key = (int(entry["dataset_index"]), int(entry["index"]))
        positive_lookup[key] = positive
        advantage_lookup[key] = float(entry["advantage"])

    payload = {
        "version": 1,
        "label": cfg.positive_label,
        "positive_percentile": cfg.advantage_positive_percentile,
        "threshold": threshold,
        "num_frames": len(entries),
        "num_positive": positive_count,
        "positive_fraction": positive_count / len(entries),
        "datasets": [{"dataset_index": i, "repo_id": dataset.repo_id} for i, dataset in enumerate(datasets)],
        "entries": entries,
    }
    _save_json(output_path, payload)
    logging.info(
        "Saved RECAP advantage labels to %s: frames=%d positive=%d (%.1f%%) threshold=%.6f",
        output_path,
        len(entries),
        positive_count,
        100.0 * positive_count / len(entries),
        threshold,
    )
    return positive_lookup, advantage_lookup


def _load_advantage_labels(path: Path) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int], float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    positive_lookup: dict[tuple[int, int], bool] = {}
    advantage_lookup: dict[tuple[int, int], float] = {}
    for entry in payload["entries"]:
        key = (int(entry["dataset_index"]), int(entry["index"]))
        positive_lookup[key] = bool(entry["positive"])
        advantage_lookup[key] = float(entry["advantage"])
    return positive_lookup, advantage_lookup, payload


_RECAP_METADATA_KEYS = ("dataset_index", "recap_positive", "recap_advantage")


def _preprocess_recap_batch(batch: dict[str, Any], preprocessor) -> dict[str, Any]:
    preserved = {key: batch[key] for key in _RECAP_METADATA_KEYS if key in batch}
    batch = preprocessor(batch)
    device = next((value.device for value in batch.values() if isinstance(value, Tensor)), torch.device("cpu"))
    for key, value in preserved.items():
        if isinstance(value, Tensor):
            value = value.to(device)
        batch[key] = value
    return batch


class RECAPPI0Dataset(Dataset):
    def __init__(
        self,
        datasets: list[LeRobotDataset],
        positive_lookup: dict[tuple[int, int], bool],
        advantage_lookup: dict[tuple[int, int], float],
        cfg: RECAPPI0TrainingConfig,
        train: bool,
    ):
        self.datasets = datasets
        self.positive_lookup = positive_lookup
        self.advantage_lookup = advantage_lookup
        self.cfg = cfg
        self.train = train
        lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = np.cumsum(lengths).tolist()
        self.image_keys = (
            cfg.image_cutout_cameras
            if cfg.image_cutout_cameras is not None
            else list(datasets[0].meta.camera_keys)
        )

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def _map_index(self, idx: int) -> tuple[int, int]:
        dataset_index = bisect_right(self.cumulative_lengths, idx)
        previous = 0 if dataset_index == 0 else self.cumulative_lengths[dataset_index - 1]
        return dataset_index, idx - previous

    def _with_positive_prompt(self, task: str) -> str:
        label = self.cfg.positive_label
        if self.cfg.positive_label_position == "suffix":
            return f"{task.rstrip()}\n{label}"
        return f"{label}\n{task}"

    def _apply_image_cutout(self, item: dict[str, Any]) -> None:
        if not self.train or self.cfg.image_cutout_prob <= 0 or random.random() >= self.cfg.image_cutout_prob:
            return
        keys = [key for key in self.image_keys if key in item]
        if not keys:
            return
        key = random.choice(keys)
        image = item[key]
        if not isinstance(image, Tensor) or image.ndim < 3:
            return

        channels_first = image.shape[0] in (1, 3, 4)
        h_dim, w_dim = (1, 2) if channels_first else (0, 1)
        height, width = int(image.shape[h_dim]), int(image.shape[w_dim])
        if height <= 1 or width <= 1:
            return

        area = height * width
        frac = random.uniform(self.cfg.image_cutout_min_area, self.cfg.image_cutout_max_area)
        cut_area = max(1, int(area * frac))
        aspect = random.uniform(0.5, 2.0)
        cut_h = min(height, max(1, int(math.sqrt(cut_area / aspect))))
        cut_w = min(width, max(1, int(math.sqrt(cut_area * aspect))))
        top = random.randint(0, height - cut_h)
        left = random.randint(0, width - cut_w)

        out = image.clone()
        min_val = float(out.min().item())
        max_val = float(out.max().item())
        if out.dtype.is_floating_point:
            fill = torch.empty_like(out[..., :cut_h, :cut_w] if channels_first else out[:cut_h, :cut_w, ...])
            fill.uniform_(min_val, max_val)
        else:
            fill = torch.randint(
                int(min_val),
                int(max_val) + 1,
                tuple(out[..., :cut_h, :cut_w].shape if channels_first else out[:cut_h, :cut_w, ...].shape),
                dtype=out.dtype,
                device=out.device,
            )
        if channels_first:
            out[:, top : top + cut_h, left : left + cut_w] = fill
        else:
            out[top : top + cut_h, left : left + cut_w, :] = fill
        item[key] = out

    def _apply_state_mask(self, item: dict[str, Any]) -> None:
        if not self.train or self.cfg.state_mask_prob <= 0 or random.random() >= self.cfg.state_mask_prob:
            return
        state = item.get(OBS_STATE)
        if not isinstance(state, Tensor) or state.ndim == 0:
            return
        flat = state.clone().reshape(-1)
        valid_dims = self.cfg.state_mask_dims if self.cfg.state_mask_dims is not None else list(range(flat.numel()))
        valid_dims = [dim for dim in valid_dims if 0 <= dim < flat.numel()]
        if not valid_dims:
            return
        num_dims = min(max(1, self.cfg.state_mask_num_dims), len(valid_dims))
        dims = random.sample(valid_dims, num_dims)
        min_val = float(flat.min().item())
        max_val = float(flat.max().item())
        values = torch.empty(num_dims, dtype=flat.dtype, device=flat.device).uniform_(min_val, max_val)
        flat[torch.tensor(dims, dtype=torch.long, device=flat.device)] = values
        item[OBS_STATE] = flat.reshape_as(state)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dataset_index, local_index = self._map_index(idx)
        item = dict(self.datasets[dataset_index][local_index])
        abs_index = int(item["index"].item() if isinstance(item["index"], Tensor) else item["index"])
        key = (dataset_index, abs_index)
        positive = self.positive_lookup.get(key, False)

        item["dataset_index"] = torch.tensor(dataset_index, dtype=torch.long)
        item["recap_positive"] = torch.tensor(positive, dtype=torch.bool)
        item["recap_advantage"] = torch.tensor(self.advantage_lookup.get(key, 0.0), dtype=torch.float32)
        if positive and (not self.train or random.random() >= self.cfg.positive_label_dropout):
            item["task"] = self._with_positive_prompt(str(item["task"]))

        self._apply_image_cutout(item)
        self._apply_state_mask(item)
        return item


def _split_subset_indices(
    datasets: list[LeRobotDataset],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    by_episode: dict[tuple[int, int], list[int]] = {}
    offset = 0
    for dataset_index, dataset in enumerate(datasets):
        abs_to_rel_idx = {
            int(abs_idx): rel_idx
            for rel_idx, abs_idx in enumerate(dataset.hf_dataset["index"])
        }
        episode_indices = dataset.episodes if dataset.episodes is not None else range(dataset.meta.total_episodes)
        for episode_index in episode_indices:
            ep = dataset.meta.episodes[episode_index]
            frame_indices = []
            for abs_index in range(int(ep["dataset_from_index"]), int(ep["dataset_to_index"])):
                rel_idx = abs_to_rel_idx.get(abs_index)
                if rel_idx is not None:
                    frame_indices.append(offset + rel_idx)
            if frame_indices:
                by_episode[(dataset_index, int(episode_index))] = frame_indices
        offset += len(dataset)

    episode_keys = sorted(by_episode)
    rng = random.Random(seed)
    rng.shuffle(episode_keys)
    val_count = int(round(len(episode_keys) * val_ratio))
    if val_ratio > 0 and val_count == 0 and len(episode_keys) > 1:
        val_count = 1
    val_keys = set(episode_keys[:val_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for key, indices in by_episode.items():
        if key in val_keys:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    if not train_indices:
        raise ValueError("Train split is empty. Reduce val_split_ratio.")
    return train_indices, val_indices


def _apply_backbone_freezing(policy: PI0Policy, cfg: RECAPPI0TrainingConfig) -> None:
    if not cfg.freeze_backbone:
        return
    paligemma = policy.model.paligemma_with_expert.paligemma
    paligemma.eval()
    for param in paligemma.parameters():
        param.requires_grad = False
    if cfg.num_unfrozen_backbone_layers <= 0:
        logging.info("PI0 PaliGemma backbone fully frozen")
        return
    layers = paligemma.model.language_model.layers
    if cfg.num_unfrozen_backbone_layers > len(layers):
        raise ValueError(
            f"num_unfrozen_backbone_layers={cfg.num_unfrozen_backbone_layers} exceeds available layers {len(layers)}"
        )
    for layer in layers[-cfg.num_unfrozen_backbone_layers :]:
        layer.train()
        for param in layer.parameters():
            param.requires_grad = True
    logging.info(
        "PI0 PaliGemma backbone frozen; unfreezing last %d/%d language layers",
        cfg.num_unfrozen_backbone_layers,
        len(layers),
    )


def _restore_freeze_state(policy: PI0Policy, cfg: RECAPPI0TrainingConfig) -> None:
    if not cfg.freeze_backbone:
        return
    paligemma = policy.model.paligemma_with_expert.paligemma
    paligemma.eval()
    if cfg.num_unfrozen_backbone_layers > 0:
        for layer in paligemma.model.language_model.layers[-cfg.num_unfrozen_backbone_layers :]:
            layer.train()


def _build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_policy_checkpoint(
    checkpoint_dir: Path,
    policy: PI0Policy,
    preprocessor,
    postprocessor,
    cfg: RECAPPI0TrainingConfig,
    metrics: dict[str, Any],
    labels_path: Path,
) -> None:
    policy.save_pretrained(checkpoint_dir)
    preprocessor.save_pretrained(checkpoint_dir)
    postprocessor.save_pretrained(checkpoint_dir)
    _save_json(checkpoint_dir / "train_config.json", asdict(cfg))
    _save_json(checkpoint_dir / "recap_train_state.json", {"metrics": metrics, "labels_path": str(labels_path)})


def _run_validation(
    policy: PI0Policy,
    loader: DataLoader,
    preprocessor,
    device: torch.device,
    max_steps: int | None,
) -> dict[str, float]:
    policy.eval()
    total_loss = 0.0
    total_positive = 0
    total_samples = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = _preprocess_recap_batch(batch, preprocessor)
        with torch.no_grad():
            loss, _ = policy.forward(batch)
        batch_size = int(batch[ACTION].shape[0])
        total_loss += float(loss.item()) * batch_size
        if "recap_positive" in batch:
            positives = batch["recap_positive"]
            if isinstance(positives, Tensor):
                total_positive += int(positives.sum().item())
        total_samples += batch_size
    if total_samples == 0:
        return {"loss": float("nan"), "positive_fraction": float("nan")}
    return {"loss": total_loss / total_samples, "positive_fraction": total_positive / total_samples}


@parser.wrap()
def run_recap_pi0_train(cfg: RECAPPI0TrainingConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    _set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "train_config.json", asdict(cfg))

    device = _resolve_device(cfg.device)
    logging.info("Using device: %s", device)

    base_datasets = _load_datasets(cfg)
    _validate_dataset_compatibility(base_datasets)
    policy_cfg = _build_pi0_config(base_datasets, cfg, device)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, base_datasets[0].meta)

    labels_path = Path(cfg.advantage_labels_path) if cfg.advantage_labels_path else output_dir / "recap_advantage_labels.json"
    if labels_path.is_file() and not cfg.recompute_advantage_labels:
        positive_lookup, advantage_lookup, label_payload = _load_advantage_labels(labels_path)
        logging.info(
            "Loaded RECAP advantage labels from %s: frames=%s positive_fraction=%.3f",
            labels_path,
            label_payload.get("num_frames"),
            label_payload.get("positive_fraction", float("nan")),
        )
    else:
        positive_lookup, advantage_lookup = _compute_advantage_labels(
            base_datasets, policy_cfg, cfg, device, labels_path
        )

    train_datasets = _load_datasets(cfg, delta_timestamps=delta_timestamps)
    _validate_dataset_compatibility(train_datasets)
    recap_dataset = RECAPPI0Dataset(train_datasets, positive_lookup, advantage_lookup, cfg, train=True)
    eval_recap_dataset = RECAPPI0Dataset(train_datasets, positive_lookup, advantage_lookup, cfg, train=False)
    train_indices, val_indices = _split_subset_indices(train_datasets, cfg.val_split_ratio, cfg.seed)
    train_dataset = Subset(recap_dataset, train_indices)
    val_dataset = Subset(eval_recap_dataset, val_indices) if val_indices else None

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, dataset_stats=_dataset_stats(train_datasets)
    )
    if cfg.pretrained_path is not None:
        policy = PI0Policy.from_pretrained(
            cfg.pretrained_path,
            config=policy_cfg,
            revision=cfg.pretrained_revision,
            local_files_only=cfg.local_files_only,
            strict=False,
        )
    else:
        policy = PI0Policy(policy_cfg)
    policy.to(device)
    _apply_backbone_freezing(policy, cfg)

    trainable_params = [param for param in policy.parameters() if param.requires_grad]
    logging.info(
        "Trainable parameters: %d / %d",
        sum(p.numel() for p in trainable_params),
        sum(p.numel() for p in policy.parameters()),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )
        if val_dataset is not None
        else None
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    if cfg.max_train_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_steps_per_epoch)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / max(1, cfg.gradient_accumulation_steps))
    scheduler = _build_scheduler(optimizer, optimizer_steps_per_epoch * cfg.epochs, cfg.warmup_ratio)

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    global_step = 0
    grad_accum = max(1, cfg.gradient_accumulation_steps)
    for epoch in range(1, cfg.epochs + 1):
        policy.train()
        _restore_freeze_state(policy, cfg)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_samples = 0
        start = time.perf_counter()
        for step, batch in enumerate(train_loader):
            if cfg.max_train_steps_per_epoch is not None and step >= cfg.max_train_steps_per_epoch:
                break
            batch = _preprocess_recap_batch(batch, preprocessor)
            loss, output_dict = policy.forward(batch)
            (loss / grad_accum).backward()
            if ((step + 1) % grad_accum == 0) or (step + 1 == len(train_loader)):
                if cfg.max_grad_norm > 0:
                    clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            batch_size = int(batch[ACTION].shape[0])
            epoch_loss += float(loss.item()) * batch_size
            epoch_samples += batch_size
            global_step += 1
            if cfg.log_every_n_steps > 0 and ((step + 1) % cfg.log_every_n_steps == 0 or step == 0):
                elapsed = max(time.perf_counter() - start, 1e-9)
                logging.info(
                    "[train epoch %d/%d] step=%d/%d loss=%.5f lr=%.2e samples/s=%.2f",
                    epoch,
                    cfg.epochs,
                    step + 1,
                    len(train_loader),
                    epoch_loss / max(1, epoch_samples),
                    optimizer.param_groups[0]["lr"],
                    epoch_samples / elapsed,
                )

        train_metrics = {"loss": epoch_loss / max(1, epoch_samples)}
        val_metrics = (
            _run_validation(policy, val_loader, preprocessor, device, cfg.max_val_steps_per_epoch)
            if val_loader is not None
            else {"loss": float("nan"), "positive_fraction": float("nan")}
        )
        metrics = {"epoch": epoch, "global_step": global_step, "train": train_metrics, "val": val_metrics}
        history.append(metrics)
        _save_json(output_dir / "metrics.json", {"history": history})
        logging.info(
            "Epoch %d/%d: train_loss=%.5f val_loss=%.5f val_positive_fraction=%.3f",
            epoch,
            cfg.epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["positive_fraction"],
        )

        checkpoint_dir = checkpoints_dir / f"epoch_{epoch:04d}"
        _save_policy_checkpoint(checkpoint_dir, policy, preprocessor, postprocessor, cfg, metrics, labels_path)
        if math.isnan(val_metrics["loss"]) or val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_dir = checkpoints_dir / "best"
            _save_policy_checkpoint(best_dir, policy, preprocessor, postprocessor, cfg, metrics, labels_path)
        last_dir = checkpoints_dir / "last"
        _save_policy_checkpoint(last_dir, policy, preprocessor, postprocessor, cfg, metrics, labels_path)


if __name__ == "__main__":
    run_recap_pi0_train()
