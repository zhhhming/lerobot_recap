"""Batched value-model inference and atomic raw-run writeback."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.value_function.configuration import ValueFunctionConfig
from lerobot.value_function.dataset import (
    RawRunValueContract,
    RawValueFrameDataset,
    ValueAugmentationConfig,
)
from lerobot.value_function.modeling_pi0_value import select_paired_subtask_head
from lerobot.value_function.raw_io import (
    fingerprint_raw_run_columns,
    merge_raw_run_extras,
    read_value_function_metadata,
    update_stage_metadata,
)
from lerobot.value_function.schema import (
    PREDICTION_SOURCE_MODEL,
    TARGET_STAGE,
    VALUE_GLOBAL_REMAINING_FRAMES_PRED,
    VALUE_GLOBAL_REMAINING_NORM_PRED,
    VALUE_INFERENCE_STAGE_PREFIX,
    VALUE_SUBTASK_CONFIDENCE,
    VALUE_SUBTASK_ID_GT,
    VALUE_SUBTASK_ID_PRED,
    VALUE_SUBTASK_ID_PRED_SMOOTH,
    VALUE_SUBTASK_NAME_PRED_SMOOTH,
    VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD,
    VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD,
    VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD,
    VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD,
)
from lerobot.value_function.training import (
    ModelFactory,
    load_value_function_checkpoint,
    resolve_device,
)

ValueMode = Literal["global", "subtask", "both"]
SubtaskInferencePath = Literal["gt_conditioned", "pred_smooth", "both"]


@dataclass(frozen=True)
class ValueInferenceConfig:
    root: str | Path
    checkpoint: str | Path
    mode: ValueMode = "both"
    batch_size: int = 8
    num_workers: int = 0
    device: str = "auto"
    image_keys: tuple[str, ...] | None = None
    subtask_inference_path: SubtaskInferencePath = "both"
    transition_penalty: float = 0.0
    allow_subtask_skip: bool = False
    progress: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"global", "subtask", "both"}:
            raise ValueError(f"Unsupported inference mode: {self.mode!r}")
        if self.subtask_inference_path not in {"gt_conditioned", "pred_smooth", "both"}:
            raise ValueError(
                f"Unsupported subtask inference path: {self.subtask_inference_path!r}"
            )
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if not math.isfinite(self.transition_penalty) or self.transition_penalty < 0:
            raise ValueError("transition_penalty must be finite and non-negative")
        if self.image_keys is not None:
            normalized = tuple(self.image_keys)
            if not normalized or any(not key for key in normalized):
                raise ValueError("image_keys must contain at least one non-empty feature key")
            if len(set(normalized)) != len(normalized):
                raise ValueError("image_keys must not contain duplicates")
            object.__setattr__(self, "image_keys", normalized)


@dataclass(frozen=True)
class _CheckpointContract:
    model: ValueFunctionConfig
    data: Mapping[str, Any]
    sha256: str
    step: int | None
    epoch: int | None


def _needs_global(mode: ValueMode) -> bool:
    return mode in {"global", "both"}


def _needs_subtask(mode: ValueMode) -> bool:
    return mode in {"subtask", "both"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_signature(feature: Mapping[str, Any] | None) -> Any:
    if feature is None:
        return None
    return {
        "dtype": feature.get("dtype"),
        "shape": list(feature.get("shape") or []),
        "names": list(feature.get("names") or []) if feature.get("names") is not None else None,
    }


def _checkpoint_supports(requested: ValueMode, checkpoint_mode: str) -> bool:
    if requested == "both":
        return checkpoint_mode == "both"
    return checkpoint_mode in {requested, "both"}


def _validate_checkpoint_contract(
    config: ValueInferenceConfig,
    checkpoint: _CheckpointContract,
    raw: RawRunValueContract,
    image_keys: Sequence[str],
) -> None:
    model = checkpoint.model
    data = checkpoint.data
    if not _checkpoint_supports(config.mode, model.mode):
        raise ValueError(
            f"Checkpoint mode {model.mode!r} does not support requested mode {config.mode!r}"
        )
    if tuple(image_keys) != tuple(model.image_keys):
        raise ValueError(
            "Inference image_keys must exactly match checkpoint model order: "
            f"{tuple(image_keys)!r} != {tuple(model.image_keys)!r}"
        )
    expected_robot = str(data.get("robot_type", ""))
    if expected_robot and raw.robot_type != expected_robot:
        raise ValueError(
            f"Checkpoint/raw robot_type mismatch: {expected_robot!r} != {raw.robot_type!r}"
        )
    expected_fps = data.get("fps")
    if expected_fps is not None and float(expected_fps) != raw.fps:
        raise ValueError(f"Checkpoint/raw fps mismatch: {expected_fps!r} != {raw.fps!r}")

    checkpoint_images = data.get("image_features") or {}
    for key in image_keys:
        if key not in raw.image_features:
            raise ValueError(f"Raw run is missing checkpoint image feature {key!r}")
        expected = checkpoint_images.get(key)
        actual = _feature_signature(raw.image_features[key])
        if expected is not None and expected != actual:
            raise ValueError(
                f"Checkpoint/raw image schema mismatch for {key!r}: {expected!r} != {actual!r}"
            )

    if model.use_state:
        if raw.state_key != model.state_key or data.get("state_key") != model.state_key:
            raise ValueError(
                "Checkpoint/raw state key mismatch: "
                f"model={model.state_key!r}, data={data.get('state_key')!r}, raw={raw.state_key!r}"
            )
        expected_state = data.get("state_feature")
        actual_state = _feature_signature(raw.state_feature)
        if raw.state_feature is None or (
            expected_state is not None and expected_state != actual_state
        ):
            raise ValueError(
                f"Checkpoint/raw state schema mismatch: {expected_state!r} != {actual_state!r}"
            )
        shape = list((raw.state_feature or {}).get("shape") or [])
        if shape != [model.state_dim]:
            raise ValueError(
                f"Checkpoint/raw state dimension mismatch: {[model.state_dim]!r} != {shape!r}"
            )

    if _needs_global(config.mode):
        if raw.global_num_bins != model.resolved_global_num_bins:
            raise ValueError(
                "Checkpoint/raw global bin mismatch: "
                f"{model.resolved_global_num_bins} != {raw.global_num_bins}"
            )
        expected_scale = data.get("global_scale_frames")
        if expected_scale is not None and float(expected_scale) != raw.global_scale_frames:
            raise ValueError(
                "Checkpoint/raw global scale mismatch: "
                f"{expected_scale!r} != {raw.global_scale_frames!r}"
            )

    if _needs_subtask(config.mode):
        if raw.subtask_num_bins != model.resolved_subtask_num_bins:
            raise ValueError(
                "Checkpoint/raw subtask bin mismatch: "
                f"{model.resolved_subtask_num_bins} != {raw.subtask_num_bins}"
            )
        expected_order = tuple(str(name) for name in data.get("subtask_order") or ())
        if expected_order != raw.subtask_order:
            raise ValueError(
                f"Checkpoint/raw subtask order mismatch: {expected_order!r} != {raw.subtask_order!r}"
            )
        if model.num_subtasks != len(raw.subtask_order):
            raise ValueError(
                "Checkpoint/raw subtask count mismatch: "
                f"{model.num_subtasks!r} != {len(raw.subtask_order)}"
            )
        expected_scales = {
            str(name): float(value)
            for name, value in (data.get("subtask_scale_frames") or {}).items()
        }
        if expected_scales != raw.subtask_scale_frames:
            raise ValueError(
                "Checkpoint/raw subtask scales mismatch: "
                f"{expected_scales!r} != {raw.subtask_scale_frames!r}"
            )

    training_roots = [str(Path(root).expanduser().resolve()) for root in data.get("roots") or []]
    current_root = str(raw.root)
    if current_root in training_roots:
        index = training_roots.index(current_root)
        fingerprints = list(data.get("target_stage_fingerprints") or [])
        if index >= len(fingerprints) or fingerprints[index] != raw.target_stage_fingerprint:
            raise ValueError(
                "Checkpoint target-stage fingerprint does not match its training raw root"
            )


def monotonic_viterbi(
    log_probabilities: np.ndarray,
    *,
    transition_penalty: float = 0.0,
    allow_skip: bool = False,
) -> np.ndarray:
    """Decode a canonical subtask path that starts at 0 and finishes at K-1."""

    scores = np.asarray(log_probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 1:
        raise ValueError("log_probabilities must have shape [frames, subtasks]")
    if not np.isfinite(scores).all():
        raise ValueError("log_probabilities must be finite")
    if not math.isfinite(transition_penalty) or transition_penalty < 0:
        raise ValueError("transition_penalty must be finite and non-negative")
    frame_count, num_subtasks = scores.shape
    if not allow_skip and frame_count < num_subtasks:
        raise ValueError(
            f"Cannot visit {num_subtasks} subtasks without skips in {frame_count} frames"
        )

    negative_infinity = -np.inf
    dynamic = np.full((frame_count, num_subtasks), negative_infinity, dtype=np.float64)
    backpointers = np.full((frame_count, num_subtasks), -1, dtype=np.int32)
    dynamic[0, 0] = scores[0, 0]
    for frame in range(1, frame_count):
        for current in range(num_subtasks):
            candidates: list[tuple[float, int]] = [(dynamic[frame - 1, current], current)]
            first_previous = 0 if allow_skip else current - 1
            for previous in range(max(first_previous, 0), current):
                distance = current - previous
                candidates.append(
                    (
                        dynamic[frame - 1, previous] - transition_penalty * distance,
                        previous,
                    )
                )
            best_score, best_previous = max(candidates, key=lambda item: item[0])
            if np.isfinite(best_score):
                dynamic[frame, current] = best_score + scores[frame, current]
                backpointers[frame, current] = best_previous

    final_state = num_subtasks - 1
    if not np.isfinite(dynamic[-1, final_state]):
        raise ValueError("No valid canonical subtask path reaches the final subtask")
    path = np.empty(frame_count, dtype=np.int32)
    path[-1] = final_state
    for frame in range(frame_count - 1, 0, -1):
        previous = backpointers[frame, path[frame]]
        if previous < 0:
            raise RuntimeError("Viterbi backpointer is incomplete")
        path[frame - 1] = previous
    return path


def _checkpoint_record(
    path: Path,
    payload: Mapping[str, Any],
    model_config: ValueFunctionConfig,
) -> _CheckpointContract:
    data = payload.get("data_contract")
    if not isinstance(data, Mapping):
        raise ValueError("Value checkpoint is missing data_contract")
    return _CheckpointContract(
        model=model_config,
        data=data,
        sha256=_file_sha256(path),
        step=int(payload["step"]) if payload.get("step") is not None else None,
        epoch=int(payload["epoch"]) if payload.get("epoch") is not None else None,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _as_float32(values: np.ndarray) -> pa.Array:
    return pa.array(np.asarray(values, dtype=np.float32).tolist(), type=pa.float32())


def _as_int32(values: np.ndarray) -> pa.Array:
    return pa.array(np.asarray(values, dtype=np.int32).tolist(), type=pa.int32())


def _validated_tensor(
    outputs: Mapping[str, Tensor],
    name: str,
    *,
    batch_size: int,
    ndim: int,
    normalized: bool = False,
) -> Tensor:
    value = outputs.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"Model output {name!r} must be a tensor")
    if value.ndim != ndim or value.shape[0] != batch_size:
        raise ValueError(
            f"Model output {name!r} must have {ndim} dimensions and batch size "
            f"{batch_size}, got {tuple(value.shape)}"
        )
    value = value.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError(f"Model output {name!r} contains non-finite values")
    if normalized and ((value < -1e-6) | (value > 1.0 + 1e-6)).any():
        raise ValueError(f"Model output {name!r} must be normalized to [0, 1]")
    return value.clamp(0.0, 1.0) if normalized else value


def _stage_config(
    config: ValueInferenceConfig,
    *,
    checkpoint_path: Path,
    checkpoint: _CheckpointContract,
    image_keys: Sequence[str],
    stage_mode: str,
) -> dict[str, Any]:
    return {
        "root": str(Path(config.root).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_step": checkpoint.step,
        "checkpoint_epoch": checkpoint.epoch,
        "mode": stage_mode,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "device": config.device,
        "image_keys": list(image_keys),
        "subtask_inference_path": config.subtask_inference_path,
        "transition_penalty": config.transition_penalty,
        "allow_subtask_skip": config.allow_subtask_skip,
        "progress": config.progress,
    }


def infer_value_function(
    config: ValueInferenceConfig,
    *,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    """Run model inference for every raw frame, then atomically merge prediction columns."""

    root = Path(config.root).expanduser().resolve()
    checkpoint_path = Path(config.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Value checkpoint not found: {checkpoint_path}")
    model, payload = load_value_function_checkpoint(
        checkpoint_path, model_factory=model_factory, map_location="cpu"
    )
    model_config = ValueFunctionConfig.from_dict(payload["model_config"])
    checkpoint = _checkpoint_record(checkpoint_path, payload, model_config)
    image_keys = tuple(config.image_keys or model_config.image_keys)

    dataset = RawValueFrameDataset(
        [root],
        mode=config.mode,
        image_keys=image_keys,
        state_key=model_config.state_key,
        use_state=model_config.use_state,
        use_elapsed_aux=False,
        augmentation=ValueAugmentationConfig(enabled=False),
    )
    raw_contract = dataset.contracts[0]
    _validate_checkpoint_contract(config, checkpoint, raw_contract, image_keys)

    device = resolve_device(config.device)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if config.num_workers else None,
    )
    model.to(device)
    model.eval()

    records: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    with torch.inference_mode(), tqdm(
        total=len(dataset),
        desc="Value inference",
        unit="frame",
        dynamic_ncols=True,
        leave=True,
        disable=not config.progress,
    ) as progress:
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            outputs = model(batch)
            batch_size = int(batch["value_frame_index"].shape[0])
            episode_ids = batch["value_episode_index"].detach().cpu().tolist()
            frame_ids = batch["value_frame_index"].detach().cpu().tolist()

            global_values = None
            if _needs_global(config.mode):
                global_values = (
                    _validated_tensor(
                        outputs,
                        "global_remaining_value",
                        batch_size=batch_size,
                        ndim=1,
                        normalized=True,
                    )
                    .cpu()
                    .numpy()
                )

            log_probabilities = remaining_values = gt_ids = raw_ids = confidences = None
            if _needs_subtask(config.mode):
                subtask_logits = _validated_tensor(
                    outputs, "subtask_logits", batch_size=batch_size, ndim=2
                )
                subtask_values = _validated_tensor(
                    outputs,
                    "subtask_remaining_value",
                    batch_size=batch_size,
                    ndim=2,
                    normalized=True,
                )
                expected_subtasks = len(raw_contract.subtask_order)
                if (
                    subtask_logits.shape[1] != expected_subtasks
                    or subtask_values.shape[1] != expected_subtasks
                ):
                    raise ValueError(
                        "Model subtask outputs do not match the raw canonical subtask count: "
                        f"logits={tuple(subtask_logits.shape)}, values={tuple(subtask_values.shape)}, "
                        f"expected={expected_subtasks}"
                    )
                log_probabilities = (
                    F.log_softmax(subtask_logits, dim=-1).cpu().numpy()
                )
                remaining_values = subtask_values.cpu().numpy()
                gt_ids = batch[VALUE_SUBTASK_ID_GT].detach().long().cpu().numpy()
                raw_ids = log_probabilities.argmax(axis=-1).astype(np.int32)
                confidences = np.exp(log_probabilities.max(axis=-1)).astype(np.float32)

            for index in range(batch_size):
                record: dict[str, Any] = {"frame": int(frame_ids[index])}
                if global_values is not None:
                    record["global"] = float(global_values[index])
                if log_probabilities is not None:
                    record.update(
                        {
                            "log_probabilities": log_probabilities[index],
                            "remaining_values": remaining_values[index],
                            "gt_id": int(gt_ids[index]),
                            "raw_id": int(raw_ids[index]),
                            "confidence": float(confidences[index]),
                        }
                    )
                records[int(episode_ids[index])].append(record)
            progress.update(batch_size)

    episode_columns: dict[int, dict[str, pa.Array]] = {}
    for episode in dataset.episodes:
        episode_records = sorted(records[episode.episode_index], key=lambda item: item["frame"])
        observed_frames = [record["frame"] for record in episode_records]
        if observed_frames != list(range(episode.frame_count)):
            raise RuntimeError(
                f"Inference did not produce exactly one ordered prediction per frame in "
                f"episode {episode.episode_index}"
            )
        columns: dict[str, pa.Array] = {}
        if _needs_global(config.mode):
            normalized = np.asarray([record["global"] for record in episode_records], dtype=np.float32)
            frames = normalized * np.float32(raw_contract.global_scale_frames)
            columns[VALUE_GLOBAL_REMAINING_NORM_PRED] = _as_float32(normalized)
            columns[VALUE_GLOBAL_REMAINING_FRAMES_PRED] = _as_float32(frames)

        if _needs_subtask(config.mode):
            log_probabilities = np.stack(
                [record["log_probabilities"] for record in episode_records]
            )
            all_head_values = np.stack(
                [record["remaining_values"] for record in episode_records]
            ).astype(np.float32)
            gt_ids = np.asarray([record["gt_id"] for record in episode_records], dtype=np.int64)
            raw_ids = np.asarray([record["raw_id"] for record in episode_records], dtype=np.int32)
            confidences = np.asarray(
                [record["confidence"] for record in episode_records], dtype=np.float32
            )
            smooth_ids = monotonic_viterbi(
                log_probabilities,
                transition_penalty=config.transition_penalty,
                allow_skip=config.allow_subtask_skip,
            )
            smooth_names = [raw_contract.subtask_order[int(index)] for index in smooth_ids]
            columns[VALUE_SUBTASK_ID_PRED] = _as_int32(raw_ids)
            columns[VALUE_SUBTASK_CONFIDENCE] = _as_float32(confidences)
            columns[VALUE_SUBTASK_ID_PRED_SMOOTH] = _as_int32(smooth_ids)
            columns[VALUE_SUBTASK_NAME_PRED_SMOOTH] = pa.array(
                smooth_names, type=pa.string()
            )

            head_tensor = torch.from_numpy(all_head_values).unsqueeze(-1)
            paths = (
                ("gt_conditioned", "pred_smooth")
                if config.subtask_inference_path == "both"
                else (config.subtask_inference_path,)
            )
            for path in paths:
                id_values = gt_ids if path == "gt_conditioned" else smooth_ids
                selected, output_name = select_paired_subtask_head(
                    head_tensor,
                    {
                        VALUE_SUBTASK_ID_GT: torch.from_numpy(gt_ids),
                        VALUE_SUBTASK_ID_PRED_SMOOTH: torch.from_numpy(smooth_ids),
                    },
                    path,
                )
                normalized = selected.squeeze(-1).numpy().astype(np.float32)
                scales = np.asarray(
                    [
                        raw_contract.subtask_scale_frames[
                            raw_contract.subtask_order[int(index)]
                        ]
                        for index in id_values
                    ],
                    dtype=np.float32,
                )
                frame_values = normalized * scales
                columns[output_name] = _as_float32(normalized)
                frame_name = {
                    VALUE_SUBTASK_REMAINING_NORM_PRED_GT_HEAD: (
                        VALUE_SUBTASK_REMAINING_FRAMES_PRED_GT_HEAD
                    ),
                    VALUE_SUBTASK_REMAINING_NORM_PRED_SMOOTH_HEAD: (
                        VALUE_SUBTASK_REMAINING_FRAMES_PRED_SMOOTH_HEAD
                    ),
                }[output_name]
                columns[frame_name] = _as_float32(frame_values)
        episode_columns[episode.episode_index] = columns

    columns_by_mode: dict[str, list[str]] = {}
    if _needs_global(config.mode):
        columns_by_mode["global"] = [
            VALUE_GLOBAL_REMAINING_NORM_PRED,
            VALUE_GLOBAL_REMAINING_FRAMES_PRED,
        ]
    if _needs_subtask(config.mode):
        columns_by_mode["subtask"] = sorted(
            name
            for name in next(iter(episode_columns.values()))
            if name
            not in {VALUE_GLOBAL_REMAINING_NORM_PRED, VALUE_GLOBAL_REMAINING_FRAMES_PRED}
        )
    columns_written = sorted({name for names in columns_by_mode.values() for name in names})
    merge_raw_run_extras(root, episode_columns)

    existing_metadata = read_value_function_metadata(root)
    all_columns_written = sorted(
        set(existing_metadata.get("columns_written") or ()) | set(columns_written)
    )
    stage_summaries: dict[str, Any] = {}
    for stage_mode, output_columns in columns_by_mode.items():
        input_columns = (
            [VALUE_SUBTASK_ID_GT]
            if stage_mode == "subtask"
            and config.subtask_inference_path in {"gt_conditioned", "both"}
            else []
        )
        input_fingerprint = fingerprint_raw_run_columns(root, input_columns)
        output_fingerprint = fingerprint_raw_run_columns(root, output_columns)
        stage_name = f"{VALUE_INFERENCE_STAGE_PREFIX}.{stage_mode}"
        stage_summary = {
            "stage": stage_name,
            "mode": stage_mode,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint.sha256,
            "checkpoint_step": checkpoint.step,
            "checkpoint_epoch": checkpoint.epoch,
            "prediction_source": PREDICTION_SOURCE_MODEL,
            "synthetic": False,
            "episodes": len(episode_columns),
            "frames": len(dataset),
            "output_columns": output_columns,
        }
        update_stage_metadata(
            root,
            stage_name,
            config=_stage_config(
                config,
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                image_keys=image_keys,
                stage_mode=stage_mode,
            ),
            input_columns=input_columns,
            input_fingerprint=input_fingerprint,
            output_columns=output_columns,
            output_fingerprint=output_fingerprint,
            prediction_source=PREDICTION_SOURCE_MODEL,
            synthetic=False,
            dependencies=[TARGET_STAGE],
            metadata_patch={
                "columns_written": all_columns_written,
                "value_inference": {stage_mode: stage_summary},
            },
        )
        stage_summaries[stage_mode] = stage_summary

    return {
        "root": str(root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint.sha256,
        "mode": config.mode,
        "subtask_inference_path": config.subtask_inference_path,
        "prediction_source": PREDICTION_SOURCE_MODEL,
        "synthetic": False,
        "episodes": len(episode_columns),
        "frames": len(dataset),
        "columns_written": columns_written,
        "stages": stage_summaries,
    }
