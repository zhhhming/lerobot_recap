"""Raw-run datasets for offline value-function training."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2

from lerobot.value_function.raw_io import (
    assert_stage_dependencies_current,
    discover_episodes,
    frame_image_paths,
    get_image_keys,
    read_extras_table,
    read_frames_table,
    read_run_meta,
    read_value_function_metadata,
)
from lerobot.value_function.schema import (
    TARGET_STAGE,
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

ValueMode = Literal["global", "subtask", "both"]
EpisodeKey = tuple[int, int]


@dataclass(frozen=True)
class ValueAugmentationConfig:
    enabled: bool = True
    crop_scale_min: float = 0.9
    crop_ratio_min: float = 0.95
    crop_ratio_max: float = 1.05
    rotation_degrees: float = 3.0
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    gaussian_blur_prob: float = 0.0
    grayscale_prob: float = 0.0
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.crop_scale_min <= 1:
            raise ValueError("crop_scale_min must be in (0, 1]")
        if not 0 < self.crop_ratio_min <= self.crop_ratio_max:
            raise ValueError("crop ratios must be positive and ordered")
        if self.rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative")
        for name in ("brightness", "contrast", "saturation", "noise_std"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("gaussian_blur_prob", "grayscale_prob"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RawRunValueContract:
    root: Path
    robot_type: str
    fps: float
    image_features: dict[str, dict[str, Any]]
    state_key: str
    state_feature: dict[str, Any] | None
    value_mode: str
    all_episodes_successful: bool
    subtask_order: tuple[str, ...]
    global_scale_frames: float | None
    subtask_scale_frames: dict[str, float]
    global_num_bins: int
    subtask_num_bins: int
    elapsed_aux: bool
    target_stage_fingerprint: str
    target_output_columns: tuple[str, ...]


@dataclass(frozen=True)
class FrameReference:
    episode_slot: int
    frame_index: int


@dataclass
class EpisodeValueData:
    root_index: int
    episode_index: int
    image_paths: dict[str, list[Path]]
    state: np.ndarray | None
    targets: dict[str, np.ndarray]
    subtask_progress: np.ndarray
    subtask_scale_frames: np.ndarray

    @property
    def frame_count(self) -> int:
        return len(self.subtask_progress)


def _feature_signature(feature: dict[str, Any] | None) -> Any:
    if feature is None:
        return None
    return {
        "dtype": feature.get("dtype"),
        "shape": list(feature.get("shape") or []),
        "names": list(feature.get("names") or []) if feature.get("names") is not None else None,
    }


def load_raw_run_value_contract(
    root: str | Path,
    *,
    state_key: str,
) -> RawRunValueContract:
    root = Path(root).expanduser().resolve()
    assert_stage_dependencies_current(root, TARGET_STAGE)
    run_meta = read_run_meta(root)
    metadata = read_value_function_metadata(root)
    target_stage = (metadata.get("stages") or {}).get(TARGET_STAGE) or {}
    if metadata.get("all_episodes_successful") is not True:
        raise ValueError(f"Raw run {root} does not declare all_episodes_successful=true")

    features = run_meta.get("features") or {}
    image_features = {key: dict(features[key]) for key in get_image_keys(run_meta)}
    global_scale = metadata.get("global_scale") or {}
    subtask_scale = metadata.get("subtask_scale") or {}
    return RawRunValueContract(
        root=root,
        robot_type=str(run_meta.get("robot_type", "")),
        fps=float(run_meta.get("fps", 0.0)),
        image_features=image_features,
        state_key=state_key,
        state_feature=dict(features[state_key]) if state_key in features else None,
        value_mode=str(metadata.get("value_mode", "")),
        all_episodes_successful=True,
        subtask_order=tuple(str(name) for name in metadata.get("subtask_order") or ()),
        global_scale_frames=(
            float(global_scale["frames"]) if global_scale.get("frames") is not None else None
        ),
        subtask_scale_frames={
            str(name): float(scale)
            for name, scale in (subtask_scale.get("frames_by_subtask") or {}).items()
        },
        global_num_bins=int(metadata.get("global_num_bins", metadata.get("num_bins", 256))),
        subtask_num_bins=int(metadata.get("subtask_num_bins", metadata.get("num_bins", 256))),
        elapsed_aux=bool(metadata.get("elapsed_aux", False)),
        target_stage_fingerprint=str(target_stage.get("stage_fingerprint", "")),
        target_output_columns=tuple(str(name) for name in target_stage.get("output_columns") or ()),
    )


def validate_compatible_value_contracts(
    contracts: Sequence[RawRunValueContract],
    *,
    mode: ValueMode,
    image_keys: Sequence[str],
    use_state: bool,
    use_elapsed_aux: bool,
) -> None:
    if not contracts:
        raise ValueError("At least one raw run root is required")
    first = contracts[0]
    required_columns: set[str] = set()
    if mode in {"global", "both"}:
        required_columns.update(
            {
                VALUE_GLOBAL_REMAINING_FRAMES_GT,
                VALUE_GLOBAL_REMAINING_NORM_GT,
                VALUE_GLOBAL_REMAINING_NORM_GT_IS_CLIPPED,
            }
        )
        if use_elapsed_aux:
            required_columns.add(VALUE_GLOBAL_ELAPSED_NORM_GT)
    if mode in {"subtask", "both"}:
        required_columns.update(
            {
                VALUE_SUBTASK_ID_GT,
                VALUE_SUBTASK_REMAINING_FRAMES_GT,
                VALUE_SUBTASK_REMAINING_NORM_GT,
                VALUE_SUBTASK_REMAINING_NORM_GT_IS_CLIPPED,
            }
        )
        if use_elapsed_aux:
            required_columns.add(VALUE_SUBTASK_ELAPSED_NORM_GT)
    for contract in contracts:
        missing_images = [key for key in image_keys if key not in contract.image_features]
        if missing_images:
            raise ValueError(f"Raw run {contract.root} is missing image features: {missing_images}")
        if use_state and contract.state_feature is None:
            raise ValueError(f"Raw run {contract.root} is missing state feature {contract.state_key!r}")
        inactive = sorted(required_columns - set(contract.target_output_columns))
        if inactive:
            raise ValueError(
                f"Raw run {contract.root} current target stage does not provide columns: {inactive}"
            )
        if mode in {"global", "both"} and contract.global_scale_frames is None:
            raise ValueError(f"Raw run {contract.root} has no global target scale")
        if mode in {"subtask", "both"}:
            if not contract.subtask_order or not contract.subtask_scale_frames:
                raise ValueError(f"Raw run {contract.root} has no canonical subtask target contract")
        if use_elapsed_aux and not contract.elapsed_aux:
            raise ValueError(f"Raw run {contract.root} has no elapsed auxiliary targets")

    fields = (
        ("robot_type", lambda c: c.robot_type),
        ("fps", lambda c: c.fps),
        ("image schema", lambda c: {k: _feature_signature(c.image_features[k]) for k in image_keys}),
        ("state schema", lambda c: _feature_signature(c.state_feature) if use_state else None),
        ("global bin count", lambda c: c.global_num_bins if mode in {"global", "both"} else None),
        ("subtask bin count", lambda c: c.subtask_num_bins if mode in {"subtask", "both"} else None),
        ("global scale", lambda c: c.global_scale_frames if mode in {"global", "both"} else None),
        ("subtask order", lambda c: c.subtask_order if mode in {"subtask", "both"} else None),
        (
            "subtask scales",
            lambda c: c.subtask_scale_frames if mode in {"subtask", "both"} else None,
        ),
    )
    for name, getter in fields:
        expected = getter(first)
        for contract in contracts[1:]:
            actual = getter(contract)
            if actual != expected:
                raise ValueError(
                    f"Incompatible {name} between raw roots {first.root} and {contract.root}: "
                    f"{expected!r} != {actual!r}"
                )


def _column_numpy(table, name: str, *, dtype: Any) -> np.ndarray:
    if name not in table.column_names:
        raise ValueError(f"Missing required value target column {name!r}")
    return np.asarray(table.column(name).to_pylist(), dtype=dtype)


def _state_numpy(frames, state_key: str, expected_dim: int) -> np.ndarray:
    if state_key not in frames.column_names:
        raise ValueError(f"Missing configured state column {state_key!r} in frames.parquet")
    state = np.asarray(frames.column(state_key).to_pylist(), dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != expected_dim:
        raise ValueError(
            f"State column {state_key!r} must have shape [N, {expected_dim}], got {state.shape}"
        )
    if not np.isfinite(state).all():
        raise ValueError(f"State column {state_key!r} contains non-finite values")
    return state


class ValueImageAugmentation:
    """Apply one sampled transform jointly to all camera views of a frame."""

    def __init__(self, config: ValueAugmentationConfig):
        self.config = config
        self._pipelines: dict[tuple[int, int], Any] = {}

    def __call__(self, images: Tensor) -> Tensor:
        if not self.config.enabled:
            return images
        height, width = images.shape[-2:]
        size = (height, width)
        if size not in self._pipelines:
            transforms: list[Any] = [
                v2.RandomResizedCrop(
                    size,
                    scale=(self.config.crop_scale_min, 1.0),
                    ratio=(self.config.crop_ratio_min, self.config.crop_ratio_max),
                    antialias=True,
                ),
                v2.RandomRotation(self.config.rotation_degrees),
                v2.ColorJitter(
                    brightness=self.config.brightness,
                    contrast=self.config.contrast,
                    saturation=self.config.saturation,
                ),
            ]
            if self.config.gaussian_blur_prob > 0:
                transforms.append(
                    v2.RandomApply(
                        [v2.GaussianBlur(kernel_size=3)], p=self.config.gaussian_blur_prob
                    )
                )
            if self.config.grayscale_prob > 0:
                transforms.append(v2.RandomGrayscale(p=self.config.grayscale_prob))
            self._pipelines[size] = v2.Compose(transforms)
        result = self._pipelines[size](images)
        if self.config.noise_std > 0:
            result = result + torch.randn_like(result) * self.config.noise_std
        return result.clamp_(0.0, 1.0)


class RawValueFrameDataset(Dataset):
    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        mode: ValueMode,
        image_keys: Sequence[str],
        state_key: str = "observation.state",
        use_state: bool = True,
        use_elapsed_aux: bool = False,
        augmentation: ValueAugmentationConfig | None = None,
    ):
        self.roots = tuple(Path(root).expanduser().resolve() for root in roots)
        self.mode = mode
        self.image_keys = tuple(image_keys)
        self.state_key = state_key
        self.use_state = use_state
        self.use_elapsed_aux = use_elapsed_aux
        self.augmentation = ValueImageAugmentation(augmentation or ValueAugmentationConfig())
        if not self.image_keys:
            raise ValueError("image_keys must not be empty")

        self.contracts = tuple(
            load_raw_run_value_contract(root, state_key=state_key) for root in self.roots
        )
        validate_compatible_value_contracts(
            self.contracts,
            mode=mode,
            image_keys=self.image_keys,
            use_state=use_state,
            use_elapsed_aux=use_elapsed_aux,
        )
        self.episodes: list[EpisodeValueData] = []
        self.references: list[FrameReference] = []
        self.episode_to_indices: dict[EpisodeKey, list[int]] = {}
        for root_index, contract in enumerate(self.contracts):
            self._load_root(root_index, contract)

    @property
    def state_dim(self) -> int | None:
        feature = self.contracts[0].state_feature
        if not self.use_state or feature is None:
            return None
        shape = list(feature.get("shape") or [])
        if len(shape) != 1 or int(shape[0]) < 1:
            raise ValueError(f"State feature must have one positive dimension, got {shape}")
        return int(shape[0])

    @property
    def subtask_order(self) -> tuple[str, ...]:
        return self.contracts[0].subtask_order

    def _required_targets(self) -> list[tuple[str, Any]]:
        columns: list[tuple[str, Any]] = []
        if self.mode in {"global", "both"}:
            columns.extend(
                [
                    (VALUE_GLOBAL_REMAINING_NORM_GT, np.float32),
                    (VALUE_GLOBAL_REMAINING_FRAMES_GT, np.float32),
                    (VALUE_GLOBAL_REMAINING_NORM_GT_IS_CLIPPED, bool),
                ]
            )
            if self.use_elapsed_aux:
                columns.append((VALUE_GLOBAL_ELAPSED_NORM_GT, np.float32))
        if self.mode in {"subtask", "both"}:
            columns.extend(
                [
                    (VALUE_SUBTASK_ID_GT, np.int64),
                    (VALUE_SUBTASK_REMAINING_FRAMES_GT, np.float32),
                    (VALUE_SUBTASK_REMAINING_NORM_GT, np.float32),
                    (VALUE_SUBTASK_REMAINING_NORM_GT_IS_CLIPPED, bool),
                ]
            )
            if self.use_elapsed_aux:
                columns.append((VALUE_SUBTASK_ELAPSED_NORM_GT, np.float32))
        return columns

    def _load_root(self, root_index: int, contract: RawRunValueContract) -> None:
        expected_state_dim = self.state_dim
        for episode in discover_episodes(contract.root):
            frames = read_frames_table(episode.path)
            extras = read_extras_table(episode.path)
            if extras is None:
                raise FileNotFoundError(f"Missing extras.parquet in {episode.path}")
            if frames.num_rows != extras.num_rows or frames.num_rows != episode.frame_count:
                raise ValueError(f"Frame/extras length mismatch in {episode.path}")
            targets = {
                name: _column_numpy(extras, name, dtype=dtype)
                for name, dtype in self._required_targets()
            }
            state = (
                _state_numpy(frames, self.state_key, int(expected_state_dim))
                if self.use_state
                else None
            )
            progress = (
                _column_numpy(extras, "subtask_progress", dtype=np.float32)
                if "subtask_progress" in extras.column_names
                else np.full(episode.frame_count, np.nan, dtype=np.float32)
            )
            if VALUE_SUBTASK_ID_GT in targets:
                ids = targets[VALUE_SUBTASK_ID_GT]
                if ((ids < 0) | (ids >= len(contract.subtask_order))).any():
                    raise ValueError(f"Invalid GT subtask id in {episode.path}")
                scales = np.asarray(
                    [contract.subtask_scale_frames[contract.subtask_order[int(i)]] for i in ids],
                    dtype=np.float32,
                )
            else:
                scales = np.ones(episode.frame_count, dtype=np.float32)
            paths_by_key = frame_image_paths(episode.path, 0, self.image_keys)
            image_paths = {
                key: [path.parent / f"{frame_index:06d}.png" for frame_index in range(episode.frame_count)]
                for key, path in paths_by_key.items()
            }
            slot = len(self.episodes)
            data = EpisodeValueData(
                root_index=root_index,
                episode_index=episode.index,
                image_paths=image_paths,
                state=state,
                targets=targets,
                subtask_progress=progress,
                subtask_scale_frames=scales,
            )
            self.episodes.append(data)
            key = (root_index, episode.index)
            indices: list[int] = []
            for frame_index in range(episode.frame_count):
                indices.append(len(self.references))
                self.references.append(FrameReference(slot, frame_index))
            self.episode_to_indices[key] = indices

    def __len__(self) -> int:
        return len(self.references)

    @staticmethod
    def _load_image(path: Path) -> Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw frame image: {path}")
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)

    def get_item(self, index: int, *, augment: bool) -> dict[str, Any]:
        reference = self.references[index]
        episode = self.episodes[reference.episode_slot]
        frame_index = reference.frame_index
        loaded_images = [
            self._load_image(episode.image_paths[key][frame_index]) for key in self.image_keys
        ]
        if augment:
            groups: dict[tuple[int, int], list[int]] = {}
            for image_index, image in enumerate(loaded_images):
                groups.setdefault(tuple(image.shape[-2:]), []).append(image_index)
            for indices in groups.values():
                transformed = self.augmentation(
                    torch.stack([loaded_images[image_index] for image_index in indices])
                )
                for offset, image_index in enumerate(indices):
                    loaded_images[image_index] = transformed[offset]
        sample: dict[str, Any] = {
            key: loaded_images[i] for i, key in enumerate(self.image_keys)
        }
        if episode.state is not None:
            sample[self.state_key] = torch.from_numpy(episode.state[frame_index].copy())
        for name, values in episode.targets.items():
            value = values[frame_index]
            if np.issubdtype(values.dtype, np.bool_):
                sample[name] = torch.tensor(bool(value), dtype=torch.bool)
            elif np.issubdtype(values.dtype, np.integer):
                sample[name] = torch.tensor(int(value), dtype=torch.long)
            else:
                sample[name] = torch.tensor(float(value), dtype=torch.float32)
        contract = self.contracts[episode.root_index]
        sample.update(
            {
                "value_root_index": torch.tensor(episode.root_index, dtype=torch.long),
                "value_episode_index": torch.tensor(episode.episode_index, dtype=torch.long),
                "value_frame_index": torch.tensor(frame_index, dtype=torch.long),
                "value_global_scale_frames": torch.tensor(
                    contract.global_scale_frames or 1.0, dtype=torch.float32
                ),
                "value_subtask_scale_frames": torch.tensor(
                    episode.subtask_scale_frames[frame_index], dtype=torch.float32
                ),
                "value_subtask_progress": torch.tensor(
                    episode.subtask_progress[frame_index], dtype=torch.float32
                ),
            }
        )
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_item(index, augment=False)


class ValueFrameSubset(Dataset):
    def __init__(self, dataset: RawValueFrameDataset, indices: Sequence[int], *, augment: bool):
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset.get_item(self.indices[index], augment=self.augment)


def parse_val_episode_keys(values: Sequence[str], *, num_roots: int) -> set[EpisodeKey]:
    parsed: set[EpisodeKey] = set()
    for value in values:
        parts = value.split(":", 1)
        if len(parts) == 1:
            if num_roots != 1:
                raise ValueError(
                    f"Multi-root --val_episodes entries must use ROOT_INDEX:EPISODE_INDEX, got {value!r}"
                )
            root_index, episode_index = 0, int(parts[0])
        else:
            root_index, episode_index = int(parts[0]), int(parts[1])
        if root_index < 0 or root_index >= num_roots or episode_index < 0:
            raise ValueError(f"Invalid validation episode key: {value!r}")
        parsed.add((root_index, episode_index))
    return parsed


def split_episode_indices(
    dataset: RawValueFrameDataset,
    *,
    val_fraction: float = 0.1,
    val_episodes: Iterable[EpisodeKey] | None = None,
    seed: int = 42,
) -> tuple[list[int], list[int], list[EpisodeKey], list[EpisodeKey]]:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    episode_keys = sorted(dataset.episode_to_indices)
    if val_episodes is not None:
        val_keys = set(val_episodes)
        unknown = sorted(val_keys - set(episode_keys))
        if unknown:
            raise ValueError(f"Unknown validation episodes: {unknown}")
    else:
        shuffled = list(episode_keys)
        random.Random(seed).shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_fraction))
        if val_fraction > 0 and val_count == 0 and len(shuffled) > 1:
            val_count = 1
        val_keys = set(shuffled[:val_count])
    train_keys = [key for key in episode_keys if key not in val_keys]
    if not train_keys:
        raise ValueError("Train split is empty; reduce or change validation episodes")
    train_indices = [index for key in train_keys for index in dataset.episode_to_indices[key]]
    ordered_val_keys = [key for key in episode_keys if key in val_keys]
    val_indices = [index for key in ordered_val_keys for index in dataset.episode_to_indices[key]]
    return train_indices, val_indices, train_keys, ordered_val_keys


def compute_state_statistics(
    dataset: RawValueFrameDataset,
    indices: Sequence[int],
) -> tuple[Tensor, Tensor]:
    if not dataset.use_state or dataset.state_dim is None:
        raise ValueError("State statistics require use_state=true")
    if not indices:
        raise ValueError("Cannot compute state statistics from an empty split")
    total = torch.zeros(dataset.state_dim, dtype=torch.float64)
    total_sq = torch.zeros(dataset.state_dim, dtype=torch.float64)
    count = 0
    for index in indices:
        reference = dataset.references[index]
        episode = dataset.episodes[reference.episode_slot]
        state = torch.from_numpy(episode.state[reference.frame_index]).double()
        total += state
        total_sq += state.square()
        count += 1
    mean = total / count
    variance = (total_sq / count - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(1e-6)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Computed non-finite state normalization statistics")
    return mean.float(), std.float()


def training_data_contract(dataset: RawValueFrameDataset) -> dict[str, Any]:
    return {
        "roots": [str(contract.root) for contract in dataset.contracts],
        "robot_type": dataset.contracts[0].robot_type,
        "fps": dataset.contracts[0].fps,
        "image_keys": list(dataset.image_keys),
        "image_features": {
            key: _feature_signature(dataset.contracts[0].image_features[key])
            for key in dataset.image_keys
        },
        "state_key": dataset.state_key if dataset.use_state else None,
        "state_feature": (
            _feature_signature(dataset.contracts[0].state_feature) if dataset.use_state else None
        ),
        "subtask_order": list(dataset.subtask_order),
        "global_scale_frames": dataset.contracts[0].global_scale_frames,
        "subtask_scale_frames": dataset.contracts[0].subtask_scale_frames,
        "target_stage_fingerprints": [
            contract.target_stage_fingerprint for contract in dataset.contracts
        ],
        "target_output_columns": list(dataset.contracts[0].target_output_columns),
    }
