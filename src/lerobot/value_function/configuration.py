"""Configuration for the offline PI0/PI0.5 value-function model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal


DEFAULT_VALUE_IMAGE_KEYS = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.third_person",
)


@dataclass
class ValueFunctionConfig:
    """Serializable architecture configuration for the offline value model."""

    mode: Literal["global", "subtask", "both"] = "global"
    backbone_type: Literal["pi0", "pi05", "vision_only"] = "pi0"
    paligemma_variant: Literal["gemma_300m", "gemma_2b"] = "gemma_2b"
    precision: Literal["float32", "bfloat16"] = "float32"
    pretrained_path: str | None = None
    local_files_only: bool = False
    revision: str | None = None

    image_keys: tuple[str, ...] = DEFAULT_VALUE_IMAGE_KEYS
    image_resolution: tuple[int, int] = (224, 224)
    num_vlm_layers: int = 3

    num_bins: int = 256
    global_num_bins: int | None = None
    subtask_num_bins: int | None = None
    num_subtasks: int | None = None
    use_elapsed_aux: bool = False

    use_state: bool = True
    state_key: str = "observation.state"
    state_dim: int = 16
    state_hidden_dim: int = 256
    state_norm_eps: float = 1e-6
    fusion_hidden_dim: int = 512

    head_hidden_dim: int = 256
    head_depth: int = 1
    dropout: float = 0.1

    freeze_vision_encoder: bool = True
    freeze_backbone: bool = True
    num_unfrozen_backbone_layers: int = 0

    remaining_loss_weight: float = 1.0
    subtask_ce_loss_weight: float = 0.2
    elapsed_loss_weight: float = 0.0

    # Milestone 2 deliberately uses visual observations and state only. Keeping this
    # explicit prevents a training caller from silently assuming task text was used.
    use_task_text: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"global", "subtask", "both"}:
            raise ValueError(f"Unsupported value mode: {self.mode!r}")
        if self.backbone_type not in {"pi0", "pi05", "vision_only"}:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type!r}")
        if self.paligemma_variant not in {"gemma_300m", "gemma_2b"}:
            raise ValueError(f"Unsupported paligemma_variant: {self.paligemma_variant!r}")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError(f"Unsupported precision: {self.precision!r}")
        if not self.image_keys or any(not key for key in self.image_keys):
            raise ValueError("image_keys must contain at least one non-empty feature key")
        if len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("image_keys must not contain duplicates")
        if len(self.image_resolution) != 2 or any(size <= 0 for size in self.image_resolution):
            raise ValueError(f"Invalid image_resolution: {self.image_resolution!r}")
        if self.image_resolution[0] != self.image_resolution[1]:
            raise ValueError("PaliGemma value backbones require a square image_resolution")
        if self.num_vlm_layers < 0:
            raise ValueError("num_vlm_layers must be non-negative")
        if self.backbone_type in {"pi0", "pi05"} and self.num_vlm_layers < 1:
            raise ValueError("pi0/pi05 value backbones require num_vlm_layers >= 1")
        if self.backbone_type == "vision_only" and self.num_unfrozen_backbone_layers:
            raise ValueError("num_unfrozen_backbone_layers is not defined for vision_only")

        for name, count in (
            ("num_bins", self.num_bins),
            ("global_num_bins", self.resolved_global_num_bins),
            ("subtask_num_bins", self.resolved_subtask_num_bins),
        ):
            if count < 2:
                raise ValueError(f"{name} must be >= 2, got {count}")
        if self.mode in {"subtask", "both"} and (self.num_subtasks is None or self.num_subtasks < 1):
            raise ValueError("subtask/both mode requires num_subtasks >= 1")
        if self.state_dim < 1 or self.state_hidden_dim < 1:
            raise ValueError("state_dim and state_hidden_dim must be positive")
        if self.use_state:
            if not self.state_key:
                raise ValueError("state_key must be non-empty when use_state=true")
            if self.state_norm_eps <= 0:
                raise ValueError("state_norm_eps must be positive")
        if self.fusion_hidden_dim < 1 or self.head_hidden_dim < 1 or self.head_depth < 0:
            raise ValueError("fusion/head dimensions must be positive and head_depth non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_unfrozen_backbone_layers < 0:
            raise ValueError("num_unfrozen_backbone_layers must be non-negative")
        if self.num_unfrozen_backbone_layers > self.num_vlm_layers:
            raise ValueError("num_unfrozen_backbone_layers cannot exceed num_vlm_layers")
        for name, weight in (
            ("remaining_loss_weight", self.remaining_loss_weight),
            ("subtask_ce_loss_weight", self.subtask_ce_loss_weight),
            ("elapsed_loss_weight", self.elapsed_loss_weight),
        ):
            if weight < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.use_task_text:
            raise ValueError("Milestone 2 value models do not accept task text; set use_task_text=false")

    @property
    def resolved_global_num_bins(self) -> int:
        return self.num_bins if self.global_num_bins is None else self.global_num_bins

    @property
    def resolved_subtask_num_bins(self) -> int:
        return self.num_bins if self.subtask_num_bins is None else self.subtask_num_bins

    @property
    def resolved_pretrained_path(self) -> str:
        if self.pretrained_path:
            return self.pretrained_path
        if self.backbone_type == "pi05":
            return "lerobot/pi05_base"
        return "lerobot/pi0_base"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValueFunctionConfig:
        known_fields = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known_fields)
        if unknown:
            raise ValueError(f"Unknown value model config fields: {unknown}")
        values = dict(payload)
        for key in ("image_keys", "image_resolution"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)
