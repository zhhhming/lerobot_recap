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

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from lerobot.policies.pi0.configuration_pi0 import DEFAULT_IMAGE_SIZE
from lerobot.policies.pi0.modeling_pi0 import get_gemma_config, make_att_2d_masks, resize_with_pad_torch
from lerobot.policies.pi_gemma import PaliGemmaForConditionalGenerationWithPiGemma
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)
from lerobot.utils.import_utils import _transformers_available

if _transformers_available:
    from safetensors.torch import load_file
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.utils import cached_file
else:
    CONFIG_MAPPING = None
    cached_file = None
    load_file = None


@dataclass
class RECAPPI0ValueNetworkConfig:
    """Configuration for a PI0/PaliGemma prefix-only RECAP value network."""

    paligemma_variant: str = "gemma_2b"
    precision: Literal["bfloat16", "float32"] = "float32"
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    image_feature_keys: list[str] | None = None
    num_value_bins: int = 101
    v_min: float = -1.0
    v_max: float = 0.0
    num_vlm_layers: int = 8
    freeze_vision_encoder: bool = True
    freeze_backbone: bool = True
    num_unfrozen_backbone_layers: int = 2
    value_head_hidden_dim: int = 512
    value_head_depth: int = 1
    dropout: float = 0.1
    pretrained_path: str | None = None
    local_files_only: bool = False
    revision: str | None = None


def _build_paligemma_config(config: RECAPPI0ValueNetworkConfig):
    if CONFIG_MAPPING is None:
        raise ImportError("transformers is required to instantiate RECAPPI0ValueNetwork.")

    gemma_config = get_gemma_config(config.paligemma_variant)
    image_size = config.image_resolution[0]
    if config.image_resolution[0] != config.image_resolution[1]:
        raise ValueError(f"PaliGemma expects square image resolution, got {config.image_resolution}")

    paligemma_config_hf = CONFIG_MAPPING["paligemma"]()
    paligemma_config_hf._vocab_size = 257152  # noqa: SLF001
    paligemma_config_hf.image_token_index = 257152
    paligemma_config_hf.text_config.hidden_size = gemma_config.width
    paligemma_config_hf.text_config.intermediate_size = gemma_config.mlp_dim
    paligemma_config_hf.text_config.num_attention_heads = gemma_config.num_heads
    paligemma_config_hf.text_config.head_dim = gemma_config.head_dim
    paligemma_config_hf.text_config.num_hidden_layers = gemma_config.depth
    paligemma_config_hf.text_config.num_key_value_heads = gemma_config.num_kv_heads
    paligemma_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
    paligemma_config_hf.text_config.dtype = "float32"
    paligemma_config_hf.text_config.vocab_size = 257152
    paligemma_config_hf.text_config.use_adarms = False
    paligemma_config_hf.text_config.adarms_cond_dim = None
    paligemma_config_hf.vision_config.image_size = image_size
    paligemma_config_hf.vision_config.intermediate_size = 4304
    paligemma_config_hf.vision_config.projection_dim = 2048
    paligemma_config_hf.vision_config.projector_hidden_act = "gelu_fast"
    paligemma_config_hf.vision_config.dtype = "float32"
    return paligemma_config_hf, gemma_config


class RECAPPI0ValueNetwork(nn.Module):
    value_bin_support: Tensor

    def __init__(self, config: RECAPPI0ValueNetworkConfig):
        super().__init__()
        self.config = config
        paligemma_config_hf, gemma_config = _build_paligemma_config(config)

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=paligemma_config_hf)
        self._apply_precision(config.precision)
        self._truncate_vlm_layers(config.num_vlm_layers)

        self.register_buffer(
            "value_bin_support",
            torch.linspace(config.v_min, config.v_max, config.num_value_bins, dtype=torch.float32),
            persistent=True,
        )

        head_layers: list[nn.Module] = [nn.LayerNorm(gemma_config.width)]
        in_dim = gemma_config.width
        for _ in range(config.value_head_depth):
            head_layers.extend(
                [
                    nn.Linear(in_dim, config.value_head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_dim = config.value_head_hidden_dim
        head_layers.append(nn.Linear(in_dim, config.num_value_bins))
        self.value_head = nn.Sequential(*head_layers)

        if config.pretrained_path:
            self.load_pi0_pretrained(config.pretrained_path, config.local_files_only, config.revision)

        self._apply_freezing()

    def _apply_precision(self, precision: str) -> None:
        if precision == "float32":
            self.paligemma.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Invalid precision: {precision}")

        self.paligemma.to(dtype=torch.bfloat16)
        for name, param in self.paligemma.named_parameters():
            if any(selector in name for selector in ("vision_tower", "multi_modal_projector", "layernorm")):
                param.data = param.data.to(dtype=torch.float32)

    def _truncate_vlm_layers(self, num_vlm_layers: int) -> None:
        if num_vlm_layers <= 0:
            return
        language_model = self.paligemma.model.language_model
        total_layers = len(language_model.layers)
        if num_vlm_layers > total_layers:
            raise ValueError(f"num_vlm_layers={num_vlm_layers} exceeds model depth {total_layers}")
        language_model.layers = language_model.layers[:num_vlm_layers]
        language_model.config.num_hidden_layers = num_vlm_layers
        self.paligemma.config.text_config.num_hidden_layers = num_vlm_layers
        logging.info(f"Using first {num_vlm_layers}/{total_layers} PaliGemma text layers")

    def _apply_freezing(self) -> None:
        if self.config.freeze_backbone:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

        if self.config.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False

        if self.config.num_unfrozen_backbone_layers > 0:
            layers = self.paligemma.model.language_model.layers
            if self.config.num_unfrozen_backbone_layers > len(layers):
                raise ValueError(
                    "num_unfrozen_backbone_layers="
                    f"{self.config.num_unfrozen_backbone_layers} exceeds available layers {len(layers)}"
                )
            for layer in layers[-self.config.num_unfrozen_backbone_layers :]:
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True

        for param in self.value_head.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.config.freeze_backbone and self.config.num_unfrozen_backbone_layers == 0:
            self.paligemma.eval()
        return self

    def _resolve_pretrained_file(
        self, pretrained_path: str, local_files_only: bool, revision: str | None
    ) -> str:
        local_path = Path(pretrained_path).expanduser()
        if local_path.is_file():
            return str(local_path)
        if local_path.is_dir():
            safetensors_path = local_path / "model.safetensors"
            if safetensors_path.exists():
                return str(safetensors_path)
            raise FileNotFoundError(f"Could not find model.safetensors under {local_path}")
        if cached_file is None:
            raise ImportError("transformers is required to load pretrained PI0 weights.")
        resolved = cached_file(
            pretrained_path,
            "model.safetensors",
            revision=revision,
            local_files_only=local_files_only,
        )
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve model.safetensors from {pretrained_path}")
        return resolved

    def load_pi0_pretrained(
        self, pretrained_path: str, local_files_only: bool = False, revision: str | None = None
    ) -> None:
        """Load the PaliGemma/VLM branch from a PI0 checkpoint or HF repo."""
        if load_file is None:
            raise ImportError("safetensors is required to load PI0 weights.")

        resolved_file = self._resolve_pretrained_file(pretrained_path, local_files_only, revision)
        logging.info(f"Loading PI0 PaliGemma weights from {resolved_file}")
        full_state_dict = load_file(resolved_file)

        prefix_mappings = (
            ("model.paligemma_with_expert.", ""),
            ("paligemma_with_expert.", ""),
            ("model.", ""),
        )
        state_dict: dict[str, Tensor] = {}
        for key, value in full_state_dict.items():
            mapped_key = key
            for prefix, replacement in prefix_mappings:
                if mapped_key.startswith(prefix):
                    mapped_key = f"{replacement}{mapped_key[len(prefix):]}"
                    break
            if not mapped_key.startswith("paligemma."):
                continue
            state_dict[mapped_key] = value

        lm_head_key = "paligemma.lm_head.weight"
        embed_key = "paligemma.model.language_model.embed_tokens.weight"
        if lm_head_key in state_dict and embed_key not in state_dict:
            state_dict[embed_key] = state_dict[lm_head_key].clone()

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        expected_missing = [
            key
            for key in missing
            if key.startswith(("value_head.", "value_bin_support"))
        ]
        truly_missing = [key for key in missing if key not in expected_missing]
        logging.info(
            "PI0 pretrained VLM load: "
            f"{len(state_dict) - len(unexpected)} tensors loaded, "
            f"{len(expected_missing)} expected value-head misses, "
            f"{len(truly_missing)} unexpected misses, "
            f"{len(unexpected)} unexpected tensors."
        )
        if truly_missing:
            logging.warning(f"Unexpected missing keys: {truly_missing[:20]}")
        if unexpected:
            logging.warning(f"Unexpected keys: {unexpected[:20]}")

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor) -> Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        image_keys = self.config.image_feature_keys
        if not image_keys:
            image_keys = sorted(key for key in batch if key.startswith("observation.images."))
        if not image_keys:
            raise ValueError("No image feature keys configured or present in batch.")

        device = next(self.parameters()).device
        images: list[Tensor] = []
        img_masks: list[Tensor] = []
        for key in image_keys:
            if key not in batch:
                if not images:
                    raise ValueError(f"Missing first configured image key: {key}")
                img = torch.ones_like(images[-1]) * -1
                mask = torch.zeros(images[-1].shape[0], dtype=torch.bool, device=device)
                images.append(img)
                img_masks.append(mask)
                continue

            img = batch[key].to(device=device)
            if img.ndim == 5:
                img = img[:, -1]
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)
            img = img * 2.0 - 1.0
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)

            images.append(img)
            img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))
        return images, img_masks

    def _embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        att_masks: list[bool] = []

        for img, img_mask in zip(images, img_masks, strict=True):
            out_dtype = img.dtype
            image_outputs = self.paligemma.model.get_image_features(img.to(torch.float32))
            img_emb = image_outputs.pooler_output * self.paligemma.config.text_config.hidden_size**0.5
            if img_emb.dtype != out_dtype:
                img_emb = img_emb.to(out_dtype)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [False] * num_img_embs

        lang_emb = self.paligemma.model.language_model.embed_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks.bool())
        att_masks += [False] * lang_emb.shape[1]

        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
        prefix_att_masks = prefix_att_masks[None, :].expand(prefix_pad_masks.shape[0], len(att_masks))
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        device = next(self.parameters()).device
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS].to(device)
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device).bool()

        prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        text_dtype = next(self.paligemma.model.language_model.parameters()).dtype
        prefix_embs = prefix_embs.to(dtype=text_dtype)
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        attention_mask = self._prepare_attention_masks_4d(att_2d_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        position_ids = position_ids.masked_fill(~prefix_pad_masks, 0).long()

        hidden_states = self.paligemma.model.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

        seq_lengths = prefix_pad_masks.sum(dim=1) - 1
        last_hidden = hidden_states[torch.arange(hidden_states.shape[0], device=device), seq_lengths.long()]
        value_logits = self.value_head(last_hidden.float())
        value_probs = torch.softmax(value_logits, dim=-1)
        support = self.value_bin_support.to(device=value_probs.device, dtype=value_probs.dtype)
        expected_value = (value_probs * support.unsqueeze(0)).sum(dim=-1)
        return {
            "value_logits": value_logits,
            "value_probs": value_probs,
            "expected_value": expected_value,
        }

    @torch.no_grad()
    def get_value(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        return self.forward(batch)["expected_value"]

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)
