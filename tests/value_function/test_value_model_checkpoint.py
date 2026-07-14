from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.value_function.configuration import ValueFunctionConfig
from lerobot.value_function.modeling_pi0_value import (
    PaliGemmaValueBackbone,
    load_selected_safetensors,
    map_pi0_value_checkpoint_key,
)


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)])
        self.norm = nn.LayerNorm(2)


class TinyPolicyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = nn.Linear(2, 2, bias=False)
        self.multi_modal_projector = nn.Linear(2, 2, bias=True)
        self.language_model = TinyLanguageModel()


def source_key(target_key: str) -> str:
    return f"model.paligemma_with_expert.paligemma.model.{target_key}"


def test_checkpoint_key_mapper_accepts_pi0_and_pi05_policy_layouts():
    suffix = "language_model.layers.2.self_attn.q_proj.weight"
    assert map_pi0_value_checkpoint_key(source_key(suffix)) == suffix
    assert map_pi0_value_checkpoint_key(f"paligemma_with_expert.paligemma.model.{suffix}") == suffix
    assert map_pi0_value_checkpoint_key("model.paligemma_with_expert.gemma_expert.foo") is None
    assert map_pi0_value_checkpoint_key(
        "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    ) is None


def test_selective_safetensors_loader_reads_only_required_backbone_tensors(tmp_path):
    module = TinyPolicyBackbone()
    tensors = {
        source_key(key): torch.full_like(value, index + 1)
        for index, (key, value) in enumerate(module.state_dict().items())
    }
    tensors["model.paligemma_with_expert.gemma_expert.ignored"] = torch.ones(100)
    checkpoint = tmp_path / "model.safetensors"
    save_file(tensors, checkpoint)

    missing, unexpected = load_selected_safetensors(module, checkpoint)

    assert missing == []
    assert unexpected == []
    for index, value in enumerate(module.state_dict().values()):
        torch.testing.assert_close(value, torch.full_like(value, index + 1))


def test_selective_safetensors_loader_rejects_missing_and_wrong_shape(tmp_path):
    module = TinyPolicyBackbone()
    items = list(module.state_dict().items())
    missing_checkpoint = tmp_path / "missing.safetensors"
    save_file({source_key(key): value for key, value in items[:-1]}, missing_checkpoint)
    with pytest.raises(ValueError, match="missing .* required"):
        load_selected_safetensors(module, missing_checkpoint)

    wrong_checkpoint = tmp_path / "wrong.safetensors"
    tensors = {source_key(key): value for key, value in items}
    tensors[source_key(items[0][0])] = torch.ones(3, 3)
    save_file(tensors, wrong_checkpoint)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_selected_safetensors(module, wrong_checkpoint)


def test_freezing_keeps_only_requested_last_vlm_layer_trainable():
    backbone = PaliGemmaValueBackbone.__new__(PaliGemmaValueBackbone)
    nn.Module.__init__(backbone)
    backbone.config = ValueFunctionConfig(
        mode="global",
        freeze_backbone=True,
        freeze_vision_encoder=True,
        num_vlm_layers=2,
        num_unfrozen_backbone_layers=1,
    )
    backbone.vision_tower = nn.Linear(2, 2)
    backbone.multi_modal_projector = nn.Linear(2, 2)
    backbone.language_model = TinyLanguageModel()

    backbone._apply_freezing()

    assert not any(parameter.requires_grad for parameter in backbone.vision_tower.parameters())
    assert not any(parameter.requires_grad for parameter in backbone.multi_modal_projector.parameters())
    assert not any(parameter.requires_grad for parameter in backbone.language_model.layers[0].parameters())
    assert all(parameter.requires_grad for parameter in backbone.language_model.layers[1].parameters())
    assert not any(parameter.requires_grad for parameter in backbone.language_model.norm.parameters())

    backbone.train()
    assert not backbone.vision_tower.training
    assert not backbone.multi_modal_projector.training
    assert not backbone.language_model.training
    assert backbone.language_model.layers[1].training


def test_resolved_pretrained_defaults_follow_backbone_type():
    assert (
        ValueFunctionConfig(mode="global", backbone_type="pi0").resolved_pretrained_path
        == "lerobot/pi0_base"
    )
    assert (
        ValueFunctionConfig(mode="global", backbone_type="pi05").resolved_pretrained_path
        == "lerobot/pi05_base"
    )
    explicit = ValueFunctionConfig(mode="global", pretrained_path="/tmp/local-model")
    assert explicit.resolved_pretrained_path == "/tmp/local-model"
