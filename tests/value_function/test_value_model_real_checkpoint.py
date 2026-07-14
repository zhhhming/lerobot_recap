from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from lerobot.value_function.configuration import ValueFunctionConfig
from lerobot.value_function.modeling_pi0_value import PI0ValueFunctionModel


RUN_REAL_SMOKE = os.environ.get("LEROBOT_RUN_REAL_VALUE_MODEL_SMOKE") == "1"


@pytest.mark.skipif(not RUN_REAL_SMOKE, reason="Set LEROBOT_RUN_REAL_VALUE_MODEL_SMOKE=1")
@pytest.mark.parametrize(
    ("backbone_type", "checkpoint_env", "mode"),
    [
        ("pi0", "LEROBOT_PI0_CHECKPOINT", "global"),
        ("pi05", "LEROBOT_PI05_CHECKPOINT", "subtask"),
        ("vision_only", "LEROBOT_PI0_CHECKPOINT", "global"),
    ],
)
def test_real_policy_checkpoint_selective_load_and_forward(backbone_type, checkpoint_env, mode):
    checkpoint_value = os.environ.get(checkpoint_env)
    if not checkpoint_value:
        pytest.skip(f"Set {checkpoint_env} to a local policy model.safetensors")
    checkpoint = Path(checkpoint_value).expanduser()
    if not checkpoint.is_file():
        pytest.fail(f"Configured real checkpoint does not exist: {checkpoint}")

    config = ValueFunctionConfig(
        mode=mode,
        backbone_type=backbone_type,
        pretrained_path=str(checkpoint),
        image_keys=("observation.images.third_person",),
        num_vlm_layers=1,
        num_bins=8,
        num_subtasks=3 if mode == "subtask" else None,
        use_state=mode == "global",
        state_dim=16,
        state_hidden_dim=8,
        fusion_hidden_dim=32,
        head_hidden_dim=16,
        freeze_backbone=True,
        dropout=0.0,
    )
    model = PI0ValueFunctionModel(config).eval()
    batch = {"observation.images.third_person": torch.rand(1, 3, 224, 224)}
    if config.use_state:
        batch[config.state_key] = torch.zeros(1, config.state_dim)

    with torch.inference_mode():
        outputs = model(batch)

    if mode == "global":
        assert outputs["global_remaining_logits"].shape == (1, 8)
        assert torch.isfinite(outputs["global_remaining_logits"]).all()
    else:
        assert outputs["subtask_logits"].shape == (1, 3)
        assert outputs["subtask_remaining_logits"].shape == (1, 3, 8)
        assert torch.isfinite(outputs["subtask_remaining_logits"]).all()
