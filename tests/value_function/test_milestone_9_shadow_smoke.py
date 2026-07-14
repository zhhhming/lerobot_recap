import os
import shutil
from pathlib import Path

import pytest
import torch

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.scripts.lerobot_build_dataset import BuildDatasetConfig, build_dataset
from lerobot.value_function.advantage import AdvantageConfig, compute_advantage
from lerobot.value_function.advantage_labeling import AdvantageLabelingConfig, export_advantage_labels
from lerobot.value_function.advantage_weights import AdvantageWeightConfig, compute_advantage_weights
from lerobot.value_function.mock_predictions import MockPredictionConfig, generate_mock_predictions
from lerobot.value_function.schema import (
    ADVANTAGE_LABEL_GLOBAL,
    ADVANTAGE_LABEL_SUBTASK,
    ADVANTAGE_LOSS_WEIGHT_GLOBAL,
    ADVANTAGE_LOSS_WEIGHT_SUBTASK,
)
from lerobot.value_function.targets import ValueTargetConfig, prepare_value_targets

SAMPLE_ROOT = Path(
    "/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3"
)


def _small_shadow_run(source: Path, destination: Path, episode_count: int = 2) -> None:
    destination.mkdir()
    for name in ("run_meta.json", "annotation_config.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    episodes = [
        episode
        for episode in sorted(source.glob("ep_*"))
        if (episode / "info.json").is_file()
    ][:episode_count]
    for source_episode in episodes:
        target_episode = destination / source_episode.name
        target_episode.mkdir()
        for name in ("info.json", "frames.parquet", "extras.parquet"):
            shutil.copy2(source_episode / name, target_episode / name)


@pytest.mark.skipif(
    os.environ.get("LEROBOT_RUN_REAL_VALUE_PIPELINE_SMOKE") != "1",
    reason="set LEROBOT_RUN_REAL_VALUE_PIPELINE_SMOKE=1 for the real-data shadow smoke",
)
def test_real_sample_shadow_build_dataset_and_batch_without_touching_source(tmp_path):
    if not SAMPLE_ROOT.is_dir():
        pytest.skip(f"sample raw run not found: {SAMPLE_ROOT}")
    source_extras = SAMPLE_ROOT / "ep_000000" / "extras.parquet"
    source_before = source_extras.read_bytes()

    root = tmp_path / "shadow"
    _small_shadow_run(SAMPLE_ROOT, root)
    prepare_value_targets(ValueTargetConfig(root=root, mode="both"))
    generate_mock_predictions(
        MockPredictionConfig(root=root, mode="both", seed=42, noise_std_frames=3.0)
    )
    for mode in ("global", "subtask"):
        compute_advantage(
            AdvantageConfig(
                root=root,
                value_mode=mode,
                value_source="mock_pred",
                chunk_size=50,
            )
        )
        export_advantage_labels(
            AdvantageLabelingConfig(
                root=root,
                value_mode=mode,
                top_percent=0.8,
                allow_synthetic=True,
            )
        )
        compute_advantage_weights(
            AdvantageWeightConfig(root=root, value_mode=mode, allow_synthetic=True)
        )

    output_root = tmp_path / "dataset"
    dataset = build_dataset(
        BuildDatasetConfig(
            runs=[str(root)],
            output_repo_id="test/milestone9_shadow",
            output_root=str(output_root),
            video=False,
            exclude_features="observation.images.*",
        )
    )
    assert dataset is not None
    assert dataset.num_episodes == 2
    for key in (
        ADVANTAGE_LABEL_GLOBAL,
        ADVANTAGE_LABEL_SUBTASK,
        ADVANTAGE_LOSS_WEIGHT_GLOBAL,
        ADVANTAGE_LOSS_WEIGHT_SUBTASK,
    ):
        assert key in dataset.features
        assert key in dataset[0]

    config = PI0Config(chunk_size=50, n_action_steps=50, device="cpu")
    delta_timestamps = resolve_delta_timestamps(config, dataset.meta)
    reloaded = LeRobotDataset(
        "test/milestone9_shadow",
        root=output_root,
        delta_timestamps=delta_timestamps,
    )
    batch = next(iter(torch.utils.data.DataLoader(reloaded, batch_size=2, shuffle=False)))
    assert batch["action"].shape[:2] == (2, 50)
    converted = transition_to_batch(batch_to_transition(batch))
    assert converted[ADVANTAGE_LOSS_WEIGHT_GLOBAL].shape == (2,)
    assert converted[ADVANTAGE_LOSS_WEIGHT_SUBTASK].shape == (2,)
    assert source_extras.read_bytes() == source_before
