import os
import shutil
from pathlib import Path

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.scripts.lerobot_build_dataset import BuildDatasetConfig, build_dataset
from lerobot.utils.advantage_weights import AdvantageWeights, sample_advantage_condition_mask
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


class _Tokenizer:
    eos_token_id = 1

    def __call__(self, text, *, max_length, return_tensors, **kwargs):
        texts = [text] if isinstance(text, str) else text
        return {
            "input_ids": torch.ones(len(texts), max_length, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), max_length, dtype=torch.long),
        }


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


def _build_shadow_dataset(root: Path, output_root: Path):
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
    return build_dataset(
        BuildDatasetConfig(
            runs=[str(root)],
            output_repo_id="test/milestone11_shadow",
            output_root=str(output_root),
            video=False,
            exclude_features="observation.images.*",
        )
    )


def _policy_config(config_cls, *, label_key, weight_key, state_shape, action_shape):
    config = config_cls(
        chunk_size=50,
        n_action_steps=50,
        device="cpu",
        use_advantage_conditioning=True,
        advantage_label_key=label_key,
        advantage_loss_weight_key=weight_key,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape)
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape)}
    return config


@pytest.mark.skipif(
    os.environ.get("LEROBOT_RUN_REAL_VALUE_PIPELINE_SMOKE") != "1",
    reason="set LEROBOT_RUN_REAL_VALUE_PIPELINE_SMOKE=1 for the real-data shadow smoke",
)
def test_real_sample_shadow_batch_dropout_and_weights_without_touching_source(
    tmp_path,
    monkeypatch,
):
    if not SAMPLE_ROOT.is_dir():
        pytest.skip(f"sample raw run not found: {SAMPLE_ROOT}")
    source_extras = SAMPLE_ROOT / "ep_000000" / "extras.parquet"
    source_before = source_extras.read_bytes()

    root = tmp_path / "shadow"
    _small_shadow_run(SAMPLE_ROOT, root)
    output_root = tmp_path / "dataset"
    dataset = _build_shadow_dataset(root, output_root)
    assert dataset is not None

    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    for config_cls, processor_factory in (
        (PI0Config, make_pi0_pre_post_processors),
        (PI05Config, make_pi05_pre_post_processors),
    ):
        for label_key, weight_key in (
            (ADVANTAGE_LABEL_GLOBAL, ADVANTAGE_LOSS_WEIGHT_GLOBAL),
            (ADVANTAGE_LABEL_SUBTASK, ADVANTAGE_LOSS_WEIGHT_SUBTASK),
        ):
            config = _policy_config(
                config_cls,
                label_key=label_key,
                weight_key=weight_key,
                state_shape=tuple(dataset.features["observation.state"]["shape"]),
                action_shape=tuple(dataset.features["action"]["shape"]),
            )
            delta_timestamps = resolve_delta_timestamps(config, dataset.meta)
            reloaded = LeRobotDataset(
                "test/milestone11_shadow",
                root=output_root,
                delta_timestamps=delta_timestamps,
            )
            batch = next(
                iter(torch.utils.data.DataLoader(reloaded, batch_size=8, shuffle=False))
            )
            assert batch[weight_key].shape == (8,)

            retained = sample_advantage_condition_mask(
                batch, label_key=label_key, dropout_prob=0.0
            )
            retained_weights, _ = AdvantageWeights(
                loss_weight_key=weight_key, label_key=label_key
            ).compute_batch_weights(retained)
            expected_retained = batch[weight_key].float().clone()
            for index, label in enumerate(batch[label_key]):
                if label == "ignore":
                    expected_retained[index] = 0.0
            assert torch.allclose(retained_weights, expected_retained)

            dropped = sample_advantage_condition_mask(
                batch, label_key=label_key, dropout_prob=1.0
            )
            preprocessor, _ = processor_factory(config, dataset_stats=reloaded.meta.stats)
            processed = preprocessor(dropped)
            assert not processed["advantage_condition_kept"].any()
            assert all("Advantage:" not in task for task in processed["task"])

            effective, stats = AdvantageWeights(
                loss_weight_key=weight_key, label_key=label_key
            ).compute_batch_weights(processed)
            expected = torch.tensor(
                [0.0 if label == "ignore" else 1.0 for label in batch[label_key]]
            )
            assert torch.equal(effective.cpu(), expected)
            assert stats["num_condition_dropped"] == sum(
                label != "ignore" for label in batch[label_key]
            )
    assert source_extras.read_bytes() == source_before
