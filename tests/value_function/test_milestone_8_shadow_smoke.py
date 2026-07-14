import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

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


def _shadow_raw_run(source: Path, destination: Path) -> None:
    destination.mkdir()
    for name in ("run_meta.json", "annotation_config.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    for source_episode in sorted(source.glob("ep_*")):
        if not (source_episode / "info.json").is_file():
            continue
        target_episode = destination / source_episode.name
        target_episode.mkdir()
        for name in ("info.json", "frames.parquet", "extras.parquet"):
            shutil.copy2(source_episode / name, target_episode / name)


@pytest.mark.skipif(
    os.environ.get("LEROBOT_RUN_REAL_VALUE_PIPELINE_SMOKE") != "1",
    reason="set LEROBOT_RUN_REAL_VALUE_PIPELINE_SMOKE=1 for the real-data shadow smoke",
)
def test_real_sample_shadow_pipeline_writes_weights_without_touching_source(tmp_path):
    if not SAMPLE_ROOT.is_dir():
        pytest.skip(f"sample raw run not found: {SAMPLE_ROOT}")
    source_extras = SAMPLE_ROOT / "ep_000000" / "extras.parquet"
    source_before = source_extras.read_bytes()
    root = tmp_path / "shadow"
    _shadow_raw_run(SAMPLE_ROOT, root)

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
        summary = compute_advantage_weights(
            AdvantageWeightConfig(root=root, value_mode=mode, allow_synthetic=True)
        )
        assert summary["total_chunks"] == 53_794
        assert summary["rank_weighted_group_count"] > 0
        assert summary["eligibility"]["experiment_eligible"] is False

    table = pq.read_table(root / "ep_000000" / "extras.parquet")
    for label_column, weight_column in (
        (ADVANTAGE_LABEL_GLOBAL, ADVANTAGE_LOSS_WEIGHT_GLOBAL),
        (ADVANTAGE_LABEL_SUBTASK, ADVANTAGE_LOSS_WEIGHT_SUBTASK),
    ):
        labels = table.column(label_column).to_pylist()
        weights = table.column(weight_column).to_pylist()
        assert len(labels) == len(weights) == table.num_rows
        assert all(weight == 0.0 for label, weight in zip(labels, weights, strict=True) if label == "ignore")
        assert all(
            weight == 1.0
            for label, weight in zip(labels, weights, strict=True)
            if label == "negative"
        )
    assert source_extras.read_bytes() == source_before
