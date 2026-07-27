import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.scripts.lerobot_build_dataset import (
    BuildDatasetConfig,
    _load_extras_schema,
    build_dataset,
)
from lerobot.value_function.raw_io import (
    StalePipelineArtifactError,
    fingerprint_raw_run_columns,
    update_stage_metadata,
)
from lerobot.value_function.schema import (
    ADVANTAGE_GLOBAL_IS_VALID,
    ADVANTAGE_GROUP_ID_GLOBAL,
    ADVANTAGE_LABEL_GLOBAL,
    ADVANTAGE_LOSS_WEIGHT_GLOBAL,
    RAW_FORMAT_VERSION,
    VALUE_GLOBAL_REMAINING_NORM_PRED,
)


class _Tokenizer:
    eos_token_id = 1

    def __init__(self):
        self.last_text = None

    def __call__(self, text, *, max_length, return_tensors, **kwargs):
        self.last_text = text
        texts = [text] if isinstance(text, str) else text
        return {
            "input_ids": torch.ones(len(texts), max_length, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), max_length, dtype=torch.long),
        }


def _write_stage(root, name, *, inputs, outputs, dependencies=()):
    update_stage_metadata(
        root,
        name,
        config={"stage": name},
        input_columns=inputs,
        input_fingerprint=fingerprint_raw_run_columns(root, inputs),
        output_columns=outputs,
        output_fingerprint=fingerprint_raw_run_columns(root, outputs),
        prediction_source="model_pred",
        synthetic=False,
        dependencies=dependencies,
    )


def _make_raw_run(tmp_path, *, with_metadata=True, episode_count=1, jpeg_images=False):
    root = tmp_path / "raw"
    root.mkdir(parents=True)
    features = {
        "action": {
            "dtype": "float32",
            "shape": [2],
            "names": ["a0", "a1"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [2],
            "names": ["s0", "s1"],
        },
    }
    if jpeg_images:
        features["observation.images.third_person"] = {
            "dtype": "video",
            "shape": [12, 16, 3],
            "names": ["height", "width", "channels"],
        }
    run_meta = {
        "version": RAW_FORMAT_VERSION,
        "fps": 10,
        "task": "integration test",
        "robot_type": "test_robot",
        "features": features,
    }
    if jpeg_images:
        run_meta["image_encoding"] = {
            "format": "jpeg",
            "extension": ".jpg",
            "quality": 95,
            "subsampling": 0,
        }
    (root / "run_meta.json").write_text(json.dumps(run_meta))
    for episode_index in range(episode_count):
        length = 6
        episode = root / f"ep_{episode_index:06d}"
        episode.mkdir()
        (episode / "info.json").write_text(json.dumps({"length": length, "task": "integration test"}))
        pq.write_table(
            pa.table(
                {
                    "frame_index": pa.array(range(length), type=pa.int64()),
                    "action": pa.array(
                        [[float(i + episode_index * length), float(i) + 0.25] for i in range(length)],
                        type=pa.list_(pa.float32(), 2),
                    ),
                    "observation.state": pa.array(
                        [[float(i), float(i) + 0.5] for i in range(length)],
                        type=pa.list_(pa.float32(), 2),
                    ),
                }
            ),
            episode / "frames.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    ADVANTAGE_LABEL_GLOBAL: pa.array(
                        ["positive", "negative", "positive", "negative", "ignore", "ignore"],
                        type=pa.string(),
                    ),
                    ADVANTAGE_GROUP_ID_GLOBAL: pa.array(["global:bin:+00001"] * length),
                    ADVANTAGE_LOSS_WEIGHT_GLOBAL: pa.array([2.0, 1.0, 0.5, 1.0, 0.0, 0.0], type=pa.float32()),
                    ADVANTAGE_GLOBAL_IS_VALID: pa.array(
                        [True, True, True, True, False, False], type=pa.bool_()
                    ),
                    VALUE_GLOBAL_REMAINING_NORM_PRED: pa.array(
                        [1.0, 0.8, 0.6, 0.4, 0.2, 0.0], type=pa.float32()
                    ),
                }
            ),
            episode / "extras.parquet",
        )
        if jpeg_images:
            camera = episode / "third_person"
            camera.mkdir()
            for frame in range(length):
                pixels = np.full((12, 16, 3), 30 + frame, dtype=np.uint8)
                Image.fromarray(pixels).save(camera / f"{frame:06d}.jpg", quality=95, subsampling=0)
    if with_metadata:
        _write_stage(
            root,
            "advantage_labeling.global",
            inputs=[],
            outputs=[ADVANTAGE_LABEL_GLOBAL],
        )
        _write_stage(
            root,
            "advantage_weights.global",
            inputs=[ADVANTAGE_LABEL_GLOBAL],
            outputs=[ADVANTAGE_GROUP_ID_GLOBAL, ADVANTAGE_LOSS_WEIGHT_GLOBAL],
            dependencies=["advantage_labeling.global"],
        )
    return root


def _build(root, output_root, *, exclude_features=""):
    return build_dataset(
        BuildDatasetConfig(
            runs=[str(root)],
            output_repo_id="test/value_extras",
            output_root=str(output_root),
            video=False,
            exclude_features=exclude_features,
        )
    )


def test_build_dataset_item_dataloader_pi0_preprocessor_and_action_chunk(tmp_path, monkeypatch):
    root = _make_raw_run(tmp_path)
    output_root = tmp_path / "dataset"
    dataset = _build(root, output_root)
    assert dataset is not None

    item = dataset[0]
    assert item[ADVANTAGE_LABEL_GLOBAL] == "positive"
    assert item[ADVANTAGE_LOSS_WEIGHT_GLOBAL].shape == ()
    assert item[ADVANTAGE_LOSS_WEIGHT_GLOBAL].item() == pytest.approx(2.0)

    config = PI0Config(
        chunk_size=3,
        n_action_steps=3,
        device="cpu",
        use_advantage_conditioning=True,
    )
    config.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    delta_timestamps = resolve_delta_timestamps(config, dataset.meta)
    reloaded = LeRobotDataset(
        "test/value_extras",
        root=output_root,
        delta_timestamps=delta_timestamps,
    )
    batch = next(iter(torch.utils.data.DataLoader(reloaded, batch_size=2, shuffle=False)))
    assert batch["action"].shape == (2, 3, 2)
    assert batch[ADVANTAGE_LOSS_WEIGHT_GLOBAL].shape == (2,)
    assert batch[ADVANTAGE_LABEL_GLOBAL] == ["positive", "negative"]

    # Also cover datasets/backends that retain the declared scalar feature
    # dimension and therefore collate numeric extras as [B, 1].
    batch[ADVANTAGE_LOSS_WEIGHT_GLOBAL] = batch[ADVANTAGE_LOSS_WEIGHT_GLOBAL].unsqueeze(-1)
    batch["advantage_condition_kept"] = torch.tensor([[True], [False]])
    tokenizer = _Tokenizer()
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    preprocessor, _ = make_pi0_pre_post_processors(config, dataset_stats=reloaded.meta.stats)
    processed = preprocessor(batch)

    assert processed["action"].shape == (2, 3, 2)
    assert processed[ADVANTAGE_LABEL_GLOBAL] == ["positive", "negative"]
    assert processed[ADVANTAGE_LOSS_WEIGHT_GLOBAL].shape == (2,)
    assert torch.equal(processed["advantage_condition_kept"], torch.tensor([True, False]))
    assert "Advantage: positive" in processed["task"][0]
    assert "Advantage:" not in processed["task"][1]
    assert tokenizer.last_text == processed["task"]


def test_build_include_exclude_keeps_training_fields_and_drops_debug(tmp_path):
    root = _make_raw_run(tmp_path)
    dataset = _build(
        root,
        tmp_path / "dataset",
        exclude_features="value_*,advantage_global_is_valid",
    )
    assert dataset is not None
    assert ADVANTAGE_LABEL_GLOBAL in dataset.features
    assert ADVANTAGE_LOSS_WEIGHT_GLOBAL in dataset.features
    assert VALUE_GLOBAL_REMAINING_NORM_PRED not in dataset.features
    assert ADVANTAGE_GLOBAL_IS_VALID not in dataset.features


def test_build_dataset_filters_raw_episode_indices(tmp_path):
    root = _make_raw_run(tmp_path, episode_count=3)
    dataset = build_dataset(
        BuildDatasetConfig(
            runs=[str(root)],
            episode_indices=[1],
            output_repo_id="test/value-extras-filtered",
            output_root=str(tmp_path / "filtered-dataset"),
            video=False,
        )
    )

    assert dataset is not None
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 6
    assert dataset[0]["action"][0].item() == pytest.approx(6.0)


def test_build_dataset_rejects_missing_raw_episode_indices(tmp_path):
    root = _make_raw_run(tmp_path, episode_count=2)
    with pytest.raises(ValueError, match=r"not found: \[7\]"):
        build_dataset(
            BuildDatasetConfig(
                runs=[str(root)],
                episode_indices=[7],
                output_repo_id="test/value-extras-missing-episode",
                output_root=str(tmp_path / "missing-episode-dataset"),
                video=False,
            )
        )


def test_build_dataset_reads_jpeg_backed_raw_images(tmp_path):
    root = _make_raw_run(tmp_path, jpeg_images=True)

    dataset = _build(root, tmp_path / "jpeg-dataset")

    assert dataset is not None
    image = dataset[0]["observation.images.third_person"]
    assert image.shape == (3, 12, 16)


def test_build_rejects_missing_and_stale_pipeline_metadata(tmp_path):
    root = _make_raw_run(tmp_path, with_metadata=False)
    with pytest.raises(ValueError, match="value_function_meta.json is missing"):
        _build(root, tmp_path / "missing-meta")

    tracked_root = _make_raw_run(tmp_path / "tracked")
    extras_path = tracked_root / "ep_000000" / "extras.parquet"
    table = pq.read_table(extras_path)
    changed = table.set_column(
        table.schema.get_field_index(ADVANTAGE_LOSS_WEIGHT_GLOBAL),
        ADVANTAGE_LOSS_WEIGHT_GLOBAL,
        pa.array([1.0] * table.num_rows, type=pa.float32()),
    )
    pq.write_table(changed, extras_path)
    with pytest.raises(StalePipelineArtifactError, match="outputs changed"):
        _build(tracked_root, tmp_path / "stale")


def test_extras_schema_requires_matching_types_across_episodes(tmp_path):
    root = _make_raw_run(tmp_path, episode_count=2)
    second_path = root / "ep_000001" / "extras.parquet"
    table = pq.read_table(second_path)
    changed = table.set_column(
        table.schema.get_field_index(ADVANTAGE_LOSS_WEIGHT_GLOBAL),
        ADVANTAGE_LOSS_WEIGHT_GLOBAL,
        pa.array([1.0] * table.num_rows, type=pa.float64()),
    )
    pq.write_table(changed, second_path)

    with pytest.raises(ValueError, match="extras.parquet schema differs"):
        _load_extras_schema([root / "ep_000000", root / "ep_000001"])


def test_extras_schema_maps_bool_and_list_columns(tmp_path):
    episode = tmp_path / "ep_000000"
    episode.mkdir()
    pq.write_table(
        pa.table(
            {
                "valid": pa.array([True], type=pa.bool_()),
                "scores": pa.array([[1.0, 2.0]], type=pa.list_(pa.float32())),
            }
        ),
        episode / "extras.parquet",
    )

    features, _ = _load_extras_schema([episode])
    assert features["valid"] == {"dtype": "bool", "shape": (1,), "names": None}
    assert features["scores"] == {"dtype": "float", "shape": (None,), "names": None}


def test_preprocessor_rejects_non_scalar_advantage_weight_shape():
    from lerobot.processor.converters import batch_to_transition

    with pytest.raises(ValueError, match="exactly one scalar per sample"):
        batch_to_transition(
            {
                "action": torch.zeros(2, 3, 1),
                ADVANTAGE_LOSS_WEIGHT_GLOBAL: torch.ones(2, 2),
            }
        )
