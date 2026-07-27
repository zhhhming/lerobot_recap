import json

import numpy as np
from PIL import Image, JpegImagePlugin

from lerobot.datasets.raw_media import make_raw_image_encoding
from lerobot.scripts.lerobot_raw_record import RawEpisodeRecorder, RawRun

FEATURES = {
    "action": {"dtype": "float32", "shape": (1,), "names": ["motor.pos"]},
    "observation.images.third_person": {
        "dtype": "video",
        "shape": (12, 16, 3),
        "names": ["height", "width", "channels"],
    },
}


def test_raw_episode_recorder_writes_jpeg_95_444(tmp_path):
    run = RawRun.create(
        root=tmp_path / "raw",
        features=FEATURES,
        fps=30,
        task="test",
        robot_type="test_robot",
        run_config={},
        image_encoding=make_raw_image_encoding("jpeg", jpeg_quality=95, jpeg_subsampling=0),
    )
    recorder = RawEpisodeRecorder(
        run=run,
        display_data=False,
        display_compressed_images=False,
        image_writer=None,
    )

    recorder.add(
        {"third_person": np.full((12, 16, 3), 128, dtype=np.uint8)},
        {"motor.pos": 0.5},
        source="teleop",
    )
    assert recorder.save() == 0

    image_path = run.root / "ep_000000/third_person/000000.jpg"
    assert image_path.is_file()
    with Image.open(image_path) as image:
        assert image.size == (16, 12)
        assert image.format == "JPEG"
        assert JpegImagePlugin.get_sampling(image) == 0
    metadata = json.loads((run.root / "run_meta.json").read_text())
    assert metadata["image_encoding"]["quality"] == 95
    assert metadata["image_encoding"]["subsampling"] == 0


def test_resume_legacy_raw_run_keeps_png_encoding(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "run_meta.json").write_text(
        json.dumps(
            {
                "version": 1,
                "fps": 30,
                "task": "test",
                "robot_type": "test_robot",
                "features": FEATURES,
            }
        )
    )

    run = RawRun.resume(root)

    assert run.image_encoding.format == "png"
    assert run.image_encoding.extension == ".png"
