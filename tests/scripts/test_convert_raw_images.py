import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image, JpegImagePlugin

from lerobot.scripts.lerobot_convert_raw_images import (
    MARKER_FILENAME,
    ConvertRawImagesConfig,
    convert_raw_images,
)


def _make_png_raw_run(tmp_path):
    root = tmp_path / "png_raw"
    episode = root / "ep_000000"
    camera = episode / "third_person"
    camera.mkdir(parents=True)
    (root / "run_meta.json").write_text(
        json.dumps(
            {
                "version": 1,
                "fps": 30,
                "task": "test",
                "robot_type": "test_robot",
                "features": {
                    "observation.images.third_person": {
                        "dtype": "video",
                        "shape": [12, 16, 3],
                        "names": ["height", "width", "channels"],
                    }
                },
            }
        )
    )
    (root / "annotations.json").write_text('{"keep": true}')
    (episode / "info.json").write_text(json.dumps({"length": 3, "task": "test"}))
    pq.write_table(pa.table({"frame_index": pa.array(range(3))}), episode / "frames.parquet")
    for frame in range(3):
        pixels = np.full((12, 16, 3), 40 + frame * 20, dtype=np.uint8)
        Image.fromarray(pixels).save(camera / f"{frame:06d}.png")
    return root


def test_convert_raw_images_is_non_destructive_and_writes_jpeg_444(tmp_path):
    source = _make_png_raw_run(tmp_path)
    output = tmp_path / "jpeg_raw"

    result = convert_raw_images(ConvertRawImagesConfig(root=source, output_root=output, workers=2))

    assert result == output.resolve()
    assert (source / "ep_000000/third_person/000000.png").is_file()
    assert not (output / MARKER_FILENAME).exists()
    assert (output / "annotations.json").read_text() == '{"keep": true}'
    metadata = json.loads((output / "run_meta.json").read_text())
    assert metadata["version"] == 2
    assert metadata["image_encoding"] == {
        "format": "jpeg",
        "extension": ".jpg",
        "quality": 95,
        "subsampling": 0,
    }
    with Image.open(output / "ep_000000/third_person/000000.jpg") as image:
        assert image.size == (16, 12)
        assert JpegImagePlugin.get_sampling(image) == 0


def test_convert_raw_images_refuses_existing_unmarked_output(tmp_path):
    source = _make_png_raw_run(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        convert_raw_images(ConvertRawImagesConfig(root=source, output_root=output))


def test_convert_raw_images_resumes_matching_partial_output(tmp_path):
    source = _make_png_raw_run(tmp_path)
    output = tmp_path / "partial"
    output.mkdir()
    (output / MARKER_FILENAME).write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "quality": 95,
                "subsampling": 0,
            }
        )
    )

    convert_raw_images(
        ConvertRawImagesConfig(
            root=source,
            output_root=output,
            workers=2,
            resume=True,
        )
    )

    assert (output / "run_meta.json").is_file()
    assert len(list((output / "ep_000000/third_person").glob("*.jpg"))) == 3
