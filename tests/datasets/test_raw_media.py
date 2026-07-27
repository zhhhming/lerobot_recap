import pytest

from lerobot.datasets.raw_media import (
    RAW_FORMAT_VERSION,
    make_raw_image_encoding,
    raw_frame_image_path,
    raw_image_encoding_from_meta,
    validate_raw_format_version,
)


def test_legacy_raw_run_defaults_to_png(tmp_path):
    encoding = raw_image_encoding_from_meta({"version": 1})

    assert encoding.format == "png"
    assert encoding.extension == ".png"
    assert encoding.mime_type == "image/png"
    assert raw_frame_image_path(tmp_path, "third_person", 12, encoding).name == "000012.png"


def test_jpeg_encoding_metadata_round_trip(tmp_path):
    expected = make_raw_image_encoding("jpeg", jpeg_quality=95, jpeg_subsampling=0)
    actual = raw_image_encoding_from_meta(
        {"version": RAW_FORMAT_VERSION, "image_encoding": expected.to_metadata()}
    )

    assert actual == expected
    assert actual.pillow_save_kwargs() == {
        "format": "JPEG",
        "quality": 95,
        "subsampling": 0,
        "optimize": False,
    }
    assert raw_frame_image_path(tmp_path, "left_wrist", 7, actual).name == "000007.jpg"


def test_raw_media_rejects_invalid_metadata():
    with pytest.raises(ValueError, match="does not match"):
        raw_image_encoding_from_meta(
            {
                "version": RAW_FORMAT_VERSION,
                "image_encoding": {"format": "jpeg", "extension": ".png"},
            }
        )
    with pytest.raises(ValueError, match="supported"):
        validate_raw_format_version({"version": 999}, "fixture")
