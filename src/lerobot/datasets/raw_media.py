"""Image storage metadata and paths for editable raw LeRobot runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

RAW_FORMAT_VERSION = 2
SUPPORTED_RAW_FORMAT_VERSIONS = (1, RAW_FORMAT_VERSION)
RAW_IMAGE_ENCODING_KEY = "image_encoding"

RawImageFormat = Literal["png", "jpeg"]


@dataclass(frozen=True)
class RawImageEncoding:
    format: RawImageFormat
    extension: str
    mime_type: str
    quality: int | None = None
    subsampling: int | None = None
    png_compress_level: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "format": self.format,
            "extension": self.extension,
        }
        if self.quality is not None:
            metadata["quality"] = self.quality
        if self.subsampling is not None:
            metadata["subsampling"] = self.subsampling
        if self.png_compress_level is not None:
            metadata["png_compress_level"] = self.png_compress_level
        return metadata

    def pillow_save_kwargs(self) -> dict[str, Any]:
        if self.format == "jpeg":
            return {
                "format": "JPEG",
                "quality": self.quality,
                "subsampling": self.subsampling,
                "optimize": False,
            }
        return {
            "format": "PNG",
            "compress_level": self.png_compress_level,
        }


def make_raw_image_encoding(
    image_format: RawImageFormat,
    *,
    jpeg_quality: int = 95,
    jpeg_subsampling: int = 0,
    png_compress_level: int = 3,
) -> RawImageEncoding:
    if image_format == "jpeg":
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if jpeg_subsampling not in (0, 1, 2):
            raise ValueError("jpeg_subsampling must be 0 (4:4:4), 1 (4:2:2), or 2 (4:2:0)")
        return RawImageEncoding(
            format="jpeg",
            extension=".jpg",
            mime_type="image/jpeg",
            quality=jpeg_quality,
            subsampling=jpeg_subsampling,
        )
    if image_format == "png":
        if not 0 <= png_compress_level <= 9:
            raise ValueError("png_compress_level must be in [0, 9]")
        return RawImageEncoding(
            format="png",
            extension=".png",
            mime_type="image/png",
            png_compress_level=png_compress_level,
        )
    raise ValueError(f"Unsupported raw image format: {image_format!r}")


def raw_image_encoding_from_meta(meta: Mapping[str, Any]) -> RawImageEncoding:
    """Read image encoding metadata, treating legacy v1 runs as PNG."""
    payload = meta.get(RAW_IMAGE_ENCODING_KEY)
    if payload is None:
        return make_raw_image_encoding("png")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{RAW_IMAGE_ENCODING_KEY} must be a mapping")

    image_format = str(payload.get("format", "")).lower()
    if image_format == "jpg":
        image_format = "jpeg"
    encoding = make_raw_image_encoding(
        image_format,  # type: ignore[arg-type]
        jpeg_quality=int(payload.get("quality", 95)),
        jpeg_subsampling=int(payload.get("subsampling", 0)),
        png_compress_level=int(payload.get("png_compress_level", 3)),
    )
    declared_extension = payload.get("extension")
    if declared_extension is not None and str(declared_extension).lower() != encoding.extension:
        raise ValueError(f"Raw image extension {declared_extension!r} does not match format {image_format!r}")
    return encoding


def validate_raw_format_version(meta: Mapping[str, Any], location: str | Path) -> int:
    version = int(meta.get("version", 1))
    if version not in SUPPORTED_RAW_FORMAT_VERSIONS:
        raise ValueError(
            f"Raw format version mismatch in '{location}': supported "
            f"{SUPPORTED_RAW_FORMAT_VERSIONS}, got {version}."
        )
    return version


def camera_subdir_name(image_key: str) -> str:
    return image_key.split(".")[-1]


def raw_frame_image_path(
    episode_dir: str | Path,
    camera: str,
    frame_index: int,
    encoding: RawImageEncoding,
) -> Path:
    return Path(episode_dir) / camera / f"{frame_index:06d}{encoding.extension}"
