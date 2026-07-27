"""Raw-run IO helpers for value-function pipeline artifacts.

The raw dataset integration point is per-episode ``extras.parquet``. These
helpers centralize the validation and merge behavior so later pipeline stages
can add columns without dropping subtask annotations or other sidecar fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from lerobot.datasets.raw_media import (
    RawImageEncoding,
    camera_subdir_name as _camera_subdir_name,
    make_raw_image_encoding,
    raw_frame_image_path,
    validate_raw_format_version,
)
from lerobot.value_function.schema import (
    EPISODE_DIR_PREFIX,
    EXTRAS_FILENAME,
    FRAMES_FILENAME,
    PIPELINE_SCHEMA_VERSION,
    RUN_META_FILENAME,
    VALUE_FUNCTION_META_FILENAME,
)

EPISODE_DIR_RE = re.compile(rf"^{re.escape(EPISODE_DIR_PREFIX)}(\d+)$")


class StalePipelineArtifactError(RuntimeError):
    """Raised when a pipeline stage depends on an outdated upstream stage."""


@dataclass(frozen=True)
class RawEpisode:
    index: int
    path: Path
    frame_count: int


def read_run_meta(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    meta_path = root / RUN_META_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {RUN_META_FILENAME} in {root}")
    with open(meta_path) as f:
        meta = json.load(f)
    validate_raw_format_version(meta, root)
    return meta


def read_frames_table(episode_dir: str | Path) -> pa.Table:
    episode_dir = Path(episode_dir).expanduser().resolve()
    frames_path = episode_dir / FRAMES_FILENAME
    if not frames_path.is_file():
        raise FileNotFoundError(f"Missing {frames_path}")
    return pq.read_table(frames_path)


def get_frame_count(episode_dir: str | Path) -> int:
    return read_frames_table(episode_dir).num_rows


def discover_episodes(root: str | Path) -> list[RawEpisode]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Raw run directory not found: {root}")
    read_run_meta(root)

    episodes: list[RawEpisode] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "info.json").is_file():
            continue
        match = EPISODE_DIR_RE.match(path.name)
        if match is None:
            continue
        episodes.append(RawEpisode(index=int(match.group(1)), path=path, frame_count=get_frame_count(path)))
    if not episodes:
        raise ValueError(f"No raw episodes found in {root}")
    episodes.sort(key=lambda ep: ep.index)
    return episodes


def get_image_keys(run_meta: Mapping[str, Any]) -> list[str]:
    features = run_meta.get("features", {})
    return [key for key, feature in features.items() if feature.get("dtype") in ("image", "video")]


def camera_subdir_name(image_key: str) -> str:
    return _camera_subdir_name(image_key)


def frame_image_paths(
    episode_dir: str | Path,
    frame_index: int,
    image_keys: Sequence[str],
    image_encoding: RawImageEncoding | None = None,
) -> dict[str, Path]:
    episode_dir = Path(episode_dir).expanduser().resolve()
    encoding = image_encoding or make_raw_image_encoding("png")
    return {
        image_key: raw_frame_image_path(
            episode_dir,
            camera_subdir_name(image_key),
            frame_index,
            encoding,
        )
        for image_key in image_keys
    }


def read_extras_table(episode_dir: str | Path) -> pa.Table | None:
    episode_dir = Path(episode_dir).expanduser().resolve()
    extras_path = episode_dir / EXTRAS_FILENAME
    if not extras_path.is_file():
        return None
    return pq.read_table(extras_path)


def _as_array(name: str, values: Sequence[Any] | pa.Array | pa.ChunkedArray) -> pa.Array:
    if isinstance(values, pa.ChunkedArray):
        return values.combine_chunks()
    if isinstance(values, pa.Array):
        return values
    try:
        return pa.array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise ValueError(f"Could not convert column '{name}' to a pyarrow array") from exc


def _table_columns(table: pa.Table) -> dict[str, pa.Array]:
    return {name: table.column(name).combine_chunks() for name in table.column_names}


def _prepare_episode_extras_table(
    episode_dir: str | Path,
    columns: Mapping[str, Sequence[Any] | pa.Array | pa.ChunkedArray],
) -> pa.Table:
    if not columns:
        raise ValueError("No columns provided for extras merge")

    episode_dir = Path(episode_dir).expanduser().resolve()
    frame_count = get_frame_count(episode_dir)
    extras_path = episode_dir / EXTRAS_FILENAME

    existing_table = read_extras_table(episode_dir)
    existing_columns: dict[str, pa.Array] = {}
    ordered_names: list[str] = []
    if existing_table is not None:
        if existing_table.num_rows != frame_count:
            raise ValueError(
                f"{extras_path} length ({existing_table.num_rows}) does not match "
                f"{FRAMES_FILENAME} length ({frame_count})"
            )
        existing_columns = _table_columns(existing_table)
        ordered_names = list(existing_table.column_names)

    merged_columns = dict(existing_columns)
    for name, values in columns.items():
        array = _as_array(name, values)
        if len(array) != frame_count:
            raise ValueError(
                f"Column '{name}' length ({len(array)}) does not match {FRAMES_FILENAME} "
                f"length ({frame_count}) in {episode_dir}"
            )
        if name not in merged_columns:
            ordered_names.append(name)
        merged_columns[name] = array

    arrays = [merged_columns[name] for name in ordered_names]
    return pa.Table.from_arrays(arrays, names=ordered_names)


def _fsync_directory(path: Path) -> None:
    """Persist directory entry changes after an atomic replace."""

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _new_sibling_temp_path(destination: Path, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=suffix,
    )
    os.close(fd)
    return Path(name)


def _write_parquet_temp(table: pa.Table, destination: Path) -> Path:
    temp_path = _new_sibling_temp_path(destination, ".tmp")
    try:
        pq.write_table(table, temp_path)
        with open(temp_path, "rb") as f:
            os.fsync(f.fileno())
        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_parquet(table: pa.Table, destination: Path) -> None:
    temp_path = _write_parquet_temp(table, destination)
    try:
        os.replace(temp_path, destination)
        _fsync_directory(destination.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def merge_episode_extras(
    episode_dir: str | Path,
    columns: Mapping[str, Sequence[Any] | pa.Array | pa.ChunkedArray],
) -> pa.Table:
    """Merge columns into one episode's ``extras.parquet``.

    Existing columns are preserved unless a new column has the same name, in
    which case it replaces the old value in the original column position.
    """

    episode_dir = Path(episode_dir).expanduser().resolve()
    extras_path = episode_dir / EXTRAS_FILENAME
    table = _prepare_episode_extras_table(episode_dir, columns)
    _atomic_write_parquet(table, extras_path)
    return table


def merge_raw_run_extras(
    root: str | Path,
    episode_columns: Mapping[int, Mapping[str, Sequence[Any] | pa.Array | pa.ChunkedArray]],
) -> dict[int, Path]:
    """Merge extras for every episode in a run and enforce final schema parity."""

    episodes = discover_episodes(root)
    episode_indices = {ep.index for ep in episodes}
    provided_indices = set(episode_columns)
    if episode_indices != provided_indices:
        missing = sorted(episode_indices - provided_indices)
        extra = sorted(provided_indices - episode_indices)
        raise ValueError(
            f"episode_columns must contain exactly one entry per episode: missing={missing}, extra={extra}"
        )

    existing_schemas = []
    for ep in episodes:
        table = read_extras_table(ep.path)
        if table is not None:
            if table.num_rows != ep.frame_count:
                raise ValueError(
                    f"{ep.path / EXTRAS_FILENAME} length ({table.num_rows}) does not match "
                    f"{FRAMES_FILENAME} length ({ep.frame_count})"
                )
            existing_schemas.append((ep.path, table.schema))
    if existing_schemas:
        first_path, first_schema = existing_schemas[0]
        for path, schema in existing_schemas[1:]:
            if schema != first_schema:
                raise ValueError(
                    f"Existing extras schema differs between {first_path} and {path}: "
                    f"{first_schema} vs {schema}"
                )
        if len(existing_schemas) != len(episodes):
            missing = [str(ep.path) for ep in episodes if read_extras_table(ep.path) is None]
            raise ValueError(
                "Some episodes have extras.parquet and others do not; refusing to create "
                f"inconsistent merged extras. Missing in: {missing[:5]}"
            )

    prepared_tables: dict[int, pa.Table] = {}
    first_final_schema: pa.Schema | None = None
    first_final_path: Path | None = None
    for ep in episodes:
        table = _prepare_episode_extras_table(ep.path, episode_columns[ep.index])
        if first_final_schema is None:
            first_final_schema = table.schema
            first_final_path = ep.path
        elif table.schema != first_final_schema:
            raise ValueError(
                f"Final extras schema differs between {first_final_path} and {ep.path}: "
                f"{first_final_schema} vs {table.schema}"
            )
        prepared_tables[ep.index] = table

    # Staging every output before the first replace is the writeability preflight.
    # A caught commit failure is rolled back across the complete run.
    staged: dict[int, Path] = {}
    backups: dict[int, Path | None] = {}
    replaced: list[int] = []
    destinations = {ep.index: ep.path / EXTRAS_FILENAME for ep in episodes}
    try:
        for ep in episodes:
            staged[ep.index] = _write_parquet_temp(prepared_tables[ep.index], destinations[ep.index])

        for ep in episodes:
            destination = destinations[ep.index]
            backup: Path | None = None
            if destination.exists():
                backup = _new_sibling_temp_path(destination, ".bak")
                backups[ep.index] = backup
                shutil.copy2(destination, backup)
                with open(backup, "rb") as f:
                    os.fsync(f.fileno())
            else:
                backups[ep.index] = None
            os.replace(staged[ep.index], destination)
            replaced.append(ep.index)

        for directory in {path.parent for path in destinations.values()}:
            _fsync_directory(directory)
    except BaseException as commit_error:
        rollback_errors: list[str] = []
        for episode_index in reversed(replaced):
            destination = destinations[episode_index]
            backup = backups.get(episode_index)
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except BaseException as rollback_error:  # pragma: no cover
                rollback_errors.append(f"episode {episode_index}: {rollback_error}")
        for directory in {path.parent for path in destinations.values()}:
            try:
                _fsync_directory(directory)
            except OSError:
                pass
        if rollback_errors:
            raise RuntimeError(
                "extras commit failed and rollback was incomplete: " + "; ".join(rollback_errors)
            ) from commit_error
        raise
    finally:
        for path in [*staged.values(), *(path for path in backups.values() if path is not None)]:
            path.unlink(missing_ok=True)

    return destinations


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def normalize_stage_config(value: Any) -> Any:
    """Return a stable, JSON-compatible representation of stage configuration."""

    if is_dataclass(value) and not isinstance(value, type):
        return normalize_stage_config(asdict(value))
    if isinstance(value, Enum):
        return normalize_stage_config(value.value)
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise TypeError(f"Stage config mapping keys must be strings, got {type(key).__name__}")
            normalized[key] = normalize_stage_config(value[key])
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [normalize_stage_config(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        # Reject NaN/Inf through the same strict encoder used for fingerprints.
        json.dumps(value, allow_nan=False)
        return value
    raise TypeError(f"Unsupported stage config value of type {type(value).__name__}")


def fingerprint_payload(value: Any) -> str:
    normalized = normalize_stage_config(value)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_raw_run_columns(root: str | Path, columns: Sequence[str]) -> str:
    """Hash selected extras columns plus episode/frame structure in canonical order."""

    column_names = list(columns)
    if len(column_names) != len(set(column_names)):
        raise ValueError(f"Duplicate fingerprint columns: {column_names}")
    digest = hashlib.sha256()
    digest.update(f"value-pipeline-input-v1\n{column_names!r}\n".encode())
    for episode in discover_episodes(root):
        digest.update(f"episode={episode.index};frames={episode.frame_count}\n".encode())
        if not column_names:
            continue
        extras = read_extras_table(episode.path)
        if extras is None:
            raise FileNotFoundError(f"Missing {EXTRAS_FILENAME} in {episode.path}")
        missing = [name for name in column_names if name not in extras.column_names]
        if missing:
            raise ValueError(f"Missing fingerprint columns in {episode.path}: {missing}")
        selected = extras.select(column_names).combine_chunks()
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, selected.schema) as writer:
            writer.write_table(selected)
        digest.update(sink.getvalue().to_pybytes())
    return digest.hexdigest()


def _atomic_write_json(destination: Path, payload: Mapping[str, Any]) -> None:
    temp_path = _new_sibling_temp_path(destination, ".tmp")
    try:
        with open(temp_path, "w") as f:
            json.dump(
                payload,
                f,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
                allow_nan=False,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, destination)
        _fsync_directory(destination.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def write_value_function_metadata(root: str | Path, metadata: Mapping[str, Any]) -> Path:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Raw run directory not found: {root}")
    payload = dict(metadata)
    payload.setdefault("created_at", iso_utc_now())
    meta_path = root / VALUE_FUNCTION_META_FILENAME
    _atomic_write_json(meta_path, payload)
    return meta_path


def merge_value_function_metadata(root: str | Path, patch: Mapping[str, Any]) -> Path:
    root = Path(root).expanduser().resolve()
    try:
        existing = read_value_function_metadata(root)
    except FileNotFoundError:
        existing = {}
    payload = _deep_merge(existing, patch)
    payload.setdefault("created_at", iso_utc_now())
    return write_value_function_metadata(root, payload)


def _stage_record(metadata: Mapping[str, Any], stage_name: str) -> Mapping[str, Any] | None:
    stages = metadata.get("stages")
    if not isinstance(stages, Mapping):
        return None
    stage = stages.get(stage_name)
    return stage if isinstance(stage, Mapping) else None


def _mark_stale_dependents(stages: dict[str, Any], changed_stage: str) -> None:
    queue = [changed_stage]
    while queue:
        upstream_name = queue.pop(0)
        upstream = stages.get(upstream_name) or {}
        upstream_fingerprint = upstream.get("stage_fingerprint")
        for name, candidate in stages.items():
            if name == upstream_name or not isinstance(candidate, Mapping):
                continue
            dependencies = candidate.get("dependencies") or {}
            if upstream_name not in dependencies:
                continue
            dependency_is_stale = (
                bool(upstream.get("stale")) or dependencies[upstream_name] != upstream_fingerprint
            )
            if dependency_is_stale and not candidate.get("stale"):
                updated = dict(candidate)
                updated["stale"] = True
                updated["stale_reason"] = f"dependency '{upstream_name}' changed or is stale"
                stages[name] = updated
                queue.append(name)


def update_stage_metadata(
    root: str | Path,
    stage_name: str,
    *,
    config: Any,
    input_columns: Sequence[str],
    input_fingerprint: str,
    output_columns: Sequence[str],
    prediction_source: str | None,
    synthetic: bool,
    output_fingerprint: str | None = None,
    dependencies: Sequence[str] = (),
    metadata_patch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically merge a stage record and invalidate outdated dependents."""

    root = Path(root).expanduser().resolve()
    try:
        metadata = read_value_function_metadata(root)
    except FileNotFoundError:
        metadata = {}
    stages = dict(metadata.get("stages") or {})
    dependency_fingerprints: dict[str, str] = {}
    for dependency in dependencies:
        record = stages.get(dependency)
        if not isinstance(record, Mapping) or not record.get("stage_fingerprint"):
            raise ValueError(f"Stage '{stage_name}' requires missing dependency '{dependency}'")
        if record.get("stale"):
            raise StalePipelineArtifactError(
                f"Stage '{stage_name}' requires stale dependency '{dependency}'; rerun it first"
            )
        dependency_fingerprints[dependency] = str(record["stage_fingerprint"])

    created_at = iso_utc_now()
    record = {
        "created_at": created_at,
        "config": normalize_stage_config(config),
        "input_columns": list(input_columns),
        "input_fingerprint": input_fingerprint,
        "output_columns": list(output_columns),
        "output_fingerprint": output_fingerprint,
        "prediction_source": prediction_source,
        "synthetic": bool(synthetic),
        "dependencies": dependency_fingerprints,
        "stale": False,
    }
    record["stage_fingerprint"] = fingerprint_payload(record)
    stages[stage_name] = record
    _mark_stale_dependents(stages, stage_name)

    patch = dict(metadata_patch or {})
    patch["pipeline_schema_version"] = PIPELINE_SCHEMA_VERSION
    patch["stages"] = stages
    merge_value_function_metadata(root, patch)
    return record


def assert_stage_dependencies_current(root: str | Path, stage_name: str) -> None:
    metadata = read_value_function_metadata(root)
    record = _stage_record(metadata, stage_name)
    if record is None:
        raise ValueError(f"Missing pipeline stage metadata for '{stage_name}'")
    if record.get("stale"):
        raise StalePipelineArtifactError(
            f"Pipeline stage '{stage_name}' is stale: {record.get('stale_reason', 'dependency changed')}"
        )
    current_input_fingerprint = fingerprint_raw_run_columns(root, list(record.get("input_columns") or []))
    if current_input_fingerprint != record.get("input_fingerprint"):
        raise StalePipelineArtifactError(
            f"Pipeline stage '{stage_name}' inputs changed; rerun '{stage_name}'"
        )
    if record.get("output_fingerprint"):
        current_output_fingerprint = fingerprint_raw_run_columns(
            root, list(record.get("output_columns") or [])
        )
        if current_output_fingerprint != record.get("output_fingerprint"):
            raise StalePipelineArtifactError(
                f"Pipeline stage '{stage_name}' outputs changed; rerun '{stage_name}'"
            )
    for dependency, expected in (record.get("dependencies") or {}).items():
        upstream = _stage_record(metadata, dependency)
        if upstream is None or upstream.get("stale") or upstream.get("stage_fingerprint") != expected:
            raise StalePipelineArtifactError(
                f"Pipeline stage '{stage_name}' has stale dependency '{dependency}'; rerun '{stage_name}'"
            )


def read_value_function_metadata(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    meta_path = root / VALUE_FUNCTION_META_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {VALUE_FUNCTION_META_FILENAME} in {root}")
    with open(meta_path) as f:
        return json.load(f)
