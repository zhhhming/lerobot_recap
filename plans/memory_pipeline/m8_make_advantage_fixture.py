#!/usr/bin/env python

"""Create a temporary, video-sharing M8 dataset with subtask advantage fields.

The source dataset is never modified.  Data/meta parquet files are copied with
two additional columns while the large videos directory is represented by one
absolute symlink to the source tree.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


LABEL_KEY = "advantage_label_subtask"
WEIGHT_KEY = "advantage_loss_weight_subtask"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not (source / "meta/info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobotDataset root: {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite destination: {destination}")

    (destination / "meta").mkdir(parents=True)
    (destination / "data").mkdir()
    for metadata_name in ("stats.json", "tasks.parquet"):
        shutil.copy2(source / "meta" / metadata_name, destination / "meta" / metadata_name)
    if (source / "meta/episodes").exists():
        shutil.copytree(source / "meta/episodes", destination / "meta/episodes")
    (destination / "videos").symlink_to(source / "videos", target_is_directory=True)

    info = json.loads((source / "meta/info.json").read_text())
    info["features"][LABEL_KEY] = {"dtype": "string", "shape": [1], "names": None}
    info["features"][WEIGHT_KEY] = {"dtype": "float32", "shape": [1], "names": None}
    (destination / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")

    rows_written = 0
    label_counts = {"positive": 0, "negative": 0, "ignore": 0}
    for source_file in sorted(source.glob("data/chunk-*/file-*.parquet")):
        relative = source_file.relative_to(source)
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source_file)
        episode = table["episode_index"].to_numpy()
        selector = episode % 3
        labels = np.where(selector == 0, "positive", np.where(selector == 1, "negative", "ignore"))
        weights = np.where(selector == 0, 2.0, np.where(selector == 1, 1.0, 0.0)).astype(
            np.float32
        )
        table = table.append_column(LABEL_KEY, pa.array(labels, type=pa.string()))
        table = table.append_column(WEIGHT_KEY, pa.array(weights, type=pa.float32()))
        pq.write_table(table, destination_file)
        rows_written += table.num_rows
        for label in label_counts:
            label_counts[label] += int(np.sum(labels == label))

    if rows_written != info["total_frames"]:
        raise AssertionError(f"Expected {info['total_frames']} rows, wrote {rows_written}")
    if not all(label_counts.values()):
        raise AssertionError(f"Fixture must contain all advantage labels: {label_counts}")
    print(
        json.dumps(
            {
                "source": str(source),
                "destination": str(destination),
                "rows": rows_written,
                "label_counts": label_counts,
                "videos_symlink": str((destination / "videos").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

