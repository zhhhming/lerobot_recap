import os
import random
from pathlib import Path

import pytest

from lerobot.scripts.lerobot_value_viz import ValueRun
from lerobot.value_function.schema import EXTRAS_FILENAME

RUN_REAL_SMOKE = os.environ.get("LEROBOT_RUN_REAL_VALUE_VIZ_SMOKE") == "1"
REAL_ROOT = Path(
    os.environ.get(
        "LEROBOT_VALUE_VIZ_REAL_ROOT",
        "/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3",
    )
)


@pytest.mark.skipif(not RUN_REAL_SMOKE, reason="Set LEROBOT_RUN_REAL_VALUE_VIZ_SMOKE=1")
def test_real_raw_run_random_episode_is_read_only_and_degrades_gracefully():
    run = ValueRun(REAL_ROOT)
    episode = random.Random(12).choice(run.episodes)
    extras_path = episode.path / EXTRAS_FILENAME
    before = extras_path.read_bytes() if extras_path.is_file() else None

    meta = run.metadata()
    assert meta["episodes"]
    assert {camera["subdir"] for camera in meta["cameras"]} >= {
        "left_wrist",
        "third_person",
        "right_wrist",
    }
    curves = run.curves(episode.index, unit="norm", boundary="gt", max_points=2_000)
    assert curves["sampled_points"] <= 2_000
    assert curves["frame_count"] == episode.frame_count
    frame = run.frame(episode.index, episode.frame_count // 2, boundary="gt")
    assert frame["frame"] == episode.frame_count // 2
    assert run.image_path(episode.index, "third_person", frame["frame"]).is_file()

    after = extras_path.read_bytes() if extras_path.is_file() else None
    assert after == before
