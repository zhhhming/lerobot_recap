"""Helpers to keep Pinocchio isolated from system ROS Python paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_ros_python_path(path: str) -> bool:
    return "python3.10" in path and (
        "/opt/ros/" in path
        or "/ros2_ws/" in path
        or "/gen3_ws/" in path
        or "/zb_ws/" in path
    )


def _split_env_paths(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [p for p in raw.split(":") if p]


def isolate_python_env_for_pinocchio() -> dict[str, object]:
    prefix = Path(sys.executable).resolve().parents[1]
    cmeel_root = prefix / "lib" / "python3.12" / "site-packages" / "cmeel.prefix"
    cmeel_lib = str(cmeel_root / "lib")

    py_before = list(sys.path)
    sys.path[:] = [p for p in sys.path if not _is_ros_python_path(str(p))]

    env_py = _split_env_paths(os.environ.get("PYTHONPATH"))
    kept_py = [p for p in env_py if not _is_ros_python_path(p)]
    if kept_py:
        os.environ["PYTHONPATH"] = ":".join(kept_py)
    else:
        os.environ.pop("PYTHONPATH", None)

    env_ld = _split_env_paths(os.environ.get("LD_LIBRARY_PATH"))
    filtered_ld = [p for p in env_ld if "/opt/ros/" not in p and "/ros2_ws/" not in p and "/gen3_ws/" not in p and "/zb_ws/" not in p]
    new_ld = [cmeel_lib, str(prefix / "lib"), *filtered_ld]
    seen: set[str] = set()
    dedup_ld = []
    for p in new_ld:
        if p and p not in seen:
            dedup_ld.append(p)
            seen.add(p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(dedup_ld)

    return {
        "prefix": str(prefix),
        "cmeel_lib": cmeel_lib,
        "sys_path_removed": [p for p in py_before if p not in sys.path],
        "ld_library_path": dedup_ld,
    }
