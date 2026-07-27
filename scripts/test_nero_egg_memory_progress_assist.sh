#!/usr/bin/env bash

set -euo pipefail

TASK_REPO=/home/zenbot-robot/repos/lerobot
TASK_PYTHON=/home/zenbot-robot/repos/.conda/envs/lerobot-main-0.5.1/bin/python
TASK_ENV_LIB=/home/zenbot-robot/repos/.conda/envs/lerobot-main-0.5.1/lib

cd "$TASK_REPO"
export LD_LIBRARY_PATH="$TASK_ENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"$TASK_PYTHON" -m pytest -q \
  tests/inference_engines/test_memory_progress_assist.py \
  tests/inference_engines/test_rtc_memory.py \
  tests/inference_engines/test_rtc_subtask_time.py \
  tests/scripts/test_nero_egg_memory_progress_assist_deploy.py \
  tests/scripts/test_subtask_time_deploy.py
