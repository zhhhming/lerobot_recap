#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"

cd "${REPO_ROOT}"
export LEROBOT_PYTHON="${PYTHON}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/utils/terminal_status.py \
  src/lerobot/scripts/lerobot_policy_deploy.py \
  tests/scripts/test_lerobot_policy_deploy_status.py

plans/memory_pipeline/validate_milestone_6.sh

"${PYTHON}" -m pytest \
  tests/scripts/test_lerobot_policy_deploy_status.py \
  tests/inference_engines/test_rtc_memory.py \
  -q

if "${PYTHON}" -m ruff --version >/dev/null 2>&1; then
  "${PYTHON}" -m ruff check \
    src/lerobot/utils/terminal_status.py \
    src/lerobot/scripts/lerobot_policy_deploy.py \
    tests/scripts/test_lerobot_policy_deploy_status.py
  "${PYTHON}" -m ruff format --check \
    src/lerobot/utils/terminal_status.py \
    src/lerobot/scripts/lerobot_policy_deploy.py \
    tests/scripts/test_lerobot_policy_deploy_status.py
else
  echo "ruff: skipped (not installed in ${PYTHON})"
fi

git diff --check
