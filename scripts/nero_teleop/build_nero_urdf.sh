#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/zenbot-robot/repos"
XACRO_PATH="${1:-$ROOT_DIR/nero/urdf/nero_with_gripper_description.xacro}"
OUT_PATH="${2:-$ROOT_DIR/lerobot/scripts/nero_teleop/generated/nero_with_gripper_description.urdf}"

mkdir -p "$(dirname "$OUT_PATH")"

set +u
source /opt/ros/humble/setup.bash
source "$ROOT_DIR/agx_arm_ws/install/setup.bash"
set -u

xacro "$XACRO_PATH" -o "$OUT_PATH"
echo "Wrote URDF to $OUT_PATH"
