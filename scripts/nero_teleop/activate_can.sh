#!/usr/bin/env bash
# Bring up the left/right Nero CAN interfaces. Thin wrapper around the vendor script.
set -euo pipefail
CAN_SCRIPT_DIR=${NERO_CAN_SCRIPT_DIR:-/home/zenbot-robot/repos/nero_pyagxarm/pyAgxArm/pyAgxArm/scripts/ubuntu}
cd "$CAN_SCRIPT_DIR"
exec bash can_muti_activate.sh "$@"
