#!/usr/bin/env bash
set -euo pipefail

# Reuse the one authoritative ROS/Python/Gazebo bootstrap.  The selected
# entry point is allowlisted by that bootstrap; this wrapper exists so an
# operator never needs to set an implementation environment variable.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENETA_GAZEBO_RUNNER="${SCRIPT_DIR}/open_sort_gazebo_tui.py"
exec "${SCRIPT_DIR}/run_normal_gazebo_acceptance.sh" "$@"
