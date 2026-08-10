#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -r /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS_NOT_READY: /opt/ros/jazzy/setup.bash is missing" >&2
  exit 2
fi
if [[ ! -r "${REPO_DIR}/extensions/gazebo/ros2_ws/install/setup.bash" ]]; then
  echo "WORKSPACE_NOT_BUILT: Gazebo overlay setup is missing" >&2
  exit 2
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "${REPO_DIR}/extensions/gazebo/ros2_ws/install/setup.bash"
set -u

exec "${REPO_DIR}/.venv/bin/python" "${SCRIPT_DIR}/tui_gazebo_acceptance.py" "$@"
