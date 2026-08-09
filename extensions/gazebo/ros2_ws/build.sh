#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
python3 "${REPO_DIR}/extensions/gazebo/asset_preflight.py"
if [[ -r /opt/ros/jazzy/setup.bash ]]; then
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u
fi
ROS2_BIN="${OPENETA_ROS2_BIN:-$(command -v ros2 || true)}"
if [[ -z "${ROS2_BIN}" || "${ROS_DISTRO:-}" != jazzy ]]; then
  echo "ROS_NOT_READY: ROS 2 Jazzy is required; source /opt/ros/jazzy/setup.bash" >&2
  exit 3
fi
if [[ "${OPENETA_SKIP_ROSDEP:-0}" != 1 ]]; then
  rosdep install --from-paths "${SCRIPT_DIR}/src" --ignore-src --rosdistro jazzy -y
fi
cd "${SCRIPT_DIR}"
colcon build --symlink-install --event-handlers console_direct+
