#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../config/runtime/m0_m2.env
source "${REPO_DIR}/config/runtime/m0_m2.env"

CHECK_PYTHON=1
CHECK_ROS=1
case "${1:-}" in
  "") ;;
  --python-only) CHECK_ROS=0 ;;
  --ros-only) CHECK_PYTHON=0 ;;
  -h|--help)
    echo "Usage: $0 [--python-only|--ros-only]"
    exit 0
    ;;
  *) echo "USAGE_ERROR: unknown option: $1" >&2; exit 64 ;;
esac

failures=()
record_failure() {
  failures+=("$1")
  printf '%s: %s\n' "$1" "$2" >&2
}

if [[ "$(uname -m)" != "x86_64" ]]; then
  record_failure PLATFORM_NOT_SUPPORTED "expected amd64/x86_64"
fi
if [[ ! -r /etc/os-release ]]; then
  record_failure PLATFORM_NOT_SUPPORTED "/etc/os-release is unavailable"
else
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04* ]]; then
    record_failure PLATFORM_NOT_SUPPORTED "expected Ubuntu 24.04, found ${ID:-unknown} ${VERSION_ID:-unknown}"
  fi
fi

if [[ "${CHECK_PYTHON}" == 1 ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    record_failure PYTHON_NOT_READY "${PYTHON_BIN} does not exist; run scripts/setup_openeta_m2.sh --python-only"
  elif ! "${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12); import gymnasium, mcp, numpy, PIL, imageio, prompt_toolkit, tiktoken' >/dev/null 2>&1; then
    record_failure PYTHON_NOT_READY "Python 3.12 or required OpenETA imports are unavailable"
  else
    echo "OK PYTHON_READY $(${PYTHON_BIN} --version 2>&1)"
  fi
fi

for asset in rm75_6fb_v_vendor robotiq_2f85_vendor; do
  if ! python3 "${REPO_DIR}/extensions/gazebo/asset_preflight.py" \
      "${REPO_DIR}/extensions/gazebo/assets/${asset}" >/dev/null; then
    record_failure MODEL_ASSET_NOT_FOUND "asset preflight failed for ${asset}"
  else
    echo "OK MODEL_ASSET ${asset}"
  fi
done

if ! python3 - "${REPO_DIR}" <<'PY'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
root = Path(sys.argv[1])
for path in (root / "extensions/gazebo/ros2_ws/src").rglob("*"):
    if path.suffix in {".sdf", ".srdf", ".urdf", ".xacro"}:
        ET.parse(path)
PY
then
  record_failure MODEL_ASSET_NOT_FOUND "repository URDF/SDF/Xacro XML is not well formed"
fi

if [[ "${CHECK_ROS}" == 1 ]]; then
  if [[ -r /opt/ros/jazzy/setup.bash ]]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    set -u
  fi
  for command_name in ros2 gz xacro colcon rosdep; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      case "${command_name}" in
        ros2|xacro) code=ROS_NOT_READY ;;
        gz) code=GAZEBO_NOT_READY ;;
        *) code=ROS_NOT_READY ;;
      esac
      record_failure "${code}" "${command_name} is not on PATH"
    fi
  done

  required_packages=(
    ros_gz_sim ros_gz_bridge gz_ros2_control controller_manager
    joint_state_broadcaster joint_trajectory_controller parallel_gripper_controller
    forward_command_controller
    moveit_ros_move_group moveit_configs_utils moveit_core robot_state_publisher tf2_ros xacro
    control_msgs geometry_msgs sensor_msgs std_msgs rclpy
  )
  if command -v ros2 >/dev/null 2>&1; then
    for package_name in "${required_packages[@]}"; do
      if ! ros2 pkg prefix "${package_name}" >/dev/null 2>&1; then
        record_failure ROS_PACKAGE_MISSING "${package_name}"
      fi
    done
  fi

  OVERLAY_SETUP="${OPENETA_GAZEBO_ROS2_WS}/install/setup.bash"
  if [[ ! -r "${OVERLAY_SETUP}" ]]; then
    record_failure WORKSPACE_NOT_BUILT "${OVERLAY_SETUP} is missing"
  elif command -v ros2 >/dev/null 2>&1; then
    set +u
    # shellcheck disable=SC1090
    source "${OVERLAY_SETUP}"
    set -u
    for package_name in openeta_rm75_v_description openeta_rm75_parallel_sim openeta_rm75_robotiq2f85_sim; do
      if ! ros2 pkg prefix "${package_name}" >/dev/null 2>&1; then
        record_failure WORKSPACE_NOT_BUILT "overlay package ${package_name} is unavailable"
      fi
    done
    if command -v xacro >/dev/null 2>&1; then
      expanded="$(mktemp --suffix=.urdf)"
      if ! xacro "${OPENETA_GAZEBO_ROS2_WS}/install/openeta_rm75_robotiq2f85_sim/share/openeta_rm75_robotiq2f85_sim/urdf/rm75_robotiq2f85.urdf.xacro" >"${expanded}" 2>/dev/null \
          || ! python3 -c 'import sys, xml.etree.ElementTree as E; E.parse(sys.argv[1])' "${expanded}"; then
        record_failure MODEL_ASSET_NOT_FOUND "Robotiq robot_description did not expand"
      fi
      rm -f -- "${expanded}"
    fi
  fi
fi

if python3 - "${OPENETA_MCP_PORT}" <<'PY'
import socket, sys
s = socket.socket()
try:
    try:
        s.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
finally:
    s.close()
PY
then
  echo "OK MCP_PORT_FREE ${OPENETA_MCP_PORT}"
else
  record_failure MCP_PORT_IN_USE "127.0.0.1:${OPENETA_MCP_PORT} is occupied"
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "INFO RENDERER NVIDIA_GPU_AVAILABLE"
else
  echo "INFO RENDERER HEADLESS_SOFTWARE_MODE (RGB-D throughput may be limited)"
fi

if ((${#failures[@]})); then
  printf 'OPENETA_M2_CHECK_FAILED count=%d codes=%s\n' "${#failures[@]}" "$(IFS=,; echo "${failures[*]}")" >&2
  exit 1
fi
echo "OPENETA_M2_CHECK_OK"
