#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SYSTEM_ROS_SETUP="${OPENETA_GAZEBO_SYSTEM_ROS_PREFIX:-/opt/ros/jazzy}/setup.bash"
OVERLAY_PREFIX="${OPENETA_GAZEBO_OVERLAY:-${REPO_DIR}/extensions/gazebo/ros2_ws/install}"
OVERLAY_SETUP="${OVERLAY_PREFIX}/setup.bash"

if [[ ! -r "${SYSTEM_ROS_SETUP}" ]]; then
  echo "ROS_NOT_READY: ${SYSTEM_ROS_SETUP} is missing" >&2
  exit 2
fi
if [[ ! -r "${OVERLAY_SETUP}" ]]; then
  echo "WORKSPACE_NOT_BUILT: ${OVERLAY_SETUP} is missing" >&2
  exit 2
fi

REQUIRED_PYTHON_MODULES="pytest,gymnasium,numpy,prompt_toolkit,mcp,starlette,uvicorn"
PYTHON_SOURCE=""
if [[ -n "${OPENETA_PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_BIN="${OPENETA_PYTHON_EXECUTABLE}"
  PYTHON_SOURCE="OPENETA_PYTHON_EXECUTABLE"
  if [[ "${PYTHON_BIN}" != /* || ! -x "${PYTHON_BIN}" ]]; then
    echo "OPENETA_PYTHON_EXECUTABLE_INVALID: require an absolute executable" >&2
    exit 3
  fi
elif [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
  PYTHON_SOURCE="repo_venv"
else
  PYTHON_BIN="$(command -v python3 || true)"
  PYTHON_SOURCE="discovered_python3"
  if [[ -z "${PYTHON_BIN}" || "${PYTHON_BIN}" != /* || ! -x "${PYTHON_BIN}" ]]; then
    echo "OPENETA_PYTHON_DISCOVERY_FAILED: python3 is unavailable" >&2
    exit 3
  fi
fi
if ! "${PYTHON_BIN}" -c "import sys; assert sys.version_info[:2] == (3, 12)" >/dev/null 2>&1; then
  echo "OPENETA_PYTHON_VERSION_UNSUPPORTED: require CPython 3.12" >&2
  exit 3
fi
if ! "${PYTHON_BIN}" -c "import pytest, gymnasium, numpy, prompt_toolkit, mcp, starlette, uvicorn" >/dev/null 2>&1; then
  echo "OPENETA_PYTHON_RUNTIME_IMPORTS_MISSING: ${REQUIRED_PYTHON_MODULES}" >&2
  exit 3
fi
export OPENETA_PYTHON_EXECUTABLE="${PYTHON_BIN}"
export OPENETA_PYTHON_SOURCE="${PYTHON_SOURCE}"
export OPENETA_PYTHON_REQUIRED_MODULES="${REQUIRED_PYTHON_MODULES}"

set +u
# shellcheck disable=SC1090
source "${SYSTEM_ROS_SETUP}"
# shellcheck disable=SC1090
source "${OVERLAY_SETUP}"
set -u

if ! "${PYTHON_BIN}" -c "import rclpy; from rosgraph_msgs.msg import Clock; Clock.__class__.__import_type_support__(); assert Clock.__class__._TYPE_SUPPORT is not None" >/dev/null 2>&1; then
  echo "OPENETA_ROS_PYTHON_ABI_UNAVAILABLE: Jazzy rclpy/typesupport load failed" >&2
  exit 3
fi
if ! command -v ros2 >/dev/null 2>&1 || ! command -v gz >/dev/null 2>&1; then
  echo "OPENETA_GAZEBO_COMMAND_UNAVAILABLE: sourced ROS environment must provide ros2 and gz" >&2
  exit 3
fi
if ! ros2 pkg prefix openeta_rm75_robotiq2f85_sim >/dev/null 2>&1; then
  echo "OPENETA_GAZEBO_OVERLAY_PACKAGE_UNAVAILABLE: rebuild/source the OpenETA Gazebo overlay" >&2
  exit 3
fi

export OPENETA_GAZEBO_SYSTEM_ROS_PREFIX="$(cd -- "$(dirname -- "${SYSTEM_ROS_SETUP}")" && pwd)"
export OPENETA_GAZEBO_OVERLAY="$(cd -- "${OVERLAY_PREFIX}" && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# The canonical operator acceptance is observable by default.  The GUI remains
# a case-owned, out-of-band client and can be disabled explicitly for CI with
# OPENETA_GAZEBO_OPERATOR_GUI=0.
export OPENETA_GAZEBO_OPERATOR_GUI="${OPENETA_GAZEBO_OPERATOR_GUI:-1}"

# Keep the formal acceptance's canonical command literal and immutable.  The
# task-neutral operator wrapper opts into the one separately allowlisted
# alternate entry point below; sourcing a ROS environment must never turn an
# arbitrary environment value into a Python entry point.
if [[ -z "${OPENETA_GAZEBO_RUNNER:-}" ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/normal_gazebo_acceptance.py" "$@"
fi

if [[ "${OPENETA_GAZEBO_RUNNER}" != "${SCRIPT_DIR}/open_sort_gazebo_tui.py" ]]; then
  echo "OPENETA_GAZEBO_RUNNER_INVALID: unsupported runner ${OPENETA_GAZEBO_RUNNER}" >&2
  exit 3
fi

exec "${PYTHON_BIN}" "${OPENETA_GAZEBO_RUNNER}" "$@"
