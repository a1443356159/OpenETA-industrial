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

# A detached clean clone deliberately has no repository-local virtualenv.  An
# operator may provide the independently verified remote interpreter, but it
# must be an absolute executable.  Falling back to python3 is allowed only
# after proving the runtime imports used by the PTY acceptance path exist.
REQUIRED_PYTHON_MODULES="pytest,gymnasium,numpy,prompt_toolkit"
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
if ! "${PYTHON_BIN}" -c "import pytest, gymnasium, numpy, prompt_toolkit" >/dev/null 2>&1; then
  echo "OPENETA_PYTHON_RUNTIME_IMPORTS_MISSING: ${REQUIRED_PYTHON_MODULES}" >&2
  exit 3
fi
export OPENETA_PYTHON_EXECUTABLE="${PYTHON_BIN}"
export OPENETA_PYTHON_SOURCE="${PYTHON_SOURCE}"
export OPENETA_PYTHON_REQUIRED_MODULES="${REQUIRED_PYTHON_MODULES}"

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "${REPO_DIR}/extensions/gazebo/ros2_ws/install/setup.bash"
set -u

# The Python used for the isolated acceptance venv can itself be hosted under
# another ROS prefix.  Gazebo workers must import and dynamically link one
# coherent stack: this vendor Jazzy installation plus this clean-clone overlay.
export OPENETA_GAZEBO_SYSTEM_ROS_PREFIX="/opt/ros/jazzy"
export OPENETA_GAZEBO_OVERLAY="${REPO_DIR}/extensions/gazebo/ros2_ws/install"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/tui_gazebo_acceptance.py" "$@"
