#!/usr/bin/env bash
# Run the noninteractive, clean-clone cloud M0--M4 formal acceptance.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${OPENETA_ACCEPTANCE_PYTHON:-${REPO_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_NOT_READY: ${PYTHON_BIN}" >&2
  exit 3
fi

if [[ ! -r /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS_NOT_READY: /opt/ros/jazzy/setup.bash is unavailable" >&2
  exit 3
fi
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
set -u

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/cloud_m0_m4_acceptance.py" \
  --source-repo "${REPO_DIR}" "$@"
