#!/usr/bin/env bash
# Report the fail-closed status of the retired M3/M4 cloud acceptance path.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${OPENETA_ACCEPTANCE_PYTHON:-${REPO_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_NOT_READY: ${PYTHON_BIN}" >&2
  exit 3
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/cloud_m0_m4_acceptance.py" "$@"
