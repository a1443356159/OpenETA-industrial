#!/usr/bin/env bash
# Prepare the fail-closed remote clean-clone PTY-TUI plan.  This wrapper never
# opens SSH itself; an authorized operator executes the reported command.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${OPENETA_ACCEPTANCE_PYTHON:-${REPO_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_NOT_READY: ${PYTHON_BIN}" >&2
  exit 3
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/cloud_m0_m4_acceptance.py" "$@"
