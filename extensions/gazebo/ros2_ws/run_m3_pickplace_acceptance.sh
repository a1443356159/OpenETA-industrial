#!/usr/bin/env bash
# M3's formal path is scripts/tui_gazebo_acceptance.py.  This wrapper only
# validates an already captured evidence JSON; it never launches Gazebo/MCP.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"
exec "${REPO_DIR}/.venv/bin/python" "${SCRIPT_DIR}/m3_pickplace_acceptance.py" "$@"
