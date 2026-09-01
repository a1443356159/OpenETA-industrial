#!/usr/bin/env bash
set -euo pipefail

readonly APP_PYTHON="${OPENETA_PYTHON_EXECUTABLE:-/opt/openeta/venvs/openeta/bin/python}"
readonly STATE_ROOT="${OPENETA_STATE_ROOT:-/srv/openeta/state}"
readonly RUNS_ROOT="${OPENETA_RUN_ROOT:-${STATE_ROOT}/runs}"
readonly WORKSPACE="${STATE_ROOT}/workspace"
readonly SIM_PORT="${OPENETA_MCP_PORT:-8765}"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
session_root="${RUNS_ROOT}/tui-${stamp}-$$"
mkdir -p "${session_root}" "${WORKSPACE}"

# shellcheck disable=SC1091
source /opt/openeta/src/deploy/ubuntu/model_services.sh
sim_pid=""
cleanup() {
  if [[ -n "${sim_pid}" ]] && kill -0 "${sim_pid}" >/dev/null 2>&1; then
    kill -TERM "${sim_pid}" >/dev/null 2>&1 || true
    wait "${sim_pid}" >/dev/null 2>&1 || true
  fi
  openeta_stop_model_services
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

openeta_prepare_model_assets "${session_root}/model-assets.json" \
  >"${session_root}/model-assets.stdout.json"
openeta_start_model_services "${session_root}/mcp-state"

if { exec 9<>"/dev/tcp/127.0.0.1/${SIM_PORT}"; } 2>/dev/null; then
  exec 9>&-
  exec 9<&-
  echo "simulator MCP port is already in use: ${SIM_PORT}" >&2
  exit 2
fi

"${APP_PYTHON}" /opt/openeta/src/deploy/ubuntu/prepare_tui_workspace.py \
  --output "${WORKSPACE}/.mcp.json" \
  --sim-port "${SIM_PORT}" \
  --sam3-port "${OPENETA_SAM3_PORT:-8773}" \
  --anyplace-port "${OPENETA_ANYPLACE_PORT:-8775}" \
  --graspgenx-port "${OPENETA_GRASPGENX_PORT:-8778}"

MCP_PORT="${SIM_PORT}" "${APP_PYTHON}" -m sim.mcp_server \
  --port "${SIM_PORT}" >"${session_root}/sim-mcp.log" 2>&1 &
sim_pid=$!

ready=0
for _attempt in {1..300}; do
  if curl --fail --silent --show-error --max-time 1 \
    "http://127.0.0.1:${SIM_PORT}/session/container-bootstrap/envs" \
    >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${sim_pid}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
if (( ready == 0 )); then
  echo "simulator MCP did not become ready; log: ${session_root}/sim-mcp.log" >&2
  exit 2
fi

cd "${WORKSPACE}"
set +e
"${APP_PYTHON}" -m agent.cli.openeta_cli "$@"
tui_status=$?
set -e
exit "${tui_status}"
