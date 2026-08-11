#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DRIVER="${SCRIPT_DIR}/m3_pickplace_acceptance.py"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
LOCK_DIR="/tmp/openeta-acceptance-locks"
mkdir -p "${LOCK_DIR}" "${REPO_DIR}/.cache/logs" "${REPO_DIR}/.cache/reports"

ORIGINAL_ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
if [[ -r "${SCRIPT_DIR}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/install/setup.bash"
fi
set -u

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_NOT_READY: ${PYTHON_BIN} is unavailable" >&2
  exit 3
fi

unset ROS_LOCALHOST_ONLY ROS_STATIC_PEERS ROS2CLI_DISABLE_DAEMON
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_HOME="${REPO_DIR}/.cache/ros/m3-acceptance-select-$$"
mkdir -p "${ROS_HOME}"
DOMAIN_SELECTION_LOG="${ROS_HOME}/domain-selection.jsonl"

terminate_group() {
  local leader="${1:-}"
  [[ "${leader}" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "${leader}" 2>/dev/null; then
    kill -TERM -- "-${leader}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${leader}" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "${leader}" 2>/dev/null; then
      kill -KILL -- "-${leader}" 2>/dev/null || true
    fi
  fi
  wait "${leader}" 2>/dev/null || true
}

run_cleanup_path_self_tests() {
  local scratch mode helper child rc expected
  scratch="$(mktemp -d)"
  for mode in normal startup_failure action_failure interrupt_signal; do
    (
      child=""
      trap 'terminate_group "${child}"' EXIT
      exec 9>"${scratch}/${mode}.lock"
      flock -n 9
      setsid bash -c 'while :; do sleep 1; done' &
      child=$!
      echo "${child}" >"${scratch}/${mode}.pid"
      case "${mode}" in
        normal) exit 0 ;;
        startup_failure) exit 17 ;;
        action_failure) sleep 0.1; exit 18 ;;
        interrupt_signal)
          trap 'exit 19' INT TERM
          while :; do sleep 0.1; done
          ;;
      esac
    ) &
    helper=$!
    if [[ "${mode}" == interrupt_signal ]]; then
      for _ in $(seq 1 50); do
        [[ -s "${scratch}/${mode}.pid" ]] && break
        sleep 0.02
      done
      kill -TERM "${helper}"
    fi
    set +e
    wait "${helper}"
    rc=$?
    set -e
    case "${mode}" in
      normal) expected=0 ;;
      startup_failure) expected=17 ;;
      action_failure) expected=18 ;;
      interrupt_signal) expected=19 ;;
    esac
    [[ "${rc}" == "${expected}" ]] || return 1
    child="$(<"${scratch}/${mode}.pid")"
    ! kill -0 "${child}" 2>/dev/null || return 1
    flock -n "${scratch}/${mode}.lock" true || return 1
  done
  rm -f -- "${scratch}"/*.pid "${scratch}"/*.lock
  rmdir -- "${scratch}"
}

DOMAIN_LOCK_FD=""
for candidate in $(seq 80 101); do
  exec {candidate_fd}>"${LOCK_DIR}/ros-domain-${candidate}.lock"
  if flock -n "${candidate_fd}" && "${PYTHON_BIN}" "${DRIVER}" probe --report /dev/null --domain "${candidate}" >>"${DOMAIN_SELECTION_LOG}"; then
    DOMAIN_LOCK_FD="${candidate_fd}"
    ROS_DOMAIN_ID="${candidate}"
    break
  fi
  echo "{\"domain\":${candidate},\"state\":\"FAILED\",\"reason_code\":\"LOCK_UNAVAILABLE_OR_NOT_EMPTY\"}" >>"${DOMAIN_SELECTION_LOG}"
  flock -u "${candidate_fd}" 2>/dev/null || true
  eval "exec ${candidate_fd}>&-"
done
[[ -n "${DOMAIN_LOCK_FD}" ]] || { echo "ISOLATION_UNAVAILABLE: no ROS domain" >&2; exit 6; }

port_is_free() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    s.close()
PY
}

PORT_LOCK_FD=""
for candidate in $(seq 18865 18964); do
  exec {candidate_fd}>"${LOCK_DIR}/mcp-port-${candidate}.lock"
  if flock -n "${candidate_fd}" && port_is_free "${candidate}"; then
    PORT_LOCK_FD="${candidate_fd}"
    OPENETA_MCP_PORT="${candidate}"
    break
  fi
  flock -u "${candidate_fd}" 2>/dev/null || true
  eval "exec ${candidate_fd}>&-"
done
[[ -n "${PORT_LOCK_FD}" ]] || { echo "ISOLATION_UNAVAILABLE: no MCP port" >&2; exit 7; }

export ROS_DOMAIN_ID OPENETA_MCP_PORT
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_PARTITION="openeta_m3_acceptance_${ROS_DOMAIN_ID}_$$"
export ROS_HOME="${REPO_DIR}/.cache/ros/m3-acceptance-${ROS_DOMAIN_ID}-$$"
export OPENETA_ISOLATION_SELECTION_LOG="${DOMAIN_SELECTION_LOG}"
export OPENETA_WORKER_LOG_DIR="${REPO_DIR}/.cache/logs/m3-acceptance-${ROS_DOMAIN_ID}-$$"
mkdir -p "${ROS_HOME}" "${OPENETA_WORKER_LOG_DIR}"

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="${REPO_DIR}/.cache/reports/m3-pickplace-${RUN_STAMP}-$$.json"
MCP_LOG="${REPO_DIR}/.cache/logs/m3-mcp-${RUN_STAMP}-$$.log"

run_cleanup_path_self_tests
"${PYTHON_BIN}" "${DRIVER}" init --report "${REPORT}" \
  --domain "${ROS_DOMAIN_ID}" --original-domain "${ORIGINAL_ROS_DOMAIN_ID}" \
  --partition "${GZ_PARTITION}" --port "${OPENETA_MCP_PORT}" \
  --world "m3_rm75_robotiq2f85_pickplace"

cleanup() {
  local main_exit=$? final_exit=0 pgid current_pgid
  trap - EXIT INT TERM
  current_pgid="$(ps -o pgid= -p "$$" | tr -d '[:space:]')"
  terminate_group "${MCP_PID:-}"
  mapfile -t isolated_groups < <(
    "${PYTHON_BIN}" "${DRIVER}" processes --report "${REPORT}" \
      --partition "${GZ_PARTITION}" 2>/dev/null || true
  )
  for pgid in "${isolated_groups[@]}"; do
    [[ "${pgid}" =~ ^[0-9]+$ && "${pgid}" != "${current_pgid}" ]] && \
      kill -TERM -- "-${pgid}" 2>/dev/null || true
  done
  for _ in $(seq 1 30); do
    mapfile -t isolated_groups < <(
      "${PYTHON_BIN}" "${DRIVER}" processes --report "${REPORT}" \
        --partition "${GZ_PARTITION}" 2>/dev/null || true
    )
    ((${#isolated_groups[@]} == 0)) && break
    sleep 0.2
  done
  for pgid in "${isolated_groups[@]}"; do
    [[ "${pgid}" =~ ^[0-9]+$ && "${pgid}" != "${current_pgid}" ]] && \
      kill -KILL -- "-${pgid}" 2>/dev/null || true
  done
  set +e
  "${PYTHON_BIN}" "${DRIVER}" finalize --report "${REPORT}" \
    --domain "${ROS_DOMAIN_ID}" --partition "${GZ_PARTITION}" \
    --port "${OPENETA_MCP_PORT}" --world "m3_rm75_robotiq2f85_pickplace" \
    --exit-code "${main_exit}"
  final_exit=$?
  set -e
  if [[ "${main_exit}" == 0 && "${final_exit}" != 0 ]]; then
    main_exit="${final_exit}"
  fi
  flock -u "${DOMAIN_LOCK_FD}" 2>/dev/null || true
  flock -u "${PORT_LOCK_FD}" 2>/dev/null || true
  eval "exec ${DOMAIN_LOCK_FD}>&-"
  eval "exec ${PORT_LOCK_FD}>&-"
  echo "M3_ACCEPTANCE_REPORT=${REPORT}"
  [[ "${main_exit}" == 0 ]] || echo "OPENETA_M3_ACCEPTANCE_BLOCKED exit=${main_exit} cleanup=${final_exit}" >&2
  exit "${main_exit}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${REPO_DIR}"
OPENETA_SKIP_ROSDEP=1 "${SCRIPT_DIR}/build.sh"
set +u
source "${SCRIPT_DIR}/install/setup.bash"
set -u
# ROS / ament setup scripts are allowed to alter shell traps.  Reinstall the
# acceptance-owned handlers after sourcing the overlay so every later failure
# still finalizes the report and releases the two locks.
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${PYTHON_BIN}" -m pytest -q \
  tests/test_gazebo_m3_profile.py tests/test_gazebo_m3_source.py \
  tests/test_gazebo_m3_verifier.py tests/test_gazebo_m3_worker.py \
  tests/test_gazebo_m3_acceptance.py \
  tests/test_gazebo_m2_assets.py tests/test_gazebo_m2_contract.py \
  tests/test_gazebo_m2_worker.py tests/test_gazebo_ros_control.py
"${PYTHON_BIN}" "${DRIVER}" gate --report "${REPORT}" --gate offline_contracts \
  --details "M3 verifier/source/worker/profile and M2 regression passed"

bash "${REPO_DIR}/scripts/check_openeta_m2.sh"
"${PYTHON_BIN}" "${DRIVER}" gate --report "${REPORT}" --gate m2_checkpoint_regression \
  --details "scripts/check_openeta_m2.sh passed; this is not formal M2 acceptance"

# The direct driver owns its launch stack in-process; give it its own session
# so the partition-scoped cleanup below can reap the whole tree (the cleanup
# deliberately never signals its own/ancestor groups).
setsid "${PYTHON_BIN}" "${DRIVER}" direct --report "${REPORT}"

setsid "${PYTHON_BIN}" -m sim.mcp_server --port "${OPENETA_MCP_PORT}" >"${MCP_LOG}" 2>&1 &
MCP_PID=$!
"${PYTHON_BIN}" "${DRIVER}" mcp --report "${REPORT}" \
  --mcp-url "http://127.0.0.1:${OPENETA_MCP_PORT}/sse"
terminate_group "${MCP_PID}"
unset MCP_PID

echo "OPENETA_M3_ACCEPTANCE_OK"
