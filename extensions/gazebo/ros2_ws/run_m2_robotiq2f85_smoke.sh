#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DRIVER="${SCRIPT_DIR}/m2_robotiq2f85_acceptance.py"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
LOCK_DIR="/tmp/openeta-m2-acceptance-locks"
mkdir -p "${LOCK_DIR}" "${REPO_DIR}/.cache/logs" "${REPO_DIR}/.cache/reports"

# shellcheck source=../../../config/runtime/m0_m2.env
source "${REPO_DIR}/config/runtime/m0_m2.env"
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
      test_cleanup() {
        terminate_group "${child}"
      }
      trap test_cleanup EXIT
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
    if [[ "${rc}" != "${expected}" ]]; then
      echo "CLEANUP_SELF_TEST_FAILED: ${mode} exit=${rc} expected=${expected}" >&2
      return 1
    fi
    child="$(<"${scratch}/${mode}.pid")"
    if kill -0 "${child}" 2>/dev/null; then
      echo "CLEANUP_SELF_TEST_FAILED: ${mode} left process ${child}" >&2
      return 1
    fi
    if ! flock -n "${scratch}/${mode}.lock" true; then
      echo "CLEANUP_SELF_TEST_FAILED: ${mode} left its lock held" >&2
      return 1
    fi
  done
  rm -f -- "${scratch}"/*.pid "${scratch}"/*.lock
  rmdir -- "${scratch}"
}

domain_is_empty() {
  local candidate="$1"
  local nodes
  nodes="$(ROS_DOMAIN_ID="${candidate}" ROS2CLI_DISABLE_DAEMON=1 \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_LOCALHOST_ONLY=1 \
    timeout 8 ros2 node list 2>/dev/null || true)"
  [[ -z "${nodes//[[:space:]]/}" ]]
}

DOMAIN_LOCK_FD=""
for candidate in $(seq 100 199); do
  exec {candidate_fd}>"${LOCK_DIR}/ros-domain-${candidate}.lock"
  if flock -n "${candidate_fd}" && domain_is_empty "${candidate}"; then
    DOMAIN_LOCK_FD="${candidate_fd}"
    ROS_DOMAIN_ID="${candidate}"
    break
  fi
  flock -u "${candidate_fd}" 2>/dev/null || true
  eval "exec ${candidate_fd}>&-"
done
if [[ -z "${DOMAIN_LOCK_FD}" ]]; then
  echo "ISOLATION_UNAVAILABLE: no empty locked ROS domain in 100..199" >&2
  exit 6
fi

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
for candidate in $(seq 18765 18864); do
  exec {candidate_fd}>"${LOCK_DIR}/mcp-port-${candidate}.lock"
  if flock -n "${candidate_fd}" && port_is_free "${candidate}"; then
    PORT_LOCK_FD="${candidate_fd}"
    OPENETA_MCP_PORT="${candidate}"
    break
  fi
  flock -u "${candidate_fd}" 2>/dev/null || true
  eval "exec ${candidate_fd}>&-"
done
if [[ -z "${PORT_LOCK_FD}" ]]; then
  echo "ISOLATION_UNAVAILABLE: no free locked MCP port in 18765..18864" >&2
  exit 7
fi

export ROS_DOMAIN_ID OPENETA_MCP_PORT
export ROS2CLI_DISABLE_DAEMON=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=1
export GZ_PARTITION="openeta_m2_acceptance_${ROS_DOMAIN_ID}_$$"
export ROS_HOME="${REPO_DIR}/.cache/ros/m2-acceptance-${ROS_DOMAIN_ID}-$$"
export OPENETA_WORKER_LOG_DIR="${REPO_DIR}/.cache/logs/m2-acceptance-${ROS_DOMAIN_ID}-$$"
mkdir -p "${ROS_HOME}" "${OPENETA_WORKER_LOG_DIR}"

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="${REPO_DIR}/.cache/reports/m2-robotiq2f85-acceptance-${RUN_STAMP}-$$.json"
DIRECT_LOG="${REPO_DIR}/.cache/logs/m2-robotiq2f85-direct-${RUN_STAMP}-$$.log"
MCP_LOG="${REPO_DIR}/.cache/logs/m2-robotiq2f85-mcp-${RUN_STAMP}-$$.log"
REGRESSION_LOG="${REPO_DIR}/.cache/logs/m2-regression-${RUN_STAMP}-$$.log"

run_cleanup_path_self_tests
"${PYTHON_BIN}" "${DRIVER}" init --report "${REPORT}" \
  --domain "${ROS_DOMAIN_ID}" --original-domain "${ORIGINAL_ROS_DOMAIN_ID}" \
  --partition "${GZ_PARTITION}" --port "${OPENETA_MCP_PORT}"

cleanup() {
  local main_exit=$? final_exit=0 pgid
  trap - EXIT INT TERM
  terminate_group "${DIRECT_PID:-}"
  terminate_group "${MCP_PID:-}"

  # Nested bench workers and Ros2LaunchProcess instances intentionally create
  # their own sessions. Resolve only processes carrying this run's unique
  # Gazebo partition, then terminate those exact process groups.
  mapfile -t isolated_groups < <(
    "${PYTHON_BIN}" "${DRIVER}" processes --report "${REPORT}" \
      --partition "${GZ_PARTITION}" 2>/dev/null || true
  )
  for pgid in "${isolated_groups[@]}"; do
    [[ "${pgid}" =~ ^[0-9]+$ ]] || continue
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
    [[ "${pgid}" =~ ^[0-9]+$ ]] || continue
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  done

  set +e
  "${PYTHON_BIN}" "${DRIVER}" finalize --report "${REPORT}" \
    --domain "${ROS_DOMAIN_ID}" --partition "${GZ_PARTITION}" \
    --port "${OPENETA_MCP_PORT}" --exit-code "${main_exit}"
  final_exit=$?
  set -e
  if [[ "${final_exit}" != 0 ]]; then
    [[ "${main_exit}" == 0 ]] && main_exit=9
  fi
  flock -u "${DOMAIN_LOCK_FD}" 2>/dev/null || true
  flock -u "${PORT_LOCK_FD}" 2>/dev/null || true
  eval "exec ${DOMAIN_LOCK_FD}>&-"
  eval "exec ${PORT_LOCK_FD}>&-"
  echo "M2_ACCEPTANCE_REPORT=${REPORT}"
  if [[ "${main_exit}" != 0 ]]; then
    echo "OPENETA_M2_ROBOTIQ2F85_ACCEPTANCE_FAILED exit=${main_exit} cleanup=${final_exit}" >&2
  fi
  exit "${main_exit}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${REPO_DIR}"
OPENETA_SKIP_ROSDEP=1 "${SCRIPT_DIR}/build.sh"
"${PYTHON_BIN}" "${DRIVER}" gate --report "${REPORT}" --gate ros_workspace_build \
  --details "colcon build completed for both M2 profiles"

bash "${REPO_DIR}/scripts/check_openeta_m2.sh"
"${PYTHON_BIN}" "${DRIVER}" gate --report "${REPORT}" --gate m2_runtime_check \
  --details "scripts/check_openeta_m2.sh passed"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${PYTHON_BIN}" -m pytest -q \
  tests/test_gazebo_m2_assets.py \
  tests/test_gazebo_m2_contract.py \
  tests/test_gazebo_m2_worker.py \
  tests/test_gazebo_ros_control.py \
  tests/test_gazebo_observation.py \
  tests/test_sim_control_codecs.py
"${PYTHON_BIN}" "${DRIVER}" gate --report "${REPORT}" --gate offline_contract_regression \
  --details "M2 assets/control/freshness/MCP routing contracts passed"

setsid ros2 launch openeta_rm75_robotiq2f85_sim m2_gazebo_moveit.launch.py \
  >"${DIRECT_LOG}" 2>&1 &
DIRECT_PID=$!
if [[ "${OPENETA_M2_ACCEPTANCE_INJECT_FAILURE:-}" == startup ]]; then
  echo "INJECTED_STARTUP_FAILURE" >&2
  exit 70
fi

required_topics=(
  /joint_states /tf /tf_static /openeta_rgbd/image
  /openeta_rgbd/depth_image /openeta_rgbd/camera_info
  /openeta_wrist_rgbd/image /openeta_wrist_rgbd/depth_image
  /openeta_wrist_rgbd/camera_info
)
ready=0
for _ in $(seq 1 150); do
  actions="$(ros2 action list 2>/dev/null || true)"
  topics="$(ros2 topic list 2>/dev/null || true)"
  controllers="$(ros2 control list_controllers 2>/dev/null || true)"
  ready=1
  grep -qx /move_action <<<"${actions}" || ready=0
  grep -qx /parallel_gripper_controller/gripper_cmd <<<"${actions}" || ready=0
  for topic in "${required_topics[@]}"; do
    grep -qx "${topic}" <<<"${topics}" || ready=0
  done
  for controller in joint_state_broadcaster rm_group_controller parallel_gripper_controller; do
    grep -Eq "^${controller}[[:space:]].*active" <<<"${controllers}" || ready=0
  done
  [[ "${ready}" == 1 ]] && break
  if ! kill -0 "${DIRECT_PID}" 2>/dev/null; then
    tail -120 "${DIRECT_LOG}" >&2
    exit 4
  fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  echo "ROS_NOT_READY: direct Robotiq profile timed out" >&2
  tail -120 "${DIRECT_LOG}" >&2
  exit 5
fi
if [[ "${OPENETA_M2_ACCEPTANCE_INJECT_FAILURE:-}" == action ]]; then
  echo "INJECTED_ACTION_FAILURE" >&2
  exit 71
fi

"${PYTHON_BIN}" "${DRIVER}" direct --report "${REPORT}"
terminate_group "${DIRECT_PID}"
unset DIRECT_PID

setsid "${PYTHON_BIN}" -m sim.mcp_server --port "${OPENETA_MCP_PORT}" >"${MCP_LOG}" 2>&1 &
MCP_PID=$!
"${PYTHON_BIN}" "${DRIVER}" mcp --report "${REPORT}" \
  --mcp-url "http://127.0.0.1:${OPENETA_MCP_PORT}/sse"
terminate_group "${MCP_PID}"
unset MCP_PID

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${PYTHON_BIN}" -m pytest -q \
  --ignore=tests/test_behavior_vector_contract.py \
  --ignore=tests/test_robocasa_legacy_contract.py \
  >"${REGRESSION_LOG}" 2>&1
tail -20 "${REGRESSION_LOG}"
"${PYTHON_BIN}" "${DRIVER}" gate --report "${REPORT}" --gate repository_regression \
  --details "full repository regression passed with optional torch suites excluded; log=${REGRESSION_LOG}"

echo "OPENETA_M2_ROBOTIQ2F85_ACCEPTANCE_OK"
