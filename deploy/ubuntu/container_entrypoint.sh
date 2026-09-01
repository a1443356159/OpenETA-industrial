#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT="${OPENETA_SOURCE_ROOT:-/opt/openeta/src}"
readonly STATE_ROOT="${OPENETA_STATE_ROOT:-/srv/openeta/state}"
readonly APP_PYTHON="${OPENETA_PYTHON_EXECUTABLE:-/opt/openeta/venvs/openeta/bin/python}"
readonly ROS_SETUP="${OPENETA_GAZEBO_SYSTEM_ROS_PREFIX:-/opt/ros/jazzy}/setup.bash"
readonly OVERLAY_SETUP="${OPENETA_GAZEBO_OVERLAY:-/opt/openeta/src/extensions/gazebo/ros2_ws/install}/setup.bash"

fail() {
  printf 'OPENETA_CONTAINER_ERROR: %s\n' "$*" >&2
  exit 2
}

[[ -d "${SOURCE_ROOT}" ]] || fail "source root is missing: ${SOURCE_ROOT}"
[[ -x "${APP_PYTHON}" ]] || fail "application Python is missing: ${APP_PYTHON}"
[[ -r "${ROS_SETUP}" ]] || fail "ROS setup is missing: ${ROS_SETUP}"
[[ -r "${OVERLAY_SETUP}" ]] || fail "Gazebo overlay is missing: ${OVERLAY_SETUP}"

mkdir -p \
  "${STATE_ROOT}/cache/huggingface/sam3" \
  "${STATE_ROOT}/cache/matplotlib" \
  "${STATE_ROOT}/home" \
  "${STATE_ROOT}/ros" \
  "${STATE_ROOT}/runs" \
  "${STATE_ROOT}/workspace"

set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
# shellcheck disable=SC1090
source "${OVERLAY_SETUP}"
set -u

export HOME="${HOME:-${STATE_ROOT}/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STATE_ROOT}/cache}"
export HF_HOME="${HF_HOME:-${STATE_ROOT}/cache/huggingface/sam3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${STATE_ROOT}/cache/matplotlib}"
export ROS_HOME="${ROS_HOME:-${STATE_ROOT}/ros}"
export OPENETA_RUN_ROOT="${OPENETA_RUN_ROOT:-${STATE_ROOT}/runs}"
export OPENETA_PYTHON_EXECUTABLE="${APP_PYTHON}"
export PATH="/opt/openeta/venvs/openeta/bin:${PATH}"
export PYTHONPATH="${SOURCE_ROOT}:/opt/openeta/src${PYTHONPATH:+:${PYTHONPATH}}"

provider_secret=/run/secrets/openeta_provider_env
if [[ -s "${provider_secret}" ]]; then
  provider_exports="$(
    "${APP_PYTHON}" /opt/openeta/src/deploy/ubuntu/load_provider_env.py \
      "${provider_secret}"
  )"
  eval "${provider_exports}"
  unset provider_exports
fi

loopback_no_proxy="127.0.0.1,localhost,::1"
if [[ -n "${NO_PROXY:-}" ]]; then
  export NO_PROXY="${loopback_no_proxy},${NO_PROXY}"
else
  export NO_PROXY="${loopback_no_proxy}"
fi
export no_proxy="${NO_PROXY}"

mode="${1:-shell}"
if (( $# > 0 )); then
  shift
fi

case "${mode}" in
  shell)
    cd "${SOURCE_ROOT}"
    exec bash "$@"
    ;;
  tui)
    exec /opt/openeta/src/deploy/ubuntu/run_tui.sh "$@"
    ;;
  smoke-normal|agentic-normal)
    exec /opt/openeta/src/deploy/ubuntu/run_normal.sh "${mode}" "$@"
    ;;
  validate-assets)
    exec "${APP_PYTHON}" /opt/openeta/src/deploy/ubuntu/prepare_assets.py \
      --model-root "${OPENETA_MODEL_ROOT:-/srv/openeta/models}" \
      --source-root /opt/openeta/third_party \
      --sam3-hf-home "${HF_HOME}" \
      "$@"
    ;;
  test)
    cd "${SOURCE_ROOT}"
    exec "${APP_PYTHON}" -m pytest "$@"
    ;;
  exec)
    (( $# > 0 )) || fail "exec requires a command"
    exec "$@"
    ;;
  *)
    fail "unknown mode '${mode}'; expected shell, tui, smoke-normal, agentic-normal, validate-assets, test, or exec"
    ;;
esac
