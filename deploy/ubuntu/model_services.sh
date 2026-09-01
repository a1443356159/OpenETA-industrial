#!/usr/bin/env bash
# Shared lifecycle helpers for the container-only SAM3, AnyPlace and GraspGenX services.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "model_services.sh must be sourced" >&2
  exit 2
fi

OPENETA_CONTAINER_APP_PYTHON="${OPENETA_PYTHON_EXECUTABLE:-/opt/openeta/venvs/openeta/bin/python}"
OPENETA_CONTAINER_SOURCE_ROOT="${OPENETA_SOURCE_ROOT:-/opt/openeta/src}"
OPENETA_CONTAINER_MODEL_ROOT="${OPENETA_MODEL_ROOT:-/srv/openeta/models}"
OPENETA_CONTAINER_MODEL_SOURCE_ROOT="/opt/openeta/third_party"
OPENETA_CONTAINER_MCP_MANAGER="${OPENETA_CONTAINER_SOURCE_ROOT}/scripts/openeta_mcp_services.py"
OPENETA_STARTED_MODEL_SERVICES=()

openeta_prepare_model_assets() {
  local manifest="$1"
  mkdir -p "$(dirname -- "${manifest}")"
  "${OPENETA_CONTAINER_APP_PYTHON}" \
    /opt/openeta/src/deploy/ubuntu/prepare_assets.py \
    --model-root "${OPENETA_CONTAINER_MODEL_ROOT}" \
    --source-root "${OPENETA_CONTAINER_MODEL_SOURCE_ROOT}" \
    --sam3-hf-home "${HF_HOME}" \
    --manifest "${manifest}"
}

openeta_model_service_args() {
  local state_dir="$1"
  OPENETA_MODEL_SERVICE_ARGS=(
    --host 127.0.0.1
    --state-dir "${state_dir}"
    --sam3-port "${OPENETA_SAM3_PORT:-8773}"
    --anyplace-port "${OPENETA_ANYPLACE_PORT:-8775}"
    --graspgenx-port "${OPENETA_GRASPGENX_PORT:-8778}"
    --sam3-python "${OPENETA_SAM3_PYTHON:-/opt/openeta/venvs/sam3/bin/python}"
    --sam3-hf-home "${HF_HOME}"
    --anyplace-python "${OPENETA_ANYPLACE_PYTHON:-/opt/openeta/venvs/anyplace/bin/python}"
    --anyplace-root "${OPENETA_ANYPLACE_ROOT:-/opt/openeta/third_party/anyplace}"
    --anyplace-config-path "${OPENETA_ANYPLACE_CONFIG_PATH:-/opt/openeta/config/anyplace-normal.yaml}"
    --graspgenx-python "${OPENETA_GRASPGENX_PYTHON:-/opt/openeta/venvs/graspgenx/bin/python}"
    --graspgenx-root "${OPENETA_GRASPGENX_ROOT:-/opt/openeta/third_party/GraspGenX}"
    --graspgenx-checkpoint-root "${OPENETA_GRASPGENX_CHECKPOINT_ROOT:-${OPENETA_CONTAINER_MODEL_ROOT}/graspgenx/GraspGenXModel/release}"
    --graspgenx-gripper-descriptions-root "${OPENETA_GRASPGENX_GRIPPER_DESCRIPTIONS_ROOT:-${OPENETA_CONTAINER_MODEL_ROOT}/graspgenx/gripper_descriptions}"
    --qualification-profile "${OPENETA_QUALIFICATION_PROFILE:-fast_v3}"
  )
}

openeta_start_model_services() {
  local state_dir="$1"
  mkdir -p "${state_dir}"
  openeta_model_service_args "${state_dir}"
  OPENETA_STARTED_MODEL_SERVICES=()
  local target
  for target in sam3 anyplace graspgenx; do
    "${OPENETA_CONTAINER_APP_PYTHON}" "${OPENETA_CONTAINER_MCP_MANAGER}" \
      start "${target}" "${OPENETA_MODEL_SERVICE_ARGS[@]}"
    OPENETA_STARTED_MODEL_SERVICES+=("${target}")
    "${OPENETA_CONTAINER_APP_PYTHON}" "${OPENETA_CONTAINER_MCP_MANAGER}" \
      health "${target}" "${OPENETA_MODEL_SERVICE_ARGS[@]}"
    "${OPENETA_CONTAINER_APP_PYTHON}" "${OPENETA_CONTAINER_MCP_MANAGER}" \
      smoke "${target}" "${OPENETA_MODEL_SERVICE_ARGS[@]}"
  done
}

openeta_stop_model_services() {
  local index target
  for ((index = ${#OPENETA_STARTED_MODEL_SERVICES[@]} - 1; index >= 0; index--)); do
    target="${OPENETA_STARTED_MODEL_SERVICES[index]}"
    "${OPENETA_CONTAINER_APP_PYTHON}" "${OPENETA_CONTAINER_MCP_MANAGER}" \
      stop "${target}" "${OPENETA_MODEL_SERVICE_ARGS[@]}" >/dev/null 2>&1 || true
  done
  OPENETA_STARTED_MODEL_SERVICES=()
}
