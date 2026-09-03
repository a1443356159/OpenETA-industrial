#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/srv/openeta/runs}"
REPO_ROOT="/workspace/openeta"
APP_PYTHON="/opt/openeta/venvs/openeta/bin/python"
MCP_MANAGER="${REPO_ROOT}/scripts/openeta_mcp_services.py"
MCP_STATE="${RUN_ROOT}/mcp-state"

cd "${REPO_ROOT}"

actual_revision="$(git rev-parse HEAD)"
if [[ "${OPENETA_IMAGE_REVISION:-unknown}" != "${actual_revision}" ]]; then
  echo "image/workspace revision mismatch: ${OPENETA_IMAGE_REVISION:-unknown} != ${actual_revision}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/home" "${RUN_ROOT}/cache" "${MCP_STATE}"

# smoke_normal is an offline, node-local control chain. Cluster proxy
# variables must not redirect loopback MCP traffic through an HTTP/SOCKS
# proxy (httpx initializes configured transports before applying NO_PROXY).
unset ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="${NO_PROXY}"

set +u
source /opt/ros/jazzy/setup.bash
source /opt/openeta/src/extensions/gazebo/ros2_ws/install/setup.bash
set -u

export HOME="${RUN_ROOT}/home"
export XDG_CACHE_HOME="${RUN_ROOT}/cache"
export HF_HUB_OFFLINE=1
export HF_HOME="${RUN_ROOT}/cache/huggingface/sam3"
export MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib"
export ROS_HOME="${RUN_ROOT}/ros"
export PYTHONPATH="${REPO_ROOT}:/opt/openeta/src"
export OPENETA_PYTHON_EXECUTABLE="${APP_PYTHON}"
export OPENETA_GAZEBO_SYSTEM_ROS_PREFIX=/opt/ros/jazzy
export OPENETA_GAZEBO_OVERLAY=/opt/openeta/src/extensions/gazebo/ros2_ws/install
export OPENETA_QUALIFICATION_PROFILE=fast_v3
# The final GraspGenX release path freezes its 512-candidate reserve once and
# searches it incrementally.  This is a pool size, not 512 eager IK calls.
export OPENETA_GRASPGENX_RAW_POOL_SIZE=512
export OPENETA_ANYPLACE_RAW_POOL_SIZE=96
export OPENETA_ANYPLACE_DIVERSITY_POOL_SIZE=96
export OPENETA_SAM3_PYTHON=/opt/openeta/venvs/sam3/bin/python
export OPENETA_ANYPLACE_PYTHON=/opt/openeta/venvs/anyplace/bin/python
export OPENETA_GRASPGENX_PYTHON=/opt/openeta/venvs/graspgenx/bin/python
export OPENETA_ANYPLACE_ROOT=/opt/openeta/third_party/anyplace
export OPENETA_ANYPLACE_CONFIG_PATH=/opt/openeta/config/anyplace-normal.yaml
export OPENETA_GRASPGENX_ROOT=/opt/openeta/third_party/GraspGenX
export OPENETA_GRASPGENX_CHECKPOINT_ROOT=/srv/openeta/models/graspgenx/GraspGenXModel/release
export OPENETA_GRASPGENX_GRIPPER_DESCRIPTIONS_ROOT=/srv/openeta/models/graspgenx/gripper_descriptions

job_number="${SLURM_JOB_ID:-0}"
port_base=$((20000 + (job_number % 1000) * 10))
sam3_port=$((port_base + 3))
anyplace_port=$((port_base + 5))
graspgenx_port=$((port_base + 8))

service_args=(
  --host 127.0.0.1
  --state-dir "${MCP_STATE}"
  --sam3-port "${sam3_port}"
  --anyplace-port "${anyplace_port}"
  --graspgenx-port "${graspgenx_port}"
  --sam3-python "${OPENETA_SAM3_PYTHON}"
  --sam3-hf-home "${HF_HOME}"
  --anyplace-python "${OPENETA_ANYPLACE_PYTHON}"
  --anyplace-root "${OPENETA_ANYPLACE_ROOT}"
  --anyplace-config-path "${OPENETA_ANYPLACE_CONFIG_PATH}"
  --graspgenx-python "${OPENETA_GRASPGENX_PYTHON}"
  --graspgenx-root "${OPENETA_GRASPGENX_ROOT}"
  --graspgenx-checkpoint-root "${OPENETA_GRASPGENX_CHECKPOINT_ROOT}"
  --graspgenx-gripper-descriptions-root "${OPENETA_GRASPGENX_GRIPPER_DESCRIPTIONS_ROOT}"
  --qualification-profile fast_v3
)

cleanup() {
  "${APP_PYTHON}" "${MCP_MANAGER}" stop graspgenx "${service_args[@]}" >/dev/null 2>&1 || true
  "${APP_PYTHON}" "${MCP_MANAGER}" stop anyplace "${service_args[@]}" >/dev/null 2>&1 || true
  "${APP_PYTHON}" "${MCP_MANAGER}" stop sam3 "${service_args[@]}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

"${APP_PYTHON}" "${REPO_ROOT}/deploy/ubuntu/prepare_assets.py" \
  --sam3-hf-home "${HF_HOME}" \
  --manifest "${RUN_ROOT}/model-assets.json"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  > "${RUN_ROOT}/gpu.txt"

for target in sam3 anyplace graspgenx; do
  "${APP_PYTHON}" "${MCP_MANAGER}" start "${target}" "${service_args[@]}"
  "${APP_PYTHON}" "${MCP_MANAGER}" health "${target}" "${service_args[@]}"
  "${APP_PYTHON}" "${MCP_MANAGER}" smoke "${target}" "${service_args[@]}"
done

for run_index in 1 2; do
  run_path="${RUN_ROOT}/run-${run_index}"
  /usr/bin/time \
    --format='wall_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kb=%M' \
    --output="${RUN_ROOT}/run-${run_index}-time.txt" \
    "${REPO_ROOT}/scripts/run_pick_place_acceptance.sh" \
      --run-root "${run_path}" \
      --scenario normal \
      --execution-profile smoke_normal \
      --qualification-profile fast_v3 \
      --grasp-backend graspgenx \
      --sam3-url "http://127.0.0.1:${sam3_port}/sse" \
      --anyplace-url "http://127.0.0.1:${anyplace_port}/sse" \
      --graspgenx-url "http://127.0.0.1:${graspgenx_port}/sse" \
      2>&1 | tee "${RUN_ROOT}/run-${run_index}.stdout.log"
done
