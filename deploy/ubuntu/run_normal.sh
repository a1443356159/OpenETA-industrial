#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_normal.sh smoke-normal|agentic-normal [--runs N] [--scenario NAME]
                     [--task-variant NAME] [--run-root PATH]
                     [-- ACCEPTANCE_ARGS...]

Scenarios:
  normal                         single-item control-chain scene
  multi_normal                   task-neutral industrial sorting scene
  multi_normal_random_12345      seeded randomized industrial sorting scene

`agentic-normal` defaults to `multi_normal`; `smoke-normal` defaults to and
only supports `normal`.  `--task-variant` selects a private verification
fixture for a natural-language work order.  It never writes a task into the
physical scene or planner context.
EOF
  exit 2
}

(( $# >= 1 )) || usage
mode="$1"
shift
case "${mode}" in
  smoke-normal) execution_profile="smoke_normal" ;;
  agentic-normal) execution_profile="agentic_normal" ;;
  *) usage ;;
esac

runs="${OPENETA_ACCEPTANCE_RUNS:-2}"
scenario="${OPENETA_ACCEPTANCE_SCENARIO:-}"
task_variant="${OPENETA_ACCEPTANCE_TASK_VARIANT:-}"
batch_root=""
extra_args=()
while (( $# > 0 )); do
  case "$1" in
    --runs)
      (( $# >= 2 )) || usage
      runs="$2"
      shift 2
      ;;
    --scenario)
      (( $# >= 2 )) || usage
      scenario="$2"
      shift 2
      ;;
    --task-variant)
      (( $# >= 2 )) || usage
      task_variant="$2"
      shift 2
      ;;
    --run-root)
      (( $# >= 2 )) || usage
      batch_root="$2"
      shift 2
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    *) usage ;;
  esac
done

[[ "${runs}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--runs must be a positive integer" >&2
  exit 2
}
if [[ -z "${scenario}" ]]; then
  if [[ "${mode}" == "agentic-normal" ]]; then
    scenario="multi_normal"
  else
    scenario="normal"
  fi
fi
case "${scenario}" in
  normal|multi_normal|multi_normal_random_12345) ;;
  *) echo "unsupported scenario: ${scenario}" >&2; exit 2 ;;
esac
if [[ "${mode}" == "smoke-normal" && "${scenario}" != "normal" ]]; then
  echo "smoke-normal requires scenario=normal" >&2
  exit 2
fi
if [[ -n "${task_variant}" && "${scenario}" == "normal" ]]; then
  echo "--task-variant requires a multi_normal scenario" >&2
  exit 2
fi

readonly SOURCE_ROOT="${OPENETA_SOURCE_ROOT:-/opt/openeta/src}"
readonly STATE_ROOT="${OPENETA_STATE_ROOT:-/srv/openeta/state}"
readonly RUNS_ROOT="${OPENETA_RUN_ROOT:-${STATE_ROOT}/runs}"
readonly APP_PYTHON="${OPENETA_PYTHON_EXECUTABLE:-/opt/openeta/venvs/openeta/bin/python}"

if [[ -z "${batch_root}" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  batch_root="${RUNS_ROOT}/${mode}-${stamp}-$$"
fi
if [[ -e "${batch_root}" ]]; then
  echo "refusing to overwrite existing run root: ${batch_root}" >&2
  exit 2
fi
mkdir -p "${batch_root}"

if [[ "${mode}" == "smoke-normal" ]]; then
  unset ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy
fi
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="${NO_PROXY}"

# shellcheck disable=SC1091
source /opt/openeta/src/deploy/ubuntu/model_services.sh
cleanup() {
  openeta_stop_model_services
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

openeta_prepare_model_assets "${batch_root}/model-assets.json" \
  >"${batch_root}/model-assets.stdout.json"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader >"${batch_root}/gpu.txt"
openeta_start_model_services "${batch_root}/mcp-state"

for ((run_index = 1; run_index <= runs; run_index++)); do
  run_path="${batch_root}/run-${run_index}"
  acceptance=(
    "${SOURCE_ROOT}/scripts/run_pick_place_acceptance.sh"
    --run-root "${run_path}"
    --scenario "${scenario}"
    --execution-profile "${execution_profile}"
    --qualification-profile "${OPENETA_QUALIFICATION_PROFILE:-fast_v3}"
    --grasp-backend graspgenx
    --sam3-url "http://127.0.0.1:${OPENETA_SAM3_PORT:-8773}/sse"
    --anyplace-url "http://127.0.0.1:${OPENETA_ANYPLACE_PORT:-8775}/sse"
    --graspgenx-url "http://127.0.0.1:${OPENETA_GRASPGENX_PORT:-8778}/sse"
    "${extra_args[@]}"
  )
  if [[ -n "${task_variant}" ]]; then
    acceptance+=(--task-variant "${task_variant}")
  fi
  /usr/bin/time \
    --format='wall_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kb=%M' \
    --output="${batch_root}/run-${run_index}-time.txt" \
    "${acceptance[@]}" 2>&1 | tee "${batch_root}/run-${run_index}.stdout.log"
done

printf 'OPENETA_DOCKER_ACCEPTANCE_OK profile=%s runs=%s run_root=%s\n' \
  "${execution_profile}" "${runs}" "${batch_root}"
