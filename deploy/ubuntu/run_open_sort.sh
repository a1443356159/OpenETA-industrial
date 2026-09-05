#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_open_sort.sh [--run-root PATH] [--scenario NAME] [-- OPEN_SORT_ARGS...]

Start one task-neutral, interactive multi-object Gazebo sorting session.  The
operator supplies the natural-language work order in the TUI; this command
does not create a fixed acceptance assignment or report task PASS on exit.

Scenarios:
  multi_normal                   task-neutral industrial sorting scene
  multi_normal_random_12345      seeded randomized industrial sorting scene
EOF
  exit 2
}

readonly SOURCE_ROOT="${OPENETA_SOURCE_ROOT:-/opt/openeta/src}"
readonly STATE_ROOT="${OPENETA_STATE_ROOT:-/srv/openeta/state}"
readonly RUNS_ROOT="${OPENETA_RUN_ROOT:-${STATE_ROOT}/runs}"

run_root=""
forward=()
while (( $# > 0 )); do
  case "$1" in
    --run-root)
      (( $# >= 2 )) || usage
      run_root="$2"
      forward+=("$1" "$2")
      shift 2
      ;;
    --scenario)
      (( $# >= 2 )) || usage
      case "$2" in
        multi_normal|multi_normal_random_12345) ;;
        *) echo "unsupported open-sort scenario: $2" >&2; exit 2 ;;
      esac
      forward+=("$1" "$2")
      shift 2
      ;;
    --)
      shift
      forward+=("$@")
      break
      ;;
    -h|--help)
      usage
      ;;
    *)
      # The Python launcher owns provider, model-service URL, grasp backend,
      # and qualification-profile validation.  Keep unfamiliar flags intact
      # instead of duplicating that public interface here.
      forward+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${run_root}" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  run_root="${RUNS_ROOT}/open-sort-${stamp}-$$"
  forward+=(--run-root "${run_root}")
fi
if [[ -e "${run_root}" ]]; then
  echo "refusing to overwrite existing run root: ${run_root}" >&2
  exit 2
fi
mkdir -p "$(dirname -- "${run_root}")"
# The Python launcher creates ``run_root`` itself to protect its evidence from
# accidental reuse.  Keep service logs in a sibling directory so model startup
# never pre-creates the directory that the launcher must own atomically.
service_root="${run_root}.services"
mkdir -p "${service_root}"

export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="${NO_PROXY}"

# shellcheck disable=SC1091
source "${SOURCE_ROOT}/deploy/ubuntu/model_services.sh"
cleanup() {
  openeta_stop_model_services
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

openeta_prepare_model_assets "${service_root}/model-assets.json" \
  >"${service_root}/model-assets.stdout.json"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader >"${service_root}/gpu.txt"
openeta_start_model_services "${service_root}/mcp-state"

"${SOURCE_ROOT}/scripts/run_open_sort_gazebo_tui.sh" \
  --grasp-backend graspgenx \
  --qualification-profile "${OPENETA_QUALIFICATION_PROFILE:-fast_v3}" \
  --sam3-url "http://127.0.0.1:${OPENETA_SAM3_PORT:-8773}/sse" \
  --anyplace-url "http://127.0.0.1:${OPENETA_ANYPLACE_PORT:-8775}/sse" \
  --graspgenx-url "http://127.0.0.1:${OPENETA_GRASPGENX_PORT:-8778}/sse" \
  "${forward[@]}"
