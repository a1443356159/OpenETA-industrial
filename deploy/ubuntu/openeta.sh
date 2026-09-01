#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/deploy/ubuntu/compose.yaml"

usage() {
  cat <<'EOF'
Usage: deploy/ubuntu/openeta.sh [--gui] [--dev] COMMAND [ARGS...]

Commands:
  build                 Build the CUDA/ROS/OpenETA image
  fetch-models          Download pinned model assets through the image toolchain
  config                Render and validate the Compose configuration
  shell                 Open a shell in the immutable image
  tui [ARGS...]         Start the interactive OpenETA TUI
  smoke-normal [ARGS]   Run normal without a Planner/VLM (two runs by default)
  agentic-normal [ARGS] Run normal with the configured Planner/VLM
  validate-assets       Validate mounted model assets and prepare writable cache
  test [PYTEST_ARGS...] Run repository tests inside the image
  exec COMMAND [...]    Run an arbitrary command after container initialization

Options:
  --gui   Forward the current X11/VNC display and NVIDIA graphics capability
  --dev   Bind the current checkout read-only at /workspace/openeta
EOF
}

gui=0
dev=0
while (( $# > 0 )); do
  case "$1" in
    --gui) gui=1; shift ;;
    --dev) dev=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) break ;;
  esac
done
(( $# > 0 )) || { usage >&2; exit 2; }
command_name="$1"
shift

command -v docker >/dev/null 2>&1 || {
  echo "Docker CLI is required" >&2
  exit 2
}
docker compose version >/dev/null 2>&1 || {
  echo "Docker Compose v2 is required" >&2
  exit 2
}

export OPENETA_UID="${OPENETA_UID:-$(id -u)}"
export OPENETA_GID="${OPENETA_GID:-$(id -g)}"
if [[ -z "${OPENETA_REVISION:-}" ]]; then
  OPENETA_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf local)"
  export OPENETA_REVISION
fi
export OPENETA_MODEL_ROOT="${OPENETA_MODEL_ROOT:-${REPO_ROOT}/.cache/docker/models}"
export OPENETA_DOCKER_STATE_ROOT="${OPENETA_DOCKER_STATE_ROOT:-${REPO_ROOT}/.cache/docker/state}"
if [[ -z "${OPENETA_PROVIDER_ENV_FILE:-}" ]]; then
  if [[ -r "${REPO_ROOT}/.env" ]]; then
    export OPENETA_PROVIDER_ENV_FILE="${REPO_ROOT}/.env"
  else
    export OPENETA_PROVIDER_ENV_FILE="${REPO_ROOT}/.env.example"
  fi
fi
[[ -r "${OPENETA_PROVIDER_ENV_FILE}" ]] || {
  echo "provider env file is not readable: ${OPENETA_PROVIDER_ENV_FILE}" >&2
  exit 2
}
mkdir -p "${OPENETA_MODEL_ROOT}" "${OPENETA_DOCKER_STATE_ROOT}"

compose=(
  docker compose
  --env-file /dev/null
  --project-directory "${REPO_ROOT}"
  -f "${COMPOSE_FILE}"
)
case "${command_name}" in
  build)
    (( gui == 0 && dev == 0 )) || {
      echo "--gui and --dev do not apply to build" >&2
      exit 2
    }
    exec "${compose[@]}" build "$@"
    ;;
  config)
    exec "${compose[@]}" config "$@"
    ;;
  fetch-models)
    (( gui == 0 && dev == 0 && $# == 0 )) || {
      echo "fetch-models does not accept --gui, --dev, or extra arguments" >&2
      exit 2
    }
    exec "${compose[@]}" run --rm \
      --volume "${OPENETA_MODEL_ROOT}:/srv/openeta/model-download:rw" \
      --env OPENETA_MODEL_ROOT=/srv/openeta/model-download \
      openeta exec bash /opt/openeta/src/deploy/ubuntu/fetch_models.sh /srv/openeta
    ;;
esac

run_options=(--rm)
if (( dev == 1 )); then
  run_options+=(
    --volume "${REPO_ROOT}:/workspace/openeta:ro"
    --env OPENETA_SOURCE_ROOT=/workspace/openeta
    --env PYTHONPATH=/workspace/openeta:/opt/openeta/src
  )
fi

if (( gui == 1 )); then
  [[ -n "${DISPLAY:-}" ]] || {
    echo "--gui requires DISPLAY (for example :3 for the VNC desktop)" >&2
    exit 2
  }
  [[ -d /tmp/.X11-unix ]] || {
    echo "--gui requires the host X11 socket directory /tmp/.X11-unix" >&2
    exit 2
  }
  xauthority="${OPENETA_XAUTHORITY:-${XAUTHORITY:-}}"
  if [[ -z "${xauthority}" && -r "${HOME}/.Xauthority" ]]; then
    xauthority="${HOME}/.Xauthority"
  fi
  [[ -n "${xauthority}" && -r "${xauthority}" ]] || {
    echo "--gui requires a readable Xauthority file; set OPENETA_XAUTHORITY" >&2
    exit 2
  }
  run_options+=(
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw
    --volume "${xauthority}:/run/openeta/xauthority:ro"
    --env "DISPLAY=${DISPLAY}"
    --env XAUTHORITY=/run/openeta/xauthority
    --env QT_QPA_PLATFORM=xcb
    --env QT_X11_NO_MITSHM=1
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
  )
fi

case "${command_name}" in
  shell|tui|smoke-normal|agentic-normal|validate-assets|test)
    exec "${compose[@]}" run "${run_options[@]}" openeta "${command_name}" "$@"
    ;;
  exec)
    (( $# > 0 )) || { echo "exec requires a command" >&2; exit 2; }
    exec "${compose[@]}" run "${run_options[@]}" openeta exec "$@"
    ;;
  *)
    echo "unknown command: ${command_name}" >&2
    usage >&2
    exit 2
    ;;
esac
