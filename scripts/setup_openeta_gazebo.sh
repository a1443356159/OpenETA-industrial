#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../config/runtime/gazebo_rm75_robotiq2f85.env
source "${REPO_DIR}/config/runtime/gazebo_rm75_robotiq2f85.env"

UV_VERSION="0.8.13"
DO_APT=1
DO_PYTHON=1
DO_ROS=1
CHECK_ONLY=0

usage() {
  echo "Usage: $0 [--check-only] [--no-apt] [--python-only|--ros-only]"
}
while (($#)); do
  case "$1" in
    --check-only) CHECK_ONLY=1 ;;
    --no-apt) DO_APT=0 ;;
    --python-only) DO_PYTHON=1; DO_ROS=0 ;;
    --ros-only) DO_PYTHON=0; DO_ROS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "USAGE_ERROR: unknown option: $1" >&2; usage >&2; exit 64 ;;
  esac
  shift
done

if [[ "$(uname -m)" != x86_64 ]]; then
  echo "PLATFORM_NOT_SUPPORTED: Ubuntu 24.04 amd64 is required" >&2
  exit 2
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04* ]]; then
  echo "PLATFORM_NOT_SUPPORTED: expected Ubuntu 24.04, found ${ID:-unknown} ${VERSION_ID:-unknown}" >&2
  exit 2
fi

if [[ "${CHECK_ONLY}" == 1 ]]; then
  scope=()
  if [[ "${DO_PYTHON}" == 0 ]]; then scope=(--ros-only); fi
  if [[ "${DO_ROS}" == 0 ]]; then scope=(--python-only); fi
  exec "${SCRIPT_DIR}/check_openeta_gazebo.sh" "${scope[@]}"
fi

if ((EUID == 0)); then
  APT=(apt-get)
  INSTALL=(install)
else
  if [[ "${DO_APT}" == 1 || "${DO_ROS}" == 1 ]] && ! command -v sudo >/dev/null 2>&1; then
    echo "APT_NOT_READY: sudo is required for system package installation; use --no-apt" >&2
    exit 2
  fi
  if command -v sudo >/dev/null 2>&1; then
    APT=(sudo apt-get)
    INSTALL=(sudo install)
  else
    APT=(apt-get)
    INSTALL=(install)
  fi
fi

BASE_PACKAGES=(
  git curl ca-certificates gnupg build-essential cmake python3.12 python3.12-venv
  python3-pip python3-colcon-common-extensions python3-rosdep
)
ROS_PACKAGES=(
  ros-jazzy-ros-base ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge
  ros-jazzy-gz-ros2-control ros-jazzy-ros2-control ros-jazzy-ros2-controllers
  ros-jazzy-controller-manager ros-jazzy-joint-state-broadcaster
  ros-jazzy-joint-trajectory-controller ros-jazzy-parallel-gripper-controller
  ros-jazzy-moveit ros-jazzy-moveit-ros-move-group ros-jazzy-moveit-configs-utils
  ros-jazzy-trac-ik-kinematics-plugin ros-jazzy-pick-ik
  ros-jazzy-xacro ros-jazzy-robot-state-publisher ros-jazzy-tf2-ros
  ros-jazzy-control-msgs ros-jazzy-geometry-msgs ros-jazzy-sensor-msgs
)

if [[ "${DO_APT}" == 1 ]]; then
  "${APT[@]}" update
  "${APT[@]}" install -y "${BASE_PACKAGES[@]}"
  if [[ "${DO_ROS}" == 1 ]]; then
    key_file="$(mktemp)"
    trap 'rm -f -- "${key_file:-}"' EXIT
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o "${key_file}"
    "${INSTALL[@]}" -m 0644 "${key_file}" /usr/share/keyrings/ros-archive-keyring.gpg
    source_file="$(mktemp)"
    printf 'deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main\n' >"${source_file}"
    "${INSTALL[@]}" -m 0644 "${source_file}" /etc/apt/sources.list.d/ros2.list
    rm -f -- "${source_file}" "${key_file}"
    trap - EXIT
    "${APT[@]}" update
    "${APT[@]}" install -y "${ROS_PACKAGES[@]}"
  fi
fi

if [[ "${DO_PYTHON}" == 1 ]]; then
  UV_ENV="${REPO_DIR}/.tools/uv-venv"
  if [[ ! -x "${UV_ENV}/bin/uv" ]] || [[ "$(${UV_ENV}/bin/uv --version | awk '{print $2}')" != "${UV_VERSION}" ]]; then
    python3.12 -m venv "${UV_ENV}"
    "${UV_ENV}/bin/python" -m pip install --disable-pip-version-check "uv==${UV_VERSION}"
  fi
  UV_PROJECT_ENVIRONMENT="${REPO_DIR}/.venv" "${UV_ENV}/bin/uv" sync --frozen --extra dev
fi

if [[ "${DO_ROS}" == 1 ]]; then
  if [[ ! -r /opt/ros/jazzy/setup.bash ]]; then
    echo "ROS_NOT_READY: /opt/ros/jazzy/setup.bash is missing; omit --no-apt or install Jazzy" >&2
    exit 3
  fi
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
  mkdir -p "${ROS_HOME}"
  if [[ ! -r /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    if ((EUID == 0)); then rosdep init; else sudo rosdep init; fi
  fi
  rosdep update --rosdistro jazzy
  rosdep install --from-paths "${OPENETA_GAZEBO_ROS2_WS}/src" --ignore-src \
    --rosdistro jazzy -y
  OPENETA_SKIP_ROSDEP=1 bash "${OPENETA_GAZEBO_ROS2_WS}/build.sh"
  set +u
  # shellcheck disable=SC1090
  source "${OPENETA_GAZEBO_ROS2_WS}/install/setup.bash"
  set -u
fi

REPORT="${REPO_DIR}/config/runtime/gazebo_rm75_robotiq2f85.versions.local.yaml"
{
  echo "generated_at_utc: \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
  echo "openeta_git_commit: \"$(git -C "${REPO_DIR}" rev-parse HEAD)\""
  echo "os: \"${PRETTY_NAME}\""
  echo "architecture: \"$(dpkg --print-architecture)\""
  echo "python: \"$(python3.12 --version 2>&1 | awk '{print $2}')\""
  if [[ -x "${REPO_DIR}/.tools/uv-venv/bin/uv" ]]; then
    echo "uv: \"$(${REPO_DIR}/.tools/uv-venv/bin/uv --version | awk '{print $2}')\""
  fi
  echo "ros_distro: jazzy"
  echo "gazebo: \"$(gz sim --versions 2>/dev/null | head -1 || echo unavailable)\""
  echo "mcp_port: ${OPENETA_MCP_PORT}"
  echo "ros_domain_id: ${ROS_DOMAIN_ID}"
  echo "ros_packages:"
  for package_name in "${ROS_PACKAGES[@]}"; do
    version="$(dpkg-query -W -f='${Version}' "${package_name}" 2>/dev/null || echo not-installed)"
    printf '  %s: "%s"\n' "${package_name}" "${version}"
  done
} >"${REPORT}"

scope=()
if [[ "${DO_PYTHON}" == 0 ]]; then scope=(--ros-only); fi
if [[ "${DO_ROS}" == 0 ]]; then scope=(--python-only); fi
"${SCRIPT_DIR}/check_openeta_gazebo.sh" "${scope[@]}"
echo "OPENETA_GAZEBO_SETUP_OK version_report=${REPORT}"
