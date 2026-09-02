#!/usr/bin/env bash
set -euo pipefail

# Operator-only Gazebo client.  The simulation server and acceptance executor
# remain independent so GUI/VNC latency cannot enter the qualification path.
if [[ -z "${GZ_PARTITION:-}" ]]; then
  echo "GZ_PARTITION must identify the already-running Gazebo server." >&2
  exit 2
fi
vglrun_bin="$(command -v vglrun || true)"
gz_bin="$(command -v gz || true)"
if [[ -z "${vglrun_bin}" ]]; then
  echo "VirtualGL (vglrun) is required for the GPU Gazebo client." >&2
  exit 2
fi
if [[ -z "${gz_bin}" ]]; then
  echo "Gazebo (gz) is required for the operator client." >&2
  exit 2
fi

# ``gz`` is a Ruby launcher with an ``/usr/bin/env ruby`` shebang.  Login
# shells on the GPU host may auto-activate a Conda environment whose Ruby and
# gems are ABI-incompatible with Gazebo's vendor extension.  Resolve the two
# executables before changing PATH (which also keeps test/operator overrides),
# then make the system Ruby authoritative for the launcher and discard only
# Ruby-specific package state.  CUDA, VirtualGL, ROS, and Gazebo paths remain
# available through the unchanged tail of PATH.
unset GEM_HOME GEM_PATH RUBYLIB RUBYOPT
export PATH="/usr/bin:/bin:${PATH}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_gui_config="${script_dir}/../extensions/gazebo/ros2_ws/src/openeta_rm75_robotiq2f85_sim/config/gazebo_operator_gui.config"
gui_config="${OPENETA_GAZEBO_GUI_CONFIG:-${default_gui_config}}"
if [[ ! -f "${gui_config}" ]]; then
  echo "Gazebo operator GUI config not found: ${gui_config}" >&2
  exit 2
fi

export DISPLAY="${OPENETA_GAZEBO_DISPLAY:-:3}"
export QT_QPA_PLATFORM=xcb
export QT_OPENGL=desktop
export __GL_FSAA_MODE="${OPENETA_GAZEBO_FSAA_MODE:-5}"
export __GL_ALLOW_FXAA_USAGE="${OPENETA_GAZEBO_ALLOW_FXAA:-1}"

wait_for_partition_server() {
  echo "Waiting for the Gazebo server on partition ${GZ_PARTITION}..." >&2
  while ! "${gz_bin}" service -l 2>/dev/null \
    | grep -Eq '^/gazebo/worlds$|^/world/[^/]+/control$'; do
    sleep 0.25
  done
}

focus_operator_view() {
  local camera_request
  camera_request='pose: {position: {x: -1.0, y: -1.1, z: 1.2}, orientation: {x: -0.07846, y: 0.20781, z: 0.34439, w: 0.91217}}'
  for _attempt in $(seq 1 60); do
    if "${gz_bin}" service \
      -s /gui/move_to/pose \
      --reqtype gz.msgs.GUICamera \
      --reptype gz.msgs.Boolean \
      --timeout 250 \
      --req "${camera_request}" >/dev/null 2>&1; then
      if command -v xdotool >/dev/null 2>&1; then
        local window_id
        window_id="$(xdotool search --onlyvisible --name 'Gazebo Sim' 2>/dev/null | tail -n 1 || true)"
        if [[ -n "${window_id}" ]]; then
          # Ask the window manager to maximize instead of assuming a fixed VNC
          # framebuffer size.  This keeps the render texture native-sized on
          # both 1080p and higher-resolution desktops.
          xdotool windowactivate --sync "${window_id}" \
            key --window "${window_id}" alt+F10 >/dev/null 2>&1 || true
        fi
      fi
      return 0
    fi
    sleep 0.25
  done
  echo "Gazebo GUI started, but its camera service did not become ready." >&2
}

wait_for_partition_server
focus_operator_view &
exec "${vglrun_bin}" \
  -d "${OPENETA_VGL_DISPLAY:-egl}" \
  -c "${OPENETA_VGL_TRANSPORT:-proxy}" \
  -fps "${OPENETA_GAZEBO_GUI_FPS:-30}" \
  -ms "${OPENETA_VGL_MSAA_SAMPLES:-8}" \
  "${gz_bin}" sim -g \
  --render-engine-gui ogre2 \
  --render-engine-gui-api-backend opengl \
  --gui-config "${gui_config}" \
  "$@"
