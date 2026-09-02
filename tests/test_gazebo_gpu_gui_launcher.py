from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_gazebo_gpu_gui.sh"
VNC_LAUNCHER = ROOT / "scripts/start_gpu_vnc_desktop.sh"
GUI_CONFIG = (
    ROOT
    / "extensions/gazebo/ros2_ws/src/openeta_rm75_robotiq2f85_sim/config"
    / "gazebo_operator_gui.config"
)


def test_gpu_gui_launcher_defaults_to_virtualgl_hd_antialiasing() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'exec "${vglrun_bin}"' in text
    assert '-c "${OPENETA_VGL_TRANSPORT:-proxy}"' in text
    assert '-fps "${OPENETA_GAZEBO_GUI_FPS:-30}"' in text
    assert '-ms "${OPENETA_VGL_MSAA_SAMPLES:-8}"' in text
    assert 'export DISPLAY="${OPENETA_GAZEBO_DISPLAY:-:3}"' in text
    assert 'export QT_OPENGL=desktop' in text
    assert 'export __GL_FSAA_MODE="${OPENETA_GAZEBO_FSAA_MODE:-5}"' in text
    assert "xdotool search --name '^Gazebo Sim$'" in text
    assert 'xdotool windowmap --sync "${window_id}"' in text
    assert 'xdotool windowraise "${window_id}"' in text
    assert 'xdotool windowactivate --sync "${window_id}"' in text
    assert "_NET_WM_STATE_MAXIMIZED_VERT" in text
    assert 'key --window "${window_id}" alt+F10' in text
    assert "--render-engine-gui ogre2" in text
    assert "--render-engine-gui-api-backend opengl" in text
    assert '--gui-config "${gui_config}"' in text
    assert "/gui/move_to/pose" in text
    assert "wait_for_partition_server" in text
    assert '"${gz_bin}" service -l' in text
    assert "unset GEM_HOME GEM_PATH RUBYLIB RUBYOPT" in text
    assert 'export PATH="/usr/bin:/bin:${PATH}"' in text
    assert text.index("wait_for_partition_server\nfocus_operator_view &") < text.index(
        'exec "${vglrun_bin}"'
    )


def test_vnc_desktop_wraps_the_complete_xfce_session_with_virtualgl() -> None:
    text = VNC_LAUNCHER.read_text(encoding="utf-8")

    assert 'if [[ -z "${DISPLAY:-}" ]]' in text
    assert "exec vglrun" in text
    assert "+wm" in text
    assert '-d "${OPENETA_VGL_DISPLAY:-egl}"' in text
    assert '-c "${OPENETA_VGL_TRANSPORT:-proxy}"' in text
    assert '-fps "${OPENETA_VNC_GPU_FPS:-30}"' in text
    assert "/usr/bin/startxfce4" in text
    assert "vncserver" not in text


def test_gpu_gui_launcher_requires_an_existing_partition() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'if [[ -z "${GZ_PARTITION:-}" ]]' in text
    assert "acceptance executor" in text


def test_gpu_gui_waits_for_its_partition_server_before_starting_client(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "gz-service-count"
    client_log = tmp_path / "client.log"
    fake_gz = fake_bin / "gz"
    fake_gz.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-} ${2:-}" == "service -l" ]]; then
  count=0
  if [[ -f "${FAKE_GZ_COUNTER}" ]]; then
    count="$(<"${FAKE_GZ_COUNTER}")"
  fi
  count="$((count + 1))"
  printf '%s\n' "${count}" >"${FAKE_GZ_COUNTER}"
  if (( count >= 3 )); then
    printf '%s\n' '/gazebo/worlds'
  fi
fi
""",
        encoding="utf-8",
    )
    fake_gz.chmod(0o755)
    fake_vglrun = fake_bin / "vglrun"
    fake_vglrun.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s:%s:%s:%s\n' \
  "${GZ_PARTITION}" \
  "$(<"${FAKE_GZ_COUNTER}")" \
  "${PATH%%:*}" \
  "${GEM_HOME-unset}" >"${FAKE_VGL_LOG}"
""",
        encoding="utf-8",
    )
    fake_vglrun.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GZ_PARTITION": "delayed-formal-case",
        "FAKE_GZ_COUNTER": str(counter),
        "FAKE_VGL_LOG": str(client_log),
        "OPENETA_GAZEBO_GUI_CONFIG": str(GUI_CONFIG),
        "OPENETA_GAZEBO_DISPLAY": ":999",
        "GEM_HOME": "/incompatible/conda/gems",
    }

    result = subprocess.run(
        [str(LAUNCHER)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert client_log.read_text(encoding="utf-8").strip() == (
        "delayed-formal-case:3:/usr/bin:unset"
    )


def test_operator_gui_uses_full_viewport_and_explicit_render_quality() -> None:
    text = GUI_CONFIG.read_text(encoding="utf-8")

    assert '<plugin filename="MinimalScene" name="3D View">' in text
    assert "<engine>ogre2</engine>" in text
    assert "<graphics_api>opengl</graphics_api>" in text
    assert "<anti_aliasing>8</anti_aliasing>" in text
    assert "<horizontal_fov>75</horizontal_fov>" in text
    assert '<plugin filename="EntityTree"' not in text
    assert '<plugin filename="ComponentInspector"' not in text
    assert '<plugin filename="WorldControl"' not in text
    assert "<start_paused>" not in text
