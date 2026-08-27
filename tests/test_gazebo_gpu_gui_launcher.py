from __future__ import annotations

from pathlib import Path


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

    assert "exec vglrun" in text
    assert '-c "${OPENETA_VGL_TRANSPORT:-proxy}"' in text
    assert '-fps "${OPENETA_GAZEBO_GUI_FPS:-30}"' in text
    assert '-ms "${OPENETA_VGL_MSAA_SAMPLES:-8}"' in text
    assert 'export DISPLAY="${OPENETA_GAZEBO_DISPLAY:-:3}"' in text
    assert 'export QT_OPENGL=desktop' in text
    assert 'export __GL_FSAA_MODE="${OPENETA_GAZEBO_FSAA_MODE:-5}"' in text
    assert 'key --window "${window_id}" alt+F10' in text
    assert "--render-engine-gui ogre2" in text
    assert "--render-engine-gui-api-backend opengl" in text
    assert '--gui-config "${gui_config}"' in text
    assert "/gui/move_to/pose" in text


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
