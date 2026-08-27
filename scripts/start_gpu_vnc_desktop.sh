#!/usr/bin/env bash
set -euo pipefail

# TigerVNC supplies the 2D X display.  VirtualGL redirects OpenGL created by
# XFCE and every application launched from that session to the NVIDIA EGL
# device, while the resulting windows remain visible in the VNC framebuffer.
if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY must be supplied by the VNC server." >&2
  exit 2
fi
if ! command -v vglrun >/dev/null 2>&1; then
  echo "VirtualGL (vglrun) is required for the GPU VNC desktop." >&2
  exit 2
fi
if [[ ! -x /usr/bin/startxfce4 ]]; then
  echo "XFCE startup command is unavailable: /usr/bin/startxfce4" >&2
  exit 2
fi

export QT_QPA_PLATFORM=xcb
export QT_OPENGL=desktop
export __GL_ALLOW_FXAA_USAGE="${OPENETA_VNC_ALLOW_FXAA:-1}"
export __GL_FSAA_MODE="${OPENETA_VNC_FSAA_MODE:-5}"

# Window-manager mode is required when VirtualGL wraps a complete desktop.
# Proxy transport is local to the VNC X server, and a bounded presentation
# rate avoids spending acceptance CPU on frames that a 60 Hz desktop cannot
# display usefully.
exec vglrun \
  +wm \
  -d "${OPENETA_VGL_DISPLAY:-egl}" \
  -c "${OPENETA_VGL_TRANSPORT:-proxy}" \
  -fps "${OPENETA_VNC_GPU_FPS:-30}" \
  /usr/bin/startxfce4
