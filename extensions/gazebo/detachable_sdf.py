"""Render the approved fixed-root SDF used by M3's stock DetachableJoint.

The generated file exists only for the launched Gazebo process.  It contains
no custom physics plugin: it merely gives DART the fixed robot root required
by the stock cross-model fixed joint.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


class DetachableSdfError(RuntimeError):
    """The approved DetachableJoint SDF could not be produced safely."""


def _plugins(model: ET.Element) -> list[ET.Element]:
    return [
        plugin for plugin in model.findall(".//plugin")
        if plugin.get("filename") == "gz-sim-detachable-joint-system"
        and plugin.get("name") == "gz::sim::systems::DetachableJoint"
    ]


def prepare_detachable_sdf(root: ET.Element, *, dart_compatible: bool = True) -> ET.Element:
    """Validate the one approved joint and add DART-only fixed-root details."""

    model = root.find("model")
    if model is None:
        raise DetachableSdfError("rendered SDF has no robot model")
    plugins = _plugins(model)
    if len(plugins) != 1:
        raise DetachableSdfError("rendered SDF must contain exactly one DetachableJoint")
    plugin = plugins[0]
    expected = {
        "parent_link": "gripper_mount_link",
        "child_model": "m3_target",
        "child_link": "target_link",
        "attach_topic": "/m3/detachable_joint/target/attach",
        "detach_topic": "/m3/detachable_joint/target/detach",
        "output_topic": "/m3/detachable_joint/target/state",
    }
    actual = {key: plugin.findtext(key) for key in expected}
    if actual != expected:
        raise DetachableSdfError("rendered SDF DetachableJoint topology is not approved")
    if model.find("link[@name='base_link']") is None:
        raise DetachableSdfError("rendered SDF has no base_link for a fixed root")
    for joint in list(model.findall("joint[@name='openeta_m3_world_to_base']")):
        model.remove(joint)
    root_joint = ET.Element("joint", {"name": "openeta_m3_world_to_base", "type": "fixed"})
    ET.SubElement(root_joint, "parent").text = "world"
    ET.SubElement(root_joint, "child").text = "base_link"
    model.append(root_joint)
    if dart_compatible:
        # Required only for the documented DART mesh-collision limitation.
        self_collide = model.find("self_collide")
        if self_collide is None:
            self_collide = ET.Element("self_collide")
            model.insert(0, self_collide)
        self_collide.text = "false"
    return root


def render_detachable_sdf(
    *,
    xacro_file: Path,
    environment: dict[str, str] | None = None,
    xacro_executable: str = "xacro",
    gz_executable: str = "gz",
    directory: Path | None = None,
) -> Path:
    """Render a caller-owned temporary fixed-root SDF or raise fail-closed."""

    env = dict(os.environ if environment is None else environment)
    xacro = shutil.which(xacro_executable) or xacro_executable
    gz = shutil.which(gz_executable) or gz_executable
    try:
        rendered = subprocess.run(
            [xacro, str(xacro_file)], capture_output=True, text=True, env=env,
            timeout=60.0, check=False,
        )
    except OSError as exc:
        raise DetachableSdfError("M3_DART_UNSUPPORTED: xacro is unavailable") from exc
    if rendered.returncode:
        raise DetachableSdfError(f"M3_DART_UNSUPPORTED: xacro failed: {rendered.stderr[-800:]}")
    urdf_fd, urdf_name = tempfile.mkstemp(prefix="openeta-m3-", suffix=".urdf", dir=directory)
    try:
        with os.fdopen(urdf_fd, "w", encoding="utf-8") as stream:
            stream.write(rendered.stdout)
        try:
            converted = subprocess.run(
                [gz, "sdf", "-p", urdf_name], capture_output=True, text=True, env=env,
                timeout=60.0, check=False,
            )
        except OSError as exc:
            raise DetachableSdfError("M3_DART_UNSUPPORTED: gz is unavailable") from exc
    finally:
        Path(urdf_name).unlink(missing_ok=True)
    if converted.returncode:
        raise DetachableSdfError(
            f"M3_DART_UNSUPPORTED: URDF-to-SDF conversion failed: {converted.stderr[-800:]}"
        )
    try:
        root = prepare_detachable_sdf(ET.fromstring(converted.stdout), dart_compatible=True)
    except ET.ParseError as exc:
        raise DetachableSdfError("M3_DART_UNSUPPORTED: invalid converted SDF") from exc
    descriptor, sdf_name = tempfile.mkstemp(prefix="openeta-m3-", suffix=".sdf", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(ET.tostring(root, encoding="unicode"))
            stream.write("\n")
    except Exception:
        Path(sdf_name).unlink(missing_ok=True)
        raise
    return Path(sdf_name)
