"""Render the detachable-only RM75 spawn model without changing physics mode.

Gazebo's URDF importer is retained for the default physics profile.  DART's
cross-model detachable joint needs a fixed root, however, so the opt-in
fallback is rendered to one short-lived SDF file where that single boundary is
expressed explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


class DetachableSdfError(RuntimeError):
    """The detached-only production SDF could not be rendered safely."""


def _detachable_plugins(model: ET.Element) -> list[ET.Element]:
    return [
        plugin
        for plugin in model.findall(".//plugin")
        if plugin.get("filename") == "gz-sim-detachable-joint-system"
        and plugin.get("name") == "gz::sim::systems::DetachableJoint"
    ]


def prepare_detachable_sdf(
    root: ET.Element, *, fixed_root: bool = True, parent_link: str = "gripper_mount_link"
) -> ET.Element:
    """Validate and minimally amend a production xacro-rendered SDF tree."""

    model = root.find("model")
    if model is None:
        raise DetachableSdfError("rendered SDF has no production robot model")
    plugins = _detachable_plugins(model)
    expected = {
        ("m3_target", "target_link"),
        ("m3_distractor", "distractor_link"),
    }
    found = {
        (plugin.findtext("child_model"), plugin.findtext("child_link"))
        for plugin in plugins
    }
    if found != expected or len(plugins) != 2:
        raise DetachableSdfError("rendered SDF must contain exactly target and distractor joints")
    if parent_link not in {"gripper_mount_link", "link_7"}:
        raise DetachableSdfError("unsupported detachable parent link")
    if any(plugin.findtext("parent_link") != parent_link for plugin in plugins):
        raise DetachableSdfError(f"detachable parent link must be {parent_link}")

    # `self_collide=true` triggers a DART mesh collision assertion on the
    # real RM75.  Keep this property explicit in the generated fallback SDF,
    # both as a guard against upstream defaults and as a reviewable boundary.
    self_collide = model.find("self_collide")
    if self_collide is None:
        self_collide = ET.Element("self_collide")
        model.insert(0, self_collide)
    self_collide.text = "false"

    base_link = next((link for link in model.findall("link") if link.get("name") == "base_link"), None)
    if base_link is None:
        raise DetachableSdfError("rendered SDF has no base_link for detachable fixed root")
    for joint in list(model.findall("joint")):
        if joint.get("name") == "openeta_detachable_world_to_base":
            model.remove(joint)
    if fixed_root:
        root_joint = ET.Element(
            "joint", {"name": "openeta_detachable_world_to_base", "type": "fixed"}
        )
        ET.SubElement(root_joint, "parent").text = "world"
        ET.SubElement(root_joint, "child").text = "base_link"
        model.append(root_joint)
    return root


def render_detachable_sdf(
    *,
    xacro_file: Path,
    environment: dict[str, str] | None = None,
    xacro_executable: str = "xacro",
    gz_executable: str = "gz",
    directory: Path | None = None,
    fixed_root: bool = True,
    parent_link: str = "gripper_mount_link",
) -> Path:
    """Render a validated temporary SDF and return the caller-owned path."""

    xacro = shutil.which(xacro_executable) or xacro_executable
    gz = shutil.which(gz_executable) or gz_executable
    env = dict(os.environ if environment is None else environment)
    try:
        urdf = subprocess.run(
            [
                xacro, str(xacro_file), "attachment_mode:=detachable",
                f"detachable_parent_link:={parent_link}",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=60.0,
        )
    except OSError as exc:
        raise DetachableSdfError("xacro is unavailable for detachable SDF rendering") from exc
    if urdf.returncode != 0:
        raise DetachableSdfError(f"production xacro failed: {urdf.stderr[-800:]}")
    urdf_fd, raw_urdf = tempfile.mkstemp(
        prefix="openeta-m3-detachable-", suffix=".urdf", dir=directory
    )
    try:
        with os.fdopen(urdf_fd, "w", encoding="utf-8") as rendered_urdf:
            rendered_urdf.write(urdf.stdout)
        try:
            converted = subprocess.run(
                [gz, "sdf", "-p", raw_urdf],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=60.0,
            )
        except OSError as exc:
            raise DetachableSdfError("gz is unavailable for detachable SDF rendering") from exc
    finally:
        Path(raw_urdf).unlink(missing_ok=True)
    if converted.returncode != 0:
        raise DetachableSdfError(f"URDF-to-SDF conversion failed: {converted.stderr[-800:]}")
    try:
        root = prepare_detachable_sdf(
            ET.fromstring(converted.stdout), fixed_root=fixed_root, parent_link=parent_link
        )
    except ET.ParseError as exc:
        raise DetachableSdfError("URDF-to-SDF conversion returned invalid XML") from exc
    fd, raw_path = tempfile.mkstemp(
        prefix="openeta-m3-detachable-", suffix=".sdf", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as rendered:
            rendered.write(ET.tostring(root, encoding="unicode"))
            rendered.write("\n")
    except Exception:
        Path(raw_path).unlink(missing_ok=True)
        raise
    return Path(raw_path)
