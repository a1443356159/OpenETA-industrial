"""Render the approved fixed-root SDF used by native-grasp's stock DetachableJoint.

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
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET


class DetachableSdfError(RuntimeError):
    """The approved DetachableJoint SDF could not be produced safely."""


ROBOT_COLLISION_FILTER_MASK = 0x0001


def _plugins(model: ET.Element) -> list[ET.Element]:
    return [
        plugin for plugin in model.findall(".//plugin")
        if plugin.get("filename") == "gz-sim-detachable-joint-system"
        and plugin.get("name") == "gz::sim::systems::DetachableJoint"
    ]


def _normalize_contact_topics(model: ET.Element) -> None:
    """Replace URDF converter placeholders with each sensor's real topic.

    ``gz sdf -p`` adds ``<contact><topic>__default_topic__</topic>`` to
    contact sensors even when the SDF sensor already has an explicit topic.
    SDFormat 1.10 requires that nested field for contact sensors, so leaving
    the placeholder prevents Gazebo from publishing native-grasp's approved per-pad
    topics.  Keep the generic sensor topic and contact-specific topic equal.
    """

    for sensor in model.findall(".//sensor[@type='contact']"):
        topic = (sensor.findtext("topic") or "").strip()
        if not topic:
            raise DetachableSdfError(
                "rendered SDF contact sensor has no explicit topic"
            )
        contact_topic = sensor.find("contact/topic")
        if contact_topic is None:
            contact = sensor.find("contact")
            if contact is None:
                raise DetachableSdfError(
                    "rendered SDF contact sensor has no contact configuration"
                )
            contact_topic = ET.SubElement(contact, "topic")
        nested_topic = (contact_topic.text or "").strip()
        if nested_topic not in {"", "__default_topic__", topic}:
            raise DetachableSdfError(
                "rendered SDF contact sensor has conflicting topic fields"
            )
        contact_topic.text = topic


def _assign_robot_collision_filter_mask(model: ET.Element) -> None:
    """Put every robot shape in one physics collision-filter class.

    Gazebo Physics uses a symmetric bitmask: two shapes collide only when the
    bitwise AND of their masks is non-zero.  The robot owns bit 0; the normal
    world mask remains ``0xffff``.  While a target is fixed to the gripper the
    authoritative world plugin recreates only that target with bit 1, which
    suppresses internal target/robot contacts while preserving target/world
    contacts.  Never accept a converter-provided conflicting mask because a
    partially classified robot would make that guarantee false.
    """

    # A contact sensor contains a scalar ``<contact><collision>name``
    # reference which ElementTree also matches with ``.//collision``.  Only
    # real collision entities are direct children of links; adding a surface
    # below the scalar reference creates invalid SDF and weakens provenance.
    for collision in model.findall(".//link/collision"):
        surface = collision.find("surface")
        if surface is None:
            surface = ET.SubElement(collision, "surface")
        contact = surface.find("contact")
        if contact is None:
            contact = ET.SubElement(surface, "contact")
        bitmask = contact.find("collide_bitmask")
        if bitmask is None:
            bitmask = ET.SubElement(contact, "collide_bitmask")
        existing = (bitmask.text or "").strip()
        if existing:
            try:
                parsed = int(existing, 0)
            except ValueError as exc:
                raise DetachableSdfError(
                    "rendered SDF robot collision bitmask is invalid"
                ) from exc
            if parsed != ROBOT_COLLISION_FILTER_MASK:
                raise DetachableSdfError(
                    "rendered SDF robot collision bitmask conflicts with the "
                    "authoritative attachment contract"
                )
        bitmask.text = str(ROBOT_COLLISION_FILTER_MASK)


def _binding_field(binding: object, key: str) -> str:
    value = binding.get(key) if isinstance(binding, Mapping) else getattr(binding, key, None)
    return str(value or "").strip()


def prepare_detachable_sdf(
    root: ET.Element,
    *,
    dart_compatible: bool = True,
    target_bindings: Sequence[object] | None = None,
) -> ET.Element:
    """Validate the template and materialize one stock joint per sort target."""

    model = root.find("model")
    if model is None:
        raise DetachableSdfError("rendered SDF has no robot model")
    plugins = _plugins(model)
    if len(plugins) != 1:
        raise DetachableSdfError("rendered SDF must contain exactly one DetachableJoint")
    plugin = plugins[0]
    expected = {
        "parent_link": "gripper_mount_link",
        "child_model": "target_object",
        "child_link": "target_link",
        "attach_topic": "/openeta/native_grasp/detachable_joint/target/attach",
        "detach_topic": "/openeta/native_grasp/detachable_joint/target/detach",
        "output_topic": "/openeta/native_grasp/detachable_joint/target/state",
    }
    actual = {key: plugin.findtext(key) for key in expected}
    if actual != expected:
        raise DetachableSdfError("rendered SDF DetachableJoint topology is not approved")
    bindings = tuple(target_bindings or ())
    if bindings:
        normalized: list[dict[str, str]] = []
        for binding in bindings:
            values = {
                "parent_link": "gripper_mount_link",
                "child_model": _binding_field(binding, "target_model"),
                "child_link": _binding_field(binding, "target_link"),
                "attach_topic": _binding_field(binding, "attach_topic"),
                "detach_topic": _binding_field(binding, "detach_topic"),
                "output_topic": _binding_field(binding, "state_topic"),
            }
            if any(not value for value in values.values()):
                raise DetachableSdfError("detachable target binding is invalid")
            normalized.append(values)
        if len({item["child_model"] for item in normalized}) != len(normalized):
            raise DetachableSdfError("detachable target binding is duplicated")
        if plugin not in list(model):
            raise DetachableSdfError("rendered SDF DetachableJoint must belong to the robot model")
        model.remove(plugin)
        for values in normalized:
            materialized = ET.fromstring(ET.tostring(plugin, encoding="unicode"))
            for key, value in values.items():
                field = materialized.find(key)
                if field is None:
                    raise DetachableSdfError("rendered SDF DetachableJoint field is missing")
                field.text = value
            model.append(materialized)
    if model.find("link[@name='base_link']") is None:
        raise DetachableSdfError("rendered SDF has no base_link for a fixed root")
    _normalize_contact_topics(model)
    _assign_robot_collision_filter_mask(model)
    for joint in list(model.findall("joint[@name='openeta_world_to_base']")):
        model.remove(joint)
    root_joint = ET.Element("joint", {"name": "openeta_world_to_base", "type": "fixed"})
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
    target_bindings: Sequence[object] | None = None,
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
        raise DetachableSdfError("NATIVE_GRASP_DART_UNSUPPORTED: xacro is unavailable") from exc
    if rendered.returncode:
        raise DetachableSdfError(f"NATIVE_GRASP_DART_UNSUPPORTED: xacro failed: {rendered.stderr[-800:]}")
    urdf_fd, urdf_name = tempfile.mkstemp(prefix="openeta-native-grasp-", suffix=".urdf", dir=directory)
    try:
        with os.fdopen(urdf_fd, "w", encoding="utf-8") as stream:
            stream.write(rendered.stdout)
        try:
            converted = subprocess.run(
                [gz, "sdf", "-p", urdf_name], capture_output=True, text=True, env=env,
                timeout=60.0, check=False,
            )
        except OSError as exc:
            raise DetachableSdfError("NATIVE_GRASP_DART_UNSUPPORTED: gz is unavailable") from exc
    finally:
        Path(urdf_name).unlink(missing_ok=True)
    if converted.returncode:
        raise DetachableSdfError(
            f"NATIVE_GRASP_DART_UNSUPPORTED: URDF-to-SDF conversion failed: {converted.stderr[-800:]}"
        )
    try:
        root = prepare_detachable_sdf(
            ET.fromstring(converted.stdout),
            dart_compatible=True,
            target_bindings=target_bindings,
        )
    except ET.ParseError as exc:
        raise DetachableSdfError("NATIVE_GRASP_DART_UNSUPPORTED: invalid converted SDF") from exc
    descriptor, sdf_name = tempfile.mkstemp(prefix="openeta-native-grasp-", suffix=".sdf", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(ET.tostring(root, encoding="unicode"))
            stream.write("\n")
    except Exception:
        Path(sdf_name).unlink(missing_ok=True)
        raise
    return Path(sdf_name)
