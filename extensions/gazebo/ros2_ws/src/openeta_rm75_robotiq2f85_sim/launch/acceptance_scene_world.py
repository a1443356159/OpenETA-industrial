"""Render one versioned acceptance-scene variant from the canonical world."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping
import xml.etree.ElementTree as ET


SCHEMA_VERSION = "openeta.gazebo_acceptance_scenes.v1"
SCENE_ENV = "OPENETA_ACCEPTANCE_SCENE"


def _numbers(value: object, count: int, *, positive: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise RuntimeError("acceptance scene vector is invalid")
    parsed = [float(item) for item in value]
    if any(not math.isfinite(item) or (positive and item <= 0.0) for item in parsed):
        raise RuntimeError("acceptance scene vector is invalid")
    return parsed


def _text(values: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def render_acceptance_world(
    *,
    base_world: Path,
    catalog_path: Path,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, str]:
    """Return the exact SDF selected by ``OPENETA_ACCEPTANCE_SCENE``."""

    source = os.environ if environment is None else environment
    requested = str(source.get(SCENE_ENV) or "normal").strip()
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    scenes = payload.get("scenes") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(scenes, Mapping)
    ):
        raise RuntimeError("acceptance scene catalog is invalid")
    scene = scenes.get(requested)
    if not isinstance(scene, Mapping):
        raise RuntimeError(f"unsupported acceptance scene: {requested}")
    world_scene = str(scene.get("world_scene") or "")
    obstacles = scene.get("static_obstacles")
    if not world_scene or not isinstance(obstacles, list):
        raise RuntimeError("acceptance scene definition is invalid")
    destination_value = scene.get("destination_center_xy")
    if not obstacles and destination_value is None:
        return base_world, world_scene

    tree = ET.parse(base_world)
    world = tree.getroot().find("world")
    if world is None:
        raise RuntimeError("canonical pick-place world is invalid")
    if destination_value is not None:
        destination = _numbers(destination_value, 2)
        marker = world.find("model[@name='placement_zone_marker']")
        if marker is None or marker.find("pose") is None:
            raise RuntimeError("canonical placement marker is invalid")
        marker.find("pose").text = _text([*destination, 0.4005, 0.0, 0.0, 0.0])
    existing_ids = {
        str(model.get("name") or "") for model in world.findall("model")
    }
    for raw in obstacles:
        if not isinstance(raw, Mapping):
            raise RuntimeError("acceptance scene obstacle is invalid")
        obstacle_id = str(raw.get("id") or "")
        if not obstacle_id or obstacle_id in existing_ids:
            raise RuntimeError("acceptance scene obstacle identity is invalid")
        existing_ids.add(obstacle_id)
        size = _numbers(raw.get("size_xyz"), 3, positive=True)
        xyz = _numbers(raw.get("pose_xyz"), 3)
        rpy = _numbers(raw.get("pose_rpy"), 3)
        rgba = _numbers(raw.get("rgba"), 4)

        model = ET.SubElement(world, "model", {"name": obstacle_id})
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = _text([*xyz, *rpy])
        link = ET.SubElement(model, "link", {"name": f"{obstacle_id}_link"})
        collision = ET.SubElement(
            link, "collision", {"name": f"{obstacle_id}_collision"}
        )
        collision_geometry = ET.SubElement(collision, "geometry")
        collision_box = ET.SubElement(collision_geometry, "box")
        ET.SubElement(collision_box, "size").text = _text(size)
        visual = ET.SubElement(link, "visual", {"name": f"{obstacle_id}_visual"})
        visual_geometry = ET.SubElement(visual, "geometry")
        visual_box = ET.SubElement(visual_geometry, "box")
        ET.SubElement(visual_box, "size").text = _text(size)
        material = ET.SubElement(visual, "material")
        ET.SubElement(material, "ambient").text = _text(rgba)
        ET.SubElement(material, "diffuse").text = _text(rgba)
        ET.SubElement(material, "specular").text = "0.05 0.05 0.05 1"

    descriptor, rendered_name = tempfile.mkstemp(
        prefix=f"openeta_{world_scene.replace('-', '_')}_",
        suffix=".sdf",
    )
    os.close(descriptor)
    rendered = Path(rendered_name)
    tree.write(rendered, encoding="utf-8", xml_declaration=True)
    return rendered, world_scene
