"""Render one versioned acceptance-scene variant from the canonical world."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping
import xml.etree.ElementTree as ET


SCHEMA_VERSION = "openeta.gazebo_acceptance_scenes.v2"
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


def _append_geometry(parent: ET.Element, primitive: Mapping[str, object]) -> None:
    geometry = ET.SubElement(parent, "geometry")
    shape = str(primitive.get("shape") or "")
    if shape == "box":
        box = ET.SubElement(geometry, "box")
        ET.SubElement(box, "size").text = _text(
            _numbers(primitive.get("size_xyz"), 3, positive=True)
        )
        return
    if shape == "cylinder":
        cylinder = ET.SubElement(geometry, "cylinder")
        radius = float(primitive.get("radius", 0.0))
        length = float(primitive.get("length", 0.0))
        if not math.isfinite(radius) or radius <= 0.0 or not math.isfinite(length) or length <= 0.0:
            raise RuntimeError("acceptance scene cylinder is invalid")
        ET.SubElement(cylinder, "radius").text = f"{radius:.12g}"
        ET.SubElement(cylinder, "length").text = f"{length:.12g}"
        return
    raise RuntimeError("acceptance scene primitive shape is invalid")


def _append_primitives(
    link: ET.Element,
    *,
    owner_id: str,
    owner: Mapping[str, object],
) -> None:
    primitives = owner.get("primitives")
    if primitives is None:
        primitives = [
            {
                "shape": "box",
                "size_xyz": owner.get("size_xyz"),
                "pose_xyz": [0.0, 0.0, 0.0],
                "pose_rpy": [0.0, 0.0, 0.0],
                "rgba": owner.get("rgba"),
            }
        ]
    if not isinstance(primitives, list) or not primitives:
        raise RuntimeError("acceptance scene primitives are invalid")
    for index, raw_primitive in enumerate(primitives):
        if not isinstance(raw_primitive, Mapping):
            raise RuntimeError("acceptance scene primitive is invalid")
        xyz = _numbers(raw_primitive.get("pose_xyz"), 3)
        rpy = _numbers(raw_primitive.get("pose_rpy"), 3)
        rgba = _numbers(raw_primitive.get("rgba"), 4)
        pose_text = _text([*xyz, *rpy])
        collision = ET.SubElement(
            link, "collision", {"name": f"{owner_id}_collision_{index}"}
        )
        ET.SubElement(collision, "pose").text = pose_text
        _append_geometry(collision, raw_primitive)
        visual = ET.SubElement(
            link, "visual", {"name": f"{owner_id}_visual_{index}"}
        )
        ET.SubElement(visual, "pose").text = pose_text
        _append_geometry(visual, raw_primitive)
        material = ET.SubElement(visual, "material")
        ET.SubElement(material, "ambient").text = _text(rgba)
        ET.SubElement(material, "diffuse").text = _text(rgba)
        ET.SubElement(material, "specular").text = "0.12 0.12 0.12 1"


def _append_static_model(
    world: ET.Element,
    *,
    raw: Mapping[str, object],
    existing_ids: set[str],
) -> None:
    obstacle_id = str(raw.get("id") or "")
    if not obstacle_id or obstacle_id in existing_ids:
        raise RuntimeError("acceptance scene obstacle identity is invalid")
    existing_ids.add(obstacle_id)
    xyz = _numbers(raw.get("pose_xyz"), 3)
    rpy = _numbers(raw.get("pose_rpy"), 3)
    _numbers(raw.get("size_xyz"), 3, positive=True)
    _numbers(raw.get("rgba"), 4)
    model = ET.SubElement(world, "model", {"name": obstacle_id})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = _text([*xyz, *rpy])
    link = ET.SubElement(model, "link", {"name": f"{obstacle_id}_link"})
    _append_primitives(link, owner_id=obstacle_id, owner=raw)


def _replace_target_model(
    world: ET.Element,
    *,
    raw: Mapping[str, object],
) -> None:
    existing = world.find("model[@name='target_object']")
    if existing is None:
        raise RuntimeError("canonical target object is invalid")
    world.remove(existing)
    xyz = _numbers(raw.get("pose_xyz"), 3)
    rpy = _numbers(raw.get("pose_rpy"), 3)
    bounds = _numbers(raw.get("bounding_box_xyz"), 3, positive=True)
    mass = float(raw.get("mass_kg", 0.0))
    if not math.isfinite(mass) or mass <= 0.0:
        raise RuntimeError("acceptance scene target mass is invalid")
    model = ET.SubElement(world, "model", {"name": "target_object"})
    ET.SubElement(model, "pose").text = _text([*xyz, *rpy])
    link = ET.SubElement(model, "link", {"name": "target_link"})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass").text = f"{mass:.12g}"
    inertia = ET.SubElement(inertial, "inertia")
    x, y, z = bounds
    values = {
        "ixx": mass * (y * y + z * z) / 12.0,
        "iyy": mass * (x * x + z * z) / 12.0,
        "izz": mass * (x * x + y * y) / 12.0,
        "ixy": 0.0,
        "ixz": 0.0,
        "iyz": 0.0,
    }
    for key, value in values.items():
        ET.SubElement(inertia, key).text = f"{value:.12g}"
    _append_primitives(link, owner_id="target", owner=raw)


def _replace_placement_regions(
    world: ET.Element,
    *,
    regions: list[object],
) -> None:
    marker = world.find("model[@name='placement_zone_marker']")
    if marker is None:
        raise RuntimeError("canonical placement marker is invalid")
    world.remove(marker)
    for raw_region in regions:
        if not isinstance(raw_region, Mapping):
            raise RuntimeError("acceptance scene placement region is invalid")
        region_id = str(raw_region.get("id") or "")
        center = _numbers(raw_region.get("center_xy"), 2)
        size = _numbers(raw_region.get("size_xy_m"), 2, positive=True)
        rgba = _numbers(raw_region.get("rgba"), 4)
        if not region_id:
            raise RuntimeError("acceptance scene placement region identity is invalid")
        model = ET.SubElement(world, "model", {"name": region_id})
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = _text(
            [*center, 0.4005, 0.0, 0.0, 0.0]
        )
        link = ET.SubElement(model, "link", {"name": f"{region_id}_floor_link"})
        visual = ET.SubElement(
            link, "visual", {"name": f"{region_id}_floor_visual"}
        )
        geometry = ET.SubElement(visual, "geometry")
        box = ET.SubElement(geometry, "box")
        ET.SubElement(box, "size").text = _text([*size, 0.001])
        material = ET.SubElement(visual, "material")
        ET.SubElement(material, "ambient").text = _text(rgba)
        ET.SubElement(material, "diffuse").text = _text(rgba)
        ET.SubElement(material, "specular").text = "0.04 0.04 0.04 1"


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
    target_value = scene.get("target_object")
    placement_regions = scene.get("placement_regions")
    if (
        not obstacles
        and destination_value is None
        and target_value is None
        and placement_regions is None
    ):
        return base_world, world_scene

    tree = ET.parse(base_world)
    world = tree.getroot().find("world")
    if world is None:
        raise RuntimeError("canonical pick-place world is invalid")
    if isinstance(placement_regions, list):
        _replace_placement_regions(world, regions=placement_regions)
    elif destination_value is not None:
        destination = _numbers(destination_value, 2)
        marker = world.find("model[@name='placement_zone_marker']")
        if marker is None or marker.find("pose") is None:
            raise RuntimeError("canonical placement marker is invalid")
        marker.find("pose").text = _text([*destination, 0.4005, 0.0, 0.0, 0.0])
    if isinstance(target_value, Mapping):
        _replace_target_model(world, raw=target_value)
    if scene.get("replace_default_distractor") is True:
        distractor = world.find("model[@name='distractor_object']")
        if distractor is None:
            raise RuntimeError("canonical distractor object is invalid")
        world.remove(distractor)
    existing_ids = {
        str(model.get("name") or "") for model in world.findall("model")
    }
    for raw in obstacles:
        if not isinstance(raw, Mapping):
            raise RuntimeError("acceptance scene obstacle is invalid")
        _append_static_model(world, raw=raw, existing_ids=existing_ids)

    descriptor, rendered_name = tempfile.mkstemp(
        prefix=f"openeta_{world_scene.replace('-', '_')}_",
        suffix=".sdf",
    )
    os.close(descriptor)
    rendered = Path(rendered_name)
    tree.write(rendered, encoding="utf-8", xml_declaration=True)
    return rendered, world_scene
