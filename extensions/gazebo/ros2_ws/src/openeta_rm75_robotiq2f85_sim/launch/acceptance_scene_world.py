"""Render one versioned acceptance-scene variant from the canonical world."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


SCHEMA_VERSION = "openeta.gazebo_acceptance_scenes.v2"
SCENE_ENV = "OPENETA_ACCEPTANCE_SCENE"
AUTHORITATIVE_SCENE_SCHEMA_VERSION = "openeta.authoritative_scene.v3"
ROBOT_COLLISION_FILTER_MASK = 0x0001
DETACHED_TARGET_COLLISION_FILTER_MASK = 0xFFFF
ATTACHED_TARGET_COLLISION_FILTER_MASK = 0x0002
ATTACHED_COLLISION_FILTER_STATE_TOPIC = (
    "/openeta/native_grasp/detachable_joint/target/collision_filter_state"
)
ATTACHED_COLLISION_FILTER_STATE_REQUEST_TOPIC = (
    "/openeta/native_grasp/detachable_joint/target/"
    "collision_filter_state/request"
)
ATTACHED_COLLISION_FILTER_STATE_ACK_TOPIC = (
    "/openeta/native_grasp/detachable_joint/target/"
    "collision_filter_state/ack"
)


def _quaternion_from_rpy(values: Sequence[float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (float(value) for value in values)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _quaternion_multiply(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = (float(value) for value in left)
    rx, ry, rz, rw = (float(value) for value in right)
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quaternion_rotate(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    qx, qy, qz, qw = (float(value) for value in quaternion)
    vx, vy, vz = (float(value) for value in vector)
    # Expanded q * [v, 0] * conjugate(q), avoiding a temporary quaternion.
    tx, ty, tz = (
        2.0 * (qy * vz - qz * vy),
        2.0 * (qz * vx - qx * vz),
        2.0 * (qx * vy - qy * vx),
    )
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _compose_pose(
    parent_xyz: Sequence[float],
    parent_quat: Sequence[float],
    child_xyz: Sequence[float],
    child_quat: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    rotated = _quaternion_rotate(parent_quat, child_xyz)
    xyz = tuple(float(parent_xyz[index]) + rotated[index] for index in range(3))
    return xyz, _quaternion_multiply(parent_quat, child_quat)


def _pose(element: ET.Element, *, owner: str) -> tuple[
    tuple[float, float, float], tuple[float, float, float, float]
]:
    pose = element.find("pose")
    if pose is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
    if pose.attrib.get("relative_to"):
        raise RuntimeError(f"authoritative scene uses unsupported relative_to pose: {owner}")
    values = [float(value) for value in (pose.text or "").split()]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"authoritative scene pose is invalid: {owner}")
    return tuple(values[:3]), _quaternion_from_rpy(values[3:])  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class AuthoritativePrimitive:
    """One Gazebo collision primitive, expressed in its model frame."""

    name: str
    shape: str
    pose_xyz: tuple[float, float, float]
    pose_quat_xyzw: tuple[float, float, float, float]
    size_xyz: tuple[float, float, float] | None = None
    radius: float | None = None
    length: float | None = None

    def moveit_spec(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "shape": self.shape,
            "pose_xyz": list(self.pose_xyz),
            "pose_quat_xyzw": list(self.pose_quat_xyzw),
        }
        if self.shape == "box" and self.size_xyz is not None:
            result["size_xyz"] = list(self.size_xyz)
        elif self.shape == "cylinder" and self.radius is not None and self.length is not None:
            result.update({"radius": self.radius, "length": self.length})
        else:
            raise RuntimeError(f"authoritative collision primitive is invalid: {self.name}")
        return result

    def local_bounds(self) -> tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]:
        if self.shape == "box" and self.size_xyz is not None:
            half = tuple(value / 2.0 for value in self.size_xyz)
            # Project each oriented local half-axis onto the model axes.
            axes = tuple(
                _quaternion_rotate(
                    self.pose_quat_xyzw,
                    tuple(1.0 if axis == basis else 0.0 for axis in range(3)),
                )
                for basis in range(3)
            )
            extent = tuple(
                sum(abs(axes[basis][axis]) * half[basis] for basis in range(3))
                for axis in range(3)
            )
        elif self.shape == "cylinder" and self.radius is not None and self.length is not None:
            axis = _quaternion_rotate(self.pose_quat_xyzw, (0.0, 0.0, 1.0))
            half_length = self.length / 2.0
            extent = tuple(
                half_length * abs(axis[index])
                + self.radius * math.sqrt(max(0.0, 1.0 - axis[index] ** 2))
                for index in range(3)
            )
        else:
            raise RuntimeError(f"authoritative collision primitive is invalid: {self.name}")
        return (
            tuple(self.pose_xyz[index] - extent[index] for index in range(3)),
            tuple(self.pose_xyz[index] + extent[index] for index in range(3)),
        )


@dataclass(frozen=True, slots=True)
class AuthoritativeObject:
    """One model shared by Gazebo physics and MoveIt PlanningScene."""

    object_id: str
    pose_xyz: tuple[float, float, float]
    pose_quat_xyzw: tuple[float, float, float, float]
    gazebo_static: bool
    visual_count: int
    mirrored_visual_count: int
    primitives: tuple[AuthoritativePrimitive, ...]
    bounding_box_xyz: tuple[float, float, float]

    def moveit_spec(self) -> dict[str, Any]:
        return {
            "id": self.object_id,
            "frame": "world",
            "shape": "compound",
            "size_xyz": list(self.bounding_box_xyz),
            "pose_xyz": list(self.pose_xyz),
            "pose_quat_xyzw": list(self.pose_quat_xyzw),
            "primitives": [primitive.moveit_spec() for primitive in self.primitives],
            "gazebo_static": self.gazebo_static,
            "gazebo_collision_names": [primitive.name for primitive in self.primitives],
        }

    def world_bounds(self) -> tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]:
        """Return a conservative world AABB from the exact collision primitives."""

        corners: list[tuple[float, float, float]] = []
        for primitive in self.primitives:
            lower, upper = primitive.local_bounds()
            for mask in range(8):
                local = tuple(
                    upper[axis] if mask & (1 << axis) else lower[axis]
                    for axis in range(3)
                )
                rotated = _quaternion_rotate(self.pose_quat_xyzw, local)
                corners.append(
                    tuple(
                        self.pose_xyz[axis] + rotated[axis]
                        for axis in range(3)
                    )
                )
        if not corners:
            raise RuntimeError(
                f"authoritative collision object has no primitives: {self.object_id}"
            )
        return (
            tuple(min(point[axis] for point in corners) for axis in range(3)),
            tuple(max(point[axis] for point in corners) for axis in range(3)),
        )


@dataclass(frozen=True, slots=True)
class CompiledAuthoritativeScene:
    """Immutable result consumed by both Gazebo launch and MoveIt reset."""

    scene_id: str
    world_scene: str
    sdf_bytes: bytes
    objects: tuple[AuthoritativeObject, ...]
    gazebo_sdf_sha256: str
    collision_manifest_sha256: str
    authority_sha256: str

    def object(self, object_id: str) -> AuthoritativeObject:
        matches = [item for item in self.objects if item.object_id == object_id]
        if len(matches) != 1:
            raise RuntimeError(f"authoritative scene object is missing: {object_id}")
        return matches[0]

    @property
    def dynamic_object_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.objects if not item.gazebo_static)

    def evidence(self) -> dict[str, Any]:
        aligned_objects = [
            item.object_id
            for item in self.objects
            if item.mirrored_visual_count == len(item.primitives)
        ]
        return {
            "schema_version": AUTHORITATIVE_SCENE_SCHEMA_VERSION,
            "scene_id": self.scene_id,
            "world_scene": self.world_scene,
            "authority_sha256": self.authority_sha256,
            "gazebo_sdf_sha256": self.gazebo_sdf_sha256,
            "collision_manifest_sha256": self.collision_manifest_sha256,
            "gazebo_collision_object_ids": [item.object_id for item in self.objects],
            "moveit_collision_object_ids": [item.object_id for item in self.objects],
            "visual_collision_aligned_object_ids": aligned_objects,
            "dynamic_object_ids": list(self.dynamic_object_ids),
            "primitive_count": sum(len(item.primitives) for item in self.objects),
            "mirrored_visual_primitive_count": sum(
                item.mirrored_visual_count for item in self.objects
            ),
            "attached_collision_filter": {
                "schema_version": "openeta.attached_collision_filter.v1",
                "state_topic": ATTACHED_COLLISION_FILTER_STATE_TOPIC,
                "state_request_topic": (
                    ATTACHED_COLLISION_FILTER_STATE_REQUEST_TOPIC
                ),
                "state_ack_topic": ATTACHED_COLLISION_FILTER_STATE_ACK_TOPIC,
                "robot_mask": ROBOT_COLLISION_FILTER_MASK,
                "detached_target_mask": DETACHED_TARGET_COLLISION_FILTER_MASK,
                "attached_target_mask": ATTACHED_TARGET_COLLISION_FILTER_MASK,
                "attached_target_robot_collision_enabled": False,
                "attached_target_environment_collision_enabled": True,
            },
        }


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
    for marker_name in ("placement_zone_marker", "blue_destination_marker"):
        existing_marker = world.find(f"model[@name='{marker_name}']")
        if existing_marker is not None:
            world.remove(existing_marker)
    for raw_region in regions:
        if not isinstance(raw_region, Mapping):
            raise RuntimeError("acceptance scene placement region is invalid")
        region_id = str(raw_region.get("id") or "")
        center = _numbers(raw_region.get("center_xy"), 2)
        size = _numbers(raw_region.get("size_xy_m"), 2, positive=True)
        rgba = _numbers(raw_region.get("rgba"), 4)
        if not region_id:
            raise RuntimeError("acceptance scene placement region identity is invalid")
        # The canonical industrial world contains physical green/blue bins.
        # A generated legacy scene may reuse either semantic ID for a flat
        # region, so replace that model instead of emitting duplicate SDF IDs.
        existing_region = world.find(f"model[@name='{region_id}']")
        if existing_region is not None:
            world.remove(existing_region)
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


def _render_scene_tree(
    *,
    base_world: Path,
    catalog_path: Path,
    scene_id: str,
) -> tuple[ET.ElementTree, str]:
    """Compile the selected catalog variant into one final Gazebo tree."""

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    scenes = payload.get("scenes") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(scenes, Mapping)
    ):
        raise RuntimeError("acceptance scene catalog is invalid")
    scene = scenes.get(scene_id)
    if not isinstance(scene, Mapping):
        raise RuntimeError(f"unsupported acceptance scene: {scene_id}")
    world_scene = str(scene.get("world_scene") or "")
    obstacles = scene.get("static_obstacles")
    if not world_scene or not isinstance(obstacles, list):
        raise RuntimeError("acceptance scene definition is invalid")
    destination_value = scene.get("destination_center_xy")
    target_value = scene.get("target_object")
    placement_regions = scene.get("placement_regions")

    tree = ET.parse(base_world)
    world = tree.getroot().find("world")
    if world is None:
        raise RuntimeError("canonical pick-place world is invalid")
    if scene.get("canonical_world_complete") is not True:
        if isinstance(placement_regions, list):
            _replace_placement_regions(world, regions=placement_regions)
        elif destination_value is not None:
            destination = _numbers(destination_value, 2)
            marker = world.find("model[@name='placement_zone_marker']")
            if marker is None or marker.find("pose") is None:
                raise RuntimeError("canonical placement marker is invalid")
            marker.find("pose").text = _text(
                [*destination, 0.4005, 0.0, 0.0, 0.0]
            )
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
    return tree, world_scene


def _primitive_from_collision(
    collision: ET.Element,
    *,
    model_id: str,
    link_xyz: Sequence[float],
    link_quat: Sequence[float],
) -> AuthoritativePrimitive:
    collision_name = str(collision.get("name") or "")
    if not collision_name:
        raise RuntimeError(f"authoritative collision identity is invalid: {model_id}")
    collision_xyz, collision_quat = _pose(
        collision, owner=f"{model_id}/{collision_name}"
    )
    local_xyz, local_quat = _compose_pose(
        link_xyz, link_quat, collision_xyz, collision_quat
    )
    geometry = collision.find("geometry")
    if geometry is None:
        raise RuntimeError(
            f"authoritative collision geometry is missing: {model_id}/{collision_name}"
        )
    box = geometry.find("box")
    cylinder = geometry.find("cylinder")
    if box is not None and cylinder is None:
        size = [float(value) for value in box.findtext("size", "").split()]
        if len(size) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in size):
            raise RuntimeError(
                f"authoritative collision box is invalid: {model_id}/{collision_name}"
            )
        return AuthoritativePrimitive(
            name=collision_name,
            shape="box",
            pose_xyz=local_xyz,
            pose_quat_xyzw=local_quat,
            size_xyz=tuple(size),  # type: ignore[arg-type]
        )
    if cylinder is not None and box is None:
        try:
            radius = float(cylinder.findtext("radius", ""))
            length = float(cylinder.findtext("length", ""))
        except ValueError as exc:
            raise RuntimeError(
                f"authoritative collision cylinder is invalid: {model_id}/{collision_name}"
            ) from exc
        if any(not math.isfinite(value) or value <= 0.0 for value in (radius, length)):
            raise RuntimeError(
                f"authoritative collision cylinder is invalid: {model_id}/{collision_name}"
            )
        return AuthoritativePrimitive(
            name=collision_name,
            shape="cylinder",
            pose_xyz=local_xyz,
            pose_quat_xyzw=local_quat,
            radius=radius,
            length=length,
        )
    raise RuntimeError(
        f"authoritative collision shape is unsupported: {model_id}/{collision_name}"
    )


def _primitive_geometry_signature(
    element: ET.Element,
    *,
    owner: str,
) -> tuple[str, tuple[float, ...]]:
    """Return the supported primitive geometry without accepting a mesh proxy."""

    geometry = element.find("geometry")
    if geometry is None:
        raise RuntimeError(f"authoritative geometry is missing: {owner}")
    box = geometry.find("box")
    cylinder = geometry.find("cylinder")
    if box is not None and cylinder is None:
        size = tuple(float(value) for value in box.findtext("size", "").split())
        if len(size) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in size
        ):
            raise RuntimeError(f"authoritative visual box is invalid: {owner}")
        return "box", size
    if cylinder is not None and box is None:
        try:
            radius = float(cylinder.findtext("radius", ""))
            length = float(cylinder.findtext("length", ""))
        except ValueError as exc:
            raise RuntimeError(
                f"authoritative visual cylinder is invalid: {owner}"
            ) from exc
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (radius, length)
        ):
            raise RuntimeError(f"authoritative visual cylinder is invalid: {owner}")
        return "cylinder", (radius, length)
    raise RuntimeError(f"authoritative visual shape is unsupported: {owner}")


def _collision_has_exact_visual_twin(
    link: ET.Element,
    collision: ET.Element,
    *,
    model_id: str,
) -> bool:
    """Prove one visible primitive is exactly the Gazebo/MoveIt primitive."""

    collision_name = str(collision.get("name") or "")
    visual = link.find(f"visual[@name='{collision_name}_visual']")
    if visual is None:
        return False
    collision_xyz, collision_quat = _pose(
        collision,
        owner=f"{model_id}/{collision_name}",
    )
    visual_xyz, visual_quat = _pose(
        visual,
        owner=f"{model_id}/{collision_name}_visual",
    )
    if not _same_numbers(collision_xyz, visual_xyz) or not _same_numbers(
        collision_quat,
        visual_quat,
    ):
        return False
    collision_geometry = _primitive_geometry_signature(
        collision,
        owner=f"{model_id}/{collision_name}",
    )
    visual_geometry = _primitive_geometry_signature(
        visual,
        owner=f"{model_id}/{collision_name}_visual",
    )
    return collision_geometry[0] == visual_geometry[0] and _same_numbers(
        collision_geometry[1],
        visual_geometry[1],
    )


def _set_collision_filter_mask(model: ET.Element, *, mask: int) -> None:
    """Materialize one explicit Gazebo Physics mask on every model shape."""

    if mask <= 0 or mask > 0xFFFF:
        raise RuntimeError("authoritative collision-filter mask is invalid")
    collisions = model.findall("link/collision")
    if not collisions:
        raise RuntimeError(
            "authoritative collision-filter target has no collision geometry"
        )
    for collision in collisions:
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
                raise RuntimeError(
                    "authoritative collision-filter mask is invalid"
                ) from exc
            if parsed != mask:
                raise RuntimeError(
                    "authoritative target collision-filter mask conflicts with "
                    "the scene manager"
                )
        bitmask.text = str(mask)


def _validate_attached_collision_filter_contract(world: ET.Element) -> None:
    plugins = [
        plugin
        for plugin in world.findall("plugin")
        if plugin.get("name") == "openeta::gazebo::AttachedCollisionFilter"
    ]
    if len(plugins) != 1:
        raise RuntimeError(
            "authoritative attached collision-filter plugin is missing"
        )
    plugin = plugins[0]
    expected = {
        "target_model": "target_object",
        "target_link": "target_link",
        "state_topic": ATTACHED_COLLISION_FILTER_STATE_TOPIC,
        "state_request_topic": ATTACHED_COLLISION_FILTER_STATE_REQUEST_TOPIC,
        "state_ack_topic": ATTACHED_COLLISION_FILTER_STATE_ACK_TOPIC,
        "robot_mask": str(ROBOT_COLLISION_FILTER_MASK),
        "detached_mask": str(DETACHED_TARGET_COLLISION_FILTER_MASK),
        "attached_mask": str(ATTACHED_TARGET_COLLISION_FILTER_MASK),
    }
    actual = {
        key: (plugin.findtext(key) or "").strip() for key in expected
    }
    if actual != expected:
        raise RuntimeError(
            "authoritative attached collision-filter contract is invalid"
        )


def _authoritative_objects(world: ET.Element) -> tuple[AuthoritativeObject, ...]:
    objects: list[AuthoritativeObject] = []
    seen_models: set[str] = set()
    for model in world.findall("model"):
        model_id = str(model.get("name") or "")
        if not model_id or model_id in seen_models:
            raise RuntimeError("authoritative model identity is invalid")
        seen_models.add(model_id)
        model_xyz, model_quat = _pose(model, owner=model_id)
        primitives: list[AuthoritativePrimitive] = []
        visual_count = 0
        mirrored_visual_count = 0
        collision_names: set[str] = set()
        for link in model.findall("link"):
            link_name = str(link.get("name") or "")
            if not link_name:
                raise RuntimeError(f"authoritative link identity is invalid: {model_id}")
            link_xyz, link_quat = _pose(link, owner=f"{model_id}/{link_name}")
            visual_count += len(link.findall("visual"))
            for collision in link.findall("collision"):
                primitive = _primitive_from_collision(
                    collision,
                    model_id=model_id,
                    link_xyz=link_xyz,
                    link_quat=link_quat,
                )
                if primitive.name in collision_names:
                    raise RuntimeError(
                        f"authoritative collision identity is duplicated: "
                        f"{model_id}/{primitive.name}"
                    )
                collision_names.add(primitive.name)
                primitives.append(primitive)
                if _collision_has_exact_visual_twin(
                    link,
                    collision,
                    model_id=model_id,
                ):
                    mirrored_visual_count += 1
        if not primitives:
            continue
        if visual_count <= 0:
            raise RuntimeError(
                f"authoritative collision model has no Gazebo visual: {model_id}"
            )
        bounds = [primitive.local_bounds() for primitive in primitives]
        lower = tuple(min(item[0][axis] for item in bounds) for axis in range(3))
        upper = tuple(max(item[1][axis] for item in bounds) for axis in range(3))
        objects.append(
            AuthoritativeObject(
                object_id=model_id,
                pose_xyz=model_xyz,
                pose_quat_xyzw=model_quat,
                gazebo_static=(model.findtext("static", "false").strip().lower() == "true"),
                visual_count=visual_count,
                mirrored_visual_count=mirrored_visual_count,
                primitives=tuple(primitives),
                bounding_box_xyz=tuple(
                    upper[axis] - lower[axis] for axis in range(3)
                ),
            )
        )
    return tuple(objects)


def _same_numbers(
    left: Sequence[float], right: Sequence[float], *, tolerance: float = 1e-9
) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right, strict=True)
    )


def _validate_catalog_bindings(
    *,
    catalog_path: Path,
    scene_id: str,
    objects: Sequence[AuthoritativeObject],
    world: ET.Element,
) -> None:
    """Reject semantic catalog data that drifted from the compiled world."""

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    scene = payload.get("scenes", {}).get(scene_id) if isinstance(payload, Mapping) else None
    if not isinstance(scene, Mapping):
        raise RuntimeError("authoritative scene catalog binding is invalid")
    by_id = {item.object_id: item for item in objects}
    target_raw = scene.get("target_object")
    if isinstance(target_raw, Mapping):
        target = by_id.get("target_object")
        if target is None:
            raise RuntimeError("authoritative target collision model is missing")
        expected_xyz = _numbers(target_raw.get("pose_xyz"), 3)
        expected_quat = _quaternion_from_rpy(
            _numbers(target_raw.get("pose_rpy"), 3)
        )
        if not _same_numbers(target.pose_xyz, expected_xyz) or not _same_numbers(
            target.pose_quat_xyzw, expected_quat
        ):
            raise RuntimeError(
                "authoritative target pose differs from the task contract"
            )
        expected_primitives = target_raw.get("primitives")
        if isinstance(expected_primitives, list):
            if len(expected_primitives) != len(target.primitives):
                raise RuntimeError(
                    "authoritative target primitive count differs from the task contract"
                )
            for raw, primitive in zip(
                expected_primitives, target.primitives, strict=True
            ):
                if not isinstance(raw, Mapping):
                    raise RuntimeError("authoritative target primitive is invalid")
                expected_shape = str(raw.get("shape") or "")
                expected_local_xyz = _numbers(raw.get("pose_xyz"), 3)
                expected_local_quat = _quaternion_from_rpy(
                    _numbers(raw.get("pose_rpy"), 3)
                )
                geometry_matches = expected_shape == primitive.shape
                if expected_shape == "box" and primitive.size_xyz is not None:
                    geometry_matches = geometry_matches and _same_numbers(
                        primitive.size_xyz,
                        _numbers(raw.get("size_xyz"), 3, positive=True),
                    )
                elif expected_shape == "cylinder":
                    geometry_matches = geometry_matches and _same_numbers(
                        (primitive.radius or 0.0, primitive.length or 0.0),
                        (float(raw.get("radius", 0.0)), float(raw.get("length", 0.0))),
                    )
                else:
                    geometry_matches = False
                if not (
                    geometry_matches
                    and _same_numbers(primitive.pose_xyz, expected_local_xyz)
                    and _same_numbers(
                        primitive.pose_quat_xyzw, expected_local_quat
                    )
                ):
                    raise RuntimeError(
                        "authoritative target geometry differs from the task contract"
                    )
        bounds = _numbers(target_raw.get("bounding_box_xyz"), 3, positive=True)
        if any(
            target.bounding_box_xyz[index] > bounds[index] + 1e-9
            for index in range(3)
        ):
            raise RuntimeError(
                "authoritative target collision exceeds its task bounding box"
            )
        target_model = world.find("model[@name='target_object']")
        mass_text = (
            target_model.findtext("link/inertial/mass", "")
            if target_model is not None
            else ""
        )
        try:
            world_mass = float(mass_text)
            task_mass = float(target_raw.get("mass_kg", 0.0))
        except ValueError as exc:
            raise RuntimeError("authoritative target mass is invalid") from exc
        if not math.isclose(world_mass, task_mass, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                "authoritative target mass differs from the task contract"
            )

    regions = scene.get("placement_regions")
    if isinstance(regions, list):
        physical_supports: list[AuthoritativeObject] = []
        for raw in regions:
            if not isinstance(raw, Mapping):
                raise RuntimeError("authoritative placement region is invalid")
            support = by_id.get(str(raw.get("id") or ""))
            if support is None:
                # Legacy flat-region scenes are supported by the work table;
                # their generated marker intentionally has no collision.
                continue
            physical_supports.append(support)
            if support.mirrored_visual_count != len(support.primitives):
                raise RuntimeError(
                    "authoritative placement support visual/collision geometry differs: "
                    f"{support.object_id}"
                )
            center = _numbers(raw.get("center_xy"), 2)
            if not _same_numbers(support.pose_xyz[:2], center):
                raise RuntimeError(
                    "authoritative placement support differs from its task region"
                )
            support_z = raw.get("support_z_m")
            if support_z is not None:
                surface_z = float(support_z)
                primitive_tops = [
                    primitive.local_bounds()[1][2] + support.pose_xyz[2]
                    for primitive in support.primitives
                ]
                if not any(
                    math.isclose(
                        top,
                        surface_z,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    for top in primitive_tops
                ):
                    raise RuntimeError(
                        "authoritative placement support height differs from its task region"
                    )
        for index, left in enumerate(physical_supports):
            left_lower, left_upper = left.world_bounds()
            for right in physical_supports[index + 1 :]:
                right_lower, right_upper = right.world_bounds()
                penetration = tuple(
                    min(left_upper[axis], right_upper[axis])
                    - max(left_lower[axis], right_lower[axis])
                    for axis in range(3)
                )
                if all(value > 1e-9 for value in penetration):
                    raise RuntimeError(
                        "authoritative placement support collisions overlap: "
                        f"{left.object_id}/{right.object_id}"
                    )


def compile_authoritative_scene(
    *,
    base_world: Path,
    catalog_path: Path,
    scene_id: str,
) -> CompiledAuthoritativeScene:
    """Compile one immutable world and its exact MoveIt collision manifest."""

    selected = str(scene_id).strip()
    if not selected:
        raise RuntimeError("authoritative scene identity is invalid")
    tree, world_scene = _render_scene_tree(
        base_world=base_world,
        catalog_path=catalog_path,
        scene_id=selected,
    )
    world = tree.getroot().find("world")
    if world is None:
        raise RuntimeError("authoritative scene world is invalid")
    target_model = world.find("model[@name='target_object']")
    if target_model is None:
        raise RuntimeError("authoritative target collision model is missing")
    _set_collision_filter_mask(
        target_model,
        mask=DETACHED_TARGET_COLLISION_FILTER_MASK,
    )
    _validate_attached_collision_filter_contract(world)
    objects = _authoritative_objects(world)
    required_ids = {"work_table", "target_object"}
    object_ids = {item.object_id for item in objects}
    if not required_ids <= object_ids:
        raise RuntimeError(
            "authoritative scene is missing required collision models: "
            + ",".join(sorted(required_ids - object_ids))
        )
    _validate_catalog_bindings(
        catalog_path=catalog_path,
        scene_id=selected,
        objects=objects,
        world=world,
    )
    sdf_bytes = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    sdf_sha256 = hashlib.sha256(sdf_bytes).hexdigest()
    collision_payload = [item.moveit_spec() for item in objects]
    collision_bytes = json.dumps(
        collision_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    collision_sha256 = hashlib.sha256(collision_bytes).hexdigest()
    authority_sha256 = hashlib.sha256(
        json.dumps(
            {
                "schema_version": AUTHORITATIVE_SCENE_SCHEMA_VERSION,
                "scene_id": selected,
                "world_scene": world_scene,
                "gazebo_sdf_sha256": sdf_sha256,
                "collision_manifest_sha256": collision_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return CompiledAuthoritativeScene(
        scene_id=selected,
        world_scene=world_scene,
        sdf_bytes=sdf_bytes,
        objects=objects,
        gazebo_sdf_sha256=sdf_sha256,
        collision_manifest_sha256=collision_sha256,
        authority_sha256=authority_sha256,
    )


def render_acceptance_world(
    *,
    base_world: Path,
    catalog_path: Path,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, str]:
    """Materialize the authoritative SDF selected by the environment."""

    source = os.environ if environment is None else environment
    requested = str(source.get(SCENE_ENV) or "normal").strip()
    compiled = compile_authoritative_scene(
        base_world=base_world,
        catalog_path=catalog_path,
        scene_id=requested,
    )
    return materialize_authoritative_scene(compiled), compiled.world_scene


def materialize_authoritative_scene(
    compiled: CompiledAuthoritativeScene,
) -> Path:
    """Write exactly the bytes covered by the authoritative scene hash."""

    descriptor, rendered_name = tempfile.mkstemp(
        prefix=(
            f"openeta_{compiled.world_scene.replace('-', '_')}_"
            f"{compiled.authority_sha256[:12]}_"
        ),
        suffix=".sdf",
    )
    os.close(descriptor)
    rendered = Path(rendered_name)
    rendered.write_bytes(compiled.sdf_bytes)
    if hashlib.sha256(rendered.read_bytes()).hexdigest() != compiled.gazebo_sdf_sha256:
        rendered.unlink(missing_ok=True)
        raise RuntimeError("materialized Gazebo world failed authoritative hash check")
    return rendered
