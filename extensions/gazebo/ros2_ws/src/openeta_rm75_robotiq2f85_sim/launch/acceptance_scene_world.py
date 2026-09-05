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


def _native_target_topic_namespace(target_model: str) -> str:
    """Return a stable transport namespace for one detachable world body."""

    model = str(target_model).strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    if not model or any(character not in allowed for character in model):
        raise RuntimeError("native grasp target identity is invalid")
    # Preserve every existing normal-scene topic byte-for-byte.
    return "target" if model == "target_object" else model


@dataclass(frozen=True, slots=True)
class NativeTargetBinding:
    """One scene-owned portable-body binding shared by launch and runtime."""

    assignment_id: str
    target_model: str
    target_link: str
    attach_topic: str
    detach_topic: str
    state_topic: str
    collision_filter_state_topic: str
    collision_filter_state_request_topic: str
    collision_filter_state_ack_topic: str

    def to_dict(self) -> dict[str, str]:
        return {
            "assignment_id": self.assignment_id,
            "target_model": self.target_model,
            "target_link": self.target_link,
            "attach_topic": self.attach_topic,
            "detach_topic": self.detach_topic,
            "state_topic": self.state_topic,
            "collision_filter_state_topic": self.collision_filter_state_topic,
            "collision_filter_state_request_topic": self.collision_filter_state_request_topic,
            "collision_filter_state_ack_topic": self.collision_filter_state_ack_topic,
        }


def native_target_binding(
    *,
    assignment_id: str,
    target_model: str,
    target_link: str,
) -> NativeTargetBinding:
    assignment = str(assignment_id).strip()
    link = str(target_link).strip()
    if not assignment or not link:
        raise RuntimeError("native grasp assignment binding is invalid")
    namespace = _native_target_topic_namespace(target_model)
    prefix = f"/openeta/native_grasp/detachable_joint/{namespace}"
    return NativeTargetBinding(
        assignment_id=assignment,
        target_model=str(target_model).strip(),
        target_link=link,
        attach_topic=f"{prefix}/attach",
        detach_topic=f"{prefix}/detach",
        state_topic=f"{prefix}/state",
        collision_filter_state_topic=f"{prefix}/collision_filter_state",
        collision_filter_state_request_topic=f"{prefix}/collision_filter_state/request",
        collision_filter_state_ack_topic=f"{prefix}/collision_filter_state/ack",
    )


def scene_target_bindings(scene: Mapping[str, object]) -> tuple[NativeTargetBinding, ...]:
    """Resolve physical target bindings, synthesizing the legacy singleton."""

    raw_targets = scene.get("manipulation_targets")
    if raw_targets is None:
        return (
            native_target_binding(
                assignment_id="default",
                target_model="target_object",
                target_link="target_link",
            ),
        )
    if not isinstance(raw_targets, list) or not raw_targets:
        raise RuntimeError("acceptance scene manipulation targets are invalid")
    bindings: list[NativeTargetBinding] = []
    for raw in raw_targets:
        if not isinstance(raw, Mapping):
            raise RuntimeError("acceptance scene manipulation target is invalid")
        bindings.append(
            native_target_binding(
                assignment_id=str(raw.get("id") or ""),
                target_model=str(raw.get("target_object_id") or ""),
                target_link=str(raw.get("target_link") or ""),
            )
        )
    if len({item.assignment_id for item in bindings}) != len(bindings):
        raise RuntimeError("acceptance scene manipulation target identity is duplicated")
    if len({item.target_model for item in bindings}) != len(bindings):
        raise RuntimeError("acceptance scene manipulation target is duplicated")
    return tuple(bindings)


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
    """One collision model shared by Gazebo physics and MoveIt PlanningScene.

    Gazebo visuals are presentation assets and may use a detailed mesh.  The
    authoritative contract is deliberately scoped to collision geometry: the
    exact same primitives are materialized in Gazebo and published to MoveIt.
    """

    object_id: str
    pose_xyz: tuple[float, float, float]
    pose_quat_xyzw: tuple[float, float, float, float]
    gazebo_static: bool
    visual_count: int
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
    target_bindings: tuple[NativeTargetBinding, ...]
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
        return {
            "schema_version": AUTHORITATIVE_SCENE_SCHEMA_VERSION,
            "scene_id": self.scene_id,
            "world_scene": self.world_scene,
            "authority_sha256": self.authority_sha256,
            "gazebo_sdf_sha256": self.gazebo_sdf_sha256,
            "collision_manifest_sha256": self.collision_manifest_sha256,
            "gazebo_collision_object_ids": [item.object_id for item in self.objects],
            "moveit_collision_object_ids": [item.object_id for item in self.objects],
            "dynamic_object_ids": list(self.dynamic_object_ids),
            "target_bindings": [item.to_dict() for item in self.target_bindings],
            "primitive_count": sum(len(item.primitives) for item in self.objects),
            "visual_policy": "independent_high_fidelity_gazebo_assets",
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


_VISUAL_ENVELOPE_COLLISION_NAME = "openeta_visual_mesh_envelope"
_VISUAL_COLLISION_COVERAGE_TOLERANCE_M = 1e-6


def _package_visual_obj_path(base_world: Path, uri: str) -> Path | None:
    """Resolve an offline visual mesh to its source OBJ when available.

    The release assets retain an OBJ beside the rendered GLB.  OBJ parsing is
    intentionally dependency-free, and an unrelated presentation-only URI is
    left independent from collision geometry rather than being guessed.
    """

    prefix = "model://"
    if not uri.startswith(prefix):
        return None
    model_path = uri[len(prefix) :]
    package_name, separator, relative_path = model_path.partition("/")
    package_root = base_world.parent.parent
    if (
        not separator
        or package_name != package_root.name
        or not relative_path
    ):
        return None
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    source = (package_root / relative).with_suffix(".obj")
    return source if source.is_file() else None


def _obj_vertex_bounds(path: Path) -> tuple[
    tuple[float, float, float], tuple[float, float, float]
] | None:
    """Return finite OBJ vertex bounds without requiring a mesh dependency."""

    lower = [math.inf, math.inf, math.inf]
    upper = [-math.inf, -math.inf, -math.inf]
    found = False
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) < 4 or fields[0] != "v":
            continue
        try:
            vertex = [float(value) for value in fields[1:4]]
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in vertex):
            continue
        found = True
        for axis, value in enumerate(vertex):
            lower[axis] = min(lower[axis], value)
            upper[axis] = max(upper[axis], value)
    if not found:
        return None
    return tuple(lower), tuple(upper)  # type: ignore[return-value]


def _visual_mesh_bounds(
    visual: ET.Element,
    *,
    base_world: Path,
    owner: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return a conservative link-frame AABB for one known visual mesh."""

    mesh = visual.find("geometry/mesh")
    uri = (mesh.findtext("uri") or "").strip() if mesh is not None else ""
    source = _package_visual_obj_path(base_world, uri) if uri else None
    vertices = _obj_vertex_bounds(source) if source is not None else None
    if vertices is None:
        return None
    scale_text = mesh.findtext("scale") if mesh is not None else None
    try:
        scale = (
            tuple(float(value) for value in scale_text.split())
            if scale_text
            else (1.0, 1.0, 1.0)
        )
    except ValueError as exc:
        raise RuntimeError(f"authoritative visual mesh scale is invalid: {owner}") from exc
    if (
        len(scale) != 3
        or any(not math.isfinite(value) or value <= 0.0 for value in scale)
    ):
        raise RuntimeError(f"authoritative visual mesh scale is invalid: {owner}")
    visual_xyz, visual_quat = _pose(visual, owner=owner)
    raw_lower, raw_upper = vertices
    lower = [math.inf, math.inf, math.inf]
    upper = [-math.inf, -math.inf, -math.inf]
    for x in (raw_lower[0] * scale[0], raw_upper[0] * scale[0]):
        for y in (raw_lower[1] * scale[1], raw_upper[1] * scale[1]):
            for z in (raw_lower[2] * scale[2], raw_upper[2] * scale[2]):
                rotated = _quaternion_rotate(visual_quat, (x, y, z))
                for axis, value in enumerate(rotated):
                    coordinate = visual_xyz[axis] + value
                    lower[axis] = min(lower[axis], coordinate)
                    upper[axis] = max(upper[axis], coordinate)
    return tuple(lower), tuple(upper)  # type: ignore[return-value]


def _materialize_visual_collision_envelopes(
    world: ET.Element,
    *,
    base_world: Path,
) -> None:
    """Fill any visual extent absent from the shared Gazebo/MoveIt geometry.

    Detailed meshes remain visual assets; low-complexity collision proxies are
    still preferred.  When a proxy fails to bound the mesh, add one exact
    mesh-AABB envelope to the *authoritative* SDF.  Gazebo and MoveIt then
    consume the same conservative geometry and a visual-only protrusion can
    no longer be planned through.
    """

    for model in world.findall("model"):
        model_id = str(model.get("name") or "")
        if not model_id:
            raise RuntimeError("authoritative model identity is invalid")
        for link in model.findall("link"):
            link_id = str(link.get("name") or "")
            if not link_id:
                raise RuntimeError(f"authoritative link identity is invalid: {model_id}")
            mesh_bounds = [
                bounds
                for visual in link.findall("visual")
                if (
                    bounds := _visual_mesh_bounds(
                        visual,
                        base_world=base_world,
                        owner=f"{model_id}/{link_id}/{visual.get('name') or 'visual'}",
                    )
                )
                is not None
            ]
            if not mesh_bounds:
                continue
            visual_lower = tuple(
                min(bounds[0][axis] for bounds in mesh_bounds) for axis in range(3)
            )
            visual_upper = tuple(
                max(bounds[1][axis] for bounds in mesh_bounds) for axis in range(3)
            )
            collisions = link.findall("collision")
            primitive_bounds = [
                _primitive_from_collision(
                    collision,
                    model_id=model_id,
                    link_xyz=(0.0, 0.0, 0.0),
                    link_quat=(0.0, 0.0, 0.0, 1.0),
                ).local_bounds()
                for collision in collisions
            ]
            if primitive_bounds:
                collision_lower = tuple(
                    min(bounds[0][axis] for bounds in primitive_bounds)
                    for axis in range(3)
                )
                collision_upper = tuple(
                    max(bounds[1][axis] for bounds in primitive_bounds)
                    for axis in range(3)
                )
                covered = all(
                    collision_lower[axis]
                    <= visual_lower[axis] + _VISUAL_COLLISION_COVERAGE_TOLERANCE_M
                    and collision_upper[axis]
                    >= visual_upper[axis] - _VISUAL_COLLISION_COVERAGE_TOLERANCE_M
                    for axis in range(3)
                )
                if covered:
                    continue
            names = {str(collision.get("name") or "") for collision in collisions}
            if _VISUAL_ENVELOPE_COLLISION_NAME in names:
                raise RuntimeError(
                    f"authoritative visual envelope does not cover mesh: {model_id}/{link_id}"
                )
            center = tuple(
                (visual_lower[axis] + visual_upper[axis]) / 2.0 for axis in range(3)
            )
            size = tuple(
                visual_upper[axis] - visual_lower[axis] for axis in range(3)
            )
            if any(value <= 0.0 or not math.isfinite(value) for value in size):
                raise RuntimeError(
                    f"authoritative visual mesh bounds are invalid: {model_id}/{link_id}"
                )
            envelope = ET.SubElement(
                link,
                "collision",
                {"name": _VISUAL_ENVELOPE_COLLISION_NAME},
            )
            ET.SubElement(envelope, "pose").text = _text([*center, 0.0, 0.0, 0.0])
            geometry = ET.SubElement(envelope, "geometry")
            box = ET.SubElement(geometry, "box")
            ET.SubElement(box, "size").text = _text(list(size))


def _apply_model_pose_overrides(
    world: ET.Element,
    *,
    overrides: object,
) -> tuple[str, ...]:
    """Move existing dynamic bodies without creating a second geometry source."""

    if not isinstance(overrides, list) or not overrides:
        raise RuntimeError("acceptance scene model pose overrides are invalid")
    overridden: list[str] = []
    for raw in overrides:
        if not isinstance(raw, Mapping):
            raise RuntimeError("acceptance scene model pose override is invalid")
        model_id = str(raw.get("id") or "").strip()
        if not model_id or model_id in overridden:
            raise RuntimeError("acceptance scene model pose override identity is invalid")
        model = world.find(f"model[@name='{model_id}']")
        if model is None:
            raise RuntimeError(
                f"acceptance scene model pose override is unknown: {model_id}"
            )
        if model.findtext("static", "false").strip().lower() == "true":
            raise RuntimeError(
                f"acceptance scene cannot move static model: {model_id}"
            )
        xyz = _numbers(raw.get("pose_xyz"), 3)
        rpy = _numbers(raw.get("pose_rpy"), 3)
        pose = model.find("pose")
        if pose is None:
            pose = ET.Element("pose")
            model.insert(0, pose)
        if pose.attrib.get("relative_to"):
            raise RuntimeError(
                f"acceptance scene cannot override relative pose: {model_id}"
            )
        pose.text = _text([*xyz, *rpy])
        overridden.append(model_id)
    return tuple(overridden)


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
    if obstacles:
        raise RuntimeError(
            "acceptance catalog cannot inject geometry into the authoritative world"
        )
    if scene.get("canonical_world_complete") is not True:
        raise RuntimeError("acceptance scene must use the authoritative world")

    tree = ET.parse(base_world)
    world = tree.getroot().find("world")
    if world is None:
        raise RuntimeError("canonical pick-place world is invalid")
    model_pose_overrides = scene.get("model_pose_overrides")
    if model_pose_overrides is not None:
        _apply_model_pose_overrides(world, overrides=model_pose_overrides)
    _materialize_visual_collision_envelopes(world, base_world=base_world)
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


def _attached_collision_filter_expected(
    binding: NativeTargetBinding,
) -> dict[str, str]:
    return {
        "target_model": binding.target_model,
        "target_link": binding.target_link,
        "state_topic": binding.collision_filter_state_topic,
        "state_request_topic": binding.collision_filter_state_request_topic,
        "state_ack_topic": binding.collision_filter_state_ack_topic,
        "robot_mask": str(ROBOT_COLLISION_FILTER_MASK),
        "detached_mask": str(DETACHED_TARGET_COLLISION_FILTER_MASK),
        "attached_mask": str(ATTACHED_TARGET_COLLISION_FILTER_MASK),
    }


def _materialize_attached_collision_filter_contract(
    world: ET.Element,
    *,
    bindings: Sequence[NativeTargetBinding],
) -> None:
    """Create exactly one independently-addressed filter per sort target."""

    plugins = [
        plugin
        for plugin in world.findall("plugin")
        if plugin.get("name") == "openeta::gazebo::AttachedCollisionFilter"
    ]
    if len(plugins) != 1:
        raise RuntimeError(
            "authoritative attached collision-filter plugin is missing"
        )
    legacy = native_target_binding(
        assignment_id="default",
        target_model="target_object",
        target_link="target_link",
    )
    expected = _attached_collision_filter_expected(legacy)
    actual = {
        key: (plugins[0].findtext(key) or "").strip() for key in expected
    }
    if actual != expected:
        raise RuntimeError(
            "authoritative attached collision-filter contract is invalid"
        )
    if len(bindings) == 1 and bindings[0].target_model == legacy.target_model:
        return
    world.remove(plugins[0])
    for binding in bindings:
        plugin = ET.SubElement(
            world,
            "plugin",
            {
                "filename": "libopeneta_attached_collision_filter_system.so",
                "name": "openeta::gazebo::AttachedCollisionFilter",
            },
        )
        for key, value in _attached_collision_filter_expected(binding).items():
            ET.SubElement(plugin, key).text = value


def _validate_attached_collision_filter_contract(
    world: ET.Element,
    *,
    bindings: Sequence[NativeTargetBinding],
) -> None:
    plugins = [
        plugin
        for plugin in world.findall("plugin")
        if plugin.get("name") == "openeta::gazebo::AttachedCollisionFilter"
    ]
    if len(plugins) != len(bindings):
        raise RuntimeError("authoritative attached collision-filter plugin count is invalid")
    actual_by_target = {
        (plugin.findtext("target_model") or "").strip(): plugin for plugin in plugins
    }
    if len(actual_by_target) != len(plugins):
        raise RuntimeError("authoritative attached collision-filter target is duplicated")
    for binding in bindings:
        plugin = actual_by_target.get(binding.target_model)
        expected = _attached_collision_filter_expected(binding)
        actual = (
            {key: (plugin.findtext(key) or "").strip() for key in expected}
            if plugin is not None
            else {}
        )
        if actual != expected:
            raise RuntimeError("authoritative attached collision-filter contract is invalid")


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
                primitives=tuple(primitives),
                bounding_box_xyz=tuple(
                    upper[axis] - lower[axis] for axis in range(3)
                ),
            )
        )
    return tuple(objects)


def _xy_bounds_are_separated(
    left: tuple[tuple[float, float, float], tuple[float, float, float]],
    right: tuple[tuple[float, float, float], tuple[float, float, float]],
    *,
    clearance_m: float,
) -> bool:
    return any(
        left[1][axis] + clearance_m <= right[0][axis]
        or right[1][axis] + clearance_m <= left[0][axis]
        for axis in range(2)
    )


def _validate_overridden_dynamic_layout(
    *,
    scene: Mapping[str, object],
    objects: Sequence[AuthoritativeObject],
) -> None:
    """Reject randomized starts that are not collision-free workcell layouts."""

    overrides = scene.get("model_pose_overrides")
    if overrides is None:
        return
    validation = scene.get("layout_validation")
    if not isinstance(validation, Mapping):
        raise RuntimeError("acceptance randomized layout validation is missing")
    support_id = str(validation.get("support_object_id") or "").strip()
    if not support_id:
        raise RuntimeError("acceptance randomized layout support is invalid")
    try:
        support_margin_m = float(validation.get("support_margin_m", 0.0))
        clearance_m = float(validation.get("minimum_xy_clearance_m", 0.0))
        maximum_initial_drop_m = float(
            validation.get("maximum_initial_drop_m", 0.0)
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("acceptance randomized layout clearance is invalid") from exc
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (support_margin_m, clearance_m, maximum_initial_drop_m)
    ):
        raise RuntimeError("acceptance randomized layout clearance is invalid")

    by_id = {item.object_id: item for item in objects}
    support = by_id.get(support_id)
    if support is None or not support.gazebo_static:
        raise RuntimeError("acceptance randomized layout support is invalid")
    support_bounds = support.world_bounds()
    support_surface_z = support_bounds[1][2]
    movable = tuple(item for item in objects if not item.gazebo_static)
    if not movable:
        raise RuntimeError("acceptance randomized layout has no dynamic objects")
    for item in movable:
        lower, upper = item.world_bounds()
        if any(
            lower[axis] < support_bounds[0][axis] + support_margin_m
            or upper[axis] > support_bounds[1][axis] - support_margin_m
            for axis in range(2)
        ):
            raise RuntimeError(
                f"acceptance randomized model leaves support bounds: {item.object_id}"
            )
        if lower[2] < support_surface_z - 1e-6:
            raise RuntimeError(
                f"acceptance randomized model penetrates support: {item.object_id}"
            )
        if lower[2] > support_surface_z + maximum_initial_drop_m:
            raise RuntimeError(
                f"acceptance randomized model starts unsupported: {item.object_id}"
            )

    regions = scene.get("placement_regions")
    region_ids = (
        tuple(
            str(raw.get("id") or "")
            for raw in regions
            if isinstance(raw, Mapping)
        )
        if isinstance(regions, list)
        else ()
    )
    physical_regions = tuple(by_id[item] for item in region_ids if item in by_id)
    for movable_object in movable:
        movable_bounds = movable_object.world_bounds()
        for region in physical_regions:
            if not _xy_bounds_are_separated(
                movable_bounds,
                region.world_bounds(),
                clearance_m=clearance_m,
            ):
                raise RuntimeError(
                    "acceptance randomized model overlaps placement support: "
                    f"{movable_object.object_id}/{region.object_id}"
                )

    exclusions = validation.get("exclusion_regions", [])
    if not isinstance(exclusions, list):
        raise RuntimeError("acceptance randomized layout exclusions are invalid")
    exclusion_bounds: list[
        tuple[str, tuple[tuple[float, float, float], tuple[float, float, float]]]
    ] = []
    seen_exclusions: set[str] = set()
    for raw in exclusions:
        if not isinstance(raw, Mapping):
            raise RuntimeError("acceptance randomized layout exclusion is invalid")
        exclusion_id = str(raw.get("id") or "").strip()
        if not exclusion_id or exclusion_id in seen_exclusions:
            raise RuntimeError("acceptance randomized layout exclusion identity is invalid")
        seen_exclusions.add(exclusion_id)
        center = _numbers(raw.get("center_xy"), 2)
        size = _numbers(raw.get("size_xy_m"), 2, positive=True)
        exclusion_bounds.append(
            (
                exclusion_id,
                (
                    (center[0] - size[0] / 2.0, center[1] - size[1] / 2.0, 0.0),
                    (center[0] + size[0] / 2.0, center[1] + size[1] / 2.0, 0.0),
                ),
            )
        )
    for movable_object in movable:
        movable_bounds = movable_object.world_bounds()
        for exclusion_id, excluded_bounds in exclusion_bounds:
            if not _xy_bounds_are_separated(
                movable_bounds,
                excluded_bounds,
                clearance_m=clearance_m,
            ):
                raise RuntimeError(
                    "acceptance randomized model enters exclusion region: "
                    f"{movable_object.object_id}/{exclusion_id}"
                )

    for index, left in enumerate(movable):
        for right in movable[index + 1 :]:
            if not _xy_bounds_are_separated(
                left.world_bounds(),
                right.world_bounds(),
                clearance_m=clearance_m,
            ):
                raise RuntimeError(
                    "acceptance randomized dynamic models overlap: "
                    f"{left.object_id}/{right.object_id}"
                )


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
    bindings = scene_target_bindings(scene)
    for binding in bindings:
        if binding.target_model not in by_id:
            raise RuntimeError(
                f"authoritative sort target collision model is missing: {binding.target_model}"
            )
        model = world.find(f"model[@name='{binding.target_model}']")
        if model is None or model.find(f"link[@name='{binding.target_link}']") is None:
            raise RuntimeError(
                f"authoritative sort target link is missing: "
                f"{binding.target_model}/{binding.target_link}"
            )
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
            # The catalog locks the hand-authored source proxies.  A visual
            # mesh envelope is a deterministic compiler product, derived
            # from the same authoritative SDF, and is verified separately
            # by its reserved collision name rather than being mistaken for
            # a catalog-editable task primitive.
            source_primitives = tuple(
                primitive
                for primitive in target.primitives
                if primitive.name != _VISUAL_ENVELOPE_COLLISION_NAME
            )
            if len(expected_primitives) != len(source_primitives):
                raise RuntimeError(
                    "authoritative target primitive count differs from the task contract"
                )
            for raw, primitive in zip(
                expected_primitives, source_primitives, strict=True
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
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    raw_scene = payload.get("scenes", {}).get(selected) if isinstance(payload, Mapping) else None
    if not isinstance(raw_scene, Mapping):
        raise RuntimeError("authoritative scene catalog binding is invalid")
    target_bindings = scene_target_bindings(raw_scene)
    for binding in target_bindings:
        target_model = world.find(f"model[@name='{binding.target_model}']")
        if target_model is None:
            raise RuntimeError(
                f"authoritative target collision model is missing: {binding.target_model}"
            )
        _set_collision_filter_mask(
            target_model,
            mask=DETACHED_TARGET_COLLISION_FILTER_MASK,
        )
    _materialize_attached_collision_filter_contract(
        world,
        bindings=target_bindings,
    )
    _validate_attached_collision_filter_contract(world, bindings=target_bindings)
    objects = _authoritative_objects(world)
    _validate_overridden_dynamic_layout(scene=raw_scene, objects=objects)
    required_ids = {"work_table", *(item.target_model for item in target_bindings)}
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
        target_bindings=target_bindings,
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
