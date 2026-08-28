"""Deterministic analytic legality gates for placement qualification.

The checks in this module are deliberately limited to proofs that can be made
from immutable candidate geometry and the cloned PlanningScene snapshot.  They
run before IK, but never replace MoveIt's state-validity or plan-only proof.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from adapter.protocol import JsonDict
from agent.runtime.collision_geometry import (
    ProjectedCollisionPrimitive,
    collision_geometry_volume_centroid,
    compound_axis_aligned_bounds,
    orientation_invariant_radius_m,
    project_collision_geometry,
    support_face_alignment_cosine,
)


GOAL_LEGALITY_SCHEMA = "openeta.placement_goal_legality.v1"
PAIR_LEGALITY_SCHEMA = "openeta.grasp_placement_pair_legality.v1"
GRASP_TARGET_CLOSING_SCHEMA = "openeta.grasp_target_closing_alignment.v1"
_ROTATION_TOLERANCE = 1e-4
_POSE_TOLERANCE_M = 1e-5
_ORIENTATION_TOLERANCE = 1e-4


Vector = tuple[float, float, float]
Rotation = tuple[Vector, Vector, Vector]
Transform = tuple[Rotation, Vector]
Obb = tuple[Vector, Rotation, Vector]


def _finite_vector(value: object, length: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != length:
        return None
    try:
        parsed = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return parsed if all(math.isfinite(item) for item in parsed) else None


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _cross(left: Sequence[float], right: Sequence[float]) -> Vector:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _determinant(rotation: Rotation) -> float:
    return _dot(rotation[0], _cross(rotation[1], rotation[2]))


def _rotation_from_quaternion(value: object) -> Rotation | None:
    quaternion = _finite_vector(value, 4)
    if quaternion is None:
        return None
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm <= 1e-12 or abs(norm - 1.0) > 5e-3:
        return None
    x, y, z, w = (item / norm for item in quaternion)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _strict_rotation(value: object) -> Rotation | None:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 3
    ):
        return None
    rows: list[list[float]] = []
    for raw_row in value:
        row = _finite_vector(raw_row, 3)
        if row is None:
            return None
        rows.append(row)
    rotation: Rotation = tuple(tuple(row) for row in rows)  # type: ignore[assignment]
    for row_index in range(3):
        for other_row_index in range(3):
            expected = 1.0 if row_index == other_row_index else 0.0
            if (
                abs(_dot(rotation[row_index], rotation[other_row_index]) - expected)
                > _ROTATION_TOLERANCE
            ):
                return None
    if abs(_determinant(rotation) - 1.0) > _ROTATION_TOLERANCE:
        return None
    return rotation


def rigid_pose(value: object) -> Transform | None:
    """Parse one finite, proper SE(3) pose without silently repairing rotation."""

    if not isinstance(value, Mapping):
        return None
    xyz = _finite_vector(
        value.get("xyz") or value.get("translation_xyz") or value.get("position"),
        3,
    )
    rotation = _strict_rotation(value.get("rotation_matrix"))
    if rotation is None and value.get("rotation_matrix") is None:
        rotation = _rotation_from_quaternion(value.get("quat_xyzw") or value.get("quaternion_xyzw"))
    if xyz is None or rotation is None:
        return None
    return rotation, tuple(xyz)  # type: ignore[return-value]


def _transform_matrix(value: object) -> Transform | None:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 4
    ):
        return None
    rows = [_finite_vector(row, 4) for row in value]
    if any(row is None for row in rows):
        return None
    matrix = [row for row in rows if row is not None]
    rotation = _strict_rotation([row[:3] for row in matrix[:3]])
    if rotation is None or any(
        abs(left - right) > 1e-6
        for left, right in zip(matrix[3], (0.0, 0.0, 0.0, 1.0), strict=True)
    ):
        return None
    return rotation, (matrix[0][3], matrix[1][3], matrix[2][3])


def _columns(rotation: Rotation) -> tuple[Vector, Vector, Vector]:
    return tuple(tuple(rotation[row][column] for row in range(3)) for column in range(3))  # type: ignore[return-value]


def _rotate(rotation: Rotation, vector: Sequence[float]) -> Vector:
    return tuple(_dot(row, vector) for row in rotation)  # type: ignore[return-value]


def _transpose(rotation: Rotation) -> Rotation:
    return _columns(rotation)


def _compose(left: Transform, right: Transform) -> Transform:
    left_rotation, left_xyz = left
    right_rotation, right_xyz = right
    right_columns = _columns(right_rotation)
    rotation: Rotation = tuple(
        tuple(_dot(left_rotation[row], right_columns[column]) for column in range(3))
        for row in range(3)
    )  # type: ignore[assignment]
    translated = _rotate(left_rotation, right_xyz)
    xyz = tuple(left_xyz[index] + translated[index] for index in range(3))
    return rotation, xyz  # type: ignore[return-value]


def _transform_payload(transform: Transform, *, convention: str) -> JsonDict:
    rotation, xyz = transform
    return {
        "frame": "world",
        "transform_matrix": [[*map(float, rotation[row]), float(xyz[row])] for row in range(3)]
        + [[0.0, 0.0, 0.0, 1.0]],
        "convention": convention,
    }


def _pose_payload(transform: Transform, *, convention: str) -> JsonDict:
    rotation, xyz = transform
    return {
        "frame": "world",
        "translation_xyz": list(xyz),
        "rotation_matrix": [list(row) for row in rotation],
        "convention": convention,
    }


def _inverse(transform: Transform) -> Transform:
    rotation, xyz = transform
    inverse_rotation = _transpose(rotation)
    translated = _rotate(inverse_rotation, xyz)
    return inverse_rotation, tuple(-value for value in translated)  # type: ignore[return-value]


def _rotation_distance(left: Rotation, right: Rotation) -> float:
    relative = _compose((left, (0.0, 0.0, 0.0)), (_transpose(right), (0.0, 0.0, 0.0)))[0]
    # ``acos((trace(R) - 1) / 2)`` is badly conditioned close to zero.  The
    # compiled poses are intentionally serialized to a fixed precision, so
    # two physically identical rotations can otherwise appear roughly 2 mrad
    # apart after composing the rounded matrices.  The skew/trace atan2 form
    # has the same SO(3) meaning but remains stable at the identity.
    skew = (
        relative[2][1] - relative[1][2],
        relative[0][2] - relative[2][0],
        relative[1][0] - relative[0][1],
    )
    sine = 0.5 * math.sqrt(_dot(skew, skew))
    cosine = 0.5 * (relative[0][0] + relative[1][1] + relative[2][2] - 1.0)
    return math.atan2(sine, cosine)


def _obb(transform: Transform, size_xyz: Sequence[float]) -> Obb:
    rotation, xyz = transform
    return xyz, rotation, tuple(float(value) / 2.0 for value in size_xyz)  # type: ignore[return-value]


def _projected_body_geometry(
    spec: Mapping[str, Any],
    transform: Transform,
) -> tuple[ProjectedCollisionPrimitive, ...]:
    """Project the same compound body supplied to MoveIt at one pose."""

    rotation, xyz = transform
    try:
        return project_collision_geometry(
            object_xyz=xyz,
            object_rotation=rotation,
            primitives=spec.get("primitives") or (),
            fallback_size_xyz=spec.get("size_xyz"),
        )
    except ValueError:
        return ()


def _camera_to_world_opencv(value: object) -> Transform | None:
    """Parse the camera convention accepted by the grasp compiler.

    New observation adapters publish OpenCV camera axes explicitly.  Missing
    frame metadata retains the compiler's legacy OpenGL interpretation so the
    legality evidence and the compiled terminal cannot disagree merely
    because an older artifact is replayed.
    """

    if not isinstance(value, Mapping):
        return None
    rotation: Rotation | None = None
    xyz: list[float] | None = None
    flat = _finite_vector(value.get("mat"), 9)
    if flat is not None:
        xyz = _finite_vector(value.get("pos"), 3)
        if str(value.get("matrix_layout") or "row_major").lower() == "column_major":
            rotation = _strict_rotation(
                [[flat[row + column * 3] for column in range(3)] for row in range(3)]
            )
        else:
            rotation = _strict_rotation([flat[0:3], flat[3:6], flat[6:9]])
    elif _finite_vector(value.get("quat_xyzw"), 4) is not None:
        xyz = _finite_vector(value.get("pos"), 3)
        rotation = _rotation_from_quaternion(value.get("quat_xyzw"))
    else:
        for key in ("camera_to_world", "pose_mat", "matrix"):
            transform = _transform_matrix(value.get(key))
            if transform is not None:
                rotation, transform_xyz = transform
                xyz = list(transform_xyz)
                break
    if rotation is None or xyz is None:
        return None
    raw_frame = str(value.get("camera_frame") or "opengl")
    camera_frame = raw_frame.strip().lower().replace("-", "_").replace(" ", "_")
    if camera_frame in {"opencv", "opencv_optical", "cv"}:
        return rotation, tuple(xyz)  # type: ignore[return-value]
    if camera_frame not in {"opengl", "opengl_renderer", "mujoco", "renderer"}:
        return None
    opencv_to_opengl: Rotation = (
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    return _compose(
        (rotation, tuple(xyz)),  # type: ignore[arg-type]
        (opencv_to_opengl, (0.0, 0.0, 0.0)),
    )


def _primitive_axis_half_extent(
    primitive: ProjectedCollisionPrimitive,
    axis: Sequence[float],
) -> float:
    primitive_axes = _columns(primitive.rotation_rows)
    if primitive.shape == "box" and primitive.size_xyz is not None:
        return sum(
            primitive.size_xyz[index] * 0.5 * abs(_dot(axis, primitive_axes[index]))
            for index in range(3)
        )
    if (
        primitive.shape == "cylinder"
        and primitive.radius_m is not None
        and primitive.length_m is not None
    ):
        axial = min(1.0, abs(_dot(axis, primitive_axes[2])))
        return (
            primitive.radius_m * math.sqrt(max(0.0, 1.0 - axial * axial))
            + primitive.length_m * 0.5 * axial
        )
    raise ValueError("collision primitive geometry is incomplete")


def evaluate_grasp_target_closing_alignment(
    descriptor: Mapping[str, Any],
    *,
    scene: object,
) -> JsonDict:
    """Measure an unchanged model grasp against exact scene target geometry.

    The result is an ordering signal, never an empirical hard rejection.  It
    projects the same compound primitives supplied to MoveIt through the
    model's GraspNet closing axis, restricted to the finger-height section at
    the reported fingertip centre.  This distinguishes a centred handle pinch
    from a visually centred pose whose jaws would need to span an entire long
    workpiece, while preserving every candidate for exhaustive recovery.
    """

    candidate = descriptor.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    parameters = candidate.get("compile_parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    raw = parameters.get("camera_pose")
    raw = raw if isinstance(raw, Mapping) else {}
    candidate_id = str(candidate.get("id") or raw.get("id") or descriptor.get("candidate_id") or "")
    result: JsonDict = {
        "schema_version": GRASP_TARGET_CLOSING_SCHEMA,
        "candidate_id": candidate_id,
        "source": "planning_scene_collision_geometry",
        "closing_axis": "graspnet_local_y",
        "binormal_axis": "graspnet_local_z",
        "ordering_only": True,
        "pose_modified": False,
        "evaluated": False,
    }

    grasp_rotation = _strict_rotation(raw.get("rotation_matrix"))
    tip_camera = _finite_vector(raw.get("gripper_tip_position_xyz"), 3)
    camera_to_world = _camera_to_world_opencv(parameters.get("camera_extrinsics"))
    try:
        finger_height_m = float(raw.get("height"))
        maximum_aperture_m = float(parameters.get("max_gripper_width_m"))
    except (TypeError, ValueError):
        finger_height_m = math.nan
        maximum_aperture_m = math.nan
    if (
        grasp_rotation is None
        or tip_camera is None
        or camera_to_world is None
        or not math.isfinite(finger_height_m)
        or finger_height_m <= 0.0
        or not math.isfinite(maximum_aperture_m)
        or maximum_aperture_m <= 0.0
    ):
        result.update(
            reason="grasp_or_calibration_geometry_unavailable",
            geometry_available=False,
        )
        return result

    camera_rotation, camera_xyz = camera_to_world
    world_grasp = _compose(
        (camera_rotation, camera_xyz),
        (grasp_rotation, (0.0, 0.0, 0.0)),
    )[0]
    tip_offset = _rotate(camera_rotation, tip_camera)
    tip_world = tuple(camera_xyz[index] + tip_offset[index] for index in range(3))
    _, closing_axis, binormal_axis = _columns(world_grasp)

    target_id = str(scene.get("target_id") or "") if isinstance(scene, Mapping) else ""
    world_specs = _scene_mapping(scene, "world_specs")
    attached_specs = _scene_mapping(scene, "attached_specs")
    target_spec = attached_specs.get(target_id) or world_specs.get(target_id)
    target_pose_value = (
        {
            "xyz": target_spec.get("pose_xyz"),
            "quat_xyzw": target_spec.get("pose_quat_xyzw"),
            **(
                {"rotation_matrix": target_spec["rotation_matrix"]}
                if isinstance(target_spec, Mapping)
                and target_spec.get("rotation_matrix") is not None
                else {}
            ),
        }
        if isinstance(target_spec, Mapping)
        else None
    )
    target_pose = rigid_pose(target_pose_value)
    primitives = (
        _projected_body_geometry(target_spec, target_pose)
        if isinstance(target_spec, Mapping) and target_pose is not None
        else ()
    )
    if not primitives:
        result.update(
            reason="scene_target_geometry_unavailable",
            geometry_available=False,
        )
        return result

    tip_binormal = _dot(tip_world, binormal_axis)
    section_low = tip_binormal - finger_height_m * 0.5
    section_high = tip_binormal + finger_height_m * 0.5
    section: list[ProjectedCollisionPrimitive] = []
    for primitive in primitives:
        center = _dot(primitive.center_xyz, binormal_axis)
        half_extent = _primitive_axis_half_extent(primitive, binormal_axis)
        if center + half_extent >= section_low and center - half_extent <= section_high:
            section.append(primitive)

    result.update(
        geometry_available=True,
        primitive_count=len(primitives),
        section_primitive_count=len(section),
        binormal_window_m=finger_height_m,
        maximum_aperture_m=maximum_aperture_m,
        gripper_tip_world_xyz=[float(value) for value in tip_world],
    )
    if not section:
        result.update(
            reason="finger_section_does_not_intersect_target",
            evaluated=True,
            section_intersects_target=False,
            aperture_feasible=False,
        )
        return result

    intervals = []
    for primitive in section:
        center = _dot(primitive.center_xyz, closing_axis)
        half_extent = _primitive_axis_half_extent(primitive, closing_axis)
        intervals.append((center - half_extent, center + half_extent))
    low = min(interval[0] for interval in intervals)
    high = max(interval[1] for interval in intervals)
    tip_closing = _dot(tip_world, closing_axis)
    target_span = high - low
    correction = (low + high) * 0.5 - tip_closing
    required_aperture = 2.0 * max(
        abs(low - tip_closing),
        abs(high - tip_closing),
    )
    denominator = max(target_span, 0.02)
    result.update(
        reason="scene_geometry_measured",
        evaluated=True,
        section_intersects_target=True,
        target_span_m=target_span,
        correction_m=correction,
        correction_world_xyz=[float(correction * value) for value in closing_axis],
        centering_ratio=abs(correction) / denominator,
        required_aperture_m=required_aperture,
        aperture_feasible=required_aperture <= maximum_aperture_m + 1e-9,
    )
    return result


def _projected_obb(primitive: ProjectedCollisionPrimitive) -> Obb:
    return (
        primitive.center_xyz,
        primitive.rotation_rows,
        primitive.enclosing_obb_half_sizes(),
    )


def _spec_obbs(spec: object) -> tuple[Obb, ...]:
    return tuple(
        _projected_obb(item) for item in _spec_projected_geometry(spec)
    )


def _spec_projected_geometry(
    spec: object,
) -> tuple[ProjectedCollisionPrimitive, ...]:
    if not isinstance(spec, Mapping):
        return ()
    pose = rigid_pose(
        {
            "xyz": spec.get("pose_xyz"),
            "quat_xyzw": spec.get("pose_quat_xyzw"),
            **(
                {"rotation_matrix": spec["rotation_matrix"]}
                if spec.get("rotation_matrix") is not None
                else {}
            ),
        }
    )
    if pose is None:
        return ()
    return _projected_body_geometry(spec, pose)


def _non_support_obbs(
    spec: object,
    *,
    support_z_m: object,
) -> tuple[tuple[Obb, ...], int]:
    """Separate a compound container's floor from its collision walls.

    Placement may touch the primitive whose upper support face is exactly the
    catalogued support height. Every other primitive remains a normal static
    obstacle. This avoids the previous whole-object exemption that treated a
    bin's base and four walls as if they were one infinite support plane.
    """

    if (
        isinstance(support_z_m, bool)
        or not isinstance(support_z_m, (int, float))
        or not math.isfinite(float(support_z_m))
    ):
        return _spec_obbs(spec), 0
    geometry = _spec_projected_geometry(spec)
    if not geometry:
        return (), 0
    support_z = float(support_z_m)
    scale = max(
        1.0,
        abs(support_z),
        *(
            abs(value)
            for primitive in geometry
            for value in primitive.center_xyz
        ),
    )
    numeric_band = 64.0 * math.ulp(scale)
    barriers: list[Obb] = []
    support_count = 0
    for primitive in geometry:
        top_z = primitive.center_xyz[2] + primitive.axis_half_extent(2)
        if abs(top_z - support_z) <= numeric_band:
            support_count += 1
        else:
            barriers.append(_projected_obb(primitive))
    return tuple(barriers), support_count


def _effective_release_z_offset(
    *,
    configured_drop_height_m: float,
    support_object_id: str,
    support_z_m: float | None,
    support_spec: object,
    clearance_m: float,
) -> tuple[float, JsonDict]:
    """Select a release height from support geometry, without moving the goal.

    A flat support keeps the configured gravity-drop height.  A compound
    support with collision walls is a container: the AnyPlace goal is already
    inside its opening, so lifting the terminal above the highest wall both
    changes the intended interior release and needlessly consumes directional
    arm reach.  For containers, retain only the scene-derived contact
    clearance and let pair legality plus MoveIt prove the gripper, payload, and
    complete path against every wall.
    """

    evidence: JsonDict = {
        "source": "configured_drop_height",
        "configured_drop_height_m": configured_drop_height_m,
        "container_clearance_m": 0.0,
        "effective_offset_m": configured_drop_height_m,
        "support_object_id": support_object_id,
        "support_geometry_available": False,
        "support_barrier_count": 0,
    }
    if support_z_m is None or not support_object_id:
        return configured_drop_height_m, evidence
    geometry = _spec_projected_geometry(support_spec)
    if not geometry:
        return configured_drop_height_m, evidence

    barriers, support_primitive_count = _non_support_obbs(
        support_spec,
        support_z_m=support_z_m,
    )
    maximum_z = max(
        primitive.center_xyz[2] + primitive.axis_half_extent(2)
        for primitive in geometry
    )
    container_clearance = (
        min(configured_drop_height_m, max(0.0, clearance_m))
        if barriers
        else configured_drop_height_m
    )
    evidence.update(
        {
            "source": (
                "container_interior_clearance"
                if barriers
                else "configured_drop_height"
            ),
            "container_clearance_m": container_clearance if barriers else 0.0,
            "effective_offset_m": container_clearance,
            "support_geometry_available": True,
            "support_primitive_count": len(geometry),
            "support_surface_primitive_count": support_primitive_count,
            "support_barrier_count": len(barriers),
            "support_z_m": support_z_m,
            "support_collision_maximum_z_m": maximum_z,
            "support_collision_height_above_surface_m": max(
                0.0, maximum_z - support_z_m
            ),
            "clearance_m": clearance_m,
        }
    )
    return container_clearance, evidence


def _obb_penetrates(left: Obb, right: Obb, *, tolerance_m: float) -> bool:
    """Exact box/box SAT; touching within tolerance is not penetration."""

    left_center, left_rotation, left_half = left
    right_center, right_rotation, right_half = right
    left_axes, right_axes = _columns(left_rotation), _columns(right_rotation)
    delta = tuple(right_center[index] - left_center[index] for index in range(3))
    axes = [*left_axes, *right_axes]
    axes.extend(_cross(a, b) for a in left_axes for b in right_axes)
    for axis in axes:
        norm = math.sqrt(_dot(axis, axis))
        if norm <= 1e-10:
            continue
        unit = tuple(value / norm for value in axis)
        center_distance = abs(_dot(delta, unit))
        left_radius = sum(
            left_half[index] * abs(_dot(left_axes[index], unit)) for index in range(3)
        )
        right_radius = sum(
            right_half[index] * abs(_dot(right_axes[index], unit)) for index in range(3)
        )
        if left_radius + right_radius - center_distance <= tolerance_m:
            return False
    return True


def _scene_mapping(scene: object, name: str) -> Mapping[str, Any]:
    if not isinstance(scene, Mapping):
        return {}
    value = scene.get(name)
    return value if isinstance(value, Mapping) else {}


def _object_goal(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("world_object_goal_pose", "object_goal_pose", "object_goal_world"):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _goal_base(goal_id: str) -> JsonDict:
    return {
        "schema_version": GOAL_LEGALITY_SCHEMA,
        "goal_id": goal_id,
        "execution_started": False,
        "checks": {},
    }


def evaluate_placement_goal_legality(
    descriptor: Mapping[str, Any],
    *,
    scene: object,
) -> JsonDict:
    """Evaluate an AnyPlace object goal exactly once, independently of grasps."""

    started = time.monotonic()
    candidate = descriptor.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    goal_id = str(
        candidate.get("source_object_goal_id")
        or candidate.get("id")
        or descriptor.get("candidate_id")
        or ""
    )
    result = _goal_base(goal_id)
    goal_value = _object_goal(candidate)
    goal = rigid_pose(goal_value)
    result["checks"]["se3"] = {
        "available": goal_value is not None,
        "finite": goal is not None,
        "proper_rotation": goal is not None,
    }
    if goal_value is None:
        result.update(
            verdict="PASS",
            reason="goal_legality_not_applicable",
            geometry_available=False,
            elapsed_s=time.monotonic() - started,
        )
        return result
    if goal is None:
        result.update(
            verdict="FAIL",
            reason="goal_invalid_se3",
            geometry_available=True,
            elapsed_s=time.monotonic() - started,
        )
        return result

    target_id = str(scene.get("target_id") or "") if isinstance(scene, Mapping) else ""
    world_specs = _scene_mapping(scene, "world_specs")
    attached_specs = _scene_mapping(scene, "attached_specs")
    scene_contract = _scene_mapping(scene, "acceptance_scene")
    if scene_contract:
        result["acceptance_scene"] = dict(scene_contract)
    target_is_attached = target_id in attached_specs
    target_spec = (
        attached_specs.get(target_id) if target_is_attached else world_specs.get(target_id)
    )
    # Pair compilation replaces the public object goal with the host-qualified
    # release goal used to derive the EEF terminal.  When the candidate also
    # carries an immutable model motion, replay legality against the original
    # AnyPlace settled goal instead of treating that release goal as fresh
    # model output.  This keeps repeated frozen-frontier waves idempotent.
    model_motion_value = candidate.get("model_object_motion_world_transform")
    model_goal_value = candidate.get("model_pointcloud_object_goal_pose")
    if (
        not target_is_attached
        and isinstance(model_motion_value, Mapping)
        and isinstance(model_goal_value, Mapping)
    ):
        model_goal = rigid_pose(model_goal_value)
        if model_goal is None:
            result.update(
                verdict="FAIL",
                reason="model_goal_invalid_se3",
                geometry_available=True,
                elapsed_s=time.monotonic() - started,
            )
            return result
        goal = model_goal
        result["checks"]["se3"]["source"] = "immutable_model_pointcloud_goal"
    if not isinstance(target_spec, Mapping):
        result.update(
            verdict="PASS",
            reason="goal_geometry_not_available",
            geometry_available=False,
            elapsed_s=time.monotonic() - started,
        )
        return result
    size = _finite_vector(target_spec.get("size_xyz"), 3)
    if size is None or any(value <= 0.0 for value in size):
        result.update(
            verdict="UNKNOWN",
            reason="scene_target_geometry_invalid",
            geometry_available=False,
            infrastructure_error=True,
            elapsed_s=time.monotonic() - started,
        )
        return result
    region = _scene_mapping(scene, "placement_region")
    support_object_id = str(region.get("support_object_id") or "")
    support_z_value = region.get("support_z_m")
    support_z = (
        float(support_z_value)
        if isinstance(support_z_value, (int, float))
        and not isinstance(support_z_value, bool)
        and math.isfinite(float(support_z_value))
        else None
    )
    support_tolerance = float(region.get("support_height_tolerance_m", 0.01))
    support_penetration_tolerance = float(
        region.get(
            "support_penetration_tolerance_m",
            region.get("penetration_tolerance_m", 0.001),
        )
    )
    # A physical release need not command exact zero-distance contact.  Keep
    # the complete attached body one calibrated support-uncertainty band above
    # the collision support, then let Gazebo gravity settle it after detach.
    # This preserves the model pose and avoids disabling target/support
    # collision for the whole MoveIt trajectory.
    release_clearance = min(
        max(0.0, support_penetration_tolerance),
        max(0.0, support_tolerance),
    )
    release_z_offset_value = region.get("release_z_offset_m", 0.0)
    if (
        isinstance(release_z_offset_value, bool)
        or not isinstance(release_z_offset_value, (int, float))
        or not math.isfinite(float(release_z_offset_value))
        or float(release_z_offset_value) < 0.0
    ):
        result.update(
            verdict="UNKNOWN",
            reason="placement_release_z_offset_invalid",
            geometry_available=False,
            infrastructure_error=True,
            elapsed_s=time.monotonic() - started,
        )
        return result
    configured_release_z_offset = float(release_z_offset_value)
    release_z_offset, release_offset_evidence = _effective_release_z_offset(
        configured_drop_height_m=configured_release_z_offset,
        support_object_id=support_object_id,
        support_z_m=support_z,
        support_spec=world_specs.get(support_object_id),
        clearance_m=release_clearance,
    )
    static_penetration_tolerance = float(
        region.get(
            "static_penetration_tolerance_m",
            region.get("penetration_tolerance_m", 0.001),
        )
    )

    # Frozen AnyPlace goals are expressed at the measured point-cloud
    # centroid, which is not generally the PlanningScene collision-box origin.
    # Bind the model's rigid world motion to the exact current scene object:
    # T_world_collision_goal = Delta_world_object * T_world_collision_current.
    # A top-view RGB-D centroid represents the visible surface, so applying
    # that motion to the complete collision body can put its lower extent a
    # few centimetres below the support plane.  When the discrepancy is no
    # larger than the body's own bounding radius, reconcile only world Z to
    # exact support contact.  The model's complete rotation and in-plane
    # target remain unchanged, and the same deterministic correction is later
    # applied to the release EEF terminal before IK.
    #
    # Post-attachment goals already carry the measured physical binding and
    # therefore legitimately fall back to their direct object_goal_pose.
    collision_goal = goal
    motion_value = candidate.get("model_object_motion_world_transform")
    if not isinstance(motion_value, Mapping):
        motion_value = candidate.get("object_motion_world_transform")
    if isinstance(motion_value, Mapping) and not target_is_attached:
        motion = _transform_matrix(motion_value.get("transform_matrix"))
        current_collision = rigid_pose(
            {
                "xyz": target_spec.get("pose_xyz"),
                "quat_xyzw": target_spec.get("pose_quat_xyzw"),
                **(
                    {"rotation_matrix": target_spec["rotation_matrix"]}
                    if target_spec.get("rotation_matrix") is not None
                    else {}
                ),
            }
        )
        if motion is None or current_collision is None:
            result.update(
                verdict="FAIL",
                reason="goal_object_frame_binding_invalid",
                geometry_available=True,
                elapsed_s=time.monotonic() - started,
            )
            return result
        raw_collision_goal = _compose(motion, current_collision)
        collision_goal = raw_collision_goal
        qualified_motion = motion
        raw_geometry = _projected_body_geometry(target_spec, raw_collision_goal)
        if not raw_geometry:
            result.update(
                verdict="UNKNOWN",
                reason="scene_target_geometry_invalid",
                geometry_available=False,
                infrastructure_error=True,
                elapsed_s=time.monotonic() - started,
            )
            return result
        raw_bounds = compound_axis_aligned_bounds(raw_geometry)
        raw_minimum_z = raw_bounds.minimum_xyz[2]
        support_reconciliation: JsonDict = {
            "available": support_z is not None,
            "applied": False,
            "raw_bottom_z_m": raw_minimum_z,
            "basis": "partial_rgbd_centroid_to_complete_collision_body",
        }
        qualified_support_z = (
            support_z + release_clearance if support_z is not None else None
        )
        support_reconciliation["release_clearance_m"] = release_clearance
        if qualified_support_z is not None and raw_minimum_z < qualified_support_z:
            correction_z = qualified_support_z - raw_minimum_z
            geometric_limit = orientation_invariant_radius_m(
                raw_geometry,
                object_xyz=raw_collision_goal[1],
            ) + max(0.0, support_tolerance)
            support_reconciliation.update(
                {
                    "required_translation_z_m": correction_z,
                    "geometric_limit_m": geometric_limit,
                }
            )
            if correction_z <= geometric_limit + 1e-12:
                correction: Transform = (
                    (
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                    ),
                    (0.0, 0.0, correction_z),
                )
                collision_goal = _compose(correction, raw_collision_goal)
                qualified_motion = _compose(correction, motion)
                support_reconciliation.update(
                    {
                        "applied": True,
                        "translation_xyz": [0.0, 0.0, correction_z],
                        "qualified_bottom_z_m": qualified_support_z,
                    }
                )
            else:
                support_reconciliation["reason"] = (
                    "required_correction_exceeds_collision_geometry_bound"
                )
        support_correction = list(
            support_reconciliation.get("translation_xyz") or [0.0, 0.0, 0.0]
        )
        release_translation = [0.0, 0.0, release_z_offset]
        release_correction = [
            support_correction[index] + release_translation[index]
            for index in range(3)
        ]
        release_shift: Transform = (
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            (0.0, 0.0, release_z_offset),
        )
        release_collision_goal = _compose(release_shift, collision_goal)
        release_motion = _compose(release_shift, qualified_motion)
        current_pointcloud = _compose(_inverse(motion), goal)
        release_pointcloud_goal = _compose(release_motion, current_pointcloud)
        release_orientation_policy = "model_settled_orientation"
        binding_method = "world_motion_times_current_planning_scene_object"
        if support_reconciliation.get("applied") is True:
            binding_method = (
                "support_contact_reconciled_world_motion_times_current_planning_scene_object"
            )
        container_drop: JsonDict = {
            "enabled": False,
            "reason": "selected_support_has_no_collision_barriers",
        }
        if (
            support_z is not None
            and int(release_offset_evidence.get("support_barrier_count") or 0) > 0
        ):
            # A container task asks for a stable final state inside a region,
            # not for the wrist to realize the model's post-gravity settled
            # orientation while the object is still rigidly attached. Keep the
            # model-selected in-plane destination, preserve the measured
            # current object orientation during transport, and let the
            # simulator settle the short final drop after detach. Pair
            # legality and MoveIt still prove the complete gripper/payload
            # state against every authoritative wall.
            provisional_release: Transform = (
                current_collision[0],
                (collision_goal[1][0], collision_goal[1][1], 0.0),
            )
            provisional_geometry = _projected_body_geometry(
                target_spec,
                provisional_release,
            )
            if provisional_geometry:
                provisional_bounds = compound_axis_aligned_bounds(
                    provisional_geometry
                )
                desired_bottom_z = (
                    support_z + release_clearance + release_z_offset
                )
                desired_release_collision: Transform = (
                    current_collision[0],
                    (
                        collision_goal[1][0],
                        collision_goal[1][1],
                        desired_bottom_z - provisional_bounds.minimum_xyz[2],
                    ),
                )
                release_delta = tuple(
                    desired_release_collision[1][index]
                    - current_collision[1][index]
                    for index in range(3)
                )
                release_motion = (
                    (
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                    ),
                    release_delta,
                )
                release_collision_goal = desired_release_collision
                release_pointcloud_goal = _compose(
                    release_motion,
                    current_pointcloud,
                )
                release_correction = [0.0, 0.0, 0.0]
                release_orientation_policy = (
                    "preserve_current_orientation_for_container_drop"
                )
                binding_method = (
                    "container_drop_translation_from_current_planning_scene_object"
                )
                container_drop = {
                    "enabled": True,
                    "reason": "authoritative_support_has_collision_barriers",
                    "path_owner": "moveit",
                    "settling_owner": "native_gravity",
                    "model_destination_xy_preserved": True,
                    "current_object_orientation_preserved": True,
                    "release_bottom_z_m": desired_bottom_z,
                    "support_barrier_count": int(
                        release_offset_evidence.get("support_barrier_count") or 0
                    ),
                }
        result["checks"]["object_frame_binding"] = {
            "available": True,
            "method": binding_method,
            "pointcloud_goal_translation_xyz": list(goal[1]),
            "collision_goal_translation_xyz": list(collision_goal[1]),
            "raw_collision_goal_pose": {
                "convention": "T_world_collision_object_goal_raw",
                "frame": "world",
                "translation_xyz": list(raw_collision_goal[1]),
                "rotation_matrix": [list(row) for row in raw_collision_goal[0]],
            },
            "collision_goal_pose": {
                "convention": "T_world_collision_object_goal",
                "frame": "world",
                "translation_xyz": list(collision_goal[1]),
                "rotation_matrix": [list(row) for row in collision_goal[0]],
            },
            "release_collision_goal_pose": {
                "convention": "T_world_collision_object_release",
                "frame": "world",
                "translation_xyz": list(release_collision_goal[1]),
                "rotation_matrix": [list(row) for row in release_collision_goal[0]],
            },
            "release_pointcloud_object_goal_pose": _pose_payload(
                release_pointcloud_goal,
                convention="T_world_pointcloud_object_release",
            ),
            "release_orientation_policy": release_orientation_policy,
            "container_drop": container_drop,
            "model_world_motion_transform": _transform_payload(
                motion,
                convention="T_world_motion_applied_left",
            ),
            "qualified_settled_world_motion_transform": _transform_payload(
                qualified_motion,
                convention="T_world_support_reconciled_motion_applied_left",
            ),
            "qualified_world_motion_transform": _transform_payload(
                release_motion,
                convention="T_world_release_motion_applied_left",
            ),
            "support_contact_translation_correction_xyz": support_correction,
            "placement_release_translation_xyz": release_translation,
            "release_target_translation_correction_xyz": release_correction,
            "release_z_offset_m": release_z_offset,
            "configured_release_z_offset_m": configured_release_z_offset,
            "release_offset_selection": release_offset_evidence,
            "support_contact_reconciliation": support_reconciliation,
        }
    else:
        prebound_release_goal = rigid_pose(
            candidate.get("qualified_release_object_goal_pose")
        )
        prebound_container_drop = (
            target_is_attached
            and candidate.get("container_drop_release_prebound") is True
            and prebound_release_goal is not None
        )
        if prebound_container_drop:
            release_collision_goal = prebound_release_goal
            release_correction = [0.0, 0.0, 0.0]
            release_orientation_policy = (
                "preserve_current_orientation_for_container_drop"
            )
            binding_method = "prebound_container_drop_physical_goal"
        else:
            release_shift = (
                (
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                (0.0, 0.0, release_z_offset),
            )
            release_collision_goal = _compose(release_shift, collision_goal)
            release_correction = [0.0, 0.0, release_z_offset]
            release_orientation_policy = "model_settled_orientation"
            binding_method = "direct_physical_object_goal"
        result["checks"]["object_frame_binding"] = {
            "available": True,
            "method": binding_method,
            "target_is_attached": target_is_attached,
            "collision_goal_translation_xyz": list(collision_goal[1]),
            "collision_goal_pose": {
                "convention": "T_world_collision_object_goal",
                "frame": "world",
                "translation_xyz": list(collision_goal[1]),
                "rotation_matrix": [list(row) for row in collision_goal[0]],
            },
            "release_collision_goal_pose": {
                "convention": "T_world_collision_object_release",
                "frame": "world",
                "translation_xyz": list(release_collision_goal[1]),
                "rotation_matrix": [list(row) for row in release_collision_goal[0]],
            },
            "release_orientation_policy": release_orientation_policy,
            "support_contact_translation_correction_xyz": [0.0, 0.0, 0.0],
            "placement_release_translation_xyz": [0.0, 0.0, release_z_offset],
            "release_target_translation_correction_xyz": release_correction,
            "release_z_offset_m": release_z_offset,
            "configured_release_z_offset_m": configured_release_z_offset,
            "release_offset_selection": release_offset_evidence,
        }
    goal_geometry = _projected_body_geometry(target_spec, collision_goal)
    if not goal_geometry:
        result.update(
            verdict="UNKNOWN",
            reason="scene_target_geometry_invalid",
            geometry_available=False,
            infrastructure_error=True,
            elapsed_s=time.monotonic() - started,
        )
        return result
    goal_bounds = compound_axis_aligned_bounds(goal_geometry)
    volume_centroid = collision_geometry_volume_centroid(goal_geometry)
    minimum_z = goal_bounds.minimum_xyz[2]
    maximum_z = goal_bounds.maximum_xyz[2]
    result["checks"]["object_bbox"] = {
        "available": True,
        "size_xyz": size,
        "minimum_z_m": minimum_z,
        "maximum_z_m": maximum_z,
        "geometry_source": (
            "compound_collision_primitives"
            if target_spec.get("primitives")
            else "centered_bounding_box"
        ),
        "primitive_count": len(goal_geometry),
    }

    if support_z is not None:
        support_alignment_cosine = support_face_alignment_cosine(goal_geometry)
        result["checks"]["support"] = {
            "available": True,
            "support_z_m": support_z,
            "bottom_z_m": minimum_z,
            "height_error_m": minimum_z - support_z,
            "height_tolerance_m": support_tolerance,
            "penetration_tolerance_m": support_penetration_tolerance,
            "release_z_offset_m": release_z_offset,
            "configured_release_z_offset_m": configured_release_z_offset,
            "release_offset_selection": release_offset_evidence,
            "tolerance_basis": "sensor_and_model_support_contact_uncertainty",
            "geometry_volume_centroid_xyz": list(volume_centroid),
            "geometry_volume_centroid_height_m": volume_centroid[2] - support_z,
            "support_energy_resolution_m": support_tolerance,
            "support_face_alignment_cosine": support_alignment_cosine,
            "support_face_alignment_error_rad": math.acos(support_alignment_cosine),
            "stability_role": "ordering_only",
        }
        if minimum_z < support_z - support_penetration_tolerance:
            result.update(
                verdict="FAIL",
                reason="goal_support_surface_penetration",
                geometry_available=True,
                elapsed_s=time.monotonic() - started,
            )
            return result
        if minimum_z > support_z + support_tolerance:
            result.update(
                verdict="FAIL",
                reason="goal_unsupported_height",
                geometry_available=True,
                elapsed_s=time.monotonic() - started,
            )
            return result
    else:
        result["checks"]["support"] = {"available": False}

    center_xy = _finite_vector(region.get("center_xy"), 2)
    size_xy = _finite_vector(region.get("size_xy_m"), 2)
    if center_xy is not None and size_xy is not None and all(value > 0.0 for value in size_xy):
        half_x, half_y = size_xy[0] / 2.0, size_xy[1] / 2.0
        acceptance_semantics = str(region.get("acceptance_semantics") or "complete_footprint")
        if acceptance_semantics not in {
            "complete_footprint",
            "stable_geometry_centroid_inside",
        }:
            result.update(
                verdict="UNKNOWN",
                reason="placement_region_semantics_invalid",
                geometry_available=False,
                infrastructure_error=True,
                elapsed_s=time.monotonic() - started,
            )
            return result
        # The exact compound-primitive footprint is the legality proof.  Its
        # orientation-invariant link-origin radius remains an ordering signal
        # only; a negative conservative clearance never deletes an otherwise
        # legal AnyPlace goal.
        conservative_radius = orientation_invariant_radius_m(
            goal_geometry,
            object_xyz=collision_goal[1],
        )
        conservative_margin = min(
            half_x - abs(collision_goal[1][0] - center_xy[0]) - conservative_radius,
            half_y - abs(collision_goal[1][1] - center_xy[1]) - conservative_radius,
        )
        footprint_margin = min(
            goal_bounds.minimum_xyz[0] - (center_xy[0] - half_x),
            center_xy[0] + half_x - goal_bounds.maximum_xyz[0],
            goal_bounds.minimum_xyz[1] - (center_xy[1] - half_y),
            center_xy[1] + half_y - goal_bounds.maximum_xyz[1],
        )
        centroid_margin = min(
            half_x - abs(volume_centroid[0] - center_xy[0]),
            half_y - abs(volume_centroid[1] - center_xy[1]),
        )
        result["checks"]["placement_region"] = {
            "available": True,
            "acceptance_semantics": acceptance_semantics,
            "complete_footprint_inside": footprint_margin >= -1e-9,
            "geometry_centroid_inside": centroid_margin >= -1e-9,
            "minimum_margin_m": footprint_margin,
            "geometry_centroid_minimum_margin_m": centroid_margin,
            "geometry_volume_centroid_xyz": list(volume_centroid),
            "footprint_rule": (
                "stable_geometry_centroid_inside"
                if acceptance_semantics == "stable_geometry_centroid_inside"
                else "compound_collision_projection_fully_inside"
            ),
            "complete_footprint_margin_role": (
                "ordering_only"
                if acceptance_semantics == "stable_geometry_centroid_inside"
                else "legality_gate"
            ),
            "projected_minimum_xy_m": list(goal_bounds.minimum_xyz[:2]),
            "projected_maximum_xy_m": list(goal_bounds.maximum_xyz[:2]),
            "conservative_footprint_radius_m": conservative_radius,
            "conservative_minimum_margin_m": conservative_margin,
            "conservative_margin_role": "ordering_only",
            "center_xy": center_xy,
            "size_xy_m": size_xy,
        }
        support_checks = result["checks"].get("support")
        if isinstance(support_checks, dict):
            alignment_error = support_checks.get("support_face_alignment_error_rad")
            if (
                isinstance(alignment_error, (int, float))
                and not isinstance(alignment_error, bool)
                and math.isfinite(float(alignment_error))
                and float(alignment_error) >= 0.0
            ):
                # If the model pose settles onto its nearest analytic support
                # face, every body point moves by at most the chord of the
                # orientation-invariant radius through that angular error.
                # Subtracting that exact swept-displacement bound from the
                # yaw-invariant region clearance predicts whether settling can
                # remain in-zone. It is deliberately ordering-only because
                # friction and non-uniform mass are not proven here.
                settling_shift = (
                    2.0 * conservative_radius * math.sin(min(math.pi, float(alignment_error)) / 2.0)
                )
                settling_clearance = conservative_margin - settling_shift
                settling = {
                    "settling_sweep_translation_bound_m": settling_shift,
                    "settling_sweep_clearance_m": settling_clearance,
                    "settling_sweep_rule": (
                        "orientation_invariant_radius_chord_to_nearest_support_face"
                    ),
                    "settling_sweep_role": "ordering_only",
                }
                support_checks.update(settling)
                result["checks"]["placement_region"].update(settling)
        if acceptance_semantics == "stable_geometry_centroid_inside" and centroid_margin < -1e-9:
            result.update(
                verdict="FAIL",
                reason="goal_geometry_centroid_outside_placement_region",
                geometry_available=True,
                elapsed_s=time.monotonic() - started,
            )
            return result
        if acceptance_semantics == "complete_footprint" and footprint_margin < -1e-9:
            result.update(
                verdict="FAIL",
                reason="goal_footprint_outside_placement_region",
                geometry_available=True,
                elapsed_s=time.monotonic() - started,
            )
            return result
    else:
        result["checks"]["placement_region"] = {"available": False}

    collisions: list[str] = []
    uncheckable: list[str] = []
    evaluated_obstacles: list[str] = []
    support_contact_primitive_count = 0
    support_barrier_primitive_count = 0
    for object_id, spec in sorted(world_specs.items(), key=lambda item: str(item[0])):
        object_id = str(object_id)
        if object_id == target_id:
            continue
        all_obstacle_boxes = _spec_obbs(spec)
        obstacle_boxes = all_obstacle_boxes
        if object_id == support_object_id:
            obstacle_boxes, support_contact_primitive_count = _non_support_obbs(
                spec,
                support_z_m=support_z,
            )
            support_barrier_primitive_count = len(obstacle_boxes)
        if not obstacle_boxes:
            if not all_obstacle_boxes:
                uncheckable.append(object_id)
            continue
        evaluated_obstacles.append(object_id)
        if any(
            _obb_penetrates(
                _projected_obb(body),
                obstacle,
                tolerance_m=static_penetration_tolerance,
            )
            for body in goal_geometry
            for obstacle in obstacle_boxes
        ):
            collisions.append(object_id)
    result["checks"]["static_scene_collision"] = {
        "available": bool(world_specs),
        "evaluated_obstacle_ids": evaluated_obstacles,
        "collision_ids": collisions,
        "uncheckable_ids": uncheckable,
        "support_contact_primitive_count": support_contact_primitive_count,
        "support_barrier_primitive_count": support_barrier_primitive_count,
    }
    if collisions:
        result.update(
            verdict="FAIL",
            reason="goal_static_obstacle_penetration",
            geometry_available=True,
            elapsed_s=time.monotonic() - started,
        )
        return result

    envelope = _scene_mapping(scene, "workspace_envelope")
    base_xyz = _finite_vector(envelope.get("base_xyz"), 3)
    outer = envelope.get("outer_radius_m")
    attachment_allowance = float(envelope.get("maximum_attachment_offset_m", 0.25))
    if (
        base_xyz is not None
        and isinstance(outer, (int, float))
        and not isinstance(outer, bool)
        and math.isfinite(float(outer))
        and float(outer) > 0.0
    ):
        object_radius = math.sqrt(sum((value / 2.0) ** 2 for value in size))
        center_distance = math.sqrt(
            sum((collision_goal[1][index] - base_xyz[index]) ** 2 for index in range(3))
        )
        reachable = center_distance - object_radius <= float(outer) + attachment_allowance
        result["checks"]["analytic_workspace"] = {
            "available": True,
            "center_distance_m": center_distance,
            "object_radius_m": object_radius,
            "outer_radius_m": float(outer),
            "maximum_attachment_offset_m": attachment_allowance,
            "pass": reachable,
        }
        if not reachable:
            result.update(
                verdict="FAIL",
                reason="goal_outside_analytic_workspace",
                geometry_available=True,
                elapsed_s=time.monotonic() - started,
            )
            return result
    else:
        result["checks"]["analytic_workspace"] = {"available": False}

    result.update(
        verdict="PASS",
        reason="goal_legality_qualified",
        geometry_available=True,
        elapsed_s=time.monotonic() - started,
    )
    return result


def bind_qualified_placement_goal(
    descriptor: JsonDict,
    goal_legality: Mapping[str, Any],
) -> None:
    """Apply one scene-derived physical goal binding before pair IK.

    Candidate identity and the model pose stay immutable.  This updates only
    the host-private compiled release terminal and world-motion evidence used
    by the same qualification RPC, so the subsequent pair, IK, state-validity,
    and L5 checks all prove the exact collision-body goal returned in the
    legality evidence.
    """

    if goal_legality.get("verdict") != "PASS":
        return
    checks = goal_legality.get("checks")
    binding = checks.get("object_frame_binding") if isinstance(checks, Mapping) else None
    if not isinstance(binding, Mapping):
        return
    qualified_motion = binding.get("qualified_world_motion_transform")
    collision_goal = binding.get("collision_goal_pose")
    release_goal = binding.get("release_collision_goal_pose")
    release_pointcloud_goal = binding.get("release_pointcloud_object_goal_pose")
    release_orientation_policy = str(
        binding.get("release_orientation_policy") or "model_settled_orientation"
    )
    candidate_value = descriptor.get("candidate")
    if not isinstance(candidate_value, Mapping):
        return
    candidate = dict(candidate_value)
    candidate["goal_legality_prebound"] = True
    if not isinstance(candidate.get("model_pointcloud_object_goal_pose"), Mapping):
        model_goal = _object_goal(candidate)
        if isinstance(model_goal, Mapping):
            candidate["model_pointcloud_object_goal_pose"] = dict(model_goal)
    if isinstance(collision_goal, Mapping):
        candidate["qualified_world_collision_object_goal_pose"] = dict(collision_goal)
    if isinstance(release_goal, Mapping):
        candidate["qualified_release_object_goal_pose"] = dict(release_goal)
    if isinstance(release_pointcloud_goal, Mapping):
        candidate["qualified_release_pointcloud_object_goal_pose"] = dict(
            release_pointcloud_goal
        )
    candidate["release_orientation_policy"] = release_orientation_policy
    if release_orientation_policy == "preserve_current_orientation_for_container_drop":
        candidate["container_drop_release_prebound"] = True
    if isinstance(qualified_motion, Mapping):
        if _transform_matrix(qualified_motion.get("transform_matrix")) is None:
            return
        if not isinstance(candidate.get("model_object_motion_world_transform"), Mapping):
            model_motion = candidate.get("object_motion_world_transform")
            if isinstance(model_motion, Mapping):
                candidate["model_object_motion_world_transform"] = dict(model_motion)
        candidate["object_motion_world_transform"] = dict(qualified_motion)
        candidate["qualified_release_world_motion_transform"] = dict(
            qualified_motion
        )
        # A virtual pre-attachment proof must attach the actual PlanningScene
        # collision body at the contact EEF, not the visible point-cloud centroid.
        candidate["physical_scene_attachment_required"] = True
        candidate["physical_scene_attachment_source"] = (
            "current_planning_scene_collision_object"
        )

    correction = _finite_vector(binding.get("release_target_translation_correction_xyz"), 3)
    if correction is not None and any(abs(value) > 1e-12 for value in correction):
        stages_value = candidate.get("qualification_stages")
        if isinstance(stages_value, list):
            stages: list[object] = []
            for raw_stage in stages_value:
                if not isinstance(raw_stage, Mapping):
                    stages.append(raw_stage)
                    continue
                stage = dict(raw_stage)
                if str(stage.get("name") or "") == "release":
                    for key in ("xyz", "translation_xyz", "position"):
                        xyz = _finite_vector(stage.get(key), 3)
                        if xyz is not None:
                            stage[key] = [xyz[index] + correction[index] for index in range(3)]
                    support_correction = _finite_vector(
                        binding.get("support_contact_translation_correction_xyz"), 3
                    )
                    release_translation = _finite_vector(
                        binding.get("placement_release_translation_xyz"), 3
                    )
                    if support_correction is not None:
                        stage["support_contact_translation_correction_xyz"] = support_correction
                    if release_translation is not None:
                        stage["placement_release_translation_xyz"] = release_translation
                        stage["placement_release_z_offset_m"] = release_translation[2]
                    stage["release_target_translation_correction_xyz"] = list(correction)
                    stage["terminal_pose_source"] = (
                        "anyplace_se3_with_physical_support_and_release_offset"
                    )
                stages.append(stage)
            candidate["qualification_stages"] = stages
    descriptor["candidate"] = candidate


def _expected_pair_eef_goal(candidate: Mapping[str, Any]) -> Transform | None:
    compile_parameters = candidate.get("compile_parameters")
    compile_parameters = compile_parameters if isinstance(compile_parameters, Mapping) else {}
    attachment = rigid_pose(compile_parameters.get("attachment_transform"))
    # Once a failed close rigidly moves the detached collision body, the
    # rebased grasp and its predicted attachment both use that physical object
    # frame.  The desired bin goal then stays fixed and is authoritative.
    # Initial model pairs are different: their attachment uses the partial
    # point-cloud frame, so mixing in the physical collision-body goal would
    # introduce the centroid/body offset a second time.
    qualified_object_goal = rigid_pose(candidate.get("qualified_world_collision_object_goal_pose"))
    qualified_release_goal = rigid_pose(candidate.get("qualified_release_object_goal_pose"))
    physical_rebase = candidate.get("frozen_object_motion_rebase")
    if (
        isinstance(physical_rebase, Mapping)
        and (qualified_release_goal is not None or qualified_object_goal is not None)
        and attachment is not None
    ):
        release_goal = qualified_release_goal or qualified_object_goal
        assert release_goal is not None
        return _compose(
            release_goal,
            _inverse(attachment),
        )
    contact = rigid_pose(candidate.get("frozen_contact_pose"))
    motion_value = candidate.get("object_motion_world_transform")
    motion = (
        _transform_matrix(motion_value.get("transform_matrix"))
        if isinstance(motion_value, Mapping)
        else None
    )
    if contact is not None and motion is not None:
        return _compose(motion, contact)
    # In the attached-object path the model supplies T_world_object_goal.
    # Derive the sole release EEF terminal from the measured native attachment;
    # a direct-EFF fallback would silently change the model/attachment contract.
    object_goal = qualified_release_goal or rigid_pose(_object_goal(candidate))
    if object_goal is not None and attachment is not None:
        return _compose(object_goal, _inverse(attachment))
    return None


def _pair_chain_check(
    candidate: Mapping[str, Any], stage_poses: Mapping[str, Transform]
) -> JsonDict:
    expected = _expected_pair_eef_goal(candidate)
    release = stage_poses.get("release")
    if expected is None or release is None:
        return {
            "available": False,
            "pass": True,
            "reason": "exact_release_derivation_unavailable",
        }
    expected_rotation, expected_xyz = expected
    release_rotation, release_xyz = release
    aligned = (
        abs(release_xyz[0] - expected_xyz[0]) <= _POSE_TOLERANCE_M
        and abs(release_xyz[1] - expected_xyz[1]) <= _POSE_TOLERANCE_M
        and abs(release_xyz[2] - expected_xyz[2]) <= _POSE_TOLERANCE_M
        and _rotation_distance(release_rotation, expected_rotation) <= _ORIENTATION_TOLERANCE
    )
    return {
        "available": True,
        "pass": aligned,
        "translation_error_m": math.dist(release_xyz, expected_xyz),
        "orientation_error_rad": _rotation_distance(release_rotation, expected_rotation),
        "terminal_pose_source": "model_goal_and_attachment_transform",
        "path_owner": "moveit",
    }


def _attachment_transform(candidate: Mapping[str, Any], scene: object) -> Transform | None:
    target_id = str(scene.get("target_id") or "") if isinstance(scene, Mapping) else ""
    attached = _scene_mapping(scene, "attached_specs").get(target_id)
    if isinstance(attached, Mapping):
        parsed = rigid_pose(
            {
                "xyz": attached.get("pose_xyz"),
                "quat_xyzw": attached.get("pose_quat_xyzw"),
            }
        )
        if parsed is not None:
            return parsed
    if candidate.get("initial_scene_transition") == "virtual_attach":
        eef = rigid_pose(candidate.get("initial_scene_transition_pose"))
        world_spec = _scene_mapping(scene, "world_specs").get(target_id)
        object_pose = (
            rigid_pose(
                {
                    "xyz": world_spec.get("pose_xyz"),
                    "quat_xyzw": world_spec.get("pose_quat_xyzw"),
                }
            )
            if isinstance(world_spec, Mapping)
            else None
        )
        if eef is not None and object_pose is not None:
            return _compose(_inverse(eef), object_pose)
    predicted = rigid_pose(candidate.get("predicted_attachment_transform"))
    if predicted is not None:
        return predicted
    compile_parameters = candidate.get("compile_parameters")
    compile_parameters = compile_parameters if isinstance(compile_parameters, Mapping) else {}
    return rigid_pose(compile_parameters.get("attachment_transform"))


def evaluate_grasp_placement_pair_legality(
    descriptor: Mapping[str, Any],
    *,
    scene: object,
    workspace_filter: Callable[[Mapping[str, Any]], bool] | None,
) -> JsonDict:
    """Validate a compiled grasp/placement chain before spending any IK calls."""

    started = time.monotonic()
    candidate = descriptor.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    candidate_id = str(descriptor.get("candidate_id") or candidate.get("id") or "")
    result: JsonDict = {
        "schema_version": PAIR_LEGALITY_SCHEMA,
        "candidate_id": candidate_id,
        "source_grasp_id": str(candidate.get("source_grasp_id") or ""),
        "source_object_goal_id": str(candidate.get("source_object_goal_id") or ""),
        "execution_started": False,
        "checks": {},
    }
    stages = candidate.get("qualification_stages")
    if not isinstance(stages, list) or not stages:
        result.update(
            verdict="FAIL",
            reason="pair_compiled_stages_missing",
            elapsed_s=time.monotonic() - started,
        )
        return result
    exact_release_contract = _expected_pair_eef_goal(candidate) is not None
    if exact_release_contract and (
        len(stages) != 1
        or not isinstance(stages[0], Mapping)
        or str(stages[0].get("name") or "") != "release"
    ):
        result.update(
            verdict="FAIL",
            reason="pair_artificial_waypoint_forbidden",
            elapsed_s=time.monotonic() - started,
        )
        return result
    stage_poses: dict[str, Transform] = {}
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            result.update(
                verdict="FAIL",
                reason="pair_stage_invalid_se3",
                failed_stage_index=index,
                elapsed_s=time.monotonic() - started,
            )
            return result
        parsed = rigid_pose(stage)
        if parsed is None:
            result.update(
                verdict="FAIL",
                reason="pair_stage_invalid_se3",
                failed_stage_index=index,
                elapsed_s=time.monotonic() - started,
            )
            return result
        stage_poses[str(stage.get("name") or f"stage_{index}")] = parsed
    result["checks"]["stage_se3"] = {"pass": True, "stage_count": len(stages)}

    chain = _pair_chain_check(candidate, stage_poses)
    result["checks"]["eef_chain"] = chain
    if chain.get("pass") is not True:
        result.update(
            verdict="FAIL",
            reason="pair_eef_chain_inconsistent",
            elapsed_s=time.monotonic() - started,
        )
        return result

    reach_checks: list[JsonDict] = []
    if workspace_filter is not None:
        for index, stage in enumerate(stages):
            try:
                passed = bool(workspace_filter(stage))
            except Exception as exc:  # noqa: BLE001 - callback boundary.
                result.update(
                    verdict="UNKNOWN",
                    reason="pair_analytic_workspace_error",
                    infrastructure_error=True,
                    error_type=type(exc).__name__,
                    elapsed_s=time.monotonic() - started,
                )
                return result
            reach_checks.append(
                {
                    "stage_index": index,
                    "name": str(stage.get("name") or f"stage_{index}"),
                    "pass": passed,
                }
            )
            if not passed:
                result["checks"]["analytic_workspace"] = reach_checks
                result.update(
                    verdict="FAIL",
                    reason="pair_stage_outside_analytic_workspace",
                    failed_stage_index=index,
                    elapsed_s=time.monotonic() - started,
                )
                return result
    result["checks"]["analytic_workspace"] = {
        "available": workspace_filter is not None,
        "stages": reach_checks,
    }

    target_id = str(scene.get("target_id") or "") if isinstance(scene, Mapping) else ""
    world_specs = _scene_mapping(scene, "world_specs")
    attached_specs = _scene_mapping(scene, "attached_specs")
    target_spec = world_specs.get(target_id) or attached_specs.get(target_id)
    size = (
        _finite_vector(target_spec.get("size_xyz"), 3) if isinstance(target_spec, Mapping) else None
    )
    attachment = _attachment_transform(candidate, scene)
    region = _scene_mapping(scene, "placement_region")
    support_object_id = str(region.get("support_object_id") or "")
    support_z = region.get("support_z_m")
    support_penetration_tolerance = float(
        region.get(
            "support_penetration_tolerance_m",
            region.get("penetration_tolerance_m", 0.001),
        )
    )
    static_penetration_tolerance = float(
        region.get(
            "static_penetration_tolerance_m",
            region.get("penetration_tolerance_m", 0.001),
        )
    )
    collision_events: list[JsonDict] = []
    support_contact_primitive_count = 0
    support_barrier_primitive_count = 0
    evaluated_obstacle_ids = sorted(
        str(object_id)
        for object_id, spec in world_specs.items()
        if str(object_id) != target_id and _spec_obbs(spec)
    )
    uncheckable_obstacle_ids = sorted(
        str(object_id)
        for object_id, spec in world_specs.items()
        if str(object_id) != target_id and not _spec_obbs(spec)
    )
    attached_active = (
        attachment is not None and size is not None and isinstance(target_spec, Mapping)
    )
    for index, stage in enumerate(stages):
        name = str(stage.get("name") or f"stage_{index}")
        stage_transform = stage_poses[name]
        if (
            attached_active
            and attachment is not None
            and size is not None
            and isinstance(target_spec, Mapping)
        ):
            object_geometry = _projected_body_geometry(
                target_spec,
                _compose(stage_transform, attachment),
            )
            if not object_geometry:
                result.update(
                    verdict="UNKNOWN",
                    reason="pair_attached_geometry_invalid",
                    infrastructure_error=True,
                    elapsed_s=time.monotonic() - started,
                )
                return result
            minimum_z = compound_axis_aligned_bounds(object_geometry).minimum_xyz[2]
            if (
                isinstance(support_z, (int, float))
                and not isinstance(support_z, bool)
                and math.isfinite(float(support_z))
                and minimum_z < float(support_z) - support_penetration_tolerance
            ):
                collision_events.append(
                    {
                        "stage_index": index,
                        "stage": name,
                        "body": target_id,
                        "obstacle": support_object_id,
                        "kind": "support_penetration",
                    }
                )
            for object_id, spec in sorted(world_specs.items(), key=lambda item: str(item[0])):
                object_id = str(object_id)
                if object_id == target_id:
                    continue
                obstacle_boxes = _spec_obbs(spec)
                if object_id == support_object_id:
                    obstacle_boxes, support_contact_primitive_count = (
                        _non_support_obbs(
                            spec,
                            support_z_m=support_z,
                        )
                    )
                    support_barrier_primitive_count = len(obstacle_boxes)
                if obstacle_boxes and any(
                    _obb_penetrates(
                        _projected_obb(body),
                        obstacle,
                        tolerance_m=static_penetration_tolerance,
                    )
                    for body in object_geometry
                    for obstacle in obstacle_boxes
                ):
                    collision_events.append(
                        {
                            "stage_index": index,
                            "stage": name,
                            "body": target_id,
                            "obstacle": object_id,
                            "kind": "attached_object_static_collision",
                        }
                    )

        for proxy_index, proxy in enumerate(
            scene.get("gripper_collision_boxes") or [] if isinstance(scene, Mapping) else []
        ):
            if not isinstance(proxy, Mapping):
                continue
            proxy_size = _finite_vector(proxy.get("size_xyz"), 3)
            proxy_local = rigid_pose(
                {
                    "xyz": proxy.get("pose_xyz"),
                    "quat_xyzw": proxy.get("pose_quat_xyzw"),
                    **(
                        {"rotation_matrix": proxy["rotation_matrix"]}
                        if proxy.get("rotation_matrix") is not None
                        else {}
                    ),
                }
            )
            if proxy_size is None or proxy_local is None:
                continue
            proxy_box = _obb(_compose(stage_transform, proxy_local), proxy_size)
            for object_id, spec in sorted(world_specs.items(), key=lambda item: str(item[0])):
                object_id = str(object_id)
                if object_id == target_id:
                    continue
                obstacle_boxes = _spec_obbs(spec)
                if obstacle_boxes and any(
                    _obb_penetrates(
                        proxy_box,
                        obstacle,
                        tolerance_m=static_penetration_tolerance,
                    )
                    for obstacle in obstacle_boxes
                ):
                    collision_events.append(
                        {
                            "stage_index": index,
                            "stage": name,
                            "body": str(proxy.get("id") or f"gripper_proxy_{proxy_index}"),
                            "obstacle": object_id,
                            "kind": "gripper_static_collision",
                        }
                    )
        if str(stage.get("scene_transition") or "") == "virtual_detach":
            attached_active = False

    result["checks"]["static_scene_collision"] = {
        "available": bool(world_specs),
        "gripper_geometry_available": bool(scene.get("gripper_collision_boxes"))
        if isinstance(scene, Mapping)
        else False,
        "attached_object_geometry_available": (
            attachment is not None and size is not None and isinstance(target_spec, Mapping)
        ),
        "attached_object_geometry_source": (
            "compound_collision_primitives"
            if isinstance(target_spec, Mapping) and target_spec.get("primitives")
            else "centered_bounding_box"
        ),
        "evaluated_obstacle_ids": evaluated_obstacle_ids,
        "uncheckable_obstacle_ids": uncheckable_obstacle_ids,
        "collisions": collision_events,
        "support_contact_primitive_count": support_contact_primitive_count,
        "support_barrier_primitive_count": support_barrier_primitive_count,
    }
    if collision_events:
        result.update(
            verdict="FAIL",
            reason=str(collision_events[0]["kind"]),
            elapsed_s=time.monotonic() - started,
        )
        return result

    result.update(
        verdict="PASS", reason="pair_legality_qualified", elapsed_s=time.monotonic() - started
    )
    return result
