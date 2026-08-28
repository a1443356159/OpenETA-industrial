"""Deterministic projections of versioned compound collision geometry.

MoveIt and Gazebo both consume the primitive poses stored on a collision body.
Legality and post-release verification must project those same primitives,
rather than silently replacing an offset compound body with a box centred on
its link origin.  This module is ROS-free so both runtime and simulator proof
layers can share one implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


Vector3 = tuple[float, float, float]
Rotation3 = tuple[Vector3, Vector3, Vector3]


def _finite_vector(value: object, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != length:
        return None
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return parsed if all(math.isfinite(item) for item in parsed) else None


def _identity_rotation() -> Rotation3:
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def _rotation_matrix(value: object) -> Rotation3 | None:
    if not isinstance(value, Sequence) or len(value) != 3:
        return None
    rows = [_finite_vector(row, 3) for row in value]
    if any(row is None for row in rows):
        return None
    return tuple(rows)  # type: ignore[return-value]


def _quaternion_rotation(value: object) -> Rotation3 | None:
    quaternion = _finite_vector(value, 4)
    if quaternion is None:
        return None
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm <= 1e-12:
        return None
    x, y, z, w = (item / norm for item in quaternion)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _rpy_rotation(value: object) -> Rotation3 | None:
    rpy = _finite_vector(value, 3)
    if rpy is None:
        return None
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _pose_rotation(spec: Mapping[str, Any]) -> Rotation3 | None:
    if spec.get("rotation_matrix") is not None:
        return _rotation_matrix(spec.get("rotation_matrix"))
    if spec.get("pose_quat_xyzw") is not None:
        return _quaternion_rotation(spec.get("pose_quat_xyzw"))
    if spec.get("pose_rpy") is not None:
        return _rpy_rotation(spec.get("pose_rpy"))
    return _identity_rotation()


def _rotate(rotation: Rotation3, vector: Sequence[float]) -> Vector3:
    return tuple(
        sum(rotation[row][column] * float(vector[column]) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _compose_rotation(left: Rotation3, right: Rotation3) -> Rotation3:
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ProjectedCollisionPrimitive:
    """One collision primitive expressed in world coordinates."""

    shape: str
    center_xyz: Vector3
    rotation_rows: Rotation3
    size_xyz: Vector3 | None = None
    radius_m: float | None = None
    length_m: float | None = None

    def axis_half_extent(self, axis: int) -> float:
        if self.shape == "box" and self.size_xyz is not None:
            return sum(
                abs(self.rotation_rows[axis][column]) * self.size_xyz[column] / 2.0
                for column in range(3)
            )
        if self.shape == "cylinder" and self.radius_m is not None and self.length_m is not None:
            # ROS SolidPrimitive cylinders use local Z as their longitudinal
            # axis.  Project the circular cap and axial segment independently.
            axial = abs(self.rotation_rows[axis][2])
            radial = self.radius_m * math.sqrt(max(0.0, 1.0 - axial * axial))
            return radial + axial * self.length_m / 2.0
        raise ValueError("collision primitive geometry is incomplete")

    def enclosing_obb_half_sizes(self) -> Vector3:
        if self.shape == "box" and self.size_xyz is not None:
            return tuple(value / 2.0 for value in self.size_xyz)  # type: ignore[return-value]
        if self.shape == "cylinder" and self.radius_m is not None and self.length_m is not None:
            return (self.radius_m, self.radius_m, self.length_m / 2.0)
        raise ValueError("collision primitive geometry is incomplete")


@dataclass(frozen=True, slots=True)
class CompoundAxisAlignedBounds:
    minimum_xyz: Vector3
    maximum_xyz: Vector3

    @property
    def extent_xyz(self) -> Vector3:
        return tuple(
            self.maximum_xyz[index] - self.minimum_xyz[index] for index in range(3)
        )  # type: ignore[return-value]


def collision_primitives_penetrate(
    left: ProjectedCollisionPrimitive,
    right: ProjectedCollisionPrimitive,
    *,
    tolerance_m: float = 0.0,
) -> bool:
    """Return exact box/box penetration using the separating-axis theorem.

    Touching within ``tolerance_m`` is separation. Cylinders are intentionally
    not coerced to their enclosing boxes here: that proxy is useful for cheap
    ordering, but it cannot be used as a hard authoritative collision proof
    without risking false rejection.
    """

    if not math.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("collision penetration tolerance is invalid")
    if (
        left.shape != "box"
        or right.shape != "box"
        or left.size_xyz is None
        or right.size_xyz is None
    ):
        raise ValueError("exact primitive penetration supports boxes only")

    def columns(rotation: Rotation3) -> tuple[Vector3, Vector3, Vector3]:
        return tuple(
            tuple(rotation[row][column] for row in range(3))
            for column in range(3)
        )  # type: ignore[return-value]

    def dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(float(a[index]) * float(b[index]) for index in range(3))

    def cross(a: Sequence[float], b: Sequence[float]) -> Vector3:
        return (
            float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
            float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
            float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
        )

    left_axes = columns(left.rotation_rows)
    right_axes = columns(right.rotation_rows)
    left_half = left.enclosing_obb_half_sizes()
    right_half = right.enclosing_obb_half_sizes()
    delta = tuple(
        right.center_xyz[index] - left.center_xyz[index] for index in range(3)
    )
    axes = [*left_axes, *right_axes]
    axes.extend(cross(a, b) for a in left_axes for b in right_axes)
    for axis in axes:
        norm = math.sqrt(dot(axis, axis))
        if norm <= 1e-10:
            continue
        unit = tuple(value / norm for value in axis)
        center_distance = abs(dot(delta, unit))
        left_radius = sum(
            left_half[index] * abs(dot(left_axes[index], unit))
            for index in range(3)
        )
        right_radius = sum(
            right_half[index] * abs(dot(right_axes[index], unit))
            for index in range(3)
        )
        if left_radius + right_radius - center_distance <= tolerance_m:
            return False
    return True


def collision_geometry_volume_centroid(
    primitives: Sequence[ProjectedCollisionPrimitive],
) -> Vector3:
    """Return the uniform-density centroid of exact collision primitives.

    Perception objects do not always carry a measured centre of mass.  The
    volume centroid of the collision body is a deterministic, geometry-only
    proxy which is materially better than assuming that every link origin is
    its centre of mass.  It is used for placement ordering and, when an
    explicit container contract requests it, for the final "body is in the
    container" region test.  It never replaces Gazebo's stability, support,
    collision, or dynamics evidence.
    """

    weighted = [0.0, 0.0, 0.0]
    total_volume = 0.0
    for primitive in primitives:
        if primitive.shape == "box" and primitive.size_xyz is not None:
            volume = math.prod(primitive.size_xyz)
        elif (
            primitive.shape == "cylinder"
            and primitive.radius_m is not None
            and primitive.length_m is not None
        ):
            volume = math.pi * primitive.radius_m * primitive.radius_m * primitive.length_m
        else:
            raise ValueError("collision primitive geometry is incomplete")
        total_volume += volume
        for axis in range(3):
            weighted[axis] += volume * primitive.center_xyz[axis]
    if not math.isfinite(total_volume) or total_volume <= 0.0:
        raise ValueError("collision geometry volume is invalid")
    return tuple(value / total_volume for value in weighted)  # type: ignore[return-value]


def support_face_alignment_cosine(
    primitives: Sequence[ProjectedCollisionPrimitive],
) -> float:
    """Measure how closely the lowest primitive has a support face level.

    A value of one means a box face (or a cylinder cap/side family) is exactly
    aligned with gravity.  A lower value predicts that release will initially
    settle or tip.  This remains an ordering signal because meshes, friction
    and mass distribution can make a geometrically tilted pose viable.
    """

    if not primitives:
        raise ValueError("compound collision geometry is empty")
    lower_bounds = [
        primitive.center_xyz[2] - primitive.axis_half_extent(2)
        for primitive in primitives
    ]
    lowest = min(lower_bounds)
    # The primitives and transforms are produced by the same deterministic
    # arithmetic.  A relative machine-scale band admits coplanar compound
    # members without turning sensor tolerances into a hidden hard filter.
    scale = max(
        1.0,
        *(abs(value) for value in lower_bounds),
        *(primitive.axis_half_extent(2) for primitive in primitives),
    )
    contact_band = 64.0 * math.ulp(scale)
    alignments: list[float] = []
    for primitive, lower in zip(primitives, lower_bounds, strict=True):
        if lower > lowest + contact_band:
            continue
        if primitive.shape == "box":
            alignment = max(
                abs(primitive.rotation_rows[2][axis]) for axis in range(3)
            )
        elif primitive.shape == "cylinder":
            axial = min(1.0, abs(primitive.rotation_rows[2][2]))
            # Both a horizontal side and a vertical cap are analytic support
            # families for a cylinder.  Midway orientations are less settled.
            alignment = max(axial, math.sqrt(max(0.0, 1.0 - axial * axial)))
        else:
            raise ValueError("collision primitive shape is unsupported")
        alignments.append(min(1.0, max(0.0, alignment)))
    if not alignments:
        raise ValueError("supporting collision primitive is unavailable")
    return max(alignments)


def project_collision_geometry(
    *,
    object_xyz: Sequence[float],
    object_rotation: Sequence[Sequence[float]],
    primitives: object = (),
    fallback_size_xyz: Sequence[float] | None = None,
) -> tuple[ProjectedCollisionPrimitive, ...]:
    """Express one body's exact primitives at an arbitrary world pose.

    Cylinders retain their exact axis projection for support and footprint
    bounds.  Their returned enclosing OBB is used only by the inexpensive SAT
    collision pre-screen; MoveIt remains the final collision authority.
    """

    parent_xyz = _finite_vector(object_xyz, 3)
    parent_rotation = _rotation_matrix(object_rotation)
    if parent_xyz is None or parent_rotation is None:
        raise ValueError("object pose is invalid")
    raw_primitives = (
        list(primitives)
        if isinstance(primitives, Sequence)
        and not isinstance(primitives, (str, bytes, bytearray))
        else []
    )
    if not raw_primitives:
        size = _finite_vector(fallback_size_xyz, 3)
        if size is None or any(value <= 0.0 for value in size):
            raise ValueError("collision geometry is unavailable")
        return (
            ProjectedCollisionPrimitive(
                shape="box",
                center_xyz=parent_xyz,  # type: ignore[arg-type]
                rotation_rows=parent_rotation,
                size_xyz=size,  # type: ignore[arg-type]
            ),
        )

    projected: list[ProjectedCollisionPrimitive] = []
    for raw in raw_primitives:
        if not isinstance(raw, Mapping):
            raise ValueError("collision primitive is invalid")
        local_xyz = _finite_vector(raw.get("pose_xyz"), 3)
        local_rotation = _pose_rotation(raw)
        if local_xyz is None or local_rotation is None:
            raise ValueError("collision primitive pose is invalid")
        offset = _rotate(parent_rotation, local_xyz)
        center: Vector3 = tuple(parent_xyz[index] + offset[index] for index in range(3))  # type: ignore[assignment]
        rotation = _compose_rotation(parent_rotation, local_rotation)
        shape = str(raw.get("shape") or "")
        if shape == "box":
            size = _finite_vector(raw.get("size_xyz"), 3)
            if size is None or any(value <= 0.0 for value in size):
                raise ValueError("box collision primitive is invalid")
            projected.append(
                ProjectedCollisionPrimitive(
                    shape=shape,
                    center_xyz=center,
                    rotation_rows=rotation,
                    size_xyz=size,  # type: ignore[arg-type]
                )
            )
            continue
        if shape == "cylinder":
            try:
                radius = float(raw.get("radius"))
                length = float(raw.get("length"))
            except (TypeError, ValueError):
                raise ValueError("cylinder collision primitive is invalid") from None
            if not all(math.isfinite(value) and value > 0.0 for value in (radius, length)):
                raise ValueError("cylinder collision primitive is invalid")
            projected.append(
                ProjectedCollisionPrimitive(
                    shape=shape,
                    center_xyz=center,
                    rotation_rows=rotation,
                    radius_m=radius,
                    length_m=length,
                )
            )
            continue
        raise ValueError("collision primitive shape is unsupported")
    return tuple(projected)


def compound_axis_aligned_bounds(
    primitives: Sequence[ProjectedCollisionPrimitive],
) -> CompoundAxisAlignedBounds:
    if not primitives:
        raise ValueError("compound collision geometry is empty")
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    for primitive in primitives:
        for axis in range(3):
            half_extent = primitive.axis_half_extent(axis)
            minimum[axis] = min(minimum[axis], primitive.center_xyz[axis] - half_extent)
            maximum[axis] = max(maximum[axis], primitive.center_xyz[axis] + half_extent)
    return CompoundAxisAlignedBounds(
        minimum_xyz=tuple(minimum),  # type: ignore[arg-type]
        maximum_xyz=tuple(maximum),  # type: ignore[arg-type]
    )


def orientation_invariant_radius_m(
    primitives: Sequence[ProjectedCollisionPrimitive],
    *,
    object_xyz: Sequence[float],
) -> float:
    """Return the exact farthest-point radius around the body's link origin.

    The maximum distance is invariant under a later rigid rotation of the
    complete object.  Computing it from primitive geometry avoids the loose
    ``centre distance + primitive radius`` triangle bound, which can turn a
    valid offset compound body into a misleading near-wall placement.
    """

    origin = _finite_vector(object_xyz, 3)
    if origin is None or not primitives:
        raise ValueError("collision geometry origin is invalid")
    radius = 0.0
    for primitive in primitives:
        center_offset = tuple(
            primitive.center_xyz[index] - origin[index] for index in range(3)
        )
        if primitive.shape == "box":
            half_sizes = primitive.enclosing_obb_half_sizes()
            primitive_radius = max(
                math.dist(
                    origin,
                    tuple(
                        primitive.center_xyz[axis]
                        + sum(
                            primitive.rotation_rows[axis][local_axis]
                            * half_sizes[local_axis]
                            * sign[local_axis]
                            for local_axis in range(3)
                        )
                        for axis in range(3)
                    ),
                )
                for sign in (
                    (-1.0, -1.0, -1.0),
                    (-1.0, -1.0, 1.0),
                    (-1.0, 1.0, -1.0),
                    (-1.0, 1.0, 1.0),
                    (1.0, -1.0, -1.0),
                    (1.0, -1.0, 1.0),
                    (1.0, 1.0, -1.0),
                    (1.0, 1.0, 1.0),
                )
            )
        else:
            assert primitive.radius_m is not None and primitive.length_m is not None
            cylinder_axis = tuple(
                primitive.rotation_rows[axis][2] for axis in range(3)
            )
            axial_offset = sum(
                center_offset[axis] * cylinder_axis[axis] for axis in range(3)
            )
            perpendicular_offset = math.sqrt(
                max(
                    0.0,
                    sum(value * value for value in center_offset)
                    - axial_offset * axial_offset,
                )
            )
            primitive_radius = math.hypot(
                abs(axial_offset) + primitive.length_m / 2.0,
                perpendicular_offset + primitive.radius_m,
            )
        radius = max(radius, primitive_radius)
    return radius
