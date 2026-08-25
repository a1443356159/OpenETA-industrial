"""Small dependency-free URDF serial-chain Jacobian evaluator.

The live qualification path needs a quality value for each concrete IK branch,
not just the target-pose capability-map prior.  MoveIt's Python service reply
does not expose a Jacobian, so this module evaluates the geometric Jacobian
from the same expanded URDF used by the running robot model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from xml.etree import ElementTree

import numpy as np


def _vector(text: str | None, *, default: Sequence[float]) -> np.ndarray:
    values = default if not text else tuple(float(value) for value in text.split())
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("URDF vector must contain three finite values")
    return np.asarray(values, dtype=np.float64)


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _transform(xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1e-12 or not math.isfinite(angle):
        raise ValueError("URDF joint axis/position is invalid")
    x, y, z = axis / norm
    cosine, sine = math.cos(angle), math.sin(angle)
    complement = 1.0 - cosine
    return np.asarray(
        [
            [cosine + x * x * complement, x * y * complement - z * sine, x * z * complement + y * sine],
            [y * x * complement + z * sine, cosine + y * y * complement, y * z * complement - x * sine],
            [z * x * complement - y * sine, z * y * complement + x * sine, cosine + z * z * complement],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class UrdfSerialChain:
    """The unique URDF joint chain between one base and tip link."""

    base_link: str
    tip_link: str
    joints: tuple[_Joint, ...]

    @classmethod
    def from_urdf(
        cls,
        urdf: str | bytes,
        *,
        base_link: str,
        tip_link: str,
    ) -> "UrdfSerialChain":
        try:
            root = ElementTree.fromstring(urdf)
        except (ElementTree.ParseError, TypeError) as exc:
            raise ValueError("expanded URDF is invalid") from exc
        by_child: dict[str, _Joint] = {}
        for element in root.findall("joint"):
            parent_element = element.find("parent")
            child_element = element.find("child")
            parent = parent_element.get("link") if parent_element is not None else None
            child = child_element.get("link") if child_element is not None else None
            name, kind = element.get("name"), element.get("type")
            if not name or not kind or not parent or not child:
                raise ValueError("URDF joint identity is incomplete")
            origin = element.find("origin")
            xyz = _vector(origin.get("xyz") if origin is not None else None, default=(0.0, 0.0, 0.0))
            rpy = _vector(origin.get("rpy") if origin is not None else None, default=(0.0, 0.0, 0.0))
            axis_element = element.find("axis")
            axis = _vector(
                axis_element.get("xyz") if axis_element is not None else None,
                default=(1.0, 0.0, 0.0),
            )
            if child in by_child:
                raise ValueError(f"URDF link {child!r} has multiple parent joints")
            by_child[child] = _Joint(
                name=name,
                kind=kind,
                parent=parent,
                child=child,
                origin_xyz=tuple(float(value) for value in xyz),
                origin_rpy=tuple(float(value) for value in rpy),
                axis=tuple(float(value) for value in axis),
            )
        reverse: list[_Joint] = []
        link = tip_link
        visited: set[str] = set()
        while link != base_link:
            if link in visited or link not in by_child:
                raise ValueError(
                    f"URDF has no unique chain from {base_link!r} to {tip_link!r}"
                )
            visited.add(link)
            joint = by_child[link]
            reverse.append(joint)
            link = joint.parent
        joints = tuple(reversed(reverse))
        if not any(joint.kind != "fixed" for joint in joints):
            raise ValueError("URDF serial chain has no movable joints")
        return cls(base_link=base_link, tip_link=tip_link, joints=joints)

    @property
    def movable_joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints if joint.kind != "fixed")

    @property
    def translation_upper_bound_m(self) -> float:
        """Triangle-inequality outer bound for the serial-chain tip origin."""

        return sum(
            math.sqrt(sum(value * value for value in joint.origin_xyz))
            for joint in self.joints
        )

    @property
    def translation_lower_bound_m(self) -> float:
        """Safe (possibly loose) inner radius implied by fixed link lengths."""

        lengths = [
            math.sqrt(sum(value * value for value in joint.origin_xyz))
            for joint in self.joints
        ]
        if not lengths:
            return 0.0
        longest = max(lengths)
        return max(0.0, longest - (sum(lengths) - longest))

    def jacobian(
        self,
        joint_names: Sequence[str],
        joint_positions: Sequence[float],
    ) -> np.ndarray:
        transform, columns = self._forward_state(joint_names, joint_positions)
        endpoint = transform[:3, 3]
        jacobian = np.zeros((6, len(columns)), dtype=np.float64)
        for index, (kind, origin, axis) in enumerate(columns):
            if kind == "prismatic":
                jacobian[:3, index] = axis
            else:
                jacobian[:3, index] = np.cross(axis, endpoint - origin)
                jacobian[3:, index] = axis
        return jacobian

    def forward_kinematics(
        self,
        joint_names: Sequence[str],
        joint_positions: Sequence[float],
    ) -> tuple[list[float], list[float]]:
        transform, _ = self._forward_state(joint_names, joint_positions)
        rotation = transform[:3, :3]
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        else:
            diagonal = int(np.argmax(np.diag(rotation)))
            first, second, third = (
                (0, 1, 2)
                if diagonal == 0
                else (1, 2, 0)
                if diagonal == 1
                else (2, 0, 1)
            )
            scale = math.sqrt(
                max(
                    0.0,
                    1.0
                    + rotation[first, first]
                    - rotation[second, second]
                    - rotation[third, third],
                )
            ) * 2.0
            if scale <= 1e-12:
                raise ValueError("forward-kinematics rotation is invalid")
            quaternion = [0.0, 0.0, 0.0, 0.0]
            quaternion[first] = 0.25 * scale
            quaternion[second] = (
                rotation[first, second] + rotation[second, first]
            ) / scale
            quaternion[third] = (
                rotation[first, third] + rotation[third, first]
            ) / scale
            quaternion[3] = (
                rotation[third, second] - rotation[second, third]
            ) / scale
        norm = math.sqrt(sum(float(value) ** 2 for value in quaternion))
        if norm <= 1e-12 or not math.isfinite(norm):
            raise ValueError("forward-kinematics quaternion is invalid")
        return (
            [float(value) for value in transform[:3, 3]],
            [float(value) / norm for value in quaternion],
        )

    def _forward_state(
        self,
        joint_names: Sequence[str],
        joint_positions: Sequence[float],
    ) -> tuple[np.ndarray, list[tuple[str, np.ndarray, np.ndarray]]]:
        if len(joint_names) != len(joint_positions):
            raise ValueError("joint names and positions differ in length")
        positions = {
            str(name): float(position)
            for name, position in zip(joint_names, joint_positions, strict=True)
        }
        if any(not math.isfinite(value) for value in positions.values()):
            raise ValueError("joint positions must be finite")
        transform = np.eye(4, dtype=np.float64)
        columns: list[tuple[str, np.ndarray, np.ndarray]] = []
        for joint in self.joints:
            transform = transform @ _transform(
                np.asarray(joint.origin_xyz),
                _rpy_rotation(np.asarray(joint.origin_rpy)),
            )
            if joint.kind == "fixed":
                continue
            if joint.kind not in {"continuous", "revolute", "prismatic"}:
                raise ValueError(f"unsupported URDF joint type {joint.kind!r}")
            if joint.name not in positions:
                raise ValueError(f"joint state omitted {joint.name!r}")
            axis = transform[:3, :3] @ np.asarray(joint.axis)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-12:
                raise ValueError(f"joint {joint.name!r} has a zero axis")
            axis /= axis_norm
            columns.append((joint.kind, transform[:3, 3].copy(), axis.copy()))
            position = positions[joint.name]
            if joint.kind == "prismatic":
                transform = transform @ _transform(
                    np.asarray(joint.axis) * position,
                    np.eye(3, dtype=np.float64),
                )
            else:
                transform = transform @ _transform(
                    np.zeros(3, dtype=np.float64),
                    _axis_rotation(np.asarray(joint.axis), position),
                )
        return transform, columns

    def minimum_singular_value(
        self,
        joint_names: Sequence[str],
        joint_positions: Sequence[float],
    ) -> float:
        singular_values = np.linalg.svd(
            self.jacobian(joint_names, joint_positions), compute_uv=False
        )
        if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
            raise ValueError("Jacobian singular values are invalid")
        return float(np.min(singular_values))


def capability_map_plugin(args):
    """Built-in provider for ``scripts/generate_capability_map.py``."""

    chain = UrdfSerialChain.from_urdf(
        args.urdf.read_bytes(),
        base_link=args.base_link,
        tip_link=args.tcp,
    )
    names = chain.movable_joint_names
    if len(names) != len(args.joint_lower):
        raise ValueError(
            "expanded URDF chain joint count differs from supplied limits"
        )

    class Plugin:
        def forward_kinematics(self, joints):
            return chain.forward_kinematics(names, joints)

        def jacobian(self, joints):
            return chain.jacobian(names, joints)

    return Plugin()
