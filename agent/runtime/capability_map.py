"""Versioned sparse 6D capability maps used only to order qualification work.

An empty or mismatched cell is deliberately represented as low confidence.  It
is never interpreted as a proof that a target is unreachable; MoveIt remains
the authority for every rejection and final plan-only proof.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree

import numpy as np

from adapter.protocol import JsonDict


CAPABILITY_MAP_SCHEMA = "openeta.sparse_capability_map.v1"
DEFAULT_CAPABILITY_SAMPLE_COUNT = 2_000_000
DEFAULT_POSITION_RESOLUTION_M = 0.02
DEFAULT_ORIENTATION_NEIGHBORHOOD_DEG = 15.0
DEFAULT_SOBOL_SEED = 0x4F50454E455441


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def robot_model_hash(
    *,
    urdf: str | bytes,
    srdf: str | bytes,
    planning_group: str,
    tcp: str,
    gripper: str,
) -> str:
    """Bind a map to all model inputs that can alter reachability."""

    def digest(value: str | bytes) -> str:
        payload = value.encode("utf-8") if isinstance(value, str) else value
        try:
            canonical = ElementTree.canonicalize(
                payload.decode("utf-8"), strip_text=True
            )
            payload = canonical.encode("utf-8")
        except (UnicodeDecodeError, ElementTree.ParseError, ValueError):
            pass
        return hashlib.sha256(payload).hexdigest()

    return _canonical_hash(
        {
            "urdf_sha256": digest(urdf),
            "srdf_sha256": digest(srdf),
            "planning_group": planning_group,
            "tcp": tcp,
            "gripper": gripper,
        }
    )


def _finite_vector(value: object, length: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _quat_from_rotation(rotation: object) -> list[float] | None:
    if not (
        isinstance(rotation, Sequence)
        and len(rotation) == 3
        and all(isinstance(row, Sequence) and len(row) == 3 for row in rotation)
    ):
        return None
    try:
        matrix = [[float(item) for item in row] for row in rotation]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for row in matrix for item in row):
        return None
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(max(0.0, trace + 1.0)) * 2.0
        if scale <= 1e-12:
            return None
        quat = [
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        ]
    else:
        diagonal = max(range(3), key=lambda index: matrix[index][index])
        if diagonal == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])) * 2.0
            if scale <= 1e-12:
                return None
            quat = [0.25 * scale, (matrix[0][1] + matrix[1][0]) / scale, (matrix[0][2] + matrix[2][0]) / scale, (matrix[2][1] - matrix[1][2]) / scale]
        elif diagonal == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])) * 2.0
            if scale <= 1e-12:
                return None
            quat = [(matrix[0][1] + matrix[1][0]) / scale, 0.25 * scale, (matrix[1][2] + matrix[2][1]) / scale, (matrix[0][2] - matrix[2][0]) / scale]
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])) * 2.0
            if scale <= 1e-12:
                return None
            quat = [(matrix[0][2] + matrix[2][0]) / scale, (matrix[1][2] + matrix[2][1]) / scale, 0.25 * scale, (matrix[1][0] - matrix[0][1]) / scale]
    return _normalize_quaternion(quat)


def _normalize_quaternion(value: object) -> list[float] | None:
    quat = _finite_vector(value, 4)
    if quat is None:
        return None
    norm = math.sqrt(sum(item * item for item in quat))
    if norm <= 1e-12:
        return None
    return [item / norm for item in quat]


def target_pose(target: Mapping[str, Any]) -> tuple[list[float], list[float]] | None:
    xyz = _finite_vector(
        target.get("xyz") or target.get("translation_xyz") or target.get("position"),
        3,
    )
    quat = _normalize_quaternion(
        target.get("quat_xyzw") or target.get("quaternion_xyzw")
    )
    if quat is None:
        quat = _quat_from_rotation(target.get("rotation_matrix"))
    if xyz is None or quat is None:
        return None
    return xyz, quat


def quaternion_angle_rad(left: Sequence[float], right: Sequence[float]) -> float:
    dot = min(1.0, max(-1.0, abs(sum(a * b for a, b in zip(left, right)))))
    return 2.0 * math.acos(dot)


@dataclass(frozen=True, slots=True)
class CapabilityScore:
    confidence: float
    reachable_density: float
    joint_margin: float
    min_singular_value: float
    matched_stages: int
    stage_count: int

    def ordering_key(self) -> tuple[float, float, float, float]:
        return (
            self.confidence,
            self.reachable_density,
            self.joint_margin,
            self.min_singular_value,
        )

    def to_dict(self) -> JsonDict:
        return {
            "confidence": self.confidence,
            "reachable_density": self.reachable_density,
            "joint_margin": self.joint_margin,
            "min_singular_value": self.min_singular_value,
            "matched_stages": self.matched_stages,
            "stage_count": self.stage_count,
        }


@dataclass(frozen=True, slots=True)
class SparseCapabilityMap:
    map_id: str
    robot_model_sha256: str
    position_resolution_m: float
    orientation_neighborhood_deg: float
    cells: Mapping[str, tuple[Mapping[str, Any], ...]]
    sample_count: int = DEFAULT_CAPABILITY_SAMPLE_COUNT
    sobol_seed: int = DEFAULT_SOBOL_SEED

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_map_id: str = "",
        expected_robot_model_sha256: str = "",
    ) -> "SparseCapabilityMap":
        if payload.get("schema_version") != CAPABILITY_MAP_SCHEMA:
            raise ValueError("unsupported capability map schema")
        identity_payload = {key: value for key, value in payload.items() if key != "map_id"}
        computed_id = _canonical_hash(identity_payload)
        map_id = str(payload.get("map_id") or computed_id)
        if map_id != computed_id:
            raise ValueError("capability map content hash mismatch")
        robot_hash = str(payload.get("robot_model_sha256") or "")
        if expected_map_id and map_id != expected_map_id:
            raise ValueError("capability map ID does not match qualification profile")
        if expected_robot_model_sha256 and robot_hash != expected_robot_model_sha256:
            raise ValueError("capability map robot/TCP/gripper hash mismatch")
        cells_value = payload.get("cells")
        if not isinstance(cells_value, Mapping):
            raise ValueError("capability map cells are missing")
        position_resolution = float(payload.get("position_resolution_m", 0.02))
        orientation_neighborhood = float(
            payload.get("orientation_neighborhood_deg", 15.0)
        )
        sample_count = int(
            payload.get("sample_count", DEFAULT_CAPABILITY_SAMPLE_COUNT)
        )
        if (
            not math.isfinite(position_resolution)
            or position_resolution <= 0.0
            or not math.isfinite(orientation_neighborhood)
            or not 0.0 < orientation_neighborhood <= 180.0
            or sample_count < 0
        ):
            raise ValueError("capability map resolutions/sample count are invalid")
        cells: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for key, entries in cells_value.items():
            if not isinstance(entries, list):
                raise ValueError("capability map cell entries are malformed")
            validated: list[Mapping[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError("capability map cell entry is not an object")
                orientation = _normalize_quaternion(entry.get("quat_xyzw"))
                try:
                    density = float(entry.get("reachable_density", 0.0))
                    margin = float(entry.get("joint_margin", 0.0))
                    singular = float(entry.get("min_singular_value", 0.0))
                    entry_samples = int(entry.get("sample_count", 0))
                except (TypeError, ValueError) as exc:
                    raise ValueError("capability map cell metrics are malformed") from exc
                if (
                    orientation is None
                    or not all(math.isfinite(value) for value in (density, margin, singular))
                    or not 0.0 <= density <= 1.0
                    or margin < 0.0
                    or singular < 0.0
                    or entry_samples < 0
                ):
                    raise ValueError("capability map cell metrics are invalid")
                validated.append(dict(entry))
            cells[str(key)] = tuple(validated)
        return cls(
            map_id=map_id,
            robot_model_sha256=robot_hash,
            position_resolution_m=position_resolution,
            orientation_neighborhood_deg=orientation_neighborhood,
            cells=cells,
            sample_count=sample_count,
            sobol_seed=int(payload.get("sobol_seed", DEFAULT_SOBOL_SEED)),
        )

    def _position_key(self, xyz: Sequence[float]) -> str:
        indices = [math.floor(value / self.position_resolution_m) for value in xyz]
        return ",".join(str(value) for value in indices)

    def lookup(self, target: Mapping[str, Any]) -> CapabilityScore:
        pose = target_pose(target)
        if pose is None:
            return CapabilityScore(0.0, 0.0, 0.0, 0.0, 0, 1)
        xyz, quat = pose
        entries = self.cells.get(self._position_key(xyz), ())
        maximum_angle = math.radians(self.orientation_neighborhood_deg)
        matches = []
        for entry in entries:
            orientation = _normalize_quaternion(entry.get("quat_xyzw"))
            if orientation is None:
                continue
            angle = quaternion_angle_rad(quat, orientation)
            if angle <= maximum_angle:
                matches.append((angle, entry))
        if not matches:
            return CapabilityScore(0.0, 0.0, 0.0, 0.0, 0, 1)
        _, best = min(matches, key=lambda item: (item[0], _canonical_hash(item[1])))
        return CapabilityScore(
            confidence=1.0,
            reachable_density=float(best.get("reachable_density", 0.0)),
            joint_margin=float(best.get("joint_margin", 0.0)),
            min_singular_value=float(best.get("min_singular_value", 0.0)),
            matched_stages=1,
            stage_count=1,
        )

    def score_chain(self, stages: Sequence[Mapping[str, Any]]) -> CapabilityScore:
        if not stages:
            return CapabilityScore(0.0, 0.0, 0.0, 0.0, 0, 0)
        scores = [self.lookup(stage) for stage in stages]
        matched = sum(score.matched_stages for score in scores)
        return CapabilityScore(
            confidence=matched / len(stages),
            reachable_density=min(score.reachable_density for score in scores),
            joint_margin=min(score.joint_margin for score in scores),
            min_singular_value=min(score.min_singular_value for score in scores),
            matched_stages=matched,
            stage_count=len(stages),
        )


# Primitive-polynomial parameters for the first eight Sobol dimensions.  RM75
# uses seven arm joints; keeping one spare dimension makes the helper reusable
# by gripper-inclusive offline tooling without adding SciPy as a runtime dep.
_SOBOL_PARAMETERS: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (0, 0, ()),
    (1, 0, (1,)),
    (2, 1, (1, 3)),
    (3, 1, (1, 3, 1)),
    (3, 2, (1, 1, 1)),
    (4, 1, (1, 3, 5, 13)),
    (4, 4, (1, 1, 5, 5)),
    (5, 2, (1, 3, 3, 9, 7)),
)


def sobol_batches(
    dimensions: int,
    count: int,
    *,
    seed: int = DEFAULT_SOBOL_SEED,
    batch_size: int = 16_384,
) -> Iterator[np.ndarray]:
    """Yield deterministic scrambled Sobol points without allocating 2M rows."""

    if dimensions < 1 or dimensions > len(_SOBOL_PARAMETERS):
        raise ValueError(f"Sobol dimensions must be in [1, {len(_SOBOL_PARAMETERS)}]")
    if count < 0 or batch_size < 1:
        raise ValueError("Sobol count and batch size are invalid")
    bits = 32
    directions = np.zeros((dimensions, bits), dtype=np.uint32)
    directions[0] = np.array([1 << (31 - index) for index in range(bits)], dtype=np.uint32)
    for dimension in range(1, dimensions):
        degree, coefficient, initial = _SOBOL_PARAMETERS[dimension]
        for index in range(degree):
            directions[dimension, index] = np.uint32(initial[index] << (31 - index))
        for index in range(degree, bits):
            value = directions[dimension, index - degree] ^ (
                directions[dimension, index - degree] >> np.uint32(degree)
            )
            for offset in range(1, degree):
                if (coefficient >> (degree - 1 - offset)) & 1:
                    value ^= directions[dimension, index - offset]
            directions[dimension, index] = value
    rng = np.random.default_rng(seed)
    scramble = rng.integers(0, 2**32, size=dimensions, dtype=np.uint32)
    state = np.zeros(dimensions, dtype=np.uint32)
    pending: list[np.ndarray] = []
    for index in range(count):
        if index:
            bit = (index & -index).bit_length() - 1
            state ^= directions[:, bit]
        pending.append(((state ^ scramble).astype(np.float64) / 2**32))
        if len(pending) >= batch_size:
            yield np.stack(pending)
            pending.clear()
    if pending:
        yield np.stack(pending)


def generate_sparse_capability_map(
    *,
    robot_model_sha256: str,
    joint_lower: Sequence[float],
    joint_upper: Sequence[float],
    forward_kinematics: Callable[[np.ndarray], tuple[Sequence[float], Sequence[float]]],
    jacobian: Callable[[np.ndarray], np.ndarray],
    sample_count: int = DEFAULT_CAPABILITY_SAMPLE_COUNT,
    sobol_seed: int = DEFAULT_SOBOL_SEED,
    position_resolution_m: float = DEFAULT_POSITION_RESOLUTION_M,
    orientation_neighborhood_deg: float = DEFAULT_ORIENTATION_NEIGHBORHOOD_DEG,
) -> JsonDict:
    """Generate a sparse map from fixed-seed Sobol joint samples.

    The callbacks keep URDF/FK dependencies in offline tooling.  Each stored
    orientation bucket records sample density, worst normalized joint-limit
    margin, and worst Jacobian minimum singular value.
    """

    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 1 or lower.size < 1:
        raise ValueError("joint limits are malformed")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(upper <= lower):
        raise ValueError("joint limits must be finite and increasing")
    if (
        sample_count < 0
        or not math.isfinite(position_resolution_m)
        or position_resolution_m <= 0.0
        or not math.isfinite(orientation_neighborhood_deg)
        or not 0.0 < orientation_neighborhood_deg <= 180.0
    ):
        raise ValueError("capability map generation resolution/count is invalid")
    aggregate: dict[tuple[int, int, int, int, int, int, int], list[float]] = defaultdict(
        lambda: [0.0, 1.0, math.inf]
    )
    orientation_resolution = math.radians(orientation_neighborhood_deg)
    for batch in sobol_batches(lower.size, sample_count, seed=sobol_seed):
        for unit in batch:
            joints = lower + unit * (upper - lower)
            xyz_raw, quat_raw = forward_kinematics(joints)
            xyz = _finite_vector(xyz_raw, 3)
            quat = _normalize_quaternion(quat_raw)
            if xyz is None or quat is None:
                continue
            # q and -q are identical; canonicalize before coarse orientation bins.
            if quat[3] < 0.0:
                quat = [-value for value in quat]
            position_bin = tuple(math.floor(value / position_resolution_m) for value in xyz)
            orientation_bin = tuple(
                math.floor(value / max(1e-9, orientation_resolution / 2.0))
                for value in quat
            )
            key = (*position_bin, *orientation_bin)
            margin = float(
                np.min(np.minimum(joints - lower, upper - joints) / (upper - lower))
            )
            jacobian_matrix = np.asarray(jacobian(joints), dtype=np.float64)
            if (
                jacobian_matrix.ndim != 2
                or jacobian_matrix.size == 0
                or not np.all(np.isfinite(jacobian_matrix))
            ):
                raise ValueError("Jacobian callback returned an invalid matrix")
            try:
                singular_values = np.linalg.svd(
                    jacobian_matrix, compute_uv=False
                )
            except np.linalg.LinAlgError as exc:
                raise ValueError("Jacobian singular-value computation failed") from exc
            sigma = float(np.min(singular_values)) if singular_values.size else 0.0
            entry = aggregate[key]
            entry[0] += 1.0
            entry[1] = min(entry[1], margin)
            entry[2] = min(entry[2], sigma)
    cells: dict[str, list[JsonDict]] = defaultdict(list)
    for key in sorted(aggregate):
        count, margin, sigma = aggregate[key]
        position_key = key[:3]
        # Store the center quaternion of the coarse bucket, normalized.
        orientation = [
            (value + 0.5) * max(1e-9, orientation_resolution / 2.0)
            for value in key[3:]
        ]
        orientation = _normalize_quaternion(orientation) or [0.0, 0.0, 0.0, 1.0]
        cells[",".join(str(value) for value in position_key)].append(
            {
                "quat_xyzw": orientation,
                "sample_count": int(count),
                "reachable_density": count / max(1, sample_count),
                "joint_margin": margin,
                "min_singular_value": 0.0 if math.isinf(sigma) else sigma,
            }
        )
    payload: JsonDict = {
        "schema_version": CAPABILITY_MAP_SCHEMA,
        "robot_model_sha256": robot_model_sha256,
        "sample_count": sample_count,
        "sobol_seed": sobol_seed,
        "position_resolution_m": position_resolution_m,
        "orientation_neighborhood_deg": orientation_neighborhood_deg,
        "cells": dict(cells),
    }
    payload["map_id"] = _canonical_hash(payload)
    return payload
