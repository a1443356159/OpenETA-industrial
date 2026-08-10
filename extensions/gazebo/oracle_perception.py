"""Gazebo oracle perception: project ground-truth object poses into a camera image.

Pure-Python geometry core for the M4 oracle perception module.  Object poses
come from the worker-side observation cache (``EnvObservation.objects``,
fed by the odometry bridge) and camera extrinsics from the static profile
config, so this module imports neither ROS nor Gazebo.

Occlusion is ignored on purpose: the oracle is ground truth, so every matched
object produces a detection even when another object would occlude it in the
rendered image.  The response shape mirrors ``tools/sam3_core.py`` field for
field so the agent-side SAM3 selection flow can consume it unchanged; the
only additions live in ``details.metadata`` (``perception_source`` and frame
matching diagnostics).
"""

from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .m3 import quaternion_rotate


# Rim samples per cylinder end disc.
_CYLINDER_RIM_SAMPLES = 16
# Points at or behind the optical centre cannot be projected.
_MIN_CAMERA_Z_M = 1e-6

_FRAME_MATCH_PIXEL = "pixel"
_FRAME_MATCH_FALLBACK_SIZE = "fallback_size"
_FRAME_NOT_FOUND = "frame_not_found"
_FRAME_AMBIGUOUS = "frame_match_ambiguous"


@dataclass(frozen=True, slots=True)
class OracleObjectSpec:
    """Static shape/label registry entry for one oracle-known object.

    ``shape`` is ``"box"`` (``dimensions = (sx, sy, sz)``) or ``"cylinder"``
    (``dimensions = (radius, length)``, axis along the object-local Z).
    """

    object_id: str
    name: str
    label: str
    shape: str
    dimensions: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("oracle object id must be non-empty")
        if self.shape not in ("box", "cylinder"):
            raise ValueError(f"unsupported oracle shape: {self.shape}")
        expected = 3 if self.shape == "box" else 2
        if len(self.dimensions) != expected:
            raise ValueError(f"{self.shape} dimensions must have {expected} values")
        if not all(math.isfinite(value) and value > 0 for value in self.dimensions):
            raise ValueError("oracle object dimensions must be positive and finite")


@dataclass(frozen=True, slots=True)
class PosedOracleObject:
    """One registry entry fused with a world-frame pose snapshot."""

    spec: OracleObjectSpec
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]  # xyzw


def oracle_registry_from_model_config(config: Any) -> list[OracleObjectSpec]:
    """Build the oracle registry from a profile ``model_config`` (duck-typed M3Config).

    Returns an empty list for configs without M3-style object declarations
    (e.g. M1/M2), which callers treat as "oracle unsupported on this env".
    """

    specs: list[OracleObjectSpec] = []
    target_id = getattr(config, "target_id", None)
    target_size = getattr(config, "target_size_m", None)
    if target_id and target_size:
        specs.append(
            OracleObjectSpec(
                object_id=str(target_id),
                name=str(target_id),
                label="target block",
                shape="box",
                dimensions=tuple(float(value) for value in target_size),
            )
        )
    distractor_id = getattr(config, "distractor_id", None)
    distractor_size = getattr(config, "distractor_size_m", None)
    if distractor_id and distractor_size:
        diameter, length = (float(value) for value in distractor_size)
        specs.append(
            OracleObjectSpec(
                object_id=str(distractor_id),
                name=str(distractor_id),
                label="distractor cylinder",
                shape="cylinder",
                dimensions=(diameter / 2.0, length),
            )
        )
    return specs


def posed_oracle_objects(
    registry: Sequence[OracleObjectSpec],
    objects: Sequence[Mapping[str, Any]],
) -> list[PosedOracleObject]:
    """Join observation object dicts with the shape registry on object id.

    Observation entries without a registry shape (e.g. the table) are
    skipped: without dimensions they cannot be projected.
    """

    by_id = {spec.object_id: spec for spec in registry}
    posed: list[PosedOracleObject] = []
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        spec = by_id.get(str(item.get("id") or item.get("object_id") or ""))
        if spec is None:
            continue
        position = item.get("position")
        orientation = item.get("orientation")
        if (
            not _is_finite_vector(position, 3)
            or not _is_finite_vector(orientation, 4)
            or math.sqrt(sum(float(value) ** 2 for value in orientation)) <= 1e-12  # type: ignore[union-attr]
        ):
            continue
        posed.append(
            PosedOracleObject(
                spec=spec,
                position=tuple(float(value) for value in position),  # type: ignore[arg-type]
                orientation=tuple(float(value) for value in orientation),  # type: ignore[arg-type]
            )
        )
    return posed


def match_camera_frame(
    image_array: Any,
    cameras: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, Mapping[str, Any] | None, str]:
    """Match a decoded RGB image against cached camera frames.

    Returns ``(frame_id, camera_entry, status)``.  Preferred status is
    ``"pixel"`` (size + pixel-exact match, so intrinsics/extrinsics are
    strictly from the same source as the image).  When no pixel match exists
    but exactly one camera shares the image size, the match falls back to
    ``"fallback_size"``; otherwise ``"frame_not_found"`` or
    ``"frame_match_ambiguous"`` with ``(None, None, ...)``.
    """

    import numpy as np

    query = np.asarray(image_array)
    query_shape = query.shape[:2]
    size_matches: list[tuple[str, Mapping[str, Any]]] = []
    for frame_id, camera in cameras.items():
        rgb = camera.get("rgb") if isinstance(camera, Mapping) else None
        if rgb is None:
            continue
        rgb_array = np.asarray(rgb)
        if rgb_array.shape[:2] != query_shape:
            continue
        if rgb_array.shape == query.shape and np.array_equal(rgb_array, query):
            return str(frame_id), camera, _FRAME_MATCH_PIXEL
        size_matches.append((str(frame_id), camera))
    if len(size_matches) == 1:
        frame_id, camera = size_matches[0]
        return frame_id, camera, _FRAME_MATCH_FALLBACK_SIZE
    if len(size_matches) > 1:
        return None, None, _FRAME_AMBIGUOUS
    return None, None, _FRAME_NOT_FOUND


def prompt_matches_object(prompt: str, spec: OracleObjectSpec) -> bool:
    """Case-insensitive substring match of *prompt* against id/name/label."""

    needle = prompt.strip().lower()
    if not needle:
        return False
    for field in (spec.object_id, spec.name, spec.label):
        value = field.strip().lower()
        if value and (needle in value or value in needle):
            return True
    return False


def oracle_segment_prompt(
    *,
    prompt: str,
    objects: Sequence[PosedOracleObject],
    intrinsics: Mapping[str, Any],
    extrinsics: Mapping[str, Any],
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project prompt-matched oracle objects into the camera and return masks.

    The response mirrors the SAM3 text-prompt contract
    (``tools/sam3_core.segment_image_prompt``) field for field, plus
    ``metadata.perception_source == "gazebo_oracle"``.
    """

    start_time = time.perf_counter()
    metadata: dict[str, Any] = {
        "perception_source": "gazebo_oracle",
        "prompt_type": "text",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    if not prompt:
        return oracle_failure_result(
            prompt=prompt,
            reason="missing_prompt",
            content="Oracle perception failed: missing prompt.",
            metadata=metadata,
        )

    extrinsics_error = _validate_extrinsics(extrinsics)
    if extrinsics_error is not None:
        reason, content = extrinsics_error
        return oracle_failure_result(
            prompt=prompt, reason=reason, content=content, metadata=metadata,
        )

    try:
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError):
        return oracle_failure_result(
            prompt=prompt,
            reason="invalid_intrinsics",
            content="Oracle perception failed: camera intrinsics are missing or invalid.",
            metadata=metadata,
        )
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        return oracle_failure_result(
            prompt=prompt,
            reason="invalid_intrinsics",
            content="Oracle perception failed: camera intrinsics are missing or invalid.",
            metadata=metadata,
        )
    metadata["image_size"] = [width, height]

    detections: list[dict[str, Any]] = []
    for backend_index, item in enumerate(objects):
        if not prompt_matches_object(prompt, item.spec):
            continue
        projected = _project_object(item, extrinsics=extrinsics, fx=fx, fy=fy, cx=cx, cy=cy)
        if len(projected) < 3:
            # Fully behind the camera or a degenerate projection: no detection.
            continue
        rendered = _render_mask(projected, width=width, height=height)
        if rendered is None:
            # Projects entirely outside the image bounds: no detection.
            continue
        mask_base64, area_px, bbox_xyxy = rendered
        detections.append(
            {
                "label": item.spec.label or prompt,
                "score": 1.0,
                "bbox_xyxy": bbox_xyxy,
                "mask": {"format": "png", "base64": mask_base64},
                "area_px": area_px,
                "backend_index": backend_index,
            }
        )

    # All oracle scores are 1.0; the sort is stable so ranking follows the
    # input object order, same as ``backend_index``.
    detections.sort(key=lambda detection: detection["score"], reverse=True)
    for rank, detection in enumerate(detections):
        detection["rank"] = rank

    metadata["duration_s"] = time.perf_counter() - start_time
    content = (
        "Oracle perception completed."
        if detections
        else "Oracle perception completed with no detections."
    )
    return {
        "success": True,
        "content": content,
        "details": {
            "tool": "oracle_perceive",
            "backend": "gazebo_oracle",
            "model": "gazebo_ground_truth",
            "prompt": prompt,
            "prompt_type": "text",
            "detection_count": len(detections),
            "detections": detections,
            "ranking": "score_descending",
            "artifacts": [],
            "metadata": metadata,
        },
    }


def oracle_failure_result(
    *,
    prompt: str,
    reason: str,
    content: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured failure in the SAM3 failure shape (``success == False``)."""

    merged = {"perception_source": "gazebo_oracle", "prompt_type": "text"}
    if metadata:
        merged.update(metadata)
    return {
        "success": False,
        "content": content,
        "details": {
            "tool": "oracle_perceive",
            "backend": "gazebo_oracle",
            "model": "gazebo_ground_truth",
            "prompt": prompt,
            "prompt_type": "text",
            "detection_count": 0,
            "detections": [],
            "artifacts": [],
            "reason": reason,
            "metadata": merged,
        },
    }


def _validate_extrinsics(extrinsics: Mapping[str, Any]) -> tuple[str, str] | None:
    if extrinsics.get("frame_transform") != "camera_to_world":
        return (
            "ORACLE_FRAME_UNSUPPORTED",
            "Oracle perception failed: camera frame has no numeric "
            f"camera_to_world extrinsics (got {extrinsics.get('frame_transform')!r}).",
        )
    if extrinsics.get("camera_frame") != "opencv":
        return (
            "ORACLE_FRAME_UNSUPPORTED",
            "Oracle perception failed: oracle projection requires the OpenCV "
            f"camera convention (got {extrinsics.get('camera_frame')!r}).",
        )
    pos = extrinsics.get("pos")
    quat = extrinsics.get("quat_xyzw")
    if not _is_finite_vector(pos, 3) or not _is_finite_vector(quat, 4):
        return (
            "ORACLE_FRAME_UNSUPPORTED",
            "Oracle perception failed: camera_to_world extrinsics lack numeric pos/quat_xyzw.",
        )
    return None


def _is_finite_vector(value: Any, length: int) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return False
    return all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in value
    )


def _sample_local_points(spec: OracleObjectSpec) -> list[tuple[float, float, float]]:
    if spec.shape == "box":
        hx, hy, hz = (value / 2.0 for value in spec.dimensions)
        return [
            (sx * hx, sy * hy, sz * hz)
            for sx in (1.0, -1.0)
            for sy in (1.0, -1.0)
            for sz in (1.0, -1.0)
        ]
    radius, length = spec.dimensions
    points: list[tuple[float, float, float]] = []
    for z in (length / 2.0, -length / 2.0):
        for index in range(_CYLINDER_RIM_SAMPLES):
            angle = 2.0 * math.pi * index / _CYLINDER_RIM_SAMPLES
            points.append((radius * math.cos(angle), radius * math.sin(angle), z))
    return points


def _project_object(
    item: PosedOracleObject,
    *,
    extrinsics: Mapping[str, Any],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> list[tuple[float, float]]:
    """Project one object's sampled surface points to pixel coordinates."""

    cam_pos = tuple(float(value) for value in extrinsics["pos"])
    cam_quat = _normalised(tuple(float(value) for value in extrinsics["quat_xyzw"]))
    # camera_to_world inverse: rotate by the conjugate quaternion.
    inv_quat = (-cam_quat[0], -cam_quat[1], -cam_quat[2], cam_quat[3])
    obj_quat = _normalised(item.orientation)

    projected: list[tuple[float, float]] = []
    for local in _sample_local_points(item.spec):
        rotated = quaternion_rotate(obj_quat, local)
        world = (
            rotated[0] + item.position[0],
            rotated[1] + item.position[1],
            rotated[2] + item.position[2],
        )
        delta = (
            world[0] - cam_pos[0],
            world[1] - cam_pos[1],
            world[2] - cam_pos[2],
        )
        x_cam, y_cam, z_cam = quaternion_rotate(inv_quat, delta)
        if z_cam <= _MIN_CAMERA_Z_M:
            continue
        projected.append((fx * x_cam / z_cam + cx, fy * y_cam / z_cam + cy))
    return projected


def _normalised(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion must be non-zero")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Monotone-chain convex hull over 2D points."""

    ordered = sorted(set(points))
    if len(ordered) <= 2:
        return list(ordered)

    def cross(origin, a, b) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _render_mask(
    projected: Sequence[tuple[float, float]],
    *,
    width: int,
    height: int,
) -> tuple[str, int, list[int]] | None:
    """Rasterise the projected convex hull into a 0/255 PNG mask.

    PIL clips polygons to the image bounds, so partially out-of-frame
    projections are cropped; a fully out-of-frame projection yields an empty
    mask and returns ``None``.  ``bbox_xyxy`` (half-open, max-exclusive) and
    ``area_px`` are recomputed from the rasterised mask, mirroring the SAM3
    backend behaviour.
    """

    import numpy as np
    from PIL import Image, ImageDraw

    hull = _convex_hull(projected)
    if len(hull) < 3:
        return None
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon([(float(x), float(y)) for x, y in hull], fill=255)
    mask = np.asarray(image) > 0
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    bbox_xyxy = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii"), int(mask.sum()), bbox_xyxy
