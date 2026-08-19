"""Gazebo perception-bridge perception bridge: selected SAM3 detection → 3D object summary.

Pure-Python geometry core for the perception-bridge perception bridge.  SAM3 (and the oracle-fixture
oracle, which shares its contract) produces pixel-level outputs — a mask and a
bbox — while the planner consumes plan.md §17 object summaries with a
world-frame ``position``.  This module closes that gap: it takes one selected
detection (``mask_ref`` + ``source_image``, or an inline PNG ``mask``) and the
:class:`adapter.protocol.CameraFrame` it was produced from, and back-projects
the mask into a world-frame position.

Geometry: the median of the valid (finite, positive) depths inside the mask
gives the range ``z``; the centroid of those valid pixels gives ``(u, v)``;
the pinhole model yields the camera-frame point, which the
``camera_to_world`` extrinsics (OpenCV camera convention) then move into the
world frame.  Depth is never invented: missing depth, an empty mask, or
unsupported extrinsics (e.g. the wrist camera's ``tf_dynamic``) yield
``position = None`` with a ``position_error`` reason, per plan.md §17
("Use ``unknown`` / missing fields where appropriate").

The module imports neither ROS nor Gazebo; numpy/PIL are imported lazily,
mirroring ``extensions/gazebo/oracle_perception.py``.
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .native_grasp import quaternion_rotate


PROVENANCE_SAM3 = "sam3_perception"

# ``position_error`` reasons when ``position`` is None.
ERR_MASK_MISSING = "mask_missing"
ERR_MASK_DECODE = "mask_decode_failed"
ERR_EMPTY_MASK = "empty_mask"
ERR_DEPTH_MISSING = "depth_missing"
ERR_NO_VALID_DEPTH = "no_valid_depth"
ERR_INTRINSICS = "invalid_intrinsics"
ERR_EXTRINSICS = "unsupported_extrinsics"


class PerceptionBridgeError(ValueError):
    """A fail-closed violation at the perception-bridge RGB-D/SAM3 boundary.

    The generic helpers in this module deliberately support a few useful
    offline conveniences (for example mask resizing).  The live perception-bridge control
    path cannot make those assumptions: it receives one particular observe
    frame and must prove that SAM3's mask belongs to that exact RGB-D frame.
    This exception gives that stricter caller a stable, auditable reason code.
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


PERCEPTION_ERR_CASE_ROOT = "perception_artifact_outside_case_root"
PERCEPTION_ERR_SOURCE_IMAGE = "perception_source_image_mismatch"
PERCEPTION_ERR_SOURCE_FRAME = "perception_source_frame_mismatch"
PERCEPTION_ERR_RGB_MISSING = "perception_rgb_missing"
PERCEPTION_ERR_DEPTH_MISSING = "perception_depth_missing"
PERCEPTION_ERR_DEPTH_DECODE = "perception_depth_decode_failed"
PERCEPTION_ERR_RGBD_SHAPE = "perception_rgb_depth_shape_mismatch"
PERCEPTION_ERR_DEPTH_SCALE = "perception_depth_scale_invalid"
PERCEPTION_ERR_MASK_MISSING = "perception_mask_missing"
PERCEPTION_ERR_MASK_DECODE = "perception_mask_decode_failed"
PERCEPTION_ERR_MASK_SHAPE = "perception_mask_depth_shape_mismatch"
PERCEPTION_ERR_POSITION = "perception_position_unavailable"


def summarize_detection(
    *,
    detection: Mapping[str, Any],
    camera_frame: Any,
    provenance: str = PROVENANCE_SAM3,
) -> dict[str, Any]:
    """Build one plan.md §17 object-summary entry from a selected detection.

    Parameters
    ----------
    detection:
        Selected SAM3/oracle detection.  The mask comes from ``mask_ref``
        (path to a single-channel PNG, foreground > 0) or an inline
        ``mask`` mapping (``{"format": "png", "base64": ...}``).
        ``source_image`` is accepted as part of the selection contract but
        not re-verified: frame correspondence (detection ↔ camera frame) is
        the caller's responsibility, same as the oracle-fixture worker-side frame
        matching.  ``id`` / ``label`` / ``score`` map to ``id`` / ``label``
        / ``confidence``.
    camera_frame:
        :class:`adapter.protocol.CameraFrame` or an equivalent mapping with
        ``frame_id`` / ``depth`` / ``intrinsics`` / ``extrinsics``.  Depth is
        a ``height × width`` array in metres.
    provenance:
        Value for the ``provenance`` field; defaults to ``"sam3_perception"``.

    Returns
    -------
    dict
        ``id`` / ``label`` / ``confidence`` / ``position`` / ``visibility`` /
        ``source_camera`` / ``provenance``.  ``position`` is ``[x, y, z]`` in
        the world frame, or ``None`` with an extra ``position_error`` reason
        when it cannot be computed honestly.
    """

    entry: dict[str, Any] = {
        "id": _detection_id(detection),
        "label": _detection_label(detection),
        "confidence": _detection_confidence(detection),
        "position": None,
        "visibility": "unknown",
        "source_camera": str(_frame_field(camera_frame, "frame_id") or ""),
        "provenance": provenance,
    }

    failure, position = _back_project(detection, camera_frame)
    if position is not None:
        entry["position"] = position
    else:
        entry["position_error"] = failure
    return entry


def build_object_summary(
    *,
    detections: Sequence[Mapping[str, Any]],
    camera_frame: Any,
    provenance: str = PROVENANCE_SAM3,
) -> dict[str, Any]:
    """Summarise several detections into the plan.md §17 ``{"objects": [...]}`` shape."""

    return {
        "objects": [
            summarize_detection(
                detection=detection, camera_frame=camera_frame, provenance=provenance,
            )
            for detection in detections
        ]
    }


def build_perception_object_summary(
    *,
    detection: Mapping[str, Any],
    camera: Mapping[str, Any],
    case_root: str | Path,
) -> dict[str, Any]:
    """Build the strict, case-local perception-bridge object summary for one SAM3 selection.

    Unlike :func:`summarize_detection`, this is intentionally not a generic
    convenience API.  It accepts only materialized artifacts from the current
    ``observe`` result: the selected detection's ``source_image`` and
    ``source_frame_id`` must match the camera's RGB path and frame id, and its
    mask must have the *same* pixels as the decoded metric depth image.  Every
    artifact must live under ``case_root``.  Any uncertainty raises
    :class:`PerceptionBridgeError`; no path, frame, scale, or image resize is
    guessed by the live control path.

    ``camera`` is the materialized camera mapping from a single MCP observe
    response (``rgb_path``, ``depth_path``, intrinsics and numeric
    ``camera_to_world`` extrinsics).  The returned shape is a durable perception-bridge
    object summary, ready to be embedded in a perception evidence report.
    """

    root = Path(case_root).resolve()
    if not root.is_dir():
        raise PerceptionBridgeError(PERCEPTION_ERR_CASE_ROOT, "case root does not exist")

    frame_id = str(camera.get("frame_id") or "").strip()
    if not frame_id:
        raise PerceptionBridgeError(PERCEPTION_ERR_SOURCE_FRAME, "camera frame_id is missing")
    selected_frame_id = str(detection.get("source_frame_id") or "").strip()
    if selected_frame_id != frame_id:
        raise PerceptionBridgeError(
            PERCEPTION_ERR_SOURCE_FRAME,
            "selected detection does not identify the current camera frame",
        )

    rgb_path = _case_file(camera.get("rgb_path"), root, PERCEPTION_ERR_RGB_MISSING)
    source_image = _case_file(
        detection.get("source_image"), root, PERCEPTION_ERR_SOURCE_IMAGE
    )
    if source_image != rgb_path:
        raise PerceptionBridgeError(
            PERCEPTION_ERR_SOURCE_IMAGE,
            "selected detection source_image is not the current RGB artifact",
        )
    depth_path = _case_file(camera.get("depth_path"), root, PERCEPTION_ERR_DEPTH_MISSING)
    mask_path = _case_file(detection.get("mask_ref"), root, PERCEPTION_ERR_MASK_MISSING)

    from PIL import Image
    import numpy as np

    try:
        with Image.open(rgb_path) as rgb_image:
            rgb_size = rgb_image.size
        with Image.open(depth_path) as depth_image:
            depth_pixels = np.asarray(depth_image)
    except (OSError, ValueError) as exc:
        raise PerceptionBridgeError(
            PERCEPTION_ERR_DEPTH_DECODE, "perception-bridge RGB-D artifact cannot be decoded"
        ) from exc
    try:
        with Image.open(mask_path) as mask_image:
            mask_pixels = np.asarray(mask_image.convert("L")) > 0
    except (OSError, ValueError) as exc:
        # A malformed mask is a SAM3 output-contract failure, whereas a
        # malformed depth artifact belongs to the observed RGB-D input.
        raise PerceptionBridgeError(
            PERCEPTION_ERR_MASK_DECODE, "SAM3 mask artifact cannot be decoded"
        ) from exc

    if depth_pixels.ndim != 2:
        raise PerceptionBridgeError(PERCEPTION_ERR_DEPTH_DECODE, "depth image is not single-channel")
    if tuple(rgb_size) != (int(depth_pixels.shape[1]), int(depth_pixels.shape[0])):
        raise PerceptionBridgeError(
            PERCEPTION_ERR_RGBD_SHAPE,
            "current RGB and depth artifacts have different dimensions",
        )
    if mask_pixels.shape != depth_pixels.shape:
        raise PerceptionBridgeError(
            PERCEPTION_ERR_MASK_SHAPE,
            "SAM3 mask dimensions do not match the current depth image",
        )

    intrinsics = camera.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise PerceptionBridgeError(ERR_INTRINSICS, "camera intrinsics are missing")
    scale_value = intrinsics.get("scale", camera.get("depth_scale", 1000.0))
    if (
        not isinstance(scale_value, (int, float))
        or isinstance(scale_value, bool)
        or not math.isfinite(float(scale_value))
        or float(scale_value) <= 0.0
    ):
        raise PerceptionBridgeError(PERCEPTION_ERR_DEPTH_SCALE, "depth scale is invalid")

    # The simulator MCP encodes metric depth as a uint16 PNG in fixed units.
    # A numeric array supplied here would be untrusted, so reconstruct only
    # from the current materialized depth artifact.
    depth_m = depth_pixels.astype("float64") / float(scale_value)
    frame = {
        "frame_id": frame_id,
        "depth": depth_m,
        "intrinsics": dict(intrinsics),
        "extrinsics": camera.get("extrinsics"),
    }
    entry = summarize_detection(detection=detection, camera_frame=frame)
    position = entry.get("position")
    if (
        not isinstance(position, list)
        or len(position) != 3
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in position
        )
    ):
        raise PerceptionBridgeError(
            PERCEPTION_ERR_POSITION,
            str(entry.get("position_error") or "world position is unavailable"),
        )
    return {
        "schema_version": "openeta.gazebo.perception_object_summary.v1",
        "objects": [entry],
        "source_frame_id": frame_id,
        "source_image": str(source_image),
        "depth_image": str(depth_path),
    }


def _case_file(value: Any, root: Path, code: str) -> Path:
    """Return an existing artifact below one perception-bridge case root, or fail closed."""

    if not isinstance(value, str) or not value.strip():
        raise PerceptionBridgeError(code, "required artifact path is missing")
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PerceptionBridgeError(PERCEPTION_ERR_CASE_ROOT, "artifact escapes case root") from exc
    if not path.is_file():
        raise PerceptionBridgeError(code, "required artifact is missing")
    return path


def _back_project(
    detection: Mapping[str, Any], camera_frame: Any
) -> tuple[str | None, list[float] | None]:
    """Return ``(error, None)`` or ``(None, [x, y, z])`` for one detection."""

    import numpy as np

    mask, mask_error = _decode_mask(detection)
    if mask_error is not None:
        return mask_error, None

    depth = _frame_field(camera_frame, "depth")
    if depth is None:
        return ERR_DEPTH_MISSING, None
    depth_array = np.asarray(depth, dtype=np.float64)
    if depth_array.ndim != 2 or depth_array.size == 0:
        return ERR_DEPTH_MISSING, None

    if mask.shape != depth_array.shape:
        # The mask was rendered on the RGB image; depth may have a different
        # resolution.  Resample nearest, mirroring the agent-side selection
        # overlay handling in ``agent/tools/handlers.py``.
        mask = _resize_mask(mask, depth_array.shape)
    if not mask.any():
        return ERR_EMPTY_MASK, None

    valid = mask & np.isfinite(depth_array) & (depth_array > 0.0)
    if not valid.any():
        return ERR_NO_VALID_DEPTH, None

    intrinsics = _frame_field(camera_frame, "intrinsics") or {}
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError):
        return ERR_INTRINSICS, None
    if not all(math.isfinite(value) and value > 0 for value in (fx, fy)):
        return ERR_INTRINSICS, None

    extrinsics = _frame_field(camera_frame, "extrinsics") or {}
    extrinsics_error = _validate_extrinsics(extrinsics)
    if extrinsics_error is not None:
        return extrinsics_error, None

    rows, cols = np.nonzero(valid)
    z = float(np.median(depth_array[valid]))
    u = float(cols.mean())
    v = float(rows.mean())
    camera_point = ((u - cx) * z / fx, (v - cy) * z / fy, z)

    cam_pos = tuple(float(value) for value in extrinsics["pos"])
    cam_quat = _normalised(tuple(float(value) for value in extrinsics["quat_xyzw"]))
    rotated = quaternion_rotate(cam_quat, camera_point)
    return None, [
        rotated[0] + cam_pos[0],
        rotated[1] + cam_pos[1],
        rotated[2] + cam_pos[2],
    ]


def _decode_mask(detection: Mapping[str, Any]) -> tuple[Any, str | None]:
    """Decode the detection mask to a boolean array.

    Returns ``(mask, None)`` on success, or ``(None, reason)`` where *reason*
    is ``ERR_MASK_MISSING`` when the detection carries neither ``mask_ref``
    nor an inline ``mask`` payload, and ``ERR_MASK_DECODE`` when a payload
    exists but cannot be decoded.
    """

    import numpy as np
    from PIL import Image

    mask_ref = detection.get("mask_ref")
    inline = detection.get("mask")
    try:
        if mask_ref:
            image = Image.open(str(mask_ref)).convert("L")
        elif isinstance(inline, Mapping) and inline.get("format") == "png" and inline.get("base64"):
            image = Image.open(io.BytesIO(base64.b64decode(str(inline["base64"])))).convert("L")
        else:
            return None, ERR_MASK_MISSING
    except (OSError, ValueError):
        return None, ERR_MASK_DECODE
    return np.asarray(image) > 0, None


def _resize_mask(mask: Any, shape: tuple[int, int]) -> Any:
    import numpy as np
    from PIL import Image

    height, width = shape
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _validate_extrinsics(extrinsics: Mapping[str, Any]) -> str | None:
    """Same acceptance rule as the oracle-fixture oracle: numeric ``camera_to_world``."""

    if extrinsics.get("frame_transform") != "camera_to_world":
        return ERR_EXTRINSICS
    if extrinsics.get("camera_frame") != "opencv":
        return ERR_EXTRINSICS
    if not _is_finite_vector(extrinsics.get("pos"), 3):
        return ERR_EXTRINSICS
    quat = extrinsics.get("quat_xyzw")
    if not _is_finite_vector(quat, 4):
        return ERR_EXTRINSICS
    if math.sqrt(sum(float(value) ** 2 for value in quat)) <= 1e-12:  # type: ignore[union-attr]
        return ERR_EXTRINSICS
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


def _normalised(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion must be non-zero")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _frame_field(camera_frame: Any, name: str) -> Any:
    """Read *name* from a ``CameraFrame`` dataclass or an equivalent mapping."""

    if isinstance(camera_frame, Mapping):
        return camera_frame.get(name)
    return getattr(camera_frame, name, None)


def _detection_id(detection: Mapping[str, Any]) -> str:
    value = detection.get("id") or detection.get("label") or "object"
    return str(value)


def _detection_label(detection: Mapping[str, Any]) -> str:
    value = detection.get("label") or detection.get("id") or "unknown"
    return str(value)


def _detection_confidence(detection: Mapping[str, Any]) -> float | None:
    score = detection.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)):
        return float(score)
    return None
