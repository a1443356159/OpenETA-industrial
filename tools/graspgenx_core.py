"""GraspGenX backend for the OpenETA GraspGenX MCP server."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.candidate_config import (
    DEFAULT_GRASPGENX_RAW_POOL_SIZE,
    DEFAULT_GRASP_RAW_POOL_SIZE,
    raw_pool_size as validate_raw_pool_size,
)


SERVER_NAME = "openeta-graspgenx"
TOOL_NAME = "predict_grasps"
LIST_TOOL_NAME = "list_grippers"
BACKEND_NAME = "graspgenx_mcp"
MODEL_NAME = "graspgenx"
PLANNER = "graspmoe"
FRAME = "camera"
CAMERA_FRAME = "opencv"
GRASP_FRAME = "graspnet"
NATIVE_GRASP_FRAME = "graspgenx"
POSE_CONVENTION = "p_camera = R @ p_grasp + t"

# The industrial scene camera observes the work surface from 1.35 m. A 1 m
# legacy near-field cutoff silently erased every masked target point. Two
# metres covers the calibrated work volume while still excluding the more
# distant simulated floor/background.
DEFAULT_DEPTH_TRUNCATION = 2.0
MIN_OBJECT_POINTS = 100
MAX_RETURNED_CANDIDATES = 20
NUM_GRASPS = 200
# GraspGenX diffusion is intentionally stochastic.  The first four draws are
# the frozen recall base used by the legacy provider pool: one complete
# GraspMoE draw followed by three diffusion-only draws.  Keep those exact
# seeds and select the 200 recall-base representatives from that prefix only.
#
# A low-profile industrial part can otherwise expose a single kinematically
# reachable mode whose model pose misses the parallel-jaw closing midplane.
# Add deterministic, decorrelated *model* draws for the bounded centering and
# physical-support reserve.  The first expansion's OBB output is deliberately
# discarded because the first full draw already supplies the deterministic OBB
# family; the remaining expansion draws are diffusion-only.  No pose is
# modified.  This broadens one immutable inference result for thin industrial
# parts, where a single reachable mode can otherwise make only one-pad contact.
RECALL_BASE_DRAW_SPECS = (
    (0, PLANNER, False),
    (1, "diffusion", False),
    (2, "diffusion", False),
    (3, "diffusion", False),
)
DIVERSITY_EXPANSION_DRAW_SPECS = (
    (12, PLANNER, True),
    (24, "diffusion", False),
    (36, "diffusion", False),
)
MODEL_INFERENCE_DRAW_SPECS = (
    *RECALL_BASE_DRAW_SPECS,
    *DIVERSITY_EXPANSION_DRAW_SPECS,
)
MODEL_INFERENCE_DRAWS = len(MODEL_INFERENCE_DRAW_SPECS)
DEFAULT_INFERENCE_SEED = 4
COLLISION_SURFACE_SEED_OFFSET = 10_000
COLLISION_SCENE_SEED_OFFSET = 20_000
MOE_NUM_YAWS = 36
GEOMETRY_DRIVEN_ANCHOR_SCHEMA = "openeta.graspgenx_geometry_anchors.v1"
MOE_OUTLIER_THRESHOLD = 0.014
MOE_OUTLIER_K = 20
MOE_OBB_MODE = "advanced"
MOE_SKIP_OBB_RULE = "auto"
MOE_OBB_DENSITY = "dense-topandside"
MOE_OBB_POSITION_SPACING_CM = 1.0
# This point-cloud filter is a coarse generator-side screen, not the final
# PlanningScene collision proof.  A 20 mm clearance erased every side/oblique
# mode for low-profile industrial parts (for example a 46 mm hex bolt), leaving
# only top-down poses that can be kinematically unreachable.  Even 5 mm erased
# a measured side grasp with 3.64 mm of positive table clearance.  This sampled
# point-cloud test is only a coarse penetration guard; MoveIt state validity
# and L5 remain the exact collision proof.
COLLISION_THRESHOLD = 0.001
MAX_COLLISION_SCENE_POINTS = 8192
NUM_COLLISION_SAMPLES = 2000
COLLISION_BATCH_SIZE = 16
MMR_TRANSLATION_SCALE_M = 0.03
MMR_ROTATION_SCALE_RAD = math.radians(30.0)
MMR_ROTATION_WEIGHT = 1.0
MMR_WRIST_ROTATION_WEIGHT = 0.35
MMR_SIMILARITY_PENALTY = 0.55
MMR_DIVERSITY_RESERVE_MULTIPLIER = 16
MMR_MIN_SOURCE_COVERAGE = 3
# A model pose is never translated by this signal.  It only prevents an
# off-centre, high-score pose from occupying a bounded provider slot ahead of
# another model-generated pose whose closing midplane already straddles the
# observed target.  Final ordering and all collision/IK/L5 proofs remain in
# the shared qualification funnel.
CENTERING_ALIGNMENT_QUANTILES = (0.02, 0.98)
CENTERING_MIN_SPAN_M = 0.02
CENTERING_MMR_PENALTY = 0.75
CENTERING_PROJECTION_BATCH_SIZE = 64
CENTERING_RISK_RATIO = 0.10
CENTERING_RESERVE_SIZE = 56
CENTERING_VARIANT_MAX_TRANSLATION_M = 0.04
CENTERING_VARIANT_MAX_APPROACH_RAD = math.radians(15.0)
CENTERING_VARIANT_MAX_ROTATION_RAD = math.radians(45.0)
CENTERING_VARIANT_MIN_IMPROVEMENT = 0.05
CENTERING_VARIANT_INSPECTION_MULTIPLIER = 16
TARGET_CLOSING_ALIGNMENT_SCHEMA = (
    "openeta.parallel_gripper_target_closing_alignment.v1"
)
# GraspGenX produces thousands of raw diffusion/OBB samples, so this is the
# model-side representative-pool rule, not the downstream qualification
# funnel's 10 mm / 10 degree scheduling cluster.  Keeping a wider conjunction
# here prevents score-rich near-neighbours from consuming all 200 provider
# slots; every returned representative is still retained and judged by the
# shared funnel.  The three conditions are conjunctive, so a materially
# different wrist rotation or approach remains eligible.
FORMAL_MIN_TRANSLATION_M = 0.015
FORMAL_MIN_APPROACH_SEPARATION_RAD = math.radians(20.0)
FORMAL_MIN_WRIST_ROTATION_RAD = math.radians(30.0)

_DEPTH_SCALE_GUIDANCE = (
    "Depth in meters is raw_depth / intrinsics.scale; for uint16 millimeter depth, use scale=1000."
)

# Columns are the GraspNet basis vectors expressed in the GraspGenX basis:
# GraspNet +X = GraspGenX +Z (approach), GraspNet +Y = GraspGenX +X
# (closing), and GraspNet +Z = GraspGenX +Y.
_NATIVE_FROM_GRASPNET_ROTATION = (
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
)


class GraspGenXInputError(Exception):
    """Input or normalized output cannot satisfy the GraspGenX contract."""

    def __init__(self, reason: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class GripperDescription:
    """Validated public geometry for one inference-ready gripper asset."""

    name: str
    gripper_type: str
    fingertip_xyz: tuple[float, float, float]
    sweep_open_extents_xyz: tuple[float, float, float]
    sweep_open_offset_xyz: tuple[float, float, float]
    sweep_mid_extents_xyz: tuple[float, float, float]
    sweep_mid_offset_xyz: tuple[float, float, float]
    asset_family: str = "x_grippers"

    @property
    def fingertip_depth(self) -> float:
        return self.fingertip_xyz[2]

    @property
    def width(self) -> float:
        return self.sweep_open_extents_xyz[0]

    @property
    def height(self) -> float:
        return self.sweep_open_extents_xyz[1]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gripper_type": self.gripper_type,
            "fingertip_depth": self.fingertip_depth,
            "sweep_volume_open": {
                "extents_xyz": list(self.sweep_open_extents_xyz),
                "offset_xyz": list(self.sweep_open_offset_xyz),
            },
            "sweep_volume_mid": {
                "extents_xyz": list(self.sweep_mid_extents_xyz),
                "offset_xyz": list(self.sweep_mid_offset_xyz),
            },
            "asset_family": self.asset_family,
        }


def scan_gripper_descriptions(
    gripper_descriptions_root: str | Path,
) -> tuple[dict[str, GripperDescription], dict[str, str]]:
    """Scan assets without importing GraspGenX or loading model weights.

    Invalid individual assets are excluded and returned in the diagnostics map.
    A missing asset root or an empty valid set is a service-configuration error.
    """

    root = Path(gripper_descriptions_root).expanduser().resolve()
    assets_root = root / "gripper_descriptions" / "assets" / "x_grippers"
    if not assets_root.is_dir():
        raise ValueError(
            "gripper descriptions root must contain gripper_descriptions/assets/x_grippers"
        )

    valid: dict[str, GripperDescription] = {}
    invalid: dict[str, str] = {}
    for asset_dir in sorted(path for path in assets_root.iterdir() if path.is_dir()):
        if asset_dir.name == "utils":
            continue
        try:
            valid[asset_dir.name] = _parse_gripper_description(asset_dir)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            invalid[asset_dir.name] = str(exc)
    if not valid:
        raise ValueError("no valid inference-ready gripper assets were found")
    return valid, invalid


def _parse_gripper_description(asset_dir: Path) -> GripperDescription:
    config_path = asset_dir / "config.json"
    collision_mesh_path = asset_dir / "coll_mesh.obj"
    if not config_path.is_file():
        raise ValueError("missing config.json")
    if not collision_mesh_path.is_file() or collision_mesh_path.stat().st_size == 0:
        raise ValueError("missing or empty coll_mesh.obj")
    _validate_collision_mesh_obj(collision_mesh_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    gripper_type = str(config["type"])
    if gripper_type not in {"parallel_2f", "revolute_2f", "revolute_3f"}:
        raise ValueError(f"unsupported gripper type: {gripper_type}")
    fingertip = _finite_vector(config["fingertip"], "fingertip")
    if fingertip[2] <= 0:
        raise ValueError("fingertip Z depth must be positive")

    sweep = config["sweep_volume"]
    open_extents = _positive_vector(sweep["extents"], "sweep_volume.extents")
    open_offset = _finite_vector(sweep["offset"], "sweep_volume.offset")
    mid_extents = _positive_vector(sweep["extents2"], "sweep_volume.extents2")
    mid_offset = _finite_vector(sweep["offset2"], "sweep_volume.offset2")
    return GripperDescription(
        name=asset_dir.name,
        gripper_type=gripper_type,
        fingertip_xyz=fingertip,
        sweep_open_extents_xyz=open_extents,
        sweep_open_offset_xyz=open_offset,
        sweep_mid_extents_xyz=mid_extents,
        sweep_mid_offset_xyz=mid_offset,
    )


def _validate_collision_mesh_obj(path: Path) -> None:
    has_vertex = False
    has_face = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            stripped = line.lstrip()
            if line_number == 0 and stripped.startswith("version https://git-lfs.github.com/spec/"):
                raise ValueError("coll_mesh.obj is an unmaterialized Git LFS pointer")
            has_vertex = has_vertex or stripped.startswith("v ")
            has_face = has_face or stripped.startswith("f ")
            if has_vertex and has_face:
                return
    raise ValueError("coll_mesh.obj must contain vertices and faces")


def _finite_vector(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three values")
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite numbers") from exc
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError(f"{label} must contain finite numbers")
    return parsed  # type: ignore[return-value]


def _positive_vector(value: Any, label: str) -> tuple[float, float, float]:
    parsed = _finite_vector(value, label)
    if not all(item > 0 for item in parsed):
        raise ValueError(f"{label} values must be positive")
    return parsed


def validate_checkpoint_layout(checkpoint_root: str | Path) -> tuple[Path, Path]:
    """Return the exact generator/discriminator checkpoints selected at startup."""

    root = Path(checkpoint_root).expanduser().resolve()
    checkpoints: list[Path] = []
    for component in ("gen", "dis"):
        component_dir = root / component
        if not component_dir.is_dir() or not (component_dir / "config.yaml").is_file():
            raise ValueError(f"checkpoint root must contain {component}/config.yaml")
        candidates = list(component_dir.glob("*.pth"))
        if not candidates:
            raise ValueError(f"checkpoint root must contain {component}/*.pth")
        checkpoints.append(max(candidates, key=_checkpoint_sort_key))
    return checkpoints[0], checkpoints[1]


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch_(\d+)", path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def validate_intrinsics(intrinsics: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(intrinsics, dict):
        raise GraspGenXInputError("missing_intrinsics")
    parsed: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy"):
        if key not in intrinsics:
            raise GraspGenXInputError("invalid_intrinsics")
        try:
            value = float(intrinsics[key])
        except (TypeError, ValueError) as exc:
            raise GraspGenXInputError("invalid_intrinsics") from exc
        if not math.isfinite(value) or (key in {"fx", "fy"} and value <= 0):
            raise GraspGenXInputError("invalid_intrinsics")
        parsed[key] = value
    if "scale" not in intrinsics:
        raise GraspGenXInputError("invalid_depth_scale")
    try:
        scale = float(intrinsics["scale"])
    except (TypeError, ValueError) as exc:
        raise GraspGenXInputError("invalid_depth_scale") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise GraspGenXInputError("invalid_depth_scale")
    parsed["scale"] = scale
    return parsed


def validate_up_direction(up_direction_camera: Any) -> Any:
    np, _Image = _load_image_dependencies()
    try:
        vector = np.asarray(up_direction_camera, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GraspGenXInputError("invalid_up_direction") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise GraspGenXInputError("invalid_up_direction")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise GraspGenXInputError("invalid_up_direction")
    return vector / norm


def decode_image_payload(
    payload: dict[str, Any] | None,
    *,
    Image: Any,
    np: Any,
    convert: str | None,
    missing_reason: str,
    decode_reason: str,
) -> Any:
    """Decode Pillow-supported bytes; the advisory format field is not trusted."""

    if not isinstance(payload, dict) or not payload.get("base64"):
        raise GraspGenXInputError(missing_reason)
    try:
        data = base64.b64decode(payload["base64"], validate=True)
        image = Image.open(io.BytesIO(data))
        source_format = (image.format or "").upper()
        source_mode = image.mode
        if convert is not None:
            image = image.convert(convert)
        array = np.asarray(image)
        if (
            convert is None
            and source_format == "PNG"
            and source_mode.startswith("I")
            and array.dtype.kind == "i"
        ):
            array = array.astype(np.uint16)
        return array
    except GraspGenXInputError:
        raise
    except Exception as exc:  # noqa: BLE001 - Pillow raises heterogeneous errors.
        raise GraspGenXInputError(decode_reason) from exc


def build_targeted_point_clouds(
    *,
    depth_array: Any,
    object_mask_array: Any,
    intrinsics: dict[str, float],
    depth_truncation: float = DEFAULT_DEPTH_TRUNCATION,
    min_object_points: int = MIN_OBJECT_POINTS,
) -> tuple[Any, Any, dict[str, Any]]:
    """Backproject aligned depth/mask into OpenCV camera-frame point clouds."""

    np, _Image = _load_image_dependencies()
    depth = np.asarray(depth_array)
    mask = np.asarray(object_mask_array)
    if depth.ndim != 2 or mask.ndim != 2 or depth.shape != mask.shape:
        raise GraspGenXInputError("image_shape_mismatch")
    if depth.dtype.kind not in {"u", "i", "f"}:
        raise GraspGenXInputError("depth_decode_failed")
    object_mask = mask > 0
    if not bool(object_mask.any()):
        raise GraspGenXInputError("empty_object_mask")

    depth_float = depth.astype(np.float64, copy=False)
    finite_raw = np.isfinite(depth_float)
    finite_values = depth_float[finite_raw]
    raw_min = float(finite_values.min()) if finite_values.size else None
    raw_max = float(finite_values.max()) if finite_values.size else None
    metric_depth = depth_float / float(intrinsics["scale"])
    valid = finite_raw & (metric_depth > 0.0) & (metric_depth < float(depth_truncation))
    metadata = {
        "depth_dtype": str(depth.dtype),
        "depth_raw_min": raw_min,
        "depth_raw_max": raw_max,
        "depth_metric_min": (None if raw_min is None else raw_min / float(intrinsics["scale"])),
        "depth_metric_max": (None if raw_max is None else raw_max / float(intrinsics["scale"])),
        "depth_truncation": float(depth_truncation),
        "valid_point_count": int(valid.sum()),
    }
    is_uint16 = depth.dtype.kind == "u" and depth.dtype.itemsize == 2
    if is_uint16 and raw_max is not None and raw_max > 10 and intrinsics["scale"] <= 1:
        raise GraspGenXInputError("depth_scale_mismatch", metadata=metadata)
    if metadata["valid_point_count"] == 0:
        raise GraspGenXInputError("empty_point_cloud_after_depth_filter", metadata=metadata)

    object_valid = valid & object_mask
    object_count = int(object_valid.sum())
    scene_valid = valid & ~object_mask
    metadata.update(
        {
            "object_point_count": object_count,
            "scene_point_count": int(scene_valid.sum()),
            "model_input_point_count": object_count,
        }
    )
    if object_count < int(min_object_points):
        raise GraspGenXInputError("insufficient_object_points", metadata=metadata)

    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    points_x = (u - intrinsics["cx"]) / intrinsics["fx"] * metric_depth
    points_y = (v - intrinsics["cy"]) / intrinsics["fy"] * metric_depth
    organized = np.stack([points_x, points_y, metric_depth], axis=-1)
    object_points = np.ascontiguousarray(organized[object_valid], dtype=np.float32)
    scene_points = np.ascontiguousarray(organized[scene_valid], dtype=np.float32)
    return object_points, scene_points, metadata


def geometry_driven_moe_z_offsets_cm(
    *,
    object_points_aligned: Any,
    scene_points_aligned: Any,
    depth_scale: float,
    gripper: GripperDescription,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Derive OBB contact anchors from observed support and gripper sweep.

    The three physical anchors are the observed object's minor-axis midplane,
    its visible surface, and the extra approach-axis reach of the articulated
    gripper while it closes.  No task label, scene ID, candidate index, or
    sample-specific distance enters this computation.
    """

    np, _Image = _load_image_dependencies()
    object_points = np.asarray(object_points_aligned, dtype=np.float64)
    scene_points = np.asarray(scene_points_aligned, dtype=np.float64)
    if (
        object_points.ndim != 2
        or object_points.shape[1:] != (3,)
        or not len(object_points)
        or not np.isfinite(object_points).all()
        or scene_points.ndim != 2
        or scene_points.shape[1:] != (3,)
        or not np.isfinite(scene_points).all()
        or not math.isfinite(float(depth_scale))
        or float(depth_scale) <= 0.0
    ):
        raise GraspGenXInputError("invalid_geometry_anchor_input")

    low_quantile, high_quantile = CENTERING_ALIGNMENT_QUANTILES
    object_low, object_high = np.quantile(
        object_points,
        [low_quantile, high_quantile],
        axis=0,
    )
    object_extents = np.maximum(object_high - object_low, 0.0)
    depth_quantum_m = 1.0 / float(depth_scale)

    # Search only the physical neighbourhood reachable by the open gripper.
    # The dominant horizontal depth level there is the support plane.  Binning
    # at two sensor quanta absorbs adjacent rasterized depth levels without
    # encoding a workcell-specific height.
    support_margin_m = max(gripper.sweep_open_extents_xyz[:2]) * 0.5
    local_scene_mask = (
        (scene_points[:, 0] >= object_low[0] - support_margin_m)
        & (scene_points[:, 0] <= object_high[0] + support_margin_m)
        & (scene_points[:, 1] >= object_low[1] - support_margin_m)
        & (scene_points[:, 1] <= object_high[1] + support_margin_m)
        & (scene_points[:, 2] <= object_high[2] - depth_quantum_m)
    )
    local_support_points = scene_points[local_scene_mask]
    support_height_m: float | None = None
    support_mode_count = 0
    if len(local_support_points):
        bin_width_m = 2.0 * depth_quantum_m
        levels = np.rint(local_support_points[:, 2] / bin_width_m).astype(np.int64)
        unique, counts = np.unique(levels, return_counts=True)
        maximum_count = int(counts.max())
        # On an exact tie, the highest plane below the object is the immediate
        # support rather than a lower shelf or floor visible in the same ROI.
        winning_level = int(unique[counts == maximum_count].max())
        support_values = local_support_points[levels == winning_level, 2]
        support_height_m = float(np.median(support_values))
        support_mode_count = len(support_values)

    observed_height_m = (
        max(0.0, float(object_high[2]) - support_height_m)
        if support_height_m is not None
        else float(object_extents[2])
    )
    positive_extents = [
        float(value)
        for value in (*object_extents[:2], observed_height_m)
        if float(value) > depth_quantum_m
    ]
    minor_extent_m = min(positive_extents) if positive_extents else depth_quantum_m
    midplane_penetration_m = -0.5 * minor_extent_m

    open_leading_m = (
        gripper.sweep_open_offset_xyz[2]
        + 0.5 * gripper.sweep_open_extents_xyz[2]
    )
    mid_leading_m = (
        gripper.sweep_mid_offset_xyz[2]
        + 0.5 * gripper.sweep_mid_extents_xyz[2]
    )
    closing_sweep_advance_m = max(0.0, mid_leading_m - open_leading_m)

    anchors_m: list[float] = []
    for value in (midplane_penetration_m, 0.0, closing_sweep_advance_m):
        if not any(abs(value - existing) <= depth_quantum_m for existing in anchors_m):
            anchors_m.append(value)
    offsets_cm = tuple(value * 100.0 for value in anchors_m)
    return offsets_cm, {
        "schema_version": GEOMETRY_DRIVEN_ANCHOR_SCHEMA,
        "source": "aligned_target_depth_support_plane_and_gripper_sweep",
        "object_robust_extents_m": _float_list(object_extents),
        "support_height_aligned_m": support_height_m,
        "support_local_point_count": int(len(local_support_points)),
        "support_mode_point_count": support_mode_count,
        "depth_quantum_m": depth_quantum_m,
        "minor_extent_m": minor_extent_m,
        "midplane_penetration_m": midplane_penetration_m,
        "closing_sweep_advance_m": closing_sweep_advance_m,
        "moe_z_offsets_cm": list(offsets_cm),
        "pose_modified": False,
    }


def rotation_aligning_up_to_z(up_direction_camera: Any) -> Any:
    """Return R such that p_aligned=R@p_camera and R@up_camera=[0,0,1]."""

    np, _Image = _load_image_dependencies()
    source = validate_up_direction(up_direction_camera)
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if cosine < -1.0 + 1e-12:
        # Any unit axis orthogonal to source is valid for the 180-degree case.
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(source[0])) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))


def validate_raw_grasp_outputs(
    grasps: Any, scores: Any, branch_tags: Any
) -> tuple[Any, Any, list[str]]:
    np, _Image = _load_image_dependencies()
    try:
        grasp_array = np.asarray(grasps, dtype=np.float64)
        score_array = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GraspGenXInputError("inconsistent_grasp_outputs") from exc
    if grasp_array.size == 0:
        raise GraspGenXInputError("no_grasp_candidates")
    if grasp_array.ndim != 3 or grasp_array.shape[1:] != (4, 4):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    count = int(grasp_array.shape[0])
    if score_array.shape != (count,) or not isinstance(branch_tags, (list, tuple)):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    tags = [str(tag) for tag in branch_tags]
    if len(tags) != count or any(tag not in {"diff", "obb"} for tag in tags):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    if not np.isfinite(grasp_array).all() or not np.isfinite(score_array).all():
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    if not all(_is_rigid_transform(matrix) for matrix in grasp_array):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    return grasp_array, score_array, tags


def transform_grasps_to_camera(grasps_aligned: Any, alignment_rotation: Any) -> Any:
    np, _Image = _load_image_dependencies()
    camera_from_aligned = np.eye(4, dtype=np.float64)
    camera_from_aligned[:3, :3] = np.asarray(alignment_rotation).T
    camera_grasps = np.einsum(
        "ij,kjl->kil", camera_from_aligned, np.asarray(grasps_aligned, dtype=np.float64)
    )
    if not all(_is_rigid_transform(matrix) for matrix in camera_grasps):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    return camera_grasps


def normalise_grasp_candidates(
    *,
    camera_native_grasps: Any,
    scores: Any,
    branch_tags: list[str],
    selected_indices: list[int],
    gripper: GripperDescription,
    centering_corrections: Any | None = None,
    centering_spans: Any | None = None,
    centering_ratios: Any | None = None,
    centering_variant_parents: dict[int, list[int]] | None = None,
) -> list[dict[str, Any]]:
    """Build dual-frame candidates in already-ranked selected order."""

    np, _Image = _load_image_dependencies()
    native_grasps = np.asarray(camera_native_grasps, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    native_from_graspnet = np.eye(4, dtype=np.float64)
    native_from_graspnet[:3, :3] = np.asarray(_NATIVE_FROM_GRASPNET_ROTATION, dtype=np.float64)
    tip_native = np.asarray(gripper.fingertip_xyz, dtype=np.float64)
    corrections = (
        None
        if centering_corrections is None
        else np.asarray(centering_corrections, dtype=np.float64)
    )
    spans = (
        None if centering_spans is None else np.asarray(centering_spans, dtype=np.float64)
    )
    ratios = (
        None
        if centering_ratios is None
        else np.asarray(centering_ratios, dtype=np.float64)
    )
    if any(
        values is not None and values.shape != (len(native_grasps),)
        for values in (corrections, spans, ratios)
    ):
        raise GraspGenXInputError("inconsistent_grasp_outputs")

    candidates: list[dict[str, Any]] = []
    for rank, backend_index in enumerate(selected_indices):
        native_pose = native_grasps[backend_index]
        graspnet_pose = native_pose @ native_from_graspnet
        tip_camera = native_pose[:3, :3] @ tip_native + native_pose[:3, 3]
        if not _is_rigid_transform(graspnet_pose) or not np.isfinite(tip_camera).all():
            raise GraspGenXInputError("inconsistent_grasp_outputs")
        candidate = {
            "id": f"graspgenx_{rank:03d}",
            "source_model": MODEL_NAME,
            "gripper_name": gripper.name,
            "candidate_source": ("diffusion" if branch_tags[backend_index] == "diff" else "obb"),
            "frame": FRAME,
            "camera_frame": CAMERA_FRAME,
            "grasp_frame": GRASP_FRAME,
            "convention": POSE_CONVENTION,
            "score": float(score_array[backend_index]),
            "translation_xyz": _float_list(graspnet_pose[:3, 3]),
            "rotation_matrix": _float_matrix(graspnet_pose[:3, :3]),
            "transform_matrix": _float_matrix(graspnet_pose),
            "gripper_tip_position_xyz": _float_list(tip_camera),
            "depth": float(gripper.fingertip_depth),
            "width": float(gripper.width),
            "height": float(gripper.height),
            "rank": rank,
            "backend_index": int(backend_index),
            "model_native_grasp_pose": {
                "frame": FRAME,
                "camera_frame": CAMERA_FRAME,
                "grasp_frame": NATIVE_GRASP_FRAME,
                "convention": POSE_CONVENTION,
                "transform_matrix": _float_matrix(native_pose),
            },
        }
        if corrections is not None and spans is not None and ratios is not None:
            closing_axis = graspnet_pose[:3, 1]
            correction = float(corrections[backend_index])
            candidate["target_closing_alignment"] = {
                "schema_version": TARGET_CLOSING_ALIGNMENT_SCHEMA,
                "source": "aligned_selected_mask_depth",
                "depth_provenance": "sensor_depth",
                "closing_axis": "graspnet_local_y",
                "quantile_bounds": list(CENTERING_ALIGNMENT_QUANTILES),
                "target_span_m": float(spans[backend_index]),
                "correction_m": correction,
                "correction_camera_xyz": _float_list(closing_axis * correction),
                "centering_ratio": float(ratios[backend_index]),
                "ordering_only": True,
                "pose_modified": False,
            }
            parent_indices = (centering_variant_parents or {}).get(backend_index, [])
            if parent_indices:
                candidate["target_closing_alignment"].update(
                    {
                        "variant_role": "same_approach_centering_reserve",
                        "compatible_parent_backend_indices": [
                            int(parent_index) for parent_index in parent_indices
                        ],
                    }
                )
        candidates.append(candidate)
    return candidates


def _parallel_gripper_centering_metrics(
    *,
    camera_native_grasps: Any,
    object_points_camera: Any,
    gripper: GripperDescription,
) -> tuple[Any, Any, Any]:
    """Measure closing-midplane error without changing any model pose."""

    np, _Image = _load_image_dependencies()
    poses = np.asarray(camera_native_grasps, dtype=np.float64)
    points = np.asarray(object_points_camera, dtype=np.float64)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or points.ndim != 2
        or points.shape[1:] != (3,)
        or not len(points)
        or not np.isfinite(poses).all()
        or not np.isfinite(points).all()
    ):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    # GraspNet local +Y is the parallel-jaw closing axis.  Under
    # _NATIVE_FROM_GRASPNET_ROTATION it is GraspGenX native +X.
    closing_axes = poses[:, :3, 0]
    tip_native = np.asarray(gripper.fingertip_xyz, dtype=np.float64)
    tips_camera = (
        np.einsum("nij,j->ni", poses[:, :3, :3], tip_native)
        + poses[:, :3, 3]
    )
    tip_projections = np.einsum("ni,ni->n", tips_camera, closing_axes)
    corrections = np.empty(len(poses), dtype=np.float64)
    spans = np.empty(len(poses), dtype=np.float64)
    lower_quantile, upper_quantile = CENTERING_ALIGNMENT_QUANTILES
    for start in range(0, len(poses), CENTERING_PROJECTION_BATCH_SIZE):
        stop = min(len(poses), start + CENTERING_PROJECTION_BATCH_SIZE)
        projections = points @ closing_axes[start:stop].T
        low, high = np.quantile(
            projections,
            [lower_quantile, upper_quantile],
            axis=0,
        )
        spans[start:stop] = high - low
        corrections[start:stop] = (
            (low + high) * 0.5 - tip_projections[start:stop]
        )
    ratios = np.abs(corrections) / np.maximum(spans, CENTERING_MIN_SPAN_M)
    if not all(np.isfinite(value).all() for value in (corrections, spans, ratios)):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    return corrections, spans, ratios


def _se3_mmr_order(
    *,
    poses: Any,
    scores: Any,
    branch_tags: list[str],
    selection_limit: int,
    centering_ratios: Any | None = None,
) -> list[int]:
    """Return a deterministic, source-aware quality/diversity ordering."""

    np, _Image = _load_image_dependencies()
    pose_array = np.asarray(poses, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    count = len(score_array)
    if pose_array.shape != (count, 4, 4) or len(branch_tags) != count:
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    ranked = sorted(range(count), key=lambda index: (-score_array[index], index))
    if not ranked:
        return []
    selection_limit = max(1, min(int(selection_limit), count))
    score_span = float(score_array.max() - score_array.min())
    quality = (
        np.ones(count, dtype=np.float64)
        if score_span <= 1e-12
        else (score_array - score_array.min()) / score_span
    )
    if centering_ratios is not None:
        ratios = np.asarray(centering_ratios, dtype=np.float64)
        if ratios.shape != (count,) or not np.isfinite(ratios).all():
            raise GraspGenXInputError("inconsistent_grasp_outputs")
        quality = quality - CENTERING_MMR_PENALTY * np.minimum(ratios, 1.0)

    translations = pose_array[:, :3, 3]
    approach_axes = pose_array[:, :3, 2]
    rotations = pose_array[:, :3, :3]

    def similarity_to(reference: int) -> Any:
        """Vectorized similarity from one pose to the complete raw pool.

        Elongated and thin objects can produce several thousand deterministic
        OBB grasps.  The old scalar implementation evaluated the same SE(3)
        formula in Python for every pair, turning an otherwise sub-second
        model draw into a roughly quadratic, 100-second ranking step.  Keeping
        one running maximum per candidate preserves the exact greedy MMR
        semantics while moving the O(N*K) arithmetic into NumPy.
        """

        translation = (
            np.linalg.norm(translations - translations[reference], axis=1)
            / MMR_TRANSLATION_SCALE_M
        )
        # GraspGenX native +Z is the approach axis.  A parallel-jaw gripper's
        # yaw about that axis is a weaker diversity signal than a genuinely
        # different top/side/oblique approach direction.
        approach_cosine = np.clip(
            approach_axes @ approach_axes[reference],
            -1.0,
            1.0,
        )
        approach = np.arccos(approach_cosine) / MMR_ROTATION_SCALE_RAD
        # trace(R_reference.T @ R_candidate) is the elementwise inner product.
        relative_trace = np.einsum(
            "nij,ij->n",
            rotations,
            rotations[reference],
        )
        wrist_rotation = (
            np.arccos(np.clip((relative_trace - 1.0) * 0.5, -1.0, 1.0))
            / MMR_ROTATION_SCALE_RAD
        )
        return np.exp(
            -(
                translation
                + MMR_ROTATION_WEIGHT * approach
                + MMR_WRIST_ROTATION_WEIGHT * wrist_rotation
            )
        )

    def deterministic_best(indices: Any, objective: Any) -> int:
        """Match ``max((objective, score, -index))`` without a Python scan."""

        candidate_indices = np.asarray(indices, dtype=np.int64)
        objective_values = np.asarray(objective, dtype=np.float64)[candidate_indices]
        best_objective = objective_values.max()
        tied = candidate_indices[objective_values == best_objective]
        if len(tied) > 1:
            tied_scores = score_array[tied]
            tied = tied[tied_scores == tied_scores.max()]
        return int(tied.min())

    # Seed each model source with its own quality/diversity modes before the
    # global MMR pass.  This is a floor, not a fixed OBB/diffusion quota: a
    # source that has no novel modes naturally contributes fewer candidates.
    selected: list[int] = []
    sources = sorted(
        set(branch_tags),
        key=lambda source: next(
            rank for rank, index in enumerate(ranked) if branch_tags[index] == source
        ),
    )
    source_limit = min(MMR_MIN_SOURCE_COVERAGE, max(1, selection_limit // len(sources)))
    for source in sources:
        pool = np.asarray(
            [index for index in ranked if branch_tags[index] == source],
            dtype=np.int64,
        )
        local: list[int] = []
        local_available = np.ones(len(pool), dtype=bool)
        local_max_similarity = np.zeros(count, dtype=np.float64)
        while bool(local_available.any()) and len(local) < source_limit:
            available = pool[local_available]
            if not local:
                best = int(available[0])
            else:
                objective = quality - MMR_SIMILARITY_PENALTY * local_max_similarity
                best = deterministic_best(available, objective)
            local.append(best)
            local_available[np.flatnonzero(pool == best)[0]] = False
            if bool(local_available.any()):
                local_max_similarity = np.maximum(
                    local_max_similarity,
                    similarity_to(best),
                )
        selected.extend(local)
    selected = selected[:selection_limit]
    remaining = np.ones(count, dtype=bool)
    remaining[selected] = False
    max_similarity = np.zeros(count, dtype=np.float64)
    for chosen in selected:
        max_similarity = np.maximum(max_similarity, similarity_to(chosen))
    while bool(remaining.any()) and len(selected) < selection_limit:
        available = np.flatnonzero(remaining)
        objective = quality - MMR_SIMILARITY_PENALTY * max_similarity
        best = deterministic_best(available, objective)
        selected.append(best)
        remaining[best] = False
        if bool(remaining.any()):
            max_similarity = np.maximum(max_similarity, similarity_to(best))
    return [*selected, *(index for index in ranked if remaining[index])]


def _is_formally_novel_grasp(
    *, poses: Any, candidate_index: int, selected_indices: list[int]
) -> bool:
    """Whether a pose is sufficiently distinct for the formal candidate pool."""

    np, _Image = _load_image_dependencies()
    candidate = np.asarray(poses[candidate_index], dtype=np.float64)
    for selected_index in selected_indices:
        selected = np.asarray(poses[selected_index], dtype=np.float64)
        translation = float(np.linalg.norm(candidate[:3, 3] - selected[:3, 3]))
        cosine = float(np.clip(np.dot(candidate[:3, 2], selected[:3, 2]), -1.0, 1.0))
        approach_separation = math.acos(cosine)
        relative_trace = float(np.trace(candidate[:3, :3].T @ selected[:3, :3]))
        wrist_rotation_separation = math.acos(
            float(np.clip((relative_trace - 1.0) * 0.5, -1.0, 1.0))
        )
        if (
            translation < FORMAL_MIN_TRANSLATION_M
            and approach_separation < FORMAL_MIN_APPROACH_SEPARATION_RAD
            and wrist_rotation_separation < FORMAL_MIN_WRIST_ROTATION_RAD
        ):
            return False
    return True


def _centering_variant_order(
    *,
    poses: Any,
    scores: Any,
    centering_ratios: Any,
    recall_base_indices: list[int],
) -> list[int]:
    """Order model-native variants that improve a risky recall-base mode.

    The legacy representative remains in the returned pool.  A variant is
    eligible only when its unchanged model pose belongs to a nearby approach
    mode and its observed closing-midplane error is materially smaller.  This
    is an ordering operation; collision filtering and every host-side proof
    still run afterwards.
    """

    np, _Image = _load_image_dependencies()
    pose_array = np.asarray(poses, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    ratio_array = np.asarray(centering_ratios, dtype=np.float64)
    count = len(score_array)
    if (
        pose_array.shape != (count, 4, 4)
        or ratio_array.shape != (count,)
        or not np.isfinite(ratio_array).all()
    ):
        raise GraspGenXInputError("inconsistent_grasp_outputs")
    base = np.asarray(recall_base_indices, dtype=np.int64)
    if not len(base):
        return []
    risky = base[ratio_array[base] > CENTERING_RISK_RATIO]
    if not len(risky):
        return []

    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[base] = True
    eligible = np.flatnonzero(
        (~selected_mask) & (ratio_array <= CENTERING_RISK_RATIO)
    )
    if not len(eligible):
        return []

    base_poses = pose_array[risky]
    rows: list[tuple[tuple[float, ...], int]] = []
    for candidate_index in eligible:
        candidate = pose_array[candidate_index]
        translation = np.linalg.norm(
            base_poses[:, :3, 3] - candidate[:3, 3], axis=1
        )
        approach = np.arccos(
            np.clip(base_poses[:, :3, 2] @ candidate[:3, 2], -1.0, 1.0)
        )
        relative_trace = np.einsum(
            "nij,ij->n", base_poses[:, :3, :3], candidate[:3, :3]
        )
        rotation = np.arccos(
            np.clip((relative_trace - 1.0) * 0.5, -1.0, 1.0)
        )
        improvement = ratio_array[risky] - ratio_array[candidate_index]
        compatible = (
            (translation <= CENTERING_VARIANT_MAX_TRANSLATION_M)
            & (approach <= CENTERING_VARIANT_MAX_APPROACH_RAD)
            & (rotation <= CENTERING_VARIANT_MAX_ROTATION_RAD)
            & (improvement >= CENTERING_VARIANT_MIN_IMPROVEMENT)
        )
        if not bool(compatible.any()):
            continue
        parents = np.flatnonzero(compatible)
        parent = int(
            min(
                parents,
                key=lambda index: (
                    -float(improvement[index]),
                    float(translation[index]),
                    float(approach[index]),
                    float(rotation[index]),
                    int(risky[index]),
                ),
            )
        )
        key = (
            -float(improvement[parent]),
            float(ratio_array[candidate_index]),
            -float(score_array[candidate_index]),
            float(translation[parent]),
            float(approach[parent]),
            float(rotation[parent]),
            float(candidate_index),
        )
        rows.append((key, int(candidate_index)))
    return [candidate_index for _key, candidate_index in sorted(rows)]


def _compatible_centering_parent_indices(
    *,
    poses: Any,
    centering_ratios: Any,
    candidate_index: int,
    recall_base_indices: list[int],
) -> list[int]:
    """Return every risky recall-base mode improved by one reserve pose."""

    np, _Image = _load_image_dependencies()
    pose_array = np.asarray(poses, dtype=np.float64)
    ratio_array = np.asarray(centering_ratios, dtype=np.float64)
    candidate = pose_array[candidate_index]
    parents: list[tuple[tuple[float, ...], int]] = []
    for parent_index in recall_base_indices:
        improvement = float(
            ratio_array[parent_index] - ratio_array[candidate_index]
        )
        if (
            ratio_array[parent_index] <= CENTERING_RISK_RATIO
            or ratio_array[candidate_index] > CENTERING_RISK_RATIO
            or improvement < CENTERING_VARIANT_MIN_IMPROVEMENT
        ):
            continue
        parent = pose_array[parent_index]
        translation = float(
            np.linalg.norm(parent[:3, 3] - candidate[:3, 3])
        )
        approach = math.acos(
            float(np.clip(np.dot(parent[:3, 2], candidate[:3, 2]), -1.0, 1.0))
        )
        relative_trace = float(np.trace(parent[:3, :3].T @ candidate[:3, :3]))
        rotation = math.acos(
            float(np.clip((relative_trace - 1.0) * 0.5, -1.0, 1.0))
        )
        if (
            translation <= CENTERING_VARIANT_MAX_TRANSLATION_M
            and approach <= CENTERING_VARIANT_MAX_APPROACH_RAD
            and rotation <= CENTERING_VARIANT_MAX_ROTATION_RAD
        ):
            parents.append(
                (
                    (
                        -improvement,
                        translation,
                        approach,
                        rotation,
                        float(parent_index),
                    ),
                    int(parent_index),
                )
            )
    return [parent_index for _key, parent_index in sorted(parents)]


class GraspGenXBackend:
    """Lazy, in-process wrapper around the official GraspGenX Python API."""

    def __init__(
        self,
        *,
        graspgenx_root: str | Path,
        checkpoint_root: str | Path,
        gripper_descriptions_root: str | Path,
        device: str = "cuda:0",
        depth_truncation: float = DEFAULT_DEPTH_TRUNCATION,
        raw_pool_size: int = DEFAULT_GRASPGENX_RAW_POOL_SIZE,
        inference_seed: int = DEFAULT_INFERENCE_SEED,
    ) -> None:
        self.graspgenx_root = Path(graspgenx_root).expanduser().resolve()
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve()
        self.gripper_descriptions_root = Path(gripper_descriptions_root).expanduser().resolve()
        self.device = validate_cuda_device_name(device)
        self.depth_truncation = float(depth_truncation)
        self.raw_pool_size = validate_raw_pool_size(raw_pool_size)
        self.inference_seed = validate_inference_seed(inference_seed)
        self.last_returned_candidate_count = 0
        self.generator_checkpoint, self.discriminator_checkpoint = validate_checkpoint_layout(
            self.checkpoint_root
        )
        self.grippers, self.invalid_grippers = scan_gripper_descriptions(
            self.gripper_descriptions_root
        )
        self.assets_root = self.gripper_descriptions_root / "gripper_descriptions" / "assets"
        self._loaded: dict[str, Any] | None = None
        self._samplers: dict[str, dict[str, Any]] = {}
        # Official GraspGenX consumes process-global Python, NumPy, and torch
        # RNGs. Serialize seeded inference so concurrent MCP requests cannot
        # interleave those states or contend for the same GPU model.
        self._inference_lock = threading.RLock()

    @property
    def model_loaded(self) -> bool:
        return self._loaded is not None

    def list_grippers(self) -> dict[str, Any]:
        descriptions = [self.grippers[name].to_public_dict() for name in sorted(self.grippers)]
        return {
            "success": True,
            "content": "GraspGenX gripper listing completed.",
            "details": {
                "tool": LIST_TOOL_NAME,
                "gripper_count": len(descriptions),
                "grippers": descriptions,
                "model_loaded": self.model_loaded,
            },
        }

    def predict_grasps(
        self,
        *,
        depth: dict[str, Any] | None,
        object_mask: dict[str, Any] | None,
        intrinsics: dict[str, Any] | None,
        gripper_name: str,
        up_direction_camera: Any,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        metadata = self._metadata_base(gripper_name=gripper_name)

        try:
            gripper = self._validate_gripper_name(gripper_name)
            parsed_intrinsics = validate_intrinsics(intrinsics)
            normalized_up = validate_up_direction(up_direction_camera)
            metadata["intrinsics"] = parsed_intrinsics
            metadata["up_direction_camera_normalized"] = _float_list(normalized_up)
            np, Image = _load_image_dependencies()
            depth_array = decode_image_payload(
                depth,
                Image=Image,
                np=np,
                convert=None,
                missing_reason="missing_depth",
                decode_reason="depth_decode_failed",
            )
            mask_array = decode_image_payload(
                object_mask,
                Image=Image,
                np=np,
                convert="L",
                missing_reason="missing_object_mask",
                decode_reason="object_mask_decode_failed",
            )
            object_points, scene_points, point_metadata = build_targeted_point_clouds(
                depth_array=depth_array,
                object_mask_array=mask_array,
                intrinsics=parsed_intrinsics,
                depth_truncation=self.depth_truncation,
            )
            metadata.update(point_metadata)
            alignment = rotation_aligning_up_to_z(normalized_up)
            object_points_aligned = np.ascontiguousarray(
                object_points @ alignment.T, dtype=np.float32
            )
            scene_points_aligned = np.ascontiguousarray(
                scene_points @ alignment.T, dtype=np.float32
            )
            moe_z_offsets_cm, geometry_anchor_metadata = (
                geometry_driven_moe_z_offsets_cm(
                    object_points_aligned=object_points_aligned,
                    scene_points_aligned=scene_points_aligned,
                    depth_scale=parsed_intrinsics["scale"],
                    gripper=gripper,
                )
            )
            metadata["geometry_driven_anchors"] = geometry_anchor_metadata
        except GraspGenXInputError as exc:
            metadata.update(exc.metadata)
            return failure_result(
                reason=exc.reason,
                metadata=_with_duration(metadata, start),
            )

        try:
            loaded = self._get_loaded_backend()
            sampler_entry = self._get_sampler_entry(loaded, gripper.name)
            metadata.update(self._model_metadata(loaded))
        except GraspGenXInputError as exc:
            metadata.update(exc.metadata)
            return failure_result(
                reason=exc.reason,
                metadata=_with_duration(metadata, start),
            )
        except Exception as exc:  # noqa: BLE001 - third-party load boundary.
            return failure_result(
                reason="model_load_failed",
                metadata=_with_duration({**metadata, "error_type": type(exc).__name__}, start),
            )

        try:
            planner_outputs = []
            # Test/lightweight backends do not expose the real torch runtime
            # and retain their single-call contract.  A loaded production
            # backend unions a frozen recall prefix and one decorrelated
            # model-native diffusion expansion.  The expansion may call the
            # complete planner but retains only its diffusion output so the
            # deterministic OBB family is not duplicated.
            draw_specs = (
                MODEL_INFERENCE_DRAW_SPECS
                if loaded.get("torch") is not None
                else ((0, PLANNER, False),)
            )
            draw_seeds = [self.inference_seed + offset for offset, _planner, _ in draw_specs]
            planners = [planner for _offset, planner, _ in draw_specs]
            discarded_expansion_obb_count = 0
            recall_base_raw_candidate_count = 0
            with self._inference_lock:
                for draw_index, ((_, planner, diffusion_only), draw_seed) in enumerate(
                    zip(draw_specs, draw_seeds)
                ):
                    self._seed_inference_rng(loaded=loaded, seed=draw_seed)
                    grasps, scores, tags = self._run_planner(
                        loaded=loaded,
                        sampler_entry=sampler_entry,
                        object_points_aligned=object_points_aligned,
                        moe_z_offsets_cm=moe_z_offsets_cm,
                        planner=planner,
                    )
                    if diffusion_only:
                        np, _Image = _load_image_dependencies()
                        retained = np.asarray(
                            [tag == "diff" for tag in tags], dtype=bool
                        )
                        discarded_expansion_obb_count += int((~retained).sum())
                        grasps = np.asarray(grasps)[retained]
                        scores = np.asarray(scores)[retained]
                        tags = [
                            str(tag)
                            for tag, keep in zip(tags, retained)
                            if bool(keep)
                        ]
                    planner_outputs.append((grasps, scores, tags))
                    if draw_index + 1 == len(RECALL_BASE_DRAW_SPECS):
                        recall_base_raw_candidate_count = sum(
                            len(output[1]) for output in planner_outputs
                        )
            np, _Image = _load_image_dependencies()
            raw_grasps = np.concatenate(
                [np.asarray(output[0]) for output in planner_outputs], axis=0
            )
            raw_scores = np.concatenate(
                [np.asarray(output[1]) for output in planner_outputs], axis=0
            )
            raw_tags = [str(tag) for output in planner_outputs for tag in output[2]]
            if recall_base_raw_candidate_count <= 0:
                recall_base_raw_candidate_count = int(len(raw_scores))
            metadata["model_inference_draw_count"] = len(draw_specs)
            metadata["model_inference_draw_seeds"] = draw_seeds
            metadata["graspmoe_draw_count"] = planners.count(PLANNER)
            metadata["diffusion_only_draw_count"] = planners.count("diffusion")
            metadata["recall_base_draw_count"] = min(
                len(RECALL_BASE_DRAW_SPECS), len(draw_specs)
            )
            metadata["recall_base_raw_candidate_count"] = (
                recall_base_raw_candidate_count
            )
            metadata["diversity_expansion_draw_count"] = max(
                0, len(draw_specs) - len(RECALL_BASE_DRAW_SPECS)
            )
            metadata["diversity_expansion_raw_candidate_count"] = (
                int(len(raw_scores)) - recall_base_raw_candidate_count
            )
            metadata["discarded_expansion_obb_candidate_count"] = (
                discarded_expansion_obb_count
            )
            metadata["deterministic_obb_reuse_policy"] = (
                "retain_first_full_draw_only"
            )
        except Exception as exc:  # noqa: BLE001 - third-party inference boundary.
            return failure_result(
                reason="model_inference_failed",
                metadata=_with_duration({**metadata, "error_type": type(exc).__name__}, start),
            )

        try:
            grasps_aligned, scores, tags = validate_raw_grasp_outputs(
                raw_grasps, raw_scores, raw_tags
            )
            camera_native_grasps = transform_grasps_to_camera(grasps_aligned, alignment)
            centering_corrections = None
            centering_spans = None
            centering_ratios = None
            if gripper.gripper_type.endswith("_2f"):
                centering_corrections, centering_spans, centering_ratios = (
                    _parallel_gripper_centering_metrics(
                        camera_native_grasps=camera_native_grasps,
                        object_points_camera=object_points,
                        gripper=gripper,
                    )
                )
                metadata.update(
                    {
                        "target_centering_ordering_applied": True,
                        "target_centering_risk_count": int(
                            (centering_ratios > CENTERING_RISK_RATIO).sum()
                        ),
                        "target_centering_ratio_p50": float(
                            np.quantile(centering_ratios, 0.50)
                        ),
                        "target_centering_ratio_p95": float(
                            np.quantile(centering_ratios, 0.95)
                        ),
                    }
                )
            else:
                metadata["target_centering_ordering_applied"] = False
            metadata.update(
                {
                    "raw_candidate_count": int(len(scores)),
                    "diffusion_candidate_count": tags.count("diff"),
                    "obb_candidate_count": tags.count("obb"),
                }
            )
            with self._inference_lock:
                selected_indices, collision_metadata = self._select_collision_free(
                    loaded=loaded,
                    sampler_entry=sampler_entry,
                    scene_points=scene_points,
                    camera_native_grasps=camera_native_grasps,
                    scores=scores,
                    branch_tags=tags,
                    selection_limit=self.raw_pool_size,
                    centering_ratios=centering_ratios,
                    recall_base_raw_count=recall_base_raw_candidate_count,
                )
            metadata.update(collision_metadata)
            raw_variant_parents = collision_metadata.get(
                "target_centering_reserve_parent_backend_indices", {}
            )
            variant_parents = (
                {
                    int(index): [int(parent) for parent in parents]
                    for index, parents in raw_variant_parents.items()
                    if isinstance(parents, list)
                }
                if isinstance(raw_variant_parents, dict)
                else {}
            )
            candidates = normalise_grasp_candidates(
                camera_native_grasps=camera_native_grasps,
                scores=scores,
                branch_tags=tags,
                selected_indices=selected_indices,
                gripper=gripper,
                centering_corrections=centering_corrections,
                centering_spans=centering_spans,
                centering_ratios=centering_ratios,
                centering_variant_parents=variant_parents,
            )
            if len(candidates) != len(selected_indices):
                raise GraspGenXInputError("inconsistent_grasp_outputs")
            metadata["returned_candidate_count"] = len(candidates)
            metadata["generated_candidate_count"] = len(candidates)
        except GraspGenXInputError as exc:
            metadata.update(exc.metadata)
            return failure_result(
                reason=exc.reason,
                metadata=_with_duration(metadata, start),
            )
        except Exception as exc:  # noqa: BLE001 - collision backend boundary.
            return failure_result(
                reason="model_inference_failed",
                metadata=_with_duration({**metadata, "error_type": type(exc).__name__}, start),
            )

        return {
            "success": True,
            "content": "GraspGenX grasp prediction completed.",
            "details": {
                "tool": TOOL_NAME,
                "backend": BACKEND_NAME,
                "model": MODEL_NAME,
                "planner": PLANNER,
                "deterministic": True,
                "frame": FRAME,
                "camera_frame": CAMERA_FRAME,
                "grasp_frame": GRASP_FRAME,
                "gripper_name": gripper.name,
                "model_raw_candidate_count": int(len(scores)),
                "raw_candidate_count": len(candidates),
                "generated_candidate_count": len(candidates),
                "candidate_count": len(candidates),
                "grasp_candidates": candidates,
                "ranking": (
                    "source_aware_se3_mmr_recall_base_with_target_centering_"
                    "reserve_and_minimum_se3_separation"
                ),
                "artifacts": [],
                "metadata": _with_duration(metadata, start),
            },
        }

    def _validate_gripper_name(self, gripper_name: Any) -> GripperDescription:
        if not isinstance(gripper_name, str) or gripper_name not in self.grippers:
            raise GraspGenXInputError("unsupported_gripper")
        return self.grippers[gripper_name]

    def _get_loaded_backend(self) -> dict[str, Any]:
        with self._inference_lock:
            return self._get_loaded_backend_locked()

    def _get_loaded_backend_locked(self) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded
        if not (self.graspgenx_root / "graspgenx" / "__init__.py").is_file():
            raise RuntimeError("GraspGenX source root is incomplete")

        # These existing paths make the official package setup hook a no-op and
        # prevent its fallback auto-clone behavior during import.
        os.environ["GRASPGENX_CHECKPOINT_DIR"] = str(self.checkpoint_root.parent)
        os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(self.gripper_descriptions_root)
        if str(self.graspgenx_root) not in sys.path:
            sys.path.insert(0, str(self.graspgenx_root))

        # GraspGenX configures third-party logging during its first imports.
        # Import while stdout points at stderr so those handlers can never
        # contaminate the stdio MCP JSON-RPC stream.
        with contextlib.redirect_stdout(sys.stderr):
            import torch

            from graspgenx.grasp_server import GraspGenXSampler, load_grasp_gen_model
            from graspgenx.samplers import run_planner_on_object
            from graspgenx.utils.checkpoint_io import load_model_cfg
            from graspgenx.utils.collision_filter import filter_colliding_grasps

        device_index = _resolve_cuda_device_index(torch, self.device)
        with torch.cuda.device(device_index), contextlib.redirect_stdout(sys.stderr):
            config = load_model_cfg(
                str(self.checkpoint_root / "gen"),
                str(self.checkpoint_root / "dis"),
                self.generator_checkpoint.name,
                self.discriminator_checkpoint.name,
            )
            model = load_grasp_gen_model(config)

        self._loaded = {
            "torch": torch,
            "device_index": device_index,
            "config": config,
            "model": model,
            "sampler_class": GraspGenXSampler,
            "run_planner": run_planner_on_object,
            "filter_collisions": filter_colliding_grasps,
            "backend_commit": _git_commit(self.graspgenx_root),
            "checkpoint_version": self.checkpoint_root.name,
            "generator_checkpoint_sha256": _sha256_file(self.generator_checkpoint),
            "discriminator_checkpoint_sha256": _sha256_file(self.discriminator_checkpoint),
        }
        return self._loaded

    def _get_sampler_entry(self, loaded: dict[str, Any], gripper_name: str) -> dict[str, Any]:
        with self._inference_lock:
            return self._get_sampler_entry_locked(loaded, gripper_name)

    def _get_sampler_entry_locked(
        self, loaded: dict[str, Any], gripper_name: str
    ) -> dict[str, Any]:
        cached = self._samplers.get(gripper_name)
        if cached is not None:
            return cached
        np, _Image = _load_image_dependencies()
        torch = loaded["torch"]
        with torch.cuda.device(loaded["device_index"]), contextlib.redirect_stdout(sys.stderr):
            sampler = loaded["sampler_class"](
                loaded["config"],
                gripper_name,
                assets_dir=str(self.assets_root),
                model=loaded["model"],
            )
        gripper_info = sampler.get_gripper_info()
        import trimesh

        # trimesh.sample_surface uses NumPy's legacy global RNG. Preserve the
        # caller state while making the cached collision geometry repeatable.
        numpy_state = np.random.get_state()
        np.random.seed((self.inference_seed + COLLISION_SURFACE_SEED_OFFSET) % (2**32))
        try:
            sampled, _faces = trimesh.sample.sample_surface(
                gripper_info.collision_mesh, NUM_COLLISION_SAMPLES
            )
        finally:
            np.random.set_state(numpy_state)
        surface_points = np.ascontiguousarray(sampled, dtype=np.float32)
        if surface_points.shape != (NUM_COLLISION_SAMPLES, 3):
            raise RuntimeError("invalid gripper collision surface sample")
        entry = {
            "sampler": sampler,
            "gripper_info": gripper_info,
            "collision_surface_points": surface_points,
        }
        self._samplers[gripper_name] = entry
        return entry

    @staticmethod
    def _seed_inference_rng(*, loaded: dict[str, Any], seed: int) -> None:
        """Seed every RNG used by one official GraspGenX model draw."""

        random.seed(seed)
        np, _Image = _load_image_dependencies()
        np.random.seed(seed % (2**32))
        manual_seed = getattr(loaded.get("torch"), "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)

    def _run_planner(
        self,
        *,
        loaded: dict[str, Any],
        sampler_entry: dict[str, Any],
        object_points_aligned: Any,
        moe_z_offsets_cm: tuple[float, ...],
        planner: str = PLANNER,
    ) -> tuple[Any, Any, Any]:
        torch = loaded["torch"]
        with (
            torch.cuda.device(loaded["device_index"]),
            torch.inference_mode(),
            contextlib.redirect_stdout(sys.stderr),
        ):
            grasps, scores, tags, _obb = loaded["run_planner"](
                object_points_aligned,
                sampler_entry["sampler"],
                planner=planner,
                grasp_threshold=-1.0,
                num_grasps=NUM_GRASPS,
                topk_num_grasps=-1,
                moe_num_yaws=MOE_NUM_YAWS,
                moe_z_offsets_cm=moe_z_offsets_cm,
                moe_outlier_threshold=MOE_OUTLIER_THRESHOLD,
                moe_outlier_k=MOE_OUTLIER_K,
                moe_obb_mode=MOE_OBB_MODE,
                moe_skip_obb_rule=MOE_SKIP_OBB_RULE,
                moe_obb_density=MOE_OBB_DENSITY,
                moe_obb_position_spacing_cm=MOE_OBB_POSITION_SPACING_CM,
            )
        return grasps, scores, tags

    def _select_collision_free(
        self,
        *,
        loaded: dict[str, Any],
        sampler_entry: dict[str, Any],
        scene_points: Any,
        camera_native_grasps: Any,
        scores: Any,
        branch_tags: list[str],
        selection_limit: int | None = None,
        centering_ratios: Any | None = None,
        recall_base_raw_count: int | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        np, _Image = _load_image_dependencies()
        score_array = np.asarray(scores, dtype=np.float64)
        poses = np.asarray(camera_native_grasps, dtype=np.float64)
        base_raw_count = (
            len(score_array)
            if recall_base_raw_count is None
            else int(recall_base_raw_count)
        )
        if base_raw_count <= 0 or base_raw_count > len(score_array):
            raise GraspGenXInputError("inconsistent_grasp_outputs")
        limit = self.raw_pool_size if selection_limit is None else int(selection_limit)
        reserve_capacity = (
            min(
                CENTERING_RESERVE_SIZE,
                max(0, limit - DEFAULT_GRASP_RAW_POOL_SIZE),
            )
            if centering_ratios is not None
            else 0
        )
        recall_base_limit = limit - reserve_capacity
        diversity_order_count = min(
            base_raw_count,
            max(
                COLLISION_BATCH_SIZE * 2,
                recall_base_limit * MMR_DIVERSITY_RESERVE_MULTIPLIER,
            ),
        )
        # The recall base deliberately uses the pre-centering MMR objective.
        # Ordering evidence must never erase a legacy model representative.
        inspection_order = _se3_mmr_order(
            poses=poses[:base_raw_count],
            scores=score_array[:base_raw_count],
            branch_tags=branch_tags[:base_raw_count],
            selection_limit=diversity_order_count,
            centering_ratios=None,
        )

        def variant_order(recall_base: list[int]) -> list[int]:
            if reserve_capacity <= 0 or centering_ratios is None:
                return []
            ordered = _centering_variant_order(
                poses=poses,
                scores=score_array,
                centering_ratios=centering_ratios,
                recall_base_indices=recall_base,
            )
            inspection_limit = max(
                COLLISION_BATCH_SIZE * 2,
                reserve_capacity * CENTERING_VARIANT_INSPECTION_MULTIPLIER,
            )
            return ordered[:inspection_limit]

        def variant_parent_evidence(
            recall_base: list[int], reserve: list[int]
        ) -> dict[str, list[int]]:
            if centering_ratios is None:
                return {}
            return {
                str(index): _compatible_centering_parent_indices(
                    poses=poses,
                    centering_ratios=centering_ratios,
                    candidate_index=index,
                    recall_base_indices=recall_base,
                )
                for index in reserve
            }

        selection_name = (
            "source_aware_se3_mmr_recall_base_with_target_centering_"
            "reserve_and_minimum_se3_separation"
        )
        if len(scene_points) == 0:
            recall_base: list[int] = []
            diversity_rejected = 0
            for index in inspection_order:
                if _is_formally_novel_grasp(
                    poses=poses,
                    candidate_index=index,
                    selected_indices=recall_base,
                ):
                    recall_base.append(index)
                    if len(recall_base) >= recall_base_limit:
                        break
                else:
                    diversity_rejected += 1
            reserve: list[int] = []
            for index in variant_order(recall_base):
                if _is_formally_novel_grasp(
                    poses=poses,
                    candidate_index=index,
                    selected_indices=reserve,
                ):
                    reserve.append(index)
                    if len(reserve) >= reserve_capacity:
                        break
                else:
                    diversity_rejected += 1
            selected = [*recall_base, *reserve]
            return selected, {
                "collision_filter_applied": False,
                "collision_filter_reason": "no_scene_points",
                "collision_scene_point_count": 0,
                "collision_checked_count": 0,
                "collision_rejected_count": 0,
                "candidate_selection": selection_name,
                "mmr_diversity_order_count": diversity_order_count,
                "formal_min_translation_m": FORMAL_MIN_TRANSLATION_M,
                "formal_min_approach_separation_rad": FORMAL_MIN_APPROACH_SEPARATION_RAD,
                "formal_min_wrist_rotation_rad": FORMAL_MIN_WRIST_ROTATION_RAD,
                "formal_diversity_rejected_count": diversity_rejected,
                "recall_base_raw_candidate_count": base_raw_count,
                "diversity_expansion_raw_candidate_count": (
                    len(score_array) - base_raw_count
                ),
                "recall_base_target_count": recall_base_limit,
                "recall_base_returned_count": len(recall_base),
                "target_centering_reserve_capacity": reserve_capacity,
                "target_centering_reserve_returned_count": len(reserve),
                "target_centering_reserve_parent_backend_indices": (
                    variant_parent_evidence(recall_base, reserve)
                ),
            }

        collision_scene = np.asarray(scene_points, dtype=np.float32)
        if len(collision_scene) > MAX_COLLISION_SCENE_POINTS:
            indices = np.random.default_rng(
                self.inference_seed + COLLISION_SCENE_SEED_OFFSET
            ).choice(len(collision_scene), MAX_COLLISION_SCENE_POINTS, replace=False)
            collision_scene = np.ascontiguousarray(collision_scene[indices])

        recall_base: list[int] = []
        checked = 0
        rejected = 0
        diversity_rejected = 0
        filter_fn = loaded["filter_collisions"]
        for offset in range(0, len(inspection_order), COLLISION_BATCH_SIZE):
            batch_indices = inspection_order[offset : offset + COLLISION_BATCH_SIZE]
            batch_poses = camera_native_grasps[batch_indices]
            free_mask = np.asarray(
                filter_fn(
                    scene_pc=collision_scene,
                    grasp_poses=batch_poses,
                    collision_threshold=COLLISION_THRESHOLD,
                    gripper_surface_points=sampler_entry["collision_surface_points"],
                    batch_size=COLLISION_BATCH_SIZE,
                    device=self.device,
                )
            )
            if free_mask.shape != (len(batch_indices),) or free_mask.dtype.kind != "b":
                raise GraspGenXInputError("inconsistent_grasp_outputs")
            checked += len(batch_indices)
            rejected += int((~free_mask).sum())
            for index, is_free in zip(batch_indices, free_mask):
                if not bool(is_free):
                    continue
                if _is_formally_novel_grasp(
                    poses=poses,
                    candidate_index=index,
                    selected_indices=recall_base,
                ):
                    recall_base.append(index)
                else:
                    diversity_rejected += 1
            selected_by_source = {
                source: sum(
                    1 for index in recall_base if branch_tags[index] == source
                )
                for source in set(branch_tags[:base_raw_count])
            }
            required_source_coverage = min(
                MMR_MIN_SOURCE_COVERAGE,
                max(1, recall_base_limit // len(selected_by_source)),
            )
            if len(recall_base) >= recall_base_limit and all(
                count >= required_source_coverage for count in selected_by_source.values()
            ):
                recall_base = recall_base[:recall_base_limit]
                break
        if not recall_base:
            raise GraspGenXInputError(
                "all_grasps_colliding",
                metadata={
                    "collision_filter_applied": True,
                    "collision_scene_point_count": int(len(collision_scene)),
                    "collision_checked_count": checked,
                    "collision_rejected_count": rejected,
                    "returned_candidate_count": 0,
                },
            )
        recall_base = recall_base[:recall_base_limit]
        reserve: list[int] = []
        ordered_variants = variant_order(recall_base)
        for offset in range(0, len(ordered_variants), COLLISION_BATCH_SIZE):
            batch_indices = ordered_variants[offset : offset + COLLISION_BATCH_SIZE]
            batch_poses = camera_native_grasps[batch_indices]
            free_mask = np.asarray(
                filter_fn(
                    scene_pc=collision_scene,
                    grasp_poses=batch_poses,
                    collision_threshold=COLLISION_THRESHOLD,
                    gripper_surface_points=sampler_entry["collision_surface_points"],
                    batch_size=COLLISION_BATCH_SIZE,
                    device=self.device,
                )
            )
            if free_mask.shape != (len(batch_indices),) or free_mask.dtype.kind != "b":
                raise GraspGenXInputError("inconsistent_grasp_outputs")
            checked += len(batch_indices)
            rejected += int((~free_mask).sum())
            for index, is_free in zip(batch_indices, free_mask):
                if not bool(is_free):
                    continue
                if _is_formally_novel_grasp(
                    poses=poses,
                    candidate_index=index,
                    selected_indices=reserve,
                ):
                    reserve.append(index)
                    if len(reserve) >= reserve_capacity:
                        break
                else:
                    diversity_rejected += 1
            if len(reserve) >= reserve_capacity:
                break
        selected = [*recall_base, *reserve][:limit]
        self.last_returned_candidate_count = len(selected)
        return selected, {
            "collision_filter_applied": True,
            "collision_scene_point_count": int(len(collision_scene)),
            "collision_checked_count": checked,
            "collision_rejected_count": rejected,
            "candidate_selection": selection_name,
            "mmr_diversity_order_count": diversity_order_count,
            "formal_min_translation_m": FORMAL_MIN_TRANSLATION_M,
            "formal_min_approach_separation_rad": FORMAL_MIN_APPROACH_SEPARATION_RAD,
            "formal_min_wrist_rotation_rad": FORMAL_MIN_WRIST_ROTATION_RAD,
            "formal_diversity_rejected_count": diversity_rejected,
            "recall_base_raw_candidate_count": base_raw_count,
            "diversity_expansion_raw_candidate_count": (
                len(score_array) - base_raw_count
            ),
            "recall_base_target_count": recall_base_limit,
            "recall_base_returned_count": len(recall_base),
            "target_centering_reserve_capacity": reserve_capacity,
            "target_centering_reserve_returned_count": len(reserve),
            "target_centering_reserve_parent_backend_indices": (
                variant_parent_evidence(recall_base, reserve)
            ),
        }

    def _metadata_base(self, *, gripper_name: Any) -> dict[str, Any]:
        return {
            "frame": FRAME,
            "camera_frame": CAMERA_FRAME,
            "grasp_frame": GRASP_FRAME,
            "native_grasp_frame": NATIVE_GRASP_FRAME,
            "planner": PLANNER,
            "deterministic": True,
            "determinism_scope": "same_hardware_software_model_and_input",
            "inference_seed": self.inference_seed,
            "gripper_name": gripper_name,
            "depth_truncation": self.depth_truncation,
            "min_object_points": MIN_OBJECT_POINTS,
            "max_returned_candidates": self.raw_pool_size,
            "raw_pool_size": self.raw_pool_size,
            "model_loaded": self.model_loaded,
            "intrinsics": {},
            "inference_options": {
                "num_grasps": NUM_GRASPS,
                "moe_num_yaws": MOE_NUM_YAWS,
                "moe_z_offsets_cm": "computed_per_observation",
                "moe_z_offset_source": GEOMETRY_DRIVEN_ANCHOR_SCHEMA,
                "moe_outlier_threshold": MOE_OUTLIER_THRESHOLD,
                "moe_outlier_k": MOE_OUTLIER_K,
                "moe_obb_mode": MOE_OBB_MODE,
                "moe_skip_obb_rule": MOE_SKIP_OBB_RULE,
                "moe_obb_density": MOE_OBB_DENSITY,
                "moe_obb_position_spacing_cm": MOE_OBB_POSITION_SPACING_CM,
                "collision_threshold": COLLISION_THRESHOLD,
                "max_collision_scene_points": MAX_COLLISION_SCENE_POINTS,
                "num_collision_samples": NUM_COLLISION_SAMPLES,
                "inference_draw_count": MODEL_INFERENCE_DRAWS,
                "inference_seed": self.inference_seed,
                "recall_base_draw_seed_offsets": [
                    offset for offset, _planner, _ in RECALL_BASE_DRAW_SPECS
                ],
                "diversity_expansion_draw_seed_offsets": [
                    offset
                    for offset, _planner, _ in DIVERSITY_EXPANSION_DRAW_SPECS
                ],
                "centering_mmr_penalty": CENTERING_MMR_PENALTY,
                "centering_risk_ratio": CENTERING_RISK_RATIO,
                "centering_reserve_size": CENTERING_RESERVE_SIZE,
                "centering_variant_max_translation_m": (
                    CENTERING_VARIANT_MAX_TRANSLATION_M
                ),
                "centering_variant_max_approach_rad": (
                    CENTERING_VARIANT_MAX_APPROACH_RAD
                ),
                "centering_variant_max_rotation_rad": (
                    CENTERING_VARIANT_MAX_ROTATION_RAD
                ),
                "centering_variant_min_improvement": (
                    CENTERING_VARIANT_MIN_IMPROVEMENT
                ),
            },
            "backend_commit": None,
            "checkpoint_version": self.checkpoint_root.name,
            "generator_checkpoint_sha256": None,
            "discriminator_checkpoint_sha256": None,
        }

    @staticmethod
    def _model_metadata(loaded: dict[str, Any]) -> dict[str, Any]:
        return {
            "backend": MODEL_NAME,
            "backend_commit": loaded["backend_commit"],
            "checkpoint_version": loaded["checkpoint_version"],
            "generator_checkpoint_sha256": loaded["generator_checkpoint_sha256"],
            "discriminator_checkpoint_sha256": loaded["discriminator_checkpoint_sha256"],
            "planner": PLANNER,
            "deterministic": True,
            "model_loaded": True,
        }


def validate_cuda_device_name(device: Any) -> str:
    if not isinstance(device, str) or not re.fullmatch(r"cuda(?::\d+)?", device):
        raise ValueError("device must be cuda or cuda:N")
    return "cuda:0" if device == "cuda" else device


def validate_inference_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("inference_seed must be an integer in [0, 2^31)")
    return seed


def _resolve_cuda_device_index(torch: Any, device: str) -> int:
    if not torch.cuda.is_available():
        raise GraspGenXInputError("device_unavailable")
    index = int(device.split(":", maxsplit=1)[1])
    if index < 0 or index >= int(torch.cuda.device_count()):
        raise GraspGenXInputError("device_unavailable")
    return index


def failure_result(*, reason: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    content = f"GraspGenX grasp prediction failed: {reason}."
    if reason in {
        "invalid_depth_scale",
        "depth_scale_mismatch",
        "empty_point_cloud_after_depth_filter",
    }:
        content = f"{content} {_DEPTH_SCALE_GUIDANCE}"
    return {
        "success": False,
        "content": content,
        "details": {
            "tool": TOOL_NAME,
            "reason": reason,
            "candidate_count": 0,
            "grasp_candidates": [],
            "metadata": dict(metadata or {}),
        },
    }


def _is_rigid_transform(matrix: Any) -> bool:
    np, _Image = _load_image_dependencies()
    matrix = np.asarray(matrix)
    return bool(
        matrix.shape == (4, 4)
        and np.isfinite(matrix).all()
        and np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        and _is_rotation_matrix(matrix[:3, :3])
    )


def _is_rotation_matrix(rotation: Any) -> bool:
    np, _Image = _load_image_dependencies()
    rotation = np.asarray(rotation)
    return bool(
        rotation.shape == (3, 3)
        and np.isfinite(rotation).all()
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    )


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if len(commit) == 40 else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_image_dependencies() -> tuple[Any, Any]:
    import numpy as np
    from PIL import Image

    return np, Image


def _float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _float_matrix(values: Any) -> list[list[float]]:
    return [_float_list(row) for row in values]


def _with_duration(metadata: dict[str, Any], start: float) -> dict[str, Any]:
    return {
        **metadata,
        "duration_ms": round((time.perf_counter() - start) * 1000.0, 3),
    }
