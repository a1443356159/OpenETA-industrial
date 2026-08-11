"""Hand-computed geometry and contract tests for the M5 perception summary core."""

from __future__ import annotations

import base64
import io
import math

import numpy as np
import pytest
from PIL import Image

from adapter.protocol import CameraFrame
from extensions.gazebo.perception_summary import (
    ERR_DEPTH_MISSING,
    ERR_EMPTY_MASK,
    ERR_EXTRINSICS,
    ERR_INTRINSICS,
    ERR_MASK_MISSING,
    ERR_NO_VALID_DEPTH,
    PROVENANCE_SAM3,
    build_object_summary,
    summarize_detection,
)


# 100x100 px, f=100 px, principal point at the image centre.  Camera at the
# world origin with identity orientation: OpenCV camera axes aligned with the
# world axes (+Z forward).
_INTRINSICS = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0, "width": 100, "height": 100}
_EXTRINSICS = {
    "frame_transform": "camera_to_world",
    "camera_frame": "opencv",
    "pos": [0.0, 0.0, 0.0],
    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
}


def _write_mask(path, mask: np.ndarray) -> str:
    image = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    image.save(str(path), format="PNG")
    return str(path)


def _frame(depth, *, extrinsics=None, intrinsics=None, frame_id="top_camera_optical_frame"):
    return {
        "frame_id": frame_id,
        "depth": depth,
        "intrinsics": intrinsics if intrinsics is not None else dict(_INTRINSICS),
        "extrinsics": extrinsics if extrinsics is not None else dict(_EXTRINSICS),
    }


def test_constant_depth_block_back_projects_to_hand_computed_point(tmp_path) -> None:
    # Mask rows 40..59, cols 60..79 → centroid (u=69.5, v=49.5); z = 2.0.
    # x = (69.5-50)*2/100 = 0.39, y = (49.5-50)*2/100 = -0.01.
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 60:80] = True
    depth = np.full((100, 100), 2.0)
    detection = {
        "id": "det_0", "label": "target block", "score": 0.96,
        "mask_ref": _write_mask(tmp_path / "mask.png", mask),
        "source_image": str(tmp_path / "rgb.png"),
    }
    entry = summarize_detection(detection=detection, camera_frame=_frame(depth))
    assert entry["position"] == pytest.approx([0.39, -0.01, 2.0], abs=1e-9)
    assert "position_error" not in entry
    assert entry["id"] == "det_0"
    assert entry["label"] == "target block"
    assert entry["confidence"] == 0.96
    assert entry["visibility"] == "unknown"
    assert entry["source_camera"] == "top_camera_optical_frame"
    assert entry["provenance"] == PROVENANCE_SAM3


def test_median_depth_selects_middle_sample(tmp_path) -> None:
    # Three pixels in row 50 with depths {4.0, 1.0, 2.5} → median 2.5,
    # centroid (u=11, v=50) → x = (11-50)*2.5/100 = -0.975, y = 0.
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 10:13] = True
    depth = np.full((100, 100), 1.0)
    depth[50, 10] = 4.0
    depth[50, 11] = 1.0
    depth[50, 12] = 2.5
    detection = {"mask_ref": _write_mask(tmp_path / "mask.png", mask)}
    entry = summarize_detection(detection=detection, camera_frame=_frame(depth))
    assert entry["position"] == pytest.approx([-0.975, 0.0, 2.5], abs=1e-9)


def test_invalid_depth_pixels_are_excluded(tmp_path) -> None:
    # NaN and 0 depths are dropped: valid pixels are cols 12,13 (depths 2,4)
    # → median 3.0, centroid (u=12.5, v=50) → x = (12.5-50)*3/100 = -1.125.
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 10:14] = True
    depth = np.full((100, 100), 1.0)
    depth[50, 10] = np.nan
    depth[50, 11] = 0.0
    depth[50, 12] = 2.0
    depth[50, 13] = 4.0
    detection = {"mask_ref": _write_mask(tmp_path / "mask.png", mask)}
    entry = summarize_detection(detection=detection, camera_frame=_frame(depth))
    assert entry["position"] == pytest.approx([-1.125, 0.0, 3.0], abs=1e-9)


def test_camera_to_world_extrinsics_rotate_and_translate(tmp_path) -> None:
    # Single pixel at (v=50, u=60), z=2 → camera point (0.2, 0, 2).
    # Extrinsics: pos (1,0,0), quat +90° about world Z maps (x,y,z)→(-y,x,z),
    # so world = (0, 0.2, 2) + (1, 0, 0) = (1.0, 0.2, 2.0).
    half_sqrt2 = math.sqrt(0.5)
    extrinsics = {
        "frame_transform": "camera_to_world",
        "camera_frame": "opencv",
        "pos": [1.0, 0.0, 0.0],
        "quat_xyzw": [0.0, 0.0, half_sqrt2, half_sqrt2],
    }
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 60] = True
    depth = np.full((100, 100), 2.0)
    detection = {"mask_ref": _write_mask(tmp_path / "mask.png", mask)}
    entry = summarize_detection(detection=detection, camera_frame=_frame(depth, extrinsics=extrinsics))
    assert entry["position"] == pytest.approx([1.0, 0.2, 2.0], abs=1e-9)


def test_missing_depth_yields_null_position(tmp_path) -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 50] = True
    detection = {"mask_ref": _write_mask(tmp_path / "mask.png", mask)}
    entry = summarize_detection(detection=detection, camera_frame=_frame(None))
    assert entry["position"] is None
    assert entry["position_error"] == ERR_DEPTH_MISSING


def test_empty_mask_yields_null_position(tmp_path) -> None:
    mask = np.zeros((100, 100), dtype=bool)
    depth = np.full((100, 100), 2.0)
    detection = {"mask_ref": _write_mask(tmp_path / "mask.png", mask)}
    entry = summarize_detection(detection=detection, camera_frame=_frame(depth))
    assert entry["position"] is None
    assert entry["position_error"] == ERR_EMPTY_MASK


def test_mask_without_valid_depth_yields_null_position(tmp_path) -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    depth = np.zeros((100, 100))  # 0.0 is not a valid range
    depth[50, 50] = np.nan
    detection = {"mask_ref": _write_mask(tmp_path / "mask.png", mask)}
    entry = summarize_detection(detection=detection, camera_frame=_frame(depth))
    assert entry["position"] is None
    assert entry["position_error"] == ERR_NO_VALID_DEPTH


def test_wrist_tf_dynamic_extrinsics_are_rejected(tmp_path) -> None:
    # Wrist camera extrinsics are not yet numeric (plan.md §8 follow-up, same
    # restriction as the M4 oracle): refuse rather than invent a position.
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 50] = True
    depth = np.full((100, 100), 2.0)
    extrinsics = {"frame_transform": "tf_dynamic", "camera_frame": "opencv"}
    detection = {"mask_ref": _write_mask(tmp_path / "mask.png", mask)}
    entry = summarize_detection(
        detection=detection, camera_frame=_frame(depth, extrinsics=extrinsics)
    )
    assert entry["position"] is None
    assert entry["position_error"] == ERR_EXTRINSICS


def test_invalid_intrinsics_yield_null_position(tmp_path) -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 50] = True
    depth = np.full((100, 100), 2.0)
    detection = {"mask_ref": _write_mask(tmp_path / "mask.png", mask)}
    for intrinsics in ({}, {"fx": 0.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}):
        entry = summarize_detection(
            detection=detection, camera_frame=_frame(depth, intrinsics=intrinsics)
        )
        assert entry["position"] is None
        assert entry["position_error"] == ERR_INTRINSICS


def test_missing_mask_yields_null_position() -> None:
    depth = np.full((100, 100), 2.0)
    entry = summarize_detection(detection={"id": "det_0"}, camera_frame=_frame(depth))
    assert entry["position"] is None
    assert entry["position_error"] == ERR_MASK_MISSING


def test_inline_base64_mask_is_accepted() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 50] = True
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    detection = {
        "mask": {"format": "png", "base64": base64.b64encode(buffer.getvalue()).decode("ascii")}
    }
    depth = np.full((100, 100), 2.0)
    entry = summarize_detection(detection=detection, camera_frame=_frame(depth))
    # Principal-point pixel: x = y = 0, z = 2.
    assert entry["position"] == pytest.approx([0.0, 0.0, 2.0], abs=1e-9)


def test_mask_resized_to_depth_resolution(tmp_path) -> None:
    # Full-frame 200x200 mask on a 100x100 depth map: resampled nearest, the
    # centroid stays at the image centre (u=v=49.5) with z=1.0.
    mask = np.ones((200, 200), dtype=bool)
    depth = np.full((100, 100), 1.0)
    detection = {"mask_ref": _write_mask(tmp_path / "mask.png", mask)}
    entry = summarize_detection(detection=detection, camera_frame=_frame(depth))
    assert entry["position"] == pytest.approx([-0.005, -0.005, 1.0], abs=1e-9)


def test_camera_frame_dataclass_input(tmp_path) -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[2, 3] = True
    frame = CameraFrame(
        frame_id="top_camera_optical_frame",
        rgb=[[[0, 0, 0]]],
        depth=[[1.0, 1.0, 1.0, 1.0]] * 4,
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": 1.5, "cy": 1.5, "width": 4, "height": 4},
        extrinsics=dict(_EXTRINSICS),
    )
    detection = {"mask_ref": _write_mask(tmp_path / "mask.png", mask)}
    entry = summarize_detection(detection=detection, camera_frame=frame)
    # u=3, v=2, z=1 → x = (3-1.5)/100 = 0.015, y = (2-1.5)/100 = 0.005.
    assert entry["position"] == pytest.approx([0.015, 0.005, 1.0], abs=1e-9)
    assert entry["source_camera"] == "top_camera_optical_frame"


def test_build_object_summary_wraps_entries(tmp_path) -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 50] = True
    mask_ref = _write_mask(tmp_path / "mask.png", mask)
    depth = np.full((100, 100), 2.0)
    summary = build_object_summary(
        detections=[
            {"id": "a", "mask_ref": mask_ref},
            {"id": "b"},  # no mask → null position entry, not an exception
        ],
        camera_frame=_frame(depth),
    )
    assert list(summary) == ["objects"]
    assert [entry["id"] for entry in summary["objects"]] == ["a", "b"]
    assert summary["objects"][0]["position"] is not None
    assert summary["objects"][1]["position"] is None
