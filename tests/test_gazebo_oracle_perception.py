"""Hand-computed geometry and contract tests for the Gazebo oracle perception core."""

from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image

from extensions.gazebo.m3 import M3Config
from extensions.gazebo.m2 import M2Config
from extensions.gazebo.oracle_perception import (
    OracleObjectSpec,
    PosedOracleObject,
    match_camera_frame,
    oracle_registry_from_model_config,
    oracle_segment_prompt,
    posed_oracle_objects,
    prompt_matches_object,
)


# Camera at the world origin with identity orientation: camera X/Y/Z aligned
# with world X/Y/Z (OpenCV convention, +Z forward), 100x100 px, f=100 px.
_INTRINSICS = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0, "width": 100, "height": 100}
_EXTRINSICS = {
    "frame_transform": "camera_to_world",
    "camera_frame": "opencv",
    "pos": [0.0, 0.0, 0.0],
    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
}

_BOX_SPEC = OracleObjectSpec(
    object_id="m3_target", name="m3_target", label="target block",
    shape="box", dimensions=(2.0, 2.0, 2.0),
)
_CYLINDER_SPEC = OracleObjectSpec(
    object_id="m3_distractor", name="m3_distractor", label="distractor cylinder",
    shape="cylinder", dimensions=(0.5, 2.0),
)


def _posed(spec: OracleObjectSpec, position, orientation=(0.0, 0.0, 0.0, 1.0)) -> PosedOracleObject:
    return PosedOracleObject(spec=spec, position=tuple(position), orientation=tuple(orientation))


def _decode_mask(detection: dict) -> np.ndarray:
    payload = detection["mask"]
    assert payload["format"] == "png"
    image = Image.open(io.BytesIO(base64.b64decode(payload["base64"])))
    assert image.mode == "L"
    return np.asarray(image)


def _only_detection(result: dict) -> dict:
    assert result["success"] is True
    details = result["details"]
    assert details["detection_count"] == 1
    assert len(details["detections"]) == 1
    return details["detections"][0]


def test_box_frontal_projection_matches_hand_computed_square() -> None:
    # Box 2x2x2 centred at (0,0,5): the near face (z=4) dominates the hull.
    # u = 100*x/z + 50 → corners at 50 ± 100*1/4 = {25, 75} on both axes.
    result = oracle_segment_prompt(
        prompt="target", objects=[_posed(_BOX_SPEC, (0.0, 0.0, 5.0))],
        intrinsics=_INTRINSICS, extrinsics=_EXTRINSICS,
    )
    detection = _only_detection(result)
    assert detection["bbox_xyxy"] == [25, 25, 76, 76]
    assert detection["area_px"] == 51 * 51
    mask = _decode_mask(detection)
    assert mask.shape == (100, 100)
    assert set(np.unique(mask)) == {0, 255}
    assert mask[50, 50] == 255 and mask[25, 25] == 255 and mask[75, 75] == 255
    assert mask[24, 50] == 0 and mask[50, 76] == 0


def test_box_oblique_projection_shifts_and_stretches_hull() -> None:
    # Box 1x1x1 centred at (1,0,5): corners x∈{0.5,1.5}, z∈{4.5,5.5}.
    # u extremes: 100*0.5/5.5+50 ≈ 59.09 (x=0.5,z=5.5) and
    # 100*1.5/4.5+50 ≈ 83.33 (x=1.5,z=4.5);
    # v extremes: 50 ± 100*0.5/4.5 ≈ {38.89, 61.11} (y=±0.5, z=4.5).
    spec = OracleObjectSpec(
        object_id="m3_target", name="m3_target", label="target block",
        shape="box", dimensions=(1.0, 1.0, 1.0),
    )
    result = oracle_segment_prompt(
        prompt="target", objects=[_posed(spec, (1.0, 0.0, 5.0))],
        intrinsics=_INTRINSICS, extrinsics=_EXTRINSICS,
    )
    detection = _only_detection(result)
    bbox = detection["bbox_xyxy"]
    assert abs(bbox[0] - 59) <= 1
    assert abs(bbox[1] - 38) <= 1
    assert abs(bbox[2] - 84) <= 1
    assert abs(bbox[3] - 62) <= 1
    mask = _decode_mask(detection)
    assert mask[50, 70] == 255  # projection of the box centre (100*1/5+50, 50)
    assert mask[10, 10] == 0
    assert mask[50, 90] == 0


def test_cylinder_axis_toward_camera_projects_to_disc() -> None:
    # Cylinder r=0.5 along camera Z at (0,0,5): near rim (z=4) projects to a
    # 16-gon of radius 100*0.5/4 = 12.5 px around the principal point.
    result = oracle_segment_prompt(
        prompt="distractor", objects=[_posed(_CYLINDER_SPEC, (0.0, 0.0, 5.0))],
        intrinsics=_INTRINSICS, extrinsics=_EXTRINSICS,
    )
    detection = _only_detection(result)
    bbox = detection["bbox_xyxy"]
    assert abs(bbox[0] - 37) <= 1  # 50 - 12.5
    assert abs(bbox[1] - 37) <= 1
    assert abs(bbox[2] - 63) <= 1  # 50 + 12.5, half-open
    assert abs(bbox[3] - 63) <= 1
    assert math.pi * 11.5**2 < detection["area_px"] < math.pi * 13.5**2
    mask = _decode_mask(detection)
    assert mask[50, 50] == 255
    assert mask[50, 63] == 0


def test_cylinder_sideways_orientation_projects_to_stadium() -> None:
    # Rotate 90° about Y so the cylinder axis lies along world X: rim points
    # map to x=±1 exactly, y=0.5·sinθ, z=5−0.5·cosθ.  Projection extremes over
    # the 16 rim samples: u: 100·(±1)/4.5+50 ≈ {27.8, 72.2} (at cosθ=±1);
    # v: 50 ± 100·0.5/5 = {40, 60} (the v extreme sits at sinθ=±1 where z=5).
    quat_y90 = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
    result = oracle_segment_prompt(
        prompt="distractor",
        objects=[_posed(_CYLINDER_SPEC, (0.0, 0.0, 5.0), orientation=quat_y90)],
        intrinsics=_INTRINSICS, extrinsics=_EXTRINSICS,
    )
    detection = _only_detection(result)
    bbox = detection["bbox_xyxy"]
    assert abs(bbox[0] - 27) <= 1
    assert abs(bbox[1] - 40) <= 1
    assert abs(bbox[2] - 73) <= 1
    assert abs(bbox[3] - 61) <= 1
    mask = _decode_mask(detection)
    assert mask[50, 50] == 255
    assert mask[20, 50] == 0


def test_partially_out_of_frame_projection_is_clipped_to_image_bounds() -> None:
    # Box 2x2x2 at (3.5,0,5): rightmost corner u = 100*4.5/4+50 = 162.5 → the
    # mask must be cropped at the image edge (bbox max-exclusive == width).
    result = oracle_segment_prompt(
        prompt="target", objects=[_posed(_BOX_SPEC, (3.5, 0.0, 5.0))],
        intrinsics=_INTRINSICS, extrinsics=_EXTRINSICS,
    )
    detection = _only_detection(result)
    bbox = detection["bbox_xyxy"]
    assert bbox[2] == 100
    assert abs(bbox[0] - 91) <= 2  # 100*2.5/6+50 ≈ 91.7
    mask = _decode_mask(detection)
    assert mask[50, 99] == 255


def test_fully_out_of_frame_object_produces_no_detection() -> None:
    result = oracle_segment_prompt(
        prompt="target", objects=[_posed(_BOX_SPEC, (50.0, 0.0, 5.0))],
        intrinsics=_INTRINSICS, extrinsics=_EXTRINSICS,
    )
    assert result["success"] is True
    assert result["details"]["detection_count"] == 0
    assert result["details"]["detections"] == []
    assert "no detections" in result["content"]


def test_object_behind_camera_produces_no_detection() -> None:
    result = oracle_segment_prompt(
        prompt="target", objects=[_posed(_BOX_SPEC, (0.0, 0.0, -5.0))],
        intrinsics=_INTRINSICS, extrinsics=_EXTRINSICS,
    )
    assert result["success"] is True
    assert result["details"]["detection_count"] == 0


def test_camera_to_world_extrinsics_are_applied() -> None:
    # Top camera profile pose: pos (0,0,1.8), 180° about (x,−y)/√2 — looking
    # straight down (R maps (x,y,z) → (−y,−x,−z)).  Box corners sit at camera
    # depths {0.9, 1.1} m with x_cam/y_cam ∈ {±0.1}; the nearest corners
    # dominate the hull: u/v extremes = 50 ± 100·0.1/0.9 ≈ {38.9, 61.1}.
    extrinsics = {
        "frame_transform": "camera_to_world",
        "camera_frame": "opencv",
        "pos": [0.0, 0.0, 1.8],
        "quat_xyzw": [0.7071067812, -0.7071067812, 0.0, 0.0],
    }
    spec = OracleObjectSpec(
        object_id="obj", name="obj", label="obj", shape="box",
        dimensions=(0.2, 0.2, 0.2),
    )
    result = oracle_segment_prompt(
        prompt="obj", objects=[_posed(spec, (0.0, 0.0, 0.8))],
        intrinsics=_INTRINSICS, extrinsics=extrinsics,
    )
    detection = _only_detection(result)
    bbox = detection["bbox_xyxy"]
    assert abs(bbox[0] - 38) <= 1
    assert abs(bbox[1] - 38) <= 1
    assert abs(bbox[2] - 62) <= 1
    assert abs(bbox[3] - 62) <= 1


def test_prompt_matching_is_case_insensitive_substring_on_id_name_label() -> None:
    objects = [
        _posed(_BOX_SPEC, (0.0, -1.0, 5.0)),
        _posed(_CYLINDER_SPEC, (0.0, 1.0, 5.0)),
    ]
    cases = {
        "TARGET": 1,                # case-insensitive, substring of id
        "target block": 1,          # exact label
        "distractor": 1,            # substring of id/name/label
        "cylinder": 1,              # substring of label
        "m3": 2,                    # substring of both ids
        "m3_target": 1,
        "nonexistent": 0,
    }
    for prompt, expected_count in cases.items():
        result = oracle_segment_prompt(
            prompt=prompt, objects=objects,
            intrinsics=_INTRINSICS, extrinsics=_EXTRINSICS,
        )
        assert result["success"] is True, prompt
        assert result["details"]["detection_count"] == expected_count, prompt


def test_prompt_matching_helper_covers_both_substring_directions() -> None:
    assert prompt_matches_object("target", _BOX_SPEC)
    assert prompt_matches_object("the target block on the table", _BOX_SPEC)
    assert prompt_matches_object("M3_TARGET", _BOX_SPEC)
    assert not prompt_matches_object("", _BOX_SPEC)
    assert not prompt_matches_object("sphere", _BOX_SPEC)


def test_response_contract_aligns_field_for_field_with_sam3() -> None:
    result = oracle_segment_prompt(
        prompt="target", objects=[_posed(_BOX_SPEC, (0.0, 0.0, 5.0))],
        intrinsics=_INTRINSICS, extrinsics=_EXTRINSICS,
        extra_metadata={"frame_match": "pixel", "camera_frame_id": "top_camera_optical_frame"},
    )
    assert set(result) == {"success", "content", "details"}
    details = result["details"]
    assert set(details) == {
        "tool", "backend", "model", "prompt", "prompt_type", "detection_count",
        "detections", "ranking", "artifacts", "metadata",
    }
    assert details["tool"] == "oracle_perceive"
    assert details["prompt"] == "target"
    assert details["prompt_type"] == "text"
    assert details["ranking"] == "score_descending"
    assert details["artifacts"] == []
    metadata = details["metadata"]
    assert metadata["perception_source"] == "gazebo_oracle"
    assert metadata["frame_match"] == "pixel"
    assert metadata["camera_frame_id"] == "top_camera_optical_frame"
    assert metadata["image_size"] == [100, 100]

    detection = result["details"]["detections"][0]
    assert set(detection) == {
        "label", "score", "bbox_xyxy", "mask", "area_px", "backend_index", "rank",
    }
    assert detection["label"] == "target block"
    assert detection["score"] == 1.0
    assert detection["backend_index"] == 0
    assert detection["rank"] == 0
    assert set(detection["mask"]) == {"format", "base64"}

    # bbox/area must be derivable from the mask itself (SAM3 backend parity).
    mask = _decode_mask(detection)
    ys, xs = np.nonzero(mask)
    assert detection["bbox_xyxy"] == [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    assert detection["area_px"] == int((mask > 0).sum())


def test_tf_dynamic_extrinsics_fail_explicitly() -> None:
    result = oracle_segment_prompt(
        prompt="target", objects=[_posed(_BOX_SPEC, (0.0, 0.0, 5.0))],
        intrinsics=_INTRINSICS,
        extrinsics={"frame_transform": "tf_dynamic", "camera_frame": "opencv"},
    )
    assert result["success"] is False
    details = result["details"]
    assert details["reason"] == "ORACLE_FRAME_UNSUPPORTED"
    assert details["detection_count"] == 0
    assert details["detections"] == []
    assert details["metadata"]["perception_source"] == "gazebo_oracle"


def test_missing_prompt_fails_like_sam3() -> None:
    result = oracle_segment_prompt(
        prompt="", objects=[_posed(_BOX_SPEC, (0.0, 0.0, 5.0))],
        intrinsics=_INTRINSICS, extrinsics=_EXTRINSICS,
    )
    assert result["success"] is False
    assert result["details"]["reason"] == "missing_prompt"


def test_registry_from_m3_config_matches_declared_geometry() -> None:
    specs = oracle_registry_from_model_config(M3Config())
    assert [spec.object_id for spec in specs] == ["m3_target", "m3_distractor"]
    box, cylinder = specs
    assert box.shape == "box" and box.dimensions == (0.04, 0.04, 0.06)
    assert cylinder.shape == "cylinder" and cylinder.dimensions == (0.025, 0.08)
    assert oracle_registry_from_model_config(M2Config()) == []
    assert oracle_registry_from_model_config(None) == []


def test_posed_oracle_objects_joins_observation_dicts_with_registry() -> None:
    registry = oracle_registry_from_model_config(M3Config())
    observation_objects = [
        {"id": "m3_target", "name": "m3_target", "label": "target block",
         "position": [0.28, -0.10, 0.43], "orientation": [0.0, 0.0, 0.0, 1.0]},
        {"id": "m3_table", "name": "m3_table", "label": "table",
         "position": [0.40, 0.0, 0.38], "orientation": [0.0, 0.0, 0.0, 1.0]},
        {"id": "m3_distractor", "name": "m3_distractor", "label": "distractor cylinder",
         "position": [0.28, 0.12, 0.44], "orientation": [0.0, 0.0, 0.0, 1.0]},
    ]
    posed = posed_oracle_objects(registry, observation_objects)
    # The table has no registry shape and must be skipped.
    assert [item.spec.object_id for item in posed] == ["m3_target", "m3_distractor"]
    assert posed[0].position == (0.28, -0.10, 0.43)


def test_match_camera_frame_pixel_fallback_and_failures() -> None:
    top_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    wrist_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    cameras = {
        "top_camera_optical_frame": {"rgb": top_rgb, "intrinsics": dict(_INTRINSICS)},
        "wrist_camera_optical_frame": {"rgb": wrist_rgb},
    }

    frame_id, camera, status = match_camera_frame(top_rgb, cameras)
    assert (frame_id, status) == ("top_camera_optical_frame", "pixel")

    shifted = top_rgb.copy()
    shifted[0, 0] = 255
    frame_id, camera, status = match_camera_frame(shifted, cameras)
    assert (frame_id, status) == ("top_camera_optical_frame", "fallback_size")

    frame_id, camera, status = match_camera_frame(np.zeros((32, 32, 3), dtype=np.uint8), cameras)
    assert (frame_id, status) == (None, "frame_not_found")

    twin = np.zeros((100, 100, 3), dtype=np.uint8)
    twin[1, 1] = 7
    cameras_ambiguous = {
        "cam_a": {"rgb": top_rgb},
        "cam_b": {"rgb": twin},
    }
    frame_id, camera, status = match_camera_frame(np.full((100, 100, 3), 3, dtype=np.uint8), cameras_ambiguous)
    assert (frame_id, status) == (None, "frame_match_ambiguous")
