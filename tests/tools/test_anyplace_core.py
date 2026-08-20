from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tools.anyplace_core import (
    AnyPlaceBackend,
    AnyPlaceInputError,
    OBJECT_POINTCLOUD_LIMITS,
    PLACEMENT_REGION_POINTCLOUD_LIMITS,
    build_masked_pointcloud_from_rgbd,
    normalise_placement_candidates,
    pad_pointcloud_for_model,
    _project_object_bottoms_to_support,
    validate_intrinsics,
    validate_pointcloud_array,
)


def _payload(array: np.ndarray) -> dict[str, str]:
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return {"format": "png", "base64": base64.b64encode(stream.getvalue()).decode()}


def _intrinsics() -> dict[str, float]:
    return {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0, "scale": 1000.0}


def _packet(mask_left: bool) -> dict[str, Any]:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), 500, dtype=np.uint16)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, :32] = 255 if mask_left else 0
    mask[:, 32:] = 0 if mask_left else 255
    return {"rgb": _payload(rgb), "depth": _payload(depth), "mask": _payload(mask), "intrinsics": _intrinsics()}


def _request() -> dict[str, Any]:
    object_packet = _packet(True)
    object_packet["object_mask"] = object_packet.pop("mask")
    placement_packet = _packet(False)
    placement_packet["placement_region_mask"] = placement_packet.pop("mask")
    return {
        "object_observation": object_packet,
        "placement_observation": placement_packet,
        "object_camera_to_placement_camera": np.eye(4).tolist(),
        "placement_camera_to_world": np.eye(4).tolist(),
    }


def _reason(expected: str, function, *args, **kwargs) -> None:
    with pytest.raises(AnyPlaceInputError) as caught:
        function(*args, **kwargs)
    assert caught.value.reason == expected


def test_validate_intrinsics() -> None:
    assert validate_intrinsics(_intrinsics()) == _intrinsics()
    _reason("missing_intrinsics", validate_intrinsics, None)


@pytest.mark.parametrize(
    ("left", "limits", "first"),
    [(True, OBJECT_POINTCLOUD_LIMITS, [0.0, 0.0, 0.5]), (False, PLACEMENT_REGION_POINTCLOUD_LIMITS, [0.16, 0.0, 0.5])],
)
def test_build_one_masked_pointcloud(left, limits, first) -> None:
    packet = _packet(left)
    def decode(value):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(value["base64"]))))
    points = build_masked_pointcloud_from_rgbd(
        rgb=decode(packet["rgb"]), depth=decode(packet["depth"]), mask=decode(packet["mask"]),
        intrinsics=_intrinsics(), limits=limits, empty_reason="empty_object_pointcloud",
        too_small_reason="object_pointcloud_too_small", depth_truncation=1.0,
    )
    assert points.shape == (2048, 3)
    np.testing.assert_allclose(points[0], first)


def test_small_but_measured_cloud_is_padded_to_anyplace_model_sample_count() -> None:
    measured = np.arange(873 * 3, dtype=np.float32).reshape(873, 3)

    model_input = pad_pointcloud_for_model(measured)

    assert model_input.shape == (1024, 3)
    np.testing.assert_allclose(model_input[0], measured[0])
    np.testing.assert_allclose(model_input[-1], measured[-1])


def test_pointcloud_limits_reject_insufficient_measured_geometry() -> None:
    points = np.zeros((127, 3), dtype=np.float32)

    _reason(
        "object_pointcloud_too_small",
        validate_pointcloud_array,
        points,
        limits=OBJECT_POINTCLOUD_LIMITS,
        empty_reason="empty_object_pointcloud",
        too_small_reason="object_pointcloud_too_small",
    )


def test_normalise_candidates_returns_only_object_transforms() -> None:
    poses = np.tile(np.eye(4), (10, 1, 1))
    poses[0, :3, 3] = [0.1, 0.2, 0.3]
    candidates = normalise_placement_candidates(poses)
    assert len(candidates) == 10
    assert candidates[0]["object_placement_transform"]["frame"] == "placement_camera"
    assert "source_grasp_id" not in candidates[0]
    assert "place_grasp_pose" not in candidates[0]


def test_project_object_bottoms_to_measured_support() -> None:
    transforms = np.tile(np.eye(4), (2, 1, 1))
    transforms[:, 2, 3] = [0.8, 1.2]
    adjusted = _project_object_bottoms_to_support(
        transforms,
        object_points=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
        support_points=np.array([[0.0, 0.0, 0.4], [0.1, 0.0, 0.4]]),
        np=np,
    )

    assert adjusted[:, 2, 3] == pytest.approx([0.398, 0.398])


@pytest.mark.parametrize("raw", [[], np.eye(4), np.tile(np.eye(4), (9, 1, 1))])
def test_normalise_candidates_checks_dynamic_count(raw) -> None:
    _reason(
        "no_placement_candidates" if isinstance(raw, list) else "inconsistent_placement_outputs",
        normalise_placement_candidates, raw,
    )


def test_backend_uses_independent_observations_and_transform(tmp_path, monkeypatch) -> None:
    backend = AnyPlaceBackend(anyplace_root=tmp_path, config_path=tmp_path / "config.yaml")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(backend, "_get_loaded_backend", lambda: {})
    monkeypatch.setattr(
        backend,
        "_predict_with_loaded_backend",
        lambda **kwargs: captured.update(kwargs) or np.tile(np.eye(4), (10, 1, 1)),
    )
    request = _request()
    request["object_camera_to_placement_camera"][0][3] = 0.25
    result = backend.predict_placement(**request)
    assert result["success"] is True
    assert result["details"]["frame"] == "placement_camera"
    assert result["details"]["candidate_count"] == 10
    assert captured["object_pcd"][0, 0] == pytest.approx(0.25)
    assert "selected_grasp" not in str(result)


def test_backend_uses_gravity_aligned_model_frame_and_restores_camera_output(
    tmp_path, monkeypatch
) -> None:
    backend = AnyPlaceBackend(anyplace_root=tmp_path, config_path=tmp_path / "config.yaml")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(backend, "_get_loaded_backend", lambda: {})
    raw = np.tile(np.eye(4), (10, 1, 1))
    # This is a world-frame relation: move +10 cm along gravity/world Z.
    raw[:, 2, 3] = 0.1
    monkeypatch.setattr(
        backend,
        "_predict_with_loaded_backend",
        lambda **kwargs: captured.update(kwargs) or raw,
    )
    request = _request()
    # OpenCV +Z points down in this top-camera calibration; world +Z points up.
    request["placement_camera_to_world"] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    result = backend.predict_placement(**request)

    assert result["success"] is True
    assert captured["placement_region_pcd"][0, 2] == pytest.approx(0.5)
    # Bottom-on-support projection removes the model's unsupported vertical offset.
    assert result["details"]["placement_candidates"][0]["object_placement_transform"]["transform_matrix"][2][3] == pytest.approx(0.0)
    assert result["details"]["metadata"]["model_frame"] == "world_gravity_aligned"


def test_backend_rejects_missing_or_invalid_observation_transform(tmp_path) -> None:
    backend = AnyPlaceBackend(anyplace_root=tmp_path, config_path=tmp_path / "config.yaml")
    request = _request()
    request["object_camera_to_placement_camera"] = [[1.0]]
    result = backend.predict_placement(**request)
    assert result["success"] is False
    assert result["details"]["reason"] == "invalid_observation_transform"


def test_backend_rejects_invalid_placement_camera_to_world(tmp_path) -> None:
    backend = AnyPlaceBackend(anyplace_root=tmp_path, config_path=tmp_path / "config.yaml")
    request = _request()
    request["placement_camera_to_world"] = [[1.0]]
    result = backend.predict_placement(**request)
    assert result["success"] is False
    assert result["details"]["reason"] == "invalid_placement_camera_to_world"


def test_mcp_unconfigured_failure_has_no_candidates(monkeypatch) -> None:
    pytest.importorskip("mcp")
    from tools import anyplace_mcp_server
    monkeypatch.setattr(anyplace_mcp_server, "_BACKEND", None)
    result = anyplace_mcp_server.predict_placement()
    assert result["success"] is False
    assert result["details"]["placement_candidates"] == []
