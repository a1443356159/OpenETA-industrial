from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from tools.anyplace_core import (
    AnyPlaceBackend,
    AnyPlaceInputError,
    OBJECT_POINTCLOUD_LIMITS,
    PLACEMENT_REGION_POINTCLOUD_LIMITS,
    build_masked_pointcloud_from_rgbd,
    normalise_placement_candidates,
    pad_pointcloud_for_model,
    _configure_cuda_extension_environment,
    _fixed_trimesh_rng,
    validate_inference_seed,
    validate_intrinsics,
    validate_pointcloud_array,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_industrial_profile_uses_one_deterministic_refinement_step() -> None:
    profile = yaml.safe_load(
        (REPO_ROOT / "config" / "anyplace_multitask.yaml").read_text(encoding="utf-8")
    )
    evaluation = profile["experiment"]["eval"]

    assert evaluation["init_orig_ori"] is True
    assert evaluation["n_refine_iters"] == 1
    assert evaluation["add_per_iter_noise"] is False


def test_validate_intrinsics() -> None:
    assert validate_intrinsics(_intrinsics()) == _intrinsics()
    _reason("missing_intrinsics", validate_intrinsics, None)


def test_validate_inference_seed_rejects_implicit_or_unbounded_values() -> None:
    assert validate_inference_seed(0) == 0
    assert validate_inference_seed(123) == 123
    for value in (True, -1, 2**31, 1.5, "4"):
        with pytest.raises(ValueError, match="inference_seed"):
            validate_inference_seed(value)


def test_fixed_trimesh_rng_replays_unseeded_stream_and_restores_function() -> None:
    def original_random_generator(seed=None):
        return np.random.default_rng(seed)

    trimesh = SimpleNamespace(
        util=SimpleNamespace(random_generator=original_random_generator)
    )

    with _fixed_trimesh_rng(trimesh=trimesh, np=np, seed=7):
        first = trimesh.util.random_generator().random(4)
        explicitly_seeded = trimesh.util.random_generator(11).random(4)
    with _fixed_trimesh_rng(trimesh=trimesh, np=np, seed=7):
        repeated = trimesh.util.random_generator().random(4)

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(
        explicitly_seeded, np.random.default_rng(11).random(4)
    )
    assert trimesh.util.random_generator is original_random_generator


def test_cuda_extension_environment_keeps_symlinked_venv_bin(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JIT tools must be found beside a symlinked virtualenv interpreter."""

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to("/usr/bin/python3")
    (venv_bin / "ninja").touch()
    monkeypatch.setattr(sys, "executable", str(venv_bin / "python"))
    monkeypatch.setenv("PATH", "/usr/bin")
    fake_torch = type("Torch", (), {"version": type("Version", (), {"cuda": ""})})

    _configure_cuda_extension_environment(fake_torch)

    assert os.environ["PATH"].split(os.pathsep)[0] == str(venv_bin)


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


@pytest.mark.parametrize("raw", [[], np.eye(4)])
def test_normalise_candidates_checks_dynamic_count(raw) -> None:
    _reason(
        "no_placement_candidates" if isinstance(raw, list) else "inconsistent_placement_outputs",
        normalise_placement_candidates, raw,
    )


def test_normalise_candidates_accepts_nonempty_dynamic_count() -> None:
    assert len(normalise_placement_candidates(np.tile(np.eye(4), (9, 1, 1)))) == 9


def test_backend_uses_independent_observations_and_transform(tmp_path, monkeypatch) -> None:
    backend = AnyPlaceBackend(
        anyplace_root=tmp_path, config_path=tmp_path / "config.yaml", seed=7
    )
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(backend, "_get_loaded_backend", lambda: {})
    monkeypatch.setattr(
        backend,
        "_predict_with_loaded_backend",
        lambda **kwargs: captured.append(kwargs) or np.tile(np.eye(4), (10, 1, 1)),
    )
    request = _request()
    request["object_camera_to_placement_camera"][0][3] = 0.25
    result = backend.predict_placement(**request)
    repeated = backend.predict_placement(**request)
    assert result["success"] is True
    assert result["details"]["frame"] == "placement_camera"
    assert result["details"]["candidate_count"] == 10
    assert captured[0]["object_pcd"][0, 0] == pytest.approx(0.25)
    assert [call["inference_seed"] for call in captured] == [7, 7]
    assert result["details"]["metadata"]["deterministic"] is True
    assert repeated["details"]["metadata"]["inference_seed"] == 7
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
    # Preserve the model's complete SE(3); downstream qualification decides feasibility.
    assert result["details"]["placement_candidates"][0]["object_placement_transform"]["transform_matrix"][2][3] == pytest.approx(-0.1)
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


def test_mcp_health_reports_fixed_inference_seed(tmp_path, monkeypatch) -> None:
    pytest.importorskip("mcp")
    from tools import anyplace_mcp_server

    backend = AnyPlaceBackend(
        anyplace_root=tmp_path, config_path=tmp_path / "config.yaml", seed=19
    )
    monkeypatch.setattr(anyplace_mcp_server, "_BACKEND", backend)

    health = anyplace_mcp_server.health_payload()

    assert health["deterministic"] is True
    assert health["inference_seed"] == 19
