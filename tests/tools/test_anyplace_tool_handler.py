from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from agent.tools.handlers import build_anyplace_handler
from agent.tools.registry import ToolExecutionContext
from agent.tools.registry import ToolResult
from agent.tools.registry import build_default_tool_registry


INTRINSICS = {"fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 32.0, "scale": 1000.0}


def _image(path: Path, *, depth: bool = False) -> str:
    Image.new("I;16" if depth else "L", (64, 64), 500 if depth else 255).save(path)
    return str(path)


def _extrinsics(x: float) -> dict[str, Any]:
    return {
        "camera_frame": "opencv",
        "camera_to_world": [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }


def _packet(tmp_path: Path, name: str, mask_key: str, x: float) -> dict[str, Any]:
    rgb = _image(tmp_path / f"{name}-rgb.png")
    depth = _image(tmp_path / f"{name}-depth.png", depth=True)
    mask = _image(tmp_path / f"{name}-mask.png")
    return {
        "rgb": rgb,
        "depth": depth,
        mask_key: {"mask_ref": mask, "source_image": rgb},
        "intrinsics": dict(INTRINSICS),
        "camera_extrinsics": _extrinsics(x),
        "camera_frame_id": name,
    }


def _parameters(tmp_path: Path) -> dict[str, Any]:
    return {
        "object_observation": _packet(tmp_path, "object", "object_mask", 0.1),
        "placement_observation": _packet(tmp_path, "placement", "placement_region_mask", 0.3),
        "scene_revision": 2,
    }


def _object_depth_tail(
    parameters: dict[str, Any],
    *,
    main_count: int = 400,
    sparse_count: int = 5,
) -> None:
    packet = parameters["object_observation"]
    depth = Image.new("I;16", (64, 64), 0)
    mask = Image.new("L", (64, 64), 0)
    main_pixels = [(index % 32, index // 32) for index in range(main_count)]
    sparse_pixels = [(40 + index % 20, 40 + index // 20) for index in range(sparse_count)]
    for pixel in main_pixels:
        depth.putpixel(pixel, 750)
        mask.putpixel(pixel, 255)
    for pixel in sparse_pixels:
        depth.putpixel(pixel, 900)
        mask.putpixel(pixel, 255)
    depth.save(packet["depth"])
    mask.save(packet["object_mask"]["mask_ref"])


def _mask_point_count(image: Image.Image | str) -> int:
    if isinstance(image, str):
        with Image.open(image) as loaded:
            return int(np.count_nonzero(np.asarray(loaded.convert("L"))))
    return int(np.count_nonzero(np.asarray(image.convert("L"))))


def _response(count: int = 10) -> dict[str, Any]:
    return {
        "success": True,
        "content": "ok",
        "details": {
            "backend": "anyplace_mcp",
            "model": "anyplace_multitask",
            "frame": "placement_camera",
            "camera_frame": "opencv",
            "candidate_count": count,
            "object_current_pose": {
                "frame": "world",
                "translation_xyz": [0.2, 0.0, 0.43],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            "placement_candidates": [
                {
                    "id": f"placement_{index:03d}",
                    "object_placement_transform": {
                        "frame": "placement_camera",
                        "camera_frame": "opencv",
                        "convention": "p_placed = R @ p_current + t",
                        "transform_matrix": [[1, 0, 0, index / 100], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    },
                }
                for index in range(count)
            ],
            "metadata": {"configured_candidate_count": count},
        },
    }


def _context(parameters: dict[str, Any]) -> ToolExecutionContext:
    return ToolExecutionContext(
        name="anyplace",
        spec=build_default_tool_registry().get("anyplace"),
        parameters=parameters,
        metadata={"session_id": "test"},
    )


def test_handler_encodes_two_independent_observations(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    result = build_anyplace_handler(
        lambda request: calls.append(request) or _response(), output_root=tmp_path / "runs"
    )(_context(_parameters(tmp_path)))
    assert result.success is True
    assert set(calls[0]) == {
        "object_observation", "placement_observation", "object_camera_to_placement_camera",
        "placement_camera_to_world",
    }
    assert calls[0]["object_camera_to_placement_camera"][0][3] == pytest.approx(-0.2)
    assert "object_mask" in calls[0]["object_observation"]
    assert "mask" not in calls[0]["object_observation"]
    assert "placement_region_mask" in calls[0]["placement_observation"]
    assert "mask" not in calls[0]["placement_observation"]
    assert "selected_grasp" not in str(calls[0])
    assert result.details["candidate_count"] == 10
    assert "object_mask_preprocessing" not in result.details
    assert all("place_grasp_pose" not in candidate for candidate in result.details["placement_candidates"])
    assert Path(result.details["candidate_image_ref"]).is_file()


def test_handler_removes_one_tiny_isolated_object_depth_tail(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    _object_depth_tail(parameters)
    source_mask = parameters["object_observation"]["object_mask"]["mask_ref"]
    calls: list[dict[str, Any]] = []

    result = build_anyplace_handler(
        lambda request: calls.append(request) or _response(96),
        output_root=tmp_path / "runs",
    )(_context(parameters))

    assert result.success is True
    assert result.details["candidate_count"] == 96
    cleanup = result.details["object_mask_preprocessing"]
    assert cleanup == {
        "schema": "openeta.anyplace.object_mask_depth_cleanup.v1",
        "applied": True,
        "source_mask_ref": source_mask,
        "filtered_mask_ref": cleanup["filtered_mask_ref"],
        "valid_depth_points_before": 405,
        "valid_depth_points_after": 400,
        "removed_depth_points": 5,
        "removed_tail": "far",
        "depth_gap_m": 0.15,
        "depth_boundary_m": 0.825,
        "limits": {
            "minimum_gap_m": 0.03,
            "maximum_sparse_fraction": 0.02,
            "maximum_sparse_points": 32,
            "minimum_retained_points": 128,
        },
    }
    assert cleanup["filtered_mask_ref"] != source_mask
    assert Path(cleanup["filtered_mask_ref"]).is_file()
    assert _mask_point_count(source_mask) == 405
    encoded_mask = calls[0]["object_observation"]["object_mask"]["base64"]
    with Image.open(io.BytesIO(base64.b64decode(encoded_mask))) as filtered:
        assert _mask_point_count(filtered) == 400
    assert result.details["artifacts"][1]["type"] == "object_mask_depth_cleanup"


def test_handler_preserves_non_tiny_object_depth_cluster(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    _object_depth_tail(parameters, sparse_count=20)
    calls: list[dict[str, Any]] = []

    result = build_anyplace_handler(
        lambda request: calls.append(request) or _response(),
        output_root=tmp_path / "runs",
    )(_context(parameters))

    assert result.success is True
    assert "object_mask_preprocessing" not in result.details
    encoded_mask = calls[0]["object_observation"]["object_mask"]["base64"]
    with Image.open(io.BytesIO(base64.b64decode(encoded_mask))) as preserved:
        assert _mask_point_count(preserved) == 420


def test_handler_preserves_ambiguous_depth_tails(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    _object_depth_tail(parameters)
    packet = parameters["object_observation"]
    with Image.open(packet["depth"]) as source_depth:
        depth = source_depth.copy()
    with Image.open(packet["object_mask"]["mask_ref"]) as source_mask:
        mask = source_mask.copy()
    for index in range(5):
        pixel = (50 + index, 20)
        depth.putpixel(pixel, 600)
        mask.putpixel(pixel, 255)
    depth.save(packet["depth"])
    mask.save(packet["object_mask"]["mask_ref"])

    result = build_anyplace_handler(
        lambda _: _response(),
        output_root=tmp_path / "runs",
    )(_context(parameters))

    assert result.success is True
    assert "object_mask_preprocessing" not in result.details


def test_handler_does_not_preprocess_when_frozen_pool_short_circuits(
    tmp_path: Path,
) -> None:
    parameters = _parameters(tmp_path)
    _object_depth_tail(parameters)
    sentinel = ToolResult(False, "frozen", {"reason": "frozen"})

    result = build_anyplace_handler(
        lambda _: pytest.fail("must not call MCP"),
        output_root=tmp_path / "runs",
        pre_inference=lambda _context, _request: sentinel,
    )(_context(parameters))

    assert result is sentinel
    assert not (tmp_path / "runs" / "test" / "preprocessing").exists()


def test_handler_allows_distinct_rgbd_packets(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    assert parameters["object_observation"]["rgb"] != parameters["placement_observation"]["rgb"]
    result = build_anyplace_handler(lambda _: _response(), output_root=tmp_path / "runs")(
        _context(parameters)
    )
    assert result.success is True


def test_handler_replays_its_frozen_normalized_observations(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    handler = build_anyplace_handler(
        lambda request: calls.append(request) or _response(),
        output_root=tmp_path / "runs",
    )
    first = handler(_context(_parameters(tmp_path)))

    frozen = first.details["source"]
    second = handler(
        _context(
            {
                "object_observation": frozen["object_observation"],
                "placement_observation": frozen["placement_observation"],
                "scene_revision": 2,
            }
        )
    )

    assert first.success is True
    assert second.success is True
    assert len(calls) == 2
    assert "object_mask" in calls[1]["object_observation"]
    assert "placement_region_mask" in calls[1]["placement_observation"]


def test_handler_rejects_mask_not_aligned_with_its_own_rgb(tmp_path: Path) -> None:
    parameters = _parameters(tmp_path)
    parameters["placement_observation"]["placement_region_mask"]["source_image"] = parameters["object_observation"]["rgb"]
    result = build_anyplace_handler(lambda _: pytest.fail("must not call MCP"))(_context(parameters))
    assert result.success is False
    assert result.details["reason"] == "invalid_independent_observation"


@pytest.mark.parametrize("packet", ["object_observation", "placement_observation"])
def test_handler_rejects_missing_packet_files(tmp_path: Path, packet: str) -> None:
    parameters = _parameters(tmp_path)
    parameters[packet]["depth"] = str(tmp_path / "missing.png")
    result = build_anyplace_handler(lambda _: pytest.fail("must not call MCP"))(_context(parameters))
    assert result.success is False
    assert result.details["reason"] == "invalid_independent_observation"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: response["details"].update(candidate_count=9),
        lambda response: response["details"]["placement_candidates"][1].update(id="placement_000"),
        lambda response: response["details"]["placement_candidates"][0]["object_placement_transform"]["transform_matrix"].__setitem__(3, [0, 0, 0, 2]),
        lambda response: response["details"]["placement_candidates"][0].update(place_grasp_pose={}),
    ],
)
def test_handler_rejects_malformed_or_legacy_outputs(tmp_path: Path, mutate) -> None:
    response = _response()
    mutate(response)
    result = build_anyplace_handler(lambda _: response, output_root=tmp_path / "runs")(
        _context(_parameters(tmp_path))
    )
    assert result.success is False
    assert result.details["reason"] == "inconsistent_placement_outputs"


def test_handler_preserves_structured_backend_failure(tmp_path: Path) -> None:
    result = build_anyplace_handler(
        lambda _: {"success": False, "content": "failed", "details": {"reason": "model_inference_failed", "metadata": {}}}
    )(_context(_parameters(tmp_path)))
    assert result.success is False
    assert result.details["reason"] == "model_inference_failed"
