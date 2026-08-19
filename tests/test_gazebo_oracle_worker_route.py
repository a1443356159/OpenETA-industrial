"""Contract tests for the worker oracle_perceive route and the MCP forwarding tool."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import types

import numpy as np
import pytest
from PIL import Image

import sim.bench_worker as bench_worker
from extensions.gazebo.profiles import gazebo_profile
from sim.mcp_server import server


_TOP_EXTRINSICS = {
    "frame_transform": "camera_to_world",
    "camera_frame": "opencv",
    "pos": [0.0, 0.0, 1.8],
    "quat_xyzw": [0.7071067812, -0.7071067812, 0.0, 0.0],
}
_INTRINSICS = {"fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
_WRIST_EXTRINSICS = {"frame_transform": "tf_dynamic", "camera_frame": "opencv"}


def _png_base64(rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _fake_env():
    return types.SimpleNamespace(profile=gazebo_profile("rm75_robotiq2f85_pickplace"))


def _objects() -> list[dict]:
    return [
        {
            "id": "target_object", "name": "target_object", "label": "target block",
            "position": [0.28, -0.10, 0.43], "orientation": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "id": "distractor_object", "name": "distractor_object", "label": "distractor cylinder",
            "position": [0.28, 0.12, 0.44], "orientation": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "id": "work_table", "name": "work_table", "label": "table",
            "position": [0.40, 0.0, 0.38], "orientation": [0.0, 0.0, 0.0, 1.0],
        },
    ]


def _seed_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    handle: str = "h1",
    top_rgb: np.ndarray | None = None,
    wrist_rgb: np.ndarray | None = None,
    with_objects: bool = True,
) -> dict[str, np.ndarray]:
    """Install a fake m3 env plus a cached raw observation on the worker."""
    frames = {
        "top": top_rgb if top_rgb is not None else np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist": wrist_rgb if wrist_rgb is not None else np.zeros((240, 320, 3), dtype=np.uint8),
    }
    obs = {
        "task": "pick",
        "cameras": {
            "top_camera_optical_frame": {
                "rgb": frames["top"],
                "intrinsics": dict(_INTRINSICS),
                "extrinsics": dict(_TOP_EXTRINSICS),
                "role": "scene_primary",
            },
            "wrist_camera_optical_frame": {
                "rgb": frames["wrist"],
                "intrinsics": {"fx": 525.0, "fy": 525.0, "cx": 160.0, "cy": 120.0,
                               "width": 320, "height": 240},
                "extrinsics": dict(_WRIST_EXTRINSICS),
                "role": "wrist",
            },
        },
        "robot": {},
        "objects": _objects() if with_objects else [],
        "metadata": {},
    }
    monkeypatch.setitem(bench_worker._envs, handle, _fake_env())
    monkeypatch.setitem(bench_worker._last_obs, handle, obs)
    return frames


def test_worker_route_pixel_match_projects_ground_truth_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _seed_worker(monkeypatch)
    result = bench_worker._oracle_perceive_frame(
        bench_worker._envs["h1"], "h1",
        {"image_base64": _png_base64(frames["top"]), "prompt": "target block"},
    )
    assert result["success"] is True
    details = result["details"]
    assert details["detection_count"] == 1
    detection = details["detections"][0]
    assert detection["label"] == "target block"
    assert detection["score"] == 1.0
    metadata = details["metadata"]
    assert metadata["perception_source"] == "gazebo_oracle"
    assert metadata["frame_match"] == "pixel"
    assert metadata["camera_frame_id"] == "top_camera_optical_frame"
    # Mask geometry: the target sits 1.37 m under the top camera, slightly
    # off-axis; the 4x4x6 cm box must cover a small patch near the principal
    # point and the bbox must be recomputable from the mask.
    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(detection["mask"]["base64"]))))
    assert mask.shape == (480, 640)
    ys, xs = np.nonzero(mask)
    assert detection["bbox_xyxy"] == [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    assert detection["area_px"] == int((mask > 0).sum())
    assert abs(float(xs.mean()) - 320.0) < 40.0


def test_worker_route_prompt_without_match_returns_zero_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _seed_worker(monkeypatch)
    result = bench_worker._oracle_perceive_frame(
        bench_worker._envs["h1"], "h1",
        {"image_base64": _png_base64(frames["top"]), "prompt": "nonexistent object"},
    )
    assert result["success"] is True
    assert result["details"]["detection_count"] == 0


def test_worker_route_falls_back_to_unique_size_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _seed_worker(monkeypatch)
    altered = frames["top"].copy()
    altered[0, 0] = [255, 255, 255]
    result = bench_worker._oracle_perceive_frame(
        bench_worker._envs["h1"], "h1",
        {"image_base64": _png_base64(altered), "prompt": "target"},
    )
    assert result["success"] is True
    assert result["details"]["metadata"]["frame_match"] == "fallback_size"
    assert result["details"]["detection_count"] == 1


def test_worker_route_rejects_ambiguous_size_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    same_size_wrist = np.zeros((480, 640, 3), dtype=np.uint8)
    same_size_wrist[10, 10] = [9, 9, 9]
    _seed_worker(monkeypatch, wrist_rgb=same_size_wrist)
    query = np.full((480, 640, 3), 3, dtype=np.uint8)
    result = bench_worker._oracle_perceive_frame(
        bench_worker._envs["h1"], "h1",
        {"image_base64": _png_base64(query), "prompt": "target"},
    )
    assert result["success"] is False
    assert result["details"]["reason"] == "frame_match_ambiguous"


def test_worker_route_rejects_wrist_frame_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _seed_worker(monkeypatch)
    result = bench_worker._oracle_perceive_frame(
        bench_worker._envs["h1"], "h1",
        {"image_base64": _png_base64(frames["wrist"]), "prompt": "target"},
    )
    assert result["success"] is False
    details = result["details"]
    assert details["reason"] == "ORACLE_FRAME_UNSUPPORTED"
    assert details["detections"] == []
    assert details["metadata"]["perception_source"] == "gazebo_oracle"
    assert details["metadata"]["camera_frame_id"] == "wrist_camera_optical_frame"


def test_worker_route_fails_without_cached_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(bench_worker._envs, "h1", _fake_env())
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    result = bench_worker._oracle_perceive_frame(
        bench_worker._envs["h1"], "h1",
        {"image_base64": _png_base64(rgb), "prompt": "target"},
    )
    assert result["success"] is False
    assert result["details"]["reason"] == "observation_unavailable"


def test_worker_route_fails_on_envs_without_oracle_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _seed_worker(monkeypatch)
    env = types.SimpleNamespace(profile=gazebo_profile("rgbd_observation"))
    monkeypatch.setitem(bench_worker._envs, "h1", env)
    result = bench_worker._oracle_perceive_frame(
        env, "h1",
        {"image_base64": _png_base64(frames["top"]), "prompt": "target"},
    )
    assert result["success"] is False
    assert result["details"]["reason"] == "oracle_unsupported_env"


def test_worker_route_fails_on_missing_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_worker(monkeypatch)
    env = bench_worker._envs["h1"]
    missing_image = bench_worker._oracle_perceive_frame(env, "h1", {"prompt": "target"})
    assert missing_image["success"] is False
    assert missing_image["details"]["reason"] == "missing_image"
    missing_prompt = bench_worker._oracle_perceive_frame(
        env, "h1", {"image_base64": _png_base64(np.zeros((480, 640, 3), dtype=np.uint8))},
    )
    assert missing_prompt["success"] is False
    assert missing_prompt["details"]["reason"] == "missing_prompt"
    bad_image = bench_worker._oracle_perceive_frame(
        env, "h1", {"image_base64": "not-base64!!!", "prompt": "target"},
    )
    assert bad_image["success"] is False
    assert bad_image["details"]["reason"] == "image_decode_failed"


class _FakeRequest:
    def __init__(self, handle: str, body: dict | None = None) -> None:
        self.path_params = {"handle": handle}
        self.method = "POST"
        self._body = body

    async def body(self) -> bytes:
        return json.dumps(self._body or {}).encode("utf-8")


def test_worker_http_route_returns_400_for_unknown_handle() -> None:
    response = asyncio.run(bench_worker.oracle_perceive(_FakeRequest("nope", {"prompt": "x"})))
    assert response.status_code == 400
    assert "Unknown handle" in json.loads(response.body)["error"]


def test_worker_http_route_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _seed_worker(monkeypatch)
    request = _FakeRequest("h1", {"image_base64": _png_base64(frames["top"]), "prompt": "distractor"})
    response = asyncio.run(bench_worker.oracle_perceive(request))
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["success"] is True
    assert payload["details"]["detection_count"] == 1
    assert payload["details"]["detections"][0]["label"] == "distractor cylinder"


def test_mcp_oracle_perceive_forwards_to_worker_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = {"worker_url": "http://worker", "remote_handle": "rh1"}
    monkeypatch.setitem(server._session_envs, "s1", {"h1": meta})
    captured: dict = {}

    def fake_proxy(meta_arg, *, image_base64, prompt):
        captured["meta"] = meta_arg
        captured["image_base64"] = image_base64
        captured["prompt"] = prompt
        return {"success": True, "details": {"detection_count": 1}}

    monkeypatch.setattr(server, "_proxy_oracle_perceive", fake_proxy)
    result = asyncio.run(server.oracle_perceive(
        handle="h1", image_base64="QUJD", prompt="target block", session_id="s1",
    ))
    assert result == {"success": True, "details": {"detection_count": 1}}
    assert captured == {"meta": meta, "image_base64": "QUJD", "prompt": "target block"}


def test_mcp_oracle_perceive_unknown_handle() -> None:
    result = asyncio.run(server.oracle_perceive(
        handle="nope", image_base64="QUJD", prompt="target", session_id="no-session",
    ))
    assert result == {"error": "Unknown: nope"}


def test_mcp_oracle_perceive_worker_mgr_builds_route_and_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sim.mcp_server import worker_mgr

    captured: dict = {}

    class FakeMgr:
        def proxy_handle_op(self, meta, path, method="GET", body=None):
            captured["meta"] = meta
            captured["path"] = path
            captured["method"] = method
            captured["body"] = body
            return {"ok": True}

    monkeypatch.setattr(worker_mgr, "_get_mgr", lambda: FakeMgr())
    meta = {"worker_url": "http://worker", "remote_handle": "rh1"}
    result = worker_mgr._proxy_oracle_perceive(meta, image_base64="QUJD", prompt="target")
    assert result == {"ok": True}
    assert captured["path"] == "/env/rh1/oracle_perceive"
    assert captured["method"] == "POST"
    assert captured["body"] == {"image_base64": "QUJD", "prompt": "target"}
