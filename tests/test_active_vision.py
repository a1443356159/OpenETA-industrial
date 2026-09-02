from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.backends.planner import CallablePlannerBackend
from agent.runtime.qualification_v3 import schedule_candidate_waves
from agent.runtime.reference_localization import BackendSemanticPointLocalizer
from agent.tools.active_vision import ActiveVisionController
from agent.tools.registry import (
    TOOL_PROFILE_GAZEBO_INDUSTRIAL,
    ToolExecutionContext,
    ToolResult,
    apply_tool_profile,
    build_default_tool_registry,
)


def _save_rgb(path: Path, *, black: bool = False) -> None:
    value = 0 if black else 96
    Image.fromarray(np.full((120, 160, 3), value, dtype=np.uint8)).save(path)


def _save_depth(path: Path, *, millimetres: int = 0, mask: np.ndarray | None = None) -> None:
    depth = np.zeros((120, 160), dtype=np.uint16)
    if mask is None:
        depth[:] = millimetres
    else:
        depth[mask] = millimetres
    Image.fromarray(depth).save(path)


def _camera(
    frame_id: str,
    *,
    role: str,
    position: list[float],
    quaternion: list[float],
) -> CameraFrame:
    return CameraFrame(
        frame_id=frame_id,
        rgb=[],
        depth=None,
        role=role,
        intrinsics={
            "fx": 120.0,
            "fy": 120.0,
            "cx": 80.0,
            "cy": 60.0,
            "width": 160,
            "height": 120,
            "scale": 1000.0,
        },
        extrinsics={
            "camera_frame": "opencv",
            "frame_transform": "camera_to_world",
            "pos": position,
            "quat_xyzw": quaternion,
        },
    )


def _observation(
    *,
    top_rgb: Path,
    top_depth: Path,
    wrist_rgb: Path,
    wrist_depth: Path,
    wrist_position: list[float] | None = None,
    wrist_quaternion: list[float] | None = None,
) -> EnvObservation:
    top = _camera(
        "top_camera_optical_frame",
        role="scene_primary",
        position=[0.3, 0.0, 1.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
    )
    wrist = _camera(
        "wrist_camera_optical_frame",
        role="wrist_primary",
        position=wrist_position or [0.0, 0.0, 0.8],
        quaternion=wrist_quaternion or [0.0, 0.0, 0.0, 1.0],
    )
    artifacts = [
        {
            "kind": "rgb",
            "frame_id": top.frame_id,
            "role": top.role,
            "path": str(top_rgb),
        },
        {
            "kind": "depth",
            "frame_id": top.frame_id,
            "role": top.role,
            "path": str(top_depth),
        },
        {
            "kind": "rgb",
            "frame_id": wrist.frame_id,
            "role": wrist.role,
            "path": str(wrist_rgb),
        },
        {
            "kind": "depth",
            "frame_id": wrist.frame_id,
            "role": wrist.role,
            "path": str(wrist_depth),
        },
    ]
    return EnvObservation(
        task="pick target",
        cameras=[top, wrist],
        robot=RobotState(
            joint_positions=[0.0] * 7,
            end_effector_pose={
                "frame": "gripper_mount_link",
                "xyz": [0.0, 0.0, 0.9],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            gripper_state={"open": True, "openness": 1.0},
        ),
        metadata={
            "scene_epoch": 1,
            "planning_scene_revision": 7,
            "image_artifacts": artifacts,
            "physical_verification": {"grasp_confirmed": False},
        },
    )


def _write_evidence(
    artifact_root: Path,
    *,
    result_id: str,
    source_rgb: Path,
    mask_path: Path,
    area_px: int,
) -> None:
    root = artifact_root / "sam3_results" / "session" / result_id
    root.mkdir(parents=True)
    (root / "tool_result.json").write_text(
        json.dumps(
            {
                "success": True,
                "details": {
                    "result_id": result_id,
                    "source_image": str(source_rgb),
                    "source_frame_id": "top_camera_optical_frame",
                    "semantic_role": "grasp_target",
                    "semantic_target": "yellow wrench",
                    "perception_bundle_id": "perception-current",
                    "observation_id": "observation-current",
                    "scene_epoch": 1,
                    "selected_detection": {
                        "id": "detection_000",
                        "label": "yellow wrench",
                        "mask_ref": str(mask_path),
                        "area_px": area_px,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _controller(
    tmp_path: Path,
    *,
    proxy: Any = None,
    sam3_handler: Any = None,
    semantic_localizer: Any = None,
) -> ActiveVisionController:
    tools = build_default_tool_registry()
    return ActiveVisionController(
        artifact_root=tmp_path / "artifacts",
        candidate_qualifier=None,
        simulator_proxy=proxy,
        sam3_handler=sam3_handler,
        semantic_localizer=semantic_localizer,
        move_spec=tools.get("move_to"),
        observe_spec=tools.get("observe"),
        sam3_spec=tools.get("sam3"),
    )


def _context(controller: ActiveVisionController, observation: EnvObservation) -> ToolExecutionContext:
    tools = build_default_tool_registry()
    return ToolExecutionContext(
        name="active_observe",
        spec=tools.get("active_observe"),
        parameters={
            "target_evidence_id": "sam-result-1",
            "semantic_role": "grasp_target",
            "quality_profile": "grasp_rgbd",
            "max_motion_attempts": 2,
        },
        observation=observation,
        metadata={"session_id": "unit-session"},
    )


def _search_context(
    controller: ActiveVisionController,
    observation: EnvObservation,
    *,
    source_image: Path,
) -> ToolExecutionContext:
    tools = build_default_tool_registry()
    return ToolExecutionContext(
        name="active_observe",
        spec=tools.get("active_observe"),
        parameters={
            "semantic_target": "red hex bolt",
            "semantic_role": "grasp_target",
            "quality_profile": "grasp_rgbd",
            "max_motion_attempts": 2,
            "target_hint": {
                "source_image": str(source_image),
                "positive_points": [{"x": 80.0, "y": 60.0, "label": 1}],
                "source": "bounded_visual_point_localization",
            },
        },
        observation=observation,
        metadata={"session_id": "unit-session"},
    )


def test_active_observe_reuses_quality_current_rgbd_without_motion(tmp_path: Path) -> None:
    top_rgb, top_depth = tmp_path / "top.png", tmp_path / "top-depth.png"
    wrist_rgb, wrist_depth = tmp_path / "wrist.png", tmp_path / "wrist-depth.png"
    mask_path = tmp_path / "mask.png"
    _save_rgb(top_rgb)
    _save_rgb(wrist_rgb)
    mask = np.zeros((120, 160), dtype=bool)
    mask[35:75, 55:95] = True
    _save_depth(top_depth, millimetres=980, mask=mask)
    _save_depth(wrist_depth)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    artifact_root = tmp_path / "artifacts"
    _write_evidence(
        artifact_root,
        result_id="sam-result-1",
        source_rgb=top_rgb,
        mask_path=mask_path,
        area_px=int(mask.sum()),
    )
    observation = _observation(
        top_rgb=top_rgb,
        top_depth=top_depth,
        wrist_rgb=wrist_rgb,
        wrist_depth=wrist_depth,
    )
    controller = _controller(tmp_path)

    result = controller.handler(_context(controller, observation))

    assert result.success is True
    assert result.details["outputs"]["status"] == "reused"
    assert result.details["outputs"]["motion_count"] == 0
    assert result.details["outputs"]["quality"]["passed"] is True
    assert Path(result.details["outputs"]["artifact_ref"]).is_file()


def test_active_observe_rejects_fixed_wrist_self_occlusion_before_moveit(tmp_path: Path) -> None:
    top_rgb, top_depth = tmp_path / "top.png", tmp_path / "top-depth.png"
    wrist_rgb, wrist_depth = tmp_path / "wrist.png", tmp_path / "wrist-depth.png"
    mask_path = tmp_path / "mask.png"
    _save_rgb(top_rgb)
    _save_rgb(wrist_rgb, black=True)
    mask = np.zeros((120, 160), dtype=bool)
    mask[50:60, 70:80] = True
    _save_depth(top_depth, millimetres=980, mask=mask)
    _save_depth(wrist_depth, millimetres=80)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    artifact_root = tmp_path / "artifacts"
    _write_evidence(
        artifact_root,
        result_id="sam-result-1",
        source_rgb=top_rgb,
        mask_path=mask_path,
        area_px=int(mask.sum()),
    )
    observation = _observation(
        top_rgb=top_rgb,
        top_depth=top_depth,
        wrist_rgb=wrist_rgb,
        wrist_depth=wrist_depth,
    )
    controller = _controller(tmp_path)

    result = controller.handler(_context(controller, observation))

    assert result.success is False
    assert result.details["outputs"]["status"] == "exhausted"
    assert result.details["outputs"]["stop_reason"] == "camera_self_occlusion_unusable"
    assert result.details["outputs"]["candidate_counts"]["generated"] == 0


class _PassQualifier:
    def qualify_result(self, result: ToolResult, **_: Any) -> ToolResult:
        candidates = list(result.details["observation_candidates"])
        result.details["observation_candidates"] = candidates[:2]
        result.details["qualification_profile"] = "fast_v3"
        result.details["solver_profile"] = "kdl_fast"
        result.details["qualification_stop_reason"] = "complete_l5_pass_found"
        result.details["qualification_evidence"] = {"infrastructure_error": False}
        result.details["qualification_artifact"] = {"path": "qualification.json"}
        return result


class _Proxy:
    def __init__(self, observation: EnvObservation) -> None:
        self.observation = observation
        self.calls: list[str] = []

    def call(self, _: ToolExecutionContext, *, tool_name: str | None = None) -> ToolResult:
        self.calls.append(str(tool_name))
        if tool_name == "move_to":
            return ToolResult(True, details={"outputs": {}, "diagnostics": []})
        return ToolResult(
            True,
            details={
                "outputs": {},
                "diagnostics": [],
                "environment_receipt": {
                    "observation_fresh": True,
                    "observation_snapshot": {"observation": self.observation.to_dict()},
                },
            },
        )


def test_active_observe_acquires_point_grounded_view_and_caches_sam3(tmp_path: Path) -> None:
    top_rgb, top_depth = tmp_path / "top.png", tmp_path / "top-depth.png"
    wrist_rgb, wrist_depth = tmp_path / "wrist.png", tmp_path / "wrist-depth.png"
    new_rgb, new_depth = tmp_path / "new-wrist.png", tmp_path / "new-wrist-depth.png"
    old_mask_path, new_mask_path = tmp_path / "old-mask.png", tmp_path / "new-mask.png"
    for path in (top_rgb, wrist_rgb, new_rgb):
        _save_rgb(path)
    old_mask = np.zeros((120, 160), dtype=bool)
    old_mask[50:70, 70:90] = True
    new_mask = np.zeros((120, 160), dtype=bool)
    new_mask[35:85, 60:100] = True
    _save_depth(top_depth, millimetres=980, mask=old_mask)
    _save_depth(wrist_depth)
    _save_depth(new_depth, millimetres=380, mask=new_mask)
    Image.fromarray((old_mask * 255).astype(np.uint8)).save(old_mask_path)
    Image.fromarray((new_mask * 255).astype(np.uint8)).save(new_mask_path)
    artifact_root = tmp_path / "artifacts"
    _write_evidence(
        artifact_root,
        result_id="sam-result-1",
        source_rgb=top_rgb,
        mask_path=old_mask_path,
        area_px=int(old_mask.sum()),
    )
    initial = _observation(
        top_rgb=top_rgb,
        top_depth=top_depth,
        wrist_rgb=wrist_rgb,
        wrist_depth=wrist_depth,
    )
    acquired = _observation(
        top_rgb=top_rgb,
        top_depth=top_depth,
        wrist_rgb=new_rgb,
        wrist_depth=new_depth,
        wrist_position=[0.3, 0.0, 0.4],
        wrist_quaternion=[1.0, 0.0, 0.0, 0.0],
    )
    proxy = _Proxy(acquired)
    sam_calls: list[dict[str, Any]] = []

    def sam3_handler(context: ToolExecutionContext) -> ToolResult:
        sam_calls.append(dict(context.parameters))
        detection = {
            "id": "detection_000",
            "label": "yellow wrench",
            "mask_ref": str(new_mask_path),
            "bbox_xyxy": [60, 35, 100, 85],
        }
        return ToolResult(
            True,
            details={
                "result_id": "active-sam-result",
                "source_image": str(new_rgb),
                "source_frame_id": "wrist_camera_optical_frame",
                "semantic_target": "yellow wrench",
                "perception_bundle_id": "perception-active",
                "observation_id": "observation-active",
                "detections": [detection],
                "selected_detection": detection,
            },
        )

    controller = _controller(tmp_path, proxy=proxy, sam3_handler=sam3_handler)
    controller.qualifier = _PassQualifier()  # type: ignore[assignment]

    first = controller.handler(_context(controller, initial))
    second = controller.handler(_context(controller, initial))

    assert first.success is True
    assert second.success is True
    assert first.details["outputs"]["status"] == "acquired"
    assert first.details["outputs"]["motion_count"] == 1
    assert first.details["outputs"]["observation_bundle_id"] == "perception-active"
    assert proxy.calls == ["move_to", "observe", "move_to", "observe"]
    assert len(sam_calls) == 1
    assert sam_calls[0]["mode"] == "points"
    assert sam_calls[0]["semantic_role"] == "grasp_target"


def test_active_observe_searches_from_calibrated_visual_point_without_mask(
    tmp_path: Path,
) -> None:
    top_rgb, top_depth = tmp_path / "top.png", tmp_path / "top-depth.png"
    wrist_rgb, wrist_depth = tmp_path / "wrist.png", tmp_path / "wrist-depth.png"
    new_rgb, new_depth = tmp_path / "new-wrist.png", tmp_path / "new-wrist-depth.png"
    new_mask_path = tmp_path / "new-mask.png"
    for path in (top_rgb, wrist_rgb, new_rgb):
        _save_rgb(path)
    top_target = np.zeros((120, 160), dtype=bool)
    top_target[56:65, 76:85] = True
    new_mask = np.zeros((120, 160), dtype=bool)
    new_mask[35:85, 60:100] = True
    _save_depth(top_depth, millimetres=980, mask=top_target)
    _save_depth(wrist_depth)
    _save_depth(new_depth, millimetres=380, mask=new_mask)
    Image.fromarray((new_mask * 255).astype(np.uint8)).save(new_mask_path)
    initial = _observation(
        top_rgb=top_rgb,
        top_depth=top_depth,
        wrist_rgb=wrist_rgb,
        wrist_depth=wrist_depth,
    )
    acquired = _observation(
        top_rgb=top_rgb,
        top_depth=top_depth,
        wrist_rgb=new_rgb,
        wrist_depth=new_depth,
        wrist_position=[0.3, 0.0, 0.4],
        wrist_quaternion=[1.0, 0.0, 0.0, 0.0],
    )
    proxy = _Proxy(acquired)
    sam_calls: list[dict[str, Any]] = []

    def sam3_handler(context: ToolExecutionContext) -> ToolResult:
        sam_calls.append(dict(context.parameters))
        detection = {
            "id": "detection_000",
            "label": "red hex bolt",
            "mask_ref": str(new_mask_path),
            "bbox_xyxy": [60, 35, 100, 85],
        }
        return ToolResult(
            True,
            details={
                "result_id": "active-search-result",
                "source_image": str(new_rgb),
                "source_frame_id": "wrist_camera_optical_frame",
                "semantic_role": "grasp_target",
                "semantic_target": "red hex bolt",
                "perception_bundle_id": "perception-active-search",
                "observation_id": "observation-active-search",
                "segmentation_mode": "point_prompt",
                "scene_epoch": 1,
                "attempt_id": "active-point-attempt",
                "attempt_fingerprint": "active-point-fingerprint",
                "detections": [detection],
                "selected_detection": detection,
            },
        )

    controller = _controller(tmp_path, proxy=proxy, sam3_handler=sam3_handler)
    controller.qualifier = _PassQualifier()  # type: ignore[assignment]

    result = controller.handler(
        _search_context(controller, initial, source_image=top_rgb)
    )

    assert result.success is True
    assert result.details["outputs"]["status"] == "acquired"
    assert result.details["outputs"]["active_vision_mode"] == "semantic_search"
    assert result.details["outputs"]["detections"][0]["label"] == "red hex bolt"
    assert result.details["outputs"]["selected_detection"]["id"] == "detection_000"
    assert result.details["outputs"]["active_vision_attempt_id"].startswith(
        "active-observe-"
    )
    assert proxy.calls == ["move_to", "observe"]
    assert len(sam_calls) == 1
    assert sam_calls[0]["mode"] == "points"
    assert sam_calls[0]["semantic_target"] == "red hex bolt"


def test_active_observe_can_seed_search_with_isolated_provider_point(
    tmp_path: Path,
) -> None:
    top_rgb, top_depth = tmp_path / "top.png", tmp_path / "top-depth.png"
    wrist_rgb, wrist_depth = tmp_path / "wrist.png", tmp_path / "wrist-depth.png"
    _save_rgb(top_rgb)
    _save_rgb(wrist_rgb)
    target_depth = np.zeros((120, 160), dtype=bool)
    target_depth[56:65, 76:85] = True
    _save_depth(top_depth, millimetres=980, mask=target_depth)
    _save_depth(wrist_depth)
    observation = _observation(
        top_rgb=top_rgb,
        top_depth=top_depth,
        wrist_rgb=wrist_rgb,
        wrist_depth=wrist_depth,
    )
    requests = []

    def localize(request):
        requests.append(request)
        return {
            "decision": "locate",
            "point": {"x": 80.0, "y": 60.0},
            "bbox_xyxy": [76.0, 56.0, 85.0, 65.0],
            "confidence": 0.87,
            "reason": "unique red bolt material remains visible",
        }

    semantic_localizer = BackendSemanticPointLocalizer(
        CallablePlannerBackend(localize, provider="fixture-vlm", model="fixture-vision")
    )
    controller = _controller(tmp_path, semantic_localizer=semantic_localizer)
    context = _search_context(controller, observation, source_image=top_rgb)
    context.parameters.pop("target_hint")

    result = controller.handler(context)

    assert result.success is False
    assert result.details["outputs"]["status"] == "infrastructure_error"
    assert result.details["outputs"]["stop_reason"] == (
        "active_vision_motion_backend_unavailable"
    )
    assert len(requests) == 1
    request = requests[0]
    assert request.conversation_messages == []
    assert request.tool_context["semantic_target"] == "red hex bolt"
    assert request.tool_context["scene_image_size"] == {"width": 160, "height": 120}
    assert request.tool_context["vision_image_paths"] == [str(top_rgb.resolve())]
    localization = result.details["outputs"]["semantic_point_localization"]
    assert localization["source"] == "isolated_provider_visual_grounding"
    assert localization["point_xy"] == [80.0, 60.0]
    assert localization["provider"] == "fixture-vlm"
    assert result.details["outputs"]["target_hint"]["source_image"] == str(
        top_rgb.resolve()
    )


def test_active_observe_records_provider_abstention_without_motion(
    tmp_path: Path,
) -> None:
    top_rgb, top_depth = tmp_path / "top.png", tmp_path / "top-depth.png"
    wrist_rgb, wrist_depth = tmp_path / "wrist.png", tmp_path / "wrist-depth.png"
    _save_rgb(top_rgb)
    _save_rgb(wrist_rgb)
    _save_depth(top_depth, millimetres=980)
    _save_depth(wrist_depth)
    observation = _observation(
        top_rgb=top_rgb,
        top_depth=top_depth,
        wrist_rgb=wrist_rgb,
        wrist_depth=wrist_depth,
    )
    semantic_localizer = BackendSemanticPointLocalizer(
        CallablePlannerBackend(
            lambda _: {
                "decision": "abstain",
                "point": None,
                "bbox_xyxy": None,
                "confidence": 0,
                "reason": "target is fully occluded",
            }
        )
    )
    controller = _controller(tmp_path, semantic_localizer=semantic_localizer)
    context = _search_context(controller, observation, source_image=top_rgb)
    context.parameters.pop("target_hint")

    result = controller.handler(context)

    assert result.success is False
    assert result.details["outputs"]["status"] == "exhausted"
    assert result.details["outputs"]["stop_reason"] == (
        "semantic_point_localization_abstained"
    )


def test_gazebo_industrial_profile_hides_irrelevant_tools_without_deleting_specs() -> None:
    registry = build_default_tool_registry()
    registry.bind_handler("observe", lambda _: ToolResult(True), replace=True)
    registry.bind_handler("web_search", lambda _: ToolResult(True), replace=True)

    apply_tool_profile(registry, TOOL_PROFILE_GAZEBO_INDUSTRIAL)

    visible = {spec.name for spec in registry.list()}
    complete = {spec.name for spec in registry.list(include_disabled=True)}
    assert "active_observe" in visible
    assert "configure_work_order" in visible
    assert "web_search" not in visible
    assert "web_search" in complete
    assert registry.can_execute("web_search") is False
    disabled = registry.call("web_search", {"query": "unused"})
    assert disabled.details["diagnostics"] == [{"code": "tool_disabled_by_profile"}]


def test_observation_candidates_use_bounded_deterministic_waves() -> None:
    descriptors = [
        {
            "candidate_id": f"view-{index}",
            "candidate": {
                "id": f"view-{index}",
                "qualification_stages": [
                    {
                        "xyz": [0.3 + index * 0.02, 0.0, 0.4],
                        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            },
        }
        for index in range(10)
    ]

    waves = schedule_candidate_waves(
        descriptors,
        purpose="observation",
        observation_waves=(4, 8, 16),
    )

    assert [len(wave.candidates) for wave in waves] == [4, 4, 2]
    assert [wave.cumulative_per_branch for wave in waves] == [4, 8, 10]
    assert [
        descriptor["candidate_id"]
        for wave in waves
        for descriptor in wave.candidates
    ] == [
        descriptor["candidate_id"]
        for wave in schedule_candidate_waves(
            descriptors,
            purpose="observation",
            observation_waves=(4, 8, 16),
        )
        for descriptor in wave.candidates
    ]
