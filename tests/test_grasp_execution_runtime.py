from __future__ import annotations

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
from agent.runtime.memory import AgentMemory
from agent.runtime import memory as memory_module
from agent.tools.grasp_geometry import DEFAULT_GRASP_PROFILE, compile_grasp_seed

import json
from pathlib import Path


def _candidate(candidate_id: str, score: float) -> dict:
    return {
        "id": candidate_id,
        "frame": "camera",
        "camera_frame": "opencv",
        "score": score,
        "width": 0.06,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def _candidate_with_approach(candidate_id: str, score: float, approach: str) -> dict:
    candidate = _candidate(candidate_id, score)
    candidate["rotation_matrix"] = {
        "top_down": [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        "front": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        "side": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }[approach]
    return candidate


def _record_handle_observation(memory: AgentMemory, *, task: str) -> None:
    memory.add_observation(
        EnvObservation(
            task=task,
            cameras=[
                CameraFrame(
                    frame_id="agentview",
                    rgb=[[[0, 0, 0]]],
                    extrinsics={
                        "pos": [0.0, 0.0, 0.0],
                        "mat": [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                    },
                )
            ],
            robot=RobotState(),
            metadata={
                "image_artifacts": [
                    {
                        "kind": "rgb",
                        "frame_id": "agentview",
                        "path": "tmp/agentview.rgb.png",
                    }
                ]
            },
        )
    )


def _memory_with_articulated_handle_candidates(
    *,
    geometry_family: str = "articulated_handle",
    task: str = "open the middle drawer of the cabinet",
    prompt: str = "middle drawer handle",
) -> AgentMemory:
    memory = AgentMemory()
    memory.start_session(task=task)
    _record_handle_observation(memory, task=task)
    memory.save_fact(
        "selected_sam3_detection",
        {
            "id": "detection_handle",
            "mask_ref": "tmp/handle.mask.png",
            "source_image": "tmp/agentview.rgb.png",
            "target_prompt": prompt,
            "target_geometry_family": geometry_family,
            "scene_epoch": memory.scene_epoch(),
        },
        source="test",
    )
    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "handle-grasps",
                "mode": "targeted",
                "selected_backend": "anygrasp",
                "source_rgb": "tmp/agentview.rgb.png",
                "target_mask": "tmp/handle.mask.png",
                "camera_frame_id": "agentview",
                "source": {
                    "mode": "targeted",
                    "rgb": "tmp/agentview.rgb.png",
                    "object_mask": "tmp/handle.mask.png",
                    "camera_frame_id": "agentview",
                },
                "grasp_candidates": [
                    _candidate_with_approach("top-0", 0.95, "top_down"),
                    _candidate_with_approach("top-1", 0.90, "top_down"),
                    _candidate_with_approach("top-2", 0.85, "top_down"),
                    _candidate_with_approach("top-3", 0.80, "top_down"),
                    _candidate_with_approach("front-0", 0.75, "front"),
                    _candidate_with_approach("side-0", 0.70, "side"),
                ],
            },
        )
    )
    return memory


def _reject_active_candidate(memory: AgentMemory) -> None:
    policy = memory.grasp_candidate_policy()
    active = policy["active_candidate"]
    memory.add_action(
        _tool_action(
            "move_to",
            {"target_pose": {"source_grasp_id": active["id"]}},
            success=False,
            outputs={
                "motion_summary": {
                    "reached_target": False,
                    "end": {"xyz": [0.1, 0.2, 0.3]},
                }
            },
        )
    )


def test_zero_moveit_pass_schedules_fresh_observation_without_backend_switch() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the block")
    memory.add_observation(
        EnvObservation(task="pick the block", cameras=[], robot=RobotState())
    )

    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "g0",
                "selected_backend": "graspgenx",
                "grasp_candidates": [],
                "generated_candidate_count": 10,
                "qualification_evidence": {"results": []},
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["reestimate_required"] == {
        "status": "pending_recovery",
        "reason": "no_moveit_qualified_candidates",
        "backend": "graspgenx",
        "requires_fresh_observation": True,
        "backend_switch_allowed": False,
    }
    recovery = memory.grasp_recovery()
    assert recovery["status"] == "required"
    assert recovery["required_action"] == {"name": "observe", "parameters": {}}


def test_zero_moveit_pass_reestimates_same_grasp_backend_with_fresh_observation() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the block")
    memory.add_observation(
        EnvObservation(task="pick the block", cameras=[], robot=RobotState())
    )
    memory.save_fact(
        "selected_sam3_detection",
        {"target_prompt": "red block"},
        source="test",
    )

    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "g0",
                "selected_backend": "graspgenx",
                "source_rgb": "fresh-grasp.png",
                "camera_frame_id": "top",
                "source": {"rgb": "fresh-grasp.png", "camera_frame_id": "top"},
                "grasp_candidates": [],
                "generated_candidate_count": 10,
                "qualification_evidence": {"results": []},
            },
        )
    )

    reestimate = memory.grasp_reestimation()
    assert reestimate["status"] == "pending_observation"
    assert reestimate["reason"] == "moveit_qualification_zero_pass"
    assert reestimate["source_backend"] == "graspgenx"
    assert reestimate["attempt_count"] == 1


def test_zero_pass_frozen_model_pool_stops_without_model_rerun() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the red block and place it in the green zone")
    memory.add_observation(
        EnvObservation(task="pick and place", cameras=[], robot=RobotState())
    )
    memory.save_fact(
        "placement_object_detection",
        {"target_prompt": "red block"},
        source="test",
    )
    memory.save_fact(
        "placement_region_detection",
        {"target_prompt": "green placement_zone_marker"},
        source="test",
    )
    memory.save_fact(
        "selected_sam3_detection",
        {"target_prompt": "green placement_zone_marker"},
        source="test",
    )
    memory.save_fact(
        "frozen_placement_goal_pool",
        {"status": "ready", "goal_count": 96},
        source="test",
    )

    outputs = {
        "result_id": "g0",
        "selected_backend": "graspgenx",
        "source_rgb": "fresh-grasp.png",
        "camera_frame_id": "top",
        "source": {"rgb": "fresh-grasp.png", "camera_frame_id": "top"},
        "grasp_candidates": [],
        "generated_candidate_count": 10,
        "qualification_evidence": {"results": []},
    }
    outputs.update(
        {
            "frozen_pair_count": 384,
            "frozen_pair_grasp_branch_limit": 4,
            "frozen_pair_lookahead_grasp_count": 4,
            "frozen_pair_full_plan_pass_count": 0,
        }
    )
    memory.add_action(_tool_action("grasp_pose_estimate", {}, outputs=outputs))

    assert memory.grasp_reestimation() is None
    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "stopped_requires_human"
    assert policy["stop_reason"] == "frozen_grasp_place_pool_exhausted"
    assert policy["failure_code"] == "CURRENT_FROZEN_MODEL_POOL_INFEASIBLE"
    assert policy["frozen_pair_count"] == 384
    assert "reestimate_required" not in policy


def test_zero_pass_reestimate_waits_for_a_new_complete_rgbd_packet(
    tmp_path: Path,
) -> None:
    def observation(bundle: str) -> EnvObservation:
        artifacts = []
        cameras = []
        for frame_id in ("top", "wrist"):
            rgb = tmp_path / f"{bundle}.{frame_id}.rgb.png"
            depth = tmp_path / f"{bundle}.{frame_id}.depth.png"
            rgb.write_bytes(b"rgb")
            depth.write_bytes(b"depth")
            artifacts.extend(
                [
                    {"kind": "rgb", "frame_id": frame_id, "path": str(rgb)},
                    {"kind": "depth", "frame_id": frame_id, "path": str(depth)},
                ]
            )
            cameras.append(
                CameraFrame(
                    frame_id=frame_id,
                    rgb=[[[0, 0, 0]]],
                    depth=[[1.0]],
                    intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
                )
            )
        return EnvObservation(
            task="pick and place",
            cameras=cameras,
            robot=RobotState(),
            metadata={"image_artifacts": artifacts},
        )

    memory = AgentMemory()
    memory.start_session(task="pick the red block and place it in the green zone")
    source = observation("source")
    memory.add_observation(source)
    memory.save_fact(
        "selected_sam3_detection",
        {"target_prompt": "red block"},
        source="test",
    )
    source_rgb = str(tmp_path / "source.top.rgb.png")
    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "g0",
                "selected_backend": "graspgenx",
                "source_rgb": source_rgb,
                "camera_frame_id": "top",
                "source": {"rgb": source_rgb, "camera_frame_id": "top"},
                "grasp_candidates": [],
                "generated_candidate_count": 10,
                "qualification_evidence": {"results": []},
            },
        )
    )

    reestimate = memory.grasp_reestimation()
    assert reestimate["status"] == "pending_observation"
    assert set(reestimate["source_observation_rgb_paths"]) == {
        str(tmp_path / "source.top.rgb.png"),
        str(tmp_path / "source.wrist.rgb.png"),
    }

    memory.add_observation(source)
    assert memory.grasp_reestimation()["status"] == "pending_observation"

    memory.add_observation(observation("fresh"))
    reestimate = memory.grasp_reestimation()
    assert reestimate["status"] == "ready"
    assert {view["frame_id"] for view in reestimate["observation_views"]} == {
        "top",
        "wrist",
    }

    fresh_rgb = str(tmp_path / "fresh.top.rgb.png")
    fresh_mask = str(tmp_path / "fresh.mask.png")
    memory.add_action(
        _tool_action(
            "sam3",
            {"image": fresh_rgb, "prompt": "red block"},
            outputs={
                "result_id": "sam3-fresh",
                "source_image": fresh_rgb,
                "prompt": "red block",
                "detections": [{"id": "d0", "mask_ref": fresh_mask}],
            },
        )
    )
    assert memory.grasp_reestimation()["status"] == "selection_pending"
    memory.resolve_sam3_selection(
        result_id="sam3-fresh",
        detection_id="d0",
        selection_source="main_agent_vlm",
    )
    assert memory.grasp_reestimation()["status"] == "target_ready"

    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "g1",
                "selected_backend": "graspgenx",
                "source_rgb": fresh_rgb,
                "camera_frame_id": "top",
                "source": {"rgb": fresh_rgb, "camera_frame_id": "top"},
                "grasp_candidates": [],
                "generated_candidate_count": 10,
                "qualification_evidence": {"results": []},
            },
        )
    )
    reestimate = memory.grasp_reestimation()
    assert reestimate["status"] == "pending_observation"
    assert reestimate["attempt_count"] == 2
    assert reestimate["source_image"] == fresh_rgb


def _tool_action(
    name: str,
    parameters: dict,
    *,
    success: bool = True,
    outputs: dict | None = None,
    grasp_outcome: str = "",
    planner_metadata: dict | None = None,
    environment_receipt: dict | None = None,
) -> EnvAction:
    details = {"parameters": parameters, "outputs": dict(outputs or {})}
    if environment_receipt is not None:
        details["environment_receipt"] = dict(environment_receipt)
    if grasp_outcome:
        details["supervision"] = {"details": {"grasp_outcome": grasp_outcome}}
    return EnvAction(
        action_type="tool_call",
        command={
            "request": {"kind": "tool_call", "name": name, "parameters": parameters},
            "status": "executed" if success else "failed",
            "metadata": {"planner_metadata": dict(planner_metadata or {})},
            "tool_calls": [
                {
                    "name": name,
                    "status": "executed" if success else "failed",
                    "result": {"success": success, "details": details},
                }
            ],
        },
    )


def test_single_post_attach_placement_region_is_retained_without_vlm_mask_copy() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube and place it in green marker")
    memory.save_fact(
        "attachment_gate",
        {"status": "resolved", "verdict": "PASS", "planning_scene_revision": 2},
        source="test",
    )
    memory.add_action(
        _tool_action(
            "sam3",
            {"image": "tmp/placement-rgb.png", "prompt": "green placement_zone_marker"},
            outputs={
                "result_id": "placement-region",
                "prompt": "green placement_zone_marker",
                "source_image": "tmp/placement-rgb.png",
                "frame_id": "top_camera",
                "selection_required": False,
                "detections": [
                    {"id": "detection_000", "mask_ref": "tmp/placement-mask.png"}
                ],
            },
        )
    )

    region = memory.placement_region_detection()
    assert region is not None
    assert region["mask_ref"] == "tmp/placement-mask.png"
    assert region["source_image"] == "tmp/placement-rgb.png"
    assert region["selection_source"] == "host_single_detection"
    # The raw result is still retained for auditable visual selection semantics.
    assert memory.pending_sam3_selection() is not None


def test_single_placement_region_clears_object_detection_from_stale_image() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube and place it in green marker")
    memory.save_fact(
        "attachment_gate",
        {"status": "resolved", "verdict": "PASS", "planning_scene_revision": 2},
        source="test",
    )
    memory.save_fact(
        "placement_object_detection",
        {
            "id": "old-object",
            "mask_ref": "tmp/old-object-mask.png",
            "source_image": "tmp/pre-attach-rgb.png",
        },
        source="test",
    )
    memory.add_action(
        _tool_action(
            "sam3",
            {"image": "tmp/post-attach-rgb.png", "prompt": "green placement_zone_marker"},
            outputs={
                "result_id": "post-attach-region",
                "prompt": "green placement_zone_marker",
                "source_image": "tmp/post-attach-rgb.png",
                "frame_id": "top_camera",
                "selection_required": False,
                "detections": [
                    {"id": "detection_000", "mask_ref": "tmp/post-attach-region.png"}
                ],
            },
        )
    )

    assert memory.placement_region_detection()["source_image"] == "tmp/post-attach-rgb.png"
    assert memory.placement_object_detection() is None




def _native_proof_receipt(*, revision: int = 2, target_id: str = "target_object") -> dict:
    return {
        "ok": True,
        "motion_outcome": "completed",
        "planning_scene_revision": revision,
        "detachable_joint": {"state": "attached"},
        "native_contact_gate": {
            "accepted": True,
            "reason_code": "NATIVE_GRASP_CONTACT_TARGET_CONFIRMED",
            "evidence": {
                "source": "gazebo_native_contacts",
                "target_id": target_id,
            },
        },
        "attachment_transform": {
            "schema_version": "openeta.attachment_transform.v1",
            "parent_frame": "eef",
            "child_frame": "object",
            "translation_xyz": [0.136, 0.0, 0.0],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "measurement_boundary": "native_attach_ack",
        },
        "physical_verification": {
            "schema_version": "openeta.gazebo.native_grasp.v1",
            "verdict": "PASS",
            "reason_code": "NATIVE_GRASP_ATTACHMENT_CONFIRMED",
            "target_id": target_id,
            "grasp_confirmed": True,
        },
    }


def test_native_attachment_proof_requires_exact_identity_and_revision() -> None:
    trusted = memory_module._trusted_native_attachment_proof(
        _native_proof_receipt(),
        planning_scene_revision=2,
    )
    wrong_revision = memory_module._trusted_native_attachment_proof(
        _native_proof_receipt(revision=3),
        planning_scene_revision=2,
    )
    wrong_target = memory_module._trusted_native_attachment_proof(
        _native_proof_receipt(target_id="distractor_object"),
        planning_scene_revision=2,
    )

    assert trusted[0] is True
    assert wrong_revision[:2] == (False, "native_planning_scene_revision_mismatch")
    assert wrong_target[:2] == (False, "native_proof_target_mismatch")


def test_native_attachment_proof_accepts_object_blocked_close_receipt() -> None:
    receipt = _native_proof_receipt()
    contact_gate = receipt.pop("native_contact_gate")
    receipt.pop("motion_outcome")
    receipt.update(
        {
            "reached_goal": False,
            "stalled": True,
            "terminal_status": "not_succeeded",
            "physical_verification": {
                **receipt["physical_verification"],
                "evidence": {"gate": contact_gate},
            },
        }
    )

    trusted = memory_module._trusted_native_attachment_proof(
        receipt,
        planning_scene_revision=2,
    )

    assert trusted[0] is True
    assert trusted[1] == "trusted_native_contact_attach_ack"
    assert trusted[2]["native_contact_gate"] == contact_gate


def test_object_blocked_close_completes_portable_attachment() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(
        _tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled)
    )
    policy = memory.grasp_candidate_policy()
    policy["status"] = "accepted"
    memory.save_fact("grasp_candidate_policy", policy, source="test")
    execution = memory.grasp_execution()
    execution.update(
        {
            "stage": "close",
            "required_action": {
                "name": "gripper_control",
                "parameters": {"position": 0},
            },
        }
    )
    memory.save_fact("grasp_execution", execution, source="test")
    receipt = _native_proof_receipt()
    contact_gate = receipt.pop("native_contact_gate")
    receipt.pop("motion_outcome")
    receipt.update(
        {
            "reached_goal": False,
            "stalled": True,
            "physical_verification": {
                **receipt["physical_verification"],
                "evidence": {"gate": contact_gate},
            },
        }
    )

    memory.add_action(
        _tool_action(
            "gripper_control",
            {"position": 0},
            environment_receipt=receipt,
        )
    )

    assert memory.attachment_gate()["verdict"] == "PASS"
    assert memory.attachment_gate()["status"] == "resolved"
    assert memory.grasp_execution()["status"] == "completed"
    assert memory.grasp_execution()["stage"] == "attached"


def _memory_with_candidates() -> AgentMemory:
    memory = AgentMemory()
    memory.start_session(task="pick up alphabat soup and place it into basket")
    memory.add_action(
        _tool_action(
            "anygrasp",
            {},
            outputs={
                "result_id": "anygrasp-1",
                "grasp_candidates": [
                    _candidate("grasp_001", 0.7),
                    _candidate("grasp_000", 0.9),
                ],
            },
        )
    )
    return memory


def test_observation_scene_epoch_allows_matching_compiled_grasp_capture() -> None:
    memory = _memory_with_candidates()
    memory.add_observation(
        EnvObservation(
            task="pick up alphabat soup and place it into basket",
            cameras=[],
            robot=RobotState(),
            metadata={"scene_epoch": 1},
        )
    )
    compiled = {**_compiled(memory), "scene_epoch": 1}

    memory.add_action(
        _tool_action("compile_grasp_seed", {"scene_epoch": 1}, outputs=compiled)
    )

    assert memory.scene_epoch() == 1
    assert memory.grasp_execution()["compiled_grasp_id"] == compiled["compiled_grasp_id"]
    assert memory.grasp_execution()["scene_epoch"] == 1
    assert any(event.event_type == "scene_epoch_synchronized" for event in memory.events)


def _memory_at_articulated_close() -> AgentMemory:
    memory = AgentMemory()
    memory.start_session(task="open the microwave")
    memory.save_fact(
        "grasp_candidate_policy",
        {
            "status": "accepted",
            "interaction_family": "articulated_handle",
            "active_candidate": {"id": "handle-1", "rank": 0, "score": 0.8},
        },
        source="test",
    )
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "close",
            "candidate_id": "handle-1",
            "compiled_grasp_id": "compiled-1",
            "scene_epoch": memory.scene_epoch(),
            "required_action": {"name": "gripper_control", "parameters": {"position": 0}},
        },
        source="test",
    )
    return memory


def test_articulated_close_routes_to_the_explicit_articulation_probe() -> None:
    memory = _memory_at_articulated_close()

    memory.add_action(_tool_action("gripper_control", {"position": 0}))

    execution = memory.grasp_execution()
    assert execution["stage"] == "prepare_probe"
    assert execution["probe_kind"] == "articulated_attachment"


def test_articulated_probe_pass_keeps_endpoint_and_completes_attachment() -> None:
    memory = _memory_at_articulated_close()
    memory.add_action(_tool_action("gripper_control", {"position": 0}))
    required_action = {
        "name": "move_to",
        "parameters": {
            "target_pose": {
                "frame": "world",
                "xyz": [0.15, 0.2, 0.3],
                "probe_type": "articulated_attachment",
                "source_grasp_id": "handle-1",
                "compiled_grasp_id": "compiled-1",
                "scene_epoch": memory.scene_epoch(),
                "probe_path_sha256": "a" * 64,
            }
        },
    }
    prepared = {
        "schema_version": "openeta.articulated_attachment_probe.v1",
        "status": "prepared",
        "candidate_id": "handle-1",
        "compiled_grasp_id": "compiled-1",
        "scene_epoch": memory.scene_epoch(),
        "motion_type": "linear",
        "distance_m": 0.05,
        "path_sha256": "a" * 64,
        "required_action": required_action,
        "pre_probe_image_paths": ["before-agent.png", "before-wrist.png"],
    }
    memory.add_action(
        _tool_action("prepare_attachment_probe", {}, outputs=prepared)
    )
    assert memory.grasp_execution()["stage"] == "probe"

    memory.add_action(_tool_action(required_action["name"], required_action["parameters"]))

    assert memory.articulated_attachment_probe()["status"] == "completed"
    assert memory.grasp_execution()["stage"] == "attachment"
    assert "pass" not in memory.grasp_execution()["attachment_actions"]

    memory.add_action(
        _tool_action(
            "assess_attachment_probe",
            {},
            outputs={
                "schema_version": "openeta.articulated_attachment_assessment.v1",
                "candidate_id": "handle-1",
                "verdict": "PASS",
                "reason": "handle co-moved",
            },
        )
    )

    assert memory.attachment_gate()["verdict"] == "PASS"
    assert memory.grasp_execution()["status"] == "completed"
    assert memory.grasp_execution()["stage"] == "attached"


def _memory_at_articulated_attachment() -> tuple[AgentMemory, dict]:
    memory = _memory_at_articulated_close()
    memory.add_action(_tool_action("gripper_control", {"position": 0}))
    required_action = {
        "name": "move_to",
        "parameters": {
            "target_pose": {
                "frame": "world",
                "xyz": [0.15, 0.2, 0.3],
                "probe_type": "articulated_attachment",
                "source_grasp_id": "handle-1",
                "compiled_grasp_id": "compiled-1",
                "scene_epoch": memory.scene_epoch(),
                "probe_path_sha256": "b" * 64,
            },
            "enable_collision_check": True,
        },
    }
    memory.add_action(
        _tool_action(
            "prepare_attachment_probe",
            {},
            outputs={
                "schema_version": "openeta.articulated_attachment_probe.v1",
                "status": "prepared",
                "candidate_id": "handle-1",
                "compiled_grasp_id": "compiled-1",
                "scene_epoch": memory.scene_epoch(),
                "motion_type": "linear",
                "distance_m": 0.05,
                "path_sha256": "b" * 64,
                "required_action": required_action,
                "pre_probe_image_paths": ["before-agent.png", "before-wrist.png"],
            },
        )
    )
    memory.add_action(_tool_action(required_action["name"], required_action["parameters"]))
    return memory, required_action


def test_articulated_unknown_allows_one_fresh_observation_then_stays_bounded() -> None:
    memory, _ = _memory_at_articulated_attachment()
    unknown = {
        "schema_version": "openeta.articulated_attachment_assessment.v1",
        "candidate_id": "handle-1",
        "verdict": "UNKNOWN",
        "reason": "wrist view is occluded",
    }
    memory.add_action(_tool_action("assess_attachment_probe", {}, outputs=unknown))
    assert memory.attachment_gate()["refresh_required"] is True

    memory.add_action(_tool_action("observe", {}))
    memory.add_observation(
        EnvObservation(task="open the microwave", cameras=[], robot=RobotState())
    )
    gate = memory.attachment_gate()
    assert gate["unknown_refresh_completed"] is True
    assert gate["refresh_required"] is False

    memory.add_action(_tool_action("assess_attachment_probe", {}, outputs=unknown))
    gate = memory.attachment_gate()
    assert gate["assessment_count"] == 2
    assert gate["verdict"] == "UNKNOWN"
    assert gate["refresh_required"] is False


def test_articulated_fail_open_advances_candidate_without_reviewer_metadata() -> None:
    memory, _ = _memory_at_articulated_attachment()
    policy = memory.grasp_candidate_policy()
    policy["candidates"] = [
        {"id": "handle-1", "rank": 0, "score": 0.8},
        {"id": "handle-2", "rank": 1, "score": 0.7},
    ]
    policy["approach_mode_order"] = ["front"]
    policy["approach_mode_queues"] = {"front": ["handle-1", "handle-2"]}
    policy["active_approach_mode"] = "front"
    policy["active_mode_rank"] = 0
    policy["mode_attempt_count"] = 0
    policy["mode_max_candidate_attempts"] = 3
    memory.save_fact("grasp_candidate_policy", policy, source="test")
    memory.add_action(
        _tool_action(
            "assess_attachment_probe",
            {},
            outputs={
                "schema_version": "openeta.articulated_attachment_assessment.v1",
                "candidate_id": "handle-1",
                "verdict": "FAIL",
                "reason": "handle stayed fixed while the gripper moved",
            },
        )
    )

    memory.add_action(_tool_action("gripper_control", {"position": 1}))

    assert memory.grasp_candidate_policy()["active_candidate"]["id"] == "handle-2"
    assert memory.grasp_execution() is None
    assert memory.articulated_attachment_probe() is None


def _compiled_candidate(candidate: dict, *, scene_epoch: int = 0) -> dict:
    profile = json.loads(DEFAULT_GRASP_PROFILE.read_text(encoding="utf-8"))
    return compile_grasp_seed(
        {
            "camera_pose": candidate,
            "camera_extrinsics": {
                "pos": [0.0, 0.0, 0.0],
                "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            "camera_frame_id": "agentview",
            "target_class": "upright_can",
            "scene_epoch": scene_epoch,
        },
        profile=profile,
        profile_sha256="profile-sha",
    )


def _compiled(memory: AgentMemory) -> dict:
    return _compiled_candidate(
        memory.anygrasp_candidate_policy()["active_candidate"],
        scene_epoch=memory.scene_epoch(),
    )


def _host_grasp_compilation_event(
    compiled: dict, *, queue_position: int, queue_count: int
) -> dict:
    return {
        "schema_version": "openeta.host_candidate_compilation.v1",
        "event_type": "candidate_compiled",
        "purpose": "grasp",
        "candidate_id": compiled["candidate_id"],
        "queue_position": queue_position,
        "queue_count": queue_count,
        "selection_policy": "stable_qualified_queue_head",
        "scene_epoch": compiled["scene_epoch"],
        "planning_scene_revision": 0,
        "execution_started": False,
        "compiled_seed": compiled,
    }


def test_host_compilation_activates_head_and_precompiled_grasp_fallback() -> None:
    first = _candidate("grasp_000", 0.9)
    second = _candidate("grasp_001", 0.8)
    first_compiled = _compiled_candidate(first)
    second_compiled = _compiled_candidate(second)
    events = [
        _host_grasp_compilation_event(
            first_compiled, queue_position=0, queue_count=2
        ),
        _host_grasp_compilation_event(
            second_compiled, queue_position=1, queue_count=2
        ),
    ]
    memory = AgentMemory()
    memory.start_session(task="pick the object")

    memory.add_action(
        _tool_action(
            "anygrasp",
            {},
            outputs={
                "result_id": "qualified-grasps",
                "grasp_candidates": [first, second],
                "selection_required": False,
                "host_selected_candidate_id": "grasp_000",
                "host_candidate_compilation": events[0],
                "host_candidate_compilation_queue": events,
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "grasp_000"
    assert policy["selection_source"] == "host_qualified_queue"
    assert set(policy["host_candidate_compilations"]) == {
        "grasp_000",
        "grasp_001",
    }
    assert memory.grasp_execution()["candidate_id"] == "grasp_000"

    memory.add_action(
        _tool_action(
            "move_to",
            {"target_pose": {"source_grasp_id": "grasp_000"}},
            success=False,
            outputs={
                "motion_summary": {
                    "reached_target": False,
                    "end": {"xyz": [0.1, 0.2, 0.3]},
                }
            },
        )
    )

    assert memory.grasp_candidate_policy()["active_candidate"]["id"] == "grasp_001"
    assert memory.grasp_execution()["candidate_id"] == "grasp_001"
    assert memory.grasp_execution()["compiled_grasp_id"] == second_compiled[
        "compiled_grasp_id"
    ]


def test_anygrasp_requires_host_compilation_but_anyplace_pose_keeps_generic_transform() -> None:
    memory = _memory_with_candidates()

    assert "host-owned candidate compiler" in memory.grasp_candidate_gate_error(
        tool_name="camera_pose_to_world",
        parameters={"camera_pose": {"id": "grasp_000"}},
    )
    assert (
        memory.grasp_candidate_gate_error(
            tool_name="camera_pose_to_world",
            parameters={"camera_pose": {"id": "place_grasp_000", "source_grasp_id": "grasp_000"}},
        )
        is None
    )


def test_exhausted_motion_candidate_queue_requires_fresh_reestimation() -> None:
    memory = _memory_with_candidates()
    policy = memory.anygrasp_candidate_policy()
    policy["candidates"] = [policy["active_candidate"]]
    policy["remaining_candidate_ids"] = []
    policy["target_detection"] = {
        "target_prompt": "black bowl",
        "source_image": "/tmp/current.rgb.png",
    }
    memory.save_fact("grasp_candidate_policy", policy, source="test")

    candidate_id = policy["active_candidate"]["id"]
    memory.add_action(
        _tool_action(
            "move_to",
            {"target_pose": {"id": candidate_id}},
            success=False,
            outputs={
                "motion_summary": {
                    "reached_target": False,
                    "end": {"xyz": [0.1, 0.2, 0.3]},
                }
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "exhausted"
    assert policy["active_candidate"] is None
    assert policy["reestimate_required"]["status"] == "pending_recovery"
    assert memory.grasp_reestimation()["status"] == "pending_recovery"
    assert memory.grasp_recovery()["status"] == "required"




def test_candidate_attempt_limit_requires_fresh_reestimation() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick bowl")
    memory.add_action(
        _tool_action(
            "anygrasp",
            {},
            outputs={
                "result_id": "four-candidates",
                "grasp_candidates": [
                    _candidate(f"grasp_{index:03d}", 1.0 - index * 0.1)
                    for index in range(4)
                ],
            },
        )
    )
    policy = memory.anygrasp_candidate_policy()
    policy["candidate_attempt_count"] = 2
    policy["target_detection"] = {
        "target_prompt": "black bowl",
        "source_image": "/tmp/current.rgb.png",
    }
    memory.save_fact("grasp_candidate_policy", policy, source="test")

    active = policy["active_candidate"]
    memory.add_action(
        _tool_action(
            "move_to",
            {"target_pose": {"id": active["id"]}},
            success=False,
            outputs={
                "motion_summary": {
                    "reached_target": False,
                    "end": {"xyz": [0.1, 0.2, 0.3]},
                }
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["status"] == "exhausted"
    assert policy["candidate_attempt_count"] == 3
    assert policy["active_candidate"] is None
    assert policy["reestimate_required"]["reason"] == "candidate_retry_limit_exceeded"
    assert memory.grasp_reestimation()["status"] == "pending_recovery"
    assert memory.grasp_recovery()["status"] == "required"


def test_reestimated_grasp_stops_on_repeated_failed_motion_fingerprint() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick red block")
    memory.save_fact(
        "selected_sam3_detection",
        {
            "id": "target-mask",
            "target_prompt": "red block",
            "source_image": "/tmp/scene.rgb.png",
            "mask_ref": "/tmp/target.mask.png",
            "scene_epoch": memory.scene_epoch(),
        },
        source="test",
    )
    estimate_outputs = {
        "mode": "targeted",
        "source_rgb": "/tmp/scene.rgb.png",
        "source_depth": "/tmp/scene.depth.png",
        "grasp_candidates": [_candidate("grasp_000", 0.9)],
    }
    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={"result_id": "estimate-1", **estimate_outputs},
        )
    )
    memory.add_action(
        _tool_action(
            "move_to",
            {"target_pose": {"source_grasp_id": "grasp_000"}},
            success=False,
            outputs={
                "request_fingerprint": "same-motion",
                "motion_summary": {"reached_target": False},
            },
        )
    )
    assert memory.grasp_candidate_policy()["failed_request_fingerprints"] == [
        "same-motion"
    ]

    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={"result_id": "estimate-2", **estimate_outputs},
        )
    )
    assert memory.grasp_candidate_policy()["failed_request_fingerprints"] == [
        "same-motion"
    ]
    memory.add_action(
        _tool_action(
            "move_to",
            {"target_pose": {"source_grasp_id": "grasp_000"}},
            success=False,
            outputs={
                "response": {"request_fingerprint": "same-motion"},
                "motion_summary": {"reached_target": False},
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "stopped_requires_human"
    assert policy["stop_reason"] == "repeated_failed_request_fingerprint"
    assert policy["active_candidate"] is None


def test_anygrasp_filters_non_executable_width_before_score_ranking() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick can")
    too_wide = _candidate("grasp_000", 0.99)
    too_wide["width"] = 0.1
    executable = _candidate("grasp_001", 0.5)

    memory.add_action(
        _tool_action(
            "anygrasp",
            {},
            outputs={
                "result_id": "anygrasp-width-filter",
                "grasp_candidates": [too_wide, executable],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["raw_candidate_count"] == 2
    assert policy["candidate_count"] == 1
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert policy["active_candidate"]["rank"] == 0
    assert policy["rejected_candidates"] == [
        {
            "candidate_id": "grasp_000",
            "rank": None,
            "score": 0.99,
            "reason": (
                "candidate width exceeds calibration max_gripper_width_m 0.0800 m"
            ),
            "source": "physical_gripper_width_filter",
        }
    ]


def test_graspgenx_memory_preserves_formal_counts_without_post_moveit_reordering() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick red block")
    candidates = [
        _candidate("vertical-0", 0.99),
        _candidate("vertical-1", 0.98),
        _candidate("vertical-2", 0.97),
        _candidate("side", 0.80),
        _candidate("tilted", 0.70),
    ]
    for candidate in candidates[:3]:
        candidate["rotation_matrix"] = [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ]
    candidates[3]["rotation_matrix"] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    candidates[4]["rotation_matrix"] = [
        [0.3, 0.0, 0.953939],
        [0.0, 1.0, 0.0],
        [0.953939, 0.0, -0.3],
    ]

    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "diverse-graspgenx",
                "selected_backend": "graspgenx",
                "raw_candidate_count": 100,
                "generated_candidate_count": 10,
                "submitted_candidate_count": 10,
                "full_plan_pass_count": 5,
                "grasp_candidates": candidates,
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["ranking"] == "score_descending"
    assert [candidate["id"] for candidate in policy["candidates"][:3]] == [
        "vertical-0",
        "vertical-1",
        "vertical-2",
    ]
    assert [candidate["score_rank"] for candidate in policy["candidates"][:3]] == [
        0,
        1,
        2,
    ]
    assert policy["raw_candidate_count"] == 100
    assert policy["generated_candidate_count"] == 10
    assert policy["submitted_candidate_count"] == 10
    assert policy["full_plan_pass_count"] == 5
    assert "qualified_candidate_count" not in policy


def test_memory_preserves_fast_joint_physical_quality_over_generator_score() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick red block")
    model_favorite = {
        **_candidate("model-favorite", 0.99),
        "grasp_place_joint_qualified": True,
        "grasp_place_frontier_quality_rank": 1,
        "grasp_place_physical_quality_rank": 0,
        "moveit_l5_qualified": True,
        "moveit_physical_quality_rank": 0,
    }
    physical_favorite = {
        **_candidate("physical-favorite", 0.60),
        "grasp_place_joint_qualified": True,
        "grasp_place_frontier_quality_rank": 0,
        "grasp_place_physical_quality_rank": 1,
        "moveit_l5_qualified": True,
        "moveit_physical_quality_rank": 1,
    }

    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "fast-joint-ranked",
                "selected_backend": "graspgenx",
                "qualification_profile": "fast_v3",
                "ranking": "grasp_place_physical_quality",
                "grasp_candidates": [model_favorite, physical_favorite],
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["ranking"] == "grasp_place_physical_quality"
    assert [candidate["id"] for candidate in policy["candidates"]] == [
        "physical-favorite",
        "model-favorite",
    ]
    assert policy["active_candidate"]["id"] == "physical-favorite"
    assert policy["active_candidate"]["score"] == 0.60


def test_selected_mask_geometry_becomes_task_strategy_compile_hint() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam3-geometry",
            "candidates": [{"id": "detection_000", "mask_ref": "mask.png"}],
        },
        source="sam3",
    )
    selected = memory.resolve_sam3_selection(
        result_id="sam3-geometry",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
        target_geometry_family="upright_can",
    )
    assert selected["target_geometry_family"] == "upright_can"

    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "grasp-geometry",
                "mode": "targeted",
                "selected_backend": "anygrasp",
                "grasp_candidates": [_candidate("grasp_000", 0.9)],
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["target_detection"]["id"] == "detection_000"
    assert policy["compile_hints"] == {
        "target_geometry_family": "upright_can",
    }


def test_articulated_handle_uses_bounded_mode_queues_then_global_fallback() -> None:
    memory = _memory_with_articulated_handle_candidates()
    policy = memory.grasp_candidate_policy()

    assert policy["interaction_family"] == "articulated_handle"
    assert policy["approach_mode_order"] == ["top_down", "front", "side"]
    assert policy["active_candidate"]["id"] == "top-0"
    assert policy["compile_hints"] == {
        "target_geometry_family": "articulated_handle",
        "approach_mode": "top_down",
    }

    _reject_active_candidate(memory)
    _reject_active_candidate(memory)
    _reject_active_candidate(memory)
    policy = memory.grasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "front-0"
    assert policy["active_approach_mode"] == "front"
    assert "top-3" not in {
        item["candidate_id"] for item in policy["rejected_candidates"]
    }
    assert policy["compile_hints"]["approach_mode"] == "front"

    _reject_active_candidate(memory)
    policy = memory.grasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "side-0"
    assert policy["compile_hints"]["approach_mode"] == "side"
    _reject_active_candidate(memory)
    policy = memory.grasp_candidate_policy()
    assert policy["candidate_fallback"] is True
    assert policy["fallback_reason"] == "all_approach_modes_failed"
    assert policy["active_candidate"]["id"] == "top-0"
    assert policy["active_candidate"]["candidate_fallback"] is True

    _reject_active_candidate(memory)
    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "exhausted"
    assert policy["reestimate_required"]["reason"] == (
        "articulated_handle_fallback_failed"
    )
    assert memory.grasp_reestimation()["status"] == "pending_recovery"


def test_legacy_drawer_handle_and_conservative_task_fallback_enable_modes() -> None:
    legacy = _memory_with_articulated_handle_candidates(geometry_family="drawer_handle")
    assert legacy.grasp_candidate_policy()["interaction_family"] == "articulated_handle"

    fallback = _memory_with_articulated_handle_candidates(
        geometry_family="unknown",
        task="open the microwave",
        prompt="microwave door handle",
    )
    assert fallback.grasp_candidate_policy()["interaction_family"] == "articulated_handle"


def test_non_articulated_tasks_and_non_handle_targets_keep_original_candidate_policy() -> None:
    cases = [
        ("open the bottle", "bottle cap", "other"),
        ("pull the basket closer", "basket handle", "other"),
        ("put the bowl inside the open microwave", "black bowl", "bowl"),
    ]
    for task, prompt, family in cases:
        memory = _memory_with_articulated_handle_candidates(
            geometry_family=family,
            task=task,
            prompt=prompt,
        )
        policy = memory.grasp_candidate_policy()
        assert "interaction_family" not in policy
        assert policy["active_candidate"]["id"] == "top-0"
        assert policy["compile_hints"] == {"target_geometry_family": family}


def test_explicit_articulated_label_does_not_override_portable_handle_exclusion() -> None:
    memory = _memory_with_articulated_handle_candidates(
        geometry_family="articulated_handle",
        task="put the mug in the cabinet",
        prompt="mug handle",
    )

    assert "interaction_family" not in memory.grasp_candidate_policy()


def test_missing_matching_extrinsics_fails_closed_to_original_policy() -> None:
    memory = AgentMemory()
    memory.start_session(task="open the middle drawer")
    memory.save_fact(
        "selected_sam3_detection",
        {
            "id": "detection_handle",
            "mask_ref": "tmp/handle.mask.png",
            "source_image": "tmp/agentview.rgb.png",
            "target_prompt": "drawer handle",
            "target_geometry_family": "articulated_handle",
            "scene_epoch": memory.scene_epoch(),
        },
        source="test",
    )
    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "handle-grasps-no-extrinsics",
                "mode": "targeted",
                "selected_backend": "anygrasp",
                "source_rgb": "tmp/agentview.rgb.png",
                "target_mask": "tmp/handle.mask.png",
                "camera_frame_id": "agentview",
                "grasp_candidates": [
                    _candidate_with_approach("top-0", 0.95, "top_down")
                ],
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert "interaction_family" not in policy
    assert policy["compile_hints"] == {
        "target_geometry_family": "articulated_handle"
    }


def test_new_grasp_queue_clears_prior_release_but_keeps_completed_ledger() -> None:
    memory = AgentMemory()
    memory.start_session(task="put both cans in the basket")
    memory.save_fact(
        "placement_release",
        {
            "status": "released",
            "candidate_id": "first-grasp",
            "placement_pose_id": "first-place",
        },
        source="test",
    )
    completed = {
        "items": [
            {
                "candidate_id": "first-grasp",
                "placement_pose_id": "first-place",
                "target_object": "alphabet soup",
            }
        ]
    }
    memory.save_fact("completed_placement_subgoals", completed, source="test")

    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "second-grasp-queue",
                "mode": "targeted",
                "selected_backend": "anygrasp",
                "grasp_candidates": [_candidate("second-grasp", 0.9)],
            },
        )
    )

    assert memory.placement_release() is None
    assert memory.get_memory("completed_placement_subgoals", namespace="facts")["facts"][
        "completed_placement_subgoals"
    ]["value"] == completed


def test_denied_host_release_invalidates_dropped_grasp_instead_of_replaying() -> None:
    memory = _memory_with_candidates()
    memory.save_fact(
        "grasp_execution",
        {"status": "completed", "stage": "attached", "candidate_id": "grasp_000"},
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {"status": "resolved", "verdict": "PASS", "candidate_id": "grasp_000"},
        source="test",
    )
    memory.save_fact(
        "placement_release",
        {
            "schema_version": "openeta.placement_release.v1",
            "status": "ready",
            "candidate_id": "grasp_000",
            "placement_pose_id": "place-1",
            "release_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]},
        },
        source="test",
    )
    memory.save_fact(
        "selected_sam3_detection",
        {"id": "basket-mask", "label": "basket"},
        source="test",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "gripper_control",
                    "parameters": {"position": 1},
                },
                "status": "failed",
                "metadata": {
                    "planner_metadata": {
                        "host_obligation": {
                            "stage": "release",
                            "tool": "gripper_control",
                        }
                    }
                },
                "tool_calls": [
                    {
                        "name": "gripper_control",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": "Target detached before the valid release pose.",
                            "details": {
                                "diagnostics": [{"code": "supervision_denied"}],
                            },
                        },
                    }
                ],
            },
        )
    )

    assert memory.placement_release()["status"] == "failed"
    assert memory.grasp_execution() is None
    assert memory.attachment_gate() is None
    assert memory.grasp_candidate_policy() is None
    assert memory.selected_sam3_detection() is None
    assert any(event.event_type == "placement_release_failed" for event in memory.events)


def test_host_grasp_execution_rejects_every_terminal_pose_adjustment() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    assert memory.grasp_execution()["stage"] == "open"
    assert memory.grasp_candidate_policy()["compile_hints"] == {
        "target_geometry_family": "upright_can",
    }

    required = memory.grasp_execution()["required_action"]
    memory.add_action(_tool_action(required["name"], required["parameters"]))
    assert memory.grasp_execution()["stage"] == "contact"
    required = memory.grasp_execution()["required_action"]
    reference_xyz = required["parameters"]["target_pose"]["xyz"]
    adjusted = {
        "target_pose": {
            **required["parameters"]["target_pose"],
            "xyz": [reference_xyz[0] + 0.02, reference_xyz[1], reference_xyz[2]],
        }
    }
    error = memory.grasp_execution_gate_error(tool_name="move_to", parameters=adjusted)
    assert error is not None
    assert "exactly" in error

    assert memory.grasp_execution_gate_error(
        tool_name=required["name"], parameters=required["parameters"]
    ) is None
    memory.add_action(_tool_action(required["name"], required["parameters"]))
    assert memory.grasp_execution()["stage"] == "close"
    assert memory.anygrasp_candidate_policy()["status"] == "accepted"






def test_point_segmentation_retains_same_frame_text_target_prompt() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the blue cylinder")
    image = "/frozen/top.rgb.png"
    memory.add_action(
        _tool_action(
            "sam3",
            {"image": image, "prompt": "blue cylinder"},
            outputs={"result_id": "empty-text", "detections": []},
        )
    )
    memory.add_action(
        _tool_action(
            "sam3",
            {"image": image, "positive_points": [[239, 51]]},
            outputs={
                "result_id": "point-result",
                "source_image": image,
                "detections": [
                    {
                        "id": "detection_000",
                        "mask_ref": "/frozen/mask.png",
                    }
                ],
            },
        )
    )
    memory.resolve_sam3_selection(
        result_id="point-result",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    assert memory.selected_sam3_detection()["target_prompt"] == "blue cylinder"


def test_known_failed_host_close_rejects_candidate_without_repeating() -> None:
    memory = _memory_with_candidates()
    policy = memory.grasp_candidate_policy()
    policy["status"] = "accepted"
    memory.save_fact("grasp_candidate_policy", policy, source="test")
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "close",
            "candidate_id": "grasp_000",
            "required_action": {
                "name": "gripper_control",
                "parameters": {"position": 0},
            },
        },
        source="test",
    )

    memory.add_action(
        _tool_action(
            "gripper_control",
            {"position": 0},
            success=False,
            outputs={"motion_outcome": "failed", "error_code": "CONTACT_REJECTED"},
        )
    )

    assert memory.grasp_candidate_policy()["active_candidate"]["id"] == "grasp_001"
    assert memory.grasp_execution() is None
    next_compiled = _compiled(memory)
    memory.add_action(
        _tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=next_compiled)
    )
    assert memory.grasp_execution()["stage"] == "open"
    assert memory.grasp_execution()["required_action"] == {
        "name": "gripper_control",
        "parameters": {"position": 1},
    }


def test_failed_close_requalifies_candidates_after_scene_epoch_changes() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    policy = memory.grasp_candidate_policy()
    policy.update(
        {
            "status": "accepted",
            "scene_epoch": 0,
            "qualification_evidence": {
                "schema_version": "openeta.moveit_candidate_qualification.v1"
            },
        }
    )
    memory.save_fact("grasp_candidate_policy", policy, source="test")
    memory.save_fact("scene_epoch", {"epoch": 3}, source="test")
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "close",
            "candidate_id": "grasp_000",
            "compiled_grasp": compiled,
            "required_action": {
                "name": "gripper_control",
                "parameters": {"position": 0},
            },
        },
        source="test",
    )

    memory.add_action(
        _tool_action(
            "gripper_control",
            {"position": 0},
            success=False,
            outputs={"motion_outcome": "failed", "error_code": "CONTACT_REJECTED"},
        )
    )

    updated = memory.grasp_candidate_policy()
    assert updated["status"] == "exhausted"
    assert updated["active_candidate"] is None
    assert updated["reestimate_required"]["reason"] == (
        "moveit_qualification_scene_changed"
    )
    recovery = memory.grasp_recovery()
    assert recovery["stage"] == "reopen"
    assert recovery["required_action"] == {
        "name": "gripper_control",
        "parameters": {"position": 1},
    }
    memory.add_action(_tool_action("gripper_control", {"position": 1}))
    recovery = memory.grasp_recovery()
    assert recovery["stage"] == "observe"
    assert recovery["required_action"] == {"name": "observe", "parameters": {}}


def test_failed_close_reuses_next_frozen_exact_terminal_without_model_rerun() -> None:
    candidates = [
        {
            **_candidate(f"grasp_{index:03d}", 0.9 - index * 0.1),
            "grasp_place_joint_qualified": True,
        }
        for index in range(4)
    ]
    compiled = [_compiled_candidate(candidate) for candidate in candidates]
    events = [
        _host_grasp_compilation_event(
            item,
            queue_position=index,
            queue_count=len(compiled),
        )
        for index, item in enumerate(compiled)
    ]
    memory = AgentMemory()
    memory.start_session(task="pick the red block and place it in the green zone")
    memory.add_action(
        _tool_action(
            "anyplace",
            {},
            outputs={
                "frozen_goal_pool_ready": True,
                "frozen_goal_pool_count": 96,
            },
        )
    )
    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "frozen-qualified-grasps",
                "selected_backend": "graspgenx",
                "grasp_candidates": candidates,
                "qualification_evidence": {
                    "schema_version": "openeta.moveit_candidate_qualification.v3",
                    "scene_epoch": 0,
                },
                "frozen_pair_grasp_branch_limit": 4,
                "frozen_pair_lookahead_grasp_count": 4,
                "frozen_pair_full_plan_pass_count": 4,
                "host_selected_candidate_id": candidates[0]["id"],
                "host_candidate_compilation": events[0],
                "host_candidate_compilation_queue": events,
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["max_candidate_attempts"] == 4
    policy["status"] = "accepted"
    memory.save_fact("grasp_candidate_policy", policy, source="test")
    memory.save_fact("scene_epoch", {"epoch": 3}, source="test")
    execution = memory.grasp_execution()
    execution.update(
        {
            "stage": "close",
            "required_action": {
                "name": "gripper_control",
                "parameters": {"position": 0},
            },
        }
    )
    memory.save_fact("grasp_execution", execution, source="test")

    memory.add_action(
        _tool_action(
            "gripper_control",
            {"position": 0},
            success=False,
            outputs={"motion_outcome": "failed", "error_code": "CONTACT_REJECTED"},
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "active"
    assert policy["active_candidate"]["id"] == candidates[1]["id"]
    assert "reestimate_required" not in policy
    assert policy["frozen_model_pool_retry"] == {
        "schema_version": "openeta.frozen_grasp_retry.v1",
        "source_scene_epoch": 0,
        "current_scene_epoch": 3,
        "candidate_id": candidates[1]["id"],
        "model_inference_invoked": False,
        "terminal_pose_reused": True,
        "path_owner": "moveit",
    }
    execution = memory.grasp_execution()
    assert execution["candidate_id"] == candidates[1]["id"]
    assert execution["stage"] == "open"
    reused = execution["compiled_grasp"]
    assert reused["scene_epoch"] == 3
    assert reused["contact_pose"]["xyz"] == compiled[1]["contact_pose"]["xyz"]
    assert reused["contact_pose"]["rotation_matrix"] == compiled[1]["contact_pose"][
        "rotation_matrix"
    ]
    assert reused["frozen_model_pool_reuse"]["model_inference_invoked"] is False

    memory.add_action(
        _tool_action(
            "gripper_control",
            {"position": 1},
            success=False,
            outputs={"motion_outcome": "unknown", "reconciliation_required": True},
        )
    )
    assert memory.motion_reconciliation()["status"] == "required"
    assert memory.grasp_recovery()["status"] == "reconciling"

    memory.add_observation(
        EnvObservation(
            task="pick and place",
            cameras=[],
            robot=RobotState(gripper_state={"open": True, "openness": 0.99}),
        )
    )

    assert memory.motion_reconciliation()["status"] == "completed"
    assert memory.grasp_recovery()["status"] == "completed"
    execution = memory.grasp_execution()
    assert execution["candidate_id"] == candidates[1]["id"]
    assert execution["stage"] == "contact"
    assert execution["required_action"]["name"] == "move_to"
    assert execution["required_action"]["parameters"]["target_pose"]["xyz"] == (
        compiled[1]["contact_pose"]["xyz"]
    )


def test_exhausted_backup_resumes_frozen_frontier_when_scene_revision_is_unchanged() -> None:
    candidate = {
        **_candidate("grasp_000", 0.9),
        "grasp_place_joint_qualified": True,
    }
    compiled = _compiled_candidate(candidate)
    compilation = _host_grasp_compilation_event(
        compiled, queue_position=0, queue_count=1
    )
    compilation["planning_scene_revision"] = 7
    memory = AgentMemory()
    memory.start_session(task="pick and place")
    memory.add_action(
        _tool_action(
            "anyplace",
            {},
            outputs={
                "frozen_goal_pool_ready": True,
                "frozen_goal_pool_count": 96,
            },
        )
    )
    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "frozen-qualified-grasps",
                "selected_backend": "graspgenx",
                "scene_revision": 7,
                "grasp_candidates": [candidate],
                "qualification_evidence": {
                    "schema_version": "openeta.moveit_candidate_qualification.v3",
                    "scene_epoch": 0,
                    "planning_scene_revision": 7,
                },
                "frozen_pair_grasp_branch_limit": 2,
                "frozen_pair_lookahead_grasp_count": 1,
                "frozen_pair_full_plan_pass_count": 1,
                "frozen_grasp_frontier_remaining_count": 12,
                "frozen_grasp_frontier_generation": 1,
                "host_selected_candidate_id": candidate["id"],
                "host_candidate_compilation": compilation,
                "host_candidate_compilation_queue": [compilation],
            },
        )
    )
    memory.save_fact("scene_epoch", {"epoch": 3}, source="test")
    execution = memory.grasp_execution()
    execution.update(
        {
            "stage": "contact",
            "required_action": {
                "name": "move_to",
                "parameters": {
                    "target_pose": {
                        "source_grasp_id": candidate["id"],
                        "xyz": compiled["contact_pose"]["xyz"],
                    }
                },
            },
        }
    )
    memory.save_fact("grasp_execution", execution, source="test")
    required = execution["required_action"]

    memory.add_action(
        _tool_action(
            required["name"],
            required["parameters"],
            success=False,
            outputs={
                "motion_summary": {"reached_target": False},
            },
            environment_receipt={
                "execution_started": True,
                "planning_scene_revision": 7,
                "request_fingerprint": "failed-g0",
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "frozen_frontier_required"
    assert policy["planning_scene_revision"] == 7
    assert policy["frozen_grasp_frontier_remaining_count"] == 12
    assert policy["last_rejection"]["execution_started"] is True
    assert policy["last_rejection"]["planning_scene_revision"] == 7


def test_acknowledged_binary_gripper_state_is_latched_and_skips_redundant_open() -> None:
    memory = _memory_with_candidates()
    memory.add_action(_tool_action("gripper_control", {"position": 1}))

    assert memory.gripper_command_state()["position"] == 1
    assert memory.gripper_command_state()["latched"] is True

    compiled = _compiled(memory)
    memory.add_action(
        _tool_action(
            "compile_grasp_seed",
            {"scene_epoch": memory.scene_epoch()},
            outputs={**compiled, "scene_epoch": memory.scene_epoch()},
        )
    )

    execution = memory.grasp_execution()
    assert execution["stage"] == "contact"
    assert execution["required_action"]["name"] == "move_to"


def test_close_timeout_visual_gap_cannot_replace_native_attachment_proof() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    execution = memory.grasp_execution()
    execution.update(
        {
            "stage": "close",
            "required_action": {"name": "gripper_control", "parameters": {"position": 0}},
        }
    )
    memory.save_fact("grasp_execution", execution, source="test")
    memory.add_action(
        _tool_action(
            "gripper_control",
            {"position": 0},
            success=False,
            outputs={"motion_outcome": "unknown", "reconciliation_required": True},
        )
    )

    memory.add_observation(
        EnvObservation(
            task="pick",
            cameras=[],
            robot=RobotState(gripper_state={"open": True, "openness": 0.45}),
        )
    )

    assert memory.motion_reconciliation()["status"] == "completed"
    assert memory.gripper_command_state()["position"] == 0
    assert memory.grasp_execution()["stage"] == "attachment_unknown"
    assert memory.grasp_execution()["status"] == "stopped_requires_human"
    assert memory.attachment_gate()["verdict"] == "UNKNOWN"








def test_planning_scene_unavailable_stops_grasp_without_candidate_replay() -> None:
    memory = _memory_with_candidates()
    required = {
        "target_pose": {
            "frame": "world",
            "source_grasp_id": "grasp_000",
            "compiled_grasp_id": "compiled-000",
            "xyz": [0.2, 0.0, 0.6],
        }
    }
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "contact",
            "candidate_id": "grasp_000",
            "compiled_grasp_id": "compiled-000",
            "required_action": {"name": "move_to", "parameters": required},
        },
        source="test",
    )

    memory.add_action(
        _tool_action(
            "move_to",
            required,
            success=False,
            environment_receipt={
                "ok": False,
                "error_code": "PLANNING_SCENE_UNAVAILABLE",
                "motion_outcome": "failed",
                "execution_started": False,
                "planning_scene_revision": 2,
            },
        )
    )

    assert memory.grasp_execution()["status"] == "stopped_requires_human"
    assert memory.grasp_execution()["stage"] == "planning_scene_failure"
    assert memory.grasp_candidate_policy()["status"] == "stopped_requires_human"
    assert memory.grasp_candidate_policy()["active_candidate"]["id"] == "grasp_000"
    assert memory.attachment_gate()["verdict"] == "UNKNOWN"
    assert memory.grasp_execution_gate_error(
        tool_name="move_to", parameters=required
    ) is not None










def test_transport_timeout_reconciles_gripper_and_partial_motion() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    required = memory.grasp_execution()["required_action"]
    timeout = _tool_action(
        required["name"],
        required["parameters"],
        success=False,
        outputs={"motion_outcome": "unknown", "reconciliation_required": True},
    )
    memory.add_action(timeout)
    assert memory.motion_reconciliation()["status"] == "required"
    assert memory.grasp_execution_gate_error(tool_name="move_to", parameters={}) is not None

    memory.add_observation(
        EnvObservation(
            task="pick",
            cameras=[],
            robot=RobotState(gripper_state={"open": True, "openness": 0.95}),
        )
    )
    assert memory.motion_reconciliation()["status"] == "completed"
    assert memory.grasp_execution()["stage"] == "contact"

    contact = memory.grasp_execution()["required_action"]
    memory.add_action(
        _tool_action(
            contact["name"],
            contact["parameters"],
            success=False,
            outputs={"motion_outcome": "unknown", "reconciliation_required": True},
        )
    )
    target = contact["parameters"]["target_pose"]["xyz"]
    memory.add_observation(
        EnvObservation(
            task="pick",
            cameras=[],
            robot=RobotState(end_effector_pose={"xyz": [target[0] + 0.05, target[1], target[2]]}),
        )
    )
    assert memory.motion_reconciliation()["status"] == "required"
    assert (
        memory.grasp_execution_gate_error(
            tool_name=contact["name"], parameters=contact["parameters"]
        )
        is not None
    )
    assert memory.grasp_execution_gate_error(tool_name="observe", parameters={}) is None
    assert (
        memory.grasp_execution_gate_error(tool_name="gripper_control", parameters={"position": 0})
        is not None
    )

    stable_partial = EnvObservation(
        task="pick",
        cameras=[],
        robot=RobotState(end_effector_pose={"xyz": [target[0] + 0.05, target[1], target[2]]}),
    )
    memory.add_observation(stable_partial)
    assert memory.motion_reconciliation()["status"] == "required"
    memory.add_observation(stable_partial)

    assert memory.motion_reconciliation()["status"] == "failed"
    assert memory.grasp_execution() is None
    assert memory.anygrasp_candidate_policy()["active_candidate"]["id"] == "grasp_001"
    assert memory.anygrasp_candidate_policy()["last_rejection"] == {
        "source": "reconciled_candidate_motion_rejected",
        "target_tool": "move_to",
        "reason": "reconciled_target_not_reached",
    }


def test_transport_timeout_continues_reconciling_after_unresolved_observation() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    open_action = memory.grasp_execution()["required_action"]
    memory.add_action(_tool_action(open_action["name"], open_action["parameters"]))
    contact = memory.grasp_execution()["required_action"]
    memory.add_action(
        _tool_action(
            contact["name"],
            contact["parameters"],
            success=False,
            outputs={"motion_outcome": "unknown", "reconciliation_required": True},
        )
    )
    target = contact["parameters"]["target_pose"]["xyz"]
    stable_far = EnvObservation(
        task="pick",
        cameras=[],
        robot=RobotState(end_effector_pose={"xyz": [target[0] + 0.3, target[1], target[2]]}),
    )

    memory.add_observation(stable_far)
    assert memory.motion_reconciliation()["status"] == "unresolved"
    memory.add_observation(stable_far)
    memory.add_observation(stable_far)

    assert memory.motion_reconciliation()["status"] == "failed"
    assert memory.grasp_execution() is None
    assert memory.anygrasp_candidate_policy()["active_candidate"]["id"] == "grasp_001"
