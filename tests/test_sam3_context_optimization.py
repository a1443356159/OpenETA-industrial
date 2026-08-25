from __future__ import annotations

import json

import pytest

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
from agent.backends.planner import (
    CallablePlannerBackend,
    PlannerBackendRequest,
    PlannerBackendResult,
)
from agent.runtime.memory import AgentMemory
from agent.runtime.planner import (
    PlannerContextConfig,
    ToolCallingPlanner,
    _default_tool_planner_system_prompt,
    _model_request_context,
    _sam3_request_identity,
    _semantic_perception_obligation,
)
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.sam3_selection import (
    BackendSam3SelectionReviewer,
    Sam3SelectionReviewError,
    Sam3SelectionParentContext,
)


def test_isolated_sam3_reviewer_receives_only_typed_bundle_and_two_images() -> None:
    requests: list[PlannerBackendRequest] = []

    def decide(request: PlannerBackendRequest) -> PlannerBackendResult:
        requests.append(request)
        return PlannerBackendResult(
            payload={
                "decision": "select",
                "detection_id": "detection_001",
                "confidence": 0.92,
                "reason": "Tile 1 covers the complete can in the original RGB.",
                "target_geometry_family": "upright_can",
            },
            provider="fixture-provider",
            model="fixture-vlm",
            details={"usage": {"prompt_tokens": 321, "completion_tokens": 42}},
        )

    reviewer = BackendSam3SelectionReviewer(CallablePlannerBackend(decide))
    review = reviewer.review(
        {
            "result_id": "sam3-result-1",
            "semantic_role": "grasp_target",
            "target_prompt": "alphabet soup can",
            "source_image": "/tmp/original.png",
            "candidates": [
                {
                    "id": "detection_000",
                    "rank": 0,
                    "score": 0.99,
                    "mask_ref": "/private/full-mask-payload.png",
                },
                {"id": "detection_001", "rank": 1, "score": 0.81},
            ],
            "selection_bundle": {
                "original_image_ref": "/tmp/original.png",
                "contact_sheet_ref": "/tmp/contact-sheet.png",
            },
        }
    )

    assert review["decision"] == "select"
    assert review["detection_id"] == "detection_001"
    assert review["selection_source"] == "isolated_main_vlm"
    assert len(requests) == 1
    request = requests[0]
    assert request.conversation_messages == []
    assert request.conversation_summary == ""
    assert request.metadata["isolated_context"] is True
    assert request.tool_context["vision_image_paths"] == [
        "/tmp/original.png",
        "/tmp/contact-sheet.png",
    ]
    assert request.tool_context["semantic_role"] == "grasp_target"
    assert request.tool_context["target_prompt"] == "alphabet soup can"
    assert "mask_ref" not in request.tool_context["candidates"][0]


def test_sam3_reviewer_forks_confirmed_bounded_parent_planner_context() -> None:
    parent_context = Sam3SelectionParentContext()
    parent_context.capture(
        PlannerBackendRequest(
            system_prompt="parent planner system contract",
            tool_context={
                "task": "pick and place the red block",
                "controller": {"phase": "semantic_perception"},
                "current_camera_artifacts": [{"path": "/tmp/stale.png"}],
                "current_rgbd_views": [{"rgb_path": "/tmp/stale.png"}],
                "semantic_perception_obligation": {"role": "grasp_target"},
                "selected_skill_guidance": [{"name": "pick"}],
            },
            conversation_messages=[
                {"role": "user", "content": "pick and place the red block"}
            ],
            conversation_summary="bounded parent summary",
            metadata={"schema_version": "openeta.planner_decision.v1"},
        )
    )
    requests: list[PlannerBackendRequest] = []

    def decide(request: PlannerBackendRequest) -> PlannerBackendResult:
        requests.append(request)
        return PlannerBackendResult(
            payload={
                "kind": "tool_call",
                "name": "select_sam3_detection",
                "parameters": {
                    "sam3_result_id": "sam3-parent-fork",
                    "detection_id": "detection_000",
                    "reason": "The labelled mask covers the complete red block.",
                },
                "reasoning": "The two images agree.",
            },
            provider="fixture-provider",
            model="fixture-vlm",
        )

    review = BackendSam3SelectionReviewer(
        CallablePlannerBackend(decide),
        parent_context=parent_context,
    ).review(
        {
            "result_id": "sam3-parent-fork",
            "semantic_role": "grasp_target",
            "target_prompt": "red rectangular block",
            "candidates": [{"id": "detection_000", "rank": 0}],
            "selection_bundle": {
                "original_image_ref": "/tmp/original.png",
                "contact_sheet_ref": "/tmp/contact-sheet.png",
            },
        }
    )

    assert review["decision"] == "select"
    assert review["detection_id"] == "detection_000"
    assert review["context_strategy"] == "parent_planner_fork"
    assert review["parent_context_fork"] is True
    assert review["isolated_context"] is False
    assert review["selection_source"] == "parent_context_main_vlm"
    assert len(requests) == 1
    request = requests[0]
    assert request.system_prompt == "parent planner system contract"
    assert request.conversation_messages == [
        {"role": "user", "content": "pick and place the red block"}
    ]
    assert request.conversation_summary == "bounded parent summary"
    assert request.metadata["parent_context_fork"] is True
    assert request.tool_context["vision_image_paths"] == [
        "/tmp/original.png",
        "/tmp/contact-sheet.png",
    ]
    assert request.tool_context["controller"]["phase"] == "semantic_selection"
    assert request.tool_context["registered_tool_handlers"] == [
        "select_sam3_detection",
        "reject_sam3_detections",
    ]
    assert "current_camera_artifacts" not in request.tool_context
    assert "current_rgbd_views" not in request.tool_context
    assert "semantic_perception_obligation" not in request.tool_context
    assert request.tool_context["selected_skill_guidance"] == [{"name": "pick"}]


def test_sam3_parent_context_retains_first_confirmed_checkpoint() -> None:
    parent_context = Sam3SelectionParentContext()
    first = PlannerBackendRequest(
        system_prompt="confirmed small request",
        tool_context={"task": "pick and place", "checkpoint": "first"},
        conversation_messages=[{"role": "user", "content": "pick and place"}],
    )
    later = PlannerBackendRequest(
        system_prompt="later accumulated request",
        tool_context={"task": "pick and place", "checkpoint": "later", "noise": "x" * 50_000},
        conversation_messages=[{"role": "assistant", "content": "many later turns"}],
    )

    assert parent_context.capture_if_empty(first) is True
    assert parent_context.capture_if_empty(later) is False

    snapshot = parent_context.snapshot()
    assert snapshot is not None
    assert snapshot.system_prompt == "confirmed small request"
    assert snapshot.tool_context == {"task": "pick and place", "checkpoint": "first"}
    assert snapshot.conversation_messages == [
        {"role": "user", "content": "pick and place"}
    ]

    parent_context.clear()
    assert parent_context.snapshot() is None


def test_runtime_session_start_clears_sam3_parent_checkpoint() -> None:
    parent_context = Sam3SelectionParentContext()
    parent_context.capture(
        PlannerBackendRequest(
            system_prompt="previous session",
            tool_context={"task": "previous"},
        )
    )
    planner = ToolCallingPlanner(
        CallablePlannerBackend(lambda _request: {"kind": "response", "name": "talk"}),
        sam3_selection_parent_context=parent_context,
    )
    runtime = OpenEtaAgentRuntime(planner=planner, rollout_enabled=False)

    runtime.start_session(task="new session")

    assert parent_context.snapshot() is None


def test_isolated_sam3_reviewer_rejects_unknown_candidate_id() -> None:
    reviewer = BackendSam3SelectionReviewer(
        CallablePlannerBackend(
            lambda _request: {
                "decision": "select",
                "detection_id": "invented-id",
                "confidence": 0.5,
                "reason": "invalid fixture",
                "target_geometry_family": "other",
            }
        )
    )

    with pytest.raises(ValueError, match="unknown detection id"):
        reviewer.review(
            {
                "result_id": "sam3-result-1",
                "semantic_role": "grasp_target",
                "target_prompt": "can",
                "candidates": [{"id": "detection_000"}],
                "selection_bundle": {
                    "original_image_ref": "/tmp/original.png",
                    "contact_sheet_ref": "/tmp/contact-sheet.png",
                },
            }
        )


def test_isolated_sam3_reviewer_retries_once_inside_bounded_subroutine() -> None:
    attempts = 0

    def decide(_request: PlannerBackendRequest):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("fixture provider timeout")
        return {
            "decision": "select",
            "detection_id": "detection_000",
            "confidence": 0.91,
            "reason": "The only tile covers the complete target.",
            "target_geometry_family": "boxed_item",
        }

    reviewer = BackendSam3SelectionReviewer(
        CallablePlannerBackend(decide),
        max_attempts=2,
    )

    review = reviewer.review(
        {
            "result_id": "sam3-result-retry",
            "semantic_role": "grasp_target",
            "target_prompt": "red block",
            "candidates": [{"id": "detection_000"}],
            "selection_bundle": {
                "original_image_ref": "/tmp/original.png",
                "contact_sheet_ref": "/tmp/contact-sheet.png",
            },
        }
    )

    assert attempts == 2
    assert review["review_attempt_count"] == 2
    assert review["infrastructure_retry_count"] == 1


def test_isolated_sam3_reviewer_reports_complete_bounded_failure() -> None:
    attempts = 0

    def fail(_request: PlannerBackendRequest):
        nonlocal attempts
        attempts += 1
        raise TimeoutError(f"fixture provider timeout {attempts}")

    reviewer = BackendSam3SelectionReviewer(
        CallablePlannerBackend(fail),
        max_attempts=2,
    )

    with pytest.raises(Sam3SelectionReviewError) as error:
        reviewer.review(
            {
                "result_id": "sam3-result-exhausted",
                "semantic_role": "grasp_target",
                "target_prompt": "red block",
                "candidates": [{"id": "detection_000"}],
                "selection_bundle": {
                    "original_image_ref": "/tmp/original.png",
                    "contact_sheet_ref": "/tmp/contact-sheet.png",
                },
            }
        )

    assert attempts == 2
    assert error.value.retry_exhausted is True
    assert error.value.attempt_count == 2
    assert [failure["attempt"] for failure in error.value.failures] == [1, 2]


def test_isolated_sam3_reviewer_recovers_backend_failure_payload_type() -> None:
    attempts = 0

    def fail(_request: PlannerBackendRequest) -> PlannerBackendResult:
        nonlocal attempts
        attempts += 1
        return PlannerBackendResult(
            payload={
                "kind": "response",
                "name": "ask_human",
                "parameters": {
                    "message": "Planner provider request failed.",
                    "error_type": "TimeoutError",
                    "provider_attempts": 1,
                },
                "reasoning": "Planner provider request failed: timed out",
            },
            provider="fixture-provider",
            model="fixture-vlm",
            details={
                "error_type": "TimeoutError",
                "error": "timed out",
                "provider_attempts": 1,
                "provider_role": "primary",
            },
        )

    reviewer = BackendSam3SelectionReviewer(
        CallablePlannerBackend(fail),
        max_attempts=2,
    )

    with pytest.raises(Sam3SelectionReviewError) as error:
        reviewer.review(
            {
                "result_id": "sam3-result-provider-timeout",
                "semantic_role": "grasp_target",
                "target_prompt": "red block",
                "candidates": [{"id": "detection_000"}],
                "selection_bundle": {
                    "original_image_ref": "/tmp/original.png",
                    "contact_sheet_ref": "/tmp/contact-sheet.png",
                },
            }
        )

    assert attempts == 2
    assert [failure["error_type"] for failure in error.value.failures] == [
        "TimeoutError",
        "TimeoutError",
    ]
    provider_details = error.value.failures[0]["provider_details"]
    assert provider_details["provider_attempts"] == 1
    assert provider_details["provider_role"] == "primary"
    assert provider_details["context_strategy"] == "isolated_minimal"
    assert provider_details["conversation_message_count"] == 0
    assert provider_details["request_estimated_chars"] > 0


def test_isolated_sam3_reviewer_defaults_missing_confidence_without_retry() -> None:
    attempts = 0

    def decide(_request: PlannerBackendRequest):
        nonlocal attempts
        attempts += 1
        return {
            "decision": "select",
            "detection_id": "detection_000",
            "reason": "The only tile covers the complete target.",
            "target_geometry_family": "boxed_item",
        }

    review = BackendSam3SelectionReviewer(
        CallablePlannerBackend(decide),
        max_attempts=2,
    ).review(
        {
            "result_id": "sam3-result-missing-confidence",
            "semantic_role": "grasp_target",
            "target_prompt": "red block",
            "candidates": [{"id": "detection_000"}],
            "selection_bundle": {
                "original_image_ref": "/tmp/original.png",
                "contact_sheet_ref": "/tmp/contact-sheet.png",
            },
        }
    )

    assert attempts == 1
    assert review["confidence"] == 0.0
    assert review["confidence_source"] == "default_missing"
    assert review["review_attempt_count"] == 1


def test_exhausted_embedded_review_does_not_start_second_retry_layer() -> None:
    reviewer_calls = 0

    def unexpected_review(_request):
        nonlocal reviewer_calls
        reviewer_calls += 1
        raise AssertionError("a second reviewer layer must not run")

    planner = ToolCallingPlanner(
        CallablePlannerBackend(lambda _request: pytest.fail("main backend must not run")),
        sam3_selection_reviewer=unexpected_review,
    )
    decision = planner._plan_isolated_sam3_selection(
        {
            "result_id": "sam3-result-exhausted",
            "semantic_role": "grasp_target",
            "target_prompt": "red block",
            "candidates": [{"id": "detection_000"}],
            "selection_bundle": {
                "original_image_ref": "/tmp/original.png",
                "contact_sheet_ref": "/tmp/contact-sheet.png",
            },
            "selection_review": {
                "decision": "deferred",
                "attempt_count": 2,
                "infrastructure_retry_exhausted": True,
                "error_type": "Sam3SelectionReviewError",
                "reason": "provider timed out twice",
            },
        },
        tool_context={},
    )

    assert reviewer_calls == 0
    assert decision.action == "ask_human"
    assert decision.parameters["failure_code"] == (
        "sam3_selection_infrastructure_failure"
    )


def test_bounded_review_failure_keeps_candidates_and_retry_evidence() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the red block")
    parameters = {
        "mode": "text",
        "image": "/tmp/original.png",
        "prompt": "red block",
        "semantic_role": "grasp_target",
        "semantic_target": "red block",
    }
    review = {
        "schema_version": "openeta.sam3_selection_review.v1",
        "decision": "deferred",
        "attempt_count": 2,
        "infrastructure_retry_exhausted": True,
        "error_type": "Sam3SelectionReviewError",
        "reason": "provider timed out twice",
    }
    outputs = {
        **parameters,
        "result_id": "sam3-result-exhausted",
        "source_image": "/tmp/original.png",
        "ranking": "score_descending",
        "detections": [{"id": "detection_000", "rank": 0, "score": 0.9}],
        "selection_bundle": {
            "original_image_ref": "/tmp/original.png",
            "contact_sheet_ref": "/tmp/contact-sheet.png",
        },
        "selection_review": review,
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"name": "sam3", "parameters": parameters},
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "parameters": parameters,
                                "outputs": outputs,
                            },
                        },
                    }
                ],
            },
        )
    )

    pending = memory.pending_sam3_selection()
    assert pending is not None
    assert pending["candidates"][0]["id"] == "detection_000"
    assert pending["selection_review"] == review


def test_embedded_selection_review_updates_only_its_semantic_role() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the can and place it in the green region")
    memory.save_fact(
        "selected_sam3_detection",
        {"id": "grasp-mask", "semantic_role": "grasp_target"},
        source="test",
    )
    memory.save_fact(
        "placement_object_detection",
        {
            "id": "held-object-mask",
            "semantic_role": "placement_object",
            "perception_bundle_id": "bundle-current",
            "source_image": "/tmp/current.png",
            "scene_epoch": 0,
        },
        source="test",
    )
    parameters = {
        "mode": "text",
        "image": "/tmp/current.png",
        "prompt": "green placement region",
        "semantic_role": "placement_region",
        "semantic_role_source": "explicit",
        "semantic_target": "placement_region",
        "perception_bundle_id": "bundle-current",
        "observation_id": "observation-7",
        "scene_epoch": 0,
        "attempt_id": "attempt-region-1",
        "attempt_fingerprint": "fingerprint-region-1",
    }
    outputs = {
        **parameters,
        "result_id": "sam3-region-result",
        "source_image": "/tmp/current.png",
        "segmentation_mode": "text",
        "ranking": "score_descending",
        "detections": [
            {
                "id": "detection_000",
                "rank": 0,
                "score": 0.88,
                "mask_ref": "/tmp/region-mask.png",
            }
        ],
        "selection_required": False,
        "selection_review": {
            "schema_version": "openeta.sam3_selection_review.v1",
            "decision": "select",
            "detection_id": "detection_000",
            "confidence": 0.94,
            "reason": "The mask covers the complete green support region.",
            "target_geometry_family": "unknown",
            "selection_source": "isolated_main_vlm",
            "isolated_context": True,
        },
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"name": "sam3", "parameters": parameters},
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {"parameters": parameters, "outputs": outputs},
                        },
                    }
                ],
            },
        )
    )

    assert memory.pending_sam3_selection() is None
    assert memory.selected_sam3_detection()["id"] == "grasp-mask"
    assert memory.placement_object_detection()["id"] == "held-object-mask"
    assert memory.placement_region_detection()["id"] == "detection_000"
    assert memory.placement_region_detection()["target_prompt"] == (
        "green placement region"
    )
    assert memory.placement_region_detection()["perception_bundle_id"] == "bundle-current"
    assert memory.sam3_role_state("placement_region")["status"] == "selected"
    duplicate_error = memory.detection_selection_gate_error(
        tool_name="sam3",
        parameters={"attempt_id": "attempt-region-1"},
    )
    assert duplicate_error is not None
    assert "already completed" in duplicate_error


def test_model_projection_bounds_context_without_deleting_host_evidence() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick and place")
    for index in range(15):
        memory.begin_user_turn(f"bounded follow-up {index}")
    private_marker = "PRIVATE_FULL_QUALIFICATION_PROOF"
    noisy_fields = {f"debug_{index}": "x" * 10_000 for index in range(48)}
    full_context = {
        "task": "pick the soup can and place it in the green region",
        "active_environment_task": {"status": "running", "env_id": "normal-1"},
        "observation": {"task": "pick and place", "camera_ids": ["top"]},
        "vision_image_paths": ["/tmp/top.png"],
        "tool_references": [
            {
                "name": "sam3",
                "category": "perception",
                "description": "segment an image",
                "parameters": {"prompt": "semantic target", **noisy_fields},
                "effect": "read_only",
            },
            {"name": "move_to", "description": "move robot", "parameters": {}},
        ],
        "registered_tool_handlers": ["sam3", "move_to"],
        "semantic_perception_obligation": {
            "status": "semantic_decision_required",
            "semantic_role": "placement_region",
            "required_tool": "sam3",
            **noisy_fields,
        },
        "grasp_candidate_policy": {
            "status": "accepted",
            "qualification_artifact": {
                "proof": private_marker,
                **noisy_fields,
            },
            **noisy_fields,
        },
        "memory": {
            "session_id": memory.session_id,
            "current_user_request": memory.current_user_request,
            "working_memory": {"compact_summary": "summary"},
            "recent_events": [
                {"type": "qualification", "payload": {"proof": private_marker}}
            ],
        },
        "selected_skill_guidance": [],
        "execution_rules": {"default": "one state-changing tool per turn"},
    }

    projected, messages = _model_request_context(
        full_context,
        memory=memory,
        config=PlannerContextConfig(),
        system_prompt=_default_tool_planner_system_prompt(),
        conversation_summary="checkpoint" * 300,
    )

    budget = projected["context_budget"]
    assert budget["projection_level"] in {"hard_compacted", "minimal_hard_bound"}
    assert budget["within_hard_limit"] is True
    assert budget["estimated_tokens"] <= budget["hard_limit_tokens"]
    assert len(messages) <= 9
    assert projected["controller"]["legal_tool_names"] == ["sam3"]
    assert [reference["name"] for reference in projected["tool_references"]] == ["sam3"]
    assert private_marker not in json.dumps(projected)
    assert (
        full_context["grasp_candidate_policy"]["qualification_artifact"]["proof"]
        == private_marker
    )


def test_placement_object_fallback_tries_views_then_one_simplified_prompt() -> None:
    observation = EnvObservation(
        task="pick and place",
        cameras=[],
        robot=RobotState(),
        metadata={"step_idx": 11},
    )
    camera_artifacts = [
        {"kind": "rgb", "frame_id": "agentview", "path": "/tmp/top.png"},
        {"kind": "rgb", "frame_id": "wrist", "path": "/tmp/wrist.png"},
    ]
    base_context = {
        "scene_epoch": 5,
        "selected_sam3_detection": {
            "id": "grasp-mask",
            "target_prompt": "red rectangular block",
        },
        "grasp_execution": {
            "status": "completed",
            "stage": "attached",
            "attachment_mode": "portable_object",
        },
        "attachment_gate": {"status": "resolved", "verdict": "PASS"},
    }
    top_failure = {
        "semantic_role": "placement_object",
        "status": "no_detection",
        "source_image": "/tmp/top.png",
        "mode": "full_frame",
        "scene_epoch": 5,
        "attempt_id": "top-text",
    }
    one_failure_context = {
        **base_context,
        "sam3_semantic_state": {
            "roles": {
                "placement_object": {"canonical_prompt": "red rectangular block"}
            },
            "attempts": [top_failure],
        },
    }

    wrist_retry = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=camera_artifacts,
        memory_context=one_failure_context,
    )

    assert wrist_retry["required_tool"] == "sam3"
    assert wrist_retry["required_parameters"]["image"] == "/tmp/wrist.png"
    assert wrist_retry["required_parameters"]["semantic_role"] == "placement_object"

    two_failure_context = json.loads(json.dumps(one_failure_context))
    two_failure_context["sam3_semantic_state"]["attempts"].append(
        {
            **top_failure,
            "source_image": "/tmp/wrist.png",
            "attempt_id": "wrist-text",
        }
    )
    text_fallback = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=camera_artifacts,
        memory_context=two_failure_context,
    )

    assert text_fallback["required_tool"] == "sam3"
    assert text_fallback["required_parameters"]["image"] == "/tmp/top.png"
    assert text_fallback["required_parameters"]["prompt"] == "red block"
    assert text_fallback["fallback"] == (
        "simplified_text_after_bounded_exact_views"
    )

    two_failure_context["sam3_semantic_state"]["attempts"].append(
        {
            **top_failure,
            "target_prompt": "red block",
            "attempt_id": "top-simplified-text",
        }
    )
    point_fallback = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=camera_artifacts,
        memory_context=two_failure_context,
    )

    assert point_fallback["required_tool"] == "molmopoint"
    assert point_fallback["required_parameters"]["images"] == ["/tmp/top.png"]
    assert point_fallback["attempt"] == 1


def test_placement_object_fallback_projects_trusted_attached_center() -> None:
    observation = EnvObservation(
        task="pick and place",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                rgb=[[[0, 0, 0]] for _ in range(100)],
                intrinsics={
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 50.0,
                    "cy": 50.0,
                    "width": 100,
                    "height": 100,
                },
                extrinsics={
                    "pos": [0.0, 0.0, 0.0],
                    "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "camera_frame": "opencv",
                },
            )
        ],
        robot=RobotState(
            end_effector_pose={
                "xyz": [0.0, 0.0, 1.0],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        ),
        metadata={"step_idx": 11},
    )
    failures = [
        {
            "semantic_role": "placement_object",
            "status": "no_detection",
            "source_image": path,
            "mode": "full_frame",
            "scene_epoch": 5,
            "attempt_id": attempt_id,
        }
        for path, attempt_id in (
            ("/tmp/top.png", "top-text"),
            ("/tmp/wrist.png", "wrist-text"),
        )
    ]
    context = {
        "scene_epoch": 5,
        "selected_sam3_detection": {
            "id": "grasp-mask",
            "target_prompt": "red rectangular block",
        },
        "grasp_execution": {
            "status": "completed",
            "stage": "attached",
            "attachment_mode": "portable_object",
        },
        "attachment_gate": {
            "status": "resolved",
            "verdict": "PASS",
            "full_lift_proof": {
                "attachment_transform": {
                    "schema_version": "openeta.attachment_transform.v1",
                    "parent_frame": "eef",
                    "child_frame": "object",
                    "measurement_boundary": "native_attach_ack",
                    "translation_xyz": [0.0, 0.0, 0.0],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            },
        },
        "sam3_semantic_state": {
            "roles": {
                "placement_object": {"canonical_prompt": "red rectangular block"}
            },
            "attempts": failures,
        },
    }

    fallback = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=[
            {
                "kind": "rgb",
                "frame_id": "agentview",
                "path": "/tmp/top.png",
                "width": 100,
                "height": 100,
            },
            {"kind": "rgb", "frame_id": "wrist", "path": "/tmp/wrist.png"},
        ],
        memory_context=context,
    )

    assert fallback["required_tool"] == "sam3"
    assert fallback["required_parameters"]["mode"] == "points"
    assert fallback["required_parameters"]["points"] == [
        {"x": 50.0, "y": 50.0, "label": 1}
    ]
    assert fallback["required_parameters"]["semantic_target"] == (
        "red rectangular block"
    )
    assert fallback["fallback"] == (
        "attachment_projection_after_bounded_exact_views"
    )


def test_placement_region_fallback_stays_on_object_bundle() -> None:
    observation = EnvObservation(
        task="pick and place",
        cameras=[],
        robot=RobotState(),
        metadata={"step_idx": 12},
    )
    camera_artifacts = [
        {"kind": "rgb", "frame_id": "agentview", "path": "/tmp/top.png"},
        {"kind": "rgb", "frame_id": "wrist", "path": "/tmp/wrist.png"},
    ]
    attached_context = {
        "scene_epoch": 8,
        "selected_sam3_detection": {
            "id": "grasp-mask",
            "target_prompt": "red rectangular block",
        },
        "grasp_execution": {
            "status": "completed",
            "stage": "attached",
            "attachment_mode": "portable_object",
        },
        "attachment_gate": {"status": "resolved", "verdict": "PASS"},
        "sam3_semantic_state": {"roles": {}, "attempts": []},
    }
    object_obligation = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=camera_artifacts,
        memory_context=attached_context,
    )
    object_bundle = object_obligation["perception_bundle_id"]
    region_context = {
        **attached_context,
        "placement_object_detection": {
            "id": "object-mask",
            "source_image": "/tmp/top.png",
            "scene_epoch": 8,
            "perception_bundle_id": object_bundle,
        },
        "sam3_semantic_state": {
            "roles": {
                "placement_region": {"canonical_prompt": "green placement region"}
            },
            "attempts": [
                {
                    "semantic_role": "placement_region",
                    "status": "no_detection",
                    "source_image": "/tmp/top.png",
                    "mode": "full_frame",
                    "scene_epoch": 8,
                    "attempt_id": "region-text",
                }
            ],
        },
    }

    region_fallback = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=camera_artifacts,
        memory_context=region_context,
    )

    assert region_fallback["required_tool"] == "molmopoint"
    assert region_fallback["preferred_image"] == "/tmp/top.png"
    assert region_fallback["required_parameters"]["images"] == ["/tmp/top.png"]
    assert region_fallback["perception_bundle_id"] == object_bundle


def test_active_compiled_placement_does_not_resegment_after_motion() -> None:
    observation = EnvObservation(
        task="pick and place",
        cameras=[],
        robot=RobotState(),
        metadata={"step_idx": 19},
    )
    camera_artifacts = [
        {"kind": "rgb", "frame_id": "agentview", "path": "/tmp/post-hover.png"}
    ]
    memory_context = {
        "scene_epoch": 9,
        "grasp_execution": {
            "status": "completed",
            "stage": "attached",
            "attachment_mode": "portable_object",
        },
        "attachment_gate": {"status": "resolved", "verdict": "PASS"},
        "placement_object_detection": {
            "id": "object-mask",
            "source_image": "/tmp/pre-placement.png",
            "scene_epoch": 8,
            "perception_bundle_id": "placement-bundle",
        },
        "placement_region_detection": {
            "id": "region-mask",
            "source_image": "/tmp/pre-placement.png",
            "scene_epoch": 8,
            "perception_bundle_id": "placement-bundle",
        },
        "placement_candidate_policy": {
            "status": "active",
            "active_candidate_id": "placement_079",
            "compiled_placement": {
                "compiled_placement_id": "compiled-079",
                "scene_epoch": 8,
            },
        },
    }

    obligation = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=camera_artifacts,
        memory_context=memory_context,
    )

    assert obligation is None


def test_molmopoint_result_preserves_host_role_and_bundle_for_point_sam() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick and place")
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "wrist-text-empty",
            "semantic_role": "placement_object",
            "target_prompt": "red rectangular block",
            "source_image": "/tmp/wrist.png",
            "perception_bundle_id": "stale-wrist-bundle",
            "observation_id": "observation-21",
            "scene_epoch": 4,
        },
        source="test",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "metadata": {
                    "planner_metadata": {
                        "host_obligation": {
                            "schema_version": "openeta.semantic_perception_obligation.v1",
                            "semantic_role": "placement_object",
                            "semantic_target": "red rectangular block",
                            "perception_bundle_id": "host-top-bundle",
                            "observation_id": "observation-21",
                            "scene_epoch": 4,
                        }
                    }
                },
                "tool_calls": [
                    {
                        "name": "molmopoint",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "points": [
                                        {"image_index": 0, "pixel_x": 120, "pixel_y": 80}
                                    ],
                                    "image_sources": ["/tmp/top.png"],
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    obligation = memory.pending_reference_localization()
    assert obligation["semantic_role"] == "placement_object"
    assert obligation["scene_image"] == "/tmp/top.png"
    assert obligation["perception_bundle_id"] == "host-top-bundle"
    assert obligation["observation_id"] == "observation-21"


def test_same_materialized_rgb_keeps_bundle_across_agent_step_numbers() -> None:
    identities = []
    for step_idx in (2, 9):
        identities.append(
            _sam3_request_identity(
                observation=EnvObservation(
                    task="pick and place",
                    cameras=[],
                    robot=RobotState(),
                    metadata={"step_idx": step_idx},
                ),
                scene_epoch=3,
                source_image="/tmp/one-materialized-capture.png",
                semantic_role="placement_object",
                semantic_target="red block",
                mode="text",
                prompt="red block",
                points=[],
                roi_bbox_xyxy=None,
            )
        )

    assert identities[0]["observation_id"] == identities[1]["observation_id"]
    assert identities[0]["perception_bundle_id"] == identities[1]["perception_bundle_id"]


def test_explicit_post_create_observe_precedes_semantic_perception() -> None:
    observation = EnvObservation(
        task="normal",
        cameras=[],
        robot=RobotState(),
        metadata={"step_idx": 2},
    )
    camera_artifacts = [
        {"kind": "rgb", "frame_id": "agentview", "path": "/tmp/reset.png"}
    ]
    memory_context = {
        "scene_epoch": 1,
        "current_user_request": (
            "创建环境后先 observe；create 返回的 initial observation 不计作显式 observe。"
        ),
        "latest_environment_receipt": {
            "info": {
                "previous_action": {"request_name": "create_simulator_env"}
            }
        },
        "sam3_semantic_state": {"roles": {}, "attempts": []},
    }

    obligation = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=camera_artifacts,
        memory_context=memory_context,
    )

    assert obligation["required_tool"] == "observe"
    assert obligation["required_parameters"] == {
        "reason": "explicit_post_create_observation_required"
    }
