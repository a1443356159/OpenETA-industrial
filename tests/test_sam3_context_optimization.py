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
    _model_phase_and_legal_tools,
    _operator_control_metadata,
    _operator_planner_mode,
    _operator_semantic_prompts,
    _sam3_request_identity,
    _semantic_camera_view_identity,
    _semantic_perception_obligation,
    _validate_sam3_parameters,
)
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.sam3_selection import (
    BackendSam3SelectionReviewer,
    Sam3SelectionReviewError,
    Sam3SelectionParentContext,
)
from agent.tools.handlers import _sam3_semantic_metadata


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
    assert request.conversation_messages == []
    assert request.conversation_summary == ""
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
    assert "task" not in request.tool_context
    assert "selected_skill_guidance" not in request.tool_context
    assert (
        request.tool_context["selection_obligation"]["target_prompt"]
        == "red rectangular block"
    )
    review_rules = request.tool_context["selection_review_contract"]["rules"]
    assert any("target_prompt is authoritative" in rule for rule in review_rules)
    assert any("original RGB for colour" in rule for rule in review_rules)
    assert any("raw RGB crop" in rule for rule in review_rules)


def test_parent_fork_does_not_leak_previous_multi_object_step_into_review() -> None:
    parent_context = Sam3SelectionParentContext()
    parent_context.capture(
        PlannerBackendRequest(
            system_prompt="parent planner system contract",
            tool_context={
                "task": "first move the yellow wrench, then move the red bolt",
                "memory": {"current_step": "yellow adjustable wrench"},
                "selected_skill_guidance": [{"name": "pick", "target": "wrench"}],
            },
            conversation_messages=[
                {
                    "role": "user",
                    "content": "First move the yellow wrench, then the red bolt.",
                }
            ],
            conversation_summary="The yellow wrench is the first requested item.",
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
                    "sam3_result_id": "red-bolt-result",
                    "detection_id": "detection_000",
                    "reason": "The mask covers the complete red bolt in the RGB.",
                },
                "reasoning": "The frozen bundle defines the current target.",
            }
        )

    review = BackendSam3SelectionReviewer(
        CallablePlannerBackend(decide),
        parent_context=parent_context,
    ).review(
        {
            "result_id": "red-bolt-result",
            "semantic_role": "grasp_target",
            "target_prompt": "red hex bolt",
            "candidates": [{"id": "detection_000", "rank": 0}],
            "selection_bundle": {
                "original_image_ref": "/tmp/original.png",
                "contact_sheet_ref": "/tmp/contact-sheet.png",
            },
        }
    )

    assert review["decision"] == "select"
    request = requests[0]
    serialized = json.dumps(
        {
            "tool_context": request.tool_context,
            "messages": request.conversation_messages,
            "summary": request.conversation_summary,
        }
    )
    assert "yellow wrench" not in serialized
    assert "current_step" not in serialized
    assert request.tool_context["selection_obligation"]["target_prompt"] == (
        "red hex bolt"
    )


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


def test_exhausted_embedded_review_waits_for_operator_before_new_retry_cycle() -> None:
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
    assert decision.parameters.get("terminal_handoff") is not True


def test_operator_answer_starts_new_bounded_review_for_unchanged_sam3_bundle() -> None:
    reviewer_calls = 0

    def review(_request):
        nonlocal reviewer_calls
        reviewer_calls += 1
        return {
            "decision": "select",
            "detection_id": "detection_000",
            "confidence": 0.9,
            "reason": "The complete red block is visible in the first tile.",
            "target_geometry_family": "boxed_item",
            "provider": "fixture-provider",
            "model": "fixture-vlm",
            "provider_details": {},
        }

    planner = ToolCallingPlanner(
        CallablePlannerBackend(lambda _request: pytest.fail("main backend must not run")),
        sam3_selection_reviewer=review,
    )
    question = (
        "The bounded SAM3 semantic reviewer exhausted its retry. Check the "
        "planner/VLM service, then answer here to retry this unchanged candidate bundle."
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
        tool_context={
            "memory": {
                "latest_human_interaction": {
                    "question": question,
                    "answer": "The VLM service is healthy now; continue.",
                }
            }
        },
    )

    assert reviewer_calls == 1
    assert decision.action == "select_sam3_detection"
    assert decision.parameters["sam3_result_id"] == "sam3-result-exhausted"
    assert decision.parameters["detection_id"] == "detection_000"


def test_isolated_selection_charges_provider_usage_to_episode_action() -> None:
    planner = ToolCallingPlanner(
        CallablePlannerBackend(lambda _request: pytest.fail("general planner must not run")),
        sam3_selection_reviewer=lambda _request: {
            "decision": "select",
            "detection_id": "detection_000",
            "confidence": 0.9,
            "reason": "The tile covers the complete red block.",
            "target_geometry_family": "boxed_item",
            "provider": "fixture-provider",
            "model": "fixture-vlm",
            "provider_details": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
                "usage_source": "provider",
            },
        },
    )

    decision = planner._plan_isolated_sam3_selection(
        {
            "result_id": "sam3-result-usage",
            "semantic_role": "grasp_target",
            "target_prompt": "red block",
            "candidates": [{"id": "detection_000"}],
            "selection_bundle": {
                "original_image_ref": "/tmp/original.png",
                "contact_sheet_ref": "/tmp/contact-sheet.png",
            },
        },
        tool_context={"planner_mode": "agentic_closed_loop"},
    )

    assert decision.metadata["backend_usage"]["total_tokens"] == 120
    assert decision.metadata["backend_usage_sources"] == {"provider": 1}
    assert decision.metadata["backend_provider"] == "fixture-provider"
    assert decision.metadata["backend_model"] == "fixture-vlm"


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
    assert budget["projection_level"] in {
        "soft_compacted",
        "hard_compacted",
        "minimal_hard_bound",
    }
    assert budget["within_hard_limit"] is True
    assert budget["estimated_tokens"] <= budget["hard_limit_tokens"]
    assert len(messages) <= 5
    assert projected["controller"]["legal_tool_names"] == ["sam3"]
    assert [reference["name"] for reference in projected["tool_references"]] == ["sam3"]
    assert projected["schema_version"] == "openeta.planner_model_context.v4"
    assert "semantic_perception_obligation" not in projected
    assert "grasp_candidate_policy" not in projected
    assert "current_camera_artifacts" not in projected
    assert "current_camera_calibrations" not in projected
    assert "current_user_request" not in projected["memory"]
    assert "recent_events" not in projected["memory"]
    assert private_marker not in json.dumps(projected)
    assert (
        full_context["grasp_candidate_policy"]["qualification_artifact"]["proof"]
        == private_marker
    )


def test_model_projection_deduplicates_task_and_omits_images_for_typed_action() -> None:
    task = "pick the red block"
    memory = AgentMemory()
    memory.start_session(task=task)
    full_context = {
        "task": task,
        "planner_mode": "agentic_closed_loop",
        "observation": {"task": task, "camera_ids": ["top"]},
        "vision_image_paths": ["/tmp/top.png"],
        "current_rgbd_views": [
            {
                "frame_id": "top",
                "rgb_path": "/tmp/top.png",
                "depth_path": "/tmp/top-depth.png",
            }
        ],
        "tool_references": [
            {"name": "observe", "description": "observe", "parameters": {}}
        ],
        "fresh_observation_obligation": {
            "status": "required",
            "required_tool": "observe",
            "required_parameters": {"reason": "fresh_state"},
        },
        "memory": {"metadata": {}, "latest_human_interaction": None},
        "selected_skill_guidance": [],
        "skill_usage": {},
    }

    projected, messages = _model_request_context(
        full_context,
        memory=memory,
        config=PlannerContextConfig(),
    )

    assert messages == []
    assert projected["vision_image_paths"] == []
    assert projected["current_rgbd_views"] == []
    obligation = projected["obligations"]["fresh_observation_obligation"]
    assert obligation["required_tool"] == "observe"
    assert obligation["parameter_mode"] == "host_hydrated"
    assert obligation["parameter_keys"] == ["reason"]
    assert len(obligation["parameter_binding_sha256"]) == 64
    assert "required_parameters" not in obligation


def test_model_projection_compacts_unique_host_bound_atomic_action() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the red block")
    skill_content = "Use the complete manipulation playbook. " * 200
    full_context = {
        "task": "pick the red block",
        "planner_mode": "agentic_closed_loop",
        "observation": {"camera_ids": ["top"]},
        "active_environment_task": {"status": "running"},
        "selected_sam3_detection": {"id": "target-mask"},
        "frozen_placement_goal_pool": {"status": "frozen"},
        "grasp_candidate_policy": {"status": "accepted"},
        "grasp_execution": {
            "status": "required",
            "stage": "contact",
            "required_action": {
                "name": "move_to",
                "parameters": {
                    "target_pose": {"frame_id": "world", "position": [0.4, 0.1, 0.3]},
                    "collision_check": True,
                },
            },
        },
        "tool_references": [
            {
                "name": "move_to",
                "description": "Move to the host-qualified target. " * 100,
                "parameters": {"target_pose": "pose", "collision_check": "boolean"},
            },
            {"name": "observe", "description": "observe", "parameters": {}},
        ],
        "memory": {"metadata": {}},
        "selected_skill_guidance": [
            {
                "name": "pick",
                "description": "Pick an object safely.",
                "allowed_tools": ["move_to", "observe"],
                "content": skill_content,
                "version": "1",
            }
        ],
        "skill_usage": {"inspection_recommended": ["pick"]},
    }

    projected, _messages = _model_request_context(
        full_context,
        memory=memory,
        config=PlannerContextConfig(),
    )

    assert projected["controller"]["legal_tool_names"] == ["move_to"]
    binding = projected["controller"]["host_parameter_binding"]
    assert binding["tool"] == "move_to"
    assert binding["parameter_mode"] == "host_hydrated"
    assert binding["parameter_keys"] == ["collision_check", "target_pose"]
    assert len(binding["parameter_binding_sha256"]) == 64
    assert binding["sources"] == ["grasp_execution"]
    assert "content" not in projected["selected_skill_guidance"][0]
    assert len(projected["tool_references"][0]["description"]) <= 240
    required_action = projected["state"]["grasp_execution"]["required_action"]
    assert "parameters" not in required_action
    assert required_action["parameter_binding_sha256"] == binding[
        "parameter_binding_sha256"
    ]


def test_model_projection_keeps_skill_for_unbound_semantic_decision() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the red block")
    skill_content = "Inspect the image and ground the requested object."
    full_context = {
        "task": "pick the red block",
        "planner_mode": "agentic_closed_loop",
        "observation": {"camera_ids": ["top"]},
        "active_environment_task": {"status": "running"},
        "tool_references": [
            {"name": "observe", "description": "observe", "parameters": {}},
            {"name": "sam3", "description": "segment", "parameters": {}},
        ],
        "memory": {"metadata": {}},
        "selected_skill_guidance": [
            {
                "name": "pick",
                "description": "Pick an object safely.",
                "allowed_tools": ["observe", "sam3"],
                "content": skill_content,
                "version": "1",
            }
        ],
        "skill_usage": {},
    }

    projected, _messages = _model_request_context(
        full_context,
        memory=memory,
        config=PlannerContextConfig(),
    )

    assert projected["controller"]["legal_tool_names"] == ["observe", "sam3"]
    assert "host_parameter_binding" not in projected["controller"]
    assert projected["selected_skill_guidance"][0]["content"] == skill_content


def test_model_projection_keeps_skill_when_host_binding_is_ambiguous() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the red block")
    skill_content = "Choose the correct host-qualified branch."
    full_context = {
        "task": "pick the red block",
        "planner_mode": "agentic_closed_loop",
        "observation": {"camera_ids": ["top"]},
        "active_environment_task": {"status": "running"},
        "selected_sam3_detection": {"id": "target-mask"},
        "frozen_placement_goal_pool": {"status": "frozen"},
        "grasp_candidate_policy": {
            "status": "required",
            "required_action": {
                "name": "move_to",
                "parameters": {"candidate_id": "candidate-a"},
            },
        },
        "grasp_execution": {
            "status": "required",
            "stage": "contact",
            "required_action": {
                "name": "move_to",
                "parameters": {"candidate_id": "candidate-b"},
            },
        },
        "tool_references": [
            {"name": "move_to", "description": "move", "parameters": {}}
        ],
        "memory": {"metadata": {}},
        "selected_skill_guidance": [
            {
                "name": "pick",
                "allowed_tools": ["move_to"],
                "content": skill_content,
            }
        ],
        "skill_usage": {},
    }

    projected, _messages = _model_request_context(
        full_context,
        memory=memory,
        config=PlannerContextConfig(),
    )

    assert projected["controller"]["legal_tool_names"] == ["move_to"]
    assert "host_parameter_binding" not in projected["controller"]
    assert projected["selected_skill_guidance"][0]["content"] == skill_content


def test_model_projection_summarizes_durable_evidence_and_keeps_only_latest_feedback() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick and place")
    private_marker = "PRIVATE_QUALIFICATION_MATRIX"
    for index in range(4):
        memory.begin_user_turn(f"follow-up {index} {private_marker * 50}")
    detection = {
        "id": "target-mask",
        "result_id": "sam3-result",
        "semantic_role": "grasp_target",
        "target_prompt": "red bolt",
        "score": 0.97,
        "mask_ref": "/tmp/target-mask.png",
        "source_image": "/tmp/top.png",
        "overlay_ref": private_marker * 100,
    }
    full_context = {
        "task": "pick and place",
        "planner_mode": "agentic_closed_loop",
        "active_environment_task": {"status": "running", "env_id": "normal"},
        "observation": {"camera_ids": ["top"], "metadata": {"scene_epoch": 4}},
        "selected_sam3_detection": dict(detection),
        "placement_object_detection": dict(detection),
        "sam3_semantic_state": {
            "schema_version": "openeta.sam3_semantic_state.v1",
            "roles": {"grasp_target": {"status": "selected", **detection}},
            "attempts": [
                {"result": private_marker * 100, "candidate_count": 4}
                for _ in range(12)
            ],
        },
        "grasp_candidate_policy": {
            "status": "frozen_frontier_required",
            "candidate_count": 232,
            "frozen_grasp_frontier_remaining_count": 168,
            "qualification_artifact": {"proof": private_marker * 100},
            "candidates_summary": [
                {"pose": private_marker * 100} for _ in range(100)
            ],
        },
        "grasp_frontier_obligation": {
            "status": "required",
            "required_tool": "grasp_pose_estimate",
            "required_parameters": {
                "mode": "frozen_frontier",
                "model_inference": False,
                "scene_revision": 4,
            },
        },
        "tool_references": [
            {
                "name": "grasp_pose_estimate",
                "description": "x" * 5_000,
                "parameters": {"mode": "string", "model_inference": "boolean"},
            }
        ],
        "memory": {"metadata": {}},
        "selected_skill_guidance": [],
        "skill_usage": {},
    }

    projected, messages = _model_request_context(
        full_context,
        memory=memory,
        config=PlannerContextConfig(),
    )

    serialized = json.dumps(projected, sort_keys=True)
    assert private_marker not in serialized
    assert len(messages) <= 1
    assert projected["state"]["grasp_candidate_policy"] == {
        "status": "frozen_frontier_required",
        "candidate_count": 232,
        "frozen_grasp_frontier_remaining_count": 168,
    }
    assert projected["state"]["sam3_semantic_state"]["attempt_count"] == 12
    assert len(projected["tool_references"][0]["description"]) <= 240
    assert projected["obligations"]["grasp_frontier_obligation"][
        "parameter_mode"
    ] == "host_hydrated"


def test_model_projection_hides_anyplace_paths_and_release_geometry() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick and place")
    private_rgb = "/private/session/very-long-uuid/object.rgb.png"
    private_depth = "/private/session/very-long-uuid/object.depth.png"
    private_pose_marker = "PRIVATE_RELEASE_POSE_MARKER"
    full_context = {
        "task": "pick and place",
        "planner_mode": "agentic_closed_loop",
        "observation": {"task": "pick and place", "camera_ids": ["top"]},
        "active_environment_task": {"status": "running"},
        "tool_references": [
            {"name": "anyplace", "description": "place", "parameters": {}},
            {"name": "move_to", "description": "move", "parameters": {}},
        ],
        "placement_obligation": {
            "schema_version": "openeta.placement_obligation.v2",
            "required_tool": "anyplace",
            "required_parameters": {
                "object_observation": {
                    "rgb": private_rgb,
                    "depth": private_depth,
                },
                "scene_revision": 4,
            },
            "phase": "frozen_goal_pool",
        },
        "placement_motion_guidance": {
            "schema_version": "openeta.placement_motion_guidance.v1",
            "status": "required",
            "stage": "release",
            "release_pose": {"private_marker": private_pose_marker},
            "required_parameters": {
                "target_pose": {"private_marker": private_pose_marker},
                "tolerance": 0.002,
            },
        },
        "memory": {"metadata": {}},
        "selected_skill_guidance": [],
        "skill_usage": {},
    }

    projected, _messages = _model_request_context(
        full_context,
        memory=memory,
        config=PlannerContextConfig(),
    )

    serialized = json.dumps(projected, sort_keys=True)
    assert private_rgb not in serialized
    assert private_depth not in serialized
    assert private_pose_marker not in serialized
    assert projected["obligations"]["placement_obligation"]["parameter_mode"] == (
        "host_hydrated"
    )
    motion = projected["obligations"]["placement_motion_guidance"]
    assert motion["required_tool"] == "move_to"
    assert motion["parameter_mode"] == "host_hydrated"
    assert "required_parameters" not in motion


def test_attached_object_never_triggers_postattach_semantic_retry() -> None:
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
    context = {
        **base_context,
        "frozen_placement_goal_pool": {
            "schema_version": "openeta.frozen_placement_goal_pool.v1",
            "status": "ready",
            "goal_count": 96,
        },
        "sam3_semantic_state": {
            "roles": {
                "placement_object": {"canonical_prompt": "red rectangular block"}
            },
            "attempts": [],
        },
    }

    obligation = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=camera_artifacts,
        memory_context=context,
    )

    assert obligation is None


def test_attached_object_projection_does_not_trigger_resegmentation() -> None:
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
            "attachment_proof": {
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

    assert fallback is None


def test_attached_object_never_starts_a_new_region_bundle() -> None:
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
    assert object_obligation is None


def test_active_compiled_placement_does_not_resegment_after_motion() -> None:
    observation = EnvObservation(
        task="pick and place",
        cameras=[],
        robot=RobotState(),
        metadata={"step_idx": 19},
    )
    camera_artifacts = [
        {"kind": "rgb", "frame_id": "agentview", "path": "/tmp/post-attach.png"}
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


def test_out_of_band_scripted_metadata_keeps_human_task_natural(monkeypatch) -> None:
    monkeypatch.setenv(
        "OPENETA_SCRIPTED_TASK_METADATA",
        "[automation=scripted_tui; planner_mode=agentic_closed_loop; "
        "grasp_target=yellow_adjustable_wrench]",
    )
    memory_context = {
        "scene_epoch": 1,
        "current_user_request": (
            "环境打开以后先看一眼工作台，再把黄色活动扳手放进绿色料箱。"
        ),
        "sam3_semantic_state": {"roles": {}, "attempts": []},
    }

    obligation = _semantic_perception_obligation(
        observation=EnvObservation(
            task="normal",
            cameras=[],
            robot=RobotState(),
            metadata={"step_idx": 2},
        ),
        camera_artifacts=[
            {"kind": "rgb", "frame_id": "agentview", "path": "/tmp/reset.png"}
        ],
        memory_context=memory_context,
    )

    assert "automation=" not in memory_context["current_user_request"]
    assert obligation["required_tool"] == "sam3"
    assert obligation["semantic_role"] == "grasp_target"


def test_scripted_fixed_semantics_dispatch_sam3_without_model_routing_turn() -> None:
    scripted_request = (
        "[automation=scripted_tui; environment_id=openeta/test-v0; "
        "environment_task=normal_pick_and_place; "
        "grasp_target=red_rectangular_block; "
        "placement_object=red_rectangular_block; "
        "placement_region=green_placement_zone_marker] run acceptance"
    )
    assert _operator_semantic_prompts(scripted_request) == {
        "grasp_target": "red rectangular block",
        "placement_object": "red rectangular block",
        "placement_region": "green placement zone marker",
    }
    obligation = _semantic_perception_obligation(
        observation=EnvObservation(
            task="normal pick and place",
            cameras=[],
            robot=RobotState(),
            metadata={"step_idx": 2},
        ),
        camera_artifacts=[
            {"kind": "rgb", "frame_id": "agentview", "path": "/tmp/top.png"},
            {"kind": "rgb", "frame_id": "wrist", "path": "/tmp/wrist.png"},
        ],
        memory_context={
            "scene_epoch": 1,
            "current_user_request": scripted_request,
            "latest_environment_receipt": {
                "info": {"previous_action": {"request_name": "observe"}}
            },
            "sam3_semantic_state": {"roles": {}, "attempts": []},
        },
    )

    assert obligation is not None
    assert obligation["status"] == "required"
    assert obligation["required_tool"] == "sam3"
    assert obligation["semantic_role"] == "grasp_target"
    assert obligation["semantic_target"] == "red rectangular block"
    assert obligation["required_parameters"]["prompt"] == "red rectangular block"
    assert obligation["required_parameters"]["image"] == "/tmp/top.png"


def test_initial_grasp_target_prefers_scene_camera_even_with_calibrated_wrist() -> None:
    observation = EnvObservation(
        task="normal pick and place",
        cameras=[
            CameraFrame(
                frame_id="top_camera_optical_frame",
                role="scene_primary",
                rgb=[],
                extrinsics={
                    "camera_frame": "opencv",
                    "pos": [0.35, 0.0, 1.3],
                    "mat": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
                },
            ),
            CameraFrame(
                frame_id="wrist_camera_optical_frame",
                role="wrist_primary",
                rgb=[],
                extrinsics={
                    "frame_transform": "camera_to_world",
                    "camera_frame": "opencv",
                    "pos": [0.0, -0.05, 0.996],
                    "quat_xyzw": [0.63, -0.73, 0.20, -0.17],
                    "calibration_source": "tf2_at_rgb_timestamp",
                },
            ),
        ],
        robot=RobotState(),
        metadata={"step_idx": 2},
    )
    camera_artifacts = [
        {
            "kind": "rgb",
            "frame_id": "top_camera_optical_frame",
            "role": "scene_primary",
            "path": "/tmp/top.png",
        },
        {
            "kind": "depth",
            "frame_id": "top_camera_optical_frame",
            "role": "scene_primary",
            "path": "/tmp/top-depth.png",
        },
        {
            "kind": "rgb",
            "frame_id": "wrist_camera_optical_frame",
            "role": "wrist_primary",
            "path": "/tmp/wrist.png",
        },
        {
            "kind": "depth",
            "frame_id": "wrist_camera_optical_frame",
            "role": "wrist_primary",
            "path": "/tmp/wrist-depth.png",
        },
    ]

    obligation = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=camera_artifacts,
        memory_context={
            "scene_epoch": 1,
            "sam3_semantic_state": {
                "roles": {
                    "grasp_target": {
                        "canonical_prompt": "red hex bolt",
                        "scene_epoch": 1,
                    }
                },
                "attempts": [],
            },
        },
    )

    assert obligation is not None
    assert obligation["semantic_role"] == "grasp_target"
    assert obligation["required_tool"] == "sam3"
    assert obligation["required_parameters"]["image"] == "/tmp/top.png"
    assert obligation["required_parameters"]["prompt"] == "red hex bolt"


def test_work_order_uses_catalog_perception_prompt_for_placement_segmentation() -> None:
    obligation = _semantic_perception_obligation(
        observation=EnvObservation(
            task="normal pick and place",
            cameras=[],
            robot=RobotState(),
            metadata={"step_idx": 2},
        ),
        camera_artifacts=[
            {
                "kind": "rgb",
                "frame_id": "top_camera_optical_frame",
                "role": "scene_primary",
                "path": "/tmp/top.png",
            }
        ],
        memory_context={
            "scene_epoch": 1,
            "selected_sam3_detection": {"id": "yellow-wrench"},
            "placement_object_detection": {
                "id": "yellow-wrench-object",
                "source_image": "/tmp/top.png",
                "scene_epoch": 1,
                "perception_bundle_id": "bundle-1",
            },
            "multi_sort_progress": {
                "active_assignment": {
                    "placement_region_prompt": "blue parts bin",
                    "placement_region_perception_prompt": (
                        "blue square area inside bin"
                    ),
                }
            },
            "sam3_semantic_state": {"roles": {}, "attempts": []},
        },
    )

    assert obligation is not None
    assert obligation["semantic_role"] == "placement_region"
    assert obligation["semantic_target"] == "blue square area inside bin"
    assert obligation["required_parameters"]["prompt"] == (
        "blue square area inside bin"
    )


def test_vlm_work_order_batches_target_and_region_on_one_frozen_observation() -> None:
    observation = EnvObservation(
        task="sort two industrial parts",
        cameras=[],
        robot=RobotState(),
        metadata={"step_idx": 2, "observation_id": "capture-42"},
    )
    obligation = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=[
            {
                "kind": "rgb",
                "frame_id": "top_camera_optical_frame",
                "role": "scene_primary",
                "path": "/tmp/top.png",
            }
        ],
        memory_context={
            "scene_epoch": 7,
            "work_order": {
                "schema_version": "openeta.work_order.v1",
                "source": "vlm_tool_call",
                "items": [
                    {
                        "target_prompt": "yellow adjustable wrench",
                        "placement_region_prompt": "green parts bin",
                    }
                ],
            },
            "multi_sort_progress": {
                "active_assignment": {
                    "target_perception_prompt": "yellow adjustable wrench",
                    "placement_region_perception_prompt": "green parts bin interior",
                }
            },
            "sam3_semantic_state": {"roles": {}, "attempts": []},
        },
    )

    assert obligation is not None
    assert obligation["required_tool"] == "sam3"
    assert obligation["semantic_roles"] == ["grasp_target", "placement_region"]
    parameters = obligation["required_parameters"]
    assert parameters["mode"] == "assignment_batch"
    requests = parameters["requests"]
    assert [request["semantic_role"] for request in requests] == [
        "grasp_target",
        "placement_region",
    ]
    assert [request["prompt"] for request in requests] == [
        "yellow adjustable wrench",
        "green parts bin interior",
    ]
    assert len({request["image"] for request in requests}) == 1
    assert len({request["perception_bundle_id"] for request in requests}) == 1
    assert len({request["observation_id"] for request in requests}) == 1
    assert len({request["attempt_id"] for request in requests}) == 2
    assert _validate_sam3_parameters(parameters) == []

    mismatched = {
        **parameters,
        "requests": [dict(requests[0]), {**requests[1], "image": "/tmp/other.png"}],
    }
    assert any("same `image`" in error for error in _validate_sam3_parameters(mismatched))


def test_assignment_batch_requires_vlm_authored_work_order() -> None:
    obligation = _semantic_perception_obligation(
        observation=EnvObservation(
            task="pick and place",
            cameras=[],
            robot=RobotState(),
            metadata={"step_idx": 1},
        ),
        camera_artifacts=[
            {"kind": "rgb", "frame_id": "agentview", "path": "/tmp/top.png"}
        ],
        memory_context={
            "scene_epoch": 1,
            "multi_sort_progress": {
                "active_assignment": {
                    "target_perception_prompt": "yellow wrench",
                    "placement_region_perception_prompt": "green bin interior",
                }
            },
            "sam3_semantic_state": {"roles": {}, "attempts": []},
        },
    )

    assert obligation is not None
    assert obligation["required_parameters"]["mode"] == "text"
    assert obligation["semantic_role"] == "grasp_target"


def test_memory_expands_assignment_batch_into_two_independent_sam3_records() -> None:
    def child(role: str, result_id: str, prompt: str, mask_ref: str) -> dict:
        parameters = {
            "mode": "text",
            "image": "/tmp/top.png",
            "prompt": prompt,
            "semantic_role": role,
            "semantic_target": prompt,
            "perception_bundle_id": "bundle-1",
            "observation_id": "observation-1",
            "scene_epoch": 0,
            "attempt_id": f"{role}-attempt",
        }
        detection = {
            "id": "detection_000",
            "label": prompt,
            "score": 0.9,
            "mask_ref": mask_ref,
        }
        return {
            "success": True,
            "content": "SAM3 segmentation completed.",
            "details": {
                "parameters": parameters,
                "outputs": {
                    **parameters,
                    "result_id": result_id,
                    "source_image": "/tmp/top.png",
                    "detection_count": 1,
                    "detections": [detection],
                    "selected_detection": detection,
                    "selection_required": False,
                    "selection_review": {
                        "decision": "select",
                        "detection_id": "detection_000",
                        "selection_source": "host_singleton_text_detection",
                    },
                },
                "artifacts": [],
            },
        }

    target = child("grasp_target", "target-result", "yellow wrench", "/tmp/target.png")
    region = child(
        "placement_region",
        "region-result",
        "green bin interior",
        "/tmp/region.png",
    )
    memory = AgentMemory()
    memory.start_session(task="sort one part")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "schema_version": "openeta.sam3.assignment_batch.v1",
                                    "semantic_batch_results": [
                                        {
                                            "index": 0,
                                            "semantic_role": "grasp_target",
                                            "result": target,
                                        },
                                        {
                                            "index": 1,
                                            "semantic_role": "placement_region",
                                            "result": region,
                                        },
                                    ],
                                }
                            },
                        },
                    }
                ]
            },
        )
    )

    assert memory.selected_sam3_detection()["mask_ref"] == "/tmp/target.png"
    assert memory.placement_object_detection()["mask_ref"] == "/tmp/target.png"
    assert memory.placement_region_detection()["mask_ref"] == "/tmp/region.png"
    assert memory.pending_sam3_selection() is None
    assert [
        attempt["semantic_role"] for attempt in memory.sam3_semantic_state()["attempts"]
    ] == ["grasp_target", "placement_region"]


def _grasp_target_retry_obligation(
    attempts: list[dict[str, object]],
    *,
    prompt: str = "red rectangular block",
    available_tool_names: set[str] | None = None,
) -> dict[str, object]:
    obligation = _semantic_perception_obligation(
        observation=EnvObservation(
            task="normal pick and place",
            cameras=[],
            robot=RobotState(),
            metadata={"step_idx": 2},
        ),
        camera_artifacts=[
            {"kind": "rgb", "frame_id": "agentview", "path": "/tmp/top.png"},
            {"kind": "rgb", "frame_id": "wrist", "path": "/tmp/wrist.png"},
        ],
        memory_context={
            "scene_epoch": 4,
            "sam3_semantic_state": {
                "roles": {
                    "grasp_target": {
                        "canonical_prompt": prompt,
                        "scene_epoch": 4,
                    }
                },
                "attempts": attempts,
            },
        },
        available_tool_names=available_tool_names,
    )
    assert obligation is not None
    return obligation


def _failed_grasp_target_attempt(
    *, source_image: str, prompt: str, attempt_id: str
) -> dict[str, object]:
    return {
        "semantic_role": "grasp_target",
        "status": "no_detection",
        "source_image": source_image,
        "mode": "text",
        "target_prompt": prompt,
        "scene_epoch": 4,
        "attempt_id": attempt_id,
        "attempt_fingerprint": f"fingerprint-{attempt_id}",
    }


def test_grasp_target_no_detection_advances_to_next_exact_view() -> None:
    obligation = _grasp_target_retry_obligation(
        [
            _failed_grasp_target_attempt(
                source_image="/tmp/top.png",
                prompt="red rectangular block",
                attempt_id="top-exact",
            )
        ]
    )

    assert obligation["required_tool"] == "sam3"
    assert obligation["required_parameters"]["image"] == "/tmp/wrist.png"
    assert obligation["required_parameters"]["prompt"] == "red rectangular block"
    assert (
        obligation["required_parameters"]["attempt_fingerprint"]
        != "fingerprint-top-exact"
    )


def test_grasp_target_exact_views_advance_to_one_simplified_prompt() -> None:
    exact_attempts = [
        _failed_grasp_target_attempt(
            source_image=source,
            prompt="red rectangular block",
            attempt_id=attempt_id,
        )
        for source, attempt_id in (
            ("/tmp/top.png", "top-exact"),
            ("/tmp/wrist.png", "wrist-exact"),
        )
    ]

    obligation = _grasp_target_retry_obligation(exact_attempts)

    assert obligation["fallback"] == "simplified_text_after_bounded_exact_views"
    assert obligation["required_parameters"]["image"] == "/tmp/top.png"
    assert obligation["required_parameters"]["prompt"] == "red block"
    assert obligation["canonical_semantic_target"] == "red rectangular block"


def test_grasp_target_long_subtype_prompt_backs_off_to_attribute_and_head() -> None:
    exact_attempts = [
        _failed_grasp_target_attempt(
            source_image=source,
            prompt="yellow adjustable wrench",
            attempt_id=attempt_id,
        )
        for source, attempt_id in (
            ("/tmp/top.png", "top-exact"),
            ("/tmp/wrist.png", "wrist-exact"),
        )
    ]

    obligation = _grasp_target_retry_obligation(
        exact_attempts,
        prompt="yellow adjustable wrench",
    )

    assert obligation["fallback"] == "simplified_text_after_bounded_exact_views"
    assert obligation["required_parameters"]["prompt"] == "yellow wrench"
    assert obligation["canonical_semantic_target"] == "yellow adjustable wrench"


def test_grasp_target_text_budget_advances_to_visual_point_localization() -> None:
    attempts = [
        _failed_grasp_target_attempt(
            source_image=source,
            prompt=prompt,
            attempt_id=attempt_id,
        )
        for source, prompt, attempt_id in (
            ("/tmp/top.png", "red rectangular block", "top-exact"),
            ("/tmp/wrist.png", "red rectangular block", "wrist-exact"),
            ("/tmp/top.png", "red block", "top-simplified"),
        )
    ]

    obligation = _grasp_target_retry_obligation(attempts)

    assert obligation["status"] == "required"
    assert obligation["required_tool"] == "molmopoint"
    assert obligation["fallback"] == "point_localization_after_bounded_text_views"


def test_missing_molmopoint_advances_to_provider_grounded_active_view() -> None:
    attempts = [
        _failed_grasp_target_attempt(
            source_image=source,
            prompt=prompt,
            attempt_id=attempt_id,
        )
        for source, prompt, attempt_id in (
            ("/tmp/top.png", "red rectangular block", "top-exact"),
            ("/tmp/wrist.png", "red rectangular block", "wrist-exact"),
            ("/tmp/top.png", "red block", "top-simplified"),
        )
    ]

    obligation = _grasp_target_retry_obligation(
        attempts,
        available_tool_names={"observe", "sam3", "active_observe"},
    )

    assert obligation["status"] == "required"
    assert obligation["required_tool"] == "active_observe"
    assert obligation["required_parameters"] == {
        "semantic_target": "red rectangular block",
        "semantic_role": "grasp_target",
        "quality_profile": "grasp_rgbd",
        "max_motion_attempts": 2,
    }
    assert obligation["fallback"] == (
        "active_view_with_isolated_provider_grounding"
    )

    phase, legal = _model_phase_and_legal_tools(
        {
            "semantic_perception_obligation": obligation,
            "active_environment_task": {"env_id": "openeta/test-v0"},
            "tool_references": [
                {"name": "observe"},
                {"name": "sam3"},
                {"name": "active_observe"},
            ],
        },
        max_tools=8,
    )
    assert phase == "target_perception"
    assert legal[0] == "active_observe"


def test_missing_placement_point_service_uses_one_current_view_active_grounding() -> None:
    failed_attempt = {
        "semantic_role": "placement_region",
        "status": "rejected",
        "source_image": "/tmp/top.png",
        "mode": "text",
        "target_prompt": "blue square area inside bin",
        "scene_epoch": 4,
        "attempt_id": "blue-bin-text",
        "attempt_fingerprint": "fingerprint-blue-bin-text",
    }

    def obligation_for(attempts):
        value = _semantic_perception_obligation(
            observation=EnvObservation(
                task="sort parts",
                cameras=[],
                robot=RobotState(),
                metadata={"step_idx": 2},
            ),
            camera_artifacts=[
                {
                    "kind": "rgb",
                    "frame_id": "top_camera_optical_frame",
                    "role": "scene_primary",
                    "path": "/tmp/top.png",
                }
            ],
            memory_context={
                "scene_epoch": 4,
                "selected_sam3_detection": {"id": "red-bolt"},
                "placement_object_detection": {
                    "id": "red-bolt-object",
                    "source_image": "/tmp/top.png",
                    "scene_epoch": 4,
                    "perception_bundle_id": "bundle-4",
                },
                "multi_sort_progress": {
                    "active_assignment": {
                        "placement_region_prompt": "blue parts bin",
                        "placement_region_perception_prompt": (
                            "blue square area inside bin"
                        ),
                    }
                },
                "sam3_semantic_state": {
                    "roles": {
                        "placement_region": {
                            "canonical_prompt": "blue square area inside bin",
                            "scene_epoch": 4,
                        }
                    },
                    "attempts": attempts,
                },
            },
            available_tool_names={"observe", "sam3", "active_observe"},
        )
        assert value is not None
        return value

    obligation = obligation_for([failed_attempt])

    assert obligation["status"] == "required"
    assert obligation["required_tool"] == "active_observe"
    assert obligation["required_parameters"] == {
        "semantic_target": "blue square area inside bin",
        "semantic_role": "placement_region",
        "quality_profile": "placement_rgbd",
        "max_motion_attempts": 0,
    }
    assert obligation["fallback"] == "current_view_provider_grounded_region"
    phase, legal = _model_phase_and_legal_tools(
        {
            "semantic_perception_obligation": obligation,
            "active_environment_task": {"env_id": "openeta/test-v0"},
            "tool_references": [
                {"name": "observe"},
                {"name": "sam3"},
                {"name": "active_observe"},
            ],
        },
        max_tools=8,
    )
    assert phase == "target_perception"
    assert legal[0] == "active_observe"

    exhausted = obligation_for(
        [
            failed_attempt,
            {
                "semantic_role": "placement_region",
                "status": "active_search_exhausted",
                "source_image": "/tmp/top.png",
                "mode": "active_search",
                "target_prompt": "blue square area inside bin",
                "scene_epoch": 4,
                "attempt_id": "active-blue-bin",
            },
        ]
    )
    assert exhausted["status"] == "exhausted"
    assert exhausted["failure_code"] == "placement_region_localization_exhausted"


def test_missing_all_point_localizers_exhausts_instead_of_observe_loop() -> None:
    attempts = [
        _failed_grasp_target_attempt(
            source_image=source,
            prompt=prompt,
            attempt_id=attempt_id,
        )
        for source, prompt, attempt_id in (
            ("/tmp/top.png", "red rectangular block", "top-exact"),
            ("/tmp/wrist.png", "red rectangular block", "wrist-exact"),
            ("/tmp/top.png", "red block", "top-simplified"),
        )
    ]

    obligation = _grasp_target_retry_obligation(
        attempts,
        available_tool_names={"observe", "sam3"},
    )

    assert obligation["status"] == "exhausted"
    assert obligation["failure_code"] == (
        "grasp_target_point_localizer_unavailable"
    )
    phase, legal = _model_phase_and_legal_tools(
        {
            "semantic_perception_obligation": obligation,
            "tool_references": [{"name": "observe"}, {"name": "sam3"}],
        },
        max_tools=8,
    )
    assert phase == "semantic_perception_exhausted"
    assert legal == []


def test_failed_point_segmentation_advances_to_host_bound_active_view() -> None:
    attempts = [
        _failed_grasp_target_attempt(
            source_image=source,
            prompt=prompt,
            attempt_id=attempt_id,
        )
        for source, prompt, attempt_id in (
            ("/tmp/top.png", "red rectangular block", "top-exact"),
            ("/tmp/wrist.png", "red rectangular block", "wrist-exact"),
            ("/tmp/top.png", "red block", "top-simplified"),
        )
    ]
    attempts.append(
        {
            "semantic_role": "grasp_target",
            "status": "no_detection",
            "source_image": "/tmp/top.png",
            "mode": "point_prompt",
            "target_prompt": "red rectangular block",
            "scene_epoch": 4,
            "attempt_id": "top-point",
            "attempt_fingerprint": "fingerprint-top-point",
        }
    )
    obligation = _semantic_perception_obligation(
        observation=EnvObservation(
            task="normal pick and place",
            cameras=[],
            robot=RobotState(),
            metadata={"step_idx": 2},
        ),
        camera_artifacts=[
            {"kind": "rgb", "frame_id": "agentview", "path": "/tmp/top.png"},
            {"kind": "rgb", "frame_id": "wrist", "path": "/tmp/wrist.png"},
        ],
        memory_context={
            "scene_epoch": 4,
            "sam3_semantic_state": {
                "roles": {
                    "grasp_target": {
                        "canonical_prompt": "red rectangular block",
                        "scene_epoch": 4,
                        "no_detection": {
                            "source_image": "/tmp/top.png",
                            "positive_points": [{"x": 420.0, "y": 315.0, "label": 1}],
                        },
                    }
                },
                "attempts": attempts,
            },
        },
    )

    assert obligation is not None
    assert obligation["required_tool"] == "active_observe"
    assert obligation["required_parameters"]["semantic_target"] == (
        "red rectangular block"
    )
    assert obligation["required_parameters"]["target_hint"] == {
        "source_image": "/tmp/top.png",
        "positive_points": [{"x": 420.0, "y": 315.0, "label": 1}],
        "source": "bounded_visual_point_localization",
    }


def test_exhausted_active_view_is_not_repeated_on_unchanged_scene() -> None:
    attempts = [
        _failed_grasp_target_attempt(
            source_image=source,
            prompt=prompt,
            attempt_id=attempt_id,
        )
        for source, prompt, attempt_id in (
            ("/tmp/top.png", "red hex bolt", "top-exact"),
            ("/tmp/wrist.png", "red hex bolt", "wrist-exact"),
            ("/tmp/top.png", "red bolt", "top-simplified"),
        )
    ]
    attempts.extend([
        {
            "semantic_role": "grasp_target",
            "status": "no_detection",
            "source_image": "/tmp/top.png",
            "mode": "point_prompt",
            "target_prompt": "red hex bolt",
            "scene_epoch": 4,
            "attempt_id": "point-attempt",
            "attempt_fingerprint": "point-fingerprint",
        },
        {
            "semantic_role": "grasp_target",
            "status": "active_search_exhausted",
            "mode": "active_search",
            "target_prompt": "red hex bolt",
            "scene_epoch": 4,
            "attempt_id": "active-attempt",
            "attempt_fingerprint": "active-fingerprint",
        },
    ])
    obligation = _semantic_perception_obligation(
        observation=EnvObservation(
            task="normal pick and place",
            cameras=[],
            robot=RobotState(),
            metadata={"step_idx": 2},
        ),
        camera_artifacts=[
            {"kind": "rgb", "frame_id": "agentview", "path": "/tmp/top.png"},
            {"kind": "rgb", "frame_id": "wrist", "path": "/tmp/wrist.png"},
        ],
        memory_context={
            "scene_epoch": 4,
            "sam3_semantic_state": {
                "roles": {
                    "grasp_target": {
                        "canonical_prompt": "red hex bolt",
                        "scene_epoch": 4,
                        "no_detection": {
                            "source_image": "/tmp/top.png",
                            "positive_points": [{"x": 420.0, "y": 315.0, "label": 1}],
                        },
                    }
                },
                "attempts": attempts,
            },
        },
    )

    assert obligation is not None
    assert obligation["status"] == "exhausted"
    assert obligation["failure_code"] == "grasp_target_localization_exhausted"
    assert "required_tool" not in obligation


def test_active_observe_detection_enters_shared_sam3_selection_memory() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the red hex bolt")
    detection = {
        "id": "detection_000",
        "label": "red hex bolt",
        "mask_ref": "/tmp/red-bolt-mask.png",
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "active_observe",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "parameters": {
                                    "semantic_target": "red hex bolt",
                                    "semantic_role": "grasp_target",
                                },
                                "outputs": {
                                    "status": "acquired",
                                    "active_vision_mode": "semantic_search",
                                    "active_vision_attempt_id": "active-observe-1",
                                    "active_vision_attempt_fingerprint": "active-fingerprint-1",
                                    "result_id": "active-sam-result",
                                    "semantic_role": "grasp_target",
                                    "semantic_target": "red hex bolt",
                                    "prompt": "red hex bolt",
                                    "scene_epoch": 0,
                                    "source_image": "/tmp/active-wrist.png",
                                    "source_frame_id": "wrist_camera_optical_frame",
                                    "segmentation_mode": "point_prompt",
                                    "attempt_id": "active-sam-attempt",
                                    "attempt_fingerprint": "active-sam-fingerprint",
                                    "detections": [detection],
                                    "selected_detection": detection,
                                },
                            },
                        },
                    }
                ]
            },
        )
    )

    pending = memory.pending_sam3_selection()
    assert pending is not None
    assert pending["result_id"] == "active-sam-result"
    assert pending["candidates"] == [detection]
    attempts = memory.sam3_semantic_state()["attempts"]
    assert any(item["mode"] == "active_search" for item in attempts)


def test_active_observe_rebinds_nested_environment_epoch_to_current_session() -> None:
    memory = AgentMemory()
    memory.start_session(task="sort several tools")
    for _ in range(4):
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "request": {"name": "move_to", "parameters": {}},
                    "tool_calls": [
                        {
                            "name": "move_to",
                            "status": "executed",
                            "result": {"success": True, "details": {}},
                        }
                    ],
                },
            )
        )
    assert memory.scene_epoch() == 4

    detection = {
        "id": "detection_002",
        "label": "point_prompt",
        "mask_ref": "/tmp/active-target-mask.png",
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "active_observe",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "parameters": {
                                    "semantic_target": "silver tool",
                                    "semantic_role": "grasp_target",
                                },
                                "outputs": {
                                    "status": "acquired",
                                    "active_vision_mode": "semantic_search",
                                    "active_vision_attempt_id": "active-observe-current",
                                    "active_vision_attempt_fingerprint": "active-current-fp",
                                    "result_id": "active-sam-current",
                                    "semantic_role": "grasp_target",
                                    "semantic_target": "silver tool",
                                    "prompt": "silver tool",
                                    # The nested environment does not know about
                                    # the agent's four acknowledged mutations.
                                    "scene_epoch": 0,
                                    "source_image": "/tmp/active-top.png",
                                    "source_frame_id": "top_camera_optical_frame",
                                    "segmentation_mode": "point_prompt",
                                    "attempt_id": "active-sam-attempt-current",
                                    "attempt_fingerprint": "active-sam-fp-current",
                                    "detections": [detection],
                                    "selected_detection": detection,
                                    "selection_review": {
                                        "decision": "select",
                                        "detection_id": "detection_002",
                                        "selection_source": (
                                            "active_vision_point_depth_geometry"
                                        ),
                                    },
                                },
                            },
                        },
                    }
                ]
            },
        )
    )

    selected = memory.selected_sam3_detection()
    placement_object = memory.placement_object_detection()
    assert selected is not None
    assert placement_object is not None
    assert selected["scene_epoch"] == 4
    assert placement_object["scene_epoch"] == 4
    assert memory.pending_sam3_selection() is None


def test_active_observe_placement_detection_enters_anyplace_memory() -> None:
    memory = AgentMemory()
    memory.start_session(task="place the bolt in the blue bin")
    detection = {
        "id": "detection_region",
        "label": "point_prompt",
        "mask_ref": "/tmp/blue-bin-mask.png",
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "active_observe",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "parameters": {
                                    "semantic_target": "blue parts bin interior",
                                    "semantic_role": "placement_region",
                                },
                                "outputs": {
                                    "status": "acquired",
                                    "active_vision_mode": "semantic_search",
                                    "active_vision_attempt_id": "active-placement-1",
                                    "active_vision_attempt_fingerprint": (
                                        "active-placement-fingerprint-1"
                                    ),
                                    "result_id": "active-placement-result",
                                    "semantic_role": "placement_region",
                                    "semantic_target": "blue parts bin interior",
                                    "prompt": "blue parts bin interior",
                                    "scene_epoch": 0,
                                    "source_image": "/tmp/top.png",
                                    "source_frame_id": "top_camera_optical_frame",
                                    "segmentation_mode": "point_prompt",
                                    "attempt_id": "active-placement-sam",
                                    "attempt_fingerprint": (
                                        "active-placement-sam-fingerprint"
                                    ),
                                    "perception_bundle_id": "placement-bundle",
                                    "observation_id": "placement-observation",
                                    "detections": [detection],
                                    "selected_detection": detection,
                                    "selection_review": {
                                        "decision": "select",
                                        "detection_id": "detection_region",
                                        "selection_source": "isolated_main_vlm",
                                    },
                                },
                            },
                        },
                    }
                ]
            },
        )
    )

    region = memory.placement_region_detection()
    assert region is not None
    assert region["id"] == "detection_region"
    assert region["perception_bundle_id"] == "placement-bundle"
    assert memory.pending_sam3_selection() is None


def _semantic_view_observation(
    *,
    wrist_position: list[float],
) -> EnvObservation:
    return EnvObservation(
        task="normal pick and place",
        cameras=[
            CameraFrame(
                frame_id="top_camera_optical_frame",
                role="scene_primary",
                rgb=[],
                intrinsics={"width": 1280, "height": 720, "fx": 700.0, "fy": 700.0},
                extrinsics={
                    "frame_transform": "camera_to_world",
                    "camera_frame": "opencv",
                    "pos": [0.38, 0.0, 1.35],
                    "quat_xyzw": [0.7071067812, -0.7071067812, 0.0, 0.0],
                },
            ),
            CameraFrame(
                frame_id="wrist_camera_optical_frame",
                role="wrist_primary",
                rgb=[],
                intrinsics={"width": 1280, "height": 720, "fx": 700.0, "fy": 700.0},
                extrinsics={
                    "frame_transform": "camera_to_world",
                    "camera_frame": "opencv",
                    "pos": wrist_position,
                    "quat_xyzw": [-0.73, -0.54, -0.27, -0.33],
                    "timestamp_s": 123.456,
                },
            ),
        ],
        robot=RobotState(),
        metadata={"step_idx": 2},
    )


def _semantic_view_artifacts(prefix: str) -> list[dict[str, object]]:
    return [
        {
            "kind": "rgb",
            "frame_id": "top_camera_optical_frame",
            "role": "scene_primary",
            "path": f"/tmp/{prefix}-top.png",
            "width": 1280,
            "height": 720,
        },
        {
            "kind": "rgb",
            "frame_id": "wrist_camera_optical_frame",
            "role": "wrist_primary",
            "path": f"/tmp/{prefix}-wrist.png",
            "width": 1280,
            "height": 720,
        },
    ]


def _view_retry_obligation(
    *,
    observation: EnvObservation,
    artifacts: list[dict[str, object]],
    attempts: list[dict[str, object]],
) -> dict[str, object]:
    obligation = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=artifacts,
        memory_context={
            "scene_epoch": 4,
            "sam3_semantic_state": {
                "roles": {
                    "grasp_target": {
                        "canonical_prompt": "yellow adjustable wrench",
                        "scene_epoch": 4,
                    }
                },
                "attempts": attempts,
            },
        },
    )
    assert obligation is not None
    return obligation


def test_new_file_for_same_physical_view_does_not_reset_retry_budget() -> None:
    first_observation = _semantic_view_observation(wrist_position=[0.55, 0.19, 0.26])
    first_artifacts = _semantic_view_artifacts("first")
    first = _view_retry_obligation(
        observation=first_observation,
        artifacts=first_artifacts,
        attempts=[],
    )
    top_view_identity = str(first["required_parameters"]["view_identity"])

    second_observation = _semantic_view_observation(wrist_position=[0.55, 0.19, 0.26])
    second_artifacts = _semantic_view_artifacts("second")
    second = _view_retry_obligation(
        observation=second_observation,
        artifacts=second_artifacts,
        attempts=[
            {
                **_failed_grasp_target_attempt(
                    source_image="/tmp/first-top.png",
                    prompt="yellow adjustable wrench",
                    attempt_id="top-exact",
                ),
                "view_identity": top_view_identity,
            }
        ],
    )

    assert second["required_parameters"]["image"] == "/tmp/second-wrist.png"
    assert second["required_parameters"]["view_identity"] != top_view_identity


def test_intentional_wrist_camera_move_creates_a_fresh_semantic_view() -> None:
    first_observation = _semantic_view_observation(wrist_position=[0.55, 0.19, 0.26])
    first_artifacts = _semantic_view_artifacts("first")
    top_view = _semantic_camera_view_identity(
        observation=first_observation,
        artifact=first_artifacts[0],
    )
    wrist_view = _semantic_camera_view_identity(
        observation=first_observation,
        artifact=first_artifacts[1],
    )
    attempts = [
        {
            **_failed_grasp_target_attempt(
                source_image="/tmp/first-top.png",
                prompt="yellow adjustable wrench",
                attempt_id="top-exact",
            ),
            "view_identity": top_view,
        },
        {
            **_failed_grasp_target_attempt(
                source_image="/tmp/first-wrist.png",
                prompt="yellow adjustable wrench",
                attempt_id="wrist-exact",
            ),
            "view_identity": wrist_view,
        },
    ]

    moved_observation = _semantic_view_observation(wrist_position=[0.47, 0.08, 0.42])
    moved_artifacts = _semantic_view_artifacts("moved")
    obligation = _view_retry_obligation(
        observation=moved_observation,
        artifacts=moved_artifacts,
        attempts=attempts,
    )

    assert obligation["required_parameters"]["image"] == "/tmp/moved-wrist.png"
    assert obligation["required_parameters"]["prompt"] == "yellow adjustable wrench"
    assert obligation["required_parameters"]["view_identity"] != wrist_view


def test_robot_move_creates_fresh_fixed_camera_occlusion_view() -> None:
    fixed_artifact = _semantic_view_artifacts("capture")[0]
    before = _semantic_view_observation(wrist_position=[0.55, 0.19, 0.26])
    before.robot.end_effector_pose = {
        "xyz": [0.54, 0.06, 0.25],
        "quat_xyzw": [0.22, 0.67, 0.65, 0.27],
    }
    after = _semantic_view_observation(wrist_position=[0.47, 0.08, 0.42])
    after.robot.end_effector_pose = {
        "xyz": [0.32, -0.24, 0.48],
        "quat_xyzw": [0.0, 0.7071, 0.0, 0.7071],
    }

    before_identity = _semantic_camera_view_identity(
        observation=before,
        artifact=fixed_artifact,
    )
    after_identity = _semantic_camera_view_identity(
        observation=after,
        artifact=fixed_artifact,
    )

    assert before_identity != after_identity


def test_planner_and_sam3_handler_share_physical_view_attempt_identity() -> None:
    observation = _semantic_view_observation(wrist_position=[0.55, 0.19, 0.26])
    view_identity = _semantic_camera_view_identity(
        observation=observation,
        artifact=_semantic_view_artifacts("capture")[0],
    )
    planned = _sam3_request_identity(
        observation=observation,
        scene_epoch=4,
        source_image="/tmp/capture-top.png",
        semantic_role="grasp_target",
        semantic_target="yellow adjustable wrench",
        mode="text",
        prompt="yellow adjustable wrench",
        points=[],
        roi_bbox_xyxy=None,
        view_identity=view_identity,
    )
    handled = _sam3_semantic_metadata(
        parameters={
            **planned,
            "semantic_role": "grasp_target",
            "semantic_target": "yellow adjustable wrench",
        },
        observation=observation,
        source_image="/tmp/capture-top.png",
        mode="text",
        prompt="yellow adjustable wrench",
        raw_points=[],
    )

    assert handled["view_identity"] == view_identity
    assert handled["attempt_id"] == planned["attempt_id"]
    assert handled["attempt_fingerprint"] == planned["attempt_fingerprint"]


def test_sam3_semantic_memory_persists_view_provenance() -> None:
    memory = AgentMemory()
    memory.start_session(task="normal pick and place")
    memory._record_sam3_semantic_result(  # noqa: SLF001 - state-machine unit test.
        {
            "result_id": "failed-top",
            "semantic_role": "grasp_target",
            "target_prompt": "yellow adjustable wrench",
            "source_image": "/tmp/top.png",
            "frame_id": "top_camera_optical_frame",
            "camera_role": "scene_primary",
            "view_identity": "camera-view-fixed-top",
            "segmentation_mode": "text",
            "scene_epoch": 4,
            "attempt_id": "top-exact",
            "attempt_fingerprint": "fingerprint-top-exact",
        },
        status="no_detection",
    )

    attempt = memory.sam3_semantic_state()["attempts"][0]
    assert attempt["frame_id"] == "top_camera_optical_frame"
    assert attempt["camera_role"] == "scene_primary"
    assert attempt["view_identity"] == "camera-view-fixed-top"


def test_exhausted_semantic_phase_advertises_no_more_tools() -> None:
    phase, tools = _model_phase_and_legal_tools(
        {
            "semantic_perception_obligation": {
                "status": "exhausted",
                "semantic_role": "grasp_target",
                "failure_code": "grasp_target_localization_exhausted",
            },
            "tool_references": [
                {"name": "observe"},
                {"name": "sam3"},
                {"name": "active_observe"},
            ],
        },
        max_tools=8,
    )

    assert phase == "semantic_perception_exhausted"
    assert tools == []


def test_simplified_retry_does_not_replace_canonical_semantic_prompt() -> None:
    memory = AgentMemory()
    memory.start_session(task="normal pick and place")
    memory._record_sam3_semantic_result(  # noqa: SLF001 - state-machine unit test.
        {
            "result_id": "top-exact",
            "semantic_role": "grasp_target",
            "target_prompt": "red rectangular block",
            "source_image": "/tmp/top.png",
            "segmentation_mode": "text",
            "scene_epoch": 4,
            "attempt_id": "top-exact",
            "attempt_fingerprint": "fingerprint-top-exact",
        },
        status="no_detection",
    )
    memory._record_sam3_semantic_result(  # noqa: SLF001 - state-machine unit test.
        {
            "result_id": "top-simplified",
            "semantic_role": "grasp_target",
            "target_prompt": "red block",
            "source_image": "/tmp/top.png",
            "segmentation_mode": "text",
            "scene_epoch": 4,
            "attempt_id": "top-simplified",
            "attempt_fingerprint": "fingerprint-top-simplified",
        },
        status="no_detection",
    )

    role = memory.sam3_semantic_state()["roles"]["grasp_target"]
    assert role["canonical_prompt"] == "red rectangular block"
