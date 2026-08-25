from __future__ import annotations

import json

import pytest

from adapter.protocol import EnvAction, EnvObservation, RobotState
from agent.backends.planner import (
    CallablePlannerBackend,
    PlannerBackendRequest,
    PlannerBackendResult,
)
from agent.runtime.memory import AgentMemory
from agent.runtime.planner import (
    PlannerContextConfig,
    _default_tool_planner_system_prompt,
    _model_request_context,
    _semantic_perception_obligation,
)
from agent.runtime.sam3_selection import BackendSam3SelectionReviewer


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
        "semantic_target": "green placement region",
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


def test_placement_object_fallback_tries_each_view_once_then_points() -> None:
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
    point_fallback = _semantic_perception_obligation(
        observation=observation,
        camera_artifacts=camera_artifacts,
        memory_context=two_failure_context,
    )

    assert point_fallback["required_tool"] == "molmopoint"
    assert point_fallback["required_parameters"]["images"] == ["/tmp/top.png"]
    assert point_fallback["attempt"] == 1


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
