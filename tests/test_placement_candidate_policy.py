from __future__ import annotations

import pytest

from adapter.protocol import EnvAction
from agent.runtime.memory import AgentMemory


def _policy(candidate_ids):
    return {
        "schema_version": "openeta.placement_candidate_policy.v2",
        "status": "active",
        "candidate_queue": list(candidate_ids),
        "active_candidate_id": candidate_ids[0],
        "rejected_candidates": [],
        "failed_request_fingerprints": [],
        "scene_revision": 7,
        "planning_scene_revision": 7,
        "revision_provenance": "native_attachment_gate",
        "compiled_placement": {"placement_candidate_id": candidate_ids[0]},
    }


def _planning_failure(
    candidate_id: str,
    fingerprint: str,
    *,
    execution_started=False,
    error_code="MOTION_PLAN_FAILED",
    motion_outcome=None,
    receipt_overrides=None,
) -> EnvAction:
    receipt = {
        "error_code": error_code,
        "moveit_error_code": 99999,
        "planned_point_count": 0,
        "execution_started": execution_started,
        "motion_outcome": motion_outcome,
        "planning_scene_revision": 7,
        "request_fingerprint": fingerprint,
    }
    receipt.update(receipt_overrides or {})
    return EnvAction(
        action_type="tool_call",
        command={
            "request": {
                "name": "move_to",
                "parameters": {
                    "target_pose": {
                        "purpose": "placement",
                        "placement_candidate_id": candidate_id,
                        "compiled_eef_pose": True,
                        "scene_revision": 7,
                        "xyz": [0.4, 0.0, 0.5],
                    }
                },
            },
            "tool_calls": [
                {
                    "name": "move_to",
                    "parameters": {
                        "target_pose": {
                            "purpose": "placement",
                            "placement_candidate_id": candidate_id,
                            "compiled_eef_pose": True,
                            "scene_revision": 7,
                            "xyz": [0.4, 0.0, 0.5],
                        }
                    },
                    "status": "failed",
                    "result": {
                        "success": False,
                        "details": {
                            "environment_receipt": receipt
                        },
                    },
                }
            ],
        },
    )


def _successful_call(name: str, parameters: dict, receipt: dict | None = None) -> EnvAction:
    return EnvAction(
        action_type="tool_call",
        command={
            "request": {"name": name, "parameters": parameters},
            "tool_calls": [
                {
                    "name": name,
                    "parameters": parameters,
                    "status": "executed",
                    "result": {
                        "success": True,
                        "details": {"environment_receipt": receipt or {}},
                    },
                }
            ],
        },
    )


def test_planning_failure_rejects_only_active_placement_candidate() -> None:
    memory = AgentMemory()
    memory.save_fact("placement_candidate_policy", _policy(["placement_000", "placement_001"]), source="test")

    memory.add_action(_planning_failure("placement_000", "fingerprint-a"))

    policy = memory.placement_candidate_policy()
    assert policy["status"] == "selection_required"
    assert policy["active_candidate_id"] is None
    assert policy["failed_request_fingerprints"] == ["fingerprint-a"]
    assert policy["rejected_candidates"][0]["moveit_error_code"] == 99999
    assert "current joint state" in policy["rejected_candidates"][0]["reason"]


def test_all_placement_candidates_failed_stops_without_new_inference() -> None:
    memory = AgentMemory()
    memory.save_fact("placement_candidate_policy", _policy(["placement_000"]), source="test")
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_000",
            "compiled_grasp": {
                "contact_pose": {"frame": "world", "xyz": [0.2, 0.0, 0.45]},
            },
        },
        source="test",
    )

    memory.add_action(_planning_failure("placement_000", "fingerprint-only"))

    policy = memory.placement_candidate_policy()
    assert policy["status"] == "stopped_requires_human"
    assert policy["stop_reason"] == "CURRENT_GRASP_PLACE_INFEASIBLE"
    assert policy["recovery"] == {
        "stage": "manual_intervention",
        "required_action": None,
    }


def test_pristine_attached_place_zero_pass_reopens_and_resumes_frozen_grasp_frontier() -> None:
    """A measured attachment miss resumes frozen model grasps, not inference."""

    memory = AgentMemory()
    memory.save_fact(
        "frozen_placement_goal_pool",
        {
            "schema_version": "openeta.frozen_placement_goal_pool.v1",
            "status": "ready",
            "goal_count": 96,
        },
        source="test",
    )
    memory.save_fact(
        "grasp_candidate_policy",
        {
            "status": "accepted",
            "active_candidate": {"id": "grasp_000"},
            "accepted_candidate": {"id": "grasp_000"},
            "frozen_grasp_frontier_remaining_count": 504,
            "frozen_grasp_frontier_generation": 1,
            "planning_scene_revision": 5,
            "rejected_candidates": [],
        },
        source="test",
    )
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_000",
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {
            "status": "resolved",
            "verdict": "PASS",
            "candidate_id": "grasp_000",
            "planning_scene_revision": 6,
        },
        source="test",
    )
    zero_pass = EnvAction(
        action_type="tool_call",
        command={
            "request": {
                "name": "anyplace",
                "parameters": {"reuse_frozen_goal_pool": True, "scene_revision": 6},
            },
            "tool_calls": [
                {
                    "name": "anyplace",
                    "result": {
                        "success": True,
                        "details": {
                            "outputs": {
                                "placement_candidates": [],
                                "qualification_evidence": {
                                    "results": [{"candidate_id": "p0"}] * 96
                                },
                                "frozen_goal_requalification": True,
                                "scene_revision": 6,
                            }
                        },
                    },
                }
            ],
        },
    )

    memory.add_action(zero_pass)

    placement = memory.placement_candidate_policy()
    recovery = memory.grasp_recovery()
    policy = memory.grasp_candidate_policy()
    assert placement["status"] == "frozen_grasp_frontier_recovery_required"
    assert recovery["status"] == "required"
    assert recovery["required_action"] == {
        "name": "gripper_control",
        "parameters": {"position": 1, "recovery_intent": "frozen_grasp_frontier"},
    }
    assert policy["status"] == "frozen_frontier_required"
    assert policy["frozen_grasp_frontier_rebase_pending"]["physically_rejected_candidate_id"] == (
        "grasp_000"
    )

    rebound_sync = {
        "schema_version": "openeta.planning_scene_target_pose_sync.v1",
        "operation": "update_world_target",
        "source_revision": 5,
        "revision": 8,
        "topology_unchanged": True,
        "static_world_unchanged": True,
        "attached_ids_before": [],
        "attached_ids_after": [],
    }
    memory.add_action(
        _successful_call(
            "gripper_control",
            {"position": 1, "recovery_intent": "frozen_grasp_frontier"},
            {
                "planning_scene_revision": 8,
                "detachable_joint": {"state": "detached"},
                "planning_scene_target_pose_sync": rebound_sync,
                "frozen_grasp_frontier_recovery": {
                    "schema_version": "openeta.frozen_grasp_frontier_recovery.v1",
                    "status": "ready",
                    "model_inference_invoked": False,
                },
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "frozen_frontier_required"
    assert policy["planning_scene_revision"] == 8
    assert policy["frozen_grasp_frontier_recovery"]["model_inference_invoked"] is False
    assert memory.grasp_recovery()["status"] == "completed"
    assert memory.grasp_execution() is None
    assert memory.attachment_gate() is None
    assert memory.placement_candidate_policy() is None
    assert memory.frozen_placement_goal_pool()["goal_count"] == 96


def test_failed_first_candidate_allows_only_the_second_candidate() -> None:
    memory = AgentMemory()
    memory.save_fact("placement_candidate_policy", _policy(["placement_000", "placement_001"]), source="test")

    memory.add_action(_planning_failure("placement_000", "fingerprint-a"))

    policy = memory.placement_candidate_policy()
    assert policy["status"] == "selection_required"
    rejected = {item["candidate_id"] for item in policy["rejected_candidates"]}
    assert rejected == {"placement_000"}
    assert "placement_001" not in rejected


def test_failed_first_candidate_activates_precompiled_host_queue_fallback() -> None:
    memory = AgentMemory()
    policy = _policy(["placement_000", "placement_001"])
    policy["scene_epoch"] = 0
    policy["host_candidate_compilations"] = {
        candidate_id: {
            "schema_version": "openeta.compiled_placement_seed.v3",
            "placement_candidate_id": candidate_id,
            "scene_epoch": 0,
            "scene_revision": 7,
            "selection_source": "host_qualified_queue",
            "release_pose": {
                "frame": "world",
                "purpose": "placement",
                "placement_candidate_id": candidate_id,
                "placement_stage": "release",
                "xyz": [0.4, 0.0, 0.5],
            },
        }
        for candidate_id in ("placement_000", "placement_001")
    }
    memory.save_fact("placement_candidate_policy", policy, source="test")

    memory.add_action(_planning_failure("placement_000", "fingerprint-a"))

    active = memory.placement_candidate_policy()
    assert active["status"] == "active"
    assert active["active_candidate_id"] == "placement_001"
    assert active["selection_source"] == "host_qualified_queue"
    assert active["compiled_placement"]["placement_candidate_id"] == "placement_001"


def test_started_or_unknown_placement_motion_stops_without_switching_candidate() -> None:
    for action in (
        _planning_failure("placement_000", "started", execution_started=True),
        _planning_failure(
            "placement_000",
            "unknown",
            execution_started=None,
            error_code="MOTION_OUTCOME_UNKNOWN",
            motion_outcome="unknown",
        ),
    ):
        memory = AgentMemory()
        memory.save_fact("placement_candidate_policy", _policy(["placement_000", "placement_001"]), source="test")
        memory.add_action(action)
        policy = memory.placement_candidate_policy()
        assert policy["status"] == "stopped_requires_human"
        assert policy["active_candidate_id"] == "placement_000"
        assert policy["rejected_candidates"] == []


@pytest.mark.parametrize(
    "terminal_error_code",
    ["MOTION_TARGET_NOT_REACHED", "MOTION_EXECUTION_FAILED"],
)
def test_known_terminal_miss_with_retained_attachment_resumes_frozen_frontier(
    terminal_error_code: str,
) -> None:
    memory = AgentMemory()
    policy = _policy(["placement_006"])
    policy.update(
        {
            "frozen_goal_requalification": True,
            "frozen_goal_frontier_count": 26,
            "frozen_goal_total_eligible_count": 26,
            "frozen_goal_frontier_generation": 0,
        }
    )
    memory.save_fact("placement_candidate_policy", policy, source="test")
    retained_receipt = {
        "physical_verification": {
            "schema_version": "openeta.gazebo.native_grasp.v1",
            "verdict": "PASS",
            "reason_code": "NATIVE_GRASP_TARGET_HELD",
            "grasp_confirmed": True,
            "target_id": "target_object",
        },
        "detachable_joint": {"state": "attached"},
        "attachment_transform": {
            "schema_version": "openeta.attachment_transform.v1",
            "parent_frame": "eef",
            "child_frame": "object",
            "measurement_boundary": "native_attach_ack",
            "translation_xyz": [0.0, 0.01, 0.16],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "child_link_proof": {
            "prior_attachment_confirmed": True,
            "capture_relative_translation_m": 0.0001,
            "maximum_capture_relative_translation_m": 0.01,
        },
        "observation_fresh": True,
        "motion": {
            "reached_target": False,
            "end": {
                "frame": "gripper_mount_link",
                "xyz": [0.4, 0.0, 0.5],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "observation_snapshot": {
            "schema_version": "openeta.observation_snapshot.v1",
            "observation": {
                "metadata": {"planning_scene_revision": 7},
                "robot": {
                    "end_effector_pose": {
                        "frame": "gripper_mount_link",
                        "xyz": [0.4, 0.0, 0.5],
                        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "joint_positions": [0.1] * 7,
                },
            },
        },
        "position_error_m": 0.008,
        "orientation_error_rad": 0.05,
        "planned_point_count": 47,
    }

    memory.add_action(
        _planning_failure(
            "placement_006",
            "terminal-miss-006",
            execution_started=True,
            error_code=terminal_error_code,
            motion_outcome="failed",
            receipt_overrides=retained_receipt,
        )
    )

    resumed = memory.placement_candidate_policy()
    assert resumed["status"] == "frozen_frontier_required"
    assert resumed["active_candidate_id"] is None
    assert resumed["compiled_placement"] is None
    assert resumed["failed_request_fingerprints"] == ["terminal-miss-006"]
    assert resumed["rejected_candidates"] == [
        {
            "candidate_id": "placement_006",
            "request_fingerprint": "terminal-miss-006",
            "error_code": terminal_error_code,
            "execution_started": True,
            "motion_outcome": "failed",
            "planned_point_count": 47,
            "scene_revision": 7,
            "position_error_m": 0.008,
            "orientation_error_rad": 0.05,
            "attachment_retained": True,
            "reason": (
                "known terminal miss with retained native attachment; requalify "
                "the next frozen goal from the observed end state"
            ),
        }
    ]


def test_repeated_failure_fingerprint_stops_fail_closed() -> None:
    memory = AgentMemory()
    policy = _policy(["placement_000"])
    policy["failed_request_fingerprints"] = ["duplicate"]
    memory.save_fact("placement_candidate_policy", policy, source="test")

    memory.add_action(_planning_failure("placement_000", "duplicate"))

    policy = memory.placement_candidate_policy()
    assert policy["status"] == "stopped_requires_human"
    assert policy["stop_reason"] == "repeated_failed_request_fingerprint"
    assert policy["rejected_candidates"] == []




def test_observe_cannot_reset_an_exhausted_frozen_pool() -> None:
    memory = AgentMemory()
    memory.save_fact("placement_candidate_policy", _policy(["placement_000"]), source="test")
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_000",
            "compiled_grasp": {
                "contact_pose": {"frame": "world", "xyz": [0.2, 0.0, 0.45]},
            },
        },
        source="test",
    )
    memory.save_fact("selected_sam3_detection", {"id": "stale"}, source="test")
    memory.save_fact("attachment_gate", {"status": "resolved", "verdict": "PASS"}, source="test")
    memory.add_action(_planning_failure("placement_000", "fingerprint-only"))

    assert memory.placement_candidate_policy()["status"] == "stopped_requires_human"

    memory.add_action(_successful_call("observe", {"reason": "fresh"}))

    assert memory.placement_candidate_policy()["status"] == "stopped_requires_human"
    assert memory.selected_sam3_detection() == {"id": "stale"}
    assert memory.grasp_execution()["stage"] == "attached"
    assert memory.attachment_gate() == {"status": "resolved", "verdict": "PASS"}


def test_exhausted_frozen_pool_has_no_automatic_recovery_action() -> None:
    memory = AgentMemory()
    memory.save_fact("placement_candidate_policy", _policy(["placement_000"]), source="test")
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "compiled_grasp": {
                "contact_pose": {"frame": "world", "xyz": [0.2, 0.0, 0.45]},
            },
        },
        source="test",
    )
    memory.add_action(_planning_failure("placement_000", "fingerprint-only"))

    policy = memory.placement_candidate_policy()
    assert policy["status"] == "stopped_requires_human"
    assert policy["recovery"]["required_action"] is None
    assert memory.grasp_execution()["stage"] == "attached"
