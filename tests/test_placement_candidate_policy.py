from __future__ import annotations

from adapter.protocol import EnvAction
from agent.runtime.memory import AgentMemory


def _policy(candidate_ids):
    return {
        "schema_version": "openeta.placement_candidate_policy.v1",
        "status": "active",
        "candidate_queue": list(candidate_ids),
        "source_grasp_id": "grasp_000",
        "active_candidate_id": candidate_ids[0],
        "rejected_candidates": [],
        "failed_request_fingerprints": [],
        "scene_revision": 7,
        "compiled_placement": {"placement_candidate_id": candidate_ids[0]},
    }


def _planning_failure(
    candidate_id: str,
    fingerprint: str,
    *,
    execution_started=False,
    error_code="MOTION_PLAN_FAILED",
    motion_outcome=None,
) -> EnvAction:
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
                            "xyz": [0.4, 0.0, 0.5],
                        }
                    },
                    "status": "failed",
                    "result": {
                        "success": False,
                        "details": {
                            "environment_receipt": {
                                "error_code": error_code,
                                "moveit_error_code": 99999,
                                "planned_point_count": 0,
                                "execution_started": execution_started,
                                "motion_outcome": motion_outcome,
                                "scene_revision": 7,
                                "request_fingerprint": fingerprint,
                            }
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


def test_all_placement_candidates_failed_requires_verified_source_return() -> None:
    memory = AgentMemory()
    memory.save_fact("placement_candidate_policy", _policy(["placement_000"]), source="test")
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_000",
            "compiled_grasp": {
                "hover_pose": {"frame": "world", "xyz": [0.2, 0.0, 0.6]},
                "contact_pose": {"frame": "world", "xyz": [0.2, 0.0, 0.45]},
            },
        },
        source="test",
    )

    memory.add_action(_planning_failure("placement_000", "fingerprint-only"))

    policy = memory.placement_candidate_policy()
    assert policy["status"] == "exhausted_return_required"
    assert policy["recovery"]["stage"] == "return_source_hover"
    assert policy["recovery"]["source_capture_pose"]["xyz"] == [0.2, 0.0, 0.45]


def test_failed_first_candidate_allows_only_the_second_candidate() -> None:
    memory = AgentMemory()
    memory.save_fact("placement_candidate_policy", _policy(["placement_000", "placement_001"]), source="test")

    memory.add_action(_planning_failure("placement_000", "fingerprint-a"))

    policy = memory.placement_candidate_policy()
    assert policy["status"] == "selection_required"
    rejected = {item["candidate_id"] for item in policy["rejected_candidates"]}
    assert rejected == {"placement_000"}
    assert "placement_001" not in rejected


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


def test_exhausted_recovery_detaches_then_invalidates_stale_perception_on_observe() -> None:
    memory = AgentMemory()
    memory.save_fact("placement_candidate_policy", _policy(["placement_000"]), source="test")
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_000",
            "compiled_grasp": {
                "hover_pose": {"frame": "world", "xyz": [0.2, 0.0, 0.6]},
                "contact_pose": {"frame": "world", "xyz": [0.2, 0.0, 0.45]},
            },
        },
        source="test",
    )
    memory.save_fact("selected_sam3_detection", {"id": "stale"}, source="test")
    memory.save_fact("attachment_gate", {"status": "resolved", "verdict": "PASS"}, source="test")
    memory.add_action(_planning_failure("placement_000", "fingerprint-only"))

    for stage, xyz in (
        ("return_source_hover", [0.2, 0.0, 0.6]),
        ("return_source_capture", [0.2, 0.0, 0.45]),
    ):
        memory.add_action(
            _successful_call(
                "move_to",
                {"target_pose": {"placement_recovery_stage": stage, "xyz": xyz}},
            )
        )
    memory.add_action(
        _successful_call(
            "gripper_control",
            {"position": 1},
            {
                "detachable_joint": {"state": "detached"},
                "planning_scene_revision": 9,
            },
        )
    )
    assert memory.placement_candidate_policy()["status"] == "reobserve_regrasp_required"

    memory.add_action(_successful_call("observe", {"reason": "fresh"}))

    policy = memory.placement_candidate_policy()
    assert policy["status"] == "regrasp_required"
    assert policy["recovery"]["stage"] == "regrasp"
    assert memory.selected_sam3_detection() is None
    assert memory.grasp_execution() is None
    assert memory.attachment_gate() is None


def test_failed_source_return_stops_instead_of_repeating_motion() -> None:
    memory = AgentMemory()
    memory.save_fact("placement_candidate_policy", _policy(["placement_000"]), source="test")
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "compiled_grasp": {
                "hover_pose": {"frame": "world", "xyz": [0.2, 0.0, 0.6]},
                "contact_pose": {"frame": "world", "xyz": [0.2, 0.0, 0.45]},
            },
        },
        source="test",
    )
    memory.add_action(_planning_failure("placement_000", "fingerprint-only"))

    memory.add_action(
        _planning_failure(
            "placement_000",
            "source-return-unknown",
            execution_started=None,
            error_code="MOTION_OUTCOME_UNKNOWN",
            motion_outcome="unknown",
        )
    )

    policy = memory.placement_candidate_policy()
    assert policy["status"] == "stopped_requires_human"
    assert policy["stop_reason"] == "source_return_motion_failed_or_unknown"
