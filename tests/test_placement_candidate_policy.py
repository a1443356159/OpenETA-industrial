from __future__ import annotations

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
                            "environment_receipt": {
                                "error_code": error_code,
                                "moveit_error_code": 99999,
                                "planned_point_count": 0,
                                "execution_started": execution_started,
                                "motion_outcome": motion_outcome,
                                "planning_scene_revision": 7,
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


def test_all_placement_candidates_failed_requires_independent_reobservation() -> None:
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
    assert policy["status"] == "reobserve_required"
    assert policy["recovery"] == {
        "stage": "observe_placement",
        "then": "resegment_placement_region_and_rerun_anyplace",
        "preserve_attachment": True,
    }


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
            "schema_version": "openeta.compiled_placement_seed.v2",
            "placement_candidate_id": candidate_id,
            "scene_epoch": 0,
            "scene_revision": 7,
            "selection_source": "host_qualified_queue",
            "hover_pose": {
                "frame": "world",
                "purpose": "placement",
                "placement_candidate_id": candidate_id,
                "placement_stage": "hover",
                "xyz": [0.4, 0.0, 0.6],
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




def test_exhausted_recovery_reobserves_placement_and_preserves_attachment() -> None:
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

    assert memory.placement_candidate_policy()["status"] == "reobserve_required"

    memory.add_action(_successful_call("observe", {"reason": "fresh"}))

    assert memory.placement_candidate_policy() is None
    assert memory.selected_sam3_detection() is None
    assert memory.grasp_execution()["stage"] == "attached"
    assert memory.attachment_gate() == {"status": "resolved", "verdict": "PASS"}


def test_reobserve_recovery_does_not_require_source_return_or_detach() -> None:
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

    policy = memory.placement_candidate_policy()
    assert policy["status"] == "reobserve_required"
    assert policy["recovery"]["preserve_attachment"] is True
    assert memory.grasp_execution()["stage"] == "attached"
