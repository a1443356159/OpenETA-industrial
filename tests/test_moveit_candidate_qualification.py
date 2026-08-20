from __future__ import annotations

import hashlib
import json

import pytest

from agent.runtime.moveit_qualification import (
    KINEMATIC_IK_TIMEOUT_S,
    PLANNING_ATTEMPTS,
    PLANNING_TIME_S,
    QUALIFICATION_SCHEMA,
    STATE_VALIDITY_TIMEOUT_S,
    MoveItCandidateQualifier,
    MoveItQualificationEngine,
    QualificationCache,
)
from agent.tools.registry import ToolResult


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _request(candidate, *, revision=4):
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "planning_scene_revision": revision,
        "qualification_binding_sha256": "binding",
        "candidates": [
            {
                "candidate_id": candidate["id"],
                "candidate_pose_sha256": _hash(candidate),
                "candidate": candidate,
            }
        ],
    }


def _engine(**overrides):
    callbacks = {
        "current_joint_state": lambda: {"names": ["j1"], "positions": [0.0]},
        "scene_revision": lambda: 4,
        "compute_ik": lambda target, start, collision: {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": [1.0]},
        },
        "check_state_validity": lambda state: {"valid": True, "collision_pairs": []},
        "plan_only": lambda target, start, timeout, attempts: {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": [0.5]}, {"positions": [1.0]}],
            "end_joint_state": {"names": ["j1"], "positions": [1.0]},
        },
    }
    callbacks.update(overrides)
    return MoveItQualificationEngine(**callbacks)


def test_engine_chains_segment_start_state_and_has_zero_execution_side_effects():
    starts = []

    def plan(target, start, timeout, attempts):
        starts.append(dict(start))
        assert timeout == PLANNING_TIME_S
        assert attempts == PLANNING_ATTEMPTS
        end = {"names": ["j1"], "positions": [float(len(starts))]}
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": end["positions"]}],
            "end_joint_state": end,
        }

    candidate = {
        "id": "g0",
        "qualification_stages": [{"name": "hover"}, {"name": "contact"}],
    }
    result = _engine(plan_only=plan).qualify(_request(candidate))["results"][0]

    assert result["verdict"] == "PASS"
    assert starts == [
        {"names": ["j1"], "positions": [0.0]},
        {"names": ["j1"], "positions": [1.0]},
    ]
    assert all(stage["execution_started"] is False for stage in result["stages"])


def test_qualification_request_binds_short_service_timeouts():
    captured = {}

    def rpc(name, request, timeout):
        captured.update(request["planning"])
        return _engine().qualify(request)

    MoveItCandidateQualifier(
        rpc,
        compile_candidate=lambda *args: {
            "qualification_stages": [{"name": "hover"}]
        },
    ).qualify_result(
        ToolResult(True, "ok", {"grasp_candidates": [{"id": "g0"}]}),
        purpose="grasp",
        scene_epoch=1,
        planning_scene_revision=4,
    )

    assert captured == {
        "allowed_planning_time_s": PLANNING_TIME_S,
        "num_planning_attempts": PLANNING_ATTEMPTS,
        "plan_only": True,
        "kinematic_ik_timeout_s": KINEMATIC_IK_TIMEOUT_S,
        "state_validity_timeout_s": STATE_VALIDITY_TIMEOUT_S,
    }


@pytest.mark.parametrize(
    ("override", "verdict", "reason"),
    [
        ({"compute_ik": lambda target, start, collision: {"ok": False}}, "FAIL", "kinematic_ik_failed"),
        ({"check_state_validity": lambda state: {"valid": False, "collision_pairs": [["table", "wrist"]]}}, "FAIL", "collision_state_invalid"),
        ({"plan_only": lambda *args: (_ for _ in ()).throw(TimeoutError())}, "UNKNOWN", "plan_only_timeout"),
        ({"plan_only": lambda *args: {"ok": True, "execution_started": False, "trajectory_points": [], "end_joint_state": {}}}, "UNKNOWN", "plan_only_empty_trajectory"),
    ],
)
def test_engine_fail_closed_outcomes(override, verdict, reason):
    candidate = {"id": "g0", "qualification_stages": [{"name": "hover"}]}
    result = _engine(**override).qualify(_request(candidate))["results"][0]
    assert (result["verdict"], result["reason"]) == (verdict, reason)
    if reason == "collision_state_invalid":
        assert result["stages"][0]["collision_pairs"] == [["table", "wrist"]]


def test_qualifier_exposes_only_pass_and_cache_rejects_failed_id(tmp_path):
    cache = QualificationCache()

    def rpc(name, request, timeout):
        engine = _engine()
        response = engine.qualify(request)
        response["results"][1].update({"verdict": "FAIL", "reason": "plan_only_failed"})
        return response

    def compile_candidate(candidate, purpose, source, epoch, revision):
        return {
            "qualification_stages": [{"name": "hover"}],
            "compile_parameters": {"camera_extrinsics": {"pose_mat": [1] * 16}},
        }

    candidates = [{"id": "g0"}, {"id": "g1"}]
    result = MoveItCandidateQualifier(
        rpc,
        cache=cache,
        artifact_root=tmp_path,
        compile_candidate=compile_candidate,
    ).qualify_result(
        ToolResult(True, "ok", {"grasp_candidates": candidates, "candidate_count": 2}),
        purpose="grasp",
        scene_epoch=3,
        planning_scene_revision=4,
        source={"calibration_sha256": "c"},
    )

    assert [candidate["id"] for candidate in result.details["grasp_candidates"]] == ["g0"]
    assert result.details["candidate_count"] == 1
    assert result.details["generated_candidate_count"] == 2
    assert cache.resolve(purpose="grasp", candidate_id="g0", scene_epoch=3, planning_scene_revision=4)
    assert cache.resolve(purpose="grasp", candidate_id="g1", scene_epoch=3) is None
    assert result.details["qualification_artifact"]["path"].endswith(".json")


def test_scene_revision_drift_is_unknown():
    revisions = iter([4, 5])
    candidate = {"id": "g0", "qualification_stages": [{"name": "hover"}]}
    result = _engine(scene_revision=lambda: next(revisions)).qualify(_request(candidate))["results"][0]
    assert (result["verdict"], result["reason"]) == (
        "UNKNOWN",
        "planning_scene_revision_drift",
    )
