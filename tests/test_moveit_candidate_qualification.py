from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.runtime.moveit_qualification import (
    GOAL_PREBIND_RPC_PER_CANDIDATE_S,
    KINEMATIC_IK_TIMEOUT_S,
    PLANNING_ATTEMPTS,
    PLANNING_TIME_S,
    QUALIFICATION_RPC_FIRST_ACK_TIMEOUT_S,
    QUALIFICATION_RPC_GRACE_S,
    QUALIFICATION_SCHEMA,
    STATE_VALIDITY_TIMEOUT_S,
    MoveItCandidateQualifier,
    MoveItQualificationEngine,
    QualificationCache,
    _goal_prebind_rpc_timeout_s,
    _qualification_rpc_timeout_s,
    private_qualification_rpc,
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


def test_private_qualification_rpc_uses_ack_watchdog_and_bound_identity():
    class Transport:
        def __init__(self):
            self.calls = []
            self.health_calls = []

        def call_tool(self, name, arguments, *, timeout_s=None):
            self.calls.append((name, dict(arguments), timeout_s))
            if len(self.calls) == 1:
                from agent.tools.sim_mcp import SimulatorMcpTransportError

                raise SimulatorMcpTransportError(
                    f"call_tool:{name}", TimeoutError("lost reply")
                )
            return {"execution_started": False, "results": []}

        def list_tools(self, *, timeout_s=None):
            self.health_calls.append(timeout_s)
            return {
                "tools": [{"name": "qualify_motion_candidates"}],
                "tool_count": 1,
            }

    transport = Transport()
    rpc = private_qualification_rpc(
        transport,
        handle_provider=lambda: "env-1",
        session_id_provider=lambda: "session-1",
    )

    result = rpc("qualify_motion_candidates", {"purpose": "grasp"}, 900.0)

    assert [call[2] for call in transport.calls] == [
        QUALIFICATION_RPC_FIRST_ACK_TIMEOUT_S,
        900.0,
    ]
    assert all(
        call[1] == {
            "purpose": "grasp",
            "handle": "env-1",
            "session_id": "session-1",
        }
        for call in transport.calls
    )
    assert transport.health_calls == [5.0]
    assert result["_openeta_transport_retry"]["retry_count"] == 1


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


def test_case_hash_is_stable_across_solver_profiles_while_binding_changes():
    requests = []

    def rpc(name, request, timeout):
        requests.append(request)
        return _engine().qualify(request)

    compiler = lambda *args: {  # noqa: E731 - compact immutable test fixture.
        "qualification_stages": [
            {
                "name": "goal",
                "xyz": [0.4, 0.0, 0.5],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        ]
    }
    for profile, solver in (("legacy", "kdl_legacy"), ("fast_v3", "kdl_fast")):
        MoveItCandidateQualifier(
            rpc,
            compile_candidate=compiler,
            qualification_profile=profile,
            solver_profile=solver,
        ).qualify_result(
            ToolResult(True, "ok", {"placement_candidates": [{"id": "p0"}]}),
            purpose="placement",
            scene_epoch=1,
            planning_scene_revision=4,
            source={"provider": "anyplace", "provider_version": "frozen"},
        )

    assert requests[0]["qualification_case_sha256"] == requests[1][
        "qualification_case_sha256"
    ]
    assert requests[0]["qualification_binding_sha256"] != requests[1][
        "qualification_binding_sha256"
    ]


def test_qualifier_prepares_candidate_compiler_once_per_batch():
    class BatchCompiler:
        def __init__(self):
            self.prepare_calls = 0
            self.compile_calls = 0

        def __call__(self, *args):
            raise AssertionError("batch-aware compiler must be prepared first")

        def prepare_batch(self, *, purpose):
            assert purpose == "grasp"
            self.prepare_calls += 1

            def compile_candidate(candidate, *_args):
                self.compile_calls += 1
                return {"qualification_stages": [{"name": candidate["id"]}]}

            return compile_candidate

    compiler = BatchCompiler()
    result = MoveItCandidateQualifier(
        lambda _name, request, _timeout: _engine().qualify(request),
        compile_candidate=compiler,
    ).qualify_result(
        ToolResult(
            True,
            "ok",
            {"grasp_candidates": [{"id": f"g{index}"} for index in range(3)]},
        ),
        purpose="grasp",
        scene_epoch=1,
        planning_scene_revision=4,
    )

    assert compiler.prepare_calls == 1
    assert compiler.compile_calls == 3
    assert result.details["workspace_pass_count"] == 3
    assert result.details["endpoint_evaluated_count"] == 2
    assert result.details["endpoint_not_evaluated_count"] == 1


def test_qualification_rpc_deadline_covers_screening_pool_and_plan_tail():
    captured = {}

    def rpc(name, request, timeout):
        del name
        captured["timeout"] = timeout
        return _engine().qualify(request)

    candidates = [
        {
            "id": f"p{i}",
            "object_goal_pose": {"translation_xyz": [i / 1000.0, 0.0, 0.0]},
        }
        for i in range(96)
    ]
    MoveItCandidateQualifier(
        rpc,
        placement_diversity_limit=96,
        placement_full_plan_limit=4,
        compile_candidate=lambda *args: {
            "qualification_stages": [
                {"name": "hover"},
                {"name": "release"},
                {"name": "retreat"},
            ]
        },
    ).qualify_result(
        ToolResult(True, "ok", {"placement_candidates": candidates}),
        purpose="placement",
        scene_epoch=1,
        planning_scene_revision=4,
    )

    assert captured["timeout"] > PLANNING_TIME_S * 4 + 10.0
    expected_screening = 96 * 2.0 * (
        KINEMATIC_IK_TIMEOUT_S + STATE_VALIDITY_TIMEOUT_S
    )
    expected_planning = 4 * 3 * PLANNING_TIME_S
    assert captured["timeout"] == pytest.approx(
        expected_screening + expected_planning + QUALIFICATION_RPC_GRACE_S
    )


def test_fast_rpc_deadline_covers_exhaustive_l5_and_recovery_without_60s_cutoff():
    candidates = [
        {"candidate": {"qualification_stages": [{}, {}, {}]}}
        for _ in range(192)
    ]

    timeout = _qualification_rpc_timeout_s(
        candidates,
        full_plan_limit=2,
        qualification_profile="fast_v3",
    )

    assert timeout > 192 * 2 * 3 * PLANNING_TIME_S


def test_goal_prebind_rpc_deadline_scales_with_complete_frozen_frontier():
    one_goal = [{"candidate": {"id": "p0"}}]
    full_frontier = [
        {"candidate": {"id": f"p{index}"}}
        for index in range(96)
    ]

    assert _goal_prebind_rpc_timeout_s(one_goal) == pytest.approx(
        QUALIFICATION_RPC_GRACE_S + GOAL_PREBIND_RPC_PER_CANDIDATE_S
    )
    assert _goal_prebind_rpc_timeout_s(full_frontier) == pytest.approx(
        QUALIFICATION_RPC_GRACE_S
        + 96 * GOAL_PREBIND_RPC_PER_CANDIDATE_S
    )


def test_qualification_rpc_error_reason_is_preserved():
    def rpc(name, request, timeout):
        del name, request, timeout
        raise TimeoutError("outer qualification deadline")

    result = MoveItCandidateQualifier(
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

    evidence = result.details["qualification_evidence"]
    assert [item["verdict"] for item in evidence["results"]] == ["UNKNOWN"]
    assert [item["reason"] for item in evidence["results"]] == [
        "qualification_rpc_error"
    ]
    assert result.details["rejection_reason_counts"] == {
        "qualification_rpc_error": 1
    }


def test_fast_qualification_retries_malformed_response_before_accepting_evidence():
    calls = 0

    def rpc(name, request, timeout):
        nonlocal calls
        del name, timeout
        calls += 1
        if calls == 1:
            return {}
        return _engine().qualify(request)

    result = MoveItCandidateQualifier(
        rpc,
        qualification_profile="fast_v3",
        compile_candidate=lambda *args: {
            "qualification_stages": [
                {
                    "name": "contact",
                    "xyz": [0.4, 0.0, 0.5],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ]
        },
    ).qualify_result(
        ToolResult(True, "ok", {"grasp_candidates": [{"id": "g0"}]}),
        purpose="grasp",
        scene_epoch=1,
        planning_scene_revision=4,
    )

    evidence = result.details["qualification_evidence"]
    assert calls == 2
    assert evidence["rpc_attempt_count"] == 2
    assert evidence.get("infrastructure_error") is not True
    assert [item["verdict"] for item in evidence["results"]] == ["PASS"]


def test_repeated_malformed_qualification_response_is_infrastructure_error():
    calls = 0

    def rpc(name, request, timeout):
        nonlocal calls
        del name, request, timeout
        calls += 1
        return {"results": []}

    result = MoveItCandidateQualifier(
        rpc,
        qualification_profile="fast_v3",
        compile_candidate=lambda *args: {
            "qualification_stages": [{"name": "contact"}]
        },
    ).qualify_result(
        ToolResult(True, "ok", {"grasp_candidates": [{"id": "g0"}]}),
        purpose="grasp",
        scene_epoch=1,
        planning_scene_revision=4,
    )

    evidence = result.details["qualification_evidence"]
    assert calls == 2
    assert evidence["rpc_attempt_count"] == 2
    assert evidence["stop_reason"] == "infrastructure_error"
    assert evidence["infrastructure_error"] is True
    assert evidence["results"][0]["verdict"] == "UNKNOWN"
    assert evidence["results"][0]["infrastructure_error"] is True


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


def test_collision_ik_tries_later_seed_after_first_solution_is_in_collision():
    collision_starts = []

    def compute_ik(target, start, collision):
        del target
        if not collision:
            return {
                "ok": True,
                "joint_state": {"names": ["j1"], "positions": [1.0]},
            }
        collision_starts.append(list(start["positions"]))
        solution = [1.0] if start["positions"] == [1.0] else [2.0]
        return {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": solution},
        }

    def state_validity(state):
        valid = state["positions"] == [2.0]
        return {
            "valid": valid,
            "collision_pairs": [] if valid else [["table", "wrist"]],
        }

    candidate = {"id": "g0", "qualification_stages": [{"name": "hover"}]}
    result = _engine(
        compute_ik=compute_ik,
        check_state_validity=state_validity,
    ).qualify(_request(candidate))["results"][0]

    assert result["verdict"] == "PASS"
    stage = result["stages"][0]
    assert stage["pure_state_valid"] is False
    assert stage["state_valid"] is True
    assert collision_starts[:2] == [[1.0], [0.0]]
    assert [attempt["state_valid"] for attempt in stage["collision_ik_attempts"]] == [
        False,
        True,
    ]


def test_qualifier_exposes_only_pass_and_cache_rejects_failed_id(tmp_path):
    cache = QualificationCache()

    def rpc(name, request, timeout):
        engine = _engine()
        response = engine.qualify(request)
        for item in response["results"]:
            item.pop("compile_parameters", None)
        response["results"][1].update({"verdict": "FAIL", "reason": "plan_only_failed"})
        response["search_exhaustion"] = {
            "fast_pool_exhausted": True,
            "recovery_pool_exhausted": True,
            "redundancy_degraded": True,
        }
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
        ToolResult(
            True,
            "ok",
            {
                "grasp_candidates": candidates,
                "candidate_count": 2,
                "raw_candidate_count": 100,
            },
        ),
        purpose="grasp",
        scene_epoch=3,
        planning_scene_revision=4,
        source={"calibration_sha256": "c"},
    )

    assert [candidate["id"] for candidate in result.details["grasp_candidates"]] == ["g0"]
    assert result.details["candidate_count"] == 1
    assert result.details["raw_candidate_count"] == 100
    assert result.details["generated_candidate_count"] == 2
    assert result.details["submitted_candidate_count"] == 2
    assert result.details["full_plan_pass_count"] == 1
    assert "qualified_candidate_count" not in result.details
    assert cache.resolve(purpose="grasp", candidate_id="g0", scene_epoch=3, planning_scene_revision=4)
    assert cache.resolve(purpose="grasp", candidate_id="g1", scene_epoch=3) is None
    assert result.details["qualification_artifact"]["path"].endswith(".json")
    assert "results" not in result.details["qualification_evidence"]
    stored = json.loads(
        Path(result.details["qualification_artifact"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert len(stored["results"]) == 2
    assert all(
        item["compile_parameters"]
        == {"camera_extrinsics": {"pose_mat": [1] * 16}}
        for item in stored["results"]
    )
    assert stored["search_exhaustion"] == {
        "fast_pool_exhausted": True,
        "recovery_pool_exhausted": True,
        "redundancy_degraded": True,
    }
    assert result.details["qualification_evidence"]["search_exhaustion"] == stored[
        "search_exhaustion"
    ]


def test_scene_revision_drift_is_unknown():
    revisions = iter([4, 5])
    candidate = {"id": "g0", "qualification_stages": [{"name": "hover"}]}
    result = _engine(scene_revision=lambda: next(revisions)).qualify(_request(candidate))["results"][0]
    assert (result["verdict"], result["reason"]) == (
        "UNKNOWN",
        "planning_scene_revision_drift",
    )
