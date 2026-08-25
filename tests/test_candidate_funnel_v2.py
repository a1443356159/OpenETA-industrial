from __future__ import annotations

import hashlib
import json

import pytest

from agent.runtime.moveit_qualification import (
    PROGRESSIVE_NOT_EVALUATED_REASON,
    PROGRESSIVE_SCREENING_MODE,
    QUALIFICATION_SCHEMA,
    MoveItCandidateQualifier,
    MoveItQualificationEngine,
)
from agent.tools.registry import ToolResult
from tools.candidate_config import CandidateFunnelConfig
from extensions.gazebo.planning_scene import CollisionBox, PlanningSceneSynchronizer
from extensions.gazebo.robot_control import GazeboControlConfig
from extensions.gazebo.ros_control import RosGazeboController, _RosRuntime


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate(index: int):
    return {"id": f"g{index}", "qualification_stages": [{"name": "hover"}]}


def _request(candidates, *, full_plan_limit=12, seed_count=8):
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "planning_scene_revision": 4,
        "qualification_binding_sha256": "binding",
        "planning": {
            "full_plan_limit": full_plan_limit,
            "ik_seed_count": seed_count,
        },
        "source": {"joint_limits": {"lower": [-1.0], "upper": [1.0]}},
        "candidates": [
            {
                "candidate_id": item["id"],
                "candidate_pose_sha256": _hash(item),
                "candidate": item,
            }
            for item in candidates
        ],
    }


def _engine(**overrides):
    values = {
        "current_joint_state": lambda: {"names": ["j1"], "positions": [0.0]},
        "scene_revision": lambda: 4,
        "compute_ik": lambda target, seed, collision: {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": [0.5]},
        },
        "check_state_validity": lambda state: {"valid": True, "collision_pairs": []},
        "plan_only": lambda target, start, timeout, attempts: {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": [0.5]}],
            "end_joint_state": {"names": ["j1"], "positions": [0.5]},
        },
    }
    values.update(overrides)
    return MoveItQualificationEngine(**values)


def test_v2_defaults_and_cross_field_constraints():
    config = CandidateFunnelConfig()
    assert (
        config.graspgenx_raw_pool_size,
        config.anygrasp_raw_pool_size,
        config.anyplace_raw_pool_size,
    ) == (200, 200, 96)
    assert (config.grasp_diversity_pool_size, config.anyplace_diversity_pool_size) == (64, 96)
    assert (config.grasp_full_plan_limit, config.anyplace_full_plan_limit) == (2, 2)
    assert config.moveit_ik_seed_count == 8
    with pytest.raises(ValueError):
        CandidateFunnelConfig(graspgenx_raw_pool_size=63)
    with pytest.raises(ValueError):
        CandidateFunnelConfig(anyplace_diversity_pool_size=97)


def test_multi_seed_ik_is_deterministic_and_can_recover_after_first_seed():
    seen = []

    def ik(target, seed, collision):
        seen.append((collision, list(seed["positions"])))
        ok = seed["positions"] != [0.0]
        return {
            "ok": ok,
            **({"joint_state": {"names": ["j1"], "positions": [0.5]}} if ok else {}),
        }

    first = _engine(compute_ik=ik).qualify(_request([_candidate(0)]))["results"][0]
    first_seeds = first["stages"][0]["ik_seeds"]
    seen.clear()
    second = _engine(compute_ik=ik).qualify(_request([_candidate(0)]))["results"][0]
    assert first["verdict"] == second["verdict"] == "PASS"
    assert first_seeds == second["stages"][0]["ik_seeds"]
    assert len(first_seeds) == 8


def test_multi_seed_ik_splits_the_total_budget_across_remaining_seeds():
    budgets = []

    def ik(target, seed, collision):
        budgets.append(float(target["ik_seed_timeout_s"]))
        ok = seed["positions"] != [0.0]
        return {
            "ok": ok,
            **({"joint_state": {"names": ["j1"], "positions": [0.5]}} if ok else {}),
        }

    result = _engine(compute_ik=ik).qualify(_request([_candidate(0)]))["results"][0]

    assert result["verdict"] == "PASS"
    assert len(budgets) >= 3  # two pure-IK attempts, then collision IK
    assert budgets[0] == pytest.approx(0.25, abs=0.001)
    assert all(0.0 < value <= 2.0 for value in budgets)


def test_full_planning_is_bounded_and_retains_every_pass():
    calls = []

    def plan(target, start, timeout, attempts):
        calls.append(target)
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{}],
            "end_joint_state": {"names": ["j1"], "positions": [0.5]},
        }

    response = _engine(plan_only=plan).qualify(
        _request([_candidate(i) for i in range(20)], full_plan_limit=4)
    )
    assert len(calls) == 4
    assert sum(item["verdict"] == "PASS" for item in response["results"]) == 4
    assert all(
        item.get("full_plan_submitted") is False
        and item["verdict"] == "NOT_EVALUATED"
        and item["reason"] == "full_plan_limit_not_submitted"
        for item in response["results"][4:]
    )


def test_progressive_screening_prechecks_all_but_stops_endpoint_work_at_capacity():
    workspace_calls = []
    ik_calls = []
    plan_calls = []
    candidates = [
        {
            "id": f"g{index}",
            "qualification_stages": [{"name": f"hover_{index}"}],
        }
        for index in range(10)
    ]
    request = _request(candidates, full_plan_limit=4)
    request["funnel"] = {
        "ik_seed_count": 8,
        "full_plan_limit": 4,
        "screening_mode": PROGRESSIVE_SCREENING_MODE,
        "endpoint_pass_target": 4,
    }

    def workspace(target):
        workspace_calls.append(target["name"])
        return True

    def ik(target, seed, collision):
        ik_calls.append((target["name"], collision))
        return {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": [0.5]},
        }

    def plan(target, start, timeout, attempts):
        plan_calls.append(target["name"])
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{}],
            "end_joint_state": {"names": ["j1"], "positions": [0.5]},
        }

    results = _engine(
        workspace_filter=workspace,
        compute_ik=ik,
        plan_only=plan,
    ).qualify(request)["results"]

    assert workspace_calls == [f"hover_{index}" for index in range(10)]
    assert {name for name, _collision in ik_calls} == {
        f"hover_{index}" for index in range(4)
    }
    assert plan_calls == [f"hover_{index}" for index in range(4)]
    assert all(item["verdict"] == "PASS" for item in results[:4])
    assert all(
        item["verdict"] == "NOT_EVALUATED"
        and item["reason"] == PROGRESSIVE_NOT_EVALUATED_REASON
        and item["workspace_pass"] is True
        and item["endpoint_evaluated"] is False
        and item["execution_started"] is False
        for item in results[4:]
    )


def test_progressive_screening_exhausts_batch_when_capacity_is_not_reached():
    seen = []
    candidates = [
        {
            "id": f"g{index}",
            "qualification_stages": [{"name": f"hover_{index}"}],
        }
        for index in range(6)
    ]
    request = _request(candidates, full_plan_limit=4)
    request["funnel"] = {
        "ik_seed_count": 8,
        "full_plan_limit": 4,
        "screening_mode": PROGRESSIVE_SCREENING_MODE,
        "endpoint_pass_target": 4,
    }

    def ik(target, seed, collision):
        seen.append(target["name"])
        ok = target["name"] == "hover_5"
        return {
            "ok": ok,
            **(
                {"joint_state": {"names": ["j1"], "positions": [0.5]}}
                if ok
                else {}
            ),
        }

    results = _engine(compute_ik=ik).qualify(request)["results"]

    assert set(seen) == {f"hover_{index}" for index in range(6)}
    assert sum(item.get("endpoint_evaluated") is True for item in results) == 6
    assert sum(item["verdict"] == "PASS" for item in results) == 1
    assert not any(item["verdict"] == "NOT_EVALUATED" for item in results)


def test_virtual_scene_transition_uses_clone_without_real_revision_change():
    real_revision = {"value": 4}
    transitions = []

    def transition(scene, name, target):
        scene["transitions"].append(name)
        transitions.append(name)
        return {"ok": True, "transition": name, "virtual": True}

    candidate = {
        "id": "g0",
        "qualification_stages": [
            {"name": "contact", "scene_transition": "virtual_attach"},
            {"name": "lift"},
        ],
    }
    result = _engine(
        scene_revision=lambda: real_revision["value"],
        clone_scene=lambda: {"revision": 4, "transitions": []},
        apply_scene_transition=transition,
    ).qualify(_request([candidate]))["results"][0]
    assert result["verdict"] == "PASS"
    assert transitions == ["virtual_attach"]
    assert real_revision["value"] == 4


def test_qualifier_reports_monotonic_funnel_counts_and_retains_every_pass():
    candidates = [{"id": f"g{i}"} for i in range(15)]

    def rpc(name, request, timeout):
        return _engine().qualify(request)

    result = MoveItCandidateQualifier(
        rpc,
        grasp_diversity_limit=12,
        grasp_full_plan_limit=12,
        compile_candidate=lambda *args: {"qualification_stages": [{"name": "hover"}]},
    ).qualify_result(
        ToolResult(True, "ok", {"grasp_candidates": candidates, "model_raw_candidate_count": 20}),
        purpose="grasp",
        scene_epoch=1,
        planning_scene_revision=4,
    )
    details = result.details
    assert details["candidate_count"] == 12
    assert "qualified_candidate_count" not in details
    assert details["raw_candidate_count"] == details["generated_candidate_count"] == 15
    assert details["diversity_selected_count"] == 12
    assert details["full_plan_submitted_count"] == details["full_plan_pass_count"] == 12
    assert details["coordinate_tcp_pass_count"] >= details["workspace_pass_count"] >= details["pure_ik_pass_count"]


def test_provider_candidates_enter_funnel_without_host_pose_variants():
    captured = {}
    candidate = {
        "id": "g0",
        "gripper_name": "robotiq_2f_85",
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }

    def rpc(_name, request, _timeout):
        captured["candidates"] = request["candidates"]
        return _engine().qualify(request)

    result = MoveItCandidateQualifier(
        rpc,
        compile_candidate=lambda item, *_args: {
            "qualification_stages": [
                {
                    "name": "contact",
                    "xyz": list(item["translation_xyz"]),
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ]
        },
    ).qualify_result(
        ToolResult(
            True,
            "ok",
            {"grasp_candidates": [candidate], "model_raw_candidate_count": 1},
        ),
        purpose="grasp",
        scene_epoch=1,
        planning_scene_revision=4,
    )

    assert [item["candidate_id"] for item in captured["candidates"]] == ["g0"]
    assert captured["candidates"][0]["candidate"]["translation_xyz"] == [0.1, 0.2, 0.3]
    assert result.details["raw_candidate_count"] == 1
    assert result.details["candidate_count"] == 1

def test_zero_pass_pool_returns_no_inference_recovery_contract():
    requested_candidate_ids = []

    def rpc(name, request, timeout):
        requested_candidate_ids.append(
            [item["candidate_id"] for item in request["candidates"]]
        )
        response = _engine().qualify(request)
        if request["purpose"] == "placement":
            for item in response["results"]:
                item.update({"verdict": "FAIL", "reason": "plan_only_failed"})
        return response

    qualifier = MoveItCandidateQualifier(
        rpc,
        placement_diversity_limit=12,
        placement_full_plan_limit=12,
        compile_candidate=lambda *args: {
            "qualification_stages": [{"name": "release"}]
        },
    )
    source = {
        "placement_observation": {"observation_id": "frozen"},
        "attachment_transform": {"translation": [0.0, 0.0, 0.1]},
    }
    result = qualifier.qualify_result(
        ToolResult(
            True,
            "ok",
            {
                "placement_candidates": [
                    {
                        "id": "p0",
                        "object_goal_world": {
                            "transform_matrix": [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        },
                    }
                ],
                "model_raw_candidate_count": 1,
            },
        ),
        purpose="placement",
        scene_epoch=1,
        planning_scene_revision=4,
        source=source,
    )
    assert "qualification_round" not in result.details
    assert "max_qualification_rounds" not in result.details
    assert "source_return_qualification" not in result.details
    assert result.details["candidate_count"] == 0
    assert result.details["full_plan_pass_count"] == 0
    assert requested_candidate_ids == [["p0"]]


def test_default_anyplace_pool_structurally_screens_all_96_then_stops_endpoint_at_2():
    captured = {}

    def rpc(name, request, timeout):
        captured["candidate_ids"] = [
            item["candidate_id"] for item in request["candidates"]
        ]
        captured["funnel"] = dict(request["funnel"])
        return _engine().qualify(request)

    candidates = [
        {
            "id": f"placement_{index:03d}",
            "object_goal_pose": {
                "translation_xyz": [float(index) / 1000.0, 0.0, 0.5],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        }
        for index in range(96)
    ]
    result = MoveItCandidateQualifier(
        rpc,
        compile_candidate=lambda candidate, *args: {
            "qualification_stages": [
                {
                    "name": "hover",
                    "xyz": list(candidate["object_goal_pose"]["translation_xyz"]),
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ]
        },
    ).qualify_result(
        ToolResult(
            True,
            "ok",
            {
                "placement_candidates": candidates,
                "model_raw_candidate_count": 96,
            },
        ),
        purpose="placement",
        scene_epoch=1,
        planning_scene_revision=4,
        source={
            "placement_observation": {"observation_id": "frozen"},
            "attachment_transform": {"translation": [0.0, 0.0, 0.1]},
        },
    )

    assert len(captured["candidate_ids"]) == 96
    assert captured["funnel"] == {
        "ik_seed_count": 8,
        "full_plan_limit": 2,
        "screening_mode": PROGRESSIVE_SCREENING_MODE,
        "endpoint_pass_target": 2,
    }
    assert result.details["diversity_selected_count"] == 96
    assert result.details["workspace_pass_count"] == 96
    assert result.details["pure_ik_pass_count"] == 2
    assert result.details["collision_ik_pass_count"] == 2
    assert result.details["endpoint_pass_count"] == 2
    assert result.details["endpoint_evaluated_count"] == 2
    assert result.details["endpoint_not_evaluated_count"] == 94
    assert result.details["full_plan_submitted_count"] == 2
    assert result.details["candidate_count"] == 2
    assert result.details["rejection_reason_counts"] == {
        PROGRESSIVE_NOT_EVALUATED_REASON: 94
    }


def test_frozen_pair_progressive_counts_distinguish_produced_and_evaluated():
    candidates = [
        {
            "id": f"pair_{index:03d}",
            "qualification_stages": [{"name": "release"}],
        }
        for index in range(12)
    ]

    def rpc(name, request, timeout):
        assert request["funnel"] == {
            "ik_seed_count": 8,
            "full_plan_limit": 2,
            "screening_mode": PROGRESSIVE_SCREENING_MODE,
            "endpoint_pass_target": 2,
        }
        return _engine().qualify(request)

    result = MoveItCandidateQualifier(rpc).qualify_result(
        ToolResult(
            True,
            "ok",
            {
                "placement_candidates": candidates,
                "model_raw_candidate_count": 12,
            },
        ),
        purpose="placement",
        scene_epoch=1,
        planning_scene_revision=4,
        qualification_mode="frozen_pair",
    )

    assert result.details["coordinate_tcp_pass_count"] == 12
    assert result.details["workspace_pass_count"] == 12
    assert result.details["endpoint_evaluated_count"] == 2
    assert result.details["endpoint_not_evaluated_count"] == 10
    assert result.details["endpoint_pass_count"] == 2
    assert result.details["full_plan_submitted_count"] == 2
    assert result.details["full_plan_pass_count"] == 2
    assert result.details["candidate_count"] == 2
    assert result.details["rejection_reason_counts"] == {
        PROGRESSIVE_NOT_EVALUATED_REASON: 10
    }


def test_ros_virtual_scene_diff_is_clone_only_and_payload_aware():
    scene = PlanningSceneSynchronizer()
    scene.reset(
        table=CollisionBox("table", (1.0, 1.0, 0.1), (0.0, 0.0, 0.0)),
        distractor=CollisionBox("other", (0.1, 0.1, 0.1), (0.5, 0.0, 0.2)),
        target=CollisionBox("target", (0.05, 0.05, 0.05), (0.2, 0.0, 0.3)),
    )
    runtime = _RosRuntime(config=GazeboControlConfig(), planning_scene=scene)
    revision = scene.revision
    clone = runtime.qualification_clone_scene()
    attached = runtime.qualification_scene_transition(
        clone,
        "virtual_attach",
        {"xyz": [0.2, 0.0, 0.3], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
    )
    assert attached["ok"] is True
    assert attached["planning_scene_diff"]["attached_objects"][0]["id"] == "target"
    detached = runtime.qualification_scene_transition(
        clone,
        "virtual_detach",
        {"xyz": [0.4, 0.0, 0.3], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
    )
    assert detached["planning_scene_diff"]["remove_attached_ids"] == ["target"]
    assert detached["planning_scene_diff"]["world_objects"][0]["pose_xyz"][0] == pytest.approx(0.4)
    assert scene.revision == revision
    assert scene.attached_ids == set()


def test_ros_controller_syncs_measured_detached_target_pose_and_validates_state():
    scene = PlanningSceneSynchronizer()
    scene.reset(
        table=CollisionBox("table", (1.0, 1.0, 0.1), (0.0, 0.0, 0.0)),
        distractor=CollisionBox("other", (0.1, 0.1, 0.1), (0.5, 0.0, 0.2)),
        target=CollisionBox("target", (0.05, 0.05, 0.05), (0.2, 0.0, 0.3)),
    )
    runtime = type("Runtime", (), {})()
    runtime.planning_scene = scene
    runtime.scene_revision = scene.revision
    runtime.planning_scene_ready = True
    runtime.current_state_validity = lambda **_kwargs: {
        "valid": True,
        "collision_pairs": [],
    }
    controller = object.__new__(RosGazeboController)
    controller.runtime = runtime
    config = type(
        "Config",
        (),
        {"target_id": "target", "target_size_m": (0.05, 0.05, 0.05)},
    )()

    revision = controller.sync_planning_scene_target_pose(
        config,
        target_xyz=(0.23, -0.01, 0.3),
        target_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    assert revision == 2
    assert runtime.scene_revision == 2
    assert runtime.planning_scene_ready is True
    assert scene.world_specs["target"]["pose_xyz"] == [0.23, -0.01, 0.3]
