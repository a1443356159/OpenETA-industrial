from __future__ import annotations

import hashlib
import json

import pytest

from agent.runtime.moveit_qualification import (
    QUALIFICATION_SCHEMA,
    MoveItCandidateQualifier,
    MoveItQualificationEngine,
    parallel_gripper_approach_reversal_variant,
    parallel_gripper_symmetry_variant,
)
from agent.tools.registry import ToolResult
from tools.candidate_config import CandidateFunnelConfig
from extensions.gazebo.planning_scene import CollisionBox, PlanningSceneSynchronizer
from extensions.gazebo.robot_control import GazeboControlConfig
from extensions.gazebo.ros_control import _RosRuntime


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate(index: int):
    return {"id": f"g{index}", "qualification_stages": [{"name": "hover"}]}


def _request(candidates, *, full_plan_limit=12, exposure_limit=10, seed_count=8):
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "planning_scene_revision": 4,
        "qualification_binding_sha256": "binding",
        "planning": {
            "full_plan_limit": full_plan_limit,
            "exposure_limit": exposure_limit,
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
    assert (config.grasp_full_plan_limit, config.anyplace_full_plan_limit) == (4, 4)
    assert config.moveit_ik_seed_count == 8
    with pytest.raises(ValueError):
        CandidateFunnelConfig(graspgenx_raw_pool_size=63)
    with pytest.raises(ValueError):
        CandidateFunnelConfig(anyplace_diversity_pool_size=9)


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


def test_full_planning_is_bounded_and_stops_at_exposure_limit():
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
        _request([_candidate(i) for i in range(20)], full_plan_limit=4, exposure_limit=10)
    )
    assert len(calls) == 4
    assert sum(item["verdict"] == "PASS" for item in response["results"]) == 4
    assert all(item.get("full_plan_submitted") is False for item in response["results"][4:])


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


def test_qualifier_reports_monotonic_funnel_counts_and_only_exposes_pass():
    candidates = [{"id": f"g{i}"} for i in range(15)]

    def rpc(name, request, timeout):
        return _engine().qualify(request)

    result = MoveItCandidateQualifier(
        rpc,
        grasp_diversity_limit=12,
        grasp_full_plan_limit=12,
        grasp_exposure_limit=10,
        compile_candidate=lambda *args: {"qualification_stages": [{"name": "hover"}]},
    ).qualify_result(
        ToolResult(True, "ok", {"grasp_candidates": candidates, "model_raw_candidate_count": 20}),
        purpose="grasp",
        scene_epoch=1,
        planning_scene_revision=4,
    )
    details = result.details
    assert details["candidate_count"] == details["qualified_candidate_count"] == 10
    assert details["raw_candidate_count"] == details["generated_candidate_count"] == 15
    assert details["diversity_selected_count"] == 12
    assert details["full_plan_submitted_count"] == details["full_plan_pass_count"] == 10
    assert details["coordinate_tcp_pass_count"] >= details["workspace_pass_count"] >= details["pure_ik_pass_count"]


def test_parallel_gripper_variant_preserves_approach_and_records_provenance():
    candidate = {
        "id": "g0",
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
    variant = parallel_gripper_symmetry_variant(candidate)
    assert variant["rotation_matrix"] == [[1.0, -0.0, -0.0], [0.0, -1.0, -0.0], [0.0, -0.0, -1.0]]
    assert variant["symmetry_parent_id"] == "g0"
    assert variant["provenance"] == "host_parallel_gripper_symmetry"


def test_parallel_gripper_approach_reversal_preserves_tip_and_closing_axis():
    candidate = {
        "id": "g0",
        "depth": 0.12,
        "translation_xyz": [0.1, 0.2, 0.3],
        "gripper_tip_position_xyz": [0.22, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "transform_matrix": [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }

    variant = parallel_gripper_approach_reversal_variant(candidate)

    assert variant["rotation_matrix"] == [
        [-1.0, 0.0, -0.0],
        [-0.0, 1.0, -0.0],
        [-0.0, 0.0, -1.0],
    ]
    assert variant["translation_xyz"] == pytest.approx([0.34, 0.2, 0.3])
    reconstructed_tip = [
        variant["translation_xyz"][index]
        + variant["depth"] * variant["rotation_matrix"][index][0]
        for index in range(3)
    ]
    assert reconstructed_tip == pytest.approx(candidate["gripper_tip_position_xyz"])
    assert variant["approach_reversal_parent_id"] == "g0"
    assert variant["provenance"] == "host_parallel_gripper_approach_reversal"


def test_robotiq_symmetry_variants_enter_the_real_grasp_funnel():
    captured = {}

    def rpc(name, request, timeout):
        captured["candidates"] = request["candidates"]
        return _engine().qualify(request)

    result = MoveItCandidateQualifier(
        rpc,
        compile_candidate=lambda candidate, *args: {
            "qualification_stages": [
                {
                    "name": "hover",
                    "xyz": [0.1, 0.0, 0.5],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ]
        },
    ).qualify_result(
        ToolResult(
            True,
            "ok",
            {
                "grasp_candidates": [
                    {
                        "id": "g0",
                        "gripper_name": "robotiq_2f_85",
                        "rotation_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    }
                ],
                "model_raw_candidate_count": 20,
            },
        ),
        purpose="grasp",
        scene_epoch=1,
        planning_scene_revision=4,
    )

    ids = [item["candidate_id"] for item in captured["candidates"]]
    assert ids == ["g0", "g0_sym180"]
    assert result.details["raw_candidate_count"] == 2


def test_second_zero_pass_round_has_no_source_return_recovery():
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
        placement_exposure_limit=10,
        placement_max_rounds=2,
        compile_candidate=lambda *args: {
            "qualification_stages": [{"name": "hover"}]
        },
    )
    source = {
        "placement_observation": {"observation_id": "frozen"},
        "attachment_transform": {"translation": [0.0, 0.0, 0.1]},
    }
    for index in range(2):
        result = qualifier.qualify_result(
            ToolResult(
                True,
                "ok",
                {
                    "placement_candidates": [
                        {
                            "id": f"p{index}",
                            "object_goal_world": {
                                "transform_matrix": [
                                    [1.0, 0.0, 0.0, float(index)],
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
    assert result.details["qualification_round"] == 2
    assert "source_return_qualification" not in result.details
    assert result.details["qualified_candidate_count"] == 0
    assert requested_candidate_ids == [["p0"], ["p1"]]


def test_default_anyplace_pool_screens_all_96_but_only_plans_top_4():
    captured = {}

    def rpc(name, request, timeout):
        captured["candidate_ids"] = [
            item["candidate_id"] for item in request["candidates"]
        ]
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
    assert result.details["diversity_selected_count"] == 96
    assert result.details["workspace_pass_count"] == 96
    assert result.details["pure_ik_pass_count"] == 96
    assert result.details["collision_ik_pass_count"] == 96
    assert result.details["endpoint_pass_count"] == 96
    assert result.details["full_plan_submitted_count"] == 4
    assert result.details["candidate_count"] == 4


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
