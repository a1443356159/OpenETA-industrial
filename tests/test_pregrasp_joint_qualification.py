from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from adapter.protocol import EnvObservation, RobotState
from agent.runtime.moveit_qualification import (
    QUALIFICATION_SCHEMA,
    MoveItCandidateQualifier,
    QualificationCache,
    SAME_RUN_QUALIFICATION_SEED_FIELD,
)
from agent.runtime.runtime_assembly import (
    _FrozenGoalPairCoordinator,
    _candidate_qualification_compiler,
    _compile_qualified_queue,
    _prepare_postattachment_frozen_goals,
    _qualifying_handler,
    _restore_frozen_model_motion_for_predicted_pair,
)
from agent.tools.handlers import build_anyplace_handler
from agent.tools.grasp_geometry import compile_placement_seed
from agent.tools.registry import ToolExecutionContext, ToolResult, build_default_tool_registry


def _pass_stage() -> dict[str, Any]:
    return {
        "kinematic_ik": True,
        "state_valid": True,
        "collision_ik": True,
        "plan_only": True,
        "execution_started": False,
        "start_joint_state_sha256": "start",
        "end_joint_state": {"joint_names": ["j1"], "positions": [0.0]},
        "beam_solutions": [
            {
                "joint_state": {"names": ["j1"], "positions": [0.25]},
                "state_valid": True,
            }
        ],
        "trajectory": {"point_count": 2},
    }


def _is_goal_prebind(request: dict[str, Any]) -> bool:
    funnel = request.get("funnel")
    return (
        isinstance(funnel, dict)
        and funnel.get("qualification_mode") == "goal_prebind"
    )


def _goal_prebind_response(
    request: dict[str, Any],
    *,
    verdict: str = "PASS",
    reason: str = "goal_legality_qualified",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected: list[str] = []
    for item in request["candidates"]:
        candidate = dict(item["candidate"])
        candidate["goal_legality_prebound"] = True
        row = {
            "candidate_id": item["candidate_id"],
            "candidate_pose_sha256": item["candidate_pose_sha256"],
            "qualification_binding_sha256": request[
                "qualification_binding_sha256"
            ],
            "execution_started": False,
            "verdict": verdict,
            "reason": reason,
        }
        if verdict == "PASS":
            row["prebound_candidate"] = candidate
            selected.append(item["candidate_id"])
        rows.append(row)
    return {
        "schema_version": request["schema_version"],
        "planning_scene_revision": request["planning_scene_revision"],
        "execution_started": False,
        "qualification_profile": "fast_v3",
        "stop_reason": "goal_legality_barrier_complete",
        "selected_candidate_ids": selected,
        "results": rows,
    }


def test_frozen_physical_goal_retires_active_model_motion_transform() -> None:
    goal = {
        "id": "p0",
        "object_goal_pose": {
            "translation_xyz": [0.45, 0.0, 0.43],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "object_motion_world_transform": {
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.17],
                [0.0, 1.0, 0.0, -0.04],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        },
    }
    physical_goal = {
        "translation_xyz": [0.48, -0.1, 0.43],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    frozen = _FrozenGoalPairCoordinator._bind_physical_collision_goal(
        goal,
        goal_id="p0",
        collision_goal=physical_goal,
    )

    assert frozen["object_goal_pose"] == physical_goal
    assert frozen["world_object_goal_pose"] == physical_goal
    assert "object_motion_world_transform" not in frozen
    assert frozen["model_object_motion_world_transform"] == (
        goal["object_motion_world_transform"]
    )
    assert frozen["frozen_goal_frame_binding"]["physical_collision_goal"] is True


def test_new_grasp_pair_replays_cached_model_motion_only_for_predicted_attachment() -> None:
    model_motion = {
        "frame": "world",
        "convention": "T_world_motion_applied_left",
        "transform_matrix": [
            [0.980066578, -0.198669331, 0.0, 0.17],
            [0.198669331, 0.980066578, 0.0, -0.04],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    pair = {
        "id": "frozen_pair_g1_p0",
        "model_object_motion_world_transform": model_motion,
        "frozen_goal_frame_binding": {"physical_collision_goal": True},
        "frozen_contact_pose": {
            "frame": "world",
            "xyz": [0.3, 0.0, 0.45],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "predicted_attachment_transform": {
            "parent_frame": "eef",
            "child_frame": "object",
            "translation_xyz": [0.0, 0.0, -0.1],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }

    _restore_frozen_model_motion_for_predicted_pair(pair)

    assert pair["object_motion_world_transform"] == model_motion
    assert pair["object_motion_world_transform"] is not model_motion
    assert pair["physical_scene_attachment_required"] is True
    assert pair["physical_scene_attachment_source"] == (
        "cached_collision_goal_with_replayed_model_motion"
    )

    rebased = {
        **pair,
        "frozen_object_motion_rebase": {
            "schema_version": "openeta.frozen_object_motion_rebase.v1"
        },
    }
    _restore_frozen_model_motion_for_predicted_pair(rebased)
    assert "object_motion_world_transform" not in rebased
    assert "model_object_motion_world_transform" not in rebased


def test_host_compiles_every_full_plan_pass_into_one_equal_status_queue() -> None:
    calls: list[dict[str, Any]] = []

    def compiler(context: ToolExecutionContext) -> ToolResult:
        calls.append(
            {
                "parameters": dict(context.parameters),
                "binding": dict(
                    context.metadata["_openeta_host_candidate_compilation_binding"]
                ),
            }
        )
        candidate_id = context.parameters["placement_candidate_id"]
        return ToolResult(
            True,
            "compiled",
            {
                "outputs": {
                    "schema_version": "openeta.compiled_placement_seed.v3",
                    "placement_candidate_id": candidate_id,
                    "scene_epoch": 3,
                    "scene_revision": 7,
                    "selection_source": "host_qualified_queue",
                }
            },
        )

    context = ToolExecutionContext(
        name="anyplace",
        spec=build_default_tool_registry().get("anyplace"),
        parameters={},
        observation=EnvObservation(
            task="pick and place", cameras=[], robot=RobotState()
        ),
        metadata={},
    )
    result = _compile_qualified_queue(
        ToolResult(
            True,
            "qualified",
            {
                "placement_candidates": [{"id": "p0"}, {"id": "p1"}],
                "candidate_count": 2,
                "full_plan_pass_count": 2,
            },
        ),
        purpose="placement",
        context=context,
        scene_epoch=3,
        planning_scene_revision=7,
        compiler=compiler,
    )

    assert result.success
    assert result.details["selection_required"] is False
    assert result.details["host_selected_candidate_id"] == "p0"
    assert [
        event["candidate_id"]
        for event in result.details["host_candidate_compilation_queue"]
    ] == ["p0", "p1"]
    assert result.details["host_candidate_compilation"] == result.details[
        "host_candidate_compilation_queue"
    ][0]
    assert all(
        event["selection_policy"] == "stable_qualified_queue_head"
        and event["execution_started"] is False
        for event in result.details["host_candidate_compilation_queue"]
    )
    assert [call["parameters"]["placement_candidate_id"] for call in calls] == [
        "p0",
        "p1",
    ]
    assert all(
        call["binding"]["planning_scene_revision"] == 7 for call in calls
    )


def test_predicted_attachment_recompiles_frozen_goal_in_model_object_frame() -> None:
    profile_path = (
        Path(__file__).parents[1]
        / "agent/calibrations/candidate/graspnet-eef-rm75-robotiq2f85.json"
    )
    compiler = _candidate_qualification_compiler(
        SimpleNamespace(grasp_profile_path=profile_path)
    ).prepare_batch(purpose="placement")
    model_goal = {
        "frame": "world",
        "translation_xyz": [0.48, -0.1, 0.443],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    physical_goal = {
        "frame": "world",
        "translation_xyz": [0.475, -0.101, 0.431],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    attachment = {
        "parent_frame": "eef",
        "child_frame": "object",
        "translation_xyz": [0.0, 0.0, 0.1],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    candidate = {
        "id": "frozen-pair",
        "object_goal_pose": physical_goal,
        "world_object_goal_pose": physical_goal,
        "qualified_world_collision_object_goal_pose": physical_goal,
        "model_pointcloud_object_goal_pose": model_goal,
        "predicted_attachment_transform": attachment,
        "qualification_start_joint_state": {"names": ["j1"], "positions": [0.0]},
    }

    predicted = compiler(candidate, "placement", {}, 2, 7)
    measured_candidate = dict(candidate)
    measured_candidate.pop("predicted_attachment_transform")
    measured = compiler(
        measured_candidate,
        "placement",
        {"attachment_transform": attachment},
        2,
        7,
    )

    assert predicted["qualification_stages"][0]["xyz"] == pytest.approx(
        [0.48, -0.1, 0.343]
    )
    assert predicted["qualification_stages"][0]["scene_transition"] == (
        "virtual_detach"
    )
    assert predicted["qualification_stages"][0][
        "qualification_post_transition_gripper_state"
    ] == "open"
    assert predicted["qualification_stages"][0][
        "qualification_settled_object_pose"
    ]["xyz"] == pytest.approx([0.475, -0.101, 0.431])
    assert predicted["compile_parameters"][
        "qualified_settled_object_pose_sha256"
    ]
    assert predicted["compile_parameters"]["placement_candidate"][
        "object_goal_pose"
    ] == model_goal
    assert predicted["compile_parameters"]["placement_candidate"][
        "qualification_object_goal_source"
    ] == "model_pointcloud_goal_with_predicted_attachment"
    assert measured["qualification_stages"][0]["xyz"] == pytest.approx(
        [0.475, -0.101, 0.331]
    )
    assert measured["compile_parameters"]["placement_candidate"][
        "object_goal_pose"
    ] == physical_goal
    assert measured["compile_parameters"]["placement_candidate"][
        "qualification_object_goal_source"
    ] == "physical_goal_with_measured_attachment"

    dynamic_overlap_candidate = {
        **candidate,
        "qualified_settled_dynamic_overlap_ids": ["previous_payload"],
    }
    dynamic_overlap = compiler(dynamic_overlap_candidate, "placement", {}, 2, 7)
    dynamic_stage = dynamic_overlap["qualification_stages"][0]
    assert "qualification_settled_object_pose" not in dynamic_stage
    assert dynamic_stage["qualification_settled_probe_policy"] == (
        "release_endpoint_only_due_dynamic_settling"
    )
    assert dynamic_overlap["compile_parameters"][
        "qualification_settled_dynamic_overlap_ids"
    ] == ["previous_payload"]

    rebased_candidate = dict(candidate)
    rebased_candidate["frozen_object_motion_rebase"] = {
        "schema_version": "openeta.frozen_object_motion_rebase.v1",
        "model_inference_invoked": False,
    }
    rebased = compiler(rebased_candidate, "placement", {}, 3, 8)

    assert rebased["qualification_stages"][0]["xyz"] == pytest.approx(
        [0.475, -0.101, 0.331]
    )
    assert rebased["compile_parameters"]["placement_candidate"][
        "object_goal_pose"
    ] == physical_goal
    assert rebased["compile_parameters"]["placement_candidate"][
        "qualification_object_goal_source"
    ] == "physical_goal_with_measured_attachment"

    container_release_pointcloud_goal = {
        "frame": "world",
        "translation_xyz": [0.50, -0.08, 0.41],
        "rotation_matrix": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    }
    container_release_physical_goal = {
        "frame": "world",
        "translation_xyz": [0.495, -0.081, 0.398],
        "rotation_matrix": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    }
    container_candidate = {
        **candidate,
        "qualified_release_pointcloud_object_goal_pose": (
            container_release_pointcloud_goal
        ),
        "qualified_release_object_goal_pose": container_release_physical_goal,
        "container_drop_release_prebound": True,
    }
    predicted_container = compiler(container_candidate, "placement", {}, 2, 7)
    measured_container_candidate = dict(container_candidate)
    measured_container_candidate.pop("predicted_attachment_transform")
    measured_container = compiler(
        measured_container_candidate,
        "placement",
        {"attachment_transform": attachment},
        2,
        7,
    )

    assert predicted_container["compile_parameters"]["placement_candidate"][
        "object_goal_pose"
    ] == container_release_pointcloud_goal
    assert predicted_container["compile_parameters"]["placement_candidate"][
        "qualification_object_goal_source"
    ] == "container_release_pointcloud_goal_with_predicted_attachment"
    assert predicted_container["qualification_stages"][0][
        "terminal_pose_source"
    ] == "anyplace_se3_with_container_release_offset"
    assert measured_container["compile_parameters"]["placement_candidate"][
        "object_goal_pose"
    ] == container_release_physical_goal
    assert measured_container["compile_parameters"]["placement_candidate"][
        "qualification_object_goal_source"
    ] == "container_release_physical_goal_with_measured_attachment"
    profile_bytes = profile_path.read_bytes()
    recompiled = compile_placement_seed(
        measured_container["compile_parameters"],
        profile=json.loads(profile_bytes.decode("utf-8")),
        profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
    )
    recompiled_hash = hashlib.sha256(
        json.dumps(
            [recompiled["release_pose"]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert recompiled_hash == measured_container["compile_parameters"][
        "qualified_compiled_pose_sha256"
    ]


def test_frozen_pair_search_materializes_full_pool_round_robin_and_filters_grasps() -> None:
    captured: dict[str, Any] = {}

    def rpc(_name: str, request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        if _is_goal_prebind(request):
            return _goal_prebind_response(request)
        captured.update(request)
        # The first four descriptors must represent four different grasps.
        first_ids = [
            item["candidate"]["source_grasp_id"]
            for item in request["candidates"][:4]
        ]
        assert first_ids == ["g0", "g1", "g2", "g3"]
        passed_grasps = {"g0"}
        submitted = [
            item["candidate"]["source_grasp_id"] in passed_grasps and index < 3
            for index, item in enumerate(request["candidates"])
        ]
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": request["planning_scene_revision"],
            "execution_started": False,
            "results": [
                {
                    "candidate_id": item["candidate_id"],
                    "candidate_pose_sha256": item["candidate_pose_sha256"],
                    "qualification_binding_sha256": request[
                        "qualification_binding_sha256"
                    ],
                    "execution_started": False,
                    "verdict": "PASS" if submitted[index] else "FAIL",
                    "reason": "qualified" if submitted[index] else "not_submitted",
                    "stages": [_pass_stage()] if submitted[index] else [],
                    "full_plan_submitted": submitted[index],
                    "goal_legality": (
                        {
                            "verdict": "PASS",
                            "checks": {
                                "object_frame_binding": {
                                    "collision_goal_pose": {
                                        "convention": "T_world_collision_object_goal",
                                        "frame": "world",
                                        "translation_xyz": [
                                            0.48 + index * 0.001,
                                            0.0,
                                            0.43,
                                        ],
                                        "rotation_matrix": [
                                            [1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                        ],
                                    }
                                }
                            },
                        }
                        if submitted[index]
                        else None
                    ),
                }
                for index, item in enumerate(request["candidates"])
            ],
        }

    cache = QualificationCache()
    qualifier = MoveItCandidateQualifier(
        rpc,
        cache=cache,
        placement_full_plan_limit=4,
        placement_diversity_limit=96,
        compile_candidate=lambda *_args: {
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.4, 0.0, 0.5],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            ]
        },
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
    )
    grasps = [{"id": f"g{index}"} for index in range(4)]
    grasps[0]["frozen_object_motion_rebase"] = {
        "schema_version": "openeta.frozen_object_motion_rebase.v1",
        "model_inference_invoked": False,
    }
    proofs: dict[str, dict[str, Any]] = {}
    for grasp in grasps:
        grasp_id = grasp["id"]
        proofs[grasp_id] = {
            "verdict": "PASS",
            "stages": [
                {
                    "name": "contact",
                    "target_pose": {
                        "xyz": [0.3, 0.0, 0.45],
                        "rotation_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    },
                    "end_joint_state": {"joint_names": ["j1"], "positions": [0.0]},
                },
            ],
        }
    cache.replace(
        purpose="grasp",
        candidates=grasps,
        proofs=proofs,
        scene_epoch=3,
        planning_scene_revision=7,
    )
    coordinator = _FrozenGoalPairCoordinator(qualifier)
    coordinator.object_current_pose = {
        "frame": "world",
        "translation_xyz": [0.3, 0.0, 0.43],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    coordinator.object_goals = [
        {
            "id": f"p{index}",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.45 + index * 0.001, 0.0, 0.43],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            "frozen_goal_frame_binding": {"physical_collision_goal": True},
            "model_object_motion_world_transform": {
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.17],
                    [0.0, 1.0, 0.0, -0.04],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            },
        }
        for index in range(96)
    ]
    coordinator.scene_epoch = 3
    coordinator.planning_scene_revision = 7

    result = coordinator.filter_grasps(
        ToolResult(True, "ok", {"grasp_candidates": grasps}),
        scene_epoch=3,
        planning_scene_revision=7,
        source={},
    )

    assert len(captured["candidates"]) == 384
    assert captured["funnel"]["full_plan_limit"] == 2
    assert captured["funnel"]["screening_mode"] == (
        "progressive_until_full_plan_capacity"
    )
    assert captured["funnel"]["endpoint_pass_target"] == 2
    assert captured["funnel"]["l5_pass_target"] == 1
    assert captured["funnel"]["l5_min_pass_target"] == 1
    assert "l5_submission_limit" not in captured["funnel"]
    assert captured["funnel"]["qualification_mode"] == "frozen_pair"
    assert captured["funnel"]["defer_recovery"] is True
    rebased_pairs = [
        item["candidate"]
        for item in captured["candidates"]
        if item["candidate"]["source_grasp_id"] == "g0"
    ]
    assert rebased_pairs
    assert all(
        "object_motion_world_transform" not in pair
        and "model_object_motion_world_transform" not in pair
        for pair in rebased_pairs
    )
    assert [item["id"] for item in result.details["grasp_candidates"]] == ["g0"]
    assert result.details["ranking"] == "grasp_place_physical_quality"
    assert [
        item["grasp_place_physical_quality_rank"]
        for item in result.details["grasp_candidates"]
    ] == [0]
    assert [item["id"] for item in coordinator.grasp_frontier_candidates] == [
        "g1",
        "g2",
        "g3",
    ]
    assert cache.resolve(purpose="grasp", candidate_id="g1") is None
    retained_cache = cache.resolve(purpose="grasp", candidate_id="g0")
    assert retained_cache is not None
    assert cache.resolve(purpose="grasp", candidate_id="g2") is None
    assert "grasp_place_joint_qualified" not in retained_cache["candidate"]
    coordinator.source_model_raw_candidate_count = 96
    coordinator.source_candidate_image_ref = "/frozen-goals/candidates.png"
    coordinator.source_candidate_artifacts = [
        {
            "type": "placement_candidate_image",
            "kind": "image",
            "tool": "anyplace",
            "path": "/frozen-goals/candidates.png",
        }
    ]
    attachment = {
        "parent_frame": "eef",
        "child_frame": "object",
        "translation_xyz": [0.0, 0.0, 0.15],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    postattach = coordinator.prepare_frozen_goal_requalification(
        source_grasp_id="g0",
        attachment_transform=attachment,
        source={"placement_observation": {"rgb": "/postattach/rgb.png"}},
        scene_revision=8,
    )
    assert postattach is not None
    assert [item["id"] for item in postattach.details["placement_candidates"]] == [
        "p0",
        *[f"p{index}" for index in range(1, 96)],
    ]
    frozen_goal = postattach.details["placement_candidates"][0]
    assert frozen_goal["world_object_goal_pose"]["translation_xyz"] == [
        0.48,
        0.0,
        0.43,
    ]
    assert frozen_goal["model_pointcloud_object_goal_pose"]["translation_xyz"] == [
        0.45,
        0.0,
        0.43,
    ]
    assert frozen_goal["frozen_goal_frame_binding"]["physical_collision_goal"] is True
    seed_evidence = frozen_goal[SAME_RUN_QUALIFICATION_SEED_FIELD]
    assert seed_evidence["provenance"] == "frozen_pair_l5_pass"
    assert seed_evidence["states"] == [
        {"names": ["j1"], "positions": [0.25]}
    ]
    assert postattach.details["frozen_goal_requalification"] is True
    assert postattach.details["discarded_postattach_model_candidate_count"] == 0
    assert postattach.details["model_raw_candidate_count"] == 96
    assert postattach.details["raw_candidate_count"] == 96
    assert postattach.details["frozen_goal_priority_count"] == 1
    assert postattach.details["frozen_goal_frontier_count"] == 96
    assert postattach.details["metadata"]["candidate_source"] == (
        "frozen_anyplace_goal_frontier"
    )
    assert postattach.details["anyplace_model_inference_invoked"] is False
    assert postattach.details["candidate_image_ref"] == "/frozen-goals/candidates.png"
    assert postattach.details["artifacts"][0]["provenance"] == (
        "frozen_anyplace_goal_pool"
    )
    assert "source_grasp_id" not in postattach.details

    assert coordinator.prepare_frozen_goal_requalification(
        source_grasp_id="g0",
        attachment_transform=attachment,
        source={},
        scene_revision=8,
    ) is None


@pytest.mark.parametrize(
    "passing_variant",
    ["full_barrier_clearance", "configured_drop_height_fallback"],
)
def test_frozen_pair_tries_geometry_release_heights_before_expanding_grasps(
    passing_variant: str,
) -> None:
    calls: list[tuple[str, str]] = []

    def rpc(_name: str, request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        if _is_goal_prebind(request):
            variant = request["funnel"]["release_height_variant"]
            calls.append(("prebind", variant))
            response = _goal_prebind_response(request)
            for row in response["results"]:
                candidate = row["prebound_candidate"]
                candidate["placement_release_offset_selection"] = {
                    "configured_drop_height_m": 0.05,
                    "effective_offset_m": {
                        "geometry_primary": 0.075,
                        "full_barrier_clearance": 0.165,
                        "configured_drop_height_fallback": 0.05,
                    }[variant],
                    "full_barrier_clearance_offset_m": 0.165,
                    "source": {
                        "geometry_primary": "container_exterior_entry_clearance",
                        "full_barrier_clearance": "container_full_barrier_clearance",
                        "configured_drop_height_fallback": (
                            "configured_drop_height_fallback"
                        ),
                    }[variant],
                }
            return response

        variant = request["candidates"][0]["candidate"].get(
            "frozen_pair_release_height_variant",
            "geometry_primary",
        )
        calls.append(("pair", variant))
        rows = []
        for index, item in enumerate(request["candidates"]):
            passed = variant == passing_variant and index == 0
            rows.append(
                {
                    "candidate_id": item["candidate_id"],
                    "candidate_pose_sha256": item["candidate_pose_sha256"],
                    "qualification_binding_sha256": request[
                        "qualification_binding_sha256"
                    ],
                    "execution_started": False,
                    "verdict": "PASS" if passed else "FAIL",
                    "reason": "qualified" if passed else "ik_no_solution",
                    "stages": [_pass_stage()] if passed else [],
                    "full_plan_submitted": passed,
                    "goal_legality": {
                        "verdict": "PASS",
                        "checks": {
                            "object_frame_binding": {
                                "collision_goal_pose": {
                                    "convention": "T_world_collision_object_goal",
                                    "frame": "world",
                                    "translation_xyz": [0.48, 0.0, 0.43],
                                    "rotation_matrix": [
                                        [1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                    ],
                                }
                            }
                        },
                    },
                }
            )
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": request["planning_scene_revision"],
            "execution_started": False,
            "selected_candidate_ids": (
                [request["candidates"][0]["candidate_id"]]
                if variant == passing_variant
                else []
            ),
            "results": rows,
        }

    cache = QualificationCache()
    qualifier = MoveItCandidateQualifier(
        rpc,
        cache=cache,
        compile_candidate=lambda *_args: {
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.4, 0.0, 0.5],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            ]
        },
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
    )
    grasp = {"id": "g0", "score": 0.9}
    cache.replace(
        purpose="grasp",
        candidates=[grasp],
        proofs={
            "g0": {
                "verdict": "PASS",
                "stages": [
                    {
                        "name": "contact",
                        "target_pose": {
                            "xyz": [0.3, 0.0, 0.45],
                            "rotation_matrix": [
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                        },
                        "end_joint_state": {
                            "joint_names": ["j1"],
                            "positions": [0.0],
                        },
                    }
                ],
            }
        },
        scene_epoch=3,
        planning_scene_revision=7,
    )
    coordinator = _FrozenGoalPairCoordinator(
        qualifier,
        object_goals=[
            {
                "id": "p0",
                "object_goal_pose": {
                    "frame": "world",
                    "translation_xyz": [0.48, 0.0, 0.43],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            }
        ],
        object_current_pose={
            "frame": "world",
            "translation_xyz": [0.3, 0.0, 0.43],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        scene_epoch=3,
        planning_scene_revision=7,
    )

    result = coordinator.filter_grasps(
        ToolResult(True, "qualified", {"grasp_candidates": [grasp]}),
        scene_epoch=3,
        planning_scene_revision=7,
        source={},
    )

    assert result.success is True
    assert [item["id"] for item in result.details["grasp_candidates"]] == ["g0"]
    assert result.details["frozen_pair_release_height_full_barrier_activated"] is True
    assert result.details["frozen_pair_release_height_fallback_activated"] is (
        passing_variant == "configured_drop_height_fallback"
    )
    assert result.details["frozen_pair_count"] == (
        3 if passing_variant == "configured_drop_height_fallback" else 2
    )
    assert result.details["frozen_pair_frontier_expansion_count"] == 0
    expected_calls = [
        ("prebind", "geometry_primary"),
        ("pair", "geometry_primary"),
        ("prebind", "full_barrier_clearance"),
        ("pair", "full_barrier_clearance"),
    ]
    if passing_variant == "configured_drop_height_fallback":
        expected_calls.extend(
            [
                ("prebind", "configured_drop_height_fallback"),
                ("pair", "configured_drop_height_fallback"),
            ]
        )
    assert calls == expected_calls
    selected = coordinator.qualified_goals_by_grasp["g0"][0]
    assert selected["placement_release_offset_selection"]["source"] == {
        "full_barrier_clearance": "container_full_barrier_clearance",
        "configured_drop_height_fallback": "configured_drop_height_fallback",
    }[passing_variant]


def test_frozen_pair_retries_full_barrier_prebind_after_primary_static_collision() -> None:
    """A tall payload must not exhaust an otherwise valid AnyPlace pool."""

    calls: list[tuple[str, str]] = []

    def rpc(_name: str, request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        if _is_goal_prebind(request):
            variant = request["funnel"]["release_height_variant"]
            calls.append(("prebind", variant))
            if variant == "geometry_primary":
                return _goal_prebind_response(
                    request,
                    verdict="FAIL",
                    reason="goal_static_obstacle_penetration",
                )
            assert variant == "full_barrier_clearance"
            response = _goal_prebind_response(request)
            for row in response["results"]:
                candidate = row["prebound_candidate"]
                candidate["placement_release_offset_selection"] = {
                    "configured_drop_height_m": 0.05,
                    "effective_offset_m": 0.165,
                    "full_barrier_clearance_offset_m": 0.165,
                    "source": "container_full_barrier_clearance",
                }
            return response

        variant = request["candidates"][0]["candidate"].get(
            "frozen_pair_release_height_variant",
            "geometry_primary",
        )
        calls.append(("pair", variant))
        rows = [
            {
                "candidate_id": item["candidate_id"],
                "candidate_pose_sha256": item["candidate_pose_sha256"],
                "qualification_binding_sha256": request[
                    "qualification_binding_sha256"
                ],
                "execution_started": False,
                "verdict": "PASS",
                "reason": "qualified",
                "stages": [_pass_stage()],
                "full_plan_submitted": True,
                "goal_legality": {
                    "verdict": "PASS",
                    "checks": {
                        "object_frame_binding": {
                            "collision_goal_pose": {
                                "convention": "T_world_collision_object_goal",
                                "frame": "world",
                                "translation_xyz": [0.48, 0.0, 0.43],
                                "rotation_matrix": [
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                ],
                            }
                        }
                    },
                },
            }
            for item in request["candidates"]
        ]
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": request["planning_scene_revision"],
            "execution_started": False,
            "selected_candidate_ids": [request["candidates"][0]["candidate_id"]],
            "results": rows,
        }

    cache = QualificationCache()
    qualifier = MoveItCandidateQualifier(
        rpc,
        cache=cache,
        compile_candidate=lambda *_args: {
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.4, 0.0, 0.5],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            ]
        },
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
    )
    grasp = {"id": "g0", "score": 0.9}
    cache.replace(
        purpose="grasp",
        candidates=[grasp],
        proofs={
            "g0": {
                "verdict": "PASS",
                "stages": [
                    {
                        "name": "contact",
                        "target_pose": {
                            "xyz": [0.3, 0.0, 0.45],
                            "rotation_matrix": [
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                        },
                        "end_joint_state": {
                            "joint_names": ["j1"],
                            "positions": [0.0],
                        },
                    }
                ],
            }
        },
        scene_epoch=3,
        planning_scene_revision=7,
    )
    coordinator = _FrozenGoalPairCoordinator(
        qualifier,
        object_goals=[
            {
                "id": "p0",
                "object_goal_pose": {
                    "frame": "world",
                    "translation_xyz": [0.48, 0.0, 0.43],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            }
        ],
        object_current_pose={
            "frame": "world",
            "translation_xyz": [0.3, 0.0, 0.43],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        scene_epoch=3,
        planning_scene_revision=7,
    )

    result = coordinator.filter_grasps(
        ToolResult(True, "qualified", {"grasp_candidates": [grasp]}),
        scene_epoch=3,
        planning_scene_revision=7,
        source={},
    )

    assert result.success is True
    assert [item["id"] for item in result.details["grasp_candidates"]] == ["g0"]
    assert result.details["frozen_goal_release_height_full_barrier_attempted"] is True
    assert result.details["frozen_goal_release_height_full_barrier_activated"] is True
    assert result.details["frozen_goal_primary_legality_summary"][
        "frozen_goal_legality_reason_counts"
    ] == {"goal_static_obstacle_penetration": 1}
    assert calls == [
        ("prebind", "geometry_primary"),
        ("prebind", "full_barrier_clearance"),
        ("pair", "full_barrier_clearance"),
    ]
    selected = coordinator.qualified_goals_by_grasp["g0"][0]
    assert selected["placement_release_offset_selection"]["source"] == (
        "container_full_barrier_clearance"
    )


def test_fast_pair_search_returns_first_complete_and_retains_frozen_tail(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def pass_stage(
        *,
        contact: bool = False,
        joint_margin: float = 0.05,
        min_singular_value: float = 0.05,
        joint_travel: float = 0.5,
    ) -> dict[str, Any]:
        stage = _pass_stage()
        stage.update(
            {
                "joint_margin": joint_margin,
                "min_singular_value": min_singular_value,
                "joint_travel": joint_travel,
            }
        )
        if contact:
            stage.update(
                {
                    "name": "contact",
                    "target_pose": {
                        "xyz": [0.3, 0.0, 0.45],
                        "rotation_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    },
                }
            )
        return stage

    def rpc(_name: str, request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        calls.append(request)
        if _is_goal_prebind(request):
            return _goal_prebind_response(request)
        purpose = request["purpose"]
        first_grasp_id = (
            request["candidates"][0]["candidate_id"]
            if purpose == "grasp" and request["candidates"]
            else ""
        )
        passing_ids: list[str] = []
        results: list[dict[str, Any]] = []
        for item in request["candidates"]:
            candidate = item["candidate"]
            candidate_id = item["candidate_id"]
            source_grasp_id = str(candidate.get("source_grasp_id") or "")
            passed = (
                candidate_id == first_grasp_id
                if purpose == "grasp"
                else source_grasp_id in {"g2", "g3"}
            )
            not_evaluated = purpose == "grasp" and candidate_id != first_grasp_id
            if passed:
                passing_ids.append(candidate_id)
            result = {
                "candidate_id": candidate_id,
                "candidate_pose_sha256": item["candidate_pose_sha256"],
                "qualification_binding_sha256": request[
                    "qualification_binding_sha256"
                ],
                "execution_started": False,
                "verdict": (
                    "PASS" if passed else "NOT_EVALUATED" if not_evaluated else "FAIL"
                ),
                "reason": (
                    "qualified"
                    if passed
                    else "complete_l5_pass_found"
                    if not_evaluated
                    else "no_pair_solution"
                ),
                "stages": [
                    pass_stage(
                        contact=purpose == "grasp",
                        joint_margin=0.20 if candidate_id == "g2" else 0.05,
                        min_singular_value=0.20 if candidate_id == "g2" else 0.05,
                        joint_travel=0.20 if candidate_id == "g2" else 0.50,
                    )
                ]
                if passed
                else [],
                "endpoint_pass": passed,
                "full_plan_submitted": passed,
            }
            if purpose == "placement" and passed:
                result["goal_legality"] = {
                    "verdict": "PASS",
                    "checks": {
                        "object_frame_binding": {
                            "collision_goal_pose": {
                                "convention": "T_world_collision_object_goal",
                                "frame": "world",
                                "translation_xyz": [0.48, 0.0, 0.43],
                                "rotation_matrix": [
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                ],
                            }
                        }
                    },
                }
            results.append(result)
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": request["planning_scene_revision"],
            "execution_started": False,
            "qualification_profile": "fast_v3",
            "stop_reason": (
                "complete_l5_pass_found" if passing_ids else "candidate_pool_exhausted"
            ),
            "selected_candidate_ids": passing_ids,
            "results": results,
        }

    cache = QualificationCache()
    qualifier = MoveItCandidateQualifier(
        rpc,
        cache=cache,
        artifact_root=tmp_path,
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
        compile_candidate=lambda *_args: {
            "qualification_stages": [
                {
                    "name": "contact",
                    "xyz": [0.3, 0.0, 0.45],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            ]
        },
    )
    initial_grasps = [{"id": "g0", "score": 0.9}, {"id": "g1", "score": 0.8}]
    cache.replace(
        purpose="grasp",
        candidates=initial_grasps,
        proofs={
            grasp["id"]: {
                "verdict": "PASS",
                "endpoint_pass": True,
                "stages": [
                    pass_stage(
                        contact=True,
                        joint_margin=0.05,
                        min_singular_value=0.05,
                        joint_travel=0.50,
                    )
                ],
            }
            for grasp in initial_grasps
        },
        scene_epoch=3,
        planning_scene_revision=7,
    )
    coordinator = _FrozenGoalPairCoordinator(qualifier)
    coordinator.object_current_pose = {
        "frame": "world",
        "translation_xyz": [0.3, 0.0, 0.43],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    coordinator.object_goals = [
        {
            "id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, 0.0, 0.43],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        }
    ]
    coordinator.scene_epoch = 3
    coordinator.planning_scene_revision = 7
    coordinator.grasp_frontier_candidates = [
        {"id": "g2", "score": 0.7},
        {"id": "g3", "score": 0.6},
    ]
    coordinator.grasp_frontier_template = {
        "backend": "graspgenx_mcp",
        "model_raw_candidate_count": 3,
    }
    coordinator.grasp_frontier_scene_epoch = 3
    coordinator.grasp_frontier_planning_scene_revision = 7
    coordinator.grasp_frontier_generation = 1

    result = coordinator.filter_grasps(
        ToolResult(True, "qualified", {"grasp_candidates": initial_grasps}),
        scene_epoch=3,
        planning_scene_revision=7,
        source={},
    )

    assert [candidate["id"] for candidate in result.details["grasp_candidates"]] == ["g2"]
    assert result.details["frozen_pair_execution_target"] == 1
    assert result.details["frozen_pair_stop_reason"] == "complete_pair_found"
    assert result.details["frozen_pair_deferred_grasp_count"] == 2
    assert result.details["frozen_pair_fast_recovery_deferred_grasp_count"] == 2
    assert result.details["frozen_pair_deferred_recovery_batch_count"] == 0
    assert result.details["frozen_pair_frontier_expansion_count"] == 1
    assert result.details["frozen_grasp_frontier_remaining_count"] == 3
    assert result.details["frozen_grasp_frontier_model_inference_invoked"] is False
    assert [candidate["id"] for candidate in coordinator.grasp_frontier_candidates] == [
        "g0",
        "g1",
        "g3",
    ]
    assert result.details["ranking"] == "grasp_place_physical_quality"
    assert [
        candidate["grasp_place_frontier_quality_rank"]
        for candidate in result.details["grasp_candidates"]
    ] == [0]
    assert len(result.details["frozen_pair_qualification_artifacts"]) == 2
    assert len(result.details["frozen_grasp_frontier_qualification_artifacts"]) == 1
    assert [
        call["funnel"]["l5_pass_target"]
        for call in calls
        if call["purpose"] == "placement" and not _is_goal_prebind(call)
    ] == [1, 1]
    assert all(
        call["funnel"]["defer_recovery"] is True
        for call in calls
        if call["purpose"] == "placement" and not _is_goal_prebind(call)
    )
    assert sum(_is_goal_prebind(call) for call in calls) == 1
    assert [
        call["funnel"]["l5_pass_target"]
        for call in calls
        if call["purpose"] == "grasp"
    ] == [1]


def test_frozen_pair_recovery_runs_after_four_peer_fast_branches(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def rpc(_name: str, request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        if _is_goal_prebind(request):
            return _goal_prebind_response(request)
        calls.append(request)
        deferred = request["funnel"].get("defer_recovery") is True
        results = []
        selected = []
        for item in request["candidates"]:
            candidate = item["candidate"]
            source_grasp_id = str(candidate.get("source_grasp_id") or "")
            passed = not deferred and source_grasp_id == "g2" and not selected
            if passed:
                selected.append(item["candidate_id"])
            result = {
                "candidate_id": item["candidate_id"],
                "candidate_pose_sha256": item["candidate_pose_sha256"],
                "qualification_binding_sha256": request[
                    "qualification_binding_sha256"
                ],
                "execution_started": False,
                "verdict": "PASS" if passed else "FAIL",
                "reason": "qualified" if passed else "no_pair_solution",
                "stages": [_pass_stage()] if passed else [],
                "endpoint_pass": passed,
                "full_plan_submitted": passed,
            }
            if passed:
                result["goal_legality"] = {
                    "verdict": "PASS",
                    "checks": {
                        "object_frame_binding": {
                            "collision_goal_pose": {
                                "convention": "T_world_collision_object_goal",
                                "frame": "world",
                                "translation_xyz": [0.48, 0.0, 0.43],
                                "rotation_matrix": [
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                ],
                            }
                        }
                    },
                }
            results.append(result)
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": request["planning_scene_revision"],
            "execution_started": False,
            "stop_reason": (
                "complete_l5_pass_found"
                if selected
                else "fast_pool_exhausted_recovery_deferred"
                if deferred
                else "candidate_and_recovery_exhausted"
            ),
            "selected_candidate_ids": selected,
            "results": results,
        }

    cache = QualificationCache()
    qualifier = MoveItCandidateQualifier(
        rpc,
        cache=cache,
        artifact_root=tmp_path,
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
        compile_candidate=lambda *_args: {
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.48, 0.0, 0.50],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            ]
        },
    )
    grasps = [{"id": f"g{index}", "score": 1.0 - index * 0.1} for index in range(4)]
    proofs = {
        grasp["id"]: {
            "verdict": "PASS",
            "endpoint_pass": True,
            "stages": [
                {
                    "name": "contact",
                    "target_pose": {
                        "xyz": [0.3, 0.0, 0.45],
                        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    },
                    "end_joint_state": {"names": ["j1"], "positions": [0.0]},
                }
            ],
        }
        for grasp in grasps
    }
    cache.replace(
        purpose="grasp",
        candidates=grasps,
        proofs=proofs,
        scene_epoch=3,
        planning_scene_revision=7,
    )
    coordinator = _FrozenGoalPairCoordinator(qualifier, grasp_branch_limit=4)
    coordinator.object_current_pose = {
        "frame": "world",
        "translation_xyz": [0.3, 0.0, 0.43],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    coordinator.object_goals = [
        {
            "id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, 0.0, 0.43],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        }
    ]
    coordinator.scene_epoch = 3
    coordinator.planning_scene_revision = 7

    result = coordinator.filter_grasps(
        ToolResult(True, "qualified", {"grasp_candidates": grasps}),
        scene_epoch=3,
        planning_scene_revision=7,
        source={},
    )

    assert [call["funnel"].get("defer_recovery") for call in calls] == [True, None]
    assert [candidate["id"] for candidate in result.details["grasp_candidates"]] == [
        "g2"
    ]
    assert result.details["frozen_pair_fast_recovery_deferred_grasp_count"] == 4
    assert result.details["frozen_pair_deferred_recovery_batch_count"] == 1
    assert [candidate["id"] for candidate in coordinator.grasp_frontier_candidates] == [
        "g0",
        "g1",
        "g3",
    ]


def test_frozen_pair_frontier_returns_after_one_bounded_wave(
    tmp_path: Path,
) -> None:
    """A failed wave must not synchronously consume the immutable tail."""

    calls: list[tuple[list[str], bool]] = []

    class _WaveCoordinator(_FrozenGoalPairCoordinator):
        def _filter_grasp_batch(
            self,
            result: ToolResult,
            *,
            scene_epoch: int,
            planning_scene_revision: int,
            source: dict[str, object],
            defer_pair_recovery: bool = False,
        ) -> ToolResult:
            candidates = result.details.get("grasp_candidates")
            candidate_ids = [
                str(candidate.get("id") or "")
                for candidate in candidates
                if isinstance(candidates, list) and isinstance(candidate, dict)
            ]
            calls.append((candidate_ids, defer_pair_recovery))
            return ToolResult(
                True,
                "no complete pair in this deterministic barrier",
                {
                    "grasp_candidates": [],
                    "frozen_pair_count": len(candidate_ids),
                    "frozen_pair_full_plan_pass_count": 0,
                },
            )

        def prepare_grasp_frontier_expansion(self, **_kwargs: object) -> ToolResult:
            raise AssertionError("the next frozen wave must be planner-dispatched")

    grasps = [{"id": f"g{index}", "score": 1.0 - index * 0.1} for index in range(4)]
    cache = QualificationCache()
    cache.replace(
        purpose="grasp",
        candidates=grasps,
        proofs={grasp["id"]: {"verdict": "PASS"} for grasp in grasps},
        scene_epoch=3,
        planning_scene_revision=7,
    )
    qualifier = SimpleNamespace(cache=cache, qualification_profile="fast_v3")
    coordinator = _WaveCoordinator(qualifier, grasp_branch_limit=4)
    coordinator.object_current_pose = {"frame": "world"}
    coordinator.object_goals = [{"id": "p0"}]
    coordinator.scene_epoch = 3
    coordinator.planning_scene_revision = 7
    coordinator.grasp_frontier_candidates = [{"id": "g4", "score": 0.5}]
    coordinator.grasp_frontier_generation = 1

    result = coordinator.filter_grasps(
        ToolResult(True, "qualified", {"grasp_candidates": grasps}),
        scene_epoch=3,
        planning_scene_revision=7,
        source={},
    )

    assert calls == [(["g0", "g1", "g2", "g3"], True), (["g0", "g1", "g2", "g3"], False)]
    assert result.details["grasp_candidates"] == []
    assert result.details["frozen_pair_stop_reason"] == (
        "frozen_grasp_frontier_wave_complete"
    )
    assert result.details["frozen_pair_frontier_resume_required"] is True
    assert result.details["frozen_pair_frontier_wave_grasp_count"] == 4
    assert result.details["frozen_pair_frontier_expansion_count"] == 0
    assert result.details["frozen_grasp_frontier_remaining_count"] == 1
    assert [candidate["id"] for candidate in coordinator.grasp_frontier_candidates] == [
        "g4"
    ]


def test_frozen_pair_search_does_not_reuse_stale_goal_pool() -> None:
    qualifier = MoveItCandidateQualifier(lambda *_args: {})
    coordinator = _FrozenGoalPairCoordinator(
        qualifier,
        object_goals=[{"id": "p0"}],
        object_current_pose={"frame": "world"},
        scene_epoch=1,
        planning_scene_revision=2,
    )
    result = ToolResult(True, "ok", {"grasp_candidates": [{"id": "g0"}]})

    returned = coordinator.filter_grasps(
        result,
        scene_epoch=2,
        planning_scene_revision=2,
        source={},
    )

    assert returned.details["grasp_candidates"] == [{"id": "g0"}]


def test_frozen_grasp_frontier_rebases_only_with_static_scene_and_detach_proof() -> None:
    raw_grasp = {
        "id": "g1",
        "frame": "camera",
        "camera_frame": "opencv",
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    coordinator = _FrozenGoalPairCoordinator(
        MoveItCandidateQualifier(lambda *_args: {}),
        object_goals=[{"id": "p0", "object_goal_pose": {"frame": "world"}}],
        object_current_pose={
            "frame": "world",
            "translation_xyz": [0.28, -0.1, 0.43],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        scene_epoch=1,
        planning_scene_revision=1,
        grasp_frontier_candidates=[raw_grasp],
        grasp_candidate_catalog={"g1": raw_grasp},
        grasp_frontier_template={
            "source": {
                "camera_extrinsics": {
                    "camera_frame": "opencv",
                    "pos": [0.0, 0.0, 0.0],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
        grasp_frontier_scene_epoch=1,
        grasp_frontier_planning_scene_revision=1,
    )
    sync = {
        "schema_version": "openeta.planning_scene_target_pose_sync.v1",
        "operation": "update_world_target",
        "source_revision": 1,
        "revision": 2,
        "source_target_pose": {
            "frame": "world",
            "translation_xyz": [0.28, -0.1, 0.43],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "target_pose": {
            "frame": "world",
            "translation_xyz": [0.29, -0.1, 0.43],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "world_ids_before": ["table", "target_object"],
        "world_ids_after": ["table", "target_object"],
        "attached_ids_before": [],
        "attached_ids_after": [],
        "topology_unchanged": True,
        "static_world_unchanged": True,
        "static_world_sha256_after": "static-scene",
        "translation_delta_m": 0.01,
        "rotation_delta_rad": 0.0,
    }
    receipt = {
        "planning_scene_target_pose_sync": sync,
        "detachable_joint": {"state": "detached"},
    }

    result = coordinator.rebase_grasp_frontier_from_target_pose_sync(
        receipt,
        scene_epoch=4,
        planning_scene_revision=2,
    )

    assert result.success is True
    assert coordinator.grasp_frontier_candidates[0][
        "translation_xyz"
    ] == pytest.approx([0.11, 0.2, 0.3])
    assert coordinator.grasp_frontier_candidates[0][
        "frozen_object_motion_rebase"
    ] == coordinator.grasp_frontier_template["frozen_object_motion_rebase"]
    assert coordinator.grasp_frontier_candidates[0][
        "frozen_object_motion_rebase"
    ]["planning_scene_revision"] == 2
    assert coordinator.object_current_pose["translation_xyz"] == [0.29, -0.1, 0.43]
    assert coordinator.planning_scene_revision == 2
    assert coordinator.grasp_frontier_planning_scene_revision == 2

    unsafe = dict(receipt)
    unsafe["detachable_joint"] = {"state": "attached"}
    rejected = coordinator.rebase_grasp_frontier_from_target_pose_sync(
        unsafe,
        scene_epoch=5,
        planning_scene_revision=3,
    )
    assert rejected.success is False
    assert rejected.details["reason"] == "frozen_grasp_frontier_rebase_proof_missing"


def test_complete_goal_rejection_never_leaks_a_grasp_only_frontier_candidate() -> None:
    calls: list[str] = []

    def rpc(_name: str, request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        calls.append(request["purpose"])
        if _is_goal_prebind(request):
            return _goal_prebind_response(
                request,
                verdict="FAIL",
                reason="goal_support_surface_penetration",
            )
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": request["planning_scene_revision"],
            "execution_started": False,
            "qualification_profile": "fast_v3",
            "stop_reason": "candidate_and_recovery_exhausted",
            "selected_candidate_ids": [],
            "results": [
                {
                    "candidate_id": item["candidate_id"],
                    "candidate_pose_sha256": item["candidate_pose_sha256"],
                    "qualification_binding_sha256": request[
                        "qualification_binding_sha256"
                    ],
                    "execution_started": False,
                    "verdict": "FAIL",
                    "reason": "goal_support_surface_penetration",
                    "stages": [],
                    "endpoint_pass": False,
                    "full_plan_submitted": False,
                    "goal_legality": {
                        "goal_id": item["candidate"]["source_object_goal_id"],
                        "verdict": "FAIL",
                        "reason": "goal_support_surface_penetration",
                        "checks": {},
                    },
                }
                for item in request["candidates"]
            ],
        }

    cache = QualificationCache()
    qualifier = MoveItCandidateQualifier(
        rpc,
        cache=cache,
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
        compile_candidate=lambda *_args: {
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.48, 0.0, 0.5],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            ]
        },
    )
    initial_grasps = [{"id": "g0"}, {"id": "g1"}]
    cache.replace(
        purpose="grasp",
        candidates=initial_grasps,
        proofs={
            grasp["id"]: {
                "verdict": "PASS",
                "stages": [
                    {
                        "name": "contact",
                        "target_pose": {
                            "xyz": [0.3, 0.0, 0.45],
                            "rotation_matrix": [
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                        },
                        "end_joint_state": {
                            "joint_names": ["j1"],
                            "positions": [0.0],
                        },
                    }
                ],
            }
            for grasp in initial_grasps
        },
        scene_epoch=3,
        planning_scene_revision=7,
    )
    coordinator = _FrozenGoalPairCoordinator(
        qualifier,
        object_goals=[
            {
                "id": "p0",
                "object_goal_pose": {
                    "frame": "world",
                    "translation_xyz": [0.48, 0.0, 0.43],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            }
        ],
        object_current_pose={
            "frame": "world",
            "translation_xyz": [0.3, 0.0, 0.43],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        scene_epoch=3,
        planning_scene_revision=7,
        grasp_frontier_candidates=[{"id": "g2"}],
        grasp_frontier_template={"backend": "graspgenx_mcp"},
        grasp_frontier_scene_epoch=3,
        grasp_frontier_planning_scene_revision=7,
    )

    result = coordinator.filter_grasps(
        ToolResult(True, "qualified", {"grasp_candidates": initial_grasps}),
        scene_epoch=3,
        planning_scene_revision=7,
        source={},
    )

    assert result.details["grasp_candidates"] == []
    assert result.details["frozen_pair_stop_reason"] == "frozen_goal_pool_exhausted"
    assert coordinator.qualified_goals_by_grasp == {}
    assert coordinator.object_goals == []
    assert calls == ["placement"]


def test_complete_goal_legality_evidence_caches_only_physical_pass_frontier() -> None:
    coordinator = _FrozenGoalPairCoordinator(
        MoveItCandidateQualifier(lambda *_args: {}),
        object_goals=[
            {
                "id": candidate_id,
                "object_goal_pose": {
                    "frame": "world",
                    "translation_xyz": [x, 0.0, 0.45],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
                "world_object_goal_pose": {
                    "frame": "world",
                    "translation_xyz": [x, 0.0, 0.45],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
            }
            for candidate_id, x in (("p-pass", 0.48), ("p-reject", 0.58))
        ],
    )
    physical_goal = {
        "frame": "world",
        "translation_xyz": [0.475, 0.0, 0.43],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    summary = coordinator._cache_goal_legality_frontier(  # noqa: SLF001
        ToolResult(
            True,
            "qualified",
            {
                "qualification_evidence": {
                    "results": [
                        {
                            "candidate_id": "pair-pass",
                            "goal_legality": {
                                "goal_id": "p-pass",
                                "verdict": "PASS",
                                "checks": {
                                    "object_frame_binding": {
                                        "collision_goal_pose": physical_goal
                                    }
                                },
                            },
                        },
                        {
                            "candidate_id": "pair-reject",
                            "goal_legality": {
                                "goal_id": "p-reject",
                                "verdict": "FAIL",
                                "checks": {},
                            },
                        },
                    ]
                }
            },
        ),
        pairs=[
            {"id": "pair-pass", "source_object_goal_id": "p-pass"},
            {"id": "pair-reject", "source_object_goal_id": "p-reject"},
        ],
    )

    assert summary["frozen_goal_legality_screen_complete"] is True
    assert summary["frozen_goal_legality_frontier_count"] == 1
    assert [goal["id"] for goal in coordinator.object_goals] == ["p-pass"]
    retained = coordinator.object_goals[0]
    assert retained["object_goal_pose"] == physical_goal
    assert retained["model_pointcloud_object_goal_pose"]["translation_xyz"] == [
        0.48,
        0.0,
        0.45,
    ]


def _identity_extrinsics() -> dict[str, Any]:
    return {
        "camera_frame": "opencv",
        "camera_to_world": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    }


def _observation_packet(
    tmp_path: Path,
    *,
    name: str,
    mask_key: str,
) -> dict[str, Any]:
    rgb = tmp_path / f"{name}.rgb.png"
    depth = tmp_path / f"{name}.depth.png"
    mask = tmp_path / f"{name}.mask.png"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(rgb)
    Image.new("I;16", (8, 8), 500).save(depth)
    Image.new("L", (8, 8), 255).save(mask)
    return {
        "rgb": str(rgb),
        "depth": str(depth),
        mask_key: {"mask_ref": str(mask), "source_image": str(rgb)},
        "intrinsics": {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 4.0,
            "cy": 4.0,
            "scale": 1000.0,
        },
        "camera_extrinsics": _identity_extrinsics(),
        "camera_frame_id": name,
    }


def _anyplace_parameters(tmp_path: Path) -> dict[str, Any]:
    return {
        "object_observation": _observation_packet(
            tmp_path, name="held-object", mask_key="object_mask"
        ),
        "placement_observation": _observation_packet(
            tmp_path, name="destination", mask_key="placement_region_mask"
        ),
        "scene_revision": 7,
    }


def _model_response() -> dict[str, Any]:
    return {
        "success": True,
        "content": "AnyPlace model inference",
        "details": {
            "backend": "anyplace_mcp",
            "model": "anyplace_multitask",
            "frame": "placement_camera",
            "camera_frame": "opencv",
            "candidate_count": 1,
            "object_current_pose": {
                "frame": "world",
                "translation_xyz": [0.3, 0.0, 0.45],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            "placement_candidates": [
                {
                    "id": "placement_model_goal",
                    "object_placement_transform": {
                        "frame": "placement_camera",
                        "camera_frame": "opencv",
                        "convention": "p_placed = R @ p_current + t",
                        "transform_matrix": [
                            [1, 0, 0, 0.1],
                            [0, 1, 0, 0.0],
                            [0, 0, 1, 0.0],
                            [0, 0, 0, 1.0],
                        ],
                    },
                }
            ],
            "metadata": {"configured_candidate_count": 1},
        },
    }


def _postattachment_memory(measured_attachment: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_epoch": 3,
        "attachment_gate": {
            "status": "resolved",
            "verdict": "PASS",
            "planning_scene_revision": 7,
            "attachment_proof": {"attachment_transform": measured_attachment},
        },
        "grasp_execution": {
            "status": "completed",
            "stage": "attached",
            "compiled_grasp": {"source_grasp_id": "g-selected"},
        },
    }


def _context(
    tmp_path: Path,
    *,
    measured_attachment: dict[str, Any],
    attached: bool = True,
) -> ToolExecutionContext:
    memory = _postattachment_memory(measured_attachment) if attached else {"scene_epoch": 3}
    return ToolExecutionContext(
        name="anyplace",
        spec=build_default_tool_registry().get("anyplace"),
        parameters=_anyplace_parameters(tmp_path),
        observation=EnvObservation(
            task="pick and place",
            cameras=[],
            robot=RobotState(
                end_effector_pose={
                    "frame": "world",
                    "xyz": [0.3, 0.0, 0.6],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            ),
            metadata={"planning_scene_revision": 7},
        ),
        metadata={
            "session_id": "test",
            "supervision_context": {"memory": memory},
        },
    )


def _postattachment_handler(
    tmp_path: Path,
    *,
    qualifier: MoveItCandidateQualifier,
    coordinator: _FrozenGoalPairCoordinator,
    predictor,
):
    raw = build_anyplace_handler(
        predictor,
        output_root=tmp_path / "anyplace-runs",
        pre_inference=lambda context, request: _prepare_postattachment_frozen_goals(
            context,
            request,
            coordinator=coordinator,
        ),
    )
    return _qualifying_handler(
        raw,
        qualifier,
        purpose="placement",
        frozen_pair_coordinator=coordinator,
    )


def test_postattach_frozen_goal_pass_skips_anyplace_model(
    tmp_path: Path,
) -> None:
    captured_request: dict[str, Any] = {}
    captured_source: dict[str, Any] = {}
    predictor_calls: list[dict[str, Any]] = []

    def compile_candidate(candidate, _purpose, source, _epoch, _revision):
        captured_source.update(source)
        assert candidate["id"] == "p-pass"
        return {
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.4, 0.0, 0.5],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            ]
        }

    def rpc(_name, request, _timeout):
        captured_request.update(request)
        item = request["candidates"][0]
        return {
            "schema_version": QUALIFICATION_SCHEMA,
            "planning_scene_revision": request["planning_scene_revision"],
            "execution_started": False,
            "results": [
                {
                    "candidate_id": item["candidate_id"],
                    "candidate_pose_sha256": item["candidate_pose_sha256"],
                    "qualification_binding_sha256": request[
                        "qualification_binding_sha256"
                    ],
                    "execution_started": False,
                    "verdict": "PASS",
                    "reason": "qualified",
                    "stages": [_pass_stage()],
                    "full_plan_submitted": True,
                }
            ],
        }

    qualifier = MoveItCandidateQualifier(
        rpc,
        compile_candidate=compile_candidate,
        placement_full_plan_limit=4,
        placement_diversity_limit=96,
    )
    coordinator = _FrozenGoalPairCoordinator(
        qualifier,
        qualified_goals_by_grasp={
            "g-selected": [
                {
                    "id": "p-pass",
                    "object_goal_pose": {
                        "frame": "world",
                        "translation_xyz": [0.48, -0.1, 0.43],
                        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    },
                }
            ]
        },
        source_model_raw_candidate_count=96,
        source_candidate_image_ref=str(tmp_path / "frozen-goal-candidates.png"),
        source_candidate_artifacts=[
            {
                "type": "placement_candidate_image",
                "kind": "image",
                "tool": "anyplace",
                "path": str(tmp_path / "frozen-goal-candidates.png"),
            }
        ],
        source_binding={
            "object_observation": {"frozen": True},
            "placement_observation": {"frozen": True},
            "object_camera_to_placement_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "placement_camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "placement_camera_extrinsics": {
                "frame": "world",
                "translation_xyz": [0, 0, 0],
                "quat_xyzw": [0, 0, 0, 1],
            },
        },
    )
    measured_attachment = {
        "parent_frame": "eef",
        "child_frame": "object",
        "translation_xyz": [0.001, -0.015, 0.153],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    handler = _postattachment_handler(
        tmp_path,
        qualifier=qualifier,
        coordinator=coordinator,
        predictor=lambda request: predictor_calls.append(request) or _model_response(),
    )

    context = _context(tmp_path, measured_attachment=measured_attachment)
    context.parameters = {"reuse_frozen_goal_pool": True, "scene_revision": 7}
    result = handler(context)
    repeated = handler(context)

    assert result.success
    assert predictor_calls == []
    assert [item["id"] for item in result.details["placement_candidates"]] == [
        "p-pass"
    ]
    assert "qualification_round" not in result.details
    assert result.details["model_raw_candidate_count"] == 96
    assert result.details["raw_candidate_count"] == 1
    assert result.details["anyplace_model_inference_invoked"] is False
    assert result.details["artifacts"][0]["provenance"] == (
        "frozen_anyplace_goal_pool"
    )
    assert "source_grasp_id" not in str(result.details["source"])
    assert captured_request["funnel"]["full_plan_limit"] == 4
    assert captured_source["attachment_transform"] == measured_attachment
    assert captured_source["frozen_goal_requalification"] is True
    assert repeated.success is False
    assert repeated.details["reason"] == "frozen_goal_requalification_already_consumed"
    assert predictor_calls == []


def test_postattach_continues_frozen_frontier_after_priority_goal_fails(
    tmp_path: Path,
) -> None:
    predictor_calls: list[dict[str, Any]] = []
    qualification_candidate_ids: list[str] = []

    def rpc(_name, request, _timeout):
        qualification_candidate_ids.extend(
            item["candidate_id"] for item in request["candidates"]
        )
        results = []
        for item in request["candidates"]:
            passed = item["candidate_id"] == "p-frontier"
            results.append(
                {
                    "candidate_id": item["candidate_id"],
                    "candidate_pose_sha256": item["candidate_pose_sha256"],
                    "qualification_binding_sha256": request[
                        "qualification_binding_sha256"
                    ],
                    "execution_started": False,
                    "verdict": "PASS" if passed else "FAIL",
                    "reason": "qualified" if passed else "collision_state_invalid",
                    "stages": [_pass_stage()] if passed else [],
                    "full_plan_submitted": passed,
                }
            )
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": request["planning_scene_revision"],
            "execution_started": False,
            "results": results,
        }

    def goal(candidate_id: str, x: float) -> dict[str, Any]:
        return {
            "id": candidate_id,
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [x, -0.1, 0.43],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        }

    qualifier = MoveItCandidateQualifier(
        rpc,
        compile_candidate=lambda *_args: {
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.4, 0.0, 0.5],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            ]
        },
        placement_full_plan_limit=4,
        placement_diversity_limit=96,
    )
    priority = goal("p-priority", 0.48)
    coordinator = _FrozenGoalPairCoordinator(
        qualifier,
        object_goals=[goal("p-priority", 0.48), goal("p-frontier", 0.58)],
        qualified_goals_by_grasp={"g-selected": [priority]},
        source_model_raw_candidate_count=96,
    )
    measured_attachment = {
        "parent_frame": "eef",
        "child_frame": "object",
        "translation_xyz": [0.001, -0.015, 0.153],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    handler = _postattachment_handler(
        tmp_path,
        qualifier=qualifier,
        coordinator=coordinator,
        predictor=lambda request: predictor_calls.append(request) or _model_response(),
    )

    result = handler(_context(tmp_path, measured_attachment=measured_attachment))

    assert result.success
    assert qualification_candidate_ids == ["p-priority", "p-frontier"]
    assert [item["id"] for item in result.details["placement_candidates"]] == [
        "p-frontier"
    ]
    assert result.details["raw_candidate_count"] == 2
    assert result.details["frozen_goal_priority_count"] == 1
    assert result.details["anyplace_model_inference_invoked"] is False
    assert predictor_calls == []


def test_postattach_resume_excludes_failed_physical_goal_without_model_inference(
    tmp_path: Path,
) -> None:
    predictor_calls: list[dict[str, Any]] = []
    qualification_rounds: list[list[str]] = []

    def rpc(_name, request, _timeout):
        candidate_ids = [item["candidate_id"] for item in request["candidates"]]
        qualification_rounds.append(candidate_ids)
        pass_id = "p-priority" if len(qualification_rounds) == 1 else "p-next"
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": request["planning_scene_revision"],
            "execution_started": False,
            "results": [
                {
                    "candidate_id": item["candidate_id"],
                    "candidate_pose_sha256": item["candidate_pose_sha256"],
                    "qualification_binding_sha256": request[
                        "qualification_binding_sha256"
                    ],
                    "execution_started": False,
                    "verdict": (
                        "PASS" if item["candidate_id"] == pass_id else "FAIL"
                    ),
                    "reason": (
                        "qualified"
                        if item["candidate_id"] == pass_id
                        else "not_evaluated"
                    ),
                    "stages": (
                        [_pass_stage()]
                        if item["candidate_id"] == pass_id
                        else []
                    ),
                    "full_plan_submitted": item["candidate_id"] == pass_id,
                }
                for item in request["candidates"]
            ],
        }

    def goal(candidate_id: str, x: float) -> dict[str, Any]:
        return {
            "id": candidate_id,
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [x, -0.1, 0.43],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        }

    qualifier = MoveItCandidateQualifier(
        rpc,
        compile_candidate=lambda *_args: {
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.4, 0.0, 0.5],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            ]
        },
        placement_full_plan_limit=4,
        placement_diversity_limit=96,
    )
    coordinator = _FrozenGoalPairCoordinator(
        qualifier,
        object_goals=[goal("p-priority", 0.48), goal("p-next", 0.58)],
        qualified_goals_by_grasp={"g-selected": [goal("p-priority", 0.48)]},
        source_model_raw_candidate_count=96,
        source_binding={"placement_observation": {"frozen": True}},
    )
    measured_attachment = {
        "parent_frame": "eef",
        "child_frame": "object",
        "translation_xyz": [0.001, -0.015, 0.153],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    handler = _postattachment_handler(
        tmp_path,
        qualifier=qualifier,
        coordinator=coordinator,
        predictor=lambda request: predictor_calls.append(request) or _model_response(),
    )
    context = _context(tmp_path, measured_attachment=measured_attachment)
    context.parameters = {"reuse_frozen_goal_pool": True, "scene_revision": 7}

    first = handler(context)
    context.parameters = {
        "reuse_frozen_goal_pool": True,
        "resume_frozen_goal_frontier": True,
        "excluded_frozen_goal_ids": ["p-priority"],
        "scene_revision": 7,
    }
    second = handler(context)
    repeated = handler(context)

    assert first.success and second.success
    assert [item["id"] for item in first.details["placement_candidates"]] == [
        "p-priority"
    ]
    assert [item["id"] for item in second.details["placement_candidates"]] == [
        "p-next"
    ]
    assert qualification_rounds == [["p-priority", "p-next"], ["p-next"]]
    assert second.details["frozen_goal_frontier_generation"] == 1
    assert second.details["frozen_goal_excluded_count"] == 1
    assert repeated.success is False
    assert predictor_calls == []


def test_frozen_zero_pass_never_reinvokes_anyplace_model(
    tmp_path: Path,
) -> None:
    predictor_calls: list[dict[str, Any]] = []
    qualification_requests: list[dict[str, Any]] = []

    def rpc(_name, request, _timeout):
        qualification_requests.append(request)
        return {
            "schema_version": QUALIFICATION_SCHEMA,
            "planning_scene_revision": request["planning_scene_revision"],
            "execution_started": False,
            "results": [
                {
                    "candidate_id": item["candidate_id"],
                    "candidate_pose_sha256": item["candidate_pose_sha256"],
                    "qualification_binding_sha256": request[
                        "qualification_binding_sha256"
                    ],
                    "execution_started": False,
                    "verdict": "FAIL",
                    "reason": "ik_failed",
                    "stages": [],
                    "full_plan_submitted": False,
                }
                for index, item in enumerate(request["candidates"])
            ],
        }

    qualifier = MoveItCandidateQualifier(
        rpc,
        compile_candidate=lambda *_args: {
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.4, 0.0, 0.5],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            ]
        },
        placement_full_plan_limit=4,
        placement_diversity_limit=96,
    )
    coordinator = _FrozenGoalPairCoordinator(
        qualifier,
        qualified_goals_by_grasp={
            "g-selected": [
                {
                    "id": "p-frozen-fail",
                    "object_goal_pose": {
                        "frame": "world",
                        "translation_xyz": [0.48, -0.1, 0.43],
                        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    },
                }
            ]
        },
        source_model_raw_candidate_count=96,
    )
    measured_attachment = {
        "parent_frame": "eef",
        "child_frame": "object",
        "translation_xyz": [0.001, -0.015, 0.153],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    handler = _postattachment_handler(
        tmp_path,
        qualifier=qualifier,
        coordinator=coordinator,
        predictor=lambda request: predictor_calls.append(request) or _model_response(),
    )
    frozen = handler(_context(tmp_path, measured_attachment=measured_attachment))
    repeated = handler(_context(tmp_path, measured_attachment=measured_attachment))
    assert frozen.success and frozen.details["candidate_count"] == 0
    assert repeated.success is False
    assert repeated.details["reason"] == "frozen_goal_requalification_already_consumed"
    assert predictor_calls == []
    assert len(qualification_requests) == 1


def test_only_unattached_goal_generation_may_call_anyplace_model(
    tmp_path: Path,
) -> None:
    predictor_calls: list[dict[str, Any]] = []
    qualifier = MoveItCandidateQualifier(
        lambda _name, request, _timeout: {
            "schema_version": QUALIFICATION_SCHEMA,
            "planning_scene_revision": request["planning_scene_revision"],
            "execution_started": False,
            "results": [],
        },
        compile_candidate=lambda *_args: {"qualification_stages": [{"name": "release"}]},
    )
    coordinator = _FrozenGoalPairCoordinator(qualifier)
    raw = build_anyplace_handler(
        lambda request: predictor_calls.append(request) or _model_response(),
        output_root=tmp_path / "anyplace-runs",
        pre_inference=lambda context, request: _prepare_postattachment_frozen_goals(
            context,
            request,
            coordinator=coordinator,
        ),
    )
    measured_attachment = {
        "translation_xyz": [0.0, 0.0, 0.15],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }

    unattached = raw(
        _context(tmp_path, measured_attachment=measured_attachment, attached=False)
    )
    attached_without_matching_pool = raw(
        _context(tmp_path, measured_attachment=measured_attachment, attached=True)
    )

    assert unattached.success
    assert attached_without_matching_pool.success is False
    assert attached_without_matching_pool.details["reason"] == "frozen_goal_pool_missing"
    assert len(predictor_calls) == 1
    assert unattached.details.get("frozen_goal_requalification") is not True
