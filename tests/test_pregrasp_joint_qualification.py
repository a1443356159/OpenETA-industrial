from __future__ import annotations

from pathlib import Path
from typing import Any

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
    _compile_qualified_queue,
    _prepare_postattachment_frozen_goals,
    _qualifying_handler,
)
from agent.tools.handlers import build_anyplace_handler
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


def test_frozen_pair_search_materializes_full_pool_round_robin_and_filters_grasps() -> None:
    captured: dict[str, Any] = {}

    def rpc(_name: str, request: dict[str, Any], _timeout: float) -> dict[str, Any]:
        captured.update(request)
        # The first four descriptors must represent four different grasps.
        first_ids = [
            item["candidate"]["source_grasp_id"]
            for item in request["candidates"][:4]
        ]
        assert first_ids == ["g0", "g1", "g2", "g3"]
        passed_grasps = {"g0", "g2"}
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
    assert captured["funnel"]["l5_pass_target"] == 2
    assert "l5_submission_limit" not in captured["funnel"]
    assert captured["funnel"]["qualification_mode"] == "frozen_pair"
    assert [item["id"] for item in result.details["grasp_candidates"]] == ["g0", "g2"]
    assert cache.resolve(purpose="grasp", candidate_id="g1") is None
    retained_cache = cache.resolve(purpose="grasp", candidate_id="g0")
    assert retained_cache is not None
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
