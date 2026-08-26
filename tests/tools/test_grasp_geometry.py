from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from adapter.protocol import EnvObservation, RobotState
from agent.tools.grasp_geometry import (
    DEFAULT_GRASP_PROFILE,
    GraspGeometryError,
    build_compile_grasp_seed_handler,
    build_compile_placement_seed_handler,
    compile_grasp_seed,
    compile_placement_seed,
    materialize_world_object_goal,
    materialize_world_object_goal_from_current_pose,
    predicted_attachment_from_grasp,
    project_attached_object_center_to_image,
    qualification_grasp_pose_chain,
)
from agent.tools.registry import ToolExecutionContext, ToolSpec


def _profile() -> dict:
    return json.loads(DEFAULT_GRASP_PROFILE.read_text(encoding="utf-8"))


def _profile_sha() -> str:
    return hashlib.sha256(DEFAULT_GRASP_PROFILE.read_bytes()).hexdigest()


def _candidate() -> dict:
    return {
        "id": "grasp_000",
        "frame": "camera",
        "camera_frame": "opencv",
        "width": 0.06,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }


def _grasp_parameters() -> dict:
    return {
        "purpose": "grasp",
        "camera_pose": _candidate(),
        "camera_extrinsics": {
            "pos": [0.0, 0.0, 0.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "camera_frame_id": "agentview",
        "target_class": "upright_can",
        "scene_epoch": 0,
    }


def test_project_attached_object_center_uses_native_ack_geometry() -> None:
    projection = project_attached_object_center_to_image(
        current_eef_pose={
            "xyz": [0.1, -0.2, 1.0],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        attachment_transform={
            "schema_version": "openeta.attachment_transform.v1",
            "parent_frame": "eef",
            "child_frame": "object",
            "measurement_boundary": "native_attach_ack",
            "translation_xyz": [0.1, 0.1, 0.0],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0},
        camera_extrinsics={
            "pos": [0.0, 0.0, 0.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "camera_frame": "opencv",
        },
        image_width=100,
        image_height=80,
    )

    assert projection["point_xy"] == pytest.approx([70.0, 30.0])
    assert projection["depth_m"] == pytest.approx(1.0)


def test_project_attached_object_center_rejects_non_ack_transform() -> None:
    with pytest.raises(GraspGeometryError, match="native T_eef_object"):
        project_attached_object_center_to_image(
            current_eef_pose={
                "xyz": [0.0, 0.0, 1.0],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            attachment_transform={
                "schema_version": "openeta.attachment_transform.v1",
                "parent_frame": "eef",
                "child_frame": "object",
                "measurement_boundary": "predicted",
                "translation_xyz": [0.0, 0.0, 0.0],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            intrinsics={"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0},
            camera_extrinsics={
                "pos": [0.0, 0.0, 0.0],
                "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "camera_frame": "opencv",
            },
            image_width=100,
            image_height=80,
        )


def _compiler_spec() -> ToolSpec:
    return ToolSpec(
        name="host_candidate_compiler",
        category="host_workflow",
        description="test-only host compiler context",
        effect="read_only",
    )


def _qualified_proof(compile_parameters: dict) -> dict:
    return {
        "verdict": "PASS",
        "qualification_binding_sha256": "a" * 64,
        "compile_parameters": compile_parameters,
        "stages": [
            {
                "plan_only": True,
                "execution_started": False,
                "end_joint_state": {
                    "names": [f"joint_{index}" for index in range(1, 8)],
                    "positions": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
                },
            }
        ],
    }


def test_grasp_compiler_preserves_model_terminal_after_frame_and_tcp_transform() -> None:
    compiled = compile_grasp_seed(
        _grasp_parameters(),
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert compiled["schema_version"] == "openeta.compiled_grasp_seed.v2"
    assert compiled["terminal_pose_source"] == (
        "model_pose_with_calibrated_frame_transform"
    )
    assert compiled["path_owner"] == "moveit"
    assert compiled["orientation_clamped"] is False
    assert compiled["contact_pose"]["xyz"] == pytest.approx([0.1036, -0.2, -0.3])
    assert compiled["contact_pose"]["rotation_matrix"] == [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    assert qualification_grasp_pose_chain(compiled) == [compiled["contact_pose"]]
    forbidden = {
        "hover_pose",
        "precontact_pose",
        "lift_pose",
        "retreat_pose",
        "grasp_strategy",
        "wrist_alignment_policy",
    }
    assert forbidden.isdisjoint(compiled)


def test_semantic_strategy_fields_cannot_change_model_contact() -> None:
    baseline = compile_grasp_seed(
        _grasp_parameters(), profile=_profile(), profile_sha256="sha"
    )
    annotated_parameters = deepcopy(_grasp_parameters())
    annotated_parameters.update(
        {
            "strategy_id": "legacy-top-down-policy",
            "target_class": "bowl",
        }
    )
    annotated = compile_grasp_seed(
        annotated_parameters, profile=_profile(), profile_sha256="sha"
    )

    assert annotated["contact_pose"]["xyz"] == baseline["contact_pose"]["xyz"]
    assert (
        annotated["contact_pose"]["rotation_matrix"]
        == baseline["contact_pose"]["rotation_matrix"]
    )
    assert "strategy_id" not in annotated


def test_grasp_compiler_rejects_only_invalid_physical_geometry() -> None:
    invalid_rotation = _grasp_parameters()
    invalid_rotation["camera_pose"]["rotation_matrix"][0][0] = 2.0
    excessive_width = _grasp_parameters()
    excessive_width["camera_pose"]["width"] = 1.0

    with pytest.raises(GraspGeometryError):
        compile_grasp_seed(
            invalid_rotation, profile=_profile(), profile_sha256="sha"
        )
    with pytest.raises(GraspGeometryError):
        compile_grasp_seed(
            excessive_width, profile=_profile(), profile_sha256="sha"
        )


def test_qualified_grasp_hash_binds_only_exact_contact() -> None:
    parameters = _grasp_parameters()
    compiled = compile_grasp_seed(
        parameters, profile=_profile(), profile_sha256=_profile_sha()
    )
    proof_parameters = {
        **parameters,
        "qualification_profile_sha256": _profile_sha(),
        "qualified_compiled_pose_sha256": hashlib.sha256(
            json.dumps(
                [compiled["contact_pose"]],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }

    class Cache:
        def resolve(self, **_kwargs):
            public_candidate = _candidate()
            public_candidate["moveit_physical_quality_rank"] = 0
            public_candidate["moveit_l5_qualified"] = True
            return {
                "candidate": public_candidate,
                "proof": _qualified_proof(proof_parameters),
                "scene_epoch": 0,
                "planning_scene_revision": 0,
            }

    context = ToolExecutionContext(
        name="host_candidate_compiler",
        spec=_compiler_spec(),
        parameters={"purpose": "grasp", "grasp_candidate_id": "grasp_000"},
        observation=EnvObservation(
            task="test",
            cameras=[],
            robot=RobotState(),
            metadata={"scene_epoch": 0, "planning_scene_revision": 0},
        ),
    )

    result = build_compile_grasp_seed_handler(qualification_cache=Cache())(
        context
    )

    assert result.success is True
    contact_pose = result.details["outputs"]["contact_pose"]
    assert contact_pose["xyz"] == compiled["contact_pose"]["xyz"]
    assert (
        contact_pose["rotation_matrix"]
        == compiled["contact_pose"]["rotation_matrix"]
    )
    assert contact_pose["qualification_goal_joint_state"] == {
        "names": [f"joint_{index}" for index in range(1, 8)],
        "positions": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
    }
    assert len(contact_pose["qualification_goal_joint_state_sha256"]) == 64
    assert contact_pose["qualification_binding_sha256"] == "a" * 64


def _placement_candidate() -> dict:
    return {
        "id": "placement_002",
        "object_goal_pose": {
            "frame": "world",
            "translation_xyz": [0.48, -0.1, 0.43],
            "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    }


def _attachment() -> dict:
    return {
        "parent_frame": "eef",
        "child_frame": "object",
        "translation_xyz": [0.136, 0.0, 0.0],
        "quat_xyzw": [0, 0, 0, 1],
    }


def test_placement_compiler_derives_one_exact_release_from_model_goal() -> None:
    compiled = compile_placement_seed(
        {
            "placement_candidate_id": "placement_002",
            "placement_candidate": _placement_candidate(),
            "attachment_transform": _attachment(),
            "scene_epoch": 4,
            "scene_revision": 7,
        },
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert compiled["schema_version"] == "openeta.compiled_placement_seed.v3"
    assert compiled["terminal_pose_source"] == (
        "anyplace_object_goal_with_measured_attachment"
    )
    assert compiled["path_owner"] == "moveit"
    assert compiled["release_pose"]["xyz"] == pytest.approx([0.344, -0.1, 0.43])
    assert compiled["release_pose"]["placement_stage"] == "release"
    assert {
        "hover_pose",
        "descend_pose",
        "retreat_pose",
        "release_clearance_m",
    }.isdisjoint(compiled)


def test_placement_handler_rejects_stale_attachment_and_accepts_frozen_proof() -> None:
    robot = RobotState(
        joint_positions=[0.1, 0.2],
        gripper_state={"openness": 0.2},
    )
    parameters = {
        "placement_candidate_id": "placement_002",
        "placement_candidate": _placement_candidate(),
        "attachment_transform": _attachment(),
        "scene_epoch": 4,
        "scene_revision": 7,
        "qualification_profile_sha256": _profile_sha(),
        "qualified_attachment_transform_sha256": hashlib.sha256(
            json.dumps(
                _attachment(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "qualified_start_state_sha256": hashlib.sha256(
            json.dumps(
                {
                    "joint_positions": robot.joint_positions,
                    "gripper_state": robot.gripper_state,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    compiled = compile_placement_seed(
        parameters, profile=_profile(), profile_sha256=_profile_sha()
    )
    parameters["qualified_compiled_pose_sha256"] = hashlib.sha256(
        json.dumps(
            [compiled["release_pose"]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    class Cache:
        def resolve(self, **_kwargs):
            return {
                "candidate": _placement_candidate(),
                "proof": _qualified_proof(parameters),
                "scene_epoch": 4,
                "planning_scene_revision": 7,
            }

    def context(attachment: dict) -> ToolExecutionContext:
        return ToolExecutionContext(
            name="host_candidate_compiler",
            spec=_compiler_spec(),
            parameters={"placement_candidate_id": "placement_002"},
            observation=EnvObservation(
                task="place",
                cameras=[],
                robot=robot,
                metadata={"scene_epoch": 1, "planning_scene_revision": 1},
            ),
            metadata={
                "_openeta_host_candidate_compilation_binding": {
                    "scene_epoch": 4,
                    "planning_scene_revision": 7,
                },
                "supervision_context": {
                    "memory": {
                        "attachment_gate": {
                            "attachment_proof": {
                                "attachment_transform": attachment
                            }
                        },
                        "placement_candidate_policy": {
                            "scene_epoch": 4,
                            "planning_scene_revision": 7,
                        },
                    }
                },
            },
        )

    handler = build_compile_placement_seed_handler(
        qualification_cache=Cache()
    )
    accepted = handler(context(_attachment()))
    changed = _attachment()
    changed["translation_xyz"][0] += 0.01
    rejected = handler(context(changed))

    assert accepted.success is True
    assert accepted.details["outputs"]["release_pose"][
        "qualification_goal_joint_state"
    ]["positions"] == [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    assert rejected.success is False
    assert "attachment transform is stale" in rejected.content


def test_materialize_anyplace_goal_preserves_full_se3() -> None:
    candidate = materialize_world_object_goal(
        {
            "id": "placement_000",
            "object_placement_transform": {
                "frame": "placement_camera",
                "transform_matrix": [
                    [0, 0, 1, 0.144],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ],
            },
        },
        placement_camera_extrinsics={
            "camera_frame": "opencv",
            "camera_to_world": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        },
        current_eef_pose={
            "xyz": [0.2, -0.1, 0.43],
            "quat_xyzw": [0, 0, 0, 1],
        },
        attachment_transform={
            "translation_xyz": [0.136, 0, 0],
            "quat_xyzw": [0, 0, 0, 1],
        },
    )

    assert candidate["object_goal_pose"]["rotation_matrix"] == [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    assert "orientation_projection" not in candidate


def test_predicted_attachment_preserves_model_contact_to_goal_rigid_motion() -> None:
    contact = {
        "frame": "world",
        "xyz": [0.30, -0.10, 0.45],
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    attachment = predicted_attachment_from_grasp(
        contact_pose={
            **contact,
        },
        object_current_pose={
            "frame": "world",
            "xyz": [0.10, -0.15, 0.45],
            "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    )
    compiled = compile_placement_seed(
        {
            "placement_candidate_id": "placement_000",
            "placement_candidate": {
                "id": "placement_000",
                "object_goal_pose": {
                    "frame": "world",
                    "translation_xyz": [0.30, -0.10, 0.45],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
            },
            "attachment_transform": attachment,
            "scene_epoch": 1,
            "scene_revision": 2,
        },
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert attachment["translation_xyz"] == pytest.approx([-0.20, -0.05, 0.0])
    assert compiled["release_pose"]["xyz"] == pytest.approx([0.50, -0.05, 0.45])
    assert compiled["release_pose"]["rotation_matrix"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_pre_attachment_goal_materialization_uses_current_object_not_eef() -> None:
    goal = materialize_world_object_goal_from_current_pose(
        {
            "id": "placement_000",
            "object_placement_transform": {
                "frame": "placement_camera",
                "transform_matrix": [
                    [1, 0, 0, 0.2],
                    [0, 1, 0, 0.0],
                    [0, 0, 1, 0.0],
                    [0, 0, 0, 1.0],
                ],
            },
        },
        placement_camera_extrinsics={
            "camera_frame": "opencv",
            "camera_to_world": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        },
        object_current_pose={
            "frame": "world",
            "translation_xyz": [0.3, -0.1, 0.43],
            "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    )

    assert goal["object_goal_pose"]["translation_xyz"] == pytest.approx(
        [0.5, -0.1, 0.43]
    )


def test_grasp_compiler_rejects_placement_contract() -> None:
    with pytest.raises(GraspGeometryError, match="only compiles grasp"):
        compile_grasp_seed(
            {"purpose": "placement"},
            profile=_profile(),
            profile_sha256="profile-sha",
        )
