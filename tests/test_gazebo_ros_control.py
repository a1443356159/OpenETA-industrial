from __future__ import annotations

import hashlib
import inspect
import json
import math
from types import SimpleNamespace
import time

import pytest

from extensions.gazebo.robot_control import (
    ARM_JOINTS,
    JOINT_NAMES,
    GazeboControlConfig,
    make_move_group_goal,
    robot_state_from_sources,
)
from extensions.gazebo.native_grasp import NativePickPlaceConfig
from extensions.gazebo.ros_control import (
    L5_TRAJECTORY_START_TOLERANCE_RAD,
    QUALIFIED_JOINT_GOAL_TOLERANCE_RAD,
    RosGazeboStateSource,
    _RosRuntime,
    _attached_support_departure_audit,
    _collision_message_geometry_record,
    _configured_qualification_solver_profile,
    _qualification_ik_response_timeout_s,
    _moveit_scene_frame,
    _move_group_failure_result,
    _merged_allowed_collision_rows,
    _state_valid_with_allowed_collision_pairs,
    _load_qualification_capability_map,
    _joint_states_within_l5_start_tolerance,
    _l5_trajectory_cache_key,
    _qualification_joint_state_with_sha256,
    _qualification_pose_target,
    _trajectory_end_joint_state_with_sha256,
    _populate_motion_start_state,
    _populate_recovery_trajectory_goal,
    _populate_state_validity_request,
    _qualification_robot_model_sha256,
)
from agent.runtime.capability_map import generate_sparse_capability_map, robot_model_hash


def _pose(xyz=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        position=SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
        orientation=SimpleNamespace(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
    )


def _support_departure_geometry():
    return (
        {
            "id": "target",
            "shape": "box",
            "size_xyz": [0.02, 0.02, 0.02],
            "pose_xyz": [0.0, 0.0, 0.0],
            "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "id": "work_table",
            "shape": "box",
            "size_xyz": [2.0, 2.0, 0.02],
            "pose_xyz": [0.0, 0.0, -0.01],
            "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )


def test_attached_support_departure_accepts_moveit_generated_separation() -> None:
    target, table = _support_departure_geometry()

    evidence = _attached_support_departure_audit(
        joint_names=["lift"],
        trajectory_positions=[[0.0], [0.01]],
        forward_kinematics=lambda _names, joints: (
            [0.0, 0.0, 0.01 + joints[0]],
            [0.0, 0.0, 0.0, 1.0],
        ),
        mount_xyz=[0.0, 0.0, 0.0],
        mount_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        attached_spec=target,
        support_spec=table,
    )

    assert evidence["valid"] is True
    assert evidence["route_owner"] == "moveit"
    assert evidence["host_offset_pose_generated"] is False
    assert evidence["initial_clearance_m"] == pytest.approx(0.0)
    assert evidence["minimum_clearance_m"] == pytest.approx(0.0)
    assert evidence["evaluated_sample_count"] == 3


def test_attached_support_departure_rejects_scraping_moveit_path() -> None:
    target, table = _support_departure_geometry()

    evidence = _attached_support_departure_audit(
        joint_names=["slide"],
        trajectory_positions=[[0.0], [0.1]],
        forward_kinematics=lambda _names, joints: (
            [joints[0], 0.0, 0.01],
            [0.0, 0.0, 0.0, 1.0],
        ),
        mount_xyz=[0.0, 0.0, 0.0],
        mount_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        attached_spec=target,
        support_spec=table,
    )

    assert evidence["valid"] is False
    assert evidence["failure"]["reason"] == (
        "support_contact_persists_after_departure"
    )
    assert evidence["failure"]["sample_kind"] == "midpoint"


def test_attached_world_audit_rejects_carried_object_crossing_bin_wall() -> None:
    target, table = _support_departure_geometry()
    bin_wall = {
        "id": "green_parts_bin",
        "shape": "box",
        "size_xyz": [0.01, 0.1, 0.1],
        "pose_xyz": [0.05, 0.0, 0.01],
        "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    evidence = _attached_support_departure_audit(
        joint_names=["slide"],
        trajectory_positions=[[0.0], [0.1]],
        forward_kinematics=lambda _names, joints: (
            [joints[0], 0.0, 0.01],
            [0.0, 0.0, 0.0, 1.0],
        ),
        mount_xyz=[0.0, 0.0, 0.0],
        mount_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        attached_spec=target,
        support_spec=table,
        obstacle_specs={"green_parts_bin": bin_wall},
    )

    assert evidence["valid"] is False
    assert evidence["authoritative_world_collision_audit"] is True
    assert evidence["failure"] == {
        "reason": "attached_object_static_collision",
        "point_index": 1,
        "sample_kind": "midpoint",
        "obstacle_id": "green_parts_bin",
        "target_primitive_index": 0,
        "obstacle_primitive_index": 0,
    }
    assert evidence["exact_static_box_pair_check_count"] == 2


def test_attached_support_departure_rejects_initial_penetration() -> None:
    target, table = _support_departure_geometry()

    evidence = _attached_support_departure_audit(
        joint_names=["lift"],
        trajectory_positions=[[0.0], [0.01]],
        forward_kinematics=lambda _names, joints: (
            [0.0, 0.0, 0.009 + joints[0]],
            [0.0, 0.0, 0.0, 1.0],
        ),
        mount_xyz=[0.0, 0.0, 0.0],
        mount_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        attached_spec=target,
        support_spec=table,
    )

    assert evidence["valid"] is False
    assert evidence["failure"]["reason"] == "initial_support_penetration"


def test_attached_support_departure_uses_native_attach_contact_as_start_baseline() -> None:
    target, table = _support_departure_geometry()
    measured_at_attach = {
        **target,
        # Native physics reports a shallow support overlap.  The independent
        # FK reconstruction below is sampled later and is 120 nm deeper, then
        # MoveIt separates the target immediately.
        "pose_xyz": [0.0, 0.0, 0.01 - 3.0e-8],
    }

    evidence = _attached_support_departure_audit(
        joint_names=["lift"],
        trajectory_positions=[[0.0], [0.01]],
        forward_kinematics=lambda _names, joints: (
            [0.0, 0.0, 0.01 - 1.5e-7 + joints[0]],
            [0.0, 0.0, 0.0, 1.0],
        ),
        mount_xyz=[0.0, 0.0, 0.0],
        mount_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        attached_spec=target,
        support_spec=table,
        support_contact_reference_target_spec=measured_at_attach,
    )

    assert evidence["valid"] is True
    assert evidence["initial_clearance_m"] == pytest.approx(-1.5e-7)
    assert evidence["initial_support_reference_clearance_m"] == pytest.approx(
        -3.0e-8
    )
    assert evidence["support_contact_pose_uncertainty_m"] == pytest.approx(1.0e-6)
    assert evidence["support_departed"] is True
    assert evidence["first_moving_clearance_m"] > 0.0


def test_attached_support_departure_rejects_start_deeper_than_native_baseline() -> None:
    target, table = _support_departure_geometry()
    measured_at_attach = {
        **target,
        "pose_xyz": [0.0, 0.0, 0.01 - 3.0e-8],
    }

    evidence = _attached_support_departure_audit(
        joint_names=["lift"],
        trajectory_positions=[[0.0], [0.01]],
        forward_kinematics=lambda _names, joints: (
            [0.0, 0.0, 0.0099 + joints[0]],
            [0.0, 0.0, 0.0, 1.0],
        ),
        mount_xyz=[0.0, 0.0, 0.0],
        mount_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        attached_spec=target,
        support_spec=table,
        support_contact_reference_target_spec=measured_at_attach,
    )

    assert evidence["valid"] is False
    assert evidence["failure"]["reason"] == "initial_support_penetration"


def test_moveit_geometry_proof_accepts_equivalent_object_pose_factoring() -> None:
    primitive = SimpleNamespace(type=1, dimensions=[1.15, 0.95, 0.06])
    outgoing = SimpleNamespace(
        id="work_table",
        header=SimpleNamespace(frame_id="base_link"),
        pose=_pose(quat=(0.0, 0.0, 0.0, 0.0)),
        primitives=[primitive],
        primitive_poses=[_pose((0.35, 0.0, -0.03))],
    )
    readback = SimpleNamespace(
        id="work_table",
        header=SimpleNamespace(frame_id="base_link"),
        pose=_pose((0.35, 0.0, -0.03)),
        primitives=[primitive],
        primitive_poses=[_pose()],
    )

    assert _collision_message_geometry_record(outgoing) == (
        _collision_message_geometry_record(readback)
    )


def test_qualification_solver_auto_matches_launch_profile(monkeypatch) -> None:
    monkeypatch.setenv("OPENETA_QUALIFICATION_SOLVER_PROFILE", "auto")
    monkeypatch.setenv("OPENETA_QUALIFICATION_PROFILE", "legacy")
    assert _configured_qualification_solver_profile() == "kdl_legacy"
    monkeypatch.setenv("OPENETA_QUALIFICATION_PROFILE", "fast_v3")
    assert _configured_qualification_solver_profile() == "kdl_fast"
    monkeypatch.setenv("OPENETA_QUALIFICATION_SOLVER_PROFILE", "trac_ik_speed")
    assert _configured_qualification_solver_profile() == "trac_ik_speed"


def test_fast_ik_solver_budget_does_not_shorten_ros_response_deadline() -> None:
    # Eight short IK requests may queue behind one MoveGroup service callback.
    # The IK request still carries 50 ms, while its transport may wait for the
    # bounded queue to drain without classifying a reachable pose as infra loss.
    assert _qualification_ik_response_timeout_s(
        0.05,
        queue_depth=8,
    ) == pytest.approx(25.0)
    assert _qualification_ik_response_timeout_s(
        0.05,
        queue_depth=8,
    ) > 8 * 2.0


@pytest.mark.parametrize("queue_depth", [0, -1, True])
def test_fast_ik_response_deadline_rejects_invalid_queue_depth(queue_depth) -> None:
    with pytest.raises(ValueError, match="response deadline inputs are invalid"):
        _qualification_ik_response_timeout_s(0.05, queue_depth=queue_depth)


def test_plan_only_rejected_trajectory_never_claims_execution() -> None:
    result = _move_group_failure_result(
        99999,
        41,
        plan_only=True,
        planning_failure_codes=set(),
        timed_out_code=-6,
        generic_failure_code=99999,
    )

    assert result["ok"] is False
    assert result["error_code"] == "MOTION_PLAN_FAILED"
    assert result["planned_point_count"] == 41
    assert result["execution_started"] is False


def test_qualified_joint_branch_uses_a_tight_terminal_tolerance() -> None:
    assert QUALIFIED_JOINT_GOAL_TOLERANCE_RAD == pytest.approx(0.001)


def test_l5_trajectory_cache_requires_same_start_scene_and_proof_binding() -> None:
    planned = {
        "names": [f"joint_{index}" for index in range(1, 8)],
        "positions": [0.0] * 7,
    }
    reordered_live = {
        "names": list(reversed(planned["names"])),
        "positions": list(
            reversed([L5_TRAJECTORY_START_TOLERANCE_RAD * 0.5] * 7)
        ),
    }
    drifted = dict(reordered_live)
    drifted["positions"] = list(drifted["positions"])
    drifted["positions"][0] = L5_TRAJECTORY_START_TOLERANCE_RAD * 2.0

    assert _joint_states_within_l5_start_tolerance(planned, reordered_live)
    assert not _joint_states_within_l5_start_tolerance(planned, drifted)

    base_goal = {
        "qualification_cache_binding_sha256": "a" * 64,
        "group_name": "rm_group",
        "link_name": "link_7",
        "requested_tool_pose": {"xyz": [0.4, 0.0, 0.5]},
        "target_pose": {"xyz": [0.4, 0.0, 0.5]},
        "motion_profile": "unloaded",
        "max_velocity_scaling_factor": 0.3,
        "max_acceleration_scaling_factor": 0.2,
    }
    private_key = _l5_trajectory_cache_key(
        base_goal,
        scene_revision=7,
        scene_sha256="scene-a",
    )
    public_goal = dict(base_goal)
    public_goal["qualification_binding_sha256"] = public_goal.pop(
        "qualification_cache_binding_sha256"
    )
    public_key = _l5_trajectory_cache_key(
        public_goal,
        scene_revision=7,
        scene_sha256="scene-a",
    )

    assert private_key == public_key
    assert private_key is not None
    assert _l5_trajectory_cache_key(
        public_goal,
        scene_revision=8,
        scene_sha256="scene-a",
    ) != private_key
    assert _l5_trajectory_cache_key(
        public_goal,
        scene_revision=7,
        scene_sha256="scene-b",
    ) != private_key
    virtual_goal = dict(public_goal)
    virtual_goal["qualification_scene_diff"] = {"remove_world_ids": ["target"]}
    assert _l5_trajectory_cache_key(
        virtual_goal,
        scene_revision=7,
        scene_sha256="scene-a",
    ) != private_key


def test_l5_trajectory_cache_accepts_euler_roundtrip_pose_noise_only() -> None:
    private_goal = {
        "qualification_cache_binding_sha256": "b" * 64,
        "qualification_goal_joint_state_sha256": "c" * 64,
        "compiled_grasp_id": "grasp_0042",
        "compiled_placement_id": "placement_0018",
        "placement_candidate_id": "placement_0018",
        "group_name": "rm_group",
        "link_name": "link_7",
        "requested_tool_pose": {
            "xyz": [0.62000004, 0.17999997, 0.17000002],
            "quat_xyzw": [
                -0.5177379270,
                0.5351307377,
                0.4898047062,
                0.4536402756,
            ],
        },
        "target_pose": {
            "xyz": [0.62000004, 0.17999997, 0.17000002],
            "quat_xyzw": [
                -0.5177379270,
                0.5351307377,
                0.4898047062,
                0.4536402756,
            ],
        },
        "motion_profile": "loaded",
        "max_velocity_scaling_factor": 0.25,
        "max_acceleration_scaling_factor": 0.15,
    }
    public_goal = json.loads(json.dumps(private_goal))
    public_goal["qualification_binding_sha256"] = public_goal.pop(
        "qualification_cache_binding_sha256"
    )
    for pose_name in ("requested_tool_pose", "target_pose"):
        public_goal[pose_name]["xyz"] = [0.62, 0.18, 0.17]
        public_goal[pose_name]["quat_xyzw"] = [
            -0.5177378617,
            0.5351306600,
            0.4898047901,
            0.4536402918,
        ]

    private_key = _l5_trajectory_cache_key(
        private_goal, scene_revision=2, scene_sha256="scene"
    )
    public_key = _l5_trajectory_cache_key(
        public_goal, scene_revision=2, scene_sha256="scene"
    )
    assert private_key == public_key

    different_candidate = dict(public_goal, compiled_placement_id="placement_0019")
    assert _l5_trajectory_cache_key(
        different_candidate, scene_revision=2, scene_sha256="scene"
    ) != private_key


def test_private_l5_goal_matches_public_joint_digest_and_candidate_identity() -> None:
    runtime = object.__new__(_RosRuntime)
    runtime.config = GazeboControlConfig()
    runtime.planning_scene = SimpleNamespace(revision=5)
    runtime.move = lambda goal, _timeout_s: goal
    joint_state = {
        "names": list(ARM_JOINTS),
        "positions": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
    }
    target = {
        "xyz": [0.42, -0.03, 0.31],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "compiled_grasp_id": "grasp_0042",
        "grasp_stage": "contact",
        "compiled_placement_id": "placement_0018",
        "placement_candidate_id": "placement_0018",
        "placement_stage": "release",
        "qualification_goal_joint_state": joint_state,
        "_qualification_cache_binding_sha256": "d" * 64,
    }

    private_goal = runtime.qualification_plan_only(
        target, joint_state, planning_time_s=1.0, planning_attempts=1
    )
    expected_state_hash = hashlib.sha256(
        json.dumps(
            joint_state, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    public_target = dict(target)
    public_target.pop("_qualification_cache_binding_sha256")
    public_target["qualification_goal_joint_state_sha256"] = expected_state_hash
    public_target["qualification_binding_sha256"] = "d" * 64
    public_goal = make_move_group_goal(public_target, config=runtime.config)
    public_goal.update(
        {
            "model_id": runtime.config.model_id,
            "planning_scene_revision": 5,
        }
    )

    assert private_goal["qualification_goal_joint_state"] == joint_state
    assert (
        private_goal["qualification_goal_joint_state_sha256"]
        == expected_state_hash
    )
    assert private_goal["compiled_grasp_id"] == "grasp_0042"
    assert private_goal["grasp_stage"] == "contact"
    assert _l5_trajectory_cache_key(
        private_goal, scene_revision=5, scene_sha256="scene"
    ) == _l5_trajectory_cache_key(
        public_goal, scene_revision=5, scene_sha256="scene"
    )


def test_private_qualification_projects_only_pose_goal_fields() -> None:
    target = {
        "xyz": [0.42, -0.03, 0.31],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "compiled_grasp_id": "grasp_0042",
        "grasp_stage": "contact",
        "qualification_allowed_collisions": {"target": ["left_tip", "right_tip"]},
        "qualification_allowed_collisions_sha256": "a" * 64,
        "qualification_goal_joint_state": {
            "names": list(ARM_JOINTS),
            "positions": [0.0] * len(ARM_JOINTS),
        },
        "qualification_binding_sha256": "b" * 64,
        "qualification_scene_diff": {"remove_world_ids": ["target"]},
        "solver_profile": "kdl_fast",
    }

    projected = _qualification_pose_target(target)

    assert projected == {
        "xyz": target["xyz"],
        "quat_xyzw": target["quat_xyzw"],
        "compiled_grasp_id": "grasp_0042",
        "grasp_stage": "contact",
    }
    goal = make_move_group_goal(projected)
    assert "qualification_allowed_collisions" not in goal
    assert "qualification_goal_joint_state" not in goal


def test_l5_cache_binds_to_time_parameterized_trajectory_endpoint() -> None:
    ik_state = {
        "names": list(ARM_JOINTS),
        "positions": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
    }
    planned_endpoint = [
        0.1004,
        -0.2002,
        0.3003,
        -0.4001,
        0.5004,
        -0.6003,
        0.7002,
    ]
    trajectory = SimpleNamespace(
        joint_names=list(ARM_JOINTS),
        points=[SimpleNamespace(positions=planned_endpoint)],
    )
    endpoint = _trajectory_end_joint_state_with_sha256(trajectory)

    assert endpoint is not None
    end_state, end_hash = endpoint
    assert end_state == {
        "names": list(ARM_JOINTS),
        "positions": planned_endpoint,
    }
    assert end_hash != _qualification_joint_state_with_sha256(ik_state)[1]

    runtime = object.__new__(_RosRuntime)
    runtime.planning_scene = SimpleNamespace(revision=5)
    runtime._l5_trajectory_cache = {}
    runtime._l5_scene_sha256 = lambda: "scene"
    goal = {
        "qualification_cache_binding_sha256": "a" * 64,
        "qualification_goal_joint_state_sha256": (
            _qualification_joint_state_with_sha256(ik_state)[1]
        ),
        "start_joint_state": ik_state,
        "planning_scene_revision": 5,
        "group_name": "rm_group",
        "link_name": "link_7",
        "requested_tool_pose": {"xyz": [0.4, 0.0, 0.5]},
        "target_pose": {"xyz": [0.4, 0.0, 0.5]},
        "motion_profile": "unloaded",
        "max_velocity_scaling_factor": 0.3,
        "max_acceleration_scaling_factor": 0.2,
    }

    stored_key = runtime._store_l5_trajectory(
        goal=goal, trajectory=trajectory, point_count=1
    )
    public_goal = dict(goal)
    public_goal["qualification_goal_joint_state_sha256"] = end_hash
    public_key = _l5_trajectory_cache_key(
        public_goal, scene_revision=5, scene_sha256="scene"
    )

    assert stored_key == public_key
    assert runtime._l5_trajectory_cache[stored_key][
        "end_joint_state_sha256"
    ] == end_hash


def test_move_initializes_cache_diagnostic_before_any_finish_path() -> None:
    source = inspect.getsource(_RosRuntime.move)

    assert source.index("cache_lookup: dict[str, Any]") < source.index(
        "def finish"
    )


def test_runtime_capability_hash_matches_offline_generator_contract() -> None:
    config = GazeboControlConfig()
    package = config.ros_workspace / "src" / "openeta_rm75_robotiq2f85_sim"

    expected = robot_model_hash(
        urdf=(package / "urdf" / "rm75_robotiq2f85.urdf.xacro").read_bytes(),
        srdf=(package / "config" / "rm75_robotiq2f85.srdf").read_bytes(),
        planning_group=config.move_group,
        tcp=config.arm_tip,
        gripper="robotiq_2f85",
    )

    assert _qualification_robot_model_sha256(config) == expected


def test_pickplace_capability_hash_uses_its_distinct_urdf_and_srdf() -> None:
    assert _qualification_robot_model_sha256(
        NativePickPlaceConfig()
    ) != _qualification_robot_model_sha256(GazeboControlConfig())


def test_ros_loads_content_addressed_capability_map_from_explicit_path(
    monkeypatch, tmp_path
) -> None:
    payload = generate_sparse_capability_map(
        robot_model_sha256="robot",
        joint_lower=[-1.0],
        joint_upper=[1.0],
        forward_kinematics=lambda _joints: (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ),
        jacobian=lambda _joints: [[1.0]],
        sample_count=0,
    )
    path = tmp_path / "map.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("OPENETA_CAPABILITY_MAP_PATH", str(path))

    loaded = _load_qualification_capability_map(
        GazeboControlConfig(),
        map_id=payload["map_id"],
        robot_model_sha256="robot",
    )

    assert loaded["map_id"] == payload["map_id"]


def test_allowed_collision_merge_preserves_srdf_rows_and_adds_target_links() -> None:
    names, rows = _merged_allowed_collision_rows(
        ["base_link", "link_1", "target_object", "stale_gripper_link"],
        [
            [False, True, False, False],
            [True, False, False, False],
            [False, False, False, True],
            [False, False, True, False],
        ],
        {"target_object": ["left_tip", "right_tip"]},
    )
    matrix = {
        (row_name, column_name): rows[row_index][column_index]
        for row_index, row_name in enumerate(names)
        for column_index, column_name in enumerate(names)
    }

    assert matrix["base_link", "link_1"] is True
    assert matrix["link_1", "base_link"] is True
    assert matrix["target_object", "left_tip"] is True
    assert matrix["left_tip", "target_object"] is True
    assert matrix["target_object", "right_tip"] is True
    assert matrix["base_link", "target_object"] is False
    assert matrix["target_object", "stale_gripper_link"] is False


def test_request_local_allowed_collision_merge_is_additive() -> None:
    names, rows = _merged_allowed_collision_rows(
        ["target_object", "work_table"],
        [[False, True], [True, False]],
        {"target_object": ["left_tip"]},
        replace_owned=False,
    )
    matrix = {
        (row_name, column_name): rows[row_index][column_index]
        for row_index, row_name in enumerate(names)
        for column_index, column_name in enumerate(names)
    }

    assert matrix["target_object", "work_table"] is True
    assert matrix["target_object", "left_tip"] is True


def test_state_validity_override_accepts_only_declared_contact_pairs() -> None:
    allowed = {"target_object": ["left_tip", "right_tip"]}

    assert _state_valid_with_allowed_collision_pairs(
        response_valid=False,
        collision_pairs=[["left_tip", "target_object"]],
        allowed_collisions=allowed,
    ) == (True, True)
    assert _state_valid_with_allowed_collision_pairs(
        response_valid=False,
        collision_pairs=[
            ["left_tip", "target_object"],
            ["left_tip", "work_table"],
        ],
        allowed_collisions=allowed,
    ) == (False, False)


def test_moveit_scene_maps_gazebo_world_to_fixed_robot_root_only() -> None:
    assert _moveit_scene_frame("world", base_link="base_link") == "base_link"
    assert _moveit_scene_frame("", base_link="base_link") == "base_link"
    assert (
        _moveit_scene_frame("gripper_mount_link", base_link="base_link")
        == "gripper_mount_link"
    )


def test_l5_joint_only_start_state_preserves_authoritative_attachment() -> None:
    state = SimpleNamespace(
        is_diff=False,
        joint_state=SimpleNamespace(name=[], position=[]),
        # The PlanningScene owns the attached bodies. The joint-only request
        # deliberately does not duplicate or replace that collection.
        attached_collision_objects=[],
    )

    _populate_motion_start_state(
        state,
        {
            "names": list(ARM_JOINTS),
            "positions": [0.1 * index for index in range(len(ARM_JOINTS))],
        },
    )

    assert state.is_diff is True
    assert state.joint_state.name == list(ARM_JOINTS)
    assert state.joint_state.position == pytest.approx(
        [0.1 * index for index in range(len(ARM_JOINTS))]
    )
    assert state.attached_collision_objects == []


def test_l5_joint_only_start_state_rejects_incomplete_vector() -> None:
    state = SimpleNamespace(
        is_diff=False,
        joint_state=SimpleNamespace(name=[], position=[]),
    )

    with pytest.raises(ValueError, match="motion start joint state is invalid"):
        _populate_motion_start_state(
            state,
            {"names": list(ARM_JOINTS), "positions": [0.0]},
        )


class _Clock:
    def now(self):
        return object()


class _Node:
    def get_clock(self):
        return _Clock()


class _Tf:
    def __init__(self, *, fail: bool = False, stamp_s: float | None = None):
        self.fail = fail
        self.stamp_s = stamp_s

    def lookup_transform(self, base, child, stamp):
        assert (base, child) == ("base_link", "gripper_mount_link")
        if self.fail:
            raise LookupError
        stamp = None
        if self.stamp_s is not None:
            sec = int(self.stamp_s)
            stamp = SimpleNamespace(
                sec=sec, nanosec=int(round((self.stamp_s - sec) * 1e9))
            )
        return SimpleNamespace(
            header=SimpleNamespace(stamp=stamp),
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=0.1, y=0.2, z=0.3),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )


def _joint_message(stamp_s: float | None = None):
    header = None
    if stamp_s is not None:
        sec = int(stamp_s)
        header = SimpleNamespace(
            stamp=SimpleNamespace(
                sec=sec, nanosec=int(round((stamp_s - sec) * 1e9))
            )
        )
    return SimpleNamespace(
        name=JOINT_NAMES,
        position=[0.0] * len(JOINT_NAMES),
        velocity=[],
        header=header,
    )


def test_ros_state_source_requires_fresh_complete_joint_state_and_tf() -> None:
    source = RosGazeboStateSource(_Node(), _Tf(), config=GazeboControlConfig(), freshness_s=0.02)
    with pytest.raises(RuntimeError, match="JOINT_STATE_TIMEOUT"):
        source.state()
    source.joint_state_callback(_joint_message())
    state = source.state()
    assert state.end_effector_pose["xyz"] == [0.1, 0.2, 0.3]
    assert state.metadata["joint_names"] == list(JOINT_NAMES)
    time.sleep(0.03)
    with pytest.raises(RuntimeError, match="JOINT_STATE_TIMEOUT"):
        source.state()


def test_ros_state_source_fails_closed_without_tf() -> None:
    source = RosGazeboStateSource(_Node(), _Tf(fail=True), config=GazeboControlConfig())
    source.joint_state_callback(_joint_message())
    with pytest.raises(RuntimeError, match="TF_TIMEOUT"):
        source.state()


def test_ros_state_source_rejects_cached_tf_from_before_action_boundary() -> None:
    tf = _Tf(stamp_s=9.5)
    source = RosGazeboStateSource(_Node(), tf, config=GazeboControlConfig())
    source.clear(min_ros_timestamp_s=10.0)
    source.joint_state_callback(_joint_message(10.1))

    with pytest.raises(RuntimeError, match="POST_ACTION_STATE_NOT_FRESH"):
        source.state()

    tf.stamp_s = 10.1
    state = source.state()
    assert state.metadata["joint_state_timestamp_s"] == pytest.approx(10.1)
    assert state.metadata["tf_timestamp_s"] == pytest.approx(10.1)


def test_recovery_ros_messages_preserve_all_seven_measured_joint_positions() -> None:
    validity = SimpleNamespace(
        group_name="",
        robot_state=SimpleNamespace(
            is_diff=False,
            joint_state=SimpleNamespace(name=[], position=[]),
        ),
    )
    candidate = [0.1, 0.2, 3.105, 0.4, 0.5, 0.6, 0.7]
    _populate_state_validity_request(validity, candidate, group_name="rm_group")

    goal = SimpleNamespace(
        trajectory=SimpleNamespace(joint_names=[], points=[])
    )
    point = SimpleNamespace(positions=[], time_from_start=None)
    duration = SimpleNamespace(sec=1, nanosec=0)
    _populate_recovery_trajectory_goal(goal, point, duration, candidate)

    assert validity.group_name == "rm_group"
    assert validity.robot_state.is_diff is True
    assert validity.robot_state.joint_state.name == [f"joint_{i}" for i in range(1, 8)]
    assert validity.robot_state.joint_state.position == candidate
    assert goal.trajectory.joint_names == validity.robot_state.joint_state.name
    assert goal.trajectory.points[0].positions == candidate
    assert goal.trajectory.points[0].time_from_start is duration


def test_qualification_open_state_adds_active_gripper_joint_to_moveit_diff() -> None:
    captured = []

    class Request:
        def __init__(self):
            self.group_name = ""
            self.robot_state = SimpleNamespace(
                is_diff=False,
                joint_state=SimpleNamespace(name=[], position=[]),
                attached_collision_objects=[],
            )

    response = SimpleNamespace(valid=True, contacts=[])

    class Client:
        def call_async(self, request):
            captured.append(request)
            return response

    config = GazeboControlConfig()
    runtime = _RosRuntime(
        state_validity_service_type=SimpleNamespace(Request=Request),
        state_validity_client=Client(),
        config=config,
        planning_scene=SimpleNamespace(world_ids={"work_table", "target_object"}),
    )
    runtime._await = lambda future, timeout: future

    result = runtime.qualification_state_validity(
        {
            "names": [f"joint_{index}" for index in range(1, 8)],
            "positions": [0.0] * 7,
            "qualification_gripper_state": "open",
        }
    )

    assert result == {
        "valid": True,
        "collision_pairs": [],
        "contact_collision_override": False,
    }
    assert captured[0].robot_state.joint_state.name == [
        *[f"joint_{index}" for index in range(1, 8)],
        config.active_joint,
    ]
    assert captured[0].robot_state.joint_state.position == [
        *([0.0] * 7),
        config.gripper_position(1),
    ]


def test_post_detach_open_state_keeps_released_object_as_collision_probe() -> None:
    captured = []

    class Request:
        def __init__(self):
            self.group_name = ""
            self.robot_state = SimpleNamespace(
                is_diff=False,
                joint_state=SimpleNamespace(name=[], position=[]),
                attached_collision_objects=[],
            )

    class AttachedCollisionObject:
        def __init__(self):
            self.link_name = ""
            self.touch_links = []
            self.object = None

    class Contact:
        contact_body_1 = "robotiq_85_base_link"
        contact_body_2 = "target_object__openeta_detached_probe"

    class Client:
        def call_async(self, request):
            captured.append(request)
            return SimpleNamespace(valid=False, contacts=[Contact()])

    config = GazeboControlConfig()
    runtime = _RosRuntime(
        state_validity_service_type=SimpleNamespace(Request=Request),
        state_validity_client=Client(),
        attached_collision_object_type=AttachedCollisionObject,
        config=config,
        planning_scene=SimpleNamespace(
            world_ids={"work_table"}, attached_ids={"target_object"}
        ),
    )
    runtime._await = lambda future, timeout: future
    removed = AttachedCollisionObject()
    removed.object = SimpleNamespace(id="target_object", operation="REMOVE")
    runtime._qualification_scene_diff_message = lambda _diff: SimpleNamespace(
        robot_state=SimpleNamespace(attached_collision_objects=[removed])
    )
    runtime._collision_object_from_spec = lambda spec: SimpleNamespace(
        id=spec["id"], frame=spec["frame"], operation="ADD"
    )

    result = runtime.qualification_state_validity(
        {
            "names": list(ARM_JOINTS),
            "positions": [0.0] * len(ARM_JOINTS),
            "qualification_gripper_state": "open",
            "qualification_scene_diff": {
                "remove_attached_ids": ["target_object"],
                "world_objects": [
                    {
                        "id": "target_object",
                        "frame": config.base_link,
                        "link_name": config.mount_child,
                    }
                ],
                "detached_collision_probe_objects": [
                    {
                        "id": "target_object",
                        "frame": config.base_link,
                        "link_name": config.mount_child,
                    }
                ],
            },
        }
    )

    attached = captured[0].robot_state.attached_collision_objects
    assert len(attached) == 2
    assert attached[0].object.operation == "REMOVE"
    assert attached[1].object.id == "target_object__openeta_detached_probe"
    assert attached[1].link_name == config.mount_child
    assert attached[1].touch_links == []
    assert result["valid"] is False
    assert result["collision_pairs"] == [
        [
            "robotiq_85_base_link",
            "target_object__openeta_detached_probe",
        ]
    ]
    assert result["qualification_detached_collision_probe_count"] == 1


def test_qualification_state_validity_stops_close_sweep_on_static_collision() -> None:
    captured = []

    class Request:
        def __init__(self):
            self.group_name = ""
            self.robot_state = SimpleNamespace(
                is_diff=False,
                joint_state=SimpleNamespace(name=[], position=[]),
                attached_collision_objects=[],
            )

    class Contact:
        contact_body_1 = "robotiq_85_left_finger_tip_link"
        contact_body_2 = "work_table"

    class Client:
        def call_async(self, request):
            captured.append(request)
            angle = request.robot_state.joint_state.position[-1]
            return SimpleNamespace(
                valid=not math.isclose(angle, 0.05),
                contacts=([Contact()] if math.isclose(angle, 0.05) else []),
            )

    config = GazeboControlConfig()
    runtime = _RosRuntime(
        state_validity_service_type=SimpleNamespace(Request=Request),
        state_validity_client=Client(),
        config=config,
        planning_scene=SimpleNamespace(world_ids={"work_table", "target_object"}),
    )
    runtime._await = lambda future, timeout: future

    result = runtime.qualification_state_validity(
        {
            "names": [f"joint_{index}" for index in range(1, 8)],
            "positions": [0.0] * 7,
            "qualification_gripper_state": "closing_sweep",
            "qualification_allowed_collisions": {
                "target_object": ["robotiq_85_left_finger_tip_link"]
            },
        }
    )

    assert result["valid"] is False
    assert result["collision_pairs"] == [
        ["robotiq_85_left_finger_tip_link", "work_table"]
    ]
    assert [
        request.robot_state.joint_state.position[-1] for request in captured
    ] == [0.05]
    assert result["qualification_gripper_sweep_checks"][0]["sample"] == (
        "near_open"
    )
    assert (
        result["qualification_seed_independent_static_collision"] is True
    )


def test_close_sweep_rejects_one_sided_target_contact_geometry() -> None:
    class Request:
        def __init__(self):
            self.group_name = ""
            self.robot_state = SimpleNamespace(
                is_diff=False,
                joint_state=SimpleNamespace(name=[], position=[]),
                attached_collision_objects=[],
            )

    class Contact:
        contact_body_1 = "robotiq_85_left_finger_tip_link"
        contact_body_2 = "target_object"

    class Client:
        def call_async(self, _request):
            return SimpleNamespace(valid=False, contacts=[Contact()])

    config = GazeboControlConfig()
    runtime = _RosRuntime(
        state_validity_service_type=SimpleNamespace(Request=Request),
        state_validity_client=Client(),
        config=config,
        planning_scene=SimpleNamespace(world_ids={"work_table", "target_object"}),
    )
    runtime._await = lambda future, timeout: future

    result = runtime.qualification_state_validity(
        {
            "names": [f"joint_{index}" for index in range(1, 8)],
            "positions": [0.0] * 7,
            "qualification_gripper_state": "closing_sweep",
            "qualification_allowed_collisions": {
                "target_object": [
                    "robotiq_85_left_finger_tip_link",
                    "robotiq_85_right_finger_tip_link",
                ]
            },
        }
    )

    assert result["valid"] is False
    assert result["qualification_bilateral_target_contact_required"] is True
    assert result["qualification_bilateral_target_contact_predicted"] is False
    assert result["qualification_seed_independent_contact_geometry_failure"] is True
    assert result["reason"] == "qualification_bilateral_target_contact_not_predicted"
    assert all(
        check["target_contact_links"]
        == ["robotiq_85_left_finger_tip_link"]
        for check in result["qualification_gripper_sweep_checks"]
    )


def test_close_sweep_accepts_simultaneous_bilateral_target_contact_geometry() -> None:
    class Request:
        def __init__(self):
            self.group_name = ""
            self.robot_state = SimpleNamespace(
                is_diff=False,
                joint_state=SimpleNamespace(name=[], position=[]),
                attached_collision_objects=[],
            )

    class Contact:
        def __init__(self, link):
            self.contact_body_1 = link
            self.contact_body_2 = "target_object"

    class Client:
        def call_async(self, request):
            angle = request.robot_state.joint_state.position[-1]
            links = ["robotiq_85_left_finger_tip_link"]
            if angle >= 0.4:
                links.append("robotiq_85_right_finger_tip_link")
            return SimpleNamespace(
                valid=False,
                contacts=[Contact(link) for link in links],
            )

    config = GazeboControlConfig()
    runtime = _RosRuntime(
        state_validity_service_type=SimpleNamespace(Request=Request),
        state_validity_client=Client(),
        config=config,
        planning_scene=SimpleNamespace(world_ids={"work_table", "target_object"}),
    )
    runtime._await = lambda future, timeout: future

    result = runtime.qualification_state_validity(
        {
            "names": [f"joint_{index}" for index in range(1, 8)],
            "positions": [0.0] * 7,
            "qualification_gripper_state": "closing_sweep",
            "qualification_allowed_collisions": {
                "target_object": [
                    "robotiq_85_left_finger_tip_link",
                    "robotiq_85_right_finger_tip_link",
                ]
            },
        }
    )

    assert result["valid"] is True
    assert result["qualification_bilateral_target_contact_predicted"] is True
    assert result["qualification_seed_independent_contact_geometry_failure"] is False
    assert any(
        check["bilateral_target_contact"] is True
        for check in result["qualification_gripper_sweep_checks"]
    )


class _ReadyActionClient:
    def wait_for_server(self, **kwargs):
        return True

    def server_is_ready(self):
        return True


class _ReadyServiceClient:
    def __init__(self, response):
        self.response = response

    def wait_for_service(self, **kwargs):
        return True

    def call_async(self, request):
        return self.response


class _RequestType:
    class Request:
        def __init__(self):
            self.names = []


def _readiness_runtime(*, command_limits: bool):
    action = _ReadyActionClient()
    controllers = SimpleNamespace(
        controller=[
            SimpleNamespace(name=name, state="active")
            for name in (
                "joint_state_broadcaster",
                "rm_group_controller",
                "parallel_gripper_controller",
            )
        ]
    )
    runtime = _RosRuntime(
        move_client=action,
        gripper_client=action,
        trajectory_client=action,
        controller_list_client=_ReadyServiceClient(controllers),
        controller_parameter_client=_ReadyServiceClient(
            SimpleNamespace(
                values=[SimpleNamespace(bool_value=command_limits)]
            )
        ),
        state_validity_client=_ReadyServiceClient(SimpleNamespace()),
        controller_service_type=_RequestType,
        controller_parameter_service_type=_RequestType,
        state_source=SimpleNamespace(wait_fresh=lambda timeout_s: _arm_state(0.0)),
    )
    runtime._await = lambda future, timeout_s: future
    return runtime


def test_runtime_readiness_requires_limits_trajectory_and_state_validity() -> None:
    _readiness_runtime(command_limits=True).wait_ready(0.1)

    with pytest.raises(RuntimeError, match="ROS_NOT_READY"):
        _readiness_runtime(command_limits=False).wait_ready(0.001)


def test_runtime_readiness_retries_a_transient_service_response_timeout() -> None:
    runtime = _readiness_runtime(command_limits=True)
    parameter_response = runtime.controller_parameter_client.response
    parameter_attempts = 0

    def await_response(future, _timeout_s):
        nonlocal parameter_attempts
        if future is parameter_response:
            parameter_attempts += 1
            if parameter_attempts == 1:
                raise TimeoutError
        return future

    runtime._await = await_response

    runtime.wait_ready(0.1)

    assert parameter_attempts == 2


def _arm_state(joint_3: float):
    positions = [0.0] * len(JOINT_NAMES)
    positions[2] = joint_3
    state = robot_state_from_sources(
        {
            "name": JOINT_NAMES,
            "position": positions,
            "velocity": [0.0] * len(JOINT_NAMES),
        },
        {
            "base_link->gripper_mount_link": {
                "xyz": [0.0, 0.0, 0.5],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
    )
    state.metadata.update(
        {
            "joint_state_timestamp_s": 7.0,
            "joint_state_received_monotonic_s": time.monotonic(),
        }
    )
    return state


class _ValidityType:
    class Request:
        def __init__(self):
            self.group_name = ""
            self.robot_state = SimpleNamespace(
                is_diff=False,
                joint_state=SimpleNamespace(name=[], position=[]),
            )


class _FollowType:
    class Result:
        SUCCESSFUL = 0

    class Goal:
        def __init__(self):
            self.trajectory = SimpleNamespace(joint_names=[], points=[])


class _Client:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call_async(self, request):
        self.calls.append(request)
        return self.response

    def send_goal_async(self, goal):
        self.calls.append(goal)
        return self.response


class _StateSource:
    def __init__(self, post_state):
        self.post_state = post_state
        self.clear_calls = 0
        self.clear_arguments = []

    def clear(self, **kwargs):
        self.clear_calls += 1
        self.clear_arguments.append(dict(kwargs))

    def wait_fresh(self, timeout_s):
        if isinstance(self.post_state, Exception):
            raise self.post_state
        return self.post_state


def _recovery_runtime(
    *, valid=True, accepted=True, result_code=0, post_joint_3=3.105
):
    wrapped = SimpleNamespace(result=SimpleNamespace(error_code=result_code))
    handle = SimpleNamespace(
        accepted=accepted,
        get_result_async=lambda: wrapped,
        cancel_goal_async=lambda: object(),
    )
    validity_client = _Client(SimpleNamespace(valid=valid))
    trajectory_client = _Client(handle)
    runtime = _RosRuntime(
        state_validity_service_type=_ValidityType,
        state_validity_client=validity_client,
        follow_trajectory_action_type=_FollowType,
        trajectory_client=trajectory_client,
        trajectory_point_type=lambda: SimpleNamespace(
            positions=[], time_from_start=None
        ),
        duration_type=lambda **values: SimpleNamespace(**values),
        state_source=_StateSource(_arm_state(post_joint_3)),
        config=GazeboControlConfig(),
    )
    runtime._await = lambda future, timeout_s: future
    runtime.ros_time_s = lambda: 100.0
    return runtime, validity_client, trajectory_client


def test_ros_recovery_validates_candidate_then_executes_physical_inset() -> None:
    runtime, validity_client, trajectory_client = _recovery_runtime()

    result = runtime.recover_start_state(_arm_state(3.106 + 4.5e-13), 5.0)

    assert result["ok"] is True
    evidence = result["start_state_recovery"]
    assert evidence["status"] == "RECOVERED"
    assert evidence["joints"][0]["recovery_target_rad"] == pytest.approx(3.105)
    assert evidence["trajectory_result_code"] == 0
    assert evidence["post_joint_state_timestamp_s"] == 7.0
    assert len(validity_client.calls) == 1
    assert len(trajectory_client.calls) == 1
    assert runtime.state_source.clear_arguments == [
        {"min_ros_timestamp_s": 100.0}
    ]
    assert trajectory_client.calls[0].trajectory.points[0].positions[2] == pytest.approx(
        3.105
    )


def test_ros_recovery_collision_rejection_never_sends_trajectory() -> None:
    runtime, validity_client, trajectory_client = _recovery_runtime(valid=False)

    result = runtime.recover_start_state(_arm_state(3.106 + 4.5e-13), 5.0)

    assert result["ok"] is False
    assert result["error_code"] == "START_STATE_RECOVERY_FAILED"
    assert result["start_state_recovery"]["reason_code"] == (
        "RECOVERY_STATE_INVALID_OR_IN_COLLISION"
    )
    assert len(validity_client.calls) == 1
    assert trajectory_client.calls == []


def test_ros_recovery_accepts_a_post_state_within_numeric_tolerance() -> None:
    runtime, _, _ = _recovery_runtime(post_joint_3=3.106 + 4.5e-13)

    result = runtime.recover_start_state(_arm_state(3.106 + 4.5e-13), 5.0)

    assert result["ok"] is True
    assert result["start_state_recovery"]["status"] == "RECOVERED"


def test_ros_recovery_trajectory_rejection_and_failure_are_auditable() -> None:
    rejected, _, _ = _recovery_runtime(accepted=False)
    failed, _, _ = _recovery_runtime(result_code=-5)

    rejected_result = rejected.recover_start_state(
        _arm_state(3.106 + 4.5e-13), 5.0
    )
    failed_result = failed.recover_start_state(
        _arm_state(3.106 + 4.5e-13), 5.0
    )

    assert rejected_result["start_state_recovery"]["reason_code"] == (
        "RECOVERY_TRAJECTORY_REJECTED"
    )
    assert failed_result["start_state_recovery"]["reason_code"] == (
        "RECOVERY_TRAJECTORY_FAILED"
    )
    assert failed_result["start_state_recovery"]["trajectory_result_code"] == -5


def test_ros_recovery_timeout_is_unknown_and_never_confirms_user_motion() -> None:
    runtime, _, trajectory_client = _recovery_runtime()
    timeout_token = object()
    trajectory_client.response.get_result_async = lambda: timeout_token
    runtime._await = lambda future, timeout_s: (
        (_ for _ in ()).throw(TimeoutError())
        if future is timeout_token
        else future
    )

    result = runtime.recover_start_state(_arm_state(3.106 + 4.5e-13), 5.0)

    assert result["error_code"] == "MOTION_OUTCOME_UNKNOWN"
    assert result["reconciliation_required"] is True
    assert result["start_state_recovery"]["status"] == "UNKNOWN"
    assert result["start_state_recovery"]["reason_code"] == (
        "RECOVERY_TRAJECTORY_TIMEOUT_UNCONFIRMED"
    )


def test_ros_recovery_requires_fresh_post_action_joint_state() -> None:
    runtime, _, _ = _recovery_runtime()
    runtime.state_source.post_state = RuntimeError("JOINT_STATE_TIMEOUT")

    result = runtime.recover_start_state(_arm_state(3.106 + 4.5e-13), 5.0)

    assert result["error_code"] == "START_STATE_RECOVERY_FAILED"
    assert result["start_state_recovery"]["reason_code"] == (
        "POST_RECOVERY_JOINT_STATE_MISSING"
    )
