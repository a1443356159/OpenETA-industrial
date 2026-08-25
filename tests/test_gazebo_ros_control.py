from __future__ import annotations

import json
from types import SimpleNamespace
import time

import pytest

from extensions.gazebo.robot_control import JOINT_NAMES, GazeboControlConfig, robot_state_from_sources
from extensions.gazebo.native_grasp import NativePickPlaceConfig
from extensions.gazebo.ros_control import (
    RosGazeboStateSource,
    _RosRuntime,
    _configured_qualification_solver_profile,
    _qualification_ik_response_timeout_s,
    _moveit_scene_frame,
    _merged_allowed_collision_rows,
    _load_qualification_capability_map,
    _populate_recovery_trajectory_goal,
    _populate_state_validity_request,
    _qualification_robot_model_sha256,
)
from agent.runtime.capability_map import generate_sparse_capability_map, robot_model_hash


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
    assert _qualification_ik_response_timeout_s(0.05) == pytest.approx(2.0)
    assert _qualification_ik_response_timeout_s(0.05) > 8 * 0.05


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
        ["base_link", "link_1"],
        [[False, True], [True, False]],
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


def test_moveit_scene_maps_gazebo_world_to_fixed_robot_root_only() -> None:
    assert _moveit_scene_frame("world", base_link="base_link") == "base_link"
    assert _moveit_scene_frame("", base_link="base_link") == "base_link"
    assert (
        _moveit_scene_frame("gripper_mount_link", base_link="base_link")
        == "gripper_mount_link"
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

    def clear(self, **_kwargs):
        self.clear_calls += 1

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
