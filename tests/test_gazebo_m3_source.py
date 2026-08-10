from __future__ import annotations

from types import SimpleNamespace
import threading

import pytest

from extensions.gazebo.m3 import M3Config, M3PlanningSceneModel, ObjectState, Pose
from extensions.gazebo.ros_physics import (
    RosM3PhysicsSource,
    RosM3PlanningScene,
    extend_allowed_collision_matrix,
    message_timestamp_s,
    parse_odometry_message,
)


def _stamp(value: float):
    seconds = int(value)
    return SimpleNamespace(sec=seconds, nanosec=int((value - seconds) * 1e9))


def test_allowed_collision_extension_preserves_srdf_and_only_adds_fingertip_target_pairs() -> None:
    names, rows = extend_allowed_collision_matrix(
        ("base_link", "link_1", "arm_tip"),
        (
            (False, True, False),
            (True, False, False),
            (False, False, False),
        ),
        object_id="m3_target",
        touch_links=("left_tip", "right_tip"),
    )
    indexes = {name: index for index, name in enumerate(names)}
    assert rows[indexes["base_link"]][indexes["link_1"]] is True
    assert rows[indexes["m3_target"]][indexes["left_tip"]] is True
    assert rows[indexes["right_tip"]][indexes["m3_target"]] is True
    assert rows[indexes["arm_tip"]][indexes["m3_target"]] is False
    assert rows[indexes["left_tip"]][indexes["right_tip"]] is False


def test_odometry_parser_converts_child_frame_twist_to_world_units() -> None:
    half = 2**-0.5
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(12.0), frame_id="world"),
        child_frame_id="m3_target",
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.28, y=-0.1, z=0.43),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=half, w=half),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=1.0, y=0.0, z=0.0),
                angular=SimpleNamespace(x=0.0, y=0.0, z=1.0),
            )
        ),
    )
    state = parse_odometry_message(
        message, object_id="m3_target", label="target block", role="target"
    )
    assert state.pose.position == (0.28, -0.1, 0.43)
    assert state.linear_velocity == pytest.approx((0.0, 1.0, 0.0))
    assert state.angular_velocity == pytest.approx((0.0, 0.0, 1.0))
    assert state.timestamp_s == 12.0


def test_odometry_parser_rejects_wrong_or_missing_identity_and_stamp() -> None:
    message = SimpleNamespace(header=SimpleNamespace(stamp=_stamp(0.0)), child_frame_id="other")
    with pytest.raises(ValueError, match="identity"):
        parse_odometry_message(
            message, object_id="m3_target", label="target block", role="target"
        )


def test_physics_source_builds_fresh_snapshot_without_contact_streams() -> None:
    source = object.__new__(RosM3PhysicsSource)
    source.config = M3Config()
    source._lock = threading.Lock()

    def state(object_id: str, xyz: tuple[float, float, float]) -> ObjectState:
        return ObjectState(
            object_id=object_id,
            name=object_id,
            label=object_id,
            role="target" if object_id == "m3_target" else "distractor",
            pose=Pose(xyz, (0.0, 0.0, 0.0, 1.0)),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            support=None,
            timestamp_s=10.0,
        )

    source._odometry = {
        "m3_target": state("m3_target", (0.28, -0.10, 0.43)),
        "m3_distractor": state("m3_distractor", (0.28, 0.12, 0.44)),
    }
    snapshot = source._try_snapshot(
        robot={
            "metadata": {"joint_state_timestamp_s": 10.0, "tf_timestamp_s": 10.0},
            "end_effector_pose": {"xyz": [0.28, -0.10, 0.55], "quat_xyzw": [0, 0, 0, 1]},
            "gripper_state": {"aperture_m": 0.04},
        },
        camera_timestamp_s=10.0,
        gripper_stalled=True,
        gripper_reached_goal=False,
    )

    assert snapshot is not None
    assert set(snapshot.stream_timestamps()) == {
        "joint_state", "tf", "rgb", "depth", "odometry_target", "odometry_distractor"
    }
    assert [item.support for item in snapshot.objects] == ["m3_table", "m3_table"]


def test_new_planning_scene_clear_is_idempotent() -> None:
    scene = object.__new__(RosM3PlanningScene)
    scene.model = M3PlanningSceneModel()
    calls = []
    scene._apply_commands = lambda *args, **kwargs: calls.append((args, kwargs))

    scene.clear()

    assert calls == []
