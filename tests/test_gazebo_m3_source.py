from __future__ import annotations

from types import SimpleNamespace

import pytest

from extensions.gazebo.ros_physics import (
    contact_object_ids,
    extend_allowed_collision_matrix,
    message_timestamp_s,
    parse_odometry_message,
)


def _stamp(value: float):
    seconds = int(value)
    return SimpleNamespace(sec=seconds, nanosec=int((value - seconds) * 1e9))


def _entity(name: str):
    return SimpleNamespace(name=name)


def test_contact_parser_preserves_exact_gazebo_collision_identity() -> None:
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(10.25)),
        contacts=[
            SimpleNamespace(
                collision1=_entity(
                    "rm75_robotiq_2f85_pickplace_sim_v1::robotiq_85_left_finger_tip_link::collision"
                ),
                collision2=_entity("m3_target::target_link::target_collision"),
            )
        ],
    )
    ids, complete = contact_object_ids(
        message,
        known_ids=("rm75_robotiq_2f85_pickplace_sim_v1", "m3_target", "m3_distractor"),
        sensor_owner_id="rm75_robotiq_2f85_pickplace_sim_v1",
    )
    assert ids == ("m3_target",)
    assert complete is True
    assert message_timestamp_s(message) == pytest.approx(10.25)


def test_contact_parser_marks_missing_collision_identity_incomplete() -> None:
    message = SimpleNamespace(
        contacts=[SimpleNamespace(collision1=_entity("m3_target::link::collision"), collision2=_entity(""))]
    )
    ids, complete = contact_object_ids(
        message, known_ids=("m3_target", "m3_table"), sensor_owner_id="m3_target"
    )
    assert ids == ()
    assert complete is False


def test_contact_parser_marks_unknown_partner_identity_incomplete() -> None:
    message = SimpleNamespace(
        contacts=[
            SimpleNamespace(
                collision1=_entity("rm75::finger::collision"),
                collision2=_entity("unregistered_object::link::collision"),
            )
        ]
    )
    ids, complete = contact_object_ids(
        message, known_ids=("rm75", "m3_target"), sensor_owner_id="rm75"
    )
    assert ids == ()
    assert complete is False


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
