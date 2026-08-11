from __future__ import annotations

from dataclasses import replace
import math

import pytest

from extensions.gazebo.m3 import (
    M3Config,
    M3PlanningSceneModel,
    M3Verifier,
    ObjectState,
    PhysicsSnapshot,
    Pose,
    ReasonCode,
    Verdict,
    namespaced_entity_id,
    oriented_box_xy_half_extents,
    relative_pose,
)


STREAMS = (
    "joint_state",
    "tf",
    "rgb",
    "depth",
    "odometry_target",
    "odometry_distractor",
)


def _object(
    object_id: str,
    *,
    xyz: tuple[float, float, float],
    stamp: float,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular: tuple[float, float, float] = (0.0, 0.0, 0.0),
    support: str | None = None,
) -> ObjectState:
    return ObjectState(
        object_id=object_id,
        name=object_id,
        label="target block" if object_id == "m3_target" else "distractor cylinder",
        role="target" if object_id == "m3_target" else "distractor",
        pose=Pose(xyz, (0.0, 0.0, 0.0, 1.0)),
        linear_velocity=velocity,
        angular_velocity=angular,
        support=support,
        timestamp_s=stamp,
    )


def _snapshot(
    stamp: float,
    *,
    target_xyz: tuple[float, float, float] = (0.28, -0.10, 0.43),
    eef_xyz: tuple[float, float, float] = (0.28, -0.10, 0.55),
    distractor_xyz: tuple[float, float, float] = (0.28, 0.12, 0.44),
    aperture: float = 0.04,
    support: str | None = "m3_table",
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular: tuple[float, float, float] = (0.0, 0.0, 0.0),
    stalled: bool | None = None,
    reached: bool | None = None,
    stream_stamp: float | None = None,
    streams: tuple[str, ...] = STREAMS,
) -> PhysicsSnapshot:
    stream_stamp = stamp if stream_stamp is None else stream_stamp
    return PhysicsSnapshot(
        timestamp_s=stamp,
        received_monotonic_s=100.0,
        eef_pose=Pose(eef_xyz, (0.0, 0.0, 0.0, 1.0)),
        aperture_m=aperture,
        objects=(
            _object(
                "m3_target",
                xyz=target_xyz,
                stamp=stamp,
                velocity=velocity,
                angular=angular,
                support=support,
            ),
            _object("m3_distractor", xyz=distractor_xyz, stamp=stamp),
        ),
        stream_timestamps_s=tuple((name, stream_stamp) for name in streams),
        gripper_stalled=stalled,
        gripper_reached_goal=reached,
    )


def _candidate(verifier: M3Verifier, stamp: float = 10.0):
    record = verifier.verify(
        _snapshot(
            stamp,
            stalled=True,
            reached=False,
        ),
        action_type="gripper_close",
        action_timestamp_s=stamp - 1.0,
    )
    assert (record.verdict, record.reason_code) == (
        Verdict.UNKNOWN,
        ReasonCode.LIFT_REQUIRED,
    )
    return record


def _held(verifier: M3Verifier, stamp: float = 10.0):
    _candidate(verifier, stamp)
    record = verifier.verify(
        _snapshot(
            stamp + 1,
            target_xyz=(0.28, -0.10, 0.50),
            eef_xyz=(0.28, -0.10, 0.62),
            support=None,
        ),
        action_type="move_to",
        action_timestamp_s=stamp + 0.5,
    )
    assert (record.verdict, record.reason_code) == (Verdict.PASS, ReasonCode.TARGET_HELD)
    return record


def test_close_candidate_never_claims_final_grasp_success() -> None:
    verifier = M3Verifier()
    record = _candidate(verifier)
    assert record.object_detection == "object_detected_closing"
    assert record.grasp_confirmed is False
    assert record.evidence["gripper_stalled"] is True


def test_empty_grasp_and_wrong_object_are_structured_failures() -> None:
    empty = M3Verifier().verify(
        _snapshot(10.0, aperture=0.005, support="m3_table", stalled=False, reached=True),
        action_type="gripper_close",
        action_timestamp_s=9.0,
    )
    assert (empty.verdict, empty.reason_code, empty.object_detection) == (
        Verdict.FAIL,
        ReasonCode.EMPTY_GRASP,
        "at_position_no_object",
    )
    verifier = M3Verifier()
    candidate = verifier.verify(
        _snapshot(
            10.0,
            eef_xyz=(0.28, 0.12, 0.56),
            stalled=True,
            reached=False,
        ),
        action_type="gripper_close",
        action_timestamp_s=9.0,
    )
    assert candidate.reason_code == ReasonCode.LIFT_REQUIRED
    wrong = verifier.verify(
        _snapshot(
            11.0,
            eef_xyz=(0.28, 0.12, 0.64),
            distractor_xyz=(0.28, 0.12, 0.52),
        ),
        action_type="move_to",
        action_timestamp_s=10.5,
    )
    assert (wrong.verdict, wrong.reason_code) == (Verdict.FAIL, ReasonCode.WRONG_OBJECT)


def test_lift_reports_identity_incomplete_when_both_objects_comove() -> None:
    verifier = M3Verifier()
    _candidate(verifier)
    ambiguous = verifier.verify(
        _snapshot(
            11.0,
            target_xyz=(0.28, -0.10, 0.51),
            distractor_xyz=(0.28, 0.12, 0.52),
            eef_xyz=(0.28, -0.10, 0.63),
        ),
        action_type="move_to",
        action_timestamp_s=10.5,
    )
    assert (ambiguous.verdict, ambiguous.reason_code) == (
        Verdict.UNKNOWN,
        ReasonCode.IDENTITY_INCOMPLETE,
    )


def test_close_uses_robotiq_result_and_fails_closed_on_other_states() -> None:
    missing_stall = M3Verifier().verify(
        _snapshot(10.0),
        action_type="gripper_close",
        action_timestamp_s=9.0,
    )
    assert missing_stall.reason_code == ReasonCode.STALL_STATUS_MISSING
    contradictory = M3Verifier().verify(
        _snapshot(10.0, aperture=0.005, stalled=True, reached=False),
        action_type="gripper_close",
        action_timestamp_s=9.0,
    )
    assert (contradictory.verdict, contradictory.reason_code) == (
        Verdict.UNKNOWN,
        ReasonCode.STALL_STATUS_MISSING,
    )
    out_of_bounds = M3Verifier().verify(
        _snapshot(10.0, aperture=0.09, stalled=True, reached=False),
        action_type="gripper_close",
        action_timestamp_s=9.0,
    )
    assert (out_of_bounds.verdict, out_of_bounds.reason_code) == (
        Verdict.UNKNOWN,
        ReasonCode.STALL_STATUS_MISSING,
    )


def test_lift_requires_height_support_release_and_relative_pose() -> None:
    verifier = M3Verifier()
    _candidate(verifier)
    not_lifted = verifier.verify(
        _snapshot(
            11.0,
            target_xyz=(0.28, -0.10, 0.48),
            eef_xyz=(0.28, -0.10, 0.60),
        ),
        action_type="move_to",
        action_timestamp_s=10.5,
    )
    assert not_lifted.reason_code == ReasonCode.TARGET_NOT_LIFTED

    held = verifier.verify(
        _snapshot(
            12.0,
            target_xyz=(0.28, -0.10, 0.50),
            eef_xyz=(0.28, -0.10, 0.62),
            support=None,
        ),
        action_type="move_to",
        action_timestamp_s=11.5,
    )
    assert held.reason_code == ReasonCode.TARGET_HELD
    assert held.grasp_confirmed is True


def test_relative_drift_fails_closed_after_grasp_is_proven() -> None:
    drift = M3Verifier()
    _held(drift)
    drifted = drift.verify(
        _snapshot(
            12.0,
            target_xyz=(0.28, -0.10, 0.50),
            eef_xyz=(0.28, -0.10, 0.64),
            support=None,
        ),
        action_type="move_to",
        action_timestamp_s=11.5,
    )
    assert (drifted.verdict, drifted.reason_code) == (
        Verdict.FAIL,
        ReasonCode.RELATIVE_POSE_DRIFT,
    )


def test_open_in_air_detects_physical_drop() -> None:
    verifier = M3Verifier()
    _held(verifier)
    dropped = verifier.verify(
        _snapshot(
            12.0,
            target_xyz=(0.28, -0.10, 0.48),
            eef_xyz=(0.28, -0.10, 0.62),
            aperture=0.085,
            support=None,
            velocity=(0.0, 0.0, -0.20),
            stalled=False,
            reached=True,
        ),
        action_type="gripper_open",
        action_timestamp_s=11.5,
    )
    assert (dropped.verdict, dropped.reason_code) == (
        Verdict.FAIL,
        ReasonCode.OBJECT_DROPPED,
    )


def test_place_requires_full_bbox_geometric_support_and_one_second_settle() -> None:
    verifier = M3Verifier()
    _held(verifier)
    first = verifier.verify(
        _snapshot(
            12.0,
            target_xyz=(0.48, -0.10, 0.43),
            eef_xyz=(0.48, -0.10, 0.55),
            aperture=0.085,
            support="m3_table",
            stalled=False,
            reached=True,
        ),
        action_type="gripper_open",
        action_timestamp_s=11.5,
    )
    assert first.reason_code == ReasonCode.NOT_SETTLED
    placed = verifier.verify(
        _snapshot(
            13.01,
            target_xyz=(0.48, -0.10, 0.43),
            eef_xyz=(0.48, -0.10, 0.55),
            aperture=0.085,
            support="m3_table",
        )
    )
    assert (placed.verdict, placed.reason_code) == (
        Verdict.PASS,
        ReasonCode.TARGET_PLACED,
    )


def test_outside_destination_and_unsettled_are_not_success() -> None:
    verifier = M3Verifier()
    _held(verifier)
    first = verifier.verify(
        _snapshot(
            12.0,
            target_xyz=(0.35, -0.10, 0.43),
            eef_xyz=(0.35, -0.10, 0.55),
            aperture=0.085,
            support="m3_table",
        ),
        action_type="gripper_open",
        action_timestamp_s=11.5,
    )
    assert first.reason_code == ReasonCode.NOT_SETTLED
    outside = verifier.verify(
        _snapshot(
            13.1,
            target_xyz=(0.35, -0.10, 0.43),
            eef_xyz=(0.35, -0.10, 0.55),
            aperture=0.085,
            support="m3_table",
        )
    )
    assert (outside.verdict, outside.reason_code) == (
        Verdict.FAIL,
        ReasonCode.OUTSIDE_DESTINATION,
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"stream_stamp": 8.0}, ReasonCode.DATA_STALE),
        ({"streams": ("joint_state", "tf", "rgb", "depth", "odometry_target")}, ReasonCode.DATA_MISSING),
    ],
)
def test_stale_or_missing_base_data_is_unknown(mutation, reason) -> None:
    record = M3Verifier().verify(
        _snapshot(10.0, **mutation),
        action_type="gripper_close",
        action_timestamp_s=9.0,
    )
    assert (record.verdict, record.reason_code) == (Verdict.UNKNOWN, reason)


def test_planning_scene_commands_never_describe_a_gazebo_attachment() -> None:
    model = M3PlanningSceneModel()
    target = Pose((0.28, -0.10, 0.43), (0.0, 0.0, 0.0, 1.0))
    distractor = Pose((0.28, 0.12, 0.44), (0.0, 0.0, 0.0, 1.0))
    initialized = model.initialize(target, distractor)
    assert [command.operation for command in initialized] == [
        "replace_world",
        "allow_target_touch",
        "allow_distractor_touch",
        "allow_table_touch",
    ]
    assert initialized[1].payload["links"] == list(M3Config().grasp_touch_links)
    assert initialized[2].payload["object_id"] == M3Config().distractor_id
    assert initialized[2].payload["links"] == list(M3Config().grasp_touch_links)
    assert initialized[3].payload["object_id"] == M3Config().table_id
    assert initialized[3].payload["links"] == list(
        M3Config().table_touch_links
    ) + [M3Config().target_id, M3Config().distractor_id]
    attached = model.attach(relative_pose(Pose((0, 0, 0), (0, 0, 0, 1)), target))
    assert attached[0].operation == "attach"
    assert all("gazebo" not in command.operation.lower() for command in (*initialized, *attached))
    assert model.release(target)[0].operation == "release"
    assert model.clear()[0].operation == "clear"
    assert model.initialized is False and model.attached is False


def test_pose_bbox_and_namespaced_identity_helpers() -> None:
    half_x, half_y = oriented_box_xy_half_extents(
        (0.04, 0.02, 0.06),
        (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
    )
    assert half_x == pytest.approx(0.01)
    assert half_y == pytest.approx(0.02)
    assert namespaced_entity_id(
        "m3_target::target_link::target_collision", ("m3_target", "m3_distractor")
    ) == "m3_target"
    assert namespaced_entity_id("prefix_m3_target_suffix", ("m3_target",)) is None
