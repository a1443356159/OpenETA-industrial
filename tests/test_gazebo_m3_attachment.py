"""Offline contracts for the user-approved M3 detachable-joint fallback."""

from __future__ import annotations

import pytest

from extensions.gazebo.deployment import GazeboDeploymentConfig
from extensions.gazebo.m3 import M3Config, ObjectState, Pose, ReasonCode, select_attachment_object
from extensions.gazebo.profiles import gazebo_profile
from extensions.gazebo.runtime import GazeboRuntime


def _object(object_id: str, xyz: tuple[float, float, float]) -> ObjectState:
    return ObjectState(
        object_id=object_id,
        name=object_id,
        label=object_id,
        role="target",
        pose=Pose(xyz, (0.0, 0.0, 0.0, 1.0)),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        support=None,
        timestamp_s=1.0,
    )


def _eef(distance_along_x: float = 0.0) -> Pose:
    return Pose((distance_along_x, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0))


def test_attach_requires_the_verified_close_stall_reason() -> None:
    target = _object("m3_target", (0.127, 0.0, 0.5))
    for reason in (
        ReasonCode.EMPTY_GRASP.value,
        ReasonCode.TARGET_HELD.value,
        ReasonCode.TARGET_NOT_LIFTED.value,
        ReasonCode.WRONG_OBJECT.value,
    ):
        assert (
            select_attachment_object(
                reason_code=reason, eef_pose=_eef(), objects=(target,)
            )
            is None
        )


def test_attach_selects_the_object_at_the_pads() -> None:
    target = _object("m3_target", (0.127, 0.0, 0.5))
    distractor = _object("m3_distractor", (0.30, 0.0, 0.5))
    assert (
        select_attachment_object(
            reason_code=ReasonCode.LIFT_REQUIRED.value,
            eef_pose=_eef(),
            objects=(target, distractor),
        )
        == "m3_target"
    )


def test_attach_selects_the_distractor_when_it_is_at_the_pads() -> None:
    # Wrong-object grasp: the distractor sits at the pads while the target is
    # far outside the gate band.
    target = _object("m3_target", (0.35, 0.0, 0.5))
    distractor = _object("m3_distractor", (0.13, 0.0, 0.5))
    assert (
        select_attachment_object(
            reason_code=ReasonCode.LIFT_REQUIRED.value,
            eef_pose=_eef(),
            objects=(target, distractor),
        )
        == "m3_distractor"
    )


def test_attach_rejects_empty_space_and_distant_objects() -> None:
    target = _object("m3_target", (0.60, 0.0, 0.5))
    distractor = _object("m3_distractor", (0.02, 0.0, 0.5))
    assert (
        select_attachment_object(
            reason_code=ReasonCode.LIFT_REQUIRED.value,
            eef_pose=_eef(),
            objects=(target, distractor),
        )
        is None
    )


def test_deployment_attachment_mode_defaults_to_physics() -> None:
    config = GazeboDeploymentConfig.from_environment({})
    assert config.m3_attachment_mode == "physics"
    assert (
        GazeboDeploymentConfig.from_environment(
            {"OPENETA_M3_ATTACHMENT_MODE": "detachable"}
        ).m3_attachment_mode
        == "detachable"
    )
    with pytest.raises(ValueError, match="OPENETA_M3_ATTACHMENT_MODE"):
        GazeboDeploymentConfig.from_environment({"OPENETA_M3_ATTACHMENT_MODE": "magnet"})


def test_runtime_creates_the_attachment_actuator_only_in_detachable_m3() -> None:
    profile = gazebo_profile("m3_pickplace")
    physics = GazeboRuntime(
        GazeboDeploymentConfig.from_environment({}), profile, task="t"
    )
    assert physics.attachment is None
    detachable = GazeboRuntime(
        GazeboDeploymentConfig.from_environment(
            {"OPENETA_M3_ATTACHMENT_MODE": "detachable"}
        ),
        profile,
        task="t",
    )
    assert detachable.attachment is not None
    m2_profile = gazebo_profile("m2_robotiq2f85")
    non_m3 = GazeboRuntime(
        GazeboDeploymentConfig.from_environment(
            {"OPENETA_M3_ATTACHMENT_MODE": "detachable"}
        ),
        m2_profile,
        task="t",
    )
    assert non_m3.attachment is None
