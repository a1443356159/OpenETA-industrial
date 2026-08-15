"""Offline hard gates for the M3 detachable fallback repair."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from extensions.gazebo.detachable_sdf import prepare_detachable_sdf
from extensions.gazebo.m3 import (
    M3Config,
    ObjectState,
    PadSnapshot,
    PadSurface,
    Pose,
    confirm_pad_contact,
    pads_are_clear,
)
from extensions.gazebo.process import DetachableJointState, GazeboDetachableJointControl
from extensions.gazebo.ros_control import gripper_action_success, gripper_terminal_succeeded


ROOT = Path(__file__).parents[1]
XACRO = ROOT / "extensions/gazebo/ros2_ws/src/openeta_rm75_robotiq2f85_sim/urdf/rm75_robotiq2f85_m3.urdf.xacro"
LAUNCH = ROOT / "extensions/gazebo/ros2_ws/src/openeta_rm75_robotiq2f85_sim/launch/m3_gazebo_pickplace.launch.py"
RM75_REPRO = ROOT / "extensions/gazebo/detachable_joint_repro/run_rm75.py"


def _rendered_sdf() -> ET.Element:
    return ET.fromstring(
        """
        <sdf version="1.9"><model name="rm75">
          <link name="base_link"/>
          <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
            <parent_link>gripper_mount_link</parent_link><child_model>m3_target</child_model><child_link>target_link</child_link>
          </plugin>
          <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
            <parent_link>gripper_mount_link</parent_link><child_model>m3_distractor</child_model><child_link>distractor_link</child_link>
          </plugin>
        </model></sdf>
        """
    )


def test_detachable_sdf_is_validated_fixed_root_and_explicitly_non_self_colliding() -> None:
    root = prepare_detachable_sdf(_rendered_sdf())
    model = root.find("model")
    assert model is not None
    assert model.findtext("self_collide") == "false"
    joint = model.find("joint[@name='openeta_detachable_world_to_base']")
    assert joint is not None
    assert (joint.findtext("parent"), joint.findtext("child")) == ("world", "base_link")
    assert [plugin.findtext("parent_link") for plugin in model.findall("plugin")] == [
        "gripper_mount_link",
        "gripper_mount_link",
    ]


def test_detachable_sdf_unfixed_root_is_an_explicit_diagnostic_override() -> None:
    root = prepare_detachable_sdf(_rendered_sdf(), fixed_root=False)
    model = root.find("model")
    assert model is not None
    assert model.find("joint[@name='openeta_detachable_world_to_base']") is None


def test_detachable_sdf_rejects_an_unexpected_parent_link() -> None:
    try:
        prepare_detachable_sdf(_rendered_sdf(), parent_link="base_link")
    except Exception as exc:
        assert "parent link" in str(exc)
    else:
        raise AssertionError("unexpected detachable parent link was accepted")


def test_physics_urdf_remains_clean_and_launch_uses_file_only_for_detachable() -> None:
    xacro = XACRO.read_text(encoding="utf-8")
    launch = LAUNCH.read_text(encoding="utf-8")
    assert "<self_collide>true</self_collide>" not in xacro
    assert "render_detachable_sdf" in launch
    assert '"-file"' in launch
    assert '"-string"' in launch
    assert "detachable_fixed_root" in launch
    assert "detachable_parent_link" in launch
    assert "OnShutdown" in launch


def test_real_rm75_repro_uses_production_launch_and_real_trajectory_not_set_pose() -> None:
    runner = RM75_REPRO.read_text(encoding="utf-8")
    assert "m3_gazebo_pickplace.launch.py" in runner
    assert "FollowJointTrajectory" in runner
    assert 'PROBE_JOINT = "joint_1"' in runner
    assert "PARENT_MOTION_DELTA_RAD = 0.35" in runner
    assert "Pose_V" in runner
    assert '"link": "target_link"' in runner
    assert '"link": "distractor_link"' in runner
    assert '"child_link"' in runner
    assert "_contact_reattach_negative" in runner
    assert "_place_object_clear_of_gripper" in runner
    assert "_request_clear_attach" in runner
    assert "--unfixed-root" in runner
    assert "detached_relative_rotation_rad" in runner
    assert "MIN_DETACHED_ROTATION_RAD" in runner
    assert "set_pose(self.parent_link" not in runner


def _object(object_id: str, position: tuple[float, float, float], stamp: float) -> ObjectState:
    return ObjectState(
        object_id=object_id,
        name=object_id,
        label=object_id,
        role="target",
        pose=Pose(position, (0.0, 0.0, 0.0, 1.0)),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        support="m3_table",
        timestamp_s=stamp,
    )


def _pad_sample(stamp: float, objects: tuple[ObjectState, ...], separation: float = 0.022) -> PadSnapshot:
    return PadSnapshot(
        timestamp_s=stamp,
        left=PadSurface((-separation, 0.0, 0.0), (1.0, 0.0, 0.0), 0.04),
        right=PadSurface((separation, 0.0, 0.0), (-1.0, 0.0, 0.0), 0.04),
        objects=objects,
    )


def test_pad_gate_requires_stable_post_action_bilateral_unique_contact() -> None:
    samples = tuple(
        _pad_sample(stamp, (_object("m3_target", (0.0, 0.0, 0.0), stamp),))
        for stamp in (1.01, 1.06, 1.11)
    )
    result = confirm_pad_contact(samples, action_timestamp_s=1.0)
    assert (result.accepted, result.candidate_id, result.sample_count) == (True, "m3_target", 3)


def test_pad_gate_rejects_short_stale_and_multiple_candidate_evidence() -> None:
    target = _object("m3_target", (0.0, 0.0, 0.0), 1.01)
    distractor = _object("m3_distractor", (0.0, 0.0, 0.0), 1.01)
    assert not confirm_pad_contact(
        (_pad_sample(1.01, (target,)), _pad_sample(1.06, (target,))),
        action_timestamp_s=1.0,
    ).accepted
    assert not confirm_pad_contact(
        tuple(_pad_sample(stamp, (target, distractor)) for stamp in (1.01, 1.06, 1.11)),
        action_timestamp_s=1.0,
    ).accepted
    assert not confirm_pad_contact(
        tuple(_pad_sample(stamp, (target,)) for stamp in (0.99, 1.06, 1.11)),
        action_timestamp_s=1.0,
    ).accepted


def test_backoff_requires_both_pad_clearances() -> None:
    target = _object("m3_target", (0.0, 0.0, 0.0), 2.0)
    assert pads_are_clear(_pad_sample(2.0, (target,), separation=0.04), "m3_target")
    assert not pads_are_clear(_pad_sample(2.0, (target,), separation=0.022), "m3_target")


def test_detachable_state_starts_unknown_and_detach_is_idempotent_only_after_ack() -> None:
    control = object.__new__(GazeboDetachableJointControl)
    control._state = {
        "target": DetachableJointState.UNKNOWN,
        "distractor": DetachableJointState.UNKNOWN,
    }
    control._physical_baselines = {}
    calls: list[tuple[str, str]] = []

    def drive(label: str, action: str):
        calls.append((label, action))
        control._state[label] = DetachableJointState.DETACHED
        return DetachableJointState.DETACHED

    control._drive = drive
    assert control.ensure_detached("target") is DetachableJointState.DETACHED
    assert control.ensure_detached("target") is DetachableJointState.DETACHED
    assert calls == [("target", "detach")]


def test_child_link_pose_gate_rejects_model_only_co_motion() -> None:
    control = object.__new__(GazeboDetachableJointControl)
    control.parent_link = "gripper_mount_link"
    control._child_links = {"target": "target_link"}
    control._physical_baselines = {}
    identity = (0.0, 0.0, 0.0, 1.0)
    yaw = (0.0, 0.0, math.sin(0.35 / 2), math.cos(0.35 / 2))
    samples = iter((
        {
            "gripper_mount_link": ((0.0, 0.0, 0.0), identity),
            "target_link": ((0.0, 0.0, 0.0), identity),
        },
        {
            "gripper_mount_link": ((0.0, 0.0, 0.0), yaw),
            "target_link": ((0.0, 0.0, 0.0), identity),
        },
        {
            "gripper_mount_link": ((0.0, 0.0, 0.0), yaw),
            "target_link": ((0.0, 0.0, 0.0), identity),
        },
    ))
    control._world_poses = lambda: next(samples)
    assert control.capture_physical_baseline("target")
    translation, rotation = control.physical_relative_drift("target") or (0.0, 0.0)
    assert translation < 1e-9
    assert rotation == pytest.approx(0.35, abs=1e-9)
    assert not control.is_physically_held("target")


def test_gripper_stall_is_not_success_after_abort_cancel_or_timeout() -> None:
    assert gripper_terminal_succeeded(4)
    assert not gripper_terminal_succeeded(5)
    assert gripper_action_success(
        reached_goal=False, stalled=True, allow_stalling=True, terminal_succeeded=True
    )
    assert not gripper_action_success(
        reached_goal=False, stalled=True, allow_stalling=True, terminal_succeeded=False
    )
