from __future__ import annotations

from types import SimpleNamespace
import time

import pytest

from extensions.gazebo.m2 import JOINT_NAMES, M2Config
from extensions.gazebo.ros_control import RosM2StateSource


class _Clock:
    def now(self):
        return object()


class _Node:
    def get_clock(self):
        return _Clock()


class _Tf:
    def __init__(self, *, fail: bool = False):
        self.fail = fail

    def lookup_transform(self, base, child, stamp):
        assert (base, child) == ("base_link", "gripper_mount_link")
        if self.fail:
            raise LookupError
        return SimpleNamespace(transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.1, y=0.2, z=0.3),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ))


def _joint_message():
    return SimpleNamespace(name=JOINT_NAMES, position=[0.0] * 7 + [0.035, 0.035], velocity=[])


def test_ros_state_source_requires_fresh_complete_joint_state_and_tf() -> None:
    source = RosM2StateSource(_Node(), _Tf(), config=M2Config(), freshness_s=0.02)
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
    source = RosM2StateSource(_Node(), _Tf(fail=True), config=M2Config())
    source.joint_state_callback(_joint_message())
    with pytest.raises(RuntimeError, match="TF_TIMEOUT"):
        source.state()
