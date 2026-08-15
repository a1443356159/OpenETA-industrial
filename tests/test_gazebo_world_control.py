from __future__ import annotations

import os
import shutil
from types import SimpleNamespace

import pytest

from extensions.gazebo import GazeboProcess, GazeboWorldControl
from extensions.gazebo import process as gazebo_process


def test_world_reset_retries_a_cold_control_service(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        SimpleNamespace(returncode=1, stdout="", stderr="unavailable"),
        SimpleNamespace(returncode=0, stdout="data: true", stderr=""),
    ]
    monkeypatch.setattr(gazebo_process.shutil, "which", lambda _name: "/usr/bin/gz")
    monkeypatch.setattr(gazebo_process.subprocess, "run", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(gazebo_process.time, "sleep", lambda _seconds: None)

    GazeboWorldControl(world_name="lidar_sensor").reset_all(seed=7)

    assert responses == []


def test_gazebo_world_reset_service() -> None:
    gz = shutil.which("gz") or ("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz" if os.path.exists("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz") else None)
    if gz is None:
        pytest.skip("Gazebo Sim is not installed")
    process = GazeboProcess(world="extensions/gazebo/worlds/m1_rgbd.sdf", gz_executable=gz)
    control = GazeboWorldControl(world_name="industrial_rgbd", gz_executable=gz)
    try:
        process.start()
        process.wait_for_topics(("/top_camera/image",))
        control.reset_all(seed=7)
    finally:
        process.close()
    assert not process.running
