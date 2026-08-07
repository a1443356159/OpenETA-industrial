from __future__ import annotations

import os
import shutil

import pytest

from extensions.gazebo import GazeboProcess, GazeboWorldControl


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

