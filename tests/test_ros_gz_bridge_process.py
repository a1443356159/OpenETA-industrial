from __future__ import annotations

import os
import shutil

import pytest

from extensions.gazebo import GazeboProcess, RosGzBridgeProcess


def test_ros_gz_clock_bridge_lifecycle() -> None:
    ros2 = shutil.which("ros2") or ("/opt/ros/jazzy/bin/ros2" if os.path.exists("/opt/ros/jazzy/bin/ros2") else None)
    gz = shutil.which("gz") or ("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz" if os.path.exists("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz") else None)
    if ros2 is None or gz is None:
        pytest.skip("ROS 2 Jazzy/Gazebo Sim is not installed")
    world = "extensions/gazebo/worlds/m1_oracle.sdf"
    sim = GazeboProcess(world=world, gz_executable=gz)
    bridge = RosGzBridgeProcess(
        ros2_executable=ros2,
        topics=("/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",),
    )
    try:
        sim.start()
        bridge.start()
        assert sim.running and bridge.running
    finally:
        bridge.close()
        sim.close()
    assert not bridge.running and not sim.running
