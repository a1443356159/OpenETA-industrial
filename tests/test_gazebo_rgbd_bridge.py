from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from extensions.gazebo import GazeboProcess, RosGzBridgeProcess


def test_m1_rgbd_bridge_process_contract() -> None:
    # A bare vendor binary is not a live Gazebo runtime: its plugin/resource
    # paths are exported only by Jazzy's setup script.  Without this gate an
    # ordinary unit-test shell waits for topics that can never be advertised.
    if os.environ.get("ROS_DISTRO") != "jazzy" or not os.environ.get(
        "GZ_SIM_RESOURCE_PATH"
    ):
        pytest.skip("source ROS 2 Jazzy before running the live Gazebo test")
    ros2 = shutil.which("ros2")
    gz = shutil.which("gz")
    if ros2 is None or gz is None:
        pytest.skip("sourced ROS 2 Jazzy/Gazebo Sim runtime is unavailable")
    sim = GazeboProcess(world="extensions/gazebo/worlds/rgbd.sdf", gz_executable=gz)
    bridge = RosGzBridgeProcess(
        ros2_executable=ros2,
        topics=(
            "/top_camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/top_camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/top_camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
        ),
    )
    try:
        sim.start()
        sim.wait_for_topics(("/top_camera/image", "/top_camera/depth_image", "/top_camera/camera_info"))
        bridge.start()
        assert sim.running and bridge.running
        raw_topics = subprocess.run(
            [gz, "topic", "-l"], check=True, capture_output=True, text=True, timeout=8.0,
        ).stdout.splitlines()
        assert "/top_camera/image" in raw_topics
        assert "/top_camera/depth_image" in raw_topics
        assert "/top_camera/camera_info" in raw_topics
    finally:
        bridge.close()
        sim.close()
    assert not sim.running and not bridge.running
