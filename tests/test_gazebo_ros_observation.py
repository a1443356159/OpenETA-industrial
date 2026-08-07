from __future__ import annotations

import os
import time

import numpy as np
import pytest

from extensions.gazebo import (
    Ros2LaunchProcess,
    RosRgbdCameraConfig,
    RosRgbdCameraSource,
)


def test_live_ros_rgbd_source_builds_camera_frame() -> None:
    if os.environ.get("OPENETA_RUN_LIVE_ROS_TEST") != "1":
        pytest.skip("opt-in: set OPENETA_RUN_LIVE_ROS_TEST=1 for live ROS discovery")
    rclpy = pytest.importorskip("rclpy")
    ros2 = "/opt/ros/jazzy/bin/ros2"
    if not os.path.exists(ros2):
        pytest.skip("ROS 2 CLI is not installed")

    launch = Ros2LaunchProcess(
        ros2_executable=ros2,
        package="ros_gz_sim_demos",
        launch_file="rgbd_camera_bridge.launch.py",
        arguments=("rviz:=false",),
        startup_timeout_s=10.0,
    )
    rclpy.init()
    source = RosRgbdCameraSource(RosRgbdCameraConfig(
        rgb_topic="/rgbd_camera/image",
        depth_topic="/rgbd_camera/depth_image",
        camera_info_topic="/rgbd_camera/camera_info",
        frame_id="rgbd_camera/link/rgbd_camera",
        extrinsics={"frame_transform": "camera_to_world", "camera_frame": "opencv",
                    "pos": [5.05, 0.05, 0.55], "quat_xyzw": [0.0, 0.0, 1.0, 0.0]},
    ))
    try:
        launch.start()
        time.sleep(8.0)
        source.start()
        frame = source.capture(timeout_s=8.0)
        assert len(frame.rgb) == 240 and len(frame.rgb[0]) == 320
        assert len(frame.depth) == 240 and len(frame.depth[0]) == 320
        assert frame.intrinsics["fx"] == pytest.approx(277.1913564)
        assert frame.extrinsics["camera_frame"] == "opencv"
        assert np.isfinite(np.asarray(frame.depth, dtype=np.float32)).all()
    finally:
        source.close()
        if rclpy.ok():
            rclpy.shutdown()
        launch.close()
