from __future__ import annotations

import os
import pytest

from extensions.gazebo import GazeboLiveSession, GazeboLiveSessionConfig, RosRgbdCameraConfig


def test_live_session_create_reset_observe_close() -> None:
    if os.environ.get("OPENETA_RUN_LIVE_ROS_TEST") != "1":
        pytest.skip("opt-in: set OPENETA_RUN_LIVE_ROS_TEST=1 for live Gazebo session")
    ros2 = "/opt/ros/jazzy/bin/ros2"
    gz = "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
    if not os.path.exists(ros2) or not os.path.exists(gz):
        pytest.skip("ROS 2 Jazzy/Gazebo Sim is not installed")
    session = GazeboLiveSession(GazeboLiveSessionConfig(
        ros2_executable=ros2, gz_executable=gz,
        launch_package="ros_gz_sim_demos", launch_file="rgbd_camera_bridge.launch.py",
        launch_arguments=("rviz:=false",), world_name="lidar_sensor",
        camera=RosRgbdCameraConfig(
            rgb_topic="/rgbd_camera/image", depth_topic="/rgbd_camera/depth_image",
            camera_info_topic="/rgbd_camera/camera_info", frame_id="rgbd_camera/link/rgbd_camera",
            extrinsics={"frame_transform": "camera_to_world", "camera_frame": "opencv",
                        "pos": [5.05, 0.05, 0.55], "quat_xyzw": [0.0, 0.0, 1.0, 0.0]},
        ), startup_settle_s=8.0,
    ), task="observe RGB-D")
    try:
        initial = session.create()
        assert initial.task == "observe RGB-D"
        reset = session.reset(seed=11)
        assert reset.metadata["scene_epoch"] == 1
        assert reset.metadata["observation_provenance"] == "gazebo_ros_live"
    finally:
        session.close()
        session.close()

