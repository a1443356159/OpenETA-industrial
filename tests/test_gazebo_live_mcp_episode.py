from __future__ import annotations

import os
from pathlib import Path

import pytest

from adapter.protocol import EnvAction
from agent.tools.sim_mcp import SimulatorMcpEpisodeConfig, SimulatorMcpEpisodeEnvironment
from extensions.gazebo import (
    GazeboLiveMcpTransport,
    GazeboLiveSession,
    GazeboLiveSessionConfig,
    RosRgbdCameraConfig,
)


def test_live_gazebo_mcp_episode_lifecycle_and_cleanup(tmp_path: Path) -> None:
    if os.environ.get("OPENETA_RUN_LIVE_ROS_TEST") != "1":
        pytest.skip("opt-in: set OPENETA_RUN_LIVE_ROS_TEST=1 for live Gazebo MCP")
    ros2 = "/opt/ros/jazzy/bin/ros2"
    gz = "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
    if not (os.path.exists(ros2) and os.path.exists(gz)):
        pytest.skip("ROS 2 Jazzy/Gazebo Sim is not installed")

    def make_session(*, task: str, seed: int) -> GazeboLiveSession:
        del seed  # The documented world-control reset receives the episode seed.
        return GazeboLiveSession(
            GazeboLiveSessionConfig(
                ros2_executable=ros2,
                gz_executable=gz,
                launch_package="ros_gz_sim_demos",
                launch_file="rgbd_camera_bridge.launch.py",
                launch_arguments=("rviz:=false",),
                world_name="lidar_sensor",
                camera=RosRgbdCameraConfig(
                    rgb_topic="/rgbd_camera/image",
                    depth_topic="/rgbd_camera/depth_image",
                    camera_info_topic="/rgbd_camera/camera_info",
                    frame_id="rgbd_camera/link/rgbd_camera",
                    extrinsics={
                        "frame_transform": "camera_to_world",
                        "camera_frame": "opencv",
                        "pos": [5.05, 0.05, 0.55],
                        "quat_xyzw": [0.0, 0.0, 1.0, 0.0],
                    },
                ),
                startup_settle_s=8.0,
            ),
            task=task,
        )

    transport = GazeboLiveMcpTransport(make_session)
    episode = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/gazebo_live-v0",
            seed=17,
            image_output_root=tmp_path,
        ),
    )
    try:
        initial = episode.reset(task="observe live RGB-D")
        assert initial.metadata["observation_provenance"] == "gazebo_ros_live"
        refreshed = episode.step(EnvAction(action_type="observe"))
        assert refreshed.info["environment_receipt_trusted"] is True
    finally:
        episode.close()
    assert transport.active_handles == ()
