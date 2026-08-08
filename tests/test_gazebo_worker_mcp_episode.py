from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from adapter.protocol import EnvAction
from agent.tools.sim_mcp import SimulatorMcpEpisodeConfig, SimulatorMcpEpisodeEnvironment
from sim.mcp_server import server
from sim.mcp_server.session import _get_mgr


class ServerToolTransport:
    """Test seam that invokes the existing MCP server tool wrappers directly."""

    def list_tools(self, *, timeout_s=None):
        del timeout_s
        return {"tools": [{"name": name} for name in ("create_env", "reset_env", "render_env", "close_env")]}

    def call_tool(self, name, arguments, *, timeout_s=None):
        del timeout_s
        tool = {
            "create_env": server.create_env,
            "reset_env": server.reset_env,
            "render_env": server.render_env,
            "close_env": server.close_env,
        }[name]
        return asyncio.run(tool(**arguments))


@pytest.mark.skipif(
    os.environ.get("OPENETA_RUN_LIVE_ROS_TEST") != "1",
    reason="opt-in: set OPENETA_RUN_LIVE_ROS_TEST=1 for the live worker test",
)
def test_gazebo_uses_existing_mcp_worker_lifecycle(monkeypatch, tmp_path: Path) -> None:
    ros2 = "/opt/ros/jazzy/bin/ros2"
    gz = "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
    if not (os.path.exists(ros2) and os.path.exists(gz)):
        pytest.skip("ROS 2 Jazzy/Gazebo Sim is not installed")
    monkeypatch.setenv("OPENETA_GAZEBO_ROS2_EXECUTABLE", ros2)
    monkeypatch.setenv("OPENETA_GAZEBO_GZ_EXECUTABLE", gz)
    monkeypatch.setenv(
        "OPENETA_GAZEBO_CAMERA_EXTRINSICS",
        json.dumps(
            {
                "frame_transform": "camera_to_world",
                "camera_frame": "opencv",
                "pos": [5.05, 0.05, 0.55],
                "quat_xyzw": [0.0, 0.0, 1.0, 0.0],
            }
        ),
    )
    session_id = "gazebo-worker-test"
    transport = ServerToolTransport()
    episode = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/gazebo_live_rgbd-v0",
            seed=13,
            session_id=session_id,
            image_output_root=tmp_path,
        ),
    )

    try:
        initial = episode.reset(task="observe worker RGB-D")
        assert initial.metadata["observation_provenance"] == "gazebo_ros_live"
        assert len(initial.cameras) == 1
        refreshed = episode.step(EnvAction(action_type="observe"))
        assert refreshed.info["environment_receipt_trusted"] is True
        assert refreshed.observation.metadata["observation_provenance"] == "gazebo_ros_live"
    finally:
        episode.close()
        _get_mgr().stop_all()
