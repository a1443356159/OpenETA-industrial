from __future__ import annotations

from pathlib import Path

from extensions.gazebo.direct_env import GazeboDirectEnv
from extensions.gazebo.profiles import gazebo_profile
from sim.env_registry import get_env_spec
from sim.mcp_server.worker_mgr import _bench_for_env_id


def test_gazebo_env_uses_existing_worker_resolution_and_registry() -> None:
    assert _bench_for_env_id("openeta/gazebo_live_rgbd-v0") == "gazebo"
    spec = get_env_spec("openeta/gazebo_live_rgbd-v0")
    assert spec is not None and spec.env_type == "gazebo"


def test_m1_registration_resolves_the_single_direct_env_profile() -> None:
    profile = gazebo_profile("m1")
    assert profile.name == "m1"
    assert profile.launch_package == "openeta_rm75_robotiq2f85_sim"
    assert profile.launch_file == "m1_gazebo_rgbd.launch.py"
    assert profile.world_name == "lidar_sensor"
    assert GazeboDirectEnv.__name__ == "GazeboDirectEnv"
    assert "fresh_observation" in profile.capabilities
    assert "authoritative_camera" in profile.capabilities


def test_m1_launch_is_server_only_and_preserves_official_rgbd_bridge() -> None:
    root = Path(__file__).parents[1]
    launch = root / "extensions/gazebo/ros2_ws/src/openeta_rm75_robotiq2f85_sim/launch/m1_gazebo_rgbd.launch.py"
    text = launch.read_text(encoding="utf-8")
    assert '"-r -s sensors_demo.sdf"' in text
    assert "rgbd_camera/depth_image" in text


def test_gazebo_unified_env_reports_the_established_backend_name() -> None:
    # The MCP wire contract and the M2/M3 acceptance gates expect the
    # historical "gazebo" backend string from the pre-UnifiedEnv workers.
    import gymnasium as gym

    env = gym.make(
        "openeta/gazebo_rm75_robotiq2f85-v0", task="backend", seed=0,
        render_mode="rgb_array",
    )
    try:
        assert getattr(env, "_backend", "") == "gazebo"
        raw = {
            "task": "t",
            "cameras": {"top_camera_optical_frame": {"rgb": "pixels"}},
            "robot": {"joint_positions": []},
            "objects": [],
            "metadata": {},
        }
        # The direct env already emits the established unified packet; the
        # normaliser must pass it through instead of dropping cameras.
        assert env._normalise_obs(raw) is raw
    finally:
        env.close()
