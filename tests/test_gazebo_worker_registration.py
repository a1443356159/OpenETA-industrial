from __future__ import annotations

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
    assert GazeboDirectEnv.__name__ == "GazeboDirectEnv"
    assert "fresh_observation" in profile.capabilities
    assert "authoritative_camera" in profile.capabilities
