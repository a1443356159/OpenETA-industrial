from __future__ import annotations

import pytest

from adapter.protocol import EnvAction, EnvObservation
from extensions.gazebo import GazeboConfig, GazeboEnvironment, GazeboObject, GazeboSimulatorAdapter
from extensions.gazebo.lifecycle import GazeboLifecycleError


def test_gazebo_create_reset_observe_is_deterministic() -> None:
    env = GazeboEnvironment(task="Put the silver screw into bin three", seed=17)
    first = env.create()
    second = env.reset(seed=17)
    assert first.to_dict()["objects"] == second.to_dict()["objects"]
    assert first.metadata["scene_epoch"] == 0
    assert first.metadata["observation_provenance"] == "gazebo_oracle"
    assert second.cameras[0].depth[0][0] == pytest.approx(1.0)
    assert second.cameras[0].extrinsics["camera_frame"] == "opencv"
    assert second.metadata["camera_topics"]["depth"] == "/top_camera/depth_image"


def test_gazebo_observation_round_trip_and_close_is_idempotent() -> None:
    env = GazeboEnvironment()
    raw = env.create().to_dict()
    restored = EnvObservation.from_dict(raw)
    assert restored.task == ""
    assert restored.cameras[0].frame_id == "top"
    env.close()
    env.close()
    with pytest.raises(GazeboLifecycleError):
        env.observe()


def test_gazebo_requires_create_before_observe() -> None:
    with pytest.raises(GazeboLifecycleError):
        GazeboEnvironment().observe()


def test_standard_adapter_exposes_lifecycle_and_rejects_control_in_m1() -> None:
    adapter = GazeboSimulatorAdapter()
    assert adapter.reset(task="observe only").metadata["backend"] == "gazebo"
    with pytest.raises(GazeboLifecycleError):
        adapter.step(EnvAction(action_type="tool_call"))
    adapter.close()


def test_gazebo_config_rejects_invalid_contract_values() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        GazeboConfig(image_width=0)
    with pytest.raises(ValueError, match="position"):
        GazeboObject("x", "x", (0.0, 1.0))
    with pytest.raises(ValueError, match="confidence"):
        GazeboObject("x", "x", (0.0, 1.0, 2.0), confidence=2.0)
    with pytest.raises(ValueError, match="absolute ROS topic"):
        GazeboConfig(top_rgb_topic="relative/image")
