from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from extensions.gazebo.m2 import M2ControlResult
from extensions.gazebo.process import GazeboProcessError
from extensions.gazebo import worker
from sim import bench_worker


class _Controller:
    def __init__(self, result: M2ControlResult | None = None) -> None:
        self.result = result or M2ControlResult(
            True,
            payload={
                "action_started_ros_time_s": 10.0,
                "action_completed_ros_time_s": 11.0,
            },
        )
        self.actions = []
        self.closed = 0

    def execute(self, action):
        self.actions.append(action)
        return self.result

    def close(self):
        self.closed += 1


def test_m2_step_executes_one_structured_action_and_requests_post_action_observation() -> None:
    environment = object.__new__(worker.GazeboM2WorkerEnv)
    environment.controller = _Controller()
    captured = []
    observation = {
        "cameras": {"top": {"timestamp_s": 12.0}, "wrist": {"timestamp_s": 12.0}},
        "robot": {"joint_positions": [0.0] * 13},
        "metadata": {"model_id": "rm75_robotiq_2f85_sim_v1"},
    }

    def refresh(self, **kwargs):
        captured.append(kwargs)
        return observation

    environment.refresh_observation = MethodType(refresh, environment)
    action = {"action_type": "gripper_open"}

    returned, reward, terminated, truncated, info = environment.step(action)

    assert environment.controller.actions == [action]
    assert captured[0]["min_camera_timestamp_s"] == 11.0
    assert captured[0]["min_received_monotonic_s"] > 0.0
    assert returned is observation and info["observation"] is observation
    assert reward == 0.0 and terminated is False and truncated is False


def test_m2_constructor_failure_closes_controller_and_launch(monkeypatch) -> None:
    class FailingSession:
        def __init__(self, *_args, **_kwargs):
            self.closed = 0
            made.append(self)

        def create(self):
            raise GazeboProcessError("ROS_NOT_READY")

        def close(self):
            self.closed += 1

    made = []
    controller = _Controller()
    monkeypatch.setattr(worker, "GazeboLiveSession", FailingSession)

    with pytest.raises(GazeboProcessError, match="ROS_NOT_READY"):
        worker.GazeboM2WorkerEnv(controller=controller)

    assert controller.closed == 1
    assert made[0].closed == 1


def test_m2_factory_failure_closes_launch(monkeypatch) -> None:
    class Session:
        def __init__(self, *_args, **_kwargs):
            self.closed = 0
            made.append(self)

        def create(self):
            return SimpleNamespace()

        def close(self):
            self.closed += 1

    class Factory:
        def create(self, _config):
            raise RuntimeError("MOVE_GROUP_UNAVAILABLE")

    made = []
    monkeypatch.setattr(worker, "GazeboLiveSession", Session)
    monkeypatch.setattr(worker, "RosM2ControllerFactory", Factory)

    with pytest.raises(GazeboProcessError, match="MOVE_GROUP_UNAVAILABLE"):
        worker.GazeboM2WorkerEnv()

    assert made[0].closed == 1


def test_bench_worker_preserves_structured_m2_observation() -> None:
    observation = {
        "task": "",
        "cameras": {},
        "robot": {},
        "objects": [],
        "metadata": {"model_id": "rm75_robotiq_2f85_sim_v1"},
    }

    class Environment:
        _openeta_structured_actions = True

        def step(self, _action):
            return observation, 0.0, False, False, {
                "ok": True,
                # This raw duplicate is intentionally not JSON serializable.
                "observation": {"rgb": np.zeros((2, 2, 3))},
            }

    payload = bench_worker._step_with_image(Environment(), {}, render=False)

    assert payload["ok"] is True
    assert isinstance(payload["observation"], dict)
    assert payload["observation"]["metadata"]["model_id"] == "rm75_robotiq_2f85_sim_v1"
    assert "observation" not in payload["info"]
