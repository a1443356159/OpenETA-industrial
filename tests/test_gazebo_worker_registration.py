from __future__ import annotations

import numpy as np

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from extensions.gazebo import GazeboWorkerEnv
from sim.mcp_server.worker_mgr import _bench_for_env_id
from sim.env_registry import get_env_spec


def test_gazebo_env_uses_existing_worker_resolution_and_registry() -> None:
    assert _bench_for_env_id("openeta/gazebo_live_rgbd-v0") == "gazebo"
    spec = get_env_spec("openeta/gazebo_live_rgbd-v0")
    assert spec is not None
    assert spec.env_type == "gazebo"


def test_gazebo_worker_env_adapts_openeta_observation(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, _config, *, task):
            self.task = task
            self.closed = False
            self.epoch = 0

        def create(self):
            return self._observation()

        def reset(self, *, seed=None):
            del seed
            self.epoch += 1
            return self._observation()

        def observe(self):
            return self._observation()

        def close(self):
            self.closed = True

        def _observation(self):
            return EnvObservation(
                task=self.task,
                cameras=[
                    CameraFrame(
                        frame_id="camera",
                        rgb=[[[1, 2, 3]]],
                        depth=[[1.0]],
                        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5},
                        extrinsics={"frame_transform": "camera_to_world"},
                    )
                ],
                robot=RobotState(),
                metadata={"scene_epoch": self.epoch},
            )

    monkeypatch.setattr("extensions.gazebo.worker.GazeboLiveSession", FakeSession)
    monkeypatch.setattr(
        "extensions.gazebo.worker.live_session_config_from_env", lambda: object()
    )
    env = GazeboWorkerEnv(task="inspect", seed=4)
    try:
        raw, info = env.reset(seed=4)
        assert info == {}
        assert raw["task"] == "inspect"
        assert raw["cameras"]["camera"]["rgb"].shape == (1, 1, 3)
        assert np.asarray(raw["cameras"]["camera"]["depth"]).dtype == np.float32
        fresh = env.refresh_observation()
        assert fresh["metadata"]["scene_epoch"] == 1
    finally:
        env.close()
    assert env._session.closed is True
