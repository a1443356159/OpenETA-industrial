from __future__ import annotations

import numpy as np

from extensions.gazebo.direct_env import GazeboDirectEnv
from extensions.gazebo.profiles import CONTROL, STRUCTURED_RECEIPT, gazebo_profile
from sim import bench_worker


def test_m2_profiles_use_the_single_direct_env_and_structured_receipt_capability() -> None:
    for name in ("m2_robotiq2f85",):
        profile = gazebo_profile(name)
        assert CONTROL in profile.capabilities
        assert STRUCTURED_RECEIPT in profile.capabilities
        assert GazeboDirectEnv.__name__ == "GazeboDirectEnv"


def test_generic_worker_codec_restores_internal_receipt_without_observation_clobber() -> None:
    observation = {"task": "", "cameras": {}, "robot": {}, "objects": [], "metadata": {"model_id": "m2"}}

    class Environment:
        def step(self, _action):
            return observation, 0.0, False, False, {
                "_openeta_receipt": {"ok": True, "debug": np.zeros((1,)).tolist()}
            }

    payload = bench_worker._step_with_image(Environment(), {}, render=False)
    assert payload["ok"] is True
    assert payload["observation"]["metadata"]["model_id"] == "m2"
    assert "_openeta_receipt" not in payload["info"]


def test_structured_receipt_raw_observation_never_clobbers_the_mcp_observation() -> None:
    # GazeboDirectEnv anchors the raw unified observation (numpy arrays,
    # cameras keyed by frame_id) inside its receipt.  The wire contract keeps
    # the StepResult-converted MCP observation at the top level.
    raw_cameras = {
        "top_camera_optical_frame": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
    }
    observation = {"task": "", "cameras": {}, "robot": {}, "objects": [], "metadata": {}}
    receipt_observation = {"task": "", "cameras": raw_cameras, "robot": {}, "objects": []}

    class Environment:
        def step(self, _action):
            return observation, 0.0, False, False, {
                "_openeta_receipt": {"ok": True, "observation": receipt_observation}
            }

    payload = bench_worker._step_with_image(Environment(), {}, render=False)
    assert payload["ok"] is True
    assert isinstance(payload["observation"]["cameras"], list)
    assert payload["observation"]["cameras"] == []
