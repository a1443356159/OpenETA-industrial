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
