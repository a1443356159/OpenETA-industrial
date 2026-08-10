from __future__ import annotations

from extensions.gazebo.direct_env import GazeboDirectEnv
from extensions.gazebo.m3 import M3Config
from extensions.gazebo.profiles import CONTROL, PHYSICS, STRUCTURED_RECEIPT, gazebo_profile


def test_m3_is_a_capability_composition_not_a_worker_subclass() -> None:
    profile = gazebo_profile("m3_pickplace")
    assert isinstance(profile.model_config, M3Config)
    assert {CONTROL, PHYSICS, STRUCTURED_RECEIPT}.issubset(profile.capabilities)
    assert GazeboDirectEnv.__mro__[1].__name__ == "Env"
