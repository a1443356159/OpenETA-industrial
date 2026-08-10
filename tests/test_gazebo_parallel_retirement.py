from pathlib import Path

import pytest

from extensions.gazebo.m2 import M2Config, M2_ENV_ID, MODEL_ID
from extensions.gazebo.profiles import gazebo_profile
from sim.env_registry import get_env_spec


ROOT = Path(__file__).parents[1]


def test_m2_has_one_robotiq_identity_and_profile() -> None:
    assert M2_ENV_ID == "openeta/gazebo_rm75_robotiq2f85-v0"
    assert MODEL_ID == "rm75_robotiq_2f85_sim_v1"
    assert M2Config().model_id == MODEL_ID
    assert get_env_spec(M2_ENV_ID) is not None
    assert gazebo_profile("m2_robotiq2f85").model_config == M2Config()
    with pytest.raises(ValueError):
        gazebo_profile("m2")


def test_retired_parallel_assets_are_absent() -> None:
    retired_package = "openeta_rm75_" + "parallel_sim"
    retired_model = "rm75_" + "parallel_gripper_sim_v1"
    retired_env = "openeta/gazebo_rm75_" + "parallel-v0"
    assert not (ROOT / "extensions/gazebo/ros2_ws/src" / retired_package).exists()
    assert not (ROOT / "extensions/gazebo/models" / retired_model).exists()
    assert get_env_spec(retired_env) is None
