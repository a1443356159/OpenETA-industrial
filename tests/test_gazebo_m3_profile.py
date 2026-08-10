from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from extensions.gazebo.m2 import M2_ENV_ID, MODEL_ID, M2Config
from extensions.gazebo.m3 import M3_DISPLAY_NAME, M3_ENV_ID, M3_MODEL_ID, M3Config
from extensions.gazebo.profiles import PHYSICS, gazebo_profile
from sim.env_registry import get_env_spec


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "extensions/gazebo/ros2_ws/src/openeta_rm75_robotiq2f85_sim"


def test_m3_identity_registration_and_profile_are_isolated_from_m2() -> None:
    m3, m2 = get_env_spec(M3_ENV_ID), get_env_spec(M2_ENV_ID)
    assert m3 is not None and m2 is not None and m3.display_name == M3_DISPLAY_NAME
    assert M3Config().model_id == M3_MODEL_ID
    assert M2Config().model_id == MODEL_ID
    profile = gazebo_profile("m3_pickplace")
    assert profile.launch_file == "m3_gazebo_pickplace.launch.py"
    assert profile.world_name == "m3_rm75_robotiq2f85_pickplace"
    assert PHYSICS in profile.capabilities


def test_m3_world_uses_official_odometry_without_contact_sensors() -> None:
    world = PACKAGE / "worlds/m3_rm75_robotiq2f85_pickplace.sdf"
    root = ET.parse(world).getroot()
    text = world.read_text(encoding="utf-8")
    assert root.find(".//world[@name='m3_rm75_robotiq2f85_pickplace']") is not None
    assert "gz-sim-contact-system" not in text
    assert "/m3/contact/" not in text
    assert text.count("gz-sim-odometry-publisher-system") == 2
    assert "detachable" not in text.lower()
    M3Config().validate_assets()
