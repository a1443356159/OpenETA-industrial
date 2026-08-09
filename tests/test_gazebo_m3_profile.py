from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from extensions.gazebo.m2 import ROBOTIQ2F85_ENV_ID, ROBOTIQ2F85_MODEL_ID, Robotiq2F85Config
from extensions.gazebo.m3 import M3_DISPLAY_NAME, M3_ENV_ID, M3_MODEL_ID, M3Config
from extensions.gazebo.worker import m2_live_session_config_from_env
from sim.env_registry import get_env_spec


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "extensions/gazebo/ros2_ws/src/openeta_rm75_robotiq2f85_sim"


def test_m3_identity_registration_and_control_profile_are_isolated_from_m2() -> None:
    m3 = get_env_spec(M3_ENV_ID)
    m2 = get_env_spec(ROBOTIQ2F85_ENV_ID)
    assert m3 is not None and m2 is not None
    assert m3.display_name == M3_DISPLAY_NAME
    assert m3.id != m2.id
    assert M3Config().model_id == M3_MODEL_ID
    assert Robotiq2F85Config().model_id == ROBOTIQ2F85_MODEL_ID
    assert M3Config().allow_stalling is True
    assert not hasattr(Robotiq2F85Config(), "allow_stalling")


def test_m3_launch_and_world_are_separate_and_use_only_official_physics_systems() -> None:
    config = m2_live_session_config_from_env(robotiq=True, m3=True)
    assert config.launch_file == "m3_gazebo_pickplace.launch.py"
    assert config.world_name == "m3_rm75_robotiq2f85_pickplace"
    world = PACKAGE / "worlds/m3_rm75_robotiq2f85_pickplace.sdf"
    root = ET.parse(world).getroot()
    text = world.read_text(encoding="utf-8")
    assert root.find(".//world[@name='m3_rm75_robotiq2f85_pickplace']") is not None
    assert "gz-sim-contact-system" in text
    assert text.count("gz-sim-odometry-publisher-system") == 2
    assert "detachable" not in text.lower()
    assert "fixed_joint" not in text.lower()
    assert "m3_destination_marker" in text


def test_m3_world_geometry_mass_topics_and_material_values_match_contract() -> None:
    world = PACKAGE / "worlds/m3_rm75_robotiq2f85_pickplace.sdf"
    root = ET.parse(world).getroot()
    models = {item.attrib["name"]: item for item in root.findall(".//world/model")}
    assert models["m3_table"].findtext("pose") == "0.40 0 0.38 0 0 0"
    assert models["m3_target"].findtext("pose") == "0.28 -0.10 0.43 0 0 0"
    assert models["m3_target"].findtext(".//mass") == "0.10"
    assert models["m3_distractor"].findtext("pose") == "0.28 0.12 0.44 0 0 0"
    assert models["m3_distractor"].findtext(".//mass") == "0.12"
    assert root.find(".//sensor[@name='m3_target_contact']") is not None
    assert root.findtext(".//sensor[@name='m3_target_contact']/contact/topic") == "/m3/contact/target"
    assert root.findtext(".//model[@name='m3_target']/plugin/odom_topic") == "/m3/target/odometry"
    assert root.findtext(".//model[@name='m3_distractor']/plugin/odom_topic") == "/m3/distractor/odometry"


def test_m3_asset_preflight_requires_checked_in_profile_files() -> None:
    M3Config().validate_assets()
