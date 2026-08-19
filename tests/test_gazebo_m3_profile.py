from __future__ import annotations

import xml.etree.ElementTree as ET

from extensions.gazebo.detachable_sdf import prepare_detachable_sdf
from extensions.gazebo.robot_control import GAZEBO_CONTROL_ENV_ID, MODEL_ID, GazeboControlConfig
from extensions.gazebo.native_grasp import (
    PICKPLACE_DISPLAY_NAME,
    PICKPLACE_ENV_ID,
    PICKPLACE_MODEL_ID,
    NativePickPlaceConfig,
    validated_pickplace_motion_guidance,
)
from extensions.gazebo.profiles import CONTROL, PHYSICS, STRUCTURED_RECEIPT, gazebo_profile
from sim.env_registry import get_env_spec


def test_m3_registration_exposes_the_approved_detachable_joint_profile() -> None:
    m3, m2 = get_env_spec(PICKPLACE_ENV_ID), get_env_spec(GAZEBO_CONTROL_ENV_ID)
    assert m3 is not None and m2 is not None and m3.display_name == PICKPLACE_DISPLAY_NAME
    assert NativePickPlaceConfig().model_id == PICKPLACE_MODEL_ID
    assert NativePickPlaceConfig().allow_stalling is True
    assert GazeboControlConfig().model_id == MODEL_ID
    profile = gazebo_profile("rm75_robotiq2f85_pickplace")
    assert profile.unavailable_reason is None
    assert {CONTROL, PHYSICS, STRUCTURED_RECEIPT} <= profile.capabilities
    assert profile.launch_file == "gazebo_pickplace.launch.py"


def test_m3_assets_are_required_before_manipulation_starts() -> None:
    config = NativePickPlaceConfig()
    config.validate_assets()
    package = config.ros_workspace / "src" / config.ros_package_name
    assert (package / "worlds/rm75_robotiq2f85_pickplace.sdf").is_file()


def test_m3_stable_motion_contract_uses_bilateral_contact_goal_tolerances() -> None:
    motion = validated_pickplace_motion_guidance()["motion_parameters"]

    assert motion["tolerance"] == 0.0002
    assert motion["ori_tolerance"] == 0.002


def test_m3_paused_launch_gives_runtime_a_bounded_detach_window() -> None:
    config = NativePickPlaceConfig()
    launch = (
        config.ros_workspace
        / "src"
        / config.ros_package_name
        / "launch/gazebo_pickplace.launch.py"
    ).read_text(encoding="utf-8")

    assert 'switch_timeout = ["--switch-timeout", "30.0"]' in launch
    assert launch.count("*switch_timeout") == 3


def test_m3_target_pose_contract_is_a_single_link_at_the_model_origin() -> None:
    """Keep the native Pose_V world/local-frame proof contract explicit."""

    config = NativePickPlaceConfig()
    world_path = (
        config.ros_workspace
        / "src"
        / config.ros_package_name
        / "worlds/rm75_robotiq2f85_pickplace.sdf"
    )
    root = ET.parse(world_path).getroot()
    target = root.find(f".//model[@name='{config.target_id}']")
    assert target is not None
    assert tuple(float(value) for value in target.findtext("pose", "").split()[:3]) == config.target_initial_xyz
    links = target.findall("link")
    assert [link.get("name") for link in links] == [config.target_link]
    assert links[0].find("pose") is None


def test_m3_contact_sensor_topics_use_the_sdf_sensor_topic_field() -> None:
    config = NativePickPlaceConfig()
    xacro_path = (
        config.ros_workspace
        / "src"
        / config.ros_package_name
        / "urdf/rm75_robotiq2f85_pickplace.urdf.xacro"
    )
    root = ET.parse(xacro_path).getroot()
    sensors = root.findall(".//sensor[@type='contact']")

    assert [sensor.findtext("topic") for sensor in sensors] == [
        config.left_contact_topic,
        config.right_contact_topic,
    ]
    assert all(sensor.find("contact/topic") is None for sensor in sensors)


def test_m3_sdf_renderer_allows_only_the_stock_fixed_joint_topology() -> None:
    root = ET.fromstring(
        """<sdf><model name="robot"><link name="base_link"/>
        <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
          <parent_link>gripper_mount_link</parent_link><child_model>target_object</child_model>
          <child_link>target_link</child_link><attach_topic>/openeta/native_grasp/detachable_joint/target/attach</attach_topic>
          <detach_topic>/openeta/native_grasp/detachable_joint/target/detach</detach_topic>
          <output_topic>/openeta/native_grasp/detachable_joint/target/state</output_topic>
        </plugin></model></sdf>"""
    )

    prepared = prepare_detachable_sdf(root)
    model = prepared.find("model")
    assert model is not None
    assert model.findtext("joint[@name='openeta_world_to_base']/parent") == "world"
    assert model.findtext("joint[@name='openeta_world_to_base']/child") == "base_link"
    assert model.findtext("self_collide") == "false"


def test_m3_sdf_renderer_replaces_converter_contact_topic_placeholders() -> None:
    root = ET.fromstring(
        """<sdf><model name="robot"><link name="base_link"/><link name="tip">
        <sensor name="pad" type="contact"><topic>/openeta/native_grasp/contacts/left_pad</topic>
          <contact><collision>tip_collision</collision><topic>__default_topic__</topic></contact>
        </sensor></link>
        <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
          <parent_link>gripper_mount_link</parent_link><child_model>target_object</child_model>
          <child_link>target_link</child_link><attach_topic>/openeta/native_grasp/detachable_joint/target/attach</attach_topic>
          <detach_topic>/openeta/native_grasp/detachable_joint/target/detach</detach_topic>
          <output_topic>/openeta/native_grasp/detachable_joint/target/state</output_topic>
        </plugin></model></sdf>"""
    )

    prepared = prepare_detachable_sdf(root)
    sensor = prepared.find(".//sensor[@name='pad']")
    assert sensor is not None
    assert sensor.findtext("topic") == "/openeta/native_grasp/contacts/left_pad"
    assert sensor.findtext("contact/topic") == "/openeta/native_grasp/contacts/left_pad"
