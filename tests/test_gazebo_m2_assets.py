from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest
import yaml

from extensions.gazebo.asset_preflight import validate_asset_root
from extensions.gazebo.m2 import ARM_JOINT_BOUNDS, ARM_JOINTS, M2Config


ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "extensions/gazebo/assets/rm75_6fb_v_vendor"
DESCRIPTION_PACKAGE = ROOT / "extensions/gazebo/ros2_ws/src/openeta_rm75_v_description"
ROBOTIQ_ASSETS = ROOT / "extensions/gazebo/assets/robotiq_2f85_vendor"
ROBOTIQ_PACKAGE = ROOT / "extensions/gazebo/ros2_ws/src/openeta_rm75_robotiq2f85_sim"
PACKAGE = ROBOTIQ_PACKAGE


def test_embedded_asset_manifest_and_references_are_closed() -> None:
    manifest = validate_asset_root(ASSETS)
    assert manifest["joint_names"] == list(ARM_JOINTS)
    assert manifest["description_id"] == "RM75-6FB-V"
    assert manifest["terminal_link"] == "link_7"
    assert manifest["camera_links"] == ["camera_rolink", "camera_link", "wrist_camera_optical_frame"]
    assert manifest["upstream_license"] == "BSD-3-Clause"


def test_asset_preflight_detects_digest_mutation(tmp_path: Path) -> None:
    copied = tmp_path / "rm75_6fb_v_vendor"
    shutil.copytree(ASSETS, copied)
    (copied / "urdf/arm_ros2_control.xacro").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_asset_root(copied)


def test_ros_package_is_relocatable_and_has_required_control_contract(tmp_path: Path) -> None:
    relocated = tmp_path / "openeta-m2-relocated"
    shutil.copytree(ROOT / "extensions/gazebo", relocated / "extensions/gazebo", ignore=shutil.ignore_patterns("build", "install", "log", "__pycache__"))
    validate_asset_root(relocated / "extensions/gazebo/assets/rm75_6fb_v_vendor")

    xacro = (PACKAGE / "urdf/ros2_control.xacro").read_text(encoding="utf-8")
    controllers = (PACKAGE / "config/controllers.yaml").read_text(encoding="utf-8")
    assert "gz_ros2_control/GazeboSimSystem" in xacro
    assert 'filename="gz_ros2_control-system"' in xacro
    assert "gz_ros2_control::GazeboSimROS2ControlPlugin" in xacro
    assert "forward_command_controller/ForwardCommandController" in controllers
    assert xacro.count("xacro:mimic_interface name=") == 5
    assert 'command_interface name="effort"' in xacro
    assert "gripper_left_finger_joint" in xacro


def test_all_repository_m2_xml_is_well_formed() -> None:
    for package in (ROBOTIQ_PACKAGE,):
        for path in package.rglob("*"):
            if path.suffix in {".xacro", ".srdf", ".sdf", ".urdf"}:
                ET.parse(path)


def test_robotiq_z_motion_world_is_test_only_and_launch_selectable() -> None:
    production = ROBOTIQ_PACKAGE / "worlds/m2_rm75_robotiq2f85.sdf"
    z_test = ROBOTIQ_PACKAGE / "worlds/m2_rm75_robotiq2f85_z_test.sdf"
    launch = (ROBOTIQ_PACKAGE / "launch/m2_gazebo_moveit.launch.py").read_text(
        encoding="utf-8"
    )

    production_root = ET.parse(production).getroot()
    z_test_root = ET.parse(z_test).getroot()
    production_world = production_root.find("world")
    z_test_world = z_test_root.find("world")
    assert production_world is not None
    assert z_test_world is not None
    assert production_world.attrib["name"] == "m2_rm75_robotiq2f85"
    assert z_test_world.attrib["name"] == "m2_rm75_robotiq2f85_z_test"

    # XML comments and the world name are the only intentional differences.
    # In particular the test asset cannot introduce a robot, joint initial
    # state, controller, or alternate physics behavior.
    z_test_world.attrib["name"] = production_world.attrib["name"]
    for root in (production_root, z_test_root):
        for element in root.iter():
            if element.text is not None and not element.text.strip():
                element.text = None
            if element.tail is not None and not element.tail.strip():
                element.tail = None
    assert ET.tostring(z_test_root) == ET.tostring(production_root)

    assert "DeclareLaunchArgument" in launch
    assert 'LaunchConfiguration("world")' in launch
    assert 'default_world = str(share / "worlds/m2_rm75_robotiq2f85.sdf")' in launch


def test_m2_profiles_use_production_command_limits_without_ineffective_adapter_knob() -> None:
    for package in (ROBOTIQ_PACKAGE,):
        controllers = (package / "config/controllers.yaml").read_text(encoding="utf-8")
        ompl = (package / "config/ompl_planning.yaml").read_text(encoding="utf-8")

        assert controllers.count("enforce_command_limits: true") == 1
        assert "start_state_max_bounds_error" not in ompl


def test_rm75_urdf_moveit_and_python_position_limits_are_identical() -> None:
    arm = ET.parse(ASSETS / "urdf/rm75_6fb_v.urdf.xacro").getroot()
    urdf_limits = {}
    for joint in arm.iter("joint"):
        limit = joint.find("limit")
        if joint.get("name") in ARM_JOINTS and limit is not None:
            urdf_limits[joint.get("name")] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
    python_limits = {
        name: (lower, upper) for name, lower, upper in ARM_JOINT_BOUNDS
    }

    assert urdf_limits == python_limits
    for package in (ROBOTIQ_PACKAGE,):
        moveit = yaml.safe_load(
            (package / "config/joint_limits.yaml").read_text(encoding="utf-8")
        )["joint_limits"]
        assert {
            name: (
                float(moveit[name]["min_position"]),
                float(moveit[name]["max_position"]),
            )
            for name in ARM_JOINTS
        } == python_limits


def test_m2_profiles_have_no_test_only_robot_or_control_assets() -> None:
    forbidden_name_parts = ("ready_pose", "z_motion", "test_controller", "test_robot")
    protected_directories = ("config", "launch", "scripts", "urdf")

    for package in (ROBOTIQ_PACKAGE,):
        protected_files = [
            path
            for directory in protected_directories
            for path in (package / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
        offenders = [
            path.relative_to(package).as_posix()
            for path in protected_files
            if any(part in path.name.lower() for part in forbidden_name_parts)
        ]
        assert offenders == []

        production_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in protected_files
        ).lower()
        for forbidden_token in (
            "z-motion mode",
            "z_motion_mode",
            "test_ready_pose",
            "publish_test_joint_state",
            "reset_to_test_pose",
        ):
            assert forbidden_token not in production_text


def test_robotiq_manifest_launch_and_control_adapter_are_complete() -> None:
    manifest = validate_asset_root(ROBOTIQ_ASSETS)
    assert manifest["model_id"] == "rm75_robotiq_2f85_sim_v1"
    assert manifest["maximum_aperture_m"] == 0.085
    assert manifest["upstream"]["commit"] == "2c047340aeb2440f7a60e429264221aab9658707"

    control = (ROBOTIQ_PACKAGE / "urdf/ros2_control.xacro").read_text(encoding="utf-8")
    controllers = (ROBOTIQ_PACKAGE / "config/controllers.yaml").read_text(encoding="utf-8")
    robot = (ROBOTIQ_PACKAGE / "urdf/rm75_robotiq2f85.urdf.xacro").read_text(encoding="utf-8")
    launch = (ROBOTIQ_PACKAGE / "launch/m2_gazebo_moveit.launch.py").read_text(encoding="utf-8")
    adapter = (ROBOTIQ_PACKAGE / "scripts/gripper_action_adapter.py").read_text(encoding="utf-8")
    shared_control = (ASSETS / "urdf/arm_ros2_control.xacro").read_text(encoding="utf-8")
    assert control.count("xacro:rm75_v_arm_control_interfaces") == 1
    assert shared_control.count("xacro:rm75_v_arm_interface name=") == 7
    assert control.count('<command_interface name="effort"/>') == 2
    assert shared_control.count('<command_interface name="position"/>') == 1
    assert 'mimic="false"' in control
    assert "mimic_interface" in control
    assert "forward_command_controller/ForwardCommandController" in controllers
    for name in (
        "gripper_left_finger_joint",
        "gripper_right_finger_joint",
        "gripper_left_inner_knuckle_joint",
        "gripper_right_inner_knuckle_joint",
        "gripper_left_finger_tip_joint",
        "gripper_right_finger_tip_joint",
    ):
        assert controllers.count(name) == 1
    assert robot.count("gz::sim::systems::JointPositionController") == 1
    assert robot.count("xacro:gz_gripper_position_controller joint=") == 6
    assert "ParallelGripperCommand" in adapter
    assert "JOINT_MULTIPLIERS" in adapter
    assert 'declare_parameter("action_timeout_s", ACTION_TIMEOUT_S)' in adapter
    assert '"action_timeout_s": 90.0' in launch
    assert launch.count("std_msgs/msg/Float64]gz.msgs.Double") == 6
    for token in (
        "rm75_robotiq_2f85_sim_v1",
        "joint_state_broadcaster",
        "rm_group_controller",
        "parallel_gripper_controller",
        "gripper_action_adapter.py",
        "moveit_ros_move_group",
        "/openeta_rgbd/depth_image",
        "use_sim_time",
        "OnProcessExit",
    ):
        assert token in launch


def test_production_config_ignores_external_vendor_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENETA_RM75_MODEL_PATH", "/does/not/exist")
    config = M2Config()
    config.validate_assets()
    assert config.asset_root == ASSETS


def test_v_description_is_shared_and_camera_is_fixed() -> None:
    arm = (ASSETS / "urdf/rm75_6fb_v.urdf.xacro").read_text(encoding="utf-8")
    assert 'name="camera_rojoint"\n    type="fixed"' in arm
    assert "openeta_wrist_rgbd" in arm
    assert "package://openeta_rm75_v_description/meshes/" in arm
    assert (DESCRIPTION_PACKAGE / "package.xml").is_file()
    for package in (PACKAGE, ROBOTIQ_PACKAGE):
        profile = next((package / "urdf").glob("rm75_*.urdf.xacro")).read_text(encoding="utf-8")
        assert "$(find openeta_rm75_v_description)/urdf/rm75_6fb_v.urdf.xacro" in profile
