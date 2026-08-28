from __future__ import annotations

import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest
import yaml

from extensions.gazebo.asset_preflight import validate_asset_root
from extensions.gazebo.robot_control import ARM_JOINT_BOUNDS, ARM_JOINTS, GazeboControlConfig


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
    assert manifest["camera_links"] == [
        "wrist_camera_bracket_link",
        "wrist_camera_housing_link",
        "wrist_camera_optical_frame",
    ]
    assert manifest["upstream_license"] == "BSD-3-Clause"


def test_asset_preflight_detects_digest_mutation(tmp_path: Path) -> None:
    copied = tmp_path / "rm75_6fb_v_vendor"
    shutil.copytree(ASSETS, copied)
    (copied / "urdf/arm_ros2_control.xacro").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_asset_root(copied)


def test_ros_package_is_relocatable_and_has_required_control_contract(tmp_path: Path) -> None:
    relocated = tmp_path / "openeta-control-relocated"
    shutil.copytree(
        ROOT / "extensions/gazebo",
        relocated / "extensions/gazebo",
        ignore=shutil.ignore_patterns("build", "install", "log", "__pycache__"),
    )
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


def test_all_repository_robot_xml_is_well_formed() -> None:
    for package in (ROBOTIQ_PACKAGE,):
        for path in package.rglob("*"):
            if path.suffix in {".xacro", ".srdf", ".sdf", ".urdf"}:
                ET.parse(path)


def test_robotiq_z_motion_world_is_test_only_and_launch_selectable() -> None:
    production = ROBOTIQ_PACKAGE / "worlds/rm75_robotiq2f85.sdf"
    z_test = ROBOTIQ_PACKAGE / "worlds/rm75_robotiq2f85_z_test.sdf"
    launch = (ROBOTIQ_PACKAGE / "launch/gazebo_moveit.launch.py").read_text(encoding="utf-8")

    production_root = ET.parse(production).getroot()
    z_test_root = ET.parse(z_test).getroot()
    production_world = production_root.find("world")
    z_test_world = z_test_root.find("world")
    assert production_world is not None
    assert z_test_world is not None
    assert production_world.attrib["name"] == "rm75_robotiq2f85"
    assert z_test_world.attrib["name"] == "rm75_robotiq2f85_z_test"

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
    assert 'default_world = str(share / "worlds/rm75_robotiq2f85.sdf")' in launch


def test_robot_profiles_use_production_command_limits_without_ineffective_adapter_knob() -> None:
    for package in (ROBOTIQ_PACKAGE,):
        controllers = (package / "config/controllers.yaml").read_text(encoding="utf-8")
        ompl = (package / "config/ompl_planning.yaml").read_text(encoding="utf-8")
        arm_interfaces = (
            ROOT / "extensions/gazebo/assets/rm75_6fb_v_vendor/urdf/arm_ros2_control.xacro"
        ).read_text(encoding="utf-8")

        assert controllers.count("enforce_command_limits: true") == 1
        # Keep controller-manager limiting enabled globally, but do not apply
        # its measured-state rate limiter a second time to already
        # time-parameterized simulated arm trajectories.  This joint-scoped
        # opt-out prevents command clipping while physical/planning limits remain.
        assert arm_interfaces.count('<limits enable="false"/>') == 1
        assert "start_state_max_bounds_error" not in ompl
        assert "longest_valid_segment_fraction: 0.002" in ompl


def test_arm_controller_proves_loaded_terminal_tracking() -> None:
    controllers = yaml.safe_load(
        (ROBOTIQ_PACKAGE / "config/controllers.yaml").read_text(encoding="utf-8")
    )["rm_group_controller"]["ros__parameters"]
    constraints = controllers["constraints"]
    control_xacro = (ROBOTIQ_PACKAGE / "urdf/ros2_control.xacro").read_text(encoding="utf-8")

    assert controllers["action_monitor_rate"] == 50.0
    assert constraints["goal_time"] == 4.0
    assert constraints["stopped_velocity_tolerance"] == 0.01
    assert {name: constraints[name] for name in ARM_JOINTS} == {
        name: {"trajectory": 0.06, "goal": 0.002} for name in ARM_JOINTS
    }
    assert "<position_proportional_gain>1.0</position_proportional_gain>" in control_xacro


def test_moveit_execution_monitor_accounts_for_simulated_time() -> None:
    execution = yaml.safe_load(
        (ROBOTIQ_PACKAGE / "config/moveit_controllers.yaml").read_text(encoding="utf-8")
    )["trajectory_execution"]

    assert execution["allowed_execution_duration_scaling"] == pytest.approx(1.8)
    assert execution["allowed_goal_duration_margin"] == pytest.approx(2.0)
    # This tuning extends only the supervisor deadline. It must not weaken the
    # state from which an already proven trajectory may start.
    assert execution["allowed_start_tolerance"] == pytest.approx(0.01)


def test_pickplace_physics_and_contact_rates_match_controller_budget() -> None:
    world = ET.parse(ROBOTIQ_PACKAGE / "worlds/rm75_robotiq2f85_pickplace.sdf").getroot()
    robot = ET.parse(ROBOTIQ_PACKAGE / "urdf/rm75_robotiq2f85_pickplace.urdf.xacro").getroot()
    controllers = yaml.safe_load(
        (ROBOTIQ_PACKAGE / "config/controllers.yaml").read_text(encoding="utf-8")
    )

    step_s = float(world.findtext(".//physics/max_step_size"))
    control_rate_hz = controllers["controller_manager"]["ros__parameters"]["update_rate"]
    assert step_s == pytest.approx(0.002)
    assert control_rate_hz == pytest.approx(1.0 / step_s)
    contact_rates = [
        float(sensor.findtext("update_rate"))
        for sensor in robot.findall(".//sensor[@type='contact']")
    ]
    assert contact_rates == [250.0, 250.0]
    assert all(rate <= control_rate_hz for rate in contact_rates)


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
    python_limits = {name: (lower, upper) for name, lower, upper in ARM_JOINT_BOUNDS}

    assert urdf_limits == python_limits
    for package in (ROBOTIQ_PACKAGE,):
        moveit = yaml.safe_load((package / "config/joint_limits.yaml").read_text(encoding="utf-8"))[
            "joint_limits"
        ]
        assert {
            name: (
                float(moveit[name]["min_position"]),
                float(moveit[name]["max_position"]),
            )
            for name in ARM_JOINTS
        } == python_limits


def test_robot_profiles_have_no_test_only_robot_or_control_assets() -> None:
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
            path.read_text(encoding="utf-8", errors="replace") for path in protected_files
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
    pickplace_robot = (ROBOTIQ_PACKAGE / "urdf/rm75_robotiq2f85_pickplace.urdf.xacro").read_text(
        encoding="utf-8"
    )
    launch = (ROBOTIQ_PACKAGE / "launch/gazebo_moveit.launch.py").read_text(encoding="utf-8")
    pickplace_launch = (ROBOTIQ_PACKAGE / "launch/gazebo_pickplace.launch.py").read_text(
        encoding="utf-8"
    )
    adapter = (ROBOTIQ_PACKAGE / "scripts/gripper_action_adapter.py").read_text(encoding="utf-8")
    shared_control = (ASSETS / "urdf/arm_ros2_control.xacro").read_text(encoding="utf-8")
    assert control.count("xacro:rm75_v_arm_control_interfaces") == 1
    assert shared_control.count("xacro:rm75_v_arm_interface name=") == 7
    assert control.count('<command_interface name="effort"/>') == 2
    assert shared_control.count('<command_interface name="position"/>') == 1
    assert shared_control.count('<limits enable="false"/>') == 1
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
    assert "<p_gain>30.0</p_gain>" in robot
    assert "<d_gain>0.8</d_gain>" in robot
    assert 'params="joint topic p_gain d_gain"' in pickplace_robot
    assert pickplace_robot.count('p_gain="4.0" d_gain="0.04"') == 2
    assert pickplace_robot.count('p_gain="0.5" d_gain="0.005"') == 4
    assert "<use_velocity_commands>" not in pickplace_robot
    assert "ParallelGripperCommand" in adapter
    assert "ros_gz_interfaces.msg import Contacts" in adapter
    assert "_target_contact_callback" in adapter
    assert "self._allow_stalling" in adapter
    assert "closing_goal" in adapter
    assert "TERMINAL_CONTACT_FRESHNESS_SIM_S" in adapter
    assert "TERMINAL_BILATERAL_CONTACT_DWELL_SIM_S" in adapter
    assert "BILATERAL_HOLD_PRELOAD_RAD" in adapter
    assert "TERMINAL_LINKAGE_SETTLE_DWELL_SIM_S" in adapter
    assert "TERMINAL_LINKAGE_MAX_VELOCITY_RAD_S" in adapter
    assert "COMMON_PROGRESS_EPSILON_RAD" in adapter
    assert "linkage_settle_complete" in adapter
    assert "functional_opening_complete" in adapter
    assert "full_open_goal" in adapter
    assert "full_open_goal = bool(\n            math.isclose(" in adapter
    assert "full_open_goal = bool(\n            opening_goal" not in adapter
    assert "CONTROLLER_BOUNDARY_INSET_RAD" in adapter
    assert "CONTACT_FRESHNESS_SIM_S" in adapter
    assert "_target_contact_sequences" in adapter
    assert "self._sim_time_s()" in adapter
    assert "sim_dt = max(0.0, sim_now_s - last_sim_tick_s)" in adapter
    assert "self._common_driver_position" in adapter
    assert "self._six_joint_positions(commanded_active_position)" in adapter
    assert "fresh_contact_sides" in adapter
    assert "sim_now_s - target_contact_sim_times.get" in adapter
    assert "fresh_bilateral_contact" in adapter
    assert "bilateral_mechanism_stationary" in adapter
    assert "self._bounded_contact_hold_position" in adapter
    assert "bilateral_contact_started_sim_time_s = None" in adapter
    assert "self._one_pad_compliance_exhausted" in adapter
    assert "compliance_exhausted" in adapter
    assert "COMMON_COMPLIANCE_DWELL_SIM_S" in adapter
    assert "side_holds" not in adapter
    assert "stall_hold_extra_rad" not in adapter
    assert "drive_mode" not in adapter
    assert "expanded_targets" not in adapter
    assert "freezing" not in adapter
    assert "JOINT_MULTIPLIERS" in adapter
    assert 'declare_parameter("action_timeout_s", ACTION_TIMEOUT_S)' in adapter
    assert '"action_timeout_s": 90.0' in launch
    assert launch.count("std_msgs/msg/Float64]gz.msgs.Double") == 6
    assert pickplace_launch.count("ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts") == 2
    assert '"stall_hold_extra_rad"' not in pickplace_launch
    assert '"drive_mode"' not in pickplace_launch
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
    config = GazeboControlConfig()
    config.validate_assets()
    assert config.asset_root == ASSETS


def test_v_description_is_shared_and_camera_is_fixed() -> None:
    arm = (ASSETS / "urdf/rm75_6fb_v.urdf.xacro").read_text(encoding="utf-8")
    arm_control = (ASSETS / "urdf/arm_ros2_control.xacro").read_text(encoding="utf-8")
    assert 'name="camera_rojoint"' not in arm
    assert 'name="camera_joint"' not in arm
    assert "openeta_wrist_rgbd" in arm
    assert 'name="joint_7" initial_value="3.141592653589793"' in arm_control
    root = ET.fromstring(arm)
    joint_1 = root.find("joint[@name='joint_1']")
    assert joint_1 is not None
    assert joint_1.find("origin").attrib == {
        "xyz": "0.20 0 0.2405",
        "rpy": "0 0 3.14159265359",
    }
    bracket_joint = root.find("joint[@name='wrist_camera_bracket_joint']")
    assert bracket_joint is not None
    assert bracket_joint.find("parent").attrib["link"] == "link_7"
    assert bracket_joint.find("child").attrib["link"] == "wrist_camera_bracket_link"
    assert bracket_joint.find("origin").attrib == {
        "xyz": "-0.0335 0.0085 -0.07388",
        "rpy": "-1.57079632679 -1.1344640138 0",
    }
    bracket_mesh = root.find("link[@name='wrist_camera_bracket_link']/visual/geometry/mesh")
    assert bracket_mesh is not None
    assert bracket_mesh.attrib["scale"] == "1 1 1"
    assert (
        root.find(
            "link[@name='wrist_camera_bracket_link']/visual[@name='wrist_camera_lateral_adapter']"
        )
        is None
    )
    housing_joint = root.find("joint[@name='wrist_camera_housing_joint']")
    assert housing_joint is not None
    assert housing_joint.find("parent").attrib["link"] == "wrist_camera_bracket_link"
    assert housing_joint.find("origin").attrib == {
        "xyz": "0.029031 -0.10022 -0.0005",
        "rpy": "-1.55172489 0.06894605 1.40558276",
    }
    optical_joint = root.find("joint[@name='wrist_camera_optical_joint']")
    assert optical_joint is not None
    assert optical_joint.find("parent").attrib["link"] == "wrist_camera_housing_link"
    assert optical_joint.find("origin").attrib == {
        "xyz": "0 0 -0.0242",
        "rpy": "0 1.57079632679 0",
    }
    assert len(root.findall("link[@name='wrist_camera_housing_link']/visual")) == 9
    assert (
        root.findtext("gazebo[@reference='wrist_camera_optical_frame']/sensor/camera/clip/near")
        == "0.02"
    )
    assert (
        root.findtext(
            "gazebo[@reference='wrist_camera_optical_frame']/sensor/camera/horizontal_fov"
        )
        == "1.308996938996"
    )
    assert "package://openeta_rm75_v_description/meshes/" in arm
    assert (DESCRIPTION_PACKAGE / "package.xml").is_file()
    for package in (PACKAGE, ROBOTIQ_PACKAGE):
        profile = next((package / "urdf").glob("rm75_*.urdf.xacro")).read_text(encoding="utf-8")
        assert "$(find openeta_rm75_v_description)/urdf/rm75_6fb_v.urdf.xacro" in profile


def test_wrist_camera_optical_ray_clears_gripper_and_converges_on_operation_axis() -> None:
    root = ET.fromstring((ASSETS / "urdf/rm75_6fb_v.urdf.xacro").read_text(encoding="utf-8"))

    def origin(name: str) -> tuple[list[float], list[float]]:
        element = root.find(f"joint[@name='{name}']/origin")
        assert element is not None
        return (
            [float(value) for value in element.attrib["xyz"].split()],
            [float(value) for value in element.attrib["rpy"].split()],
        )

    def rotation(rpy: list[float]) -> list[list[float]]:
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]

    def multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
        return [
            [
                sum(left[row][index] * right[index][column] for index in range(3))
                for column in range(3)
            ]
            for row in range(3)
        ]

    def transform(matrix: list[list[float]], vector: list[float]) -> list[float]:
        return [sum(matrix[row][index] * vector[index] for index in range(3)) for row in range(3)]

    bracket_xyz, bracket_rpy = origin("wrist_camera_bracket_joint")
    housing_xyz, housing_rpy = origin("wrist_camera_housing_joint")
    optical_xyz, optical_rpy = origin("wrist_camera_optical_joint")
    bracket_rotation = rotation(bracket_rpy)
    housing_rotation = multiply(bracket_rotation, rotation(housing_rpy))
    optical_rotation = multiply(housing_rotation, rotation(optical_rpy))
    transformed_housing = transform(bracket_rotation, housing_xyz)
    housing_position = [bracket_xyz[index] + transformed_housing[index] for index in range(3)]
    transformed_optical = transform(housing_rotation, optical_xyz)
    optical_position = [housing_position[index] + transformed_optical[index] for index in range(3)]
    optical_forward = [optical_rotation[index][0] for index in range(3)]
    operation_axis_target = [0.0, 0.0, 0.4]
    center_delta = [operation_axis_target[index] - optical_position[index] for index in range(3)]
    ray_distance = sum(center_delta[index] * optical_forward[index] for index in range(3))
    closest_point = [
        optical_position[index] + ray_distance * optical_forward[index] for index in range(3)
    ]

    # The lens is mounted outside the gripper envelope; at the finger plane
    # the optical centreline remains more than 80 mm off the tool axis.
    finger_plane_z = 0.104326
    finger_plane_distance = (finger_plane_z - optical_position[2]) / optical_forward[2]
    finger_plane_x = optical_position[0] + finger_plane_distance * optical_forward[0]
    assert optical_position[0] < -0.10
    assert finger_plane_x < -0.08
    assert ray_distance == pytest.approx(0.396300, abs=2e-5)
    assert closest_point == pytest.approx(operation_axis_target, abs=2e-5)


def test_robotiq_mount_is_flange_flush_and_matches_runtime_transform() -> None:
    fixture = ET.parse(ROBOTIQ_ASSETS / "urdf/rm75_robotiq2f85.urdf.xacro").getroot()
    fixed_joint = fixture.find(".//joint[@name='rm75_to_gripper_mount']")

    assert fixed_joint is not None
    assert fixed_joint.find("origin").attrib["xyz"] == "0 0 ${mount_z}"
    assert fixed_joint.find("origin").attrib["rpy"] == "0 0 1.57079632679"
    assert GazeboControlConfig().mount_xyz == (0.0, 0.0, 0.0)
    assert GazeboControlConfig().mount_quat_xyzw == pytest.approx((0.0, 0.0, 2**-0.5, 2**-0.5))
    for profile_name in (
        "rm75_robotiq2f85.urdf.xacro",
        "rm75_robotiq2f85_pickplace.urdf.xacro",
    ):
        profile = (ROBOTIQ_PACKAGE / "urdf" / profile_name).read_text(encoding="utf-8")
        assert 'parent="link_7" mount_z="0.0"' in profile
