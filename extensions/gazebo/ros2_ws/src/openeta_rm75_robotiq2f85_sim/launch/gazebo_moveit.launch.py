"""Launch the complete RM75 + Robotiq 2F-85 motion-control stack."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, LogInfo, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


_KINEMATICS_FILES = {
    "kdl_legacy": "config/kinematics.kdl_legacy.yaml",
    "kdl_fast": "config/kinematics.kdl_fast.yaml",
    "trac_ik_speed": "config/kinematics.trac_ik_speed.yaml",
    "trac_ik_distance": "config/kinematics.trac_ik_distance.yaml",
    "pick_ik_local": "config/kinematics.pick_ik_local.yaml",
}


def _kinematics_selection():
    qualification_profile = os.environ.get("OPENETA_QUALIFICATION_PROFILE", "legacy")
    solver_profile = os.environ.get("OPENETA_QUALIFICATION_SOLVER_PROFILE", "auto")
    if solver_profile == "auto":
        solver_profile = "kdl_fast" if qualification_profile == "fast_v3" else "kdl_legacy"
    try:
        return solver_profile, _KINEMATICS_FILES[solver_profile]
    except KeyError as exc:
        supported = ", ".join(["auto", *_KINEMATICS_FILES])
        raise RuntimeError(
            f"unsupported OPENETA_QUALIFICATION_SOLVER_PROFILE={solver_profile!r}; "
            f"expected one of {supported}"
        ) from exc


def _after_success(target, actions, label):
    """Start dependent actions only when the preceding process succeeded."""
    def on_exit(event, _context):
        if event.returncode == 0:
            return actions
        reason = f"{label} failed with return code {event.returncode}"
        return [LogInfo(msg=reason), EmitEvent(event=Shutdown(reason=reason))]

    return RegisterEventHandler(OnProcessExit(target_action=target, on_exit=on_exit))


def generate_launch_description():
    share = Path(get_package_share_directory("openeta_rm75_robotiq2f85_sim"))
    description_share = Path(get_package_share_directory("openeta_rm75_v_description"))
    default_world = str(share / "worlds/rm75_robotiq2f85.sdf")
    world_file = LaunchConfiguration("world")
    xacro_file = share / "urdf/rm75_robotiq2f85.urdf.xacro"
    robot_description_command = Command([FindExecutable(name="xacro"), " ", str(xacro_file)])
    robot_description = ParameterValue(robot_description_command, value_type=str)
    solver_profile, kinematics_file = _kinematics_selection()
    moveit = (
        MoveItConfigsBuilder(
            "rm75_robotiq_2f85_sim_v1", package_name="openeta_rm75_robotiq2f85_sim"
        )
        .robot_description(file_path="urdf/rm75_robotiq2f85.urdf.xacro")
        .robot_description_semantic(file_path="config/rm75_robotiq2f85.srdf")
        .robot_description_kinematics(file_path=kinematics_file)
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .to_moveit_configs()
    )
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory("ros_gz_sim")) / "launch/gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": ["-r -s ", world_file, " --physics-engine gz-physics-dartsim-plugin"]
        }.items(),
    )
    common = [{"use_sim_time": True}]
    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[{"robot_description": robot_description}, *common],
    )
    spawn = Node(
        package="ros_gz_sim", executable="create", output="screen",
        arguments=["-name", "rm75_robotiq_2f85_sim_v1", "-string", robot_description_command, "-z", "0.0"],
    )
    jsb = Node(package="controller_manager", executable="spawner", output="screen", arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"])
    arm = Node(package="controller_manager", executable="spawner", output="screen", arguments=["rm_group_controller", "--controller-manager", "/controller_manager"])
    gripper = Node(package="controller_manager", executable="spawner", output="screen", arguments=["parallel_gripper_controller", "--controller-manager", "/controller_manager"])
    gripper_action = Node(
        package="openeta_rm75_robotiq2f85_sim",
        executable="gripper_action_adapter.py",
        output="screen",
        # motion-control's certified mimic-relation contract is checked against the legacy
        # constant-multiplier drive; native-grasp uses the exact four-bar drive.
        parameters=[
            {
                "use_sim_time": True,
                "action_timeout_s": 90.0,
            }
        ],
    )
    move_group = Node(
        package="moveit_ros_move_group", executable="move_group", output="screen",
        parameters=[moveit.to_dict(), {"use_sim_time": True}],
    )
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="screen",
        arguments=[
            "/openeta_rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/openeta_rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/openeta_rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/openeta_wrist_rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/openeta_wrist_rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/openeta_wrist_rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/openeta/gripper/left_outer@std_msgs/msg/Float64]gz.msgs.Double",
            "/openeta/gripper/right_outer@std_msgs/msg/Float64]gz.msgs.Double",
            "/openeta/gripper/left_inner@std_msgs/msg/Float64]gz.msgs.Double",
            "/openeta/gripper/right_inner@std_msgs/msg/Float64]gz.msgs.Double",
            "/openeta/gripper/left_tip@std_msgs/msg/Float64]gz.msgs.Double",
            "/openeta/gripper/right_tip@std_msgs/msg/Float64]gz.msgs.Double",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        parameters=[{"use_sim_time": True}],
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "world",
            default_value=default_world,
            description="Absolute SDF world path; production defaults to the motion-control world",
        ),
        LogInfo(msg=f"OpenETA qualification solver profile: {solver_profile}"),
        SetEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            os.pathsep.join(
                [str(share.parent), str(description_share.parent), os.environ.get("GZ_SIM_RESOURCE_PATH", "")]
            ),
        ),
        gz_launch, rsp, spawn,
        # Start the clock/RGB-D bridge as soon as Gazebo is ready. Controllers
        # use simulated time and must not be activated before /clock exists.
        _after_success(spawn, [jsb, bridge], "Gazebo entity spawn"),
        _after_success(jsb, [arm], "joint_state_broadcaster activation"),
        _after_success(arm, [gripper], "RM75 controller activation"),
        _after_success(gripper, [gripper_action, move_group], "Robotiq controller activation"),
    ])
