"""Launch the complete RM75 M2 stack in deterministic dependency order."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    share = Path(get_package_share_directory("openeta_rm75_parallel_sim"))
    description_share = Path(get_package_share_directory("openeta_rm75_v_description"))
    robot_description_command = Command(
        [FindExecutable(name="xacro"), " ", str(share / "urdf/rm75_parallel.urdf.xacro")]
    )
    robot_description = ParameterValue(robot_description_command, value_type=str)
    moveit = (
        MoveItConfigsBuilder("rm75_parallel_gripper_sim_v1", package_name="openeta_rm75_parallel_sim")
        .robot_description(file_path="urdf/rm75_parallel.urdf.xacro")
        .robot_description_semantic(file_path="config/rm75_parallel.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
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
            "gz_args": (
                f"-r -s {share / 'worlds/m2_rm75_parallel.sdf'} "
                "--physics-engine gz-physics-bullet-featherstone-plugin"
            )
        }.items(),
    )
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen", parameters=[{"robot_description": robot_description, "use_sim_time": True}])
    spawn = Node(package="ros_gz_sim", executable="create", output="screen", arguments=["-name", "rm75_parallel_gripper_sim_v1", "-string", robot_description_command, "-z", "0.0"])
    jsb = Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"], output="screen")
    arm = Node(package="controller_manager", executable="spawner", arguments=["rm_group_controller", "--controller-manager", "/controller_manager"], output="screen")
    gripper = Node(package="controller_manager", executable="spawner", arguments=["parallel_gripper_controller", "--controller-manager", "/controller_manager"], output="screen")
    move_group = Node(package="moveit_ros_move_group", executable="move_group", output="screen", parameters=[moveit.to_dict(), {"use_sim_time": True}])
    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen", arguments=[
        "/openeta_rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image",
        "/openeta_rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
        "/openeta_rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        "/openeta_wrist_rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image",
        "/openeta_wrist_rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
        "/openeta_wrist_rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
    ])
    return LaunchDescription([
        SetEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            os.pathsep.join(
                [str(share.parent), str(description_share.parent), os.environ.get("GZ_SIM_RESOURCE_PATH", "")]
            ),
        ),
        gz_launch, rsp, spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm])),
        RegisterEventHandler(OnProcessExit(target_action=arm, on_exit=[gripper])),
        RegisterEventHandler(OnProcessExit(target_action=gripper, on_exit=[move_group, bridge])),
    ])
