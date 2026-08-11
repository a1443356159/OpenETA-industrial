"""Launch the isolated M3 RM75 / Robotiq physical pick-place stack."""

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


def _after_success(target, actions, label):
    def on_exit(event, _context):
        if event.returncode == 0:
            return actions
        reason = f"{label} failed with return code {event.returncode}"
        return [LogInfo(msg=reason), EmitEvent(event=Shutdown(reason=reason))]

    return RegisterEventHandler(OnProcessExit(target_action=target, on_exit=on_exit))


def generate_launch_description():
    share = Path(get_package_share_directory("openeta_rm75_robotiq2f85_sim"))
    description_share = Path(get_package_share_directory("openeta_rm75_v_description"))
    xacro_file = share / "urdf/rm75_robotiq2f85_m3.urdf.xacro"
    attachment_mode = LaunchConfiguration("attachment_mode", default="physics")
    robot_description_command = Command(
        [FindExecutable(name="xacro"), " ", str(xacro_file), " attachment_mode:=", attachment_mode]
    )
    robot_description = ParameterValue(robot_description_command, value_type=str)
    moveit = (
        MoveItConfigsBuilder(
            "rm75_robotiq_2f85_pickplace_sim_v1",
            package_name="openeta_rm75_robotiq2f85_sim",
        )
        .robot_description(file_path="urdf/rm75_robotiq2f85_m3.urdf.xacro")
        .robot_description_semantic(file_path="config/rm75_robotiq2f85_m3.srdf")
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
                f"-r -s {share / 'worlds/m3_rm75_robotiq2f85_pickplace.sdf'} "
                "--physics-engine gz-physics-dartsim-plugin"
            )
        }.items(),
    )
    common = [{"use_sim_time": True}]
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}, *common],
    )
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name",
            "rm75_robotiq_2f85_pickplace_sim_v1",
            "-string",
            robot_description_command,
            "-z",
            "0.0",
        ],
    )
    jsb = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )
    arm = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["rm_group_controller", "--controller-manager", "/controller_manager"],
    )
    gripper = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["parallel_gripper_controller", "--controller-manager", "/controller_manager"],
    )
    gripper_action = Node(
        package="openeta_rm75_robotiq2f85_sim",
        executable="gripper_action_adapter.py",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "allow_stalling": True,
                "stall_velocity_threshold": 0.001,
                # Short contact window: with per-side freeze the first pad to
                # touch the target locks in ~0.3 s instead of pushing it
                # across the table for a full second.  Free-space speeds in
                # the two-speed ramp stay orders of magnitude above the
                # threshold, so false stalls are not a concern.
                "stall_timeout": 0.3,
                # A pure freeze holds each joint at its measured position, so
                # the PID error (and therefore the pinch force) collapses to
                # zero and the caged target slides out as soon as the lift
                # breaks stiction.  A small per-joint offset toward the close
                # target keeps a sustained squeeze through the carry.  Earlier
                # sweeps that rejected 0.01-0.03 ran before the anti-slip
                # table (mu 1.5) and the per-side contact freeze stabilised
                # the close geometry; with a centred pinch 0.03 just adds
                # normal force.
                "stall_hold_extra_rad": 0.03,
                # M3 drives the exact four-bar solution; M2 keeps the legacy
                # multiplier vector its mimic contract is certified against.
                "drive_mode": "four_bar",
            }
        ],
    )
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit.to_dict(), {"use_sim_time": True}],
    )
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/openeta_rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/openeta_rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/openeta_rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/openeta_wrist_rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/openeta_wrist_rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/openeta_wrist_rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/m3/target/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/m3/distractor/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
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
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "attachment_mode",
                default_value="physics",
                description="M3 grasp mechanism: physics friction or the user-approved detachable fallback",
            ),
            SetEnvironmentVariable(
                "GZ_SIM_RESOURCE_PATH",
                os.pathsep.join(
                    [
                        str(share.parent),
                        str(description_share.parent),
                        os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
                    ]
                ),
            ),
            gz_launch,
            rsp,
            spawn,
            _after_success(spawn, [jsb, bridge], "Gazebo M3 entity spawn"),
            _after_success(jsb, [arm], "joint_state_broadcaster activation"),
            _after_success(arm, [gripper], "RM75 controller activation"),
            _after_success(
                gripper,
                [gripper_action, move_group],
                "Robotiq controller activation",
            ),
        ]
    )
