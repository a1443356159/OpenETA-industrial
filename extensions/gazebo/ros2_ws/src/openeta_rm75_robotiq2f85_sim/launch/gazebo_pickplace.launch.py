"""Launch native-grasp paused so DetachableJoint can be detached before any physics tick."""

import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, IncludeLaunchDescription, LogInfo, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable
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


for _directory in (Path(get_package_share_directory("openeta_rm75_robotiq2f85_sim")) / "launch", Path(__file__).resolve().parent):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))
from detachable_sdf import render_detachable_sdf  # noqa: E402
from acceptance_scene_world import (  # noqa: E402
    compile_authoritative_scene,
    materialize_authoritative_scene,
)


def _after_success(target, actions, label):
    def on_exit(event, _context):
        if event.returncode == 0:
            return actions
        reason = f"{label} failed with return code {event.returncode}"
        return [LogInfo(msg=reason), EmitEvent(event=Shutdown(reason=reason))]
    return RegisterEventHandler(OnProcessExit(target_action=target, on_exit=on_exit))


def _spawn_robot(context, *, xacro_file, generated_sdfs, post_spawn_actions):
    try:
        rendered = render_detachable_sdf(xacro_file=xacro_file, environment=os.environ.copy())
    except Exception as exc:
        return [LogInfo(msg=f"NATIVE_GRASP_DART_UNSUPPORTED: {exc}"), EmitEvent(event=Shutdown(reason="NATIVE_GRASP_DART_UNSUPPORTED"))]
    generated_sdfs.append(rendered)
    spawn = Node(
        package="ros_gz_sim", executable="create", output="screen",
        arguments=["-name", "rm75_robotiq_2f85_pickplace_sim_v1", "-file", str(rendered)],
    )
    return [spawn, _after_success(spawn, post_spawn_actions, "native-grasp Gazebo entity spawn")]


def _unlink_generated_sdf(paths):
    def cleanup(_event, _context):
        for path in tuple(paths):
            Path(path).unlink(missing_ok=True)
            paths.remove(path)
        return []
    return cleanup


def generate_launch_description():
    share = Path(get_package_share_directory("openeta_rm75_robotiq2f85_sim"))
    description_share = Path(get_package_share_directory("openeta_rm75_v_description"))
    xacro_file = share / "urdf/rm75_robotiq2f85_pickplace.urdf.xacro"
    robot_description_command = Command([FindExecutable(name="xacro"), " ", str(xacro_file)])
    robot_description = ParameterValue(robot_description_command, value_type=str)
    solver_profile, kinematics_file = _kinematics_selection()
    moveit = (
        MoveItConfigsBuilder("rm75_robotiq_2f85_pickplace_sim_v1", package_name="openeta_rm75_robotiq2f85_sim")
        .robot_description(file_path="urdf/rm75_robotiq2f85_pickplace.urdf.xacro")
        .robot_description_semantic(file_path="config/rm75_robotiq2f85_pickplace.srdf")
        .robot_description_kinematics(file_path=kinematics_file)
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .to_moveit_configs()
    )
    generated_sdfs = []
    canonical_world = share / "worlds/rm75_robotiq2f85_pickplace.sdf"
    authoritative_scene = compile_authoritative_scene(
        base_world=canonical_world,
        catalog_path=share / "config/acceptance_scenes.json",
        scene_id=str(os.environ.get("OPENETA_ACCEPTANCE_SCENE") or "normal").strip(),
    )
    selected_world = materialize_authoritative_scene(authoritative_scene)
    acceptance_scene = authoritative_scene.world_scene
    generated_sdfs.append(selected_world)
    # Deliberately omit -r: Gazebo starts paused.  GazeboRuntime sends and
    # confirms the initial detach before it calls world control pause:false.
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(Path(get_package_share_directory("ros_gz_sim")) / "launch/gz_sim.launch.py")),
        launch_arguments={"gz_args": f"-s {selected_world} --physics-engine gz-physics-dartsim-plugin"}.items(),
    )
    common = [{"use_sim_time": True}]
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen", parameters=[{"robot_description": robot_description}, *common])
    # native-grasp deliberately starts paused until the runtime obtains the initial
    # DetachableJoint detach ACK.  Controller activation needs a physics tick,
    # so retain the spawner through that bounded preflight window instead of
    # letting its five-second default tear down the whole launch.
    switch_timeout = ["--switch-timeout", "30.0"]
    service_call_timeout = ["--service-call-timeout", "30.0"]
    jsb = Node(package="controller_manager", executable="spawner", output="screen", arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager", *switch_timeout, *service_call_timeout])
    arm = Node(package="controller_manager", executable="spawner", output="screen", arguments=["rm_group_controller", "--controller-manager", "/controller_manager", *switch_timeout, *service_call_timeout])
    gripper = Node(package="controller_manager", executable="spawner", output="screen", arguments=["parallel_gripper_controller", "--controller-manager", "/controller_manager", *switch_timeout, *service_call_timeout])
    gripper_action = Node(
        package="openeta_rm75_robotiq2f85_sim",
        executable="gripper_action_adapter.py",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "allow_stalling": True,
            # One common driver produces the complete four-bar target vector.
            # Contact changes only its rate/lead; grasp admission remains the
            # independent bilateral-contact + attach-ACK gate in DirectEnv.
            "target_model_name": "target_object",
        }],
    )
    move_group = Node(package="moveit_ros_move_group", executable="move_group", output="screen", parameters=[moveit.to_dict(), {"use_sim_time": True}])
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="screen",
        arguments=[
            "/openeta_rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image", "/openeta_rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image", "/openeta_rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/openeta_wrist_rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image", "/openeta_wrist_rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image", "/openeta_wrist_rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/openeta/native_grasp/contacts/left_pad@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts", "/openeta/native_grasp/contacts/right_pad@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts",
            "/openeta/gripper/left_outer@std_msgs/msg/Float64]gz.msgs.Double", "/openeta/gripper/right_outer@std_msgs/msg/Float64]gz.msgs.Double", "/openeta/gripper/left_inner@std_msgs/msg/Float64]gz.msgs.Double", "/openeta/gripper/right_inner@std_msgs/msg/Float64]gz.msgs.Double", "/openeta/gripper/left_tip@std_msgs/msg/Float64]gz.msgs.Double", "/openeta/gripper/right_tip@std_msgs/msg/Float64]gz.msgs.Double", "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ], parameters=[{"use_sim_time": True}],
    )
    spawn = OpaqueFunction(function=_spawn_robot, kwargs={"xacro_file": xacro_file, "generated_sdfs": generated_sdfs, "post_spawn_actions": [jsb, bridge]})
    return LaunchDescription([
        LogInfo(msg=f"OpenETA qualification solver profile: {solver_profile}"),
        LogInfo(msg=f"OpenETA acceptance scene: {acceptance_scene}"),
        LogInfo(
            msg=(
                "OpenETA authoritative scene: "
                f"{authoritative_scene.authority_sha256} "
                f"({len(authoritative_scene.objects)} MoveIt objects)"
            )
        ),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.pathsep.join([str(share.parent), str(description_share.parent), os.environ.get("GZ_SIM_RESOURCE_PATH", "")])),
        gz_launch, rsp, spawn, RegisterEventHandler(OnShutdown(on_shutdown=_unlink_generated_sdf(generated_sdfs))),
        _after_success(jsb, [arm], "joint_state_broadcaster activation"),
        _after_success(arm, [gripper], "RM75 controller activation"),
        _after_success(gripper, [gripper_action, move_group], "Robotiq controller activation"),
    ])
