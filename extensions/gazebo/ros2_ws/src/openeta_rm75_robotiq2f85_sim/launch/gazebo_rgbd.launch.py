"""Start the installed RGB-D demo server-only for headless observation-only workers."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ros_gz_sim = Path(get_package_share_directory("ros_gz_sim"))
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(ros_gz_sim / "launch" / "gz_sim.launch.py")),
        # The installed sensors_demo.sdf declares the ``lidar_sensor`` world.
        # ``-s --headless-rendering`` keeps the RGB-D renderer alive without
        # starting Qt; plain server-only mode segfaults remotely during EGL
        # initialization before the world-control service is advertised.
        launch_arguments={"gz_args": "-r -s --headless-rendering sensors_demo.sdf"}.items(),
    )
    # ``parameter_bridge`` advertises an Image bridge in this deployment but
    # does not deliver raw image packets from the headless renderer.  The
    # official ``ros_gz_image`` bridge is the native image transport for RGB
    # and depth; retain a directed official bridge for CameraInfo, which the
    # image bridge intentionally does not publish.  All three messages remain
    # Gazebo-originated ROS packets--there is no rendered-frame fallback.
    image_bridge = Node(
        package="ros_gz_image", executable="image_bridge", output="screen",
        arguments=["rgbd_camera/image", "rgbd_camera/depth_image"],
    )
    camera_info_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="screen",
        arguments=[
            "/rgbd_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
    )
    return LaunchDescription([gazebo, image_bridge, camera_info_bridge])
