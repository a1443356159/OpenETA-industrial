"""Start the installed RGB-D demo server-only for headless M1 workers."""

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
        # ``-s`` prevents a Qt GUI process from aborting headless cloud runs.
        launch_arguments={"gz_args": "-r -s sensors_demo.sdf"}.items(),
    )
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="screen",
        arguments=[
            "/camera@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/rgbd_camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/rgbd_camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/rgbd_camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/rgbd_camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
        ],
        remappings=[("/camera", "/camera/image"), ("/camera_info", "/camera/camera_info")],
    )
    return LaunchDescription([gazebo, bridge])
