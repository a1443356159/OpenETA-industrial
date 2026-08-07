from __future__ import annotations

import os
import shutil
import threading
import time

import numpy as np
import pytest

from extensions.gazebo import (
    GazeboProcess,
    RosGzBridgeProcess,
    RosRgbdCameraConfig,
    RosRgbdCameraSource,
)


def test_live_ros_rgbd_source_builds_camera_frame() -> None:
    if os.environ.get("OPENETA_RUN_LIVE_ROS_TEST") != "1":
        pytest.skip("opt-in: set OPENETA_RUN_LIVE_ROS_TEST=1 for live ROS discovery")
    rclpy = pytest.importorskip("rclpy")
    CameraInfo = pytest.importorskip("sensor_msgs.msg").CameraInfo
    gz = shutil.which("gz") or ("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz" if os.path.exists("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz") else None)
    if gz is None:
        pytest.skip("Gazebo Sim is not installed")
    ros2 = "/opt/ros/jazzy/bin/ros2"
    if not os.path.exists(ros2):
        pytest.skip("ROS 2 CLI is not installed")

    sim = GazeboProcess(world="extensions/gazebo/worlds/m1_rgbd.sdf", gz_executable=gz)
    bridge = RosGzBridgeProcess(
        ros2_executable=ros2,
        topics=(
            "/top_camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/top_camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
        ),
    )
    rclpy.init()
    info_node = rclpy.create_node("openeta_camera_info_fixture")
    info_pub = info_node.create_publisher(CameraInfo, "/top_camera/camera_info", 10)
    source = RosRgbdCameraSource(RosRgbdCameraConfig(
        rgb_topic="/top_camera/image",
        depth_topic="/top_camera/depth_image",
        camera_info_topic="/top_camera/camera_info",
        frame_id="top_camera_optical_frame",
        extrinsics={"frame_transform": "camera_to_world", "camera_frame": "opencv",
                    "pos": [0.0, 0.0, 1.5], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
    ))
    stop = threading.Event()

    def publish_info() -> None:
        message = CameraInfo()
        message.width, message.height = 64, 48
        message.k = [55.4, 0.0, 32.0, 0.0, 55.4, 24.0, 0.0, 0.0, 1.0]
        while not stop.is_set():
            info_pub.publish(message)
            # Publishing does not require spinning the publisher node. The
            # source node owns the single-threaded executor used for capture.
            time.sleep(0.05)

    thread = None
    try:
        sim.start()
        sim.wait_for_topics(("/top_camera/image", "/top_camera/depth_image"))
        bridge.start()
        source.start()
        thread = threading.Thread(target=publish_info, daemon=True)
        thread.start()
        frame = source.capture(timeout_s=8.0)
        assert len(frame.rgb) == 48 and len(frame.rgb[0]) == 64
        assert len(frame.depth) == 48 and len(frame.depth[0]) == 64
        assert frame.intrinsics["fx"] == pytest.approx(55.4)
        assert frame.extrinsics["camera_frame"] == "opencv"
        assert np.isfinite(np.asarray(frame.depth, dtype=np.float32)).all()
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=2.0)
        source.close()
        info_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        bridge.close()
        sim.close()
