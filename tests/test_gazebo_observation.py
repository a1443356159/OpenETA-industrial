from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from extensions.gazebo import (
    GazeboObservationError,
    RosRgbdCameraConfig,
    RosRgbdCameraSource,
    camera_info_intrinsics,
    decode_ros_depth,
    decode_ros_rgb,
)
from extensions.gazebo.observation import _tf_camera_to_world_extrinsics


def _image(*, encoding: str, array: np.ndarray, step: int | None = None):
    array = np.asarray(array)
    return SimpleNamespace(
        height=array.shape[0], width=array.shape[1], encoding=encoding,
        step=step or array.strides[0], data=array.tobytes(),
    )


def test_decode_ros_rgb_and_depth_to_openeta_units() -> None:
    rgb = decode_ros_rgb(_image(encoding="bgr8", array=np.array([[[3, 2, 1]]], dtype=np.uint8)))
    assert rgb.tolist() == [[[1, 2, 3]]]
    depth = decode_ros_depth(_image(encoding="16UC1", array=np.array([[500, 0]], dtype=np.uint16)))
    assert depth.tolist() == [[0.5, 0.0]]


def test_camera_info_and_invalid_ros_packets_fail_closed() -> None:
    info = SimpleNamespace(width=640, height=480, k=[500, 0, 320, 0, 500, 240, 0, 0, 1])
    assert camera_info_intrinsics(info)["fx"] == 500.0
    with pytest.raises(GazeboObservationError, match="unsupported"):
        decode_ros_rgb(_image(encoding="mono8", array=np.zeros((1, 1), dtype=np.uint8)))
    with pytest.raises(GazeboObservationError, match="focal"):
        camera_info_intrinsics(SimpleNamespace(width=1, height=1, k=[0] * 9))


def test_dynamic_gazebo_camera_tf_is_resolved_to_opencv_camera_to_world() -> None:
    transform = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.35, y=-0.05, z=0.996),
            # A Gazebo top camera is +pi/2 around world Y.
            rotation=SimpleNamespace(
                x=0.0,
                y=2**-0.5,
                z=0.0,
                w=2**-0.5,
            ),
        )
    )

    extrinsics = _tf_camera_to_world_extrinsics(
        transform,
        reference_frame="base_link",
        sensor_frame="wrist_camera_optical_frame",
        timestamp_s=12.5,
    )

    assert extrinsics["frame_transform"] == "camera_to_world"
    assert extrinsics["camera_frame"] == "opencv"
    assert extrinsics["pos"] == [0.35, -0.05, 0.996]
    assert extrinsics["quat_xyzw"] == pytest.approx(
        [2**-0.5, -(2**-0.5), 0.0, 0.0]
    )
    assert extrinsics["calibration_source"] == "tf2_at_rgb_timestamp"
    assert extrinsics["timestamp_s"] == 12.5


def _stamped_image(*, encoding: str, array: np.ndarray, timestamp: float):
    message = _image(encoding=encoding, array=array)
    seconds = int(timestamp)
    message.header = SimpleNamespace(
        stamp=SimpleNamespace(sec=seconds, nanosec=int((timestamp - seconds) * 1e9))
    )
    return message


def test_live_camera_capture_consumes_new_rgb_and_depth_timestamps(monkeypatch) -> None:
    source = RosRgbdCameraSource(
        RosRgbdCameraConfig(
            rgb_topic="/rgb",
            depth_topic="/depth",
            camera_info_topic="/info",
            frame_id="camera",
            extrinsics={"frame_transform": "camera_to_world"},
        )
    )
    source._node = object()
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        SimpleNamespace(spin_once=lambda _node, timeout_sec=0.0: None),
    )
    source._info_callback(
        SimpleNamespace(width=1, height=1, k=[100.0, 0, 0.5, 0, 100.0, 0.5, 0, 0, 1])
    )
    source._rgb_callback(
        _stamped_image(
            encoding="rgb8", array=np.array([[[1, 2, 3]]], dtype=np.uint8), timestamp=1.0
        )
    )
    source._depth_callback(
        _stamped_image(
            encoding="16UC1", array=np.array([[500]], dtype=np.uint16), timestamp=1.0
        )
    )
    first = source.capture(timeout_s=0.01)
    assert first.timestamp_s == 1.0
    assert isinstance(first.rgb, np.ndarray)
    assert isinstance(first.depth, np.ndarray)
    with pytest.raises(GazeboObservationError, match="fresh"):
        source.capture(timeout_s=0.01)

    source._rgb_callback(
        _stamped_image(
            encoding="rgb8", array=np.array([[[4, 5, 6]]], dtype=np.uint8), timestamp=2.0
        )
    )
    source._depth_callback(
        _stamped_image(
            encoding="16UC1", array=np.array([[600]], dtype=np.uint16), timestamp=2.0
        )
    )
    assert source.capture(timeout_s=0.01, min_timestamp_s=1.5).timestamp_s == 2.0


def test_live_camera_fails_closed_when_only_one_stream_advances(monkeypatch) -> None:
    source = RosRgbdCameraSource(
        RosRgbdCameraConfig(
            rgb_topic="/rgb",
            depth_topic="/depth",
            camera_info_topic="/info",
            frame_id="camera",
            extrinsics={"frame_transform": "camera_to_world"},
        )
    )
    source._node = object()
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        SimpleNamespace(spin_once=lambda _node, timeout_sec=0.0: None),
    )
    source._info_callback(
        SimpleNamespace(width=1, height=1, k=[100.0, 0, 0.5, 0, 100.0, 0.5, 0, 0, 1])
    )
    rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
    depth = np.array([[500]], dtype=np.uint16)
    source._rgb_callback(_stamped_image(encoding="rgb8", array=rgb, timestamp=1.0))
    source._depth_callback(_stamped_image(encoding="16UC1", array=depth, timestamp=1.0))
    source.capture(timeout_s=0.01)
    source._rgb_callback(_stamped_image(encoding="rgb8", array=rgb, timestamp=2.0))
    with pytest.raises(GazeboObservationError, match="fresh"):
        source.capture(timeout_s=0.01)
