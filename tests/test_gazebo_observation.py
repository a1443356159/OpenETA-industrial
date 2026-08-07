from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from extensions.gazebo import GazeboObservationError, camera_info_intrinsics, decode_ros_depth, decode_ros_rgb


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

