"""ROS RGB-D to OpenETA observation conversion.

Only standard ``sensor_msgs/Image`` and ``sensor_msgs/CameraInfo`` fields are
used.  Depth conversion is explicit: ``32FC1`` is metres and ``16UC1`` uses a
configurable units-per-metre scale (default millimetres).  No frame convention
or extrinsics are inferred from a ROS topic name.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from adapter.protocol import CameraFrame


class GazeboObservationError(RuntimeError):
    """Raised when a ROS observation cannot satisfy the OpenETA contract."""


def _message_stamp(message: Any) -> float | None:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return None
    sec, nanosec = int(getattr(stamp, "sec", 0)), int(getattr(stamp, "nanosec", 0))
    return float(sec) + float(nanosec) * 1e-9


def _image_array(message: Any) -> np.ndarray:
    height, width = int(message.height), int(message.width)
    step, encoding = int(message.step), str(message.encoding).lower()
    raw = bytes(message.data)
    if height <= 0 or width <= 0 or step <= 0:
        raise GazeboObservationError("ROS image dimensions and step must be positive")
    if len(raw) < height * step:
        raise GazeboObservationError("ROS image data is shorter than height*step")
    channels = {"rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3),
                "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4)}
    if encoding in channels:
        dtype, count = channels[encoding]
        row_bytes = width * count
        if step < row_bytes:
            raise GazeboObservationError("ROS image step is shorter than packed RGB row")
        rows = np.frombuffer(raw, dtype=dtype).reshape(height, step)[:, :row_bytes]
        array = rows.reshape(height, width, count)
        if encoding in {"bgr8", "bgra8"}:
            array = array[..., ::-1]
        return array[..., :3].copy()
    if encoding in {"32fc1", "32fc"}:
        dtype, item_size = np.float32, 4
    elif encoding in {"16uc1", "16uc"}:
        dtype, item_size = np.uint16, 2
    else:
        raise GazeboObservationError(f"unsupported ROS image encoding: {message.encoding}")
    row_bytes = width * item_size
    if step < row_bytes:
        raise GazeboObservationError("ROS depth step is shorter than packed row")
    rows = np.frombuffer(raw, dtype=dtype).reshape(height, step // item_size)[:, :width]
    return rows.copy()


def decode_ros_rgb(message: Any) -> np.ndarray:
    """Decode a ROS colour image to contiguous RGB uint8 pixels."""

    array = _image_array(message)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise GazeboObservationError("ROS colour image did not decode to HxWx3 uint8")
    return array


def decode_ros_depth(message: Any, *, units_per_metre: float = 1000.0) -> np.ndarray:
    """Decode ROS depth to finite non-negative float32 metres."""

    array = _image_array(message)
    encoding = str(message.encoding).lower()
    if array.ndim != 2:
        raise GazeboObservationError("ROS depth image must be single-channel")
    if encoding in {"16uc1", "16uc"}:
        if not math.isfinite(units_per_metre) or units_per_metre <= 0:
            raise GazeboObservationError("depth units_per_metre must be positive")
        array = array.astype(np.float32) / float(units_per_metre)
    else:
        array = array.astype(np.float32)
    return np.where(np.isfinite(array) & (array >= 0.0), array, 0.0).astype(np.float32)


def camera_info_intrinsics(message: Any) -> dict[str, float | int]:
    """Extract the pinhole intrinsics from standard CameraInfo ``K``."""

    k = list(getattr(message, "k", ()))
    if len(k) < 9:
        raise GazeboObservationError("CameraInfo.K must contain nine values")
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    if not all(math.isfinite(v) and v > 0 for v in (fx, fy)):
        raise GazeboObservationError("CameraInfo focal lengths must be positive")
    if not all(math.isfinite(v) for v in (cx, cy)):
        raise GazeboObservationError("CameraInfo principal point must be finite")
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": int(message.width), "height": int(message.height)}


@dataclass(slots=True)
class RosRgbdCameraConfig:
    """ROS topic/calibration settings required for a live camera packet."""

    rgb_topic: str
    depth_topic: str
    camera_info_topic: str
    frame_id: str
    extrinsics: dict[str, Any]
    depth_units_per_metre: float = 1000.0


class RosRgbdCameraSource:
    """Optional rclpy subscriber producing one OpenETA ``CameraFrame``."""

    def __init__(self, config: RosRgbdCameraConfig, *, node_name: str = "openeta_rgbd_camera") -> None:
        self.config = config
        self.node_name = node_name
        self._node: Any | None = None
        self._owns_context = False
        self._rgb: Any | None = None
        self._depth: Any | None = None
        self._info: Any | None = None

    def start(self) -> None:
        if not self.config.extrinsics:
            raise GazeboObservationError("live camera requires explicit camera-to-world extrinsics")
        try:
            import rclpy
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:
            raise GazeboObservationError("rclpy and sensor_msgs are required for live observation") from exc
        if not rclpy.ok():
            rclpy.init()
            self._owns_context = True
        self._node = rclpy.create_node(self.node_name)
        self._node.create_subscription(Image, self.config.rgb_topic, lambda msg: setattr(self, "_rgb", msg), 10)
        self._node.create_subscription(Image, self.config.depth_topic, lambda msg: setattr(self, "_depth", msg), 10)
        self._node.create_subscription(CameraInfo, self.config.camera_info_topic, lambda msg: setattr(self, "_info", msg), 10)

    def capture(self, *, timeout_s: float = 2.0) -> CameraFrame:
        if self._node is None:
            raise GazeboObservationError("camera source must be started before capture")
        import rclpy
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
            if self._rgb is not None and self._depth is not None and self._info is not None:
                rgb, depth, info = self._rgb, self._depth, self._info
                stamp = _message_stamp(rgb) or _message_stamp(depth) or time.time()
                return CameraFrame(
                    frame_id=self.config.frame_id,
                    role="scene_primary",
                    rgb=decode_ros_rgb(rgb).tolist(),
                    depth=decode_ros_depth(depth, units_per_metre=self.config.depth_units_per_metre).tolist(),
                    intrinsics=camera_info_intrinsics(info),
                    extrinsics=dict(self.config.extrinsics),
                    timestamp_s=stamp,
                )
        raise GazeboObservationError("timed out waiting for RGB, depth, and CameraInfo")

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_context:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
            self._owns_context = False

