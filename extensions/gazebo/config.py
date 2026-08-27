"""Configuration for the optional Gazebo embodiment adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True, slots=True)
class GazeboObject:
    """Deterministic fixture object summary (SI units, world frame)."""

    name: str
    label: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.label.strip():
            raise ValueError("GazeboObject name and label must be non-empty")
        if len(self.position) != 3 or not all(math.isfinite(float(v)) for v in self.position):
            raise ValueError("GazeboObject position must be three finite values in metres")
        if len(self.orientation_xyzw) != 4 or not all(
            math.isfinite(float(v)) for v in self.orientation_xyzw
        ):
            raise ValueError("GazeboObject orientation must be four finite xyzw values")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("GazeboObject confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class GazeboConfig:
    """Backend-neutral settings; no ROS/Gazebo values are hard-coded in code."""

    world: str = "industrial_pick_place"
    robot_name: str = "panda"
    top_camera_name: str = "top"
    image_width: int = 64
    image_height: int = 64
    camera_frame: str = "top_camera_optical_frame"
    top_rgb_topic: str = "/top_camera/image"
    top_depth_topic: str = "/top_camera/depth_image"
    top_camera_info_topic: str = "/top_camera/camera_info"
    objects: tuple[GazeboObject, ...] = field(default_factory=lambda: (
        GazeboObject("silver_screw", "silver screw", (0.42, -0.11, 0.03)),
        GazeboObject("bin_3", "storage bin", (0.61, 0.20, 0.08)),
    ))

    def __post_init__(self) -> None:
        if not self.world.strip() or not self.robot_name.strip() or not self.top_camera_name.strip():
            raise ValueError("Gazebo world, robot, and camera names must be non-empty")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Gazebo camera dimensions must be positive")
        if not self.camera_frame.strip():
            raise ValueError("Gazebo camera_frame must be non-empty")
        for field_name in ("top_rgb_topic", "top_depth_topic", "top_camera_info_topic"):
            topic = getattr(self, field_name)
            if not topic.startswith("/"):
                raise ValueError(f"{field_name} must be an absolute ROS topic")
