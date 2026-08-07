"""Configuration for the optional Gazebo embodiment adapter."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GazeboObject:
    """Deterministic oracle object summary (SI units, world frame)."""

    name: str
    label: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class GazeboConfig:
    """Backend-neutral settings; no ROS/Gazebo values are hard-coded in code."""

    world: str = "industrial_pick_place"
    robot_name: str = "panda"
    top_camera_name: str = "top"
    image_width: int = 64
    image_height: int = 64
    camera_frame: str = "top_camera_optical_frame"
    objects: tuple[GazeboObject, ...] = field(default_factory=lambda: (
        GazeboObject("silver_screw", "silver screw", (0.42, -0.11, 0.03)),
        GazeboObject("bin_3", "storage bin", (0.61, 0.20, 0.08)),
    ))

