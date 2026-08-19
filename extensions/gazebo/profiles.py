"""Repository-owned Gazebo profiles and capability declarations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .robot_control import GazeboControlConfig
from .native_grasp import NativePickPlaceConfig
from .observation import RosRgbdCameraConfig


FRESH_OBSERVATION = "fresh_observation"
AUTHORITATIVE_CAMERA = "authoritative_camera"
STRUCTURED_RECEIPT = "structured_receipt"
CONTROL = "control"
PHYSICS = "physics"


_TOP_EXTRINSICS = {
    "frame_transform": "camera_to_world",
    "camera_frame": "opencv",
    "pos": [0.0, 0.0, 1.8],
    "quat_xyzw": [0.7071067812, -0.7071067812, 0.0, 0.0],
}


@dataclass(frozen=True, slots=True)
class GazeboProfile:
    name: str
    launch_package: str
    launch_file: str
    world_name: str
    cameras: tuple[RosRgbdCameraConfig, ...]
    capabilities: frozenset[str]
    model_config: GazeboControlConfig | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Gazebo profile name must be non-empty")
        if self.unavailable_reason is None and (
            not self.launch_package or not self.launch_file or not self.world_name
        ):
            raise ValueError("available Gazebo profile identifiers must be non-empty")
        if not self.cameras:
            raise ValueError("Gazebo profile must declare at least one RGB-D camera")
        if PHYSICS in self.capabilities and CONTROL not in self.capabilities:
            raise ValueError("physics capability requires control")
        if CONTROL in self.capabilities and self.model_config is None:
            raise ValueError("control capability requires a robot model config")


def _camera(*, wrist: bool = False, top_height_m: float = 1.8) -> RosRgbdCameraConfig:
    prefix = "/openeta_wrist_rgbd" if wrist else "/openeta_rgbd"
    return RosRgbdCameraConfig(
        rgb_topic=f"{prefix}/image",
        depth_topic=f"{prefix}/depth_image",
        camera_info_topic=f"{prefix}/camera_info",
        frame_id="wrist_camera_optical_frame" if wrist else "top_camera_optical_frame",
        extrinsics=(
            {"frame_transform": "tf_dynamic", "camera_frame": "opencv"}
            if wrist
            else {**_TOP_EXTRINSICS, "pos": [0.0, 0.0, float(top_height_m)]}
        ),
        role="wrist" if wrist else "scene_primary",
    )


_BASE = frozenset({FRESH_OBSERVATION, AUTHORITATIVE_CAMERA})
_PROFILES: Mapping[str, GazeboProfile] = MappingProxyType({
    "rgbd_observation": GazeboProfile(
        # The installed demo launch opens Gazebo's GUI, which aborts on
        # headless workers.  The repository-owned equivalent starts the same
        # RGB-D world server-only and keeps its official bridge contract.
        name="rgbd_observation", launch_package="openeta_rm75_robotiq2f85_sim",
        launch_file="gazebo_rgbd.launch.py", world_name="lidar_sensor",
        cameras=(RosRgbdCameraConfig(
            rgb_topic="/rgbd_camera/image", depth_topic="/rgbd_camera/depth_image",
            camera_info_topic="/rgbd_camera/camera_info",
            frame_id="rgbd_camera/link/rgbd_camera", extrinsics=dict(_TOP_EXTRINSICS),
        ),), capabilities=_BASE,
    ),
    "rm75_robotiq2f85_control": GazeboProfile(
        name="rm75_robotiq2f85_control", launch_package="openeta_rm75_robotiq2f85_sim",
        launch_file="gazebo_moveit.launch.py", world_name="rm75_robotiq2f85",
        cameras=(_camera(), _camera(wrist=True)),
        capabilities=_BASE | {CONTROL, STRUCTURED_RECEIPT}, model_config=GazeboControlConfig(),
    ),
    "rm75_robotiq2f85_pickplace": GazeboProfile(
        name="rm75_robotiq2f85_pickplace", launch_package="openeta_rm75_robotiq2f85_sim",
        launch_file="gazebo_pickplace.launch.py", world_name="rm75_robotiq2f85_pickplace",
        cameras=(_camera(top_height_m=1.3), _camera(wrist=True)),
        capabilities=_BASE | {CONTROL, STRUCTURED_RECEIPT, PHYSICS}, model_config=NativePickPlaceConfig(),
    ),
})


def gazebo_profile(name: str) -> GazeboProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown Gazebo profile: {name}") from exc


def gazebo_profiles() -> Mapping[str, GazeboProfile]:
    return _PROFILES
