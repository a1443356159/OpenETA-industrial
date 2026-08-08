"""OpenETA bench-worker adapter for the documented M1 Gazebo session.

The worker exposes the existing ``gym.Env``-shaped bench contract to
``sim/bench_worker.py``.  It does not add a second MCP server or planner API:
the worker owns one :class:`GazeboLiveSession`, while the parent MCP server
continues to proxy the standard environment tools.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from adapter.protocol import EnvObservation

from .live import GazeboLiveSession, GazeboLiveSessionConfig
from .observation import RosRgbdCameraConfig
from .process import GazeboProcessError


def _json_env(name: str, default: Any = None) -> Any:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc


def live_session_config_from_env() -> GazeboLiveSessionConfig:
    """Build deployment-owned live settings from worker environment variables."""

    extrinsics = _json_env("OPENETA_GAZEBO_CAMERA_EXTRINSICS")
    if not isinstance(extrinsics, dict) or not extrinsics:
        raise ValueError(
            "OPENETA_GAZEBO_CAMERA_EXTRINSICS must be a non-empty JSON object"
        )
    arguments = _json_env(
        "OPENETA_GAZEBO_LAUNCH_ARGUMENTS", ["rviz:=false"]
    )
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise ValueError("OPENETA_GAZEBO_LAUNCH_ARGUMENTS must be a JSON string list")
    return GazeboLiveSessionConfig(
        ros2_executable=os.environ.get("OPENETA_GAZEBO_ROS2_EXECUTABLE", "/opt/ros/jazzy/bin/ros2"),
        gz_executable=os.environ.get(
            "OPENETA_GAZEBO_GZ_EXECUTABLE", "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
        ),
        launch_package=os.environ.get("OPENETA_GAZEBO_LAUNCH_PACKAGE", "ros_gz_sim_demos"),
        launch_file=os.environ.get(
            "OPENETA_GAZEBO_LAUNCH_FILE", "rgbd_camera_bridge.launch.py"
        ),
        launch_arguments=tuple(arguments),
        world_name=os.environ.get("OPENETA_GAZEBO_WORLD", "lidar_sensor"),
        camera=RosRgbdCameraConfig(
            rgb_topic=os.environ.get("OPENETA_GAZEBO_RGB_TOPIC", "/rgbd_camera/image"),
            depth_topic=os.environ.get(
                "OPENETA_GAZEBO_DEPTH_TOPIC", "/rgbd_camera/depth_image"
            ),
            camera_info_topic=os.environ.get(
                "OPENETA_GAZEBO_CAMERA_INFO_TOPIC", "/rgbd_camera/camera_info"
            ),
            frame_id=os.environ.get(
                "OPENETA_GAZEBO_CAMERA_FRAME", "rgbd_camera/link/rgbd_camera"
            ),
            extrinsics=extrinsics,
            depth_units_per_metre=float(
                os.environ.get("OPENETA_GAZEBO_DEPTH_UNITS_PER_METRE", "1000")
            ),
        ),
        startup_settle_s=float(os.environ.get("OPENETA_GAZEBO_STARTUP_SETTLE_S", "8")),
    )


class GazeboWorkerEnv(Env):
    """Read-only M1 environment implementing the existing bench worker API."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, *, task: str = "", seed: int = 0, **_kwargs: Any) -> None:
        self._task = task
        self._seed = seed
        self._session = GazeboLiveSession(live_session_config_from_env(), task=task)
        self._latest: dict[str, Any] | None = None
        self._backend = "gazebo"
        self.openeta_control_spec = {"read_only": True, "m1": True}
        # M1 deliberately exposes no executable robot action space.
        self.action_space = spaces.Discrete(1)

    def _as_unified(self, observation: EnvObservation) -> dict[str, Any]:
        cameras: dict[str, dict[str, Any]] = {}
        for camera in observation.cameras:
            cameras[camera.frame_id] = {
                "rgb": np.asarray(camera.rgb, dtype=np.uint8),
                "depth": (
                    np.asarray(camera.depth, dtype=np.float32)
                    if camera.depth is not None
                    else None
                ),
                "intrinsics": dict(camera.intrinsics),
                "extrinsics": dict(camera.extrinsics),
                "timestamp_s": camera.timestamp_s,
                "role": camera.role,
            }
        return {
            "task": observation.task,
            "cameras": cameras,
            "robot": observation.robot.to_dict(),
            "objects": list(observation.objects),
            "metadata": dict(observation.metadata),
        }

    def refresh_observation(self) -> dict[str, Any]:
        observation = self._session.observe()
        self._latest = self._as_unified(observation)
        return self._latest

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._seed = int(seed)
        if self._latest is None:
            self._session.create()
        observation = self._session.reset(seed=self._seed)
        self._latest = self._as_unified(observation)
        return self._latest, {}

    def step(self, action: Any):
        del action
        raise GazeboProcessError("Gazebo M1 worker is read-only; control is deferred to M2")

    def render(self):
        if self._latest is None:
            self.refresh_observation()
        cameras = self._latest.get("cameras", {}) if self._latest else {}
        first = next(iter(cameras.values()), {})
        return first.get("rgb")

    def close(self) -> None:
        self._session.close()
        self._latest = None
