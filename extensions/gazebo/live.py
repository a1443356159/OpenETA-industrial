"""Composition of the documented ROS 2 launch, RGB-D source, and reset API."""

from __future__ import annotations

import time
from dataclasses import dataclass

from adapter.protocol import EnvObservation, RobotState

from .observation import RosRgbdCameraConfig, RosRgbdCameraSource
from .process import GazeboProcessError, GazeboWorldControl, Ros2LaunchProcess


@dataclass(slots=True)
class GazeboLiveSessionConfig:
    """Deployment-owned launch/world/camera settings for one live session."""

    ros2_executable: str
    gz_executable: str
    launch_package: str
    launch_file: str
    launch_arguments: tuple[str, ...]
    world_name: str
    camera: RosRgbdCameraConfig
    # Additional streams (for M2 this is the wrist camera).  ``camera`` is
    # retained as the primary/top camera for backwards compatibility.
    additional_cameras: tuple[RosRgbdCameraConfig, ...] = ()
    startup_settle_s: float = 5.0
    observation_timeout_s: float = 8.0


class GazeboLiveSession:
    """M1 live lifecycle facade; no Planner-facing ROS APIs are exposed."""

    def __init__(self, config: GazeboLiveSessionConfig, *, task: str = "") -> None:
        self.config = config
        self.task = task
        self._launch = Ros2LaunchProcess(
            package=config.launch_package, launch_file=config.launch_file,
            arguments=config.launch_arguments, ros2_executable=config.ros2_executable,
            startup_timeout_s=max(5.0, config.startup_settle_s),
        )
        camera_configs = [config.camera, *config.additional_cameras]
        self._cameras = [
            RosRgbdCameraSource(item, node_name=f"openeta_rgbd_camera_{index}")
            for index, item in enumerate(camera_configs)
        ]
        self._world = GazeboWorldControl(world_name=config.world_name, gz_executable=config.gz_executable)
        self._created = False
        self._closed = False
        self._epoch = 0

    def create(self) -> EnvObservation:
        if self._closed:
            raise GazeboProcessError("live session is closed")
        if not self._created:
            self._launch.start()
            time.sleep(max(0.0, self.config.startup_settle_s))
            for camera in self._cameras:
                camera.start()
            self._created = True
        return self.observe()

    def reset(
        self, *, seed: int | None = None, preserve_sim_time: bool = False
    ) -> EnvObservation:
        if not self._created or self._closed:
            raise GazeboProcessError("live session must be created before reset")
        if preserve_sim_time:
            self._world.reset_models(seed=seed)
        else:
            self._world.reset_all(seed=seed)
        self._epoch += 1
        observation = self.observe()
        observation.metadata.update({"scene_epoch": self._epoch, "reset_seed": seed,
                                    "observation_provenance": "gazebo_ros_live"})
        if self.task:
            observation.task = self.task
        return observation

    def observe(
        self,
        *,
        min_camera_timestamp_s: float | None = None,
        min_received_monotonic_s: float | None = None,
    ) -> EnvObservation:
        if not self._created or self._closed:
            raise GazeboProcessError("live session must be created before observe")
        return EnvObservation(
            task=self.task,
            cameras=[
                camera.capture(
                    timeout_s=self.config.observation_timeout_s,
                    min_timestamp_s=min_camera_timestamp_s,
                    min_received_monotonic_s=min_received_monotonic_s,
                )
                for camera in self._cameras
            ],
            robot=RobotState(),
            metadata={"backend": "gazebo", "observation_provenance": "gazebo_ros_live",
                      "scene_epoch": self._epoch},
        )

    def close(self) -> None:
        if self._closed:
            return
        for camera in self._cameras:
            camera.close()
        self._launch.close()
        self._closed = True
        self._created = False
