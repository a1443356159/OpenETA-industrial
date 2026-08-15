"""Immutable deployment settings for the Gazebo worker process.

This module is the only production Gazebo module allowed to translate the
worker environment into runtime configuration.  The translation is cached:
changing ``os.environ`` after the first call cannot reconfigure a live ROS
graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import shutil
import threading
from types import MappingProxyType
from typing import Any, Mapping


_CONFIG_ENV_NAMES = (
    "ROS_DOMAIN_ID",
    "GZ_PARTITION",
    "RMW_IMPLEMENTATION",
    "AMENT_PREFIX_PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "OPENETA_GAZEBO_ROS2_EXECUTABLE",
    "OPENETA_GAZEBO_GZ_EXECUTABLE",
    "OPENETA_GAZEBO_OVERLAY",
    "OPENETA_GAZEBO_CAMERA_EXTRINSICS",
    "OPENETA_GAZEBO_LAUNCH_ARGUMENTS",
    "OPENETA_GAZEBO_WORLD",
    "OPENETA_GAZEBO_STARTUP_TIMEOUT_S",
    "OPENETA_GAZEBO_OBSERVATION_TIMEOUT_S",
    # Deliberately keep this retired name in the environment snapshot so an
    # old worker fails loudly instead of silently selecting an obsolete M3
    # grasp implementation.
    "OPENETA_M3_ATTACHMENT_MODE",
)


def _json_value(snapshot: Mapping[str, str], name: str, default: Any) -> Any:
    raw = snapshot.get(name, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc


@dataclass(frozen=True, slots=True)
class GazeboDeploymentConfig:
    """Settings fixed for the complete lifetime of one bench worker."""

    ros_domain_id: int
    gz_partition: str
    ros2_executable: str
    gz_executable: str
    overlay: str | None = None
    rmw_implementation: str | None = None
    camera_extrinsics: Mapping[str, Any] = field(default_factory=dict)
    launch_arguments: tuple[str, ...] = ()
    world_override: str | None = None
    startup_timeout_s: float = 30.0
    observation_timeout_s: float = 30.0
    process_environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ros_domain_id < 0 or self.ros_domain_id > 232:
            raise ValueError("ROS_DOMAIN_ID must be in [0, 232]")
        if not self.gz_partition.strip():
            raise ValueError("GZ_PARTITION must be non-empty")
        if self.startup_timeout_s <= 0 or self.observation_timeout_s <= 0:
            raise ValueError("Gazebo deadlines must be positive")
        if any(not isinstance(item, str) for item in self.launch_arguments):
            raise ValueError("Gazebo launch arguments must be strings")
        object.__setattr__(self, "camera_extrinsics", MappingProxyType(dict(self.camera_extrinsics)))
        object.__setattr__(self, "process_environment", MappingProxyType(dict(self.process_environment)))

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "GazeboDeploymentConfig":
        source = os.environ if environment is None else environment
        snapshot = {name: str(source[name]) for name in _CONFIG_ENV_NAMES if name in source}
        if "OPENETA_M3_ATTACHMENT_MODE" in snapshot:
            raise ValueError(
                "OPENETA_M3_ATTACHMENT_MODE is retired; M3 always uses "
                "bilateral_contact_adhesion_v1"
            )
        launch_arguments = _json_value(snapshot, "OPENETA_GAZEBO_LAUNCH_ARGUMENTS", [])
        extrinsics = _json_value(snapshot, "OPENETA_GAZEBO_CAMERA_EXTRINSICS", {})
        if not isinstance(launch_arguments, list):
            raise ValueError("OPENETA_GAZEBO_LAUNCH_ARGUMENTS must be a JSON string list")
        if not isinstance(extrinsics, dict):
            raise ValueError("OPENETA_GAZEBO_CAMERA_EXTRINSICS must be a JSON object")
        domain = int(snapshot.get("ROS_DOMAIN_ID", "0"))
        partition = snapshot.get("GZ_PARTITION") or f"openeta-{os.getpid()}-{domain}"
        overlay = snapshot.get("OPENETA_GAZEBO_OVERLAY") or None
        child_env = dict(source)
        child_env["ROS_DOMAIN_ID"] = str(domain)
        child_env["GZ_PARTITION"] = partition
        child_env["ROS2CLI_NO_DAEMON"] = "1"
        if overlay:
            prefixes = [overlay, *filter(None, child_env.get("AMENT_PREFIX_PATH", "").split(os.pathsep))]
            child_env["AMENT_PREFIX_PATH"] = os.pathsep.join(dict.fromkeys(prefixes))
        return cls(
            ros_domain_id=domain,
            gz_partition=partition,
            ros2_executable=snapshot.get("OPENETA_GAZEBO_ROS2_EXECUTABLE", shutil.which("ros2") or "ros2"),
            gz_executable=snapshot.get("OPENETA_GAZEBO_GZ_EXECUTABLE", shutil.which("gz") or "gz"),
            overlay=overlay,
            rmw_implementation=snapshot.get("RMW_IMPLEMENTATION") or None,
            camera_extrinsics=extrinsics,
            launch_arguments=tuple(launch_arguments),
            world_override=snapshot.get("OPENETA_GAZEBO_WORLD") or None,
            startup_timeout_s=float(snapshot.get("OPENETA_GAZEBO_STARTUP_TIMEOUT_S", "30")),
            observation_timeout_s=float(snapshot.get("OPENETA_GAZEBO_OBSERVATION_TIMEOUT_S", "30")),
            process_environment=child_env,
        )


_deployment_lock = threading.Lock()
_deployment: GazeboDeploymentConfig | None = None


def worker_deployment_config() -> GazeboDeploymentConfig:
    """Return the worker-wide configuration, parsing its environment once."""
    global _deployment
    with _deployment_lock:
        if _deployment is None:
            _deployment = GazeboDeploymentConfig.from_environment()
        return _deployment
