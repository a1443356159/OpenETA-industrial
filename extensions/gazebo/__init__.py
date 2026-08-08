"""Minimal Gazebo embodiment adapter for the OpenETA M1 milestone.

The adapter deliberately keeps Gazebo/ROS optional.  It exposes the same
reset/observe/close lifecycle and :class:`adapter.protocol.EnvObservation`
shape used by the existing simulator boundary.  A transport implementation
can later replace the deterministic oracle provider without changing callers.
"""

from .config import GazeboConfig, GazeboObject
from .adapter import GazeboSimulatorAdapter
from .lifecycle import GazeboEnvironment, GazeboLifecycleError
from .mcp import GazeboLiveMcpTransport, GazeboOracleMcpTransport
from .process import (GazeboProcess, GazeboProcessError, GazeboWorldControl,
                      Ros2LaunchProcess, RosGzBridgeProcess)
from .observation import (GazeboObservationError, RosRgbdCameraConfig, RosRgbdCameraSource,
                          camera_info_intrinsics, decode_ros_depth, decode_ros_rgb)
from .live import GazeboLiveSession, GazeboLiveSessionConfig
from .worker import GazeboWorkerEnv, live_session_config_from_env

__all__ = ["GazeboConfig", "GazeboObject", "GazeboEnvironment", "GazeboSimulatorAdapter", "GazeboLifecycleError", "GazeboOracleMcpTransport", "GazeboLiveMcpTransport", "GazeboProcess", "GazeboProcessError", "GazeboWorldControl", "Ros2LaunchProcess", "RosGzBridgeProcess", "GazeboObservationError", "RosRgbdCameraConfig", "RosRgbdCameraSource", "GazeboLiveSession", "GazeboLiveSessionConfig", "GazeboWorkerEnv", "live_session_config_from_env", "camera_info_intrinsics", "decode_ros_depth", "decode_ros_rgb"]
