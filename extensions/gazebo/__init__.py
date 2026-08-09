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
try:  # gymnasium is an optional runtime dependency for the worker process
    from .worker import GazeboM2WorkerEnv, GazeboM3WorkerEnv, GazeboRobotiq2F85WorkerEnv, GazeboWorkerEnv, live_session_config_from_env
except ModuleNotFoundError as exc:  # keep asset/contracts importable offline
    if exc.name != "gymnasium":
        raise
    GazeboM2WorkerEnv = GazeboM3WorkerEnv = GazeboRobotiq2F85WorkerEnv = GazeboWorkerEnv = None  # type: ignore[assignment]
    live_session_config_from_env = None  # type: ignore[assignment]
from .m2 import (M2_ENV_ID, MODEL_ID, ARM_JOINTS, GRIPPER_JOINTS, JOINT_NAMES,
                  ROBOTIQ2F85_ENV_ID, ROBOTIQ2F85_MODEL_ID, M2Config, Robotiq2F85Config,
                  Robotiq2F85Calibration, robotiq_aperture_to_angle, robotiq_angle_to_aperture,
                  M2ControlResult, M2Controller, gripper_state,
                  make_move_group_goal, robot_state_from_sources)
from .ros_control import RosM2Controller, RosM2ControllerFactory, RosM2StateSource
from .m3 import (M3_ENV_ID, M3_MODEL_ID, M3_DISPLAY_NAME, M3Config, M3Verifier,
                 PhysicsSnapshot, VerificationRecord, Verdict, ReasonCode)

__all__ = ["GazeboConfig", "GazeboObject", "GazeboEnvironment", "GazeboSimulatorAdapter", "GazeboLifecycleError", "GazeboOracleMcpTransport", "GazeboLiveMcpTransport", "GazeboProcess", "GazeboProcessError", "GazeboWorldControl", "Ros2LaunchProcess", "RosGzBridgeProcess", "GazeboObservationError", "RosRgbdCameraConfig", "RosRgbdCameraSource", "GazeboLiveSession", "GazeboLiveSessionConfig", "GazeboWorkerEnv", "live_session_config_from_env", "camera_info_intrinsics", "decode_ros_depth", "decode_ros_rgb"]
__all__ += ["M2_ENV_ID", "MODEL_ID", "ARM_JOINTS", "GRIPPER_JOINTS", "JOINT_NAMES", "M2Config", "M2ControlResult", "M2Controller", "gripper_state", "make_move_group_goal", "robot_state_from_sources"]
__all__ += ["ROBOTIQ2F85_ENV_ID", "ROBOTIQ2F85_MODEL_ID", "Robotiq2F85Config", "Robotiq2F85Calibration", "robotiq_aperture_to_angle", "robotiq_angle_to_aperture"]
__all__ += ["GazeboM2WorkerEnv", "GazeboRobotiq2F85WorkerEnv"]
__all__ += ["RosM2Controller", "RosM2ControllerFactory", "RosM2StateSource"]
__all__ += ["M3_ENV_ID", "M3_MODEL_ID", "M3_DISPLAY_NAME", "M3Config", "M3Verifier", "PhysicsSnapshot", "VerificationRecord", "Verdict", "ReasonCode", "GazeboM3WorkerEnv"]
