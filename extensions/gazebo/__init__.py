"""Gazebo embodiment adapter for observation, motion, and native grasping.

The adapter deliberately keeps Gazebo/ROS optional.  It exposes the same
reset/observe/close lifecycle and :class:`adapter.protocol.EnvObservation`
shape used by the existing simulator boundary.
"""

from .config import GazeboConfig, GazeboObject
from .adapter import GazeboSimulatorAdapter
from .lifecycle import GazeboEnvironment, GazeboLifecycleError
from .process import (GazeboProcess, GazeboProcessError, GazeboWorldControl,
                      Ros2LaunchProcess, RosGzBridgeProcess)
from .observation import (GazeboObservationError, RosRgbdCameraConfig, RosRgbdCameraSource,
                          camera_info_intrinsics, decode_ros_depth, decode_ros_rgb)
try:  # gymnasium is an optional runtime dependency for the worker process
    from .direct_env import GazeboDirectEnv
    from .worker import make_gazebo_direct_env
except ModuleNotFoundError as exc:  # keep asset/contracts importable offline
    if exc.name != "gymnasium":
        raise
    GazeboDirectEnv = None  # type: ignore[assignment]
    make_gazebo_direct_env = None  # type: ignore[assignment]
from .robot_control import (GAZEBO_CONTROL_ENV_ID, MODEL_ID, ARM_JOINTS, GRIPPER_JOINTS, JOINT_NAMES,
                  GazeboControlConfig, Robotiq2F85Calibration,
                  robotiq_aperture_to_angle, robotiq_angle_to_aperture,
                  GazeboControlResult, GazeboController, gripper_state,
                  make_move_group_goal, robot_state_from_sources)
from .ros_control import RosGazeboController, RosGazeboControllerFactory, RosGazeboStateSource
from .native_grasp import PICKPLACE_ENV_ID, PICKPLACE_MODEL_ID, PICKPLACE_DISPLAY_NAME, NativePickPlaceConfig

__all__ = ["GazeboConfig", "GazeboObject", "GazeboEnvironment", "GazeboSimulatorAdapter", "GazeboLifecycleError", "GazeboProcess", "GazeboProcessError", "GazeboWorldControl", "Ros2LaunchProcess", "RosGzBridgeProcess", "GazeboObservationError", "RosRgbdCameraConfig", "RosRgbdCameraSource", "GazeboDirectEnv", "make_gazebo_direct_env", "camera_info_intrinsics", "decode_ros_depth", "decode_ros_rgb"]
__all__ += ["GAZEBO_CONTROL_ENV_ID", "MODEL_ID", "ARM_JOINTS", "GRIPPER_JOINTS", "JOINT_NAMES", "GazeboControlConfig", "GazeboControlResult", "GazeboController", "gripper_state", "make_move_group_goal", "robot_state_from_sources"]
__all__ += ["Robotiq2F85Calibration", "robotiq_aperture_to_angle", "robotiq_angle_to_aperture"]
__all__ += ["RosGazeboController", "RosGazeboControllerFactory", "RosGazeboStateSource"]
__all__ += ["PICKPLACE_ENV_ID", "PICKPLACE_MODEL_ID", "PICKPLACE_DISPLAY_NAME", "NativePickPlaceConfig"]
