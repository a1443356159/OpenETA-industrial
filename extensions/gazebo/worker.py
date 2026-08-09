"""OpenETA bench-worker adapter for the documented M1 Gazebo session.

The worker exposes the existing ``gym.Env``-shaped bench contract to
``sim/bench_worker.py``.  It does not add a second MCP server or planner API:
the worker owns one :class:`GazeboLiveSession`, while the parent MCP server
continues to proxy the standard environment tools.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Mapping

import numpy as np
from gymnasium import Env, spaces

from adapter.protocol import EnvObservation

from .live import GazeboLiveSession, GazeboLiveSessionConfig
from .observation import RosRgbdCameraConfig
from .process import GazeboProcessError
from .m2 import JOINT_NAMES, M2Config, M2Controller, Robotiq2F85Config
from .m3 import (
    M3_MODEL_ID,
    M3Config,
    M3Verifier,
    ReasonCode,
    relative_pose,
    unknown_record,
)
from .ros_control import RosM2ControllerFactory
from .ros_physics import RosM3PhysicsSourceFactory


def _json_env(name: str, default: Any = None) -> Any:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc


def _prepend_env_path(name: str, path: str) -> None:
    """Prepend one relocatable workspace path without duplicating entries."""
    entries = [item for item in os.environ.get(name, "").split(os.pathsep) if item]
    os.environ[name] = os.pathsep.join([path, *(item for item in entries if item != path)])


def live_session_config_from_env() -> GazeboLiveSessionConfig:
    """Build deployment-owned live settings from worker environment variables."""

    extrinsics = _json_env("OPENETA_GAZEBO_CAMERA_EXTRINSICS")
    if not isinstance(extrinsics, dict) or not extrinsics:
        raise ValueError("OPENETA_GAZEBO_CAMERA_EXTRINSICS must be a non-empty JSON object")
    arguments = _json_env("OPENETA_GAZEBO_LAUNCH_ARGUMENTS", ["rviz:=false"])
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise ValueError("OPENETA_GAZEBO_LAUNCH_ARGUMENTS must be a JSON string list")
    return GazeboLiveSessionConfig(
        ros2_executable=os.environ.get("OPENETA_GAZEBO_ROS2_EXECUTABLE", shutil.which("ros2") or "ros2"),
        gz_executable=os.environ.get("OPENETA_GAZEBO_GZ_EXECUTABLE", shutil.which("gz") or "gz"),
        launch_package=os.environ.get("OPENETA_GAZEBO_LAUNCH_PACKAGE", "ros_gz_sim_demos"),
        launch_file=os.environ.get("OPENETA_GAZEBO_LAUNCH_FILE", "rgbd_camera_bridge.launch.py"),
        launch_arguments=tuple(arguments),
        world_name=os.environ.get("OPENETA_GAZEBO_WORLD", "lidar_sensor"),
        camera=RosRgbdCameraConfig(
            rgb_topic=os.environ.get("OPENETA_GAZEBO_RGB_TOPIC", "/rgbd_camera/image"),
            depth_topic=os.environ.get("OPENETA_GAZEBO_DEPTH_TOPIC", "/rgbd_camera/depth_image"),
            camera_info_topic=os.environ.get(
                "OPENETA_GAZEBO_CAMERA_INFO_TOPIC", "/rgbd_camera/camera_info"
            ),
            frame_id=os.environ.get("OPENETA_GAZEBO_CAMERA_FRAME", "rgbd_camera/link/rgbd_camera"),
            extrinsics=extrinsics,
            depth_units_per_metre=float(
                os.environ.get("OPENETA_GAZEBO_DEPTH_UNITS_PER_METRE", "1000")
            ),
        ),
        startup_settle_s=float(os.environ.get("OPENETA_GAZEBO_STARTUP_SETTLE_S", "8")),
    )


def m2_live_session_config_from_env(
    *, robotiq: bool = False, m3: bool = False
) -> GazeboLiveSessionConfig:
    """Repository-owned M2 launch settings; no external workspace is accepted."""
    extrinsics = _json_env(
        "OPENETA_GAZEBO_CAMERA_EXTRINSICS",
        {
            "frame_transform": "camera_to_world",
            "camera_frame": "opencv",
            # Matches the static SDF pose (0, 0, 1.8, 0, pi/2, 0), with
            # Gazebo camera-native X-forward/Y-left/Z-up converted to the
            # OpenCV optical X-right/Y-down/Z-forward axes returned in RGB-D.
            "pos": [0.0, 0.0, 1.8],
            "quat_xyzw": [0.7071067812, -0.7071067812, 0.0, 0.0],
        },
    )
    if not isinstance(extrinsics, dict) or not extrinsics:
        raise ValueError("OPENETA_GAZEBO_CAMERA_EXTRINSICS must be a non-empty JSON object")
    return GazeboLiveSessionConfig(
        ros2_executable=os.environ.get("OPENETA_GAZEBO_ROS2_EXECUTABLE", shutil.which("ros2") or "ros2"),
        gz_executable=os.environ.get("OPENETA_GAZEBO_GZ_EXECUTABLE", shutil.which("gz") or "gz"),
        launch_package="openeta_rm75_robotiq2f85_sim" if robotiq else "openeta_rm75_parallel_sim",
        launch_file="m3_gazebo_pickplace.launch.py" if m3 else "m2_gazebo_moveit.launch.py",
        launch_arguments=(),
        world_name=(
            "m3_rm75_robotiq2f85_pickplace"
            if m3
            else "m2_rm75_robotiq2f85" if robotiq else "m2_rm75_parallel"
        ),
        camera=RosRgbdCameraConfig(
            rgb_topic="/openeta_rgbd/image",
            depth_topic="/openeta_rgbd/depth_image",
            camera_info_topic="/openeta_rgbd/camera_info",
            frame_id="top_camera_optical_frame",
            extrinsics=extrinsics,
            depth_units_per_metre=1000.0,
        ),
        additional_cameras=(RosRgbdCameraConfig(
            rgb_topic="/openeta_wrist_rgbd/image",
            depth_topic="/openeta_wrist_rgbd/depth_image",
            camera_info_topic="/openeta_wrist_rgbd/camera_info",
            frame_id="wrist_camera_optical_frame",
            extrinsics={"frame_transform": "tf_dynamic", "camera_frame": "opencv"},
            depth_units_per_metre=1000.0,
            role="wrist",
        ),),
        startup_settle_s=float(os.environ.get("OPENETA_GAZEBO_STARTUP_SETTLE_S", "3")),
        observation_timeout_s=float(
            os.environ.get("OPENETA_GAZEBO_OBSERVATION_TIMEOUT_S", "30")
        ),
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
                    np.asarray(camera.depth, dtype=np.float32) if camera.depth is not None else None
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


class GazeboM2WorkerEnv(GazeboWorkerEnv):
    """Mature backend-shaped M2 worker behind the existing bench worker API."""

    def __init__(self, *, controller: M2Controller | None = None, **kwargs: Any) -> None:
        cfg = kwargs.pop("m2_config", None) or M2Config()
        cfg.validate_assets()
        # Initialize the mature worker fields directly with the M2 profile.
        # Calling ``GazeboWorkerEnv.__init__`` first would parse M1's required
        # camera extrinsics and make a default, self-contained M2 create fail.
        self._task = str(kwargs.get("task", ""))
        self._seed = int(kwargs.get("seed", 0))
        self._session = None
        self._latest = None
        self._backend = "gazebo"
        is_robotiq = isinstance(cfg, Robotiq2F85Config)
        package_name = "openeta_rm75_robotiq2f85_sim" if is_robotiq else "openeta_rm75_parallel_sim"
        package_prefix = cfg.ros_workspace / "install" / package_name
        _prepend_env_path("AMENT_PREFIX_PATH", str(package_prefix))
        self._session = GazeboLiveSession(
            m2_live_session_config_from_env(
                robotiq=is_robotiq, m3=cfg.model_id == M3_MODEL_ID
            ),
            task=self._task,
        )
        self.openeta_control_spec = {"read_only": False, "m2": True, "model_id": cfg.model_id}
        self._openeta_structured_actions = True
        self.action_space = spaces.Discrete(1)
        self.controller = controller
        self._controller_factory = RosM2ControllerFactory()
        self._m2_config = cfg
        try:
            created = self._session.create()
            self._ensure_controller()
            self._latest = self._with_robot_state(self._as_unified(created))
        except Exception:
            # Constructor failures must not strand a Gazebo/ROS process group
            # in the bench worker. This mirrors mature backend cleanup.
            try:
                if self.controller is not None:
                    self.controller.close()
            finally:
                self._session.close()
            raise

    def _ensure_controller(self) -> M2Controller:
        if self.controller is None:
            try:
                self.controller = self._controller_factory.create(self._m2_config)
            except RuntimeError as exc:
                code = str(exc)
                raise GazeboProcessError(code if code in {
                    "ROS_NOT_READY", "MOVE_GROUP_UNAVAILABLE", "GRIPPER_UNAVAILABLE",
                    "MODEL_ASSET_NOT_FOUND",
                } else "ROS_NOT_READY") from exc
        return self.controller

    def _with_robot_state(self, raw: dict[str, Any]) -> dict[str, Any]:
        raw["robot"] = self._ensure_controller().state_provider().to_dict()
        raw.setdefault("metadata", {}).update({
            "model_id": self._m2_config.model_id,
            "eef_frame": self._m2_config.mount_child,
            "joint_names": list(getattr(self._m2_config, "joint_names", JOINT_NAMES)),
            "camera_frames": ["top_camera_optical_frame", "wrist_camera_optical_frame"],
        })
        return raw

    def refresh_observation(
        self,
        *,
        min_camera_timestamp_s: float | None = None,
        min_received_monotonic_s: float | None = None,
    ) -> dict[str, Any]:
        observation = self._session.observe(
            min_camera_timestamp_s=min_camera_timestamp_s,
            min_received_monotonic_s=min_received_monotonic_s,
        )
        self._latest = self._with_robot_state(self._as_unified(observation))
        return self._latest

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._seed = int(seed)
        if self._latest is None:
            self._session.create()
        controller = self._ensure_controller()
        reset_sources = getattr(controller, "reset_sources", None)
        if callable(reset_sources):
            reset_sources()
        # Reset entities without rewinding `/clock`; ROS action/controller
        # timers require a monotonic simulation clock across an M2 session.
        self._session.reset(seed=self._seed, preserve_sim_time=True)
        wait_ready = getattr(controller, "wait_ready", None)
        if callable(wait_ready):
            wait_ready(30.0)
        opened = controller.execute({"action_type": "gripper_open"})
        if not opened.ok:
            raise GazeboProcessError(opened.error_code or "GRIPPER_FAILED")
        barrier = opened.payload.get("action_completed_ros_time_s")
        return self.refresh_observation(
            min_camera_timestamp_s=float(barrier) if barrier is not None else None,
        ), {}

    def step(self, action: Any):
        controller = self._ensure_controller()
        raw = action if isinstance(action, dict) else {}
        result = controller.execute(raw).to_dict()
        completed_monotonic = time.monotonic()
        camera_barrier = result.get("action_completed_ros_time_s")
        # Like mature DirectEnv backends, one authoritative step returns the
        # exact fresh observation cached by bench_worker. World-mutating
        # actions never return a robot-only or stale packet.
        observation = self.refresh_observation(
            min_camera_timestamp_s=(
                float(camera_barrier) if camera_barrier is not None else None
            ),
            min_received_monotonic_s=completed_monotonic,
        )
        result["observation"] = observation
        return observation, 0.0, False, False, result

    def close(self) -> None:
        try:
            if self.controller is not None:
                self.controller.close()
        finally:
            super().close()


class GazeboRobotiq2F85WorkerEnv(GazeboM2WorkerEnv):
    """M2 worker using the frozen Robotiq 2F-85 asset closure."""

    def __init__(self, *, controller: M2Controller | None = None, **kwargs: Any) -> None:
        kwargs["m2_config"] = Robotiq2F85Config()
        super().__init__(controller=controller, **kwargs)


class GazeboM3WorkerEnv(GazeboM2WorkerEnv):
    """M3 worker that augments every M2 receipt with physical truth."""

    def __init__(
        self,
        *,
        controller: M2Controller | None = None,
        physics_source: Any | None = None,
        physics_source_factory: Any | None = None,
        m3_config: M3Config | None = None,
        **kwargs: Any,
    ) -> None:
        self._m3_config = m3_config or M3Config()
        self._verifier = M3Verifier(self._m3_config)
        self._physics_source = physics_source
        self._physics_source_factory = physics_source_factory or RosM3PhysicsSourceFactory()
        self._last_snapshot = None
        kwargs["m2_config"] = self._m3_config
        super().__init__(controller=controller, **kwargs)
        self.openeta_control_spec.update(
            {"m3": True, "physical_verification": True}
        )
        try:
            if self._physics_source is None:
                self._physics_source = self._physics_source_factory.create(
                    self._ensure_controller(), self._m3_config
                )
            self._latest, snapshot = self._merge_physics(
                self._latest or {}, require=True
            )
            self._initialize_planning_scene(snapshot)
        except Exception:
            self.close()
            raise

    def _camera_timestamp(self, raw: Mapping[str, Any]) -> float:
        timestamps = [
            float(camera.get("timestamp_s", 0.0))
            for camera in raw.get("cameras", {}).values()
            if isinstance(camera, Mapping)
        ]
        if not timestamps or min(timestamps) <= 0:
            raise RuntimeError("M3_CAMERA_TIMESTAMP_MISSING")
        return min(timestamps)

    def _merge_physics(
        self,
        raw: dict[str, Any],
        *,
        action_type: str | None = None,
        action_timestamp_s: float | None = None,
        gripper_result: Mapping[str, Any] | None = None,
        require: bool = False,
    ) -> tuple[dict[str, Any], Any | None]:
        try:
            snapshot = self._physics_source.capture(
                robot=raw["robot"],
                camera_timestamp_s=self._camera_timestamp(raw),
                min_timestamp_s=action_timestamp_s,
                timeout_s=5.0,
                gripper_stalled=(
                    bool(gripper_result["stalled"])
                    if gripper_result is not None and "stalled" in gripper_result
                    else None
                ),
                gripper_reached_goal=(
                    bool(gripper_result["reached_goal"])
                    if gripper_result is not None and "reached_goal" in gripper_result
                    else None
                ),
            )
            record = self._verifier.verify(
                snapshot,
                action_type=action_type,
                action_timestamp_s=action_timestamp_s,
            )
            self._last_snapshot = snapshot
            raw["objects"] = [item.to_dict() for item in snapshot.objects]
        except Exception as exc:
            if require:
                raise
            snapshot = None
            record = unknown_record(
                ReasonCode.DATA_MISSING,
                phase=self._verifier.phase,
                target_id=self._m3_config.target_id,
                evidence={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raw["objects"] = []
        physical = record.to_dict()
        gripper = raw.setdefault("robot", {}).setdefault("gripper_state", {})
        contacts = snapshot.contacts if snapshot is not None else None
        common_ids = (
            set(contacts.left_object_ids).intersection(contacts.right_object_ids)
            if contacts is not None
            else set()
        )
        gripper.update(
            {
                "contact_left": bool(contacts and contacts.left_object_ids),
                "contact_right": bool(contacts and contacts.right_object_ids),
                "contact_object_id": (
                    next(iter(common_ids)) if len(common_ids) == 1 else None
                ),
                "object_detection": record.object_detection,
                "grasp_confirmed": record.grasp_confirmed,
                "slip_detected": record.slip_detected,
            }
        )
        raw.setdefault("metadata", {}).update(
            {
                "model_id": self._m3_config.model_id,
                "physical_verification": physical,
            }
        )
        self._latest = raw
        return raw, snapshot

    def _initialize_planning_scene(self, snapshot: Any) -> None:
        if snapshot is None:
            raise RuntimeError("M3_PHYSICS_TIMEOUT")
        target = snapshot.object(self._m3_config.target_id)
        distractor = snapshot.object(self._m3_config.distractor_id)
        if target is None or distractor is None:
            raise RuntimeError("M3_PHYSICS_TIMEOUT")
        planning = getattr(self._physics_source, "planning_scene", None)
        if planning is not None:
            planning.initialize(target.pose, distractor.pose)

    def refresh_observation(
        self,
        *,
        min_camera_timestamp_s: float | None = None,
        min_received_monotonic_s: float | None = None,
        action_type: str | None = None,
        gripper_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = super().refresh_observation(
            min_camera_timestamp_s=min_camera_timestamp_s,
            min_received_monotonic_s=min_received_monotonic_s,
        )
        merged, _ = self._merge_physics(
            raw,
            action_type=action_type,
            action_timestamp_s=min_camera_timestamp_s,
            gripper_result=gripper_result,
        )
        return merged

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._seed = int(seed)
        controller = self._ensure_controller()
        planning = getattr(self._physics_source, "planning_scene", None)
        if planning is not None:
            planning.clear()
        self._verifier.reset()
        self._last_snapshot = None
        self._physics_source.clear()
        reset_sources = getattr(controller, "reset_sources", None)
        if callable(reset_sources):
            reset_sources()
        self._session.reset(seed=self._seed, preserve_sim_time=True)
        wait_ready = getattr(controller, "wait_ready", None)
        if callable(wait_ready):
            wait_ready(30.0)
        opened = controller.execute({"action_type": "gripper_open"}).to_dict()
        if not opened.get("ok"):
            raise GazeboProcessError(opened.get("error_code") or "GRIPPER_FAILED")
        barrier = opened.get("action_completed_ros_time_s")
        observation = super().refresh_observation(
            min_camera_timestamp_s=float(barrier) if barrier is not None else None
        )
        observation, snapshot = self._merge_physics(
            observation,
            action_type="gripper_open",
            action_timestamp_s=float(barrier) if barrier is not None else None,
            gripper_result=opened,
            require=True,
        )
        self._initialize_planning_scene(snapshot)
        return observation, {}

    def step(self, action: Any):
        controller = self._ensure_controller()
        raw_action = action if isinstance(action, dict) else {}
        action_type = raw_action.get("action_type")
        planning = getattr(self._physics_source, "planning_scene", None)
        if action_type == "gripper_open" and planning is not None and planning.attached:
            target = (
                self._last_snapshot.object(self._m3_config.target_id)
                if self._last_snapshot is not None
                else None
            )
            if target is not None:
                planning.release(target.pose)
        result = controller.execute(raw_action).to_dict()
        completed_monotonic = time.monotonic()
        barrier = result.get("action_completed_ros_time_s")
        observation = super().refresh_observation(
            min_camera_timestamp_s=float(barrier) if barrier is not None else None,
            min_received_monotonic_s=completed_monotonic,
        )
        observation, snapshot = self._merge_physics(
            observation,
            action_type=str(action_type) if action_type is not None else None,
            action_timestamp_s=float(barrier) if barrier is not None else None,
            gripper_result=result,
        )
        physical = observation["metadata"]["physical_verification"]
        # Opening at a destination or in free space needs post-release physics
        # time.  Keep returning a fresh complete observation, but wait until a
        # structured terminal verdict or the bounded settle window expires.
        deadline = time.monotonic() + 2.5
        while (
            action_type == "gripper_open"
            and physical["reason_code"] == ReasonCode.NOT_SETTLED.value
            and time.monotonic() < deadline
        ):
            observation = super().refresh_observation(
                min_camera_timestamp_s=float(barrier) if barrier is not None else None
            )
            observation, snapshot = self._merge_physics(
                observation,
                action_type=None,
                action_timestamp_s=float(barrier) if barrier is not None else None,
                gripper_result=result,
            )
            physical = observation["metadata"]["physical_verification"]
        if (
            snapshot is not None
            and planning is not None
            and physical["reason_code"] == ReasonCode.TARGET_HELD.value
            and not planning.attached
        ):
            target = snapshot.object(self._m3_config.target_id)
            if target is not None:
                planning.attach(relative_pose(snapshot.eef_pose, target.pose))
        if action_type == "gripper_open" and planning is not None and snapshot is not None:
            target = snapshot.object(self._m3_config.target_id)
            if target is not None and not planning.attached:
                planning.release(target.pose)
        result["physical_verification"] = physical
        result["observation"] = observation
        return observation, 0.0, False, False, result

    def close(self) -> None:
        try:
            if self._physics_source is not None:
                self._physics_source.close()
        finally:
            try:
                if getattr(self, "controller", None) is not None:
                    self.controller.close()
            finally:
                if getattr(self, "_session", None) is not None:
                    GazeboWorkerEnv.close(self)
