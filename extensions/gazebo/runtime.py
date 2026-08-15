"""Single lifecycle owner for a profile-driven Gazebo environment."""

from __future__ import annotations

from dataclasses import replace
import threading
import time
from typing import Any, Callable, Mapping

from adapter.protocol import EnvObservation, RobotState

from .deployment import GazeboDeploymentConfig
from .m3 import M3Config
from .observation import RosRgbdCameraConfig, RosRgbdCameraSource
from .process import (
    GazeboDetachableJointControl,
    GazeboProcessError,
    GazeboWorldControl,
    Ros2LaunchProcess,
)
from .profiles import CONTROL, PHYSICS, GazeboProfile
from .ros_control import RosM2ControllerFactory


class GazeboRuntime:
    """Own launch, ROS adapters, readiness, freshness and teardown.

    Construction is side-effect free.  ROS and Gazebo are started exactly
    once by the first :meth:`reset`.  All readiness checks consume real
    camera/action/service data under one monotonic deadline.
    """

    def __init__(
        self,
        deployment: GazeboDeploymentConfig,
        profile: GazeboProfile,
        *,
        task: str = "",
        launch_factory: Callable[..., Any] = Ros2LaunchProcess,
        camera_factory: Callable[..., Any] = RosRgbdCameraSource,
        controller_factory: Any | None = None,
        world_control: Any | None = None,
        attachment_factory: Callable[..., Any] = GazeboDetachableJointControl,
    ) -> None:
        self.deployment = deployment
        self.profile = profile
        self.task = task
        self._launch_factory = launch_factory
        self._camera_factory = camera_factory
        self._controller_factory = controller_factory or RosM2ControllerFactory(
            readiness_timeout_s=deployment.startup_timeout_s
        )
        self._world = world_control or GazeboWorldControl(
            world_name=deployment.world_override or profile.world_name,
            gz_executable=deployment.gz_executable,
            environment=deployment.process_environment,
        )
        self.attachment: Any | None = None
        if PHYSICS in profile.capabilities:
            self.attachment = attachment_factory(
                gz_executable=deployment.gz_executable,
                environment=deployment.process_environment,
                world_name=deployment.world_override or profile.world_name,
            )
        self._launch: Any | None = None
        self._cameras: list[Any] = []
        self.controller: Any | None = None
        self.started = False
        self.closed = False
        self.start_count = 0
        self.scene_epoch = 0
        self._ros_context: Any | None = None
        self._ros_executor: Any | None = None
        self._ros_thread: threading.Thread | None = None

    def _start_ros_graph(self) -> None:
        """Create the one explicit rclpy context/executor for this runtime."""
        if self._camera_factory is not RosRgbdCameraSource:
            return  # dependency-injected contract tests do not require ROS
        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
        except ImportError as exc:
            raise GazeboProcessError("ROS_NOT_READY") from exc
        context = rclpy.Context()
        rclpy.init(args=None, context=context)
        executor = MultiThreadedExecutor(num_threads=4, context=context)
        thread = threading.Thread(
            target=executor.spin, name="openeta-gazebo-ros", daemon=True
        )
        thread.start()
        self._ros_context, self._ros_executor, self._ros_thread = context, executor, thread

    def _camera_configs(self) -> tuple[RosRgbdCameraConfig, ...]:
        if not self.deployment.camera_extrinsics:
            return self.profile.cameras
        first, *rest = self.profile.cameras
        return (replace(first, extrinsics=dict(self.deployment.camera_extrinsics)), *rest)

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GazeboProcessError("GAZEBO_READINESS_TIMEOUT")
        return remaining

    def _start(self) -> None:
        if self.closed:
            raise GazeboProcessError("Gazebo runtime is closed")
        if self.started:
            return
        if self.profile.unavailable_reason:
            raise GazeboProcessError(self.profile.unavailable_reason)
        deadline = time.monotonic() + self.deployment.startup_timeout_s
        try:
            self._start_ros_graph()
            self._cameras = [
                self._camera_factory(
                    config, node_name=f"openeta_rgbd_camera_{index}",
                    context=self._ros_context, executor=self._ros_executor,
                )
                for index, config in enumerate(self._camera_configs())
            ]
            for camera in self._cameras:
                camera.start()
            arguments = (*self.deployment.launch_arguments,)
            self._launch = self._launch_factory(
                package=self.profile.launch_package,
                launch_file=self.profile.launch_file,
                arguments=arguments,
                ros2_executable=self.deployment.ros2_executable,
                startup_timeout_s=self._remaining(deadline),
                environment=self.deployment.process_environment,
            )
            self._launch.start()
            if PHYSICS in self.profile.capabilities:
                # The launch omits -r, so this command is still before the
                # first physics tick.  Do not resume for a missing stock-joint
                # topic or detached ACK: M3 must fail closed at startup.
                self._world.set_paused(True)
                if self.attachment is None:
                    raise GazeboProcessError("M3_DETACHABLE_JOINT_UNAVAILABLE")
                self.attachment.ensure_detached(require_ack=True)
                self._world.set_paused(False)
            if CONTROL in self.profile.capabilities:
                self.controller = self._controller_factory.create(
                    self.profile.model_config,
                    context=self._ros_context, executor=self._ros_executor,
                )
                wait_ready = getattr(self.controller, "wait_ready", None)
                if callable(wait_ready):
                    wait_ready(self._remaining(deadline))
            self.started = True
            self.start_count += 1
        except Exception:
            self.close()
            raise

    def _robot_state(self) -> RobotState:
        if self.controller is None:
            return RobotState()
        return self.controller.state_provider()

    def observe(
        self,
        *,
        min_camera_timestamp_s: float | None = None,
        min_received_monotonic_s: float | None = None,
        timeout_s: float | None = None,
    ) -> EnvObservation:
        if not self.started or self.closed:
            raise GazeboProcessError("Gazebo runtime must be reset before observe")
        deadline = time.monotonic() + (timeout_s or self.deployment.observation_timeout_s)
        frames = [
            camera.capture(
                timeout_s=self._remaining(deadline),
                min_timestamp_s=min_camera_timestamp_s,
                min_received_monotonic_s=min_received_monotonic_s,
            )
            for camera in self._cameras
        ]
        return EnvObservation(
            task=self.task,
            cameras=frames,
            robot=self._robot_state(),
            metadata={
                "backend": "gazebo",
                "profile": self.profile.name,
                "observation_provenance": "gazebo_ros_live",
                "scene_epoch": self.scene_epoch,
            },
        )

    def reset(self, *, seed: int | None = None) -> EnvObservation:
        self._start()
        if self.controller is not None:
            reset_sources = getattr(self.controller, "reset_sources", None)
            if callable(reset_sources):
                reset_sources()
        # M3's stock plugin starts attached.  Each reset pauses the world,
        # resets the scene, gets a *fresh* detached ACK, then resumes.  A
        # cached local detached state is not sufficient evidence.
        if PHYSICS in self.profile.capabilities:
            if self.attachment is None:
                raise GazeboProcessError("M3_DETACHABLE_JOINT_UNAVAILABLE")
            self._world.set_paused(True)
            self._world.reset_models(seed=seed)
            self.attachment.ensure_detached(require_ack=True)
            config = self.profile.model_config
            if isinstance(config, M3Config):
                for model_name, xyz in config.reset_object_poses.items():
                    self._world.set_model_pose(model_name, xyz)
            self._world.set_paused(False)
        else:
            # Preserve /clock after the ROS action stack has started.
            self._world.reset_models(seed=seed) if CONTROL in self.profile.capabilities else self._world.reset_all(seed=seed)
        self.scene_epoch += 1
        barrier: float | None = None
        if self.controller is not None:
            # A cancelled cold-start action can leave the first open command
            # without a terminal result even though the controller remains
            # healthy.  Opening is idempotent, so retry that one explicitly
            # recoverable result once; every other failure remains fail-closed.
            receipt: dict[str, Any] = {}
            for attempt in range(2):
                receipt = self.controller.execute({"action_type": "gripper_open"}).to_dict()
                if receipt.get("ok"):
                    break
                if receipt.get("error_code") != "GRIPPER_TIMEOUT" or attempt:
                    raise GazeboProcessError(receipt.get("error_code") or "GRIPPER_FAILED")
            value = receipt.get("action_completed_ros_time_s")
            barrier = float(value) if value is not None else None
        observation = self.observe(min_camera_timestamp_s=barrier)
        observation.metadata["reset_seed"] = seed
        return observation

    def execute(self, action: Mapping[str, Any]) -> tuple[EnvObservation, dict[str, Any]]:
        if self.controller is None:
            raise GazeboProcessError("Gazebo profile is read-only")
        receipt = self.controller.execute(dict(action)).to_dict()
        completed = time.monotonic()
        barrier_value = receipt.get("action_completed_ros_time_s")
        barrier = float(barrier_value) if barrier_value is not None else None
        observation = self.observe(
            min_camera_timestamp_s=barrier,
            min_received_monotonic_s=completed,
        )
        return observation, receipt

    def close(self) -> None:
        if self.closed:
            return
        errors: list[BaseException] = []
        if self.attachment is not None and self.started:
            try:
                self.attachment.ensure_detached(require_ack=True)
            except BaseException as exc:
                errors.append(exc)
        # Failures do not short-circuit reverse-order resource cleanup.
        for resource in (self.controller, *reversed(self._cameras)):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as exc:  # cleanup must continue
                errors.append(exc)
        self.controller = None
        self._cameras = []
        if self._ros_executor is not None:
            try:
                self._ros_executor.shutdown(timeout_sec=2.0)
            except BaseException as exc:
                errors.append(exc)
        if self._ros_thread is not None:
            self._ros_thread.join(timeout=2.0)
        if self._ros_context is not None:
            try:
                self._ros_context.shutdown()
            except BaseException as exc:
                errors.append(exc)
        self._ros_context = self._ros_executor = self._ros_thread = None
        if self._launch is not None:
            try:
                self._launch.close()
            except BaseException as exc:
                errors.append(exc)
        self._launch = None
        self.started = False
        self.closed = True
        if errors:
            raise GazeboProcessError(
                "Gazebo runtime cleanup failed: " + "; ".join(str(item) for item in errors)
            )
