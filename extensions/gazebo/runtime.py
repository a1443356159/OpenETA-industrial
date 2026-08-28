"""Single lifecycle owner for a profile-driven Gazebo environment."""

from __future__ import annotations

from dataclasses import replace
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Mapping

from adapter.protocol import EnvObservation, RobotState

from .deployment import GazeboDeploymentConfig
from .observation import (
    GazeboObservationError,
    RosRgbdCameraConfig,
    RosRgbdCameraSource,
)
from .process import (
    DetachableJointState,
    GazeboDetachableJointControl,
    GazeboProcessError,
    GazeboWorldControl,
    Ros2LaunchProcess,
)
from .profiles import CONTROL, PHYSICS, GazeboProfile
from .ros_control import RosGazeboControllerFactory


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
        self._controller_factory = controller_factory or RosGazeboControllerFactory(
            readiness_timeout_s=deployment.startup_timeout_s
        )
        self._world = world_control or GazeboWorldControl(
            world_name=deployment.world_override or profile.world_name,
            gz_executable=deployment.gz_executable,
            environment=deployment.process_environment,
        )
        self.attachment: Any | None = None
        if PHYSICS in profile.capabilities:
            model_config = profile.model_config
            self.attachment = attachment_factory(
                gz_executable=deployment.gz_executable,
                environment=deployment.process_environment,
                timeout_s=15.0,
                world_name=deployment.world_override or profile.world_name,
                parent_link=getattr(model_config, "parent_link", "gripper_mount_link"),
                child_model=getattr(model_config, "target_id", "target_object"),
                child_link=getattr(model_config, "target_link", "target_link"),
                collision_filter_state_topic=getattr(
                    model_config,
                    "attached_collision_filter_state_topic",
                    "/openeta/native_grasp/detachable_joint/target/"
                    "collision_filter_state",
                ),
                collision_filter_state_request_topic=getattr(
                    model_config,
                    "attached_collision_filter_state_request_topic",
                    "/openeta/native_grasp/detachable_joint/target/"
                    "collision_filter_state/request",
                ),
                collision_filter_state_ack_topic=getattr(
                    model_config,
                    "attached_collision_filter_state_ack_topic",
                    "/openeta/native_grasp/detachable_joint/target/"
                    "collision_filter_state/ack",
                ),
                robot_collision_filter_mask=getattr(
                    model_config, "robot_collision_filter_mask", 0x0001
                ),
                detached_target_collision_filter_mask=getattr(
                    model_config,
                    "detached_target_collision_filter_mask",
                    0xFFFF,
                ),
                attached_target_collision_filter_mask=getattr(
                    model_config,
                    "attached_target_collision_filter_mask",
                    0x0002,
                ),
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
        self._reset_target_pose_evidence: dict[str, Any] | None = None
        # The last fully validated observation is retained only for read-only
        # controller RPCs.  Motion qualification can spend tens of seconds in
        # MoveIt without changing either the world or robot; forcing another
        # RGB-D transfer afterwards adds no evidence and can turn a successful
        # qualification into a camera-transport timeout.
        self._last_observation: EnvObservation | None = None

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
        # Qualification uses up to eight IK and eight state-validity requests
        # across independent candidates. Twelve executor threads keep those
        # reentrant clients responsive while preserving room for state/camera
        # callbacks; request-specific semaphores enforce the actual limits.
        executor = MultiThreadedExecutor(num_threads=12, context=context)
        thread = threading.Thread(target=executor.spin, name="openeta-gazebo-ros", daemon=True)
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

    def _wait_for_camera_publishers(self, *, deadline: float) -> None:
        """Wait for every configured ROS camera publisher before subscribing.

        The world-control endpoint is advertised before the headless image and
        CameraInfo bridges necessarily create their ROS publishers.  Creating
        a Fast DDS subscription in that interval can leave it permanently
        unmatched on the deployed local-only discovery configuration.  This
        probe starts fresh short-lived ``ros2 topic list`` clients until the
        exact configured topics are advertised; the subsequent camera capture
        still proves fresh packet delivery and calibration.
        """

        required = {
            topic
            for config in self._camera_configs()
            for topic in (config.rgb_topic, config.depth_topic, config.camera_info_topic)
        }
        executable = (
            shutil.which(self.deployment.ros2_executable) or self.deployment.ros2_executable
        )
        last_error = "topics were not advertised"
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                result = subprocess.run(
                    [executable, "topic", "list"],
                    capture_output=True,
                    text=True,
                    check=False,
                    env=dict(self.deployment.process_environment),
                    timeout=min(5.0, remaining),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                advertised = {line.strip() for line in result.stdout.splitlines() if line.strip()}
                missing = sorted(required - advertised)
                if result.returncode == 0 and not missing:
                    return
                detail = (result.stderr or result.stdout)[-500:].strip()
                last_error = "missing=" + ",".join(missing)
                if detail:
                    last_error += f" detail={detail}"
            remaining = deadline - time.monotonic()
            if remaining > 0:
                threading.Event().wait(timeout=min(0.1, remaining))
        raise GazeboProcessError(f"ROS_CAMERA_TOPICS_NOT_READY: {last_error}")

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
                    config,
                    node_name=f"openeta_rgbd_camera_{index}",
                    context=self._ros_context,
                    executor=self._ros_executor,
                )
                for index, config in enumerate(self._camera_configs())
            ]
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
            # A launch parent can be alive while Gazebo is still registering
            # ``/world/<name>/control``.  Gate the first reset on a bounded,
            # no-op service probe under the existing startup deadline; this
            # is readiness only, never a reset-success fallback.
            wait_ready = getattr(self._world, "wait_ready", None)
            if callable(wait_ready):
                wait_ready(timeout_s=self._remaining(deadline))
            if PHYSICS in self.profile.capabilities:
                # The launch omits -r, so this command is still before the
                # first physics tick.  A world-control ACK alone is not proof
                # that the later-spawned stock joint endpoints exist; wait for
                # those exact endpoints before listener-first detach.  Do not
                # resume for a missing topic or detached ACK: native-grasp fails closed.
                if self.attachment is None:
                    raise GazeboProcessError("NATIVE_GRASP_DETACHABLE_JOINT_UNAVAILABLE")
                attachment_ready = getattr(self.attachment, "wait_ready", None)
                if callable(attachment_ready):
                    attachment_ready(timeout_s=self._remaining(deadline))
                self._world.set_paused(True)
                self.attachment.ensure_detached(require_ack=True)
                self._world.set_paused(False)
            if CONTROL in self.profile.capabilities:
                self.controller = self._controller_factory.create(
                    self.profile.model_config,
                    context=self._ros_context,
                    executor=self._ros_executor,
                )
                wait_ready = getattr(self.controller, "wait_ready", None)
                if callable(wait_ready):
                    wait_ready(self._remaining(deadline))
            if self._camera_factory is RosRgbdCameraSource:
                self._wait_for_camera_publishers(deadline=deadline)
            # Start ROS camera subscriptions only after the launch has made
            # its bridge publishers.  On the deployed Fast DDS stack, a
            # subscription created before a later headless image bridge can
            # remain unmatched indefinitely even while raw Gazebo frames are
            # available.  This is a startup ordering gate, not a cached-image
            # fallback: reset still requires newly received RGB/depth/info.
            for camera in self._cameras:
                camera.start()
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
        # Sample robot state before the potentially slower RGB-D transfers.
        # In particular, an action receipt may have just supplied the final
        # still-fresh state from its execution interval.  Deferring this read
        # until after two camera captures can age that evidence past the state
        # freshness bound even though the physical action completed correctly.
        robot = self._robot_state()
        frames = [
            camera.capture(
                timeout_s=self._remaining(deadline),
                min_timestamp_s=min_camera_timestamp_s,
                min_received_monotonic_s=min_received_monotonic_s,
            )
            for camera in self._cameras
        ]
        observation = EnvObservation(
            task=self.task,
            cameras=frames,
            robot=robot,
            metadata={
                "backend": "gazebo",
                "profile": self.profile.name,
                "observation_provenance": "gazebo_ros_live",
                "scene_epoch": self.scene_epoch,
            },
        )
        self._last_observation = observation
        return observation

    def reset(self, *, seed: int | None = None) -> EnvObservation:
        # Gazebo Sim's stock DetachableJoint emits an output-topic transition
        # only when its state changes.  A ``model_only`` reset leaves a known
        # detached joint detached, so a second detach request has no truthful
        # ACK to consume.  native-grasp therefore resets by recreating its isolated
        # paused world: every reset starts from the stock attached state and
        # obtains one fresh, listener-first detached ACK before unpausing.
        # This is intentionally not a soft attachment or an idempotent-ACK
        # assumption; inability to recreate and receive that ACK still fails
        # closed.
        if PHYSICS in self.profile.capabilities and self.started:
            self.close()
            self.closed = False
        self._start()
        self._reset_target_pose_evidence = None
        if self.controller is not None:
            reset_sources = getattr(self.controller, "reset_sources", None)
            if callable(reset_sources):
                reset_sources()
        # native-grasp's fresh launch has already restored the SDF-declared target and
        # distractor poses while paused, then obtained the fresh detached ACK
        # in ``_start``.  Do not issue a model-only reset here: it preserves a
        # detached stock joint and would turn the required ACK into a no-op.
        if PHYSICS in self.profile.capabilities:
            if self.attachment is None:
                raise GazeboProcessError("NATIVE_GRASP_DETACHABLE_JOINT_UNAVAILABLE")
        else:
            # Preserve /clock after the ROS action stack has started.
            self._world.reset_models(
                seed=seed
            ) if CONTROL in self.profile.capabilities else self._world.reset_all(seed=seed)
        self.scene_epoch += 1
        if self.controller is not None:
            if PHYSICS in self.profile.capabilities:
                sync_scene = getattr(self.controller, "sync_planning_scene_reset", None)
                scene_args = (self.profile.model_config,)
                try:
                    target_pose, _mount_pose, pose_attempts = (
                        self.attachment.native_target_mount_poses_with_retry(max_attempts=2)
                    )
                except Exception as exc:
                    raise GazeboProcessError(
                        f"PLANNING_SCENE_TARGET_POSE_UNAVAILABLE: {exc}"
                    ) from exc
                scene_kwargs = {
                    "target_xyz": tuple(float(value) for value in target_pose.xyz),
                    "target_quat_xyzw": tuple(float(value) for value in target_pose.quat_xyzw),
                }
                dynamic_ids = tuple(
                    str(value)
                    for value in getattr(
                        self.profile.model_config,
                        "authoritative_dynamic_obstacle_ids",
                        (),
                    )
                )
                dynamic_pose_attempts = 0
                if dynamic_ids:
                    read_model_poses = getattr(
                        self.attachment,
                        "native_model_poses_with_retry",
                        None,
                    )
                    if not callable(read_model_poses):
                        raise GazeboProcessError(
                            "AUTHORITATIVE_GAZEBO_MODEL_POSE_UNAVAILABLE"
                        )
                    try:
                        model_poses, dynamic_pose_attempts = read_model_poses(
                            dynamic_ids,
                            max_attempts=2,
                        )
                    except Exception as exc:
                        raise GazeboProcessError(
                            f"AUTHORITATIVE_GAZEBO_MODEL_POSE_UNAVAILABLE: {exc}"
                        ) from exc
                    scene_kwargs["world_model_poses"] = {
                        model_id: (
                            tuple(float(value) for value in model_poses[model_id].xyz),
                            tuple(
                                float(value)
                                for value in model_poses[model_id].quat_xyzw
                            ),
                        )
                        for model_id in dynamic_ids
                    }
                self._reset_target_pose_evidence = {
                    "source": "gazebo_native_pose_v_after_physics_settle",
                    "pose_read_attempt_count": int(pose_attempts),
                    "xyz": list(scene_kwargs["target_xyz"]),
                    "quat_xyzw": list(scene_kwargs["target_quat_xyzw"]),
                    "authoritative_scene_sha256": str(
                        getattr(
                            self.profile.model_config,
                            "authoritative_scene_sha256",
                            "",
                        )
                    ),
                    "dynamic_world_pose_source": "single_gazebo_native_pose_v_snapshot",
                    "dynamic_world_pose_read_attempt_count": int(
                        dynamic_pose_attempts
                    ),
                    "dynamic_world_pose_ids": list(dynamic_ids),
                }
            else:
                sync_scene = getattr(self.controller, "sync_planning_scene_empty", None)
                scene_args = ()
                scene_kwargs = {}
            if not callable(sync_scene):
                raise GazeboProcessError("PLANNING_SCENE_UNAVAILABLE")
            try:
                sync_scene(*scene_args, **scene_kwargs)
            except Exception as exc:
                raise GazeboProcessError(f"PLANNING_SCENE_SYNC_FAILED: {exc}") from exc
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
        if self._reset_target_pose_evidence is not None:
            observation.metadata["planning_scene_reset_target_pose"] = dict(
                self._reset_target_pose_evidence
            )
        return observation

    def execute(self, action: Mapping[str, Any]) -> tuple[EnvObservation, dict[str, Any]]:
        if self.controller is None:
            raise GazeboProcessError("Gazebo profile is read-only")
        normalized_action = dict(action)
        receipt = self.controller.execute(normalized_action).to_dict()
        if normalized_action.get("action_type") == "qualify_motion_candidates":
            # Qualification is an explicitly read-only host RPC: it computes
            # IK, validity and plan-only evidence but never executes a robot or
            # world mutation.  Reuse the observation that supplied the frozen
            # candidate frontier instead of waiting for two unrelated camera
            # streams to publish again.  A missing cache means the normal
            # reset/observe lifecycle was bypassed and must fail closed.
            if self._last_observation is None:
                raise GazeboProcessError("READ_ONLY_ACTION_OBSERVATION_UNAVAILABLE")
            receipt["observation_reused"] = True
            receipt["observation_reuse_reason"] = "read_only_motion_qualification"
            receipt["observation_scene_epoch"] = int(
                self._last_observation.metadata.get("scene_epoch", self.scene_epoch)
            )
            return self._last_observation, receipt
        barrier_value = receipt.get("action_completed_ros_time_s")
        barrier = float(barrier_value) if barrier_value is not None else None
        # Header timestamps and ``action_completed_ros_time_s`` share the
        # simulated ROS clock, and are the ordering proof for a post-action
        # image.  Do not additionally require the subscriber callback to run
        # *after this Python method returns*: the executor can have already
        # queued a correctly post-action image before the action future wakes
        # this thread.  That wall-clock race used to turn valid images into a
        # 30-second transport timeout.  ``capture`` still consumes new RGB and
        # depth sequences, so an image delivered before this action cannot be
        # reused as its observation.
        try:
            observation = self.observe(min_camera_timestamp_s=barrier)
        except GazeboObservationError:
            # The control action has already completed and its native receipt
            # must never be discarded or repeated merely because the camera
            # transport missed one bounded refresh.  Retry only the read-only
            # post-action observation, once, and only while the owned launch
            # and ROS executor still look healthy.  A second miss propagates
            # as infrastructure failure instead of candidate unreachability.
            launch_healthy = self._launch is None or bool(getattr(self._launch, "running", True))
            ros_healthy = self._ros_thread is None or self._ros_thread.is_alive()
            if not self.started or self.closed or not launch_healthy or not ros_healthy:
                raise
            receipt["observation_refresh_retry_count"] = 1
            receipt["observation_refresh_retry_reason"] = "camera_transport_timeout"
            observation = self.observe(min_camera_timestamp_s=barrier)
        return observation, receipt

    def close(self) -> None:
        if self.closed:
            return
        errors: list[BaseException] = []
        if self.attachment is not None and self.started:
            try:
                # A previous stock transition already supplied the only valid
                # detached ACK. Re-publishing detach while it is still known
                # detached emits no state transition in Gazebo Sim, so never
                # manufacture a second ACK requirement at cleanup. Unknown
                # or attached state still requires a real detach ACK.
                if (
                    getattr(self.attachment, "state", DetachableJointState.UNKNOWN)
                    != DetachableJointState.DETACHED
                ):
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
        self._last_observation = None
        self.started = False
        self.closed = True
        if errors:
            raise GazeboProcessError(
                "Gazebo runtime cleanup failed: " + "; ".join(str(item) for item in errors)
            )
