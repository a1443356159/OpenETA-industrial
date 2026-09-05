"""Single lifecycle owner for a profile-driven Gazebo environment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence

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
        self.attachments: dict[str, Any] = {}
        self.attachment: Any | None = None
        self._target_configs: tuple[Any, ...] = ()
        self._work_order_configs: tuple[Any, ...] = ()
        self._work_order_selection_scope = "explicit_items"
        self._work_order_sorting_policy: dict[str, str] | None = None
        self._active_work_order_index = 0
        self._completed_work_order_item_ids: list[str] = []
        self._multi_sort_observation_required = False
        self._manipulation_catalog: dict[str, Any] | None = None
        if PHYSICS in profile.capabilities:
            model_config = profile.model_config
            configured = getattr(model_config, "manipulation_target_configs", ())
            self._target_configs = (
                tuple(configured)
                if isinstance(configured, tuple) and configured
                else (model_config,)
            )
            if len(self._target_configs) > 1:
                contract = getattr(model_config, "acceptance_scene_contract", {})
                regions = (
                    contract.get("placement_regions")
                    if isinstance(contract, Mapping)
                    else None
                )
                self._manipulation_catalog = {
                    "schema_version": "openeta.manipulation_catalog.v1",
                    "targets": [
                        dict(item)
                        for item in getattr(model_config, "manipulation_targets", ())
                        if isinstance(item, Mapping)
                    ],
                    "placement_regions": [
                        {
                            key: value
                            for key, value in item.items()
                            if key in {"id", "prompt", "semantic_aliases"}
                        }
                        for item in (regions if isinstance(regions, list) else [])
                        if isinstance(item, Mapping)
                    ],
                }
            for target_config in self._target_configs:
                target_id = str(getattr(target_config, "target_id", "target_object"))
                if target_id in self.attachments:
                    raise ValueError("native grasp target binding is duplicated")
                self.attachments[target_id] = attachment_factory(
                    gz_executable=deployment.gz_executable,
                    environment=deployment.process_environment,
                    timeout_s=15.0,
                    world_name=deployment.world_override or profile.world_name,
                    parent_link=getattr(target_config, "parent_link", "gripper_mount_link"),
                    child_model=target_id,
                    child_link=getattr(target_config, "target_link", "target_link"),
                    attach_topic=getattr(
                        target_config,
                        "attach_topic",
                        "/openeta/native_grasp/detachable_joint/target/attach",
                    ),
                    detach_topic=getattr(
                        target_config,
                        "detach_topic",
                        "/openeta/native_grasp/detachable_joint/target/detach",
                    ),
                    state_topic=getattr(
                        target_config,
                        "state_topic",
                        "/openeta/native_grasp/detachable_joint/target/state",
                    ),
                    collision_filter_state_topic=getattr(
                        target_config,
                        "attached_collision_filter_state_topic",
                        "/openeta/native_grasp/detachable_joint/target/"
                        "collision_filter_state",
                    ),
                    collision_filter_state_request_topic=getattr(
                        target_config,
                        "attached_collision_filter_state_request_topic",
                        "/openeta/native_grasp/detachable_joint/target/"
                        "collision_filter_state/request",
                    ),
                    collision_filter_state_ack_topic=getattr(
                        target_config,
                        "attached_collision_filter_state_ack_topic",
                        "/openeta/native_grasp/detachable_joint/target/"
                        "collision_filter_state/ack",
                    ),
                    robot_collision_filter_mask=getattr(
                        target_config, "robot_collision_filter_mask", 0x0001
                    ),
                    detached_target_collision_filter_mask=getattr(
                        target_config,
                        "detached_target_collision_filter_mask",
                        0xFFFF,
                    ),
                    attached_target_collision_filter_mask=getattr(
                        target_config,
                        "attached_target_collision_filter_mask",
                        0x0002,
                    ),
                )
            first_target_id = str(getattr(self._target_configs[0], "target_id", ""))
            self.attachment = self.attachments.get(first_target_id)
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

    @property
    def active_pick_place_config(self) -> Any | None:
        if self._work_order_configs:
            return self._work_order_configs[self._active_work_order_index]
        if not self._target_configs:
            return self.profile.model_config
        return self._target_configs[0]

    @property
    def work_order_required(self) -> bool:
        """Whether the operator may author the next ordered sorting task.

        A launcher-owned workcell persists after an order completes.  The
        completed order is useful audit history, but it must not permanently
        lock the cell to its original assignment: a later operator request is
        a new order over the same physical scene.  An incomplete order remains
        exclusive so a follow-up cannot silently replace work already in
        progress.
        """

        if not self._work_order_configs:
            return True
        return len(self._completed_work_order_item_ids) == len(
            self._work_order_configs
        )

    def multi_sort_progress(self) -> dict[str, Any] | None:
        if not self._work_order_configs:
            return None
        all_completed = len(self._completed_work_order_item_ids) == len(
            self._work_order_configs
        )
        active = None if all_completed else self.active_pick_place_config
        assignment = getattr(active, "work_order_item", None)
        work_order_items = [
            dict(config.work_order_item)
            for config in self._work_order_configs
            if isinstance(getattr(config, "work_order_item", None), Mapping)
        ]
        return {
            "schema_version": "openeta.multi_sort_progress.v1",
            "source": "vlm_work_order",
            "scene_id": str(
                getattr(self._work_order_configs[0], "acceptance_scene_id", "")
            ),
            "work_order": {
                "schema_version": "openeta.work_order.v1",
                "source": "vlm_tool_call",
                "selection_scope": self._work_order_selection_scope,
                **(
                    {"sorting_policy": dict(self._work_order_sorting_policy)}
                    if self._work_order_sorting_policy is not None
                    else {}
                ),
                "items": work_order_items,
            },
            "assignment_count": len(self._work_order_configs),
            "completed_count": len(self._completed_work_order_item_ids),
            "completed_assignment_ids": list(self._completed_work_order_item_ids),
            "remaining_count": len(self._work_order_configs)
            - len(self._completed_work_order_item_ids),
            "all_completed": all_completed,
            "active_assignment_index": (
                None if all_completed else self._active_work_order_index
            ),
            "active_assignment": (
                dict(assignment) if isinstance(assignment, Mapping) else None
            ),
            "same_environment_session": True,
            "fresh_observation_required": self._multi_sort_observation_required,
        }

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
                if not self.attachments:
                    raise GazeboProcessError("NATIVE_GRASP_DETACHABLE_JOINT_UNAVAILABLE")
                for attachment in self.attachments.values():
                    attachment_ready = getattr(attachment, "wait_ready", None)
                    if callable(attachment_ready):
                        attachment_ready(timeout_s=self._remaining(deadline))
                self._world.set_paused(True)
                for attachment in self.attachments.values():
                    attachment.ensure_detached(require_ack=True)
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
        robot_state: RobotState | None = None,
    ) -> EnvObservation:
        if not self.started or self.closed:
            raise GazeboProcessError("Gazebo runtime must be reset before observe")
        deadline = time.monotonic() + (timeout_s or self.deployment.observation_timeout_s)
        # Sample robot state before the potentially slower RGB-D transfers.
        # In particular, an action receipt may have just supplied the final
        # still-fresh state from its execution interval.  Deferring this read
        # until after two camera captures can age that evidence past the state
        # freshness bound even though the physical action completed correctly.
        robot = robot_state if robot_state is not None else self._robot_state()
        # Each RGB-D source is independent. Capture them concurrently so a
        # high-resolution source cannot consume the shared deadline before a
        # second camera even begins waiting for its already-published frame.
        # Futures are consumed in configured camera order, keeping the public
        # observation deterministic regardless of callback completion order.
        def capture(camera: Any) -> Any:
            return camera.capture(
                timeout_s=self._remaining(deadline),
                min_timestamp_s=min_camera_timestamp_s,
                min_received_monotonic_s=min_received_monotonic_s,
            )

        with ThreadPoolExecutor(
            max_workers=max(1, len(self._cameras)),
            thread_name_prefix="openeta-rgbd-capture",
        ) as executor:
            futures = [executor.submit(capture, camera) for camera in self._cameras]
            frames = [future.result() for future in futures]
        progress = self.multi_sort_progress()
        if isinstance(progress, dict) and self._multi_sort_observation_required:
            progress = {
                **progress,
                "fresh_observation_required": False,
                "fresh_observation_satisfied": True,
            }
            self._multi_sort_observation_required = False
        observation = EnvObservation(
            task=self.task,
            cameras=frames,
            robot=robot,
            metadata={
                "backend": "gazebo",
                "profile": self.profile.name,
                "observation_provenance": "gazebo_ros_live",
                "scene_epoch": self.scene_epoch,
                **(
                    {
                        "manipulation_catalog": dict(self._manipulation_catalog),
                        "work_order_required": self.work_order_required,
                    }
                    if self._manipulation_catalog is not None
                    else {}
                ),
                **({"multi_sort_progress": progress} if progress is not None else {}),
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
        if self._target_configs:
            self._work_order_configs = ()
            self._work_order_selection_scope = "explicit_items"
            self._work_order_sorting_policy = None
            self._active_work_order_index = 0
            self._completed_work_order_item_ids = []
            self._multi_sort_observation_required = False
            first_target_id = str(getattr(self._target_configs[0], "target_id", ""))
            self.attachment = self.attachments.get(first_target_id)
        if PHYSICS in self.profile.capabilities and self.started:
            self.close()
            self.closed = False
        self._start()
        self._reset_target_pose_evidence = None
        if self.controller is not None and PHYSICS not in self.profile.capabilities:
            # A native pick/place reset always recreates the isolated world and
            # its controller above.  ``_start`` has therefore already waited
            # for a fresh, time-synchronised joint/TF sample.  Clearing that
            # sample here would turn a healthy static robot into a dependency
            # on an unnecessary later /tf publication, which is especially
            # fragile while Gazebo is under load.  Non-physics profiles keep
            # their controller across a model reset, so they still need the
            # post-reset source boundary.
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

    def configure_work_order(
        self,
        *,
        items: Sequence[Mapping[str, Any]],
        selection_scope: str = "explicit_items",
        sorting_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Activate the ordered work plan authored by the VLM tool call."""

        if self._manipulation_catalog is None:
            raise GazeboProcessError("WORK_ORDER_CONFIGURATION_UNAVAILABLE")
        if self._completed_work_order_item_ids and not self.work_order_required:
            raise GazeboProcessError("WORK_ORDER_ALREADY_IN_PROGRESS")
        if any(
            getattr(attachment, "state", None) != DetachableJointState.DETACHED
            for attachment in self.attachments.values()
        ):
            raise GazeboProcessError("WORK_ORDER_TARGET_NOT_DETACHED")
        model_config = self.profile.model_config
        resolver = getattr(model_config, "work_order_configs", None)
        if not callable(resolver):
            raise GazeboProcessError("WORK_ORDER_RESOLVER_UNAVAILABLE")
        try:
            configs = tuple(
                resolver(
                    items,
                    selection_scope=selection_scope,
                    sorting_policy=sorting_policy,
                )
            )
        except ValueError as exc:
            raise GazeboProcessError(f"WORK_ORDER_INVALID: {exc}") from exc
        if not configs:
            raise GazeboProcessError("WORK_ORDER_EMPTY")
        first_config = configs[0]
        first_target_id = str(getattr(first_config, "target_id", ""))
        first_attachment = self.attachments.get(first_target_id)
        if first_attachment is None:
            raise GazeboProcessError("WORK_ORDER_TARGET_BINDING_UNAVAILABLE")
        try:
            target_pose, _mount_pose, pose_attempts = (
                first_attachment.native_target_mount_poses_with_retry(max_attempts=2)
            )
        except Exception as exc:
            raise GazeboProcessError(f"WORK_ORDER_TARGET_POSE_UNAVAILABLE: {exc}") from exc
        activate = getattr(self.controller, "activate_pick_place_config", None)
        if not callable(activate):
            raise GazeboProcessError("WORK_ORDER_PLANNING_SCENE_SWITCH_UNAVAILABLE")
        revision = activate(
            first_config,
            target_xyz=tuple(float(value) for value in target_pose.xyz),
            target_quat_xyzw=tuple(float(value) for value in target_pose.quat_xyzw),
        )
        self._work_order_configs = configs
        first_item = getattr(first_config, "work_order_item", None)
        first_item = first_item if isinstance(first_item, Mapping) else {}
        self._work_order_selection_scope = str(
            first_item.get("selection_scope") or "explicit_items"
        )
        policy = first_item.get("sorting_policy")
        self._work_order_sorting_policy = (
            {str(key): str(value) for key, value in policy.items()}
            if isinstance(policy, Mapping)
            else None
        )
        self._active_work_order_index = 0
        self._completed_work_order_item_ids = []
        self.attachment = first_attachment
        self._last_observation = None
        progress = self.multi_sort_progress()
        if progress is None:
            raise GazeboProcessError("WORK_ORDER_PROGRESS_UNAVAILABLE")
        return {
            **progress,
            "transition": {
                "configured_by": "vlm_tool_call",
                "activated_assignment_index": 0,
                "activated_target_id": first_target_id,
                "planning_scene_revision": int(revision),
                "target_pose_read_attempt_count": int(pose_attempts),
                "world_recreated": False,
                "model_inference_invoked": False,
            },
        }

    def complete_active_work_order_item(
        self,
        *,
        release_evidence: Mapping[str, Any],
        post_release_observation: EnvObservation | None = None,
    ) -> dict[str, Any] | None:
        """Advance a released VLM-authored work order without recreating the world.

        The native detach and gripper-open acknowledgements close the physical
        release transaction. Placement quality is intentionally reviewed by
        the VLM from the causal post-release observation rather than by a
        blocking simulator stability poll.
        """

        if not self._work_order_configs:
            raise GazeboProcessError("WORK_ORDER_NOT_CONFIGURED")
        if not (
            release_evidence.get("schema_version")
            == "openeta.native_release_evidence.v1"
            and release_evidence.get("detached_confirmed") is True
            and release_evidence.get("gripper_open_confirmed") is True
        ):
            raise GazeboProcessError("MULTI_SORT_RELEASE_NOT_PROVEN")
        current = self.active_pick_place_config
        assignment = getattr(current, "work_order_item", None)
        assignment_id = (
            str(assignment.get("id") or "") if isinstance(assignment, Mapping) else ""
        )
        if not assignment_id or assignment_id in self._completed_work_order_item_ids:
            raise GazeboProcessError("MULTI_SORT_ASSIGNMENT_STATE_INVALID")
        if self.attachment is None or getattr(self.attachment, "state", None) != (
            DetachableJointState.DETACHED
        ):
            raise GazeboProcessError("MULTI_SORT_TARGET_NOT_DETACHED")
        if self._active_work_order_index != len(
            self._completed_work_order_item_ids
        ):
            raise GazeboProcessError("MULTI_SORT_ASSIGNMENT_STATE_INVALID")
        current_attachment = self.attachment
        current_target_id = str(getattr(current, "target_id", ""))
        if not current_target_id:
            raise GazeboProcessError("MULTI_SORT_TARGET_BINDING_UNAVAILABLE")
        transition_started = time.monotonic()
        transition: dict[str, Any] = {
            "completed_assignment_id": assignment_id,
            "world_recreated": False,
            "model_inference_invoked": False,
        }
        next_index = self._active_work_order_index + 1
        if next_index < len(self._work_order_configs):
            next_config = self._work_order_configs[next_index]
            next_target_id = str(getattr(next_config, "target_id", ""))
            next_attachment = self.attachments.get(next_target_id)
            if next_attachment is None or getattr(
                next_attachment, "state", None
            ) != DetachableJointState.DETACHED:
                raise GazeboProcessError("MULTI_SORT_NEXT_TARGET_NOT_DETACHED")
            pose_read_started = time.monotonic()
            shared_pose_reader = getattr(
                current_attachment,
                "native_target_model_poses_with_retry",
                None,
            )
            try:
                if callable(shared_pose_reader):
                    model_poses, shared_pose_attempts = shared_pose_reader(
                        {
                            current_target_id: str(
                                getattr(current, "target_link", "")
                            ),
                            next_target_id: str(
                                getattr(next_config, "target_link", "")
                            ),
                        },
                        max_attempts=2,
                    )
                    released_pose = model_poses[current_target_id]
                    target_pose = model_poses[next_target_id]
                    released_pose_attempts = target_pose_attempts = int(
                        shared_pose_attempts
                    )
                    pose_snapshot_shared = True
                else:
                    released_pose, _mount_pose, released_pose_attempts = (
                        current_attachment.native_target_mount_poses_with_retry(
                            max_attempts=2
                        )
                    )
                    target_pose, _mount_pose, target_pose_attempts = (
                        next_attachment.native_target_mount_poses_with_retry(
                            max_attempts=2
                        )
                    )
                    pose_snapshot_shared = False
            except Exception as exc:
                raise GazeboProcessError(
                    f"MULTI_SORT_NEXT_TARGET_POSE_UNAVAILABLE: {exc}"
                ) from exc
            activate = getattr(self.controller, "activate_pick_place_config", None)
            if not callable(activate):
                raise GazeboProcessError("MULTI_SORT_PLANNING_SCENE_SWITCH_UNAVAILABLE")
            scene_commit_started = time.monotonic()
            revision = activate(
                next_config,
                target_xyz=tuple(float(value) for value in target_pose.xyz),
                target_quat_xyzw=tuple(
                    float(value) for value in target_pose.quat_xyzw
                ),
                departure_contact_object_id=current_target_id,
                departure_target_xyz=tuple(
                    float(value) for value in released_pose.xyz
                ),
                departure_target_quat_xyzw=tuple(
                    float(value) for value in released_pose.quat_xyzw
                ),
            )
            self._completed_work_order_item_ids.append(assignment_id)
            self._active_work_order_index = next_index
            self.attachment = next_attachment
            self._last_observation = None
            self._multi_sort_observation_required = True
            transition.update(
                {
                    "activated_assignment_index": next_index,
                    "activated_target_id": next_target_id,
                    "planning_scene_revision": int(revision),
                    "planning_scene_transition_mode": (
                        "atomic_release_and_next_target"
                    ),
                    "target_pose_read_attempt_count": int(target_pose_attempts),
                    "released_target_pose_read_attempt_count": int(
                        released_pose_attempts
                    ),
                    "next_target_pose_read_attempt_count": int(
                        target_pose_attempts
                    ),
                    "pose_snapshot_shared": pose_snapshot_shared,
                    "pose_snapshot_wall_time_ms": round(
                        (scene_commit_started - pose_read_started) * 1000.0, 3
                    ),
                    "planning_scene_commit_and_validation_wall_time_ms": round(
                        (time.monotonic() - scene_commit_started) * 1000.0, 3
                    ),
                    "departure_contact_object_id": current_target_id,
                }
            )
        else:
            pose_read_started = time.monotonic()
            try:
                released_pose, _mount_pose, released_pose_attempts = (
                    current_attachment.native_target_mount_poses_with_retry(
                        max_attempts=2
                    )
                )
            except Exception as exc:
                raise GazeboProcessError(
                    f"MULTI_SORT_RELEASED_TARGET_POSE_UNAVAILABLE: {exc}"
                ) from exc
            sync_target_pose = getattr(
                self.controller, "sync_planning_scene_target_pose", None
            )
            if not callable(sync_target_pose):
                raise GazeboProcessError("PLANNING_SCENE_UNAVAILABLE")
            scene_commit_started = time.monotonic()
            revision = sync_target_pose(
                current,
                target_xyz=tuple(float(value) for value in released_pose.xyz),
                target_quat_xyzw=tuple(
                    float(value) for value in released_pose.quat_xyzw
                ),
                allow_target_touch=True,
            )
            self._completed_work_order_item_ids.append(assignment_id)
            transition.update(
                {
                    "planning_scene_revision": int(revision),
                    "planning_scene_transition_mode": (
                        "final_released_target_pose_sync"
                    ),
                    "released_target_pose_read_attempt_count": int(
                        released_pose_attempts
                    ),
                    "pose_snapshot_shared": False,
                    "pose_snapshot_wall_time_ms": round(
                        (scene_commit_started - pose_read_started) * 1000.0, 3
                    ),
                    "planning_scene_commit_and_validation_wall_time_ms": round(
                        (time.monotonic() - scene_commit_started) * 1000.0, 3
                    ),
                }
            )
        transition["total_wall_time_ms"] = round(
            (time.monotonic() - transition_started) * 1000.0, 3
        )
        progress = self.multi_sort_progress()
        if progress is None:
            raise GazeboProcessError("MULTI_SORT_PROGRESS_UNAVAILABLE")
        if (
            progress.get("all_completed") is not True
            and post_release_observation is not None
            and post_release_observation.cameras
            and post_release_observation.metadata.get("observation_stale") is not True
        ):
            # The gripper action already captured this RGB-D packet after the
            # irreversible detach/open boundary. Activating the next semantic
            # target changes only PlanningScene bookkeeping, not Gazebo. Reuse
            # that causal scene view instead of forcing another camera frame
            # and another TUI round before the next assignment.
            self._multi_sort_observation_required = False
            progress = {
                **progress,
                "fresh_observation_required": False,
                "fresh_observation_satisfied": True,
                "fresh_observation_source": "post_release_action",
            }
            post_release_observation.metadata = {
                **post_release_observation.metadata,
                "planning_scene_revision": int(
                    transition["planning_scene_revision"]
                ),
                "multi_sort_progress": dict(progress),
            }
            self._last_observation = post_release_observation
        return {**progress, "transition": transition}

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
        receipt_observation = receipt.get("observation")
        receipt_robot_value = (
            receipt_observation.get("robot")
            if isinstance(receipt_observation, Mapping)
            else None
        )
        receipt_robot = (
            RobotState.from_dict(dict(receipt_robot_value))
            if receipt.get("ok") is True and isinstance(receipt_robot_value, Mapping)
            else None
        )
        if receipt_robot is not None:
            receipt["post_action_robot_state_reused"] = True
            receipt["post_action_robot_state_source"] = (
                "controller_verified_terminal_receipt"
            )
        # Header timestamps and ``action_completed_ros_time_s`` share the
        # simulated ROS clock, and are the ordering proof for a post-action
        # image.  Do not additionally require the subscriber callback to run
        # *after this Python method returns*: the executor can have already
        # queued a correctly post-action image before the action future wakes
        # this thread.  That wall-clock race used to turn valid images into a
        # 30-second transport timeout.  ``capture`` still consumes new RGB and
        # depth sequences, so an image delivered before this action cannot be
        # reused as its observation.
        observation_error: GazeboObservationError | None = None
        try:
            observation = self.observe(
                min_camera_timestamp_s=barrier,
                robot_state=receipt_robot,
            )
        except GazeboObservationError as exc:
            # The control action has already completed and its native receipt
            # must never be discarded or repeated merely because the camera
            # transport missed one bounded refresh.  Retry only the read-only
            # post-action observation, once, and only while the owned launch
            # and ROS executor still look healthy.
            observation_error = exc
            launch_healthy = self._launch is None or bool(getattr(self._launch, "running", True))
            ros_healthy = self._ros_thread is None or self._ros_thread.is_alive()
            runtime_healthy = bool(
                self.started and not self.closed and launch_healthy and ros_healthy
            )
            receipt["observation_refresh_retry_count"] = 0
            receipt["observation_refresh_retry_reason"] = "camera_transport_timeout"
            receipt["observation_refresh_runtime_healthy"] = runtime_healthy
            if runtime_healthy:
                receipt["observation_refresh_retry_count"] = 1
                try:
                    observation = self.observe(
                        min_camera_timestamp_s=barrier,
                        robot_state=receipt_robot,
                    )
                    observation_error = None
                except GazeboObservationError as retry_exc:
                    observation_error = retry_exc
            if observation_error is not None:
                # The action receipt is causal evidence for a mutation which
                # may be irreversible (notably native detach then open). Do
                # not turn a later camera transport miss into an action
                # failure or invite the agent to repeat that mutation. Return
                # a deliberately image-free, explicitly stale observation;
                # the host obligation dispatcher will request an independent
                # observe call before further visual decisions.
                prior_metadata = (
                    dict(self._last_observation.metadata)
                    if self._last_observation is not None
                    else {}
                )
                observation = EnvObservation(
                    task=self.task,
                    cameras=[],
                    robot=receipt_robot or RobotState(),
                    objects=[],
                    metadata={
                        **prior_metadata,
                        "backend": "gazebo",
                        "profile": self.profile.name,
                        "scene_epoch": self.scene_epoch,
                        "observation_provenance": (
                            "gazebo_post_action_camera_refresh_unavailable"
                        ),
                        "observation_stale": True,
                        "fresh_observation_required": True,
                    },
                )
                receipt.update(
                    {
                        "post_action_observation_available": False,
                        "observation_refresh_error": str(observation_error),
                    }
                )
            else:
                receipt["post_action_observation_available"] = True
        return observation, receipt

    def close(self) -> None:
        if self.closed:
            return
        errors: list[BaseException] = []
        if self.started:
            for attachment in self.attachments.values():
                try:
                    # A previous stock transition already supplied the only valid
                    # detached ACK. Re-publishing detach while it is still known
                    # detached emits no state transition in Gazebo Sim, so never
                    # manufacture a second ACK requirement at cleanup. Unknown
                    # or attached state still requires a real detach ACK.
                    if (
                        getattr(attachment, "state", DetachableJointState.UNKNOWN)
                        != DetachableJointState.DETACHED
                    ):
                        attachment.ensure_detached(require_ack=True)
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
