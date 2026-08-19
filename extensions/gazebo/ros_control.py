"""Jazzy MoveIt and ros2_control adapter for motion-control.

All ROS imports occur inside :meth:`RosGazeboControllerFactory.create`; importing
OpenETA on a non-ROS test machine remains supported.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Mapping

from .robot_control import (
    ARM_JOINTS,
    GazeboControlConfig,
    GazeboController,
    START_STATE_RECOVERY_TRAJECTORY_S,
    assess_start_state_bounds,
    robot_state_from_sources,
    start_state_recovery_record,
)
from .planning_scene import CollisionBox, PlanningSceneError, PlanningSceneSynchronizer


def _normalized_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("pose quaternion must be finite and non-zero")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quaternion_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    q = _normalized_quaternion(quaternion)
    conjugate = (-q[0], -q[1], -q[2], q[3])
    rotated = _quaternion_multiply(
        _quaternion_multiply(q, (vector[0], vector[1], vector[2], 0.0)),
        conjugate,
    )
    return rotated[:3]


def _relative_pose(
    *,
    child_xyz: tuple[float, float, float],
    child_quat_xyzw: tuple[float, float, float, float],
    parent_xyz: tuple[float, float, float],
    parent_quat_xyzw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return ``T_parent^-1 * T_child`` without discarding either rotation."""

    parent_q = _normalized_quaternion(parent_quat_xyzw)
    child_q = _normalized_quaternion(child_quat_xyzw)
    parent_inverse = (-parent_q[0], -parent_q[1], -parent_q[2], parent_q[3])
    delta = tuple(child_xyz[index] - parent_xyz[index] for index in range(3))
    return (
        _quaternion_rotate(parent_inverse, delta),
        _normalized_quaternion(_quaternion_multiply(parent_inverse, child_q)),
    )


def _stamp_seconds(stamp: Any) -> float | None:
    if stamp is None:
        return None
    return float(int(getattr(stamp, "sec", 0))) + float(
        int(getattr(stamp, "nanosec", 0))
    ) * 1e-9


def _moveit_scene_frame(frame: object, *, base_link: str) -> str:
    """Map the co-located Gazebo world frame into MoveIt's fixed root frame."""

    value = str(frame or "")
    # This profile spawns the robot at the Gazebo world origin and its SRDF
    # has a fixed base_link root, so MoveIt has no separate `world` TF frame.
    # World collision poses are numerically base_link poses under that asset
    # contract; attached-object link frames must pass through unchanged.
    return base_link if value in {"", "world"} else value


def gripper_action_success(
    *,
    reached_goal: bool,
    stalled: bool,
    allow_stalling: bool,
    terminal_succeeded: bool = True,
) -> bool:
    """Stall is success only when the ROS action terminal state succeeded."""

    return bool(terminal_succeeded) and (
        bool(reached_goal) or (bool(allow_stalling) and bool(stalled))
    )


def gripper_terminal_succeeded(status: Any) -> bool:
    """Accept only action_msgs GoalStatus.STATUS_SUCCEEDED (numeric 4)."""

    try:
        return int(status) == 4
    except (TypeError, ValueError):
        return False


def _populate_state_validity_request(
    request: Any, candidate_positions: list[float], *, group_name: str
) -> None:
    request.group_name = group_name
    request.robot_state.is_diff = True
    request.robot_state.joint_state.name = list(ARM_JOINTS)
    request.robot_state.joint_state.position = list(candidate_positions)


def _populate_recovery_trajectory_goal(
    goal: Any,
    point: Any,
    duration: Any,
    candidate_positions: list[float],
) -> None:
    goal.trajectory.joint_names = list(ARM_JOINTS)
    point.positions = list(candidate_positions)
    point.time_from_start = duration
    goal.trajectory.points = [point]


def _merged_allowed_collision_rows(
    current_names: list[str],
    current_rows: list[list[bool]],
    additions: Mapping[str, Any],
) -> tuple[list[str], list[list[bool]]]:
    """Merge object/link allowances without erasing the SRDF self matrix."""

    enabled_pairs: set[tuple[str, str]] = set()
    for row_index, row_name in enumerate(current_names):
        values = current_rows[row_index] if row_index < len(current_rows) else []
        for column_index, column_name in enumerate(current_names):
            if column_index < len(values) and bool(values[column_index]):
                enabled_pairs.add(tuple(sorted((row_name, column_name))))
    for object_id, links in additions.items():
        for link in links:
            enabled_pairs.add(tuple(sorted((str(object_id), str(link)))))
    names = sorted(
        set(current_names)
        | {str(key) for key in additions}
        | {str(link) for links in additions.values() for link in links}
    )
    return names, [
        [tuple(sorted((row_name, column))) in enabled_pairs for column in names]
        for row_name in names
    ]


class RosGazeboStateSource:
    def __init__(self, node: Any, tf_buffer: Any, *, config: GazeboControlConfig, freshness_s: float = 2.0):
        self.node, self.tf_buffer, self.config = node, tf_buffer, config
        self.freshness_s = float(freshness_s)
        self._lock = threading.Lock()
        self._joint_state: dict[str, list] | None = None
        self._joint_received = 0.0
        self._joint_stamp: float | None = None

    def joint_state_callback(self, message: Any) -> None:
        with self._lock:
            self._joint_state = {
                "name": list(message.name),
                "position": list(message.position),
                "velocity": list(message.velocity),
            }
            self._joint_received = time.monotonic()
            stamp = getattr(getattr(message, "header", None), "stamp", None)
            self._joint_stamp = (
                float(int(getattr(stamp, "sec", 0)))
                + float(int(getattr(stamp, "nanosec", 0))) * 1e-9
                if stamp is not None
                else None
            )

    def clear(self) -> None:
        with self._lock:
            self._joint_state, self._joint_received, self._joint_stamp = None, 0.0, None

    def state(self):
        with self._lock:
            joint = dict(self._joint_state) if self._joint_state is not None else None
            received = self._joint_received
            joint_stamp = self._joint_stamp
        if joint is None or time.monotonic() - received > self.freshness_s:
            raise RuntimeError("JOINT_STATE_TIMEOUT")
        try:
            # A zero ROS time asks tf2 for the latest transform. This avoids
            # wall-clock vs Gazebo `/clock` extrapolation while the node is
            # still receiving simulated time ticks.
            try:
                from rclpy.time import Time
                lookup_time = Time()
            except ImportError:
                lookup_time = self.node.get_clock().now()
            stamped_transform = self.tf_buffer.lookup_transform(
                self.config.base_link, self.config.mount_child, lookup_time
            )
            transform = stamped_transform.transform
        except Exception as exc:
            raise RuntimeError("TF_TIMEOUT") from exc
        state = robot_state_from_sources(joint, {
            f"{self.config.base_link}->{self.config.mount_child}": {
                "xyz": [transform.translation.x, transform.translation.y, transform.translation.z],
                "quat_xyzw": [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w],
            }
        }, config=self.config)
        state.metadata.update(
            {
                "joint_state_timestamp_s": joint_stamp,
                "joint_state_received_monotonic_s": received,
                "tf_timestamp_s": _stamp_seconds(
                    getattr(getattr(stamped_transform, "header", None), "stamp", None)
                ),
            }
        )
        return state

    def wait_fresh(self, timeout_s: float = 15.0):
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self.state()
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.05)
        raise last_error or RuntimeError("ROBOT_STATE_UNAVAILABLE")


class RosGazeboController(GazeboController):
    def __init__(self, runtime: "_RosRuntime", *, config: GazeboControlConfig):
        self.runtime = runtime
        super().__init__(
            # Action implementations clear the cached JointState before
            # execution.  Reconciliation must wait for the first post-action
            # sample instead of racing an immediate non-blocking read.
            state_provider=runtime.state_source.wait_fresh,
            move_action=runtime.move,
            gripper_action=runtime.gripper,
            start_state_recovery=runtime.recover_start_state,
            cancel_pending=runtime.cancel_pending,
            close_source=runtime.close,
            scene_revision_provider=lambda: int(runtime.planning_scene.revision),
            motion_scene_ready=lambda: bool(runtime.planning_scene.ready),
            config=config,
        )

    def wait_ready(self, timeout_s: float = 30.0) -> None:
        self.runtime.wait_ready(timeout_s)

    def reset_sources(self) -> None:
        self.runtime.cancel_pending()
        self.runtime.state_source.clear()

    @property
    def planning_scene(self) -> PlanningSceneSynchronizer:
        return self.runtime.planning_scene

    def sync_planning_scene_reset(self, config: Any) -> int:
        table = CollisionBox(config.table_id, tuple(config.table_size_m), tuple(config.table_pose_xyz))
        distractor_size = tuple(config.distractor_size_m)
        if len(distractor_size) == 2:
            distractor_size = (distractor_size[0], distractor_size[0], distractor_size[1])
        revision = self.planning_scene.reset(
            table=table,
            distractor=CollisionBox(
                config.distractor_id,
                distractor_size,
                tuple(config.distractor_initial_xyz),
            ),
            target=CollisionBox(
                config.target_id,
                tuple(config.target_size_m),
                tuple(config.target_initial_xyz),
            ),
        )
        validity = self.runtime.current_state_validity(timeout_s=3.0)
        self.runtime.planning_scene_validation = validity
        if validity.get("valid") is not True:
            self.planning_scene.ready = False
            pairs = validity.get("collision_pairs") or []
            raise PlanningSceneError(
                "planning-scene current state is invalid; collision_pairs="
                + repr(pairs)
            )
        self.runtime.scene_revision = revision
        self.runtime.planning_scene_ready = self.planning_scene.ready
        return revision

    def sync_planning_scene_empty(self) -> int:
        revision = self.planning_scene.initialize_empty()
        self.runtime.scene_revision = revision
        self.runtime.planning_scene_ready = self.planning_scene.ready
        return revision

    def sync_planning_scene_attach(
        self,
        config: Any,
        *,
        target_xyz: tuple[float, float, float],
        target_quat_xyzw: tuple[float, float, float, float],
        mount_xyz: tuple[float, float, float],
        mount_quat_xyzw: tuple[float, float, float, float],
    ) -> int:
        relative_xyz, relative_quaternion = _relative_pose(
            child_xyz=target_xyz,
            child_quat_xyzw=target_quat_xyzw,
            parent_xyz=mount_xyz,
            parent_quat_xyzw=mount_quat_xyzw,
        )
        revision = self.planning_scene.attach_target(
            target=CollisionBox(
                config.target_id,
                tuple(config.target_size_m),
                target_xyz,
                target_quat_xyzw,
            ),
            link_name=config.parent_link,
            relative_pose_xyz=relative_xyz,
            relative_pose_quat_xyzw=relative_quaternion,
        )
        self.runtime.scene_revision = revision
        self.runtime.planning_scene_ready = self.planning_scene.ready
        return revision

    def sync_planning_scene_detach(
        self,
        config: Any,
        *,
        target_xyz: tuple[float, float, float],
        target_quat_xyzw: tuple[float, float, float, float],
    ) -> int:
        revision = self.planning_scene.detach_target(
            target=CollisionBox(
                config.target_id,
                tuple(config.target_size_m),
                target_xyz,
                target_quat_xyzw,
            )
        )
        self.runtime.scene_revision = revision
        self.runtime.planning_scene_ready = self.planning_scene.ready
        return revision

    def observation_barrier_s(self) -> float:
        """Current ROS/simulation timestamp for post-action camera ordering."""

        return self.runtime.ros_time_s()

    def return_home(self, timeout_s: float = 15.0):
        """Drive the arm back to the zero (spawn) joint configuration.

        A model-only world reset restores entity poses but leaves the arm at
        whatever configuration the last action ended in, with the trajectory
        controller still holding the stale setpoint.  native-grasp resets once per
        candidate/round and needs every round to start from the same state.
        """

        return dict(self.runtime.return_home(timeout_s))


@dataclass(slots=True)
class RosGazeboControllerFactory:
    readiness_timeout_s: float = 30.0

    def __call__(self, config: GazeboControlConfig | None = None) -> RosGazeboController:
        return self.create(config)

    def create(self, config: GazeboControlConfig | None = None, *, context: Any | None = None,
               executor: Any | None = None) -> RosGazeboController:
        cfg = config or GazeboControlConfig()
        cfg.validate_assets()
        try:
            import rclpy
            from builtin_interfaces.msg import Duration
            from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand
            from moveit_msgs.action import MoveGroup
            from controller_manager_msgs.srv import ListControllers
            from moveit_msgs.srv import GetStateValidity
            from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
            from moveit_msgs.msg import (
                AllowedCollisionEntry,
                AttachedCollisionObject,
                CollisionObject,
                PlanningScene,
                PlanningSceneComponents,
            )
            from rcl_interfaces.srv import GetParameters
            from rclpy.action import ActionClient
            from rclpy.executors import MultiThreadedExecutor
            from sensor_msgs.msg import JointState
            from tf2_ros import Buffer, TransformListener
            from trajectory_msgs.msg import JointTrajectoryPoint
        except ImportError as exc:
            raise RuntimeError("ROS_NOT_READY") from exc
        owns_context = context is None and not rclpy.ok()
        if owns_context:
            rclpy.init(args=None)
        from rclpy.parameter import Parameter
        node = rclpy.create_node(
            "openeta_gazebo_controller",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
            context=context,
        )
        tf_buffer = Buffer(node=node)
        listener = TransformListener(tf_buffer, node, spin_thread=False)
        source = RosGazeboStateSource(node, tf_buffer, config=cfg)
        subscription = node.create_subscription(JointState, "/joint_states", source.joint_state_callback, 10)
        move_client = ActionClient(node, MoveGroup, "/move_action")
        gripper_client = ActionClient(node, ParallelGripperCommand, "/parallel_gripper_controller/gripper_cmd")
        trajectory_client = ActionClient(
            node,
            FollowJointTrajectory,
            "/rm_group_controller/follow_joint_trajectory",
        )
        controller_list_client = node.create_client(ListControllers, "/controller_manager/list_controllers")
        controller_parameter_client = node.create_client(
            GetParameters, "/controller_manager/get_parameters"
        )
        state_validity_client = node.create_client(
            GetStateValidity, "/check_state_validity"
        )
        apply_scene_client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        get_scene_client = node.create_client(GetPlanningScene, "/get_planning_scene")
        shared_executor = executor is not None
        executor = executor or MultiThreadedExecutor(num_threads=2, context=context)
        executor.add_node(node)
        runtime = _RosRuntime(
            rclpy=rclpy, node=node, executor=executor, state_source=source,
            move_client=move_client, gripper_client=gripper_client,
            trajectory_client=trajectory_client,
            controller_list_client=controller_list_client,
            controller_parameter_client=controller_parameter_client,
            state_validity_client=state_validity_client,
            controller_service_type=ListControllers,
            controller_parameter_service_type=GetParameters,
            state_validity_service_type=GetStateValidity,
            follow_trajectory_action_type=FollowJointTrajectory,
            duration_type=Duration,
            trajectory_point_type=JointTrajectoryPoint,
            listener=listener, subscription=subscription,
            owns_context=owns_context,
            config=cfg,
            allow_stalling=bool(getattr(cfg, "allow_stalling", False)),
            shared_executor=shared_executor,
            planning_scene=None,
            scene_revision=0,
            planning_scene_ready=True,
            planning_scene_validation=None,
            apply_scene_client=apply_scene_client,
            get_scene_client=get_scene_client,
            apply_scene_service_type=ApplyPlanningScene,
            get_scene_service_type=GetPlanningScene,
            planning_scene_message_type=PlanningScene,
            planning_scene_components_type=PlanningSceneComponents,
            collision_object_type=CollisionObject,
            attached_collision_object_type=AttachedCollisionObject,
            allowed_collision_entry_type=AllowedCollisionEntry,
            solid_primitive_type=__import__("shape_msgs.msg", fromlist=["SolidPrimitive"]).SolidPrimitive,
            pose_type=__import__("geometry_msgs.msg", fromlist=["Pose"]).Pose,
        )
        runtime.planning_scene = PlanningSceneSynchronizer(runtime.apply_planning_scene)
        runtime.start()
        controller = RosGazeboController(runtime, config=cfg)
        try:
            controller.wait_ready(self.readiness_timeout_s)
        except Exception:
            controller.close()
            raise
        return controller


class _RosRuntime:
    def __init__(self, **values: Any):
        self.__dict__.update(values)
        self._thread: threading.Thread | None = None
        self._pending: list[Any] = []
        self._lock = threading.Lock()
        self._closed = False

    def current_state_validity(self, *, timeout_s: float) -> Mapping[str, Any]:
        """Read back MoveIt's verdict and collision pairs for the live arm state."""

        state = self.state_source.wait_fresh(timeout_s)
        request = self.state_validity_service_type.Request()
        _populate_state_validity_request(
            request,
            [float(value) for value in state.joint_positions[: len(ARM_JOINTS)]],
            group_name=self.config.move_group,
        )
        response = self._await(
            self.state_validity_client.call_async(request), timeout_s
        )
        pairs = sorted(
            {
                tuple(
                    sorted(
                        (
                            str(getattr(contact, "contact_body_1", "")),
                            str(getattr(contact, "contact_body_2", "")),
                        )
                    )
                )
                for contact in getattr(response, "contacts", ())
                if getattr(contact, "contact_body_1", "")
                or getattr(contact, "contact_body_2", "")
            }
        )
        return {
            "valid": bool(response.valid),
            "collision_pairs": [list(pair) for pair in pairs],
            "joint_state_timestamp_s": state.metadata.get("joint_state_timestamp_s"),
        }

    def start(self) -> None:
        if self.shared_executor:
            return
        self._thread = threading.Thread(
            target=self.executor.spin,
            name="openeta-gazebo-ros",
            daemon=True,
        )
        self._thread.start()

    def ros_time_s(self) -> float:
        now = self.node.get_clock().now()
        return float(now.nanoseconds) * 1e-9

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            actions_ready = (
                self.move_client.wait_for_server(timeout_sec=min(0.2, remaining))
                and self.gripper_client.wait_for_server(
                    timeout_sec=min(0.2, remaining)
                )
                and self.trajectory_client.wait_for_server(
                    timeout_sec=min(0.2, remaining)
                )
            )
            if actions_ready:
                services = (
                    self.controller_list_client,
                    self.controller_parameter_client,
                    self.state_validity_client,
                )
                if not all(
                    client.wait_for_service(timeout_sec=min(0.2, remaining))
                    for client in services
                ):
                    continue
                request = self.controller_service_type.Request()
                response = self._await(self.controller_list_client.call_async(request), min(0.5, remaining))
                states = {item.name: item.state for item in response.controller}
                required = {"joint_state_broadcaster", "rm_group_controller", "parallel_gripper_controller"}
                if not required.issubset({name for name, state in states.items() if state == "active"}):
                    continue
                parameter_request = self.controller_parameter_service_type.Request()
                parameter_request.names = ["enforce_command_limits"]
                parameter_response = self._await(
                    self.controller_parameter_client.call_async(parameter_request),
                    min(0.5, remaining),
                )
                if (
                    len(parameter_response.values) != 1
                    or parameter_response.values[0].bool_value is not True
                ):
                    continue
                self.state_source.wait_fresh(min(5.0, remaining))
                return
        if not self.move_client.server_is_ready():
            raise RuntimeError("MOVE_GROUP_UNAVAILABLE")
        if not self.gripper_client.server_is_ready():
            raise RuntimeError("GRIPPER_UNAVAILABLE")
        raise RuntimeError("ROS_NOT_READY")

    def _await(self, future: Any, timeout_s: float) -> Any:
        deadline = time.monotonic() + timeout_s
        with self._lock:
            self._pending.append(future)
        try:
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not future.done():
                raise TimeoutError
            error = future.exception()
            if error is not None:
                raise error
            return future.result()
        finally:
            with self._lock:
                if future in self._pending:
                    self._pending.remove(future)

    def apply_planning_scene(self, diff: dict[str, Any]) -> Mapping[str, Any]:
        """Apply one collision-scene diff and prove exact world/attached ids."""

        if not self.apply_scene_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("PLANNING_SCENE_APPLY_UNAVAILABLE")
        if not self.get_scene_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("PLANNING_SCENE_READBACK_UNAVAILABLE")
        scene = self.planning_scene_message_type()
        scene.is_diff = True
        # AttachedCollisionObject additions/removals live under robot_state.
        # Mark that nested message as a diff too; otherwise MoveIt interprets
        # the intentionally sparse state as a complete RobotState and rejects
        # the ApplyPlanningScene request.
        scene.robot_state.is_diff = True
        for object_id in diff.get("remove_world_ids", []):
            collision = self.collision_object_type()
            collision.id = str(object_id)
            collision.header.frame_id = self.config.base_link
            collision.operation = self.collision_object_type.REMOVE
            scene.world.collision_objects.append(collision)
        for spec in diff.get("world_objects", []):
            collision = self._collision_object_from_spec(spec)
            scene.world.collision_objects.append(collision)
        for object_id in diff.get("remove_attached_ids", []):
            attached = self.attached_collision_object_type()
            attached.object.id = str(object_id)
            attached.object.operation = self.collision_object_type.REMOVE
            scene.robot_state.attached_collision_objects.append(attached)
        for spec in diff.get("attached_objects", []):
            attached = self.attached_collision_object_type()
            attached.link_name = str(spec["link_name"])
            attached.touch_links = [str(value) for value in spec.get("touch_links", [])]
            attached.object = self._collision_object_from_spec(spec)
            scene.robot_state.attached_collision_objects.append(attached)
        allowed = diff.get("allowed_collisions")
        if isinstance(allowed, Mapping):
            # A sparse AllowedCollisionMatrix in a PlanningScene diff replaces
            # MoveIt's SRDF-derived matrix rather than patching it.  Read and
            # merge the live matrix first, otherwise adding target/fingertip
            # exceptions accidentally re-enables every adjacent-link
            # self-collision and all subsequent plans fail at the start state.
            components = self.planning_scene_components_type
            acm_request = self.get_scene_service_type.Request()
            acm_request.components.components = int(
                components.ALLOWED_COLLISION_MATRIX
            )
            acm_readback = self._await(
                self.get_scene_client.call_async(acm_request), 5.0
            )
            current_acm = acm_readback.scene.allowed_collision_matrix
            names, merged_rows = _merged_allowed_collision_rows(
                [str(value) for value in current_acm.entry_names],
                [list(row.enabled) for row in current_acm.entry_values],
                allowed,
            )
            scene.allowed_collision_matrix.entry_names = names
            for enabled in merged_rows:
                row = self.allowed_collision_entry_type()
                row.enabled = enabled
                scene.allowed_collision_matrix.entry_values.append(row)
            scene.allowed_collision_matrix.default_entry_names = list(
                current_acm.default_entry_names
            )
            scene.allowed_collision_matrix.default_entry_values = list(
                current_acm.default_entry_values
            )
        apply_request = self.apply_scene_service_type.Request()
        apply_request.scene = scene
        applied = self._await(self.apply_scene_client.call_async(apply_request), 5.0)
        get_request = self.get_scene_service_type.Request()
        components = self.planning_scene_components_type
        get_request.components.components = int(components.WORLD_OBJECT_NAMES) | int(
            components.ROBOT_STATE_ATTACHED_OBJECTS
        )
        readback = self._await(self.get_scene_client.call_async(get_request), 5.0)
        return {
            "applied": bool(getattr(applied, "success", False)),
            "world_ids": [item.id for item in readback.scene.world.collision_objects],
            "attached_ids": [
                item.object.id for item in readback.scene.robot_state.attached_collision_objects
            ],
        }

    def _collision_object_from_spec(self, spec: Mapping[str, Any]) -> Any:
        collision = self.collision_object_type()
        collision.id = str(spec["id"])
        collision.header.frame_id = _moveit_scene_frame(
            spec.get("frame"), base_link=self.config.base_link
        )
        primitive = self.solid_primitive_type()
        primitive.type = self.solid_primitive_type.BOX
        primitive.dimensions = [float(value) for value in spec["size_xyz"]]
        pose = self.pose_type()
        pose.position.x, pose.position.y, pose.position.z = [
            float(value) for value in spec["pose_xyz"]
        ]
        quaternion = spec.get("pose_quat_xyzw", (0.0, 0.0, 0.0, 1.0))
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = _normalized_quaternion(tuple(float(value) for value in quaternion))
        collision.primitives = [primitive]
        collision.primitive_poses = [pose]
        collision.operation = self.collision_object_type.ADD
        return collision

    def recover_start_state(
        self, state: Any, timeout_s: float
    ) -> Mapping[str, Any]:
        assessment = assess_start_state_bounds(state)
        classification = assessment["classification"]
        if classification == "WITHIN_BOUNDS":
            return {
                "ok": True,
                "start_state_recovery": start_state_recovery_record(
                    assessment, status="NOT_REQUIRED"
                ),
            }
        if classification == "INVALID":
            return {
                "ok": False,
                "error_code": "START_STATE_INVALID",
                "motion_outcome": "failed",
                "start_state_recovery": start_state_recovery_record(
                    assessment, status="REJECTED"
                ),
            }

        started = self.ros_time_s()
        deadline = time.monotonic() + float(timeout_s)

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        def failed(reason_code: str, *, result_code: int | None = None):
            return {
                "ok": False,
                "error_code": "START_STATE_RECOVERY_FAILED",
                "motion_outcome": "failed",
                "action_started_ros_time_s": started,
                "action_completed_ros_time_s": self.ros_time_s(),
                "start_state_recovery": start_state_recovery_record(
                    assessment,
                    status="FAILED",
                    reason_code=reason_code,
                    attempted=True,
                    trajectory_result_code=result_code,
                ),
            }

        def unknown(reason_code: str):
            return {
                "ok": False,
                "error_code": "MOTION_OUTCOME_UNKNOWN",
                "motion_outcome": "unknown",
                "reconciliation_required": True,
                "action_started_ros_time_s": started,
                "action_completed_ros_time_s": self.ros_time_s(),
                "start_state_recovery": start_state_recovery_record(
                    assessment,
                    status="UNKNOWN",
                    reason_code=reason_code,
                    attempted=True,
                ),
            }

        candidate_positions = list(assessment["candidate_positions"])
        try:
            validity_request = self.state_validity_service_type.Request()
            _populate_state_validity_request(
                validity_request,
                candidate_positions,
                group_name=self.config.move_group,
            )
            validity_response = self._await(
                self.state_validity_client.call_async(validity_request),
                min(1.0, remaining()),
            )
        except Exception:
            return failed("STATE_VALIDITY_CHECK_FAILED")
        if not bool(validity_response.valid):
            return failed("RECOVERY_STATE_INVALID_OR_IN_COLLISION")

        goal = self.follow_trajectory_action_type.Goal()
        point = self.trajectory_point_type()
        duration_ns = int(START_STATE_RECOVERY_TRAJECTORY_S * 1_000_000_000)
        duration = self.duration_type(
            sec=duration_ns // 1_000_000_000,
            nanosec=duration_ns % 1_000_000_000,
        )
        _populate_recovery_trajectory_goal(
            goal, point, duration, candidate_positions
        )
        try:
            handle = self._await(
                self.trajectory_client.send_goal_async(goal), min(1.0, remaining())
            )
        except Exception:
            return failed("RECOVERY_TRAJECTORY_SEND_FAILED")
        if not handle.accepted:
            return failed("RECOVERY_TRAJECTORY_REJECTED")

        result_future = handle.get_result_async()
        try:
            wrapped = self._await(result_future, remaining())
        except TimeoutError:
            try:
                self._await(handle.cancel_goal_async(), min(1.0, remaining()))
            except Exception:
                pass
            self.state_source.clear()
            return unknown("RECOVERY_TRAJECTORY_TIMEOUT_UNCONFIRMED")
        except Exception:
            self.state_source.clear()
            return unknown("RECOVERY_TRAJECTORY_RESULT_UNAVAILABLE")

        result_code = int(wrapped.result.error_code)
        if result_code != int(self.follow_trajectory_action_type.Result.SUCCESSFUL):
            self.state_source.clear()
            return failed("RECOVERY_TRAJECTORY_FAILED", result_code=result_code)

        self.state_source.clear()
        try:
            post_state = self.state_source.wait_fresh(remaining())
        except Exception:
            return failed("POST_RECOVERY_JOINT_STATE_MISSING", result_code=result_code)
        # Gazebo's joint limiter can report an endpoint one machine epsilon
        # beyond its decimal limit even after the accepted inset trajectory.
        # Apply the same certified numeric tolerance used for the preflight;
        # a zero-tolerance postcheck turns that harmless limiter round-off
        # into a false recovery failure.
        post_assessment = assess_start_state_bounds(post_state)
        post_timestamp = post_assessment.get("pre_joint_state_timestamp_s")
        if post_assessment["classification"] == "INVALID":
            return {
                **failed(
                    "POST_RECOVERY_STATE_OUT_OF_BOUNDS", result_code=result_code
                ),
                "start_state_recovery": start_state_recovery_record(
                    assessment,
                    status="FAILED",
                    reason_code="POST_RECOVERY_STATE_OUT_OF_BOUNDS",
                    attempted=True,
                    post_joint_state_timestamp_s=post_timestamp,
                    trajectory_result_code=result_code,
                ),
            }
        return {
            "ok": True,
            "action_started_ros_time_s": started,
            "action_completed_ros_time_s": self.ros_time_s(),
            "start_state_recovery": start_state_recovery_record(
                assessment,
                status="RECOVERED",
                reason_code="NUMERIC_BOUNDS_RECOVERED",
                attempted=True,
                post_joint_state_timestamp_s=post_timestamp,
                trajectory_result_code=result_code,
            ),
        }

    def return_home(self, timeout_s: float) -> Mapping[str, Any]:
        """Command all arm joints to zero through the trajectory controller."""

        goal = self.follow_trajectory_action_type.Goal()
        point = self.trajectory_point_type()
        duration_ns = int(2.0 * 1_000_000_000)
        duration = self.duration_type(
            sec=duration_ns // 1_000_000_000,
            nanosec=duration_ns % 1_000_000_000,
        )
        _populate_recovery_trajectory_goal(
            goal, point, duration, [0.0] * len(ARM_JOINTS)
        )
        handle = self._await(self.trajectory_client.send_goal_async(goal), min(5.0, timeout_s))
        if not handle.accepted:
            raise RuntimeError("HOME_TRAJECTORY_REJECTED")
        wrapped = self._await(handle.get_result_async(), timeout_s)
        self.state_source.clear()
        if int(wrapped.result.error_code) != int(self.follow_trajectory_action_type.Result.SUCCESSFUL):
            raise RuntimeError("HOME_TRAJECTORY_FAILED")
        return {"ok": True, "trajectory_result_code": int(wrapped.result.error_code)}

    def move(self, goal: dict, timeout_s: float) -> Mapping[str, Any]:
        from geometry_msgs.msg import Pose
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import Constraints, MoveItErrorCodes, OrientationConstraint, PositionConstraint
        from shape_msgs.msg import SolidPrimitive

        action_started = self.ros_time_s()

        def finish(payload: dict[str, Any]) -> dict[str, Any]:
            payload["action_started_ros_time_s"] = action_started
            payload["action_completed_ros_time_s"] = self.ros_time_s()
            # GazeboController asks for state immediately after this method.  Drop
            # every sample accumulated during planning/execution so that read
            # must observe a JointState published after the result boundary.
            self.state_source.clear()
            return payload

        # The start state read in GazeboController happened before this call.  Do
        # not permit it to double as post-action reconciliation state.
        self.state_source.clear()
        request = MoveGroup.Goal()
        request.request.group_name = goal["group_name"]
        # OMPL is stochastic: a single attempt can return a needlessly long
        # joint-space excursion (winding the redundant wrist onto its limits,
        # or swinging the open gripper into a grasp target mid-path), or fail
        # outright on an unlucky sample.  MoveGroup evaluates several attempts
        # and executes the shortest solution, which keeps both Cartesian hops
        # and physical approaches tidy without changing the goal contract.
        request.request.num_planning_attempts = 3
        # Keep the action client's deadline strictly outside MoveIt's own
        # planning deadline. Equal deadlines race cancellation against a
        # terminal result and can trigger an invalid goal-state transition in
        # MoveIt Jazzy. Plan-only reachability probes are deliberately short.
        planning_limit_s = 8.0 if goal.get("plan_only", False) else 30.0
        request.request.allowed_planning_time = min(
            planning_limit_s, max(0.1, timeout_s - 2.0)
        )
        # The 7-DOF arm is redundant, so OMPL may legitimately choose a
        # several-radian joint-space path even for the small Cartesian moves
        # used by motion-control.  At 10% scaling those collision-checked trajectories can
        # exceed the public action deadline on a software-rendered simulator
        # (which commonly runs below real time).  Thirty percent remains a
        # conservative MoveIt limit while keeping every valid planned path
        # inside the control contract's timeout.  Grasp-critical carry moves
        # may lower the scaling further through the goal to keep a jointed or
        # caged payload quiet.
        request.request.max_velocity_scaling_factor = float(
            goal.get("max_velocity_scaling_factor", 0.3)
        )
        request.request.max_acceleration_scaling_factor = float(
            goal.get("max_acceleration_scaling_factor", 0.3)
        )
        request.request.start_state.is_diff = True
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = goal["target_pose"]["xyz"]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = goal["target_pose"]["quat_xyzw"]
        primitive = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[2 * goal["position_tolerance_m"]] * 3)
        pc = PositionConstraint()
        pc.header.frame_id, pc.link_name, pc.weight = goal["base_frame"], goal["link_name"], 1.0
        pc.constraint_region.primitives = [primitive]
        pc.constraint_region.primitive_poses = [pose]
        oc = OrientationConstraint()
        oc.header.frame_id, oc.link_name, oc.orientation, oc.weight = goal["base_frame"], goal["link_name"], pose.orientation, 1.0
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = goal["orientation_tolerance_rad"]
        request.request.goal_constraints = [Constraints(position_constraints=[pc], orientation_constraints=[oc])]
        request.planning_options.plan_only = bool(goal.get("plan_only", False))
        send = self.move_client.send_goal_async(request)
        handle = self._await(send, min(5.0, timeout_s))
        if not handle.accepted:
            return finish({
                "ok": False,
                "error_code": "MOTION_PLAN_FAILED",
                "motion_outcome": "failed",
                "planned_point_count": 0,
                "execution_started": False,
            })
        result_future = handle.get_result_async()
        try:
            wrapped = self._await(result_future, timeout_s)
        except TimeoutError:
            try:
                self._await(handle.cancel_goal_async(), 2.0)
                self._await(result_future, 2.0)
            except Exception:
                pass
            return finish({
                "ok": False,
                "error_code": "MOTION_OUTCOME_UNKNOWN",
                "motion_outcome": "unknown",
                "planned_point_count": 0,
                "execution_started": None,
            })
        code = wrapped.result.error_code.val
        planned_points = list(
            getattr(
                getattr(wrapped.result.planned_trajectory, "joint_trajectory", None),
                "points",
                (),
            )
        )
        if code == MoveItErrorCodes.SUCCESS:
            return finish({
                "ok": True,
                "reached_goal": not request.planning_options.plan_only,
                "plan_only": request.planning_options.plan_only,
                "motion_outcome": "planned" if request.planning_options.plan_only else "completed",
                "planned_point_count": len(planned_points),
                "execution_started": not request.planning_options.plan_only,
            })
        planning_failures = {
            MoveItErrorCodes.PLANNING_FAILED,
            MoveItErrorCodes.INVALID_MOTION_PLAN,
            MoveItErrorCodes.START_STATE_IN_COLLISION,
            MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS,
            MoveItErrorCodes.GOAL_IN_COLLISION,
            MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS,
            MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED,
            MoveItErrorCodes.INVALID_GROUP_NAME,
            MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS,
            MoveItErrorCodes.INVALID_ROBOT_STATE,
            MoveItErrorCodes.INVALID_LINK_NAME,
            MoveItErrorCodes.INVALID_OBJECT_NAME,
            MoveItErrorCodes.FRAME_TRANSFORM_FAILURE,
            MoveItErrorCodes.START_STATE_INVALID,
            MoveItErrorCodes.GOAL_STATE_INVALID,
            MoveItErrorCodes.UNRECOGNIZED_GOAL_TYPE,
            MoveItErrorCodes.NO_IK_SOLUTION,
        }
        # MoveGroup sometimes collapses a goal-sampling failure to the generic
        # FAILURE code. An empty returned trajectory proves execution never
        # started, so classify it as a planning rejection.
        if code in planning_failures or (
            code == MoveItErrorCodes.FAILURE and not planned_points
        ):
            return finish({
                "ok": False,
                "error_code": "MOTION_PLAN_FAILED",
                "motion_outcome": "failed",
                "moveit_error_code": int(code),
                "planned_point_count": len(planned_points),
                "execution_started": False,
            })
        if code == MoveItErrorCodes.TIMED_OUT:
            return finish({
                "ok": False,
                "error_code": "MOTION_EXECUTION_TIMEOUT",
                "motion_outcome": "failed",
                "moveit_error_code": int(code),
                "planned_point_count": len(planned_points),
                "execution_started": bool(planned_points),
            })
        return finish({
            "ok": False,
            "error_code": "MOTION_EXECUTION_FAILED",
            "motion_outcome": "failed",
            "moveit_error_code": int(code),
            "planned_point_count": len(planned_points),
            "execution_started": bool(planned_points),
        })

    def gripper(self, position: float, timeout_s: float) -> Mapping[str, Any]:
        from control_msgs.action import ParallelGripperCommand

        action_started = self.ros_time_s()
        wall_started = time.monotonic()

        def finish(payload: dict[str, Any]) -> dict[str, Any]:
            payload["action_started_ros_time_s"] = action_started
            payload["action_completed_ros_time_s"] = self.ros_time_s()
            # Diagnostics only: wall-clock duration and terminal status do
            # not affect the strict success predicate below.
            payload["wall_elapsed_ms"] = round((time.monotonic() - wall_started) * 1000, 3)
            self.state_source.clear()
            return payload

        self.state_source.clear()
        goal = ParallelGripperCommand.Goal()
        goal.command.name = ["gripper_left_finger_joint"]
        goal.command.position = [float(position)]
        handle = self._await(self.gripper_client.send_goal_async(goal), min(5.0, timeout_s))
        if not handle.accepted:
            return finish(
                {
                    "ok": False,
                    "error_code": "GRIPPER_FAILED",
                    "terminal_status": "rejected",
                }
            )
        try:
            wrapped = self._await(handle.get_result_async(), timeout_s)
        except TimeoutError:
            try:
                self._await(handle.cancel_goal_async(), min(2.0, timeout_s))
            except Exception:
                pass
            return finish({
                "ok": False,
                "reached_goal": False,
                "stalled": False,
                "error_code": "GRIPPER_TIMEOUT",
                "terminal_status": "timed_out",
            })
        except Exception:
            return finish({
                "ok": False,
                "reached_goal": False,
                "stalled": False,
                "error_code": "GRIPPER_FAILED",
                "terminal_status": "result_error",
            })
        result = wrapped.result
        reached_goal = bool(result.reached_goal)
        stalled = bool(result.stalled)
        terminal_status_code = getattr(wrapped, "status", None)
        terminal_succeeded = gripper_terminal_succeeded(terminal_status_code)
        ok = gripper_action_success(
            reached_goal=reached_goal,
            stalled=stalled,
            allow_stalling=bool(self.allow_stalling),
            terminal_succeeded=terminal_succeeded,
        )
        return finish({
            "ok": ok,
            "reached_goal": reached_goal,
            "stalled": stalled,
            "error_code": None if ok else "GRIPPER_FAILED",
            "terminal_status": "succeeded" if terminal_succeeded else "not_succeeded",
            "terminal_status_code": (
                int(terminal_status_code)
                if isinstance(terminal_status_code, int) and not isinstance(terminal_status_code, bool)
                else None
            ),
        })

    def cancel_pending(self) -> None:
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        for future in pending:
            try:
                future.cancel()
            except Exception:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.cancel_pending()
        try:
            if self.shared_executor:
                self.executor.remove_node(self.node)
            else:
                self.executor.shutdown(timeout_sec=2.0)
        finally:
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self.node.destroy_node()
            if self.owns_context and self.rclpy.ok():
                self.rclpy.shutdown()
