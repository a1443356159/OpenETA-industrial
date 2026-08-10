"""Official ROS / Gazebo odometry source for the M3 physical verifier."""

from __future__ import annotations

from dataclasses import replace
import math
import threading
import time
from typing import Any, Callable, Mapping

from .m3 import (
    M3Config,
    M3PlanningSceneModel,
    ObjectState,
    PhysicsSnapshot,
    PlanningSceneCommand,
    Pose,
    geometric_table_support,
    quaternion_rotate,
)


ODOMETRY_TOPICS = {
    "m3_target": "/m3/target/odometry",
    "m3_distractor": "/m3/distractor/odometry",
}


def extend_allowed_collision_matrix(
    names: tuple[str, ...],
    rows: tuple[tuple[bool, ...], ...],
    *,
    object_id: str,
    touch_links: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[bool, ...], ...]]:
    """Preserve a complete ACM while enabling only target/fingertip pairs."""

    if len(rows) != len(names) or any(len(row) != len(names) for row in rows):
        raise ValueError("allowed collision matrix must be square")
    expanded_names = list(names)
    expanded_rows = [list(row) for row in rows]
    for name in (object_id, *touch_links):
        if name in expanded_names:
            continue
        expanded_names.append(name)
        for row in expanded_rows:
            row.append(False)
        expanded_rows.append([False] * len(expanded_names))
    target_index = expanded_names.index(object_id)
    for link in touch_links:
        link_index = expanded_names.index(link)
        expanded_rows[target_index][link_index] = True
        expanded_rows[link_index][target_index] = True
    return tuple(expanded_names), tuple(tuple(row) for row in expanded_rows)


def message_timestamp_s(message: Any) -> float | None:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return None
    value = float(int(getattr(stamp, "sec", 0))) + float(
        int(getattr(stamp, "nanosec", 0))
    ) * 1e-9
    return value if math.isfinite(value) and value > 0 else None


def parse_odometry_message(
    message: Any,
    *,
    object_id: str,
    label: str,
    role: str,
) -> ObjectState:
    stamp = message_timestamp_s(message)
    child_frame = str(getattr(message, "child_frame_id", ""))
    if stamp is None or object_id not in tuple(part for part in child_frame.split("::") if part):
        raise ValueError("odometry identity or timestamp is incomplete")
    pose_message = message.pose.pose
    pose = Pose(
        (
            float(pose_message.position.x),
            float(pose_message.position.y),
            float(pose_message.position.z),
        ),
        (
            float(pose_message.orientation.x),
            float(pose_message.orientation.y),
            float(pose_message.orientation.z),
            float(pose_message.orientation.w),
        ),
    ).normalized()
    twist = message.twist.twist
    # nav_msgs/Odometry specifies twist in child_frame_id.  Expose the public
    # object velocity in the world frame used by the pose and verifier.
    world_linear = quaternion_rotate(
        pose.orientation,
        (float(twist.linear.x), float(twist.linear.y), float(twist.linear.z)),
    )
    world_angular = quaternion_rotate(
        pose.orientation,
        (float(twist.angular.x), float(twist.angular.y), float(twist.angular.z)),
    )
    return ObjectState(
        object_id=object_id,
        name=object_id,
        label=label,
        role=role,
        pose=pose,
        linear_velocity=world_linear,
        angular_velocity=world_angular,
        support=None,
        timestamp_s=stamp,
    )


class RosM3PhysicsSource:
    """Own subscriptions to official bridged Odometry streams."""

    def __init__(
        self,
        *,
        robot_state_provider: Callable[[], Any],
        config: M3Config | None = None,
        planning_scene: "RosM3PlanningScene | None" = None,
        context: Any | None = None,
        executor: Any | None = None,
    ) -> None:
        self.config = config or M3Config()
        self.robot_state_provider = robot_state_provider
        self._lock = threading.Lock()
        self._odometry: dict[str, ObjectState] = {}
        self._closed = False
        self._thread: threading.Thread | None = None
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.parameter import Parameter
        except ImportError as exc:
            raise RuntimeError("ROS_NOT_READY") from exc
        self._rclpy = rclpy
        self._owns_context = context is None and not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self._node = rclpy.create_node(
            "openeta_m3_physics_source",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
            context=context,
        )
        self._subscriptions = [
            self._node.create_subscription(
                Odometry,
                topic,
                lambda message, object_id=object_id: self._odometry_callback(
                    object_id, message
                ),
                20,
            )
            for object_id, topic in ODOMETRY_TOPICS.items()
        ]
        self._shared_executor = executor is not None
        self._executor = executor or MultiThreadedExecutor(num_threads=2, context=context)
        self._executor.add_node(self._node)
        if not self._shared_executor:
            self._thread = threading.Thread(
                target=self._executor.spin, name="openeta-m3-physics", daemon=True
            )
            self._thread.start()
        self.planning_scene = planning_scene or RosM3PlanningScene(
            node=self._node, config=self.config
        )

    def _odometry_callback(self, object_id: str, message: Any) -> None:
        try:
            state = parse_odometry_message(
                message,
                object_id=object_id,
                label="target block" if object_id == self.config.target_id else "distractor cylinder",
                role="target" if object_id == self.config.target_id else "distractor",
            )
        except (AttributeError, TypeError, ValueError):
            return
        with self._lock:
            self._odometry[object_id] = state

    def clear(self) -> None:
        with self._lock:
            self._odometry.clear()

    def capture(
        self,
        *,
        robot: Mapping[str, Any],
        camera_timestamp_s: float,
        min_timestamp_s: float | None = None,
        timeout_s: float = 5.0,
        gripper_stalled: bool | None = None,
        gripper_reached_goal: bool | None = None,
    ) -> PhysicsSnapshot:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snapshot = self._try_snapshot(
                robot=robot,
                camera_timestamp_s=camera_timestamp_s,
                gripper_stalled=gripper_stalled,
                gripper_reached_goal=gripper_reached_goal,
            )
            if snapshot is not None and self._base_streams_after(snapshot, min_timestamp_s):
                return snapshot
            time.sleep(0.02)
        raise RuntimeError("M3_PHYSICS_TIMEOUT")

    @staticmethod
    def _base_streams_after(
        snapshot: PhysicsSnapshot, min_timestamp_s: float | None
    ) -> bool:
        required = {
            "joint_state",
            "tf",
            "rgb",
            "depth",
            "odometry_target",
            "odometry_distractor",
        }
        timestamps = snapshot.stream_timestamps()
        if not required.issubset(timestamps):
            return False
        return min_timestamp_s is None or all(
            timestamps[name] > min_timestamp_s for name in required
        )

    def _try_snapshot(
        self,
        *,
        robot: Mapping[str, Any],
        camera_timestamp_s: float,
        gripper_stalled: bool | None,
        gripper_reached_goal: bool | None,
    ) -> PhysicsSnapshot | None:
        with self._lock:
            odometry = dict(self._odometry)
        if set(odometry) != set(ODOMETRY_TOPICS):
            return None
        metadata = dict(robot.get("metadata", {}))
        joint_stamp = metadata.get("joint_state_timestamp_s")
        tf_stamp = metadata.get("tf_timestamp_s")
        if not isinstance(joint_stamp, int | float) or not isinstance(tf_stamp, int | float):
            return None
        eef = robot.get("end_effector_pose", {})
        gripper = robot.get("gripper_state", {})
        try:
            eef_pose = Pose(tuple(eef["xyz"]), tuple(eef["quat_xyzw"])).normalized()
            aperture = float(gripper["aperture_m"])
        except (KeyError, TypeError, ValueError):
            return None
        raw_target = odometry[self.config.target_id]
        target = replace(
            raw_target,
            support=geometric_table_support(
                raw_target.pose, self.config.target_size_m, self.config
            ),
        )
        raw_distractor = odometry[self.config.distractor_id]
        distractor = replace(
            raw_distractor,
            support=geometric_table_support(
                raw_distractor.pose,
                (
                    self.config.distractor_size_m[0],
                    self.config.distractor_size_m[0],
                    self.config.distractor_size_m[1],
                ),
                self.config,
            ),
        )
        objects = (target, distractor)
        stream_timestamps = {
            "joint_state": float(joint_stamp),
            "tf": float(tf_stamp),
            "rgb": float(camera_timestamp_s),
            "depth": float(camera_timestamp_s),
            "odometry_target": float(target.timestamp_s),
            "odometry_distractor": float(odometry[self.config.distractor_id].timestamp_s),
        }
        return PhysicsSnapshot(
            timestamp_s=max(stream_timestamps.values()),
            received_monotonic_s=time.monotonic(),
            eef_pose=eef_pose,
            aperture_m=aperture,
            objects=objects,
            stream_timestamps_s=tuple(sorted(stream_timestamps.items())),
            gripper_stalled=gripper_stalled,
            gripper_reached_goal=gripper_reached_goal,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.planning_scene.clear()
        except Exception:
            pass
        if self._shared_executor:
            self._executor.remove_node(self._node)
        else:
            self._executor.shutdown(timeout_sec=2.0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


class RosM3PlanningScene:
    """Synchronous ``ApplyPlanningScene`` adapter for pure M3 commands."""

    def __init__(self, *, node: Any, config: M3Config | None = None) -> None:
        self.node = node
        self.config = config or M3Config()
        self.model = M3PlanningSceneModel(self.config)
        try:
            from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
        except ImportError as exc:
            raise RuntimeError("ROS_NOT_READY") from exc
        self._service_type = ApplyPlanningScene
        self._client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._get_service_type = GetPlanningScene
        self._get_client = node.create_client(GetPlanningScene, "/get_planning_scene")

    @property
    def attached(self) -> bool:
        return self.model.attached

    def initialize(self, target_pose: Pose, distractor_pose: Pose) -> None:
        self._apply_commands(self.model.initialize(target_pose, distractor_pose))

    def attach(self, relative_pose_value: Pose) -> None:
        self._apply_commands(self.model.attach(relative_pose_value))

    def release(self, world_pose: Pose) -> None:
        self._apply_commands(self.model.release(world_pose))

    def clear(self) -> None:
        # A newly started MoveIt scene has no M3 objects to remove.  Some
        # MoveIt versions reject a REMOVE diff for a non-existent attached
        # object, so keep reset cleanup idempotent by reusing the local scene
        # model's state instead of sending a speculative clear.
        if not self.model.initialized and not self.model.attached:
            return
        self._apply_commands(self.model.clear(), allow_unavailable=True)

    def _apply_commands(
        self,
        commands: tuple[PlanningSceneCommand, ...],
        *,
        allow_unavailable: bool = False,
    ) -> None:
        if not self._client.wait_for_service(timeout_sec=5.0):
            if allow_unavailable:
                return
            raise RuntimeError("MOVE_GROUP_UNAVAILABLE")
        for command in commands:
            request = self._service_type.Request()
            request.scene = self._message(command)
            future = self._client.call_async(request)
            response = self._await(future)
            if not bool(response.success):
                raise RuntimeError("PLANNING_SCENE_FAILED")

    @staticmethod
    def _await(future: Any) -> Any:
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done() or future.exception() is not None:
            raise RuntimeError("PLANNING_SCENE_FAILED")
        return future.result()

    def _current_allowed_collision_matrix(self) -> Any:
        from moveit_msgs.msg import PlanningSceneComponents

        if not self._get_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("MOVE_GROUP_UNAVAILABLE")
        request = self._get_service_type.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        return self._await(
            self._get_client.call_async(request)
        ).scene.allowed_collision_matrix

    def _message(self, command: PlanningSceneCommand) -> Any:
        from geometry_msgs.msg import Pose as RosPose
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
        from shape_msgs.msg import SolidPrimitive

        def pose_message(value: Mapping[str, Any]) -> RosPose:
            result = RosPose()
            position, orientation = value["position"], value["orientation"]
            result.position.x, result.position.y, result.position.z = map(float, position)
            (
                result.orientation.x,
                result.orientation.y,
                result.orientation.z,
                result.orientation.w,
            ) = map(float, orientation)
            return result

        def collision_object(value: Mapping[str, Any], operation: int) -> CollisionObject:
            result = CollisionObject()
            result.header.frame_id = self.config.base_link
            result.id = str(value["id"])
            result.operation = operation
            if operation == CollisionObject.ADD:
                primitive = SolidPrimitive()
                primitive.type = (
                    SolidPrimitive.BOX if value["shape"] == "box" else SolidPrimitive.CYLINDER
                )
                primitive.dimensions = [float(item) for item in value["dimensions"]]
                result.primitives = [primitive]
                result.primitive_poses = [pose_message(value["pose"])]
            return result

        scene = PlanningScene()
        scene.is_diff = True
        payload = command.payload
        if command.operation == "replace_world":
            scene.world.collision_objects = [
                collision_object(item, CollisionObject.ADD) for item in payload["objects"]
            ]
        elif command.operation == "allow_target_touch":
            # A PlanningScene diff containing a partial matrix replaces the
            # current ACM entries instead of merging them.  Fetch and extend
            # MoveIt's complete SRDF-derived matrix so M3 never erases the
            # robot's established adjacent-link collision exemptions.
            matrix = self._current_allowed_collision_matrix()
            try:
                names, rows = extend_allowed_collision_matrix(
                    tuple(matrix.entry_names),
                    tuple(tuple(row.enabled) for row in matrix.entry_values),
                    object_id=payload["object_id"],
                    touch_links=tuple(payload["links"]),
                )
            except ValueError as exc:
                raise RuntimeError("PLANNING_SCENE_FAILED") from exc
            matrix.entry_names = names
            for row_message, values in zip(matrix.entry_values, rows):
                row_message.enabled = values
            # Newly appended rows need their message instances allocated.
            from moveit_msgs.msg import AllowedCollisionEntry

            while len(matrix.entry_values) < len(rows):
                row = AllowedCollisionEntry()
                row.enabled = rows[len(matrix.entry_values)]
                matrix.entry_values.append(row)
            scene.allowed_collision_matrix = matrix
        elif command.operation == "attach":
            remove = CollisionObject()
            remove.header.frame_id = self.config.base_link
            remove.id = payload["object_id"]
            remove.operation = CollisionObject.REMOVE
            scene.world.collision_objects = [remove]
            attached = AttachedCollisionObject()
            attached.link_name = payload["link_name"]
            attached.touch_links = list(payload["touch_links"])
            attached.object.header.frame_id = payload["link_name"]
            attached.object.id = payload["object_id"]
            attached.object.operation = CollisionObject.ADD
            primitive = SolidPrimitive(type=SolidPrimitive.BOX)
            primitive.dimensions = list(payload["dimensions"])
            attached.object.primitives = [primitive]
            attached.object.primitive_poses = [pose_message(payload["relative_pose"])]
            scene.robot_state.is_diff = True
            scene.robot_state.attached_collision_objects = [attached]
        elif command.operation == "release":
            attached = AttachedCollisionObject()
            attached.link_name = payload["link_name"]
            attached.object.id = payload["object_id"]
            attached.object.operation = CollisionObject.REMOVE
            scene.robot_state.is_diff = True
            scene.robot_state.attached_collision_objects = [attached]
            scene.world.collision_objects = [
                collision_object(
                    {
                        "id": payload["object_id"],
                        "shape": "box",
                        "dimensions": payload["dimensions"],
                        "pose": payload["world_pose"],
                    },
                    CollisionObject.ADD,
                )
            ]
        elif command.operation == "clear":
            scene.world.collision_objects = [
                collision_object({"id": object_id}, CollisionObject.REMOVE)
                for object_id in payload["world_object_ids"]
            ]
            attached = AttachedCollisionObject()
            attached.link_name = payload["link_name"]
            attached.object.id = payload["attached_object_id"]
            attached.object.operation = CollisionObject.REMOVE
            scene.robot_state.is_diff = True
            scene.robot_state.attached_collision_objects = [attached]
        else:
            raise ValueError(f"unsupported PlanningScene operation: {command.operation}")
        return scene


class RosM3PhysicsSourceFactory:
    def create(self, controller: Any, config: M3Config, *, context: Any | None = None,
               executor: Any | None = None) -> RosM3PhysicsSource:
        return RosM3PhysicsSource(
            robot_state_provider=controller.state_provider,
            config=config,
            context=context,
            executor=executor,
        )
