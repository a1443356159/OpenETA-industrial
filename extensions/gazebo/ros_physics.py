"""Official ROS / Gazebo message sources for the M3 physical verifier."""

from __future__ import annotations

from dataclasses import replace
import math
import threading
import time
from typing import Any, Callable, Mapping

from .m3 import (
    ContactState,
    M3Config,
    M3PlanningSceneModel,
    ObjectState,
    PhysicsSnapshot,
    PlanningSceneCommand,
    Pose,
    namespaced_entity_id,
    quaternion_rotate,
)


CONTACT_TOPICS = {
    "left": "/m3/contact/left_fingertip",
    "right": "/m3/contact/right_fingertip",
    "target": "/m3/contact/target",
}
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


def contact_object_ids(
    message: Any,
    *,
    known_ids: tuple[str, ...],
    sensor_owner_id: str | None = None,
) -> tuple[tuple[str, ...], bool]:
    """Extract exact model identities from ``ros_gz_interfaces/Contacts``."""

    found: set[str] = set()
    complete = True
    for contact in getattr(message, "contacts", ()):  # one entry per collision pair
        names = (
            str(getattr(getattr(contact, "collision1", None), "name", "")),
            str(getattr(getattr(contact, "collision2", None), "name", "")),
        )
        if not all(names):
            complete = False
            continue
        pair_ids = {
            item
            for item in (namespaced_entity_id(name, known_ids) for name in names)
            if item is not None
        }
        if sensor_owner_id is not None:
            pair_ids.discard(sensor_owner_id)
        if not pair_ids:
            complete = False
        found.update(pair_ids)
    return tuple(sorted(found)), complete


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
    """Own subscriptions to official bridged Contact and Odometry streams."""

    def __init__(
        self,
        *,
        robot_state_provider: Callable[[], Any],
        config: M3Config | None = None,
        planning_scene: "RosM3PlanningScene | None" = None,
    ) -> None:
        self.config = config or M3Config()
        self.robot_state_provider = robot_state_provider
        self._lock = threading.Lock()
        self._contacts: dict[str, dict[str, Any]] = {}
        self._odometry: dict[str, ObjectState] = {}
        self._closed = False
        self._thread: threading.Thread | None = None
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.parameter import Parameter
            from ros_gz_interfaces.msg import Contacts
        except ImportError as exc:
            raise RuntimeError("ROS_NOT_READY") from exc
        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self._node = rclpy.create_node(
            "openeta_m3_physics_source",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self._subscriptions = [
            self._node.create_subscription(
                Contacts,
                topic,
                lambda message, channel=channel: self._contact_callback(channel, message),
                20,
            )
            for channel, topic in CONTACT_TOPICS.items()
        ]
        self._subscriptions.extend(
            self._node.create_subscription(
                Odometry,
                topic,
                lambda message, object_id=object_id: self._odometry_callback(
                    object_id, message
                ),
                20,
            )
            for object_id, topic in ODOMETRY_TOPICS.items()
        )
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin, name="openeta-m3-physics", daemon=True
        )
        self._thread.start()
        self.planning_scene = planning_scene or RosM3PlanningScene(
            node=self._node, config=self.config
        )

    def _contact_callback(self, channel: str, message: Any) -> None:
        stamp = message_timestamp_s(message)
        if stamp is None:
            return
        known = (
            self.config.target_id,
            self.config.distractor_id,
            self.config.table_id,
            "ground",
            self.config.model_id,
        )
        owner = self.config.target_id if channel == "target" else self.config.model_id
        object_ids, complete = contact_object_ids(
            message, known_ids=known, sensor_owner_id=owner
        )
        received = time.monotonic()
        with self._lock:
            previous = self._contacts.get(channel, {})
            previous_ids = set(previous.get("object_ids", ()))
            previous_since = dict(previous.get("since", {}))
            since = {
                object_id: (
                    previous_since.get(object_id, stamp)
                    if object_id in previous_ids and stamp >= float(previous.get("stamp", 0.0))
                    else stamp
                )
                for object_id in object_ids
            }
            self._contacts[channel] = {
                "object_ids": object_ids,
                "complete": complete,
                "stamp": stamp,
                "received": received,
                "since": since,
            }

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
            self._contacts.clear()
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
        partial: PhysicsSnapshot | None = None
        partial_ready_at: float | None = None
        while time.monotonic() < deadline:
            snapshot = self._try_snapshot(
                robot=robot,
                camera_timestamp_s=camera_timestamp_s,
                gripper_stalled=gripper_stalled,
                gripper_reached_goal=gripper_reached_goal,
                allow_missing_contacts=False,
            )
            if snapshot is not None and (
                min_timestamp_s is None
                or all(value > min_timestamp_s for _, value in snapshot.stream_timestamps_s)
            ):
                return snapshot
            candidate = self._try_snapshot(
                robot=robot,
                camera_timestamp_s=camera_timestamp_s,
                gripper_stalled=gripper_stalled,
                gripper_reached_goal=gripper_reached_goal,
                allow_missing_contacts=True,
            )
            if candidate is not None and self._base_streams_after(candidate, min_timestamp_s):
                partial = candidate
                partial_ready_at = partial_ready_at or (
                    time.monotonic() + self.config.stable_contact_s + 0.05
                )
                if time.monotonic() >= partial_ready_at:
                    # Gazebo contact sensors publish while contacts exist, but
                    # do not emit empty heartbeat messages in Harmonic.  Keep
                    # the stream absent (or stale) so the verifier reports
                    # UNKNOWN; never synthesize evidence for "no contact".
                    return partial
            time.sleep(0.02)
        if partial is not None:
            return partial
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
        allow_missing_contacts: bool,
    ) -> PhysicsSnapshot | None:
        with self._lock:
            contacts = {key: dict(value) for key, value in self._contacts.items()}
            odometry = dict(self._odometry)
        if set(odometry) != set(ODOMETRY_TOPICS):
            return None
        if not allow_missing_contacts and set(contacts) != set(CONTACT_TOPICS):
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
        target_contact = contacts.get("target", {})
        target_supports = tuple(target_contact.get("object_ids", ()))
        target = replace(
            odometry[self.config.target_id],
            support=(self.config.table_id if self.config.table_id in target_supports else None),
        )
        objects = (target, odometry[self.config.distractor_id])
        contact_state = ContactState(
            left_object_ids=tuple(contacts.get("left", {}).get("object_ids", ())),
            right_object_ids=tuple(contacts.get("right", {}).get("object_ids", ())),
            target_support_ids=target_supports,
            left_durations_s=tuple(
                sorted(
                    (
                        object_id,
                        max(0.0, float(contacts["left"]["stamp"]) - float(since)),
                    )
                    for object_id, since in contacts.get("left", {}).get("since", {}).items()
                )
            ),
            right_durations_s=tuple(
                sorted(
                    (
                        object_id,
                        max(0.0, float(contacts["right"]["stamp"]) - float(since)),
                    )
                    for object_id, since in contacts.get("right", {}).get("since", {}).items()
                )
            ),
            timestamps_s=tuple(
                sorted(
                    (f"contact_{channel}", float(value["stamp"]))
                    for channel, value in contacts.items()
                )
            ),
            identities_complete=(
                set(contacts) == set(CONTACT_TOPICS)
                and all(bool(value["complete"]) for value in contacts.values())
            ),
        )
        stream_timestamps = {
            "joint_state": float(joint_stamp),
            "tf": float(tf_stamp),
            "rgb": float(camera_timestamp_s),
            "depth": float(camera_timestamp_s),
            "odometry_target": float(target.timestamp_s),
            "odometry_distractor": float(odometry[self.config.distractor_id].timestamp_s),
        }
        stream_timestamps.update(
            {
                f"contact_{channel}": float(value["stamp"])
                for channel, value in contacts.items()
            }
        )
        return PhysicsSnapshot(
            timestamp_s=max(stream_timestamps.values()),
            received_monotonic_s=time.monotonic(),
            eef_pose=eef_pose,
            aperture_m=aperture,
            contacts=contact_state,
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
    def create(self, controller: Any, config: M3Config) -> RosM3PhysicsSource:
        return RosM3PhysicsSource(
            robot_state_provider=controller.state_provider,
            config=config,
        )
