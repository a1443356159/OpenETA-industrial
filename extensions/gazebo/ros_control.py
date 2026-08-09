"""Jazzy MoveIt and ros2_control adapter for M2.

All ROS imports occur inside :meth:`RosM2ControllerFactory.create`; importing
OpenETA on a non-ROS test machine remains supported.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Mapping

from .m2 import M2Config, M2Controller, make_move_group_goal, robot_state_from_sources


def _stamp_seconds(stamp: Any) -> float | None:
    if stamp is None:
        return None
    return float(int(getattr(stamp, "sec", 0))) + float(
        int(getattr(stamp, "nanosec", 0))
    ) * 1e-9


def gripper_action_success(*, reached_goal: bool, stalled: bool, allow_stalling: bool) -> bool:
    """Documented stall-success policy, kept pure for M2/M3 isolation tests."""

    return bool(reached_goal) or (bool(allow_stalling) and bool(stalled))


class RosM2StateSource:
    def __init__(self, node: Any, tf_buffer: Any, *, config: M2Config, freshness_s: float = 2.0):
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


class RosM2Controller(M2Controller):
    def __init__(self, runtime: "_RosRuntime", *, config: M2Config):
        self.runtime = runtime
        super().__init__(
            # Action implementations clear the cached JointState before
            # execution.  Reconciliation must wait for the first post-action
            # sample instead of racing an immediate non-blocking read.
            state_provider=runtime.state_source.wait_fresh,
            move_action=runtime.move,
            gripper_action=runtime.gripper,
            cancel_pending=runtime.cancel_pending,
            close_source=runtime.close,
            config=config,
        )

    def wait_ready(self, timeout_s: float = 30.0) -> None:
        self.runtime.wait_ready(timeout_s)

    def reset_sources(self) -> None:
        self.runtime.cancel_pending()
        self.runtime.state_source.clear()

    def observation_barrier_s(self) -> float:
        """Current ROS/simulation timestamp for post-action camera ordering."""

        return self.runtime.ros_time_s()

    def plan_pose(self, target_pose: Mapping[str, Any], timeout_s: float = 30.0):
        """Collision-aware MoveGroup plan-only precheck through production routing."""

        goal = make_move_group_goal(target_pose, config=self.config)
        goal["plan_only"] = True
        return dict(self.runtime.move(goal, timeout_s))


@dataclass(slots=True)
class RosM2ControllerFactory:
    readiness_timeout_s: float = 30.0

    def __call__(self, config: M2Config | None = None) -> RosM2Controller:
        return self.create(config)

    def create(self, config: M2Config | None = None) -> RosM2Controller:
        cfg = config or M2Config()
        cfg.validate_assets()
        try:
            import rclpy
            from control_msgs.action import ParallelGripperCommand
            from moveit_msgs.action import MoveGroup
            from controller_manager_msgs.srv import ListControllers
            from rclpy.action import ActionClient
            from rclpy.executors import MultiThreadedExecutor
            from sensor_msgs.msg import JointState
            from tf2_ros import Buffer, TransformListener
        except ImportError as exc:
            raise RuntimeError("ROS_NOT_READY") from exc
        owns_context = not rclpy.ok()
        if owns_context:
            rclpy.init(args=None)
        from rclpy.parameter import Parameter
        node = rclpy.create_node(
            "openeta_m2_controller",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        tf_buffer = Buffer(node=node)
        listener = TransformListener(tf_buffer, node, spin_thread=False)
        source = RosM2StateSource(node, tf_buffer, config=cfg)
        subscription = node.create_subscription(JointState, "/joint_states", source.joint_state_callback, 10)
        move_client = ActionClient(node, MoveGroup, "/move_action")
        gripper_client = ActionClient(node, ParallelGripperCommand, "/parallel_gripper_controller/gripper_cmd")
        controller_list_client = node.create_client(ListControllers, "/controller_manager/list_controllers")
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        runtime = _RosRuntime(
            rclpy=rclpy, node=node, executor=executor, state_source=source,
            move_client=move_client, gripper_client=gripper_client,
            controller_list_client=controller_list_client,
            controller_service_type=ListControllers,
            listener=listener, subscription=subscription, owns_context=owns_context,
            allow_stalling=bool(getattr(cfg, "allow_stalling", False)),
        )
        runtime.start()
        controller = RosM2Controller(runtime, config=cfg)
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

    def start(self) -> None:
        self._thread = threading.Thread(target=self.executor.spin, name="openeta-m2-ros", daemon=True)
        self._thread.start()

    def ros_time_s(self) -> float:
        now = self.node.get_clock().now()
        return float(now.nanoseconds) * 1e-9

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if self.move_client.wait_for_server(timeout_sec=min(0.2, remaining)) and self.gripper_client.wait_for_server(timeout_sec=min(0.2, remaining)):
                if not self.controller_list_client.wait_for_service(timeout_sec=min(0.2, remaining)):
                    continue
                request = self.controller_service_type.Request()
                response = self._await(self.controller_list_client.call_async(request), min(0.5, remaining))
                states = {item.name: item.state for item in response.controller}
                required = {"joint_state_broadcaster", "rm_group_controller", "parallel_gripper_controller"}
                if not required.issubset({name for name, state in states.items() if state == "active"}):
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

    def move(self, goal: dict, timeout_s: float) -> Mapping[str, Any]:
        from geometry_msgs.msg import Pose
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import Constraints, MoveItErrorCodes, OrientationConstraint, PositionConstraint
        from shape_msgs.msg import SolidPrimitive

        action_started = self.ros_time_s()

        def finish(payload: dict[str, Any]) -> dict[str, Any]:
            payload["action_started_ros_time_s"] = action_started
            payload["action_completed_ros_time_s"] = self.ros_time_s()
            # M2Controller asks for state immediately after this method.  Drop
            # every sample accumulated during planning/execution so that read
            # must observe a JointState published after the result boundary.
            self.state_source.clear()
            return payload

        # The start state read in M2Controller happened before this call.  Do
        # not permit it to double as post-action reconciliation state.
        self.state_source.clear()
        request = MoveGroup.Goal()
        request.request.group_name = goal["group_name"]
        request.request.num_planning_attempts = 1
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
        # used by M2.  At 10% scaling those collision-checked trajectories can
        # exceed the public action deadline on a software-rendered simulator
        # (which commonly runs below real time).  Thirty percent remains a
        # conservative MoveIt limit while keeping every valid planned path
        # inside the control contract's timeout.
        request.request.max_velocity_scaling_factor = 0.3
        request.request.max_acceleration_scaling_factor = 0.3
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
            return finish({"ok": False, "error_code": "MOTION_PLAN_FAILED", "motion_outcome": "failed"})
        result_future = handle.get_result_async()
        try:
            wrapped = self._await(result_future, timeout_s)
        except TimeoutError:
            try:
                self._await(handle.cancel_goal_async(), 2.0)
                self._await(result_future, 2.0)
            except Exception:
                pass
            return finish({"ok": False, "error_code": "MOTION_OUTCOME_UNKNOWN", "motion_outcome": "unknown"})
        code = wrapped.result.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            return finish({
                "ok": True,
                "reached_goal": not request.planning_options.plan_only,
                "plan_only": request.planning_options.plan_only,
                "motion_outcome": "planned" if request.planning_options.plan_only else "completed",
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
        planned_points = list(
            getattr(
                getattr(wrapped.result.planned_trajectory, "joint_trajectory", None),
                "points",
                (),
            )
        )
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
            })
        if code == MoveItErrorCodes.TIMED_OUT:
            return finish({
                "ok": False,
                "error_code": "MOTION_EXECUTION_TIMEOUT",
                "motion_outcome": "failed",
                "moveit_error_code": int(code),
            })
        return finish({
            "ok": False,
            "error_code": "MOTION_EXECUTION_FAILED",
            "motion_outcome": "failed",
            "moveit_error_code": int(code),
        })

    def gripper(self, position: float, timeout_s: float) -> Mapping[str, Any]:
        from control_msgs.action import ParallelGripperCommand

        action_started = self.ros_time_s()

        def finish(payload: dict[str, Any]) -> dict[str, Any]:
            payload["action_started_ros_time_s"] = action_started
            payload["action_completed_ros_time_s"] = self.ros_time_s()
            self.state_source.clear()
            return payload

        self.state_source.clear()
        goal = ParallelGripperCommand.Goal()
        goal.command.name = ["gripper_left_finger_joint"]
        goal.command.position = [float(position)]
        handle = self._await(self.gripper_client.send_goal_async(goal), min(5.0, timeout_s))
        if not handle.accepted:
            return finish({"ok": False, "error_code": "GRIPPER_FAILED"})
        try:
            wrapped = self._await(handle.get_result_async(), timeout_s)
        except TimeoutError:
            handle.cancel_goal_async()
            self.state_source.clear()
            raise
        result = wrapped.result
        reached_goal = bool(result.reached_goal)
        stalled = bool(result.stalled)
        ok = gripper_action_success(
            reached_goal=reached_goal,
            stalled=stalled,
            allow_stalling=bool(self.allow_stalling),
        )
        return finish({
            "ok": ok,
            "reached_goal": reached_goal,
            "stalled": stalled,
            "error_code": None if ok else "GRIPPER_FAILED",
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
            self.executor.shutdown(timeout_sec=2.0)
        finally:
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self.node.destroy_node()
            if self.owns_context and self.rclpy.ok():
                self.rclpy.shutdown()
