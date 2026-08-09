#!/usr/bin/env python3
"""Expose the Robotiq Gazebo linkage as ParallelGripperCommand.

Gazebo Harmonic's URDF importer does not propagate all five Robotiq mimic
joints when only the outer knuckle is commanded.  The robot-local Gazebo
position systems therefore receive the complete vendor multiplier vector,
while callers retain the standard one-joint ROS action contract.
"""

from __future__ import annotations

import threading
import time
from typing import Final

import rclpy
from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


ACTIVE_JOINT: Final = "gripper_left_finger_joint"
JOINT_MULTIPLIERS: Final = {
    ACTIVE_JOINT: 1.0,
    "gripper_right_finger_joint": -1.0,
    "gripper_left_inner_knuckle_joint": 1.0,
    "gripper_right_inner_knuckle_joint": -1.0,
    "gripper_left_finger_tip_joint": -1.0,
    "gripper_right_finger_tip_joint": 1.0,
}
COMMAND_TOPICS: Final = {
    "gripper_left_finger_joint": "/openeta/gripper/left_outer",
    "gripper_right_finger_joint": "/openeta/gripper/right_outer",
    "gripper_left_inner_knuckle_joint": "/openeta/gripper/left_inner",
    "gripper_right_inner_knuckle_joint": "/openeta/gripper/right_inner",
    "gripper_left_finger_tip_joint": "/openeta/gripper/left_tip",
    "gripper_right_finger_tip_joint": "/openeta/gripper/right_tip",
}
MIN_POSITION_RAD: Final = 0.0
MAX_POSITION_RAD: Final = 0.8
GOAL_TOLERANCE_RAD: Final = 0.02
ACTION_TIMEOUT_S: Final = 20.0


def expanded_targets(active_position: float) -> dict[str, float]:
    """Expand the standard active-joint command to the vendor mimic vector."""

    return {
        name: multiplier * active_position
        for name, multiplier in JOINT_MULTIPLIERS.items()
    }


class RobotiqGripperActionAdapter(Node):
    def __init__(self) -> None:
        super().__init__("openeta_robotiq_gripper_action_adapter")
        self.declare_parameter("allow_stalling", False)
        self.declare_parameter("stall_velocity_threshold", 0.001)
        self.declare_parameter("stall_timeout", 1.0)
        self._allow_stalling = bool(self.get_parameter("allow_stalling").value)
        self._stall_velocity_threshold = float(
            self.get_parameter("stall_velocity_threshold").value
        )
        self._stall_timeout = float(self.get_parameter("stall_timeout").value)
        if self._stall_velocity_threshold < 0 or self._stall_timeout <= 0:
            raise ValueError("stall thresholds must be non-negative with a positive timeout")
        self._callback_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._positions: dict[str, float] = {}
        self._velocities: dict[str, float] = {}
        self._last_state: JointState | None = None
        self._command_publishers = {
            joint: self.create_publisher(Float64, topic, 10)
            for joint, topic in COMMAND_TOPICS.items()
        }
        self._subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
            callback_group=self._callback_group,
        )
        self._server = ActionServer(
            self,
            ParallelGripperCommand,
            "/parallel_gripper_controller/gripper_cmd",
            execute_callback=self._execute,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )

    def _joint_state_callback(self, message: JointState) -> None:
        positions = dict(zip(message.name, message.position))
        velocities = dict(zip(message.name, message.velocity))
        if not set(JOINT_MULTIPLIERS).issubset(positions):
            return
        with self._lock:
            self._positions = {name: float(positions[name]) for name in JOINT_MULTIPLIERS}
            self._velocities = {
                name: float(velocities.get(name, 0.0)) for name in JOINT_MULTIPLIERS
            }
            self._last_state = message

    def _goal_callback(self, request: ParallelGripperCommand.Goal) -> GoalResponse:
        names = list(request.command.name)
        positions = list(request.command.position)
        if names != [ACTIVE_JOINT] or len(positions) != 1:
            self.get_logger().error("gripper goal must contain only the active Robotiq joint")
            return GoalResponse.REJECT
        target = float(positions[0])
        if not MIN_POSITION_RAD <= target <= MAX_POSITION_RAD:
            self.get_logger().error(f"gripper position {target} is outside [0.0, 0.8]")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel_callback(_goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _publish_targets(self, targets: dict[str, float]) -> None:
        for name, target in targets.items():
            message = Float64()
            message.data = target
            self._command_publishers[name].publish(message)

    def _snapshot(self) -> tuple[dict[str, float], dict[str, float], JointState | None]:
        with self._lock:
            return dict(self._positions), dict(self._velocities), self._last_state

    @staticmethod
    def _result_state(
        positions: dict[str, float], velocities: dict[str, float], source: JointState | None
    ) -> JointState:
        state = JointState()
        if source is not None:
            state.header = source.header
        state.name = list(JOINT_MULTIPLIERS)
        state.position = [positions.get(name, float("nan")) for name in state.name]
        state.velocity = [velocities.get(name, float("nan")) for name in state.name]
        return state

    def _execute(self, goal_handle):
        active_position = float(goal_handle.request.command.position[0])
        targets = expanded_targets(active_position)
        result = ParallelGripperCommand.Result()
        deadline = time.monotonic() + ACTION_TIMEOUT_S
        last_movement_time = time.monotonic()

        while rclpy.ok() and time.monotonic() < deadline:
            positions, velocities, state = self._snapshot()
            if goal_handle.is_cancel_requested:
                if positions:
                    self._publish_targets(positions)
                result.state = self._result_state(positions, velocities, state)
                result.stalled = False
                result.reached_goal = False
                goal_handle.canceled()
                return result

            # Re-publishing makes startup deterministic even if the bridge is
            # still establishing its Gazebo publisher on the first sample.
            self._publish_targets(targets)
            if set(targets).issubset(positions):
                errors = {
                    name: abs(positions[name] - target)
                    for name, target in targets.items()
                }
                if max(errors.values()) <= GOAL_TOLERANCE_RAD:
                    result.state = self._result_state(positions, velocities, state)
                    result.stalled = False
                    result.reached_goal = True
                    goal_handle.succeed()
                    return result
                active_velocity = abs(float(velocities.get(ACTIVE_JOINT, 0.0)))
                if active_velocity > self._stall_velocity_threshold:
                    last_movement_time = time.monotonic()
                elif (
                    self._allow_stalling
                    and time.monotonic() - last_movement_time >= self._stall_timeout
                ):
                    # Match the documented ros2_control stall-success result:
                    # the action terminal state is successful, while physical
                    # grasp success remains exclusively the M3 verifier's job.
                    result.state = self._result_state(positions, velocities, state)
                    result.stalled = True
                    result.reached_goal = False
                    goal_handle.succeed()
                    return result
            time.sleep(0.05)

        positions, velocities, state = self._snapshot()
        result.state = self._result_state(positions, velocities, state)
        result.stalled = True
        result.reached_goal = False
        goal_handle.abort()
        return result

    def close(self) -> None:
        self._server.destroy()
        self.destroy_node()


def main() -> None:
    rclpy.init()
    node = RobotiqGripperActionAdapter()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.close()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
