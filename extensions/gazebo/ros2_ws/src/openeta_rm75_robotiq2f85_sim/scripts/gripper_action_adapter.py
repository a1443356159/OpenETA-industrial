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
# Servo the six Gazebo position systems through a short ramp instead of
# step-commanding the full stroke.  A step command slams the pads into a
# grasped object at full PID authority and the impact ejects light targets
# before the stall detector can settle.
RAMP_S: Final = 1.0
# While the active joint is blocked the ramp does not stop; it creeps at this
# fraction of the normal rate so a deterministic, gentle contact force builds
# during the stall-detection window instead of an impact-then-release.
BLOCKED_RAMP_FACTOR: Final = 0.0
# Extra closing stroke (active joint, rad) commanded per joint when a stall is
# held.  Each joint keeps its own measured position plus at most this offset,
# so a stressed four-bar linkage is never forced back into the nominal mimic
# relation (which ejects a held object).  Live sweeps found no positive value
# that reliably survives the lift (0.01-0.02 slips, 0.03 can eject), so the
# default is a pure freeze; the parameter remains for physics tuning.
STALL_HOLD_EXTRA_RAD: Final = 0.0


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
        self.declare_parameter("ramp_s", RAMP_S)
        self.declare_parameter("blocked_ramp_factor", BLOCKED_RAMP_FACTOR)
        self.declare_parameter("stall_hold_extra_rad", STALL_HOLD_EXTRA_RAD)
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
        start_positions, _, _ = self._snapshot()
        result = ParallelGripperCommand.Result()
        deadline = time.monotonic() + ACTION_TIMEOUT_S
        last_movement_time = time.monotonic()
        ramp_s = float(self.get_parameter("ramp_s").value)
        blocked_ramp_factor = float(self.get_parameter("blocked_ramp_factor").value)
        stall_hold_extra = float(self.get_parameter("stall_hold_extra_rad").value)
        ramp_clock = 0.0
        last_tick = time.monotonic()
        has_moved = False

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

            now = time.monotonic()
            dt = now - last_tick
            last_tick = now
            active_velocity_now = abs(float(velocities.get(ACTIVE_JOINT, 0.0)))
            if active_velocity_now > self._stall_velocity_threshold:
                has_moved = True
            errors_now = (
                {name: abs(positions[name] - target) for name, target in targets.items()}
                if set(targets).issubset(positions)
                else None
            )
            blocked = bool(
                has_moved
                and errors_now is not None
                and max(errors_now.values()) > GOAL_TOLERANCE_RAD
                and active_velocity_now <= self._stall_velocity_threshold
            )
            if not blocked:
                ramp_clock += dt
            else:
                ramp_clock += dt * blocked_ramp_factor

            # Re-publishing makes startup deterministic even if the bridge is
            # still establishing its Gazebo publisher on the first sample.
            if set(JOINT_MULTIPLIERS).issubset(start_positions):
                alpha = min(1.0, ramp_clock / ramp_s)
                published = {
                    name: start_positions[name] + (target - start_positions[name]) * alpha
                    for name, target in targets.items()
                }
            else:
                published = targets
            self._publish_targets(published)
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
                    # Hold every joint at its own measured position plus at
                    # most a small per-joint offset toward its target.  Snapping
                    # the stressed four-bar linkage back to the nominal mimic
                    # vector (or keeping the unreachable full-stroke target)
                    # ejects a held object; a pure freeze can be too weak to
                    # survive the lift, so the offset stays tunable.
                    if positions:
                        hold = {
                            name: position
                            + max(
                                -stall_hold_extra * abs(JOINT_MULTIPLIERS[name]),
                                min(
                                    stall_hold_extra * abs(JOINT_MULTIPLIERS[name]),
                                    targets[name] - position,
                                ),
                            )
                            for name, position in positions.items()
                        }
                        self._publish_targets(hold)
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
