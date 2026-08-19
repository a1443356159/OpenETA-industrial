#!/usr/bin/env python3
"""Expose the Robotiq Gazebo linkage as ParallelGripperCommand.

Gazebo Harmonic's URDF importer does not propagate all five Robotiq mimic
joints when only the outer knuckle is commanded.  The robot-local Gazebo
position systems therefore receive per-joint targets computed from the
one-DOF four-bar closed-form solution (``extensions.gazebo.robotiq_kinematics``),
keeping the six commanded positions geometrically consistent with the closed
linkage; the legacy constant-multiplier expansion stays available behind the
``drive_mode`` parameter for rollback comparison.  Callers retain the
standard one-joint ROS action contract.
"""

from __future__ import annotations

from pathlib import Path
import sys
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


def _load_six_joint_positions():
    """Import the repository's pure four-bar solver (no ROS dependency)."""

    try:
        from extensions.gazebo.robotiq_kinematics import six_joint_positions

        return six_joint_positions
    except ImportError:
        pass
    for parent in Path(__file__).resolve().parents:
        if (parent / "extensions" / "gazebo" / "robotiq_kinematics.py").is_file():
            sys.path.insert(0, str(parent))
            from extensions.gazebo.robotiq_kinematics import six_joint_positions

            return six_joint_positions
    raise RuntimeError(
        "extensions/gazebo/robotiq_kinematics.py not found relative to the adapter"
    )


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
# Each finger is an independent four-bar side.  Contact is watched per side
# (through its finger joint) so the first pad to touch the target freezes in
# place instead of carrying the object across the rest of the closing stroke.
SIDE_JOINTS: Final = {
    "left": (
        "gripper_left_finger_joint",
        "gripper_left_inner_knuckle_joint",
        "gripper_left_finger_tip_joint",
    ),
    "right": (
        "gripper_right_finger_joint",
        "gripper_right_inner_knuckle_joint",
        "gripper_right_finger_tip_joint",
    ),
}
SIDE_WATCH_JOINTS: Final = {
    "left": "gripper_left_finger_joint",
    "right": "gripper_right_finger_joint",
}
MIN_POSITION_RAD: Final = 0.0
MAX_POSITION_RAD: Final = 0.8
GOAL_TOLERANCE_RAD: Final = 0.02
ACTION_TIMEOUT_S: Final = 20.0
# Servo the six Gazebo position systems through a short ramp instead of
# step-commanding the full stroke.  A step command slams the pads into a
# grasped object at full PID authority and the impact ejects light targets
# before the stall detector can settle.  The stroke runs fast across the
# free-space part and slows for the final part where contact happens; a
# uniformly slow ramp lets the linkage bind and produces false stalls.
RAMP_S: Final = 1.0
SLOW_TAIL_FRACTION: Final = 0.55
SLOW_TAIL_FACTOR: Final = 0.25
# While the active joint is blocked the ramp pauses (factor 0), so the
# position error (and therefore the squeeze force) never grows past contact.
# Retained for rollback comparison; superseded by the max_lead_rad lead limit
# (a pure pause can deadlock into a false stall when a velocity sample dips).
BLOCKED_RAMP_FACTOR: Final = 0.0
# Lead limit for the command ramp: the published active-joint target stays at
# most this far ahead of the measured position, bounding the squeeze force on
# contact while always pulling the joint forward (no deadlock).
MAX_LEAD_RAD: Final = 0.06
# Extra closing stroke (active joint, rad) commanded per joint when a stall is
# held.  Each joint keeps its own measured position plus at most this offset,
# so a stressed four-bar linkage is never forced back into the nominal mimic
# relation (which ejects a held object).  The default stays a pure freeze for
# the generic contract; the native-grasp profile layers a positive offset on top of its
# per-side freeze and anti-slip scene to keep a sustained pinch through the
# carry, and the parameter remains for physics tuning.
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
        # Keep the generic/native-grasp default short.  motion-control's live acceptance launch
        # explicitly raises this to match its 90-second controller budget.
        self.declare_parameter("action_timeout_s", ACTION_TIMEOUT_S)
        self.declare_parameter("ramp_s", RAMP_S)
        self.declare_parameter("blocked_ramp_factor", BLOCKED_RAMP_FACTOR)
        self.declare_parameter("stall_hold_extra_rad", STALL_HOLD_EXTRA_RAD)
        self.declare_parameter("max_lead_rad", MAX_LEAD_RAD)
        # "four_bar" (default) drives the six joints with the geometrically
        # consistent closed-form solution; "multiplier" reproduces the legacy
        # constant-mimic expansion for rollback comparison.
        self.declare_parameter("drive_mode", "four_bar")
        self._six_joint_positions = _load_six_joint_positions()
        self._allow_stalling = bool(self.get_parameter("allow_stalling").value)
        self._stall_velocity_threshold = float(
            self.get_parameter("stall_velocity_threshold").value
        )
        self._stall_timeout = float(self.get_parameter("stall_timeout").value)
        self._action_timeout_s = float(self.get_parameter("action_timeout_s").value)
        if (
            self._stall_velocity_threshold < 0
            or self._stall_timeout <= 0
            or self._action_timeout_s <= 0
        ):
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
        if str(self.get_parameter("drive_mode").value) == "multiplier":
            targets = expanded_targets(active_position)
        else:
            targets = dict(self._six_joint_positions(active_position))
        start_positions, _, _ = self._snapshot()
        result = ParallelGripperCommand.Result()
        deadline = time.monotonic() + self._action_timeout_s
        ramp_s = float(self.get_parameter("ramp_s").value)
        stall_hold_extra = float(self.get_parameter("stall_hold_extra_rad").value)
        max_lead = float(self.get_parameter("max_lead_rad").value)
        alpha = 0.0
        last_tick = time.monotonic()
        # Per-side contact freeze: a side only becomes freeze-eligible after
        # its watch joint has demonstrably moved under this goal (otherwise
        # startup latency would read as a stall), and from then on a watch
        # joint that stays below the velocity threshold for stall_timeout is
        # treated as contact.  The frozen side holds its own measured
        # positions while the other side keeps closing, so a grasped object
        # is squeezed between the pads instead of carried by the first pad
        # to touch it.
        side_last_movement = {side: time.monotonic() for side in SIDE_JOINTS}
        side_moved = {side: False for side in SIDE_JOINTS}
        side_holds: dict[str, dict[str, float]] = {}

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
            # Time-based two-speed ramp: fast across free space, slow through
            # the contact tail.
            alpha += dt / ramp_s * (1.0 if alpha < SLOW_TAIL_FRACTION else SLOW_TAIL_FACTOR)
            # Lead limit: the published target never runs more than
            # max_lead_rad (watch joint) ahead of the measured position on any
            # unfrozen side.  A genuinely blocked joint therefore feels a
            # bounded squeeze instead of full-stroke pressure, without any
            # pause/deadlock state.  Frozen sides drop out of the cap so the
            # other side can finish its stroke.
            if set(JOINT_MULTIPLIERS).issubset(start_positions) and set(targets).issubset(positions):
                cap: float | None = None
                for side, joints in SIDE_JOINTS.items():
                    if side in side_holds:
                        continue
                    watch = SIDE_WATCH_JOINTS[side]
                    stroke = targets[watch] - start_positions[watch]
                    if abs(stroke) <= 1e-9:
                        continue
                    progress = (positions[watch] - start_positions[watch]) / stroke
                    side_cap = max(progress, 0.0) + max_lead / abs(stroke)
                    cap = side_cap if cap is None else min(cap, side_cap)
                if cap is not None:
                    alpha = min(alpha, cap)
            alpha = min(1.0, max(alpha, 0.0))

            # Re-publishing makes startup deterministic even if the bridge is
            # still establishing its Gazebo publisher on the first sample.
            if set(JOINT_MULTIPLIERS).issubset(start_positions):
                published = {
                    name: start_positions[name] + (target - start_positions[name]) * alpha
                    for name, target in targets.items()
                }
            else:
                published = dict(targets)
            for hold in side_holds.values():
                published.update(hold)
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
                if self._allow_stalling:
                    for side, joints in SIDE_JOINTS.items():
                        if side in side_holds:
                            continue
                        watch = SIDE_WATCH_JOINTS[side]
                        if abs(float(velocities.get(watch, 0.0))) > self._stall_velocity_threshold:
                            side_moved[side] = True
                            side_last_movement[side] = now
                        elif (
                            side_moved[side]
                            and now - side_last_movement[side] >= self._stall_timeout
                        ):
                            # Match the documented ros2_control stall-success
                            # result: the action terminal state is successful,
                            # while physical grasp success remains exclusively
                            # a higher-level manipulation verifier's job.
                            # Hold every joint of the
                            # contacted side at its own measured position plus
                            # at most a small per-joint offset toward its
                            # target.  Snapping the stressed four-bar linkage
                            # back to the nominal mimic vector (or keeping the
                            # unreachable full-stroke target) ejects a held
                            # object; a pure freeze can be too weak to survive
                            # the lift, so the offset stays tunable.
                            side_holds[side] = {
                                name: positions[name]
                                + max(
                                    -stall_hold_extra * abs(JOINT_MULTIPLIERS[name]),
                                    min(
                                        stall_hold_extra * abs(JOINT_MULTIPLIERS[name]),
                                        targets[name] - positions[name],
                                    ),
                                )
                                for name in joints
                            }
                    if len(side_holds) == len(SIDE_JOINTS):
                        for hold in side_holds.values():
                            published.update(hold)
                        self._publish_targets(published)
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
