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
from functools import partial
import math
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
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


def _load_robotiq_kinematics():
    """Import the repository's pure four-bar helpers (no ROS dependency)."""

    try:
        from extensions.gazebo.robotiq_kinematics import (
            controller_safe_targets,
            linkage_terminal_metrics,
            minimum_feasible_active_position,
            six_joint_positions,
        )

        return (
            six_joint_positions,
            linkage_terminal_metrics,
            controller_safe_targets,
            minimum_feasible_active_position,
        )
    except ImportError:
        pass
    for parent in Path(__file__).resolve().parents:
        if (parent / "extensions" / "gazebo" / "robotiq_kinematics.py").is_file():
            sys.path.insert(0, str(parent))
            from extensions.gazebo.robotiq_kinematics import (
                controller_safe_targets,
                linkage_terminal_metrics,
                minimum_feasible_active_position,
                six_joint_positions,
            )

            return (
                six_joint_positions,
                linkage_terminal_metrics,
                controller_safe_targets,
                minimum_feasible_active_position,
            )
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
# Reaching a position band while every linkage joint is still travelling at
# its 0.5 rad/s limit is not a completed gripper action.  Require the complete
# linkage to stay within the goal band and below a speed whose worst-case
# travel over the dwell cannot cross that band.  The values are related by the
# physical invariant below rather than tuned for an acceptance object.
TERMINAL_LINKAGE_SETTLE_DWELL_SIM_S: Final = 0.25
TERMINAL_LINKAGE_MAX_VELOCITY_RAD_S: Final = (
    GOAL_TOLERANCE_RAD / TERMINAL_LINKAGE_SETTLE_DWELL_SIM_S
)
# The controller target remains strictly inside a hard joint stop while the
# action result stays referenced to the original requested endpoint.
CONTROLLER_BOUNDARY_INSET_RAD: Final = GOAL_TOLERANCE_RAD / 2.0
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
# Once both independently frozen finger linkages have reached their terminal
# hold, wait for *fresh bilateral* target contacts before returning the action
# result.  A thin or rounded part can settle against the second pad near the
# end of the old fixed dwell; returning immediately truncated an otherwise
# stable native-contact proof.  This is only a bounded observation dwell: it
# adds no squeeze and DirectEnv still owns the authoritative 100 ms Gazebo-time
# bilateral-contact gate.
TERMINAL_CONTACT_FRESHNESS_SIM_S: Final = 0.10
TERMINAL_BILATERAL_CONTACT_DWELL_SIM_S: Final = 0.25
TERMINAL_HOLD_TIMEOUT_SIM_S: Final = 1.00
# A single contact sensor sample may be produced while a light or curved part
# is still settling between the fingers.  Do not latch that side forever: if
# the native pad stream stops, release only the contact-created hold and let
# the normal lead-limited ramp close it again.  Mechanical-endpoint holds stay
# latched because no additional stroke is available.  This is deliberately
# longer than the terminal freshness test so a slow Gazebo real-time factor
# cannot turn normal contact transport jitter into repeated hold/release
# oscillation.
CONTACT_HOLD_FRESHNESS_SIM_S: Final = 0.25


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
        # Keep the generic/native-grasp default short.  motion-control's live acceptance launch
        # explicitly raises this to match its 90-second controller budget.
        self.declare_parameter("action_timeout_s", ACTION_TIMEOUT_S)
        self.declare_parameter("ramp_s", RAMP_S)
        self.declare_parameter("blocked_ramp_factor", BLOCKED_RAMP_FACTOR)
        self.declare_parameter("stall_hold_extra_rad", STALL_HOLD_EXTRA_RAD)
        self.declare_parameter("max_lead_rad", MAX_LEAD_RAD)
        self.declare_parameter("target_model_name", "target_object")
        # "four_bar" (default) drives the six joints with the geometrically
        # consistent closed-form solution; "multiplier" reproduces the legacy
        # constant-mimic expansion for rollback comparison.
        self.declare_parameter("drive_mode", "four_bar")
        (
            self._six_joint_positions,
            self._linkage_terminal_metrics,
            self._controller_safe_targets,
            self._minimum_feasible_active_position,
        ) = _load_robotiq_kinematics()
        self._allow_stalling = bool(self.get_parameter("allow_stalling").value)
        self._action_timeout_s = float(self.get_parameter("action_timeout_s").value)
        self._target_model_name = str(
            self.get_parameter("target_model_name").value
        ).strip()
        if self._action_timeout_s <= 0 or not self._target_model_name:
            raise ValueError("action timeout and target model name are invalid")
        self._callback_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._positions: dict[str, float] = {}
        self._velocities: dict[str, float] = {}
        # Contact freshness is defined only in Gazebo simulation time.  Wall
        # time advances while a loaded simulator is slow, which previously
        # released valid pad holds every 250 ms and forced the close action to
        # oscillate until its timeout.  The monotonically increasing sequence
        # proves that a sample belongs to the current close goal.
        self._target_contact_sim_times: dict[str, float] = {}
        self._target_contact_sequences: dict[str, int] = {}
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
        self._contact_subscriptions = [
            self.create_subscription(
                Contacts,
                topic,
                partial(self._target_contact_callback, side),
                10,
                callback_group=self._callback_group,
            )
            for side, topic in (
                ("left", "/openeta/native_grasp/contacts/left_pad"),
                ("right", "/openeta/native_grasp/contacts/right_pad"),
            )
        ]
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

    def _target_contact_callback(self, side: str, message: Contacts) -> None:
        """Record only native pad contact with the configured target model.

        This signal terminates the mechanical close ramp; it never admits a
        grasp.  DirectEnv independently verifies sustained bilateral native
        contacts in the terminal close window before issuing the
        detachable-joint attach command.
        """

        stamp = message.header.stamp
        sim_time_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        # Some bridge versions emit the first Contacts sample with an unset
        # header stamp while /clock is already valid.  Treat that sample as
        # current Gazebo time instead of making it look infinitely stale; the
        # per-goal sequence barrier below still prevents pre-goal contact from
        # terminating a new close command.
        if not math.isfinite(sim_time_s) or sim_time_s <= 0.0:
            sim_time_s = self._sim_time_s()
        for contact in message.contacts:
            names = (
                str(contact.collision1.name),
                str(contact.collision2.name),
            )
            if any(self._target_model_name in name for name in names):
                with self._lock:
                    self._target_contact_sim_times[side] = sim_time_s
                    self._target_contact_sequences[side] = (
                        self._target_contact_sequences.get(side, 0) + 1
                    )
                return

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

    def _target_contact_snapshot(self) -> tuple[dict[str, float], dict[str, int]]:
        with self._lock:
            return (
                dict(self._target_contact_sim_times),
                dict(self._target_contact_sequences),
            )

    def _sim_time_s(self) -> float:
        """Read the node's Gazebo-backed ROS clock for physical dwell timing."""

        return float(self.get_clock().now().nanoseconds) * 1e-9

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
        drive_mode = str(self.get_parameter("drive_mode").value)
        feasible_open_position = MIN_POSITION_RAD
        if drive_mode == "multiplier":
            requested_targets = expanded_targets(active_position)
            effective_active_position = active_position
        else:
            # The exact linkage cannot realise the nominal zero-angle endpoint:
            # its inner knuckles would need to cross their lower hard stops.
            # Map only that unreachable open tail to the closest exact
            # four-bar state.  This preserves the public action range while
            # preventing six independent position systems from fighting an
            # impossible clamped linkage.
            feasible_open_position = self._minimum_feasible_active_position(
                boundary_inset_rad=CONTROLLER_BOUNDARY_INSET_RAD
            )
            effective_active_position = max(
                active_position, feasible_open_position
            )
            requested_targets = dict(
                self._six_joint_positions(effective_active_position)
            )
        targets = self._controller_safe_targets(
            requested_targets,
            boundary_inset_rad=CONTROLLER_BOUNDARY_INSET_RAD,
        )
        start_positions, _, _ = self._snapshot()
        # Native contact / load-stall handling is meaningful only while the
        # fingers are moving towards the closed calibration endpoint.  A
        # freshly detached workpiece can remain in contact with one pad for a
        # few samples while an open command starts; treating that contact as a
        # new close stall freezes the release and turns a sub-second recovery
        # into the full action timeout.
        closing_goal = bool(
            ACTIVE_JOINT in start_positions
            and (
                drive_mode == "multiplier"
                or active_position > feasible_open_position + GOAL_TOLERANCE_RAD
            )
            and effective_active_position
            > start_positions[ACTIVE_JOINT] + GOAL_TOLERANCE_RAD
        )
        result = ParallelGripperCommand.Result()
        goal_started = time.monotonic()
        _, goal_contact_sequences = self._target_contact_snapshot()
        deadline = goal_started + self._action_timeout_s
        ramp_s = float(self.get_parameter("ramp_s").value)
        stall_hold_extra = float(self.get_parameter("stall_hold_extra_rad").value)
        max_lead = float(self.get_parameter("max_lead_rad").value)
        alpha = 0.0
        last_tick = time.monotonic()
        # Per-side contact freeze: a side only becomes freeze-eligible after
        # its watch joint has demonstrably moved under this goal (otherwise a
        # stale contact sample could freeze an opening command).  Only native
        # target contact may freeze a side before its mechanical endpoint.
        # Four-bar joints can transiently report near-zero velocity while the
        # six position controllers settle; treating that as object contact
        # prematurely froze both pads with a 60+ mm aperture.  The frozen side
        # holds its measured linkage while the other side keeps closing.
        side_moved = {side: False for side in SIDE_JOINTS}
        side_holds: dict[str, dict[str, float]] = {}
        side_hold_sources: dict[str, str] = {}
        terminal_hold_started_sim_time_s: float | None = None
        bilateral_contact_started_sim_time_s: float | None = None
        linkage_settle_started_sim_time_s: float | None = None

        while rclpy.ok():
            if time.monotonic() >= deadline:
                break
            positions, velocities, state = self._snapshot()
            target_contact_sim_times, target_contact_sequences = (
                self._target_contact_snapshot()
            )
            if goal_handle.is_cancel_requested:
                if positions:
                    self._publish_targets(positions)
                result.state = self._result_state(positions, velocities, state)
                result.stalled = False
                result.reached_goal = False
                goal_handle.canceled()
                return result

            now = time.monotonic()
            sim_now_s = self._sim_time_s()
            bilateral_native_hold = bool(
                len(side_holds) == len(SIDE_JOINTS)
                and all(
                    side_hold_sources.get(side) == "native_target_contact"
                    for side in SIDE_JOINTS
                )
            )
            stale_contact_holds = [
                side
                for side in side_holds
                if side_hold_sources.get(side) == "native_target_contact"
                # Once both pads have independently produced target contact,
                # freeze the bounded terminal window.  DirectEnv's denser
                # history remains the authoritative bilateral-contact proof;
                # releasing one side here made the linkage oscillate until the
                # low-level timeout even when that proof was already present.
                and not bilateral_native_hold
                # A contact-created hold is valid only while that same pad
                # keeps reporting the target.  This remains true after the
                # other side has also touched: otherwise one transient sample
                # can latch both linkages until terminal timeout even though
                # the object has already separated from one pad.
                and sim_now_s - target_contact_sim_times.get(side, sim_now_s)
                > CONTACT_HOLD_FRESHNESS_SIM_S
            ]
            for side in stale_contact_holds:
                # A transient touch is not a mechanical terminal condition.
                # Resume this side under the existing max-lead bound so it can
                # follow a settling object and establish sustained contact.
                side_holds.pop(side, None)
                side_hold_sources.pop(side, None)
                self.get_logger().info(
                    f"releasing stale {side} Robotiq native-contact hold"
                )
            if stale_contact_holds:
                terminal_hold_started_sim_time_s = None
                bilateral_contact_started_sim_time_s = None
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
                try:
                    max_position_error_rad, max_abs_velocity_rad_s = (
                        self._linkage_terminal_metrics(
                            targets, positions, velocities
                        )
                    )
                except ValueError:
                    max_position_error_rad = math.inf
                    max_abs_velocity_rad_s = math.inf
                linkage_in_terminal_band = bool(
                    max_position_error_rad <= GOAL_TOLERANCE_RAD
                    and max_abs_velocity_rad_s
                    <= TERMINAL_LINKAGE_MAX_VELOCITY_RAD_S
                )
                if linkage_in_terminal_band:
                    if linkage_settle_started_sim_time_s is None:
                        linkage_settle_started_sim_time_s = sim_now_s
                else:
                    linkage_settle_started_sim_time_s = None
                linkage_settle_complete = bool(
                    linkage_settle_started_sim_time_s is not None
                    and sim_now_s - linkage_settle_started_sim_time_s
                    >= TERMINAL_LINKAGE_SETTLE_DWELL_SIM_S
                )
                if linkage_settle_complete:
                    result.state = self._result_state(positions, velocities, state)
                    result.stalled = False
                    result.reached_goal = True
                    goal_handle.succeed()
                    return result
                if self._allow_stalling and closing_goal:
                    for side, joints in SIDE_JOINTS.items():
                        if side in side_holds:
                            continue
                        watch = SIDE_WATCH_JOINTS[side]
                        if abs(positions[watch] - start_positions[watch]) > 0.005:
                            # A position delta is the authoritative movement
                            # proof when the sampled velocity falls between
                            # controller ticks.
                            side_moved[side] = True
                        native_target_contact = bool(
                            side_moved[side]
                            and target_contact_sequences.get(side, 0)
                            > goal_contact_sequences.get(side, 0)
                            and 0.0
                            <= sim_now_s - target_contact_sim_times.get(side, -math.inf)
                            <= CONTACT_HOLD_FRESHNESS_SIM_S
                        )
                        # Once the other side has reached the mechanical close
                        # endpoint it cannot produce a missing native contact
                        # by waiting longer.  Terminate the low-level action as
                        # a known stall immediately; DirectEnv's independent
                        # post-action bilateral-contact gate will reject this
                        # grasp and let the frozen high-level frontier advance.
                        # This avoids spending the full action timeout on a
                        # geometrically one-sided model grasp.
                        target_exhausted = bool(
                            side_moved[side]
                            and abs(positions[watch] - targets[watch])
                            <= GOAL_TOLERANCE_RAD
                        )
                        if native_target_contact or target_exhausted:
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
                            # object; a pure freeze can be too weak to retain
                            # contact during MoveIt transport, so the offset stays tunable.
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
                            side_hold_sources[side] = (
                                "native_target_contact"
                                if native_target_contact
                                else "mechanical_endpoint"
                            )
                            if native_target_contact:
                                self.get_logger().info(
                                    f"freezing {side} Robotiq side on native target contact"
                                )
                    if len(side_holds) == len(SIDE_JOINTS):
                        for hold in side_holds.values():
                            published.update(hold)
                        self._publish_targets(published)
                        sim_now_s = self._sim_time_s()
                        if terminal_hold_started_sim_time_s is None:
                            terminal_hold_started_sim_time_s = sim_now_s
                        fresh_bilateral_contact = all(
                            target_contact_sequences.get(side, 0)
                            > goal_contact_sequences.get(side, 0)
                            and 0.0
                            <= sim_now_s
                            - target_contact_sim_times.get(side, -math.inf)
                            <= TERMINAL_CONTACT_FRESHNESS_SIM_S
                            for side in SIDE_JOINTS
                        )
                        if fresh_bilateral_contact:
                            if bilateral_contact_started_sim_time_s is None:
                                bilateral_contact_started_sim_time_s = sim_now_s
                        else:
                            # A transient touch cannot accumulate dwell across
                            # a gap.  The independent native gate later checks
                            # the denser Gazebo-time sample history as well.
                            bilateral_contact_started_sim_time_s = None
                        bilateral_dwell_complete = bool(
                            bilateral_contact_started_sim_time_s is not None
                            and sim_now_s - bilateral_contact_started_sim_time_s
                            >= TERMINAL_BILATERAL_CONTACT_DWELL_SIM_S
                        )
                        terminal_wait_exhausted = bool(
                            sim_now_s - terminal_hold_started_sim_time_s
                            >= TERMINAL_HOLD_TIMEOUT_SIM_S
                        )
                        if bilateral_dwell_complete or terminal_wait_exhausted:
                            result.state = self._result_state(
                                positions, velocities, state
                            )
                            result.stalled = True
                            result.reached_goal = False
                            goal_handle.succeed()
                            return result
            time.sleep(0.05)

        positions, velocities, state = self._snapshot()
        try:
            terminal_error, terminal_speed = self._linkage_terminal_metrics(
                targets, positions, velocities
            )
        except ValueError:
            terminal_error, terminal_speed = math.inf, math.inf
        self.get_logger().error(
            "Robotiq action timed out: "
            f"requested_active={active_position:.6f}, "
            f"effective_active={effective_active_position:.6f}, "
            f"closing={closing_goal}, "
            f"requested_targets={requested_targets}, controller_targets={targets}, "
            f"positions={positions}, velocities={velocities}, "
            f"max_controller_error={terminal_error:.6f}, "
            f"max_abs_velocity={terminal_speed:.6f}"
        )
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
