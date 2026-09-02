#!/usr/bin/env python3
"""Expose the single-actuator Robotiq linkage as ParallelGripperCommand.

Gazebo Harmonic does not propagate the vendor mimic chain reliably, so six
position systems still receive commands.  They are never controlled as six
degrees of freedom: every cycle publishes one common driver value through the
exact four-bar solution in ``extensions.gazebo.robotiq_kinematics``.  Native
pad contact can slow or stop that common driver, but can never freeze or
advance one side independently.
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
            bounded_contact_hold_position,
            common_driver_position,
            functional_opening_complete,
            linkage_terminal_metrics,
            minimum_feasible_active_position,
            six_joint_positions,
            stroke_scaled_ramp_duration,
        )

        return (
            six_joint_positions,
            linkage_terminal_metrics,
            common_driver_position,
            bounded_contact_hold_position,
            functional_opening_complete,
            minimum_feasible_active_position,
            stroke_scaled_ramp_duration,
        )
    except ImportError:
        pass
    for parent in Path(__file__).resolve().parents:
        if (parent / "extensions" / "gazebo" / "robotiq_kinematics.py").is_file():
            sys.path.insert(0, str(parent))
            from extensions.gazebo.robotiq_kinematics import (
                bounded_contact_hold_position,
                common_driver_position,
                functional_opening_complete,
                linkage_terminal_metrics,
                minimum_feasible_active_position,
                six_joint_positions,
                stroke_scaled_ramp_duration,
            )

            return (
                six_joint_positions,
                linkage_terminal_metrics,
                common_driver_position,
                bounded_contact_hold_position,
                functional_opening_complete,
                minimum_feasible_active_position,
                stroke_scaled_ramp_duration,
            )
    raise RuntimeError("extensions/gazebo/robotiq_kinematics.py not found relative to the adapter")


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
SIDES: Final = ("left", "right")
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
# Servo the six Gazebo position systems through a simulator-time ramp instead
# of step-commanding the full stroke.  ``RAMP_S`` describes a complete driver
# stroke; shorter internal relief moves scale with their actual travel and keep
# at least one terminal-settle interval for fresh linkage feedback.
RAMP_S: Final = 1.0
SLOW_TAIL_FRACTION: Final = 0.55
SLOW_TAIL_FACTOR: Final = 0.25
MIN_STROKE_RAMP_SIM_S: Final = TERMINAL_LINKAGE_SETTLE_DWELL_SIM_S
# The single public driver is never commanded more than three terminal bands
# ahead of the slower mirrored side.  This bounds simulated squeeze while
# preserving enough finite preload for a supported workpiece to self-centre.
# It is a gripper-controller limit, independent of scene and object identity.
MAX_LEAD_RAD: Final = 3.0 * GOAL_TOLERANCE_RAD
# A first native pad contact slows the one common driver.  It does not create a
# per-side hold.  Reuse the already-defined contact-tail rate so the behaviour
# remains tied to the public ramp profile rather than an object-specific tune.
CONTACT_CLOSING_RATE_FACTOR: Final = SLOW_TAIL_FACTOR
TERMINAL_CONTACT_FRESHNESS_SIM_S: Final = 0.10
TERMINAL_BILATERAL_CONTACT_DWELL_SIM_S: Final = 0.25
# Once both pads touch, retain only half a public terminal band of preload.
# This is enough for a position-controlled simulator to preserve contact while
# preventing the old pre-contact command from squeezing an object that is
# subsequently fixed to the wrist.  It remains one common actuator target.
BILATERAL_HOLD_PRELOAD_RAD: Final = GOAL_TOLERANCE_RAD / 2.0
# A contact remains current for a simulator-time interval longer than the
# bilateral proof freshness.  Wall-clock load cannot expire physical contact.
CONTACT_FRESHNESS_SIM_S: Final = 0.25


class RobotiqGripperActionAdapter(Node):
    def __init__(self) -> None:
        super().__init__("openeta_robotiq_gripper_action_adapter")
        self.declare_parameter("allow_stalling", False)
        # Keep the generic/native-grasp default short.  motion-control's live acceptance launch
        # explicitly raises this to match its 90-second controller budget.
        self.declare_parameter("action_timeout_s", ACTION_TIMEOUT_S)
        self.declare_parameter("ramp_s", RAMP_S)
        self.declare_parameter("max_lead_rad", MAX_LEAD_RAD)
        self.declare_parameter("target_model_name", "target_object")
        self.declare_parameter("target_model_names", ["target_object"])
        (
            self._six_joint_positions,
            self._linkage_terminal_metrics,
            self._common_driver_position,
            self._bounded_contact_hold_position,
            self._functional_opening_complete,
            self._minimum_feasible_active_position,
            self._stroke_scaled_ramp_duration,
        ) = _load_robotiq_kinematics()
        self._allow_stalling = bool(self.get_parameter("allow_stalling").value)
        self._action_timeout_s = float(self.get_parameter("action_timeout_s").value)
        self._target_model_name = str(self.get_parameter("target_model_name").value).strip()
        configured_target_names = [
            str(value).strip()
            for value in self.get_parameter("target_model_names").value
            if str(value).strip()
        ]
        self._target_model_names = tuple(
            dict.fromkeys(configured_target_names or [self._target_model_name])
        )
        if self._action_timeout_s <= 0 or not self._target_model_names:
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

        This signal slows or stops the one common close driver; it never admits
        a grasp.  DirectEnv independently verifies sustained bilateral native
        contacts in the terminal close window before issuing the detachable-
        joint attach command.
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
            if any(
                target_model in name
                for target_model in self._target_model_names
                for name in names
            ):
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
        # The exact linkage cannot realise the nominal zero-angle endpoint,
        # and commanding a driver exactly onto either hard stop makes DART's
        # position system chatter.  Inset the *one common driver* first, then
        # derive all six targets from that value so every command remains on
        # the four-bar manifold.
        feasible_open_position = self._minimum_feasible_active_position(
            boundary_inset_rad=CONTROLLER_BOUNDARY_INSET_RAD
        )
        effective_active_position = min(
            max(active_position, feasible_open_position),
            MAX_POSITION_RAD - CONTROLLER_BOUNDARY_INSET_RAD,
        )
        requested_targets = dict(self._six_joint_positions(active_position))
        final_targets = dict(self._six_joint_positions(effective_active_position))
        start_positions, _, _ = self._snapshot()
        result = ParallelGripperCommand.Result()
        goal_started = time.monotonic()
        goal_started_sim_time_s = self._sim_time_s()
        _, goal_contact_sequences = self._target_contact_snapshot()
        deadline = goal_started + self._action_timeout_s
        ramp_s = float(self.get_parameter("ramp_s").value)
        max_lead = float(self.get_parameter("max_lead_rad").value)
        if ramp_s <= 0.0 or max_lead <= 0.0:
            raise ValueError("ramp duration and common-driver lead must be positive")
        try:
            left_start = float(start_positions["gripper_left_finger_joint"])
            right_start = -float(start_positions["gripper_right_finger_joint"])
            if not math.isfinite(left_start) or not math.isfinite(right_start):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            self.get_logger().error("complete mirrored outer-finger state is unavailable")
            result.state = self._result_state(start_positions, {}, None)
            result.stalled = False
            result.reached_goal = False
            goal_handle.abort()
            return result

        start_active_position = min(
            max((left_start + right_start) * 0.5, feasible_open_position),
            MAX_POSITION_RAD - CONTROLLER_BOUNDARY_INSET_RAD,
        )
        stroke = effective_active_position - start_active_position
        closing_goal = stroke > GOAL_TOLERANCE_RAD
        opening_goal = stroke < -GOAL_TOLERANCE_RAD
        motion_ramp_s = self._stroke_scaled_ramp_duration(
            full_stroke_duration_s=ramp_s,
            stroke_rad=stroke,
            full_stroke_rad=MAX_POSITION_RAD - MIN_POSITION_RAD,
            minimum_duration_s=MIN_STROKE_RAMP_SIM_S,
        )
        # A recovery open can begin in an already functionally open but
        # passively deflected linkage state after a rejected close.  Classify
        # the requested endpoint itself, not the signed stroke inferred from
        # the two exposed outer joints; otherwise that safe no-op waits for
        # the controller timeout even though the functional terminal proof
        # below already holds.
        full_open_goal = bool(
            math.isclose(
                effective_active_position,
                feasible_open_position,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        alpha = 0.0
        commanded_active_position = start_active_position
        last_sim_tick_s = goal_started_sim_time_s
        bilateral_contact_started_sim_time_s: float | None = None
        bilateral_settle_started_sim_time_s: float | None = None
        bilateral_hold_active_position: float | None = None
        linkage_settle_started_sim_time_s: float | None = None
        functional_open_settle_started_sim_time_s: float | None = None

        while rclpy.ok():
            positions, velocities, state = self._snapshot()
            target_contact_sim_times, target_contact_sequences = self._target_contact_snapshot()
            if goal_handle.is_cancel_requested:
                if positions:
                    self._publish_targets(positions)
                result.state = self._result_state(positions, velocities, state)
                result.stalled = False
                result.reached_goal = False
                goal_handle.canceled()
                return result

            sim_now_s = self._sim_time_s()
            fresh_contact_sides = {
                side
                for side in SIDES
                if target_contact_sequences.get(side, 0) > goal_contact_sequences.get(side, 0)
                and target_contact_sim_times.get(side, -math.inf) >= goal_started_sim_time_s
                and 0.0
                <= sim_now_s - target_contact_sim_times.get(side, -math.inf)
                <= CONTACT_FRESHNESS_SIM_S
            }
            fresh_bilateral_contact = fresh_contact_sides == set(SIDES)

            sim_dt = max(0.0, sim_now_s - last_sim_tick_s)
            last_sim_tick_s = max(last_sim_tick_s, sim_now_s)
            # Closing retains the low-impact terminal tail and slows further
            # on first pad contact. Opening moves away from the workpiece; its
            # existing common-driver lead bound already prevents an unbounded
            # step or asymmetric linkage command, so a second slow tail adds
            # no physical protection.
            rate_factor = (
                SLOW_TAIL_FACTOR
                if closing_goal and alpha >= SLOW_TAIL_FRACTION
                else 1.0
            )
            if closing_goal and fresh_contact_sides:
                rate_factor = min(rate_factor, CONTACT_CLOSING_RATE_FACTOR)
            alpha += sim_dt / motion_ramp_s * rate_factor
            alpha = min(1.0, max(alpha, 0.0))
            nominal_active_position = start_active_position + stroke * alpha
            try:
                common_active_position = self._common_driver_position(
                    positions,
                    closing=closing_goal,
                )
            except ValueError:
                common_active_position = math.nan
            if closing_goal and math.isfinite(common_active_position):
                # Before contact, keep the position systems close to the
                # measured common mechanism.  Once either pad touches, the
                # slowly advancing *single* target must be allowed to build
                # force and move the workpiece toward the opposing pad.  A
                # fixed angular lead ceiling turns that ordinary self-centring
                # phase into an artificial one-sided stall: the blocked side
                # defines the conservative measured driver forever, so the
                # target can never advance far enough to transfer force.
                #
                # All six targets below still come from this one scalar and
                # the exact four-bar map.  Contact only changes how far that
                # common actuator may lag its measured mechanism; it never
                # creates a per-side command.
                commanded_active_position = min(
                    nominal_active_position,
                    (
                        effective_active_position
                        if fresh_contact_sides
                        else common_active_position + max_lead
                    ),
                    effective_active_position,
                )
            elif opening_goal and math.isfinite(common_active_position):
                commanded_active_position = max(
                    nominal_active_position,
                    common_active_position - max_lead,
                    effective_active_position,
                )
            else:
                commanded_active_position = nominal_active_position

            bilateral_control_ready = bool(
                fresh_bilateral_contact
                and closing_goal
                and math.isfinite(common_active_position)
            )
            if bilateral_control_ready:
                if bilateral_contact_started_sim_time_s is None:
                    bilateral_contact_started_sim_time_s = sim_now_s
                    bilateral_hold_active_position = self._bounded_contact_hold_position(
                        measured_common_active_rad=common_active_position,
                        requested_active_rad=effective_active_position,
                        preload_rad=BILATERAL_HOLD_PRELOAD_RAD,
                    )
                commanded_active_position = float(bilateral_hold_active_position)
            else:
                bilateral_contact_started_sim_time_s = None
                bilateral_settle_started_sim_time_s = None
                bilateral_hold_active_position = None

            # Every publication is generated from exactly one common driver;
            # actual Gazebo contact deflection may differ, but software never
            # creates an asymmetric target or a second actuator.
            commanded_targets = dict(self._six_joint_positions(commanded_active_position))
            self._publish_targets(commanded_targets)

            complete_velocity_state = set(JOINT_MULTIPLIERS).issubset(velocities)
            bilateral_mechanism_stationary = bool(
                bilateral_control_ready
                and complete_velocity_state
                and all(math.isfinite(float(velocities[name])) for name in JOINT_MULTIPLIERS)
                and max(abs(float(velocities[name])) for name in JOINT_MULTIPLIERS)
                <= TERMINAL_LINKAGE_MAX_VELOCITY_RAD_S
            )
            if bilateral_mechanism_stationary:
                if bilateral_settle_started_sim_time_s is None:
                    bilateral_settle_started_sim_time_s = sim_now_s
            else:
                bilateral_settle_started_sim_time_s = None

            bilateral_dwell_complete = bool(
                self._allow_stalling
                and bilateral_contact_started_sim_time_s is not None
                and sim_now_s - bilateral_contact_started_sim_time_s
                >= TERMINAL_BILATERAL_CONTACT_DWELL_SIM_S
                and bilateral_settle_started_sim_time_s is not None
                and sim_now_s - bilateral_settle_started_sim_time_s
                >= TERMINAL_LINKAGE_SETTLE_DWELL_SIM_S
                and all(
                    0.0
                    <= sim_now_s - target_contact_sim_times.get(side, -math.inf)
                    <= TERMINAL_CONTACT_FRESHNESS_SIM_S
                    for side in SIDES
                )
            )
            if bilateral_dwell_complete:
                result.state = self._result_state(positions, velocities, state)
                result.stalled = True
                result.reached_goal = False
                goal_handle.succeed()
                return result

            functional_open_ready = bool(
                full_open_goal
                and self._functional_opening_complete(
                    positions,
                    velocities,
                    open_active_rad=feasible_open_position,
                    max_common_lead_rad=max_lead,
                    terminal_tolerance_rad=GOAL_TOLERANCE_RAD,
                    terminal_velocity_rad_s=(
                        TERMINAL_LINKAGE_MAX_VELOCITY_RAD_S
                    ),
                )
            )
            if functional_open_ready:
                if functional_open_settle_started_sim_time_s is None:
                    functional_open_settle_started_sim_time_s = sim_now_s
            else:
                functional_open_settle_started_sim_time_s = None
            functional_open_settled = bool(
                functional_open_settle_started_sim_time_s is not None
                and sim_now_s - functional_open_settle_started_sim_time_s
                >= TERMINAL_LINKAGE_SETTLE_DWELL_SIM_S
            )
            if functional_open_settled:
                self.get_logger().info(
                    "accepting full-open Robotiq recovery from bounded "
                    "common-driver state with stationary passive linkage"
                )
                result.state = self._result_state(positions, velocities, state)
                result.stalled = False
                result.reached_goal = True
                goal_handle.succeed()
                return result

            if set(final_targets).issubset(positions):
                try:
                    max_position_error_rad, max_abs_velocity_rad_s = self._linkage_terminal_metrics(
                        final_targets, positions, velocities
                    )
                except ValueError:
                    max_position_error_rad = math.inf
                    max_abs_velocity_rad_s = math.inf
                linkage_in_terminal_band = bool(
                    max_position_error_rad <= GOAL_TOLERANCE_RAD
                    and max_abs_velocity_rad_s <= TERMINAL_LINKAGE_MAX_VELOCITY_RAD_S
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
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)

        positions, velocities, state = self._snapshot()
        try:
            terminal_error, terminal_speed = self._linkage_terminal_metrics(
                final_targets, positions, velocities
            )
        except ValueError:
            terminal_error, terminal_speed = math.inf, math.inf
        self.get_logger().error(
            "Robotiq action timed out: "
            f"requested_active={active_position:.6f}, "
            f"effective_active={effective_active_position:.6f}, "
            f"closing={closing_goal}, "
            f"requested_targets={requested_targets}, controller_targets={final_targets}, "
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
