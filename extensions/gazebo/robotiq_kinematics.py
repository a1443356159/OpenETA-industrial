"""Closed-form one-DOF kinematics of the Robotiq 2F-85 four-bar finger.

The vendor URDF approximates the finger's closed linkage with constant mimic
multipliers (inner knuckle = +1, finger tip = -1).  In Gazebo those six joints
are driven by independent position systems, and the approximate relations are
geometrically inconsistent with the real loop, so the linkage binds or drifts
1-2 cm at the pads.  This module solves the actual four-bar:

* ``A`` — knuckle (driver) pivot on the base, joint ``gripper_*_finger_joint``
* ``B`` — inner-knuckle pivot on the base, joint ``gripper_*_inner_knuckle_joint``
* ``C`` — fingertip pivot on the knuckle, joint ``gripper_*_finger_tip_joint``
* ``D`` — coupler pin between the inner-knuckle distal bore and the tip link

A/B/C come from the vendor URDF joint origins (exact).  The coupler point is
fitted from the collision-mesh bore centres (sub-mm), and the inner-knuckle
length ``|BD|`` is chosen so the loop closes exactly at the zero
configuration.  Everything is plain Python for offline unit tests.
"""

from __future__ import annotations

import math
from typing import Mapping

# Base-frame pivots of the LEFT finger, expressed in the sagittal (x, z)
# plane; all joint axes are (0, -1, 0) so the mechanism is planar.
A_XZ: tuple[float, float] = (0.03060114, 0.05490452)
B_XZ: tuple[float, float] = (0.0127, 0.06142)
# Tip pivot C in the knuckle (driver) link frame: fixed knuckle->finger offset
# plus the finger->tip joint origin from the vendor URDF.
C_IN_KNUCKLE_XZ: tuple[float, float] = (
    0.03152616 + 0.00563134,
    -0.00376347 + 0.04718515,
)
# Coupler bore D in the fingertip link frame (C-relative), circle-fitted from
# left_finger_tip.stl (fit std 0.09 mm).
D_IN_TIP_XZ: tuple[float, float] = (-0.01789, 0.00644)
# Distal bore in the inner-knuckle link frame, fitted from
# left_inner_knuckle.stl; used to report the inner-knuckle joint angle.
D_IN_INNER_KNUCKLE_XZ: tuple[float, float] = (0.03706, 0.04595)
# Pad face centroid in the fingertip link frame (xmin face of the tip mesh).
PAD_IN_TIP_XZ: tuple[float, float] = (-0.0253, 0.0283)


# Inner-knuckle length, chosen so the loop closes exactly at theta=0.
def _closure_length() -> float:
    cx = A_XZ[0] + C_IN_KNUCKLE_XZ[0] + D_IN_TIP_XZ[0] - B_XZ[0]
    cz = A_XZ[1] + C_IN_KNUCKLE_XZ[1] + D_IN_TIP_XZ[1] - B_XZ[1]
    return math.hypot(cx, cz)


BD_M: float = _closure_length()

_ACTIVE_MIN_RAD = 0.0
_ACTIVE_MAX_RAD = 0.8

GRIPPER_JOINT_BOUNDS_RAD: Mapping[str, tuple[float, float]] = {
    "gripper_left_finger_joint": (0.0, 0.8),
    "gripper_right_finger_joint": (-0.8, 0.0),
    "gripper_left_inner_knuckle_joint": (0.0, 0.8),
    "gripper_right_inner_knuckle_joint": (-0.8, 0.0),
    "gripper_left_finger_tip_joint": (-0.8, 0.0),
    "gripper_right_finger_tip_joint": (0.0, 0.8),
}

# Public terminal band of the common Robotiq action adapter.  Keep control
# invariants which depend on that band here, beside the pure linkage model, so
# the ROS adapter and the host-side attachment transition cannot drift apart.
GRIPPER_GOAL_TOLERANCE_RAD: float = 0.02


class AttachedTransportReliefUnavailable(ValueError):
    """The measured grasp cannot retain one proven pad-relief band."""

# Position controllers must not servo an independently modelled linkage joint
# onto a hard stop.  Half the public terminal band leaves room for both state-
# estimation noise and the terminal settle proof.  This is a robot-control
# invariant, not an object- or scene-dependent tuning value.
DEFAULT_CONTROLLER_BOUNDARY_INSET_RAD: float = (
    GRIPPER_GOAL_TOLERANCE_RAD / 2.0
)


def _rot(alpha: float, v: tuple[float, float]) -> tuple[float, float]:
    c, s = math.cos(alpha), math.sin(alpha)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def _angle(v: tuple[float, float]) -> float:
    return math.atan2(v[1], v[0])


def solve_four_bar(active_rad: float) -> Mapping[str, float]:
    """Return (tip_joint, inner_knuckle_joint) for the LEFT finger.

    ``active_rad`` is the driver joint (``gripper_left_finger_joint``).  The
    solution picks the assembly branch that is continuous with the zero
    configuration.  Raises ValueError when the requested angle is outside the
    mechanism's reachable range.
    """

    theta = float(active_rad)
    if not -1e-9 <= theta <= _ACTIVE_MAX_RAD + 1e-9:
        raise ValueError(f"active joint outside [0, 0.8] rad: {active_rad}")
    theta = min(max(theta, _ACTIVE_MIN_RAD), _ACTIVE_MAX_RAD)
    rc = _rot(theta, C_IN_KNUCKLE_XZ)
    c = (A_XZ[0] + rc[0], A_XZ[1] + rc[1])
    e = (c[0] - B_XZ[0], c[1] - B_XZ[1])
    e_norm = math.hypot(*e)
    d_norm = math.hypot(*D_IN_TIP_XZ)
    cosarg = (BD_M * BD_M - e_norm * e_norm - d_norm * d_norm) / (2.0 * e_norm * d_norm)
    if not -1.0 <= cosarg <= 1.0:
        raise ValueError(f"four-bar cannot close at active={active_rad}")
    gamma = _angle(e) - _angle(D_IN_TIP_XZ) + math.acos(cosarg)
    rd = _rot(gamma, D_IN_TIP_XZ)
    d = (c[0] + rd[0], c[1] + rd[1])
    phi = _angle((d[0] - B_XZ[0], d[1] - B_XZ[1])) - _angle(D_IN_INNER_KNUCKLE_XZ)
    return {"tip_rad": gamma - theta, "inner_knuckle_rad": phi}


def minimum_feasible_active_position(*, boundary_inset_rad: float) -> float:
    """Return the nearest physically realisable open driver angle.

    At the nominal ``active=0`` endpoint the exact vendor four-bar solution
    asks the inner knuckle for roughly -0.03 rad, outside its URDF [0, 0.8]
    limit.  Clamping that one link to zero makes the six independently driven
    Gazebo joints fight each other.  The inner-knuckle solution is monotonic
    over the stroke, so a deterministic bisection finds the first driver angle
    whose exact linkage stays ``boundary_inset_rad`` inside the lower stop.
    """

    inset = float(boundary_inset_rad)
    if not math.isfinite(inset) or inset <= 0.0:
        raise ValueError("controller boundary inset must be positive and finite")
    if inset >= (_ACTIVE_MAX_RAD - _ACTIVE_MIN_RAD) / 2.0:
        raise ValueError("controller boundary inset exceeds active joint span")

    lower = _ACTIVE_MIN_RAD
    upper = _ACTIVE_MAX_RAD
    if solve_four_bar(upper)["inner_knuckle_rad"] < inset:
        raise ValueError("no four-bar open endpoint satisfies the requested inset")
    # 64 iterations are deterministic and resolve far below floating-point
    # precision at this scale without introducing a tuned angular epsilon.
    for _ in range(64):
        midpoint = (lower + upper) / 2.0
        if solve_four_bar(midpoint)["inner_knuckle_rad"] >= inset:
            upper = midpoint
        else:
            lower = midpoint

    solved = solve_four_bar(upper)
    driver = upper
    inner = solved["inner_knuckle_rad"]
    tip_magnitude = -solved["tip_rad"]
    if min(driver, inner, tip_magnitude) < inset - 1e-12:
        raise ValueError("computed four-bar open endpoint violates lower joint margin")
    if max(driver, inner, tip_magnitude) > _ACTIVE_MAX_RAD + 1e-12:
        raise ValueError("computed four-bar open endpoint violates upper joint bound")
    return upper


def six_joint_positions(active_rad: float) -> Mapping[str, float]:
    """Map the standard active-joint command to all six gripper joints.

    The right finger mirrors the left one; the four-bar solution replaces the
    vendor's constant multiplier approximations for the inner knuckles and
    fingertip joints.  Targets are clamped to the URDF joint limits (the exact
    closure wants the inner knuckle 0.03 rad below zero at full open).
    """

    solved = solve_four_bar(active_rad)
    theta = min(max(float(active_rad), _ACTIVE_MIN_RAD), _ACTIVE_MAX_RAD)
    tip = solved["tip_rad"]
    inner = solved["inner_knuckle_rad"]
    return {
        "gripper_left_finger_joint": min(max(theta, 0.0), 0.8),
        "gripper_right_finger_joint": -min(max(theta, 0.0), 0.8),
        "gripper_left_inner_knuckle_joint": min(max(inner, 0.0), 0.8),
        "gripper_right_inner_knuckle_joint": -min(max(inner, 0.0), 0.8),
        "gripper_left_finger_tip_joint": -min(max(-tip, 0.0), 0.8),
        "gripper_right_finger_tip_joint": min(max(-tip, 0.0), 0.8),
    }


def common_driver_position(
    positions: Mapping[str, float],
    *,
    closing: bool,
) -> float:
    """Estimate the one physical driver from the two mirrored outer joints.

    Gazebo exposes both sides as independent state variables even though the
    real 2F-85 has one actuator.  During closing the less-closed side is the
    conservative mechanism progress; during opening the more-closed side is.
    Using that directional bound prevents a common command from running ahead
    of either simulated side without inventing a second actuator.
    """

    try:
        left = float(positions["gripper_left_finger_joint"])
        right = -float(positions["gripper_right_finger_joint"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("both mirrored outer-finger states are required") from exc
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("outer-finger states must be finite")
    if not all(
        _ACTIVE_MIN_RAD - 1e-6 <= value <= _ACTIVE_MAX_RAD + 1e-6 for value in (left, right)
    ):
        raise ValueError("outer-finger state is outside the active-joint range")
    return min(left, right) if closing else max(left, right)


def bounded_contact_hold_position(
    *,
    measured_common_active_rad: float,
    requested_active_rad: float,
    preload_rad: float,
) -> float:
    """Return a low-energy common-driver target after bilateral contact.

    The simulated six-joint linkage must retain one actuator semantics even
    after both pads touch.  Holding the pre-contact command can leave the
    position systems several terminal bands ahead of the physical mechanism;
    once the workpiece is attached that becomes an over-constrained squeeze
    against the wrist.  Instead, start from the measured common driver and add
    only a bounded preload, capped by the requested closing endpoint.

    ``preload_rad`` is a controller property, not an object or scene tune.
    The action adapter derives it from its public terminal tolerance.
    """

    measured = float(measured_common_active_rad)
    requested = float(requested_active_rad)
    preload = float(preload_rad)
    if not all(math.isfinite(value) for value in (measured, requested, preload)):
        raise ValueError("contact hold inputs must be finite")
    if not _ACTIVE_MIN_RAD <= measured <= _ACTIVE_MAX_RAD:
        raise ValueError("measured common driver is outside the active-joint range")
    if not _ACTIVE_MIN_RAD <= requested <= _ACTIVE_MAX_RAD:
        raise ValueError("requested common driver is outside the active-joint range")
    if not 0.0 < preload < (_ACTIVE_MAX_RAD - _ACTIVE_MIN_RAD):
        raise ValueError("contact hold preload must be positive and bounded")
    return min(requested, measured + preload)


def attached_transport_relief_position(
    *,
    measured_common_active_rad: float,
    minimum_active_rad: float,
    terminal_tolerance_rad: float = GRIPPER_GOAL_TOLERANCE_RAD,
) -> float:
    """Return the common-driver target used immediately after native attach.

    A fixed attachment and two position-controlled pads pressing the same body
    form a redundant constraint in Gazebo physics.  Once native bilateral
    contact has proved the grasp and the fixed-joint attach is acknowledged,
    the pads therefore move apart by two terminal bands.  Even if the opening
    action finishes at the near edge of its permitted error band, this proves
    at least one full band of physical relief while retaining the object on
    the fixed attachment.  Environment collisions on the object remain
    enabled throughout.

    The transition is entirely linkage/tolerance driven.  If the mechanism is
    already too near its open boundary to prove that relief, fail closed
    instead of silently retaining an over-constrained transport state.
    """

    measured = float(measured_common_active_rad)
    minimum = float(minimum_active_rad)
    tolerance = float(terminal_tolerance_rad)
    if not all(math.isfinite(value) for value in (measured, minimum, tolerance)):
        raise ValueError("attached transport relief inputs must be finite")
    if not _ACTIVE_MIN_RAD <= minimum <= measured <= _ACTIVE_MAX_RAD:
        raise ValueError("attached transport relief position is outside the active range")
    if not 0.0 < tolerance < (_ACTIVE_MAX_RAD - _ACTIVE_MIN_RAD) / 2.0:
        raise ValueError("attached transport terminal tolerance is invalid")
    required_travel = 2.0 * tolerance
    if measured - minimum < required_travel:
        raise AttachedTransportReliefUnavailable(
            "insufficient common-driver travel for attached transport relief"
        )
    return measured - required_travel


def functional_opening_complete(
    positions: Mapping[str, float],
    velocities: Mapping[str, float],
    *,
    open_active_rad: float,
    max_common_lead_rad: float,
    terminal_tolerance_rad: float,
    terminal_velocity_rad_s: float,
) -> bool:
    """Prove a safe full-open state using the real one-actuator semantics.

    The public full-open command is a recovery boundary, not a demand that
    six simulated passive joints independently match an ideal zero-load
    four-bar pose.  Both mirrored outer joints must prove a bounded common
    opening and remain mutually consistent; every linkage joint must remain
    inside its hard limit and stationary.  The allowable common position is
    the controller's existing maximum lead plus one public terminal band.

    This predicate is intentionally unsuitable for intermediate position
    commands such as attached transport relief; those still require the exact
    six-joint terminal target.
    """

    values = (
        open_active_rad,
        max_common_lead_rad,
        terminal_tolerance_rad,
        terminal_velocity_rad_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("functional opening limits must be finite")
    target = float(open_active_rad)
    max_lead = float(max_common_lead_rad)
    tolerance = float(terminal_tolerance_rad)
    terminal_velocity = float(terminal_velocity_rad_s)
    if not _ACTIVE_MIN_RAD <= target <= _ACTIVE_MAX_RAD:
        raise ValueError("functional opening target is outside the active range")
    if min(max_lead, tolerance, terminal_velocity) <= 0.0:
        raise ValueError("functional opening limits must be positive")
    required = set(GRIPPER_JOINT_BOUNDS_RAD)
    if not required.issubset(positions) or not required.issubset(velocities):
        return False
    try:
        joint_positions = {name: float(positions[name]) for name in required}
        joint_velocities = {name: float(velocities[name]) for name in required}
    except (TypeError, ValueError):
        return False
    if not all(
        math.isfinite(value)
        for value in (*joint_positions.values(), *joint_velocities.values())
    ):
        return False
    if any(
        value < lower - 1e-6 or value > upper + 1e-6
        for name, value in joint_positions.items()
        for lower, upper in (GRIPPER_JOINT_BOUNDS_RAD[name],)
    ):
        return False
    left = joint_positions["gripper_left_finger_joint"]
    right = -joint_positions["gripper_right_finger_joint"]
    if abs(left - right) > max_lead + 1e-9:
        return False
    if max(left, right) > target + max_lead + tolerance + 1e-9:
        return False
    return max(abs(value) for value in joint_velocities.values()) <= terminal_velocity


def one_pad_compliance_exhausted(
    *,
    single_contact_age_s: float,
    no_common_progress_age_s: float,
    commanded_active_rad: float,
    nominal_active_rad: float,
    measured_common_active_rad: float,
    max_lead_rad: float,
    progress_epsilon_rad: float,
    mechanism_stationary: bool,
    remaining_close_travel_rad: float,
    goal_tolerance_rad: float,
    compliance_dwell_s: float,
) -> bool:
    """Prove that bounded one-pad self-centring travel is exhausted.

    One pad touching is a normal intermediate state for a single-actuator
    parallel gripper and is never sufficient to fail a close.  Exhaustion is
    reported only after the common command has accumulated its complete
    bounded preload, the whole linkage is stationary, and the common driver
    has made no measurable progress throughout the compliance window while
    meaningful close travel remains.
    """

    values = (
        single_contact_age_s,
        no_common_progress_age_s,
        commanded_active_rad,
        nominal_active_rad,
        measured_common_active_rad,
        max_lead_rad,
        progress_epsilon_rad,
        remaining_close_travel_rad,
        goal_tolerance_rad,
        compliance_dwell_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("one-pad compliance evidence must be finite")
    if (
        single_contact_age_s < 0.0
        or no_common_progress_age_s < 0.0
        or max_lead_rad <= 0.0
        or progress_epsilon_rad < 0.0
        or progress_epsilon_rad >= max_lead_rad
        or goal_tolerance_rad <= 0.0
        or compliance_dwell_s <= 0.0
    ):
        raise ValueError("one-pad compliance limits are invalid")
    common_preload_saturated = bool(
        commanded_active_rad - measured_common_active_rad >= max_lead_rad - progress_epsilon_rad
        and nominal_active_rad - measured_common_active_rad >= max_lead_rad - progress_epsilon_rad
    )
    return bool(
        single_contact_age_s >= compliance_dwell_s
        and no_common_progress_age_s >= compliance_dwell_s
        and common_preload_saturated
        and mechanism_stationary
        and remaining_close_travel_rad > goal_tolerance_rad
    )


def linkage_terminal_metrics(
    targets: Mapping[str, float],
    positions: Mapping[str, float],
    velocities: Mapping[str, float],
) -> tuple[float, float]:
    """Return the worst six-link position error and speed.

    The Gazebo gripper is six independently controlled joints even though its
    public action has one degree of freedom.  A terminal decision therefore
    cannot be made from the active finger alone.  Missing or non-finite state
    is rejected so a stale/partial ``JointState`` can never look settled.
    """

    names = tuple(targets)
    if not names or not set(names).issubset(positions) or not set(names).issubset(velocities):
        raise ValueError("complete gripper linkage state is required")
    target_values = [float(targets[name]) for name in names]
    position_values = [float(positions[name]) for name in names]
    velocity_values = [float(velocities[name]) for name in names]
    if not all(
        math.isfinite(value) for value in (*target_values, *position_values, *velocity_values)
    ):
        raise ValueError("gripper linkage state must be finite")
    return (
        max(
            abs(position - target)
            for position, target in zip(position_values, target_values, strict=True)
        ),
        max(abs(velocity) for velocity in velocity_values),
    )


def controller_safe_targets(
    requested: Mapping[str, float],
    *,
    boundary_inset_rad: float,
) -> dict[str, float]:
    """Inset controller targets that coincide with a hard joint limit.

    DART's saturated position controller can alternate at its velocity limit
    when asked to hold a revolute joint exactly on a hard stop.  The public
    gripper result is still checked against the original request; this helper
    only chooses an equivalent controller target inside that request's
    tolerance band.  The rule is joint-limit driven and applies to either end
    of every linkage joint, independent of object or scene identity.
    """

    inset = float(boundary_inset_rad)
    if not math.isfinite(inset) or inset <= 0.0:
        raise ValueError("controller boundary inset must be positive and finite")
    safe: dict[str, float] = {}
    for name, raw_target in requested.items():
        if name not in GRIPPER_JOINT_BOUNDS_RAD:
            raise ValueError(f"unknown gripper linkage joint: {name}")
        lower, upper = GRIPPER_JOINT_BOUNDS_RAD[name]
        if inset * 2.0 >= upper - lower:
            raise ValueError("controller boundary inset exceeds joint span")
        target = float(raw_target)
        if not math.isfinite(target) or target < lower - 1e-9 or target > upper + 1e-9:
            raise ValueError(f"gripper target outside joint bounds: {name}")
        if math.isclose(target, lower, rel_tol=0.0, abs_tol=1e-9):
            target = lower + inset
        elif math.isclose(target, upper, rel_tol=0.0, abs_tol=1e-9):
            target = upper - inset
        safe[name] = target
    return safe


def left_pad_position_xz(active_rad: float) -> tuple[float, float]:
    """World-plane position of the left pad face centroid (base frame)."""

    theta = min(max(float(active_rad), _ACTIVE_MIN_RAD), _ACTIVE_MAX_RAD)
    solved = solve_four_bar(theta)
    gamma = theta + solved["tip_rad"]
    rc = _rot(theta, C_IN_KNUCKLE_XZ)
    c = (A_XZ[0] + rc[0], A_XZ[1] + rc[1])
    rp = _rot(gamma, PAD_IN_TIP_XZ)
    return (c[0] + rp[0], c[1] + rp[1])


def aperture_from_angle_closed_form(active_rad: float) -> float:
    """Total aperture implied by the four-bar (2 x left-pad x coordinate)."""

    return 2.0 * left_pad_position_xz(active_rad)[0]
