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
