"""Offline contracts for the Robotiq 2F-85 four-bar closed-form driver."""

from __future__ import annotations

import math

import pytest

from extensions.gazebo.robot_control import Robotiq2F85Calibration
from extensions.gazebo.robotiq_kinematics import (
    BD_M,
    aperture_from_angle_closed_form,
    six_joint_positions,
    solve_four_bar,
)


def test_loop_closes_to_submillimetre_across_the_stroke() -> None:
    from extensions.gazebo.robotiq_kinematics import (
        A_XZ,
        B_XZ,
        C_IN_KNUCKLE_XZ,
        D_IN_TIP_XZ,
        _rot,
    )

    for i in range(17):
        theta = 0.8 * i / 16
        solved = solve_four_bar(theta)
        rc = _rot(theta, C_IN_KNUCKLE_XZ)
        c = (A_XZ[0] + rc[0], A_XZ[1] + rc[1])
        rd = _rot(theta + solved["tip_rad"], D_IN_TIP_XZ)
        d = (c[0] + rd[0], c[1] + rd[1])
        error = math.hypot(d[0] - B_XZ[0], d[1] - B_XZ[1]) - BD_M
        assert abs(error) < 1e-9


def test_solution_is_continuous_with_zero_and_matches_joint_limits() -> None:
    zero = solve_four_bar(0.0)
    assert zero["tip_rad"] == pytest.approx(0.0, abs=1e-6)
    previous = None
    for i in range(9):
        theta = 0.7929 * i / 8
        solved = solve_four_bar(theta)
        assert -0.8 <= solved["tip_rad"] <= 1e-9
        assert 0.0 <= solved["inner_knuckle_rad"] - solve_four_bar(0.0)["inner_knuckle_rad"] <= 0.8
        if previous is not None:
            assert solved["tip_rad"] < previous["tip_rad"]
        previous = solved


def test_aperture_endpoints_match_the_fk_calibration_table() -> None:
    calibration = Robotiq2F85Calibration()
    for theta in (0.0, 0.7929):
        assert aperture_from_angle_closed_form(theta) == pytest.approx(
            calibration.aperture_from_angle(theta), abs=1e-3
        )
    # Mid-stroke the exact four-bar differs from the coarse vendor table, but
    # must stay monotonic and inside the physical range.
    apertures = [aperture_from_angle_closed_form(0.1 * i) for i in range(8)]
    assert all(b < a for a, b in zip(apertures, apertures[1:]))
    assert 0.0 < apertures[-1] < 0.085


def test_six_joint_positions_mirror_and_respect_legacy_shape() -> None:
    positions = six_joint_positions(0.4)
    assert positions["gripper_right_finger_joint"] == -positions["gripper_left_finger_joint"]
    assert positions["gripper_right_inner_knuckle_joint"] == -positions["gripper_left_inner_knuckle_joint"]
    assert positions["gripper_right_finger_tip_joint"] == -positions["gripper_left_finger_tip_joint"]
    # The vendor constant multipliers are already within 2 mrad of the exact
    # solution at this point; the closed form keeps that behaviour.
    assert positions["gripper_left_inner_knuckle_joint"] == pytest.approx(0.4, abs=0.05)
    assert positions["gripper_left_finger_tip_joint"] == pytest.approx(-0.4, abs=0.05)


def test_out_of_range_active_angles_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        solve_four_bar(-0.1)
    with pytest.raises(ValueError, match="outside"):
        solve_four_bar(0.9)
