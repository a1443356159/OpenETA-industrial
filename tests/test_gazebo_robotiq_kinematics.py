"""Offline contracts for the Robotiq 2F-85 four-bar closed-form driver."""

from __future__ import annotations

import math

import pytest

from extensions.gazebo.robot_control import Robotiq2F85Calibration
from extensions.gazebo.robotiq_kinematics import (
    AttachedTransportReliefUnavailable,
    attached_transport_relief_position,
    bounded_contact_hold_position,
    BD_M,
    DEFAULT_CONTROLLER_BOUNDARY_INSET_RAD,
    aperture_from_angle_closed_form,
    common_driver_position,
    controller_safe_targets,
    functional_opening_complete,
    linkage_terminal_metrics,
    minimum_feasible_active_position,
    six_joint_positions,
    solve_four_bar,
    stroke_scaled_ramp_duration,
)


def test_attached_transport_relief_proves_one_terminal_band_after_action_error() -> None:
    target = attached_transport_relief_position(
        measured_common_active_rad=0.60,
        minimum_active_rad=0.04,
        terminal_tolerance_rad=0.02,
    )

    assert target == pytest.approx(0.56)
    # The opening action may finish one terminal band on the closed side of
    # its target and still guarantees one complete band of physical relief.
    assert 0.60 - (target + 0.02) == pytest.approx(0.02)


def test_attached_transport_relief_fails_near_open_boundary() -> None:
    with pytest.raises(
        AttachedTransportReliefUnavailable,
        match="insufficient common-driver travel",
    ):
        attached_transport_relief_position(
            measured_common_active_rad=0.07,
            minimum_active_rad=0.04,
            terminal_tolerance_rad=0.02,
        )


def test_gripper_ramp_duration_scales_internal_relief_without_changing_full_stroke() -> None:
    assert stroke_scaled_ramp_duration(
        full_stroke_duration_s=1.0,
        stroke_rad=0.8,
        full_stroke_rad=0.8,
        minimum_duration_s=0.25,
    ) == pytest.approx(1.0)
    assert stroke_scaled_ramp_duration(
        full_stroke_duration_s=1.0,
        stroke_rad=-0.04,
        full_stroke_rad=0.8,
        minimum_duration_s=0.25,
    ) == pytest.approx(0.25)


def test_functional_full_open_accepts_stationary_bounded_passive_deflection() -> None:
    positions = {
        "gripper_left_finger_joint": 0.1007,
        "gripper_right_finger_joint": -0.0469,
        "gripper_left_inner_knuckle_joint": 0.0215,
        "gripper_right_inner_knuckle_joint": -0.0216,
        "gripper_left_finger_tip_joint": 0.0,
        "gripper_right_finger_tip_joint": 0.0530,
    }
    velocities = {name: 0.0 for name in positions}

    assert functional_opening_complete(
        positions,
        velocities,
        open_active_rad=0.0401,
        max_common_lead_rad=0.06,
        terminal_tolerance_rad=0.02,
        terminal_velocity_rad_s=0.08,
    )


@pytest.mark.parametrize("failure", ["not_open", "asymmetric", "moving"])
def test_functional_full_open_rejects_unproven_common_states(failure: str) -> None:
    positions = dict(six_joint_positions(0.08))
    velocities = {name: 0.0 for name in positions}
    if failure == "not_open":
        positions.update(six_joint_positions(0.20))
    elif failure == "asymmetric":
        positions["gripper_left_finger_joint"] = 0.12
        positions["gripper_right_finger_joint"] = -0.04
    else:
        velocities["gripper_left_finger_joint"] = 0.081

    assert not functional_opening_complete(
        positions,
        velocities,
        open_active_rad=0.0401,
        max_common_lead_rad=0.06,
        terminal_tolerance_rad=0.02,
        terminal_velocity_rad_s=0.08,
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
    assert (
        positions["gripper_right_inner_knuckle_joint"]
        == -positions["gripper_left_inner_knuckle_joint"]
    )
    assert (
        positions["gripper_right_finger_tip_joint"] == -positions["gripper_left_finger_tip_joint"]
    )
    # The vendor constant multipliers are already within 2 mrad of the exact
    # solution at this point; the closed form keeps that behaviour.
    assert positions["gripper_left_inner_knuckle_joint"] == pytest.approx(0.4, abs=0.05)
    assert positions["gripper_left_finger_tip_joint"] == pytest.approx(-0.4, abs=0.05)


def test_open_controller_endpoint_is_exact_limit_safe_four_bar_pose() -> None:
    endpoint = minimum_feasible_active_position(
        boundary_inset_rad=DEFAULT_CONTROLLER_BOUNDARY_INSET_RAD
    )
    solved = solve_four_bar(endpoint)

    assert solved["inner_knuckle_rad"] == pytest.approx(
        DEFAULT_CONTROLLER_BOUNDARY_INSET_RAD, abs=1e-12
    )
    assert endpoint > DEFAULT_CONTROLLER_BOUNDARY_INSET_RAD
    assert -solved["tip_rad"] > DEFAULT_CONTROLLER_BOUNDARY_INSET_RAD
    # The controller-safe pose retains at least 95% of the theoretical CAD
    # aperture while avoiding every independently modelled hard stop.
    assert aperture_from_angle_closed_form(endpoint) >= 0.95 * aperture_from_angle_closed_form(0.0)

    with pytest.raises(ValueError, match="positive"):
        minimum_feasible_active_position(boundary_inset_rad=0.0)


def test_out_of_range_active_angles_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        solve_four_bar(-0.1)
    with pytest.raises(ValueError, match="outside"):
        solve_four_bar(0.9)


def test_linkage_terminal_metrics_require_all_six_finite_joint_samples() -> None:
    targets = six_joint_positions(0.0)
    positions = dict(targets)
    velocities = {name: 0.0 for name in targets}
    positions["gripper_left_finger_tip_joint"] = -0.012
    velocities["gripper_right_inner_knuckle_joint"] = -0.07

    position_error, maximum_speed = linkage_terminal_metrics(targets, positions, velocities)

    assert position_error == pytest.approx(0.012)
    assert maximum_speed == pytest.approx(0.07)
    with pytest.raises(ValueError, match="complete"):
        linkage_terminal_metrics(targets, positions, {})
    velocities["gripper_left_finger_joint"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        linkage_terminal_metrics(targets, positions, velocities)


def test_common_driver_uses_directional_slowest_mirrored_side() -> None:
    positions = {
        "gripper_left_finger_joint": 0.42,
        "gripper_right_finger_joint": -0.38,
    }

    assert common_driver_position(positions, closing=True) == pytest.approx(0.38)
    assert common_driver_position(positions, closing=False) == pytest.approx(0.42)

    with pytest.raises(ValueError, match="both mirrored"):
        common_driver_position(
            {"gripper_left_finger_joint": 0.4},
            closing=True,
        )
    positions["gripper_right_finger_joint"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        common_driver_position(positions, closing=True)


def test_bilateral_contact_hold_uses_measured_common_driver_and_bounded_preload() -> None:
    assert bounded_contact_hold_position(
        measured_common_active_rad=0.38,
        requested_active_rad=0.70,
        preload_rad=0.01,
    ) == pytest.approx(0.39)
    assert bounded_contact_hold_position(
        measured_common_active_rad=0.695,
        requested_active_rad=0.70,
        preload_rad=0.01,
    ) == pytest.approx(0.70)

    with pytest.raises(ValueError, match="finite"):
        bounded_contact_hold_position(
            measured_common_active_rad=math.nan,
            requested_active_rad=0.70,
            preload_rad=0.01,
        )
    with pytest.raises(ValueError, match="positive and bounded"):
        bounded_contact_hold_position(
            measured_common_active_rad=0.38,
            requested_active_rad=0.70,
            preload_rad=0.0,
        )


def test_controller_safe_targets_inset_each_hard_stop_inside_action_tolerance() -> None:
    requested = dict(six_joint_positions(0.0))

    safe = controller_safe_targets(requested, boundary_inset_rad=0.01)

    assert all(abs(value) == pytest.approx(0.01) for value in safe.values())
    assert all(abs(safe[name] - requested[name]) == pytest.approx(0.01) for name in safe)
    midstroke = dict(six_joint_positions(0.4))
    assert controller_safe_targets(midstroke, boundary_inset_rad=0.01) == midstroke
    with pytest.raises(ValueError, match="positive"):
        controller_safe_targets(requested, boundary_inset_rad=0.0)
