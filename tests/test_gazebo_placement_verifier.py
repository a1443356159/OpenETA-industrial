from __future__ import annotations

from dataclasses import replace

import pytest

from extensions.gazebo.native_grasp import (
    NativePickPlaceConfig,
    PlacementPoseSample,
    PlacementReasonCode,
    Verdict,
    verify_stable_placement,
)


def _sample(stamp: float, xyz=(0.48, -0.10, 0.43), quat=(0.0, 0.0, 0.0, 1.0)):
    return PlacementPoseSample(stamp, xyz, quat)


def test_stable_placement_requires_duration_drift_height_and_conservative_footprint() -> None:
    samples = [_sample(stamp) for stamp in (10.0, 10.1, 10.2, 10.3, 10.4, 10.5)]

    result = verify_stable_placement(samples)

    assert result.verdict is Verdict.PASS
    assert result.reason_code is PlacementReasonCode.PLACED
    assert result.evidence["stable_duration_s"] == 0.5
    assert result.evidence["terminal_drift_m"] == 0.0
    assert result.evidence["conservative_footprint_radius_m"] > 0.028
    assert result.evidence["final_pose"]["quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]


def test_placement_rejects_terminal_drift_and_wrong_height() -> None:
    moving = [
        _sample(10.0),
        _sample(10.3),
        _sample(10.4, (0.48, -0.10, 0.436)),
        _sample(10.5, (0.48, -0.10, 0.43)),
    ]
    assert verify_stable_placement(moving).reason_code is PlacementReasonCode.TERMINAL_DRIFT

    wrong_height = [_sample(stamp, (0.48, -0.10, 0.441)) for stamp in (10.0, 10.3, 10.4, 10.5)]
    assert (
        verify_stable_placement(wrong_height).reason_code
        is PlacementReasonCode.HEIGHT_OUT_OF_RANGE
    )


def test_placement_height_closed_boundary_tolerates_only_float_noise() -> None:
    boundary_noise = [
        _sample(stamp, (0.48, -0.10, 0.4199994))
        for stamp in (10.0, 10.3, 10.4, 10.5)
    ]
    outside_boundary = [
        _sample(stamp, (0.48, -0.10, 0.419998))
        for stamp in (10.0, 10.3, 10.4, 10.5)
    ]

    assert verify_stable_placement(boundary_noise).reason_code is PlacementReasonCode.PLACED
    assert (
        verify_stable_placement(outside_boundary).reason_code
        is PlacementReasonCode.HEIGHT_OUT_OF_RANGE
    )


def test_placement_uses_stable_terminal_window_after_initial_settling() -> None:
    samples = [
        _sample(10.0, (0.46, -0.10, 0.44)),
        _sample(10.2, (0.47, -0.10, 0.435)),
        _sample(10.5),
        _sample(10.7),
        _sample(10.9),
        _sample(11.0),
    ]

    result = verify_stable_placement(samples)

    assert result.verdict is Verdict.PASS
    assert result.evidence["stable_duration_s"] == 0.5
    assert result.evidence["terminal_drift_m"] == 0.0


def test_placement_terminal_window_includes_prior_discrete_sample() -> None:
    samples = [
        _sample(10.0, (0.46, -0.10, 0.44)),
        _sample(10.2, (0.47, -0.10, 0.435)),
        _sample(10.4),
        _sample(10.6),
        _sample(10.8),
        _sample(11.0),
    ]

    result = verify_stable_placement(samples)

    assert result.verdict is Verdict.PASS
    assert result.evidence["stable_duration_s"] == pytest.approx(0.6)
    assert result.evidence["terminal_drift_m"] == 0.0


def test_placement_uses_circumscribed_radius_not_center_only_region_check() -> None:
    samples = [_sample(stamp, (0.515, -0.10, 0.43)) for stamp in (10.0, 10.3, 10.4, 10.5)]

    result = verify_stable_placement(samples)

    assert result.reason_code is PlacementReasonCode.FOOTPRINT_OUTSIDE_DESTINATION
    assert result.evidence["footprint_margin_xy_m"][0] < 0.0


def test_placement_fails_closed_without_full_stability_evidence() -> None:
    too_short = [_sample(10.0), _sample(10.49)]
    sparse_terminal = [_sample(10.0), _sample(10.51)]

    assert (
        verify_stable_placement(too_short).reason_code
        is PlacementReasonCode.OBSERVATION_TOO_SHORT
    )
    assert (
        verify_stable_placement(sparse_terminal).reason_code
        is PlacementReasonCode.POSE_UNAVAILABLE
    )

    relaxed = replace(NativePickPlaceConfig(), placement_terminal_window_s=0.6)
    assert verify_stable_placement(sparse_terminal, relaxed).verdict is Verdict.PASS
