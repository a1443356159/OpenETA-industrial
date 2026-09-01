from __future__ import annotations

import math
from dataclasses import replace

import pytest

from agent.runtime.collision_geometry import (
    collision_geometry_volume_centroid,
    compound_axis_aligned_bounds,
    orientation_invariant_radius_m,
    project_collision_geometry,
    support_face_alignment_cosine,
)
from extensions.gazebo.native_grasp import (
    NativePickPlaceConfig,
    PLACEMENT_ACCEPTANCE_COMPLETE_FOOTPRINT,
    PLACEMENT_ACCEPTANCE_STABLE_GEOMETRY_CENTROID,
    PlacementPoseSample,
    PlacementReasonCode,
    Verdict,
    verify_stable_placement,
)


def _sample(stamp: float, xyz=None, quat=(0.0, 0.0, 0.0, 1.0)):
    if xyz is None:
        center_x, center_y = NativePickPlaceConfig().destination_center_xy
        xyz = (center_x, center_y - 0.035, 0.029)
    return PlacementPoseSample(stamp, xyz, quat)


def _supported_link_height(
    config: NativePickPlaceConfig,
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> float:
    x, y, z, w = quat
    rotation = (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )
    geometry = project_collision_geometry(
        object_xyz=(0.0, 0.0, 0.0),
        object_rotation=rotation,
        primitives=config.target_collision_primitives,
        fallback_size_xyz=config.target_size_m,
    )
    minimum_z = compound_axis_aligned_bounds(geometry).minimum_xyz[2]
    return config.destination_support_z_m - minimum_z


def test_orientation_invariant_box_radius_uses_exact_farthest_corner() -> None:
    geometry = project_collision_geometry(
        object_xyz=(0.0, 0.0, 0.0),
        object_rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        primitives=(
            {
                "shape": "box",
                "size_xyz": [2.0, 2.0, 2.0],
                "pose_xyz": [1.0, 0.0, 0.0],
            },
        ),
    )

    assert orientation_invariant_radius_m(geometry, object_xyz=(0.0, 0.0, 0.0)) == (
        pytest.approx(math.sqrt(6.0))
    )


def test_orientation_invariant_cylinder_radius_resolves_axial_and_radial_offsets() -> None:
    geometry = project_collision_geometry(
        object_xyz=(0.0, 0.0, 0.0),
        object_rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        primitives=(
            {
                "shape": "cylinder",
                "radius": 2.0,
                "length": 6.0,
                "pose_xyz": [3.0, 4.0, 0.0],
            },
        ),
    )

    assert orientation_invariant_radius_m(geometry, object_xyz=(0.0, 0.0, 0.0)) == (
        pytest.approx(math.sqrt(58.0))
    )


def test_compound_volume_centroid_uses_primitive_volume_not_outer_box_origin() -> None:
    geometry = project_collision_geometry(
        object_xyz=(0.0, 0.0, 0.0),
        object_rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        primitives=(
            {"shape": "box", "size_xyz": [2.0, 1.0, 1.0], "pose_xyz": [-1.0, 0.0, 0.0]},
            {"shape": "box", "size_xyz": [1.0, 1.0, 1.0], "pose_xyz": [2.0, 0.0, 0.0]},
        ),
    )

    assert collision_geometry_volume_centroid(geometry) == pytest.approx((0.0, 0.0, 0.0))


def test_support_face_alignment_is_exact_and_ordering_only() -> None:
    angle = math.radians(12.0)
    rotation = (
        (math.cos(angle), 0.0, math.sin(angle)),
        (0.0, 1.0, 0.0),
        (-math.sin(angle), 0.0, math.cos(angle)),
    )
    geometry = project_collision_geometry(
        object_xyz=(0.0, 0.0, 0.0),
        object_rotation=rotation,
        primitives=({"shape": "box", "size_xyz": [0.2, 0.06, 0.03], "pose_xyz": [0, 0, 0]},),
    )

    assert support_face_alignment_cosine(geometry) == pytest.approx(math.cos(angle))


def test_stable_placement_requires_duration_drift_height_and_oriented_footprint() -> None:
    samples = [_sample(stamp) for stamp in (10.0, 10.1, 10.2, 10.3, 10.4, 10.5)]

    result = verify_stable_placement(samples)

    assert result.verdict is Verdict.PASS
    assert result.reason_code is PlacementReasonCode.PLACED
    assert result.evidence["stable_duration_s"] == 0.5
    assert result.evidence["terminal_drift_m"] == 0.0
    assert result.evidence["conservative_footprint_radius_m"] > 0.11
    assert result.evidence["projected_footprint_half_extent_xy_m"] == pytest.approx(
        [0.1075, 0.031]
    )
    assert result.evidence["geometry_source"] == "compound_collision_primitives"
    assert result.evidence["final_pose"]["quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]


def test_placement_rejects_terminal_drift_and_wrong_height() -> None:
    config = NativePickPlaceConfig()
    center_x, center_y = config.destination_center_xy
    moving = [
        _sample(10.0),
        _sample(10.3),
        _sample(10.4, (center_x, center_y - 0.035, 0.035)),
        _sample(10.5, (center_x, center_y - 0.035, 0.029)),
    ]
    assert verify_stable_placement(moving).reason_code is PlacementReasonCode.TERMINAL_DRIFT

    expected = _supported_link_height(config)
    wrong_height = [
        _sample(
            stamp,
            (
                center_x,
                center_y - 0.035,
                expected + config.placement_support_height_tolerance_m + 0.001,
            ),
        )
        for stamp in (10.0, 10.3, 10.4, 10.5)
    ]
    assert (
        verify_stable_placement(wrong_height).reason_code
        is PlacementReasonCode.HEIGHT_OUT_OF_RANGE
    )


def test_placement_height_follows_oriented_compound_support_geometry() -> None:
    config = NativePickPlaceConfig()
    half_turn = math.sqrt(0.5)
    vertical_quat = (0.0, half_turn, 0.0, half_turn)
    vertical_center_z = _supported_link_height(config, vertical_quat)
    samples = [
        _sample(
            stamp,
            (config.destination_center_xy[0], config.destination_center_xy[1], vertical_center_z),
            vertical_quat,
        )
        for stamp in (10.0, 10.3, 10.4, 10.5)
    ]

    result = verify_stable_placement(samples, config)

    assert result.verdict is Verdict.PASS
    assert result.reason_code is PlacementReasonCode.PLACED
    assert result.evidence["height_rule"] == (
        "compound_collision_geometry_contacts_destination_plane"
    )
    assert result.evidence["projected_collision_minimum_z_m"] == pytest.approx(
        config.destination_support_z_m
    )
    assert result.evidence["support_height_error_m"] == pytest.approx(0.0)


def test_placement_height_closed_boundary_tolerates_only_float_noise() -> None:
    config = NativePickPlaceConfig()
    center_x, center_y = config.destination_center_xy
    expected = _supported_link_height(config)
    boundary_noise = [
        _sample(
            stamp,
            (
                center_x,
                center_y - 0.035,
                expected - config.placement_support_height_tolerance_m - 0.0000006,
            ),
        )
        for stamp in (10.0, 10.3, 10.4, 10.5)
    ]
    outside_boundary = [
        _sample(
            stamp,
            (
                center_x,
                center_y - 0.035,
                expected - config.placement_support_height_tolerance_m - 0.000002,
            ),
        )
        for stamp in (10.0, 10.3, 10.4, 10.5)
    ]

    assert verify_stable_placement(boundary_noise).reason_code is PlacementReasonCode.PLACED
    assert (
        verify_stable_placement(outside_boundary).reason_code
        is PlacementReasonCode.HEIGHT_OUT_OF_RANGE
    )


def test_placement_uses_stable_terminal_window_after_initial_settling() -> None:
    center_x, center_y = NativePickPlaceConfig().destination_center_xy
    samples = [
        _sample(10.0, (center_x - 0.02, center_y - 0.035, 0.039)),
        _sample(10.2, (center_x - 0.01, center_y - 0.035, 0.034)),
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
    center_x, center_y = NativePickPlaceConfig().destination_center_xy
    samples = [
        _sample(10.0, (center_x - 0.02, center_y - 0.035, 0.039)),
        _sample(10.2, (center_x - 0.01, center_y - 0.035, 0.034)),
        _sample(10.4),
        _sample(10.6),
        _sample(10.8),
        _sample(11.0),
    ]

    result = verify_stable_placement(samples)

    assert result.verdict is Verdict.PASS
    assert result.evidence["stable_duration_s"] == pytest.approx(0.6)
    assert result.evidence["terminal_drift_m"] == 0.0


def test_placement_uses_compound_footprint_not_center_only_region_check() -> None:
    base = NativePickPlaceConfig()
    center_x, center_y = base.destination_center_xy
    samples = [
        _sample(stamp, (center_x + 0.036, center_y - 0.035, 0.029))
        for stamp in (10.0, 10.3, 10.4, 10.5)
    ]
    config = replace(
        base,
        placement_acceptance_semantics=PLACEMENT_ACCEPTANCE_COMPLETE_FOOTPRINT,
    )

    result = verify_stable_placement(samples, config)

    assert result.reason_code is PlacementReasonCode.FOOTPRINT_OUTSIDE_DESTINATION
    assert result.evidence["footprint_margin_xy_m"][0] < 0.0


def test_physical_bin_accepts_stable_body_centroid_with_edge_overhang() -> None:
    config = NativePickPlaceConfig()
    center_x, center_y = config.destination_center_xy
    samples = [
        _sample(stamp, (center_x + 0.036, center_y - 0.035, 0.029))
        for stamp in (10.0, 10.3, 10.4, 10.5)
    ]

    result = verify_stable_placement(samples, config)

    assert config.placement_acceptance_semantics == (
        PLACEMENT_ACCEPTANCE_STABLE_GEOMETRY_CENTROID
    )
    assert result.verdict is Verdict.PASS
    assert result.evidence["complete_footprint_inside"] is False
    assert min(result.evidence["centroid_margin_xy_m"]) > 0.0
    assert result.evidence["placement_acceptance_semantics"] == (
        PLACEMENT_ACCEPTANCE_STABLE_GEOMETRY_CENTROID
    )
    assert result.evidence["placement_acceptance_authority"] == (
        "visual_primary_geometry_obvious_failure_guard"
    )
    assert result.evidence["complete_footprint_is_quality_only"] is True


def test_physical_bin_rejects_stable_body_centroid_outside_region() -> None:
    center_x, center_y = NativePickPlaceConfig().destination_center_xy
    samples = [
        _sample(stamp, (center_x + 0.28, center_y - 0.035, 0.029))
        for stamp in (10.0, 10.3, 10.4, 10.5)
    ]

    result = verify_stable_placement(samples)

    assert result.reason_code is PlacementReasonCode.CENTROID_OUTSIDE_DESTINATION
    assert result.evidence["centroid_margin_xy_m"][0] < 0.0


def test_offset_compound_body_uses_physical_bottom_after_tipping() -> None:
    # This is a generic compound-geometry regression: the link origin is not
    # the collision-body centre, so a centred outer box would falsely place
    # the bottom almost 19 mm below the physical bin floor.
    quat = (
        0.6936483971072943,
        -0.13730168767350612,
        -0.056720734639233744,
        0.7048282812249195,
    )
    center_x, center_y = NativePickPlaceConfig().destination_center_xy
    xyz = (center_x + 0.02, center_y, 0.044764681111384076)
    samples = [_sample(stamp, xyz, quat) for stamp in (10.0, 10.3, 10.4, 10.5)]

    result = verify_stable_placement(samples)

    assert result.verdict is Verdict.PASS
    assert result.evidence["projected_collision_minimum_z_m"] == pytest.approx(
        0.02, abs=1e-6
    )
    assert result.evidence["support_height_error_m"] < 1e-6


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
