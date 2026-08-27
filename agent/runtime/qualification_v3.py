"""Deterministic scheduling and Beam-2 helpers for the fast v3 funnel."""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from adapter.protocol import JsonDict
from agent.runtime.capability_map import (
    CapabilityScore,
    SparseCapabilityMap,
    quaternion_angle_rad,
    target_pose,
)


FAST_QUALIFICATION_SCHEMA = "openeta.moveit_candidate_funnel.v3"
FAST_ARTIFACT_SCHEMA = "openeta.moveit_candidate_qualification.v3"
POSITION_CLUSTER_M = 0.01
ORIENTATION_CLUSTER_DEG = 10.0
JOINT_SOLUTION_DEDUP_DISTANCE = 0.05
PARALLEL_GRIPPER_CENTERING_RISK_RATIO = 0.10
PARALLEL_GRIPPER_CENTERING_MIN_SPAN_M = 0.02


def grasp_symmetry_family_id(candidate: Mapping[str, Any]) -> str:
    """Return the physical branch identity after parallel-jaw roll symmetry.

    A 180-degree roll about the approach axis remains a separate candidate and
    keeps separate evidence, but must not consume the second grasp-diversity
    slot when a genuinely different grasp branch is available.
    """

    explicit = candidate.get("source_grasp_equivalence_id")
    if explicit:
        return str(explicit)
    parent = candidate.get("symmetry_parent_id")
    if parent:
        return str(parent)
    return str(candidate.get("id") or candidate.get("source_grasp_id") or "")


def generator_score(candidate: Mapping[str, Any]) -> float:
    for name in ("score", "confidence", "quality", "generator_score"):
        value = candidate.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = float(value)
            if math.isfinite(parsed):
                return parsed
    return 0.0


def parallel_gripper_centering_evidence(
    value: Mapping[str, Any],
) -> JsonDict | None:
    """Return copied provider evidence from either raw or compiled candidates."""

    sources: list[Mapping[str, Any]] = [value]
    camera_pose = value.get("camera_pose")
    if isinstance(camera_pose, Mapping):
        sources.append(camera_pose)
    compile_parameters = value.get("compile_parameters")
    if isinstance(compile_parameters, Mapping):
        compiled_camera_pose = compile_parameters.get("camera_pose")
        if isinstance(compiled_camera_pose, Mapping):
            sources.append(compiled_camera_pose)
    # Exact PlanningScene collision geometry is a stronger physical signal
    # than the provider's visible-depth quantiles.  Both remain immutable
    # evidence, but prefer the scene-derived measurement when available.
    alignment = next(
        (
            source.get("scene_target_closing_alignment")
            for source in sources
            if isinstance(source.get("scene_target_closing_alignment"), Mapping)
            and source["scene_target_closing_alignment"].get("evaluated") is True
        ),
        None,
    )
    if alignment is None:
        alignment = next(
            (
                source.get("target_closing_alignment")
                for source in sources
                if isinstance(source.get("target_closing_alignment"), Mapping)
            ),
            None,
        )
    return dict(alignment) if isinstance(alignment, Mapping) else None


def parallel_gripper_centering_quality(
    value: Mapping[str, Any],
) -> tuple[int, float]:
    """Return an ordering-only risk tier for an unchanged model pose.

    GraspGenX candidates may carry an RGB-D measurement of how far the model's
    closing midplane is from the selected object's midplane.  The host must not
    apply that correction to the model terminal pose, but it can prefer a pose
    whose jaws already straddle the object.  Missing evidence is neutral so
    other providers retain their existing ordering, and every candidate stays
    in the eventual exhaustive waves.
    """

    alignment = parallel_gripper_centering_evidence(value)
    if not isinstance(alignment, Mapping):
        return (0, PARALLEL_GRIPPER_CENTERING_RISK_RATIO)
    if (
        alignment.get("aperture_feasible") is False
        or alignment.get("section_intersects_target") is False
    ):
        # Ordering only: exhaustive waves still retain this model pose.
        ratio = alignment.get("centering_ratio")
        return (
            2,
            float(ratio)
            if isinstance(ratio, (int, float))
            and not isinstance(ratio, bool)
            and math.isfinite(float(ratio))
            else math.inf,
        )
    correction = alignment.get("correction_m")
    span = alignment.get("target_span_m")
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        for item in (correction, span)
    ):
        return (0, PARALLEL_GRIPPER_CENTERING_RISK_RATIO)
    denominator = max(abs(float(span)), PARALLEL_GRIPPER_CENTERING_MIN_SPAN_M)
    ratio = abs(float(correction)) / denominator
    return (
        1 if ratio > PARALLEL_GRIPPER_CENTERING_RISK_RATIO else 0,
        ratio,
    )


def parallel_gripper_target_span_quality(value: Mapping[str, Any]) -> float:
    """Prefer a wider observed pinch section without rejecting narrow ones."""

    alignment = parallel_gripper_centering_evidence(value)
    span = alignment.get("target_span_m") if isinstance(alignment, Mapping) else None
    if (
        not isinstance(span, (int, float))
        or isinstance(span, bool)
        or not math.isfinite(float(span))
        or float(span) <= 0.0
    ):
        return 0.0
    return -float(span)


def placement_clearance_quality(value: Mapping[str, Any]) -> tuple[int, float]:
    """Prefer post-settle footprint clearance without pruning any goal."""

    margin = value.get("placement_robust_clearance_m")
    if margin is None:
        legality = value.get("goal_legality")
        checks = legality.get("checks") if isinstance(legality, Mapping) else None
        region = checks.get("placement_region") if isinstance(checks, Mapping) else None
        if isinstance(region, Mapping):
            margin = region.get(
                "conservative_minimum_margin_m",
                region.get("minimum_margin_m"),
            )
    if (
        not isinstance(margin, (int, float))
        or isinstance(margin, bool)
        or not math.isfinite(float(margin))
    ):
        return (1, 0.0)
    return (0, -float(margin))


def placement_clearance_risk_tier(value: Mapping[str, Any]) -> int:
    """Classify geometric settling clearance before comparing its magnitude.

    A strictly positive projected margin is qualitatively different from a
    negative circumscribed-body margin.  Once two goals are on the same side
    of that exact geometric boundary, a millimetre of additional empty space
    must not outweigh support stability or robot joint quality.  Missing
    evidence remains schedulable at the lowest-confidence tier.
    """

    status, negative_margin = placement_clearance_quality(value)
    if status != 0:
        return 2
    margin = -negative_margin
    return 0 if margin >= 0.0 else 1


def placement_settling_sweep_clearance_quality(
    value: Mapping[str, Any],
) -> tuple[int, float]:
    """Prefer clearance that survives the pose's predicted support settling."""

    margin = value.get("placement_settling_sweep_clearance_m")
    if margin is None:
        legality = value.get("goal_legality")
        checks = legality.get("checks") if isinstance(legality, Mapping) else None
        support = checks.get("support") if isinstance(checks, Mapping) else None
        region = checks.get("placement_region") if isinstance(checks, Mapping) else None
        if isinstance(support, Mapping):
            margin = support.get("settling_sweep_clearance_m")
        if margin is None and isinstance(region, Mapping):
            margin = region.get("settling_sweep_clearance_m")
    if (
        not isinstance(margin, (int, float))
        or isinstance(margin, bool)
        or not math.isfinite(float(margin))
    ):
        return (1, 0.0)
    return (0, -float(margin))


def placement_settling_quality(value: Mapping[str, Any]) -> tuple[Any, ...]:
    """Order release poses by the kind of settling evidence available.

    The strongest tier has non-negative clearance even after subtracting the
    exact radius-chord bound for settling onto the nearest support face. A
    nominally in-zone goal whose predicted settling sweep crosses the region
    boundary remains schedulable in the next tier. Negative nominal clearance
    and missing evidence remain later ordering tiers; neither is pruned.
    """

    nominal_tier = placement_clearance_risk_tier(value)
    clearance = placement_clearance_quality(value)
    sweep_clearance = placement_settling_sweep_clearance_quality(value)
    stability = placement_support_stability_quality(value)
    if sweep_clearance[0] == 0:
        sweep_margin = -sweep_clearance[1]
        if sweep_margin >= 0.0:
            return (0, *stability, *sweep_clearance, *clearance)
        if nominal_tier == 0:
            return (1, *sweep_clearance, *stability, *clearance)
        return (2, *clearance, *sweep_clearance, *stability)
    if nominal_tier == 0:
        return (1, *stability, *clearance, *sweep_clearance)
    if nominal_tier == 1:
        return (2, *clearance, *stability, *sweep_clearance)
    return (3, *stability, *clearance, *sweep_clearance)


def placement_support_stability_quality(value: Mapping[str, Any]) -> tuple[Any, ...]:
    """Prefer low-energy, face-aligned support poses without pruning goals.

    When exact compound collision geometry is available, its volume centroid
    is the deterministic uniform-density centre-of-mass proxy. Support-face
    alignment is compared first because an unaligned model target must move
    under gravity; energy is then bucketed at the scene's own support-height
    resolution. The oriented vertical extent remains the compatibility
    fallback for older v3 artifacts. Every signal here is ordering-only.
    """

    legality = value.get("goal_legality")
    checks = legality.get("checks") if isinstance(legality, Mapping) else None
    support = checks.get("support") if isinstance(checks, Mapping) else None

    support_energy = value.get("placement_support_energy_m")
    if support_energy is None and isinstance(support, Mapping):
        support_energy = support.get("geometry_volume_centroid_height_m")
    energy_resolution = value.get("placement_support_energy_resolution_m")
    if energy_resolution is None and isinstance(support, Mapping):
        energy_resolution = support.get("support_energy_resolution_m")
    alignment_error = value.get("placement_support_face_alignment_error_rad")
    if alignment_error is None and isinstance(support, Mapping):
        alignment_error = support.get("support_face_alignment_error_rad")
    if alignment_error is None:
        alignment_cosine = value.get("placement_support_face_alignment_cosine")
        if alignment_cosine is None and isinstance(support, Mapping):
            alignment_cosine = support.get("support_face_alignment_cosine")
        if (
            isinstance(alignment_cosine, (int, float))
            and not isinstance(alignment_cosine, bool)
            and math.isfinite(float(alignment_cosine))
        ):
            alignment_error = math.acos(min(1.0, max(0.0, float(alignment_cosine))))
    if (
        isinstance(support_energy, (int, float))
        and not isinstance(support_energy, bool)
        and math.isfinite(float(support_energy))
        and float(support_energy) >= 0.0
    ):
        energy = float(support_energy)
        resolution = (
            float(energy_resolution)
            if isinstance(energy_resolution, (int, float))
            and not isinstance(energy_resolution, bool)
            and math.isfinite(float(energy_resolution))
            and float(energy_resolution) > 0.0
            else 0.0
        )
        energy_bucket = math.floor(energy / resolution + 1e-12) if resolution else energy
        alignment_status = 0
        alignment = 0.0
        if (
            not isinstance(alignment_error, (int, float))
            or isinstance(alignment_error, bool)
            or not math.isfinite(float(alignment_error))
            or float(alignment_error) < 0.0
        ):
            alignment_status = 1
        else:
            alignment = float(alignment_error)
        return (0, alignment_status, alignment, energy_bucket, energy)

    vertical_extent = value.get("placement_vertical_extent_m")
    if vertical_extent is None:
        object_bbox = checks.get("object_bbox") if isinstance(checks, Mapping) else None
        if isinstance(object_bbox, Mapping):
            minimum_z = object_bbox.get("minimum_z_m")
            maximum_z = object_bbox.get("maximum_z_m")
            if all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in (minimum_z, maximum_z)
            ):
                vertical_extent = float(maximum_z) - float(minimum_z)
    if (
        not isinstance(vertical_extent, (int, float))
        or isinstance(vertical_extent, bool)
        or not math.isfinite(float(vertical_extent))
        or float(vertical_extent) < 0.0
    ):
        return (2, 0.0, 1, 0.0, 0.0)
    extent = float(vertical_extent)
    return (1, extent, 1, 0.0, extent)


def frozen_frontier_parent_priority(value: Mapping[str, Any]) -> int:
    """Prefer a model-native sibling of the physically failed parent mode."""

    sources: list[Mapping[str, Any]] = [value]
    camera_pose = value.get("camera_pose")
    if isinstance(camera_pose, Mapping):
        sources.append(camera_pose)
    compile_parameters = value.get("compile_parameters")
    if isinstance(compile_parameters, Mapping):
        compiled_camera_pose = compile_parameters.get("camera_pose")
        if isinstance(compiled_camera_pose, Mapping):
            sources.append(compiled_camera_pose)
    return (
        0 if any(source.get("frozen_frontier_parent_priority") is True for source in sources) else 1
    )


def parallel_gripper_centering_variant_priority(
    value: Mapping[str, Any],
) -> int:
    """Prefer an unchanged model sibling that repairs a risky source mode.

    This signal never changes or removes a pose.  It only moves a bounded
    provider reserve, already proven to share the failed/high-score parent's
    approach neighborhood, into the early deep waves where IK and state
    validity can decide it cheaply.
    """

    sources: list[Mapping[str, Any]] = [value]
    camera_pose = value.get("camera_pose")
    if isinstance(camera_pose, Mapping):
        sources.append(camera_pose)
    compile_parameters = value.get("compile_parameters")
    if isinstance(compile_parameters, Mapping):
        compiled_camera_pose = compile_parameters.get("camera_pose")
        if isinstance(compiled_camera_pose, Mapping):
            sources.append(compiled_camera_pose)
    for source in sources:
        alignment = source.get("target_closing_alignment")
        if (
            isinstance(alignment, Mapping)
            and alignment.get("variant_role") == "same_approach_centering_reserve"
        ):
            return 0
    return 1


def final_target(descriptor: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = descriptor.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    stages = candidate.get("qualification_stages")
    if isinstance(stages, list) and stages and isinstance(stages[-1], Mapping):
        return stages[-1]
    return {}


def _pose_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[float, float] | None:
    left_pose, right_pose = target_pose(left), target_pose(right)
    if left_pose is None or right_pose is None:
        return None
    left_xyz, left_quat = left_pose
    right_xyz, right_quat = right_pose
    translation = math.sqrt(sum((a - b) ** 2 for a, b in zip(left_xyz, right_xyz)))
    return translation, quaternion_angle_rad(left_quat, right_quat)


def assign_se3_clusters(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    position_threshold_m: float = POSITION_CLUSTER_M,
    orientation_threshold_deg: float = ORIENTATION_CLUSTER_DEG,
) -> list[JsonDict]:
    """Greedily assign stable 10 mm / 10 degree clusters in input order."""

    representatives: list[Mapping[str, Any]] = []
    annotated: list[JsonDict] = []
    orientation_threshold = math.radians(orientation_threshold_deg)
    for fixed_index, raw in enumerate(descriptors):
        descriptor = dict(raw)
        target = final_target(descriptor)
        cluster_index: int | None = None
        for index, representative in enumerate(representatives):
            distance = _pose_distance(target, representative)
            if (
                distance is not None
                and distance[0] <= position_threshold_m
                and distance[1] <= orientation_threshold
            ):
                cluster_index = index
                break
        if cluster_index is None:
            cluster_index = len(representatives)
            representatives.append(target)
        descriptor["fixed_candidate_index"] = int(
            descriptor.get("fixed_candidate_index", fixed_index)
        )
        descriptor["se3_cluster_id"] = f"se3_{cluster_index:04d}"
        candidate = descriptor.get("candidate")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        descriptor["grasp_symmetry_family_id"] = grasp_symmetry_family_id(candidate)
        annotated.append(descriptor)
    return annotated


def _capability_score(
    descriptor: Mapping[str, Any], capability_map: SparseCapabilityMap | None
) -> CapabilityScore:
    if capability_map is None:
        return CapabilityScore(0.0, 0.0, 0.0, 0.0, 0, 0)
    candidate = descriptor.get("candidate")
    stages = candidate.get("qualification_stages") if isinstance(candidate, Mapping) else None
    return capability_map.score_chain(
        [stage for stage in stages or [] if isinstance(stage, Mapping)]
    )


def _descriptor_priority(descriptor: Mapping[str, Any]) -> tuple[Any, ...]:
    score = descriptor.get("capability_score")
    score = score if isinstance(score, Mapping) else {}
    candidate = descriptor.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    centering_risk, centering_ratio = parallel_gripper_centering_quality(candidate)
    parent_priority = frozen_frontier_parent_priority(candidate)
    span_quality = parallel_gripper_target_span_quality(candidate)
    # The initial pool keeps the measured centering order that has the best
    # early IK recall.  After a physical failure, candidates explicitly linked
    # to that frozen parent are already within one approach neighbourhood; in
    # that bounded recovery set, a wider observed pinch section is the more
    # useful bilateral-contact discriminator.
    centering_order = (
        (span_quality, centering_ratio) if parent_priority == 0 else (centering_ratio, span_quality)
    )
    return (
        parent_priority,
        centering_risk,
        *centering_order,
        parallel_gripper_centering_variant_priority(candidate),
        *placement_settling_quality(descriptor),
        -float(score.get("confidence", 0.0)),
        -float(score.get("reachable_density", 0.0)),
        -float(score.get("joint_margin", 0.0)),
        -float(score.get("min_singular_value", 0.0)),
        -generator_score(candidate),
        int(descriptor.get("fixed_candidate_index", 0)),
    )


def _descriptor_diversity_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Stable SE(3) distance used only to order, never reject, candidates."""

    distance = _pose_distance(final_target(left), final_target(right))
    if distance is None:
        return 0.0
    left_candidate = left.get("candidate")
    left_candidate = left_candidate if isinstance(left_candidate, Mapping) else {}
    right_candidate = right.get("candidate")
    right_candidate = right_candidate if isinstance(right_candidate, Mapping) else {}

    def source(candidate: Mapping[str, Any]) -> str:
        return str(
            candidate.get("candidate_source")
            or candidate.get("source_model")
            or candidate.get("source")
            or candidate.get("source_branch")
            or candidate.get("backend")
            or ""
        )

    left_source, right_source = source(left_candidate), source(right_candidate)
    source_bonus = 1.0 if left_source and right_source and left_source != right_source else 0.0
    translation_m, rotation_rad = distance
    return translation_m * 10.0 + rotation_rad + source_bonus


def _quality_seeded_farthest_first(
    descriptors: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Start with the best head, then cover distinct pose modes early.

    Keep each remaining descriptor's minimum distance to the selected set and
    update it once per newly selected pose.  This is order-equivalent to
    rescanning every selected/remaining pair on every iteration, but reduces
    the scheduler from cubic repeated SE(3) work to quadratic work for the
    full grasp pool.
    """

    remaining = sorted((dict(item) for item in descriptors), key=_descriptor_priority)
    if not remaining:
        return []
    selected = [remaining.pop(0)]
    minimum_distances = [_descriptor_diversity_distance(item, selected[0]) for item in remaining]
    while remaining:
        # Alternate exploitation and exploration. A pure farthest-first walk
        # can postpone the second-best reachable mode until a later wave;
        # pure score ordering over-focuses near-duplicates. The 4-wide first
        # wave therefore contains two quality heads and two coverage heads.
        choose_farthest = len(selected) % 2 == 1
        index = (
            min(
                range(len(remaining)),
                key=lambda candidate_index: (
                    -minimum_distances[candidate_index],
                    _descriptor_priority(remaining[candidate_index]),
                    str(remaining[candidate_index].get("se3_cluster_id") or ""),
                ),
            )
            if choose_farthest
            else 0
        )
        newly_selected = remaining.pop(index)
        minimum_distances.pop(index)
        selected.append(newly_selected)
        for remaining_index, descriptor in enumerate(remaining):
            minimum_distances[remaining_index] = min(
                minimum_distances[remaining_index],
                _descriptor_diversity_distance(descriptor, newly_selected),
            )
    return selected


def _cluster_round_robin(
    descriptors: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    clusters: dict[str, deque[JsonDict]] = {}
    for descriptor in descriptors:
        cluster = str(descriptor.get("se3_cluster_id") or "")
        clusters.setdefault(cluster, deque()).append(dict(descriptor))
    for cluster in clusters:
        clusters[cluster] = deque(sorted(clusters[cluster], key=_descriptor_priority))
    cluster_heads = _quality_seeded_farthest_first([clusters[cluster][0] for cluster in clusters])
    cluster_order = [str(head.get("se3_cluster_id") or "") for head in cluster_heads]
    ordered: list[JsonDict] = []
    while any(clusters[cluster] for cluster in cluster_order):
        for cluster in cluster_order:
            if clusters[cluster]:
                ordered.append(clusters[cluster].popleft())
    return ordered


def _quality_ordered_cluster_round_robin(
    descriptors: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Round-robin placement clusters with the safest geometry first.

    Grasp search benefits from alternating quality and farthest-pose
    exploration. Placement goals have a different physical asymmetry: for the
    same rigid body, a low supported oriented box is a better one-shot release
    than an upright high-energy pose. Order cluster heads by that measured
    geometry, then round-robin without deleting any cluster or model goal.
    """

    clusters: dict[str, deque[JsonDict]] = {}
    for descriptor in descriptors:
        cluster = str(descriptor.get("se3_cluster_id") or "")
        clusters.setdefault(cluster, deque()).append(dict(descriptor))
    for cluster in clusters:
        clusters[cluster] = deque(sorted(clusters[cluster], key=_descriptor_priority))
    cluster_order = sorted(
        clusters,
        key=lambda cluster: (
            _descriptor_priority(clusters[cluster][0]),
            cluster,
        ),
    )
    ordered: list[JsonDict] = []
    while any(clusters[cluster] for cluster in cluster_order):
        for cluster in cluster_order:
            if clusters[cluster]:
                ordered.append(clusters[cluster].popleft())
    return ordered


@dataclass(frozen=True, slots=True)
class CandidateWave:
    wave_index: int
    cumulative_per_branch: int
    candidates: tuple[JsonDict, ...]
    recovery: bool = False
    frozen_pair_batch_index: int = 0


def schedule_candidate_waves(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    purpose: str,
    grasp_waves: Sequence[int] = (4, 8, 16, 32, 64),
    placement_waves: Sequence[int] = (4, 8, 16, 32, 96),
    observation_waves: Sequence[int] = (4, 8, 16, 24),
    capability_map: SparseCapabilityMap | None = None,
) -> list[CandidateWave]:
    """Build cumulative waves without deleting any structurally valid candidate."""

    annotated = assign_se3_clusters(descriptors)
    for descriptor in annotated:
        descriptor["capability_score"] = _capability_score(descriptor, capability_map).to_dict()
    if purpose in {"grasp", "observation"}:
        ordered = _cluster_round_robin(annotated)
        configured_waves = grasp_waves if purpose == "grasp" else observation_waves
        cumulative = sorted(
            set([value for value in configured_waves if value < len(ordered)] + [len(ordered)])
        )
        waves: list[CandidateWave] = []
        previous = 0
        for wave_index, limit in enumerate(cumulative):
            batch = tuple(ordered[previous:limit])
            if batch:
                waves.append(CandidateWave(wave_index, limit, batch))
            previous = limit
        return waves

    branches: dict[str, list[JsonDict]] = {}
    branch_order: list[str] = []
    branch_batch: dict[str, int] = {}
    for descriptor in annotated:
        candidate = descriptor.get("candidate")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        branch = str(candidate.get("source_grasp_id") or "__single_branch__")
        if branch not in branches:
            branches[branch] = []
            branch_order.append(branch)
            raw_batch = candidate.get("frozen_pair_batch_index", 0)
            branch_batch[branch] = (
                raw_batch
                if isinstance(raw_batch, int) and not isinstance(raw_batch, bool) and raw_batch >= 0
                else 0
            )
        branches[branch].append(descriptor)
    ordered_branches = {
        branch: _quality_ordered_cluster_round_robin(branches[branch]) for branch in branch_order
    }
    waves: list[CandidateWave] = []
    for batch_index in sorted(set(branch_batch.values())):
        current_branches = [
            branch for branch in branch_order if branch_batch[branch] == batch_index
        ]
        maximum = max(
            (len(ordered_branches[branch]) for branch in current_branches),
            default=0,
        )
        cumulative = sorted(
            set([value for value in placement_waves if value < maximum] + [maximum])
        )
        previous = 0
        for limit in cumulative:
            # Interleave grasp branches within one batch at every depth. A
            # reserve batch gets its own later waves, so it cannot consume IK
            # or L5 capacity before the primary frozen branches are exhausted.
            batch = tuple(
                ordered_branches[branch][index]
                for index in range(previous, limit)
                for branch in current_branches
                if index < len(ordered_branches[branch])
            )
            if batch:
                waves.append(
                    CandidateWave(
                        len(waves),
                        limit,
                        batch,
                        frozen_pair_batch_index=batch_index,
                    )
                )
            previous = limit
    return waves


def _joint_limits(
    reference: Mapping[str, Any], source: Mapping[str, Any]
) -> tuple[list[float], list[float]]:
    positions = [float(value) for value in reference.get("positions") or []]
    limits = source.get("joint_limits") or reference.get("joint_limits")
    lower, upper = [-math.pi] * len(positions), [math.pi] * len(positions)
    if isinstance(limits, Mapping):
        raw_lower, raw_upper = limits.get("lower"), limits.get("upper")
        if (
            isinstance(raw_lower, list)
            and isinstance(raw_upper, list)
            and len(raw_lower) == len(raw_upper) == len(positions)
        ):
            lower = [float(value) for value in raw_lower]
            upper = [float(value) for value in raw_upper]
    return lower, upper


def normalized_joint_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> float:
    left_positions = [float(value) for value in left.get("positions") or []]
    right_positions = [float(value) for value in right.get("positions") or []]
    if not left_positions or len(left_positions) != len(right_positions):
        return math.inf
    lower, upper = _joint_limits(left, source)
    distances = [
        abs(a - b) / max(1e-9, hi - lo)
        for a, b, lo, hi in zip(left_positions, right_positions, lower, upper)
    ]
    return math.sqrt(sum(value * value for value in distances))


def joint_limit_margin(state: Mapping[str, Any], *, source: Mapping[str, Any]) -> float:
    positions = [float(value) for value in state.get("positions") or []]
    if not positions:
        return 0.0
    lower, upper = _joint_limits(state, source)
    return min(
        min(value - lo, hi - value) / max(1e-9, hi - lo)
        for value, lo, hi in zip(positions, lower, upper)
    )


def fixed_recovery_seeds(
    reference: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    count: int = 6,
) -> list[JsonDict]:
    """Return the fixed remainder of the eight-seed budget (no runtime RNG)."""

    names = list(reference.get("names") or [])
    lower, upper = _joint_limits(reference, source)
    primes = (2, 3, 5, 7, 11, 13, 17, 19)

    def radical_inverse(index: int, base: int) -> float:
        result, factor = 0.0, 1.0 / base
        while index:
            result += factor * (index % base)
            index //= base
            factor /= base
        return result

    return [
        {
            "names": names,
            "positions": [
                lo + radical_inverse(seed_index + 1, primes[joint_index]) * (hi - lo)
                for joint_index, (lo, hi) in enumerate(zip(lower, upper))
            ],
            "seed_source": f"fixed_recovery_{seed_index}",
        }
        for seed_index in range(count)
    ]


def deduplicate_beam_solutions(
    solutions: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
    limit: int,
) -> list[JsonDict]:
    """Quality-sort and remove joint solutions closer than normalized 0.05."""

    selected: list[JsonDict] = []
    for solution in sorted(solutions, key=beam_solution_quality_key):
        state = solution.get("joint_state")
        if not isinstance(state, Mapping):
            continue
        if any(
            normalized_joint_distance(
                state,
                existing["joint_state"],
                source=source,
            )
            < JOINT_SOLUTION_DEDUP_DISTANCE
            for existing in selected
        ):
            continue
        selected.append(dict(solution))
        if len(selected) >= limit:
            break
    return selected


def beam_solution_quality_key(solution: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if solution.get("state_valid") is True else 1,
        -float(solution.get("joint_margin", 0.0)),
        -float(solution.get("min_singular_value", 0.0)),
        float(solution.get("cumulative_joint_travel", math.inf)),
        int(solution.get("collision_rescues", 0)),
        -float(solution.get("generator_score", 0.0)),
        int(solution.get("fixed_candidate_index", 0)),
        int(solution.get("seed_index", 0)),
    )


def candidate_physical_quality_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    stages = result.get("stages")
    stages = stages if isinstance(stages, list) else []
    centering_risk, centering_ratio = parallel_gripper_centering_quality(result)
    parent_priority = frozen_frontier_parent_priority(result)
    span_quality = parallel_gripper_target_span_quality(result)
    centering_order = (
        (span_quality, centering_ratio) if parent_priority == 0 else (centering_ratio, span_quality)
    )
    return (
        0 if result.get("endpoint_pass") is True else 1,
        centering_risk,
        # Exact positive clearance is a feasibility-quality boundary. Within
        # the same tier, low support energy and the Beam physical metrics are
        # stronger evidence than sub-resolution differences in perception-
        # derived centering or surplus bin space.
        *placement_settling_quality(result),
        -min((float(stage.get("joint_margin", 0.0)) for stage in stages), default=0.0),
        -min(
            (float(stage.get("min_singular_value", 0.0)) for stage in stages),
            default=0.0,
        ),
        sum(float(stage.get("joint_travel", 0.0)) for stage in stages),
        sum(int(stage.get("collision_rescues", 0)) for stage in stages),
        *centering_order,
        -float(result.get("generator_score", 0.0)),
        int(result.get("fixed_candidate_index", 0)),
    )


def frozen_pair_l5_submission_order(
    candidates: Sequence[Mapping[str, Any]],
    *,
    prior_attempts: Sequence[Mapping[str, Any]] = (),
) -> list[JsonDict]:
    """Quality-sort frozen-pair L5 work while preserving branch diversity.

    A plain physical-quality sort can concentrate early plans on near-duplicate
    goals from the first grasp. Greedy novelty first prefers a new grasp and
    placement cluster, then a new grasp, then a new cluster. Quality remains
    deterministic within each tier; this function never deletes candidates.
    """

    remaining = [dict(item) for item in sorted(candidates, key=candidate_physical_quality_key)]
    anchors = [dict(item) for item in prior_attempts]
    ordered: list[JsonDict] = []
    while remaining:
        if not anchors:
            selected_index = 0
        else:
            used_grasps = {
                str(item.get("source_grasp_id") or "")
                for item in anchors
                if item.get("source_grasp_id")
            }
            used_clusters = {
                str(item.get("se3_cluster_id") or "")
                for item in anchors
                if item.get("se3_cluster_id")
            }

            def novelty_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
                grasp = str(item.get("source_grasp_id") or "")
                cluster = str(item.get("se3_cluster_id") or "")
                new_grasp = bool(grasp and grasp not in used_grasps)
                new_cluster = bool(cluster and cluster not in used_clusters)
                tier = (
                    0 if new_grasp and new_cluster else 1 if new_grasp else 2 if new_cluster else 3
                )
                return (tier, *candidate_physical_quality_key(item))

            selected_index = min(
                range(len(remaining)),
                key=lambda index: novelty_key(remaining[index]),
            )
        selected = remaining.pop(selected_index)
        ordered.append(selected)
        anchors.append(selected)
    return ordered


def select_grasp_branches(
    passed: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
    limit: int = 2,
) -> list[str]:
    """Prefer distinct physical families and SE(3) clusters, then fall back."""

    ordered = sorted(passed, key=candidate_physical_quality_key)
    if len(ordered) <= limit:
        return [str(item.get("candidate_id") or "") for item in ordered]
    primary_limit = min(2, limit)
    selected = [ordered[0]]
    for item in ordered[1:]:
        if item.get("se3_cluster_id") != selected[0].get("se3_cluster_id") and item.get(
            "grasp_symmetry_family_id"
        ) != selected[0].get("grasp_symmetry_family_id"):
            selected.append(item)
            break
    if len(selected) < primary_limit:
        for item in ordered[1:]:
            if item.get("grasp_symmetry_family_id") != selected[0].get("grasp_symmetry_family_id"):
                selected.append(item)
                break
    if len(selected) < primary_limit:
        for item in ordered[1:]:
            if item.get("se3_cluster_id") != selected[0].get("se3_cluster_id"):
                selected.append(item)
                break
    if len(selected) < primary_limit:
        best_pair: tuple[float, int, int] | None = None
        for left_index, left in enumerate(ordered):
            left_state = _result_end_state(left)
            if left_state is None:
                continue
            for right_index in range(left_index + 1, len(ordered)):
                right_state = _result_end_state(ordered[right_index])
                if right_state is None:
                    continue
                distance = normalized_joint_distance(left_state, right_state, source=source)
                key = (distance, -left_index, -right_index)
                if best_pair is None or key > best_pair:
                    best_pair = key
        if best_pair is not None:
            selected = [ordered[-best_pair[1]], ordered[-best_pair[2]]]
        else:
            selected = ordered[:limit]
    while len(selected) < limit:
        selected_ids = {str(item.get("candidate_id") or "") for item in selected}
        remaining = [
            item for item in ordered if str(item.get("candidate_id") or "") not in selected_ids
        ]
        if not remaining:
            break
        used_clusters = {str(item.get("se3_cluster_id") or "") for item in selected}
        used_families = {str(item.get("grasp_symmetry_family_id") or "") for item in selected}

        def expansion_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
            cluster = str(item.get("se3_cluster_id") or "")
            family = str(item.get("grasp_symmetry_family_id") or "")
            new_cluster = bool(cluster and cluster not in used_clusters)
            new_family = bool(family and family not in used_families)
            tier = 0 if new_cluster and new_family else 1 if new_family else 2 if new_cluster else 3
            state = _result_end_state(item)
            distances: list[float] = []
            if state is not None:
                for selected_item in selected:
                    selected_state = _result_end_state(selected_item)
                    if selected_state is not None:
                        distances.append(
                            normalized_joint_distance(state, selected_state, source=source)
                        )
            min_distance = min(distances, default=0.0)
            return (
                tier,
                -min_distance,
                *candidate_physical_quality_key(item),
            )

        selected.append(min(remaining, key=expansion_key))
    return [str(item.get("candidate_id") or "") for item in selected[:limit]]


def _result_end_state(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    stages = result.get("stages")
    if not isinstance(stages, list) or not stages:
        return None
    state = stages[-1].get("end_joint_state") if isinstance(stages[-1], Mapping) else None
    return state if isinstance(state, Mapping) else None


def latency_summary(values: Sequence[float]) -> JsonDict:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "p50_s": None, "p95_s": None}

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(finite)) - 1)
        return finite[index]

    return {
        "count": len(finite),
        "p50_s": statistics.median(finite),
        "p95_s": percentile(0.95),
    }
