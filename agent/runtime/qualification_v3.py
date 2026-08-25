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


def final_target(descriptor: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = descriptor.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    stages = candidate.get("qualification_stages")
    if isinstance(stages, list) and stages and isinstance(stages[-1], Mapping):
        return stages[-1]
    return {}


def _pose_distance(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[float, float] | None:
    left_pose, right_pose = target_pose(left), target_pose(right)
    if left_pose is None or right_pose is None:
        return None
    left_xyz, left_quat = left_pose
    right_xyz, right_quat = right_pose
    translation = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left_xyz, right_xyz))
    )
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
    stages = (
        candidate.get("qualification_stages")
        if isinstance(candidate, Mapping)
        else None
    )
    return capability_map.score_chain(
        [stage for stage in stages or [] if isinstance(stage, Mapping)]
    )


def _descriptor_priority(descriptor: Mapping[str, Any]) -> tuple[Any, ...]:
    score = descriptor.get("capability_score")
    score = score if isinstance(score, Mapping) else {}
    candidate = descriptor.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    return (
        -float(score.get("confidence", 0.0)),
        -float(score.get("reachable_density", 0.0)),
        -float(score.get("joint_margin", 0.0)),
        -float(score.get("min_singular_value", 0.0)),
        -generator_score(candidate),
        int(descriptor.get("fixed_candidate_index", 0)),
    )


def _cluster_round_robin(
    descriptors: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
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
    capability_map: SparseCapabilityMap | None = None,
) -> list[CandidateWave]:
    """Build cumulative waves without deleting any structurally valid candidate."""

    annotated = assign_se3_clusters(descriptors)
    for descriptor in annotated:
        descriptor["capability_score"] = _capability_score(
            descriptor, capability_map
        ).to_dict()
    if purpose == "grasp":
        ordered = _cluster_round_robin(annotated)
        cumulative = sorted(
            set([value for value in grasp_waves if value < len(ordered)] + [len(ordered)])
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
                if isinstance(raw_batch, int)
                and not isinstance(raw_batch, bool)
                and raw_batch >= 0
                else 0
            )
        branches[branch].append(descriptor)
    ordered_branches = {
        branch: _cluster_round_robin(branches[branch]) for branch in branch_order
    }
    waves: list[CandidateWave] = []
    for batch_index in sorted(set(branch_batch.values())):
        current_branches = [
            branch
            for branch in branch_order
            if branch_batch[branch] == batch_index
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


def joint_limit_margin(
    state: Mapping[str, Any], *, source: Mapping[str, Any]
) -> float:
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
    return (
        0 if result.get("endpoint_pass") is True else 1,
        -min((float(stage.get("joint_margin", 0.0)) for stage in stages), default=0.0),
        -min(
            (float(stage.get("min_singular_value", 0.0)) for stage in stages),
            default=0.0,
        ),
        sum(float(stage.get("joint_travel", 0.0)) for stage in stages),
        sum(int(stage.get("collision_rescues", 0)) for stage in stages),
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

    remaining = [
        dict(item)
        for item in sorted(candidates, key=candidate_physical_quality_key)
    ]
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
                    0
                    if new_grasp and new_cluster
                    else 1
                    if new_grasp
                    else 2
                    if new_cluster
                    else 3
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
        if (
            item.get("se3_cluster_id") != selected[0].get("se3_cluster_id")
            and item.get("grasp_symmetry_family_id")
            != selected[0].get("grasp_symmetry_family_id")
        ):
            selected.append(item)
            break
    if len(selected) < primary_limit:
        for item in ordered[1:]:
            if (
                item.get("grasp_symmetry_family_id")
                != selected[0].get("grasp_symmetry_family_id")
            ):
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
                distance = normalized_joint_distance(
                    left_state, right_state, source=source
                )
                key = (distance, -left_index, -right_index)
                if best_pair is None or key > best_pair:
                    best_pair = key
        if best_pair is not None:
            selected = [ordered[-best_pair[1]], ordered[-best_pair[2]]]
        else:
            selected = ordered[:limit]
    while len(selected) < limit:
        selected_ids = {
            str(item.get("candidate_id") or "") for item in selected
        }
        remaining = [
            item
            for item in ordered
            if str(item.get("candidate_id") or "") not in selected_ids
        ]
        if not remaining:
            break
        used_clusters = {
            str(item.get("se3_cluster_id") or "") for item in selected
        }
        used_families = {
            str(item.get("grasp_symmetry_family_id") or "") for item in selected
        }

        def expansion_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
            cluster = str(item.get("se3_cluster_id") or "")
            family = str(item.get("grasp_symmetry_family_id") or "")
            new_cluster = bool(cluster and cluster not in used_clusters)
            new_family = bool(family and family not in used_families)
            tier = (
                0
                if new_cluster and new_family
                else 1
                if new_family
                else 2
                if new_cluster
                else 3
            )
            state = _result_end_state(item)
            distances: list[float] = []
            if state is not None:
                for selected_item in selected:
                    selected_state = _result_end_state(selected_item)
                    if selected_state is not None:
                        distances.append(
                            normalized_joint_distance(
                                state, selected_state, source=source
                            )
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
