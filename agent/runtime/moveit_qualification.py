"""Host-only MoveIt candidate qualification and immutable proof cache.

This module deliberately has no AgentTool registration.  The runtime may call
the simulator's private ``qualify_motion_candidates`` RPC, but planners cannot
request arbitrary IK or plan-only motion.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict
from agent.tools.registry import ToolResult
from agent.runtime.capability_map import SparseCapabilityMap, target_pose
from agent.runtime.qualification_legality import (
    CONFIGURED_RELEASE_HEIGHT_FALLBACK,
    RELEASE_HEIGHT_VARIANT_FIELD,
    bind_qualified_placement_goal,
    evaluate_grasp_target_closing_alignment,
    evaluate_grasp_placement_pair_legality,
    evaluate_placement_goal_legality,
)
from agent.runtime.qualification_v3 import (
    FAST_ARTIFACT_SCHEMA,
    FAST_QUALIFICATION_SCHEMA,
    JOINT_SOLUTION_DEDUP_DISTANCE,
    CandidateWave,
    candidate_physical_quality_key,
    deduplicate_beam_solutions,
    fixed_recovery_seeds,
    generator_score,
    grasp_symmetry_family_id,
    joint_limit_margin,
    latency_summary,
    normalized_joint_distance,
    parallel_gripper_centering_evidence,
    frozen_pair_l5_submission_order,
    schedule_candidate_waves,
    select_grasp_branches,
)
from tools.candidate_config import (
    DEFAULT_ANYPLACE_DIVERSITY_POOL_SIZE,
    DEFAULT_ANYPLACE_FULL_PLAN_LIMIT,
    DEFAULT_GRASP_DIVERSITY_POOL_SIZE,
    DEFAULT_GRASP_FULL_PLAN_LIMIT,
    DEFAULT_GRASP_WAVES,
    DEFAULT_MOVEIT_IK_SEED_COUNT,
    DEFAULT_FROZEN_PAIR_FULL_PLAN_LIMIT,
)


QUALIFICATION_SCHEMA = "openeta.moveit_candidate_funnel.v2"
QUALIFICATION_SCHEMA_V3 = FAST_QUALIFICATION_SCHEMA
SUPPORTED_QUALIFICATION_SCHEMAS = (QUALIFICATION_SCHEMA, QUALIFICATION_SCHEMA_V3)
PRIVATE_RPC_NAME = "qualify_motion_candidates"
PLANNING_TIME_S = 30.0
PLANNING_ATTEMPTS = 3
KINEMATIC_IK_TIMEOUT_S = 2.0
# This is an infrastructure response deadline, not a solver/search budget.
# Attached-object collision checks can briefly queue behind an eight-request
# wave, so retain a bounded margin without turning service latency into a
# false candidate rejection.
STATE_VALIDITY_TIMEOUT_S = 5.0
QUALIFICATION_RPC_GRACE_S = 30.0
# Qualification may legitimately continue beyond this acknowledgement window.
# The first read-only MCP attempt is bounded so a lost HTTP result cannot
# strand the TUI behind the exhaustive solver deadline.  A health-gated second
# attempt retains the complete logical timeout and is served idempotently by
# the simulator's qualification-response cache.
QUALIFICATION_RPC_FIRST_ACK_TIMEOUT_S = 60.0
# Goal prebinding returns one immutable legality record per provider goal.  The
# simulator serializes that read-only call with live dashboard/control work, so
# its transport envelope must also scale with the amount of evidence encoded
# and returned.  This is deliberately an outer RPC allowance, not extra search
# time and not a candidate-specific reachability heuristic.
GOAL_PREBIND_RPC_PER_CANDIDATE_S = 0.5
PROGRESSIVE_SCREENING_MODE = "progressive_until_full_plan_capacity"
PROGRESSIVE_NOT_EVALUATED_REASON = "progressive_endpoint_capacity_reached"
SAME_RUN_QUALIFICATION_SEED_FIELD = "_openeta_same_run_qualification_seed_evidence"
SAME_RUN_QUALIFICATION_SEED_SCHEMA = "openeta.same_run_qualification_seed_evidence.v1"
SAME_RUN_QUALIFICATION_SEED_PROVENANCE = "frozen_pair_l5_pass"


def _contact_allowed_collisions(
    scene: object,
    target: Mapping[str, Any],
) -> JsonDict:
    """Return request-local target/touch-link exceptions for exact contact.

    The world PlanningScene remains strict while the arm approaches.  Only the
    contact endpoint and its plan-only proof receive this policy, so expected
    target contact cannot be confused with table, fixture, or arm collision.
    """

    if target.get("grasp_stage") != "contact" or not isinstance(scene, Mapping):
        return {}
    target_id = str(scene.get("target_id") or "")
    links = scene.get("target_touch_links")
    if not target_id or not isinstance(links, (list, tuple)):
        return {}
    normalized = sorted({str(link) for link in links if str(link)})
    return {target_id: normalized} if normalized else {}


def _with_qualification_state_policy(
    state: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    scene_diff: Mapping[str, Any] | None,
    queue_depth: int | None = None,
) -> JsonDict:
    result = dict(state)
    if queue_depth is not None:
        result["qualification_state_validity_queue_depth"] = int(queue_depth)
    if scene_diff is not None:
        result["qualification_scene_diff"] = dict(scene_diff)
    allowed = target.get("qualification_allowed_collisions")
    if isinstance(allowed, Mapping):
        result["qualification_allowed_collisions"] = {
            str(key): [str(value) for value in values]
            for key, values in allowed.items()
            if isinstance(values, (list, tuple))
        }
    return result


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _unique_joint_state_seeds(values: Sequence[Mapping[str, Any]], *, limit: int) -> list[JsonDict]:
    seeds: list[JsonDict] = []
    identities: set[str] = set()
    for value in values:
        identity = _hash(
            {
                "names": list(value.get("names") or []),
                "positions": list(value.get("positions") or []),
            }
        )
        if identity in identities:
            continue
        identities.add(identity)
        seeds.append(dict(value))
        if len(seeds) >= max(1, limit):
            break
    return seeds


def _valid_joint_state(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    names, positions = value.get("names"), value.get("positions")
    if (
        not isinstance(names, Sequence)
        or isinstance(names, (str, bytes, bytearray))
        or not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes, bytearray))
        or not names
        or len(names) != len(positions)
        or len({str(name) for name in names}) != len(names)
    ):
        return False
    try:
        parsed = [float(position) for position in positions]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(position) for position in parsed)


def _sanitized_joint_state(value: object) -> JsonDict | None:
    if not _valid_joint_state(value):
        return None
    assert isinstance(value, Mapping)
    return {
        "names": [str(name) for name in value.get("names") or []],
        "positions": [float(position) for position in value.get("positions") or []],
    }


def _selected_ik_state_sha256(proof: Mapping[str, Any]) -> str:
    """Return the exact first-stage IK state selected for L5 planning."""

    stages = proof.get("stages")
    if not isinstance(stages, list) or len(stages) != 1:
        return ""
    stage = stages[0]
    if not isinstance(stage, Mapping):
        return ""
    recorded = stage.get("selected_ik_end_joint_state_sha256")
    if isinstance(recorded, str) and recorded:
        return recorded
    state = _sanitized_joint_state(stage.get("end_joint_state"))
    return _hash(state) if state is not None else ""


def _same_run_seed_evidence(
    proof: Mapping[str, Any], *, source_candidate_id: str
) -> JsonDict | None:
    """Extract bounded, validated seeds from this run's first planned stage."""

    stages = proof.get("stages")
    if not isinstance(stages, list) or not stages:
        return None
    first = stages[0]
    if not isinstance(first, Mapping):
        return None
    raw_states: list[Mapping[str, Any]] = []
    planned_end = first.get("end_joint_state")
    if isinstance(planned_end, Mapping):
        raw_states.append(planned_end)
    beam = first.get("beam_solutions")
    if isinstance(beam, list):
        for solution in beam:
            state = solution.get("joint_state") if isinstance(solution, Mapping) else None
            if isinstance(state, Mapping):
                raw_states.append(state)
    sanitized = [
        state for value in raw_states if (state := _sanitized_joint_state(value)) is not None
    ]
    states = _unique_joint_state_seeds(sanitized, limit=2)
    if not states:
        return None
    return {
        "schema_version": SAME_RUN_QUALIFICATION_SEED_SCHEMA,
        "provenance": SAME_RUN_QUALIFICATION_SEED_PROVENANCE,
        "source_candidate_id": source_candidate_id,
        "source_stage_index": 0,
        "source_stage_name": str(first.get("name") or "stage_0"),
        "states": states,
    }


def _candidate_same_run_seed_states(candidate: Mapping[str, Any]) -> list[JsonDict]:
    evidence = candidate.get(SAME_RUN_QUALIFICATION_SEED_FIELD)
    if not (
        isinstance(evidence, Mapping)
        and evidence.get("schema_version") == SAME_RUN_QUALIFICATION_SEED_SCHEMA
        and evidence.get("provenance") == SAME_RUN_QUALIFICATION_SEED_PROVENANCE
    ):
        return []
    raw_states = evidence.get("states")
    if not isinstance(raw_states, list):
        return []
    sanitized = [
        state for value in raw_states if (state := _sanitized_joint_state(value)) is not None
    ]
    return _unique_joint_state_seeds(sanitized, limit=2)


class _QualificationInfrastructureError(RuntimeError):
    """Repeated ROS/service failure that must not be called unreachable."""


@dataclass(slots=True)
class _ServiceConcurrencyGate:
    limit: int
    _semaphore: threading.BoundedSemaphore = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    active: int = 0
    maximum_active: int = 0

    def __post_init__(self) -> None:
        self._semaphore = threading.BoundedSemaphore(max(1, self.limit))

    def call(self, callback: Callable[[], Mapping[str, Any]]) -> Mapping[str, Any]:
        with self._semaphore:
            with self._lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                return callback()
            finally:
                with self._lock:
                    self.active -= 1


@dataclass(slots=True)
class QualificationCache:
    """Runtime-owned binding from a qualified id to its unmodified geometry."""

    _entries: dict[tuple[str, str], JsonDict] = field(default_factory=dict)

    def replace(
        self,
        *,
        purpose: str,
        candidates: Sequence[Mapping[str, Any]],
        proofs: Mapping[str, Mapping[str, Any]],
        scene_epoch: int,
        planning_scene_revision: int,
    ) -> None:
        self.invalidate(purpose=purpose)
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "")
            proof = proofs.get(candidate_id)
            if candidate_id and isinstance(proof, Mapping):
                self._entries[(purpose, candidate_id)] = {
                    "candidate": dict(candidate),
                    "proof": dict(proof),
                    "scene_epoch": scene_epoch,
                    "planning_scene_revision": planning_scene_revision,
                }

    def resolve(
        self,
        *,
        purpose: str,
        candidate_id: str,
        scene_epoch: int | None = None,
        planning_scene_revision: int | None = None,
    ) -> JsonDict | None:
        entry = self._entries.get((purpose, candidate_id))
        if not entry or (scene_epoch is not None and entry["scene_epoch"] != scene_epoch):
            return None
        if (
            planning_scene_revision is not None
            and entry["planning_scene_revision"] != planning_scene_revision
        ):
            return None
        return dict(entry)

    def invalidate(self, *, purpose: str | None = None) -> None:
        if purpose is None:
            self._entries.clear()
            return
        for key in [key for key in self._entries if key[0] == purpose]:
            self._entries.pop(key, None)


class MoveItCandidateQualifier:
    """Validate one batch response and retain every candidate proven PASS."""

    def __init__(
        self,
        rpc: Callable[[str, JsonDict, float], JsonDict],
        *,
        cache: QualificationCache | None = None,
        artifact_root: str | Path | None = None,
        compile_candidate: Callable[
            [Mapping[str, Any], str, Mapping[str, Any], int, int], Mapping[str, Any]
        ]
        | None = None,
        grasp_diversity_limit: int = DEFAULT_GRASP_DIVERSITY_POOL_SIZE,
        placement_diversity_limit: int = DEFAULT_ANYPLACE_DIVERSITY_POOL_SIZE,
        observation_diversity_limit: int = 24,
        grasp_full_plan_limit: int = DEFAULT_GRASP_FULL_PLAN_LIMIT,
        placement_full_plan_limit: int = DEFAULT_ANYPLACE_FULL_PLAN_LIMIT,
        observation_full_plan_limit: int = 2,
        frozen_pair_full_plan_limit: int = DEFAULT_FROZEN_PAIR_FULL_PLAN_LIMIT,
        ik_seed_count: int = DEFAULT_MOVEIT_IK_SEED_COUNT,
        qualification_profile: str = "legacy",
        solver_profile: str = "auto",
        beam_width: int = 2,
        grasp_waves: Sequence[int] = DEFAULT_GRASP_WAVES,
        placement_waves: Sequence[int] = (4, 8, 16, 32, 96),
        observation_waves: Sequence[int] = (4, 8, 16, 24),
        max_ik_concurrency: int = 8,
        max_state_validity_concurrency: int = 8,
        fast_seed_count: int = 2,
        recovery_seed_count: int = 6,
        fast_ik_timeout_s: float = 0.05,
        recovery_ik_timeout_s: float = 0.2,
        capability_map_id: str = "",
    ) -> None:
        self.rpc = rpc
        self.cache = cache or QualificationCache()
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None
        self.compile_candidate = compile_candidate
        self.diversity_limits = {
            "grasp": int(grasp_diversity_limit),
            "placement": int(placement_diversity_limit),
            "observation": int(observation_diversity_limit),
        }
        self.full_plan_limits = {
            "grasp": int(grasp_full_plan_limit),
            "placement": int(placement_full_plan_limit),
            "observation": int(observation_full_plan_limit),
        }
        self.frozen_pair_full_plan_limit = int(frozen_pair_full_plan_limit)
        self.ik_seed_count = int(ik_seed_count)
        self.qualification_profile = str(qualification_profile)
        self.solver_profile = str(solver_profile)
        self.beam_width = int(beam_width)
        self.grasp_waves = tuple(int(value) for value in grasp_waves)
        self.placement_waves = tuple(int(value) for value in placement_waves)
        self.observation_waves = tuple(int(value) for value in observation_waves)
        self.max_ik_concurrency = int(max_ik_concurrency)
        self.max_state_validity_concurrency = int(max_state_validity_concurrency)
        self.fast_seed_count = int(fast_seed_count)
        self.recovery_seed_count = int(recovery_seed_count)
        self.fast_ik_timeout_s = float(fast_ik_timeout_s)
        self.recovery_ik_timeout_s = float(recovery_ik_timeout_s)
        self.capability_map_id = str(capability_map_id)

    def prebind_placement_goals(
        self,
        goals: Sequence[Mapping[str, Any]],
        *,
        scene_epoch: int,
        planning_scene_revision: int,
        source: Mapping[str, Any] | None = None,
        release_height_variant: str | None = None,
    ) -> tuple[list[JsonDict], JsonDict]:
        """Run the cheap complete goal barrier before compiling any pair.

        Container release semantics depend on authoritative support collision
        geometry owned by the simulator.  Resolve that geometry once for the
        frozen AnyPlace frontier, then compile every grasp/goal pair from the
        resulting immutable release goal.  This prevents a late legality pass
        from changing a pose identity after IK/L5 compilation.
        """

        if release_height_variant not in {
            None,
            CONFIGURED_RELEASE_HEIGHT_FALLBACK,
        }:
            raise ValueError("placement release-height variant is invalid")
        descriptors = []
        for index, goal in enumerate(goals):
            candidate = json.loads(json.dumps(goal))
            # Provider output cannot select a host-private recovery policy.
            candidate.pop(RELEASE_HEIGHT_VARIANT_FIELD, None)
            if release_height_variant is not None:
                candidate[RELEASE_HEIGHT_VARIANT_FIELD] = release_height_variant
            descriptors.append(
                {
                    "candidate_id": str(
                        goal.get("id") or f"placement_{index:03d}"
                    ),
                    "candidate_pose_sha256": _hash(candidate),
                    "candidate": candidate,
                }
            )
        request: JsonDict = {
            "schema_version": QUALIFICATION_SCHEMA_V3,
            "purpose": "placement",
            "scene_epoch": scene_epoch,
            "planning_scene_revision": planning_scene_revision,
            "planning": {
                "allowed_planning_time_s": PLANNING_TIME_S,
                "num_planning_attempts": PLANNING_ATTEMPTS,
                "plan_only": True,
            },
            "funnel": {
                "qualification_profile": "fast_v3",
                "qualification_mode": "goal_prebind",
                "release_height_variant": (
                    release_height_variant or "geometry_primary"
                ),
            },
            "source": dict(source or {}),
            "candidates": descriptors,
        }
        request["qualification_case_sha256"] = _hash(
            {
                "purpose": "placement_goal_prebind",
                "scene_epoch": scene_epoch,
                "planning_scene_revision": planning_scene_revision,
                "candidate_pose_sha256": [
                    descriptor["candidate_pose_sha256"] for descriptor in descriptors
                ],
            }
        )
        request["qualification_binding_sha256"] = _hash(
            {
                "purpose": request["purpose"],
                "scene_epoch": scene_epoch,
                "planning_scene_revision": planning_scene_revision,
                "funnel": request["funnel"],
                "candidate_pose_sha256": [
                    descriptor["candidate_pose_sha256"] for descriptor in descriptors
                ],
            }
        )
        rpc_timeout_s = _goal_prebind_rpc_timeout_s(descriptors)
        response = self.rpc(
            PRIVATE_RPC_NAME,
            request,
            rpc_timeout_s,
        )
        if not isinstance(response, Mapping):
            raise RuntimeError("placement goal prebind returned no evidence")
        if (
            response.get("schema_version") != QUALIFICATION_SCHEMA_V3
            or response.get("planning_scene_revision") != planning_scene_revision
            or response.get("execution_started") is not False
            or response.get("infrastructure_error") is True
        ):
            raise RuntimeError(
                str(response.get("stop_reason") or "placement goal prebind failed")
            )
        rows = response.get("results")
        if not isinstance(rows, list) or len(rows) != len(descriptors):
            raise RuntimeError("placement goal prebind evidence is incomplete")
        expected = {descriptor["candidate_id"] for descriptor in descriptors}
        received = {
            str(row.get("candidate_id") or "")
            for row in rows
            if isinstance(row, Mapping)
        }
        if received != expected:
            raise RuntimeError("placement goal prebind candidate identity mismatch")
        bound: list[JsonDict] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            verdict = str(row.get("verdict") or "UNKNOWN").upper()
            if verdict == "PASS" and isinstance(row.get("prebound_candidate"), Mapping):
                bound.append(json.loads(json.dumps(row["prebound_candidate"])))
            elif verdict not in {"PASS", "FAIL"}:
                raise RuntimeError(
                    str(row.get("reason") or "placement goal prebind indeterminate")
                )
        response_metrics = response.get("metrics")
        response_metrics = (
            response_metrics if isinstance(response_metrics, Mapping) else {}
        )
        return bound, {
            "frozen_goal_legality_screen_complete": True,
            "frozen_goal_legality_evidence_count": len(rows),
            "frozen_goal_legality_pass_count": len(bound),
            "frozen_goal_legality_reject_count": len(rows) - len(bound),
            "frozen_goal_legality_frontier_count": len(bound),
            "frozen_goal_legality_elapsed_s": response_metrics.get(
                "goal_legality_elapsed_s"
            ),
            "frozen_goal_legality_rpc_timeout_s": rpc_timeout_s,
            "frozen_goal_release_height_variant": (
                release_height_variant or "geometry_primary"
            ),
        }

    def qualify_result(
        self,
        result: ToolResult,
        *,
        purpose: str,
        scene_epoch: int,
        planning_scene_revision: int,
        source: Mapping[str, Any] | None = None,
        cache_result: bool = True,
        qualification_mode: str = "standard",
        l5_pass_target: int | None = None,
        l5_min_pass_target: int | None = None,
    ) -> ToolResult:
        if not result.success:
            return result
        if purpose not in {"grasp", "placement", "observation"}:
            raise ValueError("candidate qualification purpose is invalid")
        details = result.details
        key = {
            "grasp": "grasp_candidates",
            "placement": "placement_candidates",
            "observation": "observation_candidates",
        }[purpose]
        raw = details.get(key)
        if not isinstance(raw, list) or not raw:
            return result
        fast_profile = self.qualification_profile in {"fast_v3", "shadow"}
        shadow_legacy_input_ids: set[str] | None = None
        if purpose == "grasp":
            # Grasp providers own terminal pose generation.  Keep each model
            # result byte-for-byte through scheduling; the host may rank or
            # reject an invalid pose, but must not translate, rotate, mirror,
            # or reverse it into a new candidate.
            raw = [dict(candidate) for candidate in raw if isinstance(candidate, Mapping)]
            details[key] = raw
        if purpose == "placement" and qualification_mode == "standard":
            combined = list(raw)
            if fast_profile:
                if self.qualification_profile == "shadow":
                    shadow_legacy_input_ids = {
                        str(candidate.get("id") or "")
                        for candidate in _deduplicate_se3_candidates(
                            combined,
                            round_index=1,
                        )
                    }
                # Every AnyPlace result remains an independent scheduling
                # unit in v3. Exact-pose duplicates share a cluster, but are
                # never deleted; only missing/duplicate IDs are repaired.
                raw = _preserve_candidate_pool(
                    combined,
                    round_index=1,
                )
            else:
                raw = _deduplicate_se3_candidates(
                    combined,
                    round_index=1,
                )
            details[key] = raw
        generated = len(raw)
        compiled_descriptors = []
        compile_rejections: list[JsonDict] = []
        compile_candidate = self.compile_candidate
        compile_prepare_error: Exception | None = None
        if compile_candidate is not None:
            prepare_batch = getattr(compile_candidate, "prepare_batch", None)
            if callable(prepare_batch):
                try:
                    prepared = prepare_batch(purpose=purpose)
                    if not callable(prepared):
                        raise TypeError("prepared candidate compiler is not callable")
                    compile_candidate = prepared
                except Exception as exc:  # noqa: BLE001 - fail closed per candidate.
                    compile_prepare_error = exc
        for candidate in raw:
            if not isinstance(candidate, Mapping):
                continue
            compiled: Mapping[str, Any] = {}
            precompiled_stages = candidate.get("qualification_stages")
            if compile_candidate is not None:
                try:
                    if compile_prepare_error is not None:
                        raise compile_prepare_error
                    compiled = compile_candidate(
                        candidate,
                        purpose,
                        source or {},
                        scene_epoch,
                        planning_scene_revision,
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed before RPC.
                    if isinstance(precompiled_stages, list) and precompiled_stages:
                        compiled = {"qualification_stages": list(precompiled_stages)}
                    else:
                        compile_rejections.append(
                            {
                                "candidate_id": str(candidate.get("id") or ""),
                                "candidate_pose_sha256": _hash(candidate),
                                "verdict": "UNKNOWN",
                                "reason": "coordinate_tcp_compilation_failed",
                                "error_type": type(exc).__name__,
                                "execution_started": False,
                            }
                        )
                        continue
            elif isinstance(precompiled_stages, list) and precompiled_stages:
                compiled = {"qualification_stages": list(precompiled_stages)}
            rpc_candidate = dict(candidate)
            if isinstance(compiled.get("qualification_stages"), list):
                rpc_candidate["qualification_stages"] = list(compiled["qualification_stages"])
            if isinstance(compiled.get("compile_parameters"), Mapping):
                rpc_candidate["compile_parameters"] = dict(compiled["compile_parameters"])
            compiled_descriptors.append(
                {
                    "candidate_id": str(candidate.get("id") or ""),
                    "candidate_pose_sha256": _hash(candidate),
                    "candidate": rpc_candidate,
                }
            )
        if qualification_mode == "frozen_pair":
            # Every constructed grasp/object-goal pair reaches compilation and
            # the conservative structural precheck.  The private helper then
            # traverses L3/L4 in stable round-robin order only until it fills
            # the complete L5 submission capacity.  Unvisited pairs remain
            # explicitly NOT_EVALUATED; they are never relabelled as failures.
            request_candidates = compiled_descriptors
            full_plan_limit = self.frozen_pair_full_plan_limit
        elif fast_profile:
            # fast_v3 must eventually expose every structurally valid model
            # result. Capability and empirical scores affect wave ordering only.
            request_candidates = compiled_descriptors
            full_plan_limit = self.full_plan_limits[purpose]
        else:
            request_candidates = diversify_compiled_candidates(
                compiled_descriptors,
                purpose=purpose,
                limit=self.diversity_limits[purpose],
            )
            full_plan_limit = self.full_plan_limits[purpose]
        if l5_pass_target is not None:
            if (
                isinstance(l5_pass_target, bool)
                or not isinstance(l5_pass_target, int)
                or l5_pass_target <= 0
            ):
                raise ValueError("l5_pass_target must be a positive integer")
            if fast_profile:
                full_plan_limit = max(full_plan_limit, l5_pass_target)
        if l5_min_pass_target is not None:
            if (
                isinstance(l5_min_pass_target, bool)
                or not isinstance(l5_min_pass_target, int)
                or l5_min_pass_target <= 0
            ):
                raise ValueError("l5_min_pass_target must be a positive integer")
            effective_target = l5_pass_target or full_plan_limit
            if l5_min_pass_target > effective_target:
                raise ValueError("l5_min_pass_target cannot exceed l5_pass_target")
        funnel_config: JsonDict = {
            "ik_seed_count": self.ik_seed_count,
            "full_plan_limit": full_plan_limit,
            "screening_mode": PROGRESSIVE_SCREENING_MODE,
            # This is derived from L5 capacity.  L1/L2 still cover the
            # complete submitted batch; only candidates with no remaining
            # chance to reach L5 skip the expensive L3/L4 tail.
            "endpoint_pass_target": full_plan_limit,
        }
        if fast_profile:
            funnel_config.update(
                {
                    "qualification_profile": self.qualification_profile,
                    "solver_profile": self.solver_profile,
                    "beam_width": self.beam_width,
                    "grasp_waves": list(self.grasp_waves),
                    "placement_waves": list(self.placement_waves),
                    "observation_waves": list(self.observation_waves),
                    "max_ik_concurrency": self.max_ik_concurrency,
                    "max_state_validity_concurrency": self.max_state_validity_concurrency,
                    "fast_seed_count": self.fast_seed_count,
                    "recovery_seed_count": self.recovery_seed_count,
                    "fast_ik_timeout_s": self.fast_ik_timeout_s,
                    "recovery_ik_timeout_s": self.recovery_ik_timeout_s,
                    "capability_map_id": self.capability_map_id,
                    "l5_pass_target": (
                        l5_pass_target
                        if l5_pass_target is not None
                        else full_plan_limit
                        if qualification_mode == "frozen_pair"
                        else 2
                        if purpose == "grasp"
                        else full_plan_limit
                        if purpose == "placement"
                        else 1
                    ),
                    "l5_min_pass_target": (
                        l5_min_pass_target
                        if l5_min_pass_target is not None
                        else l5_pass_target
                        if l5_pass_target is not None
                        else full_plan_limit
                        if qualification_mode == "frozen_pair"
                        else 2
                        if purpose == "grasp"
                        else 1
                        if purpose == "placement"
                        else 1
                    ),
                }
            )
            if qualification_mode == "frozen_pair":
                # Preserve up to the configured number of independently
                # qualified frozen alternates, but continue L5 attempts until
                # that PASS target is met or the deterministic pool exhausts.
                funnel_config["qualification_mode"] = "frozen_pair"
            if self.qualification_profile == "shadow":
                shadow_legacy_candidates = diversify_compiled_candidates(
                    [
                        descriptor
                        for descriptor in compiled_descriptors
                        if shadow_legacy_input_ids is None
                        or str(descriptor.get("candidate_id") or "") in shadow_legacy_input_ids
                    ],
                    purpose=purpose,
                    limit=self.diversity_limits[purpose],
                )
                funnel_config["shadow_legacy_candidate_ids"] = [
                    str(item.get("candidate_id") or "") for item in shadow_legacy_candidates
                ]
        request: JsonDict = {
            "schema_version": (QUALIFICATION_SCHEMA_V3 if fast_profile else QUALIFICATION_SCHEMA),
            "purpose": purpose,
            "scene_epoch": scene_epoch,
            "planning_scene_revision": planning_scene_revision,
            "planning": {
                "allowed_planning_time_s": PLANNING_TIME_S,
                "num_planning_attempts": PLANNING_ATTEMPTS,
                "plan_only": True,
                "kinematic_ik_timeout_s": KINEMATIC_IK_TIMEOUT_S,
                "state_validity_timeout_s": STATE_VALIDITY_TIMEOUT_S,
            },
            "funnel": funnel_config,
            "source": dict(source or {}),
            "candidates": request_candidates,
        }
        request["qualification_case_sha256"] = str(
            (source or {}).get("case_id")
            or (source or {}).get("scenario_id")
            or _hash(
                {
                    "purpose": purpose,
                    "scene_epoch": scene_epoch,
                    "planning_scene_revision": planning_scene_revision,
                    "provider": (source or {}).get("provider"),
                    "provider_version": (source or {}).get("provider_version"),
                    "candidate_pool": [
                        {
                            "candidate_id": item["candidate_id"],
                            "candidate_pose_sha256": item["candidate_pose_sha256"],
                            "compiled_candidate_sha256": _hash(item["candidate"]),
                        }
                        for item in compiled_descriptors
                    ],
                }
            )
        )
        request["qualification_binding_sha256"] = _hash(
            {
                "purpose": purpose,
                "scene_epoch": scene_epoch,
                "planning_scene_revision": planning_scene_revision,
                "planning": request["planning"],
                "funnel": request["funnel"],
                "source": request["source"],
                "candidate_pose_sha256": [
                    item["candidate_pose_sha256"] for item in request_candidates
                ],
                "compiled_candidate_sha256": [
                    _hash(item["candidate"]) for item in request_candidates
                ],
            }
        )
        rpc_attempts = 2 if fast_profile else 1
        response: JsonDict | None = None
        validated_response: JsonDict | None = None
        rpc_error: Exception | None = None
        completed_rpc_attempts = 0
        for _attempt in range(rpc_attempts):
            try:
                response = self.rpc(
                    PRIVATE_RPC_NAME,
                    request,
                    _qualification_rpc_timeout_s(
                        request_candidates,
                        full_plan_limit=full_plan_limit,
                        qualification_profile=self.qualification_profile,
                        fast_seed_count=self.fast_seed_count,
                        recovery_seed_count=self.recovery_seed_count,
                        fast_ik_timeout_s=self.fast_ik_timeout_s,
                        recovery_ik_timeout_s=self.recovery_ik_timeout_s,
                    ),
                )
                completed_rpc_attempts = _attempt + 1
                validated_response = self._validate_response(request, response)
                if validated_response.get("infrastructure_error") is not True:
                    break
            except Exception as exc:  # noqa: BLE001 - private transport boundary.
                rpc_error = exc
                completed_rpc_attempts = _attempt + 1
        if response is None:
            exc = rpc_error or RuntimeError("qualification RPC failed")
            response = {
                "schema_version": request["schema_version"],
                "planning_scene_revision": planning_scene_revision,
                "execution_started": False,
                "stop_reason": "infrastructure_error",
                "infrastructure_error": True,
                "rpc_attempt_count": rpc_attempts,
                "results": [
                    {
                        "candidate_id": item["candidate_id"],
                        "candidate_pose_sha256": item["candidate_pose_sha256"],
                        "verdict": "UNKNOWN",
                        "reason": "qualification_rpc_error",
                        "error_type": type(exc).__name__,
                        "execution_started": False,
                        "qualification_binding_sha256": request["qualification_binding_sha256"],
                    }
                    for item in request_candidates
                ],
            }
        evidence = validated_response or self._validate_response(request, response)
        evidence["rpc_attempt_count"] = max(
            int(evidence.get("rpc_attempt_count") or 0), completed_rpc_attempts
        )
        for item in compile_rejections:
            item["qualification_binding_sha256"] = request["qualification_binding_sha256"]
            evidence["results"].append(item)
        proofs = {item["candidate_id"]: item for item in evidence["results"]}
        raw_selected_candidate_ids = evidence.get("selected_candidate_ids")
        selected_candidate_order = (
            [str(value) for value in raw_selected_candidate_ids]
            if isinstance(raw_selected_candidate_ids, list)
            else None
        )
        raw_by_id = {
            str(candidate.get("id") or ""): candidate
            for candidate in raw
            if isinstance(candidate, Mapping)
        }
        if selected_candidate_order is None:
            passed = [
                dict(candidate)
                for candidate in raw
                if isinstance(candidate, Mapping)
                and proofs.get(str(candidate.get("id") or ""), {}).get("verdict") == "PASS"
            ]
        else:
            # The engine's selected order is the deterministic physical-quality
            # order. Preserve it for the primary/reserve split instead of
            # reducing the evidence to an unordered membership set.
            passed = [
                dict(raw_by_id[candidate_id])
                for candidate_id in selected_candidate_order
                if candidate_id in raw_by_id
                and proofs.get(candidate_id, {}).get("verdict") == "PASS"
            ]
        for physical_rank, candidate in enumerate(passed):
            candidate_id = str(candidate.get("id") or "")
            proof = proofs.get(candidate_id, {})
            if self.qualification_profile == "fast_v3":
                # Preserve the engine's deterministic L5 quality order across
                # the public tool boundary.  Generator score remains a final
                # tie-breaker; it must not undo joint margin, singularity,
                # travel, rescue, and fixed-index ordering after MoveIt proof.
                candidate["moveit_physical_quality_rank"] = physical_rank
                candidate["moveit_l5_qualified"] = True
            scene_alignment = proof.get("scene_target_closing_alignment")
            if isinstance(scene_alignment, Mapping):
                candidate["scene_target_closing_alignment"] = json.loads(
                    json.dumps(scene_alignment)
                )
            goal_legality = proof.get("goal_legality")
            checks = goal_legality.get("checks") if isinstance(goal_legality, Mapping) else None
            binding = checks.get("object_frame_binding") if isinstance(checks, Mapping) else None
            collision_goal = (
                binding.get("collision_goal_pose") if isinstance(binding, Mapping) else None
            )
            if isinstance(collision_goal, Mapping):
                # Carry the exact PlanningScene collision-body goal across the
                # private qualification boundary.  AnyPlace's visible point
                # centroid remains provenance, but must not become the frozen
                # physical object pose after attachment.
                candidate["qualified_world_collision_object_goal_pose"] = dict(collision_goal)
            if qualification_mode == "frozen_pair":
                seed_evidence = _same_run_seed_evidence(
                    proof,
                    source_candidate_id=candidate_id,
                )
                if seed_evidence is not None:
                    candidate[SAME_RUN_QUALIFICATION_SEED_FIELD] = seed_evidence
        pass_proofs = {str(candidate["id"]): proofs[str(candidate["id"])] for candidate in passed}
        if cache_result:
            self.cache.replace(
                purpose=purpose,
                candidates=passed,
                proofs=pass_proofs,
                scene_epoch=scene_epoch,
                planning_scene_revision=planning_scene_revision,
            )
        counts = Counter(
            str(item.get("reason") or item.get("verdict") or "unknown").lower()
            for item in evidence["results"]
            if item.get("verdict") != "PASS"
        )
        details[key] = passed
        details["candidate_count"] = len(passed)
        details.pop("best_grasp_candidate", None)
        details.pop("active_grasp_candidate", None)
        stage_counts = _funnel_stage_counts(evidence["results"])
        legacy_raw = details.get("raw_candidate_count")
        has_v2_raw = isinstance(details.get("model_raw_candidate_count"), int)
        raw_count = generated if has_v2_raw or not isinstance(legacy_raw, int) else legacy_raw
        artifact = self._write_artifact(evidence)
        public_evidence = (
            _public_qualification_summary(evidence) if artifact is not None else evidence
        )
        details.update(
            {
                "selection_required": bool(passed),
                "model_raw_candidate_count": int(
                    details.get("model_raw_candidate_count", raw_count)
                ),
                "raw_candidate_count": raw_count,
                "diversity_selected_count": len(request_candidates),
                "coordinate_tcp_pass_count": len(compiled_descriptors),
                "workspace_pass_count": stage_counts["workspace"],
                "pure_ik_pass_count": stage_counts["pure_ik"],
                "collision_ik_pass_count": stage_counts["collision_ik"],
                "endpoint_pass_count": stage_counts["endpoint"],
                "endpoint_evaluated_count": sum(
                    item.get("endpoint_evaluated") is True for item in evidence["results"]
                ),
                "endpoint_not_evaluated_count": sum(
                    item.get("verdict") == "NOT_EVALUATED"
                    and item.get("endpoint_evaluated") is False
                    for item in evidence["results"]
                ),
                "full_plan_submitted_count": stage_counts["submitted"],
                "full_plan_pass_count": stage_counts["full_plan"],
                "generated_candidate_count": raw_count if has_v2_raw else generated,
                "submitted_candidate_count": stage_counts["submitted"],
                "rejection_reason_counts": dict(sorted(counts.items())),
                "qualification_evidence": public_evidence,
                "qualification_profile": self.qualification_profile,
                "solver_profile": self.solver_profile,
                "qualification_stop_reason": evidence.get("stop_reason"),
                "qualification_waves": list(evidence.get("waves") or []),
                "qualification_metrics": dict(evidence.get("metrics") or {}),
                "ranking": (
                    "moveit_physical_quality"
                    if self.qualification_profile == "fast_v3" and passed
                    else details.get("ranking") or "score_descending"
                ),
            }
        )
        if artifact is not None:
            details["qualification_artifact"] = artifact
            details.setdefault("artifacts", []).append(artifact)
        result.details = details
        return result

    def _validate_response(self, request: JsonDict, response: object) -> JsonDict:
        expected = {item["candidate_id"]: item for item in request["candidates"]}
        raw_results = response.get("results") if isinstance(response, Mapping) else None
        revision_ok = (
            isinstance(response, Mapping)
            and response.get("schema_version") == request["schema_version"]
            and response.get("planning_scene_revision") == request["planning_scene_revision"]
            and response.get("execution_started") is False
            and isinstance(raw_results, list)
        )
        received = {
            str(item.get("candidate_id") or ""): item
            for item in raw_results or []
            if isinstance(item, Mapping)
        }
        normalized: list[JsonDict] = []
        response_contract_invalid = not revision_ok
        for candidate_id, descriptor in expected.items():
            item = received.get(candidate_id)
            verdict = str(item.get("verdict") or "UNKNOWN").upper() if item else "UNKNOWN"
            reason = str(item.get("reason") or "evidence_missing") if item else "evidence_missing"
            valid_identity = bool(
                revision_ok
                and item
                and item.get("candidate_pose_sha256") == descriptor["candidate_pose_sha256"]
                and item.get("execution_started") is False
                and item.get("qualification_binding_sha256")
                == request["qualification_binding_sha256"]
            )
            stages = item.get("stages") if item else None
            valid_pass = (
                verdict == "PASS"
                and valid_identity
                and isinstance(stages, list)
                and bool(stages)
                and all(_valid_pass_stage(stage) for stage in stages)
            )
            valid_verdict = verdict in {"PASS", "FAIL", "UNKNOWN", "NOT_EVALUATED"}
            invalid_candidate_evidence = bool(
                not revision_ok
                or not item
                or not valid_identity
                or not valid_verdict
                or (verdict == "PASS" and not valid_pass)
            )
            if verdict == "PASS" and not valid_pass:
                verdict, reason = "UNKNOWN", "qualification_evidence_incomplete"
            elif not valid_verdict or not valid_identity:
                verdict, reason = "UNKNOWN", "qualification_evidence_incomplete"
            record = dict(item or {})
            record.update(
                {
                    "candidate_id": candidate_id,
                    "candidate_pose_sha256": descriptor["candidate_pose_sha256"],
                    "verdict": verdict,
                    "reason": reason,
                }
            )
            if invalid_candidate_evidence:
                response_contract_invalid = True
                record["infrastructure_error"] = True
                record["infrastructure_error_detail"] = "qualification_response_contract_invalid"
            compiled_candidate = descriptor.get("candidate")
            if isinstance(compiled_candidate, Mapping) and isinstance(
                compiled_candidate.get("compile_parameters"), Mapping
            ):
                record["compile_parameters"] = dict(compiled_candidate["compile_parameters"])
            else:
                record.pop("compile_parameters", None)
            normalized.append(record)
        evidence: JsonDict = {
            "schema_version": request["schema_version"],
            "purpose": request["purpose"],
            "scene_epoch": request["scene_epoch"],
            "planning_scene_revision": request["planning_scene_revision"],
            "planning": dict(request["planning"]),
            "funnel": dict(request.get("funnel") or {}),
            "qualification_binding_sha256": request["qualification_binding_sha256"],
            "qualification_case_sha256": request["qualification_case_sha256"],
            "execution_started": False,
            "results": normalized,
        }
        if isinstance(response, Mapping):
            for key in (
                "qualification_profile",
                "solver_profile",
                "requested_solver_profile",
                "solver_configuration_id",
                "provider",
                "provider_version",
                "solver_version",
                "robot_model_sha256",
                "scene_sha256",
                "authoritative_scene_sha256",
                "moveit_world_geometry_sha256",
                "moveit_attached_geometry_sha256",
                "moveit_geometry_verified_ids",
                "capability_map_id",
                "waves",
                "l5_attempts",
                "selected_candidate_ids",
                "selected_joint_branches",
                "search_exhaustion",
                "stop_reason",
                "metrics",
                "legality_screening",
                "infrastructure_error",
                "rpc_attempt_count",
                "shadow_fast_v3",
                "artifact_schema_version",
                "qualification_case_sha256",
                "case_id",
            ):
                value = response.get(key)
                if value is not None:
                    evidence[key] = value
        if response_contract_invalid:
            evidence["stop_reason"] = "infrastructure_error"
            evidence["infrastructure_error"] = True
            evidence["infrastructure_error_detail"] = "qualification_response_contract_invalid"
        return evidence

    def _write_artifact(self, evidence: JsonDict) -> JsonDict | None:
        if self.artifact_root is None:
            return None
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        artifact_evidence = dict(evidence)
        if evidence.get("schema_version") == QUALIFICATION_SCHEMA_V3:
            artifact_evidence["artifact_schema_version"] = FAST_ARTIFACT_SCHEMA
        path = self.artifact_root / f"qualification-{_hash(artifact_evidence)[:16]}.json"
        path.write_text(json.dumps(artifact_evidence, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "type": "moveit_candidate_qualification",
            "kind": "json",
            "schema_version": evidence.get("schema_version"),
            "artifact_schema_version": artifact_evidence.get(
                "artifact_schema_version", evidence.get("schema_version")
            ),
            "path": str(path),
            "sha256": _hash(artifact_evidence),
        }


def _public_qualification_summary(evidence: Mapping[str, Any]) -> JsonDict:
    """Return VLM-safe funnel proof metadata; exact per-candidate proof is an artifact."""

    results = evidence.get("results")
    results = results if isinstance(results, list) else []
    verdict_counts = Counter(
        str(item.get("verdict") or "UNKNOWN") for item in results if isinstance(item, Mapping)
    )
    reason_counts = Counter(
        str(item.get("reason") or "unknown")
        for item in results
        if isinstance(item, Mapping) and item.get("verdict") != "PASS"
    )
    return {
        "schema_version": evidence.get("schema_version"),
        "purpose": evidence.get("purpose"),
        "scene_epoch": evidence.get("scene_epoch"),
        "planning_scene_revision": evidence.get("planning_scene_revision"),
        "qualification_binding_sha256": evidence.get("qualification_binding_sha256"),
        "qualification_case_sha256": evidence.get("qualification_case_sha256"),
        "execution_started": False,
        "result_count": len(results),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "qualification_profile": evidence.get("qualification_profile", "legacy"),
        "solver_profile": evidence.get("solver_profile"),
        "solver_configuration_id": evidence.get("solver_configuration_id"),
        "robot_model_sha256": evidence.get("robot_model_sha256"),
        "scene_sha256": evidence.get("scene_sha256"),
        "authoritative_scene_sha256": evidence.get(
            "authoritative_scene_sha256"
        ),
        "moveit_world_geometry_sha256": evidence.get(
            "moveit_world_geometry_sha256"
        ),
        "moveit_attached_geometry_sha256": evidence.get(
            "moveit_attached_geometry_sha256"
        ),
        "moveit_geometry_verified_ids": list(
            evidence.get("moveit_geometry_verified_ids") or []
        ),
        "capability_map_id": evidence.get("capability_map_id"),
        "stop_reason": evidence.get("stop_reason"),
        "search_exhaustion": dict(evidence.get("search_exhaustion") or {}),
        "waves": list(evidence.get("waves") or []),
        "metrics": dict(evidence.get("metrics") or {}),
        "proof_storage": "qualification_artifact",
    }


def _valid_pass_stage(stage: object) -> bool:
    if not isinstance(stage, Mapping):
        return False
    trajectory = stage.get("trajectory")
    return bool(
        stage.get("kinematic_ik") is True
        and stage.get("state_valid") is True
        and stage.get("collision_ik") is True
        and stage.get("plan_only") is True
        and stage.get("execution_started") is False
        and isinstance(stage.get("start_joint_state_sha256"), str)
        and isinstance(stage.get("end_joint_state"), Mapping)
        and isinstance(trajectory, Mapping)
        and isinstance(trajectory.get("point_count"), int)
        and trajectory["point_count"] > 0
    )


def _funnel_stage_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Summarize monotonically decreasing candidate-level funnel stages."""

    def passed(item: Mapping[str, Any], key: str, fallback: Callable[[list], bool]) -> bool:
        value = item.get(key)
        if isinstance(value, bool):
            return value
        stages = item.get("stages")
        return fallback(stages if isinstance(stages, list) else [])

    return {
        "workspace": sum(
            passed(item, "workspace_pass", lambda stages: bool(stages)) for item in results
        ),
        "pure_ik": sum(
            passed(
                item,
                "pure_ik_pass",
                lambda stages: (
                    bool(stages)
                    and all(s.get("kinematic_ik") is True for s in stages if isinstance(s, Mapping))
                ),
            )
            for item in results
        ),
        "collision_ik": sum(
            passed(
                item,
                "collision_ik_pass",
                lambda stages: (
                    bool(stages)
                    and all(s.get("collision_ik") is True for s in stages if isinstance(s, Mapping))
                ),
            )
            for item in results
        ),
        "endpoint": sum(
            passed(
                item,
                "endpoint_pass",
                lambda stages: (
                    bool(stages)
                    and all(s.get("state_valid") is True for s in stages if isinstance(s, Mapping))
                ),
            )
            for item in results
        ),
        "submitted": sum(
            item.get("full_plan_submitted") is True or item.get("verdict") == "PASS"
            for item in results
        ),
        "full_plan": sum(item.get("verdict") == "PASS" for item in results),
    }


def diversify_compiled_candidates(
    descriptors: Sequence[Mapping[str, Any]], *, purpose: str, limit: int
) -> list[JsonDict]:
    """Deterministic SE(3) farthest-first selection after world-EEF compilation."""

    remaining = [dict(item) for item in descriptors]
    if len(remaining) <= limit:
        return remaining
    selected = [remaining.pop(0)]
    while remaining and len(selected) < limit:
        index = max(
            range(len(remaining)),
            key=lambda i: (
                min(_descriptor_distance(remaining[i], prior, purpose) for prior in selected),
                -i,
            ),
        )
        selected.append(remaining.pop(index))
    return selected


def _qualification_rpc_timeout_s(
    candidates: Sequence[Mapping[str, Any]],
    *,
    full_plan_limit: int,
    qualification_profile: str = "legacy",
    fast_seed_count: int = 2,
    recovery_seed_count: int = 6,
    fast_ik_timeout_s: float = 0.05,
    recovery_ik_timeout_s: float = 0.2,
) -> float:
    """Cover screening plus the bounded full-plan tail at the transport layer.

    The per-service IK and state-validity deadlines remain unchanged.  This
    outer deadline must not expire first merely because the diversity pool is
    larger than the full-plan pool.
    """

    stage_counts: list[int] = []
    for item in candidates:
        candidate = item.get("candidate")
        stages = candidate.get("qualification_stages") if isinstance(candidate, Mapping) else None
        stage_counts.append(len(stages) if isinstance(stages, list) and stages else 1)
    max_stage_count = max(stage_counts, default=1)
    if qualification_profile in {"fast_v3", "shadow"}:
        # Sixty seconds is a performance objective, never a transport cutoff.
        # Cover pure IK + possible collision rescue + validity for both the
        # fast and recovery seed budgets without assuming any concurrency.
        per_stage_screening_s = 2.0 * (
            max(0, fast_seed_count) * (max(0.0, fast_ik_timeout_s) + STATE_VALIDITY_TIMEOUT_S)
            + max(0, recovery_seed_count)
            * (max(0.0, recovery_ik_timeout_s) + STATE_VALIDITY_TIMEOUT_S)
        )
        screening_budget_s = len(candidates) * max_stage_count * per_stage_screening_s
        # Every L4 candidate may fail L5 once in the fast layer and again on
        # a different recovery branch. Shadow's bounded legacy tail fits
        # inside this same conservative envelope.
        planning_budget_s = len(candidates) * 2 * max_stage_count * PLANNING_TIME_S
    else:
        screening_budget_s = (
            len(candidates) * 2.0 * (KINEMATIC_IK_TIMEOUT_S + STATE_VALIDITY_TIMEOUT_S)
        )
        planning_budget_s = max(0, int(full_plan_limit)) * max_stage_count * PLANNING_TIME_S
    return screening_budget_s + planning_budget_s + QUALIFICATION_RPC_GRACE_S


def _goal_prebind_rpc_timeout_s(
    candidates: Sequence[Mapping[str, Any]],
) -> float:
    """Cover serialized worker latency and complete prebind evidence transfer.

    Goal legality itself is cheap and performs no IK or L5 planning.  The
    request still shares one serialized simulator worker with operator GUI
    captures and returns a record for every frozen goal.  Scaling this outer
    deadline by frontier size prevents load-dependent transport timeouts from
    being mislabeled as reachability failures while retaining a finite bound.
    """

    return (
        QUALIFICATION_RPC_GRACE_S
        + len(candidates) * GOAL_PREBIND_RPC_PER_CANDIDATE_S
    )


def _deduplicate_se3_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    round_index: int,
) -> list[JsonDict]:
    """Deduplicate absolute object goals within one inference round."""

    merged: list[JsonDict] = []
    fingerprints: set[str] = set()
    ids: set[str] = set()
    for index, value in enumerate(candidates):
        candidate = dict(value)
        transform = (
            candidate.get("object_goal_world")
            or candidate.get("object_goal_pose")
            or candidate.get("object_placement_transform")
            or candidate.get("object_goal_transform")
        )
        fingerprint = _hash(_rounded_transform(transform))
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        candidate_id = str(candidate.get("id") or "")
        if not candidate_id or candidate_id in ids:
            candidate_id = f"anyplace_r{round_index}_{index:03d}"
            candidate["id"] = candidate_id
            candidate["inference_round"] = round_index
            candidate["provenance"] = "merged_inference_round"
        ids.add(candidate_id)
        merged.append(candidate)
    return merged


def _preserve_candidate_pool(
    candidates: Sequence[Mapping[str, Any]],
    *,
    round_index: int,
) -> list[JsonDict]:
    """Preserve every model result while assigning deterministic unique IDs."""

    preserved: list[JsonDict] = []
    ids: set[str] = set()
    for index, value in enumerate(candidates):
        candidate = dict(value)
        candidate_id = str(candidate.get("id") or "")
        if not candidate_id or candidate_id in ids:
            candidate_id = f"anyplace_r{round_index}_{index:03d}"
            candidate["id"] = candidate_id
            candidate["inference_round"] = round_index
            candidate["provenance"] = "complete_inference_round"
        ids.add(candidate_id)
        preserved.append(candidate)
    return preserved


def _rounded_transform(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _rounded_transform(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_rounded_transform(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _pose_with_quaternion(pose: Mapping[str, Any]) -> JsonDict:
    result = dict(pose)
    if isinstance(result.get("quat_xyzw"), list):
        return result
    rotation = result.get("rotation_matrix")
    if not (
        isinstance(rotation, list)
        and len(rotation) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in rotation)
    ):
        raise ValueError("source-return pose rotation is missing")
    m = [[float(value) for value in row] for row in rotation]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = [
            (m[2][1] - m[1][2]) / scale,
            (m[0][2] - m[2][0]) / scale,
            (m[1][0] - m[0][1]) / scale,
            0.25 * scale,
        ]
    else:
        index = max(range(3), key=lambda item: m[item][item])
        if index == 0:
            scale = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
            quat = [
                0.25 * scale,
                (m[0][1] + m[1][0]) / scale,
                (m[0][2] + m[2][0]) / scale,
                (m[2][1] - m[1][2]) / scale,
            ]
        elif index == 1:
            scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
            quat = [
                (m[0][1] + m[1][0]) / scale,
                0.25 * scale,
                (m[1][2] + m[2][1]) / scale,
                (m[0][2] - m[2][0]) / scale,
            ]
        else:
            scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
            quat = [
                (m[0][2] + m[2][0]) / scale,
                (m[1][2] + m[2][1]) / scale,
                0.25 * scale,
                (m[1][0] - m[0][1]) / scale,
            ]
    result["quat_xyzw"] = quat
    return result


def _descriptor_distance(a: Mapping[str, Any], b: Mapping[str, Any], purpose: str) -> float:
    ca = a.get("candidate") if isinstance(a.get("candidate"), Mapping) else {}
    cb = b.get("candidate") if isinstance(b.get("candidate"), Mapping) else {}
    sa = ca.get("qualification_stages") if isinstance(ca, Mapping) else []
    sb = cb.get("qualification_stages") if isinstance(cb, Mapping) else []
    pa = sa[-1] if isinstance(sa, list) and sa and isinstance(sa[-1], Mapping) else {}
    pb = sb[-1] if isinstance(sb, list) and sb and isinstance(sb[-1], Mapping) else {}
    xa, xb = pa.get("xyz"), pb.get("xyz")
    translation = 0.0
    if isinstance(xa, list) and isinstance(xb, list) and len(xa) == len(xb) == 3:
        translation = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(xa, xb)))
    qa, qb = pa.get("quat_xyzw"), pb.get("quat_xyzw")
    angle = 0.0
    if isinstance(qa, list) and isinstance(qb, list) and len(qa) == len(qb) == 4:
        dot = min(1.0, abs(sum(float(x) * float(y) for x, y in zip(qa, qb))))
        angle = 2.0 * math.acos(dot)
    source_a = str(
        ca.get("candidate_source")
        or ca.get("source_model")
        or ca.get("source")
        or ca.get("source_branch")
        or ca.get("backend")
        or ""
    )
    source_b = str(
        cb.get("candidate_source")
        or cb.get("source_model")
        or cb.get("source")
        or cb.get("source_branch")
        or cb.get("backend")
        or ""
    )
    source_bonus = 1.0 if source_a and source_b and source_a != source_b else 0.0
    object_bonus = 0.0
    if purpose == "placement":
        object_bonus = (
            0.25 if ca.get("stable_contact_face") != cb.get("stable_contact_face") else 0.0
        )
    return translation * 10.0 + angle + source_bonus + object_bonus


def private_qualification_rpc(
    transport: Any,
    *,
    handle_provider: Callable[[], str],
    session_id_provider: Callable[[], str],
) -> Callable[[str, JsonDict, float], JsonDict]:
    """Bind session identity without exposing this RPC in the AgentTool registry."""

    def call(name: str, request: JsonDict, timeout_s: float) -> JsonDict:
        from agent.tools.sim_mcp import call_read_only_mcp_tool_with_retry

        arguments = dict(request)
        arguments["handle"] = handle_provider()
        arguments["session_id"] = session_id_provider()
        return call_read_only_mcp_tool_with_retry(
            transport,
            name,
            arguments,
            timeout_s=timeout_s,
            first_attempt_timeout_s=QUALIFICATION_RPC_FIRST_ACK_TIMEOUT_S,
        )

    return call


@dataclass(slots=True)
class MoveItQualificationEngine:
    """Server-side staged IK/collision/plan-only qualification core.

    The injected callbacks are intentionally small so ROS bindings can adapt
    ``/compute_ik``, ``/check_state_validity`` and ``MoveGroup`` messages while
    unit tests can exercise all fail-closed branches without ROS.
    """

    current_joint_state: Callable[[], Mapping[str, Any]]
    scene_revision: Callable[[], int]
    compute_ik: Callable[[Mapping[str, Any], Mapping[str, Any], bool], Mapping[str, Any]]
    check_state_validity: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    plan_only: Callable[[Mapping[str, Any], Mapping[str, Any], float, int], Mapping[str, Any]]
    workspace_filter: Callable[[Mapping[str, Any]], bool] | None = None
    clone_scene: Callable[[], Any] | None = None
    apply_scene_transition: Callable[[Any, str, Mapping[str, Any]], Mapping[str, Any]] | None = None

    service_health_check: Callable[[], bool] | None = None
    set_solver_mode: Callable[[str], Mapping[str, Any]] | None = None
    _active_qualification_binding_sha256: str = field(
        init=False,
        default="",
        repr=False,
    )

    def qualify(self, request: Mapping[str, Any]) -> JsonDict:
        funnel = request.get("funnel")
        funnel = funnel if isinstance(funnel, Mapping) else {}
        # The engine instance serves one private request. Carry its immutable
        # proof binding down to L5 so the ROS adapter can cache only the exact
        # trajectory that later appears on the public executable pose.
        self._active_qualification_binding_sha256 = str(
            request.get("qualification_binding_sha256") or ""
        )
        profile = str(funnel.get("qualification_profile") or "legacy")
        if profile == "legacy":
            return self._qualify_legacy(request)
        if profile == "fast_v3":
            return self._qualify_fast_v3(request)
        if profile == "shadow":
            # Legacy remains authoritative during rollout.  Both paths consume
            # only plan-only services and the fast evidence remains private.
            legacy_request = dict(request)
            shadow_ids = funnel.get("shadow_legacy_candidate_ids")
            shadow_ids = (
                {str(value) for value in shadow_ids} if isinstance(shadow_ids, list) else None
            )
            all_candidates = request.get("candidates")
            all_candidates = all_candidates if isinstance(all_candidates, list) else []
            if shadow_ids is not None:
                legacy_request["candidates"] = [
                    item
                    for item in all_candidates
                    if isinstance(item, Mapping)
                    and str(item.get("candidate_id") or "") in shadow_ids
                ]
            legacy = self._qualify_legacy(legacy_request)
            legacy_by_id = {
                str(item.get("candidate_id") or ""): item
                for item in legacy.get("results", [])
                if isinstance(item, Mapping)
            }
            binding = str(request.get("qualification_binding_sha256") or "")
            legacy["results"] = [
                legacy_by_id.get(
                    str(item.get("candidate_id") or ""),
                    {
                        "candidate_id": str(item.get("candidate_id") or ""),
                        "candidate_pose_sha256": str(item.get("candidate_pose_sha256") or ""),
                        "qualification_binding_sha256": binding,
                        "execution_started": False,
                        "verdict": "NOT_EVALUATED",
                        "reason": "shadow_legacy_diversity_not_selected",
                        "endpoint_evaluated": False,
                        "full_plan_submitted": False,
                        "stages": [],
                    },
                )
                for item in all_candidates
                if isinstance(item, Mapping)
            ]
            fast_request = dict(request)
            fast_funnel = dict(funnel)
            fast_funnel["qualification_profile"] = "fast_v3"
            fast_request["funnel"] = fast_funnel
            fast = self._qualify_fast_v3(fast_request)
            legacy["qualification_profile"] = "shadow"
            legacy["shadow_fast_v3"] = {
                key: fast.get(key)
                for key in (
                    "qualification_profile",
                    "solver_profile",
                    "requested_solver_profile",
                    "solver_configuration_id",
                    "provider",
                    "provider_version",
                    "solver_version",
                    "robot_model_sha256",
                    "scene_sha256",
                    "authoritative_scene_sha256",
                    "moveit_world_geometry_sha256",
                    "moveit_attached_geometry_sha256",
                    "moveit_geometry_verified_ids",
                    "capability_map_id",
                    "artifact_schema_version",
                    "qualification_case_sha256",
                    "case_id",
                    "selected_candidate_ids",
                    "stop_reason",
                    "waves",
                    "l5_attempts",
                    "metrics",
                    "first_l5_pass_s",
                    "infrastructure_error",
                    "results",
                )
            }
            return legacy
        raise ValueError("invalid qualification profile")

    def _qualify_legacy(self, request: Mapping[str, Any]) -> JsonDict:
        revision = request.get("planning_scene_revision")
        candidates = request.get("candidates")
        if (
            request.get("schema_version") not in SUPPORTED_QUALIFICATION_SCHEMAS
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or not isinstance(candidates, list)
        ):
            raise ValueError("invalid qualification request")
        planning = request.get("planning")
        planning = planning if isinstance(planning, Mapping) else {}
        funnel = request.get("funnel")
        funnel = funnel if isinstance(funnel, Mapping) else planning
        seed_count = int(funnel.get("ik_seed_count", DEFAULT_MOVEIT_IK_SEED_COUNT))
        full_plan_limit = int(funnel.get("full_plan_limit", DEFAULT_GRASP_FULL_PLAN_LIMIT))
        screening_mode = str(funnel.get("screening_mode") or "exhaustive")
        if screening_mode not in {"exhaustive", PROGRESSIVE_SCREENING_MODE}:
            raise ValueError("invalid qualification screening mode")
        progressive_endpoint_target: int | None = None
        if screening_mode == PROGRESSIVE_SCREENING_MODE:
            raw_target = funnel.get("endpoint_pass_target")
            if (
                isinstance(raw_target, bool)
                or not isinstance(raw_target, int)
                or raw_target <= 0
                or raw_target != full_plan_limit
            ):
                raise ValueError("progressive endpoint target must equal full-plan capacity")
            progressive_endpoint_target = raw_target
        source = request.get("source") if isinstance(request.get("source"), Mapping) else {}
        screened: list[JsonDict] = []
        ik_reservoir: list[Mapping[str, Any]] = []
        workspace_prechecks: list[JsonDict] | None = None
        if progressive_endpoint_target is not None:
            # L1 compilation happened at the host boundary.  Run the entire
            # batch through the deterministic L2 workspace/structure gate
            # before progressively spending multi-seed L3/L4 work.
            workspace_prechecks = [self._workspace_precheck(item) for item in candidates]
        endpoint_passes = 0
        for index, item in enumerate(candidates):
            workspace_precheck = (
                workspace_prechecks[index] if workspace_prechecks is not None else None
            )
            if workspace_precheck is not None and workspace_precheck.get("verdict") != "PASS":
                screened.append(workspace_precheck)
                continue
            if (
                progressive_endpoint_target is not None
                and endpoint_passes >= progressive_endpoint_target
            ):
                screened.append(
                    {
                        **dict(workspace_precheck or {}),
                        "verdict": "NOT_EVALUATED",
                        "reason": PROGRESSIVE_NOT_EVALUATED_REASON,
                        "endpoint_evaluated": False,
                        "full_plan_submitted": False,
                    }
                )
                continue
            candidate_source = dict(source)
            candidate_source["successful_ik_reservoir"] = [
                dict(value) for value in ik_reservoir[-seed_count:]
            ]
            screen = self._screen_candidate(
                item,
                revision,
                seed_count=seed_count,
                source=candidate_source,
                workspace_precheck=workspace_precheck,
            )
            screened.append(screen)
            if screen.get("endpoint_pass") is True:
                endpoint_passes += 1
                stages = screen.get("stages")
                if isinstance(stages, list) and stages:
                    state = stages[-1].get("end_joint_state")
                    if isinstance(state, Mapping):
                        ik_reservoir.append(dict(state))
        results: list[JsonDict] = []
        submitted = 0
        for descriptor, screen in zip(candidates, screened, strict=True):
            if screen.get("endpoint_pass") is not True:
                results.append(screen)
                continue
            if submitted >= full_plan_limit:
                results.append(
                    {
                        **screen,
                        "verdict": "NOT_EVALUATED",
                        "reason": "full_plan_limit_not_submitted",
                        "full_plan_submitted": False,
                    }
                )
                continue
            submitted += 1
            planned = self._plan_candidate(descriptor, revision, screen=screen)
            planned["full_plan_submitted"] = True
            results.append(planned)
        binding = str(request.get("qualification_binding_sha256") or "")
        for result in results:
            result["qualification_binding_sha256"] = binding
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": revision,
            "execution_started": False,
            "qualification_profile": "legacy",
            "solver_profile": str(
                source.get("solver_profile") or funnel.get("solver_profile") or "kdl_legacy"
            ),
            "solver_version": str(source.get("solver_version") or "unknown"),
            "solver_configuration_id": str(
                source.get("solver_profile") or funnel.get("solver_profile") or "kdl_legacy"
            ),
            "robot_model_sha256": str(source.get("robot_model_sha256") or ""),
            "scene_sha256": str(source.get("scene_sha256") or ""),
            "authoritative_scene_sha256": str(
                source.get("authoritative_scene_sha256") or ""
            ),
            "moveit_world_geometry_sha256": str(
                source.get("moveit_world_geometry_sha256") or ""
            ),
            "moveit_attached_geometry_sha256": str(
                source.get("moveit_attached_geometry_sha256") or ""
            ),
            "moveit_geometry_verified_ids": [
                str(value)
                for value in source.get("moveit_geometry_verified_ids") or []
            ],
            "qualification_case_sha256": str(request.get("qualification_case_sha256") or ""),
            "case_id": str(request.get("qualification_case_sha256") or ""),
            "results": results,
        }

    def _qualify_goal_prebind(self, request: Mapping[str, Any]) -> JsonDict:
        """Resolve one complete placement-goal legality frontier, without IK."""

        revision = int(request["planning_scene_revision"])
        candidates = request["candidates"]
        binding = str(request.get("qualification_binding_sha256") or "")
        try:
            scene = self.clone_scene() if self.clone_scene is not None else None
        except Exception as exc:  # noqa: BLE001 - PlanningScene service boundary.
            return {
                "schema_version": request["schema_version"],
                "planning_scene_revision": revision,
                "execution_started": False,
                "infrastructure_error": True,
                "stop_reason": f"placement_goal_scene_clone_failed:{type(exc).__name__}",
                "results": [],
            }
        if not isinstance(scene, Mapping):
            return {
                "schema_version": request["schema_version"],
                "planning_scene_revision": revision,
                "execution_started": False,
                "infrastructure_error": True,
                "stop_reason": "placement_goal_scene_clone_unavailable",
                "results": [],
            }
        scene_revision = scene.get("revision")
        if (
            isinstance(scene_revision, int)
            and not isinstance(scene_revision, bool)
            and scene_revision != revision
        ):
            return {
                "schema_version": request["schema_version"],
                "planning_scene_revision": revision,
                "execution_started": False,
                "infrastructure_error": True,
                "stop_reason": "placement_goal_scene_revision_drift",
                "results": [],
            }
        rows: list[JsonDict] = []
        selected: list[str] = []
        started = time.monotonic()
        for raw_descriptor in candidates:
            descriptor = (
                json.loads(json.dumps(raw_descriptor))
                if isinstance(raw_descriptor, Mapping)
                else {}
            )
            candidate_id = str(descriptor.get("candidate_id") or "")
            legality = evaluate_placement_goal_legality(
                descriptor,
                scene=scene,
            )
            bind_qualified_placement_goal(descriptor, legality)
            verdict = str(legality.get("verdict") or "UNKNOWN").upper()
            row: JsonDict = {
                "candidate_id": candidate_id,
                "candidate_pose_sha256": str(
                    descriptor.get("candidate_pose_sha256") or ""
                ),
                "verdict": verdict,
                "reason": str(legality.get("reason") or "goal_legality_unknown"),
                "goal_legality": legality,
                "execution_started": False,
                "qualification_binding_sha256": binding,
            }
            candidate = descriptor.get("candidate")
            if verdict == "PASS" and isinstance(candidate, Mapping):
                row["prebound_candidate"] = dict(candidate)
                selected.append(candidate_id)
            rows.append(row)
        infrastructure_error = any(
            isinstance(row.get("goal_legality"), Mapping)
            and row["goal_legality"].get("infrastructure_error") is True
            for row in rows
        )
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": revision,
            "execution_started": False,
            "infrastructure_error": infrastructure_error,
            "stop_reason": (
                "infrastructure_error"
                if infrastructure_error
                else "goal_legality_barrier_complete"
            ),
            "selected_candidate_ids": selected,
            "results": rows,
            "metrics": {
                "generated_count": len(candidates),
                "goal_legality_unique_count": len(candidates),
                "goal_legality_pass_count": len(selected),
                "goal_legality_reject_count": len(candidates) - len(selected),
                "goal_legality_elapsed_s": time.monotonic() - started,
            },
        }

    def _qualify_fast_v3(self, request: Mapping[str, Any]) -> JsonDict:
        """Run deterministic concurrent waves and a serial plan-only tail."""

        revision = request.get("planning_scene_revision")
        candidates = request.get("candidates")
        if (
            request.get("schema_version") not in SUPPORTED_QUALIFICATION_SCHEMAS
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or not isinstance(candidates, list)
        ):
            raise ValueError("invalid fast_v3 qualification request")
        funnel = request.get("funnel")
        funnel = funnel if isinstance(funnel, Mapping) else {}
        purpose = str(request.get("purpose") or "grasp")
        if purpose not in {"grasp", "placement", "observation"}:
            raise ValueError("invalid fast_v3 qualification purpose")
        if (
            purpose == "placement"
            and str(funnel.get("qualification_mode") or "") == "goal_prebind"
        ):
            return self._qualify_goal_prebind(request)
        beam_width = int(funnel.get("beam_width", 2))
        fast_seed_count = int(funnel.get("fast_seed_count", 2))
        recovery_seed_count = int(funnel.get("recovery_seed_count", 6))
        if (
            beam_width < 1
            or beam_width > fast_seed_count
            or fast_seed_count + recovery_seed_count
            != int(funnel.get("ik_seed_count", DEFAULT_MOVEIT_IK_SEED_COUNT))
        ):
            raise ValueError("invalid fast_v3 Beam/seed budget")
        grasp_waves = self._integer_waves(
            funnel.get("grasp_waves"), DEFAULT_GRASP_WAVES
        )
        placement_waves = self._integer_waves(funnel.get("placement_waves"), (4, 8, 16, 32, 96))
        observation_waves = self._integer_waves(funnel.get("observation_waves"), (4, 8, 16, 24))
        max_ik = max(1, int(funnel.get("max_ik_concurrency", 8)))
        max_validity = max(1, int(funnel.get("max_state_validity_concurrency", 8)))
        fast_timeout_s = float(funnel.get("fast_ik_timeout_s", 0.05))
        recovery_timeout_s = float(funnel.get("recovery_ik_timeout_s", 0.2))
        if not 0.0 < fast_timeout_s <= recovery_timeout_s:
            raise ValueError("invalid fast_v3 IK timeouts")
        source = request.get("source")
        source = dict(source) if isinstance(source, Mapping) else {}
        start_time = time.monotonic()
        try:
            current_state, _, _ = self._call_fast_service(
                lambda: self.current_joint_state(), required_boolean=None
            )
        except _QualificationInfrastructureError as exc:
            return self._fast_infrastructure_response(
                request, candidates, revision, reason=str(exc)
            )
        requested_solver_profile = str(funnel.get("solver_profile") or "auto")
        solver_profile = str(current_state.get("solver_profile") or requested_solver_profile)
        if requested_solver_profile != "auto" and solver_profile != requested_solver_profile:
            return self._fast_configuration_response(
                request,
                candidates,
                revision,
                reason="solver_profile_mismatch",
                error=(f"requested {requested_solver_profile}, ROS configured {solver_profile}"),
                robot_hash=str(current_state.get("robot_model_sha256") or ""),
                scene_hash=str(current_state.get("scene_sha256") or ""),
            )
        if solver_profile == "pick_ik_local":
            if self.set_solver_mode is None:
                return self._fast_infrastructure_response(
                    request,
                    candidates,
                    revision,
                    reason="pick_ik_mode_switch_unavailable",
                )
            try:
                self._call_fast_service(
                    lambda: self.set_solver_mode("local"),
                    required_boolean="ok",
                )
            except _QualificationInfrastructureError as exc:
                return self._fast_infrastructure_response(
                    request, candidates, revision, reason=str(exc)
                )
        robot_hash = str(
            source.get("robot_model_sha256")
            or current_state.get("robot_model_sha256")
            or _hash(
                {
                    "names": current_state.get("names"),
                    "joint_limits": source.get("joint_limits") or current_state.get("joint_limits"),
                    "planning_group": source.get("planning_group"),
                    "tcp": source.get("tcp"),
                    "gripper": source.get("gripper"),
                }
            )
        )
        scene_hash = str(
            source.get("scene_sha256")
            or current_state.get("scene_sha256")
            or _hash(
                {
                    "planning_scene_revision": revision,
                    "scene_identity": source.get("scene_identity"),
                }
            )
        )
        authoritative_scene_hash = str(
            source.get("authoritative_scene_sha256")
            or current_state.get("authoritative_scene_sha256")
            or ""
        )
        moveit_world_geometry_hash = str(
            source.get("moveit_world_geometry_sha256")
            or current_state.get("moveit_world_geometry_sha256")
            or ""
        )
        moveit_attached_geometry_hash = str(
            source.get("moveit_attached_geometry_sha256")
            or current_state.get("moveit_attached_geometry_sha256")
            or ""
        )
        raw_verified_ids = source.get("moveit_geometry_verified_ids")
        if not isinstance(raw_verified_ids, (list, tuple, set)):
            raw_verified_ids = current_state.get("moveit_geometry_verified_ids")
        moveit_geometry_verified_ids = sorted(
            {
                str(value)
                for value in (
                    raw_verified_ids
                    if isinstance(raw_verified_ids, (list, tuple, set))
                    else []
                )
            }
        )
        if current_state.get("jacobian_quality_available") is False:
            return self._fast_configuration_response(
                request,
                candidates,
                revision,
                reason="jacobian_quality_unavailable",
                error=str(
                    current_state.get("jacobian_quality_error")
                    or "expanded-URDF Jacobian evaluator is unavailable"
                ),
                robot_hash=robot_hash,
                scene_hash=scene_hash,
            )
        requested_map_id = str(funnel.get("capability_map_id") or "")
        capability_map: SparseCapabilityMap | None = None
        capability_map_status = "not_configured"
        map_load_error = source.get("capability_map_load_error")
        if requested_map_id and map_load_error:
            return self._fast_configuration_response(
                request,
                candidates,
                revision,
                reason="capability_map_unavailable",
                error=str(map_load_error),
                robot_hash=robot_hash,
                scene_hash=scene_hash,
            )
        map_payload = source.get("capability_map")
        if isinstance(map_payload, Mapping):
            try:
                capability_map = SparseCapabilityMap.from_dict(
                    map_payload,
                    expected_map_id=requested_map_id,
                    expected_robot_model_sha256=robot_hash,
                )
            except ValueError as exc:
                return self._fast_configuration_response(
                    request,
                    candidates,
                    revision,
                    reason="capability_map_invalid",
                    error=str(exc),
                    robot_hash=robot_hash,
                    scene_hash=scene_hash,
                )
            capability_map_status = "loaded"
        elif requested_map_id:
            return self._fast_configuration_response(
                request,
                candidates,
                revision,
                reason="capability_map_unavailable",
                error="configured capability map payload was not loaded by ROS",
                robot_hash=robot_hash,
                scene_hash=scene_hash,
            )

        try:
            descriptors, prechecks, schedulable, legality_metrics, legality_scene = (
                self._fast_legality_prechecks(
                    candidates,
                    purpose=purpose,
                    revision=revision,
                )
            )
        except _QualificationInfrastructureError as exc:
            return self._fast_infrastructure_response(
                request, candidates, revision, reason=str(exc)
            )
        descriptor_by_id: dict[str, JsonDict] = {}
        for descriptor in descriptors:
            candidate_id = str(descriptor.get("candidate_id") or "")
            descriptor_by_id[candidate_id] = descriptor
        waves = schedule_candidate_waves(
            schedulable,
            purpose=purpose,
            grasp_waves=grasp_waves,
            placement_waves=placement_waves,
            observation_waves=observation_waves,
            capability_map=capability_map,
        )
        for wave in waves:
            for descriptor in wave.candidates:
                candidate_id = str(descriptor.get("candidate_id") or "")
                descriptor_by_id[candidate_id] = dict(descriptor)
                prechecks[candidate_id]["se3_cluster_id"] = descriptor.get("se3_cluster_id")
                prechecks[candidate_id]["grasp_symmetry_family_id"] = descriptor.get(
                    "grasp_symmetry_family_id"
                )
                prechecks[candidate_id]["capability_score"] = dict(
                    descriptor.get("capability_score") or {}
                )

        ik_gate = _ServiceConcurrencyGate(max_ik)
        validity_gate = _ServiceConcurrencyGate(max_validity)
        latest: dict[str, JsonDict] = dict(prechecks)
        screening_history: dict[str, list[JsonDict]] = {}
        batch_cache: list[JsonDict] = []
        l5_passes: list[JsonDict] = []
        joint_branch_fallback_passes: list[JsonDict] = []
        planned_ids: set[str] = set()
        terminal_deterministic_failure_ids: set[str] = set()
        l5_attempts: list[JsonDict] = []
        wave_evidence: list[JsonDict] = []
        recovery_waves: list[CandidateWave] = []
        pair_legality_cache: dict[tuple[str, str], tuple[str, JsonDict]] = {}
        infrastructure_error = next(
            (
                str(precheck.get("reason") or "legality_precheck_error")
                for precheck in prechecks.values()
                if precheck.get("infrastructure_error") is True
            ),
            "",
        )
        stop_reason = "candidate_pool_exhausted"
        first_l5_pass_elapsed_s: float | None = None
        target = int(funnel.get("l5_pass_target", 2 if purpose == "grasp" else 1))
        minimum_target = int(funnel.get("l5_min_pass_target", target))
        if target <= 0 or minimum_target <= 0 or minimum_target > target:
            raise ValueError("invalid fast_v3 L5 pass targets")
        qualification_mode = str(funnel.get("qualification_mode") or "")

        def pass_target_reached(required: int) -> bool:
            if len(l5_passes) < required:
                return False
            if purpose == "observation":
                return True
            if purpose == "placement" and qualification_mode == "frozen_pair":
                grasp_ids = {
                    str(item.get("source_grasp_id") or item.get("candidate_id") or "")
                    for item in l5_passes
                    if str(item.get("source_grasp_id") or item.get("candidate_id") or "")
                }
                return len(grasp_ids) >= required
            if purpose == "placement":
                return True
            clusters = {str(item.get("se3_cluster_id") or "") for item in l5_passes}
            families = {str(item.get("grasp_symmetry_family_id") or "") for item in l5_passes}
            return len(clusters) >= required and len(families) >= required

        def target_reached() -> bool:
            return pass_target_reached(target)

        def minimum_target_reached() -> bool:
            return pass_target_reached(minimum_target)

        def acceptable_target_reached() -> bool:
            return target_reached() or minimum_target_reached()

        def run_wave(wave: CandidateWave, *, recovery: bool) -> bool:
            nonlocal infrastructure_error, stop_reason, first_l5_pass_elapsed_s
            wave_started = time.monotonic()
            deep_candidates = tuple(wave.candidates)
            pair_rejections: list[JsonDict] = []
            if purpose == "placement":
                deep_candidates, pair_rejections = self._apply_fast_pair_legality_wave(
                    deep_candidates,
                    prechecks=prechecks,
                    scene=legality_scene,
                    pair_cache=pair_legality_cache,
                    metrics=legality_metrics,
                )
            completed: list[JsonDict] = [dict(item) for item in pair_rejections]
            if deep_candidates:
                workers = max(
                    1,
                    min(len(deep_candidates), max(max_ik, max_validity)),
                )
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="openeta-qualification-v3",
                ) as executor:
                    futures = {
                        executor.submit(
                            self._screen_fast_candidate,
                            descriptor,
                            revision,
                            legality_precheck=prechecks[str(descriptor.get("candidate_id") or "")],
                            source=source,
                            current_state=current_state,
                            batch_cache=tuple(batch_cache),
                            beam_width=beam_width,
                            seed_count=(recovery_seed_count if recovery else fast_seed_count),
                            timeout_s=(recovery_timeout_s if recovery else fast_timeout_s),
                            recovery=recovery,
                            solver_profile=(
                                "pick_ik_global"
                                if recovery and solver_profile == "pick_ik_local"
                                else solver_profile
                            ),
                            ik_gate=ik_gate,
                            validity_gate=validity_gate,
                        ): descriptor
                        for descriptor in deep_candidates
                    }
                    for future in as_completed(futures):
                        descriptor = futures[future]
                        try:
                            screened = future.result()
                        except Exception as exc:  # noqa: BLE001 - worker boundary.
                            screened = {
                                **prechecks[str(descriptor.get("candidate_id") or "")],
                                "verdict": "UNKNOWN",
                                "reason": "qualification_worker_error",
                                "infrastructure_error": True,
                                "error_type": type(exc).__name__,
                            }
                        completed.append(screened)
            completed.sort(key=lambda item: int(item.get("fixed_candidate_index", 0)))
            for screened in completed:
                candidate_id = str(screened.get("candidate_id") or "")
                screened["wave_index"] = wave.wave_index
                screened["recovery_layer"] = recovery
                latest[candidate_id] = screened
                screening_history.setdefault(candidate_id, []).append(dict(screened))
                if screened.get("infrastructure_error") is True and not infrastructure_error:
                    infrastructure_error = str(screened.get("reason") or "infrastructure_error")
            # The cache is isolated within one qualification run and updated
            # only after the completion barrier in fixed candidate order.
            for screened in completed:
                if screened.get("endpoint_pass") is not True:
                    continue
                stages = screened.get("stages")
                if not isinstance(stages, list) or not stages:
                    continue
                for stage in stages:
                    solutions = stage.get("beam_solutions") if isinstance(stage, Mapping) else None
                    for solution in solutions if isinstance(solutions, list) else []:
                        state = (
                            solution.get("joint_state") if isinstance(solution, Mapping) else None
                        )
                        if isinstance(state, Mapping):
                            cached = dict(state)
                            cached["seed_source"] = "batch_cache"
                            batch_cache.append(cached)
            submitted_this_wave = 0
            passed_this_wave = 0
            if not infrastructure_error:
                ranked = sorted(
                    (
                        item
                        for item in completed
                        if item.get("endpoint_pass") is True
                        and (recovery or str(item.get("candidate_id") or "") not in planned_ids)
                    ),
                    key=candidate_physical_quality_key,
                )
                if qualification_mode == "frozen_pair":
                    ranked = frozen_pair_l5_submission_order(
                        ranked,
                        prior_attempts=l5_attempts,
                    )
                for screened in ranked:
                    candidate_id = str(screened.get("candidate_id") or "")
                    source_grasp_id = str(screened.get("source_grasp_id") or "")
                    if qualification_mode == "frozen_pair" and source_grasp_id:
                        passed_grasp_ids = {
                            str(item.get("source_grasp_id") or "")
                            for item in l5_passes
                            if str(item.get("source_grasp_id") or "")
                        }
                        if source_grasp_id in passed_grasp_ids and len(passed_grasp_ids) < target:
                            # Another goal for an already-proven grasp cannot
                            # fill the independent-backup slot.  Preserve it in
                            # the frozen frontier instead of spending L5 now.
                            continue
                    descriptor = descriptor_by_id[candidate_id]
                    planned_ids.add(candidate_id)
                    submitted_this_wave += 1
                    l5_started = time.monotonic()
                    planned, retry_count = self._plan_fast_candidate(
                        descriptor, revision, screen=screened
                    )
                    l5_elapsed_s = time.monotonic() - l5_started
                    planned["wave_index"] = wave.wave_index
                    planned["recovery_layer"] = recovery
                    latest[candidate_id] = planned
                    l5_attempts.append(
                        {
                            "attempt_index": len(l5_attempts),
                            "candidate_id": candidate_id,
                            "fixed_candidate_index": planned.get("fixed_candidate_index"),
                            "source_grasp_id": planned.get("source_grasp_id"),
                            "source_object_goal_id": planned.get("source_object_goal_id"),
                            "se3_cluster_id": planned.get("se3_cluster_id"),
                            "grasp_symmetry_family_id": planned.get("grasp_symmetry_family_id"),
                            "wave_index": wave.wave_index,
                            "recovery_layer": recovery,
                            "retry_count": retry_count,
                            "elapsed_s": l5_elapsed_s,
                            "verdict": planned.get("verdict"),
                            "reason": planned.get("reason"),
                            "joint_branch_index": 0,
                            "joint_branch_joint_state_sha256": _selected_ik_state_sha256(planned),
                        }
                    )
                    if planned.get("infrastructure_error") is True:
                        infrastructure_error = str(
                            planned.get("reason") or "plan_only_service_error"
                        )
                        break
                    terminal_evidence = (
                        planned.get("stages", [{}])[-1].get("terminal_gripper_state_validity")
                        if isinstance(planned.get("stages"), list)
                        and planned.get("stages")
                        and isinstance(planned.get("stages", [{}])[-1], Mapping)
                        else None
                    )
                    if (
                        planned.get("reason") == "terminal_gripper_state_invalid"
                        and isinstance(terminal_evidence, Mapping)
                        and (
                            terminal_evidence.get("seed_independent_static_collision") is True
                            or terminal_evidence.get("seed_independent_contact_geometry_failure")
                            is True
                        )
                    ):
                        terminal_deterministic_failure_ids.add(candidate_id)
                    if planned.get("verdict") == "PASS":
                        if first_l5_pass_elapsed_s is None:
                            first_l5_pass_elapsed_s = time.monotonic() - start_time
                        passed_this_wave += 1
                        l5_passes.append(planned)
                        if target_reached():
                            stop_reason = "complete_l5_pass_found"
                            break
            wave_record: JsonDict = {
                "wave_index": wave.wave_index,
                "recovery_layer": recovery,
                "cumulative_per_branch": wave.cumulative_per_branch,
                "candidate_count": len(wave.candidates),
                "deep_candidate_count": len(deep_candidates),
                "pair_legality_reject_count": len(pair_rejections),
                "endpoint_pass_count": sum(item.get("endpoint_pass") is True for item in completed),
                "l5_submitted_count": submitted_this_wave,
                "l5_pass_count": passed_this_wave,
                "elapsed_s": time.monotonic() - wave_started,
            }
            if qualification_mode == "frozen_pair":
                wave_record.update(
                    {
                        "frozen_pair_batch_index": wave.frozen_pair_batch_index,
                        "frozen_pair_batch_role": (
                            "primary" if wave.frozen_pair_batch_index == 0 else "reserve"
                        ),
                    }
                )
            wave_evidence.append(wave_record)
            return bool(infrastructure_error or acceptable_target_reached())

        if not infrastructure_error:
            for wave in waves:
                if run_wave(wave, recovery=False):
                    break

        if not infrastructure_error and not acceptable_target_reached():
            # Only after the complete fast pool fails do the fixed remaining
            # six seeds become eligible.  Preserve the same waves and barriers.
            l5_pass_ids = {str(item.get("candidate_id") or "") for item in l5_passes}
            for wave in waves:
                retry = tuple(
                    descriptor
                    for descriptor in wave.candidates
                    if str(descriptor.get("candidate_id") or "") not in l5_pass_ids
                    and str(descriptor.get("candidate_id") or "")
                    not in terminal_deterministic_failure_ids
                    and prechecks[str(descriptor.get("candidate_id") or "")].get("verdict")
                    == "PASS"
                    and (
                        purpose != "placement"
                        or prechecks[str(descriptor.get("candidate_id") or "")].get(
                            "pair_legality_pass"
                        )
                        is True
                    )
                )
                if retry:
                    recovery_waves.append(
                        CandidateWave(
                            wave_index=len(wave_evidence),
                            cumulative_per_branch=wave.cumulative_per_branch,
                            candidates=retry,
                            recovery=True,
                            frozen_pair_batch_index=(wave.frozen_pair_batch_index),
                        )
                    )
            recovery_solver_mode_switched = False
            if solver_profile == "pick_ik_local":
                if self.set_solver_mode is None:
                    infrastructure_error = "pick_ik_mode_switch_unavailable"
                else:
                    try:
                        self._call_fast_service(
                            lambda: self.set_solver_mode("global"),
                            required_boolean="ok",
                        )
                        recovery_solver_mode_switched = True
                    except _QualificationInfrastructureError as exc:
                        infrastructure_error = str(exc)
            if not infrastructure_error:
                for wave in recovery_waves:
                    if run_wave(wave, recovery=True):
                        break
            if recovery_solver_mode_switched:
                try:
                    self._call_fast_service(
                        lambda: self.set_solver_mode("local"),
                        required_boolean="ok",
                    )
                except _QualificationInfrastructureError as exc:
                    infrastructure_error = str(exc)

        # Prefer two distinct model poses.  If exhaustive fast and recovery
        # coverage leaves exactly one L5-qualified grasp pose, independently
        # prove the second Beam-2 joint branch before exposing a fallback.  The
        # Cartesian contact remains byte-for-byte the provider pose.
        if not infrastructure_error and purpose == "grasp" and len(l5_passes) == 1:
            primary = l5_passes[0]
            alternate = self._alternate_grasp_joint_branch_screen(
                primary,
                source=source,
            )
            if alternate is not None:
                alternate_screen, joint_distance, alternate_state_sha256 = alternate
                candidate_id = str(primary.get("candidate_id") or "")
                descriptor = descriptor_by_id[candidate_id]
                branch_started = time.monotonic()
                planned_branch, retry_count = self._plan_fast_candidate(
                    descriptor,
                    revision,
                    screen=alternate_screen,
                )
                branch_elapsed_s = time.monotonic() - branch_started
                l5_attempts.append(
                    {
                        "attempt_index": len(l5_attempts),
                        "candidate_id": candidate_id,
                        "fixed_candidate_index": planned_branch.get("fixed_candidate_index"),
                        "source_grasp_id": planned_branch.get("source_grasp_id"),
                        "source_object_goal_id": planned_branch.get("source_object_goal_id"),
                        "se3_cluster_id": planned_branch.get("se3_cluster_id"),
                        "grasp_symmetry_family_id": planned_branch.get("grasp_symmetry_family_id"),
                        "wave_index": primary.get("wave_index"),
                        "recovery_layer": primary.get("recovery_layer") is True,
                        "retry_count": retry_count,
                        "elapsed_s": branch_elapsed_s,
                        "verdict": planned_branch.get("verdict"),
                        "reason": planned_branch.get("reason"),
                        "joint_branch_index": 1,
                        "joint_branch_joint_state_sha256": alternate_state_sha256,
                        "joint_branch_normalized_distance": joint_distance,
                    }
                )
                if planned_branch.get("infrastructure_error") is True:
                    infrastructure_error = str(
                        planned_branch.get("reason") or "plan_only_service_error"
                    )
                elif planned_branch.get("verdict") == "PASS":
                    primary_state_sha256 = _selected_ik_state_sha256(primary)
                    joint_branch_fallback_passes = [primary, planned_branch]
                    primary["joint_branch_l5_proofs"] = [
                        {
                            "candidate_id": candidate_id,
                            "joint_branch_index": 0,
                            "verdict": "PASS",
                            "selected_ik_joint_state_sha256": primary_state_sha256,
                        },
                        {
                            "candidate_id": candidate_id,
                            "joint_branch_index": 1,
                            "verdict": "PASS",
                            "selected_ik_joint_state_sha256": alternate_state_sha256,
                            "normalized_distance_from_primary": joint_distance,
                        },
                    ]
                    primary["joint_branch_pass_count"] = 2
                    primary["joint_branch_normalized_distance"] = joint_distance
                    latest[candidate_id] = primary

        if infrastructure_error:
            stop_reason = "infrastructure_error"
        elif target_reached():
            stop_reason = "complete_l5_pass_found"
        elif minimum_target_reached():
            stop_reason = "complete_l5_pass_found_minimum_lookahead"
        elif len(joint_branch_fallback_passes) >= 2:
            stop_reason = "complete_l5_pass_found_joint_branch_fallback"
        elif purpose == "grasp" and len(l5_passes) >= minimum_target:
            # No two independent symmetry families / SE(3) clusters survived.
            # The contract then falls back to the joint-farthest pair only
            # after exhaustive coverage.
            stop_reason = "complete_l5_pass_found_joint_space_fallback"
        elif purpose == "grasp" and len(l5_passes) >= min(2, minimum_target):
            # A look-ahead request may ask for four branches even when the
            # frozen model output contains only two or three executable ones.
            # Keep the proven primary set; never turn it into a false zero-pass.
            stop_reason = "complete_l5_pass_found_partial_lookahead"
        elif (
            purpose == "grasp"
            and len(l5_passes) == 1
            and len(joint_branch_fallback_passes) < 2
            and sum(not bool(wave.get("recovery_layer")) for wave in wave_evidence) == len(waves)
            and sum(bool(wave.get("recovery_layer")) for wave in wave_evidence)
            == len(recovery_waves)
        ):
            # A fully state-valid and L5-proven primary is still executable.
            # Publish it only after the frozen fast pool, fixed-seed recovery
            # pool, and alternate Beam-2 branch have all failed to produce the
            # preferred backup.  This is an explicit redundancy degradation,
            # never an early-success shortcut and never a relaxed IK/L5 proof.
            stop_reason = "complete_l5_pass_found_single_branch_exhaustive_fallback"
        else:
            stop_reason = "candidate_and_recovery_exhausted"

        if purpose == "grasp":
            if len(l5_passes) >= min(2, minimum_target):
                selected_ids = select_grasp_branches(
                    l5_passes,
                    source=source,
                    limit=min(target, len(l5_passes)),
                )
            elif len(joint_branch_fallback_passes) >= 2:
                selected_ids = [str(l5_passes[0].get("candidate_id") or "")]
            elif stop_reason == "complete_l5_pass_found_single_branch_exhaustive_fallback":
                selected_ids = [str(l5_passes[0].get("candidate_id") or "")]
            else:
                selected_ids = []
        elif purpose == "observation":
            selected_ids = [
                str(item.get("candidate_id") or "")
                for item in sorted(l5_passes, key=candidate_physical_quality_key)[:target]
            ]
        else:
            if qualification_mode == "frozen_pair":
                ordered_passes = sorted(
                    l5_passes,
                    key=candidate_physical_quality_key,
                )
                selected_passes: list[JsonDict] = []
                selected_grasps: set[str] = set()
                for item in ordered_passes:
                    grasp_id = str(item.get("source_grasp_id") or "")
                    if grasp_id and grasp_id in selected_grasps:
                        continue
                    selected_passes.append(item)
                    if grasp_id:
                        selected_grasps.add(grasp_id)
                    if len(selected_passes) >= target:
                        break
                if len(selected_passes) < target:
                    selected_ids_seen = {
                        str(item.get("candidate_id") or "") for item in selected_passes
                    }
                    selected_passes.extend(
                        item
                        for item in ordered_passes
                        if str(item.get("candidate_id") or "") not in selected_ids_seen
                    )
                selected_ids = [
                    str(item.get("candidate_id") or "") for item in selected_passes[:target]
                ]
            else:
                selected_ids = [str(l5_passes[0].get("candidate_id") or "")] if l5_passes else []
        binding = str(request.get("qualification_binding_sha256") or "")
        results: list[JsonDict] = []
        for fixed_index, raw_descriptor in enumerate(candidates):
            descriptor = raw_descriptor if isinstance(raw_descriptor, Mapping) else {}
            candidate_id = str(descriptor.get("candidate_id") or "")
            result = dict(latest.get(candidate_id) or prechecks.get(candidate_id) or {})
            if (
                result.get("workspace_pass") is True
                and not result.get("stages")
                and result.get("verdict") == "PASS"
            ):
                result.update(
                    {
                        "verdict": "NOT_EVALUATED",
                        "reason": (
                            "infrastructure_abort"
                            if infrastructure_error
                            else "complete_l5_pass_found"
                        ),
                        "endpoint_evaluated": False,
                    }
                )
            elif result.get("endpoint_pass") is True and candidate_id not in planned_ids:
                result.update(
                    {
                        "verdict": "NOT_EVALUATED",
                        "reason": "l5_not_submitted_after_success",
                        "full_plan_submitted": False,
                    }
                )
            result["fixed_candidate_index"] = fixed_index
            result["qualification_binding_sha256"] = binding
            history = list(screening_history.get(candidate_id, []))
            result["screening_attempt_count"] = len(history)
            # One screening attempt is already represented in full by the
            # result itself.  Keep detailed history only when recovery adds a
            # genuinely distinct attempt; this avoids duplicating large stage
            # and legality proofs on the MCP wire and in the artifact.
            result["screening_attempts"] = history if len(history) > 1 else []
            # The host-side validator restores these immutable parameters from
            # the original request.  Omitting the repeated copy from every
            # wire result keeps large frozen-frontier replies small without
            # changing the stored qualification artifact.
            result.pop("compile_parameters", None)
            results.append(result)

        screening_records = [
            attempt for attempts in screening_history.values() for attempt in attempts
        ]
        attempt_latencies = [
            float(attempt.get("elapsed_s", 0.0))
            for record in screening_records
            for stage in record.get("stages", [])
            if isinstance(stage, Mapping)
            for attempt in stage.get("pure_ik_attempts", [])
            if isinstance(attempt, Mapping)
        ]
        cache_hits = sum(
            str(seed.get("seed_source") or "") == "batch_cache"
            for record in screening_records
            for stage in record.get("stages", [])
            if isinstance(stage, Mapping)
            for seed in stage.get("ik_seeds", [])
            if isinstance(seed, Mapping)
        )
        rescues = sum(
            len(stage.get("collision_ik_attempts", []))
            for record in screening_records
            for stage in record.get("stages", [])
            if isinstance(stage, Mapping)
        )
        metrics = {
            "generated_count": len(candidates),
            "workspace_pass_count": sum(result.get("workspace_pass") is True for result in results),
            "pure_ik_pass_count": sum(
                any(
                    attempt.get("pure_ik_pass") is True
                    for attempt in screening_history.get(str(result.get("candidate_id") or ""), [])
                )
                for result in results
            ),
            "collision_ik_pass_count": sum(
                any(
                    attempt.get("collision_ik_pass") is True
                    for attempt in screening_history.get(str(result.get("candidate_id") or ""), [])
                )
                for result in results
            ),
            "endpoint_evaluated_count": sum(
                result.get("endpoint_evaluated") is True for result in results
            ),
            "endpoint_pass_count": sum(
                any(
                    attempt.get("endpoint_pass") is True
                    for attempt in screening_history.get(str(result.get("candidate_id") or ""), [])
                )
                for result in results
            ),
            "l5_attempt_count": len(l5_attempts),
            "l5_pass_count": len(l5_passes),
            "l5_joint_branch_pass_count": (
                len(joint_branch_fallback_passes)
                if joint_branch_fallback_passes
                else len(l5_passes)
            ),
            "screening_attempt_count": len(screening_records),
            "cache_hit_count": cache_hits,
            "collision_rescue_count": rescues,
            "max_ik_concurrency": ik_gate.maximum_active,
            "max_state_validity_concurrency": validity_gate.maximum_active,
            "ik_latency": latency_summary(attempt_latencies),
            "first_l5_pass_s": first_l5_pass_elapsed_s,
            "total_elapsed_s": time.monotonic() - start_time,
            "capability_map_status": capability_map_status,
            "l5_pass_target": target,
            "l5_min_pass_target": minimum_target,
            **legality_metrics,
        }
        completed_fast_wave_count = sum(
            not bool(wave.get("recovery_layer")) for wave in wave_evidence
        )
        completed_recovery_wave_count = sum(
            bool(wave.get("recovery_layer")) for wave in wave_evidence
        )
        search_exhaustion = {
            "fast_wave_count_expected": len(waves),
            "fast_wave_count_completed": completed_fast_wave_count,
            "fast_pool_exhausted": completed_fast_wave_count == len(waves),
            "recovery_wave_count_expected": len(recovery_waves),
            "recovery_wave_count_completed": completed_recovery_wave_count,
            "recovery_pool_exhausted": (completed_recovery_wave_count == len(recovery_waves)),
            "preferred_grasp_branch_target": target if purpose == "grasp" else 0,
            "published_grasp_branch_count": (len(selected_ids) if purpose == "grasp" else 0),
            "redundancy_degraded": (
                stop_reason == "complete_l5_pass_found_single_branch_exhaustive_fallback"
            ),
        }
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": revision,
            "execution_started": False,
            "qualification_profile": "fast_v3",
            "artifact_schema_version": FAST_ARTIFACT_SCHEMA,
            "solver_profile": solver_profile,
            "requested_solver_profile": requested_solver_profile,
            "solver_configuration_id": (
                f"{solver_profile}@{round(fast_timeout_s * 1000)}ms/c{max_ik}"
            ),
            "provider": str(source.get("provider") or "unknown"),
            "provider_version": str(source.get("provider_version") or "unknown"),
            "solver_version": str(
                source.get("solver_version") or current_state.get("solver_version") or "unknown"
            ),
            "robot_model_sha256": robot_hash,
            "scene_sha256": scene_hash,
            "authoritative_scene_sha256": authoritative_scene_hash,
            "moveit_world_geometry_sha256": moveit_world_geometry_hash,
            "moveit_attached_geometry_sha256": moveit_attached_geometry_hash,
            "moveit_geometry_verified_ids": moveit_geometry_verified_ids,
            "capability_map_id": capability_map.map_id if capability_map else requested_map_id,
            "legality_screening": {
                key: value
                for key, value in legality_metrics.items()
                if key.startswith("goal_legality_") or key.startswith("pair_legality_")
            },
            "waves": wave_evidence,
            "l5_attempts": l5_attempts,
            "l5_pass_target": target,
            "l5_min_pass_target": minimum_target,
            "selected_candidate_ids": selected_ids,
            "selected_joint_branches": [
                dict(proof)
                for proof in (
                    l5_passes[0].get("joint_branch_l5_proofs")
                    if joint_branch_fallback_passes
                    and isinstance(l5_passes[0].get("joint_branch_l5_proofs"), list)
                    else []
                )
                if isinstance(proof, Mapping)
            ],
            "search_exhaustion": search_exhaustion,
            "stop_reason": stop_reason,
            "infrastructure_error": bool(infrastructure_error),
            "metrics": metrics,
            "first_l5_pass_s": first_l5_pass_elapsed_s,
            "qualification_case_sha256": str(request.get("qualification_case_sha256") or ""),
            "case_id": str(request.get("qualification_case_sha256") or ""),
            "results": results,
        }

    @staticmethod
    def _integer_waves(value: object, default: Sequence[int]) -> tuple[int, ...]:
        raw = value if isinstance(value, list) else default
        waves = tuple(int(item) for item in raw)
        if (
            not waves
            or any(item <= 0 for item in waves)
            or any(right <= left for left, right in zip(waves, waves[1:]))
        ):
            raise ValueError("qualification waves must be positive and increasing")
        return waves

    def _call_fast_service(
        self,
        callback: Callable[[], Mapping[str, Any]],
        *,
        gate: _ServiceConcurrencyGate | None = None,
        required_boolean: str | None,
    ) -> tuple[JsonDict, int, float]:
        """Retry one infrastructure failure, then raise a run-level error."""

        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                raw = gate.call(callback) if gate is not None else callback()
                if not isinstance(raw, Mapping):
                    raise _QualificationInfrastructureError(
                        "qualification service returned non-object evidence"
                    )
                result = dict(raw)
                if result.get("infrastructure_error") is True:
                    raise _QualificationInfrastructureError(
                        str(result.get("reason") or "qualification service error")
                    )
                if required_boolean is not None and not isinstance(
                    result.get(required_boolean), bool
                ):
                    raise _QualificationInfrastructureError(
                        f"qualification service omitted {required_boolean}"
                    )
                return result, attempt, time.monotonic() - started
            except (TimeoutError, _QualificationInfrastructureError) as exc:
                last_error = exc
            except Exception as exc:  # noqa: BLE001 - ROS/service boundary.
                last_error = exc
            if attempt == 0 and self.service_health_check is not None:
                try:
                    self.service_health_check()
                except Exception:  # noqa: BLE001 - retry still proceeds.
                    pass
        raise _QualificationInfrastructureError(
            f"qualification infrastructure failed twice: {type(last_error).__name__}"
        ) from last_error

    def _screen_fast_candidate(
        self,
        descriptor: Mapping[str, Any],
        revision: int,
        *,
        legality_precheck: Mapping[str, Any],
        source: Mapping[str, Any],
        current_state: Mapping[str, Any],
        batch_cache: Sequence[Mapping[str, Any]],
        beam_width: int,
        seed_count: int,
        timeout_s: float,
        recovery: bool,
        solver_profile: str,
        ik_gate: _ServiceConcurrencyGate,
        validity_gate: _ServiceConcurrencyGate,
    ) -> JsonDict:
        candidate = descriptor.get("candidate")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        stages = candidate.get("qualification_stages")
        base = json.loads(json.dumps(dict(legality_precheck)))
        base.update(
            {
                "fixed_candidate_index": int(descriptor.get("fixed_candidate_index", 0)),
                "se3_cluster_id": descriptor.get("se3_cluster_id"),
                "grasp_symmetry_family_id": descriptor.get("grasp_symmetry_family_id"),
                "capability_score": dict(descriptor.get("capability_score") or {}),
                "generator_score": generator_score(candidate),
                "source_grasp_id": str(candidate.get("source_grasp_id") or ""),
                "source_object_goal_id": str(candidate.get("source_object_goal_id") or ""),
                "frozen_pair_batch_index": candidate.get("frozen_pair_batch_index", 0),
                "frozen_pair_batch_role": str(candidate.get("frozen_pair_batch_role") or ""),
                "endpoint_evaluated": True,
                "stages": [],
            }
        )
        robust_clearance = descriptor.get("placement_robust_clearance_m")
        if (
            isinstance(robust_clearance, (int, float))
            and not isinstance(robust_clearance, bool)
            and math.isfinite(float(robust_clearance))
        ):
            base["placement_robust_clearance_m"] = float(robust_clearance)
        centering_evidence = parallel_gripper_centering_evidence(candidate)
        if centering_evidence is not None:
            # Preserve the immutable provider measurement after the private
            # screening boundary so the L5 rank uses the same physical signal
            # as the early-wave scheduler.  This changes order only: no model
            # pose is corrected and no candidate is removed.
            base["target_closing_alignment"] = centering_evidence
        scene_centering = candidate.get("scene_target_closing_alignment")
        if isinstance(scene_centering, Mapping):
            base["scene_target_closing_alignment"] = json.loads(json.dumps(scene_centering))
        if base.get("workspace_pass") is not True:
            return base
        if not isinstance(stages, list) or not stages:
            return {**base, "verdict": "UNKNOWN", "reason": "compiled_stages_missing"}
        candidate_start = candidate.get("qualification_start_joint_state")
        start = dict(candidate_start if isinstance(candidate_start, Mapping) else current_state)
        if "joint_limits" not in source and isinstance(current_state.get("joint_limits"), Mapping):
            source = {**dict(source), "joint_limits": current_state["joint_limits"]}
        active_scene_diff: Mapping[str, Any] | None = None
        scene: Any = None
        initial_transition = candidate.get("initial_scene_transition")
        staged_transition = any(
            isinstance(stage, Mapping) and bool(stage.get("scene_transition")) for stage in stages
        )
        if initial_transition or staged_transition:
            if self.clone_scene is None or self.apply_scene_transition is None:
                return {
                    **base,
                    "verdict": "UNKNOWN",
                    "reason": "virtual_scene_transition_unavailable",
                    "infrastructure_error": True,
                }
            try:
                scene = self.clone_scene()
            except Exception as exc:  # noqa: BLE001
                return {
                    **base,
                    "verdict": "UNKNOWN",
                    "reason": "virtual_scene_transition_error",
                    "error_type": type(exc).__name__,
                    "infrastructure_error": True,
                }
        if initial_transition:
            try:
                transition_pose = candidate.get("initial_scene_transition_pose")
                transition_target = dict(
                    transition_pose if isinstance(transition_pose, Mapping) else stages[0]
                )
                predicted_attachment = candidate.get("predicted_attachment_transform")
                if (
                    isinstance(predicted_attachment, Mapping)
                    and candidate.get("physical_scene_attachment_required") is not True
                ):
                    transition_target["attachment_transform"] = dict(predicted_attachment)
                transition = self.apply_scene_transition(
                    scene, str(initial_transition), transition_target
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    **base,
                    "verdict": "UNKNOWN",
                    "reason": "virtual_scene_transition_error",
                    "error_type": type(exc).__name__,
                    "infrastructure_error": True,
                }
            if transition.get("ok") is not True:
                return {
                    **base,
                    "verdict": "FAIL",
                    "reason": "virtual_scene_transition_failed",
                }
            if isinstance(transition.get("planning_scene_diff"), Mapping):
                active_scene_diff = dict(transition["planning_scene_diff"])

        previous_beam: list[JsonDict] = []
        for stage_index, raw_target in enumerate(stages):
            if self.scene_revision() != revision:
                return {
                    **base,
                    "verdict": "UNKNOWN",
                    "reason": "planning_scene_revision_drift",
                    "infrastructure_error": True,
                }
            if not isinstance(raw_target, Mapping):
                return {**base, "verdict": "UNKNOWN", "reason": "compiled_stage_invalid"}
            target = dict(raw_target)
            allowed_collisions = _contact_allowed_collisions(scene, target)
            if allowed_collisions:
                target["qualification_allowed_collisions"] = allowed_collisions
            if active_scene_diff is not None:
                target["qualification_scene_diff"] = dict(active_scene_diff)
            seeds = self._fast_stage_seeds(
                start,
                previous_beam=previous_beam,
                batch_cache=batch_cache,
                current_state=current_state,
                source=source,
                # Recovery expands only the first stage to its six fixed
                # seeds. Every dependent stage propagates the surviving
                # Beam-2 parents under the same bounded branching contract as
                # the fast layer.
                count=seed_count if stage_index == 0 else beam_width,
                recovery=recovery and stage_index == 0,
                initial_seed_source=(
                    "candidate_start_state"
                    if isinstance(candidate_start, Mapping)
                    else "current_robot_state"
                ),
                candidate_seed_states=(
                    _candidate_same_run_seed_states(candidate) if stage_index == 0 else ()
                ),
            )
            evidence: JsonDict = {
                "stage_index": stage_index,
                "name": str(target.get("name") or f"stage_{stage_index}"),
                "target_pose": dict(target),
                "start_joint_state_sha256": _hash(start),
                "execution_started": False,
                "ik_seeds": [self._public_seed(seed) for seed in seeds],
                "pure_ik_attempts": [],
                "collision_ik_attempts": [],
            }
            solutions: list[JsonDict] = []
            saw_pure_solution = False
            saw_collision = False
            stage_started = time.monotonic()
            try:
                for seed_index, seed in enumerate(seeds):
                    seeded_target = dict(target)
                    seeded_target["ik_seed_timeout_s"] = timeout_s
                    seeded_target["solver_profile"] = solver_profile
                    # The solver keeps its short per-seed budget, while the
                    # ROS response may legitimately queue behind every other
                    # request admitted by this deterministic wave.
                    seeded_target["qualification_ik_queue_depth"] = ik_gate.limit
                    pure, retry_count, elapsed = self._call_fast_service(
                        lambda seeded_target=seeded_target, seed=seed: self.compute_ik(
                            seeded_target, seed, False
                        ),
                        gate=ik_gate,
                        required_boolean="ok",
                    )
                    pure_attempt: JsonDict = {
                        "seed_index": seed_index,
                        "seed_source": seed.get("seed_source"),
                        "seed_sha256": _hash(self._public_seed(seed)),
                        "ok": pure.get("ok") is True,
                        "retry_count": retry_count,
                        "moveit_error_code": pure.get("moveit_error_code"),
                        "solver": pure.get("solver") or solver_profile,
                        "solver_version": pure.get("solver_version"),
                        "elapsed_s": elapsed,
                    }
                    evidence["pure_ik_attempts"].append(pure_attempt)
                    if pure.get("ok") is not True:
                        continue
                    saw_pure_solution = True
                    pure_state = pure.get("joint_state")
                    if not _valid_joint_state(pure_state):
                        raise _QualificationInfrastructureError(
                            "IK success returned an invalid joint state"
                        )
                    validity_state = _with_qualification_state_policy(
                        pure_state,
                        target=target,
                        scene_diff=active_scene_diff,
                        queue_depth=validity_gate.limit,
                    )
                    validity, validity_retry, validity_elapsed = self._call_fast_service(
                        lambda validity_state=validity_state: self.check_state_validity(
                            validity_state
                        ),
                        gate=validity_gate,
                        required_boolean="valid",
                    )
                    pure_attempt.update(
                        {
                            "state_valid": validity.get("valid") is True,
                            "state_validity_retry_count": validity_retry,
                            "state_validity_elapsed_s": validity_elapsed,
                            "collision_pairs": list(validity.get("collision_pairs") or []),
                        }
                    )
                    solution_state: Mapping[str, Any] | None = None
                    solution_result = pure
                    rescue_count = int(seed.get("_parent_rescues", 0))
                    if validity.get("valid") is True:
                        solution_state = pure_state
                    else:
                        saw_collision = True
                        rescue_target = dict(target)
                        rescue_target["ik_seed_timeout_s"] = timeout_s
                        rescue_target["solver_profile"] = solver_profile
                        rescue_target["qualification_ik_queue_depth"] = ik_gate.limit
                        rescue, rescue_retry, rescue_elapsed = self._call_fast_service(
                            lambda rescue_target=rescue_target, pure_state=pure_state: (
                                self.compute_ik(rescue_target, pure_state, True)
                            ),
                            gate=ik_gate,
                            required_boolean="ok",
                        )
                        rescue_attempt: JsonDict = {
                            "seed_index": seed_index,
                            "seed_source": seed.get("seed_source"),
                            "seed_sha256": _hash(pure_state),
                            "ok": rescue.get("ok") is True,
                            "retry_count": rescue_retry,
                            "moveit_error_code": rescue.get("moveit_error_code"),
                            "solver": rescue.get("solver") or solver_profile,
                            "elapsed_s": rescue_elapsed,
                        }
                        evidence["collision_ik_attempts"].append(rescue_attempt)
                        if rescue.get("ok") is not True:
                            continue
                        rescued_state = rescue.get("joint_state")
                        if not _valid_joint_state(rescued_state):
                            raise _QualificationInfrastructureError(
                                "collision rescue returned an invalid joint state"
                            )
                        rescued_validity_state = _with_qualification_state_policy(
                            rescued_state,
                            target=target,
                            scene_diff=active_scene_diff,
                            queue_depth=validity_gate.limit,
                        )
                        rescued_validity, rescued_retry, rescued_elapsed = self._call_fast_service(
                            lambda rescued_validity_state=rescued_validity_state: (
                                self.check_state_validity(rescued_validity_state)
                            ),
                            gate=validity_gate,
                            required_boolean="valid",
                        )
                        rescue_attempt.update(
                            {
                                "state_valid": rescued_validity.get("valid") is True,
                                "state_validity_retry_count": rescued_retry,
                                "state_validity_elapsed_s": rescued_elapsed,
                                "collision_pairs": list(
                                    rescued_validity.get("collision_pairs") or []
                                ),
                            }
                        )
                        if rescued_validity.get("valid") is not True:
                            continue
                        solution_state = rescued_state
                        solution_result = rescue
                        rescue_count += 1
                    parent_state = seed.get("_chain_parent_state")
                    parent_state = parent_state if isinstance(parent_state, Mapping) else start
                    joint_travel = normalized_joint_distance(
                        parent_state, solution_state, source=source
                    )
                    raw_singular = (
                        solution_result.get("min_singular_value")
                        if solution_result.get("min_singular_value") is not None
                        else solution_result.get("jacobian_min_singular_value", 0.0)
                    )
                    try:
                        minimum_singular = float(raw_singular)
                    except (TypeError, ValueError) as exc:
                        raise _QualificationInfrastructureError(
                            "IK returned malformed Jacobian quality"
                        ) from exc
                    if not math.isfinite(minimum_singular) or minimum_singular < 0.0:
                        raise _QualificationInfrastructureError(
                            "IK returned invalid Jacobian quality"
                        )
                    solutions.append(
                        {
                            "joint_state": dict(solution_state),
                            "state_valid": True,
                            "joint_margin": joint_limit_margin(solution_state, source=source),
                            "min_singular_value": minimum_singular,
                            "joint_travel": joint_travel,
                            "cumulative_joint_travel": float(
                                seed.get("_parent_cumulative_joint_travel", 0.0)
                            )
                            + joint_travel,
                            "collision_rescues": rescue_count,
                            "generator_score": generator_score(candidate),
                            "fixed_candidate_index": descriptor.get("fixed_candidate_index", 0),
                            "seed_index": seed_index,
                            "seed_source": seed.get("seed_source"),
                        }
                    )
            except _QualificationInfrastructureError as exc:
                evidence["elapsed_s"] = time.monotonic() - stage_started
                base["stages"].append(evidence)
                return {
                    **base,
                    "verdict": "UNKNOWN",
                    "reason": "qualification_service_error",
                    "infrastructure_error": True,
                    "infrastructure_error_detail": str(exc),
                }
            selected = deduplicate_beam_solutions(solutions, source=source, limit=beam_width)
            evidence["kinematic_ik"] = saw_pure_solution
            evidence["pure_state_valid"] = any(
                attempt.get("state_valid") is True for attempt in evidence["pure_ik_attempts"]
            )
            evidence["collision_ik_called"] = saw_collision
            evidence["collision_ik"] = bool(selected)
            evidence["state_valid"] = bool(selected)
            evidence["beam_solutions"] = selected
            evidence["beam_width"] = len(selected)
            evidence["elapsed_s"] = time.monotonic() - stage_started
            if not selected:
                base["stages"].append(evidence)
                return {
                    **base,
                    "verdict": "FAIL",
                    "reason": (
                        "collision_state_invalid" if saw_pure_solution else "kinematic_ik_failed"
                    ),
                }
            best = selected[0]
            evidence.update(
                {
                    "end_joint_state": dict(best["joint_state"]),
                    "joint_margin": best["joint_margin"],
                    "min_singular_value": best["min_singular_value"],
                    "joint_travel": best["joint_travel"],
                    "cumulative_joint_travel": best["cumulative_joint_travel"],
                    "collision_rescues": best["collision_rescues"],
                    "collision_pairs": [],
                }
            )
            base["stages"].append(evidence)
            previous_beam = selected
            start = dict(best["joint_state"])
            transition_name = target.get("scene_transition")
            if transition_name:
                if self.apply_scene_transition is None:
                    return {
                        **base,
                        "verdict": "UNKNOWN",
                        "reason": "virtual_scene_transition_unavailable",
                        "infrastructure_error": True,
                    }
                try:
                    transition = self.apply_scene_transition(scene, str(transition_name), target)
                except Exception as exc:  # noqa: BLE001
                    return {
                        **base,
                        "verdict": "UNKNOWN",
                        "reason": "virtual_scene_transition_error",
                        "error_type": type(exc).__name__,
                        "infrastructure_error": True,
                    }
                evidence["scene_transition"] = dict(transition)
                if transition.get("ok") is not True:
                    return {
                        **base,
                        "verdict": "FAIL",
                        "reason": "virtual_scene_transition_failed",
                    }
                if isinstance(transition.get("planning_scene_diff"), Mapping):
                    active_scene_diff = dict(transition["planning_scene_diff"])
                post_gripper_state = target.get("qualification_post_transition_gripper_state")
                if post_gripper_state:
                    if post_gripper_state != "open":
                        return {
                            **base,
                            "verdict": "UNKNOWN",
                            "reason": "post_transition_gripper_state_unsupported",
                            "infrastructure_error": True,
                        }
                    post_checks: list[JsonDict] = []
                    post_valid: list[JsonDict] = []
                    try:
                        for branch_index, solution in enumerate(selected):
                            post_state = _with_qualification_state_policy(
                                solution["joint_state"],
                                target={},
                                scene_diff=active_scene_diff,
                                queue_depth=validity_gate.limit,
                            )
                            post_state["qualification_gripper_state"] = "open"
                            post_result, post_retry, post_elapsed = self._call_fast_service(
                                lambda post_state=post_state: self.check_state_validity(post_state),
                                gate=validity_gate,
                                required_boolean="valid",
                            )
                            post_checks.append(
                                {
                                    "joint_branch_index": branch_index,
                                    "joint_state_sha256": _hash(solution["joint_state"]),
                                    "requested_gripper_state": "open",
                                    "valid": post_result.get("valid") is True,
                                    "retry_count": post_retry,
                                    "elapsed_s": post_elapsed,
                                    "collision_pairs": list(
                                        post_result.get("collision_pairs") or []
                                    ),
                                }
                            )
                            if post_result.get("valid") is True:
                                post_valid.append(solution)
                    except _QualificationInfrastructureError as exc:
                        evidence["post_transition_gripper_state_checks"] = post_checks
                        return {
                            **base,
                            "verdict": "UNKNOWN",
                            "reason": "qualification_service_error",
                            "infrastructure_error": True,
                            "infrastructure_error_detail": str(exc),
                        }
                    evidence["post_transition_gripper_state_checks"] = post_checks
                    selected = post_valid
                    evidence["state_valid"] = bool(selected)
                    evidence["beam_solutions"] = selected
                    evidence["beam_width"] = len(selected)
                    if not selected:
                        return {
                            **base,
                            "verdict": "FAIL",
                            "reason": "post_transition_gripper_state_invalid",
                        }
                    # A second Beam branch may be the only one that clears the
                    # open hand envelope.  Rebase the chain on that surviving
                    # deterministic branch instead of retaining the earlier
                    # closed-gripper winner.
                    best = selected[0]
                    evidence.update(
                        {
                            "end_joint_state": dict(best["joint_state"]),
                            "joint_margin": best["joint_margin"],
                            "min_singular_value": best["min_singular_value"],
                            "joint_travel": best["joint_travel"],
                            "cumulative_joint_travel": best["cumulative_joint_travel"],
                            "collision_rescues": best["collision_rescues"],
                        }
                    )
                    previous_beam = selected
                    start = dict(best["joint_state"])
        base.update(
            {
                "pure_ik_pass": True,
                "collision_ik_pass": True,
                "endpoint_pass": True,
                "verdict": "PASS",
                "reason": "endpoint_qualified",
            }
        )
        return base

    def _fast_stage_seeds(
        self,
        start: Mapping[str, Any],
        *,
        previous_beam: Sequence[Mapping[str, Any]],
        batch_cache: Sequence[Mapping[str, Any]],
        current_state: Mapping[str, Any],
        source: Mapping[str, Any],
        count: int,
        recovery: bool,
        initial_seed_source: str,
        candidate_seed_states: Sequence[Mapping[str, Any]] = (),
    ) -> list[JsonDict]:
        if recovery:
            seeds = fixed_recovery_seeds(start, source=source, count=count)
            for seed in seeds:
                seed["_chain_parent_state"] = dict(start)
            return seeds
        seeds: list[JsonDict] = []
        if previous_beam:
            for index, solution in enumerate(previous_beam[:count]):
                state = solution.get("joint_state")
                if not isinstance(state, Mapping):
                    continue
                seed = dict(state)
                seed["seed_source"] = f"parent_beam_{index}"
                seed["_chain_parent_state"] = dict(state)
                seed["_parent_cumulative_joint_travel"] = float(
                    solution.get("cumulative_joint_travel", 0.0)
                )
                seed["_parent_rescues"] = int(solution.get("collision_rescues", 0))
                seeds.append(seed)
        else:
            seed = dict(start)
            seed["seed_source"] = initial_seed_source
            seed["_chain_parent_state"] = dict(start)
            seeds.append(seed)
        if len(seeds) < count:
            start_names = [str(name) for name in start.get("names") or []]
            trusted_candidate_seeds = [
                value
                for value in candidate_seed_states
                if _valid_joint_state(value)
                and [str(name) for name in value.get("names") or []] == start_names
            ]
            ordered_supplements: list[tuple[Mapping[str, Any], str]] = [
                (value, "frozen_pair_qualified_same_run")
                for value in sorted(
                    trusted_candidate_seeds,
                    key=lambda value: (
                        normalized_joint_distance(start, value, source=source),
                        _hash(
                            {
                                "names": value.get("names"),
                                "positions": value.get("positions"),
                            }
                        ),
                    ),
                )
            ]
            ordered_supplements.extend(
                [
                    (value, "batch_cache")
                    for value in sorted(
                        batch_cache,
                        key=lambda value: (
                            normalized_joint_distance(start, value, source=source),
                            _hash(
                                {
                                    "names": value.get("names"),
                                    "positions": value.get("positions"),
                                }
                            ),
                        ),
                    )
                ]
            )
            home = source.get("home_joint_state") or current_state.get("home_joint_state")
            if isinstance(home, Mapping):
                ordered_supplements.append((home, "named_home"))
            for supplement_state, supplement_source in ordered_supplements:
                supplement = dict(supplement_state)
                supplement["seed_source"] = supplement_source
                supplement["_chain_parent_state"] = dict(start)
                if previous_beam:
                    supplement["_parent_cumulative_joint_travel"] = float(
                        previous_beam[0].get("cumulative_joint_travel", 0.0)
                    )
                    supplement["_parent_rescues"] = int(
                        previous_beam[0].get("collision_rescues", 0)
                    )
                expanded = _unique_joint_state_seeds([*seeds, supplement], limit=count)
                if len(expanded) > len(seeds):
                    seeds = expanded
                if len(seeds) >= count:
                    break
        return _unique_joint_state_seeds(seeds, limit=count)[:count]

    @staticmethod
    def _public_seed(seed: Mapping[str, Any]) -> JsonDict:
        return {
            "names": list(seed.get("names") or []),
            "positions": list(seed.get("positions") or []),
            "seed_source": seed.get("seed_source"),
        }

    @staticmethod
    def _alternate_grasp_joint_branch_screen(
        screen: Mapping[str, Any],
        *,
        source: Mapping[str, Any],
    ) -> tuple[JsonDict, float, str] | None:
        """Select the farthest independently screened Beam-2 contact branch.

        Direct-contact grasp qualification has exactly one Cartesian stage.
        This helper changes only that stage's selected joint solution; the
        provider's terminal SE(3) pose and every scene input remain unchanged.
        """

        stages = screen.get("stages")
        if not isinstance(stages, list) or len(stages) != 1:
            return None
        stage = stages[0]
        if not isinstance(stage, Mapping):
            return None
        primary_state = _sanitized_joint_state(stage.get("end_joint_state"))
        beam = stage.get("beam_solutions")
        if primary_state is None or not isinstance(beam, list):
            return None
        primary_sha256 = _hash(primary_state)
        alternatives: list[tuple[float, int, str, Mapping[str, Any], JsonDict]] = []
        for beam_index, raw_solution in enumerate(beam):
            if not isinstance(raw_solution, Mapping):
                continue
            state = _sanitized_joint_state(raw_solution.get("joint_state"))
            if state is None:
                continue
            state_sha256 = _hash(state)
            if state_sha256 == primary_sha256:
                continue
            distance = normalized_joint_distance(
                primary_state,
                state,
                source=source,
            )
            if not math.isfinite(distance) or distance < JOINT_SOLUTION_DEDUP_DISTANCE:
                continue
            alternatives.append((distance, beam_index, state_sha256, raw_solution, state))
        if not alternatives:
            return None
        distance, _, state_sha256, solution, state = sorted(
            alternatives,
            key=lambda item: (-item[0], item[1], item[2]),
        )[0]
        alternate = json.loads(json.dumps(dict(screen)))
        alternate_stage = alternate["stages"][0]
        alternate_stage["end_joint_state"] = state
        alternate_stage["selected_joint_branch_index"] = 1
        alternate_stage["selected_ik_end_joint_state_sha256"] = state_sha256
        for key in (
            "joint_margin",
            "min_singular_value",
            "joint_travel",
            "cumulative_joint_travel",
            "collision_rescues",
        ):
            if key in solution:
                alternate_stage[key] = solution[key]
        return alternate, distance, state_sha256

    def _plan_fast_candidate(
        self,
        descriptor: Mapping[str, Any],
        revision: int,
        *,
        screen: Mapping[str, Any],
    ) -> tuple[JsonDict, int]:
        infrastructure_reasons = {
            "plan_only_timeout",
            "plan_only_service_error",
            "plan_only_execution_evidence_missing",
            "plan_only_empty_trajectory",
            "plan_only_end_state_missing",
            "planning_context_unavailable",
            "planning_scene_revision_drift",
            "virtual_scene_transition_error",
            "virtual_scene_transition_unavailable",
        }
        first = self._plan_candidate(descriptor, revision, screen=screen)
        if first.get("reason") not in infrastructure_reasons:
            return first, 0
        if self.service_health_check is not None:
            try:
                self.service_health_check()
            except Exception:  # noqa: BLE001
                pass
        second = self._plan_candidate(descriptor, revision, screen=screen)
        if second.get("reason") in infrastructure_reasons:
            second["infrastructure_error"] = True
        return second, 1

    @staticmethod
    def _fast_infrastructure_response(
        request: Mapping[str, Any],
        candidates: Sequence[object],
        revision: int,
        *,
        reason: str,
    ) -> JsonDict:
        binding = str(request.get("qualification_binding_sha256") or "")
        return {
            "schema_version": request["schema_version"],
            "planning_scene_revision": revision,
            "execution_started": False,
            "qualification_profile": "fast_v3",
            "stop_reason": "infrastructure_error",
            "infrastructure_error": True,
            "metrics": {"generated_count": len(candidates)},
            "results": [
                {
                    "candidate_id": str(item.get("candidate_id") or "")
                    if isinstance(item, Mapping)
                    else "",
                    "candidate_pose_sha256": str(item.get("candidate_pose_sha256") or "")
                    if isinstance(item, Mapping)
                    else "",
                    "qualification_binding_sha256": binding,
                    "verdict": "UNKNOWN",
                    "reason": "qualification_infrastructure_error",
                    "infrastructure_error": True,
                    "infrastructure_error_detail": reason,
                    "execution_started": False,
                    "stages": [],
                }
                for item in candidates
            ],
        }

    @staticmethod
    def _fast_configuration_response(
        request: Mapping[str, Any],
        candidates: Sequence[object],
        revision: int,
        *,
        reason: str,
        error: str,
        robot_hash: str,
        scene_hash: str,
    ) -> JsonDict:
        response = MoveItQualificationEngine._fast_infrastructure_response(
            request, candidates, revision, reason=error
        )
        response.update(
            {
                "stop_reason": "configuration_error",
                "robot_model_sha256": robot_hash,
                "scene_sha256": scene_hash,
            }
        )
        for item in response["results"]:
            item["reason"] = reason
        return response

    def _screen_candidate(
        self,
        descriptor: object,
        revision: int,
        *,
        seed_count: int,
        source: Mapping[str, Any],
        workspace_precheck: Mapping[str, Any] | None = None,
    ) -> JsonDict:
        item = descriptor if isinstance(descriptor, Mapping) else {}
        candidate_id = str(item.get("candidate_id") or "")
        pose_hash = str(item.get("candidate_pose_sha256") or "")
        candidate = item.get("candidate")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        stages = candidate.get("qualification_stages")
        base: JsonDict = {
            "candidate_id": candidate_id,
            "candidate_pose_sha256": pose_hash,
            "execution_started": False,
            "stages": [],
            "workspace_pass": False,
            "pure_ik_pass": False,
            "collision_ik_pass": False,
            "endpoint_pass": False,
            "endpoint_evaluated": False,
            "full_plan_submitted": False,
        }
        if isinstance(candidate.get("compile_parameters"), Mapping):
            base["compile_parameters"] = dict(candidate["compile_parameters"])
        if not candidate_id or not pose_hash or not isinstance(stages, list) or not stages:
            return {**base, "verdict": "UNKNOWN", "reason": "compiled_stages_missing"}
        try:
            candidate_start = candidate.get("qualification_start_joint_state")
            start = dict(
                candidate_start
                if isinstance(candidate_start, Mapping)
                else self.current_joint_state()
            )
        except Exception as exc:  # noqa: BLE001
            return {
                **base,
                "verdict": "UNKNOWN",
                "reason": "start_joint_state_unavailable",
                "error_type": type(exc).__name__,
            }
        if workspace_precheck is not None:
            if (
                workspace_precheck.get("verdict") != "PASS"
                or workspace_precheck.get("candidate_id") != candidate_id
                or workspace_precheck.get("candidate_pose_sha256") != pose_hash
                or workspace_precheck.get("workspace_pass") is not True
            ):
                return dict(workspace_precheck)
        elif self.workspace_filter is not None:
            try:
                if not all(
                    self.workspace_filter(target)
                    for target in stages
                    if isinstance(target, Mapping)
                ):
                    return {**base, "verdict": "FAIL", "reason": "workspace_envelope_rejected"}
            except Exception as exc:  # noqa: BLE001
                return {
                    **base,
                    "verdict": "UNKNOWN",
                    "reason": "workspace_filter_error",
                    "error_type": type(exc).__name__,
                }
        base["workspace_pass"] = True
        base["endpoint_evaluated"] = True
        previous_solution: Mapping[str, Any] | None = None
        active_scene_diff: Mapping[str, Any] | None = None
        initial_transition = candidate.get("initial_scene_transition")
        if initial_transition:
            if self.clone_scene is None or self.apply_scene_transition is None:
                return {
                    **base,
                    "verdict": "UNKNOWN",
                    "reason": "virtual_scene_transition_unavailable",
                }
            try:
                scene = self.clone_scene()
                transition_pose = candidate.get("initial_scene_transition_pose")
                transition_target = dict(
                    transition_pose if isinstance(transition_pose, Mapping) else stages[0]
                )
                predicted_attachment = candidate.get("predicted_attachment_transform")
                if (
                    isinstance(predicted_attachment, Mapping)
                    and candidate.get("physical_scene_attachment_required") is not True
                ):
                    transition_target["attachment_transform"] = dict(predicted_attachment)
                transition = self.apply_scene_transition(
                    scene, str(initial_transition), transition_target
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    **base,
                    "verdict": "UNKNOWN",
                    "reason": "virtual_scene_transition_error",
                    "error_type": type(exc).__name__,
                }
            if transition.get("ok") is not True:
                return {
                    **base,
                    "verdict": "FAIL",
                    "reason": "virtual_scene_transition_failed",
                }
            next_diff = transition.get("planning_scene_diff")
            if isinstance(next_diff, Mapping):
                active_scene_diff = dict(next_diff)
        for index, target in enumerate(stages):
            if self.scene_revision() != revision:
                return {**base, "verdict": "UNKNOWN", "reason": "planning_scene_revision_drift"}
            if not isinstance(target, Mapping):
                return {**base, "verdict": "UNKNOWN", "reason": "compiled_stage_invalid"}
            target = dict(target)
            if active_scene_diff is not None:
                target["qualification_scene_diff"] = dict(active_scene_diff)
            evidence: JsonDict = {
                "stage_index": index,
                "name": str(target.get("name") or f"stage_{index}"),
                "start_joint_state_sha256": _hash(start),
                "execution_started": False,
                "target_pose": dict(target),
            }
            seeds = self._ik_seeds(
                start,
                count=seed_count,
                candidate_id=candidate_id,
                stage_index=index,
                source=source,
                previous_solution=previous_solution,
            )
            evidence["ik_seeds"] = [dict(seed) for seed in seeds]
            pure_ik, pure_attempts, pure_error = self._try_ik(target, seeds, False)
            evidence["pure_ik_attempts"] = pure_attempts
            if pure_error:
                return self._unknown(base, evidence, pure_error)
            evidence["kinematic_ik"] = pure_ik.get("ok") is True
            if not evidence["kinematic_ik"]:
                return self._fail(base, evidence, "kinematic_ik_failed")
            pure_state = pure_ik.get("joint_state")
            if not isinstance(pure_state, Mapping):
                return self._unknown(base, evidence, "kinematic_ik_evidence_missing")
            try:
                validity_state = dict(pure_state)
                if active_scene_diff is not None:
                    validity_state["qualification_scene_diff"] = dict(active_scene_diff)
                validity = dict(self.check_state_validity(validity_state))
            except TimeoutError:
                evidence["pure_state_valid"] = None
                evidence["pure_state_validity_error"] = "state_validity_timeout"
            except Exception as exc:  # noqa: BLE001
                evidence["pure_state_valid"] = None
                evidence["pure_state_validity_error"] = "state_validity_service_error"
                evidence["pure_state_validity_error_type"] = type(exc).__name__
            else:
                evidence["pure_state_valid"] = validity.get("valid") is True
                evidence["pure_collision_pairs"] = list(validity.get("collision_pairs") or [])
            collision_seeds = _unique_joint_state_seeds(
                [dict(pure_state), *seeds],
                limit=seed_count,
            )
            evidence["collision_ik_seeds"] = [dict(seed) for seed in collision_seeds]
            collision_ik, collision_attempts, collision_error = self._try_collision_ik(
                target,
                collision_seeds,
                active_scene_diff=active_scene_diff,
            )
            evidence["collision_ik_attempts"] = collision_attempts
            if collision_error:
                return self._unknown(base, evidence, collision_error)
            evidence["collision_ik"] = collision_ik.get("ok") is True
            if not evidence["collision_ik"]:
                evidence["collision_pairs"] = list(
                    collision_ik.get("collision_pairs")
                    or evidence.get("pure_collision_pairs")
                    or []
                )
                return self._fail(
                    base,
                    evidence,
                    (
                        "collision_state_invalid"
                        if collision_ik.get("all_solutions_state_invalid") is True
                        else "collision_ik_failed"
                    ),
                )
            collision_state = collision_ik.get("joint_state")
            if not isinstance(collision_state, Mapping):
                return self._unknown(base, evidence, "collision_ik_evidence_missing")
            collision_validity = collision_ik.get("state_validity")
            if not isinstance(collision_validity, Mapping):
                return self._unknown(base, evidence, "collision_ik_evidence_missing")
            evidence["collision_state_valid"] = collision_validity.get("valid") is True
            evidence["state_valid"] = evidence["collision_state_valid"]
            evidence["collision_pairs"] = list(collision_validity.get("collision_pairs") or [])
            if not evidence["state_valid"]:
                return self._fail(base, evidence, "collision_ik_state_invalid")
            end_state = collision_state
            if not isinstance(end_state, Mapping):
                return self._unknown(base, evidence, "collision_ik_evidence_missing")
            evidence["end_joint_state"] = dict(end_state)
            base["stages"].append(evidence)
            start = dict(end_state)
            previous_solution = dict(end_state)
        if self.scene_revision() != revision:
            return {**base, "verdict": "UNKNOWN", "reason": "planning_scene_revision_drift"}
        base.update({"pure_ik_pass": True, "collision_ik_pass": True, "endpoint_pass": True})
        return {**base, "verdict": "PASS", "reason": "endpoint_qualified"}

    def _workspace_precheck(self, descriptor: object) -> JsonDict:
        """Evaluate the complete batch through the cheap structural gate.

        This produces only L2 evidence.  A PASS here does not claim that IK,
        collision checking, endpoint ordering, or planning ran.
        """

        item = descriptor if isinstance(descriptor, Mapping) else {}
        candidate_id = str(item.get("candidate_id") or "")
        pose_hash = str(item.get("candidate_pose_sha256") or "")
        candidate = item.get("candidate")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        stages = candidate.get("qualification_stages")
        base: JsonDict = {
            "candidate_id": candidate_id,
            "candidate_pose_sha256": pose_hash,
            "execution_started": False,
            "stages": [],
            "workspace_pass": False,
            "pure_ik_pass": False,
            "collision_ik_pass": False,
            "endpoint_pass": False,
            "endpoint_evaluated": False,
            "full_plan_submitted": False,
        }
        if isinstance(candidate.get("compile_parameters"), Mapping):
            base["compile_parameters"] = dict(candidate["compile_parameters"])
        if not candidate_id or not pose_hash or not isinstance(stages, list) or not stages:
            return {
                **base,
                "verdict": "UNKNOWN",
                "reason": "compiled_stages_missing",
            }
        if self.workspace_filter is not None:
            try:
                if not all(
                    self.workspace_filter(target)
                    for target in stages
                    if isinstance(target, Mapping)
                ):
                    return {
                        **base,
                        "verdict": "FAIL",
                        "reason": "workspace_envelope_rejected",
                    }
            except Exception as exc:  # noqa: BLE001
                return {
                    **base,
                    "verdict": "UNKNOWN",
                    "reason": "workspace_filter_error",
                    "error_type": type(exc).__name__,
                }
        base["workspace_pass"] = True
        return {**base, "verdict": "PASS", "reason": "workspace_qualified"}

    def _fast_legality_prechecks(
        self,
        candidates: Sequence[object],
        *,
        purpose: str,
        revision: int,
    ) -> tuple[
        list[JsonDict],
        dict[str, JsonDict],
        list[JsonDict],
        JsonDict,
        object,
    ]:
        """Run the complete cheap breadth barrier before wave-local deep work.

        Placement-goal legality and strict analytic workspace checks are pure,
        inexpensive functions of the frozen scene/candidate packet, so every
        candidate pays them once up front.  Grasp/goal-pair collision geometry
        is deliberately left pending until its candidate reaches a deep wave;
        an early L5 success therefore avoids pair work for the untouched tail.
        """

        descriptors: list[JsonDict] = []
        for fixed_index, raw_descriptor in enumerate(candidates):
            descriptor = dict(raw_descriptor) if isinstance(raw_descriptor, Mapping) else {}
            descriptor["fixed_candidate_index"] = fixed_index
            descriptors.append(descriptor)

        scene: object = {}
        placement_geometry_present = purpose == "placement" and any(
            isinstance(descriptor.get("candidate"), Mapping)
            and any(
                isinstance(descriptor["candidate"].get(key), Mapping)
                for key in (
                    "world_object_goal_pose",
                    "object_goal_pose",
                    "object_goal_world",
                )
            )
            for descriptor in descriptors
        )
        grasp_geometry_present = purpose == "grasp" and any(
            isinstance(descriptor.get("candidate"), Mapping)
            and isinstance(descriptor["candidate"].get("compile_parameters"), Mapping)
            and isinstance(
                descriptor["candidate"]["compile_parameters"].get("camera_pose"),
                Mapping,
            )
            for descriptor in descriptors
        )
        if (placement_geometry_present or grasp_geometry_present) and self.clone_scene is not None:
            try:
                scene = self.clone_scene()
            except Exception as exc:  # noqa: BLE001 - PlanningScene boundary.
                raise _QualificationInfrastructureError(
                    f"placement legality scene clone failed: {type(exc).__name__}"
                ) from exc
            if not isinstance(scene, Mapping):
                raise _QualificationInfrastructureError(
                    "placement legality scene clone returned non-object evidence"
                )
            scene_revision = scene.get("revision")
            if (
                isinstance(scene_revision, int)
                and not isinstance(scene_revision, bool)
                and scene_revision != revision
            ):
                raise _QualificationInfrastructureError("placement legality scene revision drift")

        goal_started = time.monotonic()
        goal_by_id: dict[str, JsonDict] = {}
        if purpose == "placement":
            # This complete barrier is intentional: each AnyPlace object goal
            # is evaluated once before any grasp/goal pair is considered.
            for descriptor in descriptors:
                candidate = descriptor.get("candidate")
                candidate = candidate if isinstance(candidate, Mapping) else {}
                goal_id = str(
                    candidate.get("source_object_goal_id")
                    or candidate.get("id")
                    or descriptor.get("candidate_id")
                    or ""
                )
                if goal_id not in goal_by_id:
                    goal_by_id[goal_id] = evaluate_placement_goal_legality(
                        descriptor,
                        scene=scene,
                    )
        goal_elapsed_s = time.monotonic() - goal_started

        prechecks: dict[str, JsonDict] = {}
        schedulable: list[JsonDict] = []
        for descriptor in descriptors:
            candidate_id = str(descriptor.get("candidate_id") or "")
            candidate = descriptor.get("candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            if purpose == "placement":
                goal_id = str(
                    candidate.get("source_object_goal_id") or candidate.get("id") or candidate_id
                )
                bind_qualified_placement_goal(
                    descriptor,
                    goal_by_id[goal_id],
                )
                candidate = descriptor.get("candidate")
                candidate = candidate if isinstance(candidate, Mapping) else {}
            precheck = self._fast_workspace_precheck(descriptor)
            precheck["fixed_candidate_index"] = int(descriptor.get("fixed_candidate_index", 0))
            if purpose == "placement":
                goal_id = str(
                    candidate.get("source_object_goal_id") or candidate.get("id") or candidate_id
                )
                goal_evidence = json.loads(json.dumps(goal_by_id[goal_id]))
                precheck["goal_legality"] = goal_evidence
                checks = goal_evidence.get("checks")
                region_checks = (
                    checks.get("placement_region") if isinstance(checks, Mapping) else None
                )
                robust_clearance = (
                    region_checks.get(
                        "conservative_minimum_margin_m",
                        region_checks.get("minimum_margin_m"),
                    )
                    if isinstance(region_checks, Mapping)
                    else None
                )
                if (
                    isinstance(robust_clearance, (int, float))
                    and not isinstance(robust_clearance, bool)
                    and math.isfinite(float(robust_clearance))
                ):
                    descriptor["placement_robust_clearance_m"] = float(robust_clearance)
                    precheck["placement_robust_clearance_m"] = float(robust_clearance)
                object_bbox = checks.get("object_bbox") if isinstance(checks, Mapping) else None
                minimum_z = (
                    object_bbox.get("minimum_z_m") if isinstance(object_bbox, Mapping) else None
                )
                maximum_z = (
                    object_bbox.get("maximum_z_m") if isinstance(object_bbox, Mapping) else None
                )
                if all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in (minimum_z, maximum_z)
                ):
                    vertical_extent = max(0.0, float(maximum_z) - float(minimum_z))
                    descriptor["placement_vertical_extent_m"] = vertical_extent
                    precheck["placement_vertical_extent_m"] = vertical_extent
                support_checks = checks.get("support") if isinstance(checks, Mapping) else None
                if isinstance(support_checks, Mapping):
                    for source_key, destination_key in (
                        (
                            "geometry_volume_centroid_height_m",
                            "placement_support_energy_m",
                        ),
                        (
                            "support_energy_resolution_m",
                            "placement_support_energy_resolution_m",
                        ),
                        (
                            "support_face_alignment_cosine",
                            "placement_support_face_alignment_cosine",
                        ),
                        (
                            "support_face_alignment_error_rad",
                            "placement_support_face_alignment_error_rad",
                        ),
                        (
                            "settling_sweep_translation_bound_m",
                            "placement_settling_sweep_translation_bound_m",
                        ),
                        (
                            "settling_sweep_clearance_m",
                            "placement_settling_sweep_clearance_m",
                        ),
                    ):
                        value = support_checks.get(source_key)
                        if (
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                        ):
                            descriptor[destination_key] = float(value)
                            precheck[destination_key] = float(value)
                precheck["goal_legality_pass"] = goal_evidence.get("verdict") == "PASS"
                if goal_evidence.get("verdict") != "PASS":
                    precheck.update(
                        {
                            "workspace_pass": False,
                            "pair_legality_pass": False,
                            "legality_pass": False,
                            "verdict": str(goal_evidence.get("verdict") or "UNKNOWN"),
                            "reason": str(goal_evidence.get("reason") or "goal_legality_rejected"),
                            "infrastructure_error": bool(goal_evidence.get("infrastructure_error")),
                            "pair_legality": {
                                "verdict": "NOT_EVALUATED",
                                "reason": "goal_legality_rejected",
                                "execution_started": False,
                            },
                        }
                    )
                elif precheck.get("verdict") == "PASS":
                    precheck["pair_legality"] = {
                        "schema_version": "openeta.grasp_placement_pair_legality.v1",
                        "candidate_id": candidate_id,
                        "source_grasp_id": str(candidate.get("source_grasp_id") or ""),
                        "source_object_goal_id": goal_id,
                        "verdict": "NOT_EVALUATED",
                        "reason": "pending_candidate_wave",
                        "execution_started": False,
                    }
                    precheck["pair_legality_pass"] = None
                    precheck["legality_pass"] = True
                else:
                    precheck["pair_legality"] = {
                        "verdict": "NOT_EVALUATED",
                        "reason": "structural_workspace_rejected",
                        "execution_started": False,
                    }
                    precheck["pair_legality_pass"] = False
                    precheck["legality_pass"] = False
            else:
                family = grasp_symmetry_family_id(candidate)
                descriptor["grasp_symmetry_family_id"] = family
                precheck["grasp_symmetry_family_id"] = family
                scene_alignment = evaluate_grasp_target_closing_alignment(
                    descriptor,
                    scene=scene,
                )
                precheck["scene_target_closing_alignment"] = json.loads(json.dumps(scene_alignment))
                candidate_copy = dict(candidate)
                candidate_copy["scene_target_closing_alignment"] = json.loads(
                    json.dumps(scene_alignment)
                )
                descriptor["candidate"] = candidate_copy
                precheck["legality_pass"] = precheck.get("verdict") == "PASS"
            prechecks[candidate_id] = precheck
            if precheck.get("verdict") == "PASS":
                schedulable.append(descriptor)

        goal_values = list(goal_by_id.values())
        return (
            descriptors,
            prechecks,
            schedulable,
            {
                "goal_legality_unique_count": len(goal_values),
                "goal_legality_pass_count": sum(
                    evidence.get("verdict") == "PASS" for evidence in goal_values
                ),
                "goal_legality_reject_count": sum(
                    evidence.get("verdict") == "FAIL" for evidence in goal_values
                ),
                "goal_legality_elapsed_s": goal_elapsed_s,
                "pair_legality_candidate_count": (
                    len(descriptors) if purpose == "placement" else 0
                ),
                "pair_legality_reached_count": 0,
                "pair_legality_evaluation_count": 0,
                "pair_legality_shared_count": 0,
                "pair_legality_pass_count": 0,
                "pair_legality_reject_count": 0,
                "pair_legality_pending_count": sum(
                    precheck.get("pair_legality_pass") is None for precheck in prechecks.values()
                ),
                "pair_legality_elapsed_s": 0.0,
                "grasp_scene_alignment_evaluated_count": sum(
                    isinstance(precheck.get("scene_target_closing_alignment"), Mapping)
                    and precheck["scene_target_closing_alignment"].get("evaluated") is True
                    for precheck in prechecks.values()
                ),
                "grasp_scene_alignment_aperture_risk_count": sum(
                    isinstance(precheck.get("scene_target_closing_alignment"), Mapping)
                    and precheck["scene_target_closing_alignment"].get("aperture_feasible") is False
                    for precheck in prechecks.values()
                ),
            },
            scene,
        )

    def _apply_fast_pair_legality_wave(
        self,
        descriptors: Sequence[Mapping[str, Any]],
        *,
        prechecks: dict[str, JsonDict],
        scene: object,
        pair_cache: dict[tuple[str, str], tuple[str, JsonDict]],
        metrics: JsonDict,
    ) -> tuple[tuple[JsonDict, ...], list[JsonDict]]:
        """Advance one deterministic placement wave through its pair gate."""

        started = time.monotonic()
        deep_candidates: list[JsonDict] = []
        rejected: list[JsonDict] = []
        for raw_descriptor in descriptors:
            descriptor = dict(raw_descriptor)
            candidate_id = str(descriptor.get("candidate_id") or "")
            precheck = prechecks[candidate_id]
            already_evaluated = precheck.get("pair_legality_pass")
            if already_evaluated is True:
                deep_candidates.append(descriptor)
                continue
            if already_evaluated is False or precheck.get("verdict") != "PASS":
                rejected.append(dict(precheck))
                continue

            candidate = descriptor.get("candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            goal_id = str(
                candidate.get("source_object_goal_id") or candidate.get("id") or candidate_id
            )
            explicit_family = str(candidate.get("source_grasp_equivalence_id") or "")
            family = explicit_family or str(
                candidate.get("source_grasp_id") or candidate.get("id") or candidate_id
            )
            # Only an explicit host equivalence family may share geometry.
            # Unmarked candidates retain independent evidence even when their
            # provider labels happen to match.
            cache_key = (goal_id, family) if explicit_family else (candidate_id, candidate_id)
            cached = pair_cache.get(cache_key)
            if cached is None:
                pair_evidence = evaluate_grasp_placement_pair_legality(
                    descriptor,
                    scene=scene,
                    workspace_filter=self.workspace_filter,
                )
                pair_evidence["screening_reused"] = False
                pair_evidence["symmetry_equivalence_id"] = family
                pair_cache[cache_key] = (candidate_id, dict(pair_evidence))
                metrics["pair_legality_evaluation_count"] = (
                    int(metrics.get("pair_legality_evaluation_count") or 0) + 1
                )
            else:
                shared_from, shared_evidence = cached
                pair_evidence = json.loads(json.dumps(shared_evidence))
                pair_evidence.update(
                    {
                        "candidate_id": candidate_id,
                        "source_grasp_id": str(candidate.get("source_grasp_id") or ""),
                        "source_object_goal_id": goal_id,
                        "screening_reused": True,
                        "shared_from_candidate_id": shared_from,
                        "symmetry_equivalent": True,
                    }
                )
                metrics["pair_legality_shared_count"] = (
                    int(metrics.get("pair_legality_shared_count") or 0) + 1
                )

            passed = pair_evidence.get("verdict") == "PASS"
            precheck["pair_legality"] = pair_evidence
            precheck["pair_legality_pass"] = passed
            precheck["legality_pass"] = passed
            if passed:
                deep_candidates.append(descriptor)
            else:
                precheck.update(
                    {
                        "verdict": str(pair_evidence.get("verdict") or "UNKNOWN"),
                        "reason": str(pair_evidence.get("reason") or "pair_legality_rejected"),
                        "infrastructure_error": bool(pair_evidence.get("infrastructure_error")),
                    }
                )
                rejected.append(dict(precheck))

        pair_rows = [
            precheck
            for precheck in prechecks.values()
            if precheck.get("pair_legality_pass") is not None
        ]
        metrics.update(
            {
                "pair_legality_reached_count": len(pair_rows),
                "pair_legality_pass_count": sum(
                    precheck.get("pair_legality_pass") is True for precheck in pair_rows
                ),
                "pair_legality_reject_count": sum(
                    isinstance(precheck.get("pair_legality"), Mapping)
                    and precheck["pair_legality"].get("verdict") == "FAIL"
                    for precheck in pair_rows
                ),
                "pair_legality_pending_count": sum(
                    precheck.get("pair_legality_pass") is None for precheck in prechecks.values()
                ),
                "pair_legality_elapsed_s": float(metrics.get("pair_legality_elapsed_s") or 0.0)
                + time.monotonic()
                - started,
            }
        )
        return tuple(deep_candidates), rejected

    def _fast_workspace_precheck(self, descriptor: object) -> JsonDict:
        """Add only mathematical/structural hard rejects to the legacy L2 gate."""

        precheck = self._workspace_precheck(descriptor)
        if precheck.get("verdict") != "PASS":
            return precheck
        item = descriptor if isinstance(descriptor, Mapping) else {}
        candidate = item.get("candidate")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        stages = candidate.get("qualification_stages")
        if not isinstance(stages, list) or any(
            not isinstance(stage, Mapping) or target_pose(stage) is None for stage in stages
        ):
            return {
                **precheck,
                "workspace_pass": False,
                "verdict": "FAIL",
                "reason": "invalid_target_transform",
            }
        return precheck

    def _plan_candidate(
        self,
        descriptor: object,
        revision: int,
        *,
        screen: Mapping[str, Any],
    ) -> JsonDict:
        item = descriptor if isinstance(descriptor, Mapping) else {}
        candidate = item.get("candidate") if isinstance(item.get("candidate"), Mapping) else {}
        stages = candidate.get("qualification_stages") if isinstance(candidate, Mapping) else []
        base = {key: value for key, value in screen.items() if key != "stages"}
        base["stages"] = []
        active_scene_diff: Mapping[str, Any] | None = None
        try:
            candidate_start = candidate.get("qualification_start_joint_state")
            start = dict(
                candidate_start
                if isinstance(candidate_start, Mapping)
                else self.current_joint_state()
            )
            scene = self.clone_scene() if self.clone_scene is not None else None
        except Exception as exc:  # noqa: BLE001
            return {
                **base,
                "verdict": "UNKNOWN",
                "reason": "planning_context_unavailable",
                "error_type": type(exc).__name__,
            }
        initial_transition = candidate.get("initial_scene_transition")
        if initial_transition:
            if self.apply_scene_transition is None or scene is None:
                return {
                    **base,
                    "verdict": "UNKNOWN",
                    "reason": "virtual_scene_transition_unavailable",
                }
            try:
                transition_pose = candidate.get("initial_scene_transition_pose")
                transition_target = dict(
                    transition_pose if isinstance(transition_pose, Mapping) else stages[0]
                )
                predicted_attachment = candidate.get("predicted_attachment_transform")
                if (
                    isinstance(predicted_attachment, Mapping)
                    and candidate.get("physical_scene_attachment_required") is not True
                ):
                    transition_target["attachment_transform"] = dict(predicted_attachment)
                transition_evidence = self.apply_scene_transition(
                    scene, str(initial_transition), transition_target
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    **base,
                    "verdict": "UNKNOWN",
                    "reason": "virtual_scene_transition_error",
                    "error_type": type(exc).__name__,
                }
            if transition_evidence.get("ok") is not True:
                return {
                    **base,
                    "verdict": "FAIL",
                    "reason": "virtual_scene_transition_failed",
                }
            next_diff = transition_evidence.get("planning_scene_diff")
            if isinstance(next_diff, Mapping):
                active_scene_diff = dict(next_diff)
        for index, target in enumerate(stages if isinstance(stages, list) else []):
            evidence = dict(screen.get("stages", [])[index])
            evidence["start_joint_state_sha256"] = _hash(start)
            planning_started = time.monotonic()
            try:
                planning_target = dict(target)
                binding = str(
                    getattr(
                        self,
                        "_active_qualification_binding_sha256",
                        "",
                    )
                )
                if binding:
                    planning_target["_qualification_cache_binding_sha256"] = binding
                allowed_collisions = _contact_allowed_collisions(scene, planning_target)
                if allowed_collisions:
                    planning_target["qualification_allowed_collisions"] = allowed_collisions
                selected_ik_state = evidence.get("end_joint_state")
                if isinstance(selected_ik_state, Mapping):
                    planning_target["qualification_goal_joint_state"] = dict(selected_ik_state)
                    evidence["selected_ik_end_joint_state_sha256"] = _hash(selected_ik_state)
                if active_scene_diff is not None:
                    planning_target["qualification_scene_diff"] = dict(active_scene_diff)
                terminal_gripper_state = target.get("qualification_terminal_gripper_state")
                if terminal_gripper_state:
                    if terminal_gripper_state != "closing_sweep":
                        return self._unknown(
                            base,
                            evidence,
                            "terminal_gripper_state_unsupported",
                        )
                    if not isinstance(selected_ik_state, Mapping):
                        return self._unknown(
                            base,
                            evidence,
                            "terminal_gripper_state_joint_state_missing",
                        )
                    terminal_state = _with_qualification_state_policy(
                        selected_ik_state,
                        target=planning_target,
                        scene_diff=active_scene_diff,
                    )
                    terminal_state["qualification_gripper_state"] = "closing_sweep"
                    try:
                        terminal_result, terminal_retry, terminal_elapsed = self._call_fast_service(
                            lambda: self.check_state_validity(terminal_state),
                            required_boolean="valid",
                        )
                    except _QualificationInfrastructureError as exc:
                        return self._unknown(
                            base,
                            evidence,
                            "terminal_gripper_state_service_error",
                            exc,
                        )
                    evidence["terminal_gripper_state_validity"] = {
                        "requested_gripper_state": "closing_sweep",
                        "joint_state_sha256": _hash(selected_ik_state),
                        "valid": terminal_result.get("valid") is True,
                        "retry_count": terminal_retry,
                        "elapsed_s": terminal_elapsed,
                        "preplan_endpoint_check": True,
                        "seed_independent_static_collision": (
                            terminal_result.get("qualification_seed_independent_static_collision")
                            is True
                        ),
                        "bilateral_target_contact_required": (
                            terminal_result.get("qualification_bilateral_target_contact_required")
                            is True
                        ),
                        "bilateral_target_contact_predicted": (
                            terminal_result.get("qualification_bilateral_target_contact_predicted")
                            is True
                        ),
                        "seed_independent_contact_geometry_failure": (
                            terminal_result.get(
                                "qualification_seed_independent_contact_geometry_failure"
                            )
                            is True
                        ),
                        "reason": terminal_result.get("reason"),
                        "collision_pairs": list(terminal_result.get("collision_pairs") or []),
                        "sweep_checks": list(
                            terminal_result.get("qualification_gripper_sweep_checks") or []
                        ),
                    }
                    if terminal_result.get("valid") is not True:
                        return self._fail(
                            base,
                            evidence,
                            "terminal_gripper_state_invalid",
                        )
                planned = dict(
                    self.plan_only(
                        planning_target,
                        start,
                        PLANNING_TIME_S,
                        PLANNING_ATTEMPTS,
                    )
                )
            except TimeoutError:
                evidence["elapsed_s"] = time.monotonic() - planning_started
                return self._unknown(base, evidence, "plan_only_timeout")
            except Exception as exc:  # noqa: BLE001
                evidence["elapsed_s"] = time.monotonic() - planning_started
                return self._unknown(base, evidence, "plan_only_service_error", exc)
            evidence["moveit_error_code"] = planned.get("moveit_error_code")
            evidence["solver"] = planned.get("solver")
            evidence["l5_trajectory_cache_stored"] = (
                planned.get("l5_trajectory_cache_stored") is True
            )
            if isinstance(planned.get("trajectory_safety"), Mapping):
                evidence["trajectory_safety"] = dict(
                    planned["trajectory_safety"]
                )
            if isinstance(planned.get("l5_trajectory_cache_key"), str):
                evidence["l5_trajectory_cache_key"] = planned["l5_trajectory_cache_key"]
            reported_elapsed = planned.get("elapsed_s")
            evidence["elapsed_s"] = (
                float(reported_elapsed)
                if isinstance(reported_elapsed, (int, float))
                and not isinstance(reported_elapsed, bool)
                and math.isfinite(float(reported_elapsed))
                and float(reported_elapsed) >= 0.0
                else time.monotonic() - planning_started
            )
            planned_margin = planned.get("joint_margin")
            if (
                isinstance(planned_margin, (int, float))
                and not isinstance(planned_margin, bool)
                and math.isfinite(float(planned_margin))
            ):
                evidence["joint_margin"] = float(planned_margin)
            points = planned.get("trajectory_points")
            if planned.get("execution_started") is not False:
                return self._unknown(base, evidence, "plan_only_execution_evidence_missing")
            if planned.get("ok") is not True:
                evidence["collision_pairs"] = list(
                    planned.get("collision_pairs") or evidence.get("collision_pairs") or []
                )
                return self._fail(base, evidence, "plan_only_failed")
            if not isinstance(points, list) or not points:
                return self._unknown(base, evidence, "plan_only_empty_trajectory")
            end_state = planned.get("end_joint_state")
            if not isinstance(end_state, Mapping):
                return self._unknown(base, evidence, "plan_only_end_state_missing")
            evidence.update(
                {
                    "plan_only": True,
                    "execution_started": False,
                    "trajectory": {"point_count": len(points)},
                    "end_joint_state": dict(end_state),
                }
            )
            base["stages"].append(evidence)
            start = dict(end_state)
            transition = target.get("scene_transition") if isinstance(target, Mapping) else None
            if transition:
                transition_evidence: Mapping[str, Any] = {
                    "ok": True,
                    "transition": transition,
                    "virtual": True,
                }
                if self.apply_scene_transition is not None:
                    try:
                        transition_evidence = self.apply_scene_transition(
                            scene, str(transition), target
                        )
                    except Exception as exc:  # noqa: BLE001
                        return {
                            **base,
                            "verdict": "UNKNOWN",
                            "reason": "virtual_scene_transition_error",
                            "error_type": type(exc).__name__,
                        }
                evidence["scene_transition"] = dict(transition_evidence)
                if transition_evidence.get("ok") is not True:
                    return self._fail(base, evidence, "virtual_scene_transition_failed")
                next_diff = transition_evidence.get("planning_scene_diff")
                if isinstance(next_diff, Mapping):
                    active_scene_diff = dict(next_diff)
                post_gripper_state = target.get("qualification_post_transition_gripper_state")
                if post_gripper_state:
                    if post_gripper_state != "open":
                        return self._unknown(
                            base,
                            evidence,
                            "post_transition_gripper_state_unsupported",
                        )
                    post_state = _with_qualification_state_policy(
                        end_state,
                        target={},
                        scene_diff=active_scene_diff,
                    )
                    post_state["qualification_gripper_state"] = "open"
                    try:
                        post_result, post_retry, post_elapsed = self._call_fast_service(
                            lambda: self.check_state_validity(post_state),
                            required_boolean="valid",
                        )
                    except _QualificationInfrastructureError as exc:
                        return self._unknown(
                            base,
                            evidence,
                            "post_transition_gripper_state_service_error",
                            exc,
                        )
                    evidence["post_transition_gripper_state_validity"] = {
                        "requested_gripper_state": "open",
                        "joint_state_sha256": _hash(end_state),
                        "valid": post_result.get("valid") is True,
                        "retry_count": post_retry,
                        "elapsed_s": post_elapsed,
                        "collision_pairs": list(post_result.get("collision_pairs") or []),
                    }
                    if post_result.get("valid") is not True:
                        return self._fail(
                            base,
                            evidence,
                            "post_transition_gripper_state_invalid",
                        )
        if self.scene_revision() != revision:
            return {**base, "verdict": "UNKNOWN", "reason": "planning_scene_revision_drift"}
        return {**base, "verdict": "PASS", "reason": "qualified"}

    def _try_ik(
        self,
        target: Mapping[str, Any],
        seeds: Sequence[Mapping[str, Any]],
        collision: bool,
    ) -> tuple[JsonDict, list[JsonDict], str]:
        deadline = time.monotonic() + KINEMATIC_IK_TIMEOUT_S
        attempts: list[JsonDict] = []
        for index, seed in enumerate(seeds):
            remaining_budget = deadline - time.monotonic()
            if remaining_budget <= 0:
                return {}, attempts, "collision_ik_timeout" if collision else "kinematic_ik_timeout"
            remaining_seeds = max(1, len(seeds) - index)
            seeded_target = dict(target)
            seeded_target["ik_seed_timeout_s"] = max(0.001, remaining_budget / remaining_seeds)
            try:
                result = dict(self.compute_ik(seeded_target, seed, collision))
            except TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                attempts.append(
                    {"seed_sha256": _hash(seed), "ok": False, "error_type": type(exc).__name__}
                )
                continue
            attempts.append(
                {
                    "seed_sha256": _hash(seed),
                    "ok": result.get("ok") is True,
                    "moveit_error_code": result.get("moveit_error_code"),
                    "solver": result.get("solver"),
                    "elapsed_s": result.get("elapsed_s"),
                }
            )
            if result.get("ok") is True:
                return result, attempts, ""
        return {"ok": False}, attempts, ""

    def _try_collision_ik(
        self,
        target: Mapping[str, Any],
        seeds: Sequence[Mapping[str, Any]],
        *,
        active_scene_diff: Mapping[str, Any] | None,
    ) -> tuple[JsonDict, list[JsonDict], str]:
        deadline = time.monotonic() + KINEMATIC_IK_TIMEOUT_S
        attempts: list[JsonDict] = []
        collision_pairs: list[Any] = []
        saw_state_invalid = False
        saw_state_missing = False
        saw_validity_timeout = False
        saw_validity_error = False
        for index, seed in enumerate(seeds):
            remaining_budget = deadline - time.monotonic()
            if remaining_budget <= 0:
                return {}, attempts, "collision_ik_timeout"
            remaining_seeds = max(1, len(seeds) - index)
            seeded_target = dict(target)
            seeded_target["ik_seed_timeout_s"] = max(0.001, remaining_budget / remaining_seeds)
            attempt: JsonDict = {"seed_sha256": _hash(seed), "ok": False}
            try:
                result = dict(self.compute_ik(seeded_target, seed, True))
            except TimeoutError:
                attempt["error_type"] = "TimeoutError"
                attempts.append(attempt)
                continue
            except Exception as exc:  # noqa: BLE001
                attempt["error_type"] = type(exc).__name__
                attempts.append(attempt)
                continue
            attempt.update(
                {
                    "ik_solution": result.get("ok") is True,
                    "moveit_error_code": result.get("moveit_error_code"),
                    "solver": result.get("solver"),
                    "elapsed_s": result.get("elapsed_s"),
                }
            )
            result_pairs = list(result.get("collision_pairs") or [])
            if result_pairs:
                collision_pairs = result_pairs
                attempt["collision_pairs"] = result_pairs
            if result.get("ok") is not True:
                attempts.append(attempt)
                continue
            state = result.get("joint_state")
            if not isinstance(state, Mapping):
                saw_state_missing = True
                attempt["state_evidence"] = "missing"
                attempts.append(attempt)
                continue
            validity_state = _with_qualification_state_policy(
                state,
                target=target,
                scene_diff=active_scene_diff,
            )
            try:
                validity = dict(self.check_state_validity(validity_state))
            except TimeoutError:
                saw_validity_timeout = True
                attempt["state_validity_error"] = "timeout"
                attempts.append(attempt)
                continue
            except Exception as exc:  # noqa: BLE001
                saw_validity_error = True
                attempt["state_validity_error"] = "service_error"
                attempt["state_validity_error_type"] = type(exc).__name__
                attempts.append(attempt)
                continue
            valid = validity.get("valid") is True
            attempt["state_valid"] = valid
            attempt["collision_pairs"] = list(validity.get("collision_pairs") or [])
            if not valid:
                saw_state_invalid = True
                if attempt["collision_pairs"]:
                    collision_pairs = list(attempt["collision_pairs"])
                attempts.append(attempt)
                continue
            attempt["ok"] = True
            attempts.append(attempt)
            return (
                {
                    **result,
                    "state_validity": validity,
                },
                attempts,
                "",
            )
        if saw_validity_timeout:
            return {}, attempts, "collision_state_validity_timeout"
        if saw_validity_error:
            return {}, attempts, "collision_state_validity_service_error"
        if saw_state_missing:
            return {}, attempts, "collision_ik_evidence_missing"
        return (
            {
                "ok": False,
                "all_solutions_state_invalid": saw_state_invalid,
                "collision_pairs": collision_pairs,
            },
            attempts,
            "",
        )

    @staticmethod
    def _ik_seeds(
        current: Mapping[str, Any],
        *,
        count: int,
        candidate_id: str,
        stage_index: int,
        source: Mapping[str, Any],
        previous_solution: Mapping[str, Any] | None,
    ) -> list[JsonDict]:
        names = list(current.get("names") or [])
        positions = [float(value) for value in current.get("positions") or []]
        seed_values: list[Mapping[str, Any]] = [current]
        for value in (
            source.get("home_joint_state") or current.get("home_joint_state"),
            previous_solution,
        ):
            if isinstance(value, Mapping):
                seed_values.append(value)
        reservoir = source.get("successful_ik_reservoir")
        if isinstance(reservoir, list):
            seed_values.extend(value for value in reservoir if isinstance(value, Mapping))
        seeds = _unique_joint_state_seeds(seed_values, limit=count)
        limits = source.get("joint_limits") or current.get("joint_limits")
        lower = [-math.pi] * len(positions)
        upper = [math.pi] * len(positions)
        if isinstance(limits, Mapping):
            lower_value, upper_value = limits.get("lower"), limits.get("upper")
            if (
                isinstance(lower_value, list)
                and isinstance(upper_value, list)
                and len(lower_value) == len(upper_value) == len(positions)
            ):
                lower = [float(value) for value in lower_value]
                upper = [float(value) for value in upper_value]
        rng = random.Random(int(_hash({"candidate": candidate_id, "stage": stage_index})[:16], 16))
        duplicate_attempts = 0
        while positions and len(seeds) < max(1, count):
            generated = {
                "names": names,
                "positions": [rng.uniform(lo, hi) for lo, hi in zip(lower, upper)],
            }
            updated = _unique_joint_state_seeds([*seeds, generated], limit=count)
            if len(updated) == len(seeds):
                duplicate_attempts += 1
                if duplicate_attempts >= max(1, count):
                    break
            else:
                duplicate_attempts = 0
                seeds = updated
        return seeds[: max(1, count)]

    @staticmethod
    def _fail(base: JsonDict, evidence: JsonDict, reason: str) -> JsonDict:
        base["stages"].append(evidence)
        return {**base, "verdict": "FAIL", "reason": reason}

    @staticmethod
    def _unknown(
        base: JsonDict,
        evidence: JsonDict,
        reason: str,
        exc: Exception | None = None,
    ) -> JsonDict:
        base["stages"].append(evidence)
        result = {**base, "verdict": "UNKNOWN", "reason": reason}
        if exc is not None:
            result["error_type"] = type(exc).__name__
        return result
