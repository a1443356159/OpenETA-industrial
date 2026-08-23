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
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict
from agent.tools.registry import ToolResult
from tools.candidate_config import (
    DEFAULT_ANYPLACE_DIVERSITY_POOL_SIZE,
    DEFAULT_ANYPLACE_FULL_PLAN_LIMIT,
    DEFAULT_GRASP_DIVERSITY_POOL_SIZE,
    DEFAULT_GRASP_FULL_PLAN_LIMIT,
    DEFAULT_MOVEIT_IK_SEED_COUNT,
    DEFAULT_PREGRASP_JOINT_FULL_PLAN_LIMIT,
)


QUALIFICATION_SCHEMA = "openeta.moveit_candidate_funnel.v2"
PRIVATE_RPC_NAME = "qualify_motion_candidates"
PLANNING_TIME_S = 30.0
PLANNING_ATTEMPTS = 3
KINEMATIC_IK_TIMEOUT_S = 2.0
STATE_VALIDITY_TIMEOUT_S = 2.0
QUALIFICATION_RPC_GRACE_S = 30.0
PARALLEL_GRIPPER_TARGET_ALIGNMENT_SCHEMA = (
    "openeta.parallel_gripper_target_closing_alignment.v1"
)
PROGRESSIVE_SCREENING_MODE = "progressive_until_full_plan_capacity"
PROGRESSIVE_NOT_EVALUATED_REASON = "progressive_endpoint_capacity_reached"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _unique_joint_state_seeds(
    values: Sequence[Mapping[str, Any]], *, limit: int
) -> list[JsonDict]:
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
        if not entry or (
            scene_epoch is not None and entry["scene_epoch"] != scene_epoch
        ):
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
        grasp_full_plan_limit: int = DEFAULT_GRASP_FULL_PLAN_LIMIT,
        placement_full_plan_limit: int = DEFAULT_ANYPLACE_FULL_PLAN_LIMIT,
        pregrasp_joint_full_plan_limit: int = DEFAULT_PREGRASP_JOINT_FULL_PLAN_LIMIT,
        ik_seed_count: int = DEFAULT_MOVEIT_IK_SEED_COUNT,
        placement_max_rounds: int = 2,
    ) -> None:
        self.rpc = rpc
        self.cache = cache or QualificationCache()
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None
        self.compile_candidate = compile_candidate
        self.diversity_limits = {
            "grasp": int(grasp_diversity_limit),
            "placement": int(placement_diversity_limit),
        }
        self.full_plan_limits = {
            "grasp": int(grasp_full_plan_limit),
            "placement": int(placement_full_plan_limit),
        }
        self.pregrasp_joint_full_plan_limit = int(pregrasp_joint_full_plan_limit)
        self.ik_seed_count = int(ik_seed_count)
        self.placement_max_rounds = int(placement_max_rounds)
        self._placement_rounds: dict[str, int] = {}

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
    ) -> ToolResult:
        if not result.success:
            return result
        details = result.details
        key = "placement_candidates" if purpose == "placement" else "grasp_candidates"
        raw = details.get(key)
        if not isinstance(raw, list) or not raw:
            return result
        if purpose == "grasp":
            augmented_candidates: list[JsonDict] = []
            for candidate in raw:
                if not isinstance(candidate, Mapping):
                    continue
                gripper_name = str(candidate.get("gripper_name") or "").lower()
                parallel_gripper = (
                    "robotiq" in gripper_name or "parallel" in gripper_name
                )
                base_candidate = dict(candidate)
                if parallel_gripper:
                    try:
                        base_candidate = parallel_gripper_centering_variant(candidate)
                    except ValueError:
                        # Sensor evidence is optional.  With no valid aligned
                        # mask/depth correction, retain the unmodified model
                        # pose and let the existing funnel fail closed.
                        pass
                augmented_candidates.append(base_candidate)
                if parallel_gripper:
                    try:
                        augmented_candidates.append(
                            parallel_gripper_symmetry_variant(base_candidate)
                        )
                    except ValueError:
                        pass
                    try:
                        reversal = parallel_gripper_approach_reversal_variant(
                            base_candidate
                        )
                        augmented_candidates.extend(
                            [
                                reversal,
                                parallel_gripper_symmetry_variant(reversal),
                            ]
                        )
                    except ValueError:
                        pass
            raw = augmented_candidates
            details[key] = raw
        qualification_round = 1
        if purpose == "placement" and qualification_mode == "standard":
            pool_key = _hash(
                {
                    "placement_observation": (source or {}).get("placement_observation"),
                    "attachment_transform": (source or {}).get("attachment_transform"),
                    "planning_scene_revision": planning_scene_revision,
                }
            )
            qualification_round = self._placement_rounds.get(pool_key, 0) + 1
            self._placement_rounds[pool_key] = qualification_round
            augmented = (
                poisson_disk_placement_augment(raw, seed_key=pool_key)
                if qualification_round > 1
                else []
            )
            # Each inference round is an independent qualification pool.  Keep
            # the frozen observation/attachment binding for round accounting,
            # but do not spend a later round requalifying candidates from an
            # earlier zero-PASS round.
            raw = _deduplicate_se3_candidates(
                [*raw, *augmented],
                round_index=qualification_round,
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
                rpc_candidate["qualification_stages"] = list(
                    compiled["qualification_stages"]
                )
            if isinstance(compiled.get("compile_parameters"), Mapping):
                rpc_candidate["compile_parameters"] = dict(
                    compiled["compile_parameters"]
                )
            compiled_descriptors.append(
                {
                    "candidate_id": str(candidate.get("id") or ""),
                    "candidate_pose_sha256": _hash(candidate),
                    "candidate": rpc_candidate,
                }
            )
        if qualification_mode == "pregrasp_joint":
            # Every constructed grasp/object-goal pair reaches compilation and
            # the conservative structural precheck.  The private helper then
            # traverses L3/L4 in stable round-robin order only until it fills
            # the complete L5 submission capacity.  Unvisited pairs remain
            # explicitly NOT_EVALUATED; they are never relabelled as failures.
            request_candidates = compiled_descriptors
            full_plan_limit = self.pregrasp_joint_full_plan_limit
        else:
            request_candidates = diversify_compiled_candidates(
                compiled_descriptors,
                purpose=purpose,
                limit=self.diversity_limits[purpose],
            )
            full_plan_limit = self.full_plan_limits[purpose]
        request: JsonDict = {
            "schema_version": QUALIFICATION_SCHEMA,
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
            "funnel": {
                "ik_seed_count": self.ik_seed_count,
                "full_plan_limit": full_plan_limit,
                "screening_mode": PROGRESSIVE_SCREENING_MODE,
                # This is derived from L5 capacity.  L1/L2 still cover the
                # complete submitted batch; only candidates with no remaining
                # chance to reach L5 skip the expensive L3/L4 tail.
                "endpoint_pass_target": full_plan_limit,
            },
            "source": dict(source or {}),
            "candidates": request_candidates,
        }
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
        try:
            response = self.rpc(
                PRIVATE_RPC_NAME,
                request,
                _qualification_rpc_timeout_s(
                    request_candidates,
                    full_plan_limit=full_plan_limit,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - private transport boundary.
            response = {
                "schema_version": QUALIFICATION_SCHEMA,
                "planning_scene_revision": planning_scene_revision,
                "execution_started": False,
                "results": [
                    {
                        "candidate_id": item["candidate_id"],
                        "candidate_pose_sha256": item["candidate_pose_sha256"],
                        "verdict": "UNKNOWN",
                        "reason": "qualification_rpc_error",
                        "error_type": type(exc).__name__,
                        "execution_started": False,
                        "qualification_binding_sha256": request[
                            "qualification_binding_sha256"
                        ],
                    }
                    for item in request_candidates
                ],
            }
        evidence = self._validate_response(request, response)
        for item in compile_rejections:
            item["qualification_binding_sha256"] = request[
                "qualification_binding_sha256"
            ]
            evidence["results"].append(item)
        proofs = {item["candidate_id"]: item for item in evidence["results"]}
        passed = [
            dict(candidate)
            for candidate in raw
            if isinstance(candidate, Mapping)
            and proofs.get(str(candidate.get("id") or ""), {}).get("verdict") == "PASS"
        ]
        pass_proofs = {
            str(candidate["id"]): proofs[str(candidate["id"])] for candidate in passed
        }
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
            _public_qualification_summary(evidence)
            if artifact is not None
            else evidence
        )
        details.update(
            {
                "selection_required": bool(passed),
                "qualification_round": qualification_round,
                "max_qualification_rounds": self.placement_max_rounds if purpose == "placement" else 1,
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
                    item.get("endpoint_evaluated") is True
                    for item in evidence["results"]
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
            and response.get("schema_version") == QUALIFICATION_SCHEMA
            and response.get("planning_scene_revision")
            == request["planning_scene_revision"]
            and response.get("execution_started") is False
            and isinstance(raw_results, list)
        )
        received = {
            str(item.get("candidate_id") or ""): item
            for item in raw_results or []
            if isinstance(item, Mapping)
        }
        normalized: list[JsonDict] = []
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
            if verdict == "PASS" and not valid_pass:
                verdict, reason = "UNKNOWN", "qualification_evidence_incomplete"
            elif verdict not in {"PASS", "FAIL", "UNKNOWN", "NOT_EVALUATED"} or not valid_identity:
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
            compiled_candidate = descriptor.get("candidate")
            if isinstance(compiled_candidate, Mapping) and isinstance(
                compiled_candidate.get("compile_parameters"), Mapping
            ):
                record["compile_parameters"] = dict(
                    compiled_candidate["compile_parameters"]
                )
            else:
                record.pop("compile_parameters", None)
            normalized.append(record)
        return {
            "schema_version": QUALIFICATION_SCHEMA,
            "purpose": request["purpose"],
            "scene_epoch": request["scene_epoch"],
            "planning_scene_revision": request["planning_scene_revision"],
            "planning": dict(request["planning"]),
            "funnel": dict(request.get("funnel") or {}),
            "qualification_binding_sha256": request[
                "qualification_binding_sha256"
            ],
            "execution_started": False,
            "results": normalized,
        }

    def _write_artifact(self, evidence: JsonDict) -> JsonDict | None:
        if self.artifact_root is None:
            return None
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        path = self.artifact_root / f"qualification-{_hash(evidence)[:16]}.json"
        path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "type": "moveit_candidate_qualification",
            "kind": "json",
            "path": str(path),
            "sha256": _hash(evidence),
        }


def _public_qualification_summary(evidence: Mapping[str, Any]) -> JsonDict:
    """Return VLM-safe funnel proof metadata; exact per-candidate proof is an artifact."""

    results = evidence.get("results")
    results = results if isinstance(results, list) else []
    verdict_counts = Counter(
        str(item.get("verdict") or "UNKNOWN")
        for item in results
        if isinstance(item, Mapping)
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
        "qualification_binding_sha256": evidence.get(
            "qualification_binding_sha256"
        ),
        "execution_started": False,
        "result_count": len(results),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
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
        "workspace": sum(passed(item, "workspace_pass", lambda stages: bool(stages)) for item in results),
        "pure_ik": sum(passed(item, "pure_ik_pass", lambda stages: bool(stages) and all(s.get("kinematic_ik") is True for s in stages if isinstance(s, Mapping))) for item in results),
        "collision_ik": sum(passed(item, "collision_ik_pass", lambda stages: bool(stages) and all(s.get("collision_ik") is True for s in stages if isinstance(s, Mapping))) for item in results),
        "endpoint": sum(passed(item, "endpoint_pass", lambda stages: bool(stages) and all(s.get("state_valid") is True for s in stages if isinstance(s, Mapping))) for item in results),
        "submitted": sum(item.get("full_plan_submitted") is True or item.get("verdict") == "PASS" for item in results),
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
) -> float:
    """Cover screening plus the bounded full-plan tail at the transport layer.

    The per-service IK and state-validity deadlines remain unchanged.  This
    outer deadline must not expire first merely because the diversity pool is
    larger than the full-plan pool.
    """

    stage_counts: list[int] = []
    for item in candidates:
        candidate = item.get("candidate")
        stages = (
            candidate.get("qualification_stages")
            if isinstance(candidate, Mapping)
            else None
        )
        stage_counts.append(len(stages) if isinstance(stages, list) and stages else 1)
    max_stage_count = max(stage_counts, default=1)
    screening_budget_s = len(candidates) * 2.0 * (
        KINEMATIC_IK_TIMEOUT_S + STATE_VALIDITY_TIMEOUT_S
    )
    planning_budget_s = (
        max(0, int(full_plan_limit)) * max_stage_count * PLANNING_TIME_S
    )
    return screening_budget_s + planning_budget_s + QUALIFICATION_RPC_GRACE_S


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
            quat = [0.25 * scale, (m[0][1] + m[1][0]) / scale, (m[0][2] + m[2][0]) / scale, (m[2][1] - m[1][2]) / scale]
        elif index == 1:
            scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
            quat = [(m[0][1] + m[1][0]) / scale, 0.25 * scale, (m[1][2] + m[2][1]) / scale, (m[0][2] - m[2][0]) / scale]
        else:
            scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
            quat = [(m[0][2] + m[2][0]) / scale, (m[1][2] + m[2][1]) / scale, 0.25 * scale, (m[1][0] - m[0][1]) / scale]
    result["quat_xyzw"] = quat
    return result


def parallel_gripper_centering_variant(candidate: Mapping[str, Any]) -> JsonDict:
    """Translate a parallel-jaw grasp onto its measured target midplane.

    Only translation along GraspNet local +Y (the jaw closing axis) is
    permitted.  The estimator's approach, wrist rotation, insertion depth and
    opening width are preserved.  The correction must come from the selected
    target's aligned mask/depth evidence and remains auditable on the derived
    candidate.
    """

    variant = json.loads(json.dumps(dict(candidate)))
    evidence = variant.get("target_closing_alignment")
    rotation = variant.get("rotation_matrix")
    translation = variant.get("translation_xyz")
    tip = variant.get("gripper_tip_position_xyz")
    if not (
        isinstance(evidence, Mapping)
        and evidence.get("schema_version")
        == PARALLEL_GRIPPER_TARGET_ALIGNMENT_SCHEMA
        and evidence.get("source") == "aligned_selected_mask_depth"
        and evidence.get("depth_provenance")
        in {"sensor_depth", "sensor_safety_depth"}
        and evidence.get("closing_axis") == "graspnet_local_y"
        and isinstance(rotation, list)
        and len(rotation) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in rotation)
        and isinstance(translation, list)
        and len(translation) == 3
        and isinstance(tip, list)
        and len(tip) == 3
    ):
        raise ValueError("parallel-gripper centering requires aligned target evidence")
    try:
        rotation_values = [[float(value) for value in row] for row in rotation]
        translation_values = [float(value) for value in translation]
        tip_values = [float(value) for value in tip]
        correction = float(evidence["correction_m"])
        correction_vector = [
            float(value) for value in evidence["correction_camera_xyz"]
        ]
        target_span = float(evidence["target_span_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("parallel-gripper centering evidence is malformed") from exc
    expected_vector = [correction * row[1] for row in rotation_values]
    if not (
        all(
            math.isfinite(value)
            for value in (
                correction,
                target_span,
                *translation_values,
                *tip_values,
                *correction_vector,
                *(value for row in rotation_values for value in row),
            )
        )
        and target_span > 0.0
        and len(correction_vector) == 3
        and all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
            for actual, expected in zip(correction_vector, expected_vector)
        )
    ):
        raise ValueError("parallel-gripper centering evidence is inconsistent")

    centered_translation = [
        value + correction_vector[index]
        for index, value in enumerate(translation_values)
    ]
    centered_tip = [
        value + correction_vector[index] for index, value in enumerate(tip_values)
    ]
    variant["translation_xyz"] = centered_translation
    variant["gripper_tip_position_xyz"] = centered_tip
    transform = variant.get("transform_matrix")
    if transform is not None:
        if not (
            isinstance(transform, list)
            and len(transform) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in transform)
        ):
            raise ValueError("parallel-gripper centering transform is malformed")
        for row_index in range(3):
            if not math.isclose(
                float(transform[row_index][3]),
                translation_values[row_index],
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("parallel-gripper centering transform is inconsistent")
            transform[row_index][3] = centered_translation[row_index]
    variant.pop("model_native_grasp_pose", None)
    variant["id"] = f"{candidate.get('id', 'grasp')}_closing_centered"
    variant["centering_parent_id"] = str(candidate.get("id") or "")
    variant["centering_parent_provenance"] = str(
        candidate.get("provenance") or candidate.get("source_model") or ""
    )
    variant["centering_transform"] = "target_mask_depth_closing_midplane"
    variant["centering_offset_m"] = correction
    variant["target_closing_span_m"] = target_span
    variant["provenance"] = "host_parallel_gripper_closing_centering"
    return variant


def parallel_gripper_symmetry_variant(candidate: Mapping[str, Any]) -> JsonDict:
    """Return the provenance-marked 180° local approach-axis grasp symmetry."""

    variant = json.loads(json.dumps(dict(candidate)))
    rotation = variant.get("rotation_matrix")
    if not (
        isinstance(rotation, list)
        and len(rotation) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in rotation)
    ):
        raise ValueError("parallel-gripper symmetry requires a 3x3 rotation_matrix")
    # GraspNet's approach vector is local +X (the compiler reads matrix column
    # zero). Right multiplication by Rx(pi) therefore preserves approach and
    # flips the complete closing/binormal frame. This is a true SO(3) wrist
    # roll variant, not an Euler rewrite or a reversed approach.
    variant["rotation_matrix"] = [
        [float(row[0]), -float(row[1]), -float(row[2])] for row in rotation
    ]
    transform = variant.get("transform_matrix")
    if (
        isinstance(transform, list)
        and len(transform) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in transform)
    ):
        for row_index in range(3):
            transform[row_index][:3] = list(
                variant["rotation_matrix"][row_index]
            )
    variant["id"] = f"{candidate.get('id', 'grasp')}_sym180"
    variant["symmetry_parent_id"] = str(candidate.get("id") or "")
    variant["symmetry_parent_provenance"] = str(candidate.get("provenance") or "")
    variant["symmetry_transform"] = "graspnet_local_x_approach_axis_180deg"
    variant["provenance"] = "host_parallel_gripper_symmetry"
    return variant


def parallel_gripper_approach_reversal_variant(
    candidate: Mapping[str, Any],
) -> JsonDict:
    """Reverse approach about the same fingertip center and closing axis.

    GraspNet local +X points from the grasp origin to the fingertip center and
    local +Y is the parallel-jaw closing axis. Right multiplication by Ry(pi)
    reverses approach while preserving that closing axis. Translating the new
    origin from the unchanged fingertip center keeps the antipodal contact
    location fixed; all collision and reachability claims remain delegated to
    the host MoveIt funnel.
    """

    variant = json.loads(json.dumps(dict(candidate)))
    rotation = variant.get("rotation_matrix")
    tip = variant.get("gripper_tip_position_xyz")
    depth = variant.get("depth")
    if not (
        isinstance(rotation, list)
        and len(rotation) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in rotation)
        and isinstance(tip, list)
        and len(tip) == 3
        and isinstance(depth, (int, float))
        and not isinstance(depth, bool)
        and math.isfinite(float(depth))
        and float(depth) > 0.0
    ):
        raise ValueError("parallel-gripper approach reversal requires rotation, tip, and depth")
    reversed_rotation = [
        [-float(row[0]), float(row[1]), -float(row[2])] for row in rotation
    ]
    reversed_translation = [
        float(tip[index]) - float(depth) * reversed_rotation[index][0]
        for index in range(3)
    ]
    variant["rotation_matrix"] = reversed_rotation
    variant["translation_xyz"] = reversed_translation
    transform = variant.get("transform_matrix")
    if (
        isinstance(transform, list)
        and len(transform) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in transform)
    ):
        for row_index in range(3):
            transform[row_index][:3] = list(reversed_rotation[row_index])
            transform[row_index][3] = reversed_translation[row_index]
    variant.pop("model_native_grasp_pose", None)
    variant["id"] = f"{candidate.get('id', 'grasp')}_approach180"
    variant["approach_reversal_parent_id"] = str(candidate.get("id") or "")
    variant["approach_reversal_transform"] = "graspnet_local_y_closing_axis_180deg"
    variant["provenance"] = "host_parallel_gripper_approach_reversal"
    return variant


def poisson_disk_placement_augment(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed_key: str,
    minimum_spacing_m: float = 0.025,
    max_samples: int = 32,
) -> list[JsonDict]:
    """Sample stable poses only inside explicitly supplied eroded footprints."""

    rng = random.Random(int(_hash({"placement_poisson": seed_key})[:16], 16))
    samples: list[JsonDict] = []
    xy_points: list[tuple[float, float]] = []
    stable = [
        item
        for item in candidates
        if item.get("stable_pose") is True
        and isinstance(item.get("footprint_erosion_bounds"), Mapping)
    ]
    attempts = max_samples * 30
    for attempt in range(attempts):
        if not stable or len(samples) >= max_samples:
            break
        parent = stable[attempt % len(stable)]
        bounds = parent["footprint_erosion_bounds"]
        try:
            xmin, xmax = float(bounds["xmin"]), float(bounds["xmax"])
            ymin, ymax = float(bounds["ymin"]), float(bounds["ymax"])
        except (KeyError, TypeError, ValueError):
            continue
        if not xmin <= xmax or not ymin <= ymax:
            continue
        xy = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
        if any(math.hypot(xy[0] - x, xy[1] - y) < minimum_spacing_m for x, y in xy_points):
            continue
        transform = (
            parent.get("object_goal_world")
            or parent.get("object_goal_pose")
            or parent.get("object_placement_transform")
        )
        if not isinstance(transform, Mapping):
            continue
        clone = json.loads(json.dumps(dict(parent)))
        clone_transform = (
            clone.get("object_goal_world")
            or clone.get("object_goal_pose")
            or clone.get("object_placement_transform")
        )
        translation = (
            clone_transform.get("translation_xyz")
            if isinstance(clone_transform, dict)
            else None
        )
        matrix = (
            clone_transform.get("transform_matrix")
            if isinstance(clone_transform, dict)
            else None
        )
        if isinstance(translation, list) and len(translation) == 3:
            translation[0], translation[1] = xy
        elif isinstance(matrix, list) and len(matrix) == 4:
            matrix[0][3], matrix[1][3] = xy
        else:
            continue
        clone["id"] = f"placement_poisson_{len(samples):03d}"
        clone["provenance"] = "host_footprint_eroded_poisson_disk"
        clone["poisson_seed_key_sha256"] = _hash(seed_key)
        xy_points.append(xy)
        samples.append(clone)
    return samples


def declared_object_symmetry_variants(
    candidate: Mapping[str, Any],
    symmetry_group: Sequence[Sequence[Sequence[float]]] | None,
) -> list[JsonDict]:
    """Generate object-goal variants only from an explicitly declared group."""

    if not symmetry_group:
        return []
    transform = (
        candidate.get("object_goal_world")
        or candidate.get("object_goal_pose")
        or candidate.get("object_placement_transform")
    )
    matrix = transform.get("transform_matrix") if isinstance(transform, Mapping) else None
    pose_form = False
    if not (isinstance(matrix, list) and len(matrix) == 4):
        rotation = transform.get("rotation_matrix") if isinstance(transform, Mapping) else None
        translation = transform.get("translation_xyz") if isinstance(transform, Mapping) else None
        if not (
            isinstance(rotation, list)
            and len(rotation) == 3
            and isinstance(translation, list)
            and len(translation) == 3
        ):
            raise ValueError("object symmetry requires a complete object-goal transform")
        matrix = [
            [*map(float, rotation[row]), float(translation[row])]
            for row in range(3)
        ] + [[0.0, 0.0, 0.0, 1.0]]
        pose_form = True
    variants: list[JsonDict] = []
    for index, symmetry in enumerate(symmetry_group):
        if not (
            len(symmetry) == 4
            and all(len(row) == 4 for row in symmetry)
        ):
            raise ValueError("declared object symmetry must contain 4x4 transforms")
        composed = [
            [sum(float(matrix[i][k]) * float(symmetry[k][j]) for k in range(4)) for j in range(4)]
            for i in range(4)
        ]
        clone = json.loads(json.dumps(dict(candidate)))
        clone_transform = clone.get("object_goal_world") or clone.get("object_placement_transform")
        if pose_form:
            clone_transform = clone.get("object_goal_pose")
            clone_transform["rotation_matrix"] = [row[:3] for row in composed[:3]]
            clone_transform["translation_xyz"] = [row[3] for row in composed[:3]]
        else:
            clone_transform["transform_matrix"] = composed
        clone["id"] = f"{candidate.get('id', 'placement')}_sym{index:02d}"
        clone["symmetry_parent_id"] = str(candidate.get("id") or "")
        clone["provenance"] = "declared_object_symmetry_group"
        variants.append(clone)
    return variants


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
        object_bonus = 0.25 if ca.get("stable_contact_face") != cb.get("stable_contact_face") else 0.0
    return translation * 10.0 + angle + source_bonus + object_bonus


def private_qualification_rpc(
    transport: Any,
    *,
    handle_provider: Callable[[], str],
    session_id_provider: Callable[[], str],
) -> Callable[[str, JsonDict, float], JsonDict]:
    """Bind session identity without exposing this RPC in the AgentTool registry."""

    def call(name: str, request: JsonDict, timeout_s: float) -> JsonDict:
        arguments = dict(request)
        arguments["handle"] = handle_provider()
        arguments["session_id"] = session_id_provider()
        return transport.call_tool(name, arguments, timeout_s=timeout_s)

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
    plan_only: Callable[
        [Mapping[str, Any], Mapping[str, Any], float, int], Mapping[str, Any]
    ]
    workspace_filter: Callable[[Mapping[str, Any]], bool] | None = None
    clone_scene: Callable[[], Any] | None = None
    apply_scene_transition: Callable[[Any, str, Mapping[str, Any]], Mapping[str, Any]] | None = None

    def qualify(self, request: Mapping[str, Any]) -> JsonDict:
        revision = request.get("planning_scene_revision")
        candidates = request.get("candidates")
        if (
            request.get("schema_version") != QUALIFICATION_SCHEMA
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
            workspace_prechecks = [
                self._workspace_precheck(item) for item in candidates
            ]
        endpoint_passes = 0
        for index, item in enumerate(candidates):
            workspace_precheck = (
                workspace_prechecks[index]
                if workspace_prechecks is not None
                else None
            )
            if (
                workspace_precheck is not None
                and workspace_precheck.get("verdict") != "PASS"
            ):
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
            "schema_version": QUALIFICATION_SCHEMA,
            "planning_scene_revision": revision,
            "execution_started": False,
            "results": results,
        }

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
                if not all(self.workspace_filter(target) for target in stages if isinstance(target, Mapping)):
                    return {**base, "verdict": "FAIL", "reason": "workspace_envelope_rejected"}
            except Exception as exc:  # noqa: BLE001
                return {**base, "verdict": "UNKNOWN", "reason": "workspace_filter_error", "error_type": type(exc).__name__}
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
                    transition_pose
                    if isinstance(transition_pose, Mapping)
                    else stages[0]
                )
                predicted_attachment = candidate.get("predicted_attachment_transform")
                if isinstance(predicted_attachment, Mapping):
                    transition_target["attachment_transform"] = dict(
                        predicted_attachment
                    )
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
                    validity_state["qualification_scene_diff"] = dict(
                        active_scene_diff
                    )
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
                evidence["pure_collision_pairs"] = list(
                    validity.get("collision_pairs") or []
                )
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
            evidence["collision_pairs"] = list(
                collision_validity.get("collision_pairs") or []
            )
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
            return {**base, "verdict": "UNKNOWN", "reason": "planning_context_unavailable", "error_type": type(exc).__name__}
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
                    transition_pose
                    if isinstance(transition_pose, Mapping)
                    else stages[0]
                )
                predicted_attachment = candidate.get("predicted_attachment_transform")
                if isinstance(predicted_attachment, Mapping):
                    transition_target["attachment_transform"] = dict(
                        predicted_attachment
                    )
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
            try:
                planning_target = dict(target)
                if active_scene_diff is not None:
                    planning_target["qualification_scene_diff"] = dict(
                        active_scene_diff
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
                return self._unknown(base, evidence, "plan_only_timeout")
            except Exception as exc:  # noqa: BLE001
                return self._unknown(base, evidence, "plan_only_service_error", exc)
            evidence["moveit_error_code"] = planned.get("moveit_error_code")
            evidence["solver"] = planned.get("solver")
            evidence["elapsed_s"] = planned.get("elapsed_s")
            evidence["joint_margin"] = planned.get("joint_margin")
            points = planned.get("trajectory_points")
            if planned.get("execution_started") is not False:
                return self._unknown(base, evidence, "plan_only_execution_evidence_missing")
            if planned.get("ok") is not True:
                evidence["collision_pairs"] = list(planned.get("collision_pairs") or evidence.get("collision_pairs") or [])
                return self._fail(base, evidence, "plan_only_failed")
            if not isinstance(points, list) or not points:
                return self._unknown(base, evidence, "plan_only_empty_trajectory")
            end_state = planned.get("end_joint_state")
            if not isinstance(end_state, Mapping):
                return self._unknown(base, evidence, "plan_only_end_state_missing")
            evidence.update({"plan_only": True, "execution_started": False, "trajectory": {"point_count": len(points)}, "end_joint_state": dict(end_state)})
            base["stages"].append(evidence)
            start = dict(end_state)
            transition = target.get("scene_transition") if isinstance(target, Mapping) else None
            if transition:
                transition_evidence: Mapping[str, Any] = {"ok": True, "transition": transition, "virtual": True}
                if self.apply_scene_transition is not None:
                    try:
                        transition_evidence = self.apply_scene_transition(scene, str(transition), target)
                    except Exception as exc:  # noqa: BLE001
                        return {**base, "verdict": "UNKNOWN", "reason": "virtual_scene_transition_error", "error_type": type(exc).__name__}
                evidence["scene_transition"] = dict(transition_evidence)
                if transition_evidence.get("ok") is not True:
                    return self._fail(base, evidence, "virtual_scene_transition_failed")
                next_diff = transition_evidence.get("planning_scene_diff")
                if isinstance(next_diff, Mapping):
                    active_scene_diff = dict(next_diff)
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
            seeded_target["ik_seed_timeout_s"] = max(
                0.001, remaining_budget / remaining_seeds
            )
            try:
                result = dict(self.compute_ik(seeded_target, seed, collision))
            except TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                attempts.append({"seed_sha256": _hash(seed), "ok": False, "error_type": type(exc).__name__})
                continue
            attempts.append({"seed_sha256": _hash(seed), "ok": result.get("ok") is True, "moveit_error_code": result.get("moveit_error_code"), "solver": result.get("solver"), "elapsed_s": result.get("elapsed_s")})
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
            seeded_target["ik_seed_timeout_s"] = max(
                0.001, remaining_budget / remaining_seeds
            )
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
            validity_state = dict(state)
            if active_scene_diff is not None:
                validity_state["qualification_scene_diff"] = dict(active_scene_diff)
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
            return {
                **result,
                "state_validity": validity,
            }, attempts, ""
        if saw_validity_timeout:
            return {}, attempts, "collision_state_validity_timeout"
        if saw_validity_error:
            return {}, attempts, "collision_state_validity_service_error"
        if saw_state_missing:
            return {}, attempts, "collision_ik_evidence_missing"
        return {
            "ok": False,
            "all_solutions_state_invalid": saw_state_invalid,
            "collision_pairs": collision_pairs,
        }, attempts, ""

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
            if isinstance(lower_value, list) and isinstance(upper_value, list) and len(lower_value) == len(upper_value) == len(positions):
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
