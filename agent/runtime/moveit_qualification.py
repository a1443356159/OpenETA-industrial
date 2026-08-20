"""Host-only MoveIt candidate qualification and immutable proof cache.

This module deliberately has no AgentTool registration.  The runtime may call
the simulator's private ``qualify_motion_candidates`` RPC, but planners cannot
request arbitrary IK or plan-only motion.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict
from agent.tools.registry import ToolResult


QUALIFICATION_SCHEMA = "openeta.moveit_candidate_qualification.v1"
PRIVATE_RPC_NAME = "qualify_motion_candidates"
PLANNING_TIME_S = 30.0
PLANNING_ATTEMPTS = 3


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
    """Validate one batch response and expose only candidates proven PASS."""

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
    ) -> None:
        self.rpc = rpc
        self.cache = cache or QualificationCache()
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None
        self.compile_candidate = compile_candidate

    def qualify_result(
        self,
        result: ToolResult,
        *,
        purpose: str,
        scene_epoch: int,
        planning_scene_revision: int,
        source: Mapping[str, Any] | None = None,
        cache_result: bool = True,
    ) -> ToolResult:
        if not result.success:
            return result
        details = result.details
        key = "placement_candidates" if purpose == "placement" else "grasp_candidates"
        raw = details.get(key)
        if not isinstance(raw, list) or not raw:
            return result
        generated = len(raw)
        request_candidates = []
        for candidate in raw:
            if not isinstance(candidate, Mapping):
                continue
            compiled: Mapping[str, Any] = {}
            if self.compile_candidate is not None:
                try:
                    compiled = self.compile_candidate(
                        candidate,
                        purpose,
                        source or {},
                        scene_epoch,
                        planning_scene_revision,
                    )
                except Exception:  # noqa: BLE001 - becomes UNKNOWN at the host RPC.
                    compiled = {}
            rpc_candidate = dict(candidate)
            if isinstance(compiled.get("qualification_stages"), list):
                rpc_candidate["qualification_stages"] = list(
                    compiled["qualification_stages"]
                )
            if isinstance(compiled.get("compile_parameters"), Mapping):
                rpc_candidate["compile_parameters"] = dict(
                    compiled["compile_parameters"]
                )
            request_candidates.append(
                {
                    "candidate_id": str(candidate.get("id") or ""),
                    "candidate_pose_sha256": _hash(candidate),
                    "candidate": rpc_candidate,
                }
            )
        request: JsonDict = {
            "schema_version": QUALIFICATION_SCHEMA,
            "purpose": purpose,
            "scene_epoch": scene_epoch,
            "planning_scene_revision": planning_scene_revision,
            "planning": {
                "allowed_planning_time_s": PLANNING_TIME_S,
                "num_planning_attempts": PLANNING_ATTEMPTS,
                "plan_only": True,
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
            response = self.rpc(PRIVATE_RPC_NAME, request, PLANNING_TIME_S * generated + 10.0)
        except Exception as exc:  # noqa: BLE001 - private transport boundary.
            response = {
                "schema_version": QUALIFICATION_SCHEMA,
                "planning_scene_revision": planning_scene_revision,
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
        details.update(
            {
                "selection_required": bool(passed),
                "generated_candidate_count": generated,
                "qualified_candidate_count": generated,
                "rejection_reason_counts": dict(sorted(counts.items())),
                "qualification_evidence": evidence,
            }
        )
        artifact = self._write_artifact(evidence)
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
            elif verdict not in {"PASS", "FAIL", "UNKNOWN"} or not valid_identity:
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
        results = [self._qualify_candidate(item, revision) for item in candidates]
        binding = str(request.get("qualification_binding_sha256") or "")
        for result in results:
            result["qualification_binding_sha256"] = binding
        return {
            "schema_version": QUALIFICATION_SCHEMA,
            "planning_scene_revision": revision,
            "execution_started": False,
            "results": results,
        }

    def _qualify_candidate(self, descriptor: object, revision: int) -> JsonDict:
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
        }
        if isinstance(candidate.get("compile_parameters"), Mapping):
            base["compile_parameters"] = dict(candidate["compile_parameters"])
        if not candidate_id or not pose_hash or not isinstance(stages, list) or not stages:
            return {**base, "verdict": "UNKNOWN", "reason": "compiled_stages_missing"}
        try:
            start = dict(self.current_joint_state())
        except Exception as exc:  # noqa: BLE001
            return {
                **base,
                "verdict": "UNKNOWN",
                "reason": "start_joint_state_unavailable",
                "error_type": type(exc).__name__,
            }
        for index, target in enumerate(stages):
            if self.scene_revision() != revision:
                return {**base, "verdict": "UNKNOWN", "reason": "planning_scene_revision_drift"}
            if not isinstance(target, Mapping):
                return {**base, "verdict": "UNKNOWN", "reason": "compiled_stage_invalid"}
            evidence: JsonDict = {
                "stage_index": index,
                "name": str(target.get("name") or f"stage_{index}"),
                "start_joint_state_sha256": _hash(start),
                "execution_started": False,
            }
            try:
                pure_ik = dict(self.compute_ik(target, start, False))
            except TimeoutError:
                return self._unknown(base, evidence, "kinematic_ik_timeout")
            except Exception as exc:  # noqa: BLE001
                return self._unknown(base, evidence, "kinematic_ik_service_error", exc)
            evidence["kinematic_ik"] = pure_ik.get("ok") is True
            if not evidence["kinematic_ik"]:
                return self._fail(base, evidence, "kinematic_ik_failed")
            pure_state = pure_ik.get("joint_state")
            if not isinstance(pure_state, Mapping):
                return self._unknown(base, evidence, "kinematic_ik_evidence_missing")
            try:
                validity = dict(self.check_state_validity(pure_state))
            except TimeoutError:
                return self._unknown(base, evidence, "state_validity_timeout")
            except Exception as exc:  # noqa: BLE001
                return self._unknown(base, evidence, "state_validity_service_error", exc)
            evidence["state_valid"] = validity.get("valid") is True
            evidence["collision_pairs"] = list(validity.get("collision_pairs") or [])
            if not evidence["state_valid"]:
                return self._fail(base, evidence, "collision_state_invalid")
            try:
                collision_ik = dict(self.compute_ik(target, start, True))
            except TimeoutError:
                return self._unknown(base, evidence, "collision_ik_timeout")
            except Exception as exc:  # noqa: BLE001
                return self._unknown(base, evidence, "collision_ik_service_error", exc)
            evidence["collision_ik"] = collision_ik.get("ok") is True
            if not evidence["collision_ik"]:
                evidence["collision_pairs"] = list(
                    collision_ik.get("collision_pairs") or evidence["collision_pairs"]
                )
                return self._fail(base, evidence, "collision_ik_failed")
            try:
                planned = dict(
                    self.plan_only(
                        target,
                        start,
                        PLANNING_TIME_S,
                        PLANNING_ATTEMPTS,
                    )
                )
            except TimeoutError:
                return self._unknown(base, evidence, "plan_only_timeout")
            except Exception as exc:  # noqa: BLE001
                return self._unknown(base, evidence, "plan_only_service_error", exc)
            execution_started = planned.get("execution_started")
            points = planned.get("trajectory_points")
            if execution_started is not False:
                return self._unknown(base, evidence, "plan_only_execution_evidence_missing")
            if planned.get("ok") is not True:
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
        if self.scene_revision() != revision:
            return {**base, "verdict": "UNKNOWN", "reason": "planning_scene_revision_drift"}
        return {**base, "verdict": "PASS", "reason": "qualified"}

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
