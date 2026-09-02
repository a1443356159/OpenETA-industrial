"""Deterministic eye-in-hand observation planning behind one AgentTool.

The agent decides *when* another view is useful.  This module owns the bounded
geometry search, MoveIt proof, motion, re-observation, and point-grounded SAM3
refresh so those mechanical details do not consume planner turns or context.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Protocol

import numpy as np
from PIL import Image

from adapter.protocol import EnvObservation, JsonDict
from agent.runtime.artifact_paths import artifact_session_id, artifact_session_root
from agent.runtime.moveit_qualification import MoveItCandidateQualifier, QualificationCache
from agent.runtime.reference_localization import (
    ReferencePointLocalization,
    SemanticPointLocalizationError,
)
from agent.tools.registry import (
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
    make_tool_result,
)
from agent.tools.sim_mcp import SimulatorMcpToolProxy


ACTIVE_VISION_SCHEMA = "openeta.active_vision.v1"
ACTIVE_VISION_ARTIFACT_SCHEMA = "openeta.active_vision_artifact.v1"
SUPPORTED_SEMANTIC_ROLE = "grasp_target"
SUPPORTED_QUALITY_PROFILE = "grasp_rgbd"
MAX_MOTION_ATTEMPTS = 2
MAX_VIEW_CANDIDATES = 24
SELF_OCCLUSION_NEAR_M = 0.20
SELF_OCCLUSION_MAX_NEAR_FRACTION = 0.35
SELF_OCCLUSION_MAX_BLACK_FRACTION = 0.65
_EVIDENCE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


@dataclass(frozen=True, slots=True)
class GraspRgbdQualityProfile:
    min_mask_area_px: int = 1024
    min_mask_area_fraction: float = 0.001
    max_mask_area_fraction: float = 0.35
    min_valid_depth_fraction: float = 0.85
    min_border_margin_px: int = 3


@dataclass(frozen=True, slots=True)
class TargetEvidence:
    result_id: str
    detection_id: str
    semantic_target: str
    semantic_role: str
    source_rgb: Path
    source_depth: Path
    mask: Path
    frame_id: str
    intrinsics: JsonDict
    extrinsics: JsonDict
    perception_bundle_id: str
    observation_id: str
    scene_epoch: int
    bbox_xyxy: tuple[int, int, int, int] | None


class SemanticPointLocalizer(Protocol):
    """Fresh-context visual localizer used only to seed active perception."""

    def localize(
        self,
        *,
        semantic_target: str,
        scene_image: Path,
        image_size: tuple[int, int],
    ) -> ReferencePointLocalization:
        """Return one audited current-image foreground point or abstain."""


class ActiveVisionError(RuntimeError):
    def __init__(self, code: str, message: str, *, infrastructure: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.infrastructure = infrastructure


class ActiveVisionController:
    """Stateful per-runtime controller with a session-local SAM3 cache."""

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        candidate_qualifier: MoveItCandidateQualifier | None,
        simulator_proxy: SimulatorMcpToolProxy | None,
        sam3_handler: ToolHandler | None,
        semantic_localizer: SemanticPointLocalizer | None = None,
        move_spec: ToolSpec,
        observe_spec: ToolSpec,
        sam3_spec: ToolSpec,
        quality_profile: GraspRgbdQualityProfile | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.qualifier = _fast_observation_qualifier(candidate_qualifier)
        self.simulator_proxy = simulator_proxy
        self.sam3_handler = sam3_handler
        self.semantic_localizer = semantic_localizer
        self.move_spec = move_spec
        self.observe_spec = observe_spec
        self.sam3_spec = sam3_spec
        self.quality_profile = quality_profile or GraspRgbdQualityProfile()
        self._sam_cache: dict[str, ToolResult] = {}
        self._sam_cache_lock = threading.Lock()

    def handler(self, context: ToolExecutionContext) -> ToolResult:
        started = time.monotonic()
        record: JsonDict = {
            "schema_version": ACTIVE_VISION_ARTIFACT_SCHEMA,
            "request": dict(context.parameters),
            "status": "exhausted",
            "motion_attempts": [],
            "candidate_counts": {
                "generated": 0,
                "cheap_legal": 0,
                "moveit_l5_pass": 0,
                "executed": 0,
            },
            "rejection_reason_counts": {},
            "timings_s": {},
        }
        receipt: JsonDict | None = None
        try:
            result, receipt = self._run(context, record)
        except ActiveVisionError as exc:
            status = "infrastructure_error" if exc.infrastructure else "exhausted"
            record.update({"status": status, "stop_reason": exc.code})
            result = self._result(
                context,
                success=False,
                status=status,
                content=f"Active observation failed: {exc}",
                record=record,
                diagnostics=[{"code": exc.code, "message": str(exc)}],
            )
        except Exception as exc:  # noqa: BLE001 - fail closed at orchestration boundary.
            record.update(
                {
                    "status": "infrastructure_error",
                    "stop_reason": "active_vision_internal_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            result = self._result(
                context,
                success=False,
                status="infrastructure_error",
                content=f"Active observation infrastructure failed: {exc}",
                record=record,
                diagnostics=[
                    {
                        "code": "active_vision_internal_error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        record["timings_s"]["total"] = round(time.monotonic() - started, 6)
        artifact = self._write_artifact(context, record)
        result.details.setdefault("artifacts", []).append(artifact)
        outputs = result.details.setdefault("outputs", {})
        outputs["artifact_ref"] = artifact["path"]
        if receipt is not None:
            result.details["environment_receipt"] = receipt
        return result

    def _run(
        self,
        context: ToolExecutionContext,
        record: JsonDict,
    ) -> tuple[ToolResult, JsonDict | None]:
        params = context.parameters
        evidence_id = str(params.get("target_evidence_id") or "").strip()
        semantic_target = str(params.get("semantic_target") or "").strip()
        semantic_role = str(params.get("semantic_role") or SUPPORTED_SEMANTIC_ROLE).strip()
        quality_profile = str(
            params.get("quality_profile") or SUPPORTED_QUALITY_PROFILE
        ).strip()
        motion_budget = _bounded_integer(
            params.get("max_motion_attempts", MAX_MOTION_ATTEMPTS),
            minimum=0,
            maximum=MAX_MOTION_ATTEMPTS,
            field="max_motion_attempts",
        )
        if evidence_id and not _EVIDENCE_ID_RE.fullmatch(evidence_id):
            raise ActiveVisionError(
                "invalid_target_evidence_id",
                "target_evidence_id must be one selected SAM3 result id when supplied",
            )
        if not evidence_id and not semantic_target:
            raise ActiveVisionError(
                "missing_active_search_target",
                "semantic_target is required when no selected SAM3 evidence exists",
            )
        if semantic_role != SUPPORTED_SEMANTIC_ROLE:
            raise ActiveVisionError(
                "unsupported_active_vision_semantic_role",
                "active_vision v1 is pre-contact and supports only grasp_target",
            )
        if quality_profile != SUPPORTED_QUALITY_PROFILE:
            raise ActiveVisionError(
                "unsupported_active_vision_quality_profile",
                f"unsupported quality profile: {quality_profile}",
            )
        observation = context.observation
        if observation is None:
            raise ActiveVisionError("missing_current_observation", "a current observation is required")
        if _observation_is_attached(observation):
            raise ActiveVisionError(
                "active_vision_post_attachment_unsupported",
                "active_vision v1 cannot move an attached object",
            )

        search_mode = not evidence_id
        if search_mode:
            target_hint = params.get("target_hint")
            if target_hint is None:
                if self.semantic_localizer is None:
                    raise ActiveVisionError(
                        "semantic_point_localizer_unavailable",
                        "semantic search requires either a host point or the isolated provider localizer",
                        infrastructure=True,
                    )
                source_rgb, image_size = _semantic_search_source_image(observation)
                phase_started = time.monotonic()
                try:
                    localization = self.semantic_localizer.localize(
                        semantic_target=semantic_target,
                        scene_image=source_rgb,
                        image_size=image_size,
                    )
                except SemanticPointLocalizationError as exc:
                    record["timings_s"]["semantic_localization"] = round(
                        time.monotonic() - phase_started, 6
                    )
                    raise ActiveVisionError(
                        exc.code,
                        str(exc),
                        infrastructure=exc.infrastructure,
                    ) from exc
                except Exception as exc:  # noqa: BLE001 - custom localizer boundary.
                    record["timings_s"]["semantic_localization"] = round(
                        time.monotonic() - phase_started, 6
                    )
                    raise ActiveVisionError(
                        "semantic_point_localization_provider_error",
                        f"isolated visual localizer failed: {exc}",
                        infrastructure=True,
                    ) from exc
                record["timings_s"]["semantic_localization"] = round(
                    time.monotonic() - phase_started, 6
                )
                localization_receipt = _semantic_localization_receipt(localization)
                target_hint = {
                    "source_image": str(source_rgb),
                    "positive_points": [localization.as_prompt_point()],
                    "bbox_xyxy": (
                        list(localization.bbox_xyxy) if localization.bbox_xyxy is not None else None
                    ),
                    "source": "isolated_provider_visual_grounding",
                }
                record["semantic_point_localization"] = localization_receipt
            target_center, target_extent, hint_receipt = _target_hint_world_geometry(
                target_hint,
                observation=observation,
                artifact_root=self.artifact_root,
            )
            target_point_count = 1
            record.update(
                {
                    "mode": "semantic_search",
                    "semantic_target": semantic_target,
                    "target_hint": hint_receipt,
                }
            )
        else:
            evidence = _resolve_target_evidence(
                evidence_id,
                observation=observation,
                artifact_root=self.artifact_root,
                semantic_role=semantic_role,
            )
            semantic_target = evidence.semantic_target
            current_quality, target_points_world = _target_quality(
                evidence,
                profile=self.quality_profile,
            )
            record.update(
                {
                    "mode": "grounded_refinement",
                    "target_evidence": _evidence_receipt(evidence),
                    "current_quality": current_quality,
                    "semantic_target": semantic_target,
                }
            )
            if current_quality["passed"] is True:
                record.update(
                    {
                        "status": "reused",
                        "stop_reason": "current_observation_quality_pass",
                        "viewpoint_id": f"existing:{evidence.frame_id}",
                        "motion_count": 0,
                    }
                )
                return (
                    self._result(
                        context,
                        success=True,
                        status="reused",
                        content=(
                            "Current grounded RGB-D observation already passes grasp quality."
                        ),
                        record=record,
                        quality=current_quality,
                        observation_bundle_id=evidence.perception_bundle_id,
                        viewpoint_id=f"existing:{evidence.frame_id}",
                        motion_count=0,
                    ),
                    None,
                )
            target_center = np.median(target_points_world, axis=0)
            target_extent = np.quantile(target_points_world, 0.98, axis=0) - np.quantile(
                target_points_world, 0.02, axis=0
            )
            target_point_count = int(len(target_points_world))
        if motion_budget == 0:
            raise ActiveVisionError(
                "current_view_insufficient_motion_disabled",
                "current target view is insufficient and max_motion_attempts is zero",
            )

        wrist = _wrist_rgbd_packet(observation)
        self_occlusion = _mechanical_camera_preflight(wrist)
        record["camera_self_occlusion_preflight"] = self_occlusion
        if self_occlusion["passed"] is not True:
            raise ActiveVisionError(
                "camera_self_occlusion_unusable",
                "wrist RGB-D is mechanically self-occluded; changing arm joints cannot clear it",
            )
        if self.qualifier is None or self.simulator_proxy is None:
            raise ActiveVisionError(
                "active_vision_motion_backend_unavailable",
                "MoveIt qualification and simulator motion are required for another view",
                infrastructure=True,
            )
        if self.sam3_handler is None:
            raise ActiveVisionError(
                "active_vision_sam3_unavailable",
                "SAM3 is required to reground the target after camera motion",
                infrastructure=True,
            )

        phase_started = time.monotonic()
        generated, rejection_counts = _generate_view_candidates(
            observation=observation,
            wrist=wrist,
            target_center_world=target_center,
            target_extent_world=target_extent,
        )
        record["timings_s"]["candidate_generation"] = round(time.monotonic() - phase_started, 6)
        record["candidate_counts"]["generated"] = sum(rejection_counts.values()) + len(generated)
        record["candidate_counts"]["cheap_legal"] = len(generated)
        record["rejection_reason_counts"] = dict(sorted(rejection_counts.items()))
        record["target_geometry"] = {
            "center_world_xyz": _rounded_vector(target_center),
            "extent_world_xyz": _rounded_vector(target_extent),
            "point_count": target_point_count,
        }
        if not generated:
            raise ActiveVisionError(
                "active_view_candidate_pool_empty",
                "every deterministic active-view candidate failed strict analytic legality",
            )

        scene_epoch = _scene_epoch(observation)
        revision = _planning_scene_revision(observation)
        phase_started = time.monotonic()
        qualified, qualification = self._qualify(
            generated,
            scene_epoch=scene_epoch,
            planning_scene_revision=revision,
            pass_target=min(2, motion_budget),
        )
        record["timings_s"]["initial_qualification"] = round(time.monotonic() - phase_started, 6)
        record["qualification"] = qualification
        record["candidate_counts"]["moveit_l5_pass"] = len(qualified)
        if qualification.get("infrastructure_error") is True:
            raise ActiveVisionError(
                "active_view_qualification_infrastructure_error",
                "MoveIt active-view qualification failed its bounded health retry",
                infrastructure=True,
            )
        if not qualified:
            raise ActiveVisionError(
                "active_view_moveit_pool_exhausted",
                "no deterministic active-view candidate passed IK, state validity, and L5",
            )

        originals = {str(candidate["id"]): candidate for candidate in generated}
        last_receipt: JsonDict | None = None
        current_observation = observation
        for attempt_index, selected in enumerate(qualified[:motion_budget]):
            candidate_id = str(selected.get("id") or "")
            candidate = originals[candidate_id]
            if attempt_index > 0:
                phase_started = time.monotonic()
                refreshed, refreshed_summary = self._qualify(
                    [candidate],
                    scene_epoch=_scene_epoch(current_observation),
                    planning_scene_revision=_planning_scene_revision(current_observation),
                    pass_target=1,
                )
                record["motion_attempts"][-1].setdefault("timings_s", {})[
                    "next_candidate_requalification"
                ] = round(time.monotonic() - phase_started, 6)
                record["motion_attempts"][-1]["alternate_requalification"] = refreshed_summary
                if not refreshed:
                    continue
                selected = refreshed[0]
            attempt: JsonDict = {
                "attempt_index": attempt_index,
                "viewpoint_id": candidate_id,
                "moveit_l5_qualified": True,
                "timings_s": {},
            }
            record["motion_attempts"].append(attempt)
            phase_started = time.monotonic()
            move = self._move_to_view(context, selected)
            attempt["timings_s"]["move"] = round(time.monotonic() - phase_started, 6)
            attempt["motion_success"] = move.success
            attempt["motion_diagnostics"] = list(move.details.get("diagnostics") or [])
            if not move.success:
                if _motion_outcome_unknown(move):
                    raise ActiveVisionError(
                        "active_view_motion_outcome_unknown",
                        "active-view motion outcome is unknown and requires reconciliation",
                        infrastructure=True,
                    )
                attempt["stop_reason"] = "known_motion_failure"
                break
            record["candidate_counts"]["executed"] += 1
            phase_started = time.monotonic()
            observed = self._observe_after_motion(context)
            attempt["timings_s"]["observe"] = round(time.monotonic() - phase_started, 6)
            if not observed.success:
                raise ActiveVisionError(
                    "active_view_observation_failed",
                    "fresh RGB-D observation failed after active-view motion",
                    infrastructure=True,
                )
            last_receipt = _environment_receipt(observed)
            current_observation = _observation_from_result(observed)
            attempt["camera_evaluations"] = []
            last_rejection = "active_view_quality_failed"
            for packet in _active_rgbd_packets(current_observation):
                camera = packet["camera"]
                evaluation: JsonDict = {
                    "frame_id": camera.frame_id,
                    "role": camera.role,
                    "is_wrist": packet["is_wrist"],
                }
                attempt["camera_evaluations"].append(evaluation)
                if packet["is_wrist"] is True:
                    self_occlusion = _mechanical_camera_preflight(packet)
                    evaluation["camera_self_occlusion"] = self_occlusion
                    if self_occlusion["passed"] is not True:
                        evaluation["rejection"] = "camera_self_occlusion_unusable"
                        last_rejection = str(evaluation["rejection"])
                        continue
                point = _project_world_point(
                    target_center,
                    extrinsics=camera.extrinsics,
                    intrinsics=camera.intrinsics,
                )
                if point is None:
                    evaluation["rejection"] = "target_outside_active_view"
                    last_rejection = str(evaluation["rejection"])
                    continue
                evaluation["projected_target_point_xy"] = [
                    round(point[0], 3),
                    round(point[1], 3),
                ]
                phase_started = time.monotonic()
                sam_result, cache_hit = self._segment_new_view(
                    context,
                    observation=current_observation,
                    packet=packet,
                    semantic_target=semantic_target,
                    scene_epoch=_scene_epoch(current_observation),
                    point_xy=point,
                )
                evaluation["sam3_elapsed_s"] = round(time.monotonic() - phase_started, 6)
                evaluation["sam3_cache_hit"] = cache_hit
                evaluation["sam3_success"] = sam_result.success
                if not sam_result.success:
                    evaluation["rejection"] = "sam3_refresh_failed"
                    last_rejection = str(evaluation["rejection"])
                    continue
                phase_started = time.monotonic()
                refreshed_evidence, quality, selection = _select_point_grounded_evidence(
                    sam_result,
                    observation=current_observation,
                    artifact_root=self.artifact_root,
                    profile=self.quality_profile,
                    target_center_world=target_center,
                    target_extent_world=target_extent,
                    point_xy=point,
                )
                evaluation["geometry_selection_elapsed_s"] = round(
                    time.monotonic() - phase_started, 6
                )
                evaluation["selection"] = selection
                if refreshed_evidence is None or quality is None:
                    evaluation["rejection"] = "sam3_point_depth_geometry_rejected"
                    last_rejection = str(evaluation["rejection"])
                    continue
                evaluation["quality"] = quality
                evaluation["accepted"] = True
                attempt["sam3_cache_hit"] = cache_hit
                attempt["sam3_success"] = True
                attempt["quality"] = quality
                attempt["selected_camera_frame_id"] = camera.frame_id
                record.update(
                    {
                        "status": "acquired",
                        "stop_reason": "active_observation_quality_pass",
                        "viewpoint_id": candidate_id,
                        "motion_count": int(record["candidate_counts"]["executed"]),
                        "final_quality": quality,
                        "observation_bundle_id": (refreshed_evidence.perception_bundle_id),
                        "selected_camera_frame_id": camera.frame_id,
                    }
                )
                return (
                    self._result(
                        context,
                        success=True,
                        status="acquired",
                        content=(
                            "Active calibrated RGB-D observation acquired and "
                            "point/depth quality-gated."
                        ),
                        record=record,
                        quality=quality,
                        observation_bundle_id=(refreshed_evidence.perception_bundle_id),
                        viewpoint_id=candidate_id,
                        motion_count=int(record["candidate_counts"]["executed"]),
                        environment_receipt=last_receipt,
                        segmentation=sam_result.details,
                    ),
                    last_receipt,
                )
            attempt["quality_rejection"] = last_rejection

        record.update(
            {
                "status": "exhausted",
                "stop_reason": "qualified_active_view_alternates_exhausted",
                "motion_count": int(record["candidate_counts"]["executed"]),
            }
        )
        return (
            self._result(
                context,
                success=False,
                status="exhausted",
                content="Qualified active-view alternatives were exhausted without a quality pass.",
                record=record,
                motion_count=int(record["candidate_counts"]["executed"]),
                diagnostics=[{"code": "qualified_active_view_alternates_exhausted"}],
                environment_receipt=last_receipt,
            ),
            last_receipt,
        )

    def _qualify(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        scene_epoch: int,
        planning_scene_revision: int,
        pass_target: int,
    ) -> tuple[list[JsonDict], JsonDict]:
        assert self.qualifier is not None
        raw = ToolResult(
            True,
            details={
                "observation_candidates": [dict(candidate) for candidate in candidates],
                "candidate_count": len(candidates),
                "raw_candidate_count": len(candidates),
                "model_raw_candidate_count": len(candidates),
                "ranking": "deterministic_active_view_quality",
            },
        )
        qualified = self.qualifier.qualify_result(
            raw,
            purpose="observation",
            scene_epoch=scene_epoch,
            planning_scene_revision=planning_scene_revision,
            source={
                "provider": "active_vision",
                "provider_version": ACTIVE_VISION_SCHEMA,
                "solver_version": "moveit_runtime",
            },
            cache_result=False,
            l5_pass_target=pass_target,
            l5_min_pass_target=1,
        )
        details = qualified.details
        passed = details.get("observation_candidates")
        passed = [dict(value) for value in passed or [] if isinstance(value, Mapping)]
        evidence = details.get("qualification_evidence")
        summary = dict(evidence) if isinstance(evidence, Mapping) else {}
        summary.update(
            {
                "profile": details.get("qualification_profile"),
                "solver_profile": details.get("solver_profile"),
                "stop_reason": details.get("qualification_stop_reason"),
                "artifact": details.get("qualification_artifact"),
                "pass_count": len(passed),
                "infrastructure_error": bool(
                    summary.get("infrastructure_error")
                    or details.get("qualification_stop_reason") == "infrastructure_error"
                ),
            }
        )
        return passed, summary

    def _move_to_view(
        self,
        outer_context: ToolExecutionContext,
        candidate: Mapping[str, Any],
    ) -> ToolResult:
        assert self.simulator_proxy is not None
        stages = candidate.get("qualification_stages")
        if not isinstance(stages, list) or len(stages) != 1 or not isinstance(stages[0], Mapping):
            raise ActiveVisionError(
                "active_view_qualification_pose_missing",
                "qualified active view lacks its exact terminal stage",
                infrastructure=True,
            )
        target = dict(stages[0])
        target["qualified_candidate_id"] = str(candidate.get("id") or "")
        context = ToolExecutionContext(
            name="move_to",
            spec=self.move_spec,
            parameters={
                "target_pose": target,
                "tolerance": 0.001,
                "ori_tolerance": 0.005,
                "velocity_scaling": 0.15,
                "acceleration_scaling": 0.10,
                "enable_collision_check": True,
            },
            observation=outer_context.observation,
            metadata=dict(outer_context.metadata),
        )
        return self.simulator_proxy.call(context, tool_name="move_to")

    def _observe_after_motion(self, outer_context: ToolExecutionContext) -> ToolResult:
        assert self.simulator_proxy is not None
        context = ToolExecutionContext(
            name="observe",
            spec=self.observe_spec,
            parameters={"reason": "active_vision_post_motion_quality_gate"},
            observation=outer_context.observation,
            metadata=dict(outer_context.metadata),
        )
        return self.simulator_proxy.call(context, tool_name="observe")

    def _segment_new_view(
        self,
        outer_context: ToolExecutionContext,
        *,
        observation: EnvObservation,
        packet: JsonDict,
        semantic_target: str,
        scene_epoch: int,
        point_xy: tuple[float, float],
    ) -> tuple[ToolResult, bool]:
        assert self.sam3_handler is not None
        rgb_path = str(packet["rgb"]["path"])
        point = {"x": round(point_xy[0], 3), "y": round(point_xy[1], 3), "label": 1}
        image_sha = _file_sha256(Path(rgb_path))
        fingerprint = _stable_hash(
            {
                "image_sha256": image_sha,
                "semantic_target": semantic_target,
                "semantic_role": SUPPORTED_SEMANTIC_ROLE,
                "scene_epoch": scene_epoch,
                "point": point,
            }
        )
        with self._sam_cache_lock:
            cached = self._sam_cache.get(fingerprint)
        if cached is not None:
            return cached, True
        observation_id = f"observation-{fingerprint[:16]}"
        bundle_id = f"perception-{fingerprint[16:32]}"
        sam_context = ToolExecutionContext(
            name="sam3",
            spec=self.sam3_spec,
            parameters={
                "mode": "points",
                "image": rgb_path,
                "points": [point],
                "semantic_role": SUPPORTED_SEMANTIC_ROLE,
                "semantic_target": semantic_target,
                "scene_epoch": scene_epoch,
                "observation_id": observation_id,
                "perception_bundle_id": bundle_id,
                "attempt_id": f"active-sam3-{fingerprint[:16]}",
            },
            observation=observation,
            metadata={
                **dict(outer_context.metadata),
                "sam3_selection_policy": "active_vision_point_depth_geometry",
            },
        )
        result = self.sam3_handler(sam_context)
        if not isinstance(result, ToolResult):
            raise ActiveVisionError(
                "active_vision_sam3_contract_invalid",
                "SAM3 handler returned a non-ToolResult value",
                infrastructure=True,
            )
        with self._sam_cache_lock:
            self._sam_cache.setdefault(fingerprint, result)
        return result, False

    def _result(
        self,
        context: ToolExecutionContext,
        *,
        success: bool,
        status: str,
        content: str,
        record: Mapping[str, Any],
        quality: Mapping[str, Any] | None = None,
        observation_bundle_id: str = "",
        viewpoint_id: str = "",
        motion_count: int = 0,
        diagnostics: list[JsonDict] | None = None,
        environment_receipt: Mapping[str, Any] | None = None,
        segmentation: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        attempt_fingerprint = _stable_hash(
            {
                "mode": record.get("mode"),
                "semantic_role": context.parameters.get("semantic_role")
                or SUPPORTED_SEMANTIC_ROLE,
                "semantic_target": record.get("semantic_target"),
                "scene_epoch": (
                    _scene_epoch(context.observation)
                    if context.observation is not None
                    else None
                ),
                "target_evidence_id": context.parameters.get("target_evidence_id"),
                "target_hint": context.parameters.get("target_hint"),
            }
        )
        outputs: JsonDict = {
            "schema_version": ACTIVE_VISION_SCHEMA,
            "status": status,
            "target_evidence_id": str(context.parameters.get("target_evidence_id") or ""),
            "semantic_role": str(
                context.parameters.get("semantic_role") or SUPPORTED_SEMANTIC_ROLE
            ),
            "semantic_target": str(record.get("semantic_target") or ""),
            "scene_epoch": (
                _scene_epoch(context.observation)
                if context.observation is not None
                else None
            ),
            "active_vision_mode": record.get("mode"),
            "active_vision_attempt_id": f"active-observe-{attempt_fingerprint[:16]}",
            "active_vision_attempt_fingerprint": attempt_fingerprint,
            "target_hint": (
                dict(record.get("target_hint") or {})
                if isinstance(record.get("target_hint"), Mapping)
                else None
            ),
            "semantic_point_localization": (
                dict(record.get("semantic_point_localization") or {})
                if isinstance(record.get("semantic_point_localization"), Mapping)
                else None
            ),
            "observation_bundle_id": observation_bundle_id or None,
            "viewpoint_id": viewpoint_id or None,
            "motion_count": int(motion_count),
            "quality": dict(quality or {}),
            "candidate_counts": dict(record.get("candidate_counts") or {}),
            "rejection_reason_counts": dict(record.get("rejection_reason_counts") or {}),
            "stop_reason": record.get("stop_reason"),
        }
        if isinstance(segmentation, Mapping):
            nested = segmentation.get("outputs")
            source = nested if isinstance(nested, Mapping) else segmentation
            for key in (
                "result_id",
                "detections",
                "selected_detection",
                "selection_bundle",
                "selection_review",
                "selection_required",
                "source_image",
                "source_frame_id",
                "source_camera_role",
                "prompt",
                "semantic_role",
                "semantic_target",
                "segmentation_mode",
                "scene_epoch",
                "perception_bundle_id",
                "observation_id",
                "attempt_id",
                "attempt_fingerprint",
                "view_identity",
            ):
                if key in source:
                    outputs[key] = source[key]
        return make_tool_result(
            context,
            success=success,
            content=content,
            outputs=outputs,
            state_delta={
                "active_vision_motion_count": int(motion_count),
                "observation_acquired": status == "acquired",
            },
            environment_receipt=(
                dict(environment_receipt) if isinstance(environment_receipt, Mapping) else None
            ),
            diagnostics=list(diagnostics or []),
        )

    def _write_artifact(
        self,
        context: ToolExecutionContext,
        record: Mapping[str, Any],
    ) -> JsonDict:
        session_id = artifact_session_id(context.metadata)
        root = artifact_session_root(self.artifact_root / "active_vision", session_id)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        digest = _stable_hash(record)
        path = root / f"{stamp}-{digest[:8]}.json"
        payload = {**dict(record), "sha256": digest}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "type": "active_vision_evidence",
            "kind": "json",
            "schema_version": ACTIVE_VISION_ARTIFACT_SCHEMA,
            "path": str(path),
            "sha256": digest,
        }


def build_active_observe_handler(
    *,
    artifact_root: str | Path,
    candidate_qualifier: MoveItCandidateQualifier | None,
    simulator_proxy: SimulatorMcpToolProxy | None,
    sam3_handler: ToolHandler | None,
    semantic_localizer: SemanticPointLocalizer | None = None,
    move_spec: ToolSpec,
    observe_spec: ToolSpec,
    sam3_spec: ToolSpec,
) -> ToolHandler:
    """Build one runtime-scoped active-observation handler."""

    return ActiveVisionController(
        artifact_root=artifact_root,
        candidate_qualifier=candidate_qualifier,
        simulator_proxy=simulator_proxy,
        sam3_handler=sam3_handler,
        semantic_localizer=semantic_localizer,
        move_spec=move_spec,
        observe_spec=observe_spec,
        sam3_spec=sam3_spec,
    ).handler


def _fast_observation_qualifier(
    base: MoveItCandidateQualifier | None,
) -> MoveItCandidateQualifier | None:
    """Isolate active views from grasp/place cache and force Beam-2 fast proof."""

    if base is None:
        return None
    return MoveItCandidateQualifier(
        base.rpc,
        cache=QualificationCache(),
        artifact_root=base.artifact_root,
        compile_candidate=base.compile_candidate,
        grasp_diversity_limit=base.diversity_limits["grasp"],
        placement_diversity_limit=base.diversity_limits["placement"],
        observation_diversity_limit=MAX_VIEW_CANDIDATES,
        grasp_full_plan_limit=base.full_plan_limits["grasp"],
        placement_full_plan_limit=base.full_plan_limits["placement"],
        observation_full_plan_limit=2,
        frozen_pair_full_plan_limit=base.frozen_pair_full_plan_limit,
        ik_seed_count=base.ik_seed_count,
        qualification_profile="fast_v3",
        solver_profile=base.solver_profile,
        beam_width=base.beam_width,
        grasp_waves=base.grasp_waves,
        placement_waves=base.placement_waves,
        observation_waves=(4, 8, 16, MAX_VIEW_CANDIDATES),
        max_ik_concurrency=base.max_ik_concurrency,
        max_state_validity_concurrency=base.max_state_validity_concurrency,
        fast_seed_count=base.fast_seed_count,
        recovery_seed_count=base.recovery_seed_count,
        fast_ik_timeout_s=base.fast_ik_timeout_s,
        recovery_ik_timeout_s=base.recovery_ik_timeout_s,
        capability_map_id=base.capability_map_id,
    )


def _resolve_target_evidence(
    evidence_id: str,
    *,
    observation: EnvObservation,
    artifact_root: Path,
    semantic_role: str,
) -> TargetEvidence:
    result_id, _, requested_detection_id = evidence_id.partition(":")
    candidates = sorted(
        path
        for path in (artifact_root / "sam3_results").rglob("tool_result.json")
        if path.parent.name == result_id
    )
    if len(candidates) != 1:
        raise ActiveVisionError(
            "target_evidence_not_unique",
            f"expected one session-local SAM3 evidence record for {result_id}, found {len(candidates)}",
        )
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActiveVisionError("target_evidence_unreadable", str(exc)) from exc
    details = payload.get("details") if isinstance(payload, Mapping) else None
    if not isinstance(details, Mapping):
        raise ActiveVisionError("target_evidence_invalid", "SAM3 evidence details are missing")
    if str(details.get("semantic_role") or "") != semantic_role:
        raise ActiveVisionError(
            "target_evidence_semantic_role_mismatch",
            "SAM3 evidence does not represent the requested semantic role",
        )
    selected = details.get("selected_detection")
    if not isinstance(selected, Mapping):
        raise ActiveVisionError(
            "target_evidence_selection_missing",
            "SAM3 evidence must contain one selected detection",
        )
    detection_id = str(selected.get("id") or "")
    if requested_detection_id and requested_detection_id != detection_id:
        raise ActiveVisionError(
            "target_evidence_detection_mismatch",
            "requested detection does not match the selected SAM3 evidence",
        )
    source_rgb = _resolve_artifact_path(details.get("source_image"), artifact_root)
    mask = _resolve_artifact_path(selected.get("mask_ref"), artifact_root)
    frame_id = str(details.get("source_frame_id") or "")
    packet = _camera_packet_for_source(observation, source_rgb=source_rgb, frame_id=frame_id)
    evidence_scene_epoch = _bounded_integer(
        details.get("scene_epoch", 0), minimum=0, maximum=2**31 - 1, field="scene_epoch"
    )
    if evidence_scene_epoch != _scene_epoch(observation):
        raise ActiveVisionError(
            "stale_target_evidence",
            "SAM3 target evidence belongs to a different scene epoch",
        )
    bbox = selected.get("bbox_xyxy")
    parsed_bbox = (
        tuple(int(value) for value in bbox)
        if isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
        else None
    )
    return TargetEvidence(
        result_id=str(details.get("result_id") or result_id),
        detection_id=detection_id,
        semantic_target=str(details.get("semantic_target") or selected.get("label") or "target"),
        semantic_role=semantic_role,
        source_rgb=source_rgb,
        source_depth=Path(str(packet["depth"]["path"])),
        mask=mask,
        frame_id=str(packet["camera"].frame_id),
        intrinsics=dict(packet["camera"].intrinsics),
        extrinsics=dict(packet["camera"].extrinsics),
        perception_bundle_id=str(details.get("perception_bundle_id") or ""),
        observation_id=str(details.get("observation_id") or ""),
        scene_epoch=evidence_scene_epoch,
        bbox_xyxy=parsed_bbox,
    )


def _evidence_from_sam_result(
    result: ToolResult,
    *,
    observation: EnvObservation,
    artifact_root: Path,
) -> TargetEvidence | None:
    details = result.details
    selected = details.get("selected_detection")
    if not isinstance(selected, Mapping):
        return None
    source_rgb = _resolve_artifact_path(details.get("source_image"), artifact_root)
    mask = _resolve_artifact_path(selected.get("mask_ref"), artifact_root)
    frame_id = str(details.get("source_frame_id") or "")
    packet = _camera_packet_for_source(observation, source_rgb=source_rgb, frame_id=frame_id)
    bbox = selected.get("bbox_xyxy")
    parsed_bbox = (
        tuple(int(value) for value in bbox)
        if isinstance(bbox, list) and len(bbox) == 4
        else None
    )
    return TargetEvidence(
        result_id=str(details.get("result_id") or ""),
        detection_id=str(selected.get("id") or ""),
        semantic_target=str(details.get("semantic_target") or selected.get("label") or "target"),
        semantic_role=SUPPORTED_SEMANTIC_ROLE,
        source_rgb=source_rgb,
        source_depth=Path(str(packet["depth"]["path"])),
        mask=mask,
        frame_id=str(packet["camera"].frame_id),
        intrinsics=dict(packet["camera"].intrinsics),
        extrinsics=dict(packet["camera"].extrinsics),
        perception_bundle_id=str(details.get("perception_bundle_id") or ""),
        observation_id=str(details.get("observation_id") or ""),
        scene_epoch=_scene_epoch(observation),
        bbox_xyxy=parsed_bbox,
    )


def _point_grounded_evidence_candidates(
    result: ToolResult,
    *,
    observation: EnvObservation,
    artifact_root: Path,
) -> list[tuple[TargetEvidence, JsonDict]]:
    """Materialize every SAM point-mask without invoking semantic review."""

    details = result.details
    selected = details.get("selected_detection")
    raw_candidates = (
        [selected]
        if isinstance(selected, Mapping)
        else details.get("detections")
        if isinstance(details.get("detections"), list)
        else []
    )
    source_rgb = _resolve_artifact_path(details.get("source_image"), artifact_root)
    frame_id = str(details.get("source_frame_id") or "")
    packet = _camera_packet_for_source(
        observation,
        source_rgb=source_rgb,
        frame_id=frame_id,
    )
    candidates: list[tuple[TargetEvidence, JsonDict]] = []
    for value in raw_candidates:
        if not isinstance(value, Mapping):
            continue
        detection = dict(value)
        try:
            mask = _resolve_artifact_path(detection.get("mask_ref"), artifact_root)
        except ActiveVisionError:
            continue
        bbox = detection.get("bbox_xyxy")
        parsed_bbox = (
            tuple(int(round(float(item))) for item in bbox)
            if isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(item, int | float) and not isinstance(item, bool) for item in bbox)
            else None
        )
        candidates.append(
            (
                TargetEvidence(
                    result_id=str(details.get("result_id") or ""),
                    detection_id=str(detection.get("id") or ""),
                    semantic_target=str(
                        details.get("semantic_target") or detection.get("label") or "target"
                    ),
                    semantic_role=SUPPORTED_SEMANTIC_ROLE,
                    source_rgb=source_rgb,
                    source_depth=Path(str(packet["depth"]["path"])),
                    mask=mask,
                    frame_id=str(packet["camera"].frame_id),
                    intrinsics=dict(packet["camera"].intrinsics),
                    extrinsics=dict(packet["camera"].extrinsics),
                    perception_bundle_id=str(details.get("perception_bundle_id") or ""),
                    observation_id=str(details.get("observation_id") or ""),
                    scene_epoch=_scene_epoch(observation),
                    bbox_xyxy=parsed_bbox,
                ),
                detection,
            )
        )
    return candidates


def _point_depth_geometry_audit(
    evidence: TargetEvidence,
    *,
    target_center_world: np.ndarray,
    target_extent_world: np.ndarray,
    point_xy: tuple[float, float],
    world_points: np.ndarray,
) -> JsonDict:
    """Prove that a point-prompt mask lies on the expected 3D target surface."""

    try:
        mask = np.asarray(Image.open(evidence.mask).convert("L")) > 0
        depth_raw = np.asarray(Image.open(evidence.source_depth))
    except (OSError, ValueError) as exc:
        raise ActiveVisionError("target_rgbd_artifact_unreadable", str(exc)) from exc
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    if mask.ndim != 2 or depth_raw.ndim != 2 or mask.shape != depth_raw.shape:
        raise ActiveVisionError(
            "target_rgbd_alignment_invalid",
            "point-grounded mask and depth must be aligned two-dimensional images",
        )
    x, y = int(round(point_xy[0])), int(round(point_xy[1]))
    reasons: list[str] = []
    if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
        reasons.append("projected_target_outside_image")
    elif not bool(mask[y, x]):
        reasons.append("projected_target_point_outside_mask")
    scale = float(evidence.intrinsics.get("scale") or 1000.0)
    measured_depth = (
        float(depth_raw[y, x]) / scale
        if not reasons and math.isfinite(float(depth_raw[y, x])) and depth_raw[y, x] > 0
        else math.nan
    )
    if not math.isfinite(measured_depth):
        reasons.append("projected_target_depth_missing")
    camera_from_world = np.linalg.inv(_extrinsics_matrix(evidence.extrinsics))
    target_camera = camera_from_world[:3, :3] @ target_center_world + camera_from_world[:3, 3]
    expected_depth = float(target_camera[2])
    if not math.isfinite(expected_depth) or expected_depth <= 0.0:
        reasons.append("projected_target_behind_camera")
    fx = float(evidence.intrinsics.get("fx") or 0.0)
    fy = float(evidence.intrinsics.get("fy") or 0.0)
    if not math.isfinite(scale) or scale <= 0.0 or fx <= 0.0 or fy <= 0.0:
        raise ActiveVisionError(
            "target_intrinsics_invalid",
            "point-grounded depth geometry requires calibrated pinhole intrinsics",
        )
    half_extent_along_ray = float(
        np.abs(camera_from_world[2, :3]) @ (np.maximum(target_extent_world, 0.0) * 0.5)
    )
    pixel_footprint_m = expected_depth / min(fx, fy) if expected_depth > 0.0 else 0.0
    depth_tolerance_m = half_extent_along_ray + max(1.0 / scale, 2.0 * pixel_footprint_m)
    depth_residual_m = (
        abs(measured_depth - expected_depth)
        if math.isfinite(measured_depth) and expected_depth > 0.0
        else math.inf
    )
    if math.isfinite(depth_residual_m) and depth_residual_m > depth_tolerance_m:
        reasons.append("projected_target_depth_mismatch")
    centroid_distance_m = math.inf
    centroid_tolerance_m = 0.5 * float(np.linalg.norm(target_extent_world)) + depth_tolerance_m
    if len(world_points):
        centroid = np.median(world_points, axis=0)
        centroid_distance_m = float(np.linalg.norm(centroid - target_center_world))
        if centroid_distance_m > centroid_tolerance_m:
            reasons.append("mask_centroid_misses_target_geometry")
    else:
        reasons.append("mask_has_no_metric_points")
    return {
        "schema_version": "openeta.active_vision_point_depth_geometry.v1",
        "passed": not reasons,
        "reason_codes": reasons,
        "point_xy": [round(point_xy[0], 3), round(point_xy[1], 3)],
        "expected_depth_m": round(expected_depth, 6),
        "measured_depth_m": (round(measured_depth, 6) if math.isfinite(measured_depth) else None),
        "depth_residual_m": (
            round(depth_residual_m, 6) if math.isfinite(depth_residual_m) else None
        ),
        "depth_tolerance_m": round(depth_tolerance_m, 6),
        "mask_centroid_distance_m": (
            round(centroid_distance_m, 6) if math.isfinite(centroid_distance_m) else None
        ),
        "mask_centroid_tolerance_m": round(centroid_tolerance_m, 6),
    }


def _select_point_grounded_evidence(
    result: ToolResult,
    *,
    observation: EnvObservation,
    artifact_root: Path,
    profile: GraspRgbdQualityProfile,
    target_center_world: np.ndarray,
    target_extent_world: np.ndarray,
    point_xy: tuple[float, float],
) -> tuple[TargetEvidence | None, JsonDict | None, JsonDict]:
    """Select one nested SAM mask using only calibrated RGB-D evidence."""

    audits: list[JsonDict] = []
    passed: list[tuple[tuple[Any, ...], TargetEvidence, JsonDict, JsonDict, JsonDict]] = []
    for evidence, detection in _point_grounded_evidence_candidates(
        result,
        observation=observation,
        artifact_root=artifact_root,
    ):
        quality, points = _target_quality(evidence, profile=profile)
        geometry = _point_depth_geometry_audit(
            evidence,
            target_center_world=target_center_world,
            target_extent_world=target_extent_world,
            point_xy=point_xy,
            world_points=points,
        )
        accepted = quality.get("passed") is True and geometry.get("passed") is True
        audit = {
            "detection_id": evidence.detection_id,
            "accepted": accepted,
            "quality": quality,
            "point_depth_geometry": geometry,
        }
        audits.append(audit)
        if not accepted:
            continue
        score = detection.get("score")
        score_value = (
            float(score)
            if isinstance(score, int | float) and not isinstance(score, bool)
            else -math.inf
        )
        depth_residual = geometry.get("depth_residual_m")
        centroid_distance = geometry.get("mask_centroid_distance_m")
        passed.append(
            (
                (
                    float(depth_residual) if depth_residual is not None else math.inf,
                    float(centroid_distance) if centroid_distance is not None else math.inf,
                    float(quality.get("mask_area_fraction") or math.inf),
                    -score_value,
                    evidence.detection_id,
                ),
                evidence,
                quality,
                geometry,
                detection,
            )
        )
    selection = {
        "schema_version": "openeta.active_vision_mask_selection.v1",
        "model_review_invoked": False,
        "candidate_count": len(audits),
        "candidate_audits": audits,
        "decision": "reject",
    }
    if not passed:
        return None, None, selection
    _, evidence, quality, geometry, detection = min(passed, key=lambda item: item[0])
    selected = {
        **detection,
        "selection_source": "active_vision_point_depth_geometry",
        "selection_reason": (
            "Calibrated projected point, aligned depth, target geometry, and "
            "grasp RGB-D quality all passed."
        ),
    }
    result.details = {
        **dict(result.details),
        "selected_detection": selected,
        "selection_required": False,
        "selection_review": {
            **selection,
            "decision": "select",
            "detection_id": evidence.detection_id,
            "selection_source": "active_vision_point_depth_geometry",
        },
    }
    quality = {**quality, "point_depth_geometry": geometry}
    return evidence, quality, result.details["selection_review"]


def _target_hint_world_geometry(
    value: object,
    *,
    observation: EnvObservation,
    artifact_root: Path,
) -> tuple[np.ndarray, np.ndarray, JsonDict]:
    """Lift one host-retained visual point into a coarse metric search target.

    This path is used only when text and point segmentation could not produce a
    mask.  The point comes from the bounded visual localizer; calibrated depth
    supplies metric geometry.  It does not inspect simulator object identities
    or inject scene-specific poses.
    """

    if not isinstance(value, Mapping):
        raise ActiveVisionError(
            "active_search_hint_missing",
            "target_hint with one current image point is required for semantic search",
        )
    source_rgb = _resolve_artifact_path(value.get("source_image"), artifact_root)
    raw_points = value.get("positive_points") or value.get("points")
    points = [item for item in raw_points or [] if isinstance(item, Mapping)]
    if not points:
        raise ActiveVisionError(
            "active_search_point_missing",
            "target_hint must contain one visual foreground point",
        )
    point = points[0]
    try:
        u = float(point.get("x"))
        v = float(point.get("y"))
    except (TypeError, ValueError) as exc:
        raise ActiveVisionError(
            "active_search_point_invalid",
            "target_hint point coordinates must be finite numbers",
        ) from exc
    if not math.isfinite(u) or not math.isfinite(v):
        raise ActiveVisionError(
            "active_search_point_invalid",
            "target_hint point coordinates must be finite numbers",
        )
    packet = _camera_packet_for_source(observation, source_rgb=source_rgb, frame_id="")
    try:
        depth_raw = np.asarray(Image.open(str(packet["depth"]["path"])))
    except (OSError, ValueError) as exc:
        raise ActiveVisionError("active_search_depth_unreadable", str(exc)) from exc
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    if depth_raw.ndim != 2:
        raise ActiveVisionError(
            "active_search_depth_invalid",
            "target_hint depth must be one calibrated two-dimensional image",
        )
    height, width = depth_raw.shape
    x = int(round(u))
    y = int(round(v))
    if not 0 <= x < width or not 0 <= y < height:
        raise ActiveVisionError(
            "active_search_point_outside_image",
            "target_hint point lies outside its source image",
        )
    camera = packet["camera"]
    scale = float(camera.intrinsics.get("scale") or 1000.0)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ActiveVisionError(
            "active_search_depth_scale_invalid",
            "target_hint depth scale is invalid",
        )
    depth = depth_raw.astype(np.float64) / scale
    radius = max(2, int(round(min(width, height) * 0.01)))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    window = depth[y0:y1, x0:x1]
    valid_y, valid_x = np.nonzero(np.isfinite(window) & (window > 0.0))
    if not len(valid_x):
        raise ActiveVisionError(
            "active_search_point_depth_missing",
            "target_hint has no finite metric depth in its local image neighbourhood",
        )
    image_x = valid_x + x0
    image_y = valid_y + y0
    pixel_distance_sq = (image_x - u) ** 2 + (image_y - v) ** 2
    nearest = int(np.argmin(pixel_distance_sq))
    sample_x = float(image_x[nearest])
    sample_y = float(image_y[nearest])
    sample_depth = float(depth[int(image_y[nearest]), int(image_x[nearest])])
    fx, fy, cx, cy = (
        float(camera.intrinsics[name]) for name in ("fx", "fy", "cx", "cy")
    )
    if not all(math.isfinite(item) for item in (fx, fy, cx, cy)) or fx <= 0.0 or fy <= 0.0:
        raise ActiveVisionError(
            "active_search_intrinsics_invalid",
            "target_hint camera intrinsics are invalid",
        )
    camera_point = np.array(
        [
            (sample_x - cx) * sample_depth / fx,
            (sample_y - cy) * sample_depth / fy,
            sample_depth,
        ],
        dtype=np.float64,
    )
    camera_to_world = _extrinsics_matrix(camera.extrinsics)
    center = camera_to_world[:3, :3] @ camera_point + camera_to_world[:3, 3]
    # A tight visual box supplies lateral target scale even when segmentation
    # is unavailable.  It cannot prove depth thickness, so retain the local
    # metric uncertainty along the optical axis and rotate that conservative
    # camera-frame box into a world-axis extent.  Without a valid box, keep the
    # historical isotropic point uncertainty.
    lateral_uncertainty = max(
        sample_depth * (2.0 * radius + 1.0) / min(fx, fy),
        1e-3,
    )
    bbox = value.get("bbox_xyxy")
    parsed_bbox: tuple[float, float, float, float] | None = None
    if bbox is not None:
        if not (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(
                isinstance(item, int | float)
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in bbox
            )
        ):
            raise ActiveVisionError(
                "active_search_bbox_invalid",
                "target_hint bbox_xyxy must contain four finite pixel coordinates",
            )
        left, top, right, bottom = (float(item) for item in bbox)
        if not (
            0.0 <= left < right <= width
            and 0.0 <= top < bottom <= height
            and left <= u <= right
            and top <= v <= bottom
        ):
            raise ActiveVisionError(
                "active_search_bbox_invalid",
                "target_hint bbox must lie inside the image and contain its point",
            )
        parsed_bbox = left, top, right, bottom
        camera_extent = np.array(
            [
                (right - left) * sample_depth / fx,
                (bottom - top) * sample_depth / fy,
                lateral_uncertainty,
            ],
            dtype=np.float64,
        )
        extent = np.abs(camera_to_world[:3, :3]) @ camera_extent
        extent = np.maximum(extent, lateral_uncertainty)
    else:
        extent = np.array(
            [lateral_uncertainty, lateral_uncertainty, lateral_uncertainty],
            dtype=np.float64,
        )
    return (
        center,
        extent,
        {
            "source_image": str(source_rgb),
            "source_frame_id": camera.frame_id,
            "requested_point_xy": [round(u, 3), round(v, 3)],
            "depth_sample_point_xy": [sample_x, sample_y],
            "depth_m": round(sample_depth, 6),
            "neighbourhood_radius_px": radius,
            "center_world_xyz": _rounded_vector(center),
            "metric_uncertainty_m": round(lateral_uncertainty, 6),
            "bbox_xyxy": (
                [round(item, 3) for item in parsed_bbox] if parsed_bbox is not None else None
            ),
            "extent_world_xyz": _rounded_vector(extent),
        },
    )


def _target_quality(
    evidence: TargetEvidence,
    *,
    profile: GraspRgbdQualityProfile,
) -> tuple[JsonDict, np.ndarray]:
    try:
        mask = np.asarray(Image.open(evidence.mask).convert("L")) > 0
        depth_raw = np.asarray(Image.open(evidence.source_depth))
    except (OSError, ValueError) as exc:
        raise ActiveVisionError("target_rgbd_artifact_unreadable", str(exc)) from exc
    if mask.ndim != 2 or depth_raw.ndim not in {2, 3}:
        raise ActiveVisionError("target_rgbd_shape_invalid", "mask/depth must be aligned 2D images")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    if mask.shape != depth_raw.shape:
        raise ActiveVisionError("target_rgbd_alignment_invalid", "mask and depth sizes differ")
    area = int(mask.sum())
    total = int(mask.size)
    scale = float(evidence.intrinsics.get("scale") or 1000.0)
    if not math.isfinite(scale) or scale <= 0:
        raise ActiveVisionError("target_intrinsics_scale_invalid", "depth scale is invalid")
    depth = depth_raw.astype(np.float64) / scale
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    valid_count = int(valid.sum())
    ys, xs = np.nonzero(mask)
    if area:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        border_margin = min(bbox[0], bbox[1], mask.shape[1] - bbox[2], mask.shape[0] - bbox[3])
    else:
        bbox = (0, 0, 0, 0)
        border_margin = -1
    area_fraction = area / max(1, total)
    valid_fraction = valid_count / max(1, area)
    reasons: list[str] = []
    if area < profile.min_mask_area_px or area_fraction < profile.min_mask_area_fraction:
        reasons.append("target_too_small")
    if area_fraction > profile.max_mask_area_fraction:
        reasons.append("mask_oversegmented")
    if valid_fraction < profile.min_valid_depth_fraction:
        reasons.append("insufficient_target_depth")
    if border_margin < profile.min_border_margin_px:
        reasons.append("target_touches_image_border")
    points = _masked_world_points(
        mask=valid,
        depth=depth,
        intrinsics=evidence.intrinsics,
        extrinsics=evidence.extrinsics,
    )
    if len(points) < profile.min_mask_area_px:
        reasons.append("insufficient_metric_target_points")
    quality = {
        "profile": SUPPORTED_QUALITY_PROFILE,
        "passed": not reasons,
        "reason_codes": reasons,
        "mask_area_px": area,
        "mask_area_fraction": round(area_fraction, 8),
        "valid_depth_fraction": round(valid_fraction, 6),
        "border_margin_px": int(border_margin),
        "bbox_xyxy": list(bbox),
        "metric_point_count": int(len(points)),
        "source_frame_id": evidence.frame_id,
        "rgb_sha256": _file_sha256(evidence.source_rgb),
        "depth_sha256": _file_sha256(evidence.source_depth),
        "mask_sha256": _file_sha256(evidence.mask),
    }
    return quality, points


def _masked_world_points(
    *,
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: Mapping[str, Any],
    extrinsics: Mapping[str, Any],
) -> np.ndarray:
    fx, fy, cx, cy = (
        float(intrinsics[name]) for name in ("fx", "fy", "cx", "cy")
    )
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ActiveVisionError("target_intrinsics_invalid", "pinhole intrinsics are invalid")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.empty((0, 3), dtype=np.float64)
    if len(xs) > 100_000:
        stride = int(math.ceil(len(xs) / 100_000))
        xs, ys = xs[::stride], ys[::stride]
    zs = depth[ys, xs]
    camera_points = np.column_stack(((xs - cx) * zs / fx, (ys - cy) * zs / fy, zs))
    transform = _extrinsics_matrix(extrinsics)
    return camera_points @ transform[:3, :3].T + transform[:3, 3]


def _active_rgbd_packets(observation: EnvObservation) -> list[JsonDict]:
    """Return complete calibrated views in deterministic utility order."""

    artifacts = observation.metadata.get("image_artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    packets: list[tuple[tuple[int, int, str], JsonDict]] = []
    for camera in observation.cameras:
        role = f"{camera.role or ''} {camera.frame_id}".lower()
        rgb = next(
            (
                value
                for value in artifacts
                if isinstance(value, Mapping)
                and value.get("kind") == "rgb"
                and value.get("frame_id") == camera.frame_id
                and isinstance(value.get("path"), str)
                and Path(str(value["path"])).is_file()
            ),
            None,
        )
        depth = next(
            (
                value
                for value in artifacts
                if isinstance(value, Mapping)
                and value.get("kind") == "depth"
                and value.get("frame_id") == camera.frame_id
                and isinstance(value.get("path"), str)
                and Path(str(value["path"])).is_file()
            ),
            None,
        )
        if rgb is not None and depth is not None and camera.intrinsics and camera.extrinsics:
            is_wrist = "wrist" in role
            is_primary = "primary" in role
            packets.append(
                (
                    (int(not is_primary), int(is_wrist), camera.frame_id),
                    {
                        "camera": camera,
                        "rgb": dict(rgb),
                        "depth": dict(depth),
                        "is_wrist": is_wrist,
                    },
                )
            )
    return [packet for _, packet in sorted(packets, key=lambda item: item[0])]


def _wrist_rgbd_packet(observation: EnvObservation) -> JsonDict:
    for packet in _active_rgbd_packets(observation):
        if packet["is_wrist"] is True:
            return packet
    raise ActiveVisionError(
        "wrist_rgbd_unavailable",
        "current observation has no complete calibrated wrist RGB-D packet",
    )


def _semantic_search_source_image(
    observation: EnvObservation,
) -> tuple[Path, tuple[int, int]]:
    """Choose one current calibrated scene RGB image without sample-specific hints."""

    artifacts = observation.metadata.get("image_artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    camera_by_frame = {camera.frame_id: camera for camera in observation.cameras}
    candidates: list[tuple[tuple[int, int, int], Path]] = []
    for index, artifact in enumerate(artifacts):
        if not (
            isinstance(artifact, Mapping)
            and artifact.get("kind") == "rgb"
            and isinstance(artifact.get("path"), str)
        ):
            continue
        path = Path(str(artifact["path"])).expanduser()
        if not path.is_file():
            continue
        frame_id = str(artifact.get("frame_id") or "")
        camera = camera_by_frame.get(frame_id)
        if camera is None or not camera.intrinsics or not camera.extrinsics:
            continue
        role = f"{artifact.get('role') or ''} {camera.role or ''} {frame_id}".lower()
        is_wrist = "wrist" in role
        is_primary = "primary" in role
        candidates.append(((int(is_wrist), int(not is_primary), index), path.resolve()))
    for _, source_rgb in sorted(candidates, key=lambda item: item[0]):
        try:
            _camera_packet_for_source(observation, source_rgb=source_rgb, frame_id="")
            with Image.open(source_rgb) as image:
                width, height = image.size
        except (ActiveVisionError, OSError, ValueError):
            continue
        if width > 0 and height > 0:
            return source_rgb, (int(width), int(height))
    raise ActiveVisionError(
        "semantic_search_rgbd_unavailable",
        "current observation has no complete calibrated RGB-D image for visual grounding",
        infrastructure=True,
    )


def _semantic_localization_receipt(
    localization: ReferencePointLocalization,
) -> JsonDict:
    details = localization.details if isinstance(localization.details, dict) else {}
    return {
        "schema_version": "openeta.semantic_point_localization.v1",
        "source": "isolated_provider_visual_grounding",
        "point_xy": [round(localization.x, 3), round(localization.y, 3)],
        "bbox_xyxy": (
            [round(value, 3) for value in localization.bbox_xyxy]
            if localization.bbox_xyxy is not None
            else None
        ),
        "confidence": round(localization.confidence, 6),
        "reason": localization.reason,
        "provider": localization.provider,
        "model": localization.model,
        "latency_s": details.get("latency_s"),
        "provider_details": dict(details.get("provider_details") or {}),
    }


def _mechanical_camera_preflight(wrist: Mapping[str, Any]) -> JsonDict:
    try:
        rgb = np.asarray(Image.open(str(wrist["rgb"]["path"])).convert("RGB"))
        depth_raw = np.asarray(Image.open(str(wrist["depth"]["path"])))
    except (OSError, ValueError) as exc:
        raise ActiveVisionError("wrist_rgbd_artifact_unreadable", str(exc)) from exc
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    camera = wrist["camera"]
    scale = float(camera.intrinsics.get("scale") or 1000.0)
    depth = depth_raw.astype(np.float64) / scale
    near = np.isfinite(depth) & (depth > 0.0) & (depth < SELF_OCCLUSION_NEAR_M)
    near_fraction = float(near.mean()) if near.size else 1.0
    black_fraction = float((rgb.max(axis=2) <= 16).mean()) if rgb.size else 1.0
    reasons: list[str] = []
    if near_fraction > SELF_OCCLUSION_MAX_NEAR_FRACTION:
        reasons.append("near_robot_occupancy_excessive")
    if black_fraction > SELF_OCCLUSION_MAX_BLACK_FRACTION:
        reasons.append("near_black_silhouette_excessive")
    return {
        "passed": not reasons,
        "reason_codes": reasons,
        "near_field_threshold_m": SELF_OCCLUSION_NEAR_M,
        "near_field_fraction": round(near_fraction, 6),
        "near_black_fraction": round(black_fraction, 6),
        "frame_id": camera.frame_id,
    }


def _generate_view_candidates(
    *,
    observation: EnvObservation,
    wrist: Mapping[str, Any],
    target_center_world: np.ndarray,
    target_extent_world: np.ndarray,
) -> tuple[list[JsonDict], Counter[str]]:
    current_eef = _pose_matrix(observation.robot.end_effector_pose)
    current_camera = _extrinsics_matrix(wrist["camera"].extrinsics)
    eef_to_camera = np.linalg.inv(current_eef) @ current_camera
    outward_azimuth = math.atan2(float(target_center_world[1]), float(target_center_world[0]))
    distance = min(0.44, max(0.32, 2.5 * float(np.max(target_extent_world))))
    offsets_deg = (0, -30, 30, -60, 60, -90, 90, 180)
    elevations_deg = (55, 45, 65)
    generated: list[JsonDict] = []
    rejected: Counter[str] = Counter()
    fixed_index = 0
    for offset_deg in offsets_deg:
        for elevation_deg in elevations_deg:
            azimuth = outward_azimuth + math.radians(offset_deg)
            elevation = math.radians(elevation_deg)
            horizontal = distance * math.cos(elevation)
            camera_position = target_center_world + np.array(
                [
                    horizontal * math.cos(azimuth),
                    horizontal * math.sin(azimuth),
                    distance * math.sin(elevation),
                ],
                dtype=np.float64,
            )
            camera_rotation = _look_at_opencv(camera_position, target_center_world)
            world_camera = np.eye(4, dtype=np.float64)
            world_camera[:3, :3] = camera_rotation
            world_camera[:3, 3] = camera_position
            world_eef = world_camera @ np.linalg.inv(eef_to_camera)
            reason = _analytic_view_rejection(
                camera_position=camera_position,
                eef_position=world_eef[:3, 3],
                target_center=target_center_world,
            )
            candidate_id = f"active_view_{fixed_index:03d}"
            fixed_index += 1
            if reason:
                rejected[reason] += 1
                continue
            quaternion = _matrix_to_quaternion(world_eef[:3, :3])
            travel = float(np.linalg.norm(world_eef[:3, 3] - current_eef[:3, 3]))
            score = 1.0 / (1.0 + travel + abs(offset_deg) / 180.0)
            stage = {
                "name": "observation",
                "grasp_stage": "observation",
                "frame": "world",
                "xyz": _rounded_vector(world_eef[:3, 3]),
                "quat_xyzw": _rounded_vector(quaternion),
            }
            generated.append(
                {
                    "id": candidate_id,
                    "candidate_source": "active_vision",
                    "score": round(score, 9),
                    "viewpoint": {
                        "camera_world_xyz": _rounded_vector(camera_position),
                        "target_world_xyz": _rounded_vector(target_center_world),
                        "azimuth_offset_deg": offset_deg,
                        "elevation_deg": elevation_deg,
                        "distance_m": round(distance, 6),
                    },
                    "qualification_stages": [stage],
                }
            )
    generated.sort(
        key=lambda value: (
            -float(value["score"]),
            str(value["id"]),
        )
    )
    return generated[:MAX_VIEW_CANDIDATES], rejected


def _analytic_view_rejection(
    *,
    camera_position: np.ndarray,
    eef_position: np.ndarray,
    target_center: np.ndarray,
) -> str:
    if not np.isfinite(camera_position).all() or not np.isfinite(eef_position).all():
        return "non_finite_pose"
    camera_distance = float(np.linalg.norm(camera_position - target_center))
    if camera_position[2] < target_center[2] + 0.16 or camera_position[2] < 0.12:
        return "camera_below_strict_clearance"
    if not 0.20 <= camera_distance <= 0.55:
        return "camera_distance_out_of_bounds"
    # RM75's exact directional reach remains a MoveIt decision.  This only
    # rejects points beyond a sphere larger than the arm's summed link length.
    if float(np.linalg.norm(eef_position)) > 1.20:
        return "mathematically_outside_maximum_workspace"
    if eef_position[2] < 0.08:
        return "eef_below_work_surface_clearance"
    return ""


def _look_at_opencv(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - camera_position
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-9:
        raise ActiveVisionError("active_view_rotation_invalid", "camera coincides with target")
    forward /= norm
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) <= 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    rotation = np.column_stack((right, down, forward))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ActiveVisionError("active_view_rotation_invalid", "look-at rotation is not orthonormal")
    return rotation


def _project_world_point(
    point_world: np.ndarray,
    *,
    extrinsics: Mapping[str, Any],
    intrinsics: Mapping[str, Any],
) -> tuple[float, float] | None:
    camera_from_world = np.linalg.inv(_extrinsics_matrix(extrinsics))
    point = camera_from_world[:3, :3] @ point_world + camera_from_world[:3, 3]
    if not np.isfinite(point).all() or point[2] <= 1e-6:
        return None
    fx, fy, cx, cy = (float(intrinsics[name]) for name in ("fx", "fy", "cx", "cy"))
    u = fx * point[0] / point[2] + cx
    v = fy * point[1] / point[2] + cy
    width = int(intrinsics.get("width") or round(cx * 2))
    height = int(intrinsics.get("height") or round(cy * 2))
    if not (2 <= u < width - 2 and 2 <= v < height - 2):
        return None
    return float(u), float(v)


def _camera_packet_for_source(
    observation: EnvObservation,
    *,
    source_rgb: Path,
    frame_id: str,
) -> JsonDict:
    artifacts = observation.metadata.get("image_artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    matching_rgb = [
        value
        for value in artifacts
        if isinstance(value, Mapping)
        and value.get("kind") == "rgb"
        and isinstance(value.get("path"), str)
        and _same_file(Path(str(value["path"])), source_rgb)
    ]
    if len(matching_rgb) != 1:
        raise ActiveVisionError(
            "stale_target_evidence",
            "target RGB is not the unique current-observation artifact",
        )
    resolved_frame = str(matching_rgb[0].get("frame_id") or frame_id)
    camera = next(
        (candidate for candidate in observation.cameras if candidate.frame_id == resolved_frame),
        None,
    )
    depth = next(
        (
            value
            for value in artifacts
            if isinstance(value, Mapping)
            and value.get("kind") == "depth"
            and value.get("frame_id") == resolved_frame
            and isinstance(value.get("path"), str)
            and Path(str(value["path"])).is_file()
        ),
        None,
    )
    if camera is None or depth is None or not camera.intrinsics or not camera.extrinsics:
        raise ActiveVisionError(
            "target_rgbd_calibration_missing",
            "target evidence has no aligned current depth/calibration packet",
        )
    return {"camera": camera, "rgb": dict(matching_rgb[0]), "depth": dict(depth)}


def _observation_from_result(result: ToolResult) -> EnvObservation:
    receipt = _environment_receipt(result)
    snapshot = receipt.get("observation_snapshot") if isinstance(receipt, Mapping) else None
    payload = snapshot.get("observation") if isinstance(snapshot, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ActiveVisionError(
            "active_view_observation_snapshot_missing",
            "observe succeeded without a materialized observation snapshot",
            infrastructure=True,
        )
    return EnvObservation.from_dict(dict(payload))


def _environment_receipt(result: ToolResult) -> JsonDict:
    receipt = result.details.get("environment_receipt")
    return dict(receipt) if isinstance(receipt, Mapping) else {}


def _motion_outcome_unknown(result: ToolResult) -> bool:
    outputs = result.details.get("outputs")
    return bool(
        isinstance(outputs, Mapping)
        and (
            outputs.get("motion_outcome") == "unknown"
            or outputs.get("reconciliation_required") is True
        )
    )


def _observation_is_attached(observation: EnvObservation) -> bool:
    physical = observation.metadata.get("physical_verification")
    if isinstance(physical, Mapping) and physical.get("grasp_confirmed") is True:
        return True
    gripper = observation.robot.gripper_state
    return bool(isinstance(gripper, Mapping) and gripper.get("attached") is True)


def _scene_epoch(observation: EnvObservation) -> int:
    value = observation.metadata.get("scene_epoch", 0)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _planning_scene_revision(observation: EnvObservation) -> int:
    value = observation.metadata.get("planning_scene_revision", 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ActiveVisionError(
            "planning_scene_revision_missing",
            "current observation lacks a valid PlanningScene revision",
            infrastructure=True,
        )
    return value


def _pose_matrix(value: Mapping[str, Any]) -> np.ndarray:
    xyz = value.get("xyz") or value.get("pos")
    quaternion = value.get("quat_xyzw")
    if not _finite_sequence(xyz, 3) or not _finite_sequence(quaternion, 4):
        raise ActiveVisionError("eef_pose_invalid", "current EEF pose is incomplete")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _quaternion_to_matrix(quaternion)
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def _extrinsics_matrix(value: Mapping[str, Any]) -> np.ndarray:
    if str(value.get("frame_transform") or "camera_to_world") != "camera_to_world":
        raise ActiveVisionError(
            "camera_extrinsics_direction_invalid",
            "active_vision requires camera_to_world extrinsics",
        )
    if str(value.get("camera_frame") or "").lower() != "opencv":
        raise ActiveVisionError(
            "camera_convention_invalid",
            "active_vision requires explicit OpenCV camera calibration",
        )
    xyz = value.get("pos") or value.get("xyz")
    if not _finite_sequence(xyz, 3):
        raise ActiveVisionError("camera_extrinsics_invalid", "camera position is invalid")
    quaternion = value.get("quat_xyzw")
    flat_matrix = value.get("mat")
    if _finite_sequence(quaternion, 4):
        rotation = _quaternion_to_matrix(quaternion)
    elif _finite_sequence(flat_matrix, 9):
        rotation = np.asarray(flat_matrix, dtype=np.float64).reshape(3, 3)
    else:
        raise ActiveVisionError("camera_extrinsics_invalid", "camera rotation is invalid")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.99:
        raise ActiveVisionError("camera_extrinsics_invalid", "camera rotation is not proper")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def _quaternion_to_matrix(value: Sequence[Any]) -> np.ndarray:
    x, y, z, w = (float(item) for item in value)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ActiveVisionError("quaternion_invalid", "quaternion norm is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quaternion = np.array(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quaternion = np.array(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion *= -1
    return quaternion


def _resolve_artifact_path(value: object, artifact_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ActiveVisionError("artifact_reference_missing", "required artifact reference is empty")
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_absolute():
        raise ActiveVisionError("artifact_reference_missing", f"artifact does not exist: {path}")
    for parent in (Path.cwd(), *artifact_root.resolve().parents):
        candidate = parent / path
        if candidate.is_file():
            return candidate.resolve()
    raise ActiveVisionError("artifact_reference_missing", f"artifact does not exist: {path}")


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _finite_sequence(value: object, length: int) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    )


def _bounded_integer(value: object, *, minimum: int, maximum: int, field: str) -> int:
    if isinstance(value, bool):
        raise ActiveVisionError(f"invalid_{field}", f"{field} must be an integer")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ActiveVisionError(f"invalid_{field}", f"{field} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise ActiveVisionError(
            f"invalid_{field}", f"{field} must be in [{minimum}, {maximum}]"
        )
    return parsed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _rounded_vector(value: Sequence[Any]) -> list[float]:
    return [round(float(item), 9) for item in value]


def _evidence_receipt(evidence: TargetEvidence) -> JsonDict:
    return {
        "result_id": evidence.result_id,
        "detection_id": evidence.detection_id,
        "semantic_role": evidence.semantic_role,
        "semantic_target": evidence.semantic_target,
        "source_frame_id": evidence.frame_id,
        "perception_bundle_id": evidence.perception_bundle_id,
        "observation_id": evidence.observation_id,
        "scene_epoch": evidence.scene_epoch,
        "rgb_sha256": _file_sha256(evidence.source_rgb),
        "depth_sha256": _file_sha256(evidence.source_depth),
        "mask_sha256": _file_sha256(evidence.mask),
    }
