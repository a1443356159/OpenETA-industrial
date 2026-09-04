"""Session memory for lightweight embodied agents."""

from __future__ import annotations

import json
import hashlib
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from adapter.protocol import EnvAction, EnvObservation, JsonDict
from agent.runtime.conversation import (
    ConversationHistory,
    ConversationItem,
    checkpoint_record,
    item_record,
)
from agent.runtime.calibration_registry import (
    DEFAULT_GRASP_CALIBRATION_PROFILE,
    load_grasp_calibration_capabilities,
)
from agent.runtime.release_evidence import (
    known_gripper_open_terminal_dispatch,
    ordered_native_release_proof,
)


PENDING_SAM3_SELECTION_KEY = "pending_sam3_selection"
SELECTED_SAM3_DETECTION_KEY = "selected_sam3_detection"
# Successful SAM3 results, including a point-grounded refresh performed inside
# active_observe, feed the same shared selection flow.
SELECTION_CAPTURE_TOOL_NAMES = frozenset({"sam3", "active_observe"})
PENDING_REFERENCE_LOCALIZATION_KEY = "pending_reference_localization"
REFERENCE_LOCALIZATION_FAILURE_KEY = "reference_localization_failure"
TARGET_LOCALIZATION_BUDGET_KEY = "target_localization_budget"
TARGET_ASSET_REFERENCE_KEY = "target_asset_reference"
SAM3_NO_DETECTION_KEY = "sam3_no_detection"
SAM3_SEMANTIC_STATE_KEY = "sam3_semantic_state"
SAM3_SEMANTIC_ROLES = frozenset({"grasp_target", "placement_object", "placement_region"})
GRASP_CANDIDATE_POLICY_KEY = "grasp_candidate_policy"
LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY = "anygrasp_candidate_policy"
GRASP_REESTIMATION_KEY = "grasp_reestimation"
ARTICULATED_ATTACHMENT_PROBE_KEY = "articulated_attachment_probe"
GRASP_EXECUTION_KEY = "grasp_execution"
GRASP_RECOVERY_KEY = "grasp_recovery"
GRASP_ESTIMATION_RECOVERY_KEY = "grasp_estimation_recovery"
GRIPPER_COMMAND_STATE_KEY = "gripper_command_state"
ATTACHMENT_GATE_KEY = "attachment_gate"
PLACEMENT_RELEASE_KEY = "placement_release"
TASK_COMPLETION_EVIDENCE_KEY = "task_completion_evidence"
PLACEMENT_CANDIDATE_POLICY_KEY = "placement_candidate_policy"
PLACEMENT_OBJECT_DETECTION_KEY = "placement_object_detection"
PLACEMENT_REGION_DETECTION_KEY = "placement_region_detection"
FROZEN_PLACEMENT_POOL_KEY = "frozen_placement_goal_pool"
COMPLETED_PLACEMENT_SUBGOALS_KEY = "completed_placement_subgoals"
WORK_ORDER_KEY = "work_order"
MULTI_SORT_PROGRESS_KEY = "multi_sort_progress"
MOTION_RECONCILIATION_KEY = "motion_reconciliation"
SCENE_EPOCH_KEY = "scene_epoch"
PLANNING_SCENE_TARGET_POSE_SYNC_KEY = "planning_scene_target_pose_sync"
TRANSITION_LEDGER_KEY = "transition_ledger"
ACTIVE_ENVIRONMENT_TASK_KEY = "active_environment_task"
NATIVE_GRASP_SCHEMA_VERSION = "openeta.gazebo.native_grasp.v1"
NATIVE_GRASP_MAXIMUM_DRIFT_M = 0.01
GRASP_CANDIDATE_MAX_ATTEMPTS = 3
ARTICULATED_HANDLE_APPROACH_MODES = ("top_down", "front", "side")
TRANSITION_LEDGER_LIMIT = 32
GRASP_GEOMETRY_FAMILIES = {
    "upright_can",
    "upright_bottle",
    "boxed_item",
    "bowl",
    "apple",
    "articulated_handle",
    "drawer_handle",
    "other",
    "unknown",
}


class MemoryStore(Protocol):
    """Persistence boundary for session trace and working memory."""

    def start_session(
        self,
        *,
        session_id: str,
        task: str,
        metadata: JsonDict | None = None,
    ) -> None: ...

    def append_event(self, event: "MemoryEvent") -> None: ...

    def append_conversation_record(self, record: JsonDict) -> None: ...

    def load_conversation_records(self, session_id: str) -> list[JsonDict]: ...

    def load_working_memory(self) -> JsonDict: ...

    def save_working_memory(self, memory: "AgentMemory") -> None: ...

    def load_events(self, session_id: str, *, limit: int | None = None) -> list[JsonDict]: ...

    def load_session_metadata(self, session_id: str) -> JsonDict: ...


@dataclass(slots=True)
class MemoryEvent:
    """One durable-enough event in an agent session."""

    event_type: str
    payload: JsonDict
    timestamp_s: float = field(default_factory=time.time)


class AgentMemory:
    """Session log plus working memory.

    Without a store this remains an in-process object. With a store, session
    events are written to JSONL and working memory is loaded/saved as JSON.
    """

    def __init__(self, *, store: MemoryStore | None = None) -> None:
        self.store = store
        self.session_id: str | None = None
        self.task: str | None = None
        self.current_user_request: str = ""
        self.metadata: JsonDict = {}
        self.events: list[MemoryEvent] = []
        self.conversation = ConversationHistory()
        self.facts: dict[str, JsonDict] = {}
        self.artifacts: dict[str, JsonDict] = {}
        self.skill_notes: dict[str, list[JsonDict]] = {}
        self.compact_summary: str = ""

    def start_session(
        self,
        *,
        task: str,
        metadata: JsonDict | None = None,
        session_id: str | None = None,
    ) -> None:
        boot_facts = dict(self.facts) if self.session_id is None else {}
        boot_artifacts = dict(self.artifacts) if self.session_id is None else {}
        self.session_id = session_id or str(uuid4())
        self.task = task
        self.current_user_request = ""
        self.metadata = dict(metadata or {})
        self.events.clear()
        self.conversation.clear()
        self.facts = boot_facts
        self.facts.setdefault(
            SCENE_EPOCH_KEY,
            _memory_fact_entry({"epoch": 0}, source="runtime"),
        )
        self.artifacts = boot_artifacts
        self.skill_notes.clear()
        self.compact_summary = ""
        if self.store is not None:
            self.store.start_session(
                session_id=self.session_id,
                task=task,
                metadata=self.metadata,
            )
            self._save_working_memory()
        self.record(
            "session_start",
            {
                "session_id": self.session_id,
                "task": task,
                "metadata": self.metadata,
            },
        )
        self.begin_user_turn(task, source="session_start")

    def resume_session(
        self,
        session_id: str,
        *,
        task: str = "",
        metadata: JsonDict | None = None,
        max_events: int | None = 64,
    ) -> None:
        self.session_id = session_id
        stored_metadata: JsonDict = {}
        if self.store is not None:
            stored_metadata = self.store.load_session_metadata(session_id)
        self.task = task or str(stored_metadata.get("task") or "")
        self.metadata = {
            **(
                stored_metadata.get("metadata")
                if isinstance(stored_metadata.get("metadata"), dict)
                else {}
            ),
            **dict(metadata or {}),
        }
        self.events.clear()
        self.conversation.clear()
        self.facts.clear()
        self.artifacts.clear()
        self.skill_notes.clear()
        self.compact_summary = ""
        if self.store is not None:
            self.store.start_session(
                session_id=session_id,
                task=self.task or "(resumed)",
                metadata=self.metadata,
            )
            self._load_working_memory()
            for row in self.store.load_events(session_id, limit=max_events):
                event_type = str(row.get("event_type") or "event")
                payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
                timestamp = row.get("timestamp_s")
                try:
                    timestamp_s = float(timestamp)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    timestamp_s = time.time()
                self.events.append(
                    MemoryEvent(
                        event_type=event_type,
                        payload=dict(payload),
                        timestamp_s=timestamp_s,
                    )
                )
            records = self.store.load_conversation_records(session_id)
            if records:
                self.conversation.replay(records)
            else:
                self._reconstruct_legacy_conversation(
                    self.store.load_events(session_id, limit=None)
                )
                for item in self.conversation.items:
                    self._append_conversation_record(item_record(item))
        self.current_user_request = self.conversation.current_user_request or (self.task or "")
        self.record(
            "session_resumed",
            {
                "session_id": session_id,
                "task": self.task,
                "loaded_event_count": len(self.events),
            },
        )

    def begin_user_turn(self, text: str, *, source: str = "user") -> ConversationItem:
        """Record one exact user instruction as a durable conversation turn."""

        item = self.conversation.begin_user_turn(text, source=source)
        self.current_user_request = item.content
        self._append_conversation_record(item_record(item))
        self.record(
            "user_message",
            {
                "turn_id": item.turn_id,
                "source": source,
                "text": item.content,
            },
        )
        return item

    def record(self, event_type: str, payload: JsonDict | None = None) -> MemoryEvent:
        event = MemoryEvent(event_type=event_type, payload=dict(payload or {}))
        self.events.append(event)
        if self.store is not None:
            self.store.append_event(event)
        return event

    def add_observation(self, observation: EnvObservation) -> None:
        observed_epoch = _optional_int(
            observation.metadata.get("scene_epoch"),
            default=self.scene_epoch(),
        )
        if observed_epoch > self.scene_epoch():
            self.facts[SCENE_EPOCH_KEY] = _memory_fact_entry(
                {"epoch": observed_epoch},
                source="environment_observation",
            )
            self.record(
                "scene_epoch_synchronized",
                {"scene_epoch": observed_epoch, "source": "environment_observation"},
            )
        summary = summarize_observation(observation)
        summary["scene_epoch"] = self.scene_epoch()
        summary["runtime_camera_calibrations"] = [
            {
                "frame_id": camera.frame_id,
                "extrinsics": dict(camera.extrinsics),
            }
            for camera in observation.cameras
            if isinstance(camera.extrinsics, dict) and camera.extrinsics
        ]
        runtime_sources: list[JsonDict] = []
        current_sources: list[JsonDict] = []
        seen_sources: set[tuple[str, str]] = set()
        for artifact in observation.metadata.get("image_artifacts", []):
            if (
                not isinstance(artifact, dict)
                or artifact.get("kind") != "rgb"
                or not isinstance(artifact.get("path"), str)
            ):
                continue
            frame_id = str(artifact.get("frame_id") or "")
            rgb_path = str(artifact["path"])
            key = (frame_id, rgb_path)
            if key not in seen_sources:
                seen_sources.add(key)
                source = {"frame_id": frame_id, "rgb_path": rgb_path}
                runtime_sources.append(source)
                current_sources.append(dict(source))
        for artifact in self.artifacts.values():
            value = artifact.get("value") if isinstance(artifact, dict) else None
            if not isinstance(value, dict) or value.get("kind") != "rgbd_camera":
                continue
            frame_id = str(value.get("frame_id") or "")
            rgb_path = str(value.get("rgb_path") or "")
            key = (frame_id, rgb_path)
            if rgb_path and key not in seen_sources:
                seen_sources.add(key)
                runtime_sources.append({"frame_id": frame_id, "rgb_path": rgb_path})
        summary["runtime_camera_sources"] = runtime_sources
        summary["current_camera_sources"] = current_sources
        self.record("observation", summary)
        post_release_visual_updated = (
            self._capture_post_release_visual_observation(observation)
        )
        multi_sort_updated = self._capture_multi_sort_progress(observation)
        self._capture_grasp_reestimation_observation(observation)
        placement_release_reobserved = self._capture_failed_placement_release_reobservation(
            observation
        )
        reconciliation_updated = self._reconcile_unknown_motion(observation)
        attachment_updated = self._capture_attachment_observation_verdict(observation)
        host_completion_updated = self._capture_host_owned_task_completion()
        if (
            multi_sort_updated
            or post_release_visual_updated
            or placement_release_reobserved
            or reconciliation_updated
            or attachment_updated
            or host_completion_updated
        ):
            self._save_working_memory()

    def _capture_post_release_visual_observation(
        self, observation: EnvObservation
    ) -> bool:
        """Bind one fresh RGB view to an already completed native release."""

        release = self.placement_release()
        if not (
            isinstance(release, dict)
            and release.get("status") == "released"
        ):
            return False
        visual = release.get("post_release_visual_observation")
        if isinstance(visual, dict) and visual.get("available") is True:
            return False
        if observation.metadata.get("observation_stale") is True:
            return False
        rgb_artifacts = [
            dict(artifact)
            for artifact in observation.metadata.get("image_artifacts", [])
            if isinstance(artifact, dict)
            and artifact.get("kind") == "rgb"
            and isinstance(artifact.get("path"), str)
            and artifact["path"]
        ]
        if not rgb_artifacts:
            return False
        visual = {
            "schema_version": "openeta.post_release_visual_observation.v1",
            "required": True,
            "available": True,
            "source": "fresh_observe_after_release",
            "review_authority": "vlm",
            "camera_frame_ids": [
                str(artifact.get("frame_id") or "") for artifact in rgb_artifacts
            ],
            "captured_at_s": time.time(),
        }
        release["post_release_visual_observation"] = visual
        evidence = release.get("release_evidence")
        if isinstance(evidence, dict):
            evidence = dict(evidence)
            evidence["post_release_visual_observation"] = dict(visual)
            release["release_evidence"] = evidence
        self.facts[PLACEMENT_RELEASE_KEY] = _memory_fact_entry(
            release,
            source="post_release_visual_observation",
        )
        self.record(
            "post_release_visual_observation_available",
            {
                "candidate_id": release.get("candidate_id"),
                "placement_pose_id": release.get("placement_pose_id"),
                "camera_frame_ids": visual["camera_frame_ids"],
            },
        )
        return True

    def _capture_multi_sort_progress(self, observation: EnvObservation) -> bool:
        progress = observation.metadata.get("multi_sort_progress")
        if not isinstance(progress, dict) or progress.get("schema_version") != (
            "openeta.multi_sort_progress.v1"
        ):
            return False
        try:
            assignment_count = int(progress["assignment_count"])
            completed_count = int(progress["completed_count"])
            remaining_count = int(progress["remaining_count"])
        except (KeyError, TypeError, ValueError):
            return False
        if (
            assignment_count < 1
            or completed_count < 0
            or remaining_count < 0
            or completed_count + remaining_count != assignment_count
            or bool(progress.get("all_completed")) != (remaining_count == 0)
        ):
            return False
        work_order = progress.get("work_order")
        if not (
            progress.get("source") == "vlm_work_order"
            and isinstance(work_order, dict)
            and work_order.get("schema_version") == "openeta.work_order.v1"
            and work_order.get("source") == "vlm_tool_call"
            and isinstance(work_order.get("items"), list)
            and len(work_order["items"]) == assignment_count
        ):
            return False
        normalized = dict(progress)
        previous = _memory_fact_value(self.facts.get(MULTI_SORT_PROGRESS_KEY))
        previous_work_order = _memory_fact_value(self.facts.get(WORK_ORDER_KEY))
        self.facts[WORK_ORDER_KEY] = _memory_fact_entry(
            dict(work_order),
            source="vlm_configure_work_order",
        )
        self.facts[MULTI_SORT_PROGRESS_KEY] = _memory_fact_entry(
            normalized,
            source="environment_multi_sort_progress",
        )
        release = self.placement_release()
        release_progress = (
            release.get("multi_sort_progress") if isinstance(release, dict) else None
        )
        advanced_release = bool(
            isinstance(release, dict)
            and release.get("status") == "released"
            and isinstance(release_progress, dict)
            and int(release_progress.get("completed_count") or -1) == completed_count
            and remaining_count > 0
            and progress.get("fresh_observation_required") is False
            and progress.get("fresh_observation_satisfied") is True
        )
        if advanced_release:
            self.facts.pop(PLACEMENT_RELEASE_KEY, None)
            self.record(
                "multi_sort_next_assignment_observed",
                {
                    "completed_count": completed_count,
                    "remaining_count": remaining_count,
                    "active_assignment_index": progress.get(
                        "active_assignment_index"
                    ),
                    "scene_epoch": self.scene_epoch(),
                },
            )
        if previous != normalized:
            self.record("multi_sort_progress_updated", normalized)
        return previous != normalized or previous_work_order != work_order or advanced_release

    def _capture_failed_placement_release_reobservation(self, observation: EnvObservation) -> bool:
        """Acknowledge the fresh scene required after an unsuccessful release."""

        del observation
        release = self.placement_release()
        if not (
            isinstance(release, dict)
            and release.get("status") == "failed"
            and release.get("reobservation_required") is True
        ):
            return False
        release.update(
            {
                "reobservation_required": False,
                "reobserved_scene_epoch": self.scene_epoch(),
                "reobserved_at_s": time.time(),
            }
        )
        self.facts[PLACEMENT_RELEASE_KEY] = _memory_fact_entry(
            release,
            source="failed_placement_release_reobservation",
        )
        self.record(
            "failed_placement_release_reobserved",
            {
                "candidate_id": release.get("candidate_id"),
                "placement_pose_id": release.get("placement_pose_id"),
                "scene_epoch": self.scene_epoch(),
            },
        )
        return True

    def _capture_grasp_reestimation_observation(self, observation: EnvObservation) -> bool:
        reestimate = _memory_fact_value(self.facts.get(GRASP_REESTIMATION_KEY))
        if not isinstance(reestimate, dict) or reestimate.get("status") != "pending_observation":
            return False
        views = _complete_observation_rgbd_views(observation)
        source_paths = {
            str(path)
            for path in reestimate.get("source_observation_rgb_paths", [])
            if isinstance(path, str) and path
        }
        source_image = str(reestimate.get("source_image") or "")
        if source_image:
            source_paths.add(source_image)
        fresh_views = [
            view for view in views if str(view.get("rgb_path") or "") not in source_paths
        ]
        if not fresh_views:
            return False
        reestimate.update(
            {
                "status": "ready",
                "observed_at_s": time.time(),
                "observation_views": fresh_views,
            }
        )
        self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
            reestimate,
            source="candidate_reestimate_observation",
        )
        for key in (
            PENDING_SAM3_SELECTION_KEY,
            SELECTED_SAM3_DETECTION_KEY,
            GRASP_CANDIDATE_POLICY_KEY,
            LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY,
            ARTICULATED_ATTACHMENT_PROBE_KEY,
            GRASP_EXECUTION_KEY,
            ATTACHMENT_GATE_KEY,
        ):
            self.facts.pop(key, None)
        if reestimate.get("invalidate_frozen_placement_pool") is True:
            for key in (
                PLACEMENT_CANDIDATE_POLICY_KEY,
                PLACEMENT_OBJECT_DETECTION_KEY,
                PLACEMENT_REGION_DETECTION_KEY,
                FROZEN_PLACEMENT_POOL_KEY,
                PLANNING_SCENE_TARGET_POSE_SYNC_KEY,
            ):
                self.facts.pop(key, None)
        for key in (
            "anygrasp_grasp_candidates_latest",
            "grasp_pose_estimate_grasp_candidates_latest",
            "graspgenx_grasp_candidates_latest",
            "contact_graspnet_grasp_candidates_latest",
            "camera_pose_to_world_world_pose_latest",
        ):
            self.artifacts.pop(key, None)
        if reestimate.get("invalidate_frozen_placement_pool") is True:
            self.artifacts.pop("anyplace_placement_candidates_latest", None)
        self.record("grasp_reestimate_ready", dict(reestimate))
        self._save_working_memory()
        return True

    def add_action(self, action: EnvAction) -> None:
        environment_task_updated = self._capture_active_environment_task(action)
        self._capture_reference_localization_state(action)
        self._capture_sam3_selection_state(action)
        self._capture_active_vision_state(action)
        target_mask_invalidated = self._invalidate_failed_anygrasp_target_mask(action)
        self._capture_anygrasp_candidate_policy(action)
        placement_candidates_updated = self._capture_placement_candidates(action)
        compilation_updated = self._capture_compiled_grasp(action)
        if not compilation_updated:
            # A physical-width filter may remove the qualifier's first queue
            # entry before activation.  Compile events are already retained by
            # candidate id, so activate the first locally valid queue entry
            # without asking the planner to repeat a deterministic transition.
            compilation_updated = self._activate_host_compiled_grasp_for_active_candidate()
        captured_artifacts = _extract_action_artifacts(action)
        for artifact in captured_artifacts:
            key = _artifact_memory_key(artifact, fallback_index=len(self.artifacts))
            self.artifacts[key] = {
                "value": artifact,
                "source": "tool_result",
                "timestamp_s": time.time(),
            }
        world_mutated = self._record_successful_world_mutation(action)
        gripper_state_updated = self._capture_gripper_command_state(action)
        target_pose_sync_updated = self._capture_planning_scene_target_pose_sync(action)
        placement_release_denied = self._capture_placement_release_denial(action)
        irreversible_release_failure = (
            False
            if placement_release_denied
            else self._capture_irreversible_placement_release_failure(action)
        )
        placement_release_updated = (
            False if irreversible_release_failure else self._capture_placement_release(action)
        )
        articulated_assessment_updated = self._capture_articulated_attachment_assessment(action)
        articulated_probe_prepared = self._capture_articulated_attachment_probe(action)
        articulated_probe_updated = self._capture_articulated_attachment_probe_result(action)
        execution_updated = self._advance_grasp_execution(action)
        reconciliation_updated = self._capture_motion_reconciliation(action)
        native_grasp_infrastructure_blocked = self._capture_native_grasp_infrastructure_failure(
            action
        )
        planning_scene_stopped = (
            False
            if native_grasp_infrastructure_blocked
            else self._capture_planning_scene_control_failure(action)
        )
        placement_candidate_advanced = self._advance_placement_candidate_after_motion(action)
        recovery_updated = self._advance_grasp_recovery(action)
        candidate_advanced = (
            False
            if planning_scene_stopped or native_grasp_infrastructure_blocked
            else self._advance_anygrasp_candidate_after_rejection(action)
        )
        qualification_infrastructure_blocked = (
            False
            if candidate_advanced
            else self._capture_terminal_qualification_infrastructure_failure(action)
        )
        frozen_frontier_binding_blocked = (
            False
            if candidate_advanced or qualification_infrastructure_blocked
            else self._capture_frozen_frontier_binding_failure(action)
        )
        terminal_compile_blocked = (
            False
            if candidate_advanced
            or qualification_infrastructure_blocked
            or frozen_frontier_binding_blocked
            else self._capture_terminal_grasp_compile_failure(action)
        )
        candidate_accepted = (
            False
            if candidate_advanced
            or planning_scene_stopped
            or native_grasp_infrastructure_blocked
            or qualification_infrastructure_blocked
            or frozen_frontier_binding_blocked
            or terminal_compile_blocked
            else self._accept_anygrasp_candidate_after_motion(action)
        )
        if candidate_advanced:
            self.facts.pop(ARTICULATED_ATTACHMENT_PROBE_KEY, None)
            self.facts.pop(GRASP_EXECUTION_KEY, None)
            self.facts.pop(ATTACHMENT_GATE_KEY, None)
            invalidated = [
                key
                for key in (
                    "anyplace_placement_candidates_latest",
                    "camera_pose_to_world_world_pose_latest",
                )
                if self.artifacts.pop(key, None) is not None
            ]
            if invalidated:
                self.record(
                    "placement_plan_invalidated",
                    {
                        "reason": "grasp_candidate_changed",
                        "artifact_keys": invalidated,
                    },
                )
            self._activate_host_compiled_grasp_for_active_candidate()
        self._append_transition_ledger(action)
        if (
            captured_artifacts
            or world_mutated
            or placement_release_denied
            or irreversible_release_failure
            or placement_release_updated
            or placement_candidates_updated
            or compilation_updated
            or articulated_probe_prepared
            or articulated_probe_updated
            or execution_updated
            or articulated_assessment_updated
            or reconciliation_updated
            or placement_candidate_advanced
            or recovery_updated
            or gripper_state_updated
            or target_pose_sync_updated
            or candidate_advanced
            or candidate_accepted
            or native_grasp_infrastructure_blocked
            or qualification_infrastructure_blocked
            or frozen_frontier_binding_blocked
            or terminal_compile_blocked
            or target_mask_invalidated
            or environment_task_updated
        ):
            self._save_working_memory()
        self.record(
            "action",
            {
                "action_type": action.action_type,
                "command": action.command,
                "has_code": action.code is not None,
                "metadata": action.metadata,
                "captured_artifact_count": len(captured_artifacts),
            },
        )
        for conversation_item in self.conversation.add_action(action):
            self._append_conversation_record(item_record(conversation_item))

    def _capture_native_grasp_infrastructure_failure(self, action: EnvAction) -> bool:
        """Stop after the simulator exhausted its bounded post-attach retry."""

        call = _tool_call(action, "gripper_control")
        if not isinstance(call, dict) or _call_result_success(call):
            return False
        receipt = _environment_receipt(call)
        if (
            receipt.get("infrastructure_error") is not True
            or receipt.get("attach_acked_before_rollback") is not True
        ):
            return False
        execution = self.grasp_execution()
        if not (
            isinstance(execution, dict)
            and execution.get("status") == "required"
            and execution.get("stage") == "close"
        ):
            return False
        error_code = str(
            receipt.get("error_code") or "NATIVE_GRASP_POST_ATTACH_INFRASTRUCTURE_FAILED"
        )
        policy = self.grasp_candidate_policy()
        policy = dict(policy) if isinstance(policy, dict) else {}
        policy.update(
            {
                "status": "blocked",
                "blocked_tool": "gazebo_native_grasp",
                "terminal_failure": error_code,
                "failure_code": "GRASP_RUNTIME_INFRASTRUCTURE_FAILED",
                "model_inference_retry_allowed": False,
                "blocked_at_s": time.time(),
            }
        )
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source="native_grasp_infrastructure_failure",
        )
        execution.update(
            {
                "status": "stopped_requires_human",
                "stage": "native_grasp_infrastructure_failure",
                "required_action": None,
                "control_failure": {
                    "error_code": error_code,
                    "attach_acked_before_rollback": True,
                    "native_state_snapshot": receipt.get("native_state_snapshot"),
                    "planning_scene_rollback": receipt.get("planning_scene_rollback"),
                },
                "stopped_at_s": time.time(),
            }
        )
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="native_grasp_infrastructure_failure",
        )
        self.facts.pop(GRASP_RECOVERY_KEY, None)
        self.record(
            "native_grasp_infrastructure_terminal_failure",
            {
                "error_code": error_code,
                "candidate_id": execution.get("candidate_id"),
                "model_inference_retry_allowed": False,
                "native_state_snapshot": receipt.get("native_state_snapshot"),
            },
        )
        return True

    def _capture_terminal_qualification_infrastructure_failure(self, action: EnvAction) -> bool:
        """Block model reinference after qualification exhausted its retry."""

        call = _tool_call(action, "grasp_pose_estimate")
        if not isinstance(call, dict) or _call_result_success(call):
            return False
        outputs = _tool_call_outputs(call)
        reason = str(outputs.get("reason") or "")
        if not (
            reason == "qualification_infrastructure_error"
            or outputs.get("infrastructure_error") is True
        ):
            return False
        infrastructure_reason = str(
            outputs.get("qualification_infrastructure_reason")
            or outputs.get("qualification_stop_reason")
            or reason
        )
        policy = self.grasp_candidate_policy()
        policy = dict(policy) if isinstance(policy, dict) else {}
        policy.update(
            {
                "status": "blocked",
                "blocked_tool": "moveit_candidate_qualification",
                "terminal_failure": infrastructure_reason,
                "failure_code": "QUALIFICATION_INFRASTRUCTURE_FAILED",
                "model_inference_retry_allowed": False,
                "blocked_at_s": time.time(),
            }
        )
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source="terminal_qualification_infrastructure_failure",
        )
        self.record(
            "qualification_infrastructure_terminal_failure",
            {
                "reason": infrastructure_reason,
                "blocked_tool": "moveit_candidate_qualification",
                "model_inference_retry_allowed": False,
            },
        )
        return True

    def _capture_frozen_frontier_binding_failure(self, action: EnvAction) -> bool:
        """Invalidate an unbindable frontier and request a fresh agentic cycle.

        A failed physical close can move the detached object and advance the
        PlanningScene revision.  The host may resume the frozen model output
        only when the controller supplies a complete rigid-motion proof.  If
        that proof is missing or invalid, repeating the same frozen-frontier
        call cannot change the result.  The stale pool is discarded, while a
        fresh observation and new inference remain an agent decision.
        """

        call = _tool_call(action, "grasp_pose_estimate")
        if not isinstance(call, dict) or _call_result_success(call):
            return False
        outputs = _tool_call_outputs(call)
        reason = str(outputs.get("reason") or "")
        terminal_reasons = {
            "frozen_grasp_frontier_scene_revision_changed",
            "frozen_grasp_frontier_rebase_proof_missing",
            "frozen_grasp_frontier_rebase_invalid",
        }
        if reason not in terminal_reasons:
            return False

        policy = self.grasp_candidate_policy()
        if not isinstance(policy, dict) or policy.get("status") not in {
            "frozen_frontier_required",
            "active",
            "exhausted",
        }:
            return False
        policy = dict(policy)
        failure = {
            "reason": reason,
            "source_planning_scene_revision": outputs.get("source_planning_scene_revision"),
            "planning_scene_revision": outputs.get("planning_scene_revision"),
            "rebase_reason": outputs.get("rebase_reason"),
            "model_inference_invoked": False,
            "execution_started": False,
        }
        target = policy.get("target_detection")
        target = target if isinstance(target, dict) else {}
        reestimate = {
            "schema_version": "openeta.grasp_reestimate.v1",
            "status": "pending_observation",
            "reason": "frozen_grasp_frontier_rebase_unproven",
            "attempt_count": 1,
            "scene_epoch": self.scene_epoch(),
            "target_prompt": target.get("target_prompt"),
            "source_image": target.get("source_image"),
            "source_observation_rgb_paths": (
                self._latest_current_observation_rgb_paths()
            ),
            "invalidate_frozen_placement_pool": True,
            "created_at_s": time.time(),
        }
        policy.update(
            {
                "status": "exhausted",
                "stop_reason": "frozen_grasp_frontier_invalidated",
                "failure_code": "FROZEN_GRASP_FRONTIER_REBASE_UNPROVEN",
                "terminal_failure": failure,
                "model_inference_invoked": False,
                "model_inference_retry_allowed": True,
                "reestimate_required": dict(reestimate),
                "invalidated_at_s": time.time(),
            }
        )
        policy.pop("frozen_grasp_frontier_rebase_pending", None)
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source="frozen_grasp_frontier_binding_failure",
        )
        self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
            reestimate,
            source="frozen_grasp_frontier_binding_failure",
        )
        self.facts.pop(FROZEN_PLACEMENT_POOL_KEY, None)
        self.facts.pop(GRASP_EXECUTION_KEY, None)
        self.facts.pop(ATTACHMENT_GATE_KEY, None)
        self.facts.pop(GRASP_RECOVERY_KEY, None)
        self.record(
            "frozen_grasp_frontier_binding_failure",
            {**failure, "model_inference_retry_allowed": True},
        )
        return True

    def _capture_terminal_grasp_compile_failure(self, action: EnvAction) -> bool:
        """Stop deterministic retries after a non-candidate compile failure."""

        call = _tool_call(action, "compile_grasp_seed")
        blocked_tool = "compile_grasp_seed"
        candidate_id = ""
        if not isinstance(call, dict) or _call_result_success(call):
            call = _tool_call(action, "grasp_pose_estimate")
            if not isinstance(call, dict) or _call_result_success(call):
                return False
            outputs = _tool_call_outputs(call)
            if outputs.get("reason") != "host_candidate_compilation_failed":
                return False
            blocked_tool = "host_candidate_compiler"
            candidate_id = str(outputs.get("candidate_id") or "")
        policy = self.grasp_candidate_policy()
        policy = dict(policy) if isinstance(policy, dict) else {}
        terminal_failure = _call_failure_reason(call)
        outputs = _tool_call_outputs(call)
        diagnostics = outputs.get("compilation_diagnostics")
        if isinstance(diagnostics, list):
            diagnostic_message = next(
                (
                    str(item.get("message") or "").strip()
                    for item in diagnostics
                    if isinstance(item, dict) and str(item.get("message") or "").strip()
                ),
                "",
            )
            terminal_failure = diagnostic_message or terminal_failure
        policy.update(
            {
                "status": "blocked",
                "blocked_tool": blocked_tool,
                "terminal_failure": terminal_failure,
                "failure_code": "HOST_GRASP_PROOF_COMPILATION_FAILED",
                "model_inference_retry_allowed": False,
                "blocked_at_s": time.time(),
            }
        )
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source="terminal_grasp_compile_failure",
        )
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        request = request if isinstance(request, dict) else {}
        parameters = request.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        candidate_id = candidate_id or _parameters_grasp_candidate_id(parameters)
        self.record(
            "grasp_compile_terminal_failure",
            {
                "candidate_id": candidate_id,
                "reason": policy["terminal_failure"],
                "blocked_tool": blocked_tool,
                "model_inference_retry_allowed": False,
            },
        )
        return True

    def _invalidate_failed_anygrasp_target_mask(self, action: EnvAction) -> bool:
        call = _tool_call(action, "grasp_pose_estimate") or _tool_call(action, "anygrasp")
        if not isinstance(call, dict) or _call_result_success(call):
            return False
        outputs = _tool_call_outputs(call)
        reason = str(outputs.get("reason") or "")
        if reason in {
            "all_grasps_colliding",
            "insufficient_object_points",
            "empty_point_cloud",
        }:
            reason = "no_grasp_candidates"
        if reason == "all_backends_failed":
            attempts = outputs.get("backend_attempts")
            failed_reasons = (
                {
                    str(attempt.get("reason") or "")
                    for attempt in attempts
                    if isinstance(attempt, dict) and attempt.get("status") == "failed"
                }
                if isinstance(attempts, list)
                else set()
            )
            if failed_reasons == {"no_grasp_candidates"}:
                reason = "no_grasp_candidates"
            elif failed_reasons and failed_reasons.issubset(
                {
                    "mcp_call_failed",
                    "model_inference_failed",
                    "model_load_failed",
                    "unknown_error",
                }
            ):
                reason = "model_inference_failed"
        if reason not in {
            "empty_target_mask",
            "model_inference_failed",
            "no_grasp_candidates",
        }:
            return False
        selected = self.selected_sam3_detection()
        if not isinstance(selected, dict):
            return False
        parameters = call.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        if reason == "model_inference_failed":
            unified = str(call.get("name") or "") == "grasp_pose_estimate"
            failure_key = (
                "grasp_estimator_backend_failure" if unified else "anygrasp_backend_failure"
            )
            previous = selected.get(failure_key)
            previous_attempts = (
                int(previous.get("attempt_count") or 0) if isinstance(previous, dict) else 0
            )
            metadata = outputs.get("metadata")
            error_type = str(metadata.get("error_type") or "") if isinstance(metadata, dict) else ""
            attempt_count = previous_attempts + 1
            # A unified grasp call already walks every configured compatible
            # backend.  Its explicit non-retryable result must not trigger a
            # second inference on the identical frozen RGB-D packet.  Keep the
            # legacy direct-AnyGrasp circuit bounded at two attempts.
            retryable = outputs.get("retryable") is True if unified else True
            max_attempts = 2 if retryable else 1
            exhausted = attempt_count >= max_attempts
            selected[failure_key] = {
                "reason": reason,
                "error_type": error_type or None,
                "attempt_count": attempt_count,
                "max_attempts": max_attempts,
                "status": "exhausted" if exhausted else "retry_required",
            }
            self.facts[SELECTED_SAM3_DETECTION_KEY] = _memory_fact_entry(
                selected,
                source=failure_key,
            )
            self.record(
                "grasp_estimator_backend_retry_exhausted"
                if exhausted
                else "grasp_estimator_backend_retry_required",
                {
                    "result_id": selected.get("result_id"),
                    "detection_id": selected.get("id"),
                    "reason": reason,
                    "error_type": error_type or None,
                    "attempt_count": attempt_count,
                },
            )
            return True
        hints = parameters.get("hints")
        dense_sampling = hints.get("dense_sampling") if isinstance(hints, dict) else None
        if (
            reason == "no_grasp_candidates"
            and parameters.get("dense_grasp") is not True
            and dense_sampling is not True
        ):
            selected["dense_grasp_retry_required"] = True
            self.facts[SELECTED_SAM3_DETECTION_KEY] = _memory_fact_entry(
                selected,
                source="anygrasp_dense_retry",
            )
            self.record(
                "anygrasp_dense_retry_required",
                {
                    "result_id": selected.get("result_id"),
                    "detection_id": selected.get("id"),
                },
            )
            return True
        source_image = outputs.get("source_rgb") or selected.get("source_image")
        self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)
        self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
        self.facts[SAM3_NO_DETECTION_KEY] = _memory_fact_entry(
            {
                "result_id": selected.get("result_id"),
                "source_image": source_image,
                "target_prompt": selected.get("target_prompt"),
                "reason": reason,
                "segmentation_mode": selected.get("segmentation_mode"),
                "bbox_xyxy": selected.get("bbox_xyxy"),
            },
            source="anygrasp",
        )
        self.record(
            "sam3_detection_invalidated",
            {
                "result_id": selected.get("result_id"),
                "detection_id": selected.get("id"),
                "reason": reason,
            },
        )
        return True

    def add_external_event(self, event: JsonDict) -> None:
        event_type = str(event.get("type", "external"))
        self.record(event_type, event)
        if event_type == "human_answer":
            answer = event.get("answer")
            if isinstance(answer, str) and answer.strip():
                self.begin_user_turn(answer, source="human_answer")

    def save_fact(self, key: str, value: JsonDict, *, source: str = "") -> None:
        if key == LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY:
            key = GRASP_CANDIDATE_POLICY_KEY
            self.facts.pop(LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY, None)
        self.facts[key] = {"value": dict(value), "source": source, "timestamp_s": time.time()}
        self.record("memory_fact_saved", {"key": key, "source": source})
        self._save_working_memory()

    def save_artifact(self, key: str, value: JsonDict, *, source: str = "") -> None:
        self.artifacts[key] = {"value": dict(value), "source": source, "timestamp_s": time.time()}
        self.record("memory_artifact_saved", {"key": key, "source": source})
        self._save_working_memory()

    def save_skill_note(self, skill_name: str, note: JsonDict, *, source: str = "") -> None:
        entry = {"note": dict(note), "source": source, "timestamp_s": time.time()}
        self.skill_notes.setdefault(skill_name, []).append(entry)
        self.record("memory_skill_note_saved", {"skill": skill_name, "source": source})
        self._save_working_memory()

    def get_memory(
        self,
        key: str | None = None,
        *,
        namespace: str = "all",
    ) -> JsonDict:
        if namespace == "facts":
            return {"facts": _select_memory(self.facts, key)}
        if namespace == "artifacts":
            return {"artifacts": _select_memory(self.artifacts, key)}
        if namespace == "skill_notes":
            if key is None:
                return {"skill_notes": self.skill_notes}
            return {"skill_notes": {key: self.skill_notes.get(key, [])}}
        return {
            "facts": _select_memory(self.facts, key),
            "artifacts": _select_memory(self.artifacts, key),
            "skill_notes": (
                self.skill_notes if key is None else {key: self.skill_notes.get(key, [])}
            ),
            "compact_summary": self.compact_summary,
        }

    def pending_sam3_selection(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PENDING_SAM3_SELECTION_KEY))

    def selected_sam3_detection(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(SELECTED_SAM3_DETECTION_KEY))

    def pending_reference_localization(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PENDING_REFERENCE_LOCALIZATION_KEY))

    def target_asset_reference(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(TARGET_ASSET_REFERENCE_KEY))

    def reference_localization_failure(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(REFERENCE_LOCALIZATION_FAILURE_KEY))

    def sam3_no_detection(self, semantic_role: str = "") -> JsonDict | None:
        role = semantic_role.strip().lower()
        if role in SAM3_SEMANTIC_ROLES:
            role_state = self.sam3_role_state(role)
            value = role_state.get("no_detection") if isinstance(role_state, dict) else None
            return dict(value) if isinstance(value, dict) else None
        return _memory_fact_value(self.facts.get(SAM3_NO_DETECTION_KEY))

    def sam3_semantic_state(self) -> JsonDict:
        value = _memory_fact_value(self.facts.get(SAM3_SEMANTIC_STATE_KEY))
        if not isinstance(value, dict):
            return {
                "schema_version": "openeta.sam3_semantic_state.v1",
                "roles": {},
                "attempts": [],
            }
        return dict(value)

    def sam3_role_state(self, semantic_role: str) -> JsonDict | None:
        role = semantic_role.strip().lower()
        if role not in SAM3_SEMANTIC_ROLES:
            return None
        roles = self.sam3_semantic_state().get("roles")
        value = roles.get(role) if isinstance(roles, dict) else None
        return dict(value) if isinstance(value, dict) else None

    def grasp_candidate_policy(self) -> JsonDict | None:
        return _memory_fact_value(
            self.facts.get(GRASP_CANDIDATE_POLICY_KEY)
            or self.facts.get(LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY)
        )

    def grasp_reestimation(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRASP_REESTIMATION_KEY))

    def anygrasp_candidate_policy(self) -> JsonDict | None:
        """Backward-compatible accessor for pre-facade callers."""

        return self.grasp_candidate_policy()

    def retained_targeted_grasp(self) -> JsonDict | None:
        """Return the active targeted grasp and its exact aligned source packet."""

        policy = self.grasp_candidate_policy() or {}
        final_source = policy.get("final_fallback_source")
        final_candidate = policy.get("active_candidate")
        if (
            policy.get("final_refinable_fallback") is True
            and isinstance(final_source, dict)
            and isinstance(final_candidate, dict)
        ):
            source = final_source.get("source")
            source = dict(source) if isinstance(source, dict) else {}
            for source_key, policy_key in (
                ("rgb", "source_rgb"),
                ("depth", "source_depth"),
            ):
                if not isinstance(source.get(source_key), str) or not source[source_key]:
                    value = final_source.get(policy_key)
                    if isinstance(value, str) and value:
                        source[source_key] = value
            frame_id = final_source.get("camera_frame_id")
            if isinstance(frame_id, str) and frame_id:
                source["camera_frame_id"] = frame_id
            return {
                "artifact_key": "grasp_estimation_recovery.final_candidate",
                "result_id": policy.get("result_id"),
                "status": policy.get("status", "retained"),
                "candidate": dict(final_candidate),
                "source": source,
            }

        artifact_key = next(
            (
                key
                for key in (
                    "grasp_pose_estimate_grasp_candidates_latest",
                    "anygrasp_grasp_candidates_latest",
                    "graspgenx_grasp_candidates_latest",
                    "contact_graspnet_grasp_candidates_latest",
                )
                if key in self.artifacts
            ),
            "",
        )
        entry = self.artifacts.get(artifact_key)
        if not isinstance(entry, dict):
            return None
        artifact = entry.get("value")
        if not isinstance(artifact, dict):
            return None
        source_value = artifact.get("selected_grasp_source")
        source = dict(source_value) if isinstance(source_value, dict) else {}
        if str(source.get("mode") or "") != "targeted":
            return None

        for source_key, artifact_key in (
            ("rgb", "source_rgb"),
            ("depth", "source_depth"),
            ("object_mask", "target_mask"),
        ):
            if not isinstance(source.get(source_key), str) or not source[source_key]:
                value = artifact.get(artifact_key)
                if isinstance(value, str) and value:
                    source[source_key] = value

        candidate = policy.get("active_candidate")
        if not isinstance(candidate, dict):
            candidate = artifact.get("best_grasp_candidate")
        if not isinstance(candidate, dict):
            return None

        return {
            "artifact_key": artifact_key,
            "result_id": policy.get("result_id"),
            "status": policy.get("status", "retained"),
            "candidate": dict(candidate),
            "source": source,
        }

    def activate_final_grasp_candidate(self, *, recovery_id: str) -> JsonDict:
        recovery = self.grasp_estimation_recovery()
        policy = self.grasp_candidate_policy()
        if (
            not isinstance(recovery, dict)
            or recovery.get("status") != "required"
            or str(recovery.get("recovery_id") or "") != recovery_id
        ):
            raise ValueError("final grasp fallback requires the active recovery_id.")
        if str(recovery.get("trigger_class") or "") not in {
            "perception_refinable",
            "uncertain_review",
        }:
            raise ValueError("final grasp fallback is not allowed for a hard rejection.")
        if not isinstance(policy, dict) or policy.get("status") != "exhausted":
            raise ValueError("final grasp fallback requires an exhausted candidate policy.")
        entries = [
            dict(item)
            for item in recovery.get("fallback_candidates", [])
            if isinstance(item, dict) and isinstance(item.get("candidate"), dict)
        ]
        if not entries:
            raise ValueError("final grasp fallback has no refinable candidate.")
        selected = min(
            entries,
            key=lambda item: _grasp_candidate_sort_key(dict(item["candidate"])),
        )
        candidate = dict(selected["candidate"])
        original_id = str(candidate.get("id") or "grasp")
        candidate["original_candidate_id"] = original_id
        candidate["id"] = f"{original_id}-final-{recovery_id[-8:]}"
        candidate["final_refinable_fallback"] = True
        capabilities = _fallback_candidate_capabilities(
            selected,
            default=self._active_grasp_calibration_capabilities(),
        )
        max_gripper_width = float(capabilities["max_gripper_width_m"])
        try:
            estimated_width = float(candidate.get("width"))
        except (TypeError, ValueError):
            estimated_width = max_gripper_width
        if not math.isfinite(estimated_width):
            estimated_width = max_gripper_width
        candidate["estimated_width_m"] = estimated_width
        candidate["width"] = min(max(0.0, estimated_width), max_gripper_width)
        candidate["max_gripper_width_m"] = max_gripper_width
        candidate["grasp_calibration_id"] = capabilities["calibration_id"]
        candidate["width_clamped_to_physical_limit"] = estimated_width > max_gripper_width
        policy.update(
            {
                "result_id": selected.get("source_result_id") or policy.get("result_id"),
                "source_tool": selected.get("source_tool") or policy.get("source_tool"),
                "source_backend": (selected.get("source_backend") or policy.get("source_backend")),
                "status": "active",
                "candidate_count": 1,
                "active_rank": 0,
                "active_candidate": candidate,
                "remaining_candidate_ids": [],
                "candidates": [candidate],
                "source_rgb": selected.get("source_rgb"),
                "source_depth": selected.get("source_depth"),
                "camera_frame_id": selected.get("camera_frame_id"),
                "target_detection": selected.get("target_detection"),
                "final_refinable_fallback": True,
                "final_fallback_source": selected,
                "final_fallback_recovery_id": recovery_id,
                "physical_width_limit_m": max_gripper_width,
                "grasp_calibration_id": capabilities["calibration_id"],
                "grasp_calibration_profile_path": capabilities["profile_path"],
                "activated_at_s": time.time(),
            }
        )
        for key in ("fallback_required", "exhaustion_reason", "refinement_seed_candidate"):
            policy.pop(key, None)
        recovery.update(
            {
                "status": "final_candidate_activated",
                "final_candidate_id": candidate["id"],
                "final_source_result_id": selected.get("source_result_id"),
                "completed_at_s": time.time(),
            }
        )
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source="final_refinable_grasp_fallback",
        )
        self.facts[GRASP_ESTIMATION_RECOVERY_KEY] = _memory_fact_entry(
            recovery,
            source="final_refinable_grasp_fallback",
        )
        self.facts.pop(GRASP_EXECUTION_KEY, None)
        self.facts.pop(ARTICULATED_ATTACHMENT_PROBE_KEY, None)
        self.facts.pop(ATTACHMENT_GATE_KEY, None)
        self.record(
            "final_refinable_grasp_candidate_activated",
            {
                "recovery_id": recovery_id,
                "candidate_id": candidate["id"],
                "original_candidate_id": original_id,
                "score": candidate.get("score"),
                "source_backend": policy.get("source_backend"),
                "width_clamped": candidate["width_clamped_to_physical_limit"],
            },
        )
        self._save_working_memory()
        return {
            "recovery_id": recovery_id,
            "candidate": candidate,
            "source_backend": policy.get("source_backend"),
            "source_result_id": policy.get("result_id"),
            "max_gripper_width_m": max_gripper_width,
            "grasp_calibration_id": capabilities["calibration_id"],
        }

    def _active_grasp_calibration_capabilities(self) -> JsonDict:
        workspace = self.metadata.get("workspace")
        workspace = workspace if isinstance(workspace, dict) else {}
        profile_path = (
            workspace.get("grasp_profile_path")
            or self.metadata.get("grasp_profile_path")
            or self.metadata.get("calibration_profile_path")
            or DEFAULT_GRASP_CALIBRATION_PROFILE
        )
        return load_grasp_calibration_capabilities(str(profile_path))

    def articulated_attachment_probe(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(ARTICULATED_ATTACHMENT_PROBE_KEY))

    def grasp_execution(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRASP_EXECUTION_KEY))

    def grasp_recovery(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRASP_RECOVERY_KEY))

    def grasp_estimation_recovery(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRASP_ESTIMATION_RECOVERY_KEY))

    def gripper_command_state(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRIPPER_COMMAND_STATE_KEY))

    def attachment_gate(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(ATTACHMENT_GATE_KEY))

    def placement_release(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PLACEMENT_RELEASE_KEY))

    def placement_candidate_policy(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PLACEMENT_CANDIDATE_POLICY_KEY))

    def placement_object_detection(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PLACEMENT_OBJECT_DETECTION_KEY))

    def placement_region_detection(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PLACEMENT_REGION_DETECTION_KEY))

    def frozen_placement_goal_pool(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(FROZEN_PLACEMENT_POOL_KEY))

    def _save_placement_detection(
        self,
        key: str,
        detection: JsonDict,
        *,
        source: str,
    ) -> None:
        """Retain only object/region masks from one fresh perception bundle."""

        other_key = (
            PLACEMENT_REGION_DETECTION_KEY
            if key == PLACEMENT_OBJECT_DETECTION_KEY
            else PLACEMENT_OBJECT_DETECTION_KEY
        )
        other = _memory_fact_value(self.facts.get(other_key))
        source_image = detection.get("source_image")
        bundle_id = str(detection.get("perception_bundle_id") or "")
        other_bundle_id = (
            str(other.get("perception_bundle_id") or "") if isinstance(other, dict) else ""
        )
        same_source_image = (
            isinstance(source_image, str)
            and bool(source_image)
            and source_image == other.get("source_image")
            if isinstance(other, dict)
            else False
        )
        if isinstance(other, dict) and not same_source_image and (
            (bundle_id and other_bundle_id and bundle_id != other_bundle_id)
            or (
                not bundle_id
                and not other_bundle_id
                and isinstance(source_image, str)
                and source_image
                and other.get("source_image") != source_image
            )
        ):
            self.facts.pop(other_key, None)
            self.record(
                "stale_placement_detection_cleared",
                {
                    "cleared_key": other_key,
                    "replacement_key": key,
                    "replacement_source_image": source_image,
                    "replacement_perception_bundle_id": bundle_id or None,
                },
            )
        self.facts[key] = _memory_fact_entry(detection, source=source)

    def _record_sam3_semantic_result(
        self,
        result: JsonDict,
        *,
        status: str,
    ) -> None:
        role = str(result.get("semantic_role") or "").strip().lower()
        if role not in SAM3_SEMANTIC_ROLES:
            return
        state = self.sam3_semantic_state()
        roles = state.get("roles")
        roles = dict(roles) if isinstance(roles, dict) else {}
        role_state = roles.get(role)
        role_state = dict(role_state) if isinstance(role_state, dict) else {}
        target_prompt = str(result.get("target_prompt") or "").strip()
        result_scene_epoch = _optional_int(result.get("scene_epoch"), default=self.scene_epoch())
        previous_scene_epoch = _optional_int(role_state.get("scene_epoch"), default=-1)
        if (
            target_prompt
            and target_prompt.lower() != "point_prompt"
            and (
                not str(role_state.get("canonical_prompt") or "").strip()
                or previous_scene_epoch != result_scene_epoch
            )
        ):
            role_state["canonical_prompt"] = target_prompt
        role_state.update(
            {
                "semantic_role": role,
                "status": status,
                "scene_epoch": result_scene_epoch,
                "perception_bundle_id": result.get("perception_bundle_id"),
                "observation_id": result.get("observation_id"),
                "last_result_id": result.get("result_id"),
                "last_attempt_id": result.get("attempt_id"),
                "updated_at_s": time.time(),
            }
        )
        if status == "selected":
            role_state["selected_detection"] = dict(result)
            role_state.pop("no_detection", None)
        elif status in {"no_detection", "rejected"}:
            role_state["no_detection"] = dict(result)
        roles[role] = role_state
        attempts = state.get("attempts")
        attempts = (
            [dict(item) for item in attempts if isinstance(item, dict)]
            if isinstance(attempts, list)
            else []
        )
        fingerprint = str(result.get("attempt_fingerprint") or "")
        attempt_id = str(result.get("attempt_id") or "")
        if fingerprint or attempt_id:
            attempts = [
                item
                for item in attempts
                if not (fingerprint and str(item.get("attempt_fingerprint") or "") == fingerprint)
                and not (attempt_id and str(item.get("attempt_id") or "") == attempt_id)
            ]
            attempts.append(
                {
                    "semantic_role": role,
                    "status": status,
                    "result_id": result.get("result_id"),
                    "source_image": result.get("source_image"),
                    "frame_id": result.get("frame_id") or result.get("source_frame_id"),
                    "camera_role": result.get("camera_role")
                    or result.get("source_camera_role"),
                    "view_identity": result.get("view_identity"),
                    "mode": result.get("segmentation_mode"),
                    "target_prompt": target_prompt,
                    "scene_epoch": role_state["scene_epoch"],
                    "perception_bundle_id": result.get("perception_bundle_id"),
                    "observation_id": result.get("observation_id"),
                    "attempt_id": attempt_id or None,
                    "attempt_fingerprint": fingerprint or None,
                    "completed_at_s": time.time(),
                }
            )
        state.update(
            {
                "schema_version": "openeta.sam3_semantic_state.v1",
                "roles": roles,
                "attempts": attempts[-24:],
                "updated_at_s": time.time(),
            }
        )
        self.facts[SAM3_SEMANTIC_STATE_KEY] = _memory_fact_entry(
            state,
            source="sam3_semantic_state",
        )

    def motion_reconciliation(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(MOTION_RECONCILIATION_KEY))

    def scene_epoch(self) -> int:
        value = _memory_fact_value(self.facts.get(SCENE_EPOCH_KEY)) or {}
        try:
            return max(0, int(value.get("epoch") or 0))
        except (TypeError, ValueError):
            return 0

    def transition_ledger(self) -> list[JsonDict]:
        value = _memory_fact_value(self.facts.get(TRANSITION_LEDGER_KEY)) or {}
        rows = value.get("rows")
        return (
            [dict(row) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        )

    def latest_environment_receipt(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get("latest_environment_receipt"))

    def planning_scene_target_pose_sync(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PLANNING_SCENE_TARGET_POSE_SYNC_KEY))

    def active_environment_task(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(ACTIVE_ENVIRONMENT_TASK_KEY))

    def _capture_host_owned_task_completion(self) -> bool:
        """Prove task completion before the launcher performs lifecycle cleanup."""

        active = self.active_environment_task()
        if not (
            isinstance(active, dict)
            and active.get("lifecycle_owner") == "host"
            and active.get("host_cleanup_pending") is True
        ):
            return False
        release = self.placement_release()
        visual = (
            release.get("post_release_visual_observation")
            if isinstance(release, dict)
            else None
        )
        if not (
            isinstance(release, dict)
            and release.get("status") == "released"
            and isinstance(visual, dict)
            and visual.get("available") is True
        ):
            return False
        multi_sort = _memory_fact_value(self.facts.get(MULTI_SORT_PROGRESS_KEY))
        completed = _memory_fact_value(
            self.facts.get(COMPLETED_PLACEMENT_SUBGOALS_KEY)
        )
        completed_items = (
            list(completed.get("items") or []) if isinstance(completed, dict) else []
        )
        if isinstance(multi_sort, dict):
            try:
                remaining_count = int(multi_sort.get("remaining_count", -1))
                assignment_count = int(multi_sort.get("assignment_count") or 0)
            except (TypeError, ValueError):
                return False
            if not (
                multi_sort.get("all_completed") is True
                and remaining_count == 0
                and assignment_count > 0
                and len(completed_items) >= assignment_count
            ):
                return False
        evidence: JsonDict = {
            "schema_version": "openeta.task_completion_evidence.v2",
            "status": "proven",
            "outcome": "success",
            "source": "vlm_post_release_observation",
            "environment_closed": False,
            "lifecycle_owner": "host",
            "host_cleanup_pending": True,
            "candidate_id": release.get("candidate_id"),
            "placement_pose_id": release.get("placement_pose_id"),
            "post_release_visual_observation": dict(visual),
            "completed_at_s": time.time(),
        }
        if isinstance(multi_sort, dict):
            evidence.update(
                {
                    "multi_sort_progress": dict(multi_sort),
                    "completed_placement_subgoals": completed_items,
                }
            )
        previous = self.task_completion_evidence()
        if (
            isinstance(previous, dict)
            and previous.get("status") == "proven"
            and previous.get("outcome") == "success"
            and previous.get("lifecycle_owner") == "host"
        ):
            return False
        self.facts[TASK_COMPLETION_EVIDENCE_KEY] = _memory_fact_entry(
            evidence,
            source="host_owned_environment_completion",
        )
        self.record("task_completion_proven", dict(evidence))
        return True

    def task_completion_evidence(self) -> JsonDict | None:
        """Return host-proven evidence that the embodied task finished successfully."""

        return _memory_fact_value(self.facts.get(TASK_COMPLETION_EVIDENCE_KEY))

    def _capture_active_environment_task(self, action: EnvAction) -> bool:
        """Refresh task identity from observations inside a host-bound environment."""

        previous = self.active_environment_task() or {}
        if previous.get("lifecycle_owner") == "host":
            # A launcher-owned Gazebo world is task-neutral. Its observation
            # text describes the scene and must not replace operator intent.
            return False
        call = _successful_tool_call(action, "observe")
        if call is None:
            return False
        extracted = _assigned_task_from_tool_call(call)
        if extracted is None:
            return False
        task, source_field = extracted
        if previous.get("task") == task:
            return False
        outputs = _tool_call_outputs(call)
        environment = outputs.get("environment")
        environment = environment if isinstance(environment, dict) else {}
        mcp = outputs.get("mcp")
        mcp = mcp if isinstance(mcp, dict) else {}
        value = {
            "task": task,
            "env_id": environment.get("env_id") or previous.get("env_id"),
            "handle": environment.get("handle") or mcp.get("handle") or previous.get("handle"),
            "session_id": (
                environment.get("session_id")
                or mcp.get("session_id")
                or previous.get("session_id")
            ),
            "source_tool": "observe",
            "source_field": source_field,
            "scene_epoch": self.scene_epoch(),
            "updated_at_s": time.time(),
            "lifecycle_owner": previous.get("lifecycle_owner"),
            "host_cleanup_pending": previous.get("host_cleanup_pending"),
        }
        value = {key: item for key, item in value.items() if item not in (None, "")}
        self.facts[ACTIVE_ENVIRONMENT_TASK_KEY] = _memory_fact_entry(
            value,
            source="simulator_tool_result",
        )
        self.record("active_environment_task_updated", value)
        return True

    def grasp_candidate_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        policy = self.grasp_candidate_policy()
        if policy is None:
            return None
        if tool_name not in {
            "camera_pose_to_world",
            "move_to",
            "follow_eef_trajectory",
            "obstacle_avoidance",
        }:
            return None
        recovery = self.grasp_recovery()
        recovery_action = recovery.get("required_action") if isinstance(recovery, dict) else None
        if (
            isinstance(recovery, dict)
            and recovery.get("status") == "required"
            and isinstance(recovery_action, dict)
            and tool_name == recovery_action.get("name")
            and parameters == recovery_action.get("parameters")
        ):
            return None
        status = str(policy.get("status") or "")
        if status == "accepted":
            return None
        source_tool = str(policy.get("source_tool") or "grasp_pose_estimate")
        source_backend = str(policy.get("source_backend") or source_tool)
        source_label = _grasp_backend_label(source_backend)
        if status == "selection_required":
            return (
                "The qualified grasp queue has no host compilation event. Planner "
                "tools cannot compile or execute raw candidates."
            )
        active = policy.get("active_candidate")
        if status == "exhausted" or not isinstance(active, dict):
            return (
                f"All {source_label} candidates were rejected. Observe and rerun "
                f"{source_label} "
                "instead of reusing an exhausted pose."
            )
        active_id = str(active.get("id") or "")
        supplied_id = _parameters_grasp_candidate_id(parameters)
        if tool_name == "camera_pose_to_world":
            if _is_anyplace_pose(parameters):
                return None
            if source_tool in {"anygrasp", "grasp_pose_estimate"}:
                return (
                    "Raw grasp candidates cannot use camera_pose_to_world; only the "
                    "host-owned candidate compiler applies GraspNet-to-EEF calibration."
                )
            if not supplied_id:
                return (
                    f"camera_pose_to_world must receive the active {source_label} "
                    f"candidate {active_id!r}, including its id."
                )
            if supplied_id != active_id:
                return (
                    f"Greedy {source_label} policy requires the current active "
                    f"candidate {active_id!r}; later candidates are available only "
                    "after a candidate-linked safety or motion rejection."
                )
            return None
        if (
            source_tool in {"anygrasp", "grasp_pose_estimate"}
            and tool_name in {"move_to", "follow_eef_trajectory"}
            and self.grasp_execution() is None
        ):
            return (
                "Raw grasp motion is blocked. A valid host compilation event and "
                "host-generated grasp_execution stages are required."
            )
        if supplied_id and supplied_id != active_id:
            return (
                f"Tool {tool_name!r} references {source_label} candidate {supplied_id!r}, "
                f"but the current active candidate is {active_id!r}."
            )
        return None

    def grasp_execution_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        reconciliation = self.motion_reconciliation()
        if isinstance(reconciliation, dict) and reconciliation.get("status") in {
            "required",
            "unresolved",
        }:
            if tool_name == "observe":
                return None
            return (
                "A simulator action has transport-unknown outcome. Observe the same "
                "environment handle before issuing another action. Do not resend a "
                "partial move because the original controller may still be running."
            )
        execution = self.grasp_execution()
        if not isinstance(execution, dict):
            return None
        recovery = self.grasp_recovery()
        recovery_action = recovery.get("required_action") if isinstance(recovery, dict) else None
        if (
            isinstance(recovery, dict)
            and recovery.get("status") == "required"
            and isinstance(recovery_action, dict)
            and tool_name == recovery_action.get("name")
            and parameters == recovery_action.get("parameters")
        ):
            return None
        if execution.get("status") == "stopped_requires_human":
            if tool_name == "observe":
                return None
            return (
                "The planning scene became unavailable before motion execution. "
                "Do not resend or switch grasp candidates; human recovery is required."
            )
        if execution.get("status") != "required":
            return None
        stage = str(execution.get("stage") or "")
        if tool_name == "observe":
            return None
        if stage == "prepare_probe":
            if tool_name == "prepare_attachment_probe":
                return None
            return (
                "The closed articulated handle requires prepare_attachment_probe. "
                "Propose one bounded linear direction or short arc from the current "
                "multi-view observation before any further world mutation."
            )
        if stage == "attachment":
            if execution.get("attachment_mode") == "articulated_handle":
                gate = self.attachment_gate() or {}
                verdict = str(gate.get("verdict") or "UNKNOWN").upper()
                assessment_count = int(gate.get("assessment_count") or 0)
                refresh_required = gate.get("refresh_required") is True
                refresh_completed = gate.get("unknown_refresh_completed") is True
                if verdict == "UNKNOWN":
                    if tool_name == "assess_attachment_probe" and (
                        assessment_count == 0 or (assessment_count == 1 and refresh_completed)
                    ):
                        return None
                    if (
                        tool_name == "observe"
                        and assessment_count == 1
                        and refresh_required
                        and not refresh_completed
                    ):
                        return None
            actions = execution.get("attachment_actions")
            allowed = [
                action
                for action in (actions.values() if isinstance(actions, dict) else [])
                if isinstance(action, dict)
            ]
            if any(
                tool_name == action.get("name") and parameters == action.get("parameters")
                for action in allowed
            ):
                return None
            return (
                "Attachment evidence is pending. Keep the gripper closed and use only "
                "the host-owned action for the current verdict. Articulated UNKNOWN "
                "permits one fresh observation; a portable object proceeds only after "
                "native bilateral contact and attach ACK, and structured FAIL permits "
                "the exact recovery open."
            )
        if stage == "probe":
            articulated_probe = self.articulated_attachment_probe()
            articulated_required = (
                articulated_probe.get("required_action")
                if isinstance(articulated_probe, dict)
                else None
            )
            if (
                isinstance(articulated_required, dict)
                and articulated_probe.get("status") == "required"
                and tool_name == articulated_required.get("name")
                and parameters == articulated_required.get("parameters")
            ):
                return None
            return (
                "The attachment probe requires the exact host-generated motion "
                "parameters while keeping the gripper closed."
            )
        required = execution.get("required_action")
        if not isinstance(required, dict):
            return "The host-generated grasp execution obligation is malformed."
        return grasp_reference_action_error(
            stage=stage,
            tool_name=tool_name,
            parameters=parameters,
            required_action=required,
        )

    def resolve_sam3_selection(
        self,
        *,
        result_id: str,
        detection_id: str,
        selection_source: str,
        confidence: float | None = None,
        reason: str = "",
        target_geometry_family: str = "",
    ) -> JsonDict:
        pending = self.pending_sam3_selection()
        if pending is None:
            raise ValueError("No SAM3 detection selection is pending.")
        expected_result_id = str(pending.get("result_id") or "")
        if not result_id or result_id != expected_result_id:
            raise ValueError("select_sam3_detection requires the exact pending sam3_result_id.")
        candidates = pending.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        selected = next(
            (
                dict(candidate)
                for candidate in candidates
                if isinstance(candidate, dict) and str(candidate.get("id") or "") == detection_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("detection_id does not belong to the pending SAM3 result.")
        geometry_family = target_geometry_family.strip().lower()
        if geometry_family and geometry_family not in GRASP_GEOMETRY_FAMILIES:
            raise ValueError(
                "target_geometry_family must be one of "
                + ", ".join(sorted(GRASP_GEOMETRY_FAMILIES))
                + "."
            )
        selected.update(
            {
                "result_id": result_id,
                "source_image": pending.get("source_image"),
                # Preserve the current observation identity alongside the
                # selected mask.  perception-bridge's strict RGB-D bridge requires this
                # exact frame association and must never infer it from a file
                # name or a unique camera count.
                "source_frame_id": pending.get("frame_id"),
                "target_prompt": pending.get("target_prompt"),
                "segmentation_mode": pending.get("segmentation_mode"),
                "selection_source": selection_source or "main_agent_vlm",
                "selection_confidence": confidence,
                "selection_reason": reason,
                "selected_at_s": time.time(),
                "scene_epoch": _optional_int(
                    pending.get("scene_epoch"),
                    default=self.scene_epoch(),
                ),
                "semantic_role": pending.get("semantic_role") or "grasp_target",
                "semantic_role_source": pending.get("semantic_role_source"),
                "semantic_target": pending.get("semantic_target") or pending.get("target_prompt"),
                "perception_bundle_id": pending.get("perception_bundle_id"),
                "observation_id": pending.get("observation_id"),
                "attempt_id": pending.get("attempt_id"),
                "attempt_fingerprint": pending.get("attempt_fingerprint"),
            }
        )
        camera_role = pending.get("camera_role")
        if isinstance(camera_role, str) and camera_role:
            selected["source_camera_role"] = camera_role
        if geometry_family and geometry_family != "unknown":
            selected["target_geometry_family"] = geometry_family
        semantic_role = str(selected.get("semantic_role") or "grasp_target").strip().lower()
        if semantic_role not in SAM3_SEMANTIC_ROLES:
            semantic_role = "grasp_target"
            selected["semantic_role"] = semantic_role
        self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
        if semantic_role == "grasp_target":
            self.facts[SELECTED_SAM3_DETECTION_KEY] = _memory_fact_entry(
                selected,
                source="select_sam3_detection",
            )
        reestimate = self.grasp_reestimation()
        if (
            isinstance(reestimate, dict)
            and reestimate.get("status") == "selection_pending"
            and str(reestimate.get("segmentation_result_id") or "") == result_id
        ):
            reestimate.update(
                {
                    "status": "target_ready",
                    "selected_detection_id": detection_id,
                    "selected_source_image": selected.get("source_image"),
                    "selected_at_s": selected["selected_at_s"],
                }
            )
            self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
                reestimate,
                source="grasp_reestimate_target_selected",
            )
        placement_key = (
            PLACEMENT_REGION_DETECTION_KEY
            if semantic_role == "placement_region"
            else PLACEMENT_OBJECT_DETECTION_KEY
        )
        self._save_placement_detection(
            placement_key,
            selected,
            source="select_sam3_detection",
        )
        self._record_sam3_semantic_result(selected, status="selected")
        self.facts.pop(TARGET_LOCALIZATION_BUDGET_KEY, None)
        self.record(
            "sam3_detection_selected",
            {
                "result_id": result_id,
                "detection_id": detection_id,
                "selection_source": selected["selection_source"],
                "selection_confidence": confidence,
                "semantic_role": semantic_role,
            },
        )
        self._save_working_memory()
        return selected

    def reject_sam3_detections(self, *, result_id: str, reason: str) -> JsonDict:
        pending = self.pending_sam3_selection()
        if pending is None:
            raise ValueError("No SAM3 detection selection is pending.")
        expected_result_id = str(pending.get("result_id") or "")
        if not result_id or result_id != expected_result_id:
            raise ValueError("reject_sam3_detections requires the pending sam3_result_id.")
        rejection_reason = reason.strip()
        if not rejection_reason:
            raise ValueError("reject_sam3_detections requires a visual reason.")
        candidates = pending.get("candidates")
        rejected_ids = [
            str(candidate.get("id") or "")
            for candidate in (candidates if isinstance(candidates, list) else [])
            if isinstance(candidate, dict) and str(candidate.get("id") or "")
        ]
        no_detection = {
            "result_id": result_id,
            "source_image": pending.get("source_image"),
            "target_prompt": pending.get("target_prompt"),
            "reason": "semantic_candidates_rejected",
            "segmentation_mode": pending.get("segmentation_mode"),
            "rejected_detection_ids": rejected_ids,
            "rejection_reason": rejection_reason,
            "semantic_role": pending.get("semantic_role") or "grasp_target",
            "semantic_role_source": pending.get("semantic_role_source"),
            "semantic_target": pending.get("semantic_target") or pending.get("target_prompt"),
            "perception_bundle_id": pending.get("perception_bundle_id"),
            "observation_id": pending.get("observation_id"),
            "scene_epoch": _optional_int(pending.get("scene_epoch"), default=self.scene_epoch()),
            "attempt_id": pending.get("attempt_id"),
            "attempt_fingerprint": pending.get("attempt_fingerprint"),
        }
        self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
        if str(no_detection["semantic_role"]) == "grasp_target":
            self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)
        self.facts[SAM3_NO_DETECTION_KEY] = _memory_fact_entry(
            no_detection,
            source="reject_sam3_detections",
        )
        self._record_sam3_semantic_result(no_detection, status="rejected")
        reestimate = self.grasp_reestimation()
        if (
            isinstance(reestimate, dict)
            and reestimate.get("status") == "selection_pending"
            and str(reestimate.get("segmentation_result_id") or "") == result_id
        ):
            reestimate.update(
                {
                    "selection_rejection_reason": rejection_reason,
                    "selection_rejected_at_s": time.time(),
                }
            )
            _record_failed_grasp_reestimate_view(
                reestimate,
                source_image=str(pending.get("source_image") or ""),
                failure="selection_rejected",
            )
            self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
                reestimate,
                source="grasp_reestimate_target_rejected",
            )
        self.record("sam3_detections_rejected", dict(no_detection))
        self._save_working_memory()
        return no_detection

    def detection_selection_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
        world_mutating: bool = False,
    ) -> str | None:
        if tool_name == "sam3":
            semantic_role = str(parameters.get("semantic_role") or "").strip().lower()
            if semantic_role and semantic_role not in SAM3_SEMANTIC_ROLES:
                return (
                    "SAM3 semantic_role must be grasp_target, placement_object, or "
                    "placement_region."
                )
            attempt_id = str(parameters.get("attempt_id") or "")
            fingerprint = str(parameters.get("attempt_fingerprint") or "")
            attempts = self.sam3_semantic_state().get("attempts")
            duplicate = next(
                (
                    attempt
                    for attempt in (attempts if isinstance(attempts, list) else [])
                    if isinstance(attempt, dict)
                    and (
                        (attempt_id and str(attempt.get("attempt_id") or "") == attempt_id)
                        or (
                            fingerprint
                            and str(attempt.get("attempt_fingerprint") or "") == fingerprint
                        )
                    )
                ),
                None,
            )
            if isinstance(duplicate, dict):
                return (
                    "This deterministic SAM3 role/image/mode attempt already completed "
                    f"with status {duplicate.get('status')!r}; use the bounded next "
                    "fallback or a genuinely fresh observation instead of retrying it."
                )
        no_detection = self.sam3_no_detection()
        reference_failure = _memory_fact_value(self.facts.get(REFERENCE_LOCALIZATION_FAILURE_KEY))
        if (
            tool_name == "retrieve_asset_reference"
            and isinstance(no_detection, dict)
            and isinstance(reference_failure, dict)
            and str(reference_failure.get("sam3_result_id") or "")
            == str(no_detection.get("result_id") or "")
        ):
            return (
                "Object-memory localization already failed for the current empty SAM3 "
                "result. Use molmopoint on sam3_no_detection.source_image, then feed its "
                "exact pixel point to SAM3; do not retry retrieve_asset_reference until "
                "a new SAM3 result or changed scene exists."
            )
        localization = self.pending_reference_localization()
        if localization is not None:
            required_parameter = str(localization.get("required_parameter") or "roi_bbox_xyxy")
            if tool_name != "sam3":
                return (
                    "Reference-guided target localization is pending. The next "
                    f"perception action must call SAM3 with {required_parameter}."
                )
            expected_image = str(localization.get("scene_image") or "")
            if str(parameters.get("image") or "") != expected_image:
                return (
                    "Reference-guided SAM3 must use the exact scene image from the "
                    "pending asset-reference result."
                )
            if required_parameter == "positive_points":
                expected_points = localization.get("positive_points")
                if parameters.get("positive_points") != expected_points:
                    return (
                        "Reference-guided SAM3 requires the exact positive_points "
                        "returned by the isolated reference localizer."
                    )
            else:
                bbox = parameters.get("roi_bbox_xyxy")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    return (
                        "Reference-guided SAM3 requires roi_bbox_xyxy in original-image "
                        "pixel coordinates."
                    )
        reestimate = self.grasp_reestimation()
        if (
            tool_name == "sam3"
            and isinstance(reestimate, dict)
            and reestimate.get("status") == "ready"
        ):
            attempted = {
                str(path)
                for path in reestimate.get("attempted_view_images", [])
                if isinstance(path, str) and path
            }
            allowed_images = {
                str(view.get("rgb_path") or "")
                for view in reestimate.get("observation_views", [])
                if isinstance(view, dict)
                and str(view.get("rgb_path") or "")
                and str(view.get("rgb_path") or "") not in attempted
            }
            supplied_image = str(parameters.get("image") or "")
            if supplied_image not in allowed_images:
                return (
                    "Grasp re-estimation SAM3 must use one exact, untried rgb_path "
                    "from grasp_view_selection_obligation.candidate_views."
                )
            expected_prompt = str(reestimate.get("target_prompt") or "").strip()
            if expected_prompt and str(parameters.get("prompt") or "").strip() != expected_prompt:
                return (
                    "Grasp re-estimation SAM3 must preserve the exact target_prompt "
                    "from grasp_view_selection_obligation."
                )
        pending = self.pending_sam3_selection()
        mode = str(parameters.get("mode") or "targeted").strip().lower()
        if pending is not None and tool_name == "anygrasp" and mode != "scene":
            return (
                "Targeted AnyGrasp is blocked until the pending SAM3 candidates are "
                "resolved with select_sam3_detection."
            )
        if pending is not None and tool_name == "graspgenx":
            return (
                "GraspGenX is blocked until the pending SAM3 candidates are resolved "
                "with select_sam3_detection."
            )
        if pending is not None and world_mutating:
            return (
                "World-mutating tools are blocked while a SAM3 detection selection "
                "obligation is pending."
            )
        if tool_name not in {"anygrasp", "graspgenx"}:
            return None
        if tool_name == "anygrasp" and mode == "scene":
            return None
        selected = self.selected_sam3_detection()
        if selected is None:
            return None
        expected_mask = str(selected.get("mask_ref") or "")
        if tool_name == "graspgenx":
            object_mask = parameters.get("object_mask")
            supplied_mask = (
                str(object_mask.get("mask_ref") or "") if isinstance(object_mask, dict) else ""
            )
        else:
            supplied_mask = str(parameters.get("target_mask") or "")
        if expected_mask and supplied_mask != expected_mask:
            if tool_name == "anygrasp":
                return (
                    "Targeted AnyGrasp must use the mask_ref from the recorded "
                    "select_sam3_detection result."
                )
            return "GraspGenX must use the mask_ref from the recorded select_sam3_detection result."
        return None

    def _capture_sam3_selection_state(self, action: EnvAction) -> None:
        command = action.command if isinstance(action.command, dict) else {}
        for call in _expanded_sam3_selection_calls(command.get("tool_calls", []) or []):
            if not isinstance(call, dict):
                continue
            segmenter_tool_name = str(call.get("name") or "")
            if segmenter_tool_name not in SELECTION_CAPTURE_TOOL_NAMES:
                continue
            result = call.get("result")
            if not isinstance(result, dict) or not bool(result.get("success")):
                continue
            details = result.get("details")
            if not isinstance(details, dict):
                continue
            outputs = details.get("outputs")
            if not isinstance(outputs, dict):
                outputs = details
            detections = outputs.get("detections")
            if not isinstance(detections, list):
                continue
            candidates = [
                dict(candidate) for candidate in detections if isinstance(candidate, dict)
            ]
            result_id = str(outputs.get("result_id") or "")
            if not result_id:
                result_id = f"{segmenter_tool_name}-{int(time.time() * 1000)}"
            selection_bundle = outputs.get("selection_bundle")
            if not isinstance(selection_bundle, dict):
                selection_bundle = {}
            parameters = details.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {}
            source_image = outputs.get("source_image") or parameters.get("image")
            semantic_role = (
                str(outputs.get("semantic_role") or parameters.get("semantic_role") or "")
                .strip()
                .lower()
            )
            semantic_role_source = str(
                outputs.get("semantic_role_source") or parameters.get("semantic_role_source") or ""
            ).strip()
            # The actual text prompt is the canonical visual phrase.  A model
            # may also emit an abstract semantic_target such as
            # ``target_object``; never let that alias replace the phrase used
            # successfully by the segmenter and required by later views.
            target_prompt = (
                outputs.get("prompt")
                or parameters.get("prompt")
                or outputs.get("semantic_target")
                or parameters.get("semantic_target")
            )
            previous_no_detection = self.sam3_no_detection(
                semantic_role if semantic_role in SAM3_SEMANTIC_ROLES else ""
            )
            generic_point_prompt = str(target_prompt or "").strip().lower() in {
                "",
                "point_prompt",
            }
            previous_prompt = (
                previous_no_detection.get("target_prompt")
                if isinstance(previous_no_detection, dict)
                else None
            )
            same_scene = (
                isinstance(previous_no_detection, dict)
                and _optional_int(previous_no_detection.get("scene_epoch"), default=-1)
                == self.scene_epoch()
            )
            continuing_missing_placement_role = (
                generic_point_prompt
                and same_scene
                and isinstance(previous_prompt, str)
                and (
                    (
                        _placement_region_prompt(previous_prompt)
                        and isinstance(self.placement_object_detection(), dict)
                        and not isinstance(self.placement_region_detection(), dict)
                    )
                    or (
                        not _placement_region_prompt(previous_prompt)
                        and not isinstance(self.placement_object_detection(), dict)
                    )
                )
            )
            if (
                generic_point_prompt
                and isinstance(previous_no_detection, dict)
                and (
                    str(previous_no_detection.get("source_image") or "") == str(source_image or "")
                    or continuing_missing_placement_role
                )
            ):
                # Point-SAM has no text channel of its own.  Preserve the
                # semantic target of the immediately preceding text failure,
                # including the normal top->wrist->top fallback sequence used
                # for the placement marker.  Otherwise the literal backend
                # label ``point_prompt`` is misclassified as the object mask
                # and the host can never build the AnyPlace obligation.
                target_prompt = previous_prompt
                if semantic_role not in SAM3_SEMANTIC_ROLES:
                    semantic_role = (
                        str(previous_no_detection.get("semantic_role") or "").strip().lower()
                    )
                    semantic_role_source = "inherited_previous_attempt"
            if semantic_role not in SAM3_SEMANTIC_ROLES:
                semantic_role = (
                    "placement_region"
                    if _placement_region_prompt(target_prompt)
                    else "grasp_target"
                )
                semantic_role_source = semantic_role_source or "legacy_prompt_inference"
            source_camera_role = str(
                outputs.get("source_camera_role") or details.get("source_camera_role") or ""
            )
            source_frame_id = (
                outputs.get("frame_id")
                or outputs.get("source_frame_id")
                or (details.get("source_frame_id") if source_camera_role else None)
                or parameters.get("frame_id")
            )
            evidence_scene_epoch = _optional_int(
                outputs.get("scene_epoch") or parameters.get("scene_epoch"),
                default=self.scene_epoch(),
            )
            if segmenter_tool_name == "active_observe":
                # active_observe is a composite host action: it may execute a
                # MoveIt-qualified camera motion, acquire a fresh simulator
                # receipt, and run SAM3 before returning.  The nested simulator
                # observation carries the environment's scene epoch, which can
                # lag the agent's logical epoch after earlier acknowledged
                # motions.  The result is captured synchronously in this
                # session, so bind its evidence to the current authoritative
                # memory epoch instead of making a fresh mask stale immediately.
                evidence_scene_epoch = self.scene_epoch()
            base = {
                "result_id": result_id,
                "target_prompt": target_prompt,
                "source_image": source_image,
                "frame_id": source_frame_id,
                "ranking": outputs.get("ranking") or "score_descending",
                "candidate_count": len(candidates),
                "candidates": candidates,
                "selection_bundle": dict(selection_bundle),
                "segmentation_mode": outputs.get("segmentation_mode"),
                "scene_epoch": evidence_scene_epoch,
                "semantic_role": semantic_role,
                "semantic_role_source": semantic_role_source or "explicit",
                "semantic_target": target_prompt,
                "perception_bundle_id": outputs.get("perception_bundle_id")
                or parameters.get("perception_bundle_id"),
                "observation_id": outputs.get("observation_id") or parameters.get("observation_id"),
                "attempt_id": outputs.get("attempt_id") or parameters.get("attempt_id"),
                "attempt_fingerprint": outputs.get("attempt_fingerprint")
                or parameters.get("attempt_fingerprint"),
                "view_identity": outputs.get("view_identity")
                or parameters.get("view_identity"),
            }
            positive_points = parameters.get("positive_points") or parameters.get("points")
            if isinstance(positive_points, list) and positive_points:
                base["positive_points"] = [
                    dict(point) for point in positive_points if isinstance(point, dict)
                ]
            selection_review = outputs.get("selection_review")
            if isinstance(selection_review, dict):
                base["selection_review"] = dict(selection_review)
            if source_camera_role:
                base["camera_role"] = source_camera_role
            reestimate = self.grasp_reestimation()
            reestimate_view_paths = {
                str(view.get("rgb_path") or "")
                for view in (
                    reestimate.get("observation_views", []) if isinstance(reestimate, dict) else []
                )
                if isinstance(view, dict)
            }
            reestimate_segmentation = (
                isinstance(reestimate, dict)
                and (
                    (
                        reestimate.get("status") == "ready"
                        and str(source_image or "") in reestimate_view_paths
                    )
                    or (
                        segmenter_tool_name == "active_observe"
                        and reestimate.get("status") == "pending_observation"
                        and reestimate.get("recovery_strategy")
                        == "active_view_relocalization"
                        and str(outputs.get("active_vision_mode") or "")
                        == "semantic_search"
                        and str(outputs.get("status") or "").lower() == "acquired"
                    )
                )
                and str(reestimate.get("target_prompt") or "").strip()
                == str(target_prompt or "").strip()
            )
            if reestimate_segmentation:
                reestimate.update(
                    {
                        "status": "selection_pending",
                        "segmentation_result_id": result_id,
                        "segmentation_source_image": source_image,
                        "segmentation_frame_id": source_frame_id,
                        "segmentation_candidate_count": len(candidates),
                        "segmented_at_s": time.time(),
                    }
                )
                if not candidates:
                    _record_failed_grasp_reestimate_view(
                        reestimate,
                        source_image=str(source_image or ""),
                        failure="segmentation_failed",
                    )
                self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
                    reestimate,
                    source="grasp_reestimate_segmentation",
                )
            host_obligation = _action_host_obligation(action)
            preserve_scene_target = (
                not candidates
                and host_obligation.get("schema_version")
                == "openeta.wrist_segmentation_obligation.v1"
                and host_obligation.get("stage") == "wrist_segmentation"
            )
            self.facts.pop(REFERENCE_LOCALIZATION_FAILURE_KEY, None)
            asset_reference = self.target_asset_reference()
            if isinstance(asset_reference, dict):
                verification = asset_reference.get("exact_instance_verification")
                if (
                    isinstance(verification, dict)
                    and str(verification.get("decision") or "").lower() == "match"
                    and str(parameters.get("image") or "")
                    == str(asset_reference.get("scene_image") or "")
                    and parameters.get("positive_points") == asset_reference.get("positive_points")
                ):
                    base["reference_verification"] = dict(verification)
            self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
            if semantic_role == "grasp_target" and not preserve_scene_target:
                self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)
            if candidates:
                self.facts.pop(SAM3_NO_DETECTION_KEY, None)
                self.facts[PENDING_SAM3_SELECTION_KEY] = _memory_fact_entry(
                    base,
                    source=segmenter_tool_name,
                )
                self._record_sam3_semantic_result(
                    base,
                    status="selection_pending",
                )
                review = outputs.get("selection_review")
                review = dict(review) if isinstance(review, dict) else {}
                review_decision = str(review.get("decision") or "").strip().lower()
                attachment = self.attachment_gate()
                legacy_single_region = (
                    not review_decision
                    and semantic_role_source == "legacy_prompt_inference"
                    and semantic_role == "placement_region"
                    and len(candidates) == 1
                    and outputs.get("selection_required") is False
                    and isinstance(attachment, dict)
                    and attachment.get("status") == "resolved"
                    and str(attachment.get("verdict") or "").upper() == "PASS"
                )
                if legacy_single_region:
                    region = dict(candidates[0])
                    region.update(
                        {
                            **{
                                key: value
                                for key, value in base.items()
                                if key not in {"candidates", "selection_bundle"}
                            },
                            "source_frame_id": source_frame_id,
                            "selection_source": "host_single_detection",
                            "selected_at_s": time.time(),
                        }
                    )
                    if source_camera_role:
                        region["source_camera_role"] = source_camera_role
                    self._save_placement_detection(
                        PLACEMENT_REGION_DETECTION_KEY,
                        region,
                        source="sam3_single_placement_region",
                    )
                    self._record_sam3_semantic_result(region, status="selected")
                if review_decision == "select":
                    detection_id = str(review.get("detection_id") or "")
                    try:
                        self.resolve_sam3_selection(
                            result_id=result_id,
                            detection_id=detection_id,
                            selection_source=str(
                                review.get("selection_source") or "isolated_main_vlm"
                            ),
                            confidence=_optional_float(review.get("confidence")),
                            reason=str(review.get("reason") or ""),
                            target_geometry_family=str(review.get("target_geometry_family") or ""),
                        )
                    except ValueError as exc:
                        review_decision = "deferred"
                        self.record(
                            "sam3_selection_review_invalid",
                            {
                                "result_id": result_id,
                                "semantic_role": semantic_role,
                                "error": str(exc),
                            },
                        )
                elif review_decision == "reject":
                    try:
                        self.reject_sam3_detections(
                            result_id=result_id,
                            reason=str(
                                review.get("reason") or "Visual reviewer rejected all masks."
                            ),
                        )
                    except ValueError as exc:
                        review_decision = "deferred"
                        self.record(
                            "sam3_selection_review_invalid",
                            {
                                "result_id": result_id,
                                "semantic_role": semantic_role,
                                "error": str(exc),
                            },
                        )
                if review_decision not in {"select", "reject"}:
                    self.record(
                        "sam3_detection_selection_required",
                        {
                            "result_id": result_id,
                            "candidate_count": len(candidates),
                            "semantic_role": semantic_role,
                            "review_status": review_decision or "not_configured",
                            "verification_scope": (
                                "single_detection"
                                if len(candidates) == 1
                                else "multiple_detections"
                            ),
                        },
                    )
            else:
                self.facts[SAM3_NO_DETECTION_KEY] = _memory_fact_entry(
                    base,
                    source=segmenter_tool_name,
                )
                self._record_sam3_semantic_result(base, status="no_detection")
                if semantic_role == "grasp_target":
                    self._capture_grasp_fallback_segmentation_failure(base)
                self.record(
                    "sam3_no_detection",
                    {"result_id": result_id, "semantic_role": semantic_role},
                )
            self._save_working_memory()

    def _capture_active_vision_state(self, action: EnvAction) -> None:
        """Retain one bounded active-search outcome without fabricating a SAM mask."""

        call = _tool_call(action, "active_observe")
        if not isinstance(call, dict):
            return
        result = call.get("result")
        details = result.get("details") if isinstance(result, dict) else None
        if not isinstance(details, dict):
            return
        outputs = details.get("outputs")
        if not isinstance(outputs, dict):
            return
        mode = str(outputs.get("active_vision_mode") or "").strip()
        status = str(outputs.get("status") or "").strip().lower()
        if mode != "semantic_search" or status not in {
            "acquired",
            "exhausted",
            "infrastructure_error",
        }:
            return
        record = {
            "result_id": outputs.get("result_id")
            or outputs.get("active_vision_attempt_id"),
            "semantic_role": outputs.get("semantic_role") or "grasp_target",
            "target_prompt": outputs.get("semantic_target"),
            "source_image": (
                (outputs.get("target_hint") or {}).get("source_image")
                if isinstance(outputs.get("target_hint"), dict)
                else None
            ),
            "segmentation_mode": "active_search",
            "scene_epoch": outputs.get("scene_epoch"),
            "perception_bundle_id": outputs.get("observation_bundle_id"),
            "observation_id": outputs.get("observation_id"),
            "attempt_id": outputs.get("active_vision_attempt_id"),
            "attempt_fingerprint": outputs.get("active_vision_attempt_fingerprint"),
            "active_vision_status": status,
            "active_vision_stop_reason": outputs.get("stop_reason"),
        }
        self._record_sam3_semantic_result(record, status=f"active_search_{status}")
        reestimate = self.grasp_reestimation()
        if (
            isinstance(reestimate, dict)
            and reestimate.get("recovery_strategy") == "active_view_relocalization"
            and status in {"exhausted", "infrastructure_error"}
        ):
            reestimate.update(
                {
                    "status": (
                        "active_view_infrastructure_error"
                        if status == "infrastructure_error"
                        else "active_view_exhausted"
                    ),
                    "active_vision_stop_reason": outputs.get("stop_reason"),
                    "active_vision_attempt_id": outputs.get("active_vision_attempt_id"),
                    "active_vision_completed_at_s": time.time(),
                }
            )
            self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
                reestimate,
                source="active_vision_grasp_reestimate",
            )
        self.record(
            "active_vision_search_completed",
            {
                "semantic_role": record["semantic_role"],
                "target_prompt": record["target_prompt"],
                "status": status,
                "stop_reason": record["active_vision_stop_reason"],
            },
        )
        self._save_working_memory()

    def _capture_grasp_fallback_segmentation_failure(self, sam3_result: JsonDict) -> None:
        policy = self.grasp_candidate_policy()
        if not isinstance(policy, dict) or policy.get("fallback_required") is not True:
            return
        target_prompt = str(sam3_result.get("target_prompt") or "").strip()
        fallback_prompt = str(policy.get("fallback_target_prompt") or "").strip()
        if target_prompt and fallback_prompt and target_prompt != fallback_prompt:
            return
        source_image = str(sam3_result.get("source_image") or "")
        if not source_image:
            return
        attempts = _grasp_fallback_attempts(policy)
        max_gripper_width = _policy_max_gripper_width(
            policy,
            default=float(self._active_grasp_calibration_capabilities()["max_gripper_width_m"]),
        )
        _append_grasp_fallback_attempt(
            attempts,
            {
                "backend": str(policy.get("source_backend") or "anygrasp"),
                "camera_frame_id": _camera_frame_for_source_image(
                    policy,
                    source_image,
                ),
                "source_rgb": source_image,
                "outcome": "segmentation_no_detection",
                "raw_candidate_count": 0,
                "width_limit_m": max_gripper_width,
            },
        )
        policy["fallback_attempts"] = attempts
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source="grasp_width_fallback",
        )
        self.record(
            "grasp_estimation_fallback_camera_exhausted",
            {
                "backend": str(policy.get("source_backend") or "anygrasp"),
                "source_rgb": source_image,
                "reason": "segmentation_no_detection",
            },
        )

    def _capture_reference_localization_state(self, action: EnvAction) -> None:
        command = action.command if isinstance(action.command, dict) else {}
        for call in command.get("tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "")
            result = call.get("result")
            if (
                name == "retrieve_asset_reference"
                and isinstance(result, dict)
                and not bool(result.get("success"))
            ):
                no_detection = self.sam3_no_detection()
                if isinstance(no_detection, dict):
                    parameters = call.get("parameters")
                    if not isinstance(parameters, dict):
                        request = command.get("request")
                        parameters = (
                            request.get("parameters") if isinstance(request, dict) else None
                        )
                    parameters = parameters if isinstance(parameters, dict) else {}
                    target_object = parameters.get("target_object") or no_detection.get(
                        "target_prompt"
                    )
                    scene_image = parameters.get("scene_image") or no_detection.get("source_image")
                    budget = _memory_fact_value(self.facts.get(TARGET_LOCALIZATION_BUDGET_KEY))
                    current_epoch = self.scene_epoch()
                    same_grounding_request = (
                        isinstance(budget, dict)
                        and str(budget.get("target_object") or "").strip().lower()
                        == str(target_object or "").strip().lower()
                        and budget.get("scene_epoch") == current_epoch
                    )
                    failure = {
                        "sam3_result_id": no_detection.get("result_id"),
                        "target_object": target_object,
                        "scene_image": scene_image,
                        "semantic_role": no_detection.get("semantic_role"),
                        "perception_bundle_id": no_detection.get("perception_bundle_id"),
                        "observation_id": no_detection.get("observation_id"),
                        "scene_epoch": current_epoch,
                        "failed_at_s": time.time(),
                        "molmopoint_attempts": (
                            int(budget.get("molmopoint_attempts") or 0)
                            if same_grounding_request
                            else 0
                        ),
                    }
                    self.facts[REFERENCE_LOCALIZATION_FAILURE_KEY] = _memory_fact_entry(
                        failure,
                        source=name,
                    )
                    self.record("asset_reference_localization_failed", dict(failure))
                    self._save_working_memory()
                continue
            if name == "molmopoint":
                host_obligation = _action_host_obligation(action)
                host_semantic_role = str(host_obligation.get("semantic_role") or "").strip().lower()
                no_detection = (
                    self.sam3_no_detection(host_semantic_role)
                    if host_semantic_role in SAM3_SEMANTIC_ROLES
                    else None
                ) or self.sam3_no_detection()
                if isinstance(no_detection, dict):
                    failure = self.reference_localization_failure() or {}
                    if str(failure.get("sam3_result_id") or "") != str(
                        no_detection.get("result_id") or ""
                    ):
                        failure = {
                            "sam3_result_id": no_detection.get("result_id"),
                            "target_object": host_obligation.get("semantic_target")
                            or no_detection.get("target_prompt"),
                            "scene_image": no_detection.get("source_image"),
                            "semantic_role": host_semantic_role
                            or no_detection.get("semantic_role"),
                            "perception_bundle_id": host_obligation.get("perception_bundle_id")
                            or no_detection.get("perception_bundle_id"),
                            "observation_id": host_obligation.get("observation_id")
                            or no_detection.get("observation_id"),
                            "failed_at_s": time.time(),
                            "molmopoint_attempts": 0,
                        }
                    failure["molmopoint_attempts"] = (
                        int(failure.get("molmopoint_attempts") or 0) + 1
                    )
                    failure["scene_epoch"] = self.scene_epoch()
                    if not isinstance(result, dict) or not bool(result.get("success")):
                        failure["last_molmopoint_error"] = (
                            str(result.get("content") or "MolmoPoint failed.")
                            if isinstance(result, dict)
                            else "MolmoPoint failed."
                        )
                        failure["last_molmopoint_failed_at_s"] = time.time()
                    else:
                        failure["last_molmopoint_succeeded_at_s"] = time.time()
                    self.facts[REFERENCE_LOCALIZATION_FAILURE_KEY] = _memory_fact_entry(
                        failure,
                        source=name,
                    )
                    self.facts[TARGET_LOCALIZATION_BUDGET_KEY] = _memory_fact_entry(
                        {
                            "target_object": failure.get("target_object"),
                            "scene_epoch": failure.get("scene_epoch"),
                            "molmopoint_attempts": failure["molmopoint_attempts"],
                            "last_scene_image": failure.get("scene_image"),
                            "updated_at_s": time.time(),
                        },
                        source=name,
                    )
                    self.record(
                        (
                            "molmopoint_localization_succeeded"
                            if isinstance(result, dict) and bool(result.get("success"))
                            else "molmopoint_localization_failed"
                        ),
                        dict(failure),
                    )
                    self._save_working_memory()
                if not isinstance(result, dict) or not bool(result.get("success")):
                    continue
            if not isinstance(result, dict) or not bool(result.get("success")):
                continue
            details = result.get("details")
            if not isinstance(details, dict):
                continue
            outputs = details.get("outputs")
            if not isinstance(outputs, dict):
                outputs = details
            if name == "retrieve_asset_reference":
                self.facts.pop(REFERENCE_LOCALIZATION_FAILURE_KEY, None)
                self.facts.pop(TARGET_LOCALIZATION_BUDGET_KEY, None)
                bundle = outputs.get("localization_bundle")
                if not isinstance(bundle, dict):
                    continue
                scene_image = str(bundle.get("scene_image_ref") or outputs.get("scene_image") or "")
                references = bundle.get("reference_image_refs")
                if not isinstance(references, list):
                    references = outputs.get("reference_images")
                reference_images = [
                    str(item) for item in (references or []) if isinstance(item, str) and item
                ]
                if not scene_image or not reference_images:
                    continue
                positive_points = bundle.get("positive_points")
                if not isinstance(positive_points, list):
                    positive_points = outputs.get("positive_points")
                bbox_xyxy = bundle.get("bbox_xyxy")
                if not isinstance(bbox_xyxy, list):
                    bbox_xyxy = outputs.get("bbox_xyxy")
                point_prompt = isinstance(positive_points, list) and bool(positive_points)
                localizer = outputs.get("localizer")
                if not isinstance(localizer, dict):
                    localizer = {}
                verification = localizer.get("verification")
                exact_instance_verification = (
                    {
                        "decision": "match",
                        "confidence": verification.get("confidence"),
                        "reason": verification.get("reason"),
                        "candidate_crop": verification.get("candidate_crop"),
                        "reference_geometry": verification.get("reference_geometry"),
                        "candidate_geometry": verification.get("candidate_geometry"),
                        "grasp_geometry_family": verification.get("grasp_geometry_family"),
                    }
                    if isinstance(verification, dict)
                    and str(verification.get("decision") or "").lower() == "match"
                    else None
                )
                obligation = {
                    "environment": outputs.get("environment") or bundle.get("environment"),
                    "target_object": outputs.get("target_object") or bundle.get("target_object"),
                    "scene_image": scene_image,
                    "reference_images": reference_images,
                    "marked_scene_image": bundle.get("marked_scene_image_ref")
                    or outputs.get("marked_scene_image"),
                    "positive_points": positive_points if point_prompt else None,
                    "bbox_xyxy": bbox_xyxy,
                    "localization_bundle": dict(bundle),
                    "exact_instance_verification": exact_instance_verification,
                    "required_next_tool": "sam3",
                    "required_parameter": ("positive_points" if point_prompt else "roi_bbox_xyxy"),
                    "semantic_role": (self.sam3_no_detection() or {}).get("semantic_role"),
                    "perception_bundle_id": (self.sam3_no_detection() or {}).get(
                        "perception_bundle_id"
                    ),
                    "observation_id": (self.sam3_no_detection() or {}).get("observation_id"),
                    "scene_epoch": self.scene_epoch(),
                }
                self.facts[PENDING_REFERENCE_LOCALIZATION_KEY] = _memory_fact_entry(
                    obligation,
                    source=name,
                )
                self.facts[TARGET_ASSET_REFERENCE_KEY] = _memory_fact_entry(
                    {
                        "environment": obligation["environment"],
                        "target_object": obligation["target_object"],
                        "memory_query_key": bundle.get("memory_query_key"),
                        "resolved_asset_key": (
                            bundle.get("resolved_asset_key") or outputs.get("resolved_asset_key")
                        ),
                        "memory_resolution": (
                            bundle.get("memory_resolution") or outputs.get("memory_resolution")
                        ),
                        "scene_image": scene_image,
                        "reference_images": reference_images,
                        "positive_points": positive_points if point_prompt else None,
                        "bbox_xyxy": bbox_xyxy,
                        "exact_instance_verification": exact_instance_verification,
                    },
                    source=name,
                )
                self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
                if str(obligation.get("semantic_role") or "grasp_target") == "grasp_target":
                    self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)
                self.record(
                    "asset_reference_localization_required",
                    {
                        "environment": obligation["environment"],
                        "target_object": obligation["target_object"],
                        "reference_count": len(reference_images),
                    },
                )
                self._save_working_memory()
                continue
            if name == "molmopoint":
                host_obligation = _action_host_obligation(action)
                host_semantic_role = str(host_obligation.get("semantic_role") or "").strip().lower()
                no_detection = (
                    self.sam3_no_detection(host_semantic_role)
                    if host_semantic_role in SAM3_SEMANTIC_ROLES
                    else None
                ) or self.sam3_no_detection()
                points = outputs.get("points")
                image_sources = outputs.get("image_sources")
                if (
                    not isinstance(no_detection, dict)
                    or not isinstance(points, list)
                    or not isinstance(image_sources, list)
                ):
                    continue
                normalized: list[JsonDict] = []
                scene_image = ""
                selected_index: int | None = None
                for point in points:
                    if not isinstance(point, dict):
                        continue
                    image_index = point.get("image_index")
                    x = point.get("pixel_x")
                    y = point.get("pixel_y")
                    if (
                        not isinstance(image_index, int)
                        or isinstance(image_index, bool)
                        or not 0 <= image_index < len(image_sources)
                        or not _finite_number(x)
                        or not _finite_number(y)
                    ):
                        continue
                    source = image_sources[image_index]
                    if not isinstance(source, str) or not source:
                        continue
                    if selected_index is None:
                        selected_index = image_index
                        scene_image = source
                    if image_index != selected_index:
                        continue
                    normalized.append({"x": float(x), "y": float(y), "label": 1})
                if not scene_image or not normalized:
                    continue
                obligation = {
                    "environment": None,
                    "target_object": no_detection.get("target_prompt"),
                    "scene_image": scene_image,
                    "reference_images": [],
                    "marked_scene_image": None,
                    "positive_points": normalized,
                    "bbox_xyxy": None,
                    "localization_bundle": {
                        "source": "molmopoint",
                        "image_index": selected_index,
                    },
                    "exact_instance_verification": None,
                    "required_next_tool": "sam3",
                    "required_parameter": "positive_points",
                    "semantic_role": host_semantic_role or no_detection.get("semantic_role"),
                    "perception_bundle_id": host_obligation.get("perception_bundle_id")
                    or no_detection.get("perception_bundle_id"),
                    "observation_id": host_obligation.get("observation_id")
                    or no_detection.get("observation_id"),
                    "scene_epoch": _optional_int(
                        host_obligation.get("scene_epoch") or no_detection.get("scene_epoch"),
                        default=self.scene_epoch(),
                    ),
                }
                self.facts[PENDING_REFERENCE_LOCALIZATION_KEY] = _memory_fact_entry(
                    obligation,
                    source=name,
                )
                self.facts.pop(REFERENCE_LOCALIZATION_FAILURE_KEY, None)
                self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
                if str(obligation.get("semantic_role") or "grasp_target") == "grasp_target":
                    self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)
                self.record(
                    "molmopoint_localization_required",
                    {
                        "target_object": obligation["target_object"],
                        "scene_image": scene_image,
                        "point_count": len(normalized),
                    },
                )
                self._save_working_memory()
                continue
            if name != "sam3" or PENDING_REFERENCE_LOCALIZATION_KEY not in self.facts:
                continue
            parameters = details.get("parameters")
            if not isinstance(parameters, dict):
                parameters = call.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {}
            pending = self.pending_reference_localization() or {}
            required_parameter = str(pending.get("required_parameter") or "roi_bbox_xyxy")
            geometry_matches = (
                parameters.get("positive_points") == pending.get("positive_points")
                if required_parameter == "positive_points"
                else parameters.get("roi_bbox_xyxy") is not None
            )
            if (
                str(parameters.get("image") or "") == str(pending.get("scene_image") or "")
                and geometry_matches
            ):
                self.facts.pop(PENDING_REFERENCE_LOCALIZATION_KEY, None)
                self.record(
                    "asset_reference_localization_resolved",
                    {
                        "target_object": pending.get("target_object"),
                        required_parameter: parameters.get(required_parameter),
                    },
                )
                self._save_working_memory()

    def _capture_anygrasp_candidate_policy(self, action: EnvAction) -> None:
        command = action.command if isinstance(action.command, dict) else {}
        for call in command.get("tool_calls", []) or []:
            source_tool = str(call.get("name") or "") if isinstance(call, dict) else ""
            if not isinstance(call, dict) or source_tool not in {
                "grasp_pose_estimate",
                "anygrasp",
                "graspgenx",
                "contact_graspnet",
            }:
                continue
            result = call.get("result")
            if not isinstance(result, dict) or not bool(result.get("success")):
                continue
            details = result.get("details")
            if not isinstance(details, dict):
                continue
            outputs = details.get("outputs")
            if not isinstance(outputs, dict):
                outputs = details
            source_backend = str(outputs.get("selected_backend") or source_tool)
            candidates_value = outputs.get("grasp_candidates")
            if not isinstance(candidates_value, list):
                continue
            raw_candidates = [
                dict(candidate)
                for candidate in candidates_value
                if isinstance(candidate, dict) and str(candidate.get("id") or "")
            ]
            generated_count_value = outputs.get("generated_candidate_count")
            generated_candidate_count = (
                generated_count_value
                if isinstance(generated_count_value, int)
                and not isinstance(generated_count_value, bool)
                and generated_count_value >= 0
                else len(raw_candidates)
            )
            raw_count_value = outputs.get("raw_candidate_count")
            raw_candidate_count = (
                raw_count_value
                if isinstance(raw_count_value, int)
                and not isinstance(raw_count_value, bool)
                and raw_count_value >= generated_candidate_count
                else generated_candidate_count
            )
            submitted_count_value = outputs.get("submitted_candidate_count")
            submitted_candidate_count = (
                submitted_count_value
                if isinstance(submitted_count_value, int)
                and not isinstance(submitted_count_value, bool)
                and submitted_count_value >= 0
                else generated_candidate_count
            )
            full_plan_pass_value = outputs.get("full_plan_pass_count")
            if not (
                isinstance(full_plan_pass_value, int)
                and not isinstance(full_plan_pass_value, bool)
                and full_plan_pass_value >= 0
            ):
                # Read old traces without emitting the retired alias in new
                # policy state.
                full_plan_pass_value = outputs.get("qualified_candidate_count")
            full_plan_pass_count = (
                full_plan_pass_value
                if isinstance(full_plan_pass_value, int)
                and not isinstance(full_plan_pass_value, bool)
                and full_plan_pass_value >= 0
                else len(raw_candidates)
            )
            if not raw_candidates:
                if isinstance(outputs.get("qualification_evidence"), dict):
                    qualification_evidence = dict(outputs["qualification_evidence"])
                    source_backend = str(outputs.get("selected_backend") or source_tool)
                    frozen_pool_active = isinstance(self.frozen_placement_goal_pool(), dict)
                    frozen_pair_count = _optional_int(outputs.get("frozen_pair_count"), default=0)
                    frozen_frontier_remaining = _optional_int(
                        outputs.get("frozen_grasp_frontier_remaining_count"),
                        default=0,
                    )
                    frozen_frontier_available = frozen_pool_active and frozen_frontier_remaining > 0
                    frozen_stop_reason = (
                        "frozen_grasp_frontier_expansion_required"
                        if frozen_frontier_available
                        else "frozen_grasp_place_pool_exhausted"
                        if frozen_pair_count > 0
                        else "frozen_grasp_pool_exhausted"
                    )
                    policy = {
                        "result_id": str(outputs.get("result_id") or ""),
                        "source_tool": source_tool,
                        "source_backend": source_backend,
                        "status": (
                            "frozen_frontier_required"
                            if frozen_frontier_available
                            else "exhausted"
                        ),
                        "stop_reason": (
                            frozen_stop_reason
                            if frozen_pool_active
                            else "no_moveit_qualified_candidates"
                        ),
                        "candidate_count": 0,
                        "raw_candidate_count": raw_candidate_count,
                        "generated_candidate_count": generated_candidate_count,
                        "submitted_candidate_count": submitted_candidate_count,
                        "full_plan_pass_count": full_plan_pass_count,
                        "active_rank": None,
                        "active_candidate": None,
                        "remaining_candidate_ids": [],
                        "candidates": [],
                        "qualification_evidence": qualification_evidence,
                    }
                    planning_scene_revision = _optional_int(
                        outputs.get("planning_scene_revision"),
                        default=_optional_int(
                            qualification_evidence.get("planning_scene_revision"),
                            default=-1,
                        ),
                    )
                    if planning_scene_revision >= 0:
                        # Frozen-frontier continuation is permitted only at
                        # the exact PlanningScene revision proved by the last
                        # qualification wave.  Preserve that revision at the
                        # policy level even when the pair join exposes no
                        # executable candidate yet; otherwise the explicit
                        # continuation obligation cannot be formed and the
                        # agent is left with observation tools that cannot
                        # advance the immutable provider queue.
                        policy["planning_scene_revision"] = planning_scene_revision
                    if frozen_pool_active:
                        policy.update(
                            {
                                "frozen_pair_count": frozen_pair_count,
                                "frozen_pair_grasp_branch_limit": _optional_int(
                                    outputs.get("frozen_pair_grasp_branch_limit"),
                                    default=0,
                                ),
                                "frozen_pair_lookahead_grasp_count": _optional_int(
                                    outputs.get("frozen_pair_lookahead_grasp_count"),
                                    default=0,
                                ),
                                "frozen_pair_full_plan_pass_count": _optional_int(
                                    outputs.get("frozen_pair_full_plan_pass_count"),
                                    default=0,
                                ),
                                "frozen_grasp_frontier_remaining_count": (
                                    frozen_frontier_remaining
                                ),
                                "frozen_grasp_frontier_generation": _optional_int(
                                    outputs.get("frozen_grasp_frontier_generation"),
                                    default=0,
                                ),
                                "model_inference_retry_allowed": (
                                    not frozen_frontier_available
                                ),
                            }
                        )
                    else:
                        policy["reestimate_required"] = {
                            "status": "pending_recovery",
                            "reason": "no_moveit_qualified_candidates",
                            "backend": source_backend,
                            "requires_fresh_observation": True,
                            "backend_switch_allowed": False,
                        }
                    self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                        policy, source="moveit_qualification"
                    )
                    self.record("grasp_candidates_moveit_rejected", dict(policy))
                    if frozen_frontier_available:
                        self.facts.pop(GRASP_REESTIMATION_KEY, None)
                        self.record(
                            "frozen_grasp_frontier_expansion_required",
                            dict(policy),
                        )
                        continue
                    if frozen_pool_active:
                        self.record(
                            "frozen_grasp_pool_exhausted_reestimate_allowed",
                            dict(policy),
                        )
                    selected_target = (
                        self.placement_object_detection()
                        if frozen_pool_active
                        else self.selected_sam3_detection()
                    )
                    source = outputs.get("source")
                    source = source if isinstance(source, dict) else {}
                    target_prompt = (
                        str(selected_target.get("target_prompt") or "").strip()
                        if isinstance(selected_target, dict)
                        else ""
                    )
                    source_rgb = str(outputs.get("source_rgb") or source.get("rgb") or "")
                    if target_prompt and source_rgb:
                        existing = self.grasp_reestimation()
                        previous_attempts = (
                            _optional_int(existing.get("attempt_count"), default=0)
                            if isinstance(existing, dict)
                            else 0
                        )
                        attempt_count = previous_attempts + 1
                        reestimate = {
                            "schema_version": "openeta.grasp_reestimate.v1",
                            "status": "pending_observation",
                            "reason": "moveit_qualification_zero_pass",
                            "attempt_count": attempt_count,
                            "scene_epoch": self.scene_epoch(),
                            "target_prompt": target_prompt,
                            "source_image": source_rgb,
                            "source_observation_rgb_paths": self._latest_current_observation_rgb_paths(),
                            "previous_view": str(
                                outputs.get("camera_frame_id")
                                or source.get("camera_frame_id")
                                or ""
                            ),
                            "source_tool": source_tool,
                            "source_backend": source_backend,
                            "invalidate_frozen_placement_pool": False,
                            "preserve_frozen_placement_pool": frozen_pool_active,
                            **_zero_pass_grasp_recovery_route(outputs),
                            "created_at_s": time.time(),
                        }
                        self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
                            reestimate, source="moveit_qualification_zero_pass_reestimate"
                        )
                        self.record("grasp_moveit_zero_pass_reestimate_required", reestimate)
                    else:
                        self._schedule_grasp_recovery(
                            rejection={
                                "source": "moveit_qualification_rejected",
                                "reason": "no_moveit_qualified_candidates",
                            },
                            candidate_id="",
                        )
                continue
            capabilities = self._active_grasp_calibration_capabilities()
            max_gripper_width = float(capabilities["max_gripper_width_m"])
            width_rejections = [
                candidate
                for candidate in raw_candidates
                if not _candidate_fits_gripper(
                    candidate,
                    max_gripper_width_m=max_gripper_width,
                )
            ]
            candidates = [
                candidate
                for candidate in raw_candidates
                if _candidate_fits_gripper(
                    candidate,
                    max_gripper_width_m=max_gripper_width,
                )
            ]
            candidates.sort(key=_grasp_candidate_sort_key)
            for score_rank, candidate in enumerate(candidates):
                candidate["score_rank"] = score_rank
            for rank, candidate in enumerate(candidates):
                candidate["rank"] = rank
            result_id = str(outputs.get("result_id") or "")
            if not result_id:
                result_id = f"{source_tool}-{int(time.time() * 1000)}"
            selected_target = (
                self.placement_object_detection()
                if self.frozen_placement_goal_pool() is not None
                else self.selected_sam3_detection()
            )
            source = outputs.get("source")
            if not isinstance(source, dict):
                source = {}
            source_rgb = str(outputs.get("source_rgb") or source.get("rgb") or "")
            source_depth = str(outputs.get("source_depth") or source.get("depth") or "")
            camera_frame_id = str(
                outputs.get("camera_frame_id") or source.get("camera_frame_id") or ""
            )
            if not camera_frame_id:
                camera_frame_id = self._latest_camera_frame_for_source_image(
                    source_rgb,
                    scene_epoch=self.scene_epoch(),
                )
            previous_policy = self.grasp_candidate_policy()
            existing_reestimate = self.grasp_reestimation()
            existing_recovery = self.grasp_estimation_recovery()
            target_prompt = (
                str(selected_target.get("target_prompt") or "").strip()
                if isinstance(selected_target, dict)
                else ""
            )
            matching_recovery = (
                isinstance(existing_recovery, dict)
                and bool(target_prompt)
                and str(existing_recovery.get("target_prompt") or "").strip() == target_prompt
            )
            previous_prompt = (
                str(previous_policy.get("fallback_target_prompt") or "").strip()
                if isinstance(previous_policy, dict)
                else ""
            )
            fallback_attempts = (
                _grasp_fallback_attempts(previous_policy)
                if target_prompt and target_prompt == previous_prompt
                else []
            )
            failed_request_fingerprints = (
                list(previous_policy.get("failed_request_fingerprints") or [])
                if isinstance(previous_policy, dict)
                and target_prompt
                and target_prompt == previous_prompt
                else []
            )
            all_candidates_over_width = bool(raw_candidates) and len(width_rejections) == len(
                raw_candidates
            )
            selection_required = outputs.get("selection_required") is True
            if all_candidates_over_width:
                _append_grasp_fallback_attempt(
                    fallback_attempts,
                    {
                        "backend": source_backend,
                        "camera_frame_id": camera_frame_id,
                        "source_rgb": source_rgb,
                        "outcome": "all_candidates_over_width",
                        "raw_candidate_count": len(raw_candidates),
                        "width_limit_m": max_gripper_width,
                    },
                )
            policy = {
                "result_id": result_id,
                "source_tool": source_tool,
                "source_backend": source_backend,
                "ranking": str(outputs.get("ranking") or "score_descending"),
                "status": (
                    "selection_required"
                    if candidates and selection_required
                    else "active"
                    if candidates
                    else "exhausted"
                ),
                "candidate_count": len(candidates),
                "raw_candidate_count": raw_candidate_count,
                "generated_candidate_count": generated_candidate_count,
                "submitted_candidate_count": submitted_candidate_count,
                "full_plan_pass_count": full_plan_pass_count,
                "active_rank": 0 if candidates and not selection_required else None,
                "active_candidate": (
                    candidates[0] if candidates and not selection_required else None
                ),
                "remaining_candidate_ids": [
                    str(candidate.get("id"))
                    for candidate in (candidates if selection_required else candidates[1:])
                ],
                "candidates": candidates,
                "source_rgb": source_rgb,
                "source_depth": source_depth,
                "camera_frame_id": camera_frame_id,
                "grasp_source": dict(source),
                "physical_width_limit_m": max_gripper_width,
                "grasp_calibration_id": capabilities["calibration_id"],
                "grasp_calibration_profile_path": capabilities["profile_path"],
                "candidate_attempt_count": 0,
                "max_candidate_attempts": (
                    max(
                        GRASP_CANDIDATE_MAX_ATTEMPTS,
                        min(
                            len(candidates),
                            max(
                                _optional_int(
                                    outputs.get("frozen_pair_grasp_branch_limit"),
                                    default=0,
                                ),
                                _optional_int(
                                    outputs.get("frozen_pair_lookahead_grasp_count"),
                                    default=0,
                                ),
                            ),
                        ),
                    )
                    if isinstance(self.frozen_placement_goal_pool(), dict)
                    else GRASP_CANDIDATE_MAX_ATTEMPTS
                ),
                "candidate_fallback": False,
                "failed_request_fingerprints": failed_request_fingerprints,
                "rejected_candidates": [
                    {
                        "candidate_id": candidate.get("id"),
                        "rank": candidate.get("rank"),
                        "score": candidate.get("score"),
                        "reason": (
                            "candidate width exceeds calibration "
                            f"max_gripper_width_m {max_gripper_width:.4f} m"
                        ),
                        "source": "physical_gripper_width_filter",
                    }
                    for candidate in width_rejections
                ],
                "scene_epoch": self.scene_epoch(),
                "planning_scene_revision": _optional_int(
                    outputs.get("scene_revision"),
                    default=_optional_int(
                        (outputs.get("qualification_evidence") or {}).get("planning_scene_revision")
                        if isinstance(outputs.get("qualification_evidence"), dict)
                        else None,
                        default=-1,
                    ),
                ),
                "activated_at_s": time.time(),
            }
            if isinstance(self.frozen_placement_goal_pool(), dict):
                policy.update(
                    {
                        "frozen_pair_grasp_branch_limit": _optional_int(
                            outputs.get("frozen_pair_grasp_branch_limit"),
                            default=0,
                        ),
                        "frozen_pair_lookahead_grasp_count": _optional_int(
                            outputs.get("frozen_pair_lookahead_grasp_count"),
                            default=len(candidates),
                        ),
                        "frozen_pair_full_plan_pass_count": _optional_int(
                            outputs.get("frozen_pair_full_plan_pass_count"),
                            default=len(candidates),
                        ),
                        "frozen_grasp_frontier_remaining_count": _optional_int(
                            outputs.get("frozen_grasp_frontier_remaining_count"),
                            default=0,
                        ),
                        "frozen_grasp_frontier_generation": _optional_int(
                            outputs.get("frozen_grasp_frontier_generation"),
                            default=0,
                        ),
                    }
                )
            compilation_queue = outputs.get("host_candidate_compilation_queue")
            if isinstance(compilation_queue, list):
                policy["host_candidate_compilations"] = {
                    str(event.get("candidate_id") or ""): dict(event["compiled_seed"])
                    for event in compilation_queue
                    if isinstance(event, dict)
                    and isinstance(event.get("compiled_seed"), dict)
                    and str(event.get("candidate_id") or "")
                    == str(event["compiled_seed"].get("candidate_id") or "")
                }
            if isinstance(outputs.get("qualification_evidence"), dict):
                policy["qualification_evidence"] = dict(outputs["qualification_evidence"])
            if target_prompt:
                policy["fallback_target_prompt"] = target_prompt
            if fallback_attempts:
                policy["fallback_attempts"] = fallback_attempts
            if all_candidates_over_width:
                seed_candidate = min(raw_candidates, key=_grasp_candidate_sort_key)
                policy.update(
                    {
                        "exhaustion_reason": "physical_gripper_width_filter",
                        "fallback_required": True,
                        "refinement_seed_candidate": dict(seed_candidate),
                    }
                )
            elif candidates and matching_recovery:
                policy["recovery_lineage"] = {
                    "recovery_id": existing_recovery.get("recovery_id"),
                    "trigger_class": existing_recovery.get("trigger_class"),
                    "trigger_scope": existing_recovery.get("trigger_scope"),
                    "source_result_id": existing_recovery.get("source_result_id"),
                }
            if isinstance(selected_target, dict) and str(outputs.get("mode") or "targeted") == (
                "targeted"
            ):
                policy["target_detection"] = dict(selected_target)
                geometry_family = str(selected_target.get("target_geometry_family") or "").strip()
                if geometry_family:
                    policy["compile_hints"] = {
                        "target_geometry_family": geometry_family,
                    }
                handle_policy = _articulated_handle_candidate_policy(
                    candidates=candidates,
                    selected_target=selected_target,
                    task=_active_task_text(self),
                    source_tool=source_tool,
                    source_rgb=source_rgb,
                    source_mask=str(
                        outputs.get("target_mask")
                        or outputs.get("object_mask")
                        or source.get("object_mask")
                        or ""
                    ),
                    camera_frame_id=camera_frame_id,
                    scene_epoch=self.scene_epoch(),
                    camera_extrinsics=self._latest_camera_extrinsics(
                        camera_frame_id=camera_frame_id,
                        scene_epoch=self.scene_epoch(),
                    ),
                )
                if handle_policy is not None:
                    policy.update(handle_policy)
            previous_release = self.placement_release()
            if isinstance(previous_release, dict) and str(previous_release.get("status") or "") in {
                "released",
                "failed",
                "invalidated",
            }:
                self.facts.pop(PLACEMENT_RELEASE_KEY, None)
                self.record(
                    "placement_release_cleared_for_new_grasp",
                    {
                        "previous_candidate_id": previous_release.get("candidate_id"),
                        "previous_placement_pose_id": previous_release.get("placement_pose_id"),
                        "new_result_id": result_id,
                    },
                )
            self.facts.pop(LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY, None)
            self.facts.pop(GRASP_REESTIMATION_KEY, None)
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source=source_tool,
            )
            zero_pass_reestimate = (
                not candidates
                and isinstance(outputs.get("qualification_evidence"), dict)
                and bool(target_prompt)
                and bool(source_rgb)
            )
            if zero_pass_reestimate:
                previous_attempts = (
                    _optional_int(existing_reestimate.get("attempt_count"), default=0)
                    if isinstance(existing_reestimate, dict)
                    else 0
                )
                attempt_count = previous_attempts + 1
                reestimate = {
                    "schema_version": "openeta.grasp_reestimate.v1",
                    "status": "pending_observation",
                    "reason": "moveit_qualification_zero_pass",
                    "attempt_count": attempt_count,
                    "scene_epoch": self.scene_epoch(),
                    "target_prompt": target_prompt,
                    "source_image": source_rgb,
                    "source_observation_rgb_paths": self._latest_current_observation_rgb_paths(),
                    "previous_view": camera_frame_id,
                    "source_tool": source_tool,
                    "source_backend": source_backend,
                    "invalidate_frozen_placement_pool": False,
                    "preserve_frozen_placement_pool": isinstance(
                        self.frozen_placement_goal_pool(), dict
                    ),
                    **_zero_pass_grasp_recovery_route(outputs),
                    "created_at_s": time.time(),
                }
                self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
                    reestimate, source=f"{source_tool}_zero_pass_reestimate"
                )
                self.record("grasp_moveit_zero_pass_reestimate_required", reestimate)
            elif all_candidates_over_width:
                self._schedule_grasp_estimation_recovery(
                    policy=policy,
                    seed_candidate=policy.get("refinement_seed_candidate"),
                    source=dict(source),
                    rejection={
                        "source": "physical_gripper_width_filter",
                        "reason": "all candidates exceed the physical gripper width limit",
                        "recovery_class": "perception_refinable",
                        "recovery_scope": "view",
                    },
                )
            elif candidates and isinstance(existing_recovery, dict):
                self.facts.pop(GRASP_ESTIMATION_RECOVERY_KEY, None)
                self.record(
                    (
                        "grasp_estimation_recovery_completed"
                        if matching_recovery
                        else "grasp_estimation_recovery_superseded"
                    ),
                    {
                        "recovery_id": existing_recovery.get("recovery_id"),
                        "result_id": result_id,
                        "candidate_count": len(candidates),
                        "source_backend": source_backend,
                        "camera_frame_id": camera_frame_id,
                    },
                )
            self.record(
                f"{source_tool}_candidate_activated",
                {
                    "result_id": result_id,
                    "candidate_id": candidates[0].get("id") if candidates else None,
                    "rank": 0 if candidates else None,
                    "score": candidates[0].get("score") if candidates else None,
                    "candidate_count": len(candidates),
                    "width_rejection_count": len(width_rejections),
                },
            )
            self._save_working_memory()

    def _latest_camera_extrinsics(
        self,
        *,
        camera_frame_id: str,
        scene_epoch: int,
    ) -> JsonDict | None:
        if not camera_frame_id:
            return None
        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            try:
                observed_epoch = int(event.payload.get("scene_epoch"))
            except (TypeError, ValueError):
                continue
            if observed_epoch != scene_epoch:
                continue
            calibrations = event.payload.get("runtime_camera_calibrations")
            if not isinstance(calibrations, list):
                continue
            for calibration in calibrations:
                if not isinstance(calibration, dict):
                    continue
                if str(calibration.get("frame_id") or "") != camera_frame_id:
                    continue
                extrinsics = calibration.get("extrinsics")
                return dict(extrinsics) if isinstance(extrinsics, dict) else None
        return None

    def _latest_camera_frame_for_source_image(
        self,
        source_image: str,
        *,
        scene_epoch: int,
    ) -> str:
        if not source_image:
            return ""
        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            try:
                observed_epoch = int(event.payload.get("scene_epoch"))
            except (TypeError, ValueError):
                continue
            if observed_epoch != scene_epoch:
                continue
            sources = event.payload.get("runtime_camera_sources")
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict):
                    continue
                if str(source.get("rgb_path") or "") == source_image:
                    return str(source.get("frame_id") or "")
        return ""

    def _latest_current_observation_rgb_paths(self) -> list[str]:
        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            sources = event.payload.get("current_camera_sources")
            if not isinstance(sources, list):
                return []
            return [
                str(source["rgb_path"])
                for source in sources
                if isinstance(source, dict)
                and isinstance(source.get("rgb_path"), str)
                and source["rgb_path"]
            ]
        return []

    def _advance_anygrasp_candidate_after_rejection(self, action: EnvAction) -> bool:
        policy = self.grasp_candidate_policy()
        if policy is None or str(policy.get("status") or "") not in {"active", "accepted"}:
            return False
        active = policy.get("active_candidate")
        if not isinstance(active, dict):
            return False
        active_candidate_id = str(active.get("id") or "")
        rejection = _host_stage_motion_review_rejection(
            action,
            active_candidate_id=active_candidate_id,
            execution=self.grasp_execution(),
        )
        if rejection is None:
            rejection = _host_stage_close_rejection(
                action,
                active_candidate_id=active_candidate_id,
                execution=self.grasp_execution(),
            )
        if rejection is None:
            if str(policy.get("status") or "") == "accepted":
                rejection = _candidate_linked_grasp_outcome_rejection(
                    action,
                    active_candidate_id=active_candidate_id,
                    articulated_probe=self.articulated_attachment_probe(),
                    attachment_gate=self.attachment_gate(),
                    execution=self.grasp_execution(),
                )
            else:
                rejection = _candidate_linked_rejection(
                    action,
                    active_candidate_id=active_candidate_id,
                    artifacts=self.artifacts,
                    final_refinable_fallback=(policy.get("final_refinable_fallback") is True),
                )
        if rejection is None:
            return False
        execution = self.grasp_execution()
        if (
            isinstance(execution, dict)
            and str(execution.get("candidate_id") or "") == active_candidate_id
        ):
            rejection = {**rejection, "grasp_stage": execution.get("stage")}

        advanced = self._apply_anygrasp_candidate_rejection(
            policy=policy,
            active=active,
            rejection=rejection,
        )
        # Retained candidates are already qualified from the same model result.
        # Reopen after a failed close, then restore the captured pre-attempt pose
        # before exposing another candidate. Only an exhausted queue may request
        # fresh perception/model inference.
        next_candidate_available = str(policy.get("status") or "") == "active" and isinstance(
            policy.get("active_candidate"), dict
        )
        reestimate_required = isinstance(policy.get("reestimate_required"), dict)
        restore_required = _grasp_recovery_requires_restore(rejection)
        if advanced and (
            str(rejection.get("grasp_stage") or "") == "close"
            or restore_required
            or reestimate_required
        ):
            self._schedule_grasp_recovery(
                rejection=rejection,
                candidate_id=active_candidate_id,
                observe_after_reopen=reestimate_required and not next_candidate_available,
            )
            recovery_class = _grasp_estimation_recovery_class(rejection)
            if (
                str(policy.get("status") or "") == "exhausted"
                and policy.get("final_refinable_fallback") is not True
                and recovery_class in {"perception_refinable", "uncertain_review"}
            ):
                rejection = {
                    **rejection,
                    "recovery_class": recovery_class,
                    "recovery_scope": "candidate",
                }
                policy.update(
                    {
                        "fallback_required": True,
                        "exhaustion_reason": recovery_class,
                        "refinement_seed_candidate": dict(active),
                    }
                )
                attempts = _grasp_fallback_attempts(policy)
                _append_grasp_fallback_attempt(
                    attempts,
                    {
                        "backend": str(policy.get("source_backend") or "anygrasp"),
                        "camera_frame_id": policy.get("camera_frame_id"),
                        "source_rgb": policy.get("source_rgb"),
                        "outcome": f"all_candidates_{recovery_class}",
                        "raw_candidate_count": policy.get("raw_candidate_count"),
                    },
                )
                policy["fallback_attempts"] = attempts
                self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                    policy,
                    source="grasp_estimation_recovery",
                )
                self._schedule_grasp_estimation_recovery(
                    policy=policy,
                    seed_candidate=_highest_refinable_candidate(policy) or active,
                    source=_policy_grasp_source(policy),
                    rejection=rejection,
                )
        return advanced

    def _schedule_grasp_recovery(
        self,
        *,
        rejection: JsonDict,
        candidate_id: str,
        observe_after_reopen: bool = True,
    ) -> bool:
        if str(rejection.get("source") or "") not in {
            "candidate_motion_rejected",
            "independent_grasp_outcome_rejected",
            "reconciled_candidate_motion_rejected",
            "safety_check_rejected",
            "grasp_seed_geometry_rejected",
            "independent_host_stage_review_rejected",
            "host_gripper_close_failed",
            "articulated_attachment_assessment_failed",
            "moveit_qualification_rejected",
        }:
            return False
        target_detection = (self.grasp_candidate_policy() or {}).get("target_detection")
        target_detection = target_detection if isinstance(target_detection, dict) else {}
        previous_view = str(target_detection.get("frame_id") or "agentview")
        failed_stage = str(rejection.get("grasp_stage") or "")
        reopen_required = failed_stage == "close"
        recovery_id = f"grasp-recovery-{uuid4()}"
        restore_required = _grasp_recovery_requires_restore(rejection)
        execution = self.grasp_execution()
        restore_collision_policy = _grasp_recovery_collision_policy(
            self.grasp_candidate_policy(),
            candidate_id=candidate_id,
        )
        source_eef_pose = (
            execution.get("source_eef_pose") if isinstance(execution, dict) else None
        )
        restore_pose = _world_recovery_pose(source_eef_pose)
        if restore_required and restore_pose is None:
            recovery = {
                "schema_version": "openeta.grasp_recovery.v2",
                "status": "stopped_requires_human",
                "recovery_id": recovery_id,
                "candidate_id": candidate_id,
                "rejection_source": rejection.get("source"),
                "rejection_reason": rejection.get("reason"),
                "stage": "restore",
                "required_action": None,
                "stop_reason": "grasp_restore_anchor_unavailable",
                "created_at_s": time.time(),
                "completed_at_s": time.time(),
            }
            self.facts[GRASP_RECOVERY_KEY] = _memory_fact_entry(
                recovery,
                source="grasp_restore_anchor_unavailable",
            )
            self.record("grasp_recovery_stopped", dict(recovery))
            return True
        initial_stage = (
            "reopen"
            if reopen_required
            else "restore"
            if restore_required
            else "observe"
        )
        recovery = {
            "schema_version": "openeta.grasp_recovery.v2",
            "status": "required",
            "recovery_id": recovery_id,
            "candidate_id": candidate_id,
            "rejection_source": rejection.get("source"),
            "rejection_reason": rejection.get("reason"),
            "purpose": (
                "candidate_reestimate"
                if isinstance(
                    (self.grasp_candidate_policy() or {}).get("reestimate_required"),
                    dict,
                )
                else "candidate_fallback"
            ),
            "reestimate_strategy": "alternate_camera_view",
            "previous_view": previous_view,
            "observation_views": ["agentview", "wrist", "render"],
            "scene_epoch": self.scene_epoch(),
            "stage": initial_stage,
            "reopen_required": reopen_required,
            "restore_required": restore_required,
            "source_eef_pose": (
                dict(source_eef_pose) if isinstance(source_eef_pose, dict) else None
            ),
            "restore_pose": restore_pose,
            **(
                {"restore_collision_policy": restore_collision_policy}
                if restore_collision_policy is not None
                else {}
            ),
            "observe_after_reopen": observe_after_reopen,
            "required_action": (
                {"name": "gripper_control", "parameters": {"position": 1}}
                if reopen_required
                else _grasp_restore_action(
                    restore_pose,
                    recovery_id=recovery_id,
                    scene_epoch=self.scene_epoch(),
                    collision_policy=restore_collision_policy,
                )
                if restore_required
                else {"name": "observe", "parameters": {}}
            ),
            "created_at_s": time.time(),
        }
        self.facts[GRASP_RECOVERY_KEY] = _memory_fact_entry(
            recovery,
            source="structured_candidate_rejection",
        )
        self.record("grasp_recovery_required", dict(recovery))
        return True

    def _advance_grasp_recovery(self, action: EnvAction) -> bool:
        recovery = self.grasp_recovery()
        if not isinstance(recovery, dict) or recovery.get("status") != "required":
            return False
        required = recovery.get("required_action")
        if not isinstance(required, dict) or not _action_matches(action, required):
            return False
        name = str(required.get("name") or "")
        call = _tool_call(action, name)
        if not isinstance(call, dict):
            return False
        completed = _call_result_success(call) and not _motion_call_rejects_candidate(call)
        stage = str(recovery.get("stage") or "")
        if stage in {"reopen", "restore"}:
            if not completed:
                reconciliation = self.motion_reconciliation()
                intended_parameters = (
                    reconciliation.get("intended_parameters")
                    if isinstance(reconciliation, dict)
                    else None
                )
                if (
                    isinstance(reconciliation, dict)
                    and reconciliation.get("status") in {"required", "unresolved"}
                    and reconciliation.get("tool") == name
                    and intended_parameters == required.get("parameters")
                ):
                    recovery.update(
                        {
                            "status": "reconciling",
                            "reconciling_action": dict(required),
                            "required_action": None,
                            "reconciliation_created_at_s": reconciliation.get("created_at_s"),
                        }
                    )
                    self.facts[GRASP_RECOVERY_KEY] = _memory_fact_entry(
                        recovery,
                        source=f"candidate_{stage}_reconciling",
                    )
                    self.record(
                        f"grasp_recovery_{stage}_reconciling",
                        dict(recovery),
                    )
                    return True
                recovery.update(
                    {
                        "status": "stopped_requires_human",
                        "required_action": None,
                        "stop_reason": f"grasp_{stage}_not_completed",
                        "completed_at_s": time.time(),
                    }
                )
                self.facts[GRASP_RECOVERY_KEY] = _memory_fact_entry(
                    recovery, source=f"candidate_{stage}_stopped"
                )
                self.record(f"grasp_recovery_{stage}_stopped", dict(recovery))
                return True
            if (
                stage == "reopen"
                and recovery.get("purpose") == "attached_place_frontier_recovery"
            ):
                self._complete_attached_place_frontier_recovery(recovery, call=call)
                self.facts[GRASP_RECOVERY_KEY] = _memory_fact_entry(
                    recovery, source="attached_place_frontier_recovery_completed"
                )
                self.record(
                    (
                        "attached_place_frontier_recovery_completed"
                        if recovery.get("status") == "completed"
                        else "attached_place_frontier_recovery_stopped"
                    ),
                    dict(recovery),
                )
                return True
            self._complete_grasp_recovery_physical_stage(
                recovery,
                stage=stage,
                reconciled=False,
            )
            self.facts[GRASP_RECOVERY_KEY] = _memory_fact_entry(
                recovery, source=f"candidate_{stage}_completed"
            )
            self.record(f"grasp_recovery_{stage}_completed", dict(recovery))
            return True
        recovery.update(
            {
                "status": "completed" if completed else "failed",
                "completed_at_s": time.time(),
                "scene_epoch": self.scene_epoch(),
                "result": {
                    "tool": name,
                    "reached_target": completed,
                    "reason": "" if completed else _call_failure_reason(call),
                },
            }
        )
        self.facts[GRASP_RECOVERY_KEY] = _memory_fact_entry(
            recovery,
            source="candidate_reestimate_observation",
        )
        self.record(
            "grasp_recovery_completed" if completed else "grasp_recovery_failed",
            dict(recovery),
        )
        if completed and recovery.get("purpose") == "candidate_reestimate":
            reestimate = (self.grasp_candidate_policy() or {}).get("reestimate_required")
            if isinstance(reestimate, dict):
                reestimate.update(
                    {
                        "status": "pending_observation",
                        "recovery_completed_at_s": time.time(),
                        "previous_view": recovery.get("previous_view"),
                        "reestimate_strategy": recovery.get("reestimate_strategy"),
                    }
                )
                self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
                    reestimate,
                    source="candidate_reestimate_recovery_completed",
                )
                self.record("grasp_reestimate_observation_required", dict(reestimate))
        return True

    def _complete_attached_place_frontier_recovery(
        self,
        recovery: JsonDict,
        *,
        call: JsonDict,
    ) -> None:
        """Close an attached-place recovery only after its native rebind proof.

        The simulator has already detached/opened the gripper.  Do not expose
        another frozen model grasp until its source world pose, current world
        pose, and unchanged static-scene proof are all present.  The runtime
        assembly verifies this evidence again while rebasing the coordinator;
        this early check prevents stale attachment facts from leaking into the
        next planner turn.
        """

        receipt = _environment_receipt(call)
        evidence = receipt.get("frozen_grasp_frontier_recovery")
        evidence = evidence if isinstance(evidence, dict) else {}
        sync_entry = self.planning_scene_target_pose_sync()
        sync = (
            sync_entry.get("planning_scene_target_pose_sync")
            if isinstance(sync_entry, dict)
            else None
        )
        sync = sync if isinstance(sync, dict) else {}
        policy = self.grasp_candidate_policy()
        pending = (
            policy.get("frozen_grasp_frontier_rebase_pending")
            if isinstance(policy, dict)
            else None
        )
        candidate_id = str(recovery.get("candidate_id") or "")
        expected_source_revision = _optional_int(
            pending.get("source_planning_scene_revision")
            if isinstance(pending, dict)
            else None,
            default=-1,
        )
        revision = _optional_int(receipt.get("planning_scene_revision"), default=-1)
        detached = receipt.get("detachable_joint")
        valid = (
            isinstance(policy, dict)
            and policy.get("status") == "frozen_frontier_required"
            and isinstance(pending, dict)
            and str(pending.get("physically_rejected_candidate_id") or "") == candidate_id
            and evidence.get("schema_version") == "openeta.frozen_grasp_frontier_recovery.v1"
            and evidence.get("status") == "ready"
            and evidence.get("model_inference_invoked") is False
            and isinstance(detached, dict)
            and detached.get("state") == "detached"
            and sync.get("schema_version") == "openeta.planning_scene_target_pose_sync.v1"
            and sync.get("operation") == "update_world_target"
            and sync.get("topology_unchanged") is True
            and sync.get("static_world_unchanged") is True
            and sync.get("attached_ids_before") == []
            and sync.get("attached_ids_after") == []
            and _optional_int(sync.get("source_revision"), default=-1)
            == expected_source_revision
            and _optional_int(sync.get("revision"), default=-1) == revision
            and revision >= 0
        )
        if not valid:
            recovery.update(
                {
                    "status": "stopped_requires_human",
                    "stage": "frontier_rebind",
                    "required_action": None,
                    "stop_reason": "attached_place_frontier_rebind_proof_missing",
                    "completed_at_s": time.time(),
                }
            )
            # The physical open is irreversible even if its proof was
            # incomplete.  Never leave stale attached-state facts available
            # to a later planner decision.
            for key in (
                GRASP_EXECUTION_KEY,
                ATTACHMENT_GATE_KEY,
                PLACEMENT_CANDIDATE_POLICY_KEY,
                PLACEMENT_RELEASE_KEY,
            ):
                self.facts.pop(key, None)
            return

        assert isinstance(policy, dict)
        policy.update(
            {
                "status": "frozen_frontier_required",
                "scene_epoch": self.scene_epoch(),
                "planning_scene_revision": revision,
                "active_rank": None,
                "active_candidate": None,
                "remaining_candidate_ids": [],
                "frozen_grasp_frontier_recovery": {
                    "schema_version": "openeta.frozen_grasp_frontier_recovery.v1",
                    "source_planning_scene_revision": expected_source_revision,
                    "planning_scene_revision": revision,
                    "candidate_id": candidate_id,
                    "model_inference_invoked": False,
                },
            }
        )
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy, source="attached_place_frontier_rebound"
        )
        for key in (
            GRASP_EXECUTION_KEY,
            ATTACHMENT_GATE_KEY,
            PLACEMENT_CANDIDATE_POLICY_KEY,
            PLACEMENT_RELEASE_KEY,
        ):
            self.facts.pop(key, None)
        recovery.update(
            {
                "status": "completed",
                "stage": "completed",
                "required_action": None,
                "completed_at_s": time.time(),
                "scene_epoch": self.scene_epoch(),
                "result": {
                    "tool": "gripper_control",
                    "reached_target": True,
                    "model_inference_invoked": False,
                    "planning_scene_revision": revision,
                },
            }
        )

    def _resolve_reconciled_grasp_recovery(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
        verdict: str,
    ) -> bool:
        """Resolve an unknown physical recovery action from observed state."""

        recovery = self.grasp_recovery()
        if not isinstance(recovery, dict) or recovery.get("status") != "reconciling":
            return False
        stage = str(recovery.get("stage") or "")
        required = recovery.get("reconciling_action")
        expected_tool = "gripper_control" if stage == "reopen" else "move_to"
        if (
            stage not in {"reopen", "restore"}
            or tool_name != expected_tool
            or not isinstance(required, dict)
            or parameters != required.get("parameters")
        ):
            return False
        if verdict == "completed":
            recovery.pop("reconciling_action", None)
            self._complete_grasp_recovery_physical_stage(
                recovery,
                stage=stage,
                reconciled=True,
            )
            source = f"candidate_{stage}_reconciled"
        elif verdict == "failed":
            recovery.pop("reconciling_action", None)
            recovery.update(
                {
                    "status": "stopped_requires_human",
                    "required_action": None,
                    "stop_reason": f"grasp_{stage}_reconciliation_failed",
                    "completed_at_s": time.time(),
                }
            )
            source = f"candidate_{stage}_reconciliation_failed"
        else:
            return False
        self.facts[GRASP_RECOVERY_KEY] = _memory_fact_entry(recovery, source=source)
        self.record(f"grasp_recovery_{stage}_reconciled", dict(recovery))
        return True

    def _complete_grasp_recovery_physical_stage(
        self,
        recovery: JsonDict,
        *,
        stage: str,
        reconciled: bool,
    ) -> None:
        """Advance reopen/restore without exposing the next frozen candidate early."""

        now = time.time()
        if stage == "reopen":
            recovery["gripper_reopened_at_s"] = now
            restore_pose = recovery.get("restore_pose")
            if recovery.get("restore_required") is True and isinstance(restore_pose, dict):
                recovery.update(
                    {
                        "status": "required",
                        "stage": "restore",
                        "required_action": _grasp_restore_action(
                            restore_pose,
                            recovery_id=str(recovery.get("recovery_id") or ""),
                            scene_epoch=self.scene_epoch(),
                            collision_policy=(
                                recovery.get("restore_collision_policy")
                                if isinstance(
                                    recovery.get("restore_collision_policy"),
                                    dict,
                                )
                                else None
                            ),
                        ),
                    }
                )
            else:
                self._finish_grasp_recovery_physical_stages(recovery, now=now)
        else:
            recovery["restored_at_s"] = now
            self._rebind_pending_grasp_execution_after_restore(recovery)
            self._finish_grasp_recovery_physical_stages(recovery, now=now)
        if reconciled:
            recovery["reconciled_from_observation"] = True

    def _finish_grasp_recovery_physical_stages(
        self,
        recovery: JsonDict,
        *,
        now: float,
    ) -> None:
        observe = recovery.get("observe_after_reopen") is True
        recovery.update(
            {
                "status": "required" if observe else "completed",
                "stage": "observe" if observe else "completed",
                "required_action": (
                    {"name": "observe", "parameters": {}} if observe else None
                ),
                **({"completed_at_s": now} if not observe else {}),
            }
        )

    def _rebind_pending_grasp_execution_after_restore(
        self,
        recovery: JsonDict,
    ) -> None:
        """Bind the next retained candidate to the proven restored start pose."""

        execution = self.grasp_execution()
        policy = self.grasp_candidate_policy()
        active = policy.get("active_candidate") if isinstance(policy, dict) else None
        if (
            not isinstance(execution, dict)
            or execution.get("status") != "required"
            or not isinstance(active, dict)
            or str(execution.get("candidate_id") or "") != str(active.get("id") or "")
        ):
            return
        source_pose = recovery.get("source_eef_pose")
        if isinstance(source_pose, dict):
            execution["source_eef_pose"] = dict(source_pose)
        execution["scene_epoch"] = self.scene_epoch()
        compiled = execution.get("compiled_grasp")
        if isinstance(compiled, dict):
            compiled = dict(compiled)
            compiled["scene_epoch"] = self.scene_epoch()
            contact = compiled.get("contact_pose")
            if isinstance(contact, dict):
                contact = _pose_for_epoch(contact, self.scene_epoch())
                compiled["contact_pose"] = contact
                if execution.get("stage") == "contact":
                    execution["required_action"] = {
                        "name": "move_to",
                        "parameters": {"target_pose": contact},
                    }
            execution["compiled_grasp"] = compiled
        execution["recovery_anchor"] = {
            "schema_version": "openeta.grasp_recovery_anchor.v1",
            "recovery_id": recovery.get("recovery_id"),
            "scene_epoch": self.scene_epoch(),
        }
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="grasp_recovery_anchor_restored",
        )

    def _schedule_grasp_estimation_recovery(
        self,
        *,
        policy: JsonDict,
        seed_candidate: object,
        source: object,
        rejection: JsonDict,
    ) -> bool:
        trigger_class = str(rejection.get("recovery_class") or "")
        if trigger_class not in {"perception_refinable", "uncertain_review"}:
            return False
        if not isinstance(seed_candidate, dict):
            return False
        target_prompt = str(policy.get("fallback_target_prompt") or "").strip()
        if not target_prompt:
            return False
        existing = self.grasp_estimation_recovery()
        same_target = (
            isinstance(existing, dict)
            and str(existing.get("target_prompt") or "").strip() == target_prompt
        )
        if same_target and isinstance(existing, dict) and existing.get("status") == "blocked":
            return False
        recovery_id = (
            str(existing.get("recovery_id") or "")
            if same_target and isinstance(existing, dict)
            else ""
        ) or f"grasp-refinement-{uuid4()}"
        recovery: JsonDict = {
            "schema_version": "openeta.grasp_estimation_recovery.v1",
            "status": "required",
            "recovery_id": recovery_id,
            "trigger_class": trigger_class,
            "trigger_scope": str(rejection.get("recovery_scope") or "candidate"),
            "trigger_source": rejection.get("source"),
            "trigger_reason": rejection.get("reason"),
            "source_result_id": policy.get("result_id"),
            "source_backend": policy.get("source_backend"),
            "source_camera_frame_id": policy.get("camera_frame_id"),
            "source_rgb": policy.get("source_rgb"),
            "target_prompt": target_prompt,
            "seed_candidate": dict(seed_candidate),
            "created_scene_epoch": self.scene_epoch(),
            "created_at_s": time.time(),
        }
        fallback_candidates = (
            [
                dict(item)
                for item in existing.get("fallback_candidates", [])
                if isinstance(item, dict)
            ]
            if same_target and isinstance(existing, dict)
            else []
        )
        candidate_entry = {
            "candidate": dict(seed_candidate),
            "source_result_id": policy.get("result_id"),
            "source_tool": policy.get("source_tool"),
            "source_backend": policy.get("source_backend"),
            "camera_frame_id": policy.get("camera_frame_id"),
            "source_rgb": policy.get("source_rgb"),
            "source_depth": policy.get("source_depth"),
            "source": dict(source) if isinstance(source, dict) else {},
            "target_detection": (
                dict(policy["target_detection"])
                if isinstance(policy.get("target_detection"), dict)
                else {}
            ),
            "trigger_class": trigger_class,
            "grasp_calibration": {
                "calibration_id": policy.get("grasp_calibration_id"),
                "profile_path": policy.get("grasp_calibration_profile_path"),
                "max_gripper_width_m": policy.get("physical_width_limit_m"),
            },
        }
        _append_refinable_fallback_candidate(fallback_candidates, candidate_entry)
        recovery["fallback_candidates"] = fallback_candidates
        self.facts[GRASP_ESTIMATION_RECOVERY_KEY] = _memory_fact_entry(
            recovery,
            source="structured_grasp_estimation_rejection",
        )
        self.record("grasp_estimation_recovery_required", dict(recovery))
        return True

    def _action_end_effector_xyz(self, action: EnvAction) -> object:
        call = _tool_call(action, "move_to") or _tool_call(action, "follow_eef_trajectory")
        outputs = _tool_call_outputs(call) if isinstance(call, dict) else {}
        motion = outputs.get("motion_summary")
        end = motion.get("end") if isinstance(motion, dict) else None
        xyz = end.get("xyz") if isinstance(end, dict) else None
        return xyz if _finite_xyz(xyz) else self._latest_observed_end_effector_xyz()

    def _latest_observed_end_effector_xyz(self) -> object:
        pose = self._latest_observed_end_effector_pose()
        xyz = pose.get("xyz") if isinstance(pose, dict) else None
        return xyz if _finite_xyz(xyz) else None

    def _latest_observed_end_effector_pose(self) -> JsonDict | None:
        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            robot = event.payload.get("robot")
            pose = robot.get("end_effector_pose") if isinstance(robot, dict) else None
            xyz = pose.get("xyz") if isinstance(pose, dict) else None
            if _finite_xyz(xyz):
                return dict(pose)
        return None

    def _latest_observed_gripper_state(self) -> JsonDict | None:
        """Return a current-epoch physical gripper state, if one was observed."""

        current_epoch = self.scene_epoch()
        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            if _optional_int(event.payload.get("scene_epoch"), default=-1) != current_epoch:
                continue
            robot = event.payload.get("robot")
            state = robot.get("gripper_state") if isinstance(robot, dict) else None
            if isinstance(state, dict) and isinstance(state.get("open"), bool):
                return dict(state)
        return None

    def _apply_anygrasp_candidate_rejection(
        self,
        *,
        policy: JsonDict,
        active: JsonDict,
        rejection: JsonDict,
    ) -> bool:
        """Advance one ranked candidate from linked deterministic evidence."""

        if policy.get("interaction_family") == "articulated_handle":
            return self._apply_articulated_handle_candidate_rejection(
                policy=policy,
                active=active,
                rejection=rejection,
            )

        candidates = policy.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        current_rank = int(policy.get("active_rank") or 0)
        rejected = policy.get("rejected_candidates")
        if not isinstance(rejected, list):
            rejected = []
        request_fingerprint = str(rejection.get("request_fingerprint") or "").strip()
        failed_request_fingerprints = list(policy.get("failed_request_fingerprints") or [])
        if request_fingerprint and request_fingerprint in failed_request_fingerprints:
            policy.update(
                {
                    "status": "stopped_requires_human",
                    "active_candidate": None,
                    "active_rank": None,
                    "remaining_candidate_ids": [],
                    "last_rejection": dict(rejection),
                    "stop_reason": "repeated_failed_request_fingerprint",
                }
            )
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source="grasp_fingerprint_repeated",
            )
            self.record(
                "grasp_failed_request_fingerprint_repeated",
                {
                    "candidate_id": active.get("id"),
                    "request_fingerprint": request_fingerprint,
                },
            )
            return True
        if request_fingerprint:
            failed_request_fingerprints.append(request_fingerprint)
        rejected.append(
            {
                "candidate_id": active.get("id"),
                "rank": current_rank,
                "score": active.get("score"),
                "reason": rejection.get("reason"),
                "source": rejection.get("source"),
                "recovery_class": _grasp_estimation_recovery_class(rejection),
                "request_fingerprint": request_fingerprint,
                "rejected_at_s": time.time(),
            }
        )
        attempts = int(policy.get("candidate_attempt_count") or 0) + 1
        max_attempts = int(policy.get("max_candidate_attempts") or GRASP_CANDIDATE_MAX_ATTEMPTS)
        retry_exhausted = attempts >= max_attempts
        next_rank = current_rank + 1
        next_candidate = (
            dict(candidates[next_rank])
            if not retry_exhausted
            and next_rank < len(candidates)
            and isinstance(candidates[next_rank], dict)
            else None
        )
        qualification_scene_changed = (
            isinstance(policy.get("qualification_evidence"), dict)
            and _optional_int(policy.get("scene_epoch"), default=-1) != self.scene_epoch()
        )
        frozen_pool = self.frozen_placement_goal_pool()
        attachment = self.attachment_gate()
        current_state_frontier_retry = (
            str(rejection.get("grasp_stage") or "") == "contact"
            and str(rejection.get("target_tool") or "")
            in {"move_to", "follow_eef_trajectory"}
            and rejection.get("current_state_requalification_safe") is True
            and isinstance(frozen_pool, dict)
            and frozen_pool.get("status") == "ready"
        )
        close_failure_can_rebase_frozen_frontier = (
            str(rejection.get("source") or "") == "host_gripper_close_failed"
            and str(rejection.get("grasp_stage") or "") == "close"
            and not (
                isinstance(attachment, dict)
                and attachment.get("status") == "resolved"
                and str(attachment.get("verdict") or "").upper() == "PASS"
            )
        )
        contact_displacement_can_rebase_frozen_frontier = bool(
            str(rejection.get("source") or "") == "candidate_motion_rejected"
            and str(rejection.get("grasp_stage") or "") == "contact"
            and rejection.get("detached_target_rebase_safe") is True
        )
        target_pose_change_can_rebase_frozen_frontier = bool(
            close_failure_can_rebase_frozen_frontier
            or contact_displacement_can_rebase_frozen_frontier
        )
        frozen_candidate_retry = (
            qualification_scene_changed
            and not target_pose_change_can_rebase_frozen_frontier
            and next_candidate is not None
            and next_candidate.get("grasp_place_joint_qualified") is True
            and isinstance(frozen_pool, dict)
            and frozen_pool.get("status") == "ready"
            and not (
                isinstance(attachment, dict)
                and attachment.get("status") == "resolved"
                and str(attachment.get("verdict") or "").upper() == "PASS"
            )
        )
        qualification_invalidated = qualification_scene_changed and not frozen_candidate_retry
        if current_state_frontier_retry:
            # Physical execution changed the arm start, so every old IK/L5
            # branch is stale even though the object and PlanningScene did not
            # change.  Keep model outputs, but force their qualification to
            # restart from the proved current state.
            qualification_invalidated = True
            frozen_candidate_retry = False
            next_candidate = None
        if qualification_invalidated:
            next_candidate = None
        source_tool = str(policy.get("source_tool") or "grasp_pose_estimate")
        policy.update(
            {
                "status": "active" if next_candidate is not None else "exhausted",
                "active_rank": next_rank if next_candidate is not None else None,
                "active_candidate": next_candidate,
                "remaining_candidate_ids": [
                    str(candidate.get("id"))
                    for candidate in candidates[next_rank + 1 :]
                    if isinstance(candidate, dict)
                ]
                if next_candidate is not None
                else [],
                "rejected_candidates": rejected,
                "failed_request_fingerprints": failed_request_fingerprints,
                "candidate_attempt_count": attempts,
                "last_candidate_attempt": {
                    "candidate_id": active.get("id"),
                    "attempt_count": attempts,
                    "max_attempts": max_attempts,
                    "retry_exhausted": retry_exhausted,
                },
                "last_rejection": dict(rejection),
            }
        )
        if frozen_candidate_retry and next_candidate is not None:
            policy["frozen_model_pool_retry"] = {
                "schema_version": "openeta.frozen_grasp_retry.v1",
                "source_scene_epoch": policy.get("scene_epoch"),
                "current_scene_epoch": self.scene_epoch(),
                "candidate_id": next_candidate.get("id"),
                "model_inference_invoked": False,
                "terminal_pose_reused": True,
                "path_owner": "moveit",
            }
        if next_candidate is None:
            frozen_frontier_remaining = _optional_int(
                policy.get("frozen_grasp_frontier_remaining_count"),
                default=0,
            )
            if target_pose_change_can_rebase_frozen_frontier and not retry_exhausted:
                # Every old IK/L5 proof is pose-bound after a failed close
                # moves the detached target. Include unattempted qualified
                # backups in the frozen requalification count instead of
                # executing their stale terminal poses directly.
                rejected_ids = {
                    str(item.get("candidate_id") or "")
                    for item in rejected
                    if isinstance(item, dict)
                }
                frozen_frontier_remaining = max(
                    frozen_frontier_remaining,
                    sum(
                        1
                        for candidate in candidates
                        if isinstance(candidate, dict)
                        and str(candidate.get("id") or "")
                        not in rejected_ids
                    ),
                )
                policy["frozen_grasp_frontier_remaining_count"] = (
                    frozen_frontier_remaining
                )
            if current_state_frontier_retry and not retry_exhausted:
                rejected_ids = {
                    str(item.get("candidate_id") or "")
                    for item in rejected
                    if isinstance(item, dict)
                }
                frozen_frontier_remaining = max(
                    frozen_frontier_remaining,
                    sum(
                        1
                        for candidate in candidates
                        if isinstance(candidate, dict)
                        and str(candidate.get("id") or "") not in rejected_ids
                    ),
                )
                policy["frozen_grasp_frontier_remaining_count"] = (
                    frozen_frontier_remaining
                )
            policy_revision = _optional_int(policy.get("planning_scene_revision"), default=-1)
            rejection_revision = _optional_int(rejection.get("planning_scene_revision"), default=-2)
            planning_scene_unchanged = (
                policy_revision >= 0 and rejection_revision == policy_revision
            )
            if (
                isinstance(frozen_pool, dict)
                and frozen_pool.get("status") == "ready"
                and frozen_frontier_remaining > 0
                and (
                    not qualification_invalidated
                    or rejection.get("execution_started") is False
                    or planning_scene_unchanged
                    or target_pose_change_can_rebase_frozen_frontier
                )
            ):
                policy.update(
                    {
                        "status": "frozen_frontier_required",
                        "active_rank": None,
                        "active_candidate": None,
                        "remaining_candidate_ids": [],
                        "stop_reason": "frozen_grasp_frontier_expansion_required",
                    }
                )
                if target_pose_change_can_rebase_frozen_frontier:
                    policy["frozen_grasp_frontier_rebase_pending"] = {
                        "schema_version": ("openeta.frozen_grasp_frontier_rebase_pending.v1"),
                        "reason_code": (
                            "FAILED_CONTACT_TARGET_POSE_SYNC_REQUIRED"
                            if contact_displacement_can_rebase_frozen_frontier
                            else "FAILED_CLOSE_TARGET_POSE_SYNC_REQUIRED"
                        ),
                        "source_planning_scene_revision": (
                            rejection.get("source_planning_scene_revision")
                            if contact_displacement_can_rebase_frozen_frontier
                            else policy_revision
                        ),
                        "physically_rejected_candidate_id": str(
                            active.get("id") or ""
                        ),
                        "required_proof": (
                            "detached_native_target_pose_sync_with_unchanged_static_scene"
                        ),
                        "model_inference_invoked": False,
                    }
                if current_state_frontier_retry:
                    policy["frozen_grasp_frontier_current_state_pending"] = {
                        "schema_version": (
                            "openeta.frozen_grasp_frontier_current_state.v1"
                        ),
                        "reason_code": (
                            "FAILED_CONTACT_CURRENT_STATE_REQUALIFICATION"
                        ),
                        "source_planning_scene_revision": policy_revision,
                        "physically_rejected_candidate_id": str(
                            active.get("id") or ""
                        ),
                        "current_state_restart_sha256": rejection.get(
                            "current_state_restart_sha256"
                        ),
                        "model_inference_invoked": False,
                        "exact_anchor_restoration": False,
                    }
                    policy.pop("host_candidate_compilations", None)
                    policy.pop("qualification_evidence", None)
                    policy.pop("frozen_model_pool_retry", None)
                    policy.pop("accepted_candidate", None)
                    policy.pop("accepted_at_s", None)
                    self.facts.pop(GRASP_EXECUTION_KEY, None)
                    self.facts.pop(GRASP_RECOVERY_KEY, None)
                policy.pop("reestimate_required", None)
                self.facts.pop(GRASP_REESTIMATION_KEY, None)
                self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                    policy,
                    source="frozen_grasp_frontier",
                )
                self.record(
                    "frozen_grasp_frontier_expansion_required",
                    {
                        "remaining_count": frozen_frontier_remaining,
                        "generation": policy.get("frozen_grasp_frontier_generation"),
                        "model_inference_invoked": False,
                        "planning_scene_revision": policy_revision,
                    },
                )
                return True
            # Once the bounded candidate budget is exhausted, keep the highest
            # scoring candidate only when compilation rejected a strategy-level
            # geometry preference. Motion, safety, and structured perception
            # rejections must continue through recovery instead of reviving a pose
            # that was already shown to be unusable.
            fallback_candidate = (
                dict(candidates[0]) if candidates and isinstance(candidates[0], dict) else None
            )
            recovery_class = _grasp_estimation_recovery_class(rejection)
            score_fallback_allowed = (
                str(rejection.get("source") or "") == "grasp_seed_geometry_rejected"
                and recovery_class == "none"
                and policy.get("final_refinable_fallback") is not True
            )
            if (
                fallback_candidate is not None
                and policy.get("candidate_fallback") is not True
                and score_fallback_allowed
            ):
                fallback_candidate["candidate_fallback"] = True
                policy.update(
                    {
                        "status": "active",
                        "active_rank": int(fallback_candidate.get("rank") or 0),
                        "active_candidate": fallback_candidate,
                        "remaining_candidate_ids": [],
                        "candidate_fallback": True,
                        "fallback_reason": "all_ranked_candidates_failed",
                        "fallback_selected_at_s": time.time(),
                    }
                )
                policy.pop("reestimate_required", None)
                self.facts.pop(GRASP_REESTIMATION_KEY, None)
                self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                    policy,
                    source=f"{source_tool}_score_fallback",
                )
                self.record(
                    f"{source_tool}_candidate_fallback_activated",
                    {
                        "result_id": policy.get("result_id"),
                        "candidate_id": fallback_candidate.get("id"),
                        "rank": fallback_candidate.get("rank"),
                        "score": fallback_candidate.get("score"),
                        "attempt_count": attempts,
                        "max_attempts": max_attempts,
                        "reason": "all_ranked_candidates_failed",
                        "alignment_filter_bypassed": True,
                    },
                )
                return True
            if policy.get("final_refinable_fallback") is True:
                policy.pop("reestimate_required", None)
                self.facts.pop(GRASP_REESTIMATION_KEY, None)
                self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                    policy,
                    source=f"{source_tool}_final_fallback_exhausted",
                )
                self.record(
                    f"{source_tool}_final_fallback_exhausted",
                    {
                        "result_id": policy.get("result_id"),
                        "candidate_id": active.get("id"),
                        "attempt_count": attempts,
                        "reason": rejection.get("reason"),
                    },
                )
                return True
            frozen_pool_exhausted = (
                isinstance(frozen_pool, dict)
                and frozen_pool.get("status") == "ready"
            )
            reestimate_reason = (
                "moveit_qualification_scene_changed"
                if qualification_invalidated
                else (
                    "frozen_grasp_pool_exhausted"
                    if frozen_pool_exhausted
                    else "candidate_retry_limit_exceeded"
                    if retry_exhausted
                    else "ranked_candidate_queue_exhausted"
                )
            )
            policy["reestimate_required"] = {
                "status": "pending_recovery",
                "reason": reestimate_reason,
                "candidate_id": active.get("id"),
                "attempt_count": attempts,
                "max_attempts": max_attempts,
                "scene_epoch": self.scene_epoch(),
                "created_at_s": time.time(),
                "target_prompt": ((policy.get("target_detection") or {}).get("target_prompt")),
                "source_image": ((policy.get("target_detection") or {}).get("source_image")),
                "invalidate_frozen_placement_pool": frozen_pool_exhausted,
            }
            if frozen_pool_exhausted:
                policy["model_inference_retry_allowed"] = True
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source=f"{source_tool}_reestimate_required",
            )
            self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
                dict(policy["reestimate_required"]),
                source=f"{source_tool}_reestimate_required",
            )
            self.record(
                f"{source_tool}_reestimate_required",
                dict(policy["reestimate_required"]),
            )
            return True
        policy.pop("accepted_candidate", None)
        policy.pop("accepted_at_s", None)
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source=f"{source_tool}_greedy_fallback",
        )
        self.record(
            f"{source_tool}_candidate_rejected",
            {
                "result_id": policy.get("result_id"),
                "candidate_id": active.get("id"),
                "rank": current_rank,
                "reason": rejection.get("reason"),
                "source": rejection.get("source"),
                "next_candidate_id": (
                    next_candidate.get("id") if isinstance(next_candidate, dict) else None
                ),
                "exhausted": next_candidate is None,
            },
        )
        if next_candidate is not None:
            self.record(
                f"{source_tool}_candidate_activated",
                {
                    "result_id": policy.get("result_id"),
                    "candidate_id": next_candidate.get("id"),
                    "rank": next_rank,
                    "score": next_candidate.get("score"),
                    "activation_source": "previous_candidate_rejected",
                },
            )
        return True

    def _apply_articulated_handle_candidate_rejection(
        self,
        *,
        policy: JsonDict,
        active: JsonDict,
        rejection: JsonDict,
    ) -> bool:
        """Advance bounded top/front/side queues before one score fallback."""

        candidates_value = policy.get("candidates")
        candidates = (
            [dict(candidate) for candidate in candidates_value if isinstance(candidate, dict)]
            if isinstance(candidates_value, list)
            else []
        )
        by_id = {str(candidate.get("id") or ""): candidate for candidate in candidates}
        active_id = str(active.get("id") or "")
        active_mode = str(policy.get("active_approach_mode") or active.get("approach_mode") or "")
        rejected = policy.get("rejected_candidates")
        if not isinstance(rejected, list):
            rejected = []
        rejected.append(
            {
                "candidate_id": active_id,
                "rank": active.get("rank"),
                "score": active.get("score"),
                "approach_mode": active_mode,
                "reason": rejection.get("reason"),
                "source": rejection.get("source"),
                "recovery_class": _grasp_estimation_recovery_class(rejection),
                "rejected_at_s": time.time(),
            }
        )
        attempts = int(policy.get("candidate_attempt_count") or 0) + 1
        mode_attempts = int(policy.get("mode_attempt_count") or 0) + 1
        max_mode_attempts = int(
            policy.get("mode_max_candidate_attempts") or GRASP_CANDIDATE_MAX_ATTEMPTS
        )
        policy.update(
            {
                "rejected_candidates": rejected,
                "candidate_attempt_count": attempts,
                "mode_attempt_count": mode_attempts,
                "last_candidate_attempt": {
                    "candidate_id": active_id,
                    "attempt_count": attempts,
                    "mode_attempt_count": mode_attempts,
                    "max_mode_attempts": max_mode_attempts,
                    "approach_mode": active_mode,
                },
                "last_rejection": dict(rejection),
            }
        )
        source_tool = str(policy.get("source_tool") or "grasp_pose_estimate")
        if policy.get("candidate_fallback") is True:
            return self._mark_articulated_handle_reestimate_required(
                policy=policy,
                active=active,
                attempts=attempts,
                source_tool=source_tool,
            )

        queues = policy.get("approach_mode_queues")
        mode_order = policy.get("approach_mode_order")
        if not isinstance(queues, dict) or not isinstance(mode_order, list):
            return self._mark_articulated_handle_reestimate_required(
                policy=policy,
                active=active,
                attempts=attempts,
                source_tool=source_tool,
            )
        queue = queues.get(active_mode)
        queue = [str(value) for value in queue] if isinstance(queue, list) else []
        current_mode_rank = int(policy.get("active_mode_rank") or 0)
        next_mode_rank = current_mode_rank + 1
        next_candidate: JsonDict | None = None
        next_mode = active_mode
        activation_source = "previous_candidate_rejected"
        if mode_attempts < max_mode_attempts and next_mode_rank < len(queue):
            next_candidate = by_id.get(queue[next_mode_rank])
        else:
            try:
                active_mode_index = [str(value) for value in mode_order].index(active_mode)
            except ValueError:
                active_mode_index = -1
            for value in mode_order[active_mode_index + 1 :]:
                candidate_ids = queues.get(value)
                if not isinstance(candidate_ids, list) or not candidate_ids:
                    continue
                candidate = by_id.get(str(candidate_ids[0]))
                if candidate is None:
                    continue
                next_mode = str(value)
                next_mode_rank = 0
                next_candidate = candidate
                mode_attempts = 0
                activation_source = "approach_mode_degraded"
                break

        if next_candidate is not None:
            next_candidate = dict(next_candidate)
            next_queue = queues.get(next_mode)
            next_queue = (
                [str(value) for value in next_queue] if isinstance(next_queue, list) else []
            )
            policy.update(
                {
                    "status": "active",
                    "active_approach_mode": next_mode,
                    "active_mode_rank": next_mode_rank,
                    "active_rank": int(next_candidate.get("rank") or 0),
                    "active_candidate": next_candidate,
                    "remaining_candidate_ids": next_queue[next_mode_rank + 1 :],
                    "mode_attempt_count": mode_attempts,
                    "compile_hints": {
                        "target_geometry_family": "articulated_handle",
                        "approach_mode": next_mode,
                    },
                }
            )
            policy.pop("accepted_candidate", None)
            policy.pop("accepted_at_s", None)
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source=f"{source_tool}_articulated_handle_fallback",
            )
            self.record(
                f"{source_tool}_candidate_rejected",
                {
                    "result_id": policy.get("result_id"),
                    "candidate_id": active_id,
                    "rank": active.get("rank"),
                    "approach_mode": active_mode,
                    "reason": rejection.get("reason"),
                    "source": rejection.get("source"),
                    "next_candidate_id": next_candidate.get("id"),
                    "next_approach_mode": next_mode,
                    "activation_source": activation_source,
                },
            )
            self.record(
                f"{source_tool}_candidate_activated",
                {
                    "result_id": policy.get("result_id"),
                    "candidate_id": next_candidate.get("id"),
                    "rank": next_candidate.get("rank"),
                    "score": next_candidate.get("score"),
                    "approach_mode": next_mode,
                    "activation_source": activation_source,
                },
            )
            return True

        fallback_candidate = dict(candidates[0]) if candidates else None
        if fallback_candidate is not None:
            fallback_mode = str(fallback_candidate.get("approach_mode") or "side")
            fallback_candidate.update(
                {
                    "candidate_fallback": True,
                    "fallback_reason": "all_approach_modes_failed",
                }
            )
            policy.update(
                {
                    "status": "active",
                    "active_approach_mode": fallback_mode,
                    "active_mode_rank": None,
                    "active_rank": int(fallback_candidate.get("rank") or 0),
                    "active_candidate": fallback_candidate,
                    "remaining_candidate_ids": [],
                    "candidate_fallback": True,
                    "fallback_reason": "all_approach_modes_failed",
                    "fallback_selected_at_s": time.time(),
                    "compile_hints": {
                        "target_geometry_family": "articulated_handle",
                        "approach_mode": fallback_mode,
                        "candidate_fallback": True,
                        "fallback_reason": "all_approach_modes_failed",
                    },
                }
            )
            policy.pop("reestimate_required", None)
            self.facts.pop(GRASP_REESTIMATION_KEY, None)
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source=f"{source_tool}_articulated_handle_score_fallback",
            )
            self.record(
                f"{source_tool}_candidate_fallback_activated",
                {
                    "result_id": policy.get("result_id"),
                    "candidate_id": fallback_candidate.get("id"),
                    "rank": fallback_candidate.get("rank"),
                    "score": fallback_candidate.get("score"),
                    "reason": "all_approach_modes_failed",
                    "approach_mode": fallback_mode,
                    "alignment_filter_bypassed": True,
                },
            )
            return True
        return self._mark_articulated_handle_reestimate_required(
            policy=policy,
            active=active,
            attempts=attempts,
            source_tool=source_tool,
        )

    def _mark_articulated_handle_reestimate_required(
        self,
        *,
        policy: JsonDict,
        active: JsonDict,
        attempts: int,
        source_tool: str,
    ) -> bool:
        policy["status"] = "exhausted"
        policy["active_candidate"] = None
        policy["active_rank"] = None
        policy["remaining_candidate_ids"] = []
        policy["reestimate_required"] = {
            "status": "pending_recovery",
            "reason": "articulated_handle_fallback_failed",
            "candidate_id": active.get("id"),
            "attempt_count": attempts,
            "scene_epoch": self.scene_epoch(),
            "created_at_s": time.time(),
            "target_prompt": ((policy.get("target_detection") or {}).get("target_prompt")),
            "source_image": ((policy.get("target_detection") or {}).get("source_image")),
        }
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source=f"{source_tool}_reestimate_required",
        )
        self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
            dict(policy["reestimate_required"]),
            source=f"{source_tool}_reestimate_required",
        )
        self.record(
            f"{source_tool}_reestimate_required",
            dict(policy["reestimate_required"]),
        )
        return True

    def _capture_compiled_grasp(self, action: EnvAction) -> bool:
        placement_call = _successful_tool_call(action, "compile_placement_seed")
        if placement_call is not None:
            outputs = _tool_call_outputs(placement_call)
            if outputs.get("schema_version") == "openeta.compiled_placement_seed.v3":
                return self._capture_compiled_placement(outputs, source="legacy_compile_tool")
        placement_outputs = _host_candidate_compilation_outputs(action, purpose="placement")
        if placement_outputs is not None:
            return self._capture_compiled_placement(
                placement_outputs, source="host_qualified_queue"
            )
        call = _successful_tool_call(action, "compile_grasp_seed")
        outputs = _tool_call_outputs(call) if call is not None else None
        compilation_source = "legacy_compile_tool"
        if not isinstance(outputs, dict):
            outputs = _host_candidate_compilation_outputs(action, purpose="grasp")
            compilation_source = "host_qualified_queue"
        if not isinstance(outputs, dict):
            return False
        if outputs.get("schema_version") != "openeta.compiled_grasp_seed.v2":
            return False
        policy = self.anygrasp_candidate_policy() or {}
        active = policy.get("active_candidate")
        candidate_id = str(outputs.get("candidate_id") or "")
        if policy.get("status") == "selection_required":
            active = next(
                (
                    candidate
                    for candidate in policy.get("candidates", [])
                    if isinstance(candidate, dict)
                    and str(candidate.get("id") or "") == candidate_id
                ),
                None,
            )
            if isinstance(active, dict):
                policy.update(
                    {
                        "status": "active",
                        "active_candidate": active,
                        "active_rank": active.get("rank"),
                        "selection_source": str(
                            outputs.get("selection_source") or compilation_source
                        ),
                        "remaining_candidate_ids": [],
                    }
                )
                self.facts.pop(LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY, None)
                self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                    policy, source=compilation_source
                )
        if not isinstance(active, dict):
            return False
        if candidate_id != str(active.get("id") or ""):
            return False
        if _optional_int(outputs.get("scene_epoch"), default=-1) != self.scene_epoch():
            return False
        policy["selection_source"] = str(outputs.get("selection_source") or compilation_source)
        self.facts.pop(LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY, None)
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source=compilation_source,
        )
        compile_hints: JsonDict = {
            field: outputs.get(field)
            for field in (
                "target_geometry_family",
                "approach_mode",
                "fallback_reason",
            )
            if outputs.get(field) not in (None, "")
        }
        if outputs.get("candidate_fallback") is True:
            compile_hints["candidate_fallback"] = True
        if compile_hints:
            policy["compile_hints"] = compile_hints
            self.facts.pop(LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY, None)
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source=compilation_source,
            )
        contact = outputs.get("contact_pose")
        gripper_state = self.gripper_command_state() or {}
        observed_gripper_state = self._latest_observed_gripper_state() or {}
        last_rejection = policy.get("last_rejection")
        failed_close_requires_open = (
            isinstance(last_rejection, dict)
            and last_rejection.get("source") == "host_gripper_close_failed"
        )
        open_state_source = (
            "acknowledged_gripper_command"
            if gripper_state.get("position") == 1
            else (
                "current_epoch_physical_observation"
                if observed_gripper_state.get("open") is True
                else "none"
            )
        )
        already_open = open_state_source != "none" and not failed_close_requires_open
        if already_open and isinstance(contact, dict):
            initial_stage = "contact"
            required_action = {
                "name": "move_to",
                "parameters": {"target_pose": _pose_for_epoch(contact, self.scene_epoch())},
            }
        else:
            initial_stage = "open"
            required_action = {
                "name": "gripper_control",
                "parameters": {"position": 1},
            }
        execution = {
            "schema_version": "openeta.grasp_execution.v2",
            "status": "required",
            "stage": initial_stage,
            "candidate_id": candidate_id,
            "compiled_grasp_id": outputs.get("compiled_grasp_id"),
            "scene_epoch": self.scene_epoch(),
            "compiled_grasp": dict(outputs),
            "source_eef_pose": self._latest_observed_end_effector_pose(),
            "initial_gripper_open_proof": {
                "source": open_state_source,
                "scene_epoch": self.scene_epoch(),
                **(
                    {"observed_state": observed_gripper_state}
                    if open_state_source == "current_epoch_physical_observation"
                    else {}
                ),
            },
            "required_action": required_action,
            "transition_conditions": {
                "contact": (
                    "MoveIt plans the complete collision-aware path from the current "
                    "joint state to the unmodified model terminal contact pose"
                ),
                "close": (
                    "acknowledge binary gripper position=0; this confirms only the "
                    "latched command, never object attachment"
                ),
            },
            "created_at_s": time.time(),
        }
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source=compilation_source,
        )
        self.facts.pop(ATTACHMENT_GATE_KEY, None)
        self.facts.pop(ARTICULATED_ATTACHMENT_PROBE_KEY, None)
        self.record(
            "grasp_execution_started",
            {
                "candidate_id": candidate_id,
                "compiled_grasp_id": outputs.get("compiled_grasp_id"),
                "calibration_status": outputs.get("calibration_status"),
                "scene_epoch": self.scene_epoch(),
            },
        )
        return True

    def _activate_host_compiled_grasp_for_active_candidate(self) -> bool:
        """Project the next retained queue entry into normal grasp execution."""

        policy = self.grasp_candidate_policy()
        if not isinstance(policy, dict) or policy.get("status") != "active":
            return False
        active = policy.get("active_candidate")
        candidate_id = str(active.get("id") or "") if isinstance(active, dict) else ""
        execution = self.grasp_execution()
        if isinstance(execution, dict) and str(execution.get("candidate_id") or "") == candidate_id:
            return False
        compiled_by_id = policy.get("host_candidate_compilations")
        compiled = compiled_by_id.get(candidate_id) if isinstance(compiled_by_id, dict) else None
        if not isinstance(compiled, dict):
            return False
        compiled = json.loads(json.dumps(compiled))
        compiled_scene_epoch = _optional_int(compiled.get("scene_epoch"), default=-1)
        if compiled_scene_epoch != self.scene_epoch():
            frozen_pool = self.frozen_placement_goal_pool()
            if not (
                isinstance(active, dict)
                and active.get("grasp_place_joint_qualified") is True
                and isinstance(frozen_pool, dict)
                and frozen_pool.get("status") == "ready"
            ):
                return False
            contact = compiled.get("contact_pose")
            if not isinstance(contact, dict):
                return False
            rebound_contact = dict(contact)
            rebound_contact["scene_epoch"] = self.scene_epoch()
            compiled.update(
                {
                    "scene_epoch": self.scene_epoch(),
                    "contact_pose": rebound_contact,
                    "selection_source": "host_frozen_qualified_queue_retry",
                    "frozen_model_pool_reuse": {
                        "schema_version": "openeta.frozen_grasp_retry.v1",
                        "source_scene_epoch": compiled_scene_epoch,
                        "current_scene_epoch": self.scene_epoch(),
                        "model_inference_invoked": False,
                        "terminal_pose_reused": True,
                        "path_owner": "moveit",
                    },
                }
            )
        return self._capture_compiled_grasp(
            _host_compilation_action(purpose="grasp", compiled=compiled)
        )

    def _capture_placement_candidates(self, action: EnvAction) -> bool:
        call = _successful_tool_call(action, "anyplace")
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        previous_policy = self.placement_candidate_policy()
        if outputs.get("frozen_goal_pool_ready") is True:
            pool = {
                "schema_version": "openeta.frozen_placement_goal_pool.v1",
                "status": "ready",
                "goal_count": _optional_int(outputs.get("frozen_goal_pool_count"), default=0),
                "scene_epoch": self.scene_epoch(),
                "execution_started": False,
                "visibility": "host_private",
            }
            self.facts[FROZEN_PLACEMENT_POOL_KEY] = _memory_fact_entry(pool, source="anyplace")
            self.record("frozen_placement_goal_pool_ready", dict(pool))
            return True
        private_qualification = _qualification_artifact_evidence(
            outputs, artifact_key="qualification_artifact"
        )
        candidates = outputs.get("placement_candidates")
        if not isinstance(candidates, list) or not candidates:
            if isinstance(outputs.get("qualification_evidence"), dict):
                if self._schedule_attached_place_frontier_recovery(
                    outputs=outputs,
                    previous_policy=previous_policy,
                    private_qualification=private_qualification,
                ):
                    return True
                zero_pass_resume = (
                    isinstance(previous_policy, dict)
                    and previous_policy.get("status") == "frozen_frontier_required"
                    and outputs.get("frozen_goal_frontier_resume") is True
                )
                prior_rejected = (
                    list(previous_policy.get("rejected_candidates") or [])
                    if zero_pass_resume
                    else []
                )
                prior_fingerprints = (
                    list(previous_policy.get("failed_request_fingerprints") or [])
                    if zero_pass_resume
                    else []
                )
                policy = {
                    "schema_version": "openeta.placement_candidate_policy.v2",
                    "status": "stopped_requires_human",
                    "candidate_queue": [],
                    "active_candidate_id": None,
                    # Exact failed IDs, seeds, poses, and collision evidence
                    # remain in the host-only qualification artifact. There is
                    # no selectable queue in a zero-PASS round, so exposing
                    # those entries to planner memory has no control purpose.
                    "rejected_candidates": prior_rejected,
                    "rejected_candidate_count": len(prior_rejected)
                    + len(private_qualification.get("results", [])),
                    "failed_request_fingerprints": prior_fingerprints,
                    "scene_revision": outputs.get("scene_revision"),
                    "planning_scene_revision": outputs.get("scene_revision"),
                    "scene_epoch": self.scene_epoch(),
                    "selection_source": None,
                    "stop_reason": "CURRENT_GRASP_PLACE_INFEASIBLE",
                    "frozen_goal_requalification": outputs.get("frozen_goal_requalification")
                    is True,
                    "frozen_goal_frontier_exhausted": True,
                    "recovery": {
                        "stage": "manual_intervention",
                        "required_action": None,
                    },
                }
                self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                    policy, source="moveit_qualification"
                )
                self.record("placement_candidates_moveit_rejected", dict(policy))
                return True
            return False
        queue = [
            str(candidate.get("id") or "")
            for candidate in candidates
            if isinstance(candidate, dict) and str(candidate.get("id") or "")
        ]
        if not queue:
            return False
        attachment = self.attachment_gate()
        if (
            not isinstance(attachment, dict)
            or attachment.get("status") != "resolved"
            or str(attachment.get("verdict") or "").upper() != "PASS"
        ):
            return False
        attachment_proof = attachment.get("attachment_proof")
        attachment_transform = (
            attachment_proof.get("attachment_transform")
            if isinstance(attachment_proof, dict)
            else None
        )
        if not _valid_attachment_transform(attachment_transform):
            return False
        native_revision = attachment.get("evidence_source") == "gazebo_native_contact_attach_ack"
        if native_revision and not isinstance(attachment.get("planning_scene_revision"), int):
            return False
        latest_receipt = self.latest_environment_receipt() or {}
        scene_revision = (
            int(attachment["planning_scene_revision"])
            if native_revision and isinstance(attachment.get("planning_scene_revision"), int)
            else _optional_int(
                outputs.get(
                    "scene_revision",
                    latest_receipt.get("scene_revision", self.scene_epoch()),
                ),
                default=self.scene_epoch(),
            )
        )
        if native_revision and (
            _optional_int(outputs.get("scene_revision"), default=-1) != scene_revision
        ):
            return False
        qualification_evidence = outputs.get("qualification_evidence")
        qualification_scene_epoch = _optional_int(
            qualification_evidence.get("scene_epoch")
            if isinstance(qualification_evidence, dict)
            else None,
            default=-1,
        )
        if qualification_scene_epoch < 0:
            # The proof's compiled poses are bound to this simulator epoch;
            # never substitute the internal action counter for it.
            return False
        resuming_frontier = (
            isinstance(previous_policy, dict)
            and previous_policy.get("status") == "frozen_frontier_required"
            and outputs.get("frozen_goal_frontier_resume") is True
        )
        prior_rejected = (
            list(previous_policy.get("rejected_candidates") or []) if resuming_frontier else []
        )
        prior_fingerprints = (
            list(previous_policy.get("failed_request_fingerprints") or [])
            if resuming_frontier
            else []
        )
        policy = {
            "schema_version": "openeta.placement_candidate_policy.v2",
            "status": "selection_required",
            "candidate_queue": queue,
            "active_candidate_id": None,
            "rejected_candidates": prior_rejected,
            "failed_request_fingerprints": prior_fingerprints,
            "scene_revision": scene_revision,
            "planning_scene_revision": scene_revision,
            "revision_provenance": ("native_attachment_gate" if native_revision else "scene_epoch"),
            "scene_epoch": qualification_scene_epoch,
            "selection_source": None,
            "frozen_goal_requalification": outputs.get("frozen_goal_requalification") is True,
            "frozen_goal_frontier_count": _optional_int(
                outputs.get("frozen_goal_frontier_count"), default=len(queue)
            ),
            "frozen_goal_total_eligible_count": _optional_int(
                outputs.get("frozen_goal_total_eligible_count"),
                default=len(queue),
            ),
            "frozen_goal_frontier_generation": _optional_int(
                outputs.get("frozen_goal_frontier_generation"), default=0
            ),
            "attachment_transform_sha256": hashlib.sha256(
                json.dumps(attachment_transform, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
        }
        compilation_queue = outputs.get("host_candidate_compilation_queue")
        if isinstance(compilation_queue, list):
            policy["host_candidate_compilations"] = {
                str(event.get("candidate_id") or ""): dict(event["compiled_seed"])
                for event in compilation_queue
                if isinstance(event, dict)
                and isinstance(event.get("compiled_seed"), dict)
                and str(event.get("candidate_id") or "")
                == str(
                    event["compiled_seed"].get("placement_candidate_id")
                    or event["compiled_seed"].get("candidate_id")
                    or ""
                )
            }
        self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(policy, source="anyplace")
        self.record("placement_candidates_retained", dict(policy))
        return True

    def _schedule_attached_place_frontier_recovery(
        self,
        *,
        outputs: Mapping[str, Any],
        previous_policy: JsonDict | None,
        private_qualification: Mapping[str, Any],
    ) -> bool:
        """Continue a frozen grasp frontier after a pristine attached-place miss.

        A measured attachment can differ enough from its pre-bind prediction
        that every frozen AnyPlace target becomes infeasible.  When no
        placement motion has started, the arm is still at the original grasp
        terminal.  It is then safe to reopen, prove the detached object's
        current world pose, and requalify the next *already frozen* model
        grasp branch.  This is deliberately narrower than generic placement
        recovery: after a placement motion attempt the object may be anywhere
        along a transport path, so this automatic reopen is no longer safe.
        """

        if (
            outputs.get("frozen_goal_requalification") is not True
            or isinstance(previous_policy, dict)
        ):
            return False
        frozen_pool = self.frozen_placement_goal_pool()
        policy = self.grasp_candidate_policy()
        execution = self.grasp_execution()
        attachment = self.attachment_gate()
        if not (
            isinstance(frozen_pool, dict)
            and frozen_pool.get("status") == "ready"
            and isinstance(policy, dict)
            and str(policy.get("status") or "") in {"active", "accepted"}
            and isinstance(execution, dict)
            and execution.get("status") == "completed"
            and execution.get("stage") == "attached"
            and execution.get("attachment_mode") != "articulated_handle"
            and isinstance(attachment, dict)
            and attachment.get("status") == "resolved"
            and str(attachment.get("verdict") or "").upper() == "PASS"
        ):
            return False
        candidate_id = str(execution.get("candidate_id") or "")
        if not candidate_id:
            return False
        active = policy.get("active_candidate")
        accepted = policy.get("accepted_candidate")
        policy_candidate_id = str(
            (active if isinstance(active, dict) else accepted).get("id") or ""
        ) if isinstance(active, dict) or isinstance(accepted, dict) else ""
        if policy_candidate_id and policy_candidate_id != candidate_id:
            return False
        if str(attachment.get("candidate_id") or "") not in {"", candidate_id}:
            return False
        remaining = _optional_int(
            policy.get("frozen_grasp_frontier_remaining_count"), default=0
        )
        source_revision = _optional_int(
            policy.get("planning_scene_revision"), default=-1
        )
        attachment_revision = _optional_int(
            attachment.get("planning_scene_revision"), default=-1
        )
        output_revision = _optional_int(outputs.get("scene_revision"), default=-1)
        if (
            remaining <= 0
            or source_revision < 0
            or attachment_revision < 0
            or output_revision != attachment_revision
        ):
            return False

        rejection = {
            "candidate_id": candidate_id,
            "reason": (
                "all frozen placement goals are infeasible for the measured "
                "attachment before any placement motion began"
            ),
            "source": "attached_place_frontier_exhausted",
            "frozen_goal_requalification": True,
            "rejected_goal_count": len(private_qualification.get("results", [])),
            "attachment_planning_scene_revision": attachment_revision,
            "source_planning_scene_revision": source_revision,
        }
        rejected = [
            dict(item)
            for item in policy.get("rejected_candidates", [])
            if isinstance(item, dict)
        ]
        if candidate_id not in {
            str(item.get("candidate_id") or "") for item in rejected
        }:
            rejected.append(rejection)
        policy.update(
            {
                "status": "frozen_frontier_required",
                "active_rank": None,
                "active_candidate": None,
                "remaining_candidate_ids": [],
                "rejected_candidates": rejected,
                "stop_reason": "frozen_grasp_frontier_recovery_required",
                "frozen_grasp_frontier_rebase_pending": {
                    "schema_version": "openeta.frozen_grasp_frontier_rebase_pending.v1",
                    "reason_code": "ATTACHED_PLACE_FRONTIER_EXHAUSTED",
                    "source_planning_scene_revision": source_revision,
                    "physically_rejected_candidate_id": candidate_id,
                    "required_proof": (
                        "detached_native_target_pose_sync_with_unchanged_static_scene"
                    ),
                    "model_inference_invoked": False,
                },
            }
        )
        placement_policy = {
            "schema_version": "openeta.placement_candidate_policy.v2",
            "status": "frozen_grasp_frontier_recovery_required",
            "candidate_queue": [],
            "active_candidate_id": None,
            "rejected_candidates": [],
            "rejected_candidate_count": len(private_qualification.get("results", [])),
            "failed_request_fingerprints": [],
            "scene_revision": attachment_revision,
            "planning_scene_revision": attachment_revision,
            "scene_epoch": self.scene_epoch(),
            "selection_source": None,
            "stop_reason": "CURRENT_GRASP_PLACE_INFEASIBLE",
            "frozen_goal_requalification": True,
            "frozen_goal_frontier_exhausted": True,
            "recovery": {
                "stage": "frozen_grasp_frontier_reopen",
                "required_action": {
                    "name": "gripper_control",
                    "parameters": {
                        "position": 1,
                        "recovery_intent": "frozen_grasp_frontier",
                    },
                },
            },
        }
        recovery = {
            "schema_version": "openeta.grasp_recovery.v2",
            "status": "required",
            "recovery_id": f"attached-place-frontier-{uuid4()}",
            "candidate_id": candidate_id,
            "rejection_source": "attached_place_frontier_exhausted",
            "rejection_reason": rejection["reason"],
            "purpose": "attached_place_frontier_recovery",
            "scene_epoch": self.scene_epoch(),
            "stage": "reopen",
            "reopen_required": True,
            "restore_required": False,
            "observe_after_reopen": False,
            "required_action": {
                "name": "gripper_control",
                "parameters": {
                    "position": 1,
                    "recovery_intent": "frozen_grasp_frontier",
                },
            },
            "frontier": {
                "remaining_candidate_count": remaining,
                "source_planning_scene_revision": source_revision,
                "attachment_planning_scene_revision": attachment_revision,
                "model_inference_invoked": False,
            },
            "created_at_s": time.time(),
        }
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy, source="attached_place_frontier_recovery"
        )
        self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            placement_policy, source="attached_place_frontier_recovery"
        )
        self.facts[GRASP_RECOVERY_KEY] = _memory_fact_entry(
            recovery, source="attached_place_frontier_recovery"
        )
        self.record("attached_place_frontier_recovery_required", {
            "candidate_id": candidate_id,
            "remaining_candidate_count": remaining,
            "rejected_goal_count": placement_policy["rejected_candidate_count"],
            "model_inference_invoked": False,
        })
        return True

    def _capture_compiled_placement(
        self, outputs: JsonDict, *, source: str = "legacy_compile_tool"
    ) -> bool:
        policy = self.placement_candidate_policy()
        if not isinstance(policy, dict):
            return False
        candidate_id = str(outputs.get("placement_candidate_id") or "")
        rejected = {
            str(item.get("candidate_id") or "")
            for item in policy.get("rejected_candidates", [])
            if isinstance(item, dict)
        }
        if (
            policy.get("status") != "selection_required"
            or candidate_id not in policy.get("candidate_queue", [])
            or candidate_id in rejected
            # The compiled seed must match the epoch captured by the retained
            # qualification proof.  ``self.scene_epoch()`` may already have
            # advanced while attaching the object (which also changes the
            # planning-scene revision) without invalidating this proof.
            or _optional_int(outputs.get("scene_epoch"), default=-1)
            != _optional_int(policy.get("scene_epoch"), default=-1)
            or _optional_int(outputs.get("scene_revision"), default=-1)
            != _optional_int(policy.get("scene_revision"), default=-2)
        ):
            return False
        policy.update(
            {
                "status": "active",
                "active_candidate_id": candidate_id,
                "selection_source": str(outputs.get("selection_source") or source),
                "compiled_placement": dict(outputs),
            }
        )
        self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(policy, source=source)
        self.record(
            "placement_candidate_selected",
            {
                "placement_candidate_id": candidate_id,
                "attachment_transform_sha256": outputs.get("attachment_transform_sha256"),
                "scene_revision": policy.get("scene_revision"),
                "selection_source": policy.get("selection_source"),
            },
        )
        return True

    def _advance_placement_candidate_after_motion(self, action: EnvAction) -> bool:
        policy = self.placement_candidate_policy()
        if not isinstance(policy, dict) or policy.get("status") != "active":
            return False
        call = _tool_call(action, "move_to")
        if not isinstance(call, dict):
            return False
        request = call.get("parameters")
        if not isinstance(request, dict):
            request = call.get("request")
        target = request.get("target_pose") if isinstance(request, dict) else None
        if not isinstance(target, dict) or target.get("purpose") != "placement":
            return False
        result = call.get("result")
        details = result.get("details") if isinstance(result, dict) else None
        details = details if isinstance(details, dict) else {}
        outputs = details.get("outputs")
        outputs = outputs if isinstance(outputs, dict) else {}
        receipt = details.get("environment_receipt")
        if not isinstance(receipt, dict):
            receipt = outputs.get("response")
        receipt = receipt if isinstance(receipt, dict) else outputs
        candidate_id = str(policy.get("active_candidate_id") or "")
        requested_candidate_id = str(target.get("placement_candidate_id") or "")
        if not candidate_id or requested_candidate_id != candidate_id:
            policy.update(
                {
                    "status": "stopped_requires_human",
                    "stop_reason": "placement_candidate_identity_mismatch",
                }
            )
            self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy, source="placement_candidate_identity_mismatch"
            )
            return True
        expected_revision = _optional_int(policy.get("scene_revision"), default=-1)
        requested_revision = _optional_int(target.get("scene_revision"), default=-2)
        receipt_revision = _optional_int(receipt.get("planning_scene_revision"), default=-3)
        if policy.get("revision_provenance") == "native_attachment_gate" and (
            expected_revision < 0
            or requested_revision != expected_revision
            or receipt_revision != expected_revision
        ):
            policy.update(
                {
                    "status": "stopped_requires_human",
                    "stop_reason": "planning_scene_revision_mismatch",
                }
            )
            self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy, source="placement_scene_revision_mismatch"
            )
            return True
        if (
            receipt.get("error_code") == "MOTION_OUTCOME_UNKNOWN"
            or receipt.get("motion_outcome") == "unknown"
        ):
            policy.update(
                {
                    "status": "stopped_requires_human",
                    "stop_reason": "motion_outcome_unknown",
                }
            )
            self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy, source="placement_motion_unknown"
            )
            return True
        result_success = result.get("success") if isinstance(result, dict) else None
        retained_failure = result_success is False and (
            _trusted_retained_attachment_motion_failure(
                receipt,
                planning_scene_revision=expected_revision,
            )
        )
        if (
            result_success is False
            and not (
                receipt.get("error_code") == "MOTION_PLAN_FAILED"
                and receipt.get("execution_started") is False
            )
            and not retained_failure
        ):
            policy.update(
                {
                    "status": "stopped_requires_human",
                    "stop_reason": "placement_motion_may_have_started",
                }
            )
            self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy, source="placement_motion_uncertain"
            )
            return True
        if retained_failure:
            fingerprint = str(receipt.get("request_fingerprint") or "").strip()
            failed = list(policy.get("failed_request_fingerprints") or [])
            if fingerprint and fingerprint in failed:
                policy.update(
                    {
                        "status": "stopped_requires_human",
                        "stop_reason": "repeated_failed_request_fingerprint",
                    }
                )
                self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                    policy, source="placement_fingerprint_repeated"
                )
                return True
            if fingerprint:
                failed.append(fingerprint)
            rejected = list(policy.get("rejected_candidates") or [])
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "request_fingerprint": fingerprint,
                    "error_code": receipt.get("error_code"),
                    "execution_started": True,
                    "motion_outcome": "failed",
                    "planned_point_count": receipt.get("planned_point_count", 0),
                    "scene_revision": receipt.get("planning_scene_revision"),
                    "position_error_m": receipt.get("position_error_m"),
                    "orientation_error_rad": receipt.get("orientation_error_rad"),
                    "attachment_retained": True,
                    "reason": (
                        "known terminal miss with retained native attachment; "
                        "requalify the next frozen goal from the observed end state"
                    ),
                }
            )
            rejected_ids = {
                str(item.get("candidate_id") or "")
                for item in rejected
                if isinstance(item, dict) and str(item.get("candidate_id") or "")
            }
            total_eligible = _optional_int(
                policy.get("frozen_goal_total_eligible_count"), default=0
            )
            resume_available = policy.get(
                "frozen_goal_requalification"
            ) is True and total_eligible > len(rejected_ids)
            policy.update(
                {
                    "active_candidate_id": None,
                    "compiled_placement": None,
                    "rejected_candidates": rejected,
                    "failed_request_fingerprints": failed,
                    "status": (
                        "frozen_frontier_required" if resume_available else "stopped_requires_human"
                    ),
                    "stop_reason": (
                        "frozen_placement_frontier_requalification_required"
                        if resume_available
                        else "CURRENT_GRASP_PLACE_INFEASIBLE"
                    ),
                    "recovery": {
                        "stage": (
                            "measured_attachment_frontier_requalification"
                            if resume_available
                            else "manual_intervention"
                        ),
                        "required_action": None,
                    },
                }
            )
            self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source=(
                    "frozen_placement_frontier"
                    if resume_available
                    else "placement_motion_terminal_miss"
                ),
            )
            self.record(
                "placement_candidate_terminal_miss",
                dict(rejected[-1]),
            )
            return True
        if (
            receipt.get("error_code") != "MOTION_PLAN_FAILED"
            or receipt.get("execution_started") is not False
        ):
            return False
        fingerprint = str(receipt.get("request_fingerprint") or "").strip()
        failed = list(policy.get("failed_request_fingerprints") or [])
        if fingerprint and fingerprint in failed:
            policy.update(
                {
                    "status": "stopped_requires_human",
                    "stop_reason": "repeated_failed_request_fingerprint",
                }
            )
            self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy, source="placement_fingerprint_repeated"
            )
            return True
        if fingerprint and fingerprint not in failed:
            failed.append(fingerprint)
        rejected = list(policy.get("rejected_candidates") or [])
        rejected.append(
            {
                "candidate_id": candidate_id,
                "request_fingerprint": fingerprint,
                "moveit_error_code": receipt.get("moveit_error_code"),
                "planned_point_count": receipt.get("planned_point_count", 0),
                "scene_revision": receipt.get("planning_scene_revision"),
                "reason": (
                    "planning failed for this current joint state, target, tolerances, and scene"
                ),
            }
        )
        remaining = [
            candidate
            for candidate in policy.get("candidate_queue", [])
            if candidate not in {str(item.get("candidate_id") or "") for item in rejected}
        ]
        rejected_ids = {
            str(item.get("candidate_id") or "")
            for item in rejected
            if isinstance(item, dict) and str(item.get("candidate_id") or "")
        }
        total_eligible = _optional_int(policy.get("frozen_goal_total_eligible_count"), default=0)
        resume_available = (
            not remaining
            and policy.get("frozen_goal_requalification") is True
            and total_eligible > len(rejected_ids)
        )
        policy.update(
            {
                "active_candidate_id": None,
                "compiled_placement": None,
                "rejected_candidates": rejected,
                "failed_request_fingerprints": failed,
                "status": (
                    "selection_required"
                    if remaining
                    else "frozen_frontier_required"
                    if resume_available
                    else "stopped_requires_human"
                ),
            }
        )
        if resume_available:
            policy["stop_reason"] = "frozen_placement_frontier_requalification_required"
            policy["recovery"] = {
                "stage": "measured_attachment_frontier_requalification",
                "required_action": None,
            }
        elif not remaining:
            policy["stop_reason"] = "CURRENT_GRASP_PLACE_INFEASIBLE"
            policy["recovery"] = {
                "stage": "manual_intervention",
                "required_action": None,
            }
        self.facts[PLACEMENT_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy, source="placement_candidate_rejected"
        )
        if remaining:
            compiled_by_id = policy.get("host_candidate_compilations")
            next_compiled = (
                compiled_by_id.get(str(remaining[0])) if isinstance(compiled_by_id, dict) else None
            )
            if isinstance(next_compiled, dict):
                self._capture_compiled_placement(
                    dict(next_compiled), source="host_qualified_queue_fallback"
                )
        self.record("placement_candidate_rejected", dict(rejected[-1]))
        return True

    def _capture_articulated_attachment_probe(self, action: EnvAction) -> bool:
        call = _successful_tool_call(action, "prepare_attachment_probe")
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        if outputs.get("schema_version") != "openeta.articulated_attachment_probe.v1":
            return False
        execution = self.grasp_execution()
        policy = self.grasp_candidate_policy()
        if (
            not isinstance(execution, dict)
            or execution.get("status") != "required"
            or execution.get("stage") != "prepare_probe"
            or not isinstance(policy, dict)
            or policy.get("interaction_family") != "articulated_handle"
        ):
            return False
        if str(outputs.get("candidate_id") or "") != str(execution.get("candidate_id") or ""):
            return False
        if str(outputs.get("compiled_grasp_id") or "") != str(
            execution.get("compiled_grasp_id") or ""
        ):
            return False
        if _optional_int(outputs.get("scene_epoch"), default=-1) != self.scene_epoch():
            return False
        required_action = outputs.get("required_action")
        if not isinstance(required_action, dict):
            return False
        name = str(required_action.get("name") or "")
        parameters = required_action.get("parameters")
        if name not in {"move_to", "follow_eef_trajectory"} or not isinstance(parameters, dict):
            return False
        probe = {
            **dict(outputs),
            "status": "required",
            "prepared_at_s": time.time(),
        }
        execution.update(
            {
                "stage": "probe",
                "scene_epoch": self.scene_epoch(),
                "required_action": {"name": name, "parameters": dict(parameters)},
                "probe_kind": "articulated_attachment",
            }
        )
        self.facts[ARTICULATED_ATTACHMENT_PROBE_KEY] = _memory_fact_entry(
            probe,
            source="prepare_attachment_probe",
        )
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="prepare_attachment_probe",
        )
        self.record("articulated_attachment_probe_prepared", dict(probe))
        return True

    def _capture_articulated_attachment_probe_result(self, action: EnvAction) -> bool:
        probe = self.articulated_attachment_probe()
        if not isinstance(probe, dict) or probe.get("status") != "required":
            return False
        required = probe.get("required_action")
        if not isinstance(required, dict):
            return False
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        if not isinstance(request, dict):
            return False
        name = str(required.get("name") or "")
        parameters = required.get("parameters")
        if (
            str(request.get("name") or "") != name
            or not isinstance(parameters, dict)
            or request.get("parameters") != parameters
        ):
            return False
        call = _tool_call(action, name)
        probe["attempt_count"] = int(probe.get("attempt_count") or 0) + 1
        if (
            not isinstance(call, dict)
            or not _call_result_success(call)
            or _motion_call_rejects_candidate(call)
        ):
            probe["last_attempt_status"] = "failed"
            self.facts[ARTICULATED_ATTACHMENT_PROBE_KEY] = _memory_fact_entry(
                probe,
                source="runtime_articulated_attachment_probe",
            )
            self.record(
                "articulated_attachment_probe_failed",
                {
                    "candidate_id": probe.get("candidate_id"),
                    "attempt_count": probe["attempt_count"],
                },
            )
            return True
        probe.update(
            {
                "status": "completed",
                "completed_at_s": time.time(),
                "last_attempt_status": "executed",
            }
        )
        self.facts[ARTICULATED_ATTACHMENT_PROBE_KEY] = _memory_fact_entry(
            probe,
            source="runtime_articulated_attachment_probe",
        )
        self.record("articulated_attachment_probe_completed", dict(probe))
        execution = self.grasp_execution()
        if isinstance(execution, dict) and execution.get("stage") == "probe":
            self._advance_articulated_probe_to_attachment(execution)
        return True

    def _record_successful_world_mutation(self, action: EnvAction) -> bool:
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        name = str(request.get("name") or "") if isinstance(request, dict) else ""
        if name not in {
            "move_to",
            "follow_eef_trajectory",
            "gripper_control",
            "lower_body_control_policy",
        }:
            return False
        if _successful_tool_call(action, name) is None:
            return False
        epoch = self.scene_epoch() + 1
        self.facts[SCENE_EPOCH_KEY] = _memory_fact_entry(
            {"epoch": epoch},
            source="successful_world_mutation",
        )
        self.record("scene_epoch_advanced", {"scene_epoch": epoch, "tool": name})
        return True

    def _capture_gripper_command_state(self, action: EnvAction) -> bool:
        call = _successful_tool_call(action, "gripper_control")
        if call is None:
            return False
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        parameters = request.get("parameters") if isinstance(request, dict) else None
        requested_position = (
            parameters.get("position", parameters.get("open"))
            if isinstance(parameters, dict)
            else None
        )
        position = (
            _binary_gripper_position(requested_position) if isinstance(parameters, dict) else None
        )
        if position is None:
            return False
        self._set_gripper_command_state(position, source="acknowledged_gripper_command")
        return True

    def _capture_planning_scene_target_pose_sync(self, action: EnvAction) -> bool:
        """Retain the bounded proof needed to rebase a frozen grasp frontier."""

        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        request_name = str(request.get("name") or "") if isinstance(request, dict) else ""
        if request_name not in {
            "gripper_control",
            "move_to",
            "follow_eef_trajectory",
        }:
            return False
        call = _tool_call(action, request_name)
        if call is None:
            return False
        receipt = _environment_receipt(call)
        if not _call_result_success(call):
            rollback = receipt.get("planning_scene_rollback")
            detachable = receipt.get("detachable_joint")
            close_rollback = bool(
                receipt.get("candidate_rejection") is True
                and receipt.get("infrastructure_error") is False
                and isinstance(rollback, dict)
                and rollback.get("state") == "detached"
                and isinstance(detachable, dict)
                and detachable.get("state") == "detached"
            )
            if not close_rollback and not _trusted_detached_target_displacement(
                receipt
            ):
                return False
        sync = receipt.get("planning_scene_target_pose_sync")
        if not (
            isinstance(sync, dict)
            and sync.get("schema_version") == "openeta.planning_scene_target_pose_sync.v1"
            and sync.get("operation") == "update_world_target"
        ):
            return False
        previous = self.planning_scene_target_pose_sync()
        previous_sync = (
            previous.get("planning_scene_target_pose_sync")
            if isinstance(previous, dict)
            else None
        )
        pending = (self.grasp_candidate_policy() or {}).get(
            "frozen_grasp_frontier_rebase_pending"
        )
        recovery = self.grasp_recovery()
        preserve_recovery_lineage = (
            _call_result_success(call)
            and isinstance(previous_sync, dict)
            and isinstance(pending, dict)
            and pending.get("schema_version")
            == "openeta.frozen_grasp_frontier_rebase_pending.v1"
            and isinstance(recovery, dict)
            and recovery.get("status") == "required"
            and recovery.get("stage") == "reopen"
            and str(recovery.get("candidate_id") or "")
            == str(pending.get("physically_rejected_candidate_id") or "")
            and previous_sync.get("source_revision")
            == pending.get("source_planning_scene_revision")
            and previous_sync.get("revision") == sync.get("revision")
            and sync.get("source_revision") == sync.get("revision")
            and previous_sync.get("target_id") == sync.get("target_id")
            and sync.get("topology_unchanged") is True
            and sync.get("static_world_unchanged") is True
            and (receipt.get("detachable_joint") or {}).get("state") == "detached"
        )
        if preserve_recovery_lineage:
            payload = {
                **dict(previous),
                "detachable_joint": dict(receipt.get("detachable_joint") or {}),
                "planning_scene_revision": receipt.get("planning_scene_revision"),
                "confirmed_at_s": time.time(),
            }
            self.facts[PLANNING_SCENE_TARGET_POSE_SYNC_KEY] = _memory_fact_entry(
                payload,
                source="native_target_pose_sync_lineage_confirmed",
            )
            self.record("planning_scene_target_pose_sync_lineage_confirmed", dict(payload))
            return True
        payload = {
            "planning_scene_target_pose_sync": dict(sync),
            "detachable_joint": dict(receipt.get("detachable_joint") or {}),
            "planning_scene_revision": receipt.get("planning_scene_revision"),
            "captured_at_s": time.time(),
        }
        self.facts[PLANNING_SCENE_TARGET_POSE_SYNC_KEY] = _memory_fact_entry(
            payload,
            source="native_target_pose_sync",
        )
        self.record("planning_scene_target_pose_synchronized", dict(payload))
        return True

    def _set_gripper_command_state(self, position: int, *, source: str) -> None:
        state = {
            "schema_version": "openeta.gripper_command_state.v1",
            "position": position,
            "state": "open" if position == 1 else "closed",
            "latched": True,
            "scene_epoch": self.scene_epoch(),
            "updated_at_s": time.time(),
        }
        self.facts[GRIPPER_COMMAND_STATE_KEY] = _memory_fact_entry(state, source=source)
        self.record("gripper_command_state_changed", dict(state))

    def _capture_placement_release(self, action: EnvAction) -> bool:
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        if not isinstance(request, dict):
            return False
        name = str(request.get("name") or "")
        parameters = request.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        if name == "move_to":
            target_pose = parameters.get("target_pose")
            if not isinstance(target_pose, dict):
                return False
            placement_stage = str(target_pose.get("placement_stage") or "")
            call = _successful_tool_call(action, "move_to")
            if call is None:
                return False
            policy = self.placement_candidate_policy() or {}
            receipt = _environment_receipt(call)
            expected_revision = _optional_int(policy.get("scene_revision"), default=-1)
            if policy.get("revision_provenance") == "native_attachment_gate" and (
                _optional_int(target_pose.get("scene_revision"), default=-2) != expected_revision
                or _optional_int(receipt.get("planning_scene_revision"), default=-3)
                != expected_revision
            ):
                return False
            if _motion_call_rejects_candidate(call):
                return False
            if placement_stage != "release":
                return False
            release = {
                "schema_version": "openeta.placement_release.v1",
                "status": "ready",
                "candidate_id": target_pose.get("placement_candidate_id"),
                "placement_pose_id": target_pose.get("placement_pose_id"),
                "release_pose": dict(target_pose),
                "scene_epoch": self.scene_epoch(),
                "ready_at_s": time.time(),
                "arrival_stage": "release",
            }
            self.facts[PLACEMENT_RELEASE_KEY] = _memory_fact_entry(
                release,
                source="placement_release_pose_reached",
            )
            self.record("placement_release_ready", dict(release))
            return True
        if name != "gripper_control" or _successful_tool_call(action, name) is None:
            return False
        if _binary_gripper_position(parameters.get("position")) != 1:
            return False
        release = self.placement_release()
        if not isinstance(release, dict) or release.get("status") != "ready":
            return False
        call = _successful_tool_call(action, name)
        receipt = _environment_receipt(call) if isinstance(call, dict) else {}
        policy = self.placement_candidate_policy() or {}
        prior_revision = _optional_int(policy.get("scene_revision"), default=-1)
        detached = receipt.get("detachable_joint")
        release_sequence = receipt.get("release_sequence")
        ordered_release = (
            [dict(item) for item in release_sequence if isinstance(item, dict)]
            if isinstance(release_sequence, list)
            else []
        )
        release_proof = ordered_native_release_proof(ordered_release)
        irreversible_open = (
            receipt.get("gripper_open_executed") is True
            and isinstance(detached, dict)
            and detached.get("state") == "detached"
        )
        if (
            release_proof is None
            or not isinstance(release_proof.get("planning_scene_detach_ack"), dict)
            or not isinstance(release_proof.get("gripper_open_completed"), dict)
        ):
            if not irreversible_open:
                return False
            failure_code = "PLACEMENT_RELEASE_SEQUENCE_INVALID"
            return self._record_failed_placement_release(
                release,
                failure_code=failure_code,
                failure_reason=(
                    "The object was detached and the gripper opened, but the "
                    "ordered native release proof was invalid. A fresh observation "
                    "is required."
                ),
                evidence={
                    "schema_version": "openeta.placement_release_failure.v1",
                    "failure_code": failure_code,
                    "planning_scene_revision": receipt.get(
                        "planning_scene_revision"
                    ),
                    "detachable_joint": dict(detached),
                    "gripper_open_executed": True,
                    "release_sequence": ordered_release,
                },
                source="placement_release_sequence_invalid",
                event_type="placement_release_failed_after_irreversible_transition",
                advance_scene_epoch=True,
            )
        detach_revision = _optional_int(
            release_proof["planning_scene_detach_ack"].get("revision"),
            default=-1,
        )
        final_revision = _optional_int(receipt.get("planning_scene_revision"), default=-1)
        terminal = release_proof.get("released_target_pose_sync_ack")
        target_pose_revision = (
            _optional_int(terminal.get("revision"), default=-1)
            if isinstance(terminal, dict)
            else None
        )
        if policy.get("revision_provenance") == "native_attachment_gate":
            if (
                not isinstance(detached, dict)
                or detached.get("state") != "detached"
                or prior_revision < 0
                or detach_revision != prior_revision + 1
                or final_revision < detach_revision
                or (
                    target_pose_revision is not None
                    and (
                        target_pose_revision != detach_revision + 1
                        or final_revision != target_pose_revision
                    )
                )
                or (target_pose_revision is None and final_revision != detach_revision)
            ):
                return False
        release_evidence = receipt.get("release_evidence")
        gripper_open_terminal_dispatch_accepted = bool(
            isinstance(release_evidence, dict)
            and known_gripper_open_terminal_dispatch(
                release_evidence.get("gripper_open_terminal_dispatch")
            )
        )
        release_proven = bool(
            isinstance(release_evidence, dict)
            and release_evidence.get("schema_version")
            == "openeta.native_release_evidence.v1"
            and release_evidence.get("detached_confirmed") is True
            and (
                release_evidence.get("gripper_open_confirmed") is True
                or gripper_open_terminal_dispatch_accepted
            )
            and isinstance(
                release_evidence.get("post_release_visual_observation"), dict
            )
        )
        # Historical receipts used a blocking geometry stability verdict.
        # Keep those artifacts readable, but never generate that criterion in
        # the live release path.
        legacy_verification = receipt.get("placement_verification")
        legacy_release_proven = bool(
            isinstance(legacy_verification, dict)
            and legacy_verification.get("placement_confirmed") is True
            and str(legacy_verification.get("verdict") or "").upper() == "PASS"
        )
        if not release_proven and legacy_release_proven:
            release_evidence = {
                "schema_version": "openeta.native_release_evidence.v1",
                "detached_confirmed": True,
                "gripper_open_confirmed": True,
                "post_release_visual_observation": {
                    "schema_version": (
                        "openeta.post_release_visual_observation.v1"
                    ),
                    "required": True,
                    "available": True,
                    "source": "legacy_post_release_receipt",
                    "review_authority": "vlm",
                },
                "legacy_placement_verification": dict(legacy_verification),
            }
            release_proven = True
        if not release_proven:
            legacy_verification_failed = bool(
                isinstance(legacy_verification, dict)
                and not legacy_release_proven
            )
            failure_code = (
                "PLACEMENT_RELEASE_VERIFICATION_FAILED"
                if legacy_verification_failed
                else "PLACEMENT_RELEASE_EVIDENCE_INVALID"
            )
            return self._record_failed_placement_release(
                release,
                failure_code=failure_code,
                failure_reason=(
                    "The historical placement receipt reported a failed geometry "
                    "verdict. A fresh observation is required."
                    if legacy_verification_failed
                    else (
                        "The object release did not contain native detach plus a "
                        "confirmed or known-dispatched gripper-open boundary. A fresh "
                        "observation is required."
                    )
                ),
                evidence={
                    "schema_version": "openeta.placement_release_failure.v1",
                    "failure_code": failure_code,
                    "planning_scene_revision": final_revision,
                    "detachable_joint": (dict(detached) if isinstance(detached, dict) else {}),
                    "gripper_open_executed": receipt.get("gripper_open_executed"),
                    "release_sequence": ordered_release,
                    "release_evidence": (
                        dict(release_evidence)
                        if isinstance(release_evidence, dict)
                        else None
                    ),
                    **(
                        {"placement_verification": dict(legacy_verification)}
                        if legacy_verification_failed
                        else {}
                    ),
                },
                source="placement_release_evidence_invalid",
                event_type="placement_release_failed_after_detach",
            )
        post_release_visual = release_evidence[
            "post_release_visual_observation"
        ]
        release.update(
            {
                "status": "released",
                "scene_epoch": self.scene_epoch(),
                "released_at_s": time.time(),
                "release_evidence": dict(release_evidence),
                "post_release_visual_observation": dict(post_release_visual),
                **(
                    {"placement_verification": dict(legacy_verification)}
                    if legacy_release_proven
                    else {}
                ),
                **(
                    {"planning_scene_revision": detach_revision}
                    if policy.get("revision_provenance") == "native_attachment_gate"
                    else {}
                ),
                **(
                    {"released_target_pose_revision": target_pose_revision}
                    if target_pose_revision is not None
                    else {}
                ),
                **(
                    {
                        "attached_collision_filter": dict(
                            release_proof["attached_collision_filter_ack"]
                        )
                    }
                    if isinstance(
                        release_proof.get("attached_collision_filter_ack"),
                        dict,
                    )
                    else {}
                ),
                "release_sequence": ordered_release,
                **(
                    {"multi_sort_progress": dict(receipt["multi_sort_progress"])}
                    if isinstance(receipt.get("multi_sort_progress"), dict)
                    and receipt["multi_sort_progress"].get("schema_version")
                    == "openeta.multi_sort_progress.v1"
                    else {}
                ),
            }
        )
        if isinstance(release.get("multi_sort_progress"), dict):
            self.facts[MULTI_SORT_PROGRESS_KEY] = _memory_fact_entry(
                dict(release["multi_sort_progress"]),
                source="placement_release_multi_sort_progress",
            )
        self.facts[PLACEMENT_RELEASE_KEY] = _memory_fact_entry(
            release,
            source="placement_release_completed",
        )
        self._finalize_placement_subgoal(release)
        self.record("placement_released", dict(release))
        return True

    def _capture_irreversible_placement_release_failure(self, action: EnvAction) -> bool:
        """Require reobservation after a failed call that detached and opened.

        Detaching the object is an irreversible world transition.  Replaying
        ``gripper_control(open)`` cannot recreate the ordered release proof.
        When the receipt proves the physical transition, close the failed
        attempt and start a fresh observed agentic cycle instead of treating
        the object as still attached.
        """

        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        request = request if isinstance(request, dict) else {}
        parameters = request.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        if (
            request.get("name") != "gripper_control"
            or _binary_gripper_position(parameters.get("position")) != 1
        ):
            return False
        call = _tool_call(action, "gripper_control")
        if not isinstance(call, dict) or _call_result_success(call):
            return False
        release = self.placement_release()
        if not isinstance(release, dict) or release.get("status") != "ready":
            return False
        receipt = _environment_receipt(call)
        detachable = receipt.get("detachable_joint")
        sequence = receipt.get("release_sequence")
        ordered = (
            [dict(item) for item in sequence if isinstance(item, dict)]
            if isinstance(sequence, list)
            else []
        )
        release_proof = ordered_native_release_proof(ordered)
        irreversible = (
            receipt.get("gripper_open_executed") is True
            and isinstance(detachable, dict)
            and detachable.get("state") == "detached"
            and isinstance(release_proof, dict)
            and isinstance(release_proof.get("native_detach_ack"), dict)
            and isinstance(
                release_proof.get("planning_scene_detach_ack"), dict
            )
        )
        if not irreversible:
            return False

        failure_code = "PLACEMENT_RELEASE_POST_DETACH_VERIFICATION_FAILED"
        evidence = {
            "schema_version": "openeta.placement_release_failure.v1",
            "failure_code": failure_code,
            "error_code": receipt.get("error_code"),
            "planning_scene_revision": receipt.get("planning_scene_revision"),
            "detachable_joint": dict(detachable),
            "gripper_open_executed": True,
            "release_sequence": ordered,
            **(
                {"release_evidence": dict(receipt["release_evidence"])}
                if isinstance(receipt.get("release_evidence"), dict)
                else {}
            ),
            **(
                {"placement_verification": dict(receipt["placement_verification"])}
                if isinstance(receipt.get("placement_verification"), dict)
                else {}
            ),
        }
        return self._record_failed_placement_release(
            release,
            failure_code=failure_code,
            failure_reason=(
                "The object was detached and the gripper-open command ran, but "
                "the call did not finish its post-release world synchronization. "
                "A fresh observation is required."
            ),
            evidence=evidence,
            source="irreversible_placement_release_failure",
            event_type="placement_release_failed_after_irreversible_transition",
            advance_scene_epoch=True,
        )

    def _record_failed_placement_release(
        self,
        release: JsonDict,
        *,
        failure_code: str,
        failure_reason: str,
        evidence: Mapping[str, Any],
        source: str,
        event_type: str,
        advance_scene_epoch: bool = False,
    ) -> bool:
        """Close one placement attempt and invalidate its stale model state."""

        if advance_scene_epoch:
            epoch = self.scene_epoch() + 1
            self.facts[SCENE_EPOCH_KEY] = _memory_fact_entry(
                {"epoch": epoch},
                source="irreversible_placement_release",
            )
            self.record(
                "scene_epoch_advanced",
                {"scene_epoch": epoch, "tool": "gripper_control"},
            )
        gripper_state = self.gripper_command_state()
        if evidence.get("gripper_open_executed") is True and not (
            isinstance(gripper_state, dict) and gripper_state.get("position") == 1
        ):
            self._set_gripper_command_state(
                1,
                source="failed_placement_release_open_proof",
            )
        release.update(
            {
                "status": "failed",
                "failure_code": failure_code,
                "failure_reason": failure_reason,
                "failure_evidence": dict(evidence),
                "reobservation_required": True,
                "scene_epoch": self.scene_epoch(),
                "failed_at_s": time.time(),
            }
        )
        self.facts[PLACEMENT_RELEASE_KEY] = _memory_fact_entry(
            release,
            source=source,
        )

        invalidated_facts = []
        for key in (
            PENDING_SAM3_SELECTION_KEY,
            SELECTED_SAM3_DETECTION_KEY,
            PENDING_REFERENCE_LOCALIZATION_KEY,
            TARGET_ASSET_REFERENCE_KEY,
            SAM3_NO_DETECTION_KEY,
            SAM3_SEMANTIC_STATE_KEY,
            GRASP_CANDIDATE_POLICY_KEY,
            LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY,
            GRASP_REESTIMATION_KEY,
            ARTICULATED_ATTACHMENT_PROBE_KEY,
            GRASP_EXECUTION_KEY,
            GRASP_RECOVERY_KEY,
            GRASP_ESTIMATION_RECOVERY_KEY,
            ATTACHMENT_GATE_KEY,
            PLACEMENT_CANDIDATE_POLICY_KEY,
            PLACEMENT_OBJECT_DETECTION_KEY,
            PLACEMENT_REGION_DETECTION_KEY,
            FROZEN_PLACEMENT_POOL_KEY,
            PLANNING_SCENE_TARGET_POSE_SYNC_KEY,
            MOTION_RECONCILIATION_KEY,
        ):
            if self.facts.pop(key, None) is not None:
                invalidated_facts.append(key)
        invalidated_artifacts = []
        for key in (
            "anygrasp_grasp_candidates_latest",
            "grasp_pose_estimate_grasp_candidates_latest",
            "graspgenx_grasp_candidates_latest",
            "contact_graspnet_grasp_candidates_latest",
            "anyplace_placement_candidates_latest",
            "camera_pose_to_world_world_pose_latest",
        ):
            if self.artifacts.pop(key, None) is not None:
                invalidated_artifacts.append(key)
        self.record(
            event_type,
            {
                "candidate_id": release.get("candidate_id"),
                "placement_pose_id": release.get("placement_pose_id"),
                "failure_code": failure_code,
                "reobservation_required": True,
                "invalidated_facts": invalidated_facts,
                "invalidated_artifacts": invalidated_artifacts,
                "evidence": dict(evidence),
            },
        )
        return True

    def _capture_placement_release_denial(self, action: EnvAction) -> bool:
        obligation = _action_host_obligation(action)
        if obligation.get("stage") != "release":
            return False
        call = _tool_call(action, "gripper_control")
        if not isinstance(call, dict) or not _call_has_diagnostic(
            call,
            "supervision_denied",
        ):
            return False
        release = self.placement_release()
        if not isinstance(release, dict) or release.get("status") != "ready":
            return False
        result = call.get("result")
        reason = (
            str(result.get("content") or "Independent reviewer denied placement release.")
            if isinstance(result, dict)
            else "Independent reviewer denied placement release."
        )
        return self._record_failed_placement_release(
            release,
            failure_code="PLACEMENT_RELEASE_SUPERVISION_DENIED",
            failure_reason=reason,
            evidence={
                "schema_version": "openeta.placement_release_failure.v1",
                "failure_code": "PLACEMENT_RELEASE_SUPERVISION_DENIED",
                "supervision_denied": True,
            },
            source="placement_release_supervision_denied",
            event_type="placement_release_failed",
        )

    def _finalize_placement_subgoal(self, release: JsonDict) -> None:
        policy = self.grasp_candidate_policy() or {}
        target = policy.get("target_detection")
        target = target if isinstance(target, dict) else {}
        completed = _memory_fact_value(self.facts.get(COMPLETED_PLACEMENT_SUBGOALS_KEY))
        items = list(completed.get("items") or []) if isinstance(completed, dict) else []
        progress = release.get("multi_sort_progress")
        progress = progress if isinstance(progress, dict) else {}
        transition = progress.get("transition")
        transition = transition if isinstance(transition, dict) else {}
        items.append(
            {
                "candidate_id": release.get("candidate_id"),
                "placement_pose_id": release.get("placement_pose_id"),
                "target_object": target.get("target_prompt") or target.get("label"),
                "assignment_id": transition.get("completed_assignment_id"),
                "release_mode": release.get("release_mode", "explicit_gripper_open"),
                "completed_at_s": time.time(),
            }
        )
        self.facts[COMPLETED_PLACEMENT_SUBGOALS_KEY] = _memory_fact_entry(
            {"items": items[-16:]},
            source="placement_subgoal_completed",
        )
        for key in (
            PENDING_SAM3_SELECTION_KEY,
            SELECTED_SAM3_DETECTION_KEY,
            PENDING_REFERENCE_LOCALIZATION_KEY,
            TARGET_ASSET_REFERENCE_KEY,
            SAM3_NO_DETECTION_KEY,
            GRASP_CANDIDATE_POLICY_KEY,
            LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY,
            ARTICULATED_ATTACHMENT_PROBE_KEY,
            GRASP_EXECUTION_KEY,
            GRASP_ESTIMATION_RECOVERY_KEY,
            ATTACHMENT_GATE_KEY,
        ):
            self.facts.pop(key, None)
        if int(progress.get("remaining_count") or 0) > 0:
            for key in (
                REFERENCE_LOCALIZATION_FAILURE_KEY,
                TARGET_LOCALIZATION_BUDGET_KEY,
                SAM3_SEMANTIC_STATE_KEY,
                GRASP_RECOVERY_KEY,
                PLACEMENT_CANDIDATE_POLICY_KEY,
                PLACEMENT_OBJECT_DETECTION_KEY,
                PLACEMENT_REGION_DETECTION_KEY,
                FROZEN_PLACEMENT_POOL_KEY,
                PLANNING_SCENE_TARGET_POSE_SYNC_KEY,
                MOTION_RECONCILIATION_KEY,
            ):
                self.facts.pop(key, None)
        for key in (
            "anygrasp_grasp_candidates_latest",
            "grasp_pose_estimate_grasp_candidates_latest",
            "graspgenx_grasp_candidates_latest",
            "contact_graspnet_grasp_candidates_latest",
            "anyplace_placement_candidates_latest",
            "camera_pose_to_world_world_pose_latest",
        ):
            self.artifacts.pop(key, None)

    def _advance_grasp_execution(self, action: EnvAction) -> bool:
        execution = self.grasp_execution()
        if not isinstance(execution, dict) or execution.get("status") != "required":
            return False
        stage = str(execution.get("stage") or "")
        required = execution.get("required_action")
        if stage == "probe":
            articulated_probe = self.articulated_attachment_probe()
            if isinstance(articulated_probe, dict):
                if articulated_probe.get("status") == "completed":
                    return self._advance_articulated_probe_to_attachment(execution)
                return False
            return False
        if stage == "attachment":
            if execution.get("attachment_mode") != "articulated_handle":
                return False
            gate = self.attachment_gate() or {}
            verdict = str(gate.get("verdict") or "UNKNOWN").upper()
            actions = execution.get("attachment_actions")
            if verdict == "PASS":
                self._complete_attachment_execution(
                    execution,
                    source="runtime_articulated_attachment_pass",
                )
                self.record(
                    "articulated_attachment_confirmed",
                    {
                        "candidate_id": execution.get("candidate_id"),
                        "scene_epoch": self.scene_epoch(),
                    },
                )
                return True
            fail_action = actions.get("fail") if isinstance(actions, dict) else None
            if (
                verdict == "FAIL"
                and isinstance(fail_action, dict)
                and _action_matches(action, fail_action)
                and _successful_tool_call(action, str(fail_action.get("name") or "")) is not None
            ):
                return False
            return False
        if not isinstance(required, dict) or not _grasp_stage_action_matches(
            action,
            required,
            stage=stage,
        ):
            return False
        name = str(required.get("name") or "")
        successful_call = _successful_tool_call(action, name)
        if successful_call is None:
            return False
        if name in {"move_to", "follow_eef_trajectory"} and _motion_call_rejects_candidate(
            successful_call
        ):
            return False

        compiled = execution.get("compiled_grasp")
        compiled = compiled if isinstance(compiled, dict) else {}
        if stage == "open":
            contact = compiled.get("contact_pose")
            if not isinstance(contact, dict):
                return False
            execution.update(
                {
                    "stage": "contact",
                    "scene_epoch": self.scene_epoch(),
                    "required_action": {
                        "name": "move_to",
                        "parameters": {"target_pose": _pose_for_epoch(contact, self.scene_epoch())},
                    },
                }
            )
        elif stage == "contact":
            self._mark_active_candidate_accepted(source="compiled_contact_reached")
            execution.update(
                {
                    "stage": "close",
                    "scene_epoch": self.scene_epoch(),
                    "required_action": {
                        "name": "gripper_control",
                        "parameters": {"position": 0},
                    },
                }
            )
        elif stage == "close":
            policy = self.grasp_candidate_policy() or {}
            articulated = policy.get("interaction_family") == "articulated_handle"
            close_receipt = _environment_receipt(successful_call)
            close_revision = close_receipt.get("planning_scene_revision")
            attached = close_receipt.get("detachable_joint")
            if articulated:
                execution.update(
                    {
                        "stage": "prepare_probe",
                        "scene_epoch": self.scene_epoch(),
                        "required_action": None,
                        "probe_kind": "articulated_attachment",
                    }
                )
            else:
                proof_ok, proof_reason, proof = _trusted_native_attachment_proof(
                    close_receipt,
                    planning_scene_revision=(
                        int(close_revision)
                        if isinstance(close_revision, int) and not isinstance(close_revision, bool)
                        else -1
                    ),
                )
                gate = {
                    "schema_version": "openeta.attachment_gate.v2",
                    "status": "resolved" if proof_ok else "stopped_requires_human",
                    "verdict": "PASS" if proof_ok else "UNKNOWN",
                    "candidate_id": execution.get("candidate_id"),
                    "compiled_grasp_id": execution.get("compiled_grasp_id"),
                    "scene_epoch": self.scene_epoch(),
                    "planning_scene_revision": close_revision,
                    "evidence_source": "gazebo_native_contact_attach_ack",
                    "attachment_proof": proof,
                    **({"stop_reason": proof_reason} if not proof_ok else {}),
                }
                self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
                    gate, source="runtime_native_contact_attach_ack"
                )
                execution.update(
                    {
                        "planning_scene_revision": close_revision,
                        "attach_ack": dict(attached) if isinstance(attached, dict) else {},
                    }
                )
                if proof_ok:
                    self._complete_attachment_execution(
                        execution,
                        source="runtime_native_contact_attach_ack",
                    )
                    return True
                execution.update(
                    {
                        "status": "stopped_requires_human",
                        "stage": "attachment_unknown",
                        "required_action": None,
                    }
                )
        else:
            return False
        execution.pop("required_tool", None)
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="runtime_grasp_transition",
        )
        self.record(
            "grasp_execution_transition",
            {
                "candidate_id": execution.get("candidate_id"),
                "completed_stage": stage,
                "next_stage": execution.get("stage"),
                "scene_epoch": self.scene_epoch(),
            },
        )
        return True

    def _advance_articulated_probe_to_attachment(self, execution: JsonDict) -> bool:
        probe = self.articulated_attachment_probe()
        if (
            not isinstance(probe, dict)
            or probe.get("status") != "completed"
            or str(probe.get("candidate_id") or "") != str(execution.get("candidate_id") or "")
        ):
            return False
        recovery_open = {"name": "gripper_control", "parameters": {"position": 1}}
        execution.update(
            {
                "stage": "attachment",
                "scene_epoch": self.scene_epoch(),
                "required_action": None,
                "attachment_mode": "articulated_handle",
                "attachment_actions": {"fail": recovery_open},
            }
        )
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="runtime_articulated_attachment_gate",
        )
        self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
            {
                "status": "pending",
                "verdict": "UNKNOWN",
                "candidate_id": execution.get("candidate_id"),
                "scene_epoch": self.scene_epoch(),
                "attachment_mode": "articulated_handle",
                "unknown_observation_count": 0,
            },
            source="runtime_articulated_attachment_gate",
        )
        self.record(
            "attachment_gate_required",
            {
                "candidate_id": execution.get("candidate_id"),
                "scene_epoch": self.scene_epoch(),
                "attachment_mode": "articulated_handle",
            },
        )
        return True

    def _mark_active_candidate_accepted(self, *, source: str) -> bool:
        policy = self.anygrasp_candidate_policy()
        if not isinstance(policy, dict) or policy.get("status") != "active":
            return False
        active = policy.get("active_candidate")
        if not isinstance(active, dict):
            return False
        policy.update(
            {
                "status": "accepted",
                "accepted_candidate": dict(active),
                "accepted_at_s": time.time(),
            }
        )
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(policy, source=source)
        self.record(
            "anygrasp_candidate_accepted",
            {
                "result_id": policy.get("result_id"),
                "candidate_id": active.get("id"),
                "rank": policy.get("active_rank"),
                "score": active.get("score"),
                "source": source,
            },
        )
        return True

    def _capture_attachment_observation_verdict(
        self,
        observation: EnvObservation,
    ) -> bool:
        execution = self.grasp_execution()
        if (
            not isinstance(execution, dict)
            or execution.get("status") != "required"
            or execution.get("stage") != "attachment"
        ):
            return False
        gate = self.attachment_gate()
        if not isinstance(gate, dict):
            return False
        if execution.get("attachment_mode") != "articulated_handle":
            return False
        command = self._latest_action_command()
        request = command.get("request") if isinstance(command, dict) else None
        if (
            str(gate.get("verdict") or "UNKNOWN").upper() != "UNKNOWN"
            or not isinstance(request, dict)
            or request.get("name") != "observe"
            or gate.get("refresh_required") is not True
            or gate.get("unknown_refresh_completed") is True
        ):
            return False
        gate.update(
            {
                "unknown_refresh_completed": True,
                "refresh_required": False,
                "last_observed_scene_epoch": self.scene_epoch(),
                "last_observed_at_s": time.time(),
            }
        )
        self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
            gate,
            source="articulated_attachment_observation",
        )
        self.record(
            "articulated_attachment_probe_verdict",
            {
                "candidate_id": execution.get("candidate_id"),
                "verdict": "UNKNOWN",
                "refresh_completed": True,
                "scene_epoch": self.scene_epoch(),
            },
        )
        return True

    def _capture_articulated_attachment_assessment(self, action: EnvAction) -> bool:
        call = _successful_tool_call(action, "assess_attachment_probe")
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        if outputs.get("schema_version") != ("openeta.articulated_attachment_assessment.v1"):
            return False
        execution = self.grasp_execution()
        gate = self.attachment_gate()
        if (
            not isinstance(execution, dict)
            or execution.get("stage") != "attachment"
            or execution.get("attachment_mode") != "articulated_handle"
            or not isinstance(gate, dict)
        ):
            return False
        if str(outputs.get("candidate_id") or "") != str(execution.get("candidate_id") or ""):
            return False
        verdict = str(outputs.get("verdict") or "").upper()
        if verdict not in {"PASS", "FAIL", "UNKNOWN"}:
            return False
        assessment_count = int(gate.get("assessment_count") or 0) + 1
        gate.update(
            {
                "status": "resolved" if verdict in {"PASS", "FAIL"} else "pending",
                "verdict": verdict,
                "assessment_count": assessment_count,
                "evidence_source": "independent_attachment_reviewer",
                "assessment_reason": outputs.get("reason"),
                "updated_at_s": time.time(),
                "refresh_required": verdict == "UNKNOWN" and assessment_count == 1,
            }
        )
        self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
            gate,
            source="independent_attachment_reviewer",
        )
        self.record("articulated_attachment_probe_verdict", dict(gate))
        return True

    def _complete_attachment_execution(
        self,
        execution: JsonDict,
        *,
        source: str,
    ) -> None:
        execution.update(
            {
                "status": "completed",
                "stage": "attached",
                "scene_epoch": self.scene_epoch(),
                "required_action": None,
                "completed_at_s": time.time(),
            }
        )
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(execution, source=source)
        self.record(
            "grasp_attachment_confirmed",
            {"candidate_id": execution.get("candidate_id"), "source": source},
        )

    def _capture_motion_reconciliation(self, action: EnvAction) -> bool:
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        if not isinstance(request, dict):
            return False
        name = str(request.get("name") or "")
        if name not in {"move_to", "follow_eef_trajectory", "gripper_control"}:
            return False
        call = _tool_call(action, name)
        outputs = _tool_call_outputs(call) if isinstance(call, dict) else {}
        if outputs.get("motion_outcome") != "unknown":
            return False
        recovery = self.grasp_recovery()
        recovery_action = recovery.get("required_action") if isinstance(recovery, dict) else None
        is_recovery_action = bool(
            isinstance(recovery, dict)
            and recovery.get("status") == "required"
            and isinstance(recovery_action, dict)
            and name == recovery_action.get("name")
            and request.get("parameters") == recovery_action.get("parameters")
        )
        reconciliation = {
            "status": "required",
            "tool": name,
            "intended_parameters": dict(request.get("parameters") or {}),
            "candidate_id": _parameters_grasp_candidate_id(request.get("parameters") or {}),
            "grasp_stage": (
                None if is_recovery_action else (self.grasp_execution() or {}).get("stage")
            ),
            "grasp_recovery_stage": (
                recovery.get("stage") if is_recovery_action and isinstance(recovery, dict) else None
            ),
            "scene_epoch": self.scene_epoch(),
            "created_at_s": time.time(),
        }
        self.facts[MOTION_RECONCILIATION_KEY] = _memory_fact_entry(
            reconciliation,
            source="simulator_action_outcome_unknown",
        )
        self.record("motion_reconciliation_required", dict(reconciliation))
        return True

    def _capture_planning_scene_control_failure(self, action: EnvAction) -> bool:
        """Stop a grasp once the controller cannot prove its planning scene."""

        execution = self.grasp_execution()
        if not isinstance(execution, dict) or execution.get("status") != "required":
            return False
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        if not isinstance(request, dict):
            return False
        name = str(request.get("name") or "")
        if name not in {"move_to", "follow_eef_trajectory", "gripper_control"}:
            return False
        call = _tool_call(action, name)
        if not isinstance(call, dict) or _call_result_success(call):
            return False
        receipt = _environment_receipt(call)
        error_code = str(receipt.get("error_code") or "")
        if not (
            error_code == "PLANNING_SCENE_UNAVAILABLE"
            or error_code.startswith("PLANNING_SCENE_SYNC_FAILED")
            or (
                isinstance(receipt.get("planning_scene_rollback"), dict)
                and receipt["planning_scene_rollback"].get("state") == "failed"
            )
        ):
            return False
        if receipt.get("execution_started") not in {False, None}:
            return False
        execution.update(
            {
                "status": "stopped_requires_human",
                "stage": "planning_scene_failure",
                "required_action": None,
                "control_failure": {
                    "error_code": error_code or "PLANNING_SCENE_ROLLBACK_FAILED",
                    "planning_scene_revision": receipt.get("planning_scene_revision"),
                    "execution_started": receipt.get("execution_started"),
                },
                "stopped_at_s": time.time(),
            }
        )
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution, source="planning_scene_control_failure"
        )
        policy = self.grasp_candidate_policy()
        if isinstance(policy, dict):
            policy.update(
                {
                    "status": "stopped_requires_human",
                    "active_candidate_id": execution.get("candidate_id"),
                }
            )
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy, source="planning_scene_control_failure"
            )
        gate = {
            "schema_version": "openeta.attachment_gate.v1",
            "status": "stopped_requires_human",
            "verdict": "UNKNOWN",
            "assessment_reason": "planning_scene_control_failure",
            "candidate_id": execution.get("candidate_id"),
            "planning_scene_revision": receipt.get("planning_scene_revision"),
        }
        self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
            gate, source="planning_scene_control_failure"
        )
        self.record("planning_scene_control_failure", dict(execution["control_failure"]))
        return True

    def _reconcile_unknown_motion(self, observation: EnvObservation) -> bool:
        reconciliation = self.motion_reconciliation()
        if not isinstance(reconciliation, dict) or reconciliation.get("status") not in {
            "required",
            "unresolved",
        }:
            return False
        parameters = reconciliation.get("intended_parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        tool_name = str(reconciliation.get("tool") or "")
        target_pose = parameters.get("target_pose")
        target_xyz = target_pose.get("xyz") if isinstance(target_pose, dict) else None
        measured_xyz = observation.robot.end_effector_pose.get("xyz")
        if tool_name == "gripper_control":
            requested_position = parameters.get("position", parameters.get("open"))
            verdict = _reconcile_gripper_position(
                requested_position, observation.robot.gripper_state
            )
            if verdict == "completed":
                self.facts[SCENE_EPOCH_KEY] = _memory_fact_entry(
                    {"epoch": self.scene_epoch() + 1},
                    source="reconciled_world_mutation",
                )
                position = _binary_gripper_position(requested_position)
                if position is not None:
                    self._set_gripper_command_state(
                        position,
                        source="reconciled_gripper_command",
                    )
                self._advance_reconciled_grasp_stage(str(reconciliation.get("grasp_stage") or ""))
            reconciliation.update(
                {
                    "status": verdict,
                    "measured_gripper_state": dict(observation.robot.gripper_state),
                    "resolved_at_s": time.time(),
                }
            )
        elif _finite_xyz(target_xyz) and _finite_xyz(measured_xyz):
            distance = math.sqrt(
                sum(
                    (float(target_xyz[index]) - float(measured_xyz[index])) ** 2
                    for index in range(3)
                )
            )
            tolerance = float(parameters.get("tolerance") or 0.02)
            previous_measured_xyz = reconciliation.get("measured_eef_xyz")
            observation_count = int(reconciliation.get("observation_count") or 0) + 1
            if distance <= max(0.005, tolerance):
                verdict = "completed"
                self.facts[SCENE_EPOCH_KEY] = _memory_fact_entry(
                    {"epoch": self.scene_epoch() + 1},
                    source="reconciled_world_mutation",
                )
                self._advance_reconciled_grasp_stage(str(reconciliation.get("grasp_stage") or ""))
            elif (
                observation_count >= 3
                and _finite_xyz(previous_measured_xyz)
                and math.sqrt(
                    sum(
                        (float(previous_measured_xyz[index]) - float(measured_xyz[index])) ** 2
                        for index in range(3)
                    )
                )
                <= 0.005
            ):
                verdict = "failed"
            elif distance <= 0.20:
                verdict = "required"
            else:
                verdict = "unresolved"
            reconciliation.update(
                {
                    "status": verdict,
                    "observation_count": observation_count,
                    "distance_to_target_m": round(distance, 6),
                    "measured_eef_xyz": [float(value) for value in measured_xyz[:3]],
                    "resolved_at_s": time.time(),
                }
            )
        else:
            reconciliation.update({"status": "unresolved", "resolved_at_s": time.time()})
            verdict = "unresolved"
        self._resolve_reconciled_grasp_recovery(
            tool_name=tool_name,
            parameters=parameters,
            verdict=verdict,
        )
        self.facts[MOTION_RECONCILIATION_KEY] = _memory_fact_entry(
            reconciliation,
            source="same_handle_observation",
        )
        self.record("motion_reconciliation_result", dict(reconciliation))
        if reconciliation.get("status") == "failed":
            self._reject_candidate_after_failed_motion_reconciliation(reconciliation)
        return True

    def _reject_candidate_after_failed_motion_reconciliation(
        self,
        reconciliation: JsonDict,
    ) -> None:
        policy = self.grasp_candidate_policy()
        if not isinstance(policy, dict) or policy.get("status") != "active":
            return
        active = policy.get("active_candidate")
        if not isinstance(active, dict):
            return
        candidate_id = str(reconciliation.get("candidate_id") or "")
        if not candidate_id or candidate_id != str(active.get("id") or ""):
            return
        advanced = self._apply_anygrasp_candidate_rejection(
            policy=policy,
            active=active,
            rejection={
                "source": "reconciled_candidate_motion_rejected",
                "target_tool": reconciliation.get("tool"),
                "reason": "reconciled_target_not_reached",
            },
        )
        if advanced:
            rejection = {
                "source": "reconciled_candidate_motion_rejected",
                "target_tool": reconciliation.get("tool"),
                "grasp_stage": reconciliation.get("grasp_stage"),
                "reason": "reconciled_target_not_reached",
                "execution_started": True,
            }
            self._schedule_grasp_recovery(
                rejection=rejection,
                candidate_id=candidate_id,
            )
            self.facts.pop(ARTICULATED_ATTACHMENT_PROBE_KEY, None)
            self.facts.pop(GRASP_EXECUTION_KEY, None)
            self.facts.pop(ATTACHMENT_GATE_KEY, None)

    def _advance_reconciled_grasp_stage(self, stage: str) -> None:
        execution = self.grasp_execution()
        if not isinstance(execution, dict) or execution.get("stage") != stage:
            return
        if stage == "open":
            compiled = execution.get("compiled_grasp")
            contact = compiled.get("contact_pose") if isinstance(compiled, dict) else None
            if isinstance(contact, dict):
                execution.update(
                    {
                        "stage": "contact",
                        "required_action": {
                            "name": "move_to",
                            "parameters": {
                                "target_pose": _pose_for_epoch(contact, self.scene_epoch())
                            },
                        },
                    }
                )
        elif stage == "contact":
            self._mark_active_candidate_accepted(source="reconciled_model_contact_reached")
            execution.update(
                {
                    "stage": "close",
                    "required_action": {
                        "name": "gripper_control",
                        "parameters": {"position": 0},
                    },
                }
            )
        elif stage == "close":
            gate = {
                "schema_version": "openeta.attachment_gate.v2",
                "status": "stopped_requires_human",
                "verdict": "UNKNOWN",
                "candidate_id": execution.get("candidate_id"),
                "compiled_grasp_id": execution.get("compiled_grasp_id"),
                "scene_epoch": self.scene_epoch(),
                "assessment_reason": "native_contact_attach_receipt_unavailable",
            }
            self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
                gate, source="motion_reconciliation"
            )
            execution.update(
                {
                    "status": "stopped_requires_human",
                    "stage": "attachment_unknown",
                    "required_action": None,
                    "stop_reason": "native_contact_attach_receipt_unavailable",
                }
            )
        elif stage == "probe":
            articulated_probe = self.articulated_attachment_probe()
            if isinstance(articulated_probe, dict):
                articulated_probe.update(
                    {
                        "status": "completed",
                        "completed_at_s": time.time(),
                        "last_attempt_status": "reconciled",
                    }
                )
                self.facts[ARTICULATED_ATTACHMENT_PROBE_KEY] = _memory_fact_entry(
                    articulated_probe, source="motion_reconciliation"
                )
                self._advance_articulated_probe_to_attachment(execution)
                return
        execution["scene_epoch"] = self.scene_epoch()
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution, source="motion_reconciliation"
        )

    def _append_transition_ledger(self, action: EnvAction) -> None:
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        request = request if isinstance(request, dict) else {}
        name = str(request.get("name") or "")
        call = _tool_call(action, name)
        row = {
            "index": len(self.transition_ledger()),
            "timestamp_s": time.time(),
            "scene_epoch": self.scene_epoch(),
            "tool": name,
            "effect": "world_mutating"
            if name in {"move_to", "gripper_control", "follow_eef_trajectory"}
            else "other",
            "grasp_stage": (self.grasp_execution() or {}).get("stage"),
            "candidate_id": _parameters_grasp_candidate_id(request.get("parameters") or {}),
            "verdict": _transition_call_verdict(call),
        }
        rows = [*self.transition_ledger(), row][-TRANSITION_LEDGER_LIMIT:]
        self.facts[TRANSITION_LEDGER_KEY] = _memory_fact_entry(
            {"rows": rows},
            source="runtime_transition_ledger",
        )

    def record_environment_receipt(
        self,
        *,
        reward: object,
        terminated: bool,
        truncated: bool,
        info: JsonDict | None = None,
    ) -> None:
        try:
            reward_value = float(reward)
        except (TypeError, ValueError):
            reward_value = 0.0
        receipt = {
            "reward": reward_value,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": dict(info or {}),
            "scene_epoch": self.scene_epoch(),
            "timestamp_s": time.time(),
        }
        self.facts["latest_environment_receipt"] = _memory_fact_entry(
            receipt,
            source="environment_step",
        )
        self.record("environment_receipt", receipt)
        rows = self.transition_ledger()
        rows.append(
            {
                "index": len(rows),
                "timestamp_s": receipt["timestamp_s"],
                "scene_epoch": self.scene_epoch(),
                "tool": "environment_receipt",
                "reward": reward_value,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "verdict": "PASS" if reward_value > 0 else "UNKNOWN",
            }
        )
        self.facts[TRANSITION_LEDGER_KEY] = _memory_fact_entry(
            {"rows": rows[-TRANSITION_LEDGER_LIMIT:]},
            source="runtime_transition_ledger",
        )
        self._save_working_memory()

    def _latest_action_command(self) -> JsonDict:
        for event in reversed(self.events[:-1]):
            if event.event_type == "action":
                command = event.payload.get("command")
                return dict(command) if isinstance(command, dict) else {}
            if event.event_type in {"observation", "episode_start"}:
                break
        return {}

    def _accept_anygrasp_candidate_after_motion(self, action: EnvAction) -> bool:
        execution = self.grasp_execution()
        if isinstance(execution, dict) and execution.get("status") == "required":
            return False
        policy = self.anygrasp_candidate_policy()
        if policy is None or str(policy.get("status") or "") != "active":
            return False
        active = policy.get("active_candidate")
        if not isinstance(active, dict):
            return False
        if not _candidate_linked_motion_succeeded(
            action,
            active_candidate_id=str(active.get("id") or ""),
            artifacts=self.artifacts,
        ):
            return False
        policy.update(
            {
                "status": "accepted",
                "accepted_candidate": dict(active),
                "accepted_at_s": time.time(),
            }
        )
        source_tool = str(policy.get("source_tool") or "anygrasp")
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source=f"{source_tool}_motion_accepted",
        )
        self.record(
            f"{source_tool}_candidate_accepted",
            {
                "result_id": policy.get("result_id"),
                "candidate_id": active.get("id"),
                "rank": policy.get("active_rank"),
                "score": active.get("score"),
            },
        )
        return True

    def delete_memory(self, key: str, *, namespace: str = "all") -> JsonDict:
        deleted: JsonDict = {}
        if namespace in {"all", "facts"}:
            deleted["facts"] = self.facts.pop(key, None) is not None
        if namespace in {"all", "artifacts"}:
            deleted["artifacts"] = self.artifacts.pop(key, None) is not None
        if namespace in {"all", "skill_notes"}:
            deleted["skill_notes"] = self.skill_notes.pop(key, None) is not None
        self.record("memory_deleted", {"key": key, "namespace": namespace, "deleted": deleted})
        self._save_working_memory()
        return deleted

    def clear_working_memory(self) -> None:
        """Clear persisted working memory without deleting session trace files."""

        self.facts.clear()
        self.artifacts.clear()
        self.skill_notes.clear()
        self.compact_summary = ""
        self.record("working_memory_cleared", {})
        self._save_working_memory()

    def compact(self, *, max_events: int = 8) -> str:
        recent = self.recent_events(max_events)
        conversation_checkpoint = self.conversation.compact()
        self._append_conversation_record(checkpoint_record(conversation_checkpoint))
        parts = [
            f"task={self.task}",
            f"current_user_request={self.current_user_request}",
            f"facts={list(self.facts)}",
            f"artifacts={list(self.artifacts)}",
            f"skill_notes={list(self.skill_notes)}",
            "recent_events=" + ",".join(event.event_type for event in recent),
            "conversation="
            f"{conversation_checkpoint['source_item_count']}->"
            f"{conversation_checkpoint['retained_item_count']}",
        ]
        self.compact_summary = "; ".join(parts)
        self.record("memory_compacted", {"summary": self.compact_summary})
        self._save_working_memory()
        return self.compact_summary

    def recent_events(self, limit: int = 8) -> list[MemoryEvent]:
        if limit <= 0:
            return []
        return self.events[-limit:]

    def latest_human_interaction(self) -> JsonDict | None:
        """Return the latest operator answer from the current episode."""

        for event in reversed(self.events):
            if event.event_type == "episode_start":
                break
            if event.event_type != "human_answer":
                continue
            question = event.payload.get("question")
            answer = event.payload.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                return None
            interaction: JsonDict = {
                "answer": _compact_value(answer.strip()),
                "timestamp_s": event.timestamp_s,
            }
            if isinstance(question, str) and question.strip():
                interaction["question"] = _compact_value(question.strip())
            return interaction
        return None

    def latest_guidance_interaction(self) -> JsonDict | None:
        """Return the latest non-human guidance answer with explicit provenance."""

        for event in reversed(self.events):
            if event.event_type == "episode_start":
                break
            if event.event_type != "guidance_answer":
                continue
            answer = event.payload.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                return None
            interaction: JsonDict = {
                "answer": _compact_value(answer.strip()),
                "source": "guidance_agent",
                "timestamp_s": event.timestamp_s,
            }
            question = event.payload.get("question")
            if isinstance(question, str) and question.strip():
                interaction["question"] = _compact_value(question.strip())
            return interaction
        return None

    def planning_context(self, *, max_events: int = 8) -> JsonDict:
        """Return compact context suitable for a planner prompt or policy."""

        return {
            "session_id": self.session_id,
            "task": self.current_user_request or self.task,
            "session_initial_task": self.task,
            "current_user_request": self.current_user_request,
            "active_environment_task": self.active_environment_task(),
            "task_completion_evidence": self.task_completion_evidence(),
            "multi_sort_progress": _memory_fact_value(
                self.facts.get(MULTI_SORT_PROGRESS_KEY)
            ),
            "work_order": _memory_fact_value(self.facts.get(WORK_ORDER_KEY)),
            "conversation": self.conversation.planning_context(max_items=0),
            "metadata": self.metadata,
            "selection_obligation": self.pending_sam3_selection(),
            "selected_sam3_detection": self.selected_sam3_detection(),
            "reference_localization_obligation": self.pending_reference_localization(),
            "target_asset_reference": self.target_asset_reference(),
            "reference_localization_failure": self.reference_localization_failure(),
            "sam3_no_detection": self.sam3_no_detection(),
            "sam3_semantic_state": self.sam3_semantic_state(),
            "grasp_candidate_policy": self.anygrasp_candidate_policy(),
            "retained_targeted_grasp": self.retained_targeted_grasp(),
            "articulated_attachment_probe": self.articulated_attachment_probe(),
            "grasp_execution": self.grasp_execution(),
            "grasp_recovery": self.grasp_recovery(),
            "grasp_estimation_recovery": self.grasp_estimation_recovery(),
            "gripper_command_state": self.gripper_command_state(),
            "attachment_gate": self.attachment_gate(),
            "placement_release": self.placement_release(),
            "placement_candidate_policy": self.placement_candidate_policy(),
            "placement_object_detection": self.placement_object_detection(),
            "placement_region_detection": self.placement_region_detection(),
            "frozen_placement_goal_pool": self.frozen_placement_goal_pool(),
            "motion_reconciliation": self.motion_reconciliation(),
            "scene_epoch": self.scene_epoch(),
            "transition_ledger": self.transition_ledger()[-12:],
            "latest_environment_receipt": self.latest_environment_receipt(),
            "planning_scene_target_pose_sync": (self.planning_scene_target_pose_sync()),
            "latest_human_interaction": self.latest_human_interaction(),
            "latest_guidance_interaction": self.latest_guidance_interaction(),
            "working_memory": {
                "facts": {
                    key: value
                    for key, value in self.facts.items()
                    if key
                    not in {
                        PENDING_SAM3_SELECTION_KEY,
                        SELECTED_SAM3_DETECTION_KEY,
                        PENDING_REFERENCE_LOCALIZATION_KEY,
                        REFERENCE_LOCALIZATION_FAILURE_KEY,
                        TARGET_ASSET_REFERENCE_KEY,
                        SAM3_NO_DETECTION_KEY,
                        SAM3_SEMANTIC_STATE_KEY,
                        GRASP_CANDIDATE_POLICY_KEY,
                        LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY,
                        ARTICULATED_ATTACHMENT_PROBE_KEY,
                        GRASP_EXECUTION_KEY,
                        GRASP_RECOVERY_KEY,
                        GRASP_ESTIMATION_RECOVERY_KEY,
                        GRIPPER_COMMAND_STATE_KEY,
                        ATTACHMENT_GATE_KEY,
                        PLACEMENT_RELEASE_KEY,
                        TASK_COMPLETION_EVIDENCE_KEY,
                        MOTION_RECONCILIATION_KEY,
                        SCENE_EPOCH_KEY,
                        TRANSITION_LEDGER_KEY,
                        ACTIVE_ENVIRONMENT_TASK_KEY,
                    }
                },
                "artifacts": {
                    key: summarize_memory_artifact(value) for key, value in self.artifacts.items()
                },
                "skill_notes": self.skill_notes,
                "compact_summary": self.compact_summary,
            },
            "recent_events": [
                {
                    "type": event.event_type,
                    "timestamp_s": event.timestamp_s,
                    "payload": summarize_event_payload(event.payload),
                }
                for event in self.recent_events(max_events)
            ],
        }

    def _load_working_memory(self) -> None:
        if self.store is None:
            return
        memory = self.store.load_working_memory()
        facts = memory.get("facts", {})
        artifacts = memory.get("artifacts", {})
        skill_notes = memory.get("skill_notes", {})
        if isinstance(facts, dict):
            self.facts = dict(facts)
        if isinstance(artifacts, dict):
            self.artifacts = dict(artifacts)
        if isinstance(skill_notes, dict):
            self.skill_notes = {
                str(skill): list(notes) if isinstance(notes, list) else []
                for skill, notes in skill_notes.items()
            }
        self.compact_summary = str(memory.get("compact_summary", ""))

    def model_conversation_messages(
        self,
        *,
        omit_known_successful_execution_feedback: bool = False,
    ) -> list[JsonDict]:
        """Return provider chat messages with optional success-feedback elision."""

        return self.conversation.model_messages(
            omit_known_successful_execution_feedback=(
                omit_known_successful_execution_feedback
            )
        )

    def conversation_checkpoint_summary(self) -> str:
        summary = self.conversation.checkpoint.get("summary")
        return summary if isinstance(summary, str) else ""

    def _append_conversation_record(self, record: JsonDict) -> None:
        if self.store is not None:
            self.store.append_conversation_record(record)

    def _reconstruct_legacy_conversation(self, rows: list[JsonDict]) -> None:
        """Recover user turns from traces created before conversation.jsonl existed."""

        seen: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            event_type = str(row.get("event_type") or "")
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            if event_type == "user_message":
                text = payload.get("text")
            elif event_type in {"session_start", "episode_start"}:
                text = payload.get("task")
            else:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            normalized = text.strip()
            if seen and seen[-1] == normalized:
                continue
            item = self.conversation.begin_user_turn(normalized, source="legacy_trace")
            seen.append(item.content)

    def _save_working_memory(self) -> None:
        if self.store is not None:
            self.store.save_working_memory(self)


def _grasp_candidate_sort_key(candidate: JsonDict) -> tuple[int, int, float, int]:
    """Prefer host-proven physical order, then the provider's native score."""

    try:
        score = float(candidate.get("score"))
    except (TypeError, ValueError):
        score = float("-inf")
    try:
        rank = int(candidate.get("rank"))
    except (TypeError, ValueError):
        rank = 1_000_000
    physical_rank: int | None = None
    if candidate.get("grasp_place_joint_qualified") is True:
        for key in (
            "grasp_place_frontier_quality_rank",
            "grasp_place_physical_quality_rank",
        ):
            value = candidate.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                physical_rank = value
                break
    if physical_rank is not None:
        return (0, physical_rank, -score, rank)
    moveit_rank = candidate.get("moveit_physical_quality_rank")
    if (
        candidate.get("moveit_l5_qualified") is True
        and isinstance(moveit_rank, int)
        and not isinstance(moveit_rank, bool)
        and moveit_rank >= 0
    ):
        return (1, moveit_rank, -score, rank)
    return (2, 0, -score, rank)


def _normalize_grasp_geometry_family(value: object) -> str:
    family = str(value or "").strip().lower()
    return "articulated_handle" if family == "drawer_handle" else family


def _active_task_text(memory: AgentMemory) -> str:
    active = memory.active_environment_task()
    task = active.get("task") if isinstance(active, dict) else None
    if isinstance(task, str) and task.strip():
        return task.strip()
    return str(memory.current_user_request or memory.task or "").strip()


def _articulated_handle_candidate_policy(
    *,
    candidates: list[JsonDict],
    selected_target: JsonDict,
    task: str,
    source_tool: str,
    source_rgb: str,
    source_mask: str,
    camera_frame_id: str,
    scene_epoch: int,
    camera_extrinsics: JsonDict | None,
) -> JsonDict | None:
    """Build host-owned mode queues only for a current targeted handle grasp."""

    if source_tool not in {"grasp_pose_estimate", "anygrasp"} or not candidates:
        return None
    if not camera_frame_id or not isinstance(camera_extrinsics, dict):
        return None
    try:
        selected_epoch = int(selected_target.get("scene_epoch"))
    except (TypeError, ValueError):
        return None
    if selected_epoch != scene_epoch:
        return None
    selected_image = str(selected_target.get("source_image") or "")
    selected_mask = str(selected_target.get("mask_ref") or "")
    if not selected_image or not source_rgb or not _same_local_file(selected_image, source_rgb):
        return None
    if not selected_mask or not source_mask or selected_mask != source_mask:
        return None
    geometry_family = _normalize_grasp_geometry_family(
        selected_target.get("target_geometry_family")
    )
    target_is_handle = _selected_target_is_handle(selected_target)
    if _selected_target_is_non_articulated_handle(selected_target):
        return None
    if geometry_family != "articulated_handle" and not (
        _is_articulated_container_task(task) and target_is_handle
    ):
        return None
    try:
        from agent.tools.grasp_geometry import (
            GraspGeometryError,
            camera_optical_forward_world,
            grasp_candidate_approach_world,
        )

        optical_forward = camera_optical_forward_world(camera_extrinsics)
        grouped: dict[str, list[JsonDict]] = {
            mode: [] for mode in ARTICULATED_HANDLE_APPROACH_MODES
        }
        for candidate in candidates:
            approach_world = grasp_candidate_approach_world(candidate, camera_extrinsics)
            mode = _classify_articulated_handle_approach(
                approach_world,
                optical_forward_world=optical_forward,
            )
            candidate["approach_mode"] = mode
            candidate["native_approach_world_xyz"] = [
                round(float(value), 6) for value in approach_world
            ]
            grouped[mode].append(candidate)
    except (TypeError, ValueError, GraspGeometryError):
        return None
    ordered_modes = [mode for mode in ARTICULATED_HANDLE_APPROACH_MODES if grouped[mode]]
    if not ordered_modes:
        return None
    active_mode = ordered_modes[0]
    active_candidates = grouped[active_mode]
    active = dict(active_candidates[0])
    return {
        "interaction_family": "articulated_handle",
        "normalized_target_geometry_family": "articulated_handle",
        "approach_mode_order": ordered_modes,
        "approach_mode_queues": {
            mode: [str(candidate.get("id")) for candidate in grouped[mode]]
            for mode in ordered_modes
        },
        "active_approach_mode": active_mode,
        "active_mode_rank": 0,
        "active_rank": int(active.get("rank") or 0),
        "active_candidate": active,
        "remaining_candidate_ids": [
            str(candidate.get("id")) for candidate in active_candidates[1:]
        ],
        "candidate_attempt_count": 0,
        "mode_attempt_count": 0,
        "mode_max_candidate_attempts": GRASP_CANDIDATE_MAX_ATTEMPTS,
        "compile_hints": {
            "target_geometry_family": "articulated_handle",
            "approach_mode": active_mode,
        },
        "approach_classification": {
            "camera_frame_id": camera_frame_id,
            "scene_epoch": scene_epoch,
            "method": "world_approach_with_matching_camera_extrinsics",
        },
    }


def _classify_articulated_handle_approach(
    approach_world: list[float],
    *,
    optical_forward_world: list[float],
) -> str:
    downward_alignment = -float(approach_world[2])
    front_alignment = abs(
        sum(
            float(approach_world[index]) * float(optical_forward_world[index]) for index in range(3)
        )
    )
    if downward_alignment >= 0.5:
        return "top_down"
    if front_alignment >= 0.5:
        return "front"
    return "side"


def _selected_target_is_handle(selected_target: JsonDict) -> bool:
    values = [
        selected_target.get("target_prompt"),
        selected_target.get("label"),
        selected_target.get("class_name"),
        selected_target.get("name"),
    ]
    return any(
        any(
            token == "handle" or token.endswith("_handle")
            for token in _grasp_word_tokens(str(value or ""))
        )
        for value in values
    )


def _selected_target_is_non_articulated_handle(selected_target: JsonDict) -> bool:
    values = " ".join(
        str(selected_target.get(field) or "")
        for field in ("target_prompt", "label", "class_name", "name")
    )
    tokens = set(_grasp_word_tokens(values))
    portable_handles = {
        "mug",
        "cup",
        "basket",
        "bucket",
        "bag",
        "suitcase",
        "pan",
        "pot",
    }
    non_handle_controls = {"button", "knob", "dial", "lever", "switch"}
    return bool(tokens & portable_handles) or bool(tokens & non_handle_controls)


def _is_articulated_container_task(task: str) -> bool:
    tokens = set(_grasp_word_tokens(task))
    containers = {
        "drawer",
        "drawers",
        "microwave",
        "cabinet",
        "cupboard",
        "fridge",
        "refrigerator",
        "door",
    }
    if not tokens & containers:
        return False
    explicit_interaction = bool(tokens & {"open", "pull"})
    interior_access = bool(tokens & {"in", "inside", "into", "interior"})
    return explicit_interaction or interior_access


def _grasp_word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _same_local_file(left: str, right: str) -> bool:
    if left == right:
        return True
    try:
        left_path = Path(left)
        right_path = Path(right)
        if not left_path.is_file() or not right_path.is_file():
            return False
        if left_path.stat().st_size != right_path.stat().st_size:
            return False
        return left_path.read_bytes() == right_path.read_bytes()
    except OSError:
        return False


def _highest_refinable_candidate(policy: JsonDict) -> JsonDict | None:
    candidates = policy.get("candidates")
    rejected = policy.get("rejected_candidates")
    if not isinstance(candidates, list) or not isinstance(rejected, list):
        return None
    refinable_ids = {
        str(item.get("candidate_id") or "")
        for item in rejected
        if isinstance(item, dict)
        and str(item.get("recovery_class") or "") in {"perception_refinable", "uncertain_review"}
    }
    eligible = [
        dict(candidate)
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("id") or "") in refinable_ids
    ]
    return min(eligible, key=_grasp_candidate_sort_key) if eligible else None


def _policy_grasp_source(policy: JsonDict) -> JsonDict:
    source = policy.get("grasp_source")
    if isinstance(source, dict):
        return dict(source)
    return {
        "mode": "targeted",
        "rgb": policy.get("source_rgb"),
        "depth": policy.get("source_depth"),
        "camera_frame_id": policy.get("camera_frame_id"),
    }


def _append_refinable_fallback_candidate(
    candidates: list[JsonDict],
    entry: JsonDict,
) -> None:
    candidate = entry.get("candidate")
    if not isinstance(candidate, dict):
        return
    identity = (
        str(entry.get("source_result_id") or ""),
        str(candidate.get("id") or ""),
    )
    for item in candidates:
        stored = item.get("candidate")
        stored_identity = (
            str(item.get("source_result_id") or ""),
            str(stored.get("id") or "") if isinstance(stored, dict) else "",
        )
        if stored_identity == identity:
            return
    candidates.append(dict(entry))


def _candidate_fits_gripper(
    candidate: JsonDict,
    *,
    max_gripper_width_m: float,
) -> bool:
    try:
        width = float(candidate.get("width"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(width) and 0.0 < width <= max_gripper_width_m


def _policy_max_gripper_width(policy: JsonDict, *, default: float) -> float:
    try:
        value = float(policy.get("physical_width_limit_m"))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and 0.0 < value <= 0.2 else default


def _fallback_candidate_capabilities(
    entry: JsonDict,
    *,
    default: JsonDict,
) -> JsonDict:
    capabilities = entry.get("grasp_calibration")
    capabilities = dict(capabilities) if isinstance(capabilities, dict) else {}
    max_width = _policy_max_gripper_width(
        {"physical_width_limit_m": capabilities.get("max_gripper_width_m")},
        default=float(default["max_gripper_width_m"]),
    )
    return {
        "calibration_id": str(
            capabilities.get("calibration_id") or default.get("calibration_id") or ""
        ),
        "profile_path": str(capabilities.get("profile_path") or default.get("profile_path") or ""),
        "max_gripper_width_m": max_width,
    }


def _parameters_grasp_candidate_id(parameters: JsonDict) -> str:
    for key in ("source_grasp_id", "grasp_candidate_id"):
        value = parameters.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("camera_pose", "target_pose", "pose", "eef_pose"):
        pose = parameters.get(key)
        if not isinstance(pose, dict):
            continue
        for id_key in ("id", "source_grasp_id", "grasp_candidate_id"):
            value = pose.get(id_key)
            if isinstance(value, str) and value:
                return value
    target_parameters = parameters.get("target_parameters")
    if isinstance(target_parameters, dict):
        return _parameters_grasp_candidate_id(target_parameters)
    trajectory = parameters.get("trajectory")
    if isinstance(trajectory, list):
        for pose in trajectory:
            if not isinstance(pose, dict):
                continue
            value = pose.get("source_grasp_id") or pose.get("grasp_candidate_id")
            if isinstance(value, str) and value:
                return value
    return ""


def _candidate_linked_rejection(
    action: EnvAction,
    *,
    active_candidate_id: str,
    artifacts: dict[str, JsonDict],
    final_refinable_fallback: bool = False,
) -> JsonDict | None:
    if not active_candidate_id:
        return None
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict):
        request = {}
    request_name = str(request.get("name") or command.get("request_name") or "")
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    candidate_id = _parameters_grasp_candidate_id(parameters)
    if not candidate_id:
        candidate_id = _matching_latest_world_pose_candidate(parameters, artifacts)
    if candidate_id != active_candidate_id:
        return None

    failed_safety = next(
        (
            call
            for call in command.get("safety_checks", []) or []
            if isinstance(call, dict) and _safety_call_rejects_candidate(call)
        ),
        None,
    )
    if isinstance(failed_safety, dict):
        return {
            "source": "safety_check_rejected",
            "checker": failed_safety.get("name"),
            "target_tool": request_name,
            "reason": _call_failure_reason(failed_safety),
        }

    if request_name == "compile_grasp_seed":
        failed_compile = next(
            (
                call
                for call in command.get("tool_calls", []) or []
                if isinstance(call, dict)
                and str(call.get("name") or "") == request_name
                and _grasp_seed_compile_rejects_candidate(call)
            ),
            None,
        )
        if isinstance(failed_compile, dict):
            recovery_metadata = _grasp_seed_compile_recovery_metadata(failed_compile)
            return {
                "source": "grasp_seed_geometry_rejected",
                "target_tool": request_name,
                "reason": _call_failure_reason(failed_compile),
                **recovery_metadata,
            }
        if final_refinable_fallback:
            terminal_compile = next(
                (
                    call
                    for call in command.get("tool_calls", []) or []
                    if isinstance(call, dict)
                    and str(call.get("name") or "") == request_name
                    and not _call_result_success(call)
                ),
                None,
            )
            if isinstance(terminal_compile, dict):
                return {
                    "source": "final_fallback_compile_failed",
                    "target_tool": request_name,
                    "reason": _call_failure_reason(terminal_compile),
                }
        return None

    if request_name not in {"move_to", "follow_eef_trajectory"}:
        return None
    failed_review = next(
        (
            call
            for call in command.get("tool_calls", []) or []
            if isinstance(call, dict)
            and str(call.get("name") or "") == request_name
            and _independent_motion_review_rejects_candidate(call)
        ),
        None,
    )
    if isinstance(failed_review, dict):
        return {
            "source": "independent_host_stage_review_rejected",
            "target_tool": request_name,
            "reason": _call_failure_reason(failed_review),
            **_independent_review_recovery_metadata(failed_review),
        }
    failed_tool = next(
        (
            call
            for call in command.get("tool_calls", []) or []
            if isinstance(call, dict)
            and str(call.get("name") or "") == request_name
            and _motion_call_rejects_candidate(call)
        ),
        None,
    )
    if not isinstance(failed_tool, dict):
        return None
    return {
        "source": "candidate_motion_rejected",
        "target_tool": request_name,
        "reason": _call_failure_reason(failed_tool),
        **_motion_rejection_fingerprint(failed_tool),
    }


def _motion_rejection_fingerprint(call: JsonDict) -> JsonDict:
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    if not isinstance(details, dict):
        return {}
    outputs = details.get("outputs")
    sources = [details]
    if isinstance(outputs, dict):
        sources.extend(
            source
            for source in (
                outputs,
                outputs.get("response"),
                outputs.get("motion_summary"),
            )
            if isinstance(source, dict)
        )
    receipt = details.get("environment_receipt")
    if isinstance(receipt, dict):
        sources.append(receipt)
    evidence: JsonDict = {}
    for source in sources:
        fingerprint = str(source.get("request_fingerprint") or "").strip()
        if fingerprint and "request_fingerprint" not in evidence:
            evidence["request_fingerprint"] = fingerprint
        execution_started = source.get("execution_started")
        if isinstance(execution_started, bool):
            evidence.setdefault("execution_started", execution_started)
        revision = source.get("planning_scene_revision")
        if isinstance(revision, int) and not isinstance(revision, bool):
            evidence.setdefault("planning_scene_revision", revision)
    revision = evidence.get("planning_scene_revision")
    receipt = _environment_receipt(call)
    if _trusted_detached_target_displacement(receipt):
        sync = receipt["planning_scene_target_pose_sync"]
        evidence.update(
            {
                "detached_target_rebase_safe": True,
                "source_planning_scene_revision": sync.get("source_revision"),
            }
        )
    if (
        isinstance(revision, int)
        and not isinstance(revision, bool)
        and _trusted_open_unattached_grasp_motion_failure(
            receipt,
            planning_scene_revision=revision,
        )
    ):
        restart = receipt.get("current_state_restart")
        evidence.update(
            {
                "current_state_requalification_safe": True,
                "current_state_restart_sha256": hashlib.sha256(
                    json.dumps(
                        restart,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    return evidence


def _host_stage_motion_review_rejection(
    action: EnvAction,
    *,
    active_candidate_id: str,
    execution: JsonDict | None,
) -> JsonDict | None:
    if (
        not active_candidate_id
        or not isinstance(execution, dict)
        or str(execution.get("candidate_id") or "") != active_candidate_id
        or str(execution.get("stage") or "") not in {"open", "contact", "close"}
    ):
        return None
    required = execution.get("required_action")
    if not isinstance(required, dict):
        return None
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict):
        return None
    request_name = str(request.get("name") or "")
    parameters = request.get("parameters")
    if (
        request_name != str(required.get("name") or "")
        or not isinstance(parameters, dict)
        or parameters != required.get("parameters")
    ):
        return None
    call = _tool_call(action, request_name)
    if not isinstance(call, dict) or not _independent_motion_review_rejects_candidate(call):
        return None
    return {
        "source": "independent_host_stage_review_rejected",
        "target_tool": request_name,
        "grasp_stage": execution.get("stage"),
        "reason": _call_failure_reason(call),
        **_independent_review_recovery_metadata(call),
    }


def _host_stage_close_rejection(
    action: EnvAction,
    *,
    active_candidate_id: str,
    execution: JsonDict | None,
) -> JsonDict | None:
    """Reject one grasp candidate after a known failed close; never retry it."""

    if (
        not active_candidate_id
        or not isinstance(execution, dict)
        or execution.get("stage") != "close"
        or str(execution.get("candidate_id") or "") != active_candidate_id
    ):
        return None
    required = execution.get("required_action")
    if not isinstance(required, dict) or not _action_matches(action, required):
        return None
    call = _tool_call(action, "gripper_control")
    if not isinstance(call, dict) or _call_result_success(call):
        return None
    outputs = _tool_call_outputs(call)
    if str(outputs.get("motion_outcome") or "").lower() == "unknown" or (
        outputs.get("reconciliation_required") is True
    ):
        return None
    return {
        "source": "host_gripper_close_failed",
        "target_tool": "gripper_control",
        "grasp_stage": "close",
        "reason": _call_failure_reason(call),
        **_motion_rejection_fingerprint(call),
    }


def _candidate_linked_grasp_outcome_rejection(
    action: EnvAction,
    *,
    active_candidate_id: str,
    articulated_probe: JsonDict | None = None,
    attachment_gate: JsonDict | None = None,
    execution: JsonDict | None = None,
) -> JsonDict | None:
    if not active_candidate_id:
        return None
    if (
        not isinstance(articulated_probe, dict)
        or str(articulated_probe.get("status") or "") != "completed"
        or str(articulated_probe.get("candidate_id") or "") != active_candidate_id
    ):
        return None
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict):
        return None
    request_name = str(request.get("name") or "")
    if request_name not in {"gripper_control", "move_to", "follow_eef_trajectory"}:
        return None
    if (
        isinstance(attachment_gate, dict)
        and str(attachment_gate.get("verdict") or "").upper() == "FAIL"
        and isinstance(execution, dict)
        and execution.get("attachment_mode") == "articulated_handle"
    ):
        actions = execution.get("attachment_actions")
        fail_action = actions.get("fail") if isinstance(actions, dict) else None
        call = _tool_call(action, request_name)
        if (
            isinstance(fail_action, dict)
            and _action_matches(action, fail_action)
            and isinstance(call, dict)
            and _call_result_success(call)
        ):
            return {
                "source": "articulated_attachment_assessment_failed",
                "target_tool": request_name,
                "reason": str(
                    attachment_gate.get("assessment_reason")
                    or "Independent multi-view assessment found no attachment."
                ),
            }
    for call in command.get("tool_calls", []) or []:
        if not isinstance(call, dict) or str(call.get("name") or "") != request_name:
            continue
        supervision = _call_supervision(call)
        review = supervision.get("details") if isinstance(supervision, dict) else None
        if not isinstance(review, dict) or str(review.get("grasp_outcome") or "").lower() not in {
            "fail",
            "failed",
        }:
            continue
        if str(review.get("candidate_id") or "") != active_candidate_id:
            continue
        if request_name == "gripper_control":
            parameters = request.get("parameters")
            if not isinstance(parameters, dict):
                continue
            if _binary_gripper_position(
                parameters.get("position")
            ) != 1 or not _call_result_success(call):
                continue
        elif _call_result_success(call) or supervision.get("allowed") is not False:
            continue
        return {
            "source": "independent_grasp_outcome_rejected",
            "target_tool": request_name,
            "reason": str(supervision.get("reason") or "Visual grasp verification failed."),
        }
    return None


def _call_supervision(call: JsonDict) -> JsonDict | None:
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    if not isinstance(details, dict):
        return None
    supervision = details.get("supervision")
    if isinstance(supervision, dict):
        return supervision
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        return None
    supervision = outputs.get("supervision")
    return supervision if isinstance(supervision, dict) else None


def _successful_gripper_close_command(command: JsonDict) -> bool:
    request = command.get("request")
    if not isinstance(request, dict) or str(request.get("name") or "") != "gripper_control":
        return False
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        return False
    if _binary_gripper_position(parameters.get("position")) != 0:
        return False
    return any(
        isinstance(call, dict)
        and str(call.get("name") or "") == "gripper_control"
        and _call_result_success(call)
        for call in command.get("tool_calls", []) or []
    )


def _finite_xyz(value: object) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) >= 3
        and all(
            isinstance(component, int | float) and math.isfinite(float(component))
            for component in value[:3]
        )
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _optional_float(value: object) -> float | None:
    return float(value) if _finite_number(value) else None


def _candidate_linked_motion_succeeded(
    action: EnvAction,
    *,
    active_candidate_id: str,
    artifacts: dict[str, JsonDict],
) -> bool:
    if not active_candidate_id:
        return False
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict) or str(request.get("name") or "") != "move_to":
        return False
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        return False
    candidate_id = _parameters_grasp_candidate_id(parameters)
    if not candidate_id:
        candidate_id = _matching_latest_world_pose_candidate(parameters, artifacts)
    if candidate_id != active_candidate_id:
        return False
    return any(
        isinstance(call, dict)
        and str(call.get("name") or "") == "move_to"
        and str(call.get("status") or "") == "executed"
        and _call_result_success(call)
        and not _motion_call_rejects_candidate(call)
        for call in command.get("tool_calls", []) or []
    )


def _call_result_success(call: JsonDict) -> bool:
    result = call.get("result")
    return isinstance(result, dict) and bool(result.get("success"))


def _call_has_diagnostic(call: JsonDict, code: str) -> bool:
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    diagnostics = details.get("diagnostics") if isinstance(details, dict) else None
    return isinstance(diagnostics, list) and any(
        isinstance(diagnostic, dict) and diagnostic.get("code") == code
        for diagnostic in diagnostics
    )


def _tool_call(action: EnvAction, name: str) -> JsonDict | None:
    command = action.command if isinstance(action.command, dict) else {}
    return next(
        (
            call
            for call in command.get("tool_calls", []) or []
            if isinstance(call, dict) and str(call.get("name") or "") == name
        ),
        None,
    )


def _expanded_sam3_selection_calls(calls: object) -> list[JsonDict]:
    """Expose ordered batch children to the existing single-result state machine."""

    if not isinstance(calls, list):
        return []
    expanded: list[JsonDict] = []
    for call in calls:
        if not isinstance(call, dict) or str(call.get("name") or "") != "sam3":
            if isinstance(call, dict):
                expanded.append(call)
            continue
        outputs = _tool_call_outputs(call)
        children = outputs.get("semantic_batch_results")
        if (
            outputs.get("schema_version") != "openeta.sam3.assignment_batch.v1"
            or not isinstance(children, list)
        ):
            expanded.append(call)
            continue
        for child in children:
            if not isinstance(child, dict):
                continue
            result = child.get("result")
            if not isinstance(result, dict):
                continue
            details = result.get("details")
            parameters = details.get("parameters") if isinstance(details, dict) else None
            expanded.append(
                {
                    "name": "sam3",
                    "status": call.get("status"),
                    "parameters": dict(parameters) if isinstance(parameters, dict) else {},
                    "result": result,
                }
            )
    return expanded


def _successful_tool_call(action: EnvAction, name: str) -> JsonDict | None:
    call = _tool_call(action, name)
    if not isinstance(call, dict) or not _call_result_success(call):
        return None
    return call


def _tool_call_outputs(call: JsonDict) -> JsonDict:
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    if not isinstance(details, dict):
        return {}
    outputs = details.get("outputs")
    return dict(outputs) if isinstance(outputs, dict) else dict(details)


def _host_candidate_compilation_outputs(action: EnvAction, *, purpose: str) -> JsonDict | None:
    """Read one fail-closed host compilation event from a normal tool result."""

    command = action.command if isinstance(action.command, dict) else {}
    for call in command.get("tool_calls", []) or []:
        if not isinstance(call, dict) or not _call_result_success(call):
            continue
        result_outputs = _tool_call_outputs(call)
        event = result_outputs.get("host_candidate_compilation")
        if not isinstance(event, dict):
            continue
        compiled = event.get("compiled_seed")
        if (
            event.get("schema_version") != "openeta.host_candidate_compilation.v1"
            or event.get("event_type") != "candidate_compiled"
            or event.get("purpose") != purpose
            or event.get("execution_started") is not False
            or not isinstance(compiled, dict)
            or str(event.get("candidate_id") or "")
            != str(compiled.get("candidate_id") or compiled.get("placement_candidate_id") or "")
        ):
            continue
        return dict(compiled)
    return None


def _host_compilation_action(*, purpose: str, compiled: JsonDict) -> EnvAction:
    """Build an internal reducer input; it is never dispatched as an AgentTool."""

    candidate_id = str(compiled.get("placement_candidate_id") or compiled.get("candidate_id") or "")
    return EnvAction(
        action_type="host_transition",
        command={
            "tool_calls": [
                {
                    "name": "_host_candidate_transition",
                    "result": {
                        "success": True,
                        "details": {
                            "host_candidate_compilation": {
                                "schema_version": "openeta.host_candidate_compilation.v1",
                                "event_type": "candidate_compiled",
                                "purpose": purpose,
                                "candidate_id": candidate_id,
                                "execution_started": False,
                                "compiled_seed": dict(compiled),
                            }
                        },
                    },
                }
            ]
        },
    )


def _qualification_artifact_evidence(
    outputs: Mapping[str, Any],
    *,
    artifact_key: str,
    public_key: str = "qualification_evidence",
) -> JsonDict:
    """Load host-private exact proof while planner-visible memory keeps only a summary."""

    public = outputs.get(public_key)
    public = dict(public) if isinstance(public, Mapping) else {}
    artifact = outputs.get(artifact_key)
    if not isinstance(artifact, Mapping):
        return public
    path_value = artifact.get("path")
    if not isinstance(path_value, str) or not path_value.endswith(".json"):
        return public
    try:
        path = Path(path_value)
        if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            return public
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return public
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        not in {
            "openeta.moveit_candidate_funnel.v3",
            "openeta.moveit_candidate_qualification.v3",
            "openeta.moveit_candidate_funnel.v2",
            "openeta.moveit_candidate_qualification.v1",
        }
        or not isinstance(payload.get("results"), list)
    ):
        return public
    return payload


def _environment_receipt(call: JsonDict) -> JsonDict:
    """Return only the structured simulator receipt associated with this call."""

    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    if not isinstance(details, dict):
        return {}
    receipt = details.get("environment_receipt")
    if isinstance(receipt, dict):
        return dict(receipt)
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        return {}
    response = outputs.get("response")
    return dict(response) if isinstance(response, dict) else {}


def _trusted_retained_attachment_motion_failure(
    receipt: Mapping[str, object],
    *,
    planning_scene_revision: int,
) -> bool:
    """Return whether a failed placement motion has a safe, known restart state."""

    proof = receipt.get("physical_verification")
    attached = receipt.get("detachable_joint")
    child_proof = receipt.get("child_link_proof")
    expected_target_id = _receipt_native_target_id(receipt)
    if not (
        receipt.get("error_code")
        in {
            "MOTION_TARGET_NOT_REACHED",
            # MoveGroup CONTROL_FAILED is a terminal action result, not an
            # unknown timeout.  It is restartable only with the same fresh
            # end-state, scene-revision, and retained-attachment proofs below.
            "MOTION_EXECUTION_FAILED",
        }
        and receipt.get("execution_started") is True
        and receipt.get("motion_outcome") == "failed"
        and receipt.get("planning_scene_revision") == planning_scene_revision
        and isinstance(proof, Mapping)
        and proof.get("schema_version") == NATIVE_GRASP_SCHEMA_VERSION
        and proof.get("verdict") == "PASS"
        and proof.get("reason_code") == "NATIVE_GRASP_TARGET_HELD"
        and proof.get("grasp_confirmed") is True
        and bool(expected_target_id)
        and proof.get("target_id") == expected_target_id
        and isinstance(attached, Mapping)
        and attached.get("state") == "attached"
        and attached.get("target_id", expected_target_id) == expected_target_id
        and _valid_attachment_transform(receipt.get("attachment_transform"))
        and isinstance(child_proof, Mapping)
        and child_proof.get("prior_attachment_confirmed") is True
        and _trusted_failed_motion_end_state(
            receipt,
            planning_scene_revision=planning_scene_revision,
        )
    ):
        return False
    try:
        drift = float(child_proof.get("capture_relative_translation_m"))
        maximum = float(
            child_proof.get(
                "maximum_capture_relative_translation_m",
                NATIVE_GRASP_MAXIMUM_DRIFT_M,
            )
        )
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(drift)
        and math.isfinite(maximum)
        and 0.0
        <= drift
        <= min(
            maximum,
            NATIVE_GRASP_MAXIMUM_DRIFT_M,
        )
    )


def _trusted_failed_motion_end_state(
    receipt: Mapping[str, object],
    *,
    planning_scene_revision: int,
) -> bool:
    """Prove that a failed motion left the robot in a known restart state."""

    snapshot = receipt.get("observation_snapshot")
    if not (
        receipt.get("observation_fresh") is True
        and isinstance(snapshot, Mapping)
        and snapshot.get("schema_version") == "openeta.observation_snapshot.v1"
    ):
        return False
    observation = snapshot.get("observation")
    if not isinstance(observation, Mapping):
        return False
    metadata = observation.get("metadata")
    robot = observation.get("robot")
    if not (
        isinstance(metadata, Mapping)
        and metadata.get("planning_scene_revision") == planning_scene_revision
        and isinstance(robot, Mapping)
        and _finite_robot_end_state(robot)
    ):
        return False
    motion = receipt.get("motion")
    motion_end = motion.get("end") if isinstance(motion, Mapping) else None
    snapshot_end = robot.get("end_effector_pose")
    return (
        isinstance(motion, Mapping)
        and motion.get("reached_target") is False
        and isinstance(motion_end, Mapping)
        and isinstance(snapshot_end, Mapping)
        and _same_finite_eef_pose(motion_end, snapshot_end)
    )


def _trusted_open_unattached_grasp_motion_failure(
    receipt: Mapping[str, object],
    *,
    planning_scene_revision: int,
) -> bool:
    """Authorize frozen requalification from a failed pre-close arm state."""

    restart = receipt.get("current_state_restart")
    snapshot = receipt.get("observation_snapshot")
    observation = snapshot.get("observation") if isinstance(snapshot, Mapping) else None
    robot = observation.get("robot") if isinstance(observation, Mapping) else None
    gripper = robot.get("gripper_state") if isinstance(robot, Mapping) else None
    physical = receipt.get("physical_verification")
    detachable = receipt.get("detachable_joint")
    if not (
        receipt.get("error_code")
        in {"MOTION_TARGET_NOT_REACHED", "MOTION_EXECUTION_FAILED"}
        and receipt.get("execution_started") is True
        and receipt.get("motion_outcome") == "failed"
        and receipt.get("planning_scene_revision") == planning_scene_revision
        and isinstance(restart, Mapping)
        and restart.get("schema_version")
        == "openeta.gazebo.current_state_restart.v1"
        and restart.get("status") == "PASS"
        and restart.get("planning_scene_revision") == planning_scene_revision
        and isinstance(gripper, Mapping)
        and gripper.get("open") is True
        and _trusted_failed_motion_end_state(
            receipt,
            planning_scene_revision=planning_scene_revision,
        )
    ):
        return False
    if isinstance(physical, Mapping) and (
        physical.get("verdict") == "PASS"
        or physical.get("grasp_confirmed") is True
    ):
        return False
    if isinstance(detachable, Mapping) and detachable.get("state") != "detached":
        return False
    return True


def _trusted_detached_target_displacement(
    receipt: Mapping[str, object],
) -> bool:
    """Trust one native post-motion target rebase without model inference."""

    audit = receipt.get("detached_target_motion_audit")
    sync = receipt.get("planning_scene_target_pose_sync")
    detachable = receipt.get("detachable_joint")
    return bool(
        receipt.get("error_code") == "GRASP_CONTACT_TARGET_DISPLACED"
        and receipt.get("failure_class") == "detached_target_displacement"
        and receipt.get("candidate_rejection") is True
        and receipt.get("infrastructure_error") is False
        and receipt.get("execution_started") is True
        and receipt.get("motion_outcome") == "failed"
        and isinstance(audit, Mapping)
        and audit.get("schema_version")
        == "openeta.detached_target_motion_audit.v1"
        and audit.get("valid") is False
        and audit.get("reason_code") == "GRASP_CONTACT_TARGET_DISPLACED"
        and isinstance(sync, Mapping)
        and sync.get("schema_version")
        == "openeta.planning_scene_target_pose_sync.v1"
        and sync.get("operation") == "update_world_target"
        and sync.get("topology_unchanged") is True
        and sync.get("static_world_unchanged") is True
        and receipt.get("planning_scene_revision") == sync.get("revision")
        and isinstance(detachable, Mapping)
        and detachable.get("state") == "detached"
    )


def _finite_robot_end_state(value: Mapping[str, object]) -> bool:
    pose = value.get("end_effector_pose")
    joints = value.get("joint_positions")
    return (
        isinstance(pose, Mapping)
        and _finite_eef_pose(pose)
        and isinstance(joints, list)
        and len(joints) >= 7
        and all(_finite_number(joint) for joint in joints)
    )


def _finite_eef_pose(value: Mapping[str, object]) -> bool:
    frame = value.get("frame")
    xyz = value.get("xyz")
    quat = value.get("quat_xyzw")
    if not (
        isinstance(frame, str)
        and bool(frame.strip())
        and isinstance(xyz, list | tuple)
        and len(xyz) == 3
        and _finite_xyz(xyz)
        and isinstance(quat, list)
        and len(quat) == 4
        and all(_finite_number(component) for component in quat)
    ):
        return False
    norm = math.sqrt(sum(float(component) ** 2 for component in quat))
    return abs(norm - 1.0) <= 1e-5


def _same_finite_eef_pose(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> bool:
    if not (_finite_eef_pose(first) and _finite_eef_pose(second)):
        return False
    if first.get("frame") != second.get("frame"):
        return False
    first_xyz = first["xyz"]
    second_xyz = second["xyz"]
    first_quat = first["quat_xyzw"]
    second_quat = second["quat_xyzw"]
    return all(
        abs(float(left) - float(right)) <= 1e-9
        for left, right in zip(first_xyz, second_xyz, strict=True)
    ) and all(
        abs(float(left) - float(right)) <= 1e-9
        for left, right in zip(first_quat, second_quat, strict=True)
    )


def _trusted_native_attachment_proof(
    receipt: JsonDict,
    *,
    planning_scene_revision: int,
) -> tuple[bool, str, JsonDict]:
    """Validate the close-boundary native contact and attach acknowledgement.

    This proof authorizes attached-object planning; it does not prescribe a
    lift waypoint.  Every later MoveIt transport receipt revalidates native
    attachment retention, and the final release has its own stability proof.
    """

    proof = receipt.get("physical_verification")
    attached = receipt.get("detachable_joint")
    attachment_transform = receipt.get("attachment_transform")
    contact_gate = receipt.get("native_contact_gate")
    expected_target_id = _receipt_native_target_id(receipt)
    if not isinstance(contact_gate, dict) and isinstance(proof, dict):
        proof_evidence = proof.get("evidence")
        contact_gate = proof_evidence.get("gate") if isinstance(proof_evidence, dict) else None
    reasons: list[str] = []
    if receipt.get("ok") is not True:
        reasons.append("native_motion_not_successful")
    if not isinstance(proof, dict) or proof.get("schema_version") != NATIVE_GRASP_SCHEMA_VERSION:
        reasons.append("native_proof_schema_mismatch")
    if isinstance(proof, dict) and (
        proof.get("verdict") != "PASS"
        or proof.get("reason_code") != "NATIVE_GRASP_ATTACHMENT_CONFIRMED"
        or proof.get("grasp_confirmed") is not True
    ):
        reasons.append("native_proof_not_pass")
    if (
        not expected_target_id
        or isinstance(proof, dict)
        and str(proof.get("target_id") or "") != expected_target_id
    ):
        reasons.append("native_proof_target_mismatch")
    if (
        not isinstance(attached, dict)
        or attached.get("state") != "attached"
        or attached.get("target_id", expected_target_id) != expected_target_id
    ):
        reasons.append("native_attach_ack_missing")
    if not (
        isinstance(contact_gate, dict)
        and contact_gate.get("accepted") is True
        and contact_gate.get("reason_code") == "NATIVE_GRASP_CONTACT_TARGET_CONFIRMED"
        and isinstance(contact_gate.get("evidence"), dict)
        and contact_gate["evidence"].get("source") == "gazebo_native_contacts"
        and contact_gate["evidence"].get("target_id") == expected_target_id
    ):
        reasons.append("native_bilateral_contact_proof_missing")
    if not _valid_attachment_transform(attachment_transform):
        reasons.append("native_attachment_transform_missing")
    if (
        planning_scene_revision < 0
        or receipt.get("planning_scene_revision") != planning_scene_revision
    ):
        reasons.append("native_planning_scene_revision_mismatch")
    retained = {
        "receipt": dict(receipt),
        "physical_verification": dict(proof) if isinstance(proof, dict) else {},
        "native_contact_gate": (dict(contact_gate) if isinstance(contact_gate, dict) else {}),
        "detachable_joint": dict(attached) if isinstance(attached, dict) else {},
        "attachment_transform": (
            dict(attachment_transform) if isinstance(attachment_transform, dict) else {}
        ),
        "planning_scene_revision": receipt.get("planning_scene_revision"),
    }
    return (
        not reasons,
        reasons[0] if reasons else "trusted_native_contact_attach_ack",
        retained,
    )


def _receipt_native_target_id(receipt: Mapping[str, object]) -> str:
    binding = receipt.get("native_target_binding")
    if isinstance(binding, Mapping):
        return str(binding.get("target_id") or "").strip()
    # v1 simulator receipts predate explicit target binding evidence and were
    # defined only for the canonical singleton target.
    return "target_object"


def _valid_attachment_transform(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if (
        value.get("schema_version") != "openeta.attachment_transform.v1"
        or value.get("parent_frame") != "eef"
        or value.get("child_frame") != "object"
        or value.get("measurement_boundary") != "native_attach_ack"
    ):
        return False
    xyz = value.get("translation_xyz")
    quat = value.get("quat_xyzw")
    if not _finite_xyz(xyz) or not isinstance(quat, list) or len(quat) != 4:
        return False
    if any(
        isinstance(component, bool)
        or not isinstance(component, int | float)
        or not math.isfinite(float(component))
        for component in quat
    ):
        return False
    norm = math.sqrt(sum(float(component) ** 2 for component in quat))
    return abs(norm - 1.0) <= 1e-5


def _placement_region_prompt(value: object) -> bool:
    prompt = str(value or "").strip().lower()
    return any(
        token in prompt
        for token in (
            "placement",
            "place",
            "destination",
            "target region",
            "target area",
            "marker",
            "receptacle",
            "bin",
            "basket",
            "zone",
            "放置",
            "目标区域",
        )
    )


def _assigned_task_from_tool_call(call: JsonDict) -> tuple[str, str] | None:
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    if not isinstance(details, dict):
        return None
    outputs = details.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else {}
    state_delta = details.get("state_delta")
    state_delta = state_delta if isinstance(state_delta, dict) else {}
    candidates = (
        ("outputs.assigned_task", outputs.get("assigned_task")),
        (
            "outputs.observation_summary.task",
            _nested_mapping_value(outputs, "observation_summary", "task"),
        ),
        (
            "outputs.initial_observation.observation_summary.task",
            _nested_mapping_value(
                outputs,
                "initial_observation",
                "observation_summary",
                "task",
            ),
        ),
        (
            "outputs.response.observation_summary.task",
            _nested_mapping_value(outputs, "response", "observation_summary", "task"),
        ),
        (
            "state_delta.observation.task",
            _nested_mapping_value(state_delta, "observation", "task"),
        ),
    )
    for source_field, value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip(), source_field
    return None


def _nested_mapping_value(value: object, *path: str) -> object:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _transition_call_verdict(call: JsonDict | None) -> str:
    if not isinstance(call, dict):
        return "UNKNOWN"
    if not _call_result_success(call) or _motion_call_rejects_candidate(call):
        return "FAIL"
    return "PASS"


def _action_matches(action: EnvAction, required: JsonDict) -> bool:
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict):
        return False
    return str(request.get("name") or "") == str(required.get("name") or "") and request.get(
        "parameters"
    ) == required.get("parameters")


def grasp_reference_action_error(
    *,
    stage: str,
    tool_name: str,
    parameters: JsonDict,
    required_action: JsonDict,
) -> str | None:
    """Require an execution stage to copy its host-qualified action exactly.

    The grasp provider owns the contact terminal pose.  Once that pose has
    passed the deterministic funnel, neither the VLM nor another host layer is
    allowed to translate or rotate it.  MoveIt remains free to choose the
    complete collision-aware joint path to the frozen terminal state.
    """

    required_name = str(required_action.get("name") or "")
    required_parameters = required_action.get("parameters")
    if tool_name != required_name or not isinstance(required_parameters, dict):
        return f"Grasp stage {stage!r} requires tool {required_name!r}."
    if parameters == required_parameters:
        return None
    return (
        f"Grasp stage {stage!r} must copy the host-qualified {required_name!r} "
        "parameters exactly; model terminal-pose offsets are forbidden."
    )


def _grasp_stage_action_matches(
    action: EnvAction,
    required: JsonDict,
    *,
    stage: str,
) -> bool:
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict):
        return False
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        return False
    return (
        grasp_reference_action_error(
            stage=stage,
            tool_name=str(request.get("name") or ""),
            parameters=parameters,
            required_action=required,
        )
        is None
    )


def _pose_for_epoch(pose: JsonDict, epoch: int) -> JsonDict:
    updated = dict(pose)
    updated["scene_epoch"] = epoch
    return updated


def _grasp_recovery_requires_restore(rejection: JsonDict) -> bool:
    """Return whether a rejected candidate may have displaced the arm."""

    stage = str(rejection.get("grasp_stage") or "")
    if stage == "close":
        # Reaching the close stage proves that the contact move already ran.
        return True
    if rejection.get("current_state_requalification_safe") is True:
        # The simulator proved a causal, stationary, open and unattached end
        # state.  Requalify the frozen model frontier from that state; an exact
        # Cartesian return to the old anchor would add risk without restoring
        # any model or scene invariant.
        return False
    return bool(
        stage == "contact"
        and str(rejection.get("target_tool") or "")
        in {"move_to", "follow_eef_trajectory"}
        and rejection.get("execution_started") is True
    )


def _world_recovery_pose(value: object) -> JsonDict | None:
    """Normalize one observed EEF pose into an exact world-frame motion target."""

    if not isinstance(value, dict):
        return None
    xyz = value.get("xyz")
    quat = value.get("quat_xyzw")
    if not (
        isinstance(xyz, (list, tuple))
        and len(xyz) == 3
        and all(isinstance(item, int | float) and math.isfinite(float(item)) for item in xyz)
        and isinstance(quat, (list, tuple))
        and len(quat) == 4
        and all(isinstance(item, int | float) and math.isfinite(float(item)) for item in quat)
    ):
        return None
    norm = math.sqrt(sum(float(item) ** 2 for item in quat))
    if norm <= 1e-12:
        return None
    return {
        "frame": "world",
        "xyz": [float(item) for item in xyz],
        "quat_xyzw": [float(item) / norm for item in quat],
    }


def _grasp_restore_action(
    restore_pose: JsonDict | None,
    *,
    recovery_id: str,
    scene_epoch: int,
    collision_policy: JsonDict | None = None,
) -> JsonDict:
    if not isinstance(restore_pose, dict):
        raise ValueError("grasp restore pose is unavailable")
    target = dict(restore_pose)
    target.update(
        {
            "scene_epoch": int(scene_epoch),
            "purpose": "grasp_recovery_restore",
            "recovery_id": recovery_id,
        }
    )
    if isinstance(collision_policy, dict):
        target.update(collision_policy)
    return {"name": "move_to", "parameters": {"target_pose": target}}


def _grasp_recovery_collision_policy(
    policy: JsonDict | None,
    *,
    candidate_id: str,
) -> JsonDict | None:
    """Reuse only the failed contact's hash-bound target touch policy."""

    if not isinstance(policy, dict):
        return None
    compilations = policy.get("host_candidate_compilations")
    compiled = compilations.get(candidate_id) if isinstance(compilations, dict) else None
    contact = compiled.get("contact_pose") if isinstance(compiled, dict) else None
    if not isinstance(contact, dict):
        return None
    raw_allowed = contact.get("qualification_allowed_collisions")
    allowed_hash = str(contact.get("qualification_allowed_collisions_sha256") or "")
    compiled_grasp_id = str(contact.get("compiled_grasp_id") or "").strip()
    if not compiled_grasp_id or not isinstance(raw_allowed, dict) or len(allowed_hash) != 64:
        return None
    return {
        "compiled_grasp_id": compiled_grasp_id,
        "grasp_stage": "recovery_restore",
        "qualification_allowed_collisions": {
            str(object_id): list(links)
            for object_id, links in raw_allowed.items()
            if isinstance(links, list)
        },
        "qualification_allowed_collisions_sha256": allowed_hash,
    }


def _optional_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _zero_pass_grasp_recovery_route(outputs: Mapping[str, Any]) -> JsonDict:
    """Choose the narrowest recovery layer justified by qualification proof."""

    evidence = outputs.get("qualification_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    metrics = outputs.get("qualification_metrics")
    if not isinstance(metrics, Mapping):
        metrics = evidence.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    reasons = outputs.get("rejection_reason_counts")
    if not isinstance(reasons, Mapping):
        reasons = evidence.get("rejection_reason_counts")
    reasons = reasons if isinstance(reasons, Mapping) else {}

    generated = _optional_int(outputs.get("generated_candidate_count"), default=0)
    evaluated = _optional_int(
        metrics.get("grasp_scene_alignment_evaluated_count"),
        default=0,
    )
    aperture_risk = _optional_int(
        metrics.get("grasp_scene_alignment_aperture_risk_count"),
        default=0,
    )
    terminal_geometry_failures = _optional_int(
        reasons.get("terminal_gripper_state_invalid"),
        default=0,
    )
    complete_alignment_mismatch = (
        generated > 0
        and evaluated == generated
        and aperture_risk == evaluated
        and terminal_geometry_failures > 0
    )
    if complete_alignment_mismatch:
        return {
            "recovery_strategy": "active_view_relocalization",
            "requires_viewpoint_change": True,
            "recovery_evidence": {
                "classification": "scene_target_alignment_mismatch",
                "generated_candidate_count": generated,
                "scene_alignment_evaluated_count": evaluated,
                "scene_alignment_aperture_risk_count": aperture_risk,
                "terminal_gripper_state_invalid_count": terminal_geometry_failures,
            },
        }
    return {
        "recovery_strategy": "passive_multiview_resegmentation",
        "requires_viewpoint_change": False,
    }


def _is_anyplace_pose(parameters: JsonDict) -> bool:
    pose = parameters.get("camera_pose")
    if not isinstance(pose, dict):
        return False
    pose_id = str(pose.get("id") or "")
    return pose_id.startswith("place_grasp_") or str(pose.get("source_tool") or "") == "anyplace"


def _reconcile_gripper_position(position: object, state: JsonDict) -> str:
    requested = _binary_gripper_position(position)
    if requested is None:
        return "unresolved"
    is_open = state.get("open")
    openness = state.get("openness")
    if requested == 1:
        if is_open is True or isinstance(openness, (int, float)) and openness >= 0.8:
            return "completed"
    else:
        # A grasped object can stop the fingers well above the empty-close value.
        # Any observed departure from fully open reconciles only the binary close
        # transition. Native bilateral contact plus the attach ACK independently
        # adjudicates whether the object is actually held.
        if is_open is False or isinstance(openness, (int, float)) and openness < 0.8:
            return "completed"
    return "unresolved"


def _gripper_attachment_evidence(
    state: JsonDict,
    *,
    command_state: JsonDict | None,
) -> tuple[str, JsonDict]:
    if not isinstance(command_state, dict) or command_state.get("position") != 0:
        return "UNKNOWN", {}
    object_detection = str(state.get("object_detection") or "").strip().lower()
    if object_detection == "object_detected_closing":
        return "PASS", {
            "object_detection": object_detection,
            "position": state.get("position"),
            "position_normalized": state.get("position_normalized"),
            "requested_position": state.get("requested_position"),
        }
    if object_detection == "at_position_no_object":
        return "FAIL", {
            "object_detection": object_detection,
            "position": state.get("position"),
            "position_normalized": state.get("position_normalized"),
            "requested_position": state.get("requested_position"),
        }
    return "UNKNOWN", {}


def _binary_gripper_position(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
        return None
    return int(numeric)


def _action_grasp_outcome(action: EnvAction) -> str:
    command = action.command if isinstance(action.command, dict) else {}
    for call in command.get("tool_calls", []) or []:
        if not isinstance(call, dict):
            continue
        supervision = _call_supervision(call)
        review = supervision.get("details") if isinstance(supervision, dict) else None
        if not isinstance(review, dict):
            continue
        outcome = str(review.get("grasp_outcome") or "").strip().lower()
        aliases = {"failed": "fail", "passed": "pass", "uncertain": "unknown"}
        return aliases.get(outcome, outcome)
    return ""


def _action_host_obligation(action: EnvAction) -> JsonDict:
    command = action.command if isinstance(action.command, dict) else {}
    metadata = command.get("metadata")
    planner_metadata = metadata.get("planner_metadata") if isinstance(metadata, dict) else None
    obligation = (
        planner_metadata.get("host_obligation") if isinstance(planner_metadata, dict) else None
    )
    return obligation if isinstance(obligation, dict) else {}


def _safety_call_rejects_candidate(call: JsonDict) -> bool:
    if _call_result_success(call):
        return False
    result = call.get("result")
    if not isinstance(result, dict):
        return False
    details = result.get("details")
    if isinstance(details, dict):
        outputs = details.get("outputs")
        for source in (outputs, details):
            if not isinstance(source, dict):
                continue
            if source.get("feasible") is False or source.get("clear") is False:
                return True
            verdict = str(source.get("verdict") or "").strip().lower()
            if verdict in {"unsafe", "infeasible", "rejected", "collision", "blocked"}:
                return True
    return _failure_text_rejects_candidate(call)


def _independent_motion_review_rejects_candidate(call: JsonDict) -> bool:
    """Treat any fail-closed terminal-motion review as candidate rejection.

    The independent reviewer uses ``abstain`` when visual evidence is too
    ambiguous to approve a candidate-specific motion.  At this boundary an
    abstention still means that the active grasp candidate cannot be executed,
    so it must consume the same bounded candidate budget as an explicit reject.
    """

    if _call_result_success(call):
        return False
    supervision = _call_supervision(call)
    if not isinstance(supervision, dict):
        return False
    review = supervision.get("details")
    if not isinstance(review, dict):
        return False
    return (
        supervision.get("allowed") is False
        and str(supervision.get("source") or "") == "independent_reviewer"
        and str(review.get("decision") or "").strip().lower() in {"reject", "abstain"}
        and str(review.get("grasp_outcome") or "not_assessed").strip().lower() == "not_assessed"
    )


def _independent_review_recovery_metadata(call: JsonDict) -> JsonDict:
    supervision = _call_supervision(call)
    review = supervision.get("details") if isinstance(supervision, dict) else None
    if not isinstance(review, dict):
        return {}
    recovery_class = str(
        review.get("recovery_class") or review.get("rejection_class") or ""
    ).strip()
    if recovery_class not in {"perception_refinable", "uncertain_review"}:
        return {}
    return {"recovery_class": recovery_class}


def _grasp_seed_compile_rejects_candidate(call: JsonDict) -> bool:
    if _call_result_success(call):
        return False
    result = call.get("result")
    if not isinstance(result, dict):
        return False
    details = result.get("details")
    if not isinstance(details, dict):
        return False
    diagnostics = details.get("diagnostics")
    if not isinstance(diagnostics, list):
        return False
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        if diagnostic.get("candidate_rejection") is True:
            return True
        if str(diagnostic.get("code") or "") != "grasp_seed_compile_failed":
            continue
        message = str(diagnostic.get("message") or "").lower()
        if "candidate width" in message and (
            "restricted bounds" in message or "strategy bounds" in message
        ):
            return True
    return False


def _grasp_seed_compile_recovery_metadata(call: JsonDict) -> JsonDict:
    outputs = _tool_call_outputs(call)
    recovery_class = str(outputs.get("recovery_class") or "")
    rejection_code = str(outputs.get("rejection_code") or "")
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    diagnostics = details.get("diagnostics") if isinstance(details, dict) else None
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            recovery_class = recovery_class or str(diagnostic.get("recovery_class") or "")
            rejection_code = rejection_code or str(diagnostic.get("rejection_code") or "")
    if recovery_class not in {"perception_refinable", "uncertain_review"}:
        return {}
    return {
        "recovery_class": recovery_class,
        "rejection_code": rejection_code,
    }


def _grasp_estimation_recovery_class(rejection: JsonDict) -> str:
    structured = str(rejection.get("recovery_class") or "")
    if structured in {"perception_refinable", "uncertain_review"}:
        return structured
    source = str(rejection.get("source") or "")
    if source == "physical_gripper_width_filter":
        return "perception_refinable"
    if source in {
        "uncertain_review",
        "main_vlm_terminal_pose_uncertain",
    }:
        return "uncertain_review"
    return "none"


def _camera_frame_for_source_image(policy: JsonDict, source_image: str) -> str:
    if str(policy.get("source_rgb") or "") == source_image:
        return str(policy.get("camera_frame_id") or "")
    return ""


def _motion_call_rejects_candidate(call: JsonDict) -> bool:
    result = call.get("result")
    if not isinstance(result, dict):
        return False
    details = result.get("details")
    if not isinstance(details, dict):
        return False
    if _structured_motion_reached_target(details) is False:
        return True
    if _call_result_success(call):
        return False
    diagnostics = details.get("diagnostics")
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            if diagnostic.get("candidate_rejection") is True:
                return True
            if str(diagnostic.get("code") or "") in {
                "grasp_candidate_collision",
                "grasp_candidate_infeasible",
                "grasp_candidate_unreachable",
            }:
                return True
    outputs = details.get("outputs")
    if isinstance(outputs, dict):
        for source in (outputs, outputs.get("motion_summary"), outputs.get("response")):
            if not isinstance(source, dict):
                continue
            if source.get("candidate_rejection") is True:
                return True
            failure_class = str(source.get("failure_class") or "").strip().lower()
            if failure_class in {
                "grasp_candidate_collision",
                "grasp_candidate_infeasible",
                "grasp_candidate_unreachable",
            }:
                return True
    return False


def _structured_motion_reached_target(details: JsonDict) -> bool | None:
    outputs = details.get("outputs")
    response = outputs.get("response") if isinstance(outputs, dict) else None
    state_delta = details.get("state_delta")
    sources = (
        outputs.get("motion_summary") if isinstance(outputs, dict) else None,
        response.get("motion_summary") if isinstance(response, dict) else None,
        details.get("motion_summary"),
        state_delta.get("motion") if isinstance(state_delta, dict) else None,
    )
    for source in sources:
        if isinstance(source, dict) and isinstance(source.get("reached_target"), bool):
            return source["reached_target"]
    return None


def _failure_text_rejects_candidate(call: JsonDict) -> bool:
    text_parts = [str(call.get("reason") or "")]
    result = call.get("result")
    if isinstance(result, dict):
        text_parts.append(str(result.get("content") or ""))
        details = result.get("details")
        if isinstance(details, dict):
            text_parts.extend(
                str(details.get(key) or "") for key in ("reason", "verdict", "diagnostic")
            )
            outputs = details.get("outputs")
            if isinstance(outputs, dict):
                text_parts.extend(
                    str(outputs.get(key) or "") for key in ("reason", "verdict", "diagnostic")
                )
            diagnostics = details.get("diagnostics")
            if isinstance(diagnostics, list):
                text_parts.extend(str(item) for item in diagnostics if isinstance(item, dict))
    text = " ".join(text_parts).lower()
    non_pose_failures = (
        "timeout",
        "transport",
        "connection",
        "mcp_call_failed",
        "not configured",
        "missing_",
        "invalid_",
        "malformed",
        "schema",
        "operator_denied",
        "user denied",
        "session",
        "backend unavailable",
    )
    if any(marker in text for marker in non_pose_failures):
        return False
    pose_rejections = (
        "unsafe",
        "infeasible",
        "unreachable",
        "outside_workspace",
        "outside workspace",
        "ik failed",
        "ik target",
        "joint limit",
        "path blocked",
        "target not reached",
        "motion rejected",
        "pose rejected",
    )
    return any(marker in text for marker in pose_rejections)


def _call_failure_reason(call: JsonDict) -> str:
    result = call.get("result")
    if isinstance(result, dict):
        details = result.get("details")
        if isinstance(details, dict):
            if _structured_motion_reached_target(details) is False:
                return "Simulator motion summary reports that the target was not reached."
            outputs = details.get("outputs")
            for source in (outputs, details):
                if not isinstance(source, dict):
                    continue
                for key in ("reason", "verdict", "diagnostic"):
                    value = source.get(key)
                    if isinstance(value, str) and value:
                        return value
        content = result.get("content")
        if isinstance(content, str) and content:
            return content
    reason = call.get("reason")
    return str(reason or "candidate-linked tool rejected the grasp pose")


def _matching_latest_world_pose_candidate(
    parameters: JsonDict,
    artifacts: dict[str, JsonDict],
) -> str:
    target_xyz = _parameters_target_xyz(parameters)
    if target_xyz is None:
        target_parameters = parameters.get("target_parameters")
        if isinstance(target_parameters, dict):
            target_xyz = _parameters_target_xyz(target_parameters)
    if target_xyz is None:
        return ""
    latest: tuple[float, JsonDict] | None = None
    for entry in artifacts.values():
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if not isinstance(value, dict) or value.get("type") != "world_pose":
            continue
        source_grasp_id = value.get("source_grasp_id")
        world_xyz = value.get("translation_xyz")
        if not isinstance(source_grasp_id, str) or not _xyz_equal(target_xyz, world_xyz):
            continue
        try:
            timestamp = float(entry.get("timestamp_s") or 0.0)
        except (TypeError, ValueError):
            timestamp = 0.0
        if latest is None or timestamp >= latest[0]:
            latest = (timestamp, value)
    return str(latest[1].get("source_grasp_id") or "") if latest else ""


def _parameters_target_xyz(parameters: JsonDict) -> list[float] | None:
    pose = parameters.get("target_pose") or parameters.get("pose") or parameters.get("eef_pose")
    if not isinstance(pose, dict):
        return None
    xyz = pose.get("xyz") or pose.get("translation_xyz") or pose.get("position")
    if not isinstance(xyz, list | tuple) or len(xyz) < 3:
        return None
    try:
        return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    except (TypeError, ValueError):
        return None


def _xyz_equal(left: list[float], right: Any, *, tolerance: float = 1e-8) -> bool:
    if not isinstance(right, list | tuple) or len(right) < 3:
        return False
    try:
        return all(abs(left[idx] - float(right[idx])) <= tolerance for idx in range(3))
    except (TypeError, ValueError):
        return False


def _select_memory(items: dict[str, JsonDict], key: str | None) -> dict[str, JsonDict]:
    if key is None:
        return dict(items)
    if key in items:
        return {key: items[key]}
    return {}


def _memory_fact_entry(value: JsonDict, *, source: str) -> JsonDict:
    return {"value": dict(value), "source": source, "timestamp_s": time.time()}


def _memory_fact_value(entry: JsonDict | None) -> JsonDict | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("value")
    return dict(value) if isinstance(value, dict) else None


def _grasp_fallback_attempts(policy: JsonDict | None) -> list[JsonDict]:
    if not isinstance(policy, dict):
        return []
    value = policy.get("fallback_attempts")
    if not isinstance(value, list):
        return []
    return [dict(attempt) for attempt in value if isinstance(attempt, dict)]


def _append_grasp_fallback_attempt(
    attempts: list[JsonDict],
    attempt: JsonDict,
) -> None:
    identity = (
        str(attempt.get("backend") or ""),
        str(attempt.get("source_rgb") or ""),
        str(attempt.get("outcome") or ""),
    )
    if any(
        (
            str(existing.get("backend") or ""),
            str(existing.get("source_rgb") or ""),
            str(existing.get("outcome") or ""),
        )
        == identity
        for existing in attempts
    ):
        return
    attempts.append(dict(attempt))


def _extract_action_artifacts(action: EnvAction) -> list[JsonDict]:
    artifacts: list[JsonDict] = []
    seen_paths: set[str] = set()
    command = action.command if isinstance(action.command, dict) else {}
    for call in command.get("tool_calls", []) or []:
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        details = result.get("details")
        if not isinstance(details, dict):
            continue
        for artifact in details.get("artifacts", []) or []:
            if not isinstance(artifact, dict):
                continue
            path = artifact.get("path")
            if not isinstance(path, str) or not path:
                continue
            if path in seen_paths:
                continue
            seen_paths.add(path)
            normalized = dict(artifact)
            normalized.setdefault("tool", call.get("name"))
            artifacts.append(normalized)
        artifacts.extend(_extract_camera_packet_artifacts(call, details))
        artifacts.extend(_extract_depth_prior_artifacts(call, details))
        artifacts.extend(_extract_depth_enhancement_artifacts(call, details))
        artifacts.extend(_extract_sensor_safety_check_artifacts(call, details))
        artifacts.extend(_extract_grasp_candidate_artifacts(call, details))
        artifacts.extend(_extract_placement_candidate_artifacts(call, details))
        artifacts.extend(_extract_world_pose_artifacts(call, details))
    return artifacts


def _extract_camera_packet_artifacts(call: JsonDict, details: JsonDict) -> list[JsonDict]:
    if str(call.get("name") or "") != "observe":
        return []
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        return []
    response = outputs.get("response")
    if not isinstance(response, dict):
        return []
    response_path = response.get("response_path")
    if not isinstance(response_path, str) or not response_path:
        return []
    try:
        payload = json.loads(Path(response_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    cameras = _find_camera_payloads(payload)
    artifacts: list[JsonDict] = []
    for index, camera in enumerate(cameras):
        packet = _camera_packet_from_payload(
            camera,
            index=index,
            response_path=response_path,
            tool=str(call.get("name") or "observe"),
        )
        if packet is not None:
            artifacts.append(packet)
    return artifacts


def _find_camera_payloads(payload: Any) -> list[JsonDict]:
    if isinstance(payload, dict):
        cameras = payload.get("cameras")
        if isinstance(cameras, list):
            return [camera for camera in cameras if isinstance(camera, dict)]
        for value in payload.values():
            found = _find_camera_payloads(value)
            if found:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _find_camera_payloads(value)
            if found:
                return found
    return []


def _camera_packet_from_payload(
    camera: JsonDict,
    *,
    index: int,
    response_path: str,
    tool: str,
) -> JsonDict | None:
    frame_id = str(camera.get("frame_id") or camera.get("camera") or f"camera_{index}")
    rgb_path = _string_field(camera, "rgb_path") or _string_field(camera, "image_path")
    depth_path = _string_field(camera, "depth_path")
    intrinsics = camera.get("intrinsics")
    extrinsics = camera.get("extrinsics")
    if not rgb_path and not depth_path:
        return None
    if not isinstance(intrinsics, dict):
        intrinsics = {}
    if not isinstance(extrinsics, dict):
        extrinsics = {}
    depth_scale, depth_scale_source = _camera_depth_scale(camera, intrinsics)
    normalized_intrinsics: JsonDict = dict(intrinsics)
    if depth_scale is not None:
        normalized_intrinsics["scale"] = depth_scale

    packet: JsonDict = {
        "type": "camera_packet",
        "kind": "rgbd_camera",
        "tool": tool,
        "index": frame_id,
        "frame_id": frame_id,
        "response_path": response_path,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "intrinsics": normalized_intrinsics,
        "anygrasp_intrinsics": dict(normalized_intrinsics),
        "extrinsics": dict(extrinsics),
    }
    role = camera.get("role")
    if isinstance(role, str) and role:
        packet["role"] = role
    for field_name in (
        "width",
        "height",
        "depth_min",
        "depth_max",
        "depth_encoding",
    ):
        if field_name in camera:
            packet[field_name] = camera[field_name]
    if depth_scale is not None:
        packet["depth_scale"] = depth_scale
        packet["depth_scale_source"] = depth_scale_source
    camera_frame = extrinsics.get("camera_frame")
    if isinstance(camera_frame, str) and camera_frame:
        packet["camera_frame"] = camera_frame
    matrix_layout = extrinsics.get("matrix_layout")
    if isinstance(matrix_layout, str) and matrix_layout:
        packet["matrix_layout"] = matrix_layout
    return packet


def _camera_depth_scale(camera: JsonDict, intrinsics: JsonDict) -> tuple[float | None, str]:
    for key in ("scale", "depth_scale"):
        value = intrinsics.get(key)
        parsed = _positive_float(value)
        if parsed is not None:
            return parsed, f"intrinsics.{key}"
    for key in ("depth_scale", "scale"):
        value = camera.get(key)
        parsed = _positive_float(value)
        if parsed is not None:
            return parsed, f"camera.{key}"
    depth_path = _string_field(camera, "depth_path")
    if depth_path and depth_path.lower().endswith(".png"):
        return 1000.0, "default_png_millimeters"
    return None, "missing"


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _extract_depth_prior_artifacts(call: JsonDict, details: JsonDict) -> list[JsonDict]:
    if str(call.get("name") or "") != "estimate_depth_prior":
        return []
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        return []
    prior_depth = _string_field(outputs, "prior_depth")
    if not prior_depth:
        return []
    camera_id = str(outputs.get("camera_id") or "camera")
    return [
        {
            "type": "depth_prior",
            "kind": "metric_depth_prior",
            "tool": "estimate_depth_prior",
            "index": camera_id,
            "camera_id": camera_id,
            "source_rgb": outputs.get("source_rgb"),
            "prior_depth": prior_depth,
            "prior_confidence": outputs.get("prior_confidence"),
            "prior_confidence_semantics": outputs.get("prior_confidence_semantics"),
            "backend": outputs.get("backend"),
            "model": outputs.get("model"),
            "request_ref": outputs.get("request_ref"),
            "raw_output_ref": outputs.get("raw_output_ref"),
            "next_tool_hint": outputs.get("next_tool_hint")
            or ("Call enhance_depth with the same rgb/depth/intrinsics and this prior_depth path."),
        }
    ]


def _extract_depth_enhancement_artifacts(call: JsonDict, details: JsonDict) -> list[JsonDict]:
    if str(call.get("name") or "") != "enhance_depth":
        return []
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        return []
    report_path = _string_field(outputs, "report_path")
    fused_depth_npy = _string_field(outputs, "fused_depth_npy")
    fused_depth_png = _string_field(outputs, "fused_depth_png")
    safety_depth_npy = _string_field(outputs, "safety_depth_npy")
    safety_depth_png = _string_field(outputs, "safety_depth_png")
    point_cloud_npz = _string_field(outputs, "point_cloud_npz")
    provenance_mask_png = _string_field(outputs, "provenance_mask_png")
    if not report_path or not fused_depth_npy:
        return []
    camera_id = str(outputs.get("camera_id") or "camera")
    return [
        {
            "type": "depth_enhancement",
            "kind": "rgbd_depth_enhancement",
            "tool": "enhance_depth",
            "index": camera_id,
            "camera_id": camera_id,
            "calibration_profile_id": outputs.get("calibration_profile_id"),
            "enabled": bool(outputs.get("enabled")),
            "reason": outputs.get("reason"),
            "source_rgb": outputs.get("source_rgb"),
            "source_depth": outputs.get("source_depth"),
            "source_sensor_confidence": outputs.get("source_sensor_confidence"),
            "source_rgb_sha256": outputs.get("source_rgb_sha256"),
            "source_depth_sha256": outputs.get("source_depth_sha256"),
            "intrinsics": outputs.get("intrinsics")
            if isinstance(outputs.get("intrinsics"), dict)
            else {},
            "candidate_intrinsics": (
                outputs.get("candidate_intrinsics")
                if isinstance(outputs.get("candidate_intrinsics"), dict)
                else {}
            ),
            "scene_epoch": outputs.get("scene_epoch"),
            "rgb_timestamp_s": outputs.get("rgb_timestamp_s"),
            "depth_timestamp_s": outputs.get("depth_timestamp_s"),
            "registration_status": outputs.get("registration_status"),
            "calibration_hash": outputs.get("calibration_hash"),
            "report_path": report_path,
            "fused_depth_npy": fused_depth_npy,
            "fused_depth_png": fused_depth_png,
            "candidate_depth_npy": outputs.get("candidate_depth_npy") or fused_depth_npy,
            "candidate_depth_png": outputs.get("candidate_depth_png") or fused_depth_png,
            "safety_depth_npy": safety_depth_npy,
            "safety_depth_png": safety_depth_png,
            "point_cloud_npz": point_cloud_npz,
            "candidate_point_cloud_npz": outputs.get("candidate_point_cloud_npz")
            or point_cloud_npz,
            "safety_point_cloud_npz": outputs.get("safety_point_cloud_npz"),
            "provenance_mask_png": provenance_mask_png,
            "alignment": outputs.get("alignment")
            if isinstance(outputs.get("alignment"), dict)
            else {},
            "quality": outputs.get("quality") if isinstance(outputs.get("quality"), dict) else {},
            "next_tool_hint": (
                "Use fused_depth_npy or a derived depth PNG for perception/grasp "
                "candidate generation only when quality.use_for_grasp_candidate_generation "
                "is true; never use mono-filled geometry for final collision clearance."
            ),
        }
    ]


def _extract_sensor_safety_check_artifacts(
    call: JsonDict,
    details: JsonDict,
) -> list[JsonDict]:
    if str(call.get("name") or "") != "obstacle_avoidance":
        return []
    parameters = details.get("parameters")
    path = parameters.get("path") if isinstance(parameters, dict) else None
    outputs = details.get("outputs")
    if (
        not isinstance(path, dict)
        or path.get("kind") != "enhanced_grasp_sensor_safety_check"
        or not isinstance(outputs, dict)
        or outputs.get("clear") is not True
    ):
        return []
    candidate_id = str(path.get("candidate_id") or "")
    safety_depth = str(path.get("safety_depth_png") or "")
    safety_cloud = str(path.get("safety_point_cloud_npz") or "")
    report_path = str(path.get("report_path") or "")
    if (
        not candidate_id
        or not Path(safety_depth).is_file()
        or not Path(safety_cloud).is_file()
        or not Path(report_path).is_file()
    ):
        return []
    return [
        {
            "type": "enhanced_grasp_sensor_safety_check",
            "kind": "sensor_safety_check",
            "tool": "obstacle_avoidance",
            "index": candidate_id,
            "candidate_id": candidate_id,
            "scene_epoch": path.get("scene_epoch"),
            "safety_depth_png": safety_depth,
            "safety_point_cloud_npz": safety_cloud,
            "report_path": report_path,
            "clear": True,
        }
    ]


def _extract_grasp_candidate_artifacts(call: JsonDict, details: JsonDict) -> list[JsonDict]:
    tool_name = str(call.get("name") or "")
    if tool_name not in {
        "grasp_pose_estimate",
        "anygrasp",
        "graspgenx",
        "contact_graspnet",
    }:
        return []
    candidates = details.get("grasp_candidates")
    source = details
    if not isinstance(candidates, list):
        outputs = details.get("outputs")
        if isinstance(outputs, dict):
            candidates = outputs.get("grasp_candidates")
            source = outputs
    if not isinstance(candidates, list) or not candidates:
        return []
    compact_candidates = [
        dict(candidate) for candidate in candidates[:20] if isinstance(candidate, dict)
    ]
    if not compact_candidates:
        return []
    grasp_source = source.get("source")
    if not isinstance(grasp_source, dict):
        grasp_source = {}
    return [
        {
            "type": "grasp_candidates",
            "kind": "grasp_candidates",
            "tool": tool_name,
            "index": "latest",
            "candidate_count": len(candidates),
            "best_grasp_candidate": compact_candidates[0],
            "grasp_candidates": compact_candidates,
            "source_rgb": source.get("source_rgb") or grasp_source.get("rgb"),
            "source_depth": source.get("source_depth") or grasp_source.get("depth"),
            "target_mask": source.get("target_mask") or grasp_source.get("object_mask"),
            "selected_grasp_source": source.get("source"),
            "source_tool": grasp_source.get("source_tool") or tool_name,
            "source_backend": grasp_source.get("source_backend")
            or source.get("selected_backend")
            or tool_name,
            "gripper_name": grasp_source.get("gripper_name"),
            "raw_output_ref": source.get("raw_output_ref"),
            "next_tool_hint": (
                "Require the host-owned compilation event for the stable qualified "
                "queue head, then follow host-generated grasp_execution stages."
            ),
        }
    ]


def _grasp_backend_label(value: str) -> str:
    return {
        "anygrasp": "AnyGrasp",
        "contact_graspnet": "Contact-GraspNet",
        "graspgenx": "GraspGenX",
        "grasp_pose_estimate": "grasp estimator",
    }.get(value, value or "grasp estimator")


def _extract_placement_candidate_artifacts(
    call: JsonDict,
    details: JsonDict,
) -> list[JsonDict]:
    if str(call.get("name") or "") != "anyplace":
        return []
    candidates = details.get("placement_candidates")
    source = details
    if not isinstance(candidates, list):
        outputs = details.get("outputs")
        if isinstance(outputs, dict):
            candidates = outputs.get("placement_candidates")
            source = outputs
    if not isinstance(candidates, list) or not candidates:
        return []
    compact_candidates = [
        dict(candidate) for candidate in candidates[:20] if isinstance(candidate, dict)
    ]
    if not compact_candidates:
        return []
    return [
        {
            "type": "placement_candidates",
            "kind": "placement_candidates",
            "tool": "anyplace",
            "index": "latest",
            "candidate_count": len(candidates),
            "placement_candidates": compact_candidates,
            "source": source.get("source"),
            "candidate_image_ref": source.get("candidate_image_ref"),
            "raw_output_ref": source.get("raw_output_ref"),
            "next_tool_hint": (
                "Use the host-owned compilation event for the stable qualified queue "
                "head; raw AnyPlace transforms are not executable EEF poses."
            ),
        }
    ]


def _extract_world_pose_artifacts(call: JsonDict, details: JsonDict) -> list[JsonDict]:
    if str(call.get("name") or "") != "camera_pose_to_world":
        return []
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        return []
    world_pose = outputs.get("world_pose")
    if not isinstance(world_pose, dict):
        return []
    translation = world_pose.get("translation_xyz") or outputs.get("translation_xyz")
    if not isinstance(translation, list) or len(translation) != 3:
        return []
    is_placement_reference = str(world_pose.get("id") or "").startswith("place_grasp_")
    next_tool_hint = (
        "This is a model object goal, not an executable wrist target. Do not add "
        "hover, descent, clearance, or orientation offsets. Let the host combine it "
        "with the measured attachment and support geometry, then expose the exact "
        "qualified release EEF pose; MoveIt owns the complete current-to-release path."
        if is_placement_reference
        else (
            "Pass the complete world_pose to move_to.target_pose without changing "
            "its translation or rotation."
        )
    )
    return [
        {
            "type": "world_pose",
            "kind": "world_pose",
            "tool": "camera_pose_to_world",
            "index": "latest",
            "frame": world_pose.get("frame") or outputs.get("frame") or "world",
            "camera_frame_id": outputs.get("camera_frame_id"),
            "world_pose": dict(world_pose),
            "translation_xyz": list(translation),
            "rotation_matrix": world_pose.get("rotation_matrix") or outputs.get("rotation_matrix"),
            "gripper_tip_position_xyz": world_pose.get("gripper_tip_position_xyz")
            or outputs.get("gripper_tip_position_xyz"),
            "source_grasp_id": world_pose.get("id"),
            "next_tool_hint": next_tool_hint,
        }
    ]


def _string_field(value: JsonDict, key: str) -> str:
    field = value.get(key)
    return field if isinstance(field, str) and field else ""


def _artifact_memory_key(artifact: JsonDict, *, fallback_index: int) -> str:
    tool = str(artifact.get("tool") or "tool")
    artifact_type = str(artifact.get("type") or artifact.get("kind") or "artifact")
    index = str(artifact.get("index") or "")
    if not index:
        path = artifact.get("path")
        index = Path(str(path)).stem if path else str(fallback_index)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{tool}:{artifact_type}:{index}").strip("._")
    return safe or f"tool_artifact_{fallback_index}"


def summarize_memory_artifact(artifact: JsonDict) -> JsonDict:
    """Return a compact artifact summary for planner context."""

    value = artifact.get("value", {})
    summary: JsonDict = {
        "source": artifact.get("source", ""),
        "timestamp_s": artifact.get("timestamp_s"),
    }
    if isinstance(value, dict):
        summary["keys"] = sorted(str(key) for key in value)
        for field in (
            "id",
            "type",
            "kind",
            "index",
            "tool",
            "content",
            "path",
            "grep_hint",
            "chars",
            "response_path",
            "response_chars",
            "response_omitted",
            "image_root",
            "image_count",
            "latest_image_path",
            "paths",
            "env_id",
            "handle",
            "session_id",
            "mcp_server_url",
            "dashboard_url",
            "frame_id",
            "role",
            "rgb_path",
            "depth_path",
            "width",
            "height",
            "depth_min",
            "depth_max",
            "depth_scale",
            "depth_scale_source",
            "camera_frame_id",
            "camera_frame",
            "matrix_layout",
            "intrinsics",
            "anygrasp_intrinsics",
            "extrinsics",
            "candidate_count",
            "best_grasp_candidate",
            "grasp_candidates",
            "selected_grasp_id",
            "placement_candidates",
            "candidate_image_ref",
            "source",
            "source_rgb",
            "source_depth",
            "source_sensor_confidence",
            "target_mask",
            "selected_grasp_source",
            "source_tool",
            "gripper_name",
            "raw_output_ref",
            "prior_depth",
            "prior_confidence",
            "backend",
            "model",
            "request_ref",
            "camera_id",
            "calibration_profile_id",
            "enabled",
            "reason",
            "report_path",
            "fused_depth_npy",
            "fused_depth_png",
            "point_cloud_npz",
            "provenance_mask_png",
            "alignment",
            "quality",
            "frame",
            "world_pose",
            "translation_xyz",
            "rotation_matrix",
            "gripper_tip_position_xyz",
            "source_grasp_id",
            "next_tool_hint",
        ):
            if field in value:
                structured_fields = {
                    "best_grasp_candidate",
                    "grasp_candidates",
                    "placement_candidates",
                    "selected_grasp_source",
                    "source",
                    "world_pose",
                    "extrinsics",
                    "rotation_matrix",
                    "intrinsics",
                    "anygrasp_intrinsics",
                    "alignment",
                    "quality",
                }
                max_depth = 4 if field in structured_fields else 2
                max_items = 16 if field in structured_fields else 8
                summary[field] = _compact_value(
                    value[field],
                    max_depth=max_depth,
                    max_items=max_items,
                )
        image_paths = _extract_artifact_image_paths(value)
        if image_paths:
            summary["image_paths"] = image_paths
    else:
        summary["type"] = type(value).__name__
    return summary


def _extract_artifact_image_paths(value: JsonDict, *, limit: int = 20) -> list[str]:
    paths: list[str] = []
    for field_name in ("images", "image_artifacts"):
        images = value.get(field_name)
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict):
                continue
            for key in ("path", "rgb_path", "depth_path", "image_path"):
                path = image.get(key)
                if isinstance(path, str) and path and path not in paths:
                    paths.append(path)
                if len(paths) >= limit:
                    return paths
    return paths


def summarize_event_payload(payload: JsonDict) -> JsonDict:
    """Return a bounded event payload summary for planner context.

    Session traces may keep rich action and tool metadata for debugging, but the
    planner should not receive complete historical commands. Full payloads can
    contain prior planner contexts, image payloads, or tool outputs, which then
    recursively inflate later prompts.
    """

    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}

    summary: JsonDict = {"keys": sorted(str(key) for key in payload)}
    for key in (
        "task",
        "session_id",
        "source",
        "environment",
        "max_turns",
        "turn_index",
        "question",
        "answer",
    ):
        if key in payload:
            summary[key] = _compact_value(payload[key])

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        summary["metadata"] = _compact_metadata(metadata)

    observation = payload.get("observation")
    if isinstance(observation, dict):
        summary["observation"] = _compact_observation_payload(observation)

    step_result = payload.get("step_result")
    if isinstance(step_result, dict):
        summary["step_result"] = _compact_step_result(step_result)

    action = payload.get("action")
    if isinstance(action, dict):
        summary["action"] = _compact_action_payload(action)

    command = payload.get("command")
    if isinstance(command, dict):
        summary["command"] = _compact_command_payload(command)

    request = payload.get("request")
    if isinstance(request, dict):
        summary["request"] = _compact_request_payload(request)

    if "interfaces" in payload and isinstance(payload["interfaces"], list):
        summary["interfaces"] = [
            _compact_named_item(item)
            for item in payload["interfaces"][:8]
            if isinstance(item, dict)
        ]
        summary["interface_count"] = len(payload["interfaces"])
    if "tools" in payload and isinstance(payload["tools"], list):
        summary["tools"] = list(payload["tools"][:32])
        summary["tool_count"] = len(payload["tools"])
    if "skills" in payload and isinstance(payload["skills"], list):
        summary["skills"] = list(payload["skills"][:16])
        summary["skill_count"] = len(payload["skills"])

    return summary


def _compact_step_result(step_result: JsonDict) -> JsonDict:
    return {
        "reward": step_result.get("reward"),
        "terminated": step_result.get("terminated"),
        "truncated": step_result.get("truncated"),
        "observation": _compact_observation_payload(step_result.get("observation"))
        if isinstance(step_result.get("observation"), dict)
        else None,
        "info": _compact_metadata(step_result.get("info") or {})
        if isinstance(step_result.get("info"), dict)
        else {},
    }


def _compact_observation_payload(observation: JsonDict) -> JsonDict:
    robot = observation.get("robot")
    if not isinstance(robot, dict):
        robot = {}
    objects = observation.get("objects")
    if not isinstance(objects, list):
        objects = []
    return {
        "task": _compact_value(observation.get("task")),
        "camera_ids": list(observation.get("camera_ids") or []),
        "num_cameras": observation.get("num_cameras")
        if "num_cameras" in observation
        else len(observation.get("camera_ids") or []),
        "object_count": len(objects),
        "objects": [_compact_value(obj) for obj in objects[:5]],
        "robot": {
            "end_effector_pose": _compact_value(robot.get("end_effector_pose")),
            "gripper_state": _compact_value(robot.get("gripper_state")),
            "base_pose": _compact_value(robot.get("base_pose")),
        },
        "metadata": _compact_metadata(observation.get("metadata") or {})
        if isinstance(observation.get("metadata"), dict)
        else {},
    }


def _compact_action_payload(action: JsonDict) -> JsonDict:
    return {
        "action_type": action.get("action_type"),
        "request_kind": action.get("request_kind"),
        "request_name": action.get("request_name"),
        "status": action.get("status"),
        "tool_calls": [
            _compact_tool_call(call)
            for call in (action.get("tool_calls") or [])[:8]
            if isinstance(call, dict)
        ],
        "metadata": _compact_metadata(action.get("metadata") or {})
        if isinstance(action.get("metadata"), dict)
        else {},
    }


def _compact_command_payload(command: JsonDict) -> JsonDict:
    return {
        "status": command.get("status"),
        "schema_version": command.get("schema_version"),
        "request": _compact_request_payload(command.get("request") or {})
        if isinstance(command.get("request"), dict)
        else {},
        "tool_calls": [
            _compact_tool_call(call)
            for call in (command.get("tool_calls") or [])[:8]
            if isinstance(call, dict)
        ],
        "metadata": _compact_metadata(command.get("metadata") or {})
        if isinstance(command.get("metadata"), dict)
        else {},
    }


def _compact_tool_call(call: JsonDict) -> JsonDict:
    result = call.get("result")
    compact_result: JsonDict | None = None
    if isinstance(result, dict):
        compact_result = {
            "success": result.get("success"),
            "content": _compact_value(result.get("content")),
        }
        details = result.get("details")
        if isinstance(details, dict):
            compact_details = _compact_tool_result_details(details)
            if compact_details:
                compact_result["details"] = compact_details
    return {
        "name": call.get("name"),
        "status": call.get("status"),
        "reason": _compact_value(call.get("reason")),
        "result": compact_result,
    }


def _compact_tool_result_details(details: JsonDict) -> JsonDict:
    compact: JsonDict = {}
    for key in (
        "candidate_count",
        "best_grasp_candidate",
        "grasp_candidates",
        "ranking",
        "result_id",
        "selection_required",
        "selected_detection",
        "selection_bundle",
        "source_rgb",
        "source_depth",
        "target_mask",
        "raw_output_ref",
        "frame",
        "camera_frame_id",
        "world_pose",
        "translation_xyz",
        "rotation_matrix",
        "gripper_tip_position_xyz",
        "observation_summary",
        "motion_summary",
    ):
        if key in details:
            structured = {
                "best_grasp_candidate",
                "grasp_candidates",
                "world_pose",
                "rotation_matrix",
                "selected_detection",
                "selection_bundle",
                "observation_summary",
                "motion_summary",
            }
            max_depth = 5 if key in structured else 2
            max_items = 32 if key in {"observation_summary", "motion_summary"} else 16
            compact[key] = _compact_value(
                details[key],
                max_depth=max_depth,
                max_items=max_items,
            )
    outputs = details.get("outputs")
    if isinstance(outputs, dict):
        useful_outputs: JsonDict = {}
        for key in (
            "result",
            "detection_count",
            "detections",
            "candidate_count",
            "best_grasp_candidate",
            "grasp_candidates",
            "ranking",
            "result_id",
            "selection_required",
            "selected_detection",
            "selection_bundle",
            "source_rgb",
            "source_depth",
            "target_mask",
            "frame",
            "camera_frame_id",
            "world_pose",
            "translation_xyz",
            "rotation_matrix",
            "gripper_tip_position_xyz",
            "observation_summary",
            "motion_summary",
            "response",
            "mcp",
            "schema_version",
            "query",
            "answer",
            "answer_truncated",
            "status",
            "requested_roles",
            "completed_roles",
            "barrier_order",
            "stop_reason",
            "result_count",
            "results",
            "search_call_count",
            "provider_role",
            "provider",
            "model",
            "url",
            "content_type",
            "title",
            "text",
            "truncated",
            "returned_char_count",
            "source_byte_count",
            "untrusted_external_content",
        ):
            if key in outputs:
                if key in {"answer", "text"}:
                    useful_outputs[key] = _compact_web_text(outputs[key])
                    continue
                structured = {
                    "best_grasp_candidate",
                    "grasp_candidates",
                    "world_pose",
                    "rotation_matrix",
                    "selected_detection",
                    "selection_bundle",
                    "observation_summary",
                    "motion_summary",
                    "results",
                }
                max_depth = 5 if key in structured else 2
                max_items = 32 if key in {"observation_summary", "motion_summary"} else 16
                useful_outputs[key] = _compact_value(
                    outputs[key],
                    max_depth=max_depth,
                    max_items=max_items,
                )
        if useful_outputs:
            compact["outputs"] = useful_outputs
    state_delta = details.get("state_delta")
    if isinstance(state_delta, dict) and state_delta:
        compact["state_delta"] = _compact_value(
            state_delta,
            max_depth=5,
            max_items=32,
        )
    artifacts = details.get("artifacts")
    if isinstance(artifacts, list):
        compact_artifacts = []
        for artifact in artifacts[:8]:
            if not isinstance(artifact, dict):
                continue
            compact_artifacts.append(
                {
                    key: _compact_value(artifact[key], max_depth=1)
                    for key in (
                        "type",
                        "kind",
                        "tool",
                        "index",
                        "label",
                        "path",
                        "mask_ref",
                        "overlay_ref",
                        "crop_ref",
                        "frame_id",
                        "rgb_path",
                        "depth_path",
                    )
                    if key in artifact
                }
            )
        if compact_artifacts:
            compact["artifacts"] = compact_artifacts
    diagnostics = details.get("diagnostics")
    if isinstance(diagnostics, list) and diagnostics:
        compact["diagnostics"] = _compact_value(diagnostics, max_depth=2)
    return compact


def _compact_request_payload(request: JsonDict) -> JsonDict:
    return {
        "kind": request.get("kind"),
        "name": request.get("name"),
        "parameters": _compact_value(request.get("parameters")),
        "reasoning": _compact_value(request.get("reasoning")),
    }


def _compact_metadata(metadata: JsonDict) -> JsonDict:
    compact: JsonDict = {}
    for key, value in metadata.items():
        if key in {"planner_metadata", "tool_context", "raw_backend_payload"}:
            compact[key] = "<omitted>"
        elif key == "control_spec":
            # Control specs are small host-owned contracts.  Flattening their
            # nested target lists into type/count descriptors makes the
            # planner guess motion parameters that the backend explicitly
            # advertised.  Keep the contract values while retaining strict
            # depth/item/string bounds and the normal inline-blob filtering.
            compact[key] = _compact_value(
                value,
                max_depth=8,
                max_items=32,
                preserve_control_specs=False,
            )
        elif key == "previous_action":
            compact[key] = _compact_previous_action(value)
        elif key in {"observation", "raw_payload"}:
            compact[key] = _compact_value(value, max_depth=1)
        else:
            compact[key] = _compact_value(value)
    return compact


def _compact_previous_action(value: Any) -> Any:
    if not isinstance(value, dict):
        return _compact_value(value, max_depth=1)
    if "command" in value and isinstance(value.get("command"), dict):
        return {
            "action_type": value.get("action_type"),
            "command": _compact_command_payload(value["command"]),
            "metadata": _compact_metadata(value.get("metadata") or {})
            if isinstance(value.get("metadata"), dict)
            else {},
        }
    if "tool_calls" in value or "request_name" in value or "request_kind" in value:
        return _compact_action_payload(value)
    return _compact_value(value, max_depth=1)


def _compact_named_item(item: JsonDict) -> JsonDict:
    return {
        "name": item.get("name"),
        "kind": item.get("kind"),
        "implemented": item.get("implemented"),
    }


def _compact_value(
    value: Any,
    *,
    max_depth: int = 2,
    max_items: int = 8,
    preserve_control_specs: bool = True,
) -> Any:
    if max_depth <= 0:
        if isinstance(value, dict):
            return {"type": "dict", "keys": sorted(str(key) for key in value)[:max_items]}
        if isinstance(value, list):
            return {"type": "list", "count": len(value)}
    if isinstance(value, str):
        return value if len(value) <= 300 else value[:300] + "...[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        compact: JsonDict = {}
        for idx, key in enumerate(sorted(value)):
            if idx >= max_items:
                compact["..."] = f"{len(value) - max_items} more keys"
                break
            if _looks_like_inline_blob_key(str(key)):
                compact[str(key)] = "<omitted>"
            elif key == "control_spec" and preserve_control_specs:
                compact[str(key)] = _compact_value(
                    value[key],
                    max_depth=8,
                    max_items=32,
                    preserve_control_specs=False,
                )
            else:
                compact[str(key)] = _compact_value(
                    value[key],
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    preserve_control_specs=preserve_control_specs,
                )
        return compact
    if isinstance(value, (list, tuple)):
        compact_list = [
            _compact_value(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                preserve_control_specs=preserve_control_specs,
            )
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            compact_list.append(f"... {len(value) - max_items} more items")
        return compact_list
    return str(type(value).__name__)


def _compact_web_text(value: Any, *, max_chars: int = 4000) -> Any:
    if not isinstance(value, str):
        return _compact_value(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"


def _looks_like_inline_blob_key(key: str) -> bool:
    lowered = key.lower()
    return "base64" in lowered or lowered in {
        "rgb",
        "depth",
        "image",
        "pixels",
        "array",
        "raw_payload",
    }


def _complete_observation_rgbd_views(observation: EnvObservation) -> list[JsonDict]:
    artifacts = observation.metadata.get("image_artifacts")
    if not isinstance(artifacts, list):
        return []
    cameras = {camera.frame_id: camera for camera in observation.cameras}
    views: list[JsonDict] = []
    for rgb in artifacts:
        if (
            not isinstance(rgb, dict)
            or rgb.get("kind") != "rgb"
            or not isinstance(rgb.get("path"), str)
            or not rgb["path"]
        ):
            continue
        frame_id = str(rgb.get("frame_id") or "")
        depth = next(
            (
                artifact
                for artifact in artifacts
                if isinstance(artifact, dict)
                and artifact.get("kind") == "depth"
                and str(artifact.get("frame_id") or "") == frame_id
                and isinstance(artifact.get("path"), str)
                and artifact["path"]
            ),
            None,
        )
        camera = cameras.get(frame_id)
        if not isinstance(depth, dict) or camera is None or not camera.intrinsics:
            continue
        view: JsonDict = {
            "frame_id": frame_id,
            "rgb_path": str(rgb["path"]),
            "depth_path": str(depth["path"]),
        }
        role = str(rgb.get("role") or camera.role or "")
        if role:
            view["role"] = role
        views.append(view)
    return views


def _record_failed_grasp_reestimate_view(
    reestimate: JsonDict,
    *,
    source_image: str,
    failure: str,
) -> None:
    """Advance a grasp retry to another passive view without reusing failed pixels."""

    attempted = [
        str(path)
        for path in reestimate.get("attempted_view_images", [])
        if isinstance(path, str) and path
    ]
    if source_image and source_image not in attempted:
        attempted.append(source_image)
    remaining = [
        view
        for view in reestimate.get("observation_views", [])
        if isinstance(view, dict)
        and str(view.get("rgb_path") or "")
        and str(view.get("rgb_path") or "") not in attempted
    ]
    reestimate.update(
        {
            "status": "ready" if remaining else "passive_views_exhausted",
            "attempted_view_images": attempted,
            "remaining_view_count": len(remaining),
            "last_view_failure": failure,
        }
    )


def summarize_observation(observation: EnvObservation) -> JsonDict:
    """Create a compact, JSON-friendly observation summary for memory."""

    return {
        "task": observation.task,
        "camera_ids": [camera.frame_id for camera in observation.cameras],
        "num_cameras": len(observation.cameras),
        "robot": {
            "joint_positions": observation.robot.joint_positions,
            "joint_velocities": observation.robot.joint_velocities,
            "end_effector_pose": observation.robot.end_effector_pose,
            "gripper_state": observation.robot.gripper_state,
            "base_pose": observation.robot.base_pose,
            "metadata": _compact_metadata(observation.robot.metadata),
        },
        "objects": [_compact_value(obj) for obj in observation.objects[:8]],
        "object_count": len(observation.objects),
        "metadata": _compact_metadata(observation.metadata),
    }
