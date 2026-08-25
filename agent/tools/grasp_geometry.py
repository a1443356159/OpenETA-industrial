"""Deterministic geometry tools for compiling and refining grasp seeds."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict
from agent.runtime.calibration_registry import DEFAULT_GRASP_CALIBRATION_PROFILE
from agent.tools.registry import ToolExecutionContext, ToolHandler, ToolResult, make_tool_result


LEGACY_GRASP_CALIBRATION_SCHEMA = "libero.grasp_to_eef_calibration.v1"
GRASP_CALIBRATION_SCHEMA = "libero.grasp_to_eef_calibration.v2"
SUPPORTED_GRASP_CALIBRATION_SCHEMAS = {
    LEGACY_GRASP_CALIBRATION_SCHEMA,
    GRASP_CALIBRATION_SCHEMA,
}
COMPILED_GRASP_SCHEMA = "openeta.compiled_grasp_seed.v2"
COMPILED_PLACEMENT_SCHEMA = "openeta.compiled_placement_seed.v3"
DEFAULT_GRASP_PROFILE = DEFAULT_GRASP_CALIBRATION_PROFILE
_OPENCV_TO_OPENGL = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
_ARTICULATED_HANDLE_APPROACH_MODES = {"top_down", "front", "side"}


class GraspGeometryError(ValueError):
    """Raised when a grasp geometry contract cannot be satisfied."""


def qualification_grasp_pose_chain(outputs: Mapping[str, Any]) -> list[JsonDict]:
    """Return the model terminal contact pose bound by MoveIt qualification.

    The estimator owns the terminal grasp geometry.  MoveIt owns the complete
    collision-aware path from the current state to that terminal pose; the
    compiler must not invent spatial waypoints around it.
    """

    contact = outputs.get("contact_pose")
    if not isinstance(contact, Mapping):
        raise GraspGeometryError("compiled grasp contact pose is missing for qualification")
    return [dict(contact)]


def build_compile_grasp_seed_handler(
    profile_path: str | Path = DEFAULT_GRASP_PROFILE,
    *,
    qualification_cache: Any | None = None,
) -> ToolHandler:
    """Build an exact terminal-pose compiler with fixed embodiment calibration."""

    resolved_profile = Path(profile_path)

    def handler(context: ToolExecutionContext) -> ToolResult:
        try:
            profile, profile_sha256 = _load_profile(resolved_profile)
            parameters = dict(context.parameters)
            purpose = str(parameters.get("purpose") or "grasp").strip().lower()
            if purpose != "grasp":
                raise GraspGeometryError("compile_grasp_seed only accepts grasp candidates")
            if qualification_cache is not None:
                candidate_id = str(parameters.get("grasp_candidate_id") or "").strip()
                # A read-only compile may be repeated by a planner without the
                # convenience selector. Recover only the candidate identifier;
                # all geometry and proof still come exclusively from the host
                # PASS cache below.
                if not candidate_id:
                    camera_pose = parameters.get("camera_pose")
                    if isinstance(camera_pose, Mapping):
                        candidate_id = str(camera_pose.get("id") or "").strip()
                observation_metadata = (
                    context.observation.metadata
                    if context.observation is not None
                    else {}
                )
                supervision = context.metadata.get("supervision_context")
                memory = (
                    supervision.get("memory")
                    if isinstance(supervision, Mapping)
                    else None
                )
                # The runtime's epoch is the invalidation counter for candidate
                # proofs.  Gazebo observations may retain the reset epoch while
                # the host has advanced this counter after real mutations.
                # Prefer the runtime value so cache lookup and memory capture
                # bind the same scene version.
                host_binding = context.metadata.get(
                    "_openeta_host_candidate_compilation_binding"
                )
                host_binding = (
                    host_binding if isinstance(host_binding, Mapping) else {}
                )
                requested_epoch = host_binding.get("scene_epoch")
                if requested_epoch is None:
                    requested_epoch = (
                        memory.get("scene_epoch")
                        if isinstance(memory, Mapping)
                        else observation_metadata.get("scene_epoch")
                    )
                if requested_epoch is None:
                    requested_epoch = observation_metadata.get("scene_epoch")
                scene_epoch = (
                    _nonnegative_int(requested_epoch, "scene_epoch")
                    if requested_epoch is not None
                    else None
                )
                revision = host_binding.get(
                    "planning_scene_revision",
                    observation_metadata.get("planning_scene_revision"),
                )
                if isinstance(revision, bool) or (
                    revision is not None and not isinstance(revision, int)
                ):
                    raise GraspGeometryError("planning_scene_revision must be an integer")
                entry = qualification_cache.resolve(
                    purpose=purpose,
                    candidate_id=candidate_id,
                    scene_epoch=scene_epoch,
                    planning_scene_revision=revision,
                )
                if entry is None:
                    raise GraspGeometryError(
                        f"{purpose} candidate id has no current MoveIt PASS proof"
                    )
                scene_epoch = int(entry["scene_epoch"])
                proof_parameters = entry["proof"].get("compile_parameters")
                if not isinstance(proof_parameters, Mapping):
                    raise GraspGeometryError("qualified compile parameters are missing")
                parameters = dict(proof_parameters)
                parameters["purpose"] = purpose
                parameters["grasp_candidate_id"] = candidate_id
                parameters["camera_pose"] = dict(entry["candidate"])
                parameters["scene_epoch"] = scene_epoch
                if host_binding:
                    parameters["selection_source"] = "host_qualified_queue"
                if parameters.get("qualification_profile_sha256") != profile_sha256:
                    raise GraspGeometryError(
                        "MoveIt qualification calibration proof is stale"
                    )
            outputs = compile_grasp_seed(
                parameters,
                profile=profile,
                profile_sha256=profile_sha256,
            )
            qualified_pose_hash = parameters.get("qualified_compiled_pose_sha256")
            if qualified_pose_hash:
                pose_chain = qualification_grasp_pose_chain(outputs)
                actual_pose_hash = hashlib.sha256(
                    json.dumps(
                        pose_chain, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                if actual_pose_hash != qualified_pose_hash:
                    raise GraspGeometryError(
                        "compiled grasp pose differs from MoveIt qualification proof"
                    )
        except (
            OSError,
            json.JSONDecodeError,
            GraspGeometryError,
        ) as exc:
            return make_tool_result(
                context,
                success=False,
                content=f"grasp seed compilation failed: {exc}",
                outputs={"reason": "grasp_seed_compile_failed"},
                diagnostics=[
                    {
                        "code": "grasp_seed_compile_failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        return make_tool_result(
            context,
            success=True,
            content=(
                "retained placement candidate compiled to one exact world-frame EEF release pose"
                if outputs.get("purpose") == "placement"
                else "normalized grasp seed compiled to one exact world-frame EEF contact pose"
            ),
            outputs=outputs,
        )

    return handler


def build_compile_placement_seed_handler(
    profile_path: str | Path = DEFAULT_GRASP_PROFILE,
    *,
    qualification_cache: Any | None = None,
) -> ToolHandler:
    """Build the id-only placement compiler backed by host qualification proof."""

    resolved_profile = Path(profile_path)

    def handler(context: ToolExecutionContext) -> ToolResult:
        try:
            profile, profile_sha256 = _load_profile(resolved_profile)
            candidate_id = str(context.parameters.get("placement_candidate_id") or "").strip()
            if not candidate_id:
                raise GraspGeometryError("placement_candidate_id is required")
            if qualification_cache is None:
                raise GraspGeometryError("placement compilation requires a MoveIt PASS proof")
            observation_metadata = context.observation.metadata if context.observation else {}
            supervision = context.metadata.get("supervision_context")
            memory = supervision.get("memory") if isinstance(supervision, Mapping) else None
            placement_policy = (
                memory.get("placement_candidate_policy")
                if isinstance(memory, Mapping)
                else None
            )
            # Qualification is keyed by the runtime invalidation epoch, not the
            # simulator reset epoch carried by an observation.  The retained
            # placement policy records that host-owned binding alongside the
            # candidate queue; use it to resolve the exact proof that was
            # exposed to the planner.
            host_binding = context.metadata.get(
                "_openeta_host_candidate_compilation_binding"
            )
            host_binding = host_binding if isinstance(host_binding, Mapping) else {}
            epoch_value = host_binding.get("scene_epoch")
            if epoch_value is None:
                epoch_value = (
                    placement_policy.get("scene_epoch")
                    if isinstance(placement_policy, Mapping)
                    else observation_metadata.get("scene_epoch")
                )
            revision_value = host_binding.get("planning_scene_revision")
            if revision_value is None:
                revision_value = (
                    placement_policy.get("planning_scene_revision")
                    if isinstance(placement_policy, Mapping)
                    else observation_metadata.get("planning_scene_revision")
                )
            entry = qualification_cache.resolve(
                purpose="placement",
                candidate_id=candidate_id,
                scene_epoch=(epoch_value if isinstance(epoch_value, int) and not isinstance(epoch_value, bool) else None),
                planning_scene_revision=(revision_value if isinstance(revision_value, int) and not isinstance(revision_value, bool) else None),
            )
            if entry is None:
                raise GraspGeometryError("placement candidate id has no current MoveIt PASS proof")
            proof_parameters = entry["proof"].get("compile_parameters")
            if not isinstance(proof_parameters, Mapping):
                raise GraspGeometryError("qualified placement compile parameters are missing")
            parameters = dict(proof_parameters)
            qualified_candidate = parameters.get("placement_candidate")
            if not isinstance(qualified_candidate, Mapping):
                raise GraspGeometryError(
                    "qualified placement candidate geometry is missing"
                )
            qualified_candidate_id = str(qualified_candidate.get("id") or "")
            if qualified_candidate_id and qualified_candidate_id != candidate_id:
                raise GraspGeometryError(
                    "qualified placement candidate id does not match its proof"
                )
            parameters["placement_candidate_id"] = candidate_id
            # Compile the exact immutable geometry that produced the MoveIt
            # proof.  The public/cache candidate may carry later evidence-only
            # annotations (for example the physical collision-body goal); such
            # annotations must not perturb the compiled-placement hash.
            parameters["placement_candidate"] = dict(qualified_candidate)
            if host_binding:
                parameters["selection_source"] = "host_qualified_queue"
            if parameters.get("qualification_profile_sha256") != profile_sha256:
                raise GraspGeometryError("MoveIt qualification calibration proof is stale")
            expected_start_hash = parameters.get("qualified_start_state_sha256")
            if expected_start_hash:
                if context.observation is None:
                    raise GraspGeometryError("current robot state is unavailable")
                robot = context.observation.robot.to_dict()
                current_start_hash = hashlib.sha256(
                    json.dumps(
                        {
                            "joint_positions": robot.get("joint_positions", []),
                            "gripper_state": robot.get("gripper_state", {}),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                if current_start_hash != expected_start_hash:
                    raise GraspGeometryError(
                        "MoveIt qualification start joint or gripper state is stale"
                    )
            expected_attachment_hash = parameters.get(
                "qualified_attachment_transform_sha256"
            )
            if expected_attachment_hash:
                supervision = context.metadata.get("supervision_context")
                memory = supervision.get("memory") if isinstance(supervision, Mapping) else None
                gate = memory.get("attachment_gate") if isinstance(memory, Mapping) else None
                attachment_proof = (
                    gate.get("attachment_proof")
                    if isinstance(gate, Mapping)
                    else None
                )
                live_attachment = (
                    attachment_proof.get("attachment_transform")
                    if isinstance(attachment_proof, Mapping)
                    else None
                )
                if not isinstance(live_attachment, Mapping):
                    raise GraspGeometryError("current attachment transform is unavailable")
                live_attachment_hash = hashlib.sha256(
                    json.dumps(
                        live_attachment, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                if live_attachment_hash != expected_attachment_hash:
                    raise GraspGeometryError(
                        "MoveIt qualification attachment transform is stale"
                    )
            outputs = compile_placement_seed(
                parameters, profile=profile, profile_sha256=profile_sha256
            )
            pose_hash = hashlib.sha256(
                json.dumps(
                    [outputs["release_pose"]],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if pose_hash != parameters.get("qualified_compiled_pose_sha256"):
                raise GraspGeometryError("compiled placement pose differs from MoveIt qualification proof")
        except (GraspGeometryError, OSError, ValueError) as exc:
            return make_tool_result(
                context,
                success=False,
                content=f"placement seed compilation failed: {exc}",
                outputs={"reason": "placement_seed_compile_failed"},
                diagnostics=[
                    {
                        "code": "placement_seed_compile_failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        return make_tool_result(
            context,
            success=True,
            content="placement seed compiled from object goal and measured attachment",
            outputs=outputs,
        )

    return handler


def compile_grasp_seed(
    parameters: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    profile_sha256: str,
) -> JsonDict:
    purpose = str(parameters.get("purpose") or "grasp").strip().lower()
    if purpose != "grasp":
        raise GraspGeometryError("compile_grasp_seed only compiles grasp candidates")
    candidate = _mapping(parameters.get("camera_pose"), "camera_pose")
    extrinsics = _mapping(parameters.get("camera_extrinsics"), "camera_extrinsics")
    target_geometry_family = str(
        parameters.get("target_geometry_family") or parameters.get("target_class") or ""
    ).strip()
    approach_mode = str(parameters.get("approach_mode") or "").strip().lower()
    if approach_mode and approach_mode not in _ARTICULATED_HANDLE_APPROACH_MODES:
        raise GraspGeometryError(
            "approach_mode must be one of front, side, or top_down"
        )
    scene_epoch = _nonnegative_int(parameters.get("scene_epoch"), "scene_epoch")
    candidate_fallback = (
        parameters.get("candidate_fallback") is True
        or candidate.get("candidate_fallback") is True
    )
    _validate_profile(profile, target_class=target_geometry_family)

    candidate_id = str(candidate.get("id") or "").strip()
    final_refinable_fallback = candidate.get("final_refinable_fallback") is True
    if not candidate_id:
        raise GraspGeometryError("camera_pose.id is required")
    if str(candidate.get("frame") or "") != "camera":
        raise GraspGeometryError("camera_pose.frame must be 'camera'")
    if str(candidate.get("camera_frame") or "opencv").lower() != "opencv":
        raise GraspGeometryError("camera_pose.camera_frame must be 'opencv'")
    max_gripper_width = _bounded_float(
        profile.get("max_gripper_width_m"),
        "max_gripper_width_m",
        0.001,
        0.2,
    )
    width = _bounded_float(
        candidate.get("width"),
        "camera_pose.width",
        0.0,
        max_gripper_width,
    )
    calibration_id = str(profile.get("calibration_id") or "")
    if approach_mode and target_geometry_family not in {
        "articulated_handle",
        "drawer_handle",
    }:
        raise GraspGeometryError(
            "approach_mode is reserved for articulated_handle geometry"
        )

    r_camera_grasp = _rotation(candidate.get("rotation_matrix"), "camera_pose.rotation_matrix")
    p_camera_grasp = _vector(candidate.get("translation_xyz"), 3, "camera_pose.translation_xyz")
    r_world_cv, p_world_camera = _opencv_camera_to_world(extrinsics)

    transform = _mapping(profile.get("T_grasp_eef"), "T_grasp_eef")
    r_grasp_eef = _rotation(transform.get("rotation_matrix"), "T_grasp_eef.rotation_matrix")
    p_grasp_eef = _vector(transform.get("translation_xyz"), 3, "T_grasp_eef.translation_xyz")

    r_world_grasp = _matmul3(r_world_cv, r_camera_grasp)
    r_world_eef = _matmul3(r_world_grasp, r_grasp_eef)
    p_world_grasp = _add(_matvec3(r_world_cv, p_camera_grasp), p_world_camera)
    p_world_eef = _add(p_world_grasp, _matvec3(r_world_grasp, p_grasp_eef))
    approach_world = _normalise([r_world_grasp[row][0] for row in range(3)], "approach")
    native_downward_alignment = max(-1.0, min(1.0, -approach_world[2]))
    compiled_identity: JsonDict = {
        "candidate_id": candidate_id,
        "candidate": candidate,
        "extrinsics": extrinsics,
        "profile_sha256": profile_sha256,
        "scene_epoch": scene_epoch,
    }
    if approach_mode:
        compiled_identity["approach_mode"] = approach_mode
    compiled_id = hashlib.sha256(
        json.dumps(
            compiled_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]

    pose_common = {
        "frame": "world",
        "rotation_matrix": _round_matrix(r_world_eef),
        "source_grasp_id": candidate_id,
        "compiled_grasp_id": compiled_id,
        "calibration_id": calibration_id,
        "scene_epoch": scene_epoch,
    }
    if approach_mode:
        pose_common["approach_mode"] = approach_mode
    return {
        "schema_version": COMPILED_GRASP_SCHEMA,
        "compiled_grasp_id": compiled_id,
        "candidate_id": candidate_id,
        "selection_source": str(
            parameters.get("selection_source") or "host_qualified_queue"
        ),
        "camera_frame_id": str(parameters.get("camera_frame_id") or ""),
        "scene_epoch": scene_epoch,
        "target_class": target_geometry_family,
        "target_geometry_family": target_geometry_family,
        "source_backend": str(candidate.get("source_backend") or ""),
        **({"approach_mode": approach_mode} if approach_mode else {}),
        "calibration_id": calibration_id,
        "calibration_status": str(profile.get("status") or ""),
        "not_validated": profile.get("status") != "validated",
        "profile_sha256": profile_sha256,
        "approach_world_xyz": _round_vector(approach_world),
        "native_downward_alignment": round(native_downward_alignment, 6),
        "gripper_width_m": width,
        "final_refinable_fallback": final_refinable_fallback,
        "orientation_clamped": False,
        "terminal_pose_source": "model_pose_with_calibrated_frame_transform",
        "path_owner": "moveit",
        "candidate_fallback": candidate_fallback,
        **(
            {
                "fallback_reason": str(
                    parameters.get("fallback_reason")
                    or candidate.get("fallback_reason")
                )
            }
            if parameters.get("fallback_reason") or candidate.get("fallback_reason")
            else {}
        ),
        "contact_pose": {
            **pose_common,
            "xyz": _round_vector(p_world_eef),
            "grasp_stage": "contact",
        },
        "warning": (
            (
                "All articulated-handle approach modes failed; this is the one "
                "global score-selected fallback and remains subject to motion and "
                "attachment gates. "
            )
            if candidate_fallback
            and str(
                parameters.get("fallback_reason")
                or candidate.get("fallback_reason")
                or ""
            )
            == "all_approach_modes_failed"
            else (
                "All ranked candidates failed their strategy geometry checks; this is a "
                "score-selected fallback and remains subject to motion and attachment "
                "gates. "
            )
            if candidate_fallback
            else ""
        )
        + (
            "The model terminal pose is preserved exactly after calibrated frame/TCP "
            "conversion; MoveIt owns the complete path and the native attachment gate "
            "remains mandatory."
        ),
    }


def compile_placement_seed(
    parameters: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    profile_sha256: str,
) -> JsonDict:
    """Compile an object goal through the frozen measured attachment transform."""

    _validate_profile(profile, target_class="")
    candidate = _mapping(parameters.get("placement_candidate"), "placement_candidate")
    requested_id = str(parameters.get("placement_candidate_id") or "").strip()
    candidate_id = str(candidate.get("id") or "").strip()
    if not requested_id or requested_id != candidate_id:
        raise GraspGeometryError("placement candidate selection does not match retained candidate")
    pose = _mapping(candidate.get("object_goal_pose"), "placement_candidate.object_goal_pose")
    if str(pose.get("frame") or "") != "world":
        raise GraspGeometryError("placement object goal must be in the world frame")
    attachment = _mapping(parameters.get("attachment_transform"), "attachment_transform")
    if (
        str(attachment.get("parent_frame") or "") != "eef"
        or str(attachment.get("child_frame") or "") != "object"
    ):
        raise GraspGeometryError("attachment transform must be T_eef_object_attached")
    scene_epoch = _nonnegative_int(parameters.get("scene_epoch"), "scene_epoch")
    scene_revision = _nonnegative_int(
        parameters.get("scene_revision", scene_epoch), "scene_revision"
    )
    t_world_object = _pose_transform(pose, "object_goal_pose")
    t_eef_object = _pose_transform(attachment, "attachment_transform")
    t_world_eef = _matmul4(t_world_object, _inverse_rigid_transform(t_eef_object))
    r_world_eef = [row[:3] for row in t_world_eef[:3]]
    p_world_eef = [row[3] for row in t_world_eef[:3]]
    identity = {
        "purpose": "placement",
        "placement_candidate_id": candidate_id,
        "placement_candidate": candidate,
        "attachment_transform": attachment,
        "profile_sha256": profile_sha256,
        "scene_epoch": scene_epoch,
        "scene_revision": scene_revision,
    }
    compiled_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    common = {
        "frame": "world",
        "rotation_matrix": _round_matrix(r_world_eef),
        "placement_candidate_id": candidate_id,
        "compiled_placement_id": compiled_id,
        "calibration_id": str(profile.get("calibration_id") or ""),
        "scene_epoch": scene_epoch,
        "scene_revision": scene_revision,
        "compiled_eef_pose": True,
        "purpose": "placement",
    }
    return {
        "schema_version": COMPILED_PLACEMENT_SCHEMA,
        "purpose": "placement",
        "compiled_placement_id": compiled_id,
        "placement_candidate_id": candidate_id,
        "candidate_id": candidate_id,
        "attachment_transform_sha256": hashlib.sha256(
            json.dumps(attachment, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "scene_epoch": scene_epoch,
        "scene_revision": scene_revision,
        "selection_source": str(
            parameters.get("selection_source") or "host_qualified_queue"
        ),
        "profile_sha256": profile_sha256,
        "calibration_id": str(profile.get("calibration_id") or ""),
        "orientation_clamped": False,
        "terminal_pose_source": "anyplace_object_goal_with_measured_attachment",
        "path_owner": "moveit",
        "release_pose": {
            **common,
            "xyz": _round_vector(p_world_eef),
            "placement_stage": "release",
        },
    }


def materialize_world_object_goal(
    candidate: Mapping[str, Any],
    *,
    placement_camera_extrinsics: Mapping[str, Any],
    current_eef_pose: Mapping[str, Any],
    attachment_transform: Mapping[str, Any],
) -> JsonDict:
    """Bind an AnyPlace transform to current measured robot/attachment state."""

    placement = _mapping(
        candidate.get("object_placement_transform"),
        "object_placement_transform",
    )
    if str(placement.get("frame") or "") != "placement_camera":
        raise GraspGeometryError("AnyPlace transform must use the placement camera frame")
    raw = placement.get("transform_matrix")
    if not isinstance(raw, list) or len(raw) != 4:
        raise GraspGeometryError("object placement transform must be a 4x4 matrix")
    t_place_goal = [_vector(row, 4, "object_placement_transform") for row in raw]
    _rotation([row[:3] for row in t_place_goal[:3]], "object_placement_transform")
    if any(abs(a - b) > 1e-6 for a, b in zip(t_place_goal[3], [0, 0, 0, 1])):
        raise GraspGeometryError("object placement transform is not rigid")
    r_world_camera, p_world_camera = _opencv_camera_to_world(placement_camera_extrinsics)
    t_world_camera = _transform_matrix(r_world_camera, p_world_camera)
    t_world_eef = _pose_transform(current_eef_pose, "current_eef_pose")
    t_eef_object = _pose_transform(attachment_transform, "attachment_transform")
    t_world_object_current = _matmul4(t_world_eef, t_eef_object)
    t_world_object_goal = _matmul4(
        _matmul4(_matmul4(t_world_camera, t_place_goal), _inverse_rigid_transform(t_world_camera)),
        t_world_object_current,
    )
    # Preserve the complete SE(3) object goal predicted by AnyPlace.  Whether
    # that goal is feasible for the currently attached object is decided only
    # after deriving T_world_eef_goal with the measured T_eef_object and
    # running MoveIt qualification.  The host must not silently substitute a
    # gravity/yaw projection for the model's object orientation.
    r_world_object_goal = [row[:3] for row in t_world_object_goal[:3]]
    result = dict(candidate)
    result["object_goal_pose"] = {
        "frame": "world",
        "translation_xyz": _round_vector([row[3] for row in t_world_object_goal[:3]]),
        "rotation_matrix": _round_matrix(r_world_object_goal),
        "convention": "T_world_object_goal",
    }
    return result


def materialize_world_object_goal_from_current_pose(
    candidate: Mapping[str, Any],
    *,
    placement_camera_extrinsics: Mapping[str, Any],
    object_current_pose: Mapping[str, Any],
) -> JsonDict:
    """Bind an AnyPlace point transform before a physical attachment exists."""

    placement = _mapping(
        candidate.get("object_placement_transform"),
        "object_placement_transform",
    )
    raw = placement.get("transform_matrix")
    if str(placement.get("frame") or "") != "placement_camera" or not (
        isinstance(raw, list) and len(raw) == 4
    ):
        raise GraspGeometryError("object placement transform is invalid")
    t_place_goal = [_vector(row, 4, "object_placement_transform") for row in raw]
    _rotation([row[:3] for row in t_place_goal[:3]], "object_placement_transform")
    r_world_camera, p_world_camera = _opencv_camera_to_world(
        placement_camera_extrinsics
    )
    t_world_camera = _transform_matrix(r_world_camera, p_world_camera)
    t_world_object_current = _pose_transform(
        object_current_pose, "object_current_pose"
    )
    object_motion_world = _matmul4(
        _matmul4(t_world_camera, t_place_goal),
        _inverse_rigid_transform(t_world_camera),
    )
    t_world_object_goal = _matmul4(object_motion_world, t_world_object_current)
    result = dict(candidate)
    result["object_goal_pose"] = {
        "frame": "world",
        "translation_xyz": _round_vector([row[3] for row in t_world_object_goal[:3]]),
        "rotation_matrix": _round_matrix([row[:3] for row in t_world_object_goal[:3]]),
        "convention": "T_world_object_goal",
    }
    result["object_motion_world_transform"] = {
        "frame": "world",
        "transform_matrix": _round_matrix(object_motion_world),
        "convention": "T_world_motion_applied_left",
    }
    return result


def predicted_attachment_from_grasp(
    *,
    contact_pose: Mapping[str, Any],
    object_current_pose: Mapping[str, Any],
) -> JsonDict:
    """Predict T_eef_object for a candidate contact without mutating the scene."""

    t_world_eef = _pose_transform(contact_pose, "contact_pose")
    t_world_object = _pose_transform(object_current_pose, "object_current_pose")
    t_eef_object = _matmul4(_inverse_rigid_transform(t_world_eef), t_world_object)
    rotation = [row[:3] for row in t_eef_object[:3]]
    return {
        "schema_version": "openeta.predicted_attachment_transform.v1",
        "parent_frame": "eef",
        "child_frame": "object",
        "translation_xyz": _round_vector([row[3] for row in t_eef_object[:3]]),
        "rotation_matrix": _round_matrix(rotation),
        "quat_xyzw": _round_vector(_rotation_matrix_quat_xyzw(rotation)),
        "provenance": "model_contact_and_measured_object_frame",
    }


def _rotation_matrix_quat_xyzw(rotation: Sequence[Sequence[float]]) -> list[float]:
    m = [[float(value) for value in row] for row in rotation]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return [
            (m[2][1] - m[1][2]) / scale,
            (m[0][2] - m[2][0]) / scale,
            (m[1][0] - m[0][1]) / scale,
            0.25 * scale,
        ]
    index = max(range(3), key=lambda item: m[item][item])
    if index == 0:
        scale = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return [0.25 * scale, (m[0][1] + m[1][0]) / scale, (m[0][2] + m[2][0]) / scale, (m[2][1] - m[1][2]) / scale]
    if index == 1:
        scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return [(m[0][1] + m[1][0]) / scale, 0.25 * scale, (m[1][2] + m[2][1]) / scale, (m[0][2] - m[2][0]) / scale]
    scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return [(m[0][2] + m[2][0]) / scale, (m[1][2] + m[2][1]) / scale, 0.25 * scale, (m[1][0] - m[0][1]) / scale]


def grasp_candidate_approach_world(
    camera_pose: Mapping[str, Any],
    camera_extrinsics: Mapping[str, Any],
) -> list[float]:
    """Return a normalized candidate approach using compiler frame conventions."""

    r_camera_grasp = _rotation(
        camera_pose.get("rotation_matrix"),
        "camera_pose.rotation_matrix",
    )
    r_world_cv, _ = _opencv_camera_to_world(camera_extrinsics)
    r_world_grasp = _matmul3(r_world_cv, r_camera_grasp)
    return _normalise(
        [r_world_grasp[row][0] for row in range(3)],
        "approach",
    )


def camera_optical_forward_world(camera_extrinsics: Mapping[str, Any]) -> list[float]:
    """Return the OpenCV optical +Z direction in world coordinates."""

    r_world_cv, _ = _opencv_camera_to_world(camera_extrinsics)
    return _normalise([r_world_cv[row][2] for row in range(3)], "camera optical forward")


def _load_profile(path: Path) -> tuple[JsonDict, str]:
    data = path.read_bytes()
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise GraspGeometryError("calibration profile must contain one JSON object")
    return payload, hashlib.sha256(data).hexdigest()


def _validate_profile(profile: Mapping[str, Any], *, target_class: str) -> None:
    del target_class
    schema_version = profile.get("schema_version")
    if schema_version not in SUPPORTED_GRASP_CALIBRATION_SCHEMAS:
        raise GraspGeometryError("unsupported calibration profile schema")
    if profile.get("status") not in {"candidate", "validated"}:
        raise GraspGeometryError("calibration status must be candidate or validated")
    for key in ("robot_model", "gripper_model", "eef_frame"):
        if not isinstance(profile.get(key), str) or not str(profile.get(key)).strip():
            raise GraspGeometryError(f"calibration {key} must be a non-empty string")
    required = {
        "grasp_frame": "graspnet",
        "length_unit": "m",
        "rotation_convention": "active_column_vectors",
    }
    for key, expected in required.items():
        if profile.get(key) != expected:
            raise GraspGeometryError(f"calibration {key} does not match {expected}")


def _camera_to_world(extrinsics: Mapping[str, Any]) -> tuple[list[list[float]], list[float]]:
    rotation_value = extrinsics.get("mat")
    if isinstance(rotation_value, list) and len(rotation_value) == 9:
        position = _vector(extrinsics.get("pos"), 3, "camera_extrinsics.pos")
        flat = _vector(rotation_value, 9, "camera_extrinsics.mat")
        layout = str(extrinsics.get("matrix_layout") or "row_major").lower()
        if layout == "column_major":
            rotation = [[flat[row + col * 3] for col in range(3)] for row in range(3)]
        else:
            rotation = [flat[0:3], flat[3:6], flat[6:9]]
        _rotation(rotation, "camera_extrinsics.mat")
        return rotation, position
    quaternion = extrinsics.get("quat_xyzw")
    if isinstance(quaternion, list) and len(quaternion) == 4:
        position = _vector(extrinsics.get("pos"), 3, "camera_extrinsics.pos")
        qx, qy, qz, qw = _vector(quaternion, 4, "camera_extrinsics.quat_xyzw")
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-9:
            raise GraspGeometryError("camera_extrinsics.quat_xyzw must be non-zero")
        qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
        rotation = [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ]
        return _rotation(rotation, "camera_extrinsics.quat_xyzw"), position
    for key in ("camera_to_world", "pose_mat", "matrix"):
        matrix = extrinsics.get(key)
        if isinstance(matrix, list) and len(matrix) == 4:
            rows = [_vector(row, 4, f"camera_extrinsics.{key}") for row in matrix]
            rotation = _rotation([row[:3] for row in rows[:3]], f"camera_extrinsics.{key}")
            return rotation, [rows[0][3], rows[1][3], rows[2][3]]
    raise GraspGeometryError(
        "camera_extrinsics must contain pos+mat, pos+quat_xyzw, or a 4x4 matrix"
    )


def _opencv_camera_to_world(
    extrinsics: Mapping[str, Any],
) -> tuple[list[list[float]], list[float]]:
    """Return a transform whose local axes are OpenCV optical axes.

    Missing ``camera_frame`` keeps the historical OpenGL interpretation used
    by LIBERO and older simulator packets.  New simulator adapters must tag
    their normalized packet explicitly with ``camera_frame="opencv"``.
    """

    rotation, position = _camera_to_world(extrinsics)
    raw_frame = str(extrinsics.get("camera_frame") or "opengl")
    camera_frame = raw_frame.strip().lower().replace("-", "_").replace(" ", "_")
    if camera_frame in {"opencv", "opencv_optical", "cv"}:
        return rotation, position
    if camera_frame in {"opengl", "opengl_renderer", "mujoco", "renderer"}:
        return _matmul3(rotation, _OPENCV_TO_OPENGL), position
    raise GraspGeometryError(
        f"camera_extrinsics.camera_frame has unsupported value {raw_frame!r}"
    )


def _pose_transform(value: Mapping[str, Any], label: str) -> list[list[float]]:
    translation = _vector(
        value.get("translation_xyz", value.get("xyz")), 3, f"{label}.translation_xyz"
    )
    rotation_value = value.get("rotation_matrix")
    if rotation_value is not None:
        rotation = _rotation(rotation_value, f"{label}.rotation_matrix")
    else:
        quaternion = _vector(value.get("quat_xyzw"), 4, f"{label}.quat_xyzw")
        qx, qy, qz, qw = quaternion
        norm = math.sqrt(sum(component * component for component in quaternion))
        if norm <= 1e-9:
            raise GraspGeometryError(f"{label}.quat_xyzw must be non-zero")
        qx, qy, qz, qw = [component / norm for component in quaternion]
        rotation = [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    return _transform_matrix(rotation, translation)


def _transform_matrix(
    rotation: Sequence[Sequence[float]], translation: Sequence[float]
) -> list[list[float]]:
    return [
        [float(rotation[row][column]) for column in range(3)] + [float(translation[row])]
        for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def _matmul4(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [sum(float(left[row][k]) * float(right[k][column]) for k in range(4)) for column in range(4)]
        for row in range(4)
    ]


def _inverse_rigid_transform(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    rotation_t = [[float(matrix[column][row]) for column in range(3)] for row in range(3)]
    translation = [float(matrix[row][3]) for row in range(3)]
    inverse_translation = [-value for value in _matvec3(rotation_t, translation)]
    return _transform_matrix(rotation_t, inverse_translation)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GraspGeometryError(f"{label} must be an object")
    return value


def _vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise GraspGeometryError(f"{label} must contain {length} finite numbers")
    parsed = [_finite_float(item, label) for item in value]
    return parsed


def _rotation(value: Any, label: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise GraspGeometryError(f"{label} must be a 3x3 rotation matrix")
    matrix = [_vector(row, 3, label) for row in value]
    for row in range(3):
        norm = sum(matrix[row][col] * matrix[row][col] for col in range(3))
        if not math.isclose(norm, 1.0, abs_tol=1e-4):
            raise GraspGeometryError(f"{label} is not orthonormal")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1e-4):
        raise GraspGeometryError(f"{label} must have determinant +1")
    return matrix


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise GraspGeometryError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GraspGeometryError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise GraspGeometryError(f"{label} must be finite")
    return parsed


def _positive_float(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed <= 0:
        raise GraspGeometryError(f"{label} must be positive")
    return parsed


def _bounded_float(value: Any, label: str, lower: float, upper: float) -> float:
    parsed = _finite_float(value, label)
    if parsed < lower or parsed > upper:
        raise GraspGeometryError(f"{label} must be in [{lower}, {upper}]")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise GraspGeometryError(f"{label} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise GraspGeometryError(f"{label} must be a non-negative integer") from exc
    if parsed < 0:
        raise GraspGeometryError(f"{label} must be a non-negative integer")
    return parsed


def _matmul3(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _matvec3(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)]


def _add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [left[index] + right[index] for index in range(3)]


def _normalise(vector: Sequence[float], label: str) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-9:
        raise GraspGeometryError(f"{label} has zero length")
    return [value / norm for value in vector]


def _round_vector(vector: Sequence[float]) -> list[float]:
    return [0.0 if abs(value) < 1e-12 else round(float(value), 12) for value in vector]


def _round_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [_round_vector(row) for row in matrix]
