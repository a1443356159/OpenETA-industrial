"""The sole Gym-shaped Gazebo DirectEnv implementation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
from gymnasium import Env, spaces

from adapter.protocol import EnvObservation
from agent.runtime.collision_geometry import (
    orientation_invariant_radius_m,
    project_collision_geometry,
)

from .deployment import GazeboDeploymentConfig, worker_deployment_config
from .robot_control import JOINT_NAMES, neutral_relative_motion_guidance
from .native_grasp import (
    NativePickPlaceConfig,
    NativeGraspVerifier,
    ReasonCode,
    validated_pickplace_motion_guidance,
)
from .profiles import CONTROL, PHYSICS, STRUCTURED_RECEIPT, GazeboProfile, gazebo_profile
from .process import GazeboProcessError
from .process import GazeboNativeContactWindow
from .planning_scene import PlanningSceneError
from .robotiq_kinematics import AttachedTransportReliefUnavailable
from .runtime import GazeboRuntime
from .ros_control import _relative_pose


def _native_close_failure_classification(
    exc: Exception,
    *,
    attach_acked: bool,
) -> tuple[bool, bool, str]:
    """Classify only deterministic candidate failures as queue rejections."""

    detail = str(exc) or type(exc).__name__
    measured_collision = bool(
        attach_acked
        and isinstance(exc, PlanningSceneError)
        and detail.startswith(
            "planning-scene current state is invalid; collision_pairs="
        )
    )
    relief_unavailable = bool(
        attach_acked and isinstance(exc, AttachedTransportReliefUnavailable)
    )
    candidate_rejection = measured_collision or relief_unavailable
    infrastructure_error = bool(attach_acked and not candidate_rejection)
    failure_class = (
        "measured_attachment_collision"
        if measured_collision
        else "attached_transport_relief_unavailable"
        if relief_unavailable
        else "post_attach_infrastructure_failure"
        if infrastructure_error
        else "native_attach_unacknowledged"
    )
    return candidate_rejection, infrastructure_error, failure_class


def _detached_target_motion_audit(
    *,
    source_spec: Mapping[str, Any],
    before_xyz: tuple[float, float, float],
    before_quat_xyzw: tuple[float, float, float, float],
    after_xyz: tuple[float, float, float],
    after_quat_xyzw: tuple[float, float, float, float],
    physical_tolerance_m: float,
) -> dict[str, Any]:
    """Measure qualified-to-native target drift as maximum surface motion."""

    if not math.isfinite(physical_tolerance_m) or physical_tolerance_m <= 0.0:
        raise GazeboProcessError("DETACHED_TARGET_MOTION_TOLERANCE_INVALID")
    source_xyz = tuple(float(value) for value in source_spec.get("pose_xyz") or ())
    source_quat = tuple(
        float(value) for value in source_spec.get("pose_quat_xyzw") or ()
    )
    if len(source_xyz) != 3 or len(source_quat) != 4:
        raise GazeboProcessError("PLANNING_SCENE_TARGET_SOURCE_POSE_UNAVAILABLE")
    try:
        local_geometry = project_collision_geometry(
            object_xyz=(0.0, 0.0, 0.0),
            object_rotation=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            primitives=source_spec.get("primitives") or (),
            fallback_size_xyz=source_spec.get("size_xyz"),
        )
        radius_m = orientation_invariant_radius_m(
            local_geometry,
            object_xyz=(0.0, 0.0, 0.0),
        )
    except (TypeError, ValueError) as exc:
        raise GazeboProcessError(
            "PLANNING_SCENE_TARGET_GEOMETRY_UNAVAILABLE"
        ) from exc

    def displacement(
        left_xyz: tuple[float, float, float],
        left_quat: tuple[float, float, float, float],
        right_xyz: tuple[float, float, float],
        right_quat: tuple[float, float, float, float],
    ) -> dict[str, float]:
        translation = math.dist(left_xyz, right_xyz)
        rotation = _quaternion_distance_rad(left_quat, right_quat)
        rotational_surface = 2.0 * radius_m * math.sin(rotation / 2.0)
        return {
            "translation_m": translation,
            "rotation_rad": rotation,
            "rotational_surface_displacement_m": rotational_surface,
            "maximum_surface_displacement_m": translation + rotational_surface,
        }

    source_to_before = displacement(
        source_xyz, source_quat, before_xyz, before_quat_xyzw
    )
    before_to_after = displacement(
        before_xyz, before_quat_xyzw, after_xyz, after_quat_xyzw
    )
    source_to_after = displacement(
        source_xyz, source_quat, after_xyz, after_quat_xyzw
    )
    maximum_surface_displacement = max(
        source_to_after["maximum_surface_displacement_m"],
        before_to_after["maximum_surface_displacement_m"],
    )
    valid = maximum_surface_displacement <= physical_tolerance_m
    return {
        "schema_version": "openeta.detached_target_motion_audit.v1",
        "valid": valid,
        "reason_code": (
            "DETACHED_TARGET_STATIONARY"
            if valid
            else "GRASP_CONTACT_TARGET_DISPLACED"
        ),
        "measurement_boundary": (
            "planning_scene_source_and_native_pose_before_after_contact_motion"
        ),
        "target_geometry_radius_m": radius_m,
        "physical_tolerance_m": physical_tolerance_m,
        "source_to_before": source_to_before,
        "before_to_after": before_to_after,
        "source_to_after": source_to_after,
        "maximum_surface_displacement_m": maximum_surface_displacement,
        "failure_phase": (
            "none"
            if valid
            else "preexisting_target_pose_drift"
            if source_to_before["maximum_surface_displacement_m"]
            > physical_tolerance_m
            else "moveit_contact_approach"
        ),
        "geometry_source": "authoritative_planning_scene_collision_geometry",
        "host_offset_pose_generated": False,
    }


def build_gazebo_control_spec(profile: GazeboProfile) -> dict[str, Any]:
    """Expose profile-owned control capabilities through the existing receipt."""

    spec: dict[str, Any] = {
        "read_only": CONTROL not in profile.capabilities,
        "rgbd_observation": profile.name == "rgbd_observation",
        "motion_control": CONTROL in profile.capabilities,
        "native_grasp": PHYSICS in profile.capabilities,
        "physical_verification": PHYSICS in profile.capabilities,
        "model_id": getattr(profile.model_config, "model_id", None),
    }
    if profile.name == "rm75_robotiq2f85_control":
        spec["validated_relative_motion"] = neutral_relative_motion_guidance()
    if isinstance(profile.model_config, NativePickPlaceConfig):
        spec["validated_pickplace_motion"] = validated_pickplace_motion_guidance(
            profile.model_config
        )
    return spec


class GazeboDirectEnv(Env):
    """Profile-driven DirectEnv for observation-only, motion-control, and guarded native-grasp.

    No Gazebo or ROS resource is started in ``__init__``.  The first reset is
    the authoritative lazy-start boundary.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        profile: GazeboProfile | str = "rgbd_observation",
        deployment: GazeboDeploymentConfig | None = None,
        runtime: GazeboRuntime | None = None,
        task: str = "",
        seed: int = 0,
        **_kwargs: Any,
    ) -> None:
        self.profile = gazebo_profile(profile) if isinstance(profile, str) else profile
        self.deployment = deployment or worker_deployment_config()
        if self.profile.model_config is not None:
            self.profile.model_config.validate_assets()
        self.runtime = runtime or GazeboRuntime(self.deployment, self.profile, task=task)
        self._task = task
        self._seed = int(seed)
        self._latest: dict[str, Any] | None = None
        self._backend = "gazebo"
        self.openeta_capabilities = self.profile.capabilities
        self.openeta_control_spec = build_gazebo_control_spec(self.profile)
        self.action_space = spaces.Discrete(1)
        runtime_config = getattr(self.runtime, "active_pick_place_config", None)
        self._native_grasp_config = (
            runtime_config
            if isinstance(runtime_config, NativePickPlaceConfig)
            else self.profile.model_config
            if isinstance(self.profile.model_config, NativePickPlaceConfig)
            else None
        )
        self._native_grasp_verifier = NativeGraspVerifier(self._native_grasp_config) if self._native_grasp_config is not None else None
        self._native_grasp_transport_locked = False
        self._attachment_transform: dict[str, Any] | None = None

    @property
    def controller(self) -> Any | None:
        return self.runtime.controller

    @staticmethod
    def _collision_filter_evidence(
        attachment: Any, *, attached: bool
    ) -> dict[str, Any] | None:
        """Validate the real runtime's ACKed target collision semantics.

        Lightweight dependency-injected test doubles predating the Gazebo
        plugin may omit this optional evidence method.  The production
        attachment controller always implements it and has already failed
        closed at the attach / detach boundary if the ACK was unavailable.
        """

        read = getattr(attachment, "collision_filter_evidence", None)
        if not callable(read):
            return None
        evidence = dict(read())
        expected_state = "robot_excluded" if attached else "full"
        if (
            evidence.get("schema_version")
            != "openeta.attached_collision_filter.v1"
            or evidence.get("state") != expected_state
            or evidence.get("target_environment_collision_enabled") is not True
            or evidence.get("target_robot_collision_enabled") is not (not attached)
        ):
            raise GazeboProcessError(
                "NATIVE_GRASP_COLLISION_FILTER_ACK_INVALID"
            )
        return evidence

    @staticmethod
    def _as_unified(observation: EnvObservation) -> dict[str, Any]:
        cameras: dict[str, dict[str, Any]] = {}
        for camera in observation.cameras:
            cameras[camera.frame_id] = {
                "rgb": np.asarray(camera.rgb, dtype=np.uint8),
                "depth": np.asarray(camera.depth, dtype=np.float32) if camera.depth is not None else None,
                "intrinsics": dict(camera.intrinsics),
                "extrinsics": dict(camera.extrinsics),
                "timestamp_s": camera.timestamp_s,
                "role": camera.role,
            }
        raw = {
            "task": observation.task,
            "cameras": cameras,
            "robot": observation.robot.to_dict(),
            "objects": list(observation.objects),
            "metadata": dict(observation.metadata),
        }
        return raw

    def _decorate_robot(self, raw: dict[str, Any]) -> dict[str, Any]:
        config = self.profile.model_config
        if config is not None:
            raw.setdefault("metadata", {}).update({
                "model_id": config.model_id,
                "eef_frame": config.mount_child,
                "joint_names": list(getattr(config, "joint_names", JOINT_NAMES)),
                "camera_frames": [item.frame_id for item in self.profile.cameras],
            })
        if self._native_grasp_config is not None:
            native_metadata = {
                "grasp_mechanism": "gazebo_sim8_detachable_joint",
                "contact_provenance": "gazebo_native_contacts",
            }
            dynamic_catalog = len(self._native_grasp_config.manipulation_targets) > 1
            if not dynamic_catalog or self._native_grasp_config.work_order_item is not None:
                native_metadata["attachment_target"] = self._native_grasp_config.target_id
            raw.setdefault("metadata", {}).update(native_metadata)
            progress = getattr(self.runtime, "multi_sort_progress", lambda: None)()
            if (
                isinstance(progress, Mapping)
                and not isinstance(
                    raw["metadata"].get("multi_sort_progress"), Mapping
                )
            ):
                raw["metadata"]["multi_sort_progress"] = dict(progress)
        # Work-order activation can change the selected target, destination,
        # and placement semantics without recreating the environment. Carry
        # the current profile contract with every observation; the MCP proxy's
        # creation-time copy is only a fallback for older environments.
        control_spec = getattr(self, "openeta_control_spec", None)
        if isinstance(control_spec, Mapping):
            raw.setdefault("metadata", {})["control_spec"] = dict(control_spec)
        return raw

    def observe(self) -> dict[str, Any]:
        raw = self._decorate_robot(self._as_unified(self.runtime.observe()))
        scene_revision = self._planning_scene_revision()
        if scene_revision is not None:
            raw.setdefault("metadata", {})["planning_scene_revision"] = scene_revision
        self._latest = raw
        return raw

    def _planning_scene_revision(self) -> int | None:
        controller = self.controller
        planning_scene = getattr(controller, "planning_scene", None)
        revision = getattr(planning_scene, "revision", None)
        if not isinstance(revision, int) or isinstance(revision, bool):
            revision = getattr(self.runtime, "scene_revision", None)
        return revision if isinstance(revision, int) and not isinstance(revision, bool) else None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._seed = int(seed)
        if self._native_grasp_verifier is not None:
            self._native_grasp_verifier.reset()
            self._native_grasp_transport_locked = False
            self._attachment_transform = None
        observation = self.runtime.reset(seed=self._seed)
        active_config = getattr(self.runtime, "active_pick_place_config", None)
        if isinstance(active_config, NativePickPlaceConfig):
            self._native_grasp_config = active_config
            self._native_grasp_verifier = NativeGraspVerifier(active_config)
        raw = self._decorate_robot(self._as_unified(observation))
        scene_revision = self._planning_scene_revision()
        if scene_revision is not None:
            raw.setdefault("metadata", {})["planning_scene_revision"] = scene_revision
        if self._native_grasp_verifier is not None:
            raw.setdefault("metadata", {})["physical_verification"] = self._native_grasp_verifier.last_record.to_dict()
        self._latest = raw
        reset_receipt = {
            "ok": True,
            "reset_seed": self._seed,
            **(
                {"planning_scene_revision": scene_revision}
                if scene_revision is not None
                else {}
            ),
        }
        return raw, {"_openeta_receipt": reset_receipt}

    def step(self, action: Any):
        raw_action = action if isinstance(action, Mapping) else {}
        action_type = str(raw_action.get("action_type") or "")
        contact_window: GazeboNativeContactWindow | None = None
        release_before_open: dict[str, Any] | None = None
        coordinated_open_result: tuple[EnvObservation, dict[str, Any]] | None = None
        detached_contact_motion_source: dict[str, Any] | None = None
        target_pose = raw_action.get("target_pose")
        detached_contact_motion = bool(
            self._native_grasp_config is not None
            and action_type in {"move_to", "follow_eef_trajectory"}
            and isinstance(target_pose, Mapping)
            and target_pose.get("grasp_stage") == "contact"
            and not self._native_grasp_transport_locked
        )
        if detached_contact_motion:
            attachment = getattr(self.runtime, "attachment", None)
            if attachment is None or getattr(attachment, "state", None) != "detached":
                detached_contact_motion = False
            else:
                try:
                    scene_source = self._planning_scene_target_pose_sync_source()
                    pose_reader = getattr(
                        attachment,
                        "native_target_mount_poses_with_retry",
                        None,
                    )
                    if callable(pose_reader):
                        before_pose, _mount_pose, before_attempts = pose_reader(
                            max_attempts=2
                        )
                    else:
                        before_pose, _mount_pose = (
                            attachment.native_target_mount_poses()
                        )
                        before_attempts = 1
                    detached_contact_motion_source = {
                        "scene_source": scene_source,
                        "before_xyz": tuple(float(value) for value in before_pose.xyz),
                        "before_quat_xyzw": tuple(
                            float(value) for value in before_pose.quat_xyzw
                        ),
                        "before_pose_read_attempt_count": int(before_attempts),
                    }
                except Exception as exc:
                    observation = self.runtime.observe()
                    raw = self._decorate_robot(self._as_unified(observation))
                    receipt = {
                        "ok": False,
                        "error_code": str(exc) or type(exc).__name__,
                        "failure_class": "detached_target_pose_infrastructure_failure",
                        "candidate_rejection": False,
                        "infrastructure_error": True,
                        "motion_outcome": "failed",
                        "execution_started": False,
                        "observation": raw,
                    }
                    self._latest = raw
                    return raw, 0.0, False, False, {
                        "_openeta_receipt": receipt
                    } if STRUCTURED_RECEIPT in self.profile.capabilities else {}
        if action_type == "configure_work_order":
            if self._native_grasp_transport_locked:
                raise GazeboProcessError("WORK_ORDER_RECONFIGURATION_DURING_TRANSPORT")
            configure = getattr(self.runtime, "configure_work_order", None)
            if not callable(configure):
                raise GazeboProcessError("WORK_ORDER_CONFIGURATION_UNAVAILABLE")
            items = raw_action.get("items")
            if not isinstance(items, list):
                raise GazeboProcessError("WORK_ORDER_ITEMS_INVALID")
            progress = configure(items=items)
            active_config = getattr(self.runtime, "active_pick_place_config", None)
            if not isinstance(active_config, NativePickPlaceConfig):
                raise GazeboProcessError("WORK_ORDER_ACTIVE_CONFIG_UNAVAILABLE")
            self._native_grasp_config = active_config
            self._native_grasp_verifier = NativeGraspVerifier(active_config)
            self.openeta_control_spec["validated_pickplace_motion"] = (
                validated_pickplace_motion_guidance(active_config)
            )
            observation = self.runtime.observe()
            raw = self._decorate_robot(self._as_unified(observation))
            raw.setdefault("metadata", {})["multi_sort_progress"] = dict(progress)
            raw["metadata"]["work_order"] = dict(progress["work_order"])
            revision = (progress.get("transition") or {}).get(
                "planning_scene_revision"
            )
            if isinstance(revision, int) and not isinstance(revision, bool):
                raw["metadata"]["planning_scene_revision"] = revision
            receipt = {
                "ok": True,
                "work_order": dict(progress["work_order"]),
                "multi_sort_progress": dict(progress),
                "native_target_binding": {
                    "target_id": active_config.target_id,
                    "target_link": active_config.target_link,
                    "assignment_id": active_config.work_order_item.get("id")
                    if active_config.work_order_item is not None
                    else None,
                },
                **(
                    {"planning_scene_revision": revision}
                    if isinstance(revision, int) and not isinstance(revision, bool)
                    else {}
                ),
                "observation": raw,
            }
            self._latest = raw
            return raw, 0.0, False, False, {
                "_openeta_receipt": receipt
            } if STRUCTURED_RECEIPT in self.profile.capabilities else {}
        if self._native_grasp_config is not None and action_type == "gripper_close":
            contact_window = GazeboNativeContactWindow(
                gz_executable=self.deployment.gz_executable,
                environment=dict(self.deployment.process_environment),
                simulation_time_provider=getattr(
                    self.controller,
                    "observation_barrier_s",
                    None,
                ),
            )
            try:
                contact_window.arm()
            except GazeboProcessError as exc:
                observation = self.runtime.observe()
                receipt = {"ok": False, "error_code": str(exc)}
                raw = self._decorate_robot(self._as_unified(observation))
                raw.setdefault("metadata", {})["physical_verification"] = self._native_grasp_verifier.last_record.to_dict() if self._native_grasp_verifier else {}
                receipt["observation"] = raw
                return raw, 0.0, False, False, {"_openeta_receipt": receipt}
        if self._native_grasp_config is not None and self._native_grasp_transport_locked and action_type in {"move_to", "follow_eef_trajectory"}:
            attachment = getattr(self.runtime, "attachment", None)
            if (
                (attachment is None or getattr(attachment, "state", None) != "attached")
            ):
                observation = self.runtime.observe()
                receipt = {
                    "ok": False,
                    "error_code": ReasonCode.ATTACH_ACK_MISSING.value,
                    "native_grasp_recovery_diagnostic": {
                        "verifier_phase": getattr(self._native_grasp_verifier, "phase", None),
                        "action_type": action_type,
                        "required_recovery": "gripper_open",
                    },
                }
                raw = self._decorate_robot(self._as_unified(observation))
                raw.setdefault("metadata", {})["physical_verification"] = self._native_grasp_verifier.last_record.to_dict() if self._native_grasp_verifier else {}
                receipt["observation"] = raw
                return raw, 0.0, False, False, {"_openeta_receipt": receipt}
        if (
            self._native_grasp_config is not None
            and self._native_grasp_verifier is not None
            and action_type == "gripper_open"
        ):
            attachment = getattr(self.runtime, "attachment", None)
            attached_before_open = self._native_grasp_verifier.attached or (
                attachment is not None
                and getattr(attachment, "state", None) == "attached"
            )
            if attached_before_open:
                release_sequence: list[dict[str, Any]] = []
                detached_acked = False
                open_future: Future[tuple[EnvObservation, dict[str, Any]]] | None = None

                with ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="openeta-release",
                ) as release_executor:
                    try:
                        if attachment is None:
                            raise GazeboProcessError(
                                "NATIVE_GRASP_DETACHABLE_JOINT_UNAVAILABLE"
                            )
                        target_pose, _ = attachment.native_target_mount_poses()
                        attachment.ensure_detached(require_ack=True)
                        detached_acked = True
                        release_sequence.append(
                            {
                                "sequence": 1,
                                "event": "native_detach_ack",
                                "state": "detached",
                            }
                        )
                        # Once Gazebo has acknowledged both the physical detach
                        # and collision-filter transition, begin opening at once.
                        # MoveIt's representation can be synchronized in parallel;
                        # neither operation changes the already-detached object.
                        open_future = release_executor.submit(
                            self.runtime.execute,
                            raw_action,
                        )
                        collision_filter = self._collision_filter_evidence(
                            attachment, attached=False
                        )
                        if collision_filter is not None:
                            release_sequence.append(
                                {
                                    "sequence": 2,
                                    "event": "attached_collision_filter_ack",
                                    **collision_filter,
                                }
                            )
                        sync_detach = getattr(
                            self.controller, "sync_planning_scene_detach", None
                        )
                        if not callable(sync_detach):
                            raise GazeboProcessError("PLANNING_SCENE_UNAVAILABLE")
                        scene_revision = sync_detach(
                            self._native_grasp_config,
                            target_xyz=target_pose.xyz,
                            target_quat_xyzw=target_pose.quat_xyzw,
                        )
                        release_sequence.append(
                            {
                                "sequence": len(release_sequence) + 1,
                                "event": "planning_scene_detach_ack",
                                "revision": int(scene_revision),
                            }
                        )
                        if open_future is None:
                            raise GazeboProcessError(
                                "GRIPPER_OPEN_DISPATCH_FAILED"
                            )
                        coordinated_open_result = open_future.result()
                        record = self._native_grasp_verifier.release_result(
                            detached_acked=True
                        )
                        self._native_grasp_transport_locked = False
                        self._attachment_transform = None
                        release_before_open = {
                            "release_sequence": release_sequence,
                            "target_pose": target_pose,
                            "planning_scene_revision": int(scene_revision),
                            "record": record,
                            "attached_collision_filter": collision_filter,
                            "release_coordination": {
                                "schema_version": (
                                    "openeta.native_release_coordination.v1"
                                ),
                                "mode": "detach_confirmation_triggers_open",
                                "native_detach_confirmed_before_open_dispatch": True,
                                "planning_scene_sync_concurrent_with_open_dispatch": True,
                                "blocking_stability_polling": False,
                                "placement_review": "vlm_post_release_observation",
                            },
                        }
                    except Exception as exc:
                        open_result = None
                        if open_future is not None:
                            try:
                                open_result = open_future.result()
                            except Exception:
                                open_result = None
                        record = self._native_grasp_verifier.release_result(
                            detached_acked=detached_acked
                        )
                        self._native_grasp_transport_locked = not detached_acked
                        if detached_acked:
                            self._attachment_transform = None
                        if open_result is None:
                            observation = self.runtime.observe()
                            open_receipt: dict[str, Any] = {}
                        else:
                            observation, open_receipt = open_result
                        raw = self._decorate_robot(self._as_unified(observation))
                        receipt = {
                            **open_receipt,
                            "ok": False,
                            "error_code": str(exc),
                            "gripper_open_executed": (
                                open_receipt.get("ok") is True
                            ),
                            "release_sequence": release_sequence,
                            "physical_verification": record.to_dict(),
                            "observation": raw,
                            "infrastructure_error": bool(detached_acked),
                        }
                        raw.setdefault("metadata", {})[
                            "physical_verification"
                        ] = record.to_dict()
                        return raw, 0.0, False, False, {
                            "_openeta_receipt": receipt
                        }
        try:
            if coordinated_open_result is None:
                observation, receipt = self.runtime.execute(raw_action)
            else:
                observation, receipt = coordinated_open_result
        except Exception:
            if contact_window is not None:
                contact_window.close()
            raise
        raw = self._decorate_robot(self._as_unified(observation))
        if (
            detached_contact_motion_source is not None
            and (receipt.get("execution_started") is True or receipt.get("ok") is True)
        ):
            attachment = getattr(self.runtime, "attachment", None)
            try:
                pose_reader = getattr(
                    attachment,
                    "native_target_mount_poses_with_retry",
                    None,
                )
                if callable(pose_reader):
                    after_pose, _mount_pose, after_attempts = pose_reader(
                        max_attempts=2
                    )
                else:
                    after_pose, _mount_pose = attachment.native_target_mount_poses()
                    after_attempts = 1
                scene_source = detached_contact_motion_source["scene_source"]
                planning_scene = getattr(self.controller, "planning_scene", None)
                source_spec = (
                    getattr(planning_scene, "world_specs", {}).get(
                        self._native_grasp_config.target_id
                    )
                    if planning_scene is not None
                    else None
                )
                if not isinstance(source_spec, Mapping):
                    raise GazeboProcessError(
                        "PLANNING_SCENE_TARGET_SOURCE_POSE_UNAVAILABLE"
                    )
                motion_audit = _detached_target_motion_audit(
                    source_spec=source_spec,
                    before_xyz=detached_contact_motion_source["before_xyz"],
                    before_quat_xyzw=detached_contact_motion_source[
                        "before_quat_xyzw"
                    ],
                    after_xyz=tuple(float(value) for value in after_pose.xyz),
                    after_quat_xyzw=tuple(
                        float(value) for value in after_pose.quat_xyzw
                    ),
                    physical_tolerance_m=float(
                        self._native_grasp_config.static_collision_penetration_tolerance_m
                    ),
                )
                motion_audit["native_pose_read_attempts"] = {
                    "before": detached_contact_motion_source[
                        "before_pose_read_attempt_count"
                    ],
                    "after": int(after_attempts),
                    "maximum_per_read": 2,
                }
                receipt["detached_target_motion_audit"] = motion_audit
                receipt["detachable_joint"] = {
                    "state": "detached",
                    "target_id": self._native_grasp_config.target_id,
                    "state_topic": self._native_grasp_config.state_topic,
                }
                if motion_audit["valid"] is not True:
                    sync_target_pose = getattr(
                        self.controller,
                        "sync_planning_scene_target_pose",
                        None,
                    )
                    if not callable(sync_target_pose):
                        raise GazeboProcessError("PLANNING_SCENE_UNAVAILABLE")
                    revision = sync_target_pose(
                        self._native_grasp_config,
                        target_xyz=tuple(float(value) for value in after_pose.xyz),
                        target_quat_xyzw=tuple(
                            float(value) for value in after_pose.quat_xyzw
                        ),
                        allow_target_touch=True,
                    )
                    pose_sync = self._planning_scene_target_pose_sync_evidence(
                        scene_source,
                        target_xyz=tuple(float(value) for value in after_pose.xyz),
                        target_quat_xyzw=tuple(
                            float(value) for value in after_pose.quat_xyzw
                        ),
                        revision=int(revision),
                        execution_started=True,
                    )
                    receipt.update(
                        {
                            "ok": False,
                            "error_code": "GRASP_CONTACT_TARGET_DISPLACED",
                            "failure_class": "detached_target_displacement",
                            "candidate_rejection": True,
                            "infrastructure_error": False,
                            "motion_outcome": "failed",
                            "execution_started": True,
                            "planning_scene_revision": int(revision),
                            "planning_scene_target_pose_sync": pose_sync,
                        }
                    )
                    raw.setdefault("metadata", {})[
                        "planning_scene_revision"
                    ] = int(revision)
            except Exception as exc:
                receipt.update(
                    {
                        "ok": False,
                        "error_code": str(exc) or type(exc).__name__,
                        "failure_class": (
                            "detached_target_pose_infrastructure_failure"
                        ),
                        "candidate_rejection": False,
                        "infrastructure_error": True,
                        "motion_outcome": "unknown",
                        "execution_started": True,
                    }
                )
        scene_revision = self._planning_scene_revision()
        if scene_revision is not None:
            receipt["planning_scene_revision"] = scene_revision
            raw.setdefault("metadata", {})["planning_scene_revision"] = scene_revision
        if self._native_grasp_config is not None and self._native_grasp_verifier is not None:
            attachment = getattr(self.runtime, "attachment", None)
            if action_type == "gripper_close":
                gate = None
                attach_acked = False
                attached_transport_hold: dict[str, Any] | None = None
                rollback_pose_sync_source: dict[str, Any] | None = None
                pose_snapshot_attempt_count = 0
                baseline_pose_snapshot_attempt_count = 0
                self._native_grasp_transport_locked = True
                try:
                    if receipt.get("ok") is not True:
                        raise GazeboProcessError(str(receipt.get("error_code") or "GRIPPER_FAILED"))
                    barrier_value = receipt.get("action_completed_ros_time_s")
                    barrier = float(barrier_value) if isinstance(barrier_value, int | float) else None
                    assert contact_window is not None
                    gate = contact_window.evaluate(close_completed_sim_time_s=barrier, config=self._native_grasp_config)
                    if not gate.accepted or attachment is None:
                        record = self._native_grasp_verifier.close_result(gate, attach_acked=False)
                        receipt.update({"ok": False, "error_code": record.reason_code.value, "native_contact_gate": gate.to_dict()})
                    else:
                        try:
                            rollback_pose_sync_source = (
                                self._planning_scene_target_pose_sync_source()
                            )
                        except Exception:
                            # Recovery evidence must never change a valid close.
                            # If unavailable, a later retry simply fails closed
                            # and requests fresh model evidence.
                            rollback_pose_sync_source = None
                        attachment.attach()
                        attach_acked = True
                        collision_filter = self._collision_filter_evidence(
                            attachment, attached=True
                        )
                        retrying_pose_reader = getattr(
                            attachment,
                            "native_target_mount_poses_with_retry",
                            None,
                        )
                        if callable(retrying_pose_reader):
                            (
                                target_pose,
                                mount_pose,
                                pose_snapshot_attempt_count,
                            ) = retrying_pose_reader(max_attempts=2)
                        else:
                            target_pose, mount_pose = (
                                attachment.native_target_mount_poses()
                            )
                            pose_snapshot_attempt_count = 1
                        relative_xyz, relative_quat = _relative_pose(
                            child_xyz=target_pose.xyz,
                            child_quat_xyzw=target_pose.quat_xyzw,
                            parent_xyz=mount_pose.xyz,
                            parent_quat_xyzw=mount_pose.quat_xyzw,
                        )
                        self._attachment_transform = {
                            "schema_version": "openeta.attachment_transform.v1",
                            "parent_frame": "eef",
                            "child_frame": "object",
                            "translation_xyz": list(relative_xyz),
                            "quat_xyzw": list(relative_quat),
                            "measurement_boundary": "native_attach_ack",
                        }
                        establish_transport_hold = getattr(
                            self.controller,
                            "establish_attached_transport_hold",
                            None,
                        )
                        if not callable(establish_transport_hold):
                            raise GazeboProcessError("GRIPPER_UNAVAILABLE")
                        transport_hold_attempts: list[dict[str, Any]] = []
                        while True:
                            hold_step = dict(establish_transport_hold())
                            if (
                                hold_step.get("schema_version")
                                != "openeta.attached_transport_hold.v1"
                                or hold_step.get(
                                    "object_environment_collision_enabled"
                                )
                                is not True
                            ):
                                raise GazeboProcessError(
                                    "ATTACHED_TRANSPORT_HOLD_FAILED"
                                )
                            hold_completed = hold_step.get(
                                "action_completed_ros_time_s"
                            )
                            if not isinstance(hold_completed, int | float):
                                raise GazeboProcessError(
                                    "ATTACHED_TRANSPORT_HOLD_FAILED"
                                )
                            prove_clearance = getattr(
                                contact_window,
                                "prove_contact_clearance",
                                None,
                            )
                            if not callable(prove_clearance):
                                raise GazeboProcessError(
                                    "NATIVE_GRASP_CONTACT_CLEARANCE_UNAVAILABLE"
                                )
                            clearance = dict(
                                prove_clearance(
                                    after_sim_time_s=float(hold_completed),
                                    duration_sim_s=(
                                        self._native_grasp_config.contact_post_close_hold_s
                                    ),
                                )
                            )
                            if (
                                clearance.get("schema_version")
                                != "openeta.native_pad_clearance.v1"
                                or type(clearance.get("cleared")) is not bool
                            ):
                                raise GazeboProcessError(
                                    "NATIVE_GRASP_CONTACT_CLEARANCE_INVALID"
                                )
                            transport_hold_attempts.append(
                                {
                                    **hold_step,
                                    "target_contact_clearance": clearance,
                                }
                            )
                            if clearance.get("cleared") is True:
                                break
                        first_hold = transport_hold_attempts[0]
                        last_hold = transport_hold_attempts[-1]
                        attached_transport_hold = {
                            "schema_version": "openeta.attached_transport_hold.v2",
                            "actuator_model": "single_common_driver",
                            "object_environment_collision_enabled": True,
                            "selection_policy": (
                                "open_by_terminal_bands_until_native_pads_clear"
                            ),
                            "attempt_count": len(transport_hold_attempts),
                            "attempts": transport_hold_attempts,
                            "measured_common_before_rad": first_hold[
                                "measured_common_before_rad"
                            ],
                            "measured_common_after_rad": last_hold[
                                "measured_common_after_rad"
                            ],
                            "measured_relief_rad": (
                                float(first_hold["measured_common_before_rad"])
                                - float(last_hold["measured_common_after_rad"])
                            ),
                            "target_contact_clearance": last_hold[
                                "target_contact_clearance"
                            ],
                            **(
                                {
                                    "attached_collision_filter": dict(
                                        collision_filter
                                    )
                                }
                                if collision_filter is not None
                                else {}
                            ),
                        }
                        sync_attach = getattr(self.controller, "sync_planning_scene_attach", None)
                        if not callable(sync_attach):
                            raise GazeboProcessError("PLANNING_SCENE_UNAVAILABLE")
                        scene_revision = sync_attach(
                            self._native_grasp_config,
                            target_xyz=target_pose.xyz,
                            target_quat_xyzw=target_pose.quat_xyzw,
                            mount_xyz=mount_pose.xyz,
                            mount_quat_xyzw=mount_pose.quat_xyzw,
                        )
                        record = self._native_grasp_verifier.close_result(gate, attach_acked=True)
                        baseline_attempts = attachment.capture_baseline()
                        if isinstance(baseline_attempts, int) and not isinstance(
                            baseline_attempts, bool
                        ):
                            baseline_pose_snapshot_attempt_count = baseline_attempts
                        self._native_grasp_transport_locked = False
                        receipt.update({
                            "native_contact_gate": gate.to_dict(),
                            "native_target_binding": {
                                "target_id": self._native_grasp_config.target_id,
                                "target_link": self._native_grasp_config.target_link,
                                "assignment_id": (
                                    self._native_grasp_config.work_order_item.get("id")
                                    if self._native_grasp_config.work_order_item is not None
                                    else None
                                ),
                            },
                            "detachable_joint": {
                                "state": "attached",
                                "target_id": self._native_grasp_config.target_id,
                                "attach_topic": self._native_grasp_config.attach_topic,
                                "state_topic": self._native_grasp_config.state_topic,
                                "collision_filter_state_topic": (
                                    self._native_grasp_config
                                    .attached_collision_filter_state_topic
                                ),
                            },
                            "planning_scene_revision": scene_revision,
                            "attachment_transform": dict(self._attachment_transform),
                            "attached_transport_hold": dict(
                                attached_transport_hold
                            ),
                            **(
                                {
                                    "attached_collision_filter": dict(
                                        collision_filter
                                    )
                                }
                                if collision_filter is not None
                                else {}
                            ),
                            "native_state_snapshot": {
                                "post_attach_attempt_count": pose_snapshot_attempt_count,
                                "baseline_attempt_count": (
                                    baseline_pose_snapshot_attempt_count
                                ),
                                "maximum_attempts_per_read": 2,
                            },
                        })
                except Exception as exc:
                    if pose_snapshot_attempt_count == 0:
                        pose_snapshot_attempt_count = int(
                            getattr(
                                attachment,
                                "_last_native_pose_read_attempt_count",
                                0,
                            )
                            or 0
                        )
                    attached_before_cleanup = getattr(attachment, "state", None) == "attached"
                    rollback_target_pose = None
                    if attached_before_cleanup:
                        try:
                            rollback_target_pose, _ = attachment.native_target_mount_poses()
                        except Exception:
                            rollback_target_pose = None
                    if attached_before_cleanup:
                        try:
                            attachment.ensure_detached(require_ack=True)
                            receipt["detachable_joint"] = {
                                "state": "detached",
                                "detach_topic": self._native_grasp_config.detach_topic,
                                "state_topic": self._native_grasp_config.state_topic,
                            }
                        except Exception:
                            pass
                    planning_scene = getattr(self.controller, "planning_scene", None)
                    if (
                        rollback_target_pose is not None
                        and self._native_grasp_config.target_id
                        in set(getattr(planning_scene, "attached_ids", set()))
                    ):
                        try:
                            sync_detach = getattr(
                                self.controller, "sync_planning_scene_detach", None
                            )
                            if not callable(sync_detach):
                                raise GazeboProcessError("PLANNING_SCENE_UNAVAILABLE")
                            scene_revision = sync_detach(
                                self._native_grasp_config,
                                target_xyz=rollback_target_pose.xyz,
                                target_quat_xyzw=rollback_target_pose.quat_xyzw,
                            )
                            receipt["planning_scene_revision"] = scene_revision
                            receipt["planning_scene_rollback"] = {
                                "state": "detached",
                                "revision": scene_revision,
                            }
                            if rollback_pose_sync_source is not None:
                                receipt["planning_scene_target_pose_sync"] = (
                                    self._planning_scene_target_pose_sync_evidence(
                                        rollback_pose_sync_source,
                                        target_xyz=rollback_target_pose.xyz,
                                        target_quat_xyzw=(
                                            rollback_target_pose.quat_xyzw
                                        ),
                                        revision=int(scene_revision),
                                        execution_started=True,
                                    )
                                )
                            raw.setdefault("metadata", {})[
                                "planning_scene_revision"
                            ] = scene_revision
                        except Exception as rollback_exc:
                            receipt["planning_scene_rollback"] = {
                                "state": "failed",
                                "detail": str(rollback_exc),
                            }
                    original_error = str(exc) or type(exc).__name__
                    (
                        candidate_attachment_failure,
                        post_attach_infrastructure_failure,
                        failure_class,
                    ) = _native_close_failure_classification(
                        exc,
                        attach_acked=attach_acked,
                    )
                    terminal_gate = (
                        gate
                        if gate is not None
                        else self._contact_unavailable_result()
                    )
                    record = (
                        self._native_grasp_verifier.attachment_state_rejected(
                            terminal_gate,
                            detail=original_error,
                        )
                        if candidate_attachment_failure
                        else self._native_grasp_verifier.close_result(
                            terminal_gate,
                            attach_acked=False,
                        )
                    )
                    receipt.update({
                        "ok": False,
                        "error_code": (
                            original_error
                            if post_attach_infrastructure_failure
                            else record.reason_code.value
                        ),
                        "physical_verification": record.to_dict(),
                        "detail": original_error,
                        "attach_acked_before_rollback": attach_acked,
                        "infrastructure_error": post_attach_infrastructure_failure,
                        "candidate_rejection": candidate_attachment_failure,
                        "failure_class": failure_class,
                        "motion_outcome": "failed",
                        "execution_started": True,
                        **(
                            {
                                "attached_transport_hold": dict(
                                    attached_transport_hold
                                )
                            }
                            if attached_transport_hold is not None
                            else {}
                        ),
                        "native_state_snapshot": {
                            "post_attach_attempt_count": pose_snapshot_attempt_count,
                            "baseline_attempt_count": (
                                baseline_pose_snapshot_attempt_count
                            ),
                            "maximum_attempts_per_read": 2,
                            "retry_exhausted": (
                                post_attach_infrastructure_failure
                                and original_error
                                == "NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"
                            ),
                        },
                    })
                    self._native_grasp_transport_locked = True
                    self._attachment_transform = None
                finally:
                    if contact_window is not None:
                        contact_window.close()
                raw.setdefault("metadata", {})["physical_verification"] = record.to_dict()
                receipt["physical_verification"] = record.to_dict()
            elif action_type == "gripper_open":
                if release_before_open is not None:
                    release_sequence = list(
                        release_before_open["release_sequence"]
                    )
                    open_executed = receipt.get("ok") is True
                    post_release_stage = "gripper_result"
                    try:
                        if not open_executed:
                            raise GazeboProcessError(
                                str(receipt.get("error_code") or "GRIPPER_FAILED")
                            )
                        release_sequence.append(
                            {
                                "sequence": len(release_sequence) + 1,
                                "event": "gripper_open_completed",
                                "ok": True,
                            }
                        )
                        # Detach and the physical open are already irreversible
                        # at this point. Publish their complete causal proof
                        # before PlanningScene bookkeeping or a multi-sort
                        # transition can fail independently.
                        receipt["release_sequence"] = release_sequence
                        receipt["gripper_open_executed"] = True
                        receipt["release_coordination"] = dict(
                            release_before_open["release_coordination"]
                        )
                        receipt["detachable_joint"] = {
                            "state": "detached",
                            "detach_topic": self._native_grasp_config.detach_topic,
                            "state_topic": self._native_grasp_config.state_topic,
                            "collision_filter_state_topic": (
                                self._native_grasp_config
                                .attached_collision_filter_state_topic
                            ),
                        }
                        if release_before_open.get(
                            "attached_collision_filter"
                        ) is not None:
                            receipt["attached_collision_filter"] = dict(
                                release_before_open[
                                    "attached_collision_filter"
                                ]
                            )
                        visual_observation_available = bool(
                            observation.cameras
                            and observation.metadata.get("observation_stale") is not True
                        )
                        post_release_visual_observation = {
                            "schema_version": (
                                "openeta.post_release_visual_observation.v1"
                            ),
                            "required": True,
                            "available": visual_observation_available,
                            "source": (
                                "causal_post_release_rgbd"
                                if visual_observation_available
                                else "fresh_observation_required"
                            ),
                            "camera_frame_ids": [
                                str(camera.frame_id) for camera in observation.cameras
                            ],
                            "review_authority": "vlm",
                        }
                        release_evidence = {
                            "schema_version": "openeta.native_release_evidence.v1",
                            "detached_confirmed": True,
                            "gripper_open_confirmed": True,
                            "post_release_visual_observation": (
                                post_release_visual_observation
                            ),
                            "geometry_obvious_failure_guard": {
                                "blocking_stability_polling": False,
                                "native_attachment_state": "detached",
                                "review_authority": "vlm",
                            },
                        }
                        receipt["release_evidence"] = release_evidence
                        # MoveIt still needs the released body at its measured
                        # world pose before another motion can be planned. One
                        # finite authoritative read is sufficient for that
                        # synchronization; repeated pose polling and a fixed
                        # stability dwell belong to visual adjudication, not
                        # to the gripper-open transaction.
                        post_release_stage = "released_target_pose_read"
                        pose_reader = getattr(
                            attachment,
                            "native_target_mount_poses_with_retry",
                            None,
                        )
                        if callable(pose_reader):
                            target_pose, _mount_pose, pose_attempts = pose_reader(
                                max_attempts=2
                            )
                        else:
                            target_pose, _mount_pose = (
                                attachment.native_target_mount_poses()
                            )
                            pose_attempts = 1
                        receipt["released_target_pose_read_attempt_count"] = int(
                            pose_attempts
                        )
                        sync_target_pose = getattr(
                            self.controller,
                            "sync_planning_scene_target_pose",
                            None,
                        )
                        if not callable(sync_target_pose):
                            raise GazeboProcessError("PLANNING_SCENE_UNAVAILABLE")
                        post_release_stage = "released_target_pose_sync"
                        scene_revision = sync_target_pose(
                            self._native_grasp_config,
                            target_xyz=target_pose.xyz,
                            target_quat_xyzw=target_pose.quat_xyzw,
                            # Detach is intentionally acknowledged before the
                            # fingers open. Permit only transient
                            # target-to-gripper touch while synchronizing the
                            # current state; every other robot/world collision
                            # remains a hard failure.
                            allow_target_touch=True,
                        )
                        release_sequence.append(
                            {
                                "sequence": len(release_sequence) + 1,
                                "event": "released_target_pose_sync_ack",
                                "revision": int(scene_revision),
                            }
                        )
                        record = release_before_open["record"]
                        receipt["planning_scene_revision"] = scene_revision
                        receipt["release_sequence"] = release_sequence
                        receipt["native_target_binding"] = {
                            "target_id": self._native_grasp_config.target_id,
                            "target_link": self._native_grasp_config.target_link,
                            "assignment_id": (
                                self._native_grasp_config.work_order_item.get("id")
                                if self._native_grasp_config.work_order_item is not None
                                else None
                            ),
                        }
                        raw.setdefault("metadata", {})[
                            "planning_scene_revision"
                        ] = scene_revision
                        post_release_stage = "work_order_transition"
                        advance = getattr(
                            self.runtime,
                            "complete_active_work_order_item",
                            None,
                        )
                        progress = (
                            advance(
                                release_evidence=release_evidence,
                                post_release_observation=(
                                    observation
                                    if visual_observation_available
                                    else None
                                ),
                            )
                            if callable(advance)
                            else None
                        )
                        if isinstance(progress, Mapping):
                            progress = dict(progress)
                            receipt["multi_sort_progress"] = progress
                            raw["metadata"]["multi_sort_progress"] = progress
                            next_revision = (progress.get("transition") or {}).get(
                                "planning_scene_revision"
                            )
                            if isinstance(next_revision, int) and not isinstance(
                                next_revision, bool
                            ):
                                raw["metadata"]["planning_scene_revision"] = next_revision
                                receipt[
                                    "next_assignment_planning_scene_revision"
                                ] = next_revision
                            if progress.get("all_completed") is not True:
                                next_config = getattr(
                                    self.runtime,
                                    "active_pick_place_config",
                                    None,
                                )
                                if not isinstance(
                                    next_config, NativePickPlaceConfig
                                ):
                                    raise GazeboProcessError(
                                        "MULTI_SORT_ACTIVE_CONFIG_UNAVAILABLE"
                                    )
                                self._native_grasp_config = next_config
                                self._native_grasp_verifier = NativeGraspVerifier(
                                    next_config
                                )
                                self.openeta_control_spec[
                                    "validated_pickplace_motion"
                                ] = validated_pickplace_motion_guidance(
                                    next_config
                                )
                                raw["metadata"]["control_spec"] = dict(
                                    self.openeta_control_spec
                                )
                        post_release_stage = "complete"
                    except Exception as exc:
                        record = release_before_open["record"]
                        error = str(exc).strip() or type(exc).__name__
                        receipt.update(
                            {
                                "ok": False,
                                "error_code": error,
                                "error_type": type(exc).__name__,
                                "infrastructure_error": open_executed,
                                "post_release_failure_stage": post_release_stage,
                                "gripper_open_executed": open_executed,
                                "release_sequence": release_sequence,
                            }
                        )
                else:
                    try:
                        if receipt.get("ok") is not True:
                            raise GazeboProcessError(
                                str(receipt.get("error_code") or "GRIPPER_FAILED")
                            )
                        if self._native_grasp_verifier.phase in {
                            "contact_rejected",
                            "attach_unacknowledged",
                            "attachment_rejected",
                        }:
                            pose_sync = self._sync_failed_close_target_pose()
                            receipt["planning_scene_revision"] = pose_sync[
                                "revision"
                            ]
                            raw.setdefault("metadata", {})[
                                "planning_scene_revision"
                            ] = pose_sync["revision"]
                            receipt["planning_scene_target_pose_sync"] = pose_sync
                        record = self._native_grasp_verifier.detached_open_result()
                        receipt["detachable_joint"] = {
                            "state": "detached",
                            "already_detached": True,
                        }
                        self._native_grasp_transport_locked = False
                    except Exception as exc:
                        record = self._native_grasp_verifier.last_record
                        receipt.update({"ok": False, "error_code": str(exc)})
                        self._native_grasp_transport_locked = True
                raw.setdefault("metadata", {})["physical_verification"] = record.to_dict()
                receipt["physical_verification"] = record.to_dict()
            elif (
                action_type in {"move_to", "follow_eef_trajectory"}
                and attachment is not None
                and getattr(attachment, "state", None) == "attached"
            ):
                try:
                    proof = attachment.child_link_proof()
                    record = self._native_grasp_verifier.prove_retention(
                        proof, dart_supported=True
                    )
                except Exception:
                    record = self._native_grasp_verifier.prove_retention(
                        None, dart_supported=True
                    )
                raw.setdefault("metadata", {})["physical_verification"] = record.to_dict()
                receipt["physical_verification"] = record.to_dict()
                receipt["detachable_joint"] = {
                    "state": "attached",
                    "target_id": self._native_grasp_config.target_id,
                    "state_topic": self._native_grasp_config.state_topic,
                }
                receipt["native_target_binding"] = {
                    "target_id": self._native_grasp_config.target_id,
                    "target_link": self._native_grasp_config.target_link,
                    "assignment_id": (
                        self._native_grasp_config.work_order_item.get("id")
                        if self._native_grasp_config.work_order_item is not None
                        else None
                    ),
                }
                collision_filter = self._collision_filter_evidence(
                    attachment, attached=True
                )
                if collision_filter is not None:
                    receipt["attached_collision_filter"] = collision_filter
                if self._attachment_transform is not None:
                    receipt["attachment_transform"] = dict(self._attachment_transform)
                proof_evidence = dict(record.evidence)
                receipt["child_link_proof"] = (
                    proof_evidence
                    if {
                        "vertical_displacement_m",
                        "capture_relative_translation_m",
                    }
                    <= proof_evidence.keys()
                    else {"available": False, "reason_code": record.reason_code.value}
                )
                if record.reason_code is not ReasonCode.TARGET_HELD:
                    self._native_grasp_transport_locked = True
                    receipt.update({"ok": False, "error_code": record.reason_code.value})
            else:
                raw.setdefault("metadata", {})["physical_verification"] = self._native_grasp_verifier.last_record.to_dict()
        self._latest = raw
        # The Direct/Gym boundary owns the public unified observation.  Keep
        # the structured receipt anchored to that exact post-action object so
        # Direct acceptance and the MCP codec share the same freshness proof.
        receipt["observation"] = raw
        # Receipt is deliberately namespaced inside Gym info.  The generic MCP
        # control codec restores the established top-level wire fields.
        info = {"_openeta_receipt": receipt} if STRUCTURED_RECEIPT in self.profile.capabilities else {}
        return raw, 0.0, False, False, info

    def _contact_unavailable_result(self):
        from .native_grasp import ContactGateResult
        return ContactGateResult(False, ReasonCode.CONTACT_WINDOW_NOT_ARMED, 0, 0)

    def _sync_failed_close_target_pose(self) -> dict[str, Any]:
        """Synchronize a detached target pushed by a rejected close."""

        attachment = getattr(self.runtime, "attachment", None)
        if attachment is None or getattr(attachment, "state", None) == "attached":
            raise GazeboProcessError("NATIVE_GRASP_TARGET_POSE_UNAVAILABLE")
        source = self._planning_scene_target_pose_sync_source()
        target_pose, _ = attachment.native_target_mount_poses()
        sync_target_pose = getattr(
            self.controller, "sync_planning_scene_target_pose", None
        )
        if not callable(sync_target_pose):
            raise GazeboProcessError("PLANNING_SCENE_UNAVAILABLE")
        revision = sync_target_pose(
            self._native_grasp_config,
            target_xyz=target_pose.xyz,
            target_quat_xyzw=target_pose.quat_xyzw,
            allow_target_touch=True,
        )
        return self._planning_scene_target_pose_sync_evidence(
            source,
            target_xyz=target_pose.xyz,
            target_quat_xyzw=target_pose.quat_xyzw,
            revision=int(revision),
            execution_started=False,
        )

    def _planning_scene_target_pose_sync_source(self) -> dict[str, Any]:
        """Capture the exact detached scene boundary before target motion."""

        planning_scene = getattr(self.controller, "planning_scene", None)
        target_id = self._native_grasp_config.target_id
        source_spec = (
            getattr(planning_scene, "world_specs", {}).get(target_id)
            if planning_scene is not None
            else None
        )
        source_revision = getattr(planning_scene, "revision", None)
        source_world_ids = sorted(getattr(planning_scene, "world_ids", set()))
        source_attached_ids = sorted(
            getattr(planning_scene, "attached_ids", set())
        )
        if not (
            isinstance(source_spec, Mapping)
            and isinstance(source_revision, int)
            and not isinstance(source_revision, bool)
        ):
            raise GazeboProcessError("PLANNING_SCENE_TARGET_SOURCE_POSE_UNAVAILABLE")
        source_pose = {
            "frame": "world",
            "translation_xyz": list(source_spec.get("pose_xyz") or []),
            "quat_xyzw": list(source_spec.get("pose_quat_xyzw") or []),
        }
        static_world_sha256_before = _static_world_sha256(
            planning_scene, target_id=target_id
        )
        return {
            "target_id": target_id,
            "source_revision": source_revision,
            "source_target_pose": source_pose,
            "world_ids_before": source_world_ids,
            "attached_ids_before": source_attached_ids,
            "static_world_sha256_before": static_world_sha256_before,
        }

    def _planning_scene_target_pose_sync_evidence(
        self,
        source: Mapping[str, Any],
        *,
        target_xyz: tuple[float, float, float],
        target_quat_xyzw: tuple[float, float, float, float],
        revision: int,
        execution_started: bool,
    ) -> dict[str, Any]:
        """Prove one source-to-current detached target pose transition."""

        planning_scene = getattr(self.controller, "planning_scene", None)
        target_id = str(source.get("target_id") or "")
        source_revision = source.get("source_revision")
        source_pose = source.get("source_target_pose")
        source_world_ids = source.get("world_ids_before")
        source_attached_ids = source.get("attached_ids_before")
        static_world_sha256_before = source.get("static_world_sha256_before")
        if not (
            target_id == self._native_grasp_config.target_id
            and isinstance(source_revision, int)
            and not isinstance(source_revision, bool)
            and isinstance(source_pose, Mapping)
            and isinstance(source_world_ids, list)
            and isinstance(source_attached_ids, list)
            and isinstance(static_world_sha256_before, str)
        ):
            raise GazeboProcessError("PLANNING_SCENE_TARGET_SOURCE_POSE_UNAVAILABLE")
        measured_pose = {
            "frame": "world",
            "translation_xyz": list(target_xyz),
            "quat_xyzw": list(target_quat_xyzw),
        }
        target_world_ids = sorted(getattr(planning_scene, "world_ids", set()))
        target_attached_ids = sorted(
            getattr(planning_scene, "attached_ids", set())
        )
        static_world_sha256_after = _static_world_sha256(
            planning_scene, target_id=target_id
        )
        translation_delta_m = math.dist(
            source_pose["translation_xyz"], measured_pose["translation_xyz"]
        )
        rotation_delta_rad = _quaternion_distance_rad(
            source_pose["quat_xyzw"], measured_pose["quat_xyzw"]
        )
        return {
            "schema_version": "openeta.planning_scene_target_pose_sync.v1",
            "status": (
                "verified_unchanged_from_native_world_pose"
                if int(revision) == source_revision
                else "synchronized_from_native_world_pose"
            ),
            "operation": "update_world_target",
            "source_revision": source_revision,
            "revision": int(revision),
            "target_id": target_id,
            "source_target_pose": source_pose,
            "target_pose": measured_pose,
            "translation_delta_m": translation_delta_m,
            "rotation_delta_rad": rotation_delta_rad,
            "world_ids_before": source_world_ids,
            "world_ids_after": target_world_ids,
            "attached_ids_before": source_attached_ids,
            "attached_ids_after": target_attached_ids,
            "static_world_sha256_before": static_world_sha256_before,
            "static_world_sha256_after": static_world_sha256_after,
            "topology_unchanged": (
                source_world_ids == target_world_ids
                and source_attached_ids == target_attached_ids
            ),
            "static_world_unchanged": (
                static_world_sha256_before == static_world_sha256_after
            ),
            "execution_started": execution_started,
        }

    def render(self):
        if self._latest is None:
            return None
        camera = next(iter(self._latest.get("cameras", {}).values()), {})
        return camera.get("rgb")

    def close(self) -> None:
        self.runtime.close()
        self._latest = None


def _static_world_sha256(planning_scene: object, *, target_id: str) -> str:
    specs = getattr(planning_scene, "world_specs", {})
    if not isinstance(specs, Mapping):
        raise GazeboProcessError("PLANNING_SCENE_STATIC_WORLD_UNAVAILABLE")
    payload = {
        str(object_id): specs[object_id]
        for object_id in sorted(specs)
        if str(object_id) != target_id
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _quaternion_distance_rad(source: object, target: object) -> float:
    try:
        left = [float(value) for value in source]
        right = [float(value) for value in target]
    except (TypeError, ValueError) as exc:
        raise GazeboProcessError("PLANNING_SCENE_TARGET_ORIENTATION_INVALID") from exc
    if len(left) != 4 or len(right) != 4:
        raise GazeboProcessError("PLANNING_SCENE_TARGET_ORIENTATION_INVALID")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise GazeboProcessError("PLANNING_SCENE_TARGET_ORIENTATION_INVALID")
    dot = abs(
        sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))
