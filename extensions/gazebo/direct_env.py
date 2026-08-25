"""The sole Gym-shaped Gazebo DirectEnv implementation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from gymnasium import Env, spaces

from adapter.protocol import EnvObservation

from .deployment import GazeboDeploymentConfig, worker_deployment_config
from .robot_control import JOINT_NAMES, neutral_relative_motion_guidance
from .native_grasp import (
    NativePickPlaceConfig,
    NativeGraspVerifier,
    ReasonCode,
    validated_pickplace_motion_guidance,
    verify_stable_placement,
)
from .profiles import CONTROL, PHYSICS, STRUCTURED_RECEIPT, GazeboProfile, gazebo_profile
from .process import GazeboProcessError
from .process import GazeboNativeContactWindow
from .runtime import GazeboRuntime
from .ros_control import _relative_pose


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
        self._native_grasp_config = self.profile.model_config if isinstance(self.profile.model_config, NativePickPlaceConfig) else None
        self._native_grasp_verifier = NativeGraspVerifier(self._native_grasp_config) if self._native_grasp_config is not None else None
        self._native_grasp_transport_locked = False
        self._attachment_transform: dict[str, Any] | None = None

    @property
    def controller(self) -> Any | None:
        return self.runtime.controller

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
            raw.setdefault("metadata", {}).update({
                "grasp_mechanism": "gazebo_sim8_detachable_joint",
                "contact_provenance": "gazebo_native_contacts",
                "attachment_target": self._native_grasp_config.target_id,
            })
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
        if self._native_grasp_config is not None and action_type == "gripper_close":
            contact_window = GazeboNativeContactWindow(
                gz_executable=self.deployment.gz_executable,
                environment=dict(self.deployment.process_environment),
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
        try:
            observation, receipt = self.runtime.execute(raw_action)
        except Exception:
            if contact_window is not None:
                contact_window.close()
            raise
        raw = self._decorate_robot(self._as_unified(observation))
        scene_revision = self._planning_scene_revision()
        if scene_revision is not None:
            receipt["planning_scene_revision"] = scene_revision
            raw.setdefault("metadata", {})["planning_scene_revision"] = scene_revision
        if self._native_grasp_config is not None and self._native_grasp_verifier is not None:
            attachment = getattr(self.runtime, "attachment", None)
            if action_type == "gripper_close":
                gate = None
                self._native_grasp_transport_locked = True
                try:
                    if receipt.get("ok") is not True:
                        raise GazeboProcessError(str(receipt.get("error_code") or "GRIPPER_FAILED"))
                    barrier_value = receipt.get("action_completed_ros_time_s")
                    barrier = float(barrier_value) if isinstance(barrier_value, int | float) else None
                    assert contact_window is not None
                    contact_window.begin_post_close()
                    gate = contact_window.evaluate(close_completed_sim_time_s=barrier, config=self._native_grasp_config)
                    if not gate.accepted or attachment is None:
                        record = self._native_grasp_verifier.close_result(gate, attach_acked=False)
                        receipt.update({"ok": False, "error_code": record.reason_code.value, "native_contact_gate": gate.to_dict()})
                    else:
                        attachment.attach()
                        target_pose, mount_pose = attachment.native_target_mount_poses()
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
                        attachment.capture_baseline()
                        self._native_grasp_transport_locked = False
                        receipt.update({
                            "native_contact_gate": gate.to_dict(),
                            "detachable_joint": {
                                "state": "attached",
                                "attach_topic": self._native_grasp_config.attach_topic,
                                "state_topic": self._native_grasp_config.state_topic,
                            },
                            "planning_scene_revision": scene_revision,
                            "attachment_transform": dict(self._attachment_transform),
                        })
                except Exception as exc:
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
                            raw.setdefault("metadata", {})[
                                "planning_scene_revision"
                            ] = scene_revision
                        except Exception as rollback_exc:
                            receipt["planning_scene_rollback"] = {
                                "state": "failed",
                                "detail": str(rollback_exc),
                            }
                    record = self._native_grasp_verifier.close_result(
                        gate if gate is not None else self._contact_unavailable_result(),
                        attach_acked=False,
                    )
                    receipt.update({"ok": False, "error_code": record.reason_code.value, "physical_verification": record.to_dict(), "detail": str(exc)})
                    self._native_grasp_transport_locked = True
                    self._attachment_transform = None
                finally:
                    if contact_window is not None:
                        contact_window.close()
                raw.setdefault("metadata", {})["physical_verification"] = record.to_dict()
                receipt["physical_verification"] = record.to_dict()
            elif action_type == "gripper_open":
                attached_before_open = self._native_grasp_verifier.attached or (
                    attachment is not None
                    and getattr(attachment, "state", None) == "attached"
                )
                if not attached_before_open:
                    try:
                        if receipt.get("ok") is not True:
                            raise GazeboProcessError(
                                str(receipt.get("error_code") or "GRIPPER_FAILED")
                            )
                        if self._native_grasp_verifier.phase == "contact_rejected":
                            pose_sync = self._sync_failed_close_target_pose()
                            receipt["planning_scene_revision"] = pose_sync["revision"]
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
                else:
                    try:
                        if receipt.get("ok") is not True:
                            raise GazeboProcessError(str(receipt.get("error_code") or "GRIPPER_FAILED"))
                        if attachment is None:
                            raise GazeboProcessError("NATIVE_GRASP_DETACHABLE_JOINT_UNAVAILABLE")
                        attachment.ensure_detached(require_ack=True)
                        receipt["detachable_joint"] = {
                            "state": "detached",
                            "detach_topic": self._native_grasp_config.detach_topic,
                            "state_topic": self._native_grasp_config.state_topic,
                        }
                        samples = attachment.sample_detached_target_poses(
                            duration_s=(
                                self._native_grasp_config.placement_settling_observation_s
                                + self._native_grasp_config.placement_stability_duration_s
                                + self._native_grasp_config.placement_terminal_window_s
                            ),
                            interval_s=self._native_grasp_config.placement_sample_interval_s,
                        )
                        placement = verify_stable_placement(samples, self._native_grasp_config)
                        target_pose = samples[-1]
                        sync_detach = getattr(self.controller, "sync_planning_scene_detach", None)
                        if not callable(sync_detach):
                            raise GazeboProcessError("PLANNING_SCENE_UNAVAILABLE")
                        scene_revision = sync_detach(
                            self._native_grasp_config,
                            target_xyz=target_pose.xyz,
                            target_quat_xyzw=target_pose.quat_xyzw,
                        )
                        record = self._native_grasp_verifier.release_result(detached_acked=True)
                        receipt["planning_scene_revision"] = scene_revision
                        receipt["placement_verification"] = placement.to_dict()
                        self._native_grasp_transport_locked = False
                        self._attachment_transform = None
                    except Exception as exc:
                        record = self._native_grasp_verifier.release_result(detached_acked=False)
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
                    "state_topic": self._native_grasp_config.state_topic,
                }
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
        )
        return {
            "status": "synchronized_from_native_world_pose",
            "revision": int(revision),
            "target_id": self._native_grasp_config.target_id,
            "execution_started": False,
        }

    def render(self):
        if self._latest is None:
            return None
        camera = next(iter(self._latest.get("cameras", {}).values()), {})
        return camera.get("rgb")

    def close(self) -> None:
        self.runtime.close()
        self._latest = None
