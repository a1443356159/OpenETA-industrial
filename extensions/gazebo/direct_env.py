"""The sole Gym-shaped Gazebo DirectEnv implementation."""

from __future__ import annotations

from contextlib import suppress
import time
from typing import Any, Mapping

import numpy as np
from gymnasium import Env, spaces

from adapter.protocol import EnvObservation

from .deployment import GazeboDeploymentConfig, worker_deployment_config
from .m2 import JOINT_NAMES
from .m3 import (
    AttachmentLifecycle,
    M3Config,
    M3Verifier,
    PadSnapshot,
    PadSurface,
    ReasonCode,
    confirm_pad_contact,
    fingertip_collision_bounds_m,
    pads_are_clear,
    quaternion_rotate,
    relative_pose,
    unknown_record,
    vector_norm,
)
from .profiles import CONTROL, PHYSICS, STRUCTURED_RECEIPT, GazeboProfile, gazebo_profile
from .runtime import GazeboRuntime


class GazeboDirectEnv(Env):
    """Profile-driven DirectEnv for M1, M2 and M3.

    No Gazebo or ROS resource is started in ``__init__``.  The first reset is
    the authoritative lazy-start boundary.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        profile: GazeboProfile | str = "m1",
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
        self.openeta_control_spec = {
            "read_only": CONTROL not in self.profile.capabilities,
            "m1": self.profile.name == "m1",
            "m2": CONTROL in self.profile.capabilities,
            "m3": PHYSICS in self.profile.capabilities,
            "physical_verification": PHYSICS in self.profile.capabilities,
            "model_id": getattr(self.profile.model_config, "model_id", None),
        }
        self.action_space = spaces.Discrete(1)
        self._m3_config = self.profile.model_config if isinstance(self.profile.model_config, M3Config) else None
        self._verifier = M3Verifier(self._m3_config) if self._m3_config is not None else None
        self._last_snapshot: Any | None = None
        self._attached_object: str | None = None
        self._attachment_lifecycle = AttachmentLifecycle.DETACHED

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
        if self._m3_config is not None:
            # Rollout metadata must state which attachment mechanism produced
            # any grasp evidence (physics friction vs the detachable fallback).
            raw.setdefault("metadata", {})["attachment_mode"] = getattr(
                self.runtime.deployment, "m3_attachment_mode", "physics"
            )
        return raw

    def _camera_timestamp(self, raw: Mapping[str, Any]) -> float:
        stamps = [
            float(value.get("timestamp_s", 0.0))
            for value in raw.get("cameras", {}).values()
            if isinstance(value, Mapping)
        ]
        if not stamps or min(stamps) <= 0:
            raise RuntimeError("M3_CAMERA_TIMESTAMP_MISSING")
        return min(stamps)

    def _pad_snapshot(self, snapshot: Any) -> PadSnapshot:
        """Read both frozen fingertip meshes through live TF for M3 attach."""

        config = self._m3_config
        runtime = getattr(self.runtime.controller, "runtime", None)
        if config is None or runtime is None:
            raise RuntimeError("M3_PAD_FEEDBACK_UNAVAILABLE")
        from rclpy.time import Time

        mesh_dir = config.gripper_asset_root / "meshes" / "collision" / "2f_85"
        values: dict[str, tuple[tuple[float, float, float], float]] = {}
        for link in config.fingertip_links:
            side = "left" if "left" in link else "right"
            transform = runtime.state_source.tf_buffer.lookup_transform(
                config.base_link, link, Time()
            ).transform
            translation = (
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
            )
            rotation = (
                float(transform.rotation.x),
                float(transform.rotation.y),
                float(transform.rotation.z),
                float(transform.rotation.w),
            )
            minimum, maximum = fingertip_collision_bounds_m(
                mesh_dir / f"{side}_finger_tip.stl"
            )
            local_centre = tuple((minimum[index] + maximum[index]) / 2 for index in range(3))
            offset = quaternion_rotate(rotation, local_centre)
            centre = tuple(translation[index] + offset[index] for index in range(3))
            values[side] = (
                centre,
                max((maximum[index] - minimum[index]) / 2 for index in range(3)),
            )
        left_centre, left_extent = values["left"]
        right_centre, right_extent = values["right"]
        axis = tuple(right_centre[index] - left_centre[index] for index in range(3))
        length = vector_norm(axis)
        if length <= 1e-9:
            raise RuntimeError("M3_PAD_FEEDBACK_DEGENERATE")
        left_normal = tuple(value / length for value in axis)
        return PadSnapshot(
            timestamp_s=snapshot.timestamp_s,
            left=PadSurface(left_centre, left_normal, left_extent),
            right=PadSurface(right_centre, tuple(-value for value in left_normal), right_extent),
            objects=snapshot.objects,
        )

    def _pad_feedback_samples(self, action_timestamp_s: float | None) -> tuple[PadSnapshot, ...]:
        """Collect post-action dual-TF/Odometry geometry without contacts."""

        config = self._m3_config
        source, controller = self.runtime.physics_source, self.runtime.controller
        if config is None or source is None or controller is None:
            raise RuntimeError("M3_PAD_FEEDBACK_UNAVAILABLE")
        samples: list[PadSnapshot] = []
        for index in range(config.pad_evidence_samples):
            if index:
                time.sleep(config.pad_evidence_window_s / (config.pad_evidence_samples - 1))
            robot = controller.state_provider().to_dict()
            ros_runtime = getattr(controller, "runtime", None)
            camera_stamp = (
                float(ros_runtime.ros_time_s()) if ros_runtime is not None else time.monotonic()
            )
            snapshot = source.capture(
                robot=robot,
                camera_timestamp_s=camera_stamp,
                min_timestamp_s=action_timestamp_s,
                timeout_s=5.0,
            )
            samples.append(self._pad_snapshot(snapshot))
        return tuple(samples)

    def _require_detachable_child_link_evidence(self, record: Any) -> Any:
        """Fail closed if model odometry disagrees with the joint child link.

        Gazebo Sim's stock OdometryPublisher samples a model entity, whereas
        DetachableJoint constrains the configured child link.  The two are
        known to disagree on the current RM75/DART graph, so the existing M3
        Odometry verdict cannot be accepted alone while detachable mode is
        enabled.  This check remains private and reuses an existing verifier
        reason instead of changing the observation or MCP contracts.
        """

        attachment = getattr(self.runtime, "attachment", None)
        if (
            attachment is None
            or self._attached_object is None
            or getattr(record, "reason_code", None) is not ReasonCode.TARGET_HELD
        ):
            return record
        label = (
            "target"
            if self._attached_object == self._m3_config.target_id
            else "distractor"
        )
        held_reader = getattr(attachment, "is_physically_held", None)
        if callable(held_reader) and held_reader(label):
            return record
        # Do not let a stale model pose promote the PlanningScene or preserve
        # a false held state.  The existing TARGET_NOT_LIFTED result already
        # expresses the only supported public semantics: no physical proof.
        self._verifier.reset()
        return unknown_record(
            ReasonCode.TARGET_NOT_LIFTED,
            phase="failed",
            target_id=self._m3_config.target_id,
        )

    def _merge_physics(
        self,
        raw: dict[str, Any],
        *,
        action_type: str | None = None,
        action_timestamp_s: float | None = None,
        gripper_result: Mapping[str, Any] | None = None,
        require: bool = False,
    ) -> tuple[dict[str, Any], Any | None]:
        if self._m3_config is None or self._verifier is None:
            return raw, None
        source = self.runtime.physics_source
        try:
            if source is None:
                raise RuntimeError("M3_PHYSICS_SOURCE_MISSING")
            snapshot = source.capture(
                robot=raw["robot"], camera_timestamp_s=self._camera_timestamp(raw),
                min_timestamp_s=action_timestamp_s, timeout_s=5.0,
                gripper_stalled=(bool(gripper_result["stalled"]) if gripper_result and "stalled" in gripper_result else None),
                gripper_reached_goal=(bool(gripper_result["reached_goal"]) if gripper_result and "reached_goal" in gripper_result else None),
            )
            record = self._verifier.verify(snapshot, action_type=action_type, action_timestamp_s=action_timestamp_s)
            record = self._require_detachable_child_link_evidence(record)
            self._last_snapshot = snapshot
            raw["objects"] = [item.to_dict() for item in snapshot.objects]
        except Exception as exc:
            if require:
                raise
            snapshot = None
            record = unknown_record(
                ReasonCode.DATA_MISSING, phase=self._verifier.phase,
                target_id=self._m3_config.target_id,
                evidence={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raw["objects"] = []
        physical = record.to_dict()
        raw.setdefault("robot", {}).setdefault("gripper_state", {}).update({
            "object_detection": record.object_detection,
            "grasp_confirmed": record.grasp_confirmed,
            "slip_detected": record.slip_detected,
        })
        raw.setdefault("metadata", {})["physical_verification"] = physical
        return raw, snapshot

    def observe(self) -> dict[str, Any]:
        raw = self._decorate_robot(self._as_unified(self.runtime.observe()))
        raw, _ = self._merge_physics(raw)
        self._latest = raw
        return raw

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._seed = int(seed)
        if self._verifier is not None:
            self._verifier.reset()
            self._last_snapshot = None
            self._attached_object = None
            self._attachment_lifecycle = AttachmentLifecycle.DETACHED
        observation = self.runtime.reset(seed=self._seed)
        raw = self._decorate_robot(self._as_unified(observation))
        raw, snapshot = self._merge_physics(raw, action_type="gripper_open", require=PHYSICS in self.profile.capabilities)
        if snapshot is not None:
            planning = getattr(self.runtime.physics_source, "planning_scene", None)
            target = snapshot.object(self._m3_config.target_id) if self._m3_config else None
            distractor = snapshot.object(self._m3_config.distractor_id) if self._m3_config else None
            if planning is not None and target is not None and distractor is not None:
                planning.initialize(target.pose, distractor.pose)
        self._latest = raw
        return raw, {}

    def step(self, action: Any):
        raw_action = action if isinstance(action, Mapping) else {}
        action_type = raw_action.get("action_type")
        planning = getattr(self.runtime.physics_source, "planning_scene", None)
        attachment = getattr(self.runtime, "attachment", None)
        if action_type == "gripper_open" and attachment is not None:
            # The release order is intentional: confirmed detach ACK first,
            # then physically open, then obtain a fresh observation before
            # releasing MoveIt's planning object.
            self._ensure_detached(attachment)
        observation, receipt = self.runtime.execute(raw_action)
        raw = self._decorate_robot(self._as_unified(observation))
        barrier_value = receipt.get("action_completed_ros_time_s")
        barrier = float(barrier_value) if barrier_value is not None else None
        raw, snapshot = self._merge_physics(
            raw, action_type=str(action_type) if action_type is not None else None,
            action_timestamp_s=barrier, gripper_result=receipt,
        )
        if self._m3_config is not None:
            physical = raw["metadata"]["physical_verification"]
            deadline = time.monotonic() + 2.5
            while (
                action_type == "gripper_open"
                and physical["reason_code"] == ReasonCode.NOT_SETTLED.value
                and time.monotonic() < deadline
            ):
                followup = self._decorate_robot(self._as_unified(
                    self.runtime.observe(min_camera_timestamp_s=barrier)
                ))
                raw, snapshot = self._merge_physics(
                    followup, action_timestamp_s=barrier, gripper_result=receipt
                )
                physical = raw["metadata"]["physical_verification"]
            receipt["physical_verification"] = physical
            if (
                attachment is not None
                and self._attached_object is not None
                and physical["reason_code"]
                in {
                    ReasonCode.WRONG_OBJECT.value,
                    ReasonCode.TARGET_NOT_LIFTED.value,
                    ReasonCode.RELATIVE_POSE_DRIFT.value,
                    ReasonCode.OBJECT_DROPPED.value,
                }
            ):
                # A joint topic ACK is not a held-object proof.  Any failed
                # lift/identity result immediately returns the fallback to a
                # known detached state.
                self._ensure_detached(attachment)
            if snapshot is not None and planning is not None and physical["reason_code"] == ReasonCode.TARGET_HELD.value and not planning.attached:
                target = snapshot.object(self._m3_config.target_id)
                if target is not None:
                    planning.attach(relative_pose(snapshot.eef_pose, target.pose))
                    if self._attachment_lifecycle is AttachmentLifecycle.ATTACH_ACKED_UNPROVEN:
                        self._attachment_lifecycle = AttachmentLifecycle.HELD_PROVEN
            if action_type == "gripper_open" and planning is not None and snapshot is not None and planning.attached:
                target = snapshot.object(self._m3_config.target_id)
                if target is not None:
                    planning.release(target.pose)
            if (
                action_type == "gripper_close"
                and attachment is not None
                and snapshot is not None
                and physical["reason_code"] == ReasonCode.LIFT_REQUIRED.value
            ):
                # The previous EEF-centre band admitted off-pad objects.  A
                # closed gripper now needs stable dual-pad geometry, a low
                # object speed, and exactly one candidate for 100 ms.
                gate = confirm_pad_contact(
                    self._pad_feedback_samples(barrier),
                    action_timestamp_s=barrier,
                    config=self._m3_config,
                )
                if gate.accepted and gate.candidate_id is not None:
                    selected = gate.candidate_id
                    self._attachment_lifecycle = AttachmentLifecycle.CONTACT_CONFIRMED
                    backoff = self._release_gripper_squeeze()
                    backoff_barrier = backoff.get("action_completed_ros_time_s") if backoff else None
                    if backoff and backoff.get("ok") and isinstance(backoff_barrier, int | float):
                        clear_samples = self._pad_feedback_samples(float(backoff_barrier))
                        if clear_samples and pads_are_clear(
                            clear_samples[-1], selected, config=self._m3_config
                        ):
                            label = (
                                "target"
                                if selected == self._m3_config.target_id
                                else "distractor"
                            )
                            attachment.attach(label)
                            capture = getattr(attachment, "capture_physical_baseline", None)
                            if callable(capture) and capture(label):
                                self._attached_object = selected
                                self._attachment_lifecycle = AttachmentLifecycle.ATTACH_ACKED_UNPROVEN
                            else:
                                # A request ACK without a readable child-link
                                # reference cannot be a detachable fallback.
                                self._ensure_detached(attachment)
        self._latest = raw
        # The Direct/Gym boundary owns the public unified observation.  Keep
        # the structured receipt anchored to that exact post-action object so
        # Direct acceptance and the MCP codec share the same freshness proof.
        receipt["observation"] = raw
        # Receipt is deliberately namespaced inside Gym info.  The generic MCP
        # control codec restores the established top-level wire fields.
        info = {"_openeta_receipt": receipt} if STRUCTURED_RECEIPT in self.profile.capabilities else {}
        return raw, 0.0, False, False, info

    def _ensure_detached(self, attachment: Any) -> None:
        """Detach both fallback plugins without relying on local assumptions."""

        for label in ("target", "distractor"):
            attachment.ensure_detached(label)
        self._attached_object = None
        self._attachment_lifecycle = AttachmentLifecycle.DETACHED

    def _release_gripper_squeeze(self) -> Mapping[str, Any] | None:
        """Back the pads a few millimetres off a freshly grasped object.

        Detachable-mode prerequisite: gz-sim's DetachableJoint cannot attach
        while the child model is in contact with the parent model (documented
        DART limitation), so the squeezed pads must break contact before the
        attach is published.  As a side benefit the position-controlled
        fingers no longer double-constrain the jointed object during carries.
        Returns True when the backoff goal completed.
        """

        controller = getattr(self.runtime, "controller", None)
        gripper_action = getattr(controller, "gripper_action", None)
        if gripper_action is None:
            return None
        # This is a deliberate 0.15 rad backoff, not a user-visible open.
        # The following fresh dual-pad sample must prove both pads cleared.
        with suppress(Exception):
            return dict(gripper_action(0.15, 15.0))
        return None

    def render(self):
        if self._latest is None:
            return None
        camera = next(iter(self._latest.get("cameras", {}).values()), {})
        return camera.get("rgb")

    def close(self) -> None:
        attachment = getattr(self.runtime, "attachment", None)
        if attachment is not None:
            with suppress(Exception):
                self._ensure_detached(attachment)
        self._attached_object = None
        self.runtime.close()
        self._latest = None
