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
from .m3 import M3Config, M3Verifier, ReasonCode, relative_pose, select_attachment_object, unknown_record
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
        if (
            action_type == "gripper_open" and planning is not None
            and planning.attached and self._m3_config is not None
        ):
            target = self._last_snapshot.object(self._m3_config.target_id) if self._last_snapshot is not None else None
            if target is not None:
                planning.release(target.pose)
        if (
            action_type == "gripper_open"
            and self._attached_object is not None
            and attachment is not None
            and self._m3_config is not None
        ):
            # The detachable fallback releases the joint exactly when the
            # gripper is asked to open, so drop/place physics stay honest.
            attachment.detach(
                "target" if self._attached_object == self._m3_config.target_id else "distractor"
            )
            self._attached_object = None
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
            if snapshot is not None and planning is not None and physical["reason_code"] == ReasonCode.TARGET_HELD.value and not planning.attached:
                target = snapshot.object(self._m3_config.target_id)
                if target is not None:
                    planning.attach(relative_pose(snapshot.eef_pose, target.pose))
            if action_type == "gripper_open" and planning is not None and snapshot is not None:
                target = snapshot.object(self._m3_config.target_id)
                if target is not None and not planning.attached:
                    planning.release(target.pose)
            if (
                action_type == "gripper_close"
                and attachment is not None
                and snapshot is not None
                and physical["reason_code"] == ReasonCode.LIFT_REQUIRED.value
            ):
                # Fallback attach only after the verifier's own stall/aperture
                # evidence AND with an object geometrically at the pads; the
                # verifier never learns whether a joint was created.
                selected = select_attachment_object(
                    reason_code=physical["reason_code"],
                    eef_pose=snapshot.eef_pose,
                    objects=snapshot.objects,
                    config=self._m3_config,
                )
                if selected is not None:
                    # gz-sim's DetachableJoint cannot (re)attach while the
                    # child model touches the parent model (documented DART
                    # limitation): the joint then never becomes rigid and the
                    # object is only cage-carried until the first horizontal
                    # move drops it.  Back the pads off FIRST so the attach
                    # happens out of contact; the object rests on the table
                    # during the backoff, so its pose is unchanged.  If the
                    # backoff cannot be confirmed, skip the attach and let the
                    # round fail honestly instead of faking a rigid joint.
                    if self._release_gripper_squeeze(receipt):
                        attachment.attach(
                            "target" if selected == self._m3_config.target_id else "distractor"
                        )
                        self._attached_object = selected
        self._latest = raw
        # The Direct/Gym boundary owns the public unified observation.  Keep
        # the structured receipt anchored to that exact post-action object so
        # Direct acceptance and the MCP codec share the same freshness proof.
        receipt["observation"] = raw
        # Receipt is deliberately namespaced inside Gym info.  The generic MCP
        # control codec restores the established top-level wire fields.
        info = {"_openeta_receipt": receipt} if STRUCTURED_RECEIPT in self.profile.capabilities else {}
        return raw, 0.0, False, False, info

    def _release_gripper_squeeze(self, receipt: Mapping[str, Any]) -> bool:
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
        calibration = getattr(getattr(controller, "config", None), "calibration", None)
        gripper_state = receipt.get("gripper_state") or {}
        aperture = gripper_state.get("aperture_m")
        if gripper_action is None or calibration is None or not isinstance(aperture, (int, float)):
            return False
        # Open to a fixed wide aperture (~8 cm): the 2F-85 fingertip hooks
        # curve ~1 cm inwards, so a few-millimetre backoff still leaves the
        # hooks embracing the box and the attach stays "in contact".
        open_angle = min(max(calibration.angles_rad[0], 0.15), calibration.angles_rad[-1])
        if calibration.angle_from_aperture(float(aperture)) <= open_angle:
            return True  # already clear
        with suppress(Exception):
            result = gripper_action(open_angle, 15.0)
            return bool(result.get("ok"))
        return False

    def render(self):
        if self._latest is None:
            return None
        camera = next(iter(self._latest.get("cameras", {}).values()), {})
        return camera.get("rgb")

    def close(self) -> None:
        attachment = getattr(self.runtime, "attachment", None)
        if self._attached_object is not None and attachment is not None:
            with suppress(Exception):
                attachment.detach(
                    "target" if self._attached_object == self._m3_config.target_id else "distractor"
                )
        self._attached_object = None
        self.runtime.close()
        self._latest = None
