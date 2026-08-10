"""The sole Gym-shaped Gazebo DirectEnv implementation."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
from gymnasium import Env, spaces

from adapter.protocol import EnvObservation

from .deployment import GazeboDeploymentConfig, worker_deployment_config
from .m2 import JOINT_NAMES
from .m3 import M3Config, M3Verifier, ReasonCode, relative_pose, unknown_record
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
        contacts = snapshot.contacts if snapshot is not None else None
        common = set(contacts.left_object_ids).intersection(contacts.right_object_ids) if contacts else set()
        raw.setdefault("robot", {}).setdefault("gripper_state", {}).update({
            "contact_left": bool(contacts and contacts.left_object_ids),
            "contact_right": bool(contacts and contacts.right_object_ids),
            "contact_object_id": next(iter(common)) if len(common) == 1 else None,
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
        if (
            action_type == "gripper_open" and planning is not None
            and planning.attached and self._m3_config is not None
        ):
            target = self._last_snapshot.object(self._m3_config.target_id) if self._last_snapshot is not None else None
            if target is not None:
                planning.release(target.pose)
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
        self._latest = raw
        # Receipt is deliberately namespaced inside Gym info.  The generic MCP
        # control codec restores the established top-level wire fields.
        info = {"_openeta_receipt": receipt} if STRUCTURED_RECEIPT in self.profile.capabilities else {}
        return raw, 0.0, False, False, info

    def render(self):
        if self._latest is None:
            return None
        camera = next(iter(self._latest.get("cameras", {}).values()), {})
        return camera.get("rgb")

    def close(self) -> None:
        self.runtime.close()
        self._latest = None
