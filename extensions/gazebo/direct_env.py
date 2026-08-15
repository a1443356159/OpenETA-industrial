"""The sole Gym-shaped Gazebo DirectEnv implementation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from gymnasium import Env, spaces

from adapter.protocol import EnvObservation

from .deployment import GazeboDeploymentConfig, worker_deployment_config
from .m2 import JOINT_NAMES
from .m3 import M3Config, M3Verifier, ReasonCode
from .profiles import CONTROL, PHYSICS, STRUCTURED_RECEIPT, GazeboProfile, gazebo_profile
from .process import GazeboProcessError
from .process import GazeboNativeContactWindow
from .runtime import GazeboRuntime


class GazeboDirectEnv(Env):
    """Profile-driven DirectEnv for M1, M2, and guarded M3.

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
        self._m3_verifier = M3Verifier(self._m3_config) if self._m3_config is not None else None
        self._m3_transport_locked = False
        self._m3_lift_proof_pending = False

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
            raw.setdefault("metadata", {}).update({
                "grasp_mechanism": "gazebo_sim8_detachable_joint",
                "contact_provenance": "gazebo_native_contacts",
                "attachment_target": self._m3_config.target_id,
            })
        return raw

    def observe(self) -> dict[str, Any]:
        raw = self._decorate_robot(self._as_unified(self.runtime.observe()))
        self._latest = raw
        return raw

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._seed = int(seed)
        if self._m3_verifier is not None:
            self._m3_verifier.reset()
            self._m3_transport_locked = False
            self._m3_lift_proof_pending = False
        observation = self.runtime.reset(seed=self._seed)
        raw = self._decorate_robot(self._as_unified(observation))
        if self._m3_verifier is not None:
            raw.setdefault("metadata", {})["physical_verification"] = self._m3_verifier.last_record.to_dict()
        self._latest = raw
        return raw, {}

    def step(self, action: Any):
        raw_action = action if isinstance(action, Mapping) else {}
        action_type = str(raw_action.get("action_type") or "")
        contact_window: GazeboNativeContactWindow | None = None
        if self._m3_config is not None and action_type == "gripper_close":
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
                raw.setdefault("metadata", {})["physical_verification"] = self._m3_verifier.last_record.to_dict() if self._m3_verifier else {}
                receipt["observation"] = raw
                return raw, 0.0, False, False, {"_openeta_receipt": receipt}
        if self._m3_config is not None and self._m3_transport_locked and action_type in {"move_to", "follow_eef_trajectory"}:
            attachment = getattr(self.runtime, "attachment", None)
            if attachment is None or getattr(attachment, "state", None) != "attached":
                observation = self.runtime.observe()
                receipt = {"ok": False, "error_code": ReasonCode.ATTACH_ACK_MISSING.value}
                raw = self._decorate_robot(self._as_unified(observation))
                raw.setdefault("metadata", {})["physical_verification"] = self._m3_verifier.last_record.to_dict() if self._m3_verifier else {}
                receipt["observation"] = raw
                return raw, 0.0, False, False, {"_openeta_receipt": receipt}
        try:
            observation, receipt = self.runtime.execute(raw_action)
        except Exception:
            if contact_window is not None:
                contact_window.close()
            raise
        raw = self._decorate_robot(self._as_unified(observation))
        if self._m3_config is not None and self._m3_verifier is not None:
            attachment = getattr(self.runtime, "attachment", None)
            if action_type == "gripper_close":
                gate = None
                self._m3_transport_locked = True
                try:
                    if receipt.get("ok") is not True:
                        raise GazeboProcessError(str(receipt.get("error_code") or "GRIPPER_FAILED"))
                    barrier_value = receipt.get("action_completed_ros_time_s")
                    barrier = float(barrier_value) if isinstance(barrier_value, int | float) else None
                    assert contact_window is not None
                    contact_window.begin_post_close()
                    gate = contact_window.evaluate(close_completed_sim_time_s=barrier, config=self._m3_config)
                    if not gate.accepted or attachment is None:
                        record = self._m3_verifier.close_result(gate, attach_acked=False)
                        receipt.update({"ok": False, "error_code": record.reason_code.value, "native_contact_gate": gate.to_dict()})
                    else:
                        attachment.attach()
                        record = self._m3_verifier.close_result(gate, attach_acked=True)
                        attachment.capture_baseline()
                        self._m3_transport_locked = False
                        # The first transport command after an acknowledged
                        # capture is M3's configured lift-proof step.  Later
                        # place moves retain this successful evidence instead
                        # of reclassifying a deliberately lowered object.
                        self._m3_lift_proof_pending = True
                        receipt.update({
                            "native_contact_gate": gate.to_dict(),
                            "detachable_joint": {
                                "state": "attached",
                                "attach_topic": self._m3_config.attach_topic,
                                "state_topic": self._m3_config.state_topic,
                            },
                        })
                except Exception as exc:
                    attached_before_cleanup = getattr(attachment, "state", None) == "attached"
                    if attached_before_cleanup:
                        try:
                            attachment.ensure_detached(require_ack=True)
                            receipt["detachable_joint"] = {
                                "state": "detached",
                                "detach_topic": self._m3_config.detach_topic,
                                "state_topic": self._m3_config.state_topic,
                            }
                        except Exception:
                            pass
                    if gate is not None and gate.accepted and attached_before_cleanup:
                        record = self._m3_verifier.prove_lift(None, dart_supported=True)
                    else:
                        record = self._m3_verifier.close_result(
                            gate if gate is not None else self._contact_unavailable_result(), attach_acked=False
                        )
                    receipt.update({"ok": False, "error_code": record.reason_code.value, "physical_verification": record.to_dict(), "detail": str(exc)})
                    self._m3_transport_locked = True
                    self._m3_lift_proof_pending = False
                finally:
                    if contact_window is not None:
                        contact_window.close()
                raw.setdefault("metadata", {})["physical_verification"] = record.to_dict()
                receipt["physical_verification"] = record.to_dict()
            elif action_type == "gripper_open":
                try:
                    if receipt.get("ok") is not True:
                        raise GazeboProcessError(str(receipt.get("error_code") or "GRIPPER_FAILED"))
                    if attachment is None:
                        raise GazeboProcessError("M3_DETACHABLE_JOINT_UNAVAILABLE")
                    attachment.ensure_detached(require_ack=True)
                    record = self._m3_verifier.release_result(detached_acked=True)
                    receipt["detachable_joint"] = {
                        "state": "detached",
                        "detach_topic": self._m3_config.detach_topic,
                        "state_topic": self._m3_config.state_topic,
                    }
                    self._m3_transport_locked = False
                    self._m3_lift_proof_pending = False
                except Exception as exc:
                    record = self._m3_verifier.release_result(detached_acked=False)
                    receipt.update({"ok": False, "error_code": str(exc)})
                    self._m3_transport_locked = True
                    self._m3_lift_proof_pending = False
                raw.setdefault("metadata", {})["physical_verification"] = record.to_dict()
                receipt["physical_verification"] = record.to_dict()
            elif action_type in {"move_to", "follow_eef_trajectory"} and self._m3_lift_proof_pending:
                try:
                    proof = attachment.child_link_proof() if attachment is not None else None
                    record = self._m3_verifier.prove_lift(proof, dart_supported=True)
                except Exception:
                    record = self._m3_verifier.prove_lift(None, dart_supported=True)
                raw.setdefault("metadata", {})["physical_verification"] = record.to_dict()
                receipt["physical_verification"] = record.to_dict()
                proof_evidence = dict(record.evidence)
                receipt["child_link_proof"] = (
                    proof_evidence
                    if {"lift_m", "capture_relative_translation_m"} <= proof_evidence.keys()
                    else {"available": False, "reason_code": record.reason_code.value}
                )
                self._m3_lift_proof_pending = False
                if record.reason_code is not ReasonCode.TARGET_HELD:
                    self._m3_transport_locked = True
                    try:
                        if attachment is None:
                            raise GazeboProcessError("M3_DETACHABLE_JOINT_UNAVAILABLE")
                        attachment.ensure_detached(require_ack=True)
                        receipt["detachable_joint"] = {
                            "state": "detached",
                            "detach_topic": self._m3_config.detach_topic,
                            "state_topic": self._m3_config.state_topic,
                        }
                    except Exception:
                        receipt["detach_cleanup_error"] = ReasonCode.DETACH_ACK_MISSING.value
                    receipt.update({"ok": False, "error_code": record.reason_code.value})
            else:
                raw.setdefault("metadata", {})["physical_verification"] = self._m3_verifier.last_record.to_dict()
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
        from .m3 import ContactGateResult
        return ContactGateResult(False, ReasonCode.CONTACT_WINDOW_NOT_ARMED, 0, 0)

    def render(self):
        if self._latest is None:
            return None
        camera = next(iter(self._latest.get("cameras", {}).values()), {})
        return camera.get("rgb")

    def close(self) -> None:
        self.runtime.close()
        self._latest = None
