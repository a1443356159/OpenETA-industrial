"""The sole Gym-shaped Gazebo DirectEnv implementation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from gymnasium import Env, spaces

from adapter.protocol import EnvObservation

from .deployment import GazeboDeploymentConfig, worker_deployment_config
from .m2 import JOINT_NAMES
from .profiles import CONTROL, STRUCTURED_RECEIPT, GazeboProfile, gazebo_profile
from .process import GazeboProcessError
from .runtime import GazeboRuntime


class GazeboDirectEnv(Env):
    """Profile-driven DirectEnv for M1 and M2.

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
        if self.profile.unavailable_reason:
            raise GazeboProcessError(self.profile.unavailable_reason)
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
            "model_id": getattr(self.profile.model_config, "model_id", None),
        }
        self.action_space = spaces.Discrete(1)

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

    def observe(self) -> dict[str, Any]:
        raw = self._decorate_robot(self._as_unified(self.runtime.observe()))
        self._latest = raw
        return raw

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._seed = int(seed)
        observation = self.runtime.reset(seed=self._seed)
        raw = self._decorate_robot(self._as_unified(observation))
        self._latest = raw
        return raw, {}

    def step(self, action: Any):
        raw_action = action if isinstance(action, Mapping) else {}
        observation, receipt = self.runtime.execute(raw_action)
        raw = self._decorate_robot(self._as_unified(observation))
        self._latest = raw
        # The Direct/Gym boundary owns the public unified observation.  Keep
        # the structured receipt anchored to that exact post-action object so
        # Direct acceptance and the MCP codec share the same freshness proof.
        receipt["observation"] = raw
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
