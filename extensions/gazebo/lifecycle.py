"""Deterministic, oracle-only Gazebo lifecycle boundary (M1).

The class is intentionally transport agnostic: a real Gazebo process can be
attached behind the same methods in a later milestone.  It never pretends that
an action was executed and therefore exposes no manipulation controls yet.
"""

from __future__ import annotations

import time

import numpy as np

from adapter.protocol import CameraFrame, EnvObservation, RobotState

from .config import GazeboConfig


class GazeboLifecycleError(RuntimeError):
    """Raised when lifecycle methods are called in an invalid state."""


class GazeboEnvironment:
    """Minimal create/reset/observe/close environment with oracle objects."""

    def __init__(self, *, config: GazeboConfig | None = None, task: str = "", seed: int = 0) -> None:
        self.config = config or GazeboConfig()
        self.task = task
        self.seed = int(seed)
        self._created = False
        self._closed = False
        self._epoch = 0
        self._objects = tuple(self.config.objects)
        self._observation: EnvObservation | None = None

    def create(self) -> EnvObservation:
        if self._created and not self._closed:
            return self.observe()
        self._closed = False
        self._created = True
        return self.reset(seed=self.seed)

    def reset(self, *, task: str | None = None, seed: int | None = None) -> EnvObservation:
        if self._closed:
            raise GazeboLifecycleError("environment is closed")
        if not self._created:
            self._created = True
        if task is not None:
            self.task = task
        if seed is not None:
            self.seed = int(seed)
        # Reset is deterministic: object configuration is immutable and the
        # seed is recorded as provenance rather than used to perturb poses.
        self._objects = tuple(self.config.objects)
        self._epoch = 0
        self._observation = self._build_observation()
        return self._observation

    def observe(self) -> EnvObservation:
        if self._closed or not self._created:
            raise GazeboLifecycleError("environment must be created before observe")
        if self._observation is None:
            self._observation = self._build_observation()
        return self._observation

    def close(self) -> None:
        # Idempotent cleanup is required by the MCP contract.
        self._observation = None
        self._closed = True
        self._created = False

    @property
    def scene_epoch(self) -> int:
        return self._epoch

    def _build_observation(self) -> EnvObservation:
        h, w = self.config.image_height, self.config.image_width
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.ones((h, w), dtype=np.float32)
        camera = CameraFrame(
            frame_id=self.config.top_camera_name,
            role="scene_primary",
            rgb=rgb.tolist(),
            depth=depth.tolist(),
            intrinsics={"fx": float(w), "fy": float(h), "cx": (w - 1) / 2.0, "cy": (h - 1) / 2.0,
                        "width": w, "height": h},
            extrinsics={"pos": [0.0, 0.0, 1.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "frame_transform": "camera_to_world", "camera_frame": "opencv",
                        "matrix_layout": "row_major", "image_origin": "top_left"},
            timestamp_s=time.time(),
        )
        objects = [
            {"id": o.name, "name": o.name, "label": o.label, "confidence": o.confidence,
             "position": list(o.position), "orientation": list(o.orientation_xyzw),
             "visibility": "clear", "source_camera": self.config.top_camera_name,
             "provenance": "gazebo_oracle"}
            for o in self._objects
        ]
        return EnvObservation(
            task=self.task, cameras=[camera], robot=RobotState(), objects=objects,
            metadata={"backend": "gazebo", "world": self.config.world, "seed": self.seed,
                      "camera_topics": {"rgb": self.config.top_rgb_topic,
                                         "depth": self.config.top_depth_topic,
                                         "camera_info": self.config.top_camera_info_topic},
                      "scene_epoch": self._epoch, "observation_provenance": "gazebo_oracle"},
        )
