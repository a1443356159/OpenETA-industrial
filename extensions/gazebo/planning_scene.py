"""Fail-closed MoveIt planning-scene state for the Gazebo manipulation adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


LEFT_FINGERTIP = "robotiq_85_left_finger_tip_link"
RIGHT_FINGERTIP = "robotiq_85_right_finger_tip_link"
TARGET_TOUCH_LINKS = (LEFT_FINGERTIP, RIGHT_FINGERTIP)


class PlanningSceneError(RuntimeError):
    """The applied MoveIt scene could not be proven by readback."""


@dataclass(frozen=True, slots=True)
class CollisionBox:
    object_id: str
    size_xyz: tuple[float, float, float]
    pose_xyz: tuple[float, float, float]
    pose_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.object_id,
            "frame": "world",
            "shape": "box",
            "size_xyz": list(self.size_xyz),
            "pose_xyz": list(self.pose_xyz),
            "pose_quat_xyzw": list(self.pose_quat_xyzw),
        }


ApplyReadback = Callable[[dict[str, Any]], Mapping[str, Any]]


class PlanningSceneSynchronizer:
    """Own scene revisions and require exact object/attachment readback."""

    def __init__(self, apply_readback: ApplyReadback | None = None) -> None:
        self._apply_readback = apply_readback
        self.revision = 0
        self.ready = apply_readback is None
        self.world_ids: set[str] = set()
        self.attached_ids: set[str] = set()
        self.last_error = ""

    def reset(
        self,
        *,
        table: CollisionBox,
        distractor: CollisionBox,
        target: CollisionBox,
    ) -> int:
        return self._commit(
            {
                "operation": "reset",
                "world_objects": [table.to_dict(), distractor.to_dict(), target.to_dict()],
                "attached_objects": [],
                "allowed_collisions": {
                    target.object_id: list(TARGET_TOUCH_LINKS),
                },
            },
            expected_world={table.object_id, distractor.object_id, target.object_id},
            expected_attached=set(),
        )

    def attach_target(
        self,
        *,
        target: CollisionBox,
        link_name: str = "gripper_mount_link",
        relative_pose_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        relative_pose_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> int:
        if target.object_id not in self.world_ids:
            return self._fail("target is missing from the world scene before attach")
        return self._commit(
            {
                "operation": "attach",
                "remove_world_ids": [target.object_id],
                "attached_objects": [
                    {
                        **target.to_dict(),
                        "frame": link_name,
                        "pose_xyz": list(relative_pose_xyz),
                        "pose_quat_xyzw": list(relative_pose_quat_xyzw),
                        "link_name": link_name,
                        "touch_links": list(TARGET_TOUCH_LINKS),
                    }
                ],
            },
            expected_world=self.world_ids - {target.object_id},
            expected_attached={target.object_id},
        )

    def detach_target(self, *, target: CollisionBox) -> int:
        if target.object_id not in self.attached_ids:
            return self._fail("target is not attached in the planning scene")
        return self._commit(
            {
                "operation": "detach",
                "remove_attached_ids": [target.object_id],
                "world_objects": [target.to_dict()],
            },
            expected_world=self.world_ids | {target.object_id},
            expected_attached=set(),
        )

    def _commit(
        self,
        diff: dict[str, Any],
        *,
        expected_world: set[str],
        expected_attached: set[str],
    ) -> int:
        self.ready = False
        try:
            readback = (
                self._apply_readback(diff)
                if self._apply_readback is not None
                else {
                    "applied": True,
                    "world_ids": sorted(expected_world),
                    "attached_ids": sorted(expected_attached),
                }
            )
            if readback.get("applied") is not True:
                raise PlanningSceneError("planning-scene apply was not acknowledged")
            world = {str(value) for value in readback.get("world_ids", [])}
            attached = {str(value) for value in readback.get("attached_ids", [])}
            if world != expected_world or attached != expected_attached:
                raise PlanningSceneError(
                    f"planning-scene readback mismatch: world={sorted(world)}, "
                    f"attached={sorted(attached)}"
                )
        except Exception as exc:
            self.last_error = str(exc)
            raise PlanningSceneError(self.last_error) from exc
        self.world_ids = set(expected_world)
        self.attached_ids = set(expected_attached)
        self.revision += 1
        self.ready = True
        self.last_error = ""
        return self.revision

    def _fail(self, reason: str) -> int:
        self.ready = False
        self.last_error = reason
        raise PlanningSceneError(reason)
