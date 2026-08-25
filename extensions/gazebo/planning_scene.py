"""Fail-closed MoveIt planning-scene state for the Gazebo manipulation adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


LEFT_FINGERTIP = "robotiq_85_left_finger_tip_link"
RIGHT_FINGERTIP = "robotiq_85_right_finger_tip_link"
# MoveIt touch links describe every gripper link that a physically verified
# held object may contact, not only the two Gazebo links used as the bilateral
# contact sensor gate.  The 2F-85 four-bar linkage brings the inner knuckles
# against a correctly centered object at closure, so excluding them makes a
# valid native grasp invalidate the planning scene immediately after attach.
TARGET_TOUCH_LINKS = (
    "robotiq_85_left_knuckle_link",
    "robotiq_85_right_knuckle_link",
    "robotiq_85_left_finger_link",
    "robotiq_85_right_finger_link",
    "robotiq_85_left_inner_knuckle_link",
    "robotiq_85_right_inner_knuckle_link",
    LEFT_FINGERTIP,
    RIGHT_FINGERTIP,
)


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
        self.world_specs: dict[str, dict[str, Any]] = {}
        self.attached_specs: dict[str, dict[str, Any]] = {}
        self.target_id = ""
        self.last_error = ""

    def initialize_empty(self) -> int:
        """Prove that a motion-only environment has no scene objects."""

        revision = self._commit(
            {"operation": "initialize_empty"},
            expected_world=set(),
            expected_attached=set(),
        )
        self.world_specs = {}
        self.attached_specs = {}
        self.target_id = ""
        return revision

    def reset(
        self,
        *,
        table: CollisionBox,
        distractor: CollisionBox,
        target: CollisionBox,
    ) -> int:
        revision = self._commit(
            {
                "operation": "reset",
                "world_objects": [table.to_dict(), distractor.to_dict(), target.to_dict()],
                "attached_objects": [],
                "allowed_collisions": {
                    # The target begins exactly supported by the table, so
                    # FCL reports their coincident surfaces as contact. This
                    # is the one support-surface exception required for native
                    # attachment and exact release; all other world
                    # objects and non-touch robot links remain collidable.
                    target.object_id: [*TARGET_TOUCH_LINKS, table.object_id],
                },
            },
            expected_world={table.object_id, distractor.object_id, target.object_id},
            expected_attached=set(),
        )
        self.world_specs = {
            item.object_id: item.to_dict() for item in (table, distractor, target)
        }
        self.attached_specs = {}
        self.target_id = target.object_id
        return revision

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
        attached_spec = {
            **target.to_dict(),
            "frame": link_name,
            "pose_xyz": list(relative_pose_xyz),
            "pose_quat_xyzw": list(relative_pose_quat_xyzw),
            "link_name": link_name,
            "touch_links": list(TARGET_TOUCH_LINKS),
        }
        revision = self._commit(
            {
                "operation": "attach",
                # MoveIt automatically removes a same-id world object when an
                # AttachedCollisionObject ADD is applied. Sending an explicit
                # REMOVE in the same diff removes it twice and makes the apply
                # service return success=false.
                "attached_objects": [
                    attached_spec
                ],
            },
            expected_world=self.world_ids - {target.object_id},
            expected_attached={target.object_id},
        )
        self.world_specs.pop(target.object_id, None)
        self.attached_specs = {target.object_id: attached_spec}
        return revision

    def update_world_target(self, *, target: CollisionBox) -> int:
        """Replace the detached target pose without changing scene identity.

        A rejected physical close can push the object even though no attach is
        acknowledged.  Before another ordinary grasp cycle, synchronize that
        measured world pose so collision checking cannot use the pre-contact
        location.
        """

        if not self.target_id or target.object_id != self.target_id:
            return self._fail("world target identity does not match the planning scene")
        if target.object_id in self.attached_ids:
            return self._fail("attached target cannot be updated as a world object")
        if target.object_id not in self.world_ids:
            return self._fail("target is missing from the world scene before pose update")
        revision = self._commit(
            {
                "operation": "update_world_target",
                "world_objects": [target.to_dict()],
            },
            expected_world=set(self.world_ids),
            expected_attached=set(self.attached_ids),
        )
        self.world_specs[target.object_id] = target.to_dict()
        return revision

    def detach_target(self, *, target: CollisionBox) -> int:
        if target.object_id not in self.attached_ids:
            return self._fail("target is not attached in the planning scene")
        revision = self._commit(
            {
                "operation": "detach",
                "remove_attached_ids": [target.object_id],
                "world_objects": [target.to_dict()],
            },
            expected_world=self.world_ids | {target.object_id},
            expected_attached=set(),
        )
        self.attached_specs.pop(target.object_id, None)
        self.world_specs[target.object_id] = target.to_dict()
        return revision

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
