"""Fail-closed MoveIt planning-scene state for the Gazebo manipulation adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
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
_TARGET_POSE_POSITION_ABS_TOL_M = 1e-6
_TARGET_POSE_QUATERNION_ABS_TOL = 1e-8


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


@dataclass(frozen=True, slots=True)
class CollisionPrimitive:
    """One primitive expressed in its owning rigid body's local frame."""

    shape: str
    pose_xyz: tuple[float, float, float]
    pose_quat_xyzw: tuple[float, float, float, float]
    size_xyz: tuple[float, float, float] | None = None
    radius: float | None = None
    length: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "shape": self.shape,
            "pose_xyz": list(self.pose_xyz),
            "pose_quat_xyzw": list(self.pose_quat_xyzw),
        }
        if self.shape == "box" and self.size_xyz is not None:
            result["size_xyz"] = list(self.size_xyz)
        elif self.shape == "cylinder" and self.radius is not None and self.length is not None:
            result.update({"radius": self.radius, "length": self.length})
        else:
            raise PlanningSceneError("collision primitive geometry is invalid")
        return result


@dataclass(frozen=True, slots=True)
class CollisionBody:
    """A rigid body whose exact collision model contains several primitives."""

    object_id: str
    bounding_box_xyz: tuple[float, float, float]
    pose_xyz: tuple[float, float, float]
    pose_quat_xyzw: tuple[float, float, float, float]
    primitives: tuple[CollisionPrimitive, ...]

    def to_dict(self) -> dict[str, Any]:
        if not self.primitives:
            raise PlanningSceneError("compound collision body has no primitives")
        return {
            "id": self.object_id,
            "frame": "world",
            "shape": "compound",
            # Preserve the conservative outer dimensions for placement and
            # workspace reasoning; MoveIt consumes ``primitives`` below.
            "size_xyz": list(self.bounding_box_xyz),
            "pose_xyz": list(self.pose_xyz),
            "pose_quat_xyzw": list(self.pose_quat_xyzw),
            "primitives": [primitive.to_dict() for primitive in self.primitives],
        }


CollisionGeometry = CollisionBox | CollisionBody


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
        # The detached target starts in exact support contact.  MoveIt needs
        # that one pair in its ACM to admit the measured start state, while a
        # later trajectory-level audit proves that an attached path actually
        # leaves the support instead of inheriting the exception forever.
        self.support_contact_object_id = ""
        self.support_contact_reference_target_spec: dict[str, Any] = {}
        # A stable, detached placement can still touch an open fingertip at
        # the exact release endpoint.  Multi-sort departure may temporarily
        # allow only that completed object against gripper touch links; the
        # allowance is cleared atomically when the next object is attached.
        self.transient_departure_contact_object_id = ""
        self.authoritative_scene_sha256 = ""
        self.world_geometry_sha256 = ""
        self.attached_geometry_sha256 = ""
        self.geometry_verified_ids: tuple[str, ...] = ()
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
        self.support_contact_object_id = ""
        self.support_contact_reference_target_spec = {}
        self.transient_departure_contact_object_id = ""
        self.authoritative_scene_sha256 = ""
        self.world_geometry_sha256 = ""
        self.attached_geometry_sha256 = ""
        self.geometry_verified_ids = ()
        return revision

    def reset(
        self,
        *,
        table: CollisionGeometry,
        distractor: CollisionGeometry | None = None,
        distractors: Sequence[CollisionGeometry] = (),
        target: CollisionGeometry,
        obstacles: Sequence[CollisionGeometry] = (),
        robot_support_link: str | None = None,
        authoritative_scene_sha256: str = "",
    ) -> int:
        scene_distractors = [
            *([distractor] if distractor is not None else []),
            *distractors,
        ]
        obstacle_ids = [obstacle.object_id for obstacle in obstacles]
        distractor_ids = [item.object_id for item in scene_distractors]
        all_non_target_ids = [table.object_id, *distractor_ids, *obstacle_ids]
        if (
            len(all_non_target_ids) != len(set(all_non_target_ids))
            or target.object_id in set(all_non_target_ids)
        ):
            return self._fail("planning-scene obstacle identity is not unique")
        world_objects = [table, *scene_distractors, target, *obstacles]
        expected_world = {item.object_id for item in world_objects}
        support_link = str(robot_support_link or "").strip()
        allowed_collisions: dict[str, list[str]] = {
            # The target begins exactly supported by the table, so FCL reports
            # their coincident surfaces as contact. This does not permit the
            # open gripper to sweep through the target.
            target.object_id: [table.object_id],
        }
        if support_link:
            # A workcell-mounted robot intentionally shares a fixed contact
            # interface with its support surface. Scope the exception to the
            # configured root link; every moving-link/table pair remains live.
            allowed_collisions[table.object_id] = [support_link]
        revision = self._commit(
            {
                "operation": "reset",
                "world_objects": [item.to_dict() for item in world_objects],
                "attached_objects": [],
                "allowed_collisions": allowed_collisions,
                "authoritative_scene_sha256": str(
                    authoritative_scene_sha256
                ),
            },
            expected_world=expected_world,
            expected_attached=set(),
        )
        self.world_specs = {
            item.object_id: item.to_dict() for item in world_objects
        }
        self.attached_specs = {}
        self.target_id = target.object_id
        self.support_contact_object_id = table.object_id
        self.support_contact_reference_target_spec = target.to_dict()
        self.transient_departure_contact_object_id = ""
        self.authoritative_scene_sha256 = str(authoritative_scene_sha256)
        return revision

    def attach_target(
        self,
        *,
        target: CollisionGeometry,
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
        diff: dict[str, Any] = {
            "operation": "attach",
            # MoveIt automatically removes a same-id world object when an
            # AttachedCollisionObject ADD is applied. Sending an explicit
            # REMOVE in the same diff removes it twice and makes the apply
            # service return success=false.
            "attached_objects": [attached_spec],
        }
        departure_object_id = self.transient_departure_contact_object_id
        if departure_object_id:
            diff["allowed_collisions"] = {departure_object_id: []}
        revision = self._commit(
            diff,
            expected_world=self.world_ids - {target.object_id},
            expected_attached={target.object_id},
        )
        # Preserve the native attach-time world pose as the physical support
        # baseline.  The attached start state reconstructed through FK can
        # differ by a few floating-point / timestamp interpolation bits.  The
        # trajectory audit may admit that measured contact at sample zero, but
        # it still requires positive separation as soon as motion begins.
        self.support_contact_reference_target_spec = target.to_dict()
        self.world_specs.pop(target.object_id, None)
        self.attached_specs = {target.object_id: attached_spec}
        self.transient_departure_contact_object_id = ""
        return revision

    def update_world_target(self, *, target: CollisionGeometry) -> int:
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
        existing = self.world_specs.get(target.object_id)
        target_spec = target.to_dict()
        if isinstance(existing, Mapping) and _same_collision_geometry_pose(
            existing, target_spec
        ):
            # Native pose readback is allowed to contain sub-micrometre numeric
            # noise.  An idempotent synchronization is not a PlanningScene
            # mutation and must not invalidate a frozen qualification frontier.
            self.ready = True
            self.last_error = ""
            return self.revision
        revision = self._commit(
            {
                "operation": "update_world_target",
                "world_objects": [target_spec],
            },
            expected_world=set(self.world_ids),
            expected_attached=set(self.attached_ids),
        )
        self.world_specs[target.object_id] = target_spec
        self.support_contact_reference_target_spec = target_spec
        return revision

    def activate_target(
        self,
        *,
        target: CollisionGeometry,
        support_object_id: str,
        departure_contact_object_id: str = "",
    ) -> int:
        """Switch the qualification target while retaining one physical world."""

        if self.attached_ids:
            return self._fail("sort target cannot change while an object is attached")
        if target.object_id not in self.world_ids:
            return self._fail("next sort target is missing from the world scene")
        support = str(support_object_id).strip()
        if not support or support not in self.world_ids:
            return self._fail("next sort support is missing from the world scene")
        previous_target_id = self.target_id
        departure_object_id = str(departure_contact_object_id).strip()
        if departure_object_id and departure_object_id != previous_target_id:
            return self._fail(
                "departure contact object does not match the completed sort target"
            )
        target_spec = target.to_dict()
        existing = self.world_specs.get(target.object_id)
        geometry_changed = not (
            isinstance(existing, Mapping)
            and _same_collision_geometry_pose(existing, target_spec)
        )
        allowed_collisions = {
            target.object_id: [support],
            **(
                {
                    previous_target_id: (
                        list(TARGET_TOUCH_LINKS)
                        if departure_object_id == previous_target_id
                        else []
                    )
                }
                if previous_target_id and previous_target_id != target.object_id
                else {}
            ),
        }
        revision = self._commit(
            {
                "operation": "activate_target",
                **({"world_objects": [target_spec]} if geometry_changed else {}),
                # The active object starts in measured support contact. Replace
                # the prior target's owned ACM row so a same-session sort does
                # not mistake the next object's ordinary support for a crash.
                "allowed_collisions": allowed_collisions,
            },
            expected_world=set(self.world_ids),
            expected_attached=set(),
        )
        if geometry_changed:
            self.world_specs[target.object_id] = target_spec
        self.target_id = target.object_id
        self.support_contact_object_id = support
        self.support_contact_reference_target_spec = target_spec
        self.transient_departure_contact_object_id = departure_object_id
        return revision

    def detach_target(self, *, target: CollisionGeometry) -> int:
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
            if (
                diff.get("authoritative_scene_sha256")
                and self._apply_readback is not None
                and not readback.get("world_geometry_sha256")
            ):
                raise PlanningSceneError(
                    "authoritative MoveIt geometry readback proof is missing"
                )
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
        if readback.get("world_geometry_sha256") is not None:
            self.world_geometry_sha256 = str(readback["world_geometry_sha256"])
        if readback.get("attached_geometry_sha256") is not None:
            self.attached_geometry_sha256 = str(
                readback["attached_geometry_sha256"]
            )
        if readback.get("geometry_verified_ids") is not None:
            verified_now = {
                str(value) for value in readback["geometry_verified_ids"]
            }
            if diff.get("operation") == "reset":
                verified = verified_now
            else:
                # Geometry not mentioned by a scene diff is unchanged. Keep
                # its earlier exact-readback proof while replacing the proof
                # for every object verified by this commit. Objects removed
                # from both world and attached sets cannot retain evidence.
                verified = set(self.geometry_verified_ids) | verified_now
            self.geometry_verified_ids = tuple(
                sorted(verified & (expected_world | expected_attached))
            )
        self.revision += 1
        self.ready = True
        self.last_error = ""
        return self.revision

    def _fail(self, reason: str) -> int:
        self.ready = False
        self.last_error = reason
        raise PlanningSceneError(reason)


def _same_collision_geometry_pose(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> bool:
    """Return whether two rigid-body specs differ only by native pose noise."""

    if any(source.get(key) != target.get(key) for key in ("id", "frame", "shape")):
        return False
    if source.get("primitives") != target.get("primitives"):
        return False
    try:
        source_size = [float(value) for value in source["size_xyz"]]
        target_size = [float(value) for value in target["size_xyz"]]
        source_xyz = [float(value) for value in source["pose_xyz"]]
        target_xyz = [float(value) for value in target["pose_xyz"]]
        source_quat = [float(value) for value in source["pose_quat_xyzw"]]
        target_quat = [float(value) for value in target["pose_quat_xyzw"]]
    except (KeyError, TypeError, ValueError):
        return False
    if not (
        len(source_size) == len(target_size) == 3
        and len(source_xyz) == len(target_xyz) == 3
        and len(source_quat) == len(target_quat) == 4
        and all(
            math.isfinite(value)
            for value in (
                *source_size,
                *target_size,
                *source_xyz,
                *target_xyz,
                *source_quat,
                *target_quat,
            )
        )
    ):
        return False
    if any(abs(left - right) > 1e-12 for left, right in zip(source_size, target_size)):
        return False
    if any(
        abs(left - right) > _TARGET_POSE_POSITION_ABS_TOL_M
        for left, right in zip(source_xyz, target_xyz)
    ):
        return False
    # q and -q encode the same orientation.
    direct = math.sqrt(
        sum((left - right) ** 2 for left, right in zip(source_quat, target_quat))
    )
    negated = math.sqrt(
        sum((left + right) ** 2 for left, right in zip(source_quat, target_quat))
    )
    return min(direct, negated) <= _TARGET_POSE_QUATERNION_ABS_TOL
