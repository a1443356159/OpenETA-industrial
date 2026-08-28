"""Shared validation for the ordered native placement-release protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def ordered_native_release_proof(value: object) -> dict[str, Any] | None:
    """Parse release evidence without relying on fixed event indexes.

    The authoritative attached-collision filter adds an acknowledgement
    between Gazebo's native detach and MoveIt's scene detach. Historical
    receipts predate that event, so both exact event families remain readable;
    arbitrary omissions, reordering, duplicate sequence numbers, or malformed
    filter evidence fail closed.
    """

    if not isinstance(value, list) or not value:
        return None
    sequence = [dict(item) for item in value if isinstance(item, Mapping)]
    if len(sequence) != len(value):
        return None
    numbers = [item.get("sequence") for item in sequence]
    if any(type(number) is not int for number in numbers) or numbers != list(
        range(1, len(sequence) + 1)
    ):
        return None
    events = [str(item.get("event") or "") for item in sequence]
    legacy_order = [
        "native_detach_ack",
        "planning_scene_detach_ack",
        "gripper_open_completed",
        "released_target_pose_sync_ack",
    ]
    authoritative_order = [
        "native_detach_ack",
        "attached_collision_filter_ack",
        "planning_scene_detach_ack",
        "gripper_open_completed",
        "released_target_pose_sync_ack",
    ]
    if not any(
        events == order[: len(events)]
        for order in (legacy_order, authoritative_order)
    ):
        return None
    by_event = {str(item["event"]): dict(item) for item in sequence}
    native = by_event.get("native_detach_ack")
    if not isinstance(native, dict) or native.get("state") != "detached":
        return None
    filter_ack = by_event.get("attached_collision_filter_ack")
    if filter_ack is not None:
        robot_mask = filter_ack.get("robot_mask")
        target_mask = filter_ack.get("target_mask")
        if not (
            filter_ack.get("schema_version")
            == "openeta.attached_collision_filter.v1"
            and filter_ack.get("joint_state") == "detached"
            and filter_ack.get("state") == "full"
            and filter_ack.get("target_robot_collision_enabled") is True
            and filter_ack.get("target_environment_collision_enabled") is True
            and type(robot_mask) is int
            and type(target_mask) is int
            and robot_mask > 0
            and target_mask > 0
            and (robot_mask & target_mask) != 0
        ):
            return None
    planning_ack = by_event.get("planning_scene_detach_ack")
    if planning_ack is not None and (
        type(planning_ack.get("revision")) is not int
        or planning_ack["revision"] < 0
    ):
        return None
    gripper_ack = by_event.get("gripper_open_completed")
    if gripper_ack is not None and gripper_ack.get("ok") is not True:
        return None
    target_sync_ack = by_event.get("released_target_pose_sync_ack")
    if target_sync_ack is not None and (
        type(target_sync_ack.get("revision")) is not int
        or target_sync_ack["revision"] < 0
    ):
        return None
    return {
        "schema_version": "openeta.ordered_native_release.v2",
        "protocol": (
            "authoritative_collision_filter"
            if filter_ack is not None
            else "legacy"
        ),
        **by_event,
    }
