from __future__ import annotations

import pytest

from agent.runtime.release_evidence import ordered_native_release_proof


def _authoritative_release_sequence():
    return [
        {"sequence": 1, "event": "native_detach_ack", "state": "detached"},
        {
            "sequence": 2,
            "event": "attached_collision_filter_ack",
            "schema_version": "openeta.attached_collision_filter.v1",
            "joint_state": "detached",
            "state": "full",
            "robot_mask": 1,
            "target_mask": 65535,
            "target_robot_collision_enabled": True,
            "target_environment_collision_enabled": True,
        },
        {
            "sequence": 3,
            "event": "planning_scene_detach_ack",
            "revision": 3,
        },
        {"sequence": 4, "event": "gripper_open_completed", "ok": True},
        {
            "sequence": 5,
            "event": "released_target_pose_sync_ack",
            "revision": 4,
        },
    ]


def test_authoritative_release_protocol_restores_full_target_collisions() -> None:
    proof = ordered_native_release_proof(_authoritative_release_sequence())

    assert proof is not None
    assert proof["protocol"] == "authoritative_collision_filter"
    assert proof["planning_scene_detach_ack"]["revision"] == 3
    assert proof["released_target_pose_sync_ack"]["revision"] == 4


@pytest.mark.parametrize("mutation", ["wrong_mask_state", "wrong_order", "gap"])
def test_authoritative_release_protocol_rejects_malformed_evidence(
    mutation: str,
) -> None:
    sequence = _authoritative_release_sequence()
    if mutation == "wrong_mask_state":
        sequence[1]["state"] = "robot_excluded"
    elif mutation == "wrong_order":
        sequence[2]["event"], sequence[3]["event"] = (
            sequence[3]["event"],
            sequence[2]["event"],
        )
    else:
        sequence[3]["sequence"] = 5

    assert ordered_native_release_proof(sequence) is None


def test_historical_release_protocol_remains_readable() -> None:
    sequence = _authoritative_release_sequence()
    sequence.pop(1)
    for index, item in enumerate(sequence, start=1):
        item["sequence"] = index

    proof = ordered_native_release_proof(sequence)

    assert proof is not None
    assert proof["protocol"] == "legacy"
