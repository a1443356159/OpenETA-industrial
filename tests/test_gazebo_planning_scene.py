from __future__ import annotations

import pytest

from extensions.gazebo.planning_scene import (
    CollisionBody,
    CollisionBox,
    CollisionPrimitive,
    PlanningSceneError,
    PlanningSceneSynchronizer,
    TARGET_TOUCH_LINKS,
)
from extensions.gazebo.ros_control import _relative_pose


def test_motion_only_scene_requires_empty_readback() -> None:
    calls = []

    def apply(diff):
        calls.append(diff)
        return {"applied": True, "world_ids": [], "attached_ids": []}

    scene = PlanningSceneSynchronizer(apply)

    assert scene.initialize_empty() == 1
    assert scene.ready is True
    assert calls == [{"operation": "initialize_empty"}]


def test_motion_only_scene_fails_closed_on_nonempty_readback() -> None:
    scene = PlanningSceneSynchronizer(
        lambda _diff: {
            "applied": True,
            "world_ids": ["stale_object"],
            "attached_ids": [],
        }
    )

    with pytest.raises(PlanningSceneError, match="readback mismatch"):
        scene.initialize_empty()
    assert scene.ready is False


def _boxes():
    return (
        CollisionBox("table", (0.7, 0.6, 0.04), (0.4, 0.0, 0.38)),
        CollisionBox("distractor", (0.05, 0.05, 0.08), (0.28, 0.12, 0.44)),
        CollisionBox("target", (0.04, 0.04, 0.06), (0.28, -0.1, 0.43)),
    )


def test_planning_scene_reset_and_attach_detach_switch() -> None:
    calls = []

    def apply(diff):
        calls.append(diff)
        if diff["operation"] == "reset":
            return {"applied": True, "world_ids": ["table", "distractor", "target"], "attached_ids": []}
        if diff["operation"] == "attach":
            return {"applied": True, "world_ids": ["table", "distractor"], "attached_ids": ["target"]}
        return {"applied": True, "world_ids": ["table", "distractor", "target"], "attached_ids": []}

    scene = PlanningSceneSynchronizer(apply)
    table, distractor, target = _boxes()
    assert scene.reset(table=table, distractor=distractor, target=target) == 1
    # Before native attach, MoveIt must reject any route whose open gripper
    # sweeps through the target.  Only the real support contact is allowed.
    assert calls[0]["allowed_collisions"]["target"] == ["table"]
    assert scene.support_contact_object_id == "table"
    assert scene.support_contact_reference_target_spec == target.to_dict()
    assert scene.attach_target(target=target, relative_pose_xyz=(0.0, 0.0, -0.04)) == 2
    assert scene.world_ids == {"table", "distractor"}
    assert scene.attached_ids == {"target"}
    assert calls[1]["attached_objects"][0]["touch_links"] == list(TARGET_TOUCH_LINKS)
    assert "robotiq_85_left_inner_knuckle_link" in TARGET_TOUCH_LINKS
    assert "robotiq_85_right_inner_knuckle_link" in TARGET_TOUCH_LINKS
    assert "remove_world_ids" not in calls[1]
    assert scene.detach_target(target=target) == 3
    assert scene.world_ids == {"table", "distractor", "target"}


def test_motion_only_scene_clears_authoritative_support_identity() -> None:
    scene = PlanningSceneSynchronizer()
    table, distractor, target = _boxes()
    scene.reset(table=table, distractor=distractor, target=target)

    assert scene.support_contact_object_id == "table"
    scene.initialize_empty()
    assert scene.support_contact_object_id == ""
    assert scene.support_contact_reference_target_spec == {}


def test_planning_scene_switches_sort_target_without_rebuilding_world_geometry() -> None:
    calls = []

    def apply(diff):
        calls.append(diff)
        return {
            "applied": True,
            "world_ids": ["table", "distractor", "target"],
            "attached_ids": [],
        }

    scene = PlanningSceneSynchronizer(apply)
    table, next_target, first_target = _boxes()
    scene.reset(table=table, distractor=next_target, target=first_target)
    world_before = set(scene.world_ids)

    revision = scene.activate_target(
        target=next_target,
        support_object_id=table.object_id,
    )

    assert revision == 2
    assert scene.world_ids == world_before
    assert scene.attached_ids == set()
    assert scene.target_id == next_target.object_id
    assert scene.support_contact_object_id == table.object_id
    assert calls[1] == {
        "operation": "activate_target",
        "allowed_collisions": {
            next_target.object_id: [table.object_id],
            first_target.object_id: [],
        },
    }


def test_multi_sort_departure_touch_is_scoped_until_next_attach() -> None:
    calls = []

    def apply(diff):
        calls.append(diff)
        if diff["operation"] == "attach":
            return {
                "applied": True,
                "world_ids": ["table", "target"],
                "attached_ids": ["distractor"],
            }
        return {
            "applied": True,
            "world_ids": ["table", "distractor", "target"],
            "attached_ids": [],
        }

    scene = PlanningSceneSynchronizer(apply)
    table, next_target, completed_target = _boxes()
    scene.reset(table=table, distractor=next_target, target=completed_target)

    scene.activate_target(
        target=next_target,
        support_object_id=table.object_id,
        departure_contact_object_id=completed_target.object_id,
    )

    assert calls[1]["allowed_collisions"] == {
        next_target.object_id: [table.object_id],
        completed_target.object_id: list(TARGET_TOUCH_LINKS),
    }
    assert scene.transient_departure_contact_object_id == completed_target.object_id

    scene.attach_target(target=next_target)

    assert calls[2]["allowed_collisions"] == {completed_target.object_id: []}
    assert scene.transient_departure_contact_object_id == ""


def test_multi_sort_departure_contact_must_name_completed_target() -> None:
    scene = PlanningSceneSynchronizer()
    table, next_target, completed_target = _boxes()
    scene.reset(table=table, distractor=next_target, target=completed_target)

    with pytest.raises(PlanningSceneError, match="completed sort target"):
        scene.activate_target(
            target=next_target,
            support_object_id=table.object_id,
            departure_contact_object_id="unrelated_object",
        )


def test_planning_scene_allows_only_the_fixed_robot_support_contact() -> None:
    calls = []

    def apply(diff):
        calls.append(diff)
        return {
            "applied": True,
            "world_ids": ["table", "distractor", "target"],
            "attached_ids": [],
        }

    scene = PlanningSceneSynchronizer(apply)
    table, distractor, target = _boxes()

    scene.reset(
        table=table,
        distractor=distractor,
        target=target,
        robot_support_link="base_link",
    )

    assert calls[0]["allowed_collisions"] == {
        "target": ["table"],
        "table": ["base_link"],
    }


def test_authoritative_scene_requires_moveit_geometry_readback_proof() -> None:
    table, distractor, target = _boxes()
    scene = PlanningSceneSynchronizer(
        lambda _diff: {
            "applied": True,
            "world_ids": ["table", "distractor", "target"],
            "attached_ids": [],
        }
    )

    with pytest.raises(PlanningSceneError, match="geometry readback proof is missing"):
        scene.reset(
            table=table,
            distractor=distractor,
            target=target,
            authoritative_scene_sha256="a" * 64,
        )
    assert scene.ready is False


def test_authoritative_scene_retains_moveit_geometry_readback_hash() -> None:
    table, distractor, target = _boxes()
    scene = PlanningSceneSynchronizer(
        lambda _diff: {
            "applied": True,
            "world_ids": ["table", "distractor", "target"],
            "attached_ids": [],
            "world_geometry_sha256": "b" * 64,
            "attached_geometry_sha256": "c" * 64,
            "geometry_verified_ids": ["distractor", "table", "target"],
        }
    )

    scene.reset(
        table=table,
        distractor=distractor,
        target=target,
        authoritative_scene_sha256="a" * 64,
    )

    assert scene.authoritative_scene_sha256 == "a" * 64
    assert scene.world_geometry_sha256 == "b" * 64
    assert scene.attached_geometry_sha256 == "c" * 64
    assert scene.geometry_verified_ids == ("distractor", "table", "target")


def test_authoritative_geometry_proofs_survive_target_attach_diff() -> None:
    table, distractor, target = _boxes()

    def apply(diff):
        if diff["operation"] == "reset":
            return {
                "applied": True,
                "world_ids": ["table", "distractor", "target"],
                "attached_ids": [],
                "world_geometry_sha256": "b" * 64,
                "attached_geometry_sha256": "c" * 64,
                "geometry_verified_ids": ["distractor", "table", "target"],
            }
        assert diff["operation"] == "attach"
        return {
            "applied": True,
            "world_ids": ["table", "distractor"],
            "attached_ids": ["target"],
            "world_geometry_sha256": "d" * 64,
            "attached_geometry_sha256": "e" * 64,
            "geometry_verified_ids": ["target"],
        }

    scene = PlanningSceneSynchronizer(apply)
    scene.reset(
        table=table,
        distractor=distractor,
        target=target,
        authoritative_scene_sha256="a" * 64,
    )
    scene.attach_target(target=target)

    assert scene.geometry_verified_ids == ("distractor", "table", "target")
    assert scene.world_geometry_sha256 == "d" * 64
    assert scene.attached_geometry_sha256 == "e" * 64


def test_planning_scene_preserves_authoritative_obstacles_across_attachment() -> None:
    scene = PlanningSceneSynchronizer()
    table, distractor, target = _boxes()
    bin_body = CollisionBox(
        "green_parts_bin",
        (0.32, 0.31, 0.18),
        (0.62, -0.18, 0.09),
    )

    scene.reset(
        table=table,
        distractor=distractor,
        target=target,
        obstacles=(bin_body,),
    )
    assert scene.world_ids == {"table", "distractor", "target", "green_parts_bin"}
    assert scene.world_specs["green_parts_bin"] == bin_body.to_dict()

    scene.attach_target(target=target)
    assert scene.world_ids == {"table", "distractor", "green_parts_bin"}
    scene.detach_target(target=target)
    assert scene.world_ids == {"table", "distractor", "target", "green_parts_bin"}


def test_planning_scene_accepts_authoritative_world_without_legacy_distractor() -> None:
    scene = PlanningSceneSynchronizer()
    table, _, target = _boxes()
    frame = CollisionBox(
        "workbench_frame",
        (1.0, 0.8, 0.7),
        (0.35, 0.0, 0.02),
    )
    bin_body = CollisionBox(
        "green_parts_bin",
        (0.32, 0.31, 0.18),
        (0.62, -0.18, 0.09),
    )

    scene.reset(
        table=table,
        target=target,
        distractor=None,
        obstacles=(frame, bin_body),
    )

    assert scene.world_ids == {
        "table",
        "target",
        "workbench_frame",
        "green_parts_bin",
    }
    assert "distractor" not in scene.world_ids


def test_planning_scene_preserves_compound_target_geometry_across_lifecycle() -> None:
    calls = []

    def apply(diff):
        calls.append(diff)
        operation = diff["operation"]
        if operation == "reset":
            return {"applied": True, "world_ids": ["table", "target"], "attached_ids": []}
        if operation == "attach":
            return {"applied": True, "world_ids": ["table"], "attached_ids": ["target"]}
        return {"applied": True, "world_ids": ["table", "target"], "attached_ids": []}

    table, _, _ = _boxes()
    target = CollisionBody(
        "target",
        (0.22, 0.062, 0.030),
        (0.29, -0.105, 0.015),
        (0.0, 0.0, 0.0, 1.0),
        (
            CollisionPrimitive(
                "box",
                (-0.025, 0.0, -0.002),
                (0.0, 0.0, 0.0, 1.0),
                size_xyz=(0.165, 0.025, 0.026),
            ),
            CollisionPrimitive(
                "box",
                (0.080, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                size_xyz=(0.055, 0.062, 0.030),
            ),
        ),
    )

    scene = PlanningSceneSynchronizer(apply)
    scene.reset(table=table, target=target)
    scene.attach_target(target=target, relative_pose_xyz=(0.0, 0.0, 0.16))
    scene.detach_target(target=target)

    reset_target = calls[0]["world_objects"][1]
    attached_target = calls[1]["attached_objects"][0]
    detached_target = calls[2]["world_objects"][0]
    assert reset_target["shape"] == "compound"
    assert len(reset_target["primitives"]) == 2
    assert attached_target["primitives"] == reset_target["primitives"]
    assert detached_target["primitives"] == reset_target["primitives"]


def test_planning_scene_rejects_duplicate_acceptance_obstacle_identity() -> None:
    scene = PlanningSceneSynchronizer()
    table, distractor, target = _boxes()

    with pytest.raises(PlanningSceneError, match="identity is not unique"):
        scene.reset(
            table=table,
            distractor=distractor,
            target=target,
            obstacles=(CollisionBox("target", (0.1, 0.1, 0.1), (0.0, 0.0, 0.0)),),
        )


def test_planning_scene_preserves_world_and_attached_rotations() -> None:
    calls = []

    def apply(diff):
        calls.append(diff)
        if diff["operation"] == "reset":
            return {"applied": True, "world_ids": ["table", "distractor", "target"], "attached_ids": []}
        if diff["operation"] == "attach":
            return {"applied": True, "world_ids": ["table", "distractor"], "attached_ids": ["target"]}
        return {"applied": True, "world_ids": ["table", "distractor", "target"], "attached_ids": []}

    scene = PlanningSceneSynchronizer(apply)
    table, distractor, _ = _boxes()
    target = CollisionBox(
        "target",
        (0.04, 0.04, 0.06),
        (0.28, -0.1, 0.43),
        (0.0, 0.0, 2**-0.5, 2**-0.5),
    )
    scene.reset(table=table, distractor=distractor, target=target)
    scene.attach_target(
        target=target,
        relative_pose_xyz=(0.02, 0.01, -0.04),
        relative_pose_quat_xyzw=(2**-0.5, 0.0, 0.0, 2**-0.5),
    )
    scene.detach_target(target=target)

    assert calls[0]["world_objects"][2]["pose_quat_xyzw"] == list(target.pose_quat_xyzw)
    assert calls[1]["attached_objects"][0]["pose_quat_xyzw"] == [
        2**-0.5, 0.0, 0.0, 2**-0.5
    ]
    assert calls[2]["world_objects"][0]["pose_quat_xyzw"] == list(target.pose_quat_xyzw)


def test_planning_scene_updates_detached_target_pose_with_new_revision() -> None:
    calls = []

    def apply(diff):
        calls.append(diff)
        return {
            "applied": True,
            "world_ids": ["table", "distractor", "target"],
            "attached_ids": [],
        }

    scene = PlanningSceneSynchronizer(apply)
    table, distractor, target = _boxes()
    scene.reset(table=table, distractor=distractor, target=target)
    moved_target = CollisionBox(
        "target",
        target.size_xyz,
        (0.31, -0.08, 0.43),
        (0.0, 0.0, 2**-0.5, 2**-0.5),
    )

    assert scene.update_world_target(target=moved_target) == 2
    assert scene.world_ids == {"table", "distractor", "target"}
    assert scene.attached_ids == set()
    assert scene.world_specs["target"] == moved_target.to_dict()
    assert calls[1] == {
        "operation": "update_world_target",
        "world_objects": [moved_target.to_dict()],
    }


def test_planning_scene_target_pose_sync_is_idempotent_with_native_noise() -> None:
    calls = []

    def apply(diff):
        calls.append(diff)
        return {
            "applied": True,
            "world_ids": ["table", "distractor", "target"],
            "attached_ids": [],
        }

    scene = PlanningSceneSynchronizer(apply)
    table, distractor, target = _boxes()
    assert scene.reset(table=table, distractor=distractor, target=target) == 1
    native_readback = CollisionBox(
        target.object_id,
        target.size_xyz,
        (target.pose_xyz[0] + 5e-7, *target.pose_xyz[1:]),
        tuple(-value for value in target.pose_quat_xyzw),
    )

    assert scene.update_world_target(target=native_readback) == 1
    assert scene.revision == 1
    assert len(calls) == 1


def test_planning_scene_rejects_world_pose_update_while_target_is_attached() -> None:
    scene = PlanningSceneSynchronizer()
    table, distractor, target = _boxes()
    scene.reset(table=table, distractor=distractor, target=target)
    scene.attach_target(target=target)

    with pytest.raises(PlanningSceneError, match="attached target"):
        scene.update_world_target(target=target)
    assert scene.ready is False


def test_relative_pose_uses_parent_rotation_for_translation_and_orientation() -> None:
    root_half = 2**-0.5
    relative_xyz, relative_quaternion = _relative_pose(
        child_xyz=(1.0, 1.0, 0.0),
        child_quat_xyzw=(0.0, 0.0, 1.0, 0.0),
        parent_xyz=(1.0, 0.0, 0.0),
        parent_quat_xyzw=(0.0, 0.0, root_half, root_half),
    )

    assert relative_xyz == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert relative_quaternion == pytest.approx(
        (0.0, 0.0, root_half, root_half), abs=1e-9
    )


def test_planning_scene_readback_mismatch_fails_closed() -> None:
    scene = PlanningSceneSynchronizer(
        lambda diff: {"applied": True, "world_ids": ["table"], "attached_ids": []}
    )
    table, distractor, target = _boxes()
    with pytest.raises(PlanningSceneError, match="readback mismatch"):
        scene.reset(table=table, distractor=distractor, target=target)
    assert scene.ready is False
