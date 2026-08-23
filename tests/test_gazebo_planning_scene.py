from __future__ import annotations

import pytest

from extensions.gazebo.planning_scene import (
    CollisionBox,
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
    assert calls[0]["allowed_collisions"]["target"] == [
        *TARGET_TOUCH_LINKS,
        "table",
    ]
    assert scene.attach_target(target=target, relative_pose_xyz=(0.0, 0.0, -0.04)) == 2
    assert scene.world_ids == {"table", "distractor"}
    assert scene.attached_ids == {"target"}
    assert calls[1]["attached_objects"][0]["touch_links"] == list(TARGET_TOUCH_LINKS)
    assert "robotiq_85_left_inner_knuckle_link" in TARGET_TOUCH_LINKS
    assert "robotiq_85_right_inner_knuckle_link" in TARGET_TOUCH_LINKS
    assert "remove_world_ids" not in calls[1]
    assert scene.detach_target(target=target) == 3
    assert scene.world_ids == {"table", "distractor", "target"}


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
