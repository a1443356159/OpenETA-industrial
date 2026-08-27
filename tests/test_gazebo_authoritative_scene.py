from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from extensions.gazebo.native_grasp import (
    NativePickPlaceConfig,
    load_acceptance_scene_contract,
)
from extensions.gazebo.ros2_ws.src.openeta_rm75_robotiq2f85_sim.launch.acceptance_scene_world import (
    AUTHORITATIVE_SCENE_SCHEMA_VERSION,
    compile_authoritative_scene,
    materialize_authoritative_scene,
)


def _package() -> Path:
    config = NativePickPlaceConfig()
    return config.ros_workspace / "src" / config.ros_package_name


def _compile(scene_id: str = "normal"):
    package = _package()
    return compile_authoritative_scene(
        base_world=package / "worlds/rm75_robotiq2f85_pickplace.sdf",
        catalog_path=package / "config/acceptance_scenes.json",
        scene_id=scene_id,
    )


@pytest.mark.parametrize(
    "scene_id",
    [
        "normal",
        "reject-first",
        "narrow-pick",
        "barrier-transfer",
        "fastener-bin-sort",
        "tool-bin-sort",
    ],
)
def test_authoritative_compiler_emits_the_same_object_set_for_gazebo_and_moveit(
    scene_id: str,
) -> None:
    compiled = _compile(scene_id)
    world = ET.fromstring(compiled.sdf_bytes).find("world")

    assert world is not None
    gazebo_models = {
        model.get("name"): len(model.findall("link/collision"))
        for model in world.findall("model")
        if model.findall("link/collision")
    }
    moveit_models = {
        item.object_id: len(item.primitives) for item in compiled.objects
    }
    assert gazebo_models == moveit_models
    assert all(item.visual_count > 0 for item in compiled.objects)
    assert compiled.evidence()["schema_version"] == AUTHORITATIVE_SCENE_SCHEMA_VERSION
    assert compiled.evidence()["gazebo_collision_object_ids"] == compiled.evidence()[
        "moveit_collision_object_ids"
    ]


def test_authoritative_normal_bin_is_one_exact_compound_body_with_base_and_four_walls() -> (
    None
):
    compiled = _compile()

    for bin_id in ("green_parts_bin", "blue_parts_bin"):
        bin_object = compiled.object(bin_id)
        assert [primitive.name for primitive in bin_object.primitives] == [
            "base",
            "left_wall",
            "right_wall",
            "front_wall",
            "rear_wall",
        ]
        moveit = bin_object.moveit_spec()
        assert moveit["shape"] == "compound"
        assert len(moveit["primitives"]) == 5
        assert moveit["gazebo_collision_names"] == [
            "base",
            "left_wall",
            "right_wall",
            "front_wall",
            "rear_wall",
        ]


def test_authoritative_scene_hash_is_deterministic_and_covers_materialized_sdf() -> None:
    first = _compile()
    second = _compile()

    assert first.sdf_bytes == second.sdf_bytes
    assert first.gazebo_sdf_sha256 == second.gazebo_sdf_sha256
    assert first.collision_manifest_sha256 == second.collision_manifest_sha256
    assert first.authority_sha256 == second.authority_sha256
    rendered = materialize_authoritative_scene(first)
    try:
        assert rendered.read_bytes() == first.sdf_bytes
        assert first.authority_sha256[:12] in rendered.name
    finally:
        rendered.unlink(missing_ok=True)


def test_authoritative_compiler_rejects_overlapping_physical_placement_supports(
    tmp_path: Path,
) -> None:
    package = _package()
    tree = ET.parse(package / "worlds/rm75_robotiq2f85_pickplace.sdf")
    blue_pose = tree.getroot().find(
        "world/model[@name='blue_parts_bin']/pose"
    )
    assert blue_pose is not None
    pose_values = (blue_pose.text or "").split()
    pose_values[:2] = ["0.43", "-0.300"]
    blue_pose.text = " ".join(pose_values)
    world_path = tmp_path / "overlapping.sdf"
    tree.write(world_path, encoding="utf-8", xml_declaration=True)

    payload = json.loads(
        (package / "config/acceptance_scenes.json").read_text(encoding="utf-8")
    )
    blue_region = next(
        region
        for region in payload["scenes"]["normal"]["placement_regions"]
        if region["id"] == "blue_parts_bin"
    )
    blue_region["center_xy"] = [0.43, -0.300]
    catalog_path = tmp_path / "acceptance_scenes.json"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="authoritative placement support collisions overlap",
    ):
        compile_authoritative_scene(
            base_world=world_path,
            catalog_path=catalog_path,
            scene_id="normal",
        )


def test_acceptance_catalog_cannot_reintroduce_a_second_moveit_geometry_source(
    tmp_path: Path,
) -> None:
    source = _package() / "config/acceptance_scenes.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["scenes"]["normal"]["planning_scene_obstacles"] = [
        {
            "id": "untrusted_box",
            "size_xyz": [1.0, 1.0, 1.0],
            "pose_xyz": [0.0, 0.0, 0.0],
            "pose_rpy": [0.0, 0.0, 0.0],
            "rgba": [1.0, 0.0, 0.0, 1.0],
        }
    ]
    catalog = tmp_path / "acceptance_scenes.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden"):
        load_acceptance_scene_contract("normal", path=catalog)


def test_native_config_publishes_all_visible_collision_models_from_authority() -> None:
    config = NativePickPlaceConfig()
    authority = config.authoritative_scene
    expected = {
        item.object_id
        for item in authority.objects
        if item.object_id not in {config.table_id, config.target_id}
    }

    assert {item["id"] for item in config.static_obstacle_specs} == expected
    assert "planning_scene_obstacles" not in config.acceptance_scene_contract
    assert config.authoritative_scene_sha256 == authority.authority_sha256
    assert set(config.authoritative_dynamic_obstacle_ids) == {
        item.object_id
        for item in authority.objects
        if not item.gazebo_static and item.object_id != config.target_id
    }
