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
        "multi_normal",
        "multi_normal_random_12345",
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
    assert compiled.evidence()["visual_policy"] == (
        "independent_high_fidelity_gazebo_assets"
    )


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
        assert bin_object.visual_count == 1
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
        primitives = {primitive.name: primitive for primitive in bin_object.primitives}
        assert primitives["front_wall"].pose_xyz[2] == pytest.approx(0.045)
        assert primitives["front_wall"].size_xyz == pytest.approx((0.32, 0.047, 0.09))
        assert primitives["rear_wall"].size_xyz == pytest.approx((0.32, 0.016, 0.18))


def test_authoritative_scene_materializes_attachment_collision_masks_and_plugin() -> None:
    compiled = _compile()
    world = ET.fromstring(compiled.sdf_bytes).find("world")

    assert world is not None
    target_masks = {
        collision.findtext("surface/contact/collide_bitmask")
        for collision in world.findall(
            "model[@name='target_object']/link/collision"
        )
    }
    assert target_masks == {"65535"}
    plugin = world.find(
        "plugin[@name='openeta::gazebo::AttachedCollisionFilter']"
    )
    assert plugin is not None
    assert plugin.get("filename") == (
        "libopeneta_attached_collision_filter_system.so"
    )
    assert plugin.findtext("robot_mask") == "1"
    assert plugin.findtext("detached_mask") == "65535"
    assert plugin.findtext("attached_mask") == "2"
    assert plugin.findtext("state_request_topic") == (
        "/openeta/native_grasp/detachable_joint/target/"
        "collision_filter_state/request"
    )
    assert plugin.findtext("state_ack_topic") == (
        "/openeta/native_grasp/detachable_joint/target/"
        "collision_filter_state/ack"
    )
    evidence = compiled.evidence()["attached_collision_filter"]
    assert evidence["attached_target_robot_collision_enabled"] is False
    assert evidence["attached_target_environment_collision_enabled"] is True


def test_authoritative_multi_normal_owns_two_filtered_dynamic_targets() -> None:
    compiled = _compile("multi_normal")
    world = ET.fromstring(compiled.sdf_bytes).find("world")

    assert world is not None
    assert [binding.target_model for binding in compiled.target_bindings] == [
        "target_object",
        "red_m24_hex_bolt",
    ]
    plugins = world.findall(
        "plugin[@name='openeta::gazebo::AttachedCollisionFilter']"
    )
    assert [plugin.findtext("target_model") for plugin in plugins] == [
        "target_object",
        "red_m24_hex_bolt",
    ]
    for target_id in ("target_object", "red_m24_hex_bolt"):
        assert {
            collision.findtext("surface/contact/collide_bitmask")
            for collision in world.findall(
                f"model[@name='{target_id}']/link/collision"
            )
        } == {"65535"}


def test_multi_normal_is_one_task_neutral_physical_world() -> None:
    compiled = _compile("multi_normal")
    contract = load_acceptance_scene_contract("multi_normal")

    assert compiled.world_scene == "multi_normal"
    assert "task" not in contract
    assert all("selected" not in region for region in contract["placement_regions"])
    assert {binding.target_model for binding in compiled.target_bindings} == {
        "target_object",
        "red_m24_hex_bolt",
    }
    with pytest.raises(ValueError, match="unsupported acceptance scene"):
        load_acceptance_scene_contract("multi_normal_prompt_variant")


def test_seeded_multi_normal_layout_is_task_neutral_and_authoritative() -> None:
    canonical = _compile("multi_normal")
    randomized = _compile("multi_normal_random_12345")
    contract = load_acceptance_scene_contract("multi_normal_random_12345")

    assert contract["seed"] == 12345
    assert "task" not in contract
    assert randomized.world_scene == "multi_normal_random_12345"
    assert randomized.authority_sha256 != canonical.authority_sha256
    assert randomized.object("target_object").pose_xyz == pytest.approx(
        (0.30, -0.30, 0.015)
    )
    assert randomized.object("red_m24_hex_bolt").pose_xyz == pytest.approx(
        (0.35, -0.10, 0.002)
    )
    assert [binding.target_model for binding in randomized.target_bindings] == [
        "target_object",
        "red_m24_hex_bolt",
    ]
    assert {item.object_id for item in randomized.objects} == {
        item.object_id for item in canonical.objects
    }


def test_seeded_layout_rejects_overlapping_dynamic_models(tmp_path: Path) -> None:
    package = _package()
    source = package / "config/acceptance_scenes.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    scene = payload["scenes"]["multi_normal_random_12345"]
    target_pose = scene["model_pose_overrides"][0]["pose_xyz"]
    scene["model_pose_overrides"][5]["pose_xyz"][:2] = target_pose[:2]
    catalog = tmp_path / "acceptance_scenes.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="randomized dynamic models overlap",
    ):
        compile_authoritative_scene(
            base_world=package / "worlds/rm75_robotiq2f85_pickplace.sdf",
            catalog_path=catalog,
            scene_id="multi_normal_random_12345",
        )


def test_seeded_layout_cannot_move_static_workcell_geometry(tmp_path: Path) -> None:
    package = _package()
    source = package / "config/acceptance_scenes.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["scenes"]["multi_normal_random_12345"]["model_pose_overrides"][0][
        "id"
    ] = "work_table"
    catalog = tmp_path / "acceptance_scenes.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot move static model"):
        compile_authoritative_scene(
            base_world=package / "worlds/rm75_robotiq2f85_pickplace.sdf",
            catalog_path=catalog,
            scene_id="multi_normal_random_12345",
        )


def test_seeded_layout_rejects_an_unsupported_floating_model(tmp_path: Path) -> None:
    package = _package()
    source = package / "config/acceptance_scenes.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["scenes"]["multi_normal_random_12345"]["model_pose_overrides"][2][
        "pose_xyz"
    ][2] = 0.05
    catalog = tmp_path / "acceptance_scenes.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="starts unsupported"):
        compile_authoritative_scene(
            base_world=package / "worlds/rm75_robotiq2f85_pickplace.sdf",
            catalog_path=catalog,
            scene_id="multi_normal_random_12345",
        )


def test_authoritative_compiler_keeps_detailed_visual_independent_from_collision(
    tmp_path: Path,
) -> None:
    package = _package()
    tree = ET.parse(package / "worlds/rm75_robotiq2f85_pickplace.sdf")
    visual_uri = tree.getroot().find(
        "world/model[@name='green_parts_bin']/link/"
        "visual[@name='bin_mesh']/geometry/mesh/uri"
    )
    assert visual_uri is not None
    visual_uri.text = "model://presentation-only/revised-bin.glb"
    world_path = tmp_path / "independent-detailed-visual.sdf"
    tree.write(world_path, encoding="utf-8", xml_declaration=True)

    compiled = compile_authoritative_scene(
        base_world=world_path,
        catalog_path=package / "config/acceptance_scenes.json",
        scene_id="normal",
    )
    assert len(compiled.object("green_parts_bin").primitives) == 5
    assert compiled.object("green_parts_bin").visual_count == 1


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
    pose_values[:2] = ["0.62", "0.180"]
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
    blue_region["center_xy"] = [0.62, 0.180]
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
