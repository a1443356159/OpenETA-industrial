from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from extensions.gazebo.detachable_sdf import prepare_detachable_sdf
from extensions.gazebo.robot_control import GAZEBO_CONTROL_ENV_ID, MODEL_ID, GazeboControlConfig
from extensions.gazebo.native_grasp import (
    PICKPLACE_DISPLAY_NAME,
    PICKPLACE_ENV_ID,
    PICKPLACE_MODEL_ID,
    NativePickPlaceConfig,
    load_acceptance_scene_contract,
    validated_pickplace_motion_guidance,
)
from extensions.gazebo.ros2_ws.src.openeta_rm75_robotiq2f85_sim.launch.acceptance_scene_world import (
    render_acceptance_world,
)
from extensions.gazebo.profiles import CONTROL, PHYSICS, STRUCTURED_RECEIPT, gazebo_profile
from sim.env_registry import get_env_spec


def test_m3_registration_exposes_the_approved_detachable_joint_profile() -> None:
    m3, m2 = get_env_spec(PICKPLACE_ENV_ID), get_env_spec(GAZEBO_CONTROL_ENV_ID)
    assert m3 is not None and m2 is not None and m3.display_name == PICKPLACE_DISPLAY_NAME
    assert NativePickPlaceConfig().model_id == PICKPLACE_MODEL_ID
    assert NativePickPlaceConfig().allow_stalling is True
    assert GazeboControlConfig().model_id == MODEL_ID
    profile = gazebo_profile("rm75_robotiq2f85_pickplace")
    assert profile.unavailable_reason is None
    assert {CONTROL, PHYSICS, STRUCTURED_RECEIPT} <= profile.capabilities
    assert profile.launch_file == "gazebo_pickplace.launch.py"
    assert profile.cameras[0].extrinsics["pos"] == [0.35, 0.0, 1.3]


def test_m3_top_camera_profile_matches_the_world_model_pose() -> None:
    config = NativePickPlaceConfig()
    world_path = (
        config.ros_workspace
        / "src"
        / config.ros_package_name
        / "worlds/rm75_robotiq2f85_pickplace.sdf"
    )
    root = ET.parse(world_path).getroot()
    camera = root.find(".//model[@name='openeta_top_camera']")
    assert camera is not None
    world_xyz = [float(value) for value in camera.findtext("pose", "").split()[:3]]

    profile_xyz = gazebo_profile("rm75_robotiq2f85_pickplace").cameras[0].extrinsics["pos"]
    assert profile_xyz == world_xyz


def test_m3_world_uses_soft_ambient_light_and_black_industrial_floor() -> None:
    config = NativePickPlaceConfig()
    world_path = (
        config.ros_workspace
        / "src"
        / config.ros_package_name
        / "worlds/rm75_robotiq2f85_pickplace.sdf"
    )
    world = ET.parse(world_path).getroot().find("world")

    assert world is not None
    lights = world.findall("light")
    assert [light.get("name") for light in lights] == ["soft_key_light"]
    assert lights[0].findtext("cast_shadows") == "false"
    assert lights[0].findtext("direction") == "-0.45 0.25 -0.86"
    assert world.find("model[@name='ground']") is None
    floor = world.find("model[@name='industrial_floor']")
    assert floor is not None
    assert floor.findtext("pose") == "0 0 -0.01 0 0 0"
    assert floor.findtext("link/collision/geometry/box/size") == "10 10 0.02"
    assert floor.findtext("link/visual/pose") == "0 0 0.001 0 0 0"
    assert floor.findtext("link/visual/geometry/box/size") == "10 10 0.02"
    assert floor.findtext("link/visual/material/ambient") == "0.015 0.018 0.025 1"
    assert floor.findtext("link/visual/material/specular") == "0.01 0.01 0.01 1"
    table = world.find("model[@name='work_table']")
    assert table is not None
    assert table.findtext("link/visual/material/ambient") == "0.58 0.60 0.64 1"
    assert world.findtext("scene/ambient") == "0.42 0.43 0.45 1"
    assert world.findtext("scene/background") == "0.16 0.19 0.24 1"
    assert world.findtext("scene/shadows") == "false"


def test_m3_assets_are_required_before_manipulation_starts() -> None:
    config = NativePickPlaceConfig()
    config.validate_assets()
    package = config.ros_workspace / "src" / config.ros_package_name
    assert (package / "worlds/rm75_robotiq2f85_pickplace.sdf").is_file()


@pytest.mark.parametrize(
    ("scene_id", "seed", "obstacle_ids"),
    [
        ("narrow-pick", 17, ["pick_guard_left", "pick_guard_right"]),
        ("barrier-transfer", 29, ["transfer_barrier"]),
    ],
)
def test_m3_complex_acceptance_scenes_share_one_versioned_geometry_contract(
    scene_id: str,
    seed: int,
    obstacle_ids: list[str],
) -> None:
    contract = load_acceptance_scene_contract(scene_id)
    config = NativePickPlaceConfig(acceptance_scene_id=scene_id)

    assert contract["scene_id"] == scene_id
    assert contract["world_scene"] == scene_id
    assert contract["seed"] == seed
    assert len(contract["contract_sha256"]) == 64
    assert [row["id"] for row in contract["static_obstacles"]] == obstacle_ids
    assert config.acceptance_scene_seed == seed
    assert [row["id"] for row in config.static_obstacle_specs] == obstacle_ids
    assert config.acceptance_scene_evidence()["contract_sha256"] == contract["contract_sha256"]


def test_m3_narrow_pick_corridor_is_constrained_without_excluding_the_full_gripper() -> None:
    contract = load_acceptance_scene_contract("narrow-pick")
    left, right = contract["static_obstacles"]
    center_separation = abs(left["pose_xyz"][1] - right["pose_xyz"][1])
    inner_gap = center_separation - (left["size_xyz"][1] + right["size_xyz"][1]) / 2.0

    # Robotiq's full outer body needs substantially more room than its 85 mm
    # finger opening.  A 164 mm corridor still rejects oblique approaches but
    # preserves the model's physically useful side-contact family.
    assert 0.16 <= inner_gap <= 0.17
    assert left["pose_xyz"][2] + left["size_xyz"][2] / 2.0 == pytest.approx(0.425)
    assert right["pose_xyz"][2] + right["size_xyz"][2] / 2.0 == pytest.approx(0.425)


def test_m3_barrier_transfer_blocks_only_the_diagonal_path_not_its_endpoints() -> None:
    config = NativePickPlaceConfig(acceptance_scene_id="barrier-transfer")
    barrier = config.acceptance_scene_contract["static_obstacles"][0]
    start = config.target_initial_xyz[:2]
    destination = config.destination_center_xy

    assert destination == (0.48, 0.10)
    assert barrier["pose_xyz"][:2] == pytest.approx(
        [(start[index] + destination[index]) / 2.0 for index in range(2)]
    )
    assert barrier["size_xyz"] == [0.025, 0.08, 0.12]


@pytest.mark.parametrize(
    ("scene_id", "obstacle_ids"),
    [
        ("narrow-pick", ["pick_guard_left", "pick_guard_right"]),
        ("barrier-transfer", ["transfer_barrier"]),
    ],
)
def test_m3_complex_scene_renderer_adds_real_static_collision_geometry(
    scene_id: str,
    obstacle_ids: list[str],
) -> None:
    config = NativePickPlaceConfig(acceptance_scene_id=scene_id)
    package = config.ros_workspace / "src" / config.ros_package_name
    rendered, selected = render_acceptance_world(
        base_world=package / "worlds/rm75_robotiq2f85_pickplace.sdf",
        catalog_path=package / "config/acceptance_scenes.json",
        environment={"OPENETA_ACCEPTANCE_SCENE": scene_id},
    )
    try:
        world = ET.parse(rendered).getroot().find("world")
        assert world is not None
        assert selected == scene_id
        for obstacle in config.acceptance_scene_contract["static_obstacles"]:
            model = world.find(f"model[@name='{obstacle['id']}']")
            assert model is not None
            assert model.findtext("static") == "true"
            assert [
                float(value)
                for value in model.findtext("link/collision/geometry/box/size", "").split()
            ] == obstacle["size_xyz"]
        if scene_id == "barrier-transfer":
            marker = world.find("model[@name='placement_zone_marker']")
            assert marker is not None
            marker_pose = [float(value) for value in marker.findtext("pose", "").split()]
            assert marker_pose[:2] == [0.48, 0.10]
        assert [
            model.get("name")
            for model in world.findall("model")
            if model.get("name") in obstacle_ids
        ] == obstacle_ids
    finally:
        rendered.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("scene_id", "seed", "shape_class", "target_prompt", "region_prompt"),
    [
        (
            "fastener-bin-sort",
            41,
            "hex_bolt",
            "red object",
            "blue square area inside bin",
        ),
        (
            "tool-bin-sort",
            53,
            "open_end_wrench",
            "yellow open end tool",
            "green square area inside bin",
        ),
    ],
)
def test_m3_industrial_scenes_bind_composite_target_to_one_of_two_bins(
    scene_id: str,
    seed: int,
    shape_class: str,
    target_prompt: str,
    region_prompt: str,
) -> None:
    contract = load_acceptance_scene_contract(scene_id)
    config = NativePickPlaceConfig(acceptance_scene_id=scene_id)

    assert contract["seed"] == seed
    assert contract["target_object"]["shape_class"] == shape_class
    assert contract["task"]["target_prompt"] == target_prompt
    assert contract["task"]["placement_region_prompt"] == region_prompt
    assert len(contract["target_object"]["primitives"]) >= 2
    assert len(contract["placement_regions"]) == 2
    assert sum(region["selected"] for region in contract["placement_regions"]) == 1
    assert config.replace_default_distractor is True
    assert list(config.target_size_m) == contract["target_object"][
        "bounding_box_xyz"
    ]
    assert list(config.target_initial_xyz) == contract["target_object"]["pose_xyz"]
    assert list(config.destination_center_xy) == contract["destination_center_xy"]
    assert list(config.destination_size_xy_m) == contract["destination_size_xy_m"]
    assert config.placement_center_height_m == pytest.approx(
        config.table_top_z_m + config.target_size_m[2] / 2.0
    )


@pytest.mark.parametrize(
    ("scene_id", "target_collision_count", "placement_ids"),
    [
        ("fastener-bin-sort", 2, ["blue_parts_bin", "orange_parts_bin"]),
        ("tool-bin-sort", 3, ["purple_tool_bin", "green_tool_bin"]),
    ],
)
def test_m3_industrial_renderer_materializes_real_parts_and_bin_floors(
    scene_id: str,
    target_collision_count: int,
    placement_ids: list[str],
) -> None:
    config = NativePickPlaceConfig(acceptance_scene_id=scene_id)
    package = config.ros_workspace / "src" / config.ros_package_name
    rendered, selected = render_acceptance_world(
        base_world=package / "worlds/rm75_robotiq2f85_pickplace.sdf",
        catalog_path=package / "config/acceptance_scenes.json",
        environment={"OPENETA_ACCEPTANCE_SCENE": scene_id},
    )
    try:
        world = ET.parse(rendered).getroot().find("world")
        assert world is not None and selected == scene_id
        assert world.find("model[@name='distractor_object']") is None
        target = world.find("model[@name='target_object']")
        assert target is not None
        assert len(target.findall("link/collision")) == target_collision_count
        assert len(target.findall("link/visual")) == target_collision_count
        for placement_id in placement_ids:
            floor = world.find(f"model[@name='{placement_id}']")
            assert floor is not None
            assert floor.find("link/collision") is None
            assert floor.find("link/visual/geometry/box/size") is not None
        for obstacle in config.acceptance_scene_contract["static_obstacles"]:
            model = world.find(f"model[@name='{obstacle['id']}']")
            assert model is not None
            expected_primitives = len(obstacle.get("primitives") or [obstacle])
            assert len(model.findall("link/collision")) == expected_primitives
            assert len(model.findall("link/visual")) == expected_primitives
    finally:
        rendered.unlink(missing_ok=True)


def test_m3_normal_scene_renderer_reuses_the_canonical_world() -> None:
    config = NativePickPlaceConfig()
    package = config.ros_workspace / "src" / config.ros_package_name
    canonical = package / "worlds/rm75_robotiq2f85_pickplace.sdf"

    rendered, selected = render_acceptance_world(
        base_world=canonical,
        catalog_path=package / "config/acceptance_scenes.json",
        environment={"OPENETA_ACCEPTANCE_SCENE": "normal"},
    )

    assert rendered == canonical
    assert selected == "normal"


def test_m3_stable_motion_contract_uses_bilateral_contact_goal_tolerances() -> None:
    motion = validated_pickplace_motion_guidance()["motion_parameters"]

    assert motion["tolerance"] == 0.0002
    assert motion["ori_tolerance"] == 0.002


def test_m3_paused_launch_gives_runtime_a_bounded_detach_window() -> None:
    config = NativePickPlaceConfig()
    launch = (
        config.ros_workspace
        / "src"
        / config.ros_package_name
        / "launch/gazebo_pickplace.launch.py"
    ).read_text(encoding="utf-8")

    assert 'switch_timeout = ["--switch-timeout", "30.0"]' in launch
    assert launch.count("*switch_timeout") == 3


def test_m3_target_pose_contract_is_a_single_link_at_the_model_origin() -> None:
    """Keep the native Pose_V world/local-frame proof contract explicit."""

    config = NativePickPlaceConfig()
    world_path = (
        config.ros_workspace
        / "src"
        / config.ros_package_name
        / "worlds/rm75_robotiq2f85_pickplace.sdf"
    )
    root = ET.parse(world_path).getroot()
    target = root.find(f".//model[@name='{config.target_id}']")
    assert target is not None
    assert tuple(float(value) for value in target.findtext("pose", "").split()[:3]) == config.target_initial_xyz
    links = target.findall("link")
    assert [link.get("name") for link in links] == [config.target_link]
    assert links[0].find("pose") is None


def test_m3_contact_sensor_topics_use_the_sdf_sensor_topic_field() -> None:
    config = NativePickPlaceConfig()
    xacro_path = (
        config.ros_workspace
        / "src"
        / config.ros_package_name
        / "urdf/rm75_robotiq2f85_pickplace.urdf.xacro"
    )
    root = ET.parse(xacro_path).getroot()
    sensors = root.findall(".//sensor[@type='contact']")

    assert [sensor.findtext("topic") for sensor in sensors] == [
        config.left_contact_topic,
        config.right_contact_topic,
    ]
    assert all(sensor.find("contact/topic") is None for sensor in sensors)


def test_m3_sdf_renderer_allows_only_the_stock_fixed_joint_topology() -> None:
    root = ET.fromstring(
        """<sdf><model name="robot"><link name="base_link"/>
        <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
          <parent_link>gripper_mount_link</parent_link><child_model>target_object</child_model>
          <child_link>target_link</child_link><attach_topic>/openeta/native_grasp/detachable_joint/target/attach</attach_topic>
          <detach_topic>/openeta/native_grasp/detachable_joint/target/detach</detach_topic>
          <output_topic>/openeta/native_grasp/detachable_joint/target/state</output_topic>
        </plugin></model></sdf>"""
    )

    prepared = prepare_detachable_sdf(root)
    model = prepared.find("model")
    assert model is not None
    assert model.findtext("joint[@name='openeta_world_to_base']/parent") == "world"
    assert model.findtext("joint[@name='openeta_world_to_base']/child") == "base_link"
    assert model.findtext("self_collide") == "false"


def test_m3_sdf_renderer_replaces_converter_contact_topic_placeholders() -> None:
    root = ET.fromstring(
        """<sdf><model name="robot"><link name="base_link"/><link name="tip">
        <sensor name="pad" type="contact"><topic>/openeta/native_grasp/contacts/left_pad</topic>
          <contact><collision>tip_collision</collision><topic>__default_topic__</topic></contact>
        </sensor></link>
        <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
          <parent_link>gripper_mount_link</parent_link><child_model>target_object</child_model>
          <child_link>target_link</child_link><attach_topic>/openeta/native_grasp/detachable_joint/target/attach</attach_topic>
          <detach_topic>/openeta/native_grasp/detachable_joint/target/detach</detach_topic>
          <output_topic>/openeta/native_grasp/detachable_joint/target/state</output_topic>
        </plugin></model></sdf>"""
    )

    prepared = prepare_detachable_sdf(root)
    sensor = prepared.find(".//sensor[@name='pad']")
    assert sensor is not None
    assert sensor.findtext("topic") == "/openeta/native_grasp/contacts/left_pad"
    assert sensor.findtext("contact/topic") == "/openeta/native_grasp/contacts/left_pad"
