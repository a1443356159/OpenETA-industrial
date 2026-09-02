from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from extensions.gazebo.detachable_sdf import (
    DetachableSdfError,
    prepare_detachable_sdf,
)
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
    compile_authoritative_scene,
    render_acceptance_world,
    scene_target_bindings,
)
from extensions.gazebo.profiles import CONTROL, PHYSICS, STRUCTURED_RECEIPT, gazebo_profile
from sim.env_registry import get_env_spec


def test_native_grasp_registration_exposes_the_approved_detachable_joint_profile() -> None:
    pickplace, control = get_env_spec(PICKPLACE_ENV_ID), get_env_spec(GAZEBO_CONTROL_ENV_ID)
    assert (
        pickplace is not None
        and control is not None
        and pickplace.display_name == PICKPLACE_DISPLAY_NAME
    )
    assert NativePickPlaceConfig().model_id == PICKPLACE_MODEL_ID
    assert NativePickPlaceConfig().allow_stalling is True
    assert NativePickPlaceConfig().placement_release_z_offset_m == pytest.approx(0.05)
    assert GazeboControlConfig().model_id == MODEL_ID
    profile = gazebo_profile("rm75_robotiq2f85_pickplace")
    assert profile.unavailable_reason is None
    assert {CONTROL, PHYSICS, STRUCTURED_RECEIPT} <= profile.capabilities
    assert profile.launch_file == "gazebo_pickplace.launch.py"
    assert profile.cameras[0].extrinsics["pos"] == [0.38, 0.0, 1.35]
    wrist = profile.cameras[1]
    assert wrist.frame_id == "wrist_camera_optical_frame"
    assert dict(wrist.extrinsics) == {
        "frame_transform": "tf_dynamic",
        "camera_frame": "opencv",
        "reference_frame": "base_link",
        "sensor_frame": "wrist_camera_optical_frame",
        "sensor_frame_convention": "gazebo_camera",
    }


def test_rm75_physical_pedestal_uses_one_workcell_mounting_datum() -> None:
    config = NativePickPlaceConfig()
    description = (
        config.asset_root / "urdf" / "rm75_6fb_v.urdf.xacro"
    )
    root = ET.parse(description).getroot()
    base = root.find("link[@name='base_link']")
    joint = root.find("joint[@name='joint_1']")

    assert base is not None and joint is not None
    origins = [
        base.find("inertial/origin"),
        base.find("visual/origin"),
        base.find("collision/origin"),
        joint.find("origin"),
    ]
    assert all(origin is not None for origin in origins)
    inertial_x, visual_x, collision_x, joint_x = [
        float(origin.get("xyz", "").split()[0])  # type: ignore[union-attr]
        for origin in origins
    ]

    assert visual_x == pytest.approx(0.20)
    assert collision_x == pytest.approx(visual_x)
    assert joint_x == pytest.approx(visual_x)
    assert inertial_x - 0.00049987 == pytest.approx(visual_x)


def test_native_grasp_top_camera_profile_matches_the_world_model_pose() -> None:
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


def test_native_grasp_world_uses_soft_ambient_light_and_black_industrial_floor() -> None:
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
    assert [light.get("name") for light in lights] == ["workcell_key"]
    assert lights[0].findtext("cast_shadows") == "true"
    assert lights[0].findtext("direction") == "-0.42 0.25 -0.87"
    assert world.find("model[@name='ground']") is None
    floor = world.find("model[@name='industrial_floor']")
    assert floor is not None
    assert floor.findtext("pose") == "0 0 -0.81 0 0 0"
    assert floor.findtext("link/collision/geometry/box/size") == "8 8 0.02"
    assert floor.findtext("link/visual/geometry/box/size") == "8 8 0.02"
    assert floor.findtext("link/visual/material/ambient") == "0.012 0.015 0.021 1"
    assert floor.findtext("link/visual/material/specular") == "0.025 0.028 0.034 1"
    table = world.find("model[@name='work_table']")
    assert table is not None
    assert table.findtext("pose") == "0.35 0 -0.03 0 0 0"
    assert table.findtext("link/collision/geometry/box/size") == "1.15 0.95 0.06"
    assert table.findtext("link/visual/material/ambient") == "0.25 0.28 0.32 1"
    assert world.findtext("scene/ambient") == "0.28 0.30 0.34 1"
    assert world.findtext("scene/background") == "0.10 0.13 0.18 1"
    assert world.findtext("scene/shadows") == "true"


def test_native_grasp_assets_are_required_before_manipulation_starts() -> None:
    config = NativePickPlaceConfig()
    config.validate_assets()
    package = config.ros_workspace / "src" / config.ros_package_name
    assert (package / "worlds/rm75_robotiq2f85_pickplace.sdf").is_file()


def test_native_grasp_normal_scene_renderer_materializes_the_hashed_authoritative_world() -> None:
    config = NativePickPlaceConfig()
    package = config.ros_workspace / "src" / config.ros_package_name
    canonical = package / "worlds/rm75_robotiq2f85_pickplace.sdf"

    rendered, selected = render_acceptance_world(
        base_world=canonical,
        catalog_path=package / "config/acceptance_scenes.json",
        environment={"OPENETA_ACCEPTANCE_SCENE": "normal"},
    )

    try:
        compiled = compile_authoritative_scene(
            base_world=canonical,
            catalog_path=package / "config/acceptance_scenes.json",
            scene_id="normal",
        )
        assert rendered != canonical
        assert selected == "normal"
        assert rendered.read_bytes() == compiled.sdf_bytes
        assert compiled.authority_sha256[:12] in rendered.name
    finally:
        rendered.unlink(missing_ok=True)


def test_native_grasp_normal_bin_admits_target_and_complete_release_envelope() -> None:
    config = NativePickPlaceConfig()
    package = config.ros_workspace / "src" / config.ros_package_name
    world = ET.parse(package / "worlds/rm75_robotiq2f85_pickplace.sdf").getroot().find("world")
    assert world is not None

    contract = load_acceptance_scene_contract("normal")
    green = world.find("model[@name='green_parts_bin']")
    blue = world.find("model[@name='blue_parts_bin']")
    assert green is not None and blue is not None
    selected_region = next(
        region for region in contract["placement_regions"] if region["selected"]
    )
    assert contract["destination_center_xy"] == selected_region["center_xy"]
    assert contract["destination_size_xy_m"] == [0.285, 0.260]

    side_centers = sorted(
        abs(float(collision.findtext("pose").split()[0]))
        for collision in green.findall("link/collision")
        if collision.get("name") in {"left_wall", "right_wall"}
    )
    wall_thickness = float(
        green.findtext("link/collision[@name='left_wall']/geometry/box/size").split()[0]
    )
    clear_aperture = sum(side_centers) - wall_thickness
    target_length = float(contract["target_object"]["bounding_box_xyz"][0])
    complete_gripper_envelope = 0.149345541

    # Gazebo physics and PlanningScene must describe the same bin, including
    # the detailed asset's half-height operator-facing front wall.
    # The operator-facing visual remains the detailed mesh rather than an
    # opaque rendering of these conservative collision primitives.
    assert {collision.get("name") for collision in green.findall("link/collision")} == {
        "base",
        "left_wall",
        "right_wall",
        "front_wall",
        "rear_wall",
    }
    authority = config.authoritative_scene
    region_centers = {
        str(region["id"]): tuple(float(value) for value in region["center_xy"])
        for region in contract["placement_regions"]
    }
    for prefix in ("green", "blue"):
        bin_object = authority.object(f"{prefix}_parts_bin")
        assert bin_object.pose_xyz == pytest.approx(
            (*region_centers[f"{prefix}_parts_bin"], 0.0)
        )
        assert [primitive.name for primitive in bin_object.primitives] == [
            "base",
            "left_wall",
            "right_wall",
            "front_wall",
            "rear_wall",
        ]
        assert bin_object.visual_count == 1
        assert len(bin_object.moveit_spec()["primitives"]) == len(
            world.find(f"model[@name='{prefix}_parts_bin']").findall(
                "link/collision"
            )
        )
        primitives = {primitive.name: primitive for primitive in bin_object.primitives}
        assert primitives["front_wall"].pose_xyz[2] == pytest.approx(0.045)
        assert primitives["front_wall"].size_xyz == pytest.approx((0.32, 0.047, 0.09))
        assert primitives["rear_wall"].size_xyz == pytest.approx((0.32, 0.016, 0.18))

    # The short aperture follows the detailed mesh's measured inner wall,
    # while the semantic region remains strictly inside it.  The target and
    # complete native Robotiq envelope both retain physical release room.
    assert clear_aperture == pytest.approx(0.266)
    assert contract["destination_size_xy_m"][1] < clear_aperture
    assert clear_aperture >= target_length + 0.04
    assert clear_aperture >= complete_gripper_envelope + 0.05
    visuals = green.findall("link/visual")
    assert [visual.get("name") for visual in visuals] == ["bin_mesh"]
    assert visuals[0].find("geometry/mesh") is not None
    assert float(green.findtext("link/visual/geometry/mesh/scale").split()[0]) == 0.04
    assert float(blue.findtext("link/visual/geometry/mesh/scale").split()[0]) == 0.04


def test_normal_target_compound_geometry_matches_thick_canonical_wrench() -> None:
    config = NativePickPlaceConfig(acceptance_scene_id="normal")
    primitives = config.target_collision_primitives
    package = config.ros_workspace / "src" / config.ros_package_name
    target = (
        ET.parse(package / "worlds/rm75_robotiq2f85_pickplace.sdf")
        .getroot()
        .find("world/model[@name='target_object']")
    )

    assert target is not None
    assert config.target_size_m == (0.22, 0.062, 0.030)
    assert config.target_initial_xyz[2] == pytest.approx(0.015)
    assert len(primitives) == 2
    assert [primitive["shape"] for primitive in primitives] == ["box", "box"]
    assert [primitive["size_xyz"] for primitive in primitives] == [
        [0.165, 0.025, 0.026],
        [0.055, 0.062, 0.030],
    ]
    assert target.findtext("pose").split()[2] == "0.015"
    assert target.findtext("link/inertial/mass") == "0.30"
    assert target.findtext("link/visual/geometry/mesh/scale") == "1 1 2.0"


def test_native_grasp_stable_motion_contract_uses_bilateral_contact_goal_tolerances() -> None:
    motion = validated_pickplace_motion_guidance()["motion_parameters"]

    assert motion["tolerance"] == 0.0002
    assert motion["ori_tolerance"] == 0.002
    assert motion["profile"] == "unloaded"
    assert motion["velocity_scaling"] == 0.10
    assert motion["acceleration_scaling"] == 0.04


def test_native_grasp_paused_launch_gives_runtime_a_bounded_detach_window() -> None:
    config = NativePickPlaceConfig()
    launch = (
        config.ros_workspace / "src" / config.ros_package_name / "launch/gazebo_pickplace.launch.py"
    ).read_text(encoding="utf-8")

    assert 'switch_timeout = ["--switch-timeout", "30.0"]' in launch
    assert 'service_call_timeout = ["--service-call-timeout", "30.0"]' in launch
    assert launch.count("*switch_timeout") == 3
    assert launch.count("*service_call_timeout") == 3


def test_native_grasp_target_pose_contract_is_a_single_link_at_the_model_origin() -> None:
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
    assert (
        tuple(float(value) for value in target.findtext("pose", "").split()[:3])
        == config.target_initial_xyz
    )
    links = target.findall("link")
    assert [link.get("name") for link in links] == [config.target_link]
    assert links[0].find("pose") is None


def test_native_grasp_contact_sensor_topics_use_the_sdf_sensor_topic_field() -> None:
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


def test_native_grasp_sdf_renderer_allows_only_the_stock_fixed_joint_topology() -> None:
    root = ET.fromstring(
        """<sdf><model name="robot"><link name="base_link">
        <collision name="base_collision"><geometry><box><size>1 1 1</size></box></geometry></collision>
        </link>
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
    assert model.findtext(
        "link/collision/surface/contact/collide_bitmask"
    ) == "1"


def test_multi_normal_materializes_two_independent_stock_detachable_joints() -> None:
    root = ET.fromstring(
        """<sdf><model name="robot"><link name="base_link"/>
        <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
          <parent_link>gripper_mount_link</parent_link><child_model>target_object</child_model>
          <child_link>target_link</child_link><attach_topic>/openeta/native_grasp/detachable_joint/target/attach</attach_topic>
          <detach_topic>/openeta/native_grasp/detachable_joint/target/detach</detach_topic>
          <output_topic>/openeta/native_grasp/detachable_joint/target/state</output_topic>
        </plugin></model></sdf>"""
    )
    contract = load_acceptance_scene_contract("multi_normal")

    prepared = prepare_detachable_sdf(
        root,
        target_bindings=scene_target_bindings(contract),
    )
    plugins = prepared.findall(
        "model/plugin[@name='gz::sim::systems::DetachableJoint']"
    )

    assert [plugin.findtext("child_model") for plugin in plugins] == [
        "target_object",
        "red_m24_hex_bolt",
    ]
    assert [plugin.findtext("output_topic") for plugin in plugins] == [
        "/openeta/native_grasp/detachable_joint/target/state",
        "/openeta/native_grasp/detachable_joint/red_m24_hex_bolt/state",
    ]


def test_multi_normal_physical_catalog_does_not_preselect_a_task() -> None:
    config = NativePickPlaceConfig(acceptance_scene_id="multi_normal")
    contract = load_acceptance_scene_contract("multi_normal")
    evidence = config.acceptance_scene_evidence()

    assert "task" not in contract
    assert all("selected" not in region for region in contract["placement_regions"])
    assert [item.target_id for item in config.manipulation_target_configs] == [
        "target_object",
        "red_m24_hex_bolt",
    ]
    assert evidence["manipulation_catalog"]["targets"][0]["target_prompt"] == (
        "yellow wrench"
    )
    assert "target_object" not in evidence
    assert "selected_placement_region_id" not in evidence
    assert "destination_center_xy" not in evidence


@pytest.mark.parametrize(
    ("items", "target_ids", "region_ids"),
    [
        (
            [
                {
                    "target_prompt": "yellow wrench",
                    "placement_region_prompt": "blue bin",
                },
                {
                    "target_prompt": "red hex bolt",
                    "placement_region_prompt": "green parts bin",
                },
            ],
            ["target_object", "red_m24_hex_bolt"],
            ["blue_parts_bin", "green_parts_bin"],
        ),
        (
            [
                {
                    "target_prompt": "红色六角螺栓",
                    "placement_region_prompt": "蓝色零件箱",
                },
                {
                    "target_prompt": "黄色活动扳手",
                    "placement_region_prompt": "绿色零件箱",
                },
            ],
            ["red_m24_hex_bolt", "target_object"],
            ["blue_parts_bin", "green_parts_bin"],
        ),
        (
            [
                {
                    "target_prompt": "red M24 hex bolt",
                    "placement_region_prompt": "green bin",
                },
                {
                    "target_prompt": "yellow adjustable wrench",
                    "placement_region_prompt": "blue parts bin",
                },
            ],
            ["red_m24_hex_bolt", "target_object"],
            ["green_parts_bin", "blue_parts_bin"],
        ),
    ],
)
def test_vlm_work_order_selects_order_and_bin_binding(
    items: list[dict[str, str]],
    target_ids: list[str],
    region_ids: list[str],
) -> None:
    configs = NativePickPlaceConfig(
        acceptance_scene_id="multi_normal"
    ).work_order_configs(items)

    assert [item.target_id for item in configs] == target_ids
    assert [item.selected_placement_region_id for item in configs] == region_ids
    assert all(item.work_order_item["source"] == "vlm_work_order" for item in configs)
    assert [item.destination_center_xy for item in configs] == [
        (0.62, -0.18) if region_id == "blue_parts_bin" else (0.62, 0.18)
        for region_id in region_ids
    ]
    assert [item.placement_acceptance_semantics for item in configs] == [
        "stable_geometry_centroid_inside"
        for _region_id in region_ids
    ]
    assert [
        item.work_order_item["placement_region_perception_prompt"]
        for item in configs
    ] == [
        "blue square area inside bin"
        if region_id == "blue_parts_bin"
        else "green parts bin"
        for region_id in region_ids
    ]


def test_vlm_work_order_rejects_duplicate_or_unknown_physical_targets() -> None:
    config = NativePickPlaceConfig(acceptance_scene_id="multi_normal")

    with pytest.raises(ValueError, match="cannot select one physical target twice"):
        config.work_order_configs(
            [
                {
                    "target_prompt": "yellow wrench",
                    "placement_region_prompt": "green parts bin",
                },
                {
                    "target_prompt": "黄色活动扳手",
                    "placement_region_prompt": "blue parts bin",
                },
            ]
        )
    with pytest.raises(ValueError, match="does not resolve uniquely"):
        config.work_order_configs(
            [
                {
                    "target_prompt": "orange screwdriver",
                    "placement_region_prompt": "green parts bin",
                }
            ]
        )


def test_native_grasp_sdf_renderer_rejects_a_conflicting_robot_collision_mask() -> None:
    root = ET.fromstring(
        """<sdf><model name="robot"><link name="base_link">
        <collision name="base_collision"><surface><contact><collide_bitmask>2</collide_bitmask></contact></surface><geometry><box><size>1 1 1</size></box></geometry></collision>
        </link>
        <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
          <parent_link>gripper_mount_link</parent_link><child_model>target_object</child_model>
          <child_link>target_link</child_link><attach_topic>/openeta/native_grasp/detachable_joint/target/attach</attach_topic>
          <detach_topic>/openeta/native_grasp/detachable_joint/target/detach</detach_topic>
          <output_topic>/openeta/native_grasp/detachable_joint/target/state</output_topic>
        </plugin></model></sdf>"""
    )

    with pytest.raises(
        DetachableSdfError,
        match="collision bitmask conflicts",
    ):
        prepare_detachable_sdf(root)


def test_native_grasp_sdf_renderer_replaces_converter_contact_topic_placeholders() -> None:
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
    assert sensor.find("contact/collision/surface") is None
