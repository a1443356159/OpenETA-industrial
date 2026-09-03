from __future__ import annotations

from extensions.gazebo.deployment import GazeboDeploymentConfig
from extensions.gazebo.direct_env import GazeboDirectEnv
from extensions.gazebo.native_grasp import NativePickPlaceConfig
from extensions.gazebo.profiles import gazebo_profile


def test_native_grasp_profile_constructs_without_starting_a_worker() -> None:
    config = NativePickPlaceConfig()
    deployment = GazeboDeploymentConfig(
        ros_domain_id=17,
        gz_partition="test-partition",
        ros2_executable="ros2",
        gz_executable="gz",
        process_environment={"ROS_DOMAIN_ID": "17"},
    )
    environment = GazeboDirectEnv(profile=gazebo_profile("rm75_robotiq2f85_pickplace"), deployment=deployment)
    try:
        assert environment.openeta_control_spec["native_grasp"] is True
        motion = environment.openeta_control_spec["validated_pickplace_motion"]
        assert "poses" not in motion
        assert motion["terminal_poses"] == {
            "grasp_contact": "grasp_provider_model_pose_after_calibrated_frame_transform",
            "placement_release": "anyplace_object_goal_times_inverse_measured_attachment",
            "path_owner": "moveit",
            "host_pose_offsets_forbidden": True,
        }
        assert motion["atomic_order"][0] == {
            "tool": "move_to",
            "pose": "grasp_contact",
            "path": "moveit_full_path",
        }
        assert motion["atomic_order"][1]["requires_receipt"] == [
            "native_bilateral_contact",
            "attached_ack",
        ]
        assert motion["success_evidence"]["placement"] == {
            "verification_authority": "vlm_post_release_observation",
            "visual_source": "causal_post_release_rgbd",
            "blocking_simulator_stability_poll": False,
            "release_completion": "native_detach_and_gripper_open_ack",
            "geometry_role": "obvious_failure_veto_only",
            "support_plane_height_m": 0.02,
            "height_rule": "reject_support_penetration_only",
            "support_height_tolerance_m": 0.01,
            "destination_center_xy": list(config.destination_center_xy),
            "destination_size_xy_m": [0.285, 0.260],
            "footprint_rule": "reject_only_no_destination_overlap",
            "complete_footprint_margin_role": "ordering_and_evidence_only",
        }
        assert environment.runtime.started is False
    finally:
        environment.close()
