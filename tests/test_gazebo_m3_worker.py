from __future__ import annotations

from extensions.gazebo.deployment import GazeboDeploymentConfig
from extensions.gazebo.direct_env import GazeboDirectEnv
from extensions.gazebo.profiles import gazebo_profile


def test_m3_profile_constructs_without_starting_a_worker() -> None:
    deployment = GazeboDeploymentConfig(
        ros_domain_id=17,
        gz_partition="test-partition",
        ros2_executable="ros2",
        gz_executable="gz",
        process_environment={"ROS_DOMAIN_ID": "17"},
    )
    environment = GazeboDirectEnv(profile=gazebo_profile("m3_pickplace"), deployment=deployment)
    try:
        assert environment.openeta_control_spec["m3"] is True
        motion = environment.openeta_control_spec["validated_pickplace_motion"]
        assert [item["name"] for item in motion["poses"]] == [
            "approach",
            "capture",
            "lift",
        ]
        assert motion["poses"][1]["target_pose"]["xyz"] == [0.1552, -0.1, 0.4976]
        assert motion["atomic_order"][2]["requires_receipt"] == [
            "native_bilateral_contact",
            "attached_ack",
        ]
        assert environment.runtime.started is False
    finally:
        environment.close()
