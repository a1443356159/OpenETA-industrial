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
        assert environment.runtime.started is False
    finally:
        environment.close()
