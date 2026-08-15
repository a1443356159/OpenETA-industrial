from __future__ import annotations

import pytest

from extensions.gazebo.deployment import GazeboDeploymentConfig
from extensions.gazebo.direct_env import GazeboDirectEnv
from extensions.gazebo.m3 import M3_UNAVAILABLE_REASON
from extensions.gazebo.profiles import gazebo_profile
from extensions.gazebo.process import GazeboProcessError


def test_m3_profile_refuses_direct_worker_construction() -> None:
    deployment = GazeboDeploymentConfig(
        ros_domain_id=17,
        gz_partition="test-partition",
        ros2_executable="ros2",
        gz_executable="gz",
        process_environment={"ROS_DOMAIN_ID": "17"},
    )
    with pytest.raises(GazeboProcessError, match=M3_UNAVAILABLE_REASON):
        GazeboDirectEnv(profile=gazebo_profile("m3_pickplace"), deployment=deployment)
