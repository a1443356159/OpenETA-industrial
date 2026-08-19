"""Gazebo bench-worker construction boundary.

The implementation lives in :mod:`extensions.gazebo.direct_env`; this module
contains no profile-specific worker classes or lifecycle behavior.
"""

from __future__ import annotations

from typing import Any

from .deployment import GazeboDeploymentConfig, worker_deployment_config
from .direct_env import GazeboDirectEnv
from .profiles import gazebo_profile


def make_gazebo_direct_env(
    *, profile: str = "rgbd_observation", deployment: GazeboDeploymentConfig | None = None,
    **kwargs: Any,
) -> GazeboDirectEnv:
    return GazeboDirectEnv(
        profile=gazebo_profile(profile),
        deployment=deployment or worker_deployment_config(),
        **kwargs,
    )
