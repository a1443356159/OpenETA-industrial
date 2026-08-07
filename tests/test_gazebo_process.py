from __future__ import annotations

import os
import shutil

import pytest

from extensions.gazebo import GazeboProcess


def test_gazebo_process_lifecycle_when_gz_is_installed() -> None:
    gz = shutil.which("gz") or ("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz" if os.path.exists("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz") else None)
    if gz is None:
        pytest.skip("Gazebo Sim executable is not installed")
    process = GazeboProcess(world="extensions/gazebo/worlds/m1_oracle.sdf", gz_executable=gz)
    try:
        pid = process.start()
        assert pid > 0
        assert process.running
    finally:
        process.close()
        process.close()
    assert not process.running

