from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from extensions.gazebo import GazeboProcess, Ros2LaunchProcess


def test_gazebo_process_lifecycle_when_gz_is_installed() -> None:
    gz = shutil.which("gz") or ("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz" if os.path.exists("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz") else None)
    if gz is None:
        pytest.skip("Gazebo Sim executable is not installed")
    process = GazeboProcess(world="extensions/gazebo/worlds/m1_oracle.sdf", gz_executable=gz)
    try:
        pid = process.start()
        assert pid > 0
        assert process.running
        process.wait_for_topics(())
    finally:
        process.close()
        process.close()
    assert not process.running


def test_ros_launch_close_waits_for_every_process_group_member() -> None:
    child_program = (
        "import signal,sys,time;"
        "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(0.4), sys.exit(0)));"
        "print('ready', flush=True);"
        "time.sleep(60)"
    )
    leader_program = (
        "import subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_program!r}],"
        "stdout=subprocess.PIPE,text=True);"
        "p.stdout.readline();"
        "print(p.pid, flush=True);"
        "time.sleep(60)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_program],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )
    assert leader.stdout is not None
    child_pid = int(leader.stdout.readline().strip())
    launch = Ros2LaunchProcess(package="unused", launch_file="unused")
    launch._process = leader

    launch.close()

    assert leader.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
