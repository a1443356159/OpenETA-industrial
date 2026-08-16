from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from extensions.gazebo import GazeboProcess, Ros2LaunchProcess
from extensions.gazebo import process as gazebo_process


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


def test_ros_launch_inherits_worker_streams_for_case_local_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker log must retain Gazebo's own launch diagnostics."""

    captured = {}

    class Process:
        pid = 12345

        @staticmethod
        def poll():
            return None

    def popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(gazebo_process.subprocess, "Popen", popen)
    launch = Ros2LaunchProcess(
        package="openeta_rm75_robotiq2f85_sim",
        launch_file="m1_gazebo_rgbd.launch.py",
        ros2_executable=sys.executable,
    )

    assert launch.start() == 12345
    assert captured["kwargs"]["stdout"] is None
    assert captured["kwargs"]["stderr"] is None


def test_detachable_joint_wait_ready_requires_the_stock_endpoint_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M3 must not publish its one-shot detach before plugin endpoints exist."""

    seen = []

    def run(command, **kwargs):
        seen.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(
                [
                    "/m3/detachable_joint/target/attach",
                    "/m3/detachable_joint/target/detach",
                    "/m3/detachable_joint/target/state",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(gazebo_process.subprocess, "run", run)
    control = gazebo_process.GazeboDetachableJointControl(
        gz_executable=sys.executable,
        environment={"GZ_PARTITION": "isolated"},
    )

    control.wait_ready(timeout_s=1.0)

    assert seen[0][0] == [sys.executable, "topic", "-l"]
    assert seen[0][1]["env"] == {"GZ_PARTITION": "isolated"}


def test_detachable_joint_wait_ready_fails_closed_when_an_endpoint_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="/m3/detachable_joint/target/attach\n",
            stderr="",
        )

    monkeypatch.setattr(gazebo_process.subprocess, "run", run)
    control = gazebo_process.GazeboDetachableJointControl(gz_executable=sys.executable)

    with pytest.raises(gazebo_process.GazeboProcessError, match="M3_DETACHABLE_JOINT_NOT_READY"):
        control.wait_ready(timeout_s=0.001)


def test_detachable_joint_proof_uses_the_target_model_world_pose_for_m3() -> None:
    """M3's target link is local-to-model in Gazebo Pose_V, not world-space."""

    poses = {
        "gripper_mount_link": (0.10, -0.20, 0.50),
        "m3_target": (0.20, -0.20, 0.40),
        "target_link": (0.0, 0.0, 0.0),
    }
    control = gazebo_process.GazeboDetachableJointControl()
    control._state = gazebo_process.DetachableJointState.ATTACHED
    control._world_link_positions = lambda: dict(poses)  # type: ignore[method-assign]
    control.capture_baseline()

    poses.update(
        {
            "gripper_mount_link": (0.10, -0.20, 0.60),
            "m3_target": (0.20, -0.20, 0.50),
        }
    )
    proof = control.child_link_proof()

    assert proof.lift_m == pytest.approx(0.10)
    assert proof.capture_relative_translation_m == pytest.approx(0.0)


def test_detachable_joint_proof_rejects_a_noncanonical_child_link_frame() -> None:
    control = gazebo_process.GazeboDetachableJointControl()
    control._state = gazebo_process.DetachableJointState.ATTACHED
    control._world_link_positions = lambda: {
        "gripper_mount_link": (0.0, 0.0, 0.50),
        "m3_target": (0.0, 0.0, 0.40),
        "target_link": (0.0, 0.0, 0.01),
    }  # type: ignore[method-assign]

    with pytest.raises(gazebo_process.GazeboProcessError, match="M3_CHILD_LINK_STATE_UNAVAILABLE"):
        control.capture_baseline()
