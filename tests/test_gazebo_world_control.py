from __future__ import annotations

import os
import shutil
from types import SimpleNamespace

import pytest

from extensions.gazebo import GazeboProcess, GazeboWorldControl
from extensions.gazebo import process as gazebo_process


def test_world_reset_retries_a_cold_control_service(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        SimpleNamespace(returncode=1, stdout="", stderr="unavailable"),
        SimpleNamespace(returncode=0, stdout="data: true", stderr=""),
    ]
    monkeypatch.setattr(gazebo_process.shutil, "which", lambda _name: "/usr/bin/gz")
    monkeypatch.setattr(gazebo_process.subprocess, "run", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(gazebo_process.time, "sleep", lambda _seconds: None)

    GazeboWorldControl(world_name="lidar_sensor").reset_all(seed=7)

    assert responses == []


def test_world_control_waits_for_advertisement_and_responsive_noop_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch liveness is insufficient: require exact service + Boolean ACK."""

    responses = [
        SimpleNamespace(returncode=0, stdout="/world/lidar_sensor/control/state\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="/world/lidar_sensor/control\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="data: true", stderr=""),
    ]
    calls = []
    monkeypatch.setattr(gazebo_process.shutil, "which", lambda _name: "/usr/bin/gz")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(gazebo_process.subprocess, "run", run)
    monkeypatch.setattr(gazebo_process.time, "sleep", lambda _seconds: None)

    GazeboWorldControl(world_name="lidar_sensor").wait_ready(timeout_s=1.0)

    assert responses == []
    assert calls[0][0] == ["/usr/bin/gz", "service", "-l"]
    assert calls[1][0] == ["/usr/bin/gz", "service", "-l"]
    assert calls[2][0][-2:] == ["--req", ""]


def test_world_control_readiness_fails_closed_without_the_exact_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Service discovery must never be treated as a successful reset fallback."""

    clock = iter((0.0, 0.0, 0.0, 0.0, 0.2))
    monkeypatch.setattr(gazebo_process.shutil, "which", lambda _name: "/usr/bin/gz")
    monkeypatch.setattr(gazebo_process.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(gazebo_process.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        gazebo_process.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="/world/lidar_sensor/control/state\n", stderr=""
        ),
    )

    with pytest.raises(gazebo_process.GazeboProcessError, match="not ready"):
        GazeboWorldControl(world_name="lidar_sensor").wait_ready(timeout_s=0.1)


def test_gazebo_world_reset_service() -> None:
    gz = shutil.which("gz") or ("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz" if os.path.exists("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz") else None)
    if gz is None:
        pytest.skip("Gazebo Sim is not installed")
    process = GazeboProcess(world="extensions/gazebo/worlds/m1_rgbd.sdf", gz_executable=gz)
    control = GazeboWorldControl(world_name="industrial_rgbd", gz_executable=gz)
    try:
        process.start()
        process.wait_for_topics(("/top_camera/image",))
        control.reset_all(seed=7)
    finally:
        process.close()
    assert not process.running
