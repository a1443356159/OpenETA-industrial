"""Gazebo Sim process lifecycle used by the M1 embodiment boundary."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


class GazeboProcessError(RuntimeError):
    """Raised when the configured Gazebo process cannot be managed."""


class GazeboProcess:
    """Own one headless ``gz sim`` process and clean it up deterministically.

    The class owns process lifecycle only.  ROS 2 nodes, camera topics, and
    robot controllers remain separate adapters until their configuration is
    provided by the deployment.
    """

    def __init__(
        self,
        *,
        world: str | Path,
        gz_executable: str = "gz",
        startup_timeout_s: float = 10.0,
    ) -> None:
        self.world = Path(world)
        self.gz_executable = gz_executable
        self.startup_timeout_s = float(startup_timeout_s)
        self._process: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def start(self) -> int:
        if self.running:
            return int(self._process.pid)
        executable = shutil.which(self.gz_executable) or self.gz_executable
        if not Path(executable).exists() and shutil.which(self.gz_executable) is None:
            raise GazeboProcessError(f"Gazebo executable not found: {self.gz_executable}")
        if not self.world.is_file():
            raise GazeboProcessError(f"Gazebo world does not exist: {self.world}")
        try:
            self._process = subprocess.Popen(
                [executable, "sim", "-s", "-r", str(self.world)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
                text=True,
            )
        except OSError as exc:
            raise GazeboProcessError(f"failed to start Gazebo: {exc}") from exc
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.running:
                return int(self._process.pid)
            output = self._process.stderr.read() if self._process.stderr else ""
            raise GazeboProcessError(f"Gazebo exited during startup: {output[-1000:]}")
        self.close()
        raise GazeboProcessError("Gazebo did not stay running before startup timeout")

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2.0)
        if process.stderr is not None:
            process.stderr.close()

    def __enter__(self) -> "GazeboProcess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RosGzBridgeProcess:
    """Own an official ``ros_gz_bridge parameter_bridge`` process.

    Topic expressions use the syntax documented by the installed Jazzy
    bridge executable, for example ``/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock``.
    This class deliberately does not invent camera or controller mappings.
    """

    def __init__(
        self,
        *,
        topics: tuple[str, ...],
        ros2_executable: str = "ros2",
        startup_timeout_s: float = 5.0,
    ) -> None:
        self.topics = tuple(topics)
        self.ros2_executable = ros2_executable
        self.startup_timeout_s = float(startup_timeout_s)
        self._process: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> int:
        if self.running:
            return int(self._process.pid)
        if not self.topics or any("@" not in topic for topic in self.topics):
            raise GazeboProcessError("ros_gz_bridge topics must use documented @ type syntax")
        executable = shutil.which(self.ros2_executable) or self.ros2_executable
        if not Path(executable).exists() and shutil.which(self.ros2_executable) is None:
            raise GazeboProcessError(f"ROS 2 executable not found: {self.ros2_executable}")
        bridge_executable = Path(executable).parent.parent / "lib" / "ros_gz_bridge" / "parameter_bridge"
        command = ([str(bridge_executable), *self.topics]
                   if bridge_executable.is_file()
                   else [executable, "run", "ros_gz_bridge", "parameter_bridge", *self.topics])
        self._process = subprocess.Popen(
            command,
            # ros_gz_bridge's underlying rclcpp process does not initialize
            # its lazy subscriptions correctly when stdin is /dev/null;
            # retain the parent stdin as in the official launch action.
            stdin=None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.running:
                return int(self._process.pid)
            output = self._process.stderr.read() if self._process.stderr else ""
            raise GazeboProcessError(f"ROS-Gazebo bridge exited during startup: {output[-1000:]}")
        self.close()
        raise GazeboProcessError("ROS-Gazebo bridge did not stay running")

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2.0)
        if process.stderr is not None:
            process.stderr.close()
