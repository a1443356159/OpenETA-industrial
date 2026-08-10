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


def _process_group_exists(pgid: int) -> bool:
    """Return whether an owned POSIX process group still has members."""

    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process: subprocess.Popen,
    pgid: int,
    *,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        # Reap the launch leader as soon as it exits.  Its exit alone is not a
        # cleanup result: Gazebo descendants can remain in the same group.
        process.poll()
        if not _process_group_exists(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _terminate_process_group(
    process: subprocess.Popen,
    *,
    terminate_timeout_s: float,
    kill_timeout_s: float,
) -> None:
    """Stop every member of a process group before reporting success."""

    pgid = process.pid  # every owned process is created with start_new_session
    if not _process_group_exists(pgid):
        process.poll()
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return
    if _wait_for_process_group_exit(process, pgid, timeout_s=terminate_timeout_s):
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        process.poll()
        return
    if not _wait_for_process_group_exit(process, pgid, timeout_s=kill_timeout_s):
        raise GazeboProcessError(
            f"process group {pgid} still exists after SIGTERM and SIGKILL"
        )


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
        process = self._process
        if process is None:
            return
        _terminate_process_group(
            process, terminate_timeout_s=5.0, kill_timeout_s=2.0
        )
        self._process = None
        if process.stderr is not None:
            process.stderr.close()

    def wait_for_topics(self, topics: tuple[str, ...], *, timeout_s: float = 15.0) -> None:
        """Wait until Gazebo's transport advertises every requested topic."""

        if not self.running:
            raise GazeboProcessError("Gazebo must be running before waiting for topics")
        if not topics:
            return
        executable = shutil.which(self.gz_executable) or self.gz_executable
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    [executable, "topic", "-l"], capture_output=True, text=True, timeout=5.0,
                )
                available = set(result.stdout.splitlines())
                if set(topics).issubset(available):
                    return
            except (OSError, subprocess.TimeoutExpired):
                pass
            time.sleep(0.1)
        raise GazeboProcessError(f"Gazebo topics not ready before timeout: {topics}")


class GazeboWorldControl:
    """Invoke Gazebo's documented ``/world/<name>/control`` reset service."""

    def __init__(self, *, world_name: str, gz_executable: str = "gz", timeout_ms: int = 3000,
                 environment: dict[str, str] | None = None) -> None:
        if not world_name.strip() or "/" in world_name:
            raise ValueError("world_name must be a non-empty Gazebo world identifier")
        self.world_name = world_name
        self.gz_executable = gz_executable
        self.timeout_ms = int(timeout_ms)
        self.environment = dict(environment) if environment is not None else None

    def reset_all(self, *, seed: int | None = None) -> None:
        self._reset("all", seed=seed)

    def reset_models(self, *, seed: int | None = None) -> None:
        """Reset world entities while preserving the monotonic simulation clock."""
        self._reset("model_only", seed=seed)

    def _reset(self, mode: str, *, seed: int | None = None) -> None:
        executable = shutil.which(self.gz_executable) or self.gz_executable
        if mode not in {"all", "model_only"}:
            raise ValueError("unsupported Gazebo reset mode")
        request = f"reset: {{{mode}: true}}"
        if seed is not None:
            request += f" seed: {int(seed)}"
        result = subprocess.run(
            [executable, "service", "-s", f"/world/{self.world_name}/control",
             "--reqtype", "gz.msgs.WorldControl", "--reptype", "gz.msgs.Boolean",
             "--timeout", str(self.timeout_ms), "--req", request],
            capture_output=True, text=True, timeout=max(1.0, self.timeout_ms / 1000.0 + 2.0),
            env=self.environment,
        )
        if result.returncode != 0 or "data: true" not in result.stdout.lower():
            raise GazeboProcessError(
                f"Gazebo world reset failed for {self.world_name}: {result.stdout[-500:]} {result.stderr[-500:]}"
            )

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
        process = self._process
        if process is None:
            return
        _terminate_process_group(
            process, terminate_timeout_s=5.0, kill_timeout_s=2.0
        )
        self._process = None
        if process.stderr is not None:
            process.stderr.close()


class Ros2LaunchProcess:
    """Own a documented ROS 2 launch description and its child processes."""

    def __init__(
        self,
        *,
        package: str,
        launch_file: str,
        arguments: tuple[str, ...] = (),
        ros2_executable: str = "ros2",
        startup_timeout_s: float = 5.0,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.package = package
        self.launch_file = launch_file
        self.arguments = tuple(arguments)
        self.ros2_executable = ros2_executable
        self.startup_timeout_s = float(startup_timeout_s)
        self.environment = dict(environment) if environment is not None else None
        self._process: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> int:
        if self.running:
            return int(self._process.pid)
        executable = shutil.which(self.ros2_executable) or self.ros2_executable
        if not Path(executable).exists() and shutil.which(self.ros2_executable) is None:
            raise GazeboProcessError(f"ROS 2 executable not found: {self.ros2_executable}")
        self._process = subprocess.Popen(
            [executable, "launch", self.package, self.launch_file, *self.arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            # The bench worker is the only process-group/session owner.
            start_new_session=False,
            text=True,
            env=self.environment,
        )
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.running:
                return int(self._process.pid)
            output = self._process.stderr.read() if self._process.stderr else ""
            raise GazeboProcessError(f"ROS 2 launch exited during startup: {output[-1000:]}")
        self.close()
        raise GazeboProcessError("ROS 2 launch did not stay running")

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        # A standalone launch created as a session leader owns its group (the
        # low-level class remains useful in isolation).  Production launch is
        # a member of the bench worker group and must never signal that group.
        if os.getpgid(process.pid) == process.pid:
            _terminate_process_group(
                process, terminate_timeout_s=8.0, kill_timeout_s=3.0
            )
        else:
            process.terminate()
            try:
                process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        self._process = None
        if process.stderr is not None:
            process.stderr.close()
