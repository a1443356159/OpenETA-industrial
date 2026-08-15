"""Gazebo Sim process lifecycle used by the M1 embodiment boundary."""

from __future__ import annotations

from enum import StrEnum
import math
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path


class GazeboProcessError(RuntimeError):
    """Raised when the configured Gazebo process cannot be managed."""


DETACHABLE_MAX_RELATIVE_TRANSLATION_M = 0.001
DETACHABLE_MAX_RELATIVE_ROTATION_RAD = math.radians(0.5)


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

    def wait_for_topics(self, topics: tuple[str, ...], *, timeout_s: float = 30.0) -> None:
        """Wait until Gazebo's transport advertises every requested topic.

        The default budget doubles the original 15 s allowance: on WSL2 a
        cold ``gz sim`` start plus transport discovery can exceed 15 s even
        though discovery itself is healthy, so the readiness gate polls for
        longer instead of reporting a false discovery failure.
        """

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

    def set_model_pose(self, model_name: str, xyz: tuple[float, float, float]) -> None:
        """Teleport one model through the documented ``set_pose`` service.

        Harmonic's ``model_only`` world reset restores neither free-object
        poses nor robot joint states (verified live on gz-sim 8.11), so M3
        restores its manipulated objects explicitly without rewinding the
        monotonic simulation clock.  The control-plane service can stall
        briefly under simulator load, so one bounded retry is allowed.
        """

        if not model_name.strip() or "/" in model_name:
            raise ValueError("model_name must be a non-empty model identifier")
        executable = shutil.which(self.gz_executable) or self.gz_executable
        request = (
            f'name: "{model_name}", position: {{x: {float(xyz[0])}, y: {float(xyz[1])}, '
            f"z: {float(xyz[2])}}}, orientation: {{w: 1.0}}"
        )
        last_error = ""
        for _attempt in range(2):
            try:
                result = subprocess.run(
                    [executable, "service", "-s", f"/world/{self.world_name}/set_pose",
                     "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                     "--timeout", str(self.timeout_ms), "--req", request],
                    capture_output=True, text=True,
                    timeout=max(1.0, self.timeout_ms / 1000.0 + 2.0) * 2.0,
                    env=self.environment,
                )
            except subprocess.TimeoutExpired:
                last_error = "service call timed out"
                continue
            if result.returncode == 0 and "data: true" in result.stdout.lower():
                return
            last_error = f"{result.stdout[-500:]} {result.stderr[-500:]}"
        raise GazeboProcessError(
            f"Gazebo set_pose failed for {model_name}: {last_error}"
        )

    def __enter__(self) -> "GazeboProcess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class DetachableJointState(StrEnum):
    """What the plugin topic has acknowledged, never a physics proof."""

    UNKNOWN = "UNKNOWN"
    DETACHED = "DETACHED"
    ATTACHED = "ATTACHED"


class GazeboDetachableJointControl:
    """Drive the official ``DetachableJoint`` plugin topics for M3's fallback.

    The fallback was re-approved by the user on 2026-08-10 as a controlled
    mechanism: the OpenETA side publishes ``gz.msgs.Empty`` to the plugin's
    attach/detach topics only after the verifier's own contact and geometry
    conditions hold, so the joint never fabricates grasp evidence on its own.
    """

    def __init__(
        self,
        *,
        gz_executable: str = "gz",
        timeout_ms: int = 3000,
        environment: dict[str, str] | None = None,
        world_name: str = "m3_rm75_robotiq2f85_pickplace",
        parent_link: str = "gripper_mount_link",
    ) -> None:
        self.gz_executable = gz_executable
        self.timeout_ms = int(timeout_ms)
        self.environment = dict(environment) if environment is not None else None
        self.world_name = world_name
        self.parent_link = parent_link
        self._child_links = {
            "target": "target_link",
            "distractor": "distractor_link",
        }
        self._physical_baselines: dict[str, tuple[
            tuple[float, float, float], tuple[float, float, float, float]
        ]] = {}
        # Do not guess the initial state.  In particular, a state topic
        # message only acknowledges a plugin request; it cannot show that a
        # DART fixed joint was actually created.  The direct environment uses
        # later Odometry co-motion as the sole held-object proof.
        self._state = {
            "target": DetachableJointState.UNKNOWN,
            "distractor": DetachableJointState.UNKNOWN,
        }

    def _publish(self, topic: str) -> None:
        executable = shutil.which(self.gz_executable) or self.gz_executable
        result = subprocess.run(
            [executable, "topic", "-t", topic, "-m", "gz.msgs.Empty", "-p", ""],
            capture_output=True, text=True,
            timeout=max(1.0, self.timeout_ms / 1000.0 + 2.0),
            env=self.environment,
        )
        if result.returncode != 0:
            raise GazeboProcessError(
                f"detachable joint publish failed for {topic}: {result.stderr[-500:]}"
            )

    @staticmethod
    def _field(block: str, name: str, *, default: float) -> float:
        found = re.search(rf"\b{name}:\s*([-+0-9.eE]+)", block)
        return float(found.group(1)) if found else default

    @staticmethod
    def _normalised_quaternion(
        value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        length = math.sqrt(sum(component * component for component in value))
        if length <= 1e-12:
            raise ValueError("Pose_V contained a zero quaternion")
        return tuple(component / length for component in value)  # type: ignore[return-value]

    @classmethod
    def _relative_pose(
        cls,
        parent: tuple[tuple[float, float, float], tuple[float, float, float, float]],
        child: tuple[tuple[float, float, float], tuple[float, float, float, float]],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Return child pose in the parent frame from Gazebo world poses."""

        parent_position, parent_quaternion = parent
        child_position, child_quaternion = child
        px, py, pz, pw = cls._normalised_quaternion(parent_quaternion)
        inverse = (-px, -py, -pz, pw)

        def multiply(
            left: tuple[float, float, float, float], right: tuple[float, float, float, float]
        ) -> tuple[float, float, float, float]:
            x1, y1, z1, w1 = left
            x2, y2, z2, w2 = right
            return (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * z2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            )

        delta = (
            child_position[0] - parent_position[0],
            child_position[1] - parent_position[1],
            child_position[2] - parent_position[2],
            0.0,
        )
        rotated = multiply(multiply(inverse, delta), (px, py, pz, pw))
        orientation = cls._normalised_quaternion(multiply(inverse, child_quaternion))
        return rotated[:3], orientation

    def _world_poses(self) -> dict[
        str, tuple[tuple[float, float, float], tuple[float, float, float, float]]
    ]:
        """Read physical link poses directly, never through model odometry."""

        executable = shutil.which(self.gz_executable) or self.gz_executable
        try:
            result = subprocess.run(
                [
                    executable, "topic", "-e", "-n", "1", "-t",
                    f"/world/{self.world_name}/pose/info",
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
                env=self.environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GazeboProcessError("unable to read detachable child Pose_V") from exc
        if result.returncode != 0 or not result.stdout.strip():
            raise GazeboProcessError("unable to read detachable child Pose_V")
        blocks: list[str] = []
        current: list[str] | None = None
        depth = 0
        for line in result.stdout.splitlines():
            if current is None:
                if line.strip() == "pose {":
                    current, depth = [line], 1
                continue
            current.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0:
                blocks.append("\n".join(current))
                current = None
        poses: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
        for block in blocks:
            name = re.search(r'\bname:\s*"([^"]+)"', block)
            position = re.search(r"position\s*\{(.*?)\}", block, re.DOTALL)
            orientation = re.search(r"orientation\s*\{(.*?)\}", block, re.DOTALL)
            if not name or not position or not orientation:
                continue
            poses[name.group(1)] = (
                tuple(self._field(position.group(1), axis, default=0.0) for axis in ("x", "y", "z")),
                self._normalised_quaternion(tuple(
                    self._field(orientation.group(1), axis, default=1.0 if axis == "w" else 0.0)
                    for axis in ("x", "y", "z", "w")
                )),
            )
        return poses

    def capture_physical_baseline(self, object_label: str) -> bool:
        """Record a post-ACK child-link reference without treating it as proof."""

        child_link = self._child_links.get(object_label)
        if child_link is None:
            raise ValueError(f"unknown detachable object label: {object_label}")
        try:
            poses = self._world_poses()
            self._physical_baselines[object_label] = self._relative_pose(
                poses[self.parent_link], poses[child_link]
            )
        except (KeyError, ValueError, GazeboProcessError):
            self._physical_baselines.pop(object_label, None)
            return False
        return True

    def physical_relative_drift(self, object_label: str) -> tuple[float, float] | None:
        """Return child-link drift from the post-ACK reference, if readable.

        Gazebo Sim 8's OdometryPublisher reads the model entity.  Cross-model
        DART detachable transitions can leave that entity and the constrained
        child link inconsistent, so an Odometry-only held verdict is unsafe.
        """

        baseline = self._physical_baselines.get(object_label)
        child_link = self._child_links.get(object_label)
        if baseline is None or child_link is None:
            return None
        try:
            poses = self._world_poses()
            current = self._relative_pose(poses[self.parent_link], poses[child_link])
        except (KeyError, ValueError, GazeboProcessError):
            return None
        translation = math.dist(baseline[0], current[0])
        dot = abs(sum(left * right for left, right in zip(baseline[1], current[1])))
        rotation = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
        return translation, rotation

    def is_physically_held(self, object_label: str) -> bool:
        """Apply the detachable-only `<1 mm / <0.5°` child-link hard gate."""

        drift = self.physical_relative_drift(object_label)
        return bool(
            drift is not None
            and drift[0] < DETACHABLE_MAX_RELATIVE_TRANSLATION_M
            and drift[1] < DETACHABLE_MAX_RELATIVE_ROTATION_RAD
        )

    def state(self, object_label: str) -> DetachableJointState:
        try:
            return self._state[object_label]
        except KeyError as exc:
            raise ValueError(f"unknown detachable object label: {object_label}") from exc

    def _drive(self, object_label: str, action: str) -> DetachableJointState:
        if action not in {"attach", "detach"}:
            raise ValueError(f"unsupported detachable action: {action}")
        want = (
            DetachableJointState.ATTACHED
            if action == "attach"
            else DetachableJointState.DETACHED
        )
        if self.state(object_label) is want:
            return want
        expected = want.value.lower()
        executable = shutil.which(self.gz_executable) or self.gz_executable
        for _attempt in range(3):
            # Subscribe BEFORE publishing: the plugin publishes its state once
            # per transition, so a listener started after the publish can miss
            # it.
            echo = subprocess.Popen(
                [executable, "topic", "-e", "-t", f"/m3/detachable_joint/{object_label}/state",
                 "-n", "1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                env=self.environment,
            )
            try:
                time.sleep(0.5)  # let the CLI subscription establish
                self._publish(f"/m3/detachable_joint/{object_label}/{action}")
                out, _ = echo.communicate(timeout=5.0)
                if expected in out:
                    self._state[object_label] = want
                    return want
            except subprocess.TimeoutExpired:
                pass
            finally:
                if echo.poll() is None:
                    echo.kill()
                    echo.wait()
        raise GazeboProcessError(
            f"detachable joint {action} for {object_label} was not confirmed by the plugin"
        )

    def attach(self, object_label: str) -> DetachableJointState:
        # Attach is only meaningful after an explicit detached ACK.  This
        # avoids silently accepting a reattach request that the plugin drops.
        if self.state(object_label) is not DetachableJointState.DETACHED:
            raise GazeboProcessError(
                f"detachable joint {object_label} is not confirmed detached before attach"
            )
        return self._drive(object_label, "attach")

    def ensure_detached(self, object_label: str) -> DetachableJointState:
        """Best-effort-safe idempotent detach with no assumed initial state."""

        self._physical_baselines.pop(object_label, None)
        if self.state(object_label) is DetachableJointState.DETACHED:
            return DetachableJointState.DETACHED
        return self._drive(object_label, "detach")

    def detach(self, object_label: str) -> DetachableJointState:
        return self.ensure_detached(object_label)


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
