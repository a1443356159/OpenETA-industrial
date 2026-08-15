"""Gazebo Sim process lifecycle used by the M1 embodiment boundary."""

from __future__ import annotations

import os
import math
import re
import shutil
import signal
import subprocess
import threading
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

    def set_paused(self, paused: bool) -> None:
        """Pause or resume through Gazebo's documented WorldControl service."""

        self._control(f"pause: {'true' if paused else 'false'}")

    def wait_ready(self, *, timeout_s: float) -> None:
        """Wait for a responsive documented world-control service.

        A live ``ros2 launch`` process only proves that the launch parent has
        started.  In particular, a cold headless ``sensors_demo.sdf`` can
        still be registering Gazebo transport services when M1 issues its
        first reset.  Listing the exact documented service and sending an
        empty ``WorldControl`` request proves that the server is responsive
        without changing pause, reset, or model state.
        """

        if timeout_s <= 0:
            raise ValueError("world-control readiness timeout must be positive")
        executable = shutil.which(self.gz_executable) or self.gz_executable
        service = f"/world/{self.world_name}/control"
        deadline = time.monotonic() + float(timeout_s)
        last_error = "service was not advertised"
        while time.monotonic() < deadline:
            try:
                listed = subprocess.run(
                    [executable, "service", "-l"], capture_output=True, text=True,
                    timeout=min(5.0, max(0.1, deadline - time.monotonic())),
                    env=self.environment,
                )
            except subprocess.TimeoutExpired:
                last_error = "service listing timed out"
            else:
                services = set(listed.stdout.splitlines())
                if listed.returncode == 0 and service in services:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        # The default WorldControl message is a documented
                        # no-op, unlike pause/reset.  Require its Boolean ACK
                        # so discovery alone can never be mistaken for a
                        # usable reset path.
                        probe = self._service_request(
                            executable,
                            "",
                            timeout_ms=max(1, min(self.timeout_ms, int(remaining * 1000))),
                            command_timeout_s=remaining,
                        )
                    except subprocess.TimeoutExpired:
                        last_error = "empty WorldControl probe timed out"
                    else:
                        if probe.returncode == 0 and "data: true" in probe.stdout.lower():
                            return
                        last_error = f"{probe.stdout[-500:]} {probe.stderr[-500:]}"
                elif listed.returncode:
                    last_error = f"{listed.stdout[-500:]} {listed.stderr[-500:]}"
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.1, remaining))
        raise GazeboProcessError(
            f"Gazebo world control not ready for {self.world_name}: {last_error}"
        )

    def _reset(self, mode: str, *, seed: int | None = None) -> None:
        if mode not in {"all", "model_only"}:
            raise ValueError("unsupported Gazebo reset mode")
        request = f"reset: {{{mode}: true}}"
        if seed is not None:
            request += f" seed: {int(seed)}"
        self._control(request)

    def _control(self, request: str) -> None:
        executable = shutil.which(self.gz_executable) or self.gz_executable
        last_error = ""
        # ros2 launch is alive before Gazebo advertises its control service.
        # One bounded retry covers that cold-start race without accepting a
        # failed world reset later in the runtime lifecycle.
        for attempt in range(2):
            try:
                result = self._service_request(executable, request)
            except subprocess.TimeoutExpired:
                last_error = "service call timed out"
            else:
                if result.returncode == 0 and "data: true" in result.stdout.lower():
                    return
                last_error = f"{result.stdout[-500:]} {result.stderr[-500:]}"
            if attempt == 0:
                time.sleep(0.2)
        raise GazeboProcessError(
            f"Gazebo world control failed for {self.world_name}: {last_error}"
        )

    def _service_request(
        self,
        executable: str,
        request: str,
        *,
        timeout_ms: int | None = None,
        command_timeout_s: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Call the documented WorldControl endpoint with the normal budget."""

        effective_timeout_ms = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        if effective_timeout_ms <= 0:
            raise ValueError("WorldControl timeout must be positive")
        outer_timeout_s = (
            max(0.1, float(command_timeout_s))
            if command_timeout_s is not None
            else max(1.0, effective_timeout_ms / 1000.0 + 2.0)
        )
        return subprocess.run(
            [executable, "service", "-s", f"/world/{self.world_name}/control",
             "--reqtype", "gz.msgs.WorldControl", "--reptype", "gz.msgs.Boolean",
             "--timeout", str(effective_timeout_ms), "--req", request],
            capture_output=True, text=True,
            timeout=outer_timeout_s,
            env=self.environment,
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


class DetachableJointState:
    """Acknowledged state of M3's one stock Gazebo joint."""

    UNKNOWN = "unknown"
    DETACHED = "detached"
    ATTACHED = "attached"


class GazeboDetachableJointControl:
    """Request and prove M3's stock ``DetachableJoint`` without fallback.

    The state topic is an ACK boundary only.  ``child_link_proof`` separately
    reads the native world ``pose/info`` stream and is the only source for the
    80 mm lift / 10 mm capture-relative proof.
    """

    def __init__(
        self,
        *,
        gz_executable: str = "gz",
        timeout_s: float = 5.0,
        environment: dict[str, str] | None = None,
        world_name: str = "m3_rm75_robotiq2f85_pickplace",
        parent_link: str = "gripper_mount_link",
        child_link: str = "target_link",
    ) -> None:
        self.gz_executable = gz_executable
        self.timeout_s = float(timeout_s)
        self.environment = dict(environment) if environment is not None else None
        self.world_name = world_name
        self.parent_link = parent_link
        self.child_link = child_link
        self._state = DetachableJointState.UNKNOWN
        self._baseline: tuple[float, tuple[float, float, float]] | None = None

    @property
    def state(self) -> str:
        return self._state

    @staticmethod
    def _executable(value: str) -> str:
        return shutil.which(value) or value

    def _publish_empty(self, topic: str) -> None:
        try:
            result = subprocess.run(
                [self._executable(self.gz_executable), "topic", "-t", topic, "-m", "gz.msgs.Empty", "-p", ""],
                capture_output=True, text=True, check=False, env=self.environment,
                timeout=self.timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GazeboProcessError("M3_DETACHABLE_JOINT_UNAVAILABLE") from exc
        if result.returncode:
            raise GazeboProcessError("M3_DETACHABLE_JOINT_UNAVAILABLE")

    def _request(self, action: str) -> str:
        if action not in {"attach", "detach"}:
            raise ValueError("unsupported DetachableJoint action")
        expected = action + "ed"
        topic = f"/m3/detachable_joint/target/{action}"
        # The plugin emits a transition message once.  Listener first avoids
        # accepting a command for which the state ACK was missed.
        try:
            listener = subprocess.Popen(
                [self._executable(self.gz_executable), "topic", "-e", "-n", "1", "-t", "/m3/detachable_joint/target/state"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=self.environment, start_new_session=True,
            )
        except OSError as exc:
            raise GazeboProcessError("M3_DETACHABLE_JOINT_UNAVAILABLE") from exc
        try:
            time.sleep(0.15)
            self._publish_empty(topic)
            output, _ = listener.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise GazeboProcessError(
                "M3_ATTACH_ACK_MISSING" if action == "attach" else "M3_DETACH_ACK_MISSING"
            ) from exc
        finally:
            if listener.poll() is None:
                listener.terminate()
                try:
                    listener.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    listener.kill()
                    listener.wait(timeout=1.0)
        if not re.search(rf'\bdata:\s*"?{expected}"?', output, flags=re.IGNORECASE):
            raise GazeboProcessError(
                "M3_ATTACH_ACK_MISSING" if action == "attach" else "M3_DETACH_ACK_MISSING"
            )
        self._state = DetachableJointState.ATTACHED if action == "attach" else DetachableJointState.DETACHED
        if action == "detach":
            self._baseline = None
        return self._state

    def ensure_detached(self, *, require_ack: bool = True) -> str:
        """Always request a fresh detach ACK when ``require_ack`` is true."""

        if not require_ack and self._state == DetachableJointState.DETACHED:
            return self._state
        return self._request("detach")

    def attach(self) -> str:
        if self._state != DetachableJointState.DETACHED:
            raise GazeboProcessError("M3_DETACH_ACK_MISSING")
        return self._request("attach")

    @staticmethod
    def _field(block: str, name: str, default: float = 0.0) -> float:
        match = re.search(rf"\b{name}:\s*([-+0-9.eE]+)", block)
        return float(match.group(1)) if match else default

    @staticmethod
    def _pose_blocks(text: str) -> dict[str, tuple[float, float, float]]:
        blocks: list[str] = []
        current: list[str] | None = None
        depth = 0
        for line in text.splitlines():
            if current is None:
                if line.strip() == "pose {":
                    current, depth = [line], 1
                continue
            current.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0:
                blocks.append("\n".join(current))
                current = None
        poses: dict[str, tuple[float, float, float]] = {}
        for block in blocks:
            name = re.search(r'\bname:\s*"([^"]+)"', block)
            position = re.search(r"position\s*\{(.*?)\}", block, re.DOTALL)
            if name is None or position is None:
                continue
            poses[name.group(1)] = tuple(
                GazeboDetachableJointControl._field(position.group(1), axis)
                for axis in ("x", "y", "z")
            )
        return poses

    def _world_link_positions(self) -> dict[str, tuple[float, float, float]]:
        try:
            result = subprocess.run(
                [self._executable(self.gz_executable), "topic", "-e", "-n", "1", "-t", f"/world/{self.world_name}/pose/info"],
                capture_output=True, text=True, check=False, env=self.environment,
                timeout=self.timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GazeboProcessError("M3_CHILD_LINK_STATE_UNAVAILABLE") from exc
        if result.returncode or not result.stdout.strip():
            raise GazeboProcessError("M3_CHILD_LINK_STATE_UNAVAILABLE")
        return self._pose_blocks(result.stdout)

    @staticmethod
    def _named_position(poses: dict[str, tuple[float, float, float]], link: str) -> tuple[float, float, float]:
        if link in poses:
            return poses[link]
        matches = [value for name, value in poses.items() if name.endswith(f"::{link}")]
        if len(matches) != 1:
            raise GazeboProcessError("M3_CHILD_LINK_STATE_UNAVAILABLE")
        return matches[0]

    def capture_baseline(self) -> None:
        """Record native target child-link state after the attach ACK."""

        if self._state != DetachableJointState.ATTACHED:
            raise GazeboProcessError("M3_ATTACH_ACK_MISSING")
        poses = self._world_link_positions()
        parent = self._named_position(poses, self.parent_link)
        child = self._named_position(poses, self.child_link)
        self._baseline = (child[2], tuple(child[index] - parent[index] for index in range(3)))

    def child_link_proof(self):
        """Return the approved child-link lift proof, or fail closed."""

        if self._baseline is None:
            raise GazeboProcessError("M3_CHILD_LINK_STATE_UNAVAILABLE")
        try:
            from .m3 import ChildLinkProof
            poses = self._world_link_positions()
            parent = self._named_position(poses, self.parent_link)
            child = self._named_position(poses, self.child_link)
            baseline_z, baseline_relative = self._baseline
            relative = tuple(child[index] - parent[index] for index in range(3))
            return ChildLinkProof(
                baseline_target_z_m=baseline_z,
                target_z_m=child[2],
                capture_relative_translation_m=math.dist(baseline_relative, relative),
            )
        except GazeboProcessError:
            raise
        except Exception as exc:
            raise GazeboProcessError("M3_CHILD_LINK_STATE_UNAVAILABLE") from exc


class GazeboNativeContactWindow:
    """Collect only raw Gazebo contact messages for an armed M3 close."""

    def __init__(
        self,
        *,
        gz_executable: str = "gz",
        environment: dict[str, str] | None = None,
        timeout_s: float = 3.0,
    ) -> None:
        self.gz_executable = gz_executable
        self.environment = dict(environment) if environment is not None else None
        self.timeout_s = float(timeout_s)
        self._processes: list[subprocess.Popen] = []
        self._threads: list[threading.Thread] = []
        self._samples: list[object] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._armed = False

    @staticmethod
    def _contact_message(block: str, side: str):
        from .m3 import NativeContactSample
        stamp = re.search(r"stamp\s*\{(.*?)\}", block, re.DOTALL)
        if stamp is None:
            return None
        seconds = GazeboDetachableJointControl._field(stamp.group(1), "sec")
        nanoseconds = GazeboDetachableJointControl._field(stamp.group(1), "nanosec", GazeboDetachableJointControl._field(stamp.group(1), "nsec"))
        collision_blocks = re.findall(r"collision[12]\s*\{(.*?)\}", block, re.DOTALL)
        names = tuple(
            match.group(1)
            for collision in collision_blocks
            for match in [re.search(r'\bname:\s*"([^"]+)"', collision)]
            if match is not None
        )
        if not names:
            return None
        try:
            return NativeContactSample(side, seconds + nanoseconds * 1e-9, time.monotonic(), names)
        except ValueError:
            return None

    def _read(self, stream, side: str) -> None:
        current: list[str] = []
        for line in iter(stream.readline, ""):
            # Every gz.msgs.Contacts begins with its Header.  Flush the prior
            # message at the next Header so bracket depth inside repeated
            # contacts cannot merge simulator samples.
            if line.strip() == "header {" and current:
                sample = self._contact_message("".join(current), side)
                if sample is not None:
                    with self._lock:
                        self._samples.append(sample)
                current = []
            current.append(line)
        if current:
            sample = self._contact_message("".join(current), side)
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)

    def arm(self) -> None:
        if self._armed:
            raise GazeboProcessError("M3_CONTACT_WINDOW_ALREADY_ARMED")
        executable = GazeboDetachableJointControl._executable(self.gz_executable)
        try:
            for side, topic in (("left", "/m3/contacts/left_pad"), ("right", "/m3/contacts/right_pad")):
                process = subprocess.Popen(
                    [executable, "topic", "-e", "-t", topic], stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, env=self.environment,
                    start_new_session=True, bufsize=1,
                )
                if process.stdout is None:
                    raise OSError("contact subscription has no stdout")
                thread = threading.Thread(target=self._read, args=(process.stdout, side), daemon=True)
                thread.start()
                self._processes.append(process)
                self._threads.append(thread)
            self._armed = True
        except OSError as exc:
            self.close()
            raise GazeboProcessError("M3_NATIVE_CONTACT_UNAVAILABLE") from exc

    def begin_post_close(self) -> None:
        """Discard pre-close transport backlog while preserving subscriptions."""

        if not self._armed:
            raise GazeboProcessError("M3_CONTACT_WINDOW_NOT_ARMED")
        with self._lock:
            self._samples.clear()

    def evaluate(self, *, close_completed_sim_time_s: float | None, config=None):
        from .m3 import ReasonCode, confirm_native_bilateral_contact
        deadline = time.monotonic() + self.timeout_s
        result = None
        while time.monotonic() < deadline:
            with self._lock:
                samples = tuple(self._samples)
            result = confirm_native_bilateral_contact(
                samples, close_completed_sim_time_s=close_completed_sim_time_s,
                now_monotonic_s=time.monotonic(), config=config,
            )
            if result.accepted or result.reason_code not in {
                ReasonCode.CONTACT_INSUFFICIENT_SAMPLES, ReasonCode.CONTACT_WINDOW_TOO_SHORT,
            }:
                return result
            time.sleep(0.02)
        return result or confirm_native_bilateral_contact(
            (), close_completed_sim_time_s=close_completed_sim_time_s,
            now_monotonic_s=time.monotonic(), config=config,
        )

    def close(self) -> None:
        for process in self._processes:
            if process.poll() is None:
                process.terminate()
        for process in self._processes:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._processes.clear()
        self._threads.clear()
        self._armed = False


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
            # A launch description can start Gazebo and bridge descendants.
            # Give the runtime an owned session so close() reaps the complete
            # tree without ever signalling the bench worker itself.
            start_new_session=True,
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
