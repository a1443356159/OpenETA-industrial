"""Gazebo Sim process lifecycle used by the observation-only embodiment boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
import math
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GazeboProcessError(RuntimeError):
    """Raised when the configured Gazebo process cannot be managed."""


@dataclass(frozen=True, slots=True)
class GazeboNativePose:
    xyz: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]


def _relative_translation(
    *, child: GazeboNativePose, parent: GazeboNativePose
) -> tuple[float, float, float]:
    """Express the parent-to-child translation in the parent link frame."""

    x, y, z, w = parent.quat_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE")
    x, y, z, w = (-x / norm, -y / norm, -z / norm, w / norm)
    dx, dy, dz = (child.xyz[index] - parent.xyz[index] for index in range(3))
    # Rotate the world-frame displacement by the inverse parent quaternion.
    # This is the translation component of T_world_parent^-1 T_world_child.
    return (
        (1.0 - 2.0 * (y * y + z * z)) * dx
        + 2.0 * (x * y - z * w) * dy
        + 2.0 * (x * z + y * w) * dz,
        2.0 * (x * y + z * w) * dx
        + (1.0 - 2.0 * (x * x + z * z)) * dy
        + 2.0 * (y * z - x * w) * dz,
        2.0 * (x * z - y * w) * dx
        + 2.0 * (y * z + x * w) * dy
        + (1.0 - 2.0 * (x * x + y * y)) * dz,
    )


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
        still be registering Gazebo transport services when observation-only issues its
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
        poses nor robot joint states (verified live on gz-sim 8.11), so native-grasp
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
    """Acknowledged state of native-grasp's one stock Gazebo joint."""

    UNKNOWN = "unknown"
    DETACHED = "detached"
    ATTACHED = "attached"


class GazeboDetachableJointControl:
    """Request and prove native-grasp's stock ``DetachableJoint`` without fallback.

    The state topic is an ACK boundary only.  ``child_link_proof`` separately
    reads the native world ``pose/info`` stream to revalidate relative-pose
    retention during whatever MoveIt transport the task actually requires.
    """

    def __init__(
        self,
        *,
        gz_executable: str = "gz",
        timeout_s: float = 5.0,
        environment: dict[str, str] | None = None,
        world_name: str = "rm75_robotiq2f85_pickplace",
        parent_link: str = "gripper_mount_link",
        child_model: str = "target_object",
        child_link: str = "target_link",
        attach_topic: str = "/openeta/native_grasp/detachable_joint/target/attach",
        detach_topic: str = "/openeta/native_grasp/detachable_joint/target/detach",
        state_topic: str = "/openeta/native_grasp/detachable_joint/target/state",
        collision_filter_state_topic: str = (
            "/openeta/native_grasp/detachable_joint/target/"
            "collision_filter_state"
        ),
        collision_filter_state_request_topic: str = (
            "/openeta/native_grasp/detachable_joint/target/"
            "collision_filter_state/request"
        ),
        collision_filter_state_ack_topic: str = (
            "/openeta/native_grasp/detachable_joint/target/"
            "collision_filter_state/ack"
        ),
        robot_collision_filter_mask: int = 0x0001,
        detached_target_collision_filter_mask: int = 0xFFFF,
        attached_target_collision_filter_mask: int = 0x0002,
    ) -> None:
        self.gz_executable = gz_executable
        self.timeout_s = float(timeout_s)
        self.environment = dict(environment) if environment is not None else None
        self.world_name = world_name
        self.parent_link = parent_link
        self.child_model = child_model
        self.child_link = child_link
        self.attach_topic = str(attach_topic).strip()
        self.detach_topic = str(detach_topic).strip()
        self.state_topic = str(state_topic).strip()
        self.collision_filter_state_topic = str(
            collision_filter_state_topic
        ).strip()
        self.collision_filter_state_request_topic = str(
            collision_filter_state_request_topic
        ).strip()
        self.collision_filter_state_ack_topic = str(
            collision_filter_state_ack_topic
        ).strip()
        self.robot_collision_filter_mask = int(robot_collision_filter_mask)
        self.detached_target_collision_filter_mask = int(
            detached_target_collision_filter_mask
        )
        self.attached_target_collision_filter_mask = int(
            attached_target_collision_filter_mask
        )
        if (
            not self.attach_topic
            or not self.detach_topic
            or not self.state_topic
            or not self.collision_filter_state_topic
            or not self.collision_filter_state_request_topic
            or not self.collision_filter_state_ack_topic
            or self.robot_collision_filter_mask <= 0
            or self.detached_target_collision_filter_mask <= 0
            or self.attached_target_collision_filter_mask <= 0
            or (
                self.robot_collision_filter_mask
                & self.detached_target_collision_filter_mask
            )
            == 0
            or (
                self.robot_collision_filter_mask
                & self.attached_target_collision_filter_mask
            )
            != 0
        ):
            raise ValueError("invalid attached collision-filter contract")
        self._state = DetachableJointState.UNKNOWN
        self._collision_filter_attached: bool | None = None
        self._baseline: tuple[float, tuple[float, float, float]] | None = None
        self._last_native_pose_read_attempt_count = 0

    @property
    def state(self) -> str:
        return self._state

    @staticmethod
    def _executable(value: str) -> str:
        return shutil.which(value) or value

    def _publish_empty(self, topic: str, *, timeout_s: float | None = None) -> None:
        """Publish one Empty transport request within an explicit deadline.

        Attach / detach commands retain the controller-wide timeout.  The
        collision-filter state request is retryable, however, so each attempt
        must leave time for another fresh listener instead of allowing one
        overloaded ``gz topic`` client to consume the whole proof window.
        """

        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("Gazebo transport timeout must be positive")
        try:
            result = subprocess.run(
                [self._executable(self.gz_executable), "topic", "-t", topic, "-m", "gz.msgs.Empty", "-p", ""],
                capture_output=True, text=True, check=False, env=self.environment,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GazeboProcessError("NATIVE_GRASP_DETACHABLE_JOINT_UNAVAILABLE") from exc
        if result.returncode:
            raise GazeboProcessError("NATIVE_GRASP_DETACHABLE_JOINT_UNAVAILABLE")

    def wait_ready(self, *, timeout_s: float) -> None:
        """Wait for the stock joint's three transport endpoints.

        A live world-control service proves only that Gazebo itself is ready.
        The robot model and its stock DetachableJoint system are spawned later
        in the launch graph, so publishing the first mandatory detach before
        these endpoints exist loses the one-shot ACK.  Endpoint discovery is
        therefore a readiness gate, never an attach/detach retry or a physics
        fallback.  The caller still requires the subsequent listener-first
        state ACK from :meth:`ensure_detached`.
        """

        if timeout_s <= 0:
            raise ValueError("DetachableJoint readiness timeout must be positive")
        required_topics = {
            self.attach_topic,
            self.detach_topic,
            self.state_topic,
            self.collision_filter_state_topic,
            self.collision_filter_state_request_topic,
            self.collision_filter_state_ack_topic,
        }
        deadline = time.monotonic() + float(timeout_s)
        last_error = "endpoint discovery did not run"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                result = subprocess.run(
                    [self._executable(self.gz_executable), "topic", "-l"],
                    capture_output=True,
                    text=True,
                    check=False,
                    env=self.environment,
                    timeout=min(self.timeout_s, max(0.1, remaining)),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                topics = {line.strip() for line in result.stdout.splitlines() if line.strip()}
                if (
                    result.returncode == 0
                    and required_topics <= topics
                ):
                    return
                missing = sorted(required_topics - topics)
                detail = (
                    result.stderr or result.stdout
                )[-500:].strip()
                last_error = (
                    f"missing={','.join(missing)}"
                    + (f" detail={detail}" if detail else "")
                )
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        raise GazeboProcessError(f"NATIVE_GRASP_DETACHABLE_JOINT_NOT_READY: {last_error}")

    def _wait_collision_filter_state(self, *, attached: bool) -> None:
        """Request and prove the authoritative physics mask.

        This mirrors the stock DetachableJoint transport contract: start the
        state listener, prove its Boolean publisher and our subscription are
        discoverable, then publish one Empty request.  Receiving the dedicated
        ACK also proves the request subscription end-to-end.  The plugin
        answers from an atomic stable-state snapshot even while simulation
        physics is paused.
        """

        deadline = time.monotonic() + self.timeout_s
        # ``gz topic`` is a short-lived transport client.  On a loaded host a
        # listener can occasionally fail to receive its first response even
        # though the native filter has transitioned correctly.  Bound every
        # attempt so the remaining proof window can establish a new listener
        # and issue a new request.  This is a transport reliability policy,
        # independent of scene/object identity and collision geometry.
        attempt_budget_s = min(3.0, self.timeout_s)
        last_error = "no collision-filter state received"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            attempt_deadline = min(deadline, time.monotonic() + attempt_budget_s)
            listener: subprocess.Popen[str] | None = None
            output = ""
            listener_stderr = ""
            listener_returncode: int | None = None
            try:
                listener = subprocess.Popen(
                    [
                        self._executable(self.gz_executable),
                        "topic",
                        "-e",
                        "-n",
                        "1",
                        "-t",
                        self.collision_filter_state_ack_topic,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=self.environment,
                    start_new_session=True,
                )
                listener_ready = False
                publisher = (
                    r"Publishers\s*\[[^\]]*\]:\s*\n"
                    r"(?:\s+\S+,[^\n]*\n)*?\s+\S+,\s*gz\.msgs\.Boolean\b"
                )
                subscriber = r"Subscribers\s*\[[^\]]*\]:\s*\n\s+\S"
                while time.monotonic() < attempt_deadline:
                    remaining = attempt_deadline - time.monotonic()
                    try:
                        state_info = subprocess.run(
                            [
                                self._executable(self.gz_executable),
                                "topic",
                                "-i",
                                "-t",
                                self.collision_filter_state_ack_topic,
                            ],
                            capture_output=True,
                            text=True,
                            check=False,
                            env=self.environment,
                            timeout=min(1.0, max(0.1, remaining)),
                        )
                    except subprocess.TimeoutExpired as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        time.sleep(
                            min(0.02, max(0.0, deadline - time.monotonic()))
                        )
                        continue
                    if (
                        state_info.returncode == 0
                        and re.search(publisher, state_info.stdout)
                        and re.search(subscriber, state_info.stdout)
                    ):
                        listener_ready = True
                        break
                    detail = (state_info.stderr or state_info.stdout)[-500:].strip()
                    last_error = (
                        "collision-filter ACK endpoint not ready"
                        + (f": {detail}" if detail else "")
                    )
                    time.sleep(
                        min(0.02, max(0.0, deadline - time.monotonic()))
                    )
                if not listener_ready:
                    continue
                remaining = attempt_deadline - time.monotonic()
                if remaining <= 0:
                    last_error = "collision-filter ACK attempt expired before request"
                    continue
                self._publish_empty(
                    self.collision_filter_state_request_topic,
                    timeout_s=min(attempt_budget_s, remaining),
                )
                output, listener_stderr = listener.communicate(
                    timeout=max(0.1, attempt_deadline - time.monotonic())
                )
                listener_returncode = listener.poll()
            except (OSError, subprocess.TimeoutExpired, GazeboProcessError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                # The finally block retires this listener.  Retry from a
                # fresh transport endpoint while the explicit proof deadline
                # remains; do not turn one client-side timeout into a false
                # candidate rejection.
                continue
            finally:
                if listener is not None and listener.poll() is None:
                    listener.terminate()
                    try:
                        listener.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        listener.kill()
                        listener.wait(timeout=1.0)
            match = re.search(
                r"\bdata:\s*(true|false)\b",
                output,
                flags=re.IGNORECASE,
            )
            if match is not None:
                observed = match.group(1).lower() == "true"
            elif (
                listener_returncode == 0
                and not output.strip()
                and not listener_stderr.strip()
            ):
                # ``gz topic -e`` prints a blank record for the proto3 default
                # Boolean value.  A clean one-message exit plus the typed
                # publisher proof above therefore represents ``false``; an
                # absent message cannot make ``-n 1`` exit successfully.
                observed = False
            else:
                observed = None
            if observed is not None:
                if observed == attached:
                    self._collision_filter_attached = observed
                    return
                last_error = (
                    f"observed={'attached' if observed else 'detached'} "
                    f"expected={'attached' if attached else 'detached'}"
                )
            else:
                detail = (listener_stderr or output)[-500:].strip()
                last_error = "collision-filter response was not a Boolean"
                if detail:
                    last_error += f": {detail}"
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        self._collision_filter_attached = None
        raise GazeboProcessError(
            f"NATIVE_GRASP_COLLISION_FILTER_ACK_MISSING: {last_error}"
        )

    def collision_filter_evidence(self) -> dict[str, Any]:
        """Return the last ACKed Gazebo target/robot collision semantics."""

        if self._collision_filter_attached is None:
            raise GazeboProcessError(
                "NATIVE_GRASP_COLLISION_FILTER_ACK_MISSING"
            )
        mask = (
            self.attached_target_collision_filter_mask
            if self._collision_filter_attached
            else self.detached_target_collision_filter_mask
        )
        return {
            "schema_version": "openeta.attached_collision_filter.v1",
            "state": (
                "robot_excluded" if self._collision_filter_attached else "full"
            ),
            "joint_state": (
                DetachableJointState.ATTACHED
                if self._collision_filter_attached
                else DetachableJointState.DETACHED
            ),
            "state_topic": self.collision_filter_state_topic,
            "state_request_topic": self.collision_filter_state_request_topic,
            "state_ack_topic": self.collision_filter_state_ack_topic,
            "robot_mask": self.robot_collision_filter_mask,
            "target_mask": mask,
            "target_robot_collision_enabled": (
                self.robot_collision_filter_mask & mask
            )
            != 0,
            "target_environment_collision_enabled": mask != 0,
        }

    def _request(self, action: str) -> str:
        if action not in {"attach", "detach"}:
            raise ValueError("unsupported DetachableJoint action")
        expected = action + "ed"
        topic = self.attach_topic if action == "attach" else self.detach_topic
        state_topic = self.state_topic
        # The stock joint emits a transition message once.  Listener first
        # avoids accepting a command whose joint ACK was missed.  Collision
        # semantics are queried separately from the state service below.
        try:
            listener = subprocess.Popen(
                [
                    self._executable(self.gz_executable),
                    "topic",
                    "-e",
                    "-n",
                    "1",
                    "-t",
                    state_topic,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.environment,
                start_new_session=True,
            )
        except OSError as exc:
            raise GazeboProcessError("NATIVE_GRASP_DETACHABLE_JOINT_UNAVAILABLE") from exc
        output = ""
        try:
            # Process creation does not mean the transport subscription is
            # discoverable yet.  The state transition is published once, so
            # prove the listener appears in topic introspection before
            # sending the command instead of relying on a timing sleep.
            deadline = time.monotonic() + self.timeout_s
            listener_ready = False
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    state_info = subprocess.run(
                        [self._executable(self.gz_executable), "topic", "-i", "-t",
                         state_topic],
                        capture_output=True, text=True, check=False,
                        env=self.environment, timeout=min(5.0, max(0.1, remaining)),
                    )
                    remaining = deadline - time.monotonic()
                    command_info = subprocess.run(
                        [self._executable(self.gz_executable), "topic", "-i", "-t", topic],
                        capture_output=True, text=True, check=False,
                        env=self.environment, timeout=min(5.0, max(0.1, remaining)),
                    )
                except subprocess.TimeoutExpired:
                    pass
                else:
                    publisher = r"Publishers\s*\[[^\]]*\]:\s*\n\s+\S"
                    subscriber = r"Subscribers\s*\[[^\]]*\]:\s*\n\s+\S"
                    if (
                        state_info.returncode == 0
                        and command_info.returncode == 0
                        and re.search(publisher, state_info.stdout)
                        and re.search(subscriber, state_info.stdout)
                        and re.search(subscriber, command_info.stdout)
                    ):
                        listener_ready = True
                        break
                time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
            if not listener_ready:
                raise GazeboProcessError(
                    "NATIVE_GRASP_ATTACH_ACK_MISSING" if action == "attach" else "NATIVE_GRASP_DETACH_ACK_MISSING"
                )
            self._publish_empty(topic)
            remaining = max(0.1, deadline - time.monotonic())
            output, _ = listener.communicate(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise GazeboProcessError(
                "NATIVE_GRASP_ATTACH_ACK_MISSING" if action == "attach" else "NATIVE_GRASP_DETACH_ACK_MISSING"
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
                "NATIVE_GRASP_ATTACH_ACK_MISSING" if action == "attach" else "NATIVE_GRASP_DETACH_ACK_MISSING"
            )
        # The stock DetachableJoint acknowledgement establishes this state
        # before the independent collision-filter proof completes.  Keep that
        # fact available to cleanup and failure classification: a later
        # collision-filter transport timeout is infrastructure uncertainty,
        # not evidence that the physical candidate never attached.
        self._state = DetachableJointState.ATTACHED if action == "attach" else DetachableJointState.DETACHED
        expected_filter_state = action == "attach"
        self._wait_collision_filter_state(attached=expected_filter_state)
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
            raise GazeboProcessError("NATIVE_GRASP_DETACH_ACK_MISSING")
        return self._request("attach")

    @staticmethod
    def _field(block: str, name: str, default: float = 0.0) -> float:
        match = re.search(rf"\b{name}:\s*([-+0-9.eE]+)", block)
        return float(match.group(1)) if match else default

    @staticmethod
    def _pose_records(text: str) -> dict[str, GazeboNativePose]:
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
        poses: dict[str, GazeboNativePose] = {}
        for block in blocks:
            name = re.search(r'\bname:\s*"([^"]+)"', block)
            position = re.search(r"position\s*\{(.*?)\}", block, re.DOTALL)
            if name is None or position is None:
                continue
            xyz = tuple(
                GazeboDetachableJointControl._field(position.group(1), axis)
                for axis in ("x", "y", "z")
            )
            orientation = re.search(r"orientation\s*\{(.*?)\}", block, re.DOTALL)
            orientation_block = orientation.group(1) if orientation is not None else ""
            quaternion = tuple(
                GazeboDetachableJointControl._field(
                    orientation_block,
                    axis,
                    1.0 if axis == "w" else 0.0,
                )
                for axis in ("x", "y", "z", "w")
            )
            norm = math.sqrt(sum(value * value for value in quaternion))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE")
            poses[name.group(1)] = GazeboNativePose(
                xyz=xyz,  # type: ignore[arg-type]
                quat_xyzw=tuple(value / norm for value in quaternion),  # type: ignore[arg-type]
            )
        return poses

    @staticmethod
    def _pose_blocks(text: str) -> dict[str, tuple[float, float, float]]:
        """Compatibility projection of native poses: retain position only."""

        return {
            name: pose.xyz
            for name, pose in GazeboDetachableJointControl._pose_records(text).items()
        }

    def _world_link_poses(self) -> dict[str, GazeboNativePose]:
        try:
            result = subprocess.run(
                [self._executable(self.gz_executable), "topic", "-e", "-n", "1", "-t", f"/world/{self.world_name}/pose/info"],
                capture_output=True, text=True, check=False, env=self.environment,
                timeout=self.timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE") from exc
        if result.returncode or not result.stdout.strip():
            raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE")
        return self._pose_records(result.stdout)

    def _world_link_positions(self) -> dict[str, tuple[float, float, float]]:
        return {name: pose.xyz for name, pose in self._world_link_poses().items()}

    def native_model_poses_with_retry(
        self,
        model_names: Sequence[str],
        *,
        max_attempts: int = 2,
    ) -> tuple[dict[str, GazeboNativePose], int]:
        """Read one settled authoritative pose snapshot for world clutter.

        Gazebo and MoveIt share geometry from the compiled SDF, but dynamic
        non-target objects can settle a few millimetres after physics starts.
        This method overlays all such model poses from one Pose_V generation;
        a partial snapshot is never published to PlanningScene.
        """

        requested = tuple(str(name).strip() for name in model_names)
        if (
            max_attempts <= 0
            or any(not name for name in requested)
            or len(requested) != len(set(requested))
        ):
            raise ValueError("authoritative Gazebo model pose request is invalid")
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                poses = self._world_link_poses()
                selected: dict[str, GazeboNativePose] = {}
                for model_name in requested:
                    if model_name in poses:
                        selected[model_name] = poses[model_name]
                        continue
                    matches = [
                        pose
                        for name, pose in poses.items()
                        if name.endswith(f"::{model_name}")
                    ]
                    if len(matches) != 1:
                        raise GazeboProcessError(
                            "AUTHORITATIVE_GAZEBO_MODEL_POSE_UNAVAILABLE"
                        )
                    selected[model_name] = matches[0]
                return selected, attempt
            except Exception as exc:  # noqa: BLE001 - one bounded native retry.
                last_error = exc
        raise GazeboProcessError(
            f"AUTHORITATIVE_GAZEBO_MODEL_POSE_UNAVAILABLE: {last_error}"
        ) from last_error

    @staticmethod
    def _named_position(poses: dict[str, tuple[float, float, float]], link: str) -> tuple[float, float, float]:
        if link in poses:
            return poses[link]
        matches = [value for name, value in poses.items() if name.endswith(f"::{link}")]
        if len(matches) != 1:
            raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE")
        return matches[0]

    def _child_world_position(
        self, poses: dict[str, tuple[float, float, float]]
    ) -> tuple[float, float, float]:
        """Return native-grasp's child-link position in the world frame, or fail closed.

        Gazebo's native ``Pose_V`` stream reports the native-grasp target *model* in
        world coordinates but its sole ``target_link`` in that model's local
        coordinates.  The approved native-grasp SDF contract fixes that one link at the
        model origin (validated alongside the asset), so the model pose is the
        native world pose of the child link.  A non-origin link makes this
        assumption invalid and must fail rather than silently mixing frames.

        This deliberately reads only Gazebo pose state and the static SDF
        contract; it performs no TF lookup, geometry inference, or kinematic
        following.
        """

        child_local = self._named_position(poses, self.child_link)
        if any(abs(component) > 1e-6 for component in child_local):
            raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE")
        return self._named_position(poses, self.child_model)

    def capture_baseline(
        self, *, settle_duration_s: float = 0.10, sample_interval_s: float = 0.02
    ) -> int:
        """Record the settled native child-link state after an attach ACK.

        Gazebo acknowledges a detachable joint before the next physics tick has
        necessarily propagated the new fixed constraint into Pose_V.  Freezing
        the baseline on that first pre-settle sample makes harmless constraint
        convergence look like post-grasp slip.  The gripper is stationary at
        this call site; consume a short, bounded settling window and retain
        the final native pose.  This does not relax the later 10 mm drift
        proof: it merely defines capture at the first settled attached state.
        """

        if self._state != DetachableJointState.ATTACHED:
            raise GazeboProcessError("NATIVE_GRASP_ATTACH_ACK_MISSING")
        if settle_duration_s < 0.0 or sample_interval_s <= 0.0:
            raise ValueError("attachment baseline settling values are invalid")
        deadline = time.monotonic() + settle_duration_s
        child, parent, pose_read_attempt_count = (
            self.native_target_mount_poses_with_retry()
        )
        while time.monotonic() < deadline:
            time.sleep(min(sample_interval_s, max(0.0, deadline - time.monotonic())))
            child, parent, attempts = self.native_target_mount_poses_with_retry()
            pose_read_attempt_count += attempts
        self._baseline = (
            child.xyz[2],
            _relative_translation(child=child, parent=parent),
        )
        return pose_read_attempt_count

    def native_target_mount_positions(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Read the native Gazebo target and gripper-mount world positions."""

        poses = self._world_link_positions()
        return self._child_world_position(poses), self._named_position(poses, self.parent_link)

    @staticmethod
    def _named_pose(
        poses: dict[str, GazeboNativePose], link: str
    ) -> GazeboNativePose:
        if link in poses:
            return poses[link]
        matches = [value for name, value in poses.items() if name.endswith(f"::{link}")]
        if len(matches) != 1:
            raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE")
        return matches[0]

    @classmethod
    def _target_model_pose_from_snapshot(
        cls,
        poses: dict[str, GazeboNativePose],
        *,
        child_model: str,
        child_link: str,
    ) -> GazeboNativePose:
        """Resolve a target link world pose under the validated SDF contract."""

        child_local = cls._named_pose(poses, child_link)
        local_quat = child_local.quat_xyzw
        if (
            any(abs(component) > 1e-6 for component in child_local.xyz)
            or any(abs(component) > 1e-6 for component in local_quat[:3])
            or abs(abs(local_quat[3]) - 1.0) > 1e-6
        ):
            raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE")
        return cls._named_pose(poses, child_model)

    def native_target_model_poses_with_retry(
        self,
        target_links: Mapping[str, str],
        *,
        max_attempts: int = 2,
    ) -> tuple[dict[str, GazeboNativePose], int]:
        """Read several detachable targets from one validated Pose_V snapshot."""

        bindings = {
            str(model).strip(): str(link).strip()
            for model, link in target_links.items()
        }
        if (
            isinstance(max_attempts, bool)
            or max_attempts < 1
            or not bindings
            or any(not model or not link for model, link in bindings.items())
        ):
            raise ValueError("native target pose request is invalid")
        last_error: Exception | None = None
        for attempt in range(1, int(max_attempts) + 1):
            try:
                poses = self._world_link_poses()
                return (
                    {
                        model: self._target_model_pose_from_snapshot(
                            poses,
                            child_model=model,
                            child_link=link,
                        )
                        for model, link in bindings.items()
                    },
                    attempt,
                )
            except GazeboProcessError as exc:
                last_error = exc
                if (
                    str(exc) != "NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"
                    or attempt >= max_attempts
                ):
                    raise
        raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE") from last_error

    def native_target_mount_poses(
        self,
    ) -> tuple[GazeboNativePose, GazeboNativePose]:
        """Read full native world poses for planning-scene attach/detach."""

        poses = self._world_link_poses()
        return self._target_model_pose_from_snapshot(
            poses,
            child_model=self.child_model,
            child_link=self.child_link,
        ), self._named_pose(poses, self.parent_link)

    def native_target_mount_poses_with_retry(
        self,
        *,
        max_attempts: int = 2,
    ) -> tuple[GazeboNativePose, GazeboNativePose, int]:
        """Read a fresh native pose frame with one bounded transport retry.

        The detachable-joint ACK is emitted at the command transition while the
        independent ``Pose_V`` stream advances on a physics update.  The first
        frame after a valid ACK can therefore be incomplete.  Consume at most
        one additional fresh frame for that exact infrastructure condition; do
        not turn malformed geometry or arbitrary exceptions into retries.
        """

        if isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("native pose max_attempts must be a positive integer")
        self._last_native_pose_read_attempt_count = 0
        for attempt in range(1, int(max_attempts) + 1):
            self._last_native_pose_read_attempt_count = attempt
            try:
                child, parent = self.native_target_mount_poses()
                return child, parent, attempt
            except GazeboProcessError as exc:
                if (
                    str(exc) != "NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"
                    or attempt >= max_attempts
                ):
                    raise
        raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE")

    def sample_detached_target_poses(
        self,
        *,
        duration_s: float,
        interval_s: float,
    ):
        """Sample the released target long enough to prove terminal stability."""

        if self._state != DetachableJointState.DETACHED:
            raise GazeboProcessError("NATIVE_GRASP_DETACH_ACK_MISSING")
        if duration_s <= 0.0 or interval_s <= 0.0:
            raise ValueError("placement sampling duration and interval must be positive")
        from .native_grasp import PlacementPoseSample

        samples: list[PlacementPoseSample] = []
        while True:
            # Detach ACK and the first following Pose_V publication are two
            # independent Gazebo transports.  Reuse the same one-retry bound
            # as attach snapshots so one incomplete fresh frame cannot turn a
            # successful physical release into a false infrastructure error.
            # Every stability sample still comes from native Gazebo state.
            target, _, _ = self.native_target_mount_poses_with_retry(
                max_attempts=2
            )
            sampled_s = time.monotonic()
            samples.append(
                PlacementPoseSample(
                    monotonic_s=sampled_s,
                    xyz=target.xyz,
                    quat_xyzw=target.quat_xyzw,
                )
            )
            elapsed_s = sampled_s - samples[0].monotonic_s
            if elapsed_s >= duration_s:
                return samples
            time.sleep(min(interval_s, max(0.0, duration_s - elapsed_s)))

    def child_link_proof(self):
        """Return native relative-pose retention evidence, or fail closed."""

        if self._baseline is None:
            raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE")
        try:
            from .native_grasp import ChildLinkProof
            child, parent = self.native_target_mount_poses()
            baseline_z, baseline_relative = self._baseline
            relative = _relative_translation(child=child, parent=parent)
            return ChildLinkProof(
                baseline_target_z_m=baseline_z,
                target_z_m=child.xyz[2],
                capture_relative_translation_m=math.dist(baseline_relative, relative),
            )
        except GazeboProcessError:
            raise
        except Exception as exc:
            raise GazeboProcessError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE") from exc


class GazeboNativeContactWindow:
    """Collect raw Gazebo contacts and prove the terminal close hold."""

    def __init__(
        self,
        *,
        gz_executable: str = "gz",
        environment: dict[str, str] | None = None,
        timeout_s: float = 3.0,
        simulation_time_provider: Callable[[], float] | None = None,
    ) -> None:
        self.gz_executable = gz_executable
        self.environment = dict(environment) if environment is not None else None
        self.timeout_s = float(timeout_s)
        self.simulation_time_provider = simulation_time_provider
        self._processes: list[subprocess.Popen] = []
        self._threads: list[threading.Thread] = []
        self._samples: list[object] = []
        self._latest_message_sim_times: dict[str, float] = {}
        self._pending_message_lines: dict[str, list[str]] = {}
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._armed = False

    @staticmethod
    def _message_sim_time(block: str) -> float | None:
        stamp = re.search(r"stamp\s*\{(.*?)\}", block, re.DOTALL)
        if stamp is None:
            return None
        seconds = GazeboDetachableJointControl._field(stamp.group(1), "sec")
        nanoseconds = GazeboDetachableJointControl._field(stamp.group(1), "nanosec", GazeboDetachableJointControl._field(stamp.group(1), "nsec"))
        sim_time_s = seconds + nanoseconds * 1e-9
        return sim_time_s if math.isfinite(sim_time_s) and sim_time_s >= 0.0 else None

    @staticmethod
    def _contact_message(block: str, side: str):
        from .native_grasp import NativeContactSample
        sim_time_s = GazeboNativeContactWindow._message_sim_time(block)
        if sim_time_s is None:
            return None
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
            return NativeContactSample(side, sim_time_s, time.monotonic(), names)
        except ValueError:
            return None

    def _record_message(self, block: str, side: str) -> None:
        sim_time_s = self._message_sim_time(block)
        sample = self._contact_message(block, side)
        if sim_time_s is None and sample is None:
            return
        with self._lock:
            if sim_time_s is not None:
                self._latest_message_sim_times[side] = max(
                    sim_time_s,
                    self._latest_message_sim_times.get(side, -math.inf),
                )
            if sample is not None:
                self._samples.append(sample)

    def _read(self, stream, side: str) -> None:
        current: list[str] = []
        self._pending_message_lines[side] = current
        for line in iter(stream.readline, ""):
            # Every gz.msgs.Contacts begins with its Header.  Flush the prior
            # message at the next Header so bracket depth inside repeated
            # contacts cannot merge simulator samples.
            if line.strip() == "header {" and current:
                self._record_message("".join(current), side)
                current = []
                self._pending_message_lines[side] = current
            current.append(line)
        if current:
            self._record_message("".join(current), side)
        self._pending_message_lines[side] = []

    def arm(self) -> None:
        if self._armed:
            raise GazeboProcessError("NATIVE_GRASP_CONTACT_WINDOW_ALREADY_ARMED")
        executable = GazeboDetachableJointControl._executable(self.gz_executable)
        try:
            for side, topic in (("left", "/openeta/native_grasp/contacts/left_pad"), ("right", "/openeta/native_grasp/contacts/right_pad")):
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
            raise GazeboProcessError("NATIVE_GRASP_NATIVE_CONTACT_UNAVAILABLE") from exc

    def evaluate(self, *, close_completed_sim_time_s: float | None, config=None):
        from .native_grasp import (
            ContactGateResult,
            NativePickPlaceConfig,
            ReasonCode,
            confirm_native_bilateral_contact,
        )
        if not self._armed:
            raise GazeboProcessError("NATIVE_GRASP_CONTACT_WINDOW_NOT_ARMED")
        cfg = config or NativePickPlaceConfig()
        if (
            close_completed_sim_time_s is None
            or not math.isfinite(close_completed_sim_time_s)
        ):
            return confirm_native_bilateral_contact(
                (),
                verification_window_started_sim_time_s=None,
                now_monotonic_s=time.monotonic(),
                config=cfg,
            )
        terminal_window_end = close_completed_sim_time_s
        post_close_hold_end = (
            close_completed_sim_time_s + cfg.contact_post_close_hold_s
        )
        deadline = time.monotonic() + self.timeout_s
        while True:
            now = time.monotonic()
            with self._lock:
                latest_by_side = {
                    side: max(
                        (
                            sample.timestamp_s
                            for sample in self._samples
                            if sample.side == side
                        ),
                        default=-math.inf,
                    )
                    for side in ("left", "right")
                }
                if all(
                    timestamp >= post_close_hold_end
                    for timestamp in latest_by_side.values()
                ):
                    terminal_window_end = post_close_hold_end
                terminal_window_start = (
                    terminal_window_end - cfg.contact_terminal_lookback_s
                )
                # The simulator can stop publishing a static contact as soon
                # as the action result is emitted.  Preserve the samples that
                # already prove a sustained hold immediately before that
                # result.  If both contact streams continue, extend only to
                # the fixed post-close hold boundary.  Reader-thread transport
                # lag is harmless because Gazebo time defines membership.
                samples = tuple(
                    sample
                    for sample in self._samples
                    if terminal_window_start < sample.timestamp_s
                    <= terminal_window_end
                )
            result = confirm_native_bilateral_contact(
                samples,
                verification_window_started_sim_time_s=terminal_window_start,
                verification_window_ended_sim_time_s=terminal_window_end,
                now_monotonic_s=now, config=cfg,
            )
            post_close_hold_completed = terminal_window_end > close_completed_sim_time_s
            evidence = {
                **dict(result.evidence),
                "proof_boundary": (
                    "bounded_post_close_bilateral_hold"
                    if post_close_hold_completed
                    else "terminal_bilateral_hold_before_close_result"
                ),
                "close_completed_sim_time_s": close_completed_sim_time_s,
                "verification_window_ended_sim_time_s": terminal_window_end,
                "terminal_lookback_s": cfg.contact_terminal_lookback_s,
                "post_close_hold_required_s": cfg.contact_post_close_hold_s,
                "post_close_hold_completed": post_close_hold_completed,
            }
            result = ContactGateResult(
                result.accepted,
                result.reason_code,
                result.left_sample_count,
                result.right_sample_count,
                result.left_span_s,
                result.right_span_s,
                evidence,
            )
            if result.accepted or result.reason_code not in {
                ReasonCode.CONTACT_INSUFFICIENT_SAMPLES, ReasonCode.CONTACT_WINDOW_TOO_SHORT,
            }:
                return result
            if now >= deadline:
                return result
            time.sleep(0.02)

    def prove_contact_clearance(
        self,
        *,
        after_sim_time_s: float,
        duration_sim_s: float,
    ) -> dict[str, object]:
        """Prove both previously-live native pads stayed clear after attach.

        Gazebo contact sensors publish contacts, but are silent while clear;
        silence is therefore not an empty-message heartbeat.  The two streams
        must already have produced samples and their subscription processes
        must remain alive.  The authoritative ROS/Gazebo clock then advances
        across the requested window plus an equal transport-drain window.  No
        pad contact may occur anywhere in that closed simulator-time proof.
        """

        if not self._armed:
            raise GazeboProcessError("NATIVE_GRASP_CONTACT_WINDOW_NOT_ARMED")
        started = float(after_sim_time_s)
        duration = float(duration_sim_s)
        if (
            not math.isfinite(started)
            or started < 0.0
            or not math.isfinite(duration)
            or duration <= 0.0
        ):
            raise ValueError("native contact-clearance window is invalid")
        ended = started + duration
        proof_ended = ended + duration
        deadline = time.monotonic() + self.timeout_s
        while True:
            with self._lock:
                pending_blocks = {
                    side: "".join(tuple(self._pending_message_lines.get(side, ())))
                    for side in ("left", "right")
                }
                samples = list(self._samples)
                for side, block in pending_blocks.items():
                    pending_sample = self._contact_message(block, side)
                    if pending_sample is not None:
                        samples.append(pending_sample)
                unique_samples = {
                    (
                        sample.side,
                        sample.timestamp_s,
                        tuple(sample.collision_names),
                    ): sample
                    for sample in samples
                }
                pending_times = {
                    side: self._message_sim_time(block)
                    for side, block in pending_blocks.items()
                }
                latest = {
                    side: max(
                        self._latest_message_sim_times.get(side, -math.inf),
                        (
                            pending_times[side]
                            if pending_times[side] is not None
                            else -math.inf
                        ),
                    )
                    for side in ("left", "right")
                }
                counts = {
                    side: sum(
                        1
                        for sample in unique_samples.values()
                        if sample.side == side
                        and started < sample.timestamp_s <= proof_ended
                    )
                    for side in ("left", "right")
                }
                reader_errors = tuple(self._errors)
            subscriptions_live = len(self._processes) == 2 and all(
                process.poll() is None for process in self._processes
            )
            streams_proven = all(math.isfinite(value) for value in latest.values())
            try:
                clock_now = (
                    float(self.simulation_time_provider())
                    if self.simulation_time_provider is not None
                    else -math.inf
                )
            except Exception:  # noqa: BLE001 - clock/provider boundary.
                clock_now = -math.inf
            if (
                subscriptions_live
                and streams_proven
                and not reader_errors
                and math.isfinite(clock_now)
                and clock_now >= proof_ended
            ):
                return {
                    "schema_version": "openeta.native_pad_clearance.v1",
                    "cleared": not any(counts.values()),
                    "window_started_sim_time_s": started,
                    "window_ended_sim_time_s": ended,
                    "duration_sim_s": duration,
                    "proof_window_ended_sim_time_s": proof_ended,
                    "transport_drain_duration_sim_s": duration,
                    "left_contact_sample_count": counts["left"],
                    "right_contact_sample_count": counts["right"],
                    "latest_live_contact_stream_message_sim_time_s": latest,
                    "proof_clock_sim_time_s": clock_now,
                    "contact_subscriptions_live": True,
                    "time_basis": "gazebo_clock_after_previously_live_contact_streams",
                }
            if time.monotonic() >= deadline:
                raise GazeboProcessError(
                    "NATIVE_GRASP_CONTACT_CLEARANCE_STREAM_TIMEOUT"
                )
            time.sleep(0.02)

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
            # A formal case captures the bench worker stdout/stderr in its
            # case-local worker log.  Inherit those streams so Gazebo/ROS
            # launch diagnostics survive a readiness failure instead of being
            # stranded in an unread private PIPE.
            stdout=None,
            stderr=None,
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
            raise GazeboProcessError("ROS 2 launch exited during startup; inspect worker launch log")
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
