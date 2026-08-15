#!/usr/bin/env python3
"""Formal non-interactive M0/M1 live acceptance segments.

The cloud M0--M4 coordinator calls this script from its clean checkout.  It
uses a fresh ROS domain, Gazebo partition, and localhost MCP port for every
direct/MCP segment and records the same cleanup evidence used by the M2/M3
drivers.  No deterministic oracle transport is used here.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid


try:  # script execution from the checkout
    from extensions.gazebo.ros2_ws.acceptance_isolation import (
        FAILED,
        PASSED,
        aggregate_cleanup,
        candidate_domain_evidence,
        empty_domain_evidence,
        world_partition_evidence,
    )
except ImportError:  # pragma: no cover - package execution compatibility
    from acceptance_isolation import (  # type: ignore[no-redef]
        FAILED, PASSED, aggregate_cleanup, candidate_domain_evidence,
        empty_domain_evidence, world_partition_evidence,
    )


SCHEMA_VERSION = "openeta.cloud_m0_m1_acceptance.v1"
M0_ENV_ID = "openeta/dummy_sim-v0"
M1_ENV_ID = "openeta/gazebo_live_rgbd-v0"
M1_WORLD = "lidar_sensor"
DOMAIN_CANDIDATES = tuple(range(110, 152))
PORT_CANDIDATES = tuple(range(19000, 19100))
LOCK_DIR = Path("/tmp/openeta-acceptance-locks")


class AcceptanceError(RuntimeError):
    pass


@dataclass(slots=True)
class Allocation:
    milestone: str
    segment: str
    domain: int
    port: int
    partition: str
    domain_lock: Any
    port_lock: Any

    def close(self) -> None:
        for lock in (self.domain_lock, self.port_lock):
            with suppress(OSError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            with suppress(OSError):
                lock.close()

    def evidence(self) -> dict[str, Any]:
        return {
            "milestone": self.milestone,
            "segment": self.segment,
            "ros_domain_id": self.domain,
            "mcp_port": self.port,
            "gz_partition": self.partition,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _compact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _compact(item)
            for key, item in value.items()
            if key not in {"rgb_base64", "depth_base64", "rgb", "depth"}
        }
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value]
    return value


def _write_final(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _locked(path: Path) -> Any | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def allocate(milestone: str, segment: str) -> Allocation:
    """Acquire cooperative locks only after live graph / port evidence is clean."""
    domain_lock = None
    domain = -1
    for candidate in DOMAIN_CANDIDATES:
        lock = _locked(LOCK_DIR / f"cloud-ros-domain-{candidate}.lock")
        if lock is None:
            continue
        evidence = candidate_domain_evidence(candidate)
        if evidence.get("state") == PASSED:
            domain, domain_lock = candidate, lock
            break
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    if domain_lock is None:
        raise AcceptanceError("ISOLATION_UNAVAILABLE: no empty ROS domain in 110..151")
    for candidate in PORT_CANDIDATES:
        lock = _locked(LOCK_DIR / f"cloud-mcp-port-{candidate}.lock")
        if lock is None:
            continue
        if _port_is_free(candidate):
            token = uuid.uuid4().hex[:12]
            return Allocation(
                milestone=milestone, segment=segment, domain=domain, port=candidate,
                partition=f"openeta-cloud-{milestone}-{segment}-{token}",
                domain_lock=domain_lock, port_lock=lock,
            )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    fcntl.flock(domain_lock.fileno(), fcntl.LOCK_UN)
    domain_lock.close()
    raise AcceptanceError("ISOLATION_UNAVAILABLE: no free MCP port in 19000..19099")


def _ancestors() -> set[int]:
    result: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            status = (Path("/proc") / str(pid) / "status").read_text().splitlines()
            pid = int(next(row for row in status if row.startswith("PPid:")).split()[1])
        except (OSError, StopIteration, ValueError):
            break
    return result


def _partition_process_groups(partition: str) -> set[int]:
    """Resolve only groups that explicitly inherited this run's partition."""
    own = _ancestors()
    groups: set[int] = set()
    expected = f"GZ_PARTITION={partition}".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            environment = (entry / "environ").read_bytes().split(b"\0")
            stat = (entry / "stat").read_text(encoding="utf-8")
        except (OSError, ValueError):
            continue
        if expected not in environment:
            continue
        try:
            groups.add(int(stat.rsplit(")", 1)[1].split()[2]))
        except (IndexError, ValueError):
            continue
    return groups


def _terminate_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    pgid = os.getpgid(process.pid)
    os.killpg(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        process.wait(timeout=5)


def _terminate_partition_groups(partition: str) -> list[int]:
    """Reap only independent groups carrying this exact unique partition."""
    groups = _partition_process_groups(partition)
    own_group = os.getpgrp()
    targets = sorted(group for group in groups if group != own_group)
    for group in targets:
        with suppress(ProcessLookupError):
            os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and _partition_process_groups(partition):
        time.sleep(0.2)
    for group in _partition_process_groups(partition):
        if group != own_group:
            with suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)
    return targets


def _segment_env(allocation: Allocation) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "ROS_DOMAIN_ID": str(allocation.domain),
        "GZ_PARTITION": allocation.partition,
        "OPENETA_MCP_PORT": str(allocation.port),
        "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
        "ROS2CLI_NO_DAEMON": "1",
        "ROS_HOME": str(Path.cwd() / ".cache" / "ros" / f"{allocation.partition}"),
    })
    for legacy in ("ROS_LOCALHOST_ONLY", "ROS_STATIC_PEERS", "ROS2CLI_DISABLE_DAEMON"):
        env.pop(legacy, None)
    Path(env["ROS_HOME"]).mkdir(parents=True, exist_ok=True)
    return env


def _timestamp(camera: Mapping[str, Any]) -> float:
    value = camera.get("timestamp_s")
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise AcceptanceError(f"CAMERA_TIMESTAMP_INVALID: {camera.get('frame_id')}")
    return float(value)


def _validate_m1_observation(
    observation: Mapping[str, Any], *, previous: Mapping[str, float] | None,
) -> dict[str, float]:
    """Require fresh real RGB-D packets, numeric calibration, and live provenance."""
    raw_cameras = observation.get("cameras")
    cameras: Sequence[Mapping[str, Any]]
    if isinstance(raw_cameras, Mapping):
        cameras = [item for item in raw_cameras.values() if isinstance(item, Mapping)]
    elif isinstance(raw_cameras, list):
        cameras = [item for item in raw_cameras if isinstance(item, Mapping)]
    else:
        cameras = []
    if not cameras:
        raise AcceptanceError("M1_RGBD_MISSING")
    metadata = observation.get("metadata")
    provenance = str((metadata or {}).get("observation_provenance") or "").lower()
    if provenance != "gazebo_ros_live":
        raise AcceptanceError(f"M1_NONLIVE_PROVENANCE: {provenance or 'missing'}")
    result: dict[str, float] = {}
    for camera in cameras:
        frame = str(camera.get("frame_id") or "")
        if not frame:
            raise AcceptanceError("M1_CAMERA_FRAME_ID_MISSING")
        if not (camera.get("rgb") is not None or camera.get("rgb_base64")):
            raise AcceptanceError(f"M1_RGB_MISSING: {frame}")
        if not (camera.get("depth") is not None or camera.get("depth_base64")):
            raise AcceptanceError(f"M1_DEPTH_MISSING: {frame}")
        intrinsics = camera.get("intrinsics")
        if not isinstance(intrinsics, Mapping) or not all(
            isinstance(intrinsics.get(key), (int, float)) and float(intrinsics[key]) > 0
            for key in ("fx", "fy")
        ):
            raise AcceptanceError(f"M1_INTRINSICS_INVALID: {frame}")
        extrinsics = camera.get("extrinsics")
        if not isinstance(extrinsics, Mapping) or extrinsics.get("frame_transform") != "camera_to_world":
            raise AcceptanceError(f"M1_EXTRINSICS_INVALID: {frame}")
        stamp = _timestamp(camera)
        if previous is not None and frame in previous and stamp <= previous[frame]:
            raise AcceptanceError(f"M1_STALE_RGBD: {frame}")
        result[frame] = stamp
    return result


def _wait_for_mcp(transport: Any, process: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AcceptanceError(f"MCP_EXITED_EARLY: {process.returncode}")
        try:
            names = {str(item.get("name")) for item in transport.list_tools(timeout_s=10).get("tools", [])}
            required = {"create_env", "reset_env", "observe_env", "close_env"}
            if required.issubset(names):
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise AcceptanceError("MCP_NOT_READY")


def _start_mcp(python: str, allocation: Allocation, env: Mapping[str, str], log: Path) -> subprocess.Popen[Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [python, "-m", "sim.mcp_server", "--port", str(allocation.port)],
            stdout=stream, stderr=subprocess.STDOUT, cwd=Path.cwd(), env=dict(env),
            start_new_session=True, text=True,
        )
    finally:
        stream.close()
    return process


def _cleanup(allocation: Allocation, process: subprocess.Popen[Any] | None, *, world: str | None) -> dict[str, Any]:
    if process is not None:
        with suppress(Exception):
            _terminate_group(process)
    reaped = _terminate_partition_groups(allocation.partition)
    checks: dict[str, dict[str, Any]] = {
        "ros_graph": empty_domain_evidence(allocation.domain),
        "mcp_port": {
            "state": PASSED if _port_is_free(allocation.port) else FAILED,
            "ok": _port_is_free(allocation.port),
        },
    }
    if world:
        environment = dict(os.environ)
        environment["GZ_PARTITION"] = allocation.partition
        checks["gz_partition"] = world_partition_evidence(world, environment=environment)
    return {
        "status": aggregate_cleanup(checks), "checks": checks,
        "reaped_partition_process_groups": reaped,
    }


def _m0_mcp(python: str, artifact_dir: Path) -> dict[str, Any]:
    allocation = allocate("m0", "mcp")
    process = None
    record: dict[str, Any] = {"allocation": allocation.evidence(), "cycles": []}
    try:
        env = _segment_env(allocation)
        process = _start_mcp(python, allocation, env, artifact_dir / "mcp.log")
        from agent.tools.sim_mcp import SseSimulatorMcpTransport

        transport = SseSimulatorMcpTransport(f"http://127.0.0.1:{allocation.port}/sse")
        _wait_for_mcp(transport, process)
        for seed in (101, 102, 103):
            handle = session_id = ""
            close_calls: list[dict[str, Any]] = []
            try:
                created = transport.call_tool("create_env", {"env_id": M0_ENV_ID, "seed": seed, "task": "M0 formal lifecycle"}, timeout_s=120)
                if created.get("error"):
                    raise AcceptanceError(f"M0_CREATE_FAILED: {created['error']}")
                handle, session_id = str(created.get("handle") or ""), str(created.get("session_id") or "")
                if not handle or created.get("env_id") != M0_ENV_ID:
                    raise AcceptanceError("M0_CREATE_RECEIPT_INVALID")
                common = {"handle": handle, "session_id": session_id}
                reset = transport.call_tool("reset_env", {**common, "seed": seed}, timeout_s=120)
                observed = transport.call_tool("observe_env", common, timeout_s=60)
                first = transport.call_tool("close_env", common, timeout_s=60)
                close_calls.append(first)
                second = transport.call_tool("close_env", common, timeout_s=60)
                close_calls.append(second)
                if first.get("already_closed") is not False or second.get("already_closed") is not True:
                    raise AcceptanceError("M0_CLOSE_NOT_IDEMPOTENT")
                record["cycles"].append({
                    "seed": seed, "create": _compact(created), "reset": _compact(reset),
                    "observe": _compact(observed), "close": _compact(close_calls),
                })
                handle = ""
            finally:
                if handle:
                    with suppress(Exception):
                        transport.call_tool("close_env", {"handle": handle, "session_id": session_id}, timeout_s=60)
        record["status"] = "passed"
    finally:
        record["cleanup"] = _cleanup(allocation, process, world=None)
        allocation.close()
    if record["cleanup"]["status"] != "passed":
        raise AcceptanceError("M0_CLEANUP_FAILED")
    return record


def _m1_direct_child(report_path: Path) -> int:
    """Run DirectEnv after the OS-level segment environment is established.

    rclpy and Gazebo's native libraries read process environment during their
    first initialization.  Mutating ``os.environ`` inside an already-running
    acceptance interpreter is not equivalent to starting a child with that
    environment, so this helper is intentionally a separate CLI mode.
    """
    record: dict[str, Any] = {"observations": []}
    environment = None
    try:
        from extensions.gazebo.direct_env import GazeboDirectEnv

        environment = GazeboDirectEnv(profile="m1", task="M1 formal RGB-D", seed=201)
        environment.reset(seed=201)
        previous: dict[str, float] | None = None
        for index in range(3):
            # DirectEnv has no read-only step action.  Ask its runtime for a
            # packet received after this monotonic barrier instead.
            source = environment.runtime.observe(min_received_monotonic_s=time.monotonic())
            observation = environment._decorate_robot(environment._as_unified(source))
            timestamps = _validate_m1_observation(observation, previous=previous)
            previous = timestamps
            record["observations"].append({
                "sample": index + 1, "timestamps": timestamps,
                "observation": _compact(observation),
            })
        record["status"] = "passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception as exc:
                record["status"] = "failed"
                record.setdefault("error", f"RUNTIME_CLOSE_FAILED: {type(exc).__name__}: {exc}")
        _write_final(report_path, record)
    return 0 if record["status"] == "passed" else 1


def _m1_direct(python: str, artifact_dir: Path) -> dict[str, Any]:
    allocation = allocate("m1", "direct")
    record: dict[str, Any] = {"allocation": allocation.evidence(), "observations": []}
    child_report = artifact_dir / "direct.json"
    child_log = artifact_dir / "direct.log"
    child_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [python, str(Path(__file__).resolve()), "m1-direct", "--report", str(child_report),
             "--artifact-dir", str(artifact_dir)],
            cwd=Path.cwd(), env=_segment_env(allocation), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            timeout=180, start_new_session=True,
        )
        child_log.write_text(completed.stdout or "", encoding="utf-8")
        try:
            child = json.loads(child_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record["status"] = "failed"
            record["error"] = f"M1_DIRECT_REPORT_INVALID: {type(exc).__name__}"
        else:
            record["child"] = child
            record["observations"] = list(child.get("observations", []))
            if completed.returncode == 0 and child.get("status") == "passed":
                record["status"] = "passed"
            else:
                record["status"] = "failed"
                record["error"] = str(child.get("error") or f"M1_DIRECT_EXIT_{completed.returncode}")
    except subprocess.TimeoutExpired:
        record["status"] = "failed"
        record["error"] = "M1_DIRECT_TIMEOUT"
    finally:
        # Let the child finish its rclpy teardown before proving the selected
        # ROS domain and Gazebo partition are empty.
        time.sleep(0.5)
        record["cleanup"] = _cleanup(allocation, None, world=M1_WORLD)
        allocation.close()
    if record["cleanup"]["status"] != "passed":
        record["status"] = "failed"
        record.setdefault("error", "M1_DIRECT_CLEANUP_FAILED")
    return record


def _m1_mcp(python: str, artifact_dir: Path) -> dict[str, Any]:
    allocation = allocate("m1", "mcp")
    process = None
    record: dict[str, Any] = {"allocation": allocation.evidence(), "observations": []}
    handle = session_id = ""
    try:
        env = _segment_env(allocation)
        process = _start_mcp(python, allocation, env, artifact_dir / "mcp.log")
        from agent.tools.sim_mcp import SseSimulatorMcpTransport

        transport = SseSimulatorMcpTransport(f"http://127.0.0.1:{allocation.port}/sse")
        _wait_for_mcp(transport, process)
        created = transport.call_tool("create_env", {"env_id": M1_ENV_ID, "seed": 211, "task": "M1 formal RGB-D"}, timeout_s=180)
        if created.get("error"):
            raise AcceptanceError(f"M1_MCP_CREATE_FAILED: {created['error']}")
        handle, session_id = str(created.get("handle") or ""), str(created.get("session_id") or "")
        if not handle or created.get("env_id") != M1_ENV_ID or created.get("backend") != "gazebo":
            raise AcceptanceError("M1_MCP_CREATE_RECEIPT_INVALID")
        common = {"handle": handle, "session_id": session_id}
        reset = transport.call_tool("reset_env", {**common, "seed": 211}, timeout_s=180)
        previous = _validate_m1_observation(reset, previous=None)
        record["reset"] = _compact(reset)
        for index in range(3):
            observation = transport.call_tool("observe_env", common, timeout_s=60)
            timestamps = _validate_m1_observation(observation, previous=previous)
            previous = timestamps
            record["observations"].append({"sample": index + 1, "timestamps": timestamps, "observation": _compact(observation)})
        first = transport.call_tool("close_env", common, timeout_s=60)
        second = transport.call_tool("close_env", common, timeout_s=60)
        if first.get("already_closed") is not False or second.get("already_closed") is not True:
            raise AcceptanceError("M1_CLOSE_NOT_IDEMPOTENT")
        record["close"] = _compact([first, second])
        handle = ""
        record["status"] = "passed"
    except Exception as exc:
        # See _m1_direct: a failed MCP segment still has valuable cleanup
        # receipts and must not disappear behind the top-level error.
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if handle:
            with suppress(Exception):
                transport.call_tool("close_env", {"handle": handle, "session_id": session_id}, timeout_s=60)  # type: ignore[possibly-undefined]
        record["cleanup"] = _cleanup(allocation, process, world=M1_WORLD)
        allocation.close()
    if record["cleanup"]["status"] != "passed":
        record["status"] = "failed"
        record.setdefault("error", "M1_MCP_CLEANUP_FAILED")
    return record


def run_milestone(milestone: str, *, python: str, artifact_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "milestone": milestone,
        "started_at_utc": _now(), "segments": {},
    }
    try:
        if milestone == "m0":
            report["segments"]["mcp"] = _m0_mcp(python, artifact_dir / "mcp")
        else:
            report["segments"]["direct"] = _m1_direct(python, artifact_dir / "direct")
            if report["segments"]["direct"].get("status") != "passed":
                raise AcceptanceError(str(report["segments"]["direct"].get("error") or "M1_DIRECT_FAILED"))
            report["segments"]["mcp"] = _m1_mcp(python, artifact_dir / "mcp")
            if report["segments"]["mcp"].get("status") != "passed":
                raise AcceptanceError(str(report["segments"]["mcp"].get("error") or "M1_MCP_FAILED"))
        report["status"] = "passed"
    except AcceptanceError as exc:
        text = str(exc)
        report["status"] = "blocked" if any(token in text for token in ("ISOLATION_", "MCP_NOT_READY", "ROS_", "GAZEBO_")) else "failed"
        report["error"] = text
    finally:
        report["finished_at_utc"] = _now()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("milestone", choices=("m0", "m1", "m1-direct"))
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.milestone == "m1-direct":
        return _m1_direct_child(args.report)
    try:
        report = run_milestone(args.milestone, python=sys.executable, artifact_dir=args.artifact_dir)
    except Exception as exc:
        # Keep failed/blocked reports immutable and useful even if the segment
        # failed before it could add a detailed result.
        report = {
            "schema_version": SCHEMA_VERSION, "milestone": args.milestone,
            "started_at_utc": _now(), "finished_at_utc": _now(),
            "status": "blocked" if "ISOLATION_" in str(exc) or "MCP_NOT_READY" in str(exc) else "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_final(args.report, report)
        print(report["error"], file=sys.stderr)
        return 2 if report["status"] == "blocked" else 1
    _write_final(args.report, report)
    if report["status"] == "passed":
        return 0
    return 2 if report["status"] == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
