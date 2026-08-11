#!/usr/bin/env python3
"""Deterministic Direct / SSE acceptance driver for M3 physical pick-place."""

from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

try:  # package import for tests; script import for the live runner
    from .acceptance_isolation import (
        FAILED, INCONCLUSIVE, PASSED, aggregate_cleanup, candidate_domain_evidence, empty_domain_evidence,
        node_multiset, probe_ros_graph, world_partition_evidence,
    )
except ImportError:
    from acceptance_isolation import (
        FAILED, INCONCLUSIVE, PASSED, aggregate_cleanup, candidate_domain_evidence, empty_domain_evidence,
        node_multiset, probe_ros_graph, world_partition_evidence,
    )


ROOT = Path(__file__).resolve().parents[3]
REPORT_VERSION = "openeta.m3_physical_acceptance.v2"
ENV_ID = "openeta/gazebo_rm75_robotiq2f85_pickplace-v0"
MODEL_ID = "rm75_robotiq_2f85_pickplace_sim_v1"
WORLD_NAME = "m3_rm75_robotiq2f85_pickplace"
POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = 0.08
DOCUMENTATION = {
    "gazebo_odometry_publisher": "https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1OdometryPublisher.html",
    "ros_gz_bridge_mappings": "https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_bridge/README.md",
    "moveit_planning_scene": "https://moveit.picknik.ai/main/doc/tutorials/planning_around_objects/planning_around_objects.html",
    "ros2_control_gripper_stall": "https://control.ros.org/jazzy/doc/ros2_controllers/gripper_controllers/doc/userdoc.html",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _assert_report_mutable(path: Path) -> None:
    if _load(path).get("finished_at_utc"):
        raise RuntimeError("REPORT_ALREADY_FINALIZED")


def _compact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _compact(item)
            for key, item in value.items()
            if key not in {"rgb_base64", "depth_base64"}
            and not (
                key in {"rgb", "depth"}
                and not isinstance(item, (int, float, str, bool, type(None)))
            )
        }
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value]
    return value


def _versions() -> dict[str, str]:
    packages = {
        "ros_gz": "ros-jazzy-ros-gz",
        "moveit": "ros-jazzy-moveit",
        "ros2_control": "ros-jazzy-ros2-control",
        "ros2_controllers": "ros-jazzy-ros2-controllers",
    }
    result: dict[str, str] = {}
    for label, package in packages.items():
        completed = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", package],
            capture_output=True,
            text=True,
            check=False,
        )
        result[label] = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    gazebo = subprocess.run(
        ["gz", "sim", "--force-version", "8", "--versions"],
        capture_output=True,
        text=True,
        check=False,
    )
    result["gazebo_sim"] = gazebo.stdout.strip() if gazebo.returncode == 0 else "unavailable"
    return result


def _base(path: Path) -> dict[str, Any]:
    report = _load(path)
    report.setdefault("schema_version", REPORT_VERSION)
    report.setdefault("started_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    report.setdefault("gates", {})
    report["documentation"] = DOCUMENTATION
    report["installed_versions"] = _versions()
    report["attachment_mode"] = os.environ.get("OPENETA_M3_ATTACHMENT_MODE", "physics")
    with suppress(Exception):
        report["git_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    return report


def _process_row(pid: int) -> dict[str, Any] | None:
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        pgid = int(stat.rsplit(")", 1)[1].split()[2])
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(
            b"\0", b" "
        ).decode(errors="replace")
        start_ticks = int(stat.rsplit(")", 1)[1].split()[19])
        return {"pid": pid, "pgid": pgid, "start_ticks": start_ticks, "command": command[:500]}
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None


def _ancestors() -> set[int]:
    result: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            status = (Path("/proc") / str(pid) / "status").read_text().splitlines()
            pid = int(next(line for line in status if line.startswith("PPid:")).split()[1])
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            break
    return result


def _ancestor_process_groups() -> set[int]:
    """Return process groups that the acceptance cleanup must never signal."""
    groups: set[int] = set()
    for pid in _ancestors():
        row = _process_row(pid)
        if row is not None:
            groups.add(int(row["pgid"]))
    return groups


def _isolated_processes(partition: str) -> list[dict[str, Any]]:
    expected = f"GZ_PARTITION={partition}".encode()
    excluded = _ancestors()
    excluded_groups = _ancestor_process_groups()
    rows: list[dict[str, Any]] = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) in excluded:
            continue
        try:
            environment = (item / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        row = _process_row(int(item.name)) if expected in environment else None
        # A short-lived helper may inherit both the partition and the invoking
        # shell's process group.  Killing that group would terminate the
        # acceptance harness before it can finalize the report and unlock its
        # resources.  Only independently managed groups are cleanup targets.
        if row is not None and int(row["pgid"]) not in excluded_groups:
            rows.append(row)
    return rows


def _preexisting_processes() -> list[dict[str, Any]]:
    tokens = ("gz sim", "sim.mcp_server", "bench_worker.py --bench gazebo", "move_group")
    rows: list[dict[str, Any]] = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        row = _process_row(int(item.name))
        if row is not None and any(token in row["command"] for token in tokens):
            rows.append(row)
    return rows


def _process_still_matches(item: Mapping[str, Any]) -> bool:
    current = _process_row(int(item["pid"]))
    return bool(current and current.get("start_ticks") == item.get("start_ticks") and current.get("command") == item.get("command"))


def _port_free(port: int) -> bool:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def init_report(path: Path, domain: int, original_domain: int, partition: str, port: int, world: str) -> None:
    _assert_report_mutable(path)
    if not world:
        raise ValueError("WORLD_REQUIRED")
    report = _base(path)
    selections = []
    selection_log = os.environ.get("OPENETA_ISOLATION_SELECTION_LOG", "")
    if selection_log:
        with suppress(OSError, json.JSONDecodeError):
            selections = [json.loads(line) for line in Path(selection_log).read_text(encoding="utf-8").splitlines() if line]
    report["isolation"] = {
        "ros_domain_id": domain,
        "original_ros_domain_id": original_domain,
        "gz_partition": partition,
        "mcp_port": port,
        "world": world,
        "preexisting_processes": _preexisting_processes(),
        "preexisting_default_domain_graph": probe_ros_graph(original_domain),
        "isolation_evidence_version": "openeta.acceptance_isolation.v2",
        "domain_selection": selections,
    }
    report["gates"]["isolation_cleanup"] = {"status": "running"}
    _write(path, report)


def finalize_report(path: Path, domain: int, partition: str, port: int, world: str, exit_code: int) -> bool:
    report = _base(path)
    if report.get("finished_at_utc"):
        raise RuntimeError("REPORT_ALREADY_FINALIZED")
    if not world:
        raise ValueError("WORLD_REQUIRED")
    isolation = report.get("isolation", {})
    if (
        isolation.get("ros_domain_id"), isolation.get("gz_partition"),
        isolation.get("mcp_port"), isolation.get("world"),
    ) != (domain, partition, port, world):
        raise ValueError("FINALIZE_ARGUMENT_MISMATCH")
    # DDS discovery can retain departed participants beyond process teardown.
    # Reuse the existing two-snapshot empty-domain probe for up to one lease
    # interval before classifying cleanup as a confirmed residual.  Gazebo
    # transport discovery lags the same way on WSL2, so the partition probe
    # gets an identical bounded retry instead of a single sample.
    deadline = time.monotonic() + 60.0
    domain_evidence: dict[str, Any] = {"state": INCONCLUSIVE, "ok": None, "reason_code": "ROS_GRAPH_UNAVAILABLE"}
    while time.monotonic() < deadline:
        domain_evidence = empty_domain_evidence(domain)
        if domain_evidence["state"] != FAILED:
            break
        time.sleep(0.5)
    partition_evidence: dict[str, Any] = {"state": INCONCLUSIVE, "ok": None, "reason_code": "GZ_GRAPH_UNAVAILABLE"}
    while time.monotonic() < deadline:
        partition_evidence = world_partition_evidence(world)
        if partition_evidence["state"] != FAILED:
            break
        time.sleep(0.5)
    isolated = _isolated_processes(partition)
    port_free = _port_free(port)
    checks = {
        "isolated_processes_gone": {"state": PASSED if not isolated else FAILED, "ok": not bool(isolated), "reason_code": "PROCESSES_GONE" if not isolated else "ISOLATED_PROCESSES_REMAIN", "residual": isolated},
        "test_domain_empty": domain_evidence,
        "test_partition_empty": partition_evidence,
        "mcp_port_rebind": {"state": PASSED if port_free else FAILED, "ok": port_free, "reason_code": "MCP_PORT_REBIND" if port_free else "MCP_PORT_STILL_BOUND"},
    }
    preexisting = report.get("isolation", {}).get("preexisting_processes", [])
    vanished = [item for item in preexisting if not _process_still_matches(item)]
    checks["preexisting_processes_alive"] = {"state": PASSED if not vanished else FAILED, "ok": not bool(vanished), "reason_code": "PREEXISTING_PROCESSES_ALIVE" if not vanished else "PREEXISTING_PROCESS_CHANGED", "vanished": vanished}
    original_domain = int(report.get("isolation", {}).get("original_ros_domain_id", domain))
    before = isolation.get("preexisting_default_domain_graph", {})
    after = probe_ros_graph(original_domain)
    if before.get("availability") != "AVAILABLE" or after.get("availability") != "AVAILABLE":
        checks["preexisting_default_domain_healthy"] = {"state": INCONCLUSIVE, "ok": None, "reason_code": "DEFAULT_GRAPH_UNAVAILABLE", "before": before, "after": after}
    else:
        missing = node_multiset(before) - node_multiset(after)
        checks["preexisting_default_domain_healthy"] = {"state": PASSED if not missing else FAILED, "ok": not bool(missing), "reason_code": "DEFAULT_GRAPH_HEALTHY" if not missing else "DEFAULT_GRAPH_NODES_MISSING", "missing": [{"namespace": key[0], "name": key[1], "count": value} for key, value in sorted(missing.items())]}
    cleanup_status = aggregate_cleanup(checks)
    report["gates"]["isolation_cleanup"] = {
        "status": cleanup_status,
        "checks": checks,
        "main_exit_code": exit_code,
    }
    report["overall_status"] = (
        "passed"
        if exit_code == 0 and cleanup_status == "passed"
        else "inconclusive" if exit_code == 0 and cleanup_status == "inconclusive"
        else "blocked" if report.get("gates", {}).get("direct_live", {}).get("status") == "blocked"
        else "failed"
    )
    report["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(path, report)
    return cleanup_status == "passed"


def _q_multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _q_axis(axis: Sequence[float], angle: float) -> tuple[float, float, float, float]:
    scale = math.sin(angle / 2.0)
    return axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(angle / 2.0)


def _q_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    return _q_multiply(
        _q_axis((0, 0, 1), yaw),
        _q_multiply(_q_axis((0, 1, 0), pitch), _q_axis((1, 0, 0), roll)),
    )


def _grasp_orientation(tilt_deg: float, azimuth_deg: float) -> tuple[float, float, float, float]:
    """Top-grasp orientation with a horizontal gripper closing axis.

    The plain ``_q_euler(pi, pitch, yaw)`` family tilts the closing axis by
    the same pitch: live probes showed the lower fingertip then sweeps through
    a 4 cm table-top box during the descent and topples it before closing.
    Keeping the closing axis horizontal (fingers pinch the box faces at equal
    height) was measured to leave the box undisturbed at contact.
    """

    tilt = math.radians(tilt_deg)
    azimuth = math.radians(azimuth_deg)
    # Gripper local axes: +z is the finger (approach) direction, x the closing
    # direction.  Map +z onto the tilted approach axis and x onto the
    # horizontal perpendicular.
    approach = (
        math.sin(tilt) * math.cos(azimuth),
        math.sin(tilt) * math.sin(azimuth),
        -math.cos(tilt),
    )
    closing = (-math.sin(azimuth), math.cos(azimuth), 0.0)
    third = (
        approach[1] * closing[2] - approach[2] * closing[1],
        approach[2] * closing[0] - approach[0] * closing[2],
        approach[0] * closing[1] - approach[1] * closing[0],
    )
    # Rotation matrix with columns [closing, third, approach].
    m00, m01, m02 = closing[0], third[0], approach[0]
    m10, m11, m12 = closing[1], third[1], approach[1]
    m20, m21, m22 = closing[2], third[2], approach[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)


def _q_rotate(q: Sequence[float], value: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = q
    vx, vy, vz = value
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def _stl_center(path: Path) -> tuple[float, float, float]:
    data = path.read_bytes()
    triangles = struct.unpack_from("<I", data, 80)[0]
    _assert(len(data) == 84 + triangles * 50, f"invalid frozen STL: {path}")
    vertices = [
        struct.unpack_from("<fff", data, 84 + triangle * 50 + 12 + vertex * 12)
        for triangle in range(triangles)
        for vertex in range(3)
    ]
    return tuple(
        (min(item[index] for item in vertices) + max(item[index] for item in vertices)) / 2
        for index in range(3)
    )


def _grasp_center_offset(environment: Any) -> tuple[float, float, float]:
    from rclpy.time import Time

    # Measure at the aperture the pads will actually have when they close onto
    # the 4 cm target: the four-bar linkage carries the fingertip collision
    # centres ~1 cm forward as the gripper closes, so an open-aperture
    # measurement systematically misses the box on one side.
    runtime = environment.controller.runtime
    mid = runtime.gripper(0.41, 15.0)
    _assert(mid.get("ok") is True, f"mid-aperture grip for offset measurement failed: {mid}")
    buffer = runtime.state_source.tf_buffer
    asset = ROOT / "extensions/gazebo/assets/robotiq_2f85_vendor/meshes/collision/2f_85"
    centers = []
    for side, link in zip(("left", "right"), environment._m3_config.fingertip_links):
        transform = buffer.lookup_transform("gripper_mount_link", link, Time()).transform
        translation = (transform.translation.x, transform.translation.y, transform.translation.z)
        rotation = (
            transform.rotation.x, transform.rotation.y,
            transform.rotation.z, transform.rotation.w,
        )
        local = _stl_center(asset / f"{side}_finger_tip.stl")
        rotated = _q_rotate(rotation, local)
        centers.append(tuple(translation[index] + rotated[index] for index in range(3)))
    runtime.gripper(0.0, 15.0)
    return tuple(sum(item[index] for item in centers) / 2 for index in range(3))

def _mount_pose(
    grasp_center: Sequence[float],
    orientation: Sequence[float],
    offset: Sequence[float],
) -> dict[str, list[float]]:
    rotated = _q_rotate(orientation, offset)
    return {
        "xyz": [float(grasp_center[index] - rotated[index]) for index in range(3)],
        "quat_xyzw": [float(item) for item in orientation],
    }


def _target(observation: Mapping[str, Any], object_id: str = "m3_target") -> Mapping[str, Any]:
    return next(item for item in observation["objects"] if item["id"] == object_id)


def _orientation_error(a: Iterable[float], b: Iterable[float]) -> float:
    qa, qb = list(map(float, a)), list(map(float, b))
    dot = abs(sum(x * y for x, y in zip(qa, qb)))
    norms = math.sqrt(sum(x * x for x in qa) * sum(x * x for x in qb))
    return 2 * math.acos(max(-1.0, min(1.0, dot / norms)))


def _validate_move(observation: Mapping[str, Any], target_pose: Mapping[str, Any]) -> dict[str, float]:
    actual = observation["robot"]["end_effector_pose"]
    position_error = math.dist(actual["xyz"], target_pose["xyz"])
    orientation_error = _orientation_error(actual["quat_xyzw"], target_pose["quat_xyzw"])
    _assert(position_error <= POSITION_TOLERANCE_M, f"position error {position_error:.6f} m")
    _assert(orientation_error <= ORIENTATION_TOLERANCE_RAD, f"orientation error {orientation_error:.6f} rad")
    return {"position_error_m": position_error, "orientation_error_rad": orientation_error}


def _physical(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    record = observation.get("metadata", {}).get("physical_verification")
    _assert(isinstance(record, Mapping), "physical verification record missing")
    _assert(record.get("schema_version") == "m3_physical_verification_v1", "schema mismatch")
    return record


def _step(environment: Any, action: Mapping[str, Any], gate: dict[str, Any]) -> Mapping[str, Any]:
    observation, _, _, _, info = environment.step(dict(action))
    receipt = info.get("_openeta_receipt", info)
    row = {"action": dict(action), "receipt": _compact(receipt)}
    if action.get("action_type") == "move_to" and receipt.get("ok"):
        row["pose_error"] = _validate_move(observation, action["target_pose"])
    gate["actions"].append(row)
    _assert(receipt.get("observation") is observation, "receipt does not reference fresh observation")
    return observation


def _joint_inventory() -> dict[str, Any]:
    completed = subprocess.run(
        ["gz", "model", "-m", MODEL_ID, "-j"], capture_output=True,
        text=True, timeout=10.0, check=False,
    )
    _assert(completed.returncode == 0, f"Gazebo joint inventory failed: {completed.stderr}")
    output = completed.stdout
    names = sorted(set(re.findall(r"(?m)^\s*- Name:\s*(\S+)\s*$", output)))
    _assert(bool(names), f"Gazebo joint inventory contained no joint identities: {output}")
    return {
        "sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "joint_names": names,
    }


def _select_candidate(environment: Any, observation: Mapping[str, Any], offset: Sequence[float], gate: dict[str, Any]) -> dict[str, Any]:
    del observation
    # Fixed (tilt, azimuth) candidates with a horizontal closing axis; live
    # sweeps found the arm reaches only a narrow tilt band near 60-75 degrees
    # around azimuth 0 for this table layout.
    candidates = ((65, 0), (70, 0), (75, 0), (60, 15))
    for pitch_degrees, yaw_degrees in candidates:
        observation, _ = environment.reset()
        target = _target(observation)["position"]
        orientation = _grasp_orientation(float(pitch_degrees), float(yaw_degrees))
        pregrasp = _mount_pose(
            (target[0], target[1], target[2] + 0.080), orientation, offset
        )
        # OMPL goal sampling is stochastic: the same pose can fail one call
        # and succeed on the next.  Probe the pregrasp with bounded retries
        # before declaring the candidate unreachable.
        result: Mapping[str, Any] = {"ok": False}
        plan_attempts = 0
        for plan_attempts in range(1, 4):
            try:
                result = environment.controller.plan_pose(pregrasp, timeout_s=12.0)
            except Exception as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if result.get("ok") is True:
                break
        row = {
            "pitch_degrees": pitch_degrees,
            "yaw_degrees": yaw_degrees,
            "pregrasp": pregrasp,
            "result": _compact(result),
            "plan_attempts": plan_attempts,
            "stages": [],
        }
        gate["plan_only_candidates"].append(row)
        if result.get("ok") is not True:
            row["blocker_stage"] = "pregrasp_plan"
            continue
        try:
            observation = _step(
                environment,
                {"action_type": "move_to", "target_pose": pregrasp, "timeout_s": 60.0},
                gate,
            )
        except Exception as exc:
            row["blocker_stage"] = "pregrasp_execute"
            row["error"] = f"{type(exc).__name__}: {exc}"
            continue
        row["stages"].append("pregrasp")
        if gate["actions"][-1]["receipt"].get("ok") is not True:
            row["blocker_stage"] = "pregrasp_execute"
            continue
        target = _target(observation)["position"]
        contact_pose = _mount_pose(target, orientation, offset)
        try:
            observation = _step(
                environment,
                {"action_type": "move_to", "target_pose": contact_pose, "timeout_s": 60.0},
                gate,
            )
        except Exception as exc:
            row["blocker_stage"] = "contact_execute"
            row["error"] = f"{type(exc).__name__}: {exc}"
            continue
        row["contact_pose"] = contact_pose
        row["stages"].append("contact")
        if gate["actions"][-1]["receipt"].get("ok") is not True:
            row["blocker_stage"] = "contact_execute"
            continue
        try:
            observation = _step(
                environment, {"action_type": "gripper_close", "timeout_s": 30.0}, gate
            )
        except Exception as exc:
            row["blocker_stage"] = "close_stall"
            row["error"] = f"{type(exc).__name__}: {exc}"
            continue
        close_reason = _physical(observation).get("reason_code")
        row["close_reason_code"] = close_reason
        row["stages"].append("close")
        if close_reason != "LIFT_REQUIRED":
            row["blocker_stage"] = "close_stall"
            continue
        target = _target(observation)["position"]
        lift = _mount_pose(
            (target[0], target[1], target[2] + 0.080), orientation, offset
        )
        try:
            observation = _step(
                environment,
                {"action_type": "move_to", "target_pose": lift, "timeout_s": 60.0},
                gate,
            )
        except Exception as exc:
            row["blocker_stage"] = "lift_comovement"
            row["error"] = f"{type(exc).__name__}: {exc}"
            continue
        lift_reason = _physical(observation).get("reason_code")
        row["lift_pose"] = lift
        row["lift_reason_code"] = lift_reason
        row["stages"].append("lift")
        if lift_reason == "TARGET_HELD":
            row["status"] = "passed"
            return {**row, "orientation": list(orientation)}
        row["blocker_stage"] = "lift_comovement"
    blockers = [
        {
            "pitch_degrees": row["pitch_degrees"],
            "yaw_degrees": row["yaw_degrees"],
            "blocker_stage": row.get("blocker_stage"),
            "close_reason_code": row.get("close_reason_code"),
            "lift_reason_code": row.get("lift_reason_code"),
        }
        for row in gate["plan_only_candidates"]
    ]
    raise AssertionError(f"no complete M3 grasp candidate: {blockers}")


def _positive_round(environment: Any, gate: dict[str, Any], candidate: Mapping[str, Any], offset: Sequence[float]) -> None:
    observation, _ = environment.reset()
    orientation = candidate["orientation"]
    target = _target(observation)["position"]
    pregrasp = _mount_pose((target[0], target[1], target[2] + 0.080), orientation, offset)
    observation = _step(environment, {"action_type": "move_to", "target_pose": pregrasp, "timeout_s": 60.0}, gate)
    _assert(gate["actions"][-1]["receipt"].get("ok") is True, "pregrasp motion failed")
    target = _target(observation)["position"]
    contact = _mount_pose(target, orientation, offset)
    observation = _step(environment, {"action_type": "move_to", "target_pose": contact, "timeout_s": 60.0}, gate)
    _assert(gate["actions"][-1]["receipt"].get("ok") is True, "contact motion failed")
    observation = _step(environment, {"action_type": "gripper_close", "timeout_s": 30.0}, gate)
    _assert(_physical(observation).get("reason_code") == "LIFT_REQUIRED", "close did not produce a bilateral stall candidate")
    target = _target(observation)["position"]
    lift = _mount_pose((target[0], target[1], target[2] + 0.080), orientation, offset)
    observation = _step(environment, {"action_type": "move_to", "target_pose": lift, "timeout_s": 60.0}, gate)
    _assert(_physical(observation).get("reason_code") == "TARGET_HELD", "80 mm lift did not prove target held")
    destination = environment._m3_config.destination_center_xy
    transport = _mount_pose((destination[0], destination[1], target[2] + 0.080), orientation, offset)
    observation = _step(environment, {"action_type": "move_to", "target_pose": transport, "timeout_s": 60.0}, gate)
    _assert(_physical(observation).get("reason_code") == "TARGET_HELD", "transport lost target")
    lower = _mount_pose(
        (destination[0], destination[1], environment._m3_config.table_top_z_m + environment._m3_config.target_size_m[2] / 2),
        orientation,
        offset,
    )
    observation = _step(environment, {"action_type": "move_to", "target_pose": lower, "timeout_s": 60.0}, gate)
    observation = _step(environment, {"action_type": "gripper_open", "timeout_s": 30.0}, gate)
    _assert(_physical(observation).get("reason_code") == "TARGET_PLACED", "release did not prove target placed")


def _approach_and_close(
    environment: Any,
    gate: dict[str, Any],
    candidate: Mapping[str, Any],
    offset: Sequence[float],
    grasp_center: Sequence[float],
) -> Mapping[str, Any]:
    orientation = candidate["orientation"]
    pregrasp = _mount_pose(
        (grasp_center[0], grasp_center[1], grasp_center[2] + 0.080),
        orientation,
        offset,
    )
    observation = _step(
        environment,
        {"action_type": "move_to", "target_pose": pregrasp, "timeout_s": 60.0},
        gate,
    )
    _assert(gate["actions"][-1]["receipt"].get("ok") is True, "negative pregrasp failed")
    contact = _mount_pose(grasp_center, orientation, offset)
    observation = _step(
        environment,
        {"action_type": "move_to", "target_pose": contact, "timeout_s": 60.0},
        gate,
    )
    _assert(gate["actions"][-1]["receipt"].get("ok") is True, "negative contact motion failed")
    return _step(
        environment,
        {"action_type": "gripper_close", "timeout_s": 30.0},
        gate,
    )


def _negative_cases(
    environment: Any,
    gate: dict[str, Any],
    candidate: Mapping[str, Any],
    offset: Sequence[float],
) -> None:
    cases: list[dict[str, Any]] = []

    environment.reset(seed=51)
    empty_center = (0.48, 0.12, environment._m3_config.table_top_z_m + 0.030)
    observation = _approach_and_close(environment, gate, candidate, offset, empty_center)
    reason = _physical(observation).get("reason_code")
    cases.append({"name": "empty_grasp", "reason_code": reason})
    _assert(reason == "EMPTY_GRASP", f"empty grasp returned {reason}")

    observation, _ = environment.reset(seed=52)
    distractor = _target(observation, "m3_distractor")["position"]
    observation = _approach_and_close(environment, gate, candidate, offset, distractor)
    _assert(_physical(observation).get("reason_code") == "LIFT_REQUIRED", "wrong-object close did not stall")
    distractor = _target(observation, "m3_distractor")["position"]
    wrong_lift = _mount_pose(
        (distractor[0], distractor[1], distractor[2] + 0.080),
        candidate["orientation"],
        offset,
    )
    observation = _step(
        environment,
        {"action_type": "move_to", "target_pose": wrong_lift, "timeout_s": 60.0},
        gate,
    )
    reason = _physical(observation).get("reason_code")
    cases.append({"name": "wrong_object", "reason_code": reason})
    _assert(reason == "WRONG_OBJECT", f"wrong-object grasp returned {reason}")

    # Establish a fresh proven grasp, then release with the target airborne.
    observation, _ = environment.reset(seed=53)
    target = _target(observation)["position"]
    observation = _approach_and_close(environment, gate, candidate, offset, target)
    _assert(_physical(observation).get("reason_code") == "LIFT_REQUIRED", "drop setup close failed")
    target = _target(observation)["position"]
    lift = _mount_pose(
        (target[0], target[1], target[2] + 0.080), candidate["orientation"], offset
    )
    observation = _step(
        environment,
        {"action_type": "move_to", "target_pose": lift, "timeout_s": 60.0},
        gate,
    )
    _assert(_physical(observation).get("reason_code") == "TARGET_HELD", "drop setup lift failed")
    observation = _step(
        environment, {"action_type": "gripper_open", "timeout_s": 30.0}, gate
    )
    reason = _physical(observation).get("reason_code")
    cases.append({"name": "airborne_drop", "reason_code": reason})
    _assert(reason == "OBJECT_DROPPED", f"airborne release returned {reason}")

    # Repeat the proven grasp and release on the table outside the marker.
    observation, _ = environment.reset(seed=54)
    target = _target(observation)["position"]
    observation = _approach_and_close(environment, gate, candidate, offset, target)
    _assert(_physical(observation).get("reason_code") == "LIFT_REQUIRED", "outside setup close failed")
    target = _target(observation)["position"]
    lift = _mount_pose(
        (target[0], target[1], target[2] + 0.080), candidate["orientation"], offset
    )
    observation = _step(environment, {"action_type": "move_to", "target_pose": lift, "timeout_s": 60.0}, gate)
    _assert(_physical(observation).get("reason_code") == "TARGET_HELD", "outside setup lift failed")
    outside = (0.48, 0.08)
    lower = _mount_pose(
        (outside[0], outside[1], environment._m3_config.table_top_z_m + environment._m3_config.target_size_m[2] / 2),
        candidate["orientation"],
        offset,
    )
    observation = _step(environment, {"action_type": "move_to", "target_pose": lower, "timeout_s": 60.0}, gate)
    observation = _step(environment, {"action_type": "gripper_open", "timeout_s": 30.0}, gate)
    reason = _physical(observation).get("reason_code")
    cases.append({"name": "outside_destination", "reason_code": reason})
    _assert(reason == "OUTSIDE_DESTINATION", f"outside release returned {reason}")
    gate["negative_cases"] = {"status": "passed", "cases": cases}


def run_direct(path: Path) -> None:
    _assert_report_mutable(path)
    from extensions.gazebo.direct_env import GazeboDirectEnv

    report = _base(path)
    gate: dict[str, Any] = {"status": "running", "actions": [], "plan_only_candidates": [], "positive_rounds": []}
    report["gates"]["direct_live"] = gate
    _write(path, report)
    environment = None
    try:
        environment = GazeboDirectEnv(
            profile="m3_pickplace", task="M3 physical acceptance", seed=31
        )
        observation, _ = environment.reset(seed=31)
        _assert(environment.openeta_control_spec.get("physical_verification") is True, "wrong worker profile")
        _physical(observation)
        offset = _grasp_center_offset(environment)
        gate["grasp_center_offset_mount_m"] = list(offset)
        candidate = _select_candidate(environment, observation, offset, gate)
        gate["frozen_candidate"] = candidate
        before = _joint_inventory()
        for round_number in range(1, 6):
            start = len(gate["actions"])
            _positive_round(environment, gate, candidate, offset)
            gate["positive_rounds"].append(
                {"round": round_number, "actions": list(range(start, len(gate["actions"]))), "status": "passed"}
            )
        after = _joint_inventory()
        gate["gazebo_joint_inventory"] = {"before": before, "after": after}
        _assert(before["sha256"] == after["sha256"], "Gazebo joint topology changed during grasp")
        _negative_cases(environment, gate, candidate, offset)
        gate["status"] = "passed"
    except Exception as exc:
        if gate.get("status") == "running":
            gate["status"] = "blocked"
            gate["blocker"] = f"{type(exc).__name__}: {exc}"
        gate["traceback"] = traceback.format_exc()
        raise
    finally:
        if environment is not None:
            with suppress(Exception):
                environment.close()
        _write(path, report)


def _quat_to_euler(quaternion: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sine = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sine) if abs(sine) >= 1 else math.asin(sine)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(item) for item in (roll, pitch, yaw))


def run_mcp(path: Path, url: str) -> None:
    _assert_report_mutable(path)
    from agent.tools.sim_mcp import SseSimulatorMcpTransport

    report = _base(path)
    direct = report["gates"].get("direct_live", {})
    _assert(direct.get("status") == "passed", "direct gate must pass before MCP")
    gate: dict[str, Any] = {"status": "running", "actions": []}
    report["gates"]["mcp_live"] = gate
    _write(path, report)
    transport = SseSimulatorMcpTransport(url)
    handle = session_id = ""
    try:
        deadline = time.monotonic() + 60.0
        while True:
            try:
                names = {item["name"] for item in transport.list_tools(timeout_s=10)["tools"]}
                _assert({"create_env", "reset_env", "observe_env", "move_to", "gripper_open", "gripper_close", "close_env"}.issubset(names), "MCP catalogue incomplete")
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1.0)
        created = transport.call_tool("create_env", {"env_id": ENV_ID, "seed": 41, "task": "M3 MCP acceptance"}, timeout_s=180)
        handle, session_id = created["handle"], created["session_id"]
        _assert(created.get("control_spec", {}).get("m3") is True, "MCP selected wrong profile")
        common = {"handle": handle, "session_id": session_id}
        gate["create_env"] = _compact(created)
        reset = transport.call_tool("reset_env", {**common, "seed": 41}, timeout_s=180)
        gate["reset_env"] = _compact(reset)
        # Direct acceptance freezes exact collision-aware poses for the MCP
        # replay; MCP deliberately exposes no second TF or planning API.
        candidate = direct["frozen_candidate"]
        orientation = candidate["orientation"]
        offset = direct["grasp_center_offset_mount_m"]

        def observation_of(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
            observation = receipt.get("observation", receipt)
            _physical(observation)
            _assert(len(observation.get("cameras", [])) == 2, "MCP RGB-D observation incomplete")
            _assert({item.get("id") for item in observation.get("objects", [])} == {"m3_target", "m3_distractor"}, "MCP Gazebo objects incomplete")
            return observation

        def action(name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            receipt = transport.call_tool(name, {**common, **arguments}, timeout_s=180)
            observation_of(receipt)
            gate["actions"].append({"name": name, "receipt": _compact(receipt)})
            return receipt

        def move(pose: Mapping[str, Any]) -> Mapping[str, Any]:
            roll, pitch, yaw = _quat_to_euler(pose["quat_xyzw"])
            return action(
                "move_to",
                {
                    "x": pose["xyz"][0], "y": pose["xyz"][1], "z": pose["xyz"][2],
                    "roll": roll, "pitch": pitch, "yaw": yaw,
                    "tolerance": 0.002, "ori_tolerance": 0.05,
                },
            )

        def reset_round(seed: int) -> Mapping[str, Any]:
            return observation_of(
                transport.call_tool("reset_env", {**common, "seed": seed}, timeout_s=180)
            )

        def approach_close(center: Sequence[float]) -> Mapping[str, Any]:
            move(_mount_pose((center[0], center[1], center[2] + 0.080), orientation, offset))
            move(_mount_pose(center, orientation, offset))
            return action("gripper_close", {})

        for round_number in range(2):
            observation = reset_round(42 + round_number)
            target = _target(observation)["position"]
            close = approach_close(target)
            _assert(_physical(observation_of(close)).get("reason_code") == "LIFT_REQUIRED", "MCP close candidate failed")
            target = _target(observation_of(close))["position"]
            lift = _mount_pose((target[0], target[1], target[2] + 0.080), orientation, offset)
            lifted = move(lift)
            _assert(_physical(observation_of(lifted)).get("reason_code") == "TARGET_HELD", "MCP lift failed")
            destination = (0.48, -0.10)
            move(_mount_pose((destination[0], destination[1], target[2] + 0.080), orientation, offset))
            move(_mount_pose((destination[0], destination[1], 0.430), orientation, offset))
            opened = action("gripper_open", {})
            _assert(_physical(observation_of(opened)).get("reason_code") == "TARGET_PLACED", "MCP placement failed")

        negative: list[dict[str, Any]] = []
        reset_round(61)
        empty = approach_close((0.48, 0.12, 0.430))
        negative.append({"name": "empty_grasp", "reason_code": _physical(observation_of(empty)).get("reason_code")})
        _assert(negative[-1]["reason_code"] == "EMPTY_GRASP", "MCP empty-grasp mismatch")

        observation = reset_round(62)
        distractor = _target(observation, "m3_distractor")["position"]
        wrong = approach_close(distractor)
        _assert(_physical(observation_of(wrong)).get("reason_code") == "LIFT_REQUIRED", "MCP wrong-object close did not stall")
        distractor = _target(observation_of(wrong), "m3_distractor")["position"]
        wrong = move(
            _mount_pose(
                (distractor[0], distractor[1], distractor[2] + 0.080),
                orientation,
                offset,
            )
        )
        negative.append({"name": "wrong_object", "reason_code": _physical(observation_of(wrong)).get("reason_code")})
        _assert(negative[-1]["reason_code"] == "WRONG_OBJECT", "MCP wrong-object mismatch")

        for seed, name, destination in (
            (63, "airborne_drop", None),
            (64, "outside_destination", (0.48, 0.08)),
        ):
            observation = reset_round(seed)
            target = _target(observation)["position"]
            closed = approach_close(target)
            _assert(_physical(observation_of(closed)).get("reason_code") == "LIFT_REQUIRED", f"MCP {name} setup close failed")
            target = _target(observation_of(closed))["position"]
            lifted = move(_mount_pose((target[0], target[1], target[2] + 0.080), orientation, offset))
            _assert(_physical(observation_of(lifted)).get("reason_code") == "TARGET_HELD", f"MCP {name} setup lift failed")
            if destination is not None:
                move(_mount_pose((destination[0], destination[1], 0.430), orientation, offset))
            opened = action("gripper_open", {})
            reason = _physical(observation_of(opened)).get("reason_code")
            expected = "OBJECT_DROPPED" if destination is None else "OUTSIDE_DESTINATION"
            negative.append({"name": name, "reason_code": reason})
            _assert(reason == expected, f"MCP {name} returned {reason}")
        gate["negative_cases"] = negative
        observed = transport.call_tool("observe_env", common, timeout_s=60)
        _physical(observed.get("observation", observed))
        gate["observe_env"] = _compact(observed)
        first = transport.call_tool("close_env", common, timeout_s=60)
        second = transport.call_tool("close_env", common, timeout_s=60)
        _assert(first.get("already_closed") is False and second.get("already_closed") is True, "close is not idempotent")
        gate["close_env"] = [_compact(first), _compact(second)]
        gate["status"] = "passed"
    except Exception as exc:
        gate.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        raise
    finally:
        if handle and gate.get("status") != "passed":
            with suppress(Exception):
                transport.call_tool("close_env", {"handle": handle, "session_id": session_id}, timeout_s=60)
        _write(path, report)


def record_gate(path: Path, name: str, status: str, details: str) -> None:
    _assert_report_mutable(path)
    report = _base(path)
    report["gates"][name] = {"status": status, "details": details}
    _write(path, report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init", "direct", "mcp", "processes", "finalize", "gate", "probe", "graph"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--original-domain", type=int, default=0)
    parser.add_argument("--partition", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--world", default="")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--mcp-url", default="http://127.0.0.1:8765/sse")
    parser.add_argument("--gate", default="")
    parser.add_argument("--status", choices=("passed", "failed", "skipped"), default="passed")
    parser.add_argument("--details", default="")
    arguments = parser.parse_args()
    path = arguments.report.resolve()
    if arguments.mode == "init":
        init_report(
            path, arguments.domain, arguments.original_domain,
            arguments.partition, arguments.port, arguments.world,
        )
    elif arguments.mode == "direct":
        run_direct(path)
    elif arguments.mode == "mcp":
        run_mcp(path, arguments.mcp_url)
    elif arguments.mode == "graph":
        evidence = probe_ros_graph(arguments.domain)
        print(json.dumps(evidence, sort_keys=True))
        return 0 if evidence.get("availability") == "AVAILABLE" else 1
    elif arguments.mode == "probe":
        evidence = candidate_domain_evidence(arguments.domain)
        print(json.dumps(evidence, sort_keys=True))
        return 0 if evidence.get("state") == PASSED else 1
    elif arguments.mode == "processes":
        for pgid in sorted({item["pgid"] for item in _isolated_processes(arguments.partition)}):
            print(pgid)
    elif arguments.mode == "gate":
        _assert(bool(arguments.gate), "--gate is required")
        record_gate(path, arguments.gate, arguments.status, arguments.details)
    else:
        try:
            finalize_report(
                path, arguments.domain, arguments.partition, arguments.port,
                arguments.world, arguments.exit_code,
            )
        except (RuntimeError, ValueError) as exc:
            print(str(exc), file=os.sys.stderr)
            return 11
        status = _load(path).get("gates", {}).get("isolation_cleanup", {}).get("status")
        return 0 if status == "passed" else 9 if status == "failed" else 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
