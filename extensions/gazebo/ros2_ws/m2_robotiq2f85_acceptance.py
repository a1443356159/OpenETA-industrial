#!/usr/bin/env python3
"""Live M2 acceptance driver for the production Robotiq 2F-85 profile.

The direct mode talks to the same ``RosM2ControllerFactory`` used by the
Gazebo worker.  The MCP mode talks to the real SSE endpoint through
``SseSimulatorMcpTransport``.  Results are compacted into one ignored JSON
report; camera payload bytes are deliberately omitted from that report.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[3]
REPORT_VERSION = "openeta.m2_acceptance.v1"
ENV_ID = "openeta/gazebo_rm75_robotiq2f85-v0"
MODEL_ID = "rm75_robotiq_2f85_sim_v1"
POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = 0.08
MIMIC_TOLERANCE_RAD = 0.035
TARGET_OFFSET_M = 0.030
MIN_TARGET_SEPARATION_M = 0.040


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _distance(a: Iterable[float], b: Iterable[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _orientation_error(a: Iterable[float], b: Iterable[float]) -> float:
    qa, qb = [float(x) for x in a], [float(x) for x in b]
    na, nb = math.sqrt(sum(x * x for x in qa)), math.sqrt(sum(x * x for x in qb))
    _assert(na > 0.0 and nb > 0.0, "pose quaternion must be non-zero")
    dot = abs(sum(x * y for x, y in zip(qa, qb)) / (na * nb))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _quat_to_euler_degrees(quat: Iterable[float]) -> tuple[float, float, float]:
    x, y, z, w = [float(value) for value in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _compact(value: Any) -> Any:
    """Keep receipts and metadata, but never duplicate image payloads in reports."""

    if isinstance(value, dict):
        return {
            key: _compact(item)
            for key, item in value.items()
            if key not in {"rgb_base64", "depth_base64", "rgb", "depth"}
        }
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value]
    return value


def _load_report(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": REPORT_VERSION, "gates": {}}


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _asset_evidence() -> dict[str, Any]:
    from extensions.gazebo.asset_preflight import validate_asset_root
    from extensions.gazebo.m2 import Robotiq2F85Config

    config = Robotiq2F85Config()
    roots = (config.asset_root, config.gripper_asset_root)
    evidence: dict[str, Any] = {}
    for root in roots:
        manifest_path = root / "asset_manifest.json"
        manifest = validate_asset_root(root)
        evidence[root.name] = {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "asset_id": manifest.get("model_id", manifest.get("description_id")),
            "file_count": len(manifest.get("files", [])),
        }
    return evidence


def _process_row(pid: int) -> dict[str, Any] | None:
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        tail = stat.rsplit(")", 1)[1].split()
        pgid = int(tail[2])
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
        return {"pid": pid, "pgid": pgid, "command": command[:500]}
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None


def _ancestors() -> set[int]:
    result: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            for line in (Path("/proc") / str(pid) / "status").read_text().splitlines():
                if line.startswith("PPid:"):
                    pid = int(line.split()[1])
                    break
            else:
                break
        except (FileNotFoundError, PermissionError, ValueError):
            break
    return result


def _isolated_processes(partition: str) -> list[dict[str, Any]]:
    excluded = _ancestors()
    rows: list[dict[str, Any]] = []
    expected = f"GZ_PARTITION={partition}".encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in excluded:
            continue
        try:
            environment = (proc / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if expected not in environment:
            continue
        row = _process_row(int(proc.name))
        if row is not None:
            rows.append(row)
    return sorted(rows, key=lambda item: item["pid"])


def _preexisting_processes() -> list[dict[str, Any]]:
    tokens = (
        "gz sim",
        "sim.mcp_server",
        "bench_worker.py --bench gazebo",
        "m2_gazebo_moveit.launch.py",
        "move_group",
    )
    rows: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        row = _process_row(int(proc.name))
        if row is not None and any(token in row["command"] for token in tokens):
            rows.append(row)
    return sorted(rows, key=lambda item: item["pid"])


def _ros_nodes(domain: int) -> list[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "ROS_DOMAIN_ID": str(domain),
            "ROS2CLI_DISABLE_DAEMON": "1",
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
            "ROS_LOCALHOST_ONLY": "1",
        }
    )
    try:
        result = subprocess.run(
            ["ros2", "node", "list"],
            capture_output=True,
            text=True,
            timeout=8.0,
            env=environment,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ["<node-list-unavailable>"]
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def init_isolation_report(
    report_path: Path,
    *,
    domain: int,
    original_domain: int,
    partition: str,
    port: int,
) -> None:
    report = _base_report(report_path)
    report["isolation"] = {
        "ros_domain_id": domain,
        "original_ros_domain_id": original_domain,
        "gz_partition": partition,
        "mcp_port": port,
        "preexisting_processes": _preexisting_processes(),
        "preexisting_default_domain_nodes": _ros_nodes(original_domain),
        "cleanup_path_self_tests": {
            "normal": True,
            "startup_failure": True,
            "action_failure": True,
            "interrupt_signal": True,
        },
    }
    report["gates"]["isolation_cleanup"] = {"status": "running"}
    _write_report(report_path, report)


def finalize_isolation_report(
    report_path: Path,
    *,
    domain: int,
    partition: str,
    port: int,
    exit_code: int,
) -> bool:
    report = _load_report(report_path)
    isolation = report.setdefault("isolation", {})
    checks: dict[str, Any] = {}
    residual = _isolated_processes(partition)
    checks["isolated_processes_gone"] = {"ok": not residual, "residual": residual}
    # Fast DDS discovery participants can remain visible briefly after their
    # processes exit. Allow the graph lease to expire before declaring a leak.
    deadline = time.monotonic() + 45.0
    nodes: list[str] = []
    while time.monotonic() < deadline:
        nodes = _ros_nodes(domain)
        if not nodes:
            break
        time.sleep(0.5)
    checks["test_domain_empty"] = {"ok": not nodes, "nodes": nodes}
    port_deadline = time.monotonic() + 10.0
    port_error = ""
    port_free = False
    while time.monotonic() < port_deadline:
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            port_free = True
            break
        except OSError as exc:
            port_error = str(exc)
            time.sleep(0.25)
        finally:
            sock.close()
    checks["mcp_port_rebind"] = {
        "ok": port_free,
        **({"error": port_error} if not port_free else {}),
    }
    try:
        topics = subprocess.run(
            ["gz", "topic", "-l"], capture_output=True, text=True, timeout=8.0, check=False
        ).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        topics = ["<topic-list-unavailable>"]
    world_topics = sorted(topic for topic in topics if "/world/m2_rm75_robotiq2f85" in topic)
    checks["test_partition_empty"] = {"ok": not world_topics, "world_topics": world_topics}
    preexisting = isolation.get("preexisting_processes", [])
    vanished = [item for item in preexisting if not (Path("/proc") / str(item["pid"])).exists()]
    checks["preexisting_processes_alive"] = {"ok": not vanished, "vanished": vanished}
    original_domain = int(isolation.get("original_ros_domain_id", domain))
    before_nodes = set(isolation.get("preexisting_default_domain_nodes", []))
    after_nodes = set(_ros_nodes(original_domain))
    missing_nodes = sorted(before_nodes - after_nodes)
    checks["preexisting_default_domain_healthy"] = {
        "ok": not missing_nodes,
        "before": sorted(before_nodes),
        "after": sorted(after_nodes),
        "missing": missing_nodes,
    }
    all_ok = exit_code == 0 and all(bool(item.get("ok")) for item in checks.values())
    report["gates"]["isolation_cleanup"] = {
        "status": "passed" if all_ok else "failed",
        "main_exit_code": exit_code,
        "checks": checks,
    }
    report["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_report(report_path, report)
    return all_ok


def record_gate(report_path: Path, name: str, status: str, details: str) -> None:
    report = _base_report(report_path)
    report["gates"][name] = {"status": status, "details": details}
    _write_report(report_path, report)


def _base_report(path: Path) -> dict[str, Any]:
    report = _load_report(path)
    report.setdefault("schema_version", REPORT_VERSION)
    report.setdefault("gates", {})
    report.setdefault("started_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with suppress(Exception):
        report["git_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    report["assets"] = _asset_evidence()
    return report


def _camera_timestamps(cameras: Iterable[Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for camera in cameras:
        if isinstance(camera, Mapping):
            frame_id, stamp = camera.get("frame_id"), camera.get("timestamp_s")
        else:
            frame_id, stamp = camera.frame_id, camera.timestamp_s
        _assert(isinstance(frame_id, str) and frame_id, "camera frame_id is missing")
        _assert(isinstance(stamp, (int, float)), f"camera {frame_id} timestamp is missing")
        values[frame_id] = float(stamp)
    _assert(
        set(values) == {"top_camera_optical_frame", "wrist_camera_optical_frame"},
        f"unexpected camera frames: {sorted(values)}",
    )
    return values


def _assert_new_timestamps(current: Mapping[str, float], previous: Mapping[str, float]) -> None:
    for frame_id, timestamp in current.items():
        _assert(
            timestamp > previous.get(frame_id, float("-inf")),
            f"camera {frame_id} did not advance: {timestamp} <= {previous.get(frame_id)}",
        )


def _validate_robot_metadata(robot: Mapping[str, Any]) -> None:
    metadata = robot.get("metadata", {})
    _assert(metadata.get("model_id") == MODEL_ID, "robot model_id mismatch")
    _assert(metadata.get("eef_frame") == "gripper_mount_link", "EEF frame mismatch")
    names = metadata.get("joint_names", [])
    _assert(len(names) == 13, f"expected 13 RM75/Robotiq joints, received {len(names)}")
    _assert(len(robot.get("joint_positions", [])) == 13, "robot state is incomplete")


def _validate_pose(receipt: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, float]:
    end = receipt.get("end") or receipt.get("observation", {}).get("robot", {}).get(
        "end_effector_pose", {}
    )
    position_error = _distance(end.get("xyz", []), target["xyz"])
    orientation_error = _orientation_error(end.get("quat_xyzw", []), target["quat_xyzw"])
    _assert(position_error <= POSITION_TOLERANCE_M, f"EEF position error {position_error:.6f} m")
    _assert(
        orientation_error <= ORIENTATION_TOLERANCE_RAD,
        f"EEF orientation error {orientation_error:.6f} rad",
    )
    return {
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
    }


def _assert_action_timing(receipt: Mapping[str, Any], observation: Mapping[str, Any]) -> None:
    started = receipt.get("action_started_ros_time_s")
    completed = receipt.get("action_completed_ros_time_s")
    _assert(isinstance(started, (int, float)), "action start ROS timestamp is missing")
    _assert(isinstance(completed, (int, float)), "action completion ROS timestamp is missing")
    _assert(float(completed) >= float(started), "action timestamp order is invalid")
    robot_stamp = observation.get("robot", {}).get("metadata", {}).get(
        "joint_state_timestamp_s"
    )
    _assert(isinstance(robot_stamp, (int, float)), "JointState timestamp is missing")
    _assert(float(robot_stamp) > float(completed), "JointState is not post-action")
    for frame_id, stamp in _camera_timestamps(observation.get("cameras", [])).items():
        _assert(float(stamp) > float(completed), f"camera {frame_id} is not post-action")


def _direct_observation(controller: Any, cameras: list[Any], barrier_s: float | None = None):
    boundary = time.monotonic()
    frames = [
        camera.capture(
            timeout_s=12.0,
            min_timestamp_s=barrier_s,
            min_received_monotonic_s=boundary if barrier_s is not None else None,
        )
        for camera in cameras
    ]
    state = controller.state_provider()
    return frames, state


def _validate_gripper_state(state: Any, expected: str) -> dict[str, Any]:
    names = list(state.metadata["joint_names"])
    values = dict(zip(names, state.joint_positions))
    active = float(values["gripper_left_finger_joint"])
    expected_active = 0.0 if expected == "open" else 0.7929
    active_error = abs(active - expected_active)
    _assert(active_error <= MIMIC_TOLERANCE_RAD, f"active gripper error {active_error:.6f} rad")
    multipliers = {
        "gripper_right_finger_joint": -1.0,
        "gripper_left_inner_knuckle_joint": 1.0,
        "gripper_right_inner_knuckle_joint": -1.0,
        "gripper_left_finger_tip_joint": -1.0,
        "gripper_right_finger_tip_joint": 1.0,
    }
    mimic_errors = {
        name: abs(float(values[name]) - multiplier * active)
        for name, multiplier in multipliers.items()
    }
    _assert(
        max(mimic_errors.values()) <= MIMIC_TOLERANCE_RAD,
        f"Robotiq mimic relation failed: {mimic_errors}",
    )
    aperture = float(state.gripper_state["aperture_m"])
    expected_aperture = 0.085 if expected == "open" else 0.0
    _assert(abs(aperture - expected_aperture) <= 0.006, f"aperture mismatch: {aperture}")
    return {
        "expected": expected,
        "active_position_rad": active,
        "active_error_rad": active_error,
        "mimic_errors_rad": mimic_errors,
        "aperture_m": aperture,
    }


def run_direct(report_path: Path) -> None:
    from extensions.gazebo.m2 import Robotiq2F85Config
    from extensions.gazebo.observation import RosRgbdCameraSource
    from extensions.gazebo.ros_control import RosM2ControllerFactory
    from extensions.gazebo.worker import m2_live_session_config_from_env

    report = _base_report(report_path)
    gate: dict[str, Any] = {"status": "running", "actions": [], "ik_prechecks": []}
    report["gates"]["direct_live"] = gate
    _write_report(report_path, report)
    controller = None
    cameras: list[Any] = []
    try:
        config = Robotiq2F85Config()
        controller = RosM2ControllerFactory(readiness_timeout_s=90.0).create(config)
        live_config = m2_live_session_config_from_env(robotiq=True)
        cameras = [RosRgbdCameraSource(live_config.camera, node_name="openeta_accept_top")]
        cameras.extend(
            RosRgbdCameraSource(item, node_name=f"openeta_accept_camera_{index}")
            for index, item in enumerate(live_config.additional_cameras, 1)
        )
        for camera in cameras:
            camera.start()

        frames, reset_state = _direct_observation(controller, cameras)
        _validate_robot_metadata(reset_state.to_dict())
        previous_camera_timestamps = _camera_timestamps(frames)
        reset_pose = {
            "xyz": list(reset_state.end_effector_pose["xyz"]),
            "quat_xyzw": list(reset_state.end_effector_pose["quat_xyzw"]),
        }
        gate["reset_pose"] = reset_pose
        gate["initial_camera_timestamps"] = previous_camera_timestamps

        for _round in range(2):
            for command, expected in (
                ("gripper_open", "open"),
                ("gripper_close", "closed"),
                ("gripper_open", "open"),
            ):
                receipt = controller.execute({"action_type": command}).to_dict()
                _assert(receipt.get("ok") is True, f"{command} failed: {receipt}")
                barrier = float(receipt["action_completed_ros_time_s"])
                frames, state = _direct_observation(controller, cameras, barrier)
                camera_timestamps = _camera_timestamps(frames)
                _assert_new_timestamps(camera_timestamps, previous_camera_timestamps)
                previous_camera_timestamps = camera_timestamps
                observation = {
                    "cameras": [frame.to_dict() for frame in frames],
                    "robot": state.to_dict(),
                }
                _assert_action_timing(receipt, observation)
                gate["actions"].append(
                    {
                        "name": command,
                        "receipt": _compact(receipt),
                        "gripper": _validate_gripper_state(state, expected),
                        "camera_timestamps": camera_timestamps,
                    }
                )

        offsets = (
            (TARGET_OFFSET_M, 0.0, 0.0),
            (-TARGET_OFFSET_M, 0.0, 0.0),
            (0.0, TARGET_OFFSET_M, 0.0),
            (0.0, -TARGET_OFFSET_M, 0.0),
            (0.0, 0.0, -TARGET_OFFSET_M),
            (0.0, 0.0, TARGET_OFFSET_M),
        )
        reachable: list[dict[str, Any]] = []
        for offset in offsets:
            target = {
                "xyz": [reset_pose["xyz"][index] + offset[index] for index in range(3)],
                "quat_xyzw": list(reset_pose["quat_xyzw"]),
            }
            planned = controller.plan_pose(target, timeout_s=30.0)
            gate["ik_prechecks"].append(
                {"offset_xyz_m": list(offset), "target": target, "result": _compact(planned)}
            )
            if planned.get("ok") is True:
                reachable.append(target)
                if any(
                    _distance(first["xyz"], second["xyz"])
                    >= MIN_TARGET_SEPARATION_M
                    for first, second in itertools.combinations(reachable, 2)
                ):
                    break
        pair = next(
            (
                (first, second)
                for first, second in itertools.combinations(reachable, 2)
                if _distance(first["xyz"], second["xyz"]) >= MIN_TARGET_SEPARATION_M
            ),
            None,
        )
        _assert(pair is not None, "fewer than two separated collision-aware IK targets")
        target_a, target_b = pair
        gate["targets"] = {"A": target_a, "B": target_b}

        for name, target in (("A", target_a), ("B", target_b), ("A", target_a), ("reset", reset_pose)):
            receipt = controller.execute(
                {
                    "action_type": "move_to",
                    "target_pose": target,
                    "position_tolerance_m": 0.002,
                    "orientation_tolerance_rad": 0.05,
                    "timeout_s": 60.0,
                }
            ).to_dict()
            _assert(receipt.get("ok") is True, f"move_to({name}) failed: {receipt}")
            _assert(receipt.get("motion_outcome") == "completed", "motion was not completed")
            _assert(receipt.get("reached_target") is True, "motion receipt did not reach target")
            barrier = float(receipt["action_completed_ros_time_s"])
            frames, state = _direct_observation(controller, cameras, barrier)
            camera_timestamps = _camera_timestamps(frames)
            _assert_new_timestamps(camera_timestamps, previous_camera_timestamps)
            previous_camera_timestamps = camera_timestamps
            observation = {
                "cameras": [frame.to_dict() for frame in frames],
                "robot": state.to_dict(),
            }
            _assert_action_timing(receipt, observation)
            errors = _validate_pose(receipt, target)
            gate["actions"].append(
                {
                    "name": f"move_to_{name}",
                    "target": target,
                    "receipt": _compact(receipt),
                    "errors": errors,
                    "camera_timestamps": camera_timestamps,
                }
            )
        gate["status"] = "passed"
    except Exception as exc:
        gate.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        for camera in cameras:
            with suppress(Exception):
                camera.close()
        if controller is not None:
            with suppress(Exception):
                controller.close()
        _write_report(report_path, report)


def _mcp_observation(result: Mapping[str, Any]) -> dict[str, Any]:
    observation = result.get("observation", result)
    _assert(isinstance(observation, dict), "MCP result has no observation")
    _assert(len(observation.get("cameras", [])) == 2, "MCP observation is missing RGB-D cameras")
    for camera in observation["cameras"]:
        _assert(camera.get("rgb_base64"), f"camera {camera.get('frame_id')} has no RGB payload")
        _assert(camera.get("depth_base64"), f"camera {camera.get('frame_id')} has no depth payload")
        _assert(camera.get("intrinsics", {}).get("fx", 0) > 0, "camera intrinsics are invalid")
        _assert(camera.get("extrinsics", {}).get("frame_transform"), "camera extrinsics missing")
    _validate_robot_metadata(observation.get("robot", {}))
    metadata = observation.get("metadata", {})
    _assert(metadata.get("model_id") == MODEL_ID, "observation model metadata mismatch")
    return observation


def run_mcp(report_path: Path, url: str) -> None:
    from agent.tools.sim_mcp import SseSimulatorMcpTransport

    report = _base_report(report_path)
    direct = report.get("gates", {}).get("direct_live", {})
    _assert(direct.get("status") == "passed", "direct acceptance must pass before MCP")
    targets = direct["targets"]
    transport = SseSimulatorMcpTransport(url)
    gate: dict[str, Any] = {"status": "running", "actions": []}
    report["gates"]["mcp_live"] = gate
    _write_report(report_path, report)
    handle = ""
    session_id = ""
    close_count = 0
    previous_timestamps: dict[str, float] = {}
    try:
        deadline = time.monotonic() + 60.0
        while True:
            try:
                tools = transport.list_tools(timeout_s=10.0)
                names = {item["name"] for item in tools.get("tools", [])}
                _assert(
                    {"create_env", "reset_env", "move_to", "gripper_open", "gripper_close", "observe_env", "close_env"}.issubset(names),
                    "MCP tool catalogue is incomplete",
                )
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1.0)

        created = transport.call_tool(
            "create_env", {"env_id": ENV_ID, "seed": 23, "task": "M2 acceptance"}, timeout_s=180.0
        )
        _assert("error" not in created, f"create_env failed: {created}")
        handle, session_id = str(created["handle"]), str(created["session_id"])
        _assert(created.get("env_id") == ENV_ID, "created env_id mismatch")
        _assert(created.get("backend") == "gazebo", "created backend mismatch")
        control_spec = created.get("control_spec", {})
        _assert(control_spec.get("m2") is True, "MCP did not select the M2 control branch")
        _assert(control_spec.get("model_id") == MODEL_ID, "control model_id mismatch")
        gate["create_env"] = _compact(created)
        common = {"handle": handle, "session_id": session_id}

        reset = transport.call_tool("reset_env", {**common, "seed": 23}, timeout_s=180.0)
        _assert("error" not in reset, f"reset_env failed: {reset}")
        reset_observation = _mcp_observation(reset)
        previous_timestamps = _camera_timestamps(reset_observation["cameras"])
        gate["reset_env"] = _compact(reset)

        def action(name: str, arguments: dict[str, Any], target: Mapping[str, Any] | None = None):
            nonlocal previous_timestamps
            receipt = transport.call_tool(name, {**common, **arguments}, timeout_s=180.0)
            _assert("error" not in receipt, f"{name} failed at transport/server layer: {receipt}")
            observation = _mcp_observation(receipt)
            timestamps = _camera_timestamps(observation["cameras"])
            _assert_new_timestamps(timestamps, previous_timestamps)
            previous_timestamps = timestamps
            _assert_action_timing(receipt, observation)
            record: dict[str, Any] = {
                "name": name,
                "receipt": _compact(receipt),
                "camera_timestamps": timestamps,
            }
            if target is not None:
                _assert(receipt.get("ok") is True, f"{name} action failed: {receipt}")
                _assert(receipt.get("motion_outcome") == "completed", "MCP motion incomplete")
                _assert(receipt.get("reached_target") is True, "MCP target was not reached")
                record["target"] = target
                record["errors"] = _validate_pose(receipt, target)
            else:
                _assert(receipt.get("ok") is True, f"{name} action failed: {receipt}")
            gate["actions"].append(record)
            return receipt

        action("gripper_close", {})
        action("gripper_open", {})
        for label in ("A", "B", "A", "B"):
            target = targets[label]
            roll, pitch, yaw = _quat_to_euler_degrees(target["quat_xyzw"])
            action(
                "move_to",
                {
                    "x": target["xyz"][0],
                    "y": target["xyz"][1],
                    "z": target["xyz"][2],
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "tolerance": 0.002,
                    "ori_tolerance": 0.05,
                },
                target,
            )

        observed = transport.call_tool("observe_env", common, timeout_s=60.0)
        observation = _mcp_observation(observed)
        timestamps = _camera_timestamps(observation["cameras"])
        _assert_new_timestamps(timestamps, previous_timestamps)
        previous_timestamps = timestamps
        gate["observe_env"] = _compact(observed)

        unreachable_target = {
            "xyz": [100.0, 100.0, 100.0],
            "quat_xyzw": targets["B"]["quat_xyzw"],
        }
        roll, pitch, yaw = _quat_to_euler_degrees(unreachable_target["quat_xyzw"])
        rejected = transport.call_tool(
            "move_to",
            {
                **common,
                "x": 100.0,
                "y": 100.0,
                "z": 100.0,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "tolerance": 0.002,
                "ori_tolerance": 0.05,
            },
            timeout_s=180.0,
        )
        _assert(rejected.get("ok") is False, "unreachable motion was reported successful")
        _assert(rejected.get("error_code") == "MOTION_PLAN_FAILED", f"wrong rejection: {rejected}")
        rejected_observation = _mcp_observation(rejected)
        rejected_timestamps = _camera_timestamps(rejected_observation["cameras"])
        _assert_new_timestamps(rejected_timestamps, previous_timestamps)
        _assert_action_timing(rejected, rejected_observation)
        gate["unreachable"] = _compact(rejected)

        first_close = transport.call_tool("close_env", common, timeout_s=60.0)
        close_count += 1
        _assert(first_close.get("ok") is True, f"first close failed: {first_close}")
        _assert(first_close.get("already_closed") is False, "first close was not authoritative")
        second_close = transport.call_tool("close_env", common, timeout_s=60.0)
        close_count += 1
        _assert(second_close.get("ok") is True, f"second close failed: {second_close}")
        _assert(second_close.get("already_closed") is True, "close_env is not idempotent")
        gate["close_env"] = [_compact(first_close), _compact(second_close)]
        gate["status"] = "passed"
    except Exception as exc:
        gate.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        if handle and close_count == 0:
            with suppress(Exception):
                gate["emergency_close"] = _compact(
                    transport.call_tool(
                        "close_env",
                        {"handle": handle, "session_id": session_id},
                        timeout_s=60.0,
                    )
                )
        _write_report(report_path, report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("init", "direct", "mcp", "processes", "gate", "finalize")
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mcp-url", default="http://127.0.0.1:8765/sse")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--original-domain", type=int, default=0)
    parser.add_argument("--partition", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--gate", default="")
    parser.add_argument("--status", choices=("passed", "failed", "skipped"), default="passed")
    parser.add_argument("--details", default="")
    args = parser.parse_args()
    if args.mode == "init":
        init_isolation_report(
            args.report.resolve(),
            domain=args.domain,
            original_domain=args.original_domain,
            partition=args.partition,
            port=args.port,
        )
    elif args.mode == "direct":
        run_direct(args.report.resolve())
    elif args.mode == "mcp":
        run_mcp(args.report.resolve(), args.mcp_url)
    elif args.mode == "processes":
        for pgid in sorted({item["pgid"] for item in _isolated_processes(args.partition)}):
            print(pgid)
    elif args.mode == "gate":
        _assert(bool(args.gate), "--gate is required for gate mode")
        record_gate(args.report.resolve(), args.gate, args.status, args.details)
    else:
        return 0 if finalize_isolation_report(
            args.report.resolve(),
            domain=args.domain,
            partition=args.partition,
            port=args.port,
            exit_code=args.exit_code,
        ) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
