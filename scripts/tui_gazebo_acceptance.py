#!/usr/bin/env python3
"""Human-operated TUI acceptance coordinator for the Gazebo M0--M3 chain.

The coordinator owns isolation, evidence locations, process groups and report
assembly.  It intentionally does not impersonate the operator: all mutating
AgentTool calls happen in the real TUI with the ``human_gated`` profile.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, UTC
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid


SCHEMA_VERSION = "openeta.tui_gazebo_acceptance.v1"
MILESTONES = ("m0", "m1", "m2", "m3")
DETERMINISTIC = "deterministic"
AUTONOMY = "planner_autonomy"
PROTECTED_DOMAINS = frozenset({42, 100})
DOMAIN_CANDIDATES = tuple(i for i in range(80, 102) if i not in PROTECTED_DOMAINS)
MUTATING_TOOLS = frozenset(
    {
        "create_simulator_env",
        "close_simulator_env",
        "move_to",
        "follow_eef_trajectory",
        "gripper_control",
    }
)
SIX_SIMULATOR_TOOLS = frozenset(
    {
        "create_simulator_env",
        "close_simulator_env",
        "observe",
        "move_to",
        "follow_eef_trajectory",
        "gripper_control",
    }
)
ENV_IDS = {
    "m0": "openeta/dummy_sim-v0",
    "m1": "openeta/gazebo_live_rgbd-v0",
    "m2": "openeta/gazebo_rm75_robotiq2f85-v0",
    "m3": "openeta/gazebo_rm75_robotiq2f85_pickplace-v0",
}
INFRA_CODES = frozenset(
    {
        "ROS_NOT_READY",
        "GAZEBO_NOT_READY",
        "MODEL_ASSET_NOT_FOUND",
        "JOINT_STATE_TIMEOUT",
        "TF_TIMEOUT",
        "MOVE_GROUP_UNAVAILABLE",
        "MCP_NOT_READY",
        "TUI_NOT_READY",
        "TOPIC_DISCOVERY_TIMEOUT",
    }
)


class AcceptanceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Allocation:
    ros_domain_id: int
    gz_partition: str
    port: int
    run_id: str


@dataclass(frozen=True, slots=True)
class CasePaths:
    root: Path
    transcript: Path
    mcp_log: Path
    trace_root: Path
    receipt: Path
    instructions: Path
    mcp_config: Path
    pid_record: Path


def _json_dump(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _json_load(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def allocate(case_name: str, occupied_domains: Iterable[int] = ()) -> Allocation:
    occupied = set(occupied_domains) | set(PROTECTED_DOMAINS)
    domain = next((item for item in DOMAIN_CANDIDATES if item not in occupied), None)
    if domain is None:
        raise AcceptanceError("no isolated ROS_DOMAIN_ID is available")
    token = uuid.uuid4().hex[:12]
    return Allocation(
        ros_domain_id=domain,
        gz_partition=f"openeta-tui-{case_name}-{token}",
        port=_free_port(),
        run_id=token,
    )


def _command_output(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command), text=False, capture_output=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    raw = result.stdout or result.stderr
    if not raw:
        return ""
    encoding = "utf-16-le" if b"\x00" in raw[:16] else "utf-8"
    return raw.decode(encoding, errors="replace").strip()


def _process_snapshot() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    proc = Path("/proc")
    for entry in proc.iterdir() if proc.is_dir() else ():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if not any(token in raw for token in ("ros2", "gz sim", "bench_worker")):
                continue
            result.append({"pid": int(entry.name), "cmdline": raw.strip()})
        except (OSError, UnicodeDecodeError):
            continue
    return sorted(result, key=lambda item: item["pid"])


def environment_receipt(
    repo: Path,
    allocation: Allocation,
    *,
    case_name: str,
    before: list[dict[str, Any]],
    capture_protected: bool = True,
) -> dict[str, Any]:
    overlay = repo / "extensions/gazebo/ros2_ws/install/setup.bash"
    payload: dict[str, Any] = {
        "schema_version": "openeta.gazebo_environment_receipt.v1",
        "trusted": True,
        "case": case_name,
        "captured_at": datetime.now(UTC).isoformat(),
        "ros_domain_id": allocation.ros_domain_id,
        "gz_partition": allocation.gz_partition,
        "mcp_port": allocation.port,
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        "ros_distro": os.environ.get("ROS_DISTRO", "jazzy"),
        "overlay": str(overlay.resolve()),
        "overlay_present": overlay.is_file(),
        "platform": platform.platform(),
        "wsl_version": _command_output(["wsl.exe", "--version"]),
        "git_head": _command_output(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "git_dirty_summary": _command_output(
            ["git", "-C", str(repo), "status", "--short"]
        ),
        "protected_domains": sorted(PROTECTED_DOMAINS),
        "preexisting_processes": before,
    }
    if capture_protected:
        from extensions.gazebo.ros2_ws.acceptance_isolation import probe_ros_graph

        payload["protected_ros_graphs"] = {
            str(domain): probe_ros_graph(domain) for domain in sorted(PROTECTED_DOMAINS)
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def verify_receipt(receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if receipt.get("schema_version") != "openeta.gazebo_environment_receipt.v1":
        errors.append("environment receipt schema is missing")
    if receipt.get("trusted") is not True:
        errors.append("environment receipt is not trusted")
    domain = receipt.get("ros_domain_id")
    if type(domain) is not int or domain in PROTECTED_DOMAINS:
        errors.append("case reused a protected ROS domain")
    if not str(receipt.get("gz_partition") or "").startswith("openeta-tui-"):
        errors.append("case did not use a dedicated Gazebo partition")
    supplied_hash = str(receipt.get("receipt_sha256") or "")
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    actual_hash = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if supplied_hash != actual_hash:
        errors.append("environment receipt hash mismatch")
    return errors


def case_paths(root: Path, milestone: str, mode: str) -> CasePaths:
    case = root / milestone / mode
    return CasePaths(
        root=case,
        transcript=case / "tui.transcript",
        mcp_log=case / "mcp.log",
        trace_root=case / ".openeta_memory",
        receipt=case / "environment-receipt.json",
        instructions=case / "operator-instructions.txt",
        mcp_config=case / ".mcp.json",
        pid_record=case / "processes.json",
    )


def instructions_for(milestone: str, mode: str) -> str:
    if mode == AUTONOMY:
        prompts = {
            "m1": "检查 Gazebo 现场，并基于传感器证据报告所见。完成后关闭环境。",
            "m2": "在碰撞安全的 A/B 位姿之间安全移动两次，并验证夹爪开合。完成后关闭环境。",
            "m3": (
                "依据 observation 的 object state 和 physical verification，"
                "将 m3_target 放入 destination；不要使用 SAM3。完成后关闭环境。"
            ),
        }
        return prompts.get(milestone, "M0 不运行 Planner 自主性 case。") + "\n"
    rows = {
        "m0": """依次执行 /config、/tools、/session、/memory all --json。
要求 Planner 通过 AgentTool 创建 openeta/dummy_sim-v0，observe，然后 close。
逐一批准所有 human_gated 提示，确认关闭后用 /quit 退出。
""",
        "m1": """创建 openeta/gazebo_live_rgbd-v0，连续 observe 两次，然后 close。
逐一批准 create/reset/close 的 human_gated 提示，确认关闭后用 /quit 退出。
""",
        "m2": """仅使用 direct-gate.json 冻结的 collision_checked poses A/B。
执行两轮 A↔B、两轮 open→close→open、一次冻结的不可达目标、observe、close。
不得改写任何目标。逐一批准所有 human_gated 提示，最后 /quit。
""",
        "m3": """direct gate 必须已是 5/5。正向拾放两次；再分别用 seed 51、52、53、54
创建四个独立负例环境，得到 EMPTY_GRASP、WRONG_OBJECT、OBJECT_DROPPED、
OUTSIDE_DESTINATION。每个环境均须 close；逐一批准 human_gated 提示，最后 /quit。
""",
    }
    return rows[milestone]


def prepare_case(
    repo: Path, run_root: Path, milestone: str, mode: str, allocation: Allocation
) -> CasePaths:
    paths = case_paths(run_root, milestone, mode)
    paths.root.mkdir(parents=True, exist_ok=False)
    _json_dump(
        paths.mcp_config,
        {"mcpServers": {"openeta-sim": {"url": f"http://127.0.0.1:{allocation.port}/sse"}}},
    )
    paths.instructions.write_text(instructions_for(milestone, mode), encoding="utf-8")
    receipt = environment_receipt(
        repo, allocation, case_name=f"{milestone}-{mode}", before=_process_snapshot()
    )
    _json_dump(paths.receipt, receipt)
    return paths


def _wait_ready(port: int, process: subprocess.Popen[Any], timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AcceptanceError(f"MCP exited before readiness (code {process.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise AcceptanceError("MCP_NOT_READY")


def _terminate_owned_group(process: subprocess.Popen[Any], pgid: int) -> None:
    if process.poll() is not None:
        return
    if os.getpgid(process.pid) != pgid:
        raise AcceptanceError("refusing cleanup: recorded MCP PGID changed")
    os.killpg(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        process.wait(timeout=5)


def _partition_cleanup(partition: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["GZ_PARTITION"] = partition
    try:
        result = subprocess.run(
            ["gz", "topic", "-l"], env=env, capture_output=True, text=True,
            timeout=8, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"state": "INCONCLUSIVE", "reason": f"{type(exc).__name__}: {exc}"}
    topics = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    if result.returncode:
        return {
            "state": "INCONCLUSIVE", "returncode": result.returncode,
            "stderr": result.stderr[:500],
        }
    return {"state": "PASSED" if not topics else "FAILED", "topics": topics}


def run_case(repo: Path, paths: CasePaths, allocation: Allocation) -> int:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(repo),
            "ROS_DOMAIN_ID": str(allocation.ros_domain_id),
            "GZ_PARTITION": allocation.gz_partition,
            "MCP_PORT": str(allocation.port),
            "ROS_LOCALHOST_ONLY": "1",
            "OPENETA_SUPERVISION_PROFILE": "human_gated",
            "RMW_IMPLEMENTATION": env.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        }
    )
    python = repo / ".venv/bin/python"
    if not python.is_file():
        raise AcceptanceError("TUI_NOT_READY: repository .venv is missing")
    log = paths.mcp_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(python), "-m", "sim.mcp_server.server", "--transport", "sse", "--port", str(allocation.port)],
        cwd=paths.root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    pgid = os.getpgid(process.pid)
    _json_dump(
        paths.pid_record,
        {"mcp_pid": process.pid, "mcp_pgid": pgid, "owned": True, "run_id": allocation.run_id},
    )
    tui_code = 2
    try:
        _wait_ready(allocation.port, process)
        print(f"\n=== {paths.root.name} ===")
        print(paths.instructions.read_text(encoding="utf-8"))
        command = f"{python} -m agent.cli.openeta_cli"
        if shutil.which("script") is None:
            raise AcceptanceError("TUI_NOT_READY: util-linux script is missing")
        completed = subprocess.run(
            ["script", "--flush", "--return", "--command", command, str(paths.transcript)],
            cwd=paths.root,
            env=env,
            check=False,
        )
        tui_code = int(completed.returncode)
    finally:
        try:
            _terminate_owned_group(process, pgid)
        finally:
            log.close()
    after = _process_snapshot()
    receipt = _json_load(paths.receipt)
    before_pids = {item["pid"] for item in receipt["preexisting_processes"]}
    owned_residuals = [item for item in after if item["pid"] not in before_pids]
    from extensions.gazebo.ros2_ws.acceptance_isolation import (
        candidate_domain_evidence,
        probe_ros_graph,
    )

    protected_after = {
        str(domain): probe_ros_graph(domain) for domain in sorted(PROTECTED_DOMAINS)
    }
    protected_before = receipt.get("protected_ros_graphs", {})
    protected_graph_unchanged = all(
        protected_before.get(str(domain), {}).get("availability") == snapshot.get("availability")
        and protected_before.get(str(domain), {}).get("nodes") == snapshot.get("nodes")
        and protected_before.get(str(domain), {}).get("topics") == snapshot.get("topics")
        for domain, snapshot in (
            (domain, protected_after[str(domain)]) for domain in sorted(PROTECTED_DOMAINS)
        )
    )
    cleanup = {
        "mcp_group_exited": process.poll() is not None,
        "port_free": _port_is_free(allocation.port),
        "owned_process_residuals": owned_residuals,
        "preexisting_process_snapshot_unchanged": all(
            any(row["pid"] == item["pid"] for row in after)
            for item in receipt["preexisting_processes"]
        ),
        "ros_graph": candidate_domain_evidence(allocation.ros_domain_id),
        "gz_partition": _partition_cleanup(allocation.gz_partition),
        "protected_ros_graphs_after": protected_after,
        "protected_ros_graphs_unchanged": protected_graph_unchanged,
    }
    _json_dump(paths.root / "cleanup.json", cleanup)
    return tui_code


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk(child)


def _tool_calls(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    calls: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    action_events = [event for event in events if event.get("event_type") == "action"]
    for event in action_events or events:
        for node in _walk(event):
            if not isinstance(node, Mapping) or id(node) in seen:
                continue
            name = str(node.get("name") or node.get("tool_name") or "")
            if name in SIX_SIMULATOR_TOOLS and isinstance(node.get("result"), Mapping):
                calls.append(node)
                seen.add(id(node))
    return calls


def _load_trace_events(trace_root: Path) -> tuple[list[Mapping[str, Any]], list[Path]]:
    paths = sorted(trace_root.glob("sessions/*/trace.jsonl"))
    events: list[Mapping[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AcceptanceError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, Mapping):
                events.append(value)
    return events, paths


def _result(call: Mapping[str, Any]) -> Mapping[str, Any]:
    result = call.get("result")
    return result if isinstance(result, Mapping) else call


def _contains(node: Any, key: str, expected: Any = None) -> bool:
    for item in _walk(node):
        if isinstance(item, Mapping) and key in item:
            if expected is None or item[key] == expected:
                return True
    return False


def _values(node: Any, key: str) -> list[Any]:
    return [item[key] for item in _walk(node) if isinstance(item, Mapping) and key in item]


def _successful(call: Mapping[str, Any]) -> bool:
    result = _result(call)
    success = result.get("success", result.get("ok"))
    status = str(call.get("status") or "")
    return success is True or status in {"executed", "approved", "success"}


def _human_approved(call: Mapping[str, Any]) -> bool:
    if str(call.get("name") or call.get("tool_name") or "") not in MUTATING_TOOLS:
        return True
    for item in _walk(call):
        if not isinstance(item, Mapping):
            continue
        profile = str(
            item.get("supervision_profile") or item.get("profile") or item.get("policy") or ""
        ).lower()
        decision = str(item.get("decision") or item.get("status") or "").lower()
        approved = (
            item.get("approved") is True
            or item.get("allowed") is True
            or decision in {"approved", "executed", "allow"}
        )
        gated = profile == "human_gated" or _contains(item, "profile", "human_gated")
        if gated and approved:
            return True
    return False


def _camera_frames(node: Any) -> list[Mapping[str, Any]]:
    frames: list[Mapping[str, Any]] = []
    for item in _walk(node):
        if not isinstance(item, Mapping):
            continue
        if ("rgb_path" in item or "rgb_base64" in item) and (
            "depth_path" in item or "depth_base64" in item
        ):
            frames.append(item)
    return frames


def _timestamps(node: Any) -> list[float]:
    values: list[float] = []
    for key in ("timestamp_s", "ros_timestamp_s", "capture_timestamp_s"):
        for value in _values(node, key):
            if isinstance(value, (int, float)):
                values.append(float(value))
    return values


def _base_errors(paths: CasePaths, events: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = verify_receipt(_json_load(paths.receipt))
    cleanup_path = paths.root / "cleanup.json"
    if not cleanup_path.is_file():
        errors.append("cleanup evidence missing")
    else:
        cleanup = _json_load(cleanup_path)
        if not cleanup.get("mcp_group_exited") or not cleanup.get("port_free"):
            errors.append("owned MCP process group or port was not cleaned")
        if cleanup.get("owned_process_residuals"):
            errors.append("owned ROS/Gazebo worker residuals remain")
        if cleanup.get("preexisting_process_snapshot_unchanged") is not True:
            errors.append("preexisting long-lived process snapshot changed")
        if cleanup.get("protected_ros_graphs_unchanged") is not True:
            errors.append("protected ROS domain 42/100 graph changed")
        for name in ("ros_graph", "gz_partition"):
            state = str((cleanup.get(name) or {}).get("state") or "INCONCLUSIVE")
            if state == "FAILED":
                errors.append(f"cleanup {name} is not empty")
            elif state != "PASSED":
                errors.append(f"cleanup {name} is inconclusive")
    if not events:
        errors.append("trace.jsonl is missing or empty")
    return errors


def _verify_m0(calls: Sequence[Mapping[str, Any]], paths: CasePaths) -> list[str]:
    errors: list[str] = []
    names = [str(call.get("name") or call.get("tool_name") or "") for call in calls]
    expected = ["create_simulator_env", "observe", "close_simulator_env"]
    cursor = 0
    for name in names:
        if cursor < len(expected) and name == expected[cursor]:
            cursor += 1
    if cursor != len(expected):
        errors.append("M0 create→observe→close sequence missing")
    create = next((call for call in calls if str(call.get("name")) == expected[0]), {})
    if not _contains(create, "env_id", ENV_IDS["m0"]):
        errors.append("M0 dummy environment identity missing")
    if not _contains(create, "reset_response") and not _contains(create, "initial_observation"):
        errors.append("M0 create did not retain automatic reset evidence")
    transcript = paths.transcript.read_text(encoding="utf-8", errors="replace") if paths.transcript.exists() else ""
    for command in ("/config", "/tools", "/session", "/memory all --json"):
        if command not in transcript:
            errors.append(f"M0 transcript lacks {command}")
    for name in SIX_SIMULATOR_TOOLS:
        if name not in transcript:
            errors.append(f"M0 tool listing lacks handler {name}")
    if not list(paths.trace_root.glob("sessions/*/working/artifacts.json")):
        errors.append("M0 memory artifact missing")
    return errors


def _verify_m1(calls: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    observes = [call for call in calls if str(call.get("name") or call.get("tool_name")) == "observe"]
    if len(observes) < 2:
        return ["M1 requires two observe calls"]
    last_timestamp: float | None = None
    for index, call in enumerate(observes[:2], 1):
        result = _result(call)
        cameras = _camera_frames(result)
        if not cameras:
            errors.append(f"M1 observe {index} lacks RGB-D")
        if not _contains(result, "intrinsics") and not all(
            _contains(result, key) for key in ("fx", "fy", "cx", "cy")
        ):
            errors.append(f"M1 observe {index} lacks intrinsics/CameraInfo")
        scales = [value for value in _values(result, "scale") if isinstance(value, (int, float))]
        if not scales or any(float(value) <= 0 for value in scales):
            errors.append(f"M1 observe {index} lacks metric depth scale")
        if not _contains(result, "extrinsics"):
            errors.append(f"M1 observe {index} lacks explicit extrinsics")
        timestamps = _timestamps(result)
        current = max(timestamps) if timestamps else None
        if current is None:
            errors.append(f"M1 observe {index} lacks timestamp")
        elif last_timestamp is not None and current <= last_timestamp:
            errors.append("M1 observation timestamps are not strictly increasing")
        last_timestamp = current
        provenance = " ".join(
            str(v)
            for key in ("provenance", "observation_provenance")
            for v in _values(result, key)
        ).lower()
        if "gazebo" not in provenance or "live" not in provenance:
            errors.append(f"M1 observe {index} lacks live Gazebo provenance")
        if _contains(result, "render_frame"):
            errors.append("M1 contains forbidden generic render-frame injection")
    return errors


def _verify_m2(calls: Sequence[Mapping[str, Any]], direct_gate: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if direct_gate.get("status") != "passed" or not direct_gate.get("collision_checked"):
        errors.append("M2 direct fixture gate is not passed/collision-checked")
    moves = [call for call in calls if str(call.get("name") or call.get("tool_name")) == "move_to"]
    successes = [call for call in moves if _successful(call)]
    failures = [call for call in moves if _contains(_result(call), "error_code", "MOTION_PLAN_FAILED")]
    if len(successes) < 4:
        errors.append("M2 requires two A↔B rounds (four successful moves)")
    if len(failures) != 1:
        errors.append("M2 requires exactly one stable MOTION_PLAN_FAILED target")
    successful_controls = successes + [
        call
        for call in calls
        if str(call.get("name") or call.get("tool_name")) == "gripper_control"
        and _successful(call)
    ]
    for call in successful_controls:
        result = _result(call)
        if len(_camera_frames(result)) < 2:
            errors.append("M2 successful action lacks fresh dual RGB-D")
        if not _contains(result, "robot") and not _contains(result, "joint_positions"):
            errors.append("M2 successful action lacks RobotState")
        if not _contains(result, "schema_version", "m2_start_state_recovery_v1"):
            errors.append("M2 successful action lacks recovery receipt")
        if not _contains(result, "action_completed_ros_time_s"):
            errors.append("M2 successful action lacks completion barrier")
        if not _contains(result, "observation_fresh", True):
            errors.append("M2 successful action lacks fresh-observation receipt")
    if failures and (_successful(failures[0]) or _contains(_result(failures[0]), "reached_target", True)):
        errors.append("M2 unreachable target did not fail closed")
    grippers = [
        call for call in calls if str(call.get("name") or call.get("tool_name")) == "gripper_control"
    ]
    positions = [value for call in grippers for value in _values(call, "position")]
    pattern = [1, 0, 1, 1, 0, 1]
    cursor = 0
    for position in positions:
        if cursor < len(pattern) and position == pattern[cursor]:
            cursor += 1
    if cursor != len(pattern):
        errors.append("M2 requires two open→close→open rounds")
    frozen_targets = direct_gate.get("targets")
    if isinstance(frozen_targets, Mapping):
        submitted = [value for call in moves for value in _values(call, "target_pose")]
        allowed = list(frozen_targets.values())
        if any(target not in allowed for target in submitted):
            errors.append("M2 submitted target differs from frozen direct-gate target")
    return errors


def _verify_m3(calls: Sequence[Mapping[str, Any]], direct_gate: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        direct_gate.get("status") != "passed"
        or direct_gate.get("passes") != 5
        or direct_gate.get("attempts") != 5
        or direct_gate.get("collision_checked") is not True
    ):
        errors.append("M3 direct gate is not frozen at 5/5")
    reasons = [str(value) for call in calls for value in _values(_result(call), "reason_code")]
    for reason, count in (("LIFT_REQUIRED", 2), ("TARGET_HELD", 2), ("TARGET_PLACED", 2)):
        if reasons.count(reason) < count:
            errors.append(f"M3 positive path lacks two {reason} verdicts")
    for reason in ("EMPTY_GRASP", "WRONG_OBJECT", "OBJECT_DROPPED", "OUTSIDE_DESTINATION"):
        if reasons.count(reason) != 1:
            errors.append(f"M3 requires exactly one {reason} negative verdict")
    seeds = [value for call in calls for value in _values(call, "seed")]
    if not all(seed in seeds for seed in (51, 52, 53, 54)):
        errors.append("M3 negative environments do not use seeds 51–54")
    physical = [
        item for call in calls for item in _walk(_result(call))
        if isinstance(item, Mapping) and item.get("schema_version") == "m3_physical_verification_v1"
    ]
    if not physical:
        errors.append("M3 physical verification receipts missing")
    for item in physical:
        if not (_contains(item, "contacts") or _contains(item, "contact_evidence")):
            errors.append("M3 receipt lacks real contact evidence")
        if not (_contains(item, "objects") or _contains(item, "object_evidence")):
            errors.append("M3 receipt lacks object evidence")
    physical_calls = [
        call
        for call in calls
        if _contains(_result(call), "schema_version", "m3_physical_verification_v1")
    ]
    for call in physical_calls:
        result = _result(call)
        if len(_camera_frames(result)) < 2 or not _contains(result, "observation_fresh", True):
            errors.append("M3 physical verdict lacks fresh dual RGB-D")
        if not _contains(result, "schema_version", "openeta.environment_receipt.v1"):
            errors.append("M3 physical verdict lacks trusted environment receipt")
    return errors


def verify_case(
    paths: CasePaths,
    milestone: str,
    mode: str,
    *,
    direct_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        events, trace_paths = _load_trace_events(paths.trace_root)
        calls = _tool_calls(events)
        errors = _base_errors(paths, events)
        for call in calls:
            name = str(call.get("name") or call.get("tool_name") or "")
            if name in MUTATING_TOOLS and not _human_approved(call):
                errors.append(f"{name} lacks explicit human_gated approval evidence")
        if mode == AUTONOMY:
            expected = ENV_IDS[milestone]
            if not any(_contains(call, "env_id", expected) for call in calls):
                errors.append("Planner did not create the requested milestone environment")
            if not any(str(call.get("name") or call.get("tool_name")) == "close_simulator_env" for call in calls):
                errors.append("Planner did not close its environment")
        elif milestone == "m0":
            errors.extend(_verify_m0(calls, paths))
        elif milestone == "m1":
            errors.extend(_verify_m1(calls))
        elif milestone == "m2":
            errors.extend(_verify_m2(calls, direct_gate or {}))
        elif milestone == "m3":
            errors.extend(_verify_m3(calls, direct_gate or {}))
        error_codes = {str(value) for event in events for value in _values(event, "error_code")}
        infra = bool(error_codes & INFRA_CODES) or any(
            "inconclusive" in error for error in errors
        )
        status = "blocked" if infra else ("failed" if errors else "passed")
        return {
            "status": status,
            "errors": list(dict.fromkeys(errors)),
            "trace_paths": [str(path.resolve()) for path in trace_paths],
            "tool_call_count": len(calls),
            "infrastructure_codes": sorted(error_codes & INFRA_CODES),
        }
    except (OSError, ValueError, AcceptanceError) as exc:
        return {
            "status": "blocked",
            "errors": [f"evidence unreadable: {type(exc).__name__}: {exc}"],
            "trace_paths": [],
            "tool_call_count": 0,
            "infrastructure_codes": ["EVIDENCE_UNREADABLE"],
        }


def assemble_report(run_root: Path, *, direct_gates: Mapping[str, Any]) -> dict[str, Any]:
    milestones: dict[str, Any] = {}
    stop = False
    overall = "passed"
    for milestone in MILESTONES:
        if stop:
            milestones[milestone] = {
                "backend_chain_status": {"status": "not_run", "errors": ["formal predecessor gate did not pass"]},
                "planner_autonomy_status": {"status": "not_run", "errors": ["backend gate not passed"]},
            }
            continue
        backend_paths = case_paths(run_root, milestone, DETERMINISTIC)
        backend = verify_case(
            backend_paths,
            milestone,
            DETERMINISTIC,
            direct_gate=direct_gates.get(milestone) if isinstance(direct_gates, Mapping) else None,
        )
        if backend["status"] != "passed":
            stop = True
            overall = "inconclusive" if backend["status"] == "blocked" else "failed"
            autonomy = {"status": "not_run", "errors": ["backend gate not passed"]}
        elif milestone == "m0":
            autonomy = {"status": "not_applicable", "errors": []}
        else:
            autonomy = verify_case(case_paths(run_root, milestone, AUTONOMY), milestone, AUTONOMY)
        milestones[milestone] = {
            "backend_chain_status": backend,
            "planner_autonomy_status": autonomy,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_root": str(run_root.resolve()),
        "overall_status": overall,
        "milestones": milestones,
    }


def report_exit_code(report: Mapping[str, Any]) -> int:
    status = report.get("overall_status")
    return 0 if status == "passed" else (2 if status == "inconclusive" else 1)


def _new_run_root(repo: Path, requested: str) -> Path:
    if requested:
        return Path(requested).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return repo / ".cache/reports" / f"tui-gazebo-{stamp}-{os.getpid()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="")
    parser.add_argument("--direct-gates", default="", help="JSON with frozen M2/M3 direct gates")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    run_root = _new_run_root(repo, args.run_root)
    direct_gates: Mapping[str, Any] = {}
    if args.direct_gates:
        direct_gates = _json_load(Path(args.direct_gates))
    if args.verify_only:
        report = assemble_report(run_root, direct_gates=direct_gates)
        report_path = run_root / "acceptance-report.json"
        if report_path.exists():
            raise AcceptanceError(f"immutable report already exists: {report_path}")
        _json_dump(report_path, report, exclusive=True)
        report_path.chmod(0o444)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report_exit_code(report)

    run_root.mkdir(parents=True, exist_ok=False)
    occupied: set[int] = set()
    try:
        for milestone in MILESTONES:
            modes = (DETERMINISTIC,) if milestone == "m0" else (DETERMINISTIC, AUTONOMY)
            for mode in modes:
                allocation = allocate(f"{milestone}-{mode}", occupied)
                occupied.add(allocation.ros_domain_id)
                paths = prepare_case(repo, run_root, milestone, mode, allocation)
                if args.prepare_only:
                    continue
                code = run_case(repo, paths, allocation)
                if code == 130:
                    return 130
                if mode == DETERMINISTIC:
                    gate = verify_case(
                        paths,
                        milestone,
                        mode,
                        direct_gate=direct_gates.get(milestone),
                    )
                    _json_dump(paths.root / "verification.json", gate)
                    if gate["status"] != "passed":
                        report = assemble_report(run_root, direct_gates=direct_gates)
                        _json_dump(run_root / "acceptance-report.json", report, exclusive=True)
                        return report_exit_code(report)
        if args.prepare_only:
            print(run_root)
            return 0
        report = assemble_report(run_root, direct_gates=direct_gates)
        report_path = run_root / "acceptance-report.json"
        _json_dump(report_path, report, exclusive=True)
        report_path.chmod(0o444)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report_exit_code(report)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
