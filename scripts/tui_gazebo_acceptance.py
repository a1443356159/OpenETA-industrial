#!/usr/bin/env python3
"""PTY TUI acceptance coordinator for the ordered Gazebo M0--M4 chain.

The coordinator owns isolation, evidence locations, process groups and report
assembly.  It uses the real PTY TUI for every case.  The default profile is
``human_gated``; the explicitly selected ``scripted_tui`` profile records an
automation approval rather than impersonating a human operator.
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
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid


SCHEMA_VERSION = "openeta.tui_gazebo_acceptance.v2"
MILESTONES = ("m0", "m1", "m2", "m3", "m4")
DETERMINISTIC = "deterministic"
AUTONOMY = "planner_autonomy"
SCRIPTED_TUI = "scripted_tui"
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
    "m4": "openeta/gazebo_rm75_robotiq2f85_pickplace-v0",
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
            # A recently closed MCP/SSE listener can leave TCP state behind
            # after every owned process has exited.  Reusing that harmless
            # state is not a residual server; ``listen()`` still rejects any
            # live process that owns the endpoint.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
        except OSError:
            return False
    return True


def _wait_for_free_port(port: int, *, timeout_s: float = 5.0) -> bool:
    """Bound cleanup on actual listener release, not process-exit timing.

    A terminated MCP process can be reaped marginally before its listening
    socket is released.  Polling the same bind check for a small bounded
    interval avoids recording that kernel teardown race as a false residual;
    a port which remains bound is still an acceptance failure.
    """

    if timeout_s <= 0:
        raise ValueError("port-release timeout must be positive")
    deadline = time.monotonic() + float(timeout_s)
    while True:
        if _port_is_free(port):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


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
            row: dict[str, Any] = {"pid": int(entry.name), "cmdline": raw.strip()}
            # A worker deliberately owns a separate process group.  The
            # acceptance runner can therefore not infer ownership from a
            # parent PID after the MCP server has stopped.  Its inherited
            # per-case run id is the narrow ownership proof used below.
            environment = (entry / "environ").read_bytes().split(b"\0")
            for item in environment:
                if item.startswith(b"OPENETA_TUI_RUN_ID="):
                    row["openeta_tui_run_id"] = item.partition(b"=")[2].decode(
                        "utf-8", errors="replace"
                    )
                    break
            result.append(row)
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
        "python_executable": str(
            Path(os.environ.get("OPENETA_PYTHON_EXECUTABLE") or sys.executable).absolute()
        ),
        "python_source": os.environ.get("OPENETA_PYTHON_SOURCE", "direct_python"),
        "python_required_modules": os.environ.get(
            "OPENETA_PYTHON_REQUIRED_MODULES", ""
        ),
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
            "m3": "完成真实 close、native-contact DetachableJoint attach ACK、80 mm lift 物理证明，然后 close。",
            "m4": "使用 Oracle perception 和 contractual fake candidate；仍需通过 M3 的 contact、ACK 和 lift 证明，然后 close。",
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
        "m2": """仅使用真实 MCP/MoveIt 回执执行碰撞检查动作。
执行两轮 A↔B、两轮 open→close→open、一次不可达目标（必须返回 MOTION_PLAN_FAILED）、observe、close。
不得绕过 MoveIt 或把失败动作当作成功。逐一批准所有 human_gated 提示，最后 /quit。
""",
        "m3": """创建 M3 环境，先接近，再执行一次真实 close。
只有回执显示双垫 native contact 与 attached ACK 后才能执行第一段 lift；记录 child-link 的 >=80 mm lift 和 <=10 mm 相对位移。最后 open、detach ACK、close。
""",
        "m4": """创建 M4 Oracle case，实际调用 oracle_perceive 并记录 perception_source=gazebo_oracle 与 contractual fake candidate。
它不能替代 M3：执行真实 close、native contact、attached ACK 和 child-link lift 证明，最后 close。
""",
    }
    prefix = "[automation=scripted_tui; this is not human approval]\n" if mode == SCRIPTED_TUI else ""
    return prefix + rows[milestone]


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


def _process_group_exited(pgid: int) -> bool:
    # ``killpg(..., 0)`` still succeeds for an unreaped zombie leader.  That
    # is not a live residual (and cannot receive a further signal), so inspect
    # the Linux process table first and require a non-zombie member before
    # declaring the group live.  This matters after the MCP server has already
    # sent SIGTERM to its own worker pool.
    proc = Path("/proc")
    observed_group = False
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
                tail = stat.rsplit(")", 1)[1].split()
                state = tail[0]
                process_group = int(tail[2])
            except (IndexError, OSError, ValueError):
                continue
            if process_group != pgid:
                continue
            observed_group = True
            if state != "Z":
                return False
        if observed_group:
            return True
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _terminate_owned_worker_groups(
    *,
    run_id: str,
    before: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]] | None = None,
    timeout_s: float = 15.0,
) -> list[dict[str, Any]]:
    """Terminate only bench-worker groups proven to belong to this case.

    ``BenchWorkerManager`` starts workers with ``start_new_session=True`` so
    a runner cannot rely on the MCP server's process group for cleanup.  Each
    runner supplies a unique ``OPENETA_TUI_RUN_ID`` inherited by those workers.
    We additionally reject PIDs present before the case began and require the
    worker to be its own process-group leader before sending a signal.
    """

    before_pids = {int(item["pid"]) for item in before if isinstance(item.get("pid"), int)}
    if candidates is None:
        candidates = _process_snapshot()
    owned_candidates = [
        row
        for row in candidates
        if isinstance(row.get("pid"), int)
        and row["pid"] not in before_pids
        and "bench_worker" in str(row.get("cmdline") or "")
        and row.get("openeta_tui_run_id") == run_id
    ]
    evidence: list[dict[str, Any]] = []
    for row in owned_candidates:
        pid = int(row["pid"])
        record: dict[str, Any] = {
            "pid": pid,
            "cmdline": str(row.get("cmdline") or ""),
            "run_id": run_id,
            "owned": True,
        }
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            record.update({"state": "already_exited", "group_exited": True})
            evidence.append(record)
            continue
        record["pgid"] = pgid
        if pgid != pid:
            # Never signal an unrelated shared group.  A worker launched by
            # BenchWorkerManager must be the leader because it uses
            # start_new_session=True; otherwise cleanup must fail closed.
            record.update(
                {
                    "state": "refused_non_leader_group",
                    "group_exited": False,
                }
            )
            evidence.append(record)
            continue
        try:
            os.killpg(pgid, signal.SIGTERM)
            record["termination_signal"] = "SIGTERM"
        except ProcessLookupError:
            record.update({"state": "already_exited", "group_exited": True})
            evidence.append(record)
            continue
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not _process_group_exited(pgid):
            time.sleep(0.05)
        if not _process_group_exited(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
                record["escalation_signal"] = "SIGKILL"
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not _process_group_exited(pgid):
                time.sleep(0.05)
        record["group_exited"] = _process_group_exited(pgid)
        record["state"] = "exited" if record["group_exited"] else "residual"
        evidence.append(record)
    return evidence


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


def scripted_tui_input(paths: CasePaths) -> str:
    """Return keystrokes forwarded to the real PTY for scripted automation."""

    commands: list[str] = []
    if paths.root.parent.name == "m0":
        commands.extend(("/config", "/tools", "/session", "/memory all --json"))
    # prompt_toolkit submits every newline independently.  The automation
    # provenance and the actual task must therefore be one physical prompt:
    # sending ``[automation=scripted_tui]`` alone lets a planner start generic
    # work before it receives the required M0/M3 task.  Preserve the complete
    # instruction text but collapse formatting whitespace for one real PTY
    # submission after the console-evidence slash commands.
    task = " ".join(paths.instructions.read_text(encoding="utf-8").split())
    if not task:
        raise AcceptanceError("TUI_NOT_READY: scripted task instructions are empty")
    commands.extend((task, "/quit"))
    return "\n".join(commands) + "\n"


def run_case(repo: Path, paths: CasePaths, allocation: Allocation) -> int:
    milestone = paths.root.parent.name
    scripted = paths.root.name == SCRIPTED_TUI
    env = os.environ.copy()
    # The wrapper has already sourced Jazzy and the workspace overlay.  Keep
    # their generated Python paths: BenchWorkerManager deliberately removes
    # the repository path before launching its Gazebo child, but preserves
    # these ROS paths so rclpy and ros_gz_sim remain importable there.
    inherited_python_path = env.get("PYTHONPATH", "")
    python_path = os.pathsep.join(
        item for item in (str(repo), inherited_python_path) if item
    )
    env.update(
        {
            "PYTHONPATH": python_path,
            "ROS_DOMAIN_ID": str(allocation.ros_domain_id),
            "GZ_PARTITION": allocation.gz_partition,
            "MCP_PORT": str(allocation.port),
            "ROS_LOCALHOST_ONLY": "1",
            # This is inherited by BenchWorkerManager's independently
            # sessioned children and is the only ownership marker accepted by
            # runner-side worker cleanup.
            "OPENETA_TUI_RUN_ID": allocation.run_id,
            # BenchWorkerManager drains the Gazebo worker's stdout/stderr to
            # this case-local directory.  Without this explicit path, MCP
            # server logs omit launch/ROS diagnostics needed to explain a
            # fail-closed live case.
            "OPENETA_WORKER_LOG_DIR": str(paths.root / "worker-logs"),
            "OPENETA_SUPERVISION_PROFILE": SCRIPTED_TUI if scripted else "human_gated",
            "OPENETA_SCRIPTED_TUI": "1" if scripted else "0",
            # M4 explicitly selects the existing Gazebo Oracle tool.  The
            # candidate it emits is a contractual fixture, never a model
            # prediction and never an alternative M3 grasp gate.
            "OPENETA_PERCEPTION_PROFILE": "oracle" if milestone == "m4" else "",
            "OPENETA_M4_CONTRACTUAL_FAKE_CANDIDATE": "1" if milestone == "m4" else "0",
            "RMW_IMPLEMENTATION": env.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        }
    )
    python = Path(os.environ.get("OPENETA_PYTHON_EXECUTABLE") or sys.executable).absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise AcceptanceError("TUI_NOT_READY: selected Python executable is unavailable")
    log = paths.mcp_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(python),
            "-u",
            "-m",
            "sim.mcp_server.server",
            "--transport",
            "sse",
            "--port",
            str(allocation.port),
        ],
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
    receipt = _json_load(paths.receipt)
    tui_code = 2
    mcp_termination_error = ""
    worker_candidates: list[dict[str, Any]] = []
    try:
        _wait_ready(allocation.port, process)
        print(f"\n=== {paths.root.name} ===")
        print(paths.instructions.read_text(encoding="utf-8"))
        command = f"{shlex.quote(str(python))} -m agent.cli.openeta_cli"
        if shutil.which("script") is None:
            raise AcceptanceError("TUI_NOT_READY: util-linux script is missing")
        scripted_input: str | None = None
        if scripted:
            # Feed these through the actual PTY TUI rather than shortcut mode,
            # so M0's operator-console evidence lands in the transcript.
            scripted_input = scripted_tui_input(paths)
        completed = subprocess.run(
            ["script", "--flush", "--return", "--command", command, str(paths.transcript)],
            cwd=paths.root,
            env=env,
            check=False,
            input=scripted_input,
            text=scripted_input is not None,
        )
        tui_code = int(completed.returncode)
    finally:
        # Capture the run-id-marked worker while the MCP server still owns its
        # bookkeeping.  Its own shutdown may stop the group before runner-side
        # cleanup executes; keeping this snapshot makes cleanup evidence prove
        # both discovery and exit rather than reporting an ambiguous empty set.
        worker_candidates = _process_snapshot()
        try:
            _terminate_owned_group(process, pgid)
        except Exception as exc:  # noqa: BLE001 - evidence must survive cleanup failure.
            mcp_termination_error = f"{type(exc).__name__}: {exc}"
        finally:
            log.close()
    worker_groups = _terminate_owned_worker_groups(
        run_id=allocation.run_id,
        before=receipt["preexisting_processes"],
        candidates=worker_candidates,
    )
    after = _process_snapshot()
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
        "mcp_termination_error": mcp_termination_error,
        "port_free": _wait_for_free_port(allocation.port),
        "owned_worker_groups": worker_groups,
        "owned_worker_groups_exited": all(
            item.get("group_exited") is True for item in worker_groups
        ),
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
    if mcp_termination_error:
        raise AcceptanceError(f"MCP_CLEANUP_FAILED: {mcp_termination_error}")
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
    """Extract durable, full-fidelity tool calls from session envelopes.

    An episode also records compact ``payload.action.tool_calls`` summaries for
    planner context.  Those summaries deliberately omit heavy ``outputs`` such
    as the MCP request/response/receipt chain.  Recursively walking every
    ``action`` envelope therefore selected the summary before the raw command
    and made a real formal run look uncorrelated.  Formal verification must
    consume only the original command records: ``payload.command.tool_calls``.

    Some older/runtime tool-execution records contain a direct
    ``payload.tool_calls`` array instead.  It is used only when the case has no
    command records at all, so one physical call cannot be double-counted by
    its command and per-tool trace entries.
    """

    action_command_calls: list[Mapping[str, Any]] = []
    command_calls: list[Mapping[str, Any]] = []
    direct_calls: list[Mapping[str, Any]] = []

    def append_raw(
        destination: list[Mapping[str, Any]],
        candidate: Any,
    ) -> None:
        if not isinstance(candidate, Sequence) or isinstance(
            candidate, (str, bytes, bytearray)
        ):
            return
        for call in candidate:
            if not isinstance(call, Mapping):
                continue
            name = str(call.get("name") or call.get("tool_name") or "")
            # ``parameters`` and/or the pipeline ``kind`` are present on the
            # source command.  Compact action summaries intentionally have
            # neither, even though they retain name/status/result.
            if (
                name
                and isinstance(call.get("result"), Mapping)
                and ("parameters" in call or "kind" in call)
            ):
                destination.append(call)

    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        command = payload.get("command")
        if isinstance(command, Mapping):
            destination = (
                action_command_calls
                if event.get("event_type") == "action"
                else command_calls
            )
            append_raw(destination, command.get("tool_calls"))
        append_raw(direct_calls, payload.get("tool_calls"))
    # The runtime writes the same complete command to ``pipeline_plan`` before
    # it writes its durable ``action`` event.  Prefer the latter so retries and
    # M2's repeated motions are counted exactly once rather than once per trace
    # representation.
    return action_command_calls or command_calls or direct_calls


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
        nested = item.get("details")
        nested_profile = nested.get("profile") if isinstance(nested, Mapping) else ""
        profile = str(
            item.get("supervision_profile")
            or item.get("profile")
            or item.get("policy")
            or nested_profile
            or ""
        ).lower()
        source = str(item.get("source") or "").lower()
        decision = str(item.get("decision") or item.get("status") or "").lower()
        approved = (
            item.get("approved") is True
            or item.get("allowed") is True
            or decision in {"approved", "executed", "allow"}
        )
        gated = (
            profile == "human_gated"
            or source == "human"
            or _contains(item, "profile", "human_gated")
        )
        if gated and approved:
            return True
    return False


def _scripted_approved(call: Mapping[str, Any]) -> bool:
    """Recognise automation honestly; it never counts as human approval."""

    if str(call.get("name") or call.get("tool_name") or "") not in MUTATING_TOOLS:
        return True
    for item in _walk(call):
        if not isinstance(item, Mapping):
            continue
        nested = item.get("details")
        nested_profile = nested.get("profile") if isinstance(nested, Mapping) else ""
        profile = str(
            item.get("supervision_profile") or item.get("profile") or nested_profile or ""
        ).lower()
        source = str(item.get("source") or "").lower()
        if (
            (profile == SCRIPTED_TUI or source == SCRIPTED_TUI)
            and (item.get("allowed") is True or item.get("approved") is True)
        ):
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
        if cleanup.get("owned_worker_groups_exited") is not True:
            errors.append("owned bench-worker process groups lack clean-exit evidence")
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
    remaining_names = iter(names)
    if not all(
        any(name == expected_name for name in remaining_names)
        for expected_name in expected
    ):
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


def _verify_m2(calls: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
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
    return errors


def _mapping_with(node: Any, key: str) -> Mapping[str, Any] | None:
    """Return the first mapping that contains a mapping-valued ``key``."""

    for item in _walk(node):
        value = item.get(key) if isinstance(item, Mapping) else None
        if isinstance(value, Mapping):
            return value
    return None


def _mcp_response_payloads(
    calls: Sequence[Mapping[str, Any]],
    paths: CasePaths,
    *,
    required_tools: frozenset[str] = SIX_SIMULATOR_TOOLS,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Validate every required simulator MCP call (including M4 Oracle).

    The proxy emits a local MCP request descriptor, a response file rooted in
    this case, and an environment receipt carrying the same request id.  A
    create call contains two linked RPCs (``create_env`` then its automatic
    ``reset_env``); all other simulator tools contain exactly one.  This is
    intentionally strict: a formal case cannot pass with a raw response, a
    trace-only request, or a receipt associated with a different RPC.
    """

    payloads: list[Mapping[str, Any]] = []
    errors: list[str] = []
    root = paths.root.resolve()
    simulator_calls = [
        call
        for call in calls
        if str(call.get("name") or call.get("tool_name") or "") in required_tools
    ]
    if not simulator_calls:
        errors.append("formal case has no simulator tool calls")
    for call in simulator_calls:
        agent_tool = str(call.get("name") or call.get("tool_name") or "")
        outputs = _mapping_with(_result(call), "outputs")
        entries = outputs.get("mcp_calls") if isinstance(outputs, Mapping) else None
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
            errors.append(f"{agent_tool} lacks MCP request/response/receipt evidence")
            continue
        expected_count = 2 if agent_tool == "create_simulator_env" else 1
        if len(entries) != expected_count:
            errors.append(
                f"{agent_tool} requires {expected_count} correlated MCP RPC evidence entries"
            )
            continue
        for index, entry in enumerate(entries, 1):
            if not isinstance(entry, Mapping):
                errors.append(f"{agent_tool} MCP evidence {index} is not an object")
                continue
            request = entry.get("request")
            response = entry.get("response")
            receipt = entry.get("environment_receipt")
            if not isinstance(request, Mapping):
                errors.append(f"{agent_tool} MCP evidence {index} lacks request descriptor")
                continue
            request_id = str(request.get("request_id") or "")
            remote_tool = str(request.get("tool") or "")
            arguments = request.get("arguments")
            if not request_id or not remote_tool or not isinstance(arguments, Mapping):
                errors.append(f"{agent_tool} MCP evidence {index} has malformed request descriptor")
                continue
            if not isinstance(response, Mapping) or not isinstance(response.get("response_path"), str):
                errors.append(f"MCP {remote_tool} response artifact is missing")
                continue
            if str(response.get("request_id") or "") != request_id or str(response.get("tool") or "") != remote_tool:
                errors.append(f"MCP {remote_tool} response is not correlated to its request")
                continue
            if not isinstance(receipt, Mapping):
                errors.append(f"MCP {remote_tool} response has no correlated environment receipt")
                continue
            if (
                str(receipt.get("mcp_request_id") or "") != request_id
                or str(receipt.get("remote_tool") or "") != remote_tool
            ):
                errors.append(f"MCP {remote_tool} receipt is not correlated to its request")
                continue
            for key, receipt_key in (("handle", "handle"), ("session_id", "simulator_session_id")):
                request_value = str(request.get("arguments", {}).get(key) or "")
                response_value = str(response.get(key) or "")
                receipt_value = str(receipt.get(receipt_key) or "")
                if response_value != receipt_value:
                    errors.append(f"MCP {remote_tool} {key} does not match response/receipt")
                if request_value and (request_value != response_value or request_value != receipt_value):
                    errors.append(f"MCP {remote_tool} {key} does not match request/response/receipt")
            artifact = Path(response["response_path"]).resolve()
            try:
                artifact.relative_to(root)
            except ValueError:
                errors.append(f"MCP {remote_tool} response artifact escapes the case directory")
                continue
            try:
                value = _json_load(artifact)
            except (OSError, ValueError) as exc:
                errors.append(f"MCP {remote_tool} response artifact is unreadable: {exc}")
                continue
            if not isinstance(value, Mapping):
                errors.append(f"MCP {remote_tool} response artifact is not an object")
                continue
            payloads.append(value)
    if not paths.mcp_log.is_file() or not paths.mcp_log.read_text(encoding="utf-8", errors="replace").strip():
        errors.append("MCP server log is missing or empty")
    return payloads, errors


def _verify_m3(
    calls: Sequence[Mapping[str, Any]],
    payloads: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not any(_contains(call, "env_id", ENV_IDS["m3"]) for call in calls):
        errors.append("M3 environment identity missing")
    records = [
        item for payload in payloads for item in _walk(payload)
        if isinstance(item, Mapping) and item.get("schema_version") == "openeta.m3.detachable_joint.v1"
    ]
    if not any(item.get("reason_code") == "M3_TARGET_HELD" and item.get("grasp_confirmed") is True for item in records):
        errors.append("M3 child-link held proof missing")
    held = [item for item in records if item.get("reason_code") == "M3_TARGET_HELD"]
    if not any(
        isinstance(item.get("evidence"), Mapping)
        and isinstance(item["evidence"].get("lift_m"), (int, float))
        and float(item["evidence"]["lift_m"]) >= 0.080
        and isinstance(item["evidence"].get("capture_relative_translation_m"), (int, float))
        and float(item["evidence"]["capture_relative_translation_m"]) <= 0.010
        for item in held
    ):
        errors.append("M3 numeric child-link lift/relative-translation proof missing")
    gates = [
        item for payload in payloads for item in _walk(payload)
        if isinstance(item, Mapping) and "left_sample_count" in item and "right_sample_count" in item
    ]
    if not any(
        item.get("accepted") is True
        and isinstance(item.get("evidence"), Mapping)
        and item["evidence"].get("target_id") == "m3_target"
        and isinstance(item.get("left_sample_count"), int) and item["left_sample_count"] >= 3
        and isinstance(item.get("right_sample_count"), int) and item["right_sample_count"] >= 3
        and isinstance(item.get("left_span_s"), (int, float)) and float(item["left_span_s"]) >= 0.100
        and isinstance(item.get("right_span_s"), (int, float)) and float(item["right_span_s"]) >= 0.100
        for item in gates
    ):
        errors.append("M3 native bilateral contact evidence missing")
    if not any(_contains(payload, "state", "attached") for payload in payloads):
        errors.append("M3 attached ACK evidence missing")
    if not any(_contains(payload, "state", "detached") for payload in payloads):
        errors.append("M3 detached ACK evidence missing")
    if not any(str(call.get("name") or call.get("tool_name")) == "close_simulator_env" for call in calls):
        errors.append("M3 environment close missing")
    return errors


def _verify_m4(
    calls: Sequence[Mapping[str, Any]],
    payloads: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors = _verify_m3(calls, payloads)
    oracle_calls = [
        call
        for call in calls
        if str(call.get("name") or call.get("tool_name") or "") == "oracle_perceive"
    ]
    if not oracle_calls:
        errors.append("M4 has no executed oracle_perceive tool output")
        return errors
    valid_oracle = False
    for call in oracle_calls:
        result = _result(call)
        candidate_values = _values(result, "fake_grasp_candidate")
        for candidate in candidate_values:
            if not isinstance(candidate, Mapping):
                continue
            if (
                _successful(call)
                and _contains(result, "perception_source", "gazebo_oracle")
                and candidate.get("schema_version")
                == "openeta.m4.contractual_fake_grasp_candidate.v1"
                and candidate.get("kind") == "contractual_fake_grasp_candidate"
                and candidate.get("is_model_prediction") is False
                and candidate.get("perception_source") == "gazebo_oracle"
            ):
                valid_oracle = True
    if not valid_oracle:
        errors.append("M4 Oracle output lacks a correctly labelled contractual fake candidate")
    return errors


def verify_case(
    paths: CasePaths,
    milestone: str,
    mode: str,
) -> dict[str, Any]:
    try:
        events, trace_paths = _load_trace_events(paths.trace_root)
        calls = _tool_calls(events)
        errors = _base_errors(paths, events)
        payloads: list[Mapping[str, Any]] = []
        if mode in {DETERMINISTIC, SCRIPTED_TUI}:
            required_mcp_tools = (
                SIX_SIMULATOR_TOOLS | frozenset({"oracle_perceive"})
                if milestone == "m4"
                else SIX_SIMULATOR_TOOLS
            )
            payloads, mcp_errors = _mcp_response_payloads(
                calls,
                paths,
                required_tools=required_mcp_tools,
            )
            errors.extend(mcp_errors)
        for call in calls:
            name = str(call.get("name") or call.get("tool_name") or "")
            approved = _scripted_approved(call) if mode == SCRIPTED_TUI else _human_approved(call)
            if name in MUTATING_TOOLS and not approved:
                profile = SCRIPTED_TUI if mode == SCRIPTED_TUI else "human_gated"
                errors.append(f"{name} lacks explicit {profile} approval evidence")
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
            errors.extend(_verify_m2(calls))
        elif milestone == "m3":
            errors.extend(_verify_m3(calls, payloads))
        elif milestone == "m4":
            errors.extend(_verify_m4(calls, payloads))
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


def assemble_report(
    run_root: Path,
    *,
    formal_mode: str = DETERMINISTIC,
) -> dict[str, Any]:
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
        backend_paths = case_paths(run_root, milestone, formal_mode)
        backend = verify_case(
            backend_paths,
            milestone,
            formal_mode,
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
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--scripted-tui",
        action="store_true",
        help="Run real PTY TUI cases with explicit scripted_tui approvals, never human approval.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    run_root = _new_run_root(repo, args.run_root)
    formal_mode = SCRIPTED_TUI if args.scripted_tui else DETERMINISTIC
    if args.verify_only:
        report = assemble_report(run_root, formal_mode=formal_mode)
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
            modes = (
                (SCRIPTED_TUI,)
                if args.scripted_tui
                else (DETERMINISTIC,)
                if milestone in {"m0", "m3", "m4"}
                else (DETERMINISTIC, AUTONOMY)
            )
            for mode in modes:
                allocation = allocate(f"{milestone}-{mode}", occupied)
                occupied.add(allocation.ros_domain_id)
                paths = prepare_case(repo, run_root, milestone, mode, allocation)
                if args.prepare_only:
                    continue
                code = run_case(repo, paths, allocation)
                if code == 130:
                    return 130
                if mode == formal_mode:
                    gate = verify_case(
                        paths,
                        milestone,
                        mode,
                    )
                    _json_dump(paths.root / "verification.json", gate)
                    if gate["status"] != "passed":
                        report = assemble_report(
                            run_root,
                            formal_mode=formal_mode,
                        )
                        _json_dump(run_root / "acceptance-report.json", report, exclusive=True)
                        return report_exit_code(report)
        if args.prepare_only:
            print(run_root)
            return 0
        report = assemble_report(run_root, formal_mode=formal_mode)
        report_path = run_root / "acceptance-report.json"
        _json_dump(report_path, report, exclusive=True)
        report_path.chmod(0o444)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report_exit_code(report)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
