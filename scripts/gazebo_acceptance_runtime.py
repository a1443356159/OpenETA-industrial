"""Shared isolation, PTY, evidence and cleanup runtime for Gazebo acceptance."""


from __future__ import annotations


from dataclasses import dataclass


from datetime import datetime, UTC


import hashlib


import json


import os


from pathlib import Path


import platform


import re


import shutil


import shlex


import signal


import socket


import subprocess


import sys


import time


from typing import Any, Iterable, Mapping, Sequence


from urllib.parse import urlparse


import uuid


from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    PlannerBackendRequest,
    list_openai_compatible_models,
)


from agent.backends.provider_config import (
    PlannerProviderConfig,
    load_planner_provider_config,
)


from agent.runtime.actions import PipelineStatus


SCRIPTED_TUI = "scripted_tui"
HUMAN_TUI = "human_tui"


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


PROVIDER_PREFLIGHT_FILENAME = "provider-preflight.json"


_PROVIDER_ENV_PREFIX = "OPENETA_LLM_"


# A formal profile may tune bounded request behaviour without replacing the
# repository-selected primary provider, endpoint, model, or credentials.  It
# may also name one evaluated fallback model on that same endpoint.  These are
# deliberately the only provider settings accepted from ``extra_environment``.
_PROVIDER_PROFILE_KEYS = frozenset(
    {
        "OPENETA_LLM_TIMEOUT_S",
        "OPENETA_LLM_MAX_ATTEMPTS",
        "OPENETA_LLM_RETRY_BACKOFF_S",
        "OPENETA_LLM_MAX_TOKENS",
        "OPENETA_LLM_FALLBACK_MODEL",
        "OPENETA_LLM_FALLBACK_TIMEOUT_S",
    }
)


_OPERATOR_GUI_ENV = "OPENETA_GAZEBO_OPERATOR_GUI"


_MCP_EVIDENCE_REQUEST_ID = "__openeta_acceptance_mcp_request_id"


_MCP_EVIDENCE_AGENT_TOOL = "__openeta_acceptance_agent_tool"


class AcceptanceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Allocation:
    ros_domain_id: int
    gz_partition: str
    port: int
    run_id: str
    # Recorded only for live coordinator allocations.  Unit fixtures may use
    # the deterministic allocator without probing a ROS installation.
    candidate_domain_preflight: Mapping[str, Any] | None = None


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


def _candidate_domain_preflight(domain: int) -> dict[str, Any]:
    """Return fail-closed evidence that a candidate ROS domain is unused.

    A case must never reuse a pre-existing graph or ros2cli daemon just
    because its numeric domain is not in this invocation's local ``occupied``
    set.  The isolation probe uses a short-lived rclpy context rather than
    ``ros2`` CLI graph listing, so preflight itself does not create a daemon.
    """

    try:
        from extensions.gazebo.ros2_ws.acceptance_isolation import (
            candidate_domain_evidence,
        )

        evidence = candidate_domain_evidence(domain)
    except Exception as exc:
        return {
            "state": "INCONCLUSIVE",
            "ok": None,
            "reason_code": "ROS_DOMAIN_PREFLIGHT_UNAVAILABLE",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    if not isinstance(evidence, Mapping):
        return {
            "state": "INCONCLUSIVE",
            "ok": None,
            "reason_code": "ROS_DOMAIN_PREFLIGHT_MALFORMED",
        }
    return dict(evidence)


def allocate(
    case_name: str,
    occupied_domains: Iterable[int] = (),
    *,
    preflight: bool = False,
) -> Allocation:
    occupied = set(occupied_domains) | set(PROTECTED_DOMAINS)
    preflight_failures: dict[int, Mapping[str, Any]] = {}
    domain: int | None = None
    selected_preflight: Mapping[str, Any] | None = None
    for candidate in DOMAIN_CANDIDATES:
        if candidate in occupied:
            continue
        if preflight:
            evidence = _candidate_domain_preflight(candidate)
            if evidence.get("state") != "PASSED":
                preflight_failures[candidate] = evidence
                continue
            selected_preflight = evidence
        domain = candidate
        break
    if domain is None:
        detail = json.dumps(preflight_failures, sort_keys=True)[:2000]
        raise AcceptanceError(
            "no isolated ROS_DOMAIN_ID is available"
            + (f" after preflight: {detail}" if preflight else "")
        )
    token = uuid.uuid4().hex[:12]
    return Allocation(
        ros_domain_id=domain,
        gz_partition=f"openeta-tui-{case_name}-{token}",
        port=_free_port(),
        run_id=token,
        candidate_domain_preflight=selected_preflight,
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


def _is_snapshot_candidate_argv(argv: Sequence[str]) -> bool:
    """Select actual ROS/Gazebo workload processes, never shell prose.

    The prior substring scan recorded a diagnostic command such as
    ``bash -c 'ros2 launch ...'`` as a pre-existing workload.  Its natural
    exit then made cleanup claim that our run had changed a process it never
    owned.  Inspect the null-separated argv instead and accept only concrete
    runtime executables or their Python entry points.
    """

    if not argv:
        return False
    # VirtualGL's ``vglrun`` helper can expose the launched Gazebo GUI as one
    # packed argv[0] string (``gz sim -g ...``) rather than ordinary null-
    # separated arguments. Accept only that exact executable prefix; a shell
    # command remains multi-argv and is rejected below. This lets a GPU GUI
    # carrying the case run-id participate in the normal ownership cleanup.
    if len(argv) == 1 and argv[0].startswith("gz sim "):
        argv = shlex.split(argv[0])
    executable = Path(argv[0]).name
    if executable in {"bash", "dash", "sh", "zsh", "fish"}:
        return False
    names = {Path(argument).name for argument in argv if argument}
    if "bench_worker.py" in names:
        return True
    if "ros2" in names and "launch" in argv:
        return True
    if executable == "gz" and "sim" in argv[1:]:
        return True
    # On this Gazebo installation /usr/bin/gz is a Ruby entry point.  Record
    # a real Ruby->gz sim process, but not an arbitrary shell command that
    # happens to quote those two words.
    if executable == "ruby" and "gz" in names and "sim" in argv:
        return True
    return executable in {"move_group", "parameter_bridge", "image_bridge"}


def _process_snapshot() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    proc = Path("/proc")
    for entry in proc.iterdir() if proc.is_dir() else ():
        if not entry.name.isdigit():
            continue
        try:
            argv = [
                item.decode("utf-8", errors="replace")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            if not _is_snapshot_candidate_argv(argv):
                continue
            row: dict[str, Any] = {"pid": int(entry.name), "cmdline": " ".join(argv)}
            # PID equality alone cannot prove continuity after a multi-minute
            # acceptance run because Linux may reuse a departed PID. Field 22
            # of /proc/<pid>/stat is the process start time in clock ticks.
            stat_tail = (
                (entry / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
            )
            row["start_time_ticks"] = int(stat_tail.split()[19])
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


def _preexisting_process_continuity(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Separate protected external processes from independent acceptance jobs.

    A process carrying another OPENETA_TUI_RUN_ID owns its own bounded
    lifecycle and may finish naturally while this case runs. Unmarked
    ROS/Gazebo processes are operator/external workloads and must remain. The
    current case cleanup is still restricted to its unique run id elsewhere.
    """

    def _same_process(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        if left.get("pid") != right.get("pid"):
            return False
        left_start = left.get("start_time_ticks")
        right_start = right.get("start_time_ticks")
        return (
            left_start is None
            or right_start is None
            or left_start == right_start
        )

    missing = [
        dict(item)
        for item in before
        if not any(_same_process(item, row) for row in after)
    ]
    missing_managed = [
        item for item in missing if str(item.get("openeta_tui_run_id") or "")
    ]
    missing_unmanaged = [
        item for item in missing if not str(item.get("openeta_tui_run_id") or "")
    ]
    return {
        "preexisting_process_snapshot_unchanged": not missing,
        "preexisting_unmanaged_process_snapshot_unchanged": not missing_unmanaged,
        "preexisting_missing_processes": missing,
        "preexisting_missing_managed_processes": missing_managed,
        "preexisting_missing_unmanaged_processes": missing_unmanaged,
    }


def _owned_process_residuals(
    candidates: Sequence[Mapping[str, Any]], *, run_id: str
) -> list[dict[str, Any]]:
    """Return only live processes with this case's explicit ownership marker.

    A broad ROS/Gazebo command-line snapshot is useful for recording the
    preexisting environment, but it is not an ownership proof: a diagnostic
    shell can legitimately mention ``ros2`` or ``gz sim`` in its arguments.
    Runner cleanup may never label or signal such a process as its own.
    """

    return [
        dict(item)
        for item in candidates
        if item.get("openeta_tui_run_id") == run_id
    ]


def environment_receipt(
    repo: Path,
    allocation: Allocation,
    *,
    case_name: str,
    before: list[dict[str, Any]],
    capture_protected: bool = True,
) -> dict[str, Any]:
    overlay_prefix = Path(
        os.environ.get("OPENETA_GAZEBO_OVERLAY")
        or repo / "extensions/gazebo/ros2_ws/install"
    )
    overlay = overlay_prefix / "setup.bash"
    payload: dict[str, Any] = {
        "schema_version": "openeta.gazebo_environment_receipt.v1",
        "trusted": True,
        "case": case_name,
        "captured_at": datetime.now(UTC).isoformat(),
        "ros_domain_id": allocation.ros_domain_id,
        "gz_partition": allocation.gz_partition,
        # Public ownership token for optional operator-side processes (for
        # example the GPU Gazebo GUI).  Processes that inherit this exact
        # marker are folded into the same bounded cleanup proof as the MCP
        # worker, so a visible GUI cannot race partition verification.
        "run_id": allocation.run_id,
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
    if allocation.candidate_domain_preflight is not None:
        payload["candidate_domain_preflight"] = dict(
            allocation.candidate_domain_preflight
        )
    if capture_protected:
        from extensions.gazebo.ros2_ws.acceptance_isolation import probe_ros_graph

        payload["protected_ros_graphs"] = {
            str(domain): probe_ros_graph(domain) for domain in sorted(PROTECTED_DOMAINS)
        }
    return seal_environment_receipt(payload)


def seal_environment_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a receipt hash covering all fields present at the call boundary."""

    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
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
    preflight = receipt.get("candidate_domain_preflight")
    if preflight is not None and (
        not isinstance(preflight, Mapping) or preflight.get("state") != "PASSED"
    ):
        errors.append("case ROS domain preflight did not prove an empty candidate")
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


def _terminate_owned_group(
    process: subprocess.Popen[Any], pgid: int, *, label: str = "MCP"
) -> None:
    if process.poll() is not None:
        return
    if os.getpgid(process.pid) != pgid:
        raise AcceptanceError(f"refusing cleanup: recorded {label} PGID changed")
    os.killpg(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        process.wait(timeout=5)


def _operator_gui_requested(environment: Mapping[str, str]) -> bool:
    return str(environment.get(_OPERATOR_GUI_ENV) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _launch_operator_gui(
    repo: Path,
    paths: CasePaths,
    environment: Mapping[str, str],
) -> tuple[subprocess.Popen[Any], int, Any]:
    """Launch one case-owned GPU GUI in an independent process group."""

    script = repo / "scripts" / "run_gazebo_gpu_gui.sh"
    if not script.is_file() or not os.access(script, os.X_OK):
        raise AcceptanceError(
            f"TUI_NOT_READY: operator GUI launcher is unavailable: {script}"
        )
    gui_log = (paths.root / "operator-gui.log").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [str(script)],
            cwd=repo,
            env=dict(environment),
            stdout=gui_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        pgid = os.getpgid(process.pid)
        # Catch missing VNC / VirtualGL prerequisites before the planner starts
        # an otherwise unobservable formal episode. Gazebo discovery itself
        # remains asynchronous until create_simulator_env starts the server.
        time.sleep(0.25)
        if process.poll() is not None:
            raise AcceptanceError(
                "TUI_NOT_READY: case-owned Gazebo operator GUI exited during startup"
            )
        return process, pgid, gui_log
    except Exception:
        gui_log.close()
        raise


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


def _terminate_owned_residual_groups(
    *,
    run_id: str,
    before: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]] | None = None,
    timeout_s: float = 15.0,
) -> list[dict[str, Any]]:
    """Terminate remaining process groups carrying this case's exact run id.

    ROS launch and Gazebo may create sessions below the bench worker, so
    stopping the worker group alone does not necessarily stop those adopted
    descendants.  The inherited ``OPENETA_TUI_RUN_ID`` is the ownership
    boundary.  Never signal the coordinator's own group or a process that was
    already present when the case began.
    """

    before_pids = {int(item["pid"]) for item in before if isinstance(item.get("pid"), int)}
    if candidates is None:
        candidates = _process_snapshot()
    owned = [
        row
        for row in candidates
        if isinstance(row.get("pid"), int)
        and row["pid"] not in before_pids
        and row.get("openeta_tui_run_id") == run_id
    ]
    groups: dict[int, list[Mapping[str, Any]]] = {}
    evidence: list[dict[str, Any]] = []
    current_pgid = os.getpgrp()
    for row in owned:
        pid = int(row["pid"])
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            evidence.append(
                {
                    "pgid": None,
                    "member_pids": [pid],
                    "run_id": run_id,
                    "owned": True,
                    "state": "already_exited",
                    "group_exited": True,
                }
            )
            continue
        groups.setdefault(pgid, []).append(row)

    for pgid, rows in sorted(groups.items()):
        member_pids = sorted(int(row["pid"]) for row in rows)
        record: dict[str, Any] = {
            "pgid": pgid,
            "member_pids": member_pids,
            "run_id": run_id,
            "owned": True,
        }
        if pgid == current_pgid:
            record.update({"state": "refused_coordinator_group", "group_exited": False})
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


def _partition_cleanup(
    partition: str,
    *,
    timeout_s: float = 20.0,
    poll_interval_s: float = 0.5,
) -> dict[str, Any]:
    """Wait for Gazebo transport discovery to forget exited participants.

    ``gz topic -l`` can retain a GUI or server's advertised topics for several
    seconds after every owning process has exited.  Treat that bounded discovery
    lease as settling, while remaining fail-closed when topics are still present
    at the deadline or the probe itself is inconclusive.
    """

    env = os.environ.copy()
    env["GZ_PARTITION"] = partition
    started = time.monotonic()
    deadline = started + max(0.0, float(timeout_s))
    attempts = 0
    observed_topic_sets: list[list[str]] = []
    while True:
        attempts += 1
        try:
            result = subprocess.run(
                ["gz", "topic", "-l"], env=env, capture_output=True, text=True,
                timeout=8, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "state": "INCONCLUSIVE",
                "reason": f"{type(exc).__name__}: {exc}",
                "settle_attempts": attempts,
            }
        topics = sorted(
            line.strip() for line in result.stdout.splitlines() if line.strip()
        )
        if result.returncode:
            return {
                "state": "INCONCLUSIVE", "returncode": result.returncode,
                "stderr": result.stderr[:500], "settle_attempts": attempts,
            }
        if not observed_topic_sets or observed_topic_sets[-1] != topics:
            observed_topic_sets.append(topics)
        elapsed_s = round(time.monotonic() - started, 3)
        if not topics:
            return {
                "state": "PASSED",
                "topics": [],
                "settle_attempts": attempts,
                "settle_duration_s": elapsed_s,
                "observed_topic_sets": observed_topic_sets,
            }
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            return {
                "state": "FAILED",
                "topics": topics,
                "settle_attempts": attempts,
                "settle_duration_s": elapsed_s,
                "observed_topic_sets": observed_topic_sets,
            }
        time.sleep(min(max(0.0, poll_interval_s), remaining_s))


def scripted_tui_input(paths: CasePaths) -> str:
    """Return keystrokes forwarded to the real PTY for scripted automation."""

    # prompt_toolkit submits every newline independently.  The automation
    # provenance and the actual task must therefore be one physical prompt:
    # sending ``[automation=scripted_tui]`` alone lets a planner start generic
    # work before it receives the complete task. Preserve the instruction text
    # but collapse formatting whitespace for one real PTY submission.
    task = " ".join(paths.instructions.read_text(encoding="utf-8").split())
    if not task:
        raise AcceptanceError("TUI_NOT_READY: scripted task instructions are empty")
    commands = [task, "/quit"]
    return "\n".join(commands) + "\n"


def _scripted_tui_initial_input(paths: CasePaths) -> str:
    """Return console setup plus the task, withholding `/quit` until completion.

    Prompt-toolkit consumes stdin while a planner turn is in flight.  Sending
    `/quit` in the same initial pipe can therefore lose it before the episode
    reaches its final response.  Keep the public scripted-input contract for
    evidence/tests, but make process driving explicitly sequential.
    """

    lines = scripted_tui_input(paths).splitlines()
    if not lines or lines[-1] != "/quit":
        raise AcceptanceError("TUI_NOT_READY: scripted input lacks a final /quit")
    return "\n".join(lines[:-1]) + "\n"


def _scripted_tui_trace_state(paths: CasePaths) -> str:
    """Return the newest durable scripted-TUI state from the trace tail.

    A scripted run has no human operator.  ``ask_human`` is therefore a
    terminal gate for the driver, rather than a prompt to which an automation
    command could accidentally become an answer.  Read JSONL events in reverse
    order so a later human answer or episode result overrides an older pause.
    """

    state = "running"
    traces = sorted(
        paths.trace_root.glob("sessions/*/trace.jsonl"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    for trace in traces:
        try:
            stream = trace.open("r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with stream:
            # Observation-bearing JSONL records can exceed 128 KiB. Stream
            # complete records instead of seeking into the middle of the last
            # line, which could hide an unattended ask_human gate.
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, Mapping):
                    continue
                event_type = event.get("event_type")
                if event_type == "episode_result":
                    payload = event.get("payload")
                    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
                    if isinstance(metadata, Mapping) and (
                        metadata.get("waiting_for_human") is True
                        or metadata.get("stop_reason") == "ask_human"
                    ):
                        state = "human_input_required"
                    else:
                        state = "completed"
                    continue
                if event_type in {"human_answer", "user_message"}:
                    state = "running"
                    continue
                if event_type != "episode_step":
                    continue
                payload = event.get("payload")
                if not isinstance(payload, Mapping):
                    continue
                step_result = payload.get("step_result")
                if not isinstance(step_result, Mapping):
                    continue
                info = step_result.get("info")
                if isinstance(info, Mapping) and info.get("pause_reason") == "ask_human":
                    state = "human_input_required"
    return state


def _wait_for_scripted_tui_episode(
    paths: CasePaths,
    process: subprocess.Popen[Any],
    *,
    timeout_s: float,
) -> str:
    """Wait for completion or an unattended human-input gate.

    The result deliberately does not label a completed episode as passed: the
    existing milestone verifier remains the sole authority for that decision.
    """

    deadline = time.monotonic() + timeout_s
    while process.poll() is None and time.monotonic() < deadline:
        state = _scripted_tui_trace_state(paths)
        if state != "running":
            return state
        time.sleep(0.1)
    state = _scripted_tui_trace_state(paths)
    if state != "running":
        return state
    return "exited" if process.poll() is not None else "timed_out"


def _terminate_scripted_tui_process(process: subprocess.Popen[Any]) -> None:
    """Terminate only the PTY process group created for one scripted case."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=5)


def _scripted_tui_driver_evidence(paths: CasePaths, *, status: str, reason_code: str) -> None:
    """Persist a bounded, secret-free reason when the PTY driver stops early."""

    _json_dump(
        paths.root / "scripted-tui-driver.json",
        {
            "schema_version": "openeta.scripted_tui_driver.v1",
            "status": status,
            "reason_code": reason_code,
        },
    )


def _run_scripted_tui(command: str, paths: CasePaths, env: Mapping[str, str]) -> int:
    """Drive the real PTY TUI, then submit `/quit` after episode completion."""

    timeout_raw = str(env.get("OPENETA_SCRIPTED_TUI_TIMEOUT_S", "3600"))
    try:
        timeout_s = float(timeout_raw)
    except ValueError:
        timeout_s = 3600.0
    if timeout_s <= 0:
        raise AcceptanceError("TUI_NOT_READY: scripted TUI timeout must be positive")
    process = subprocess.Popen(
        ["script", "--flush", "--return", "--command", command, str(paths.transcript)],
        cwd=paths.root,
        env=dict(env),
        stdin=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    if process.stdin is None:
        raise AcceptanceError("TUI_NOT_READY: scripted PTY stdin is unavailable")
    try:
        process.stdin.write(_scripted_tui_initial_input(paths))
        process.stdin.flush()
        state = _wait_for_scripted_tui_episode(paths, process, timeout_s=timeout_s)
        if state != "completed":
            reason_code = {
                "human_input_required": "TUI_HUMAN_INPUT_REQUIRED",
                "timed_out": "TUI_SCRIPTED_TASK_TIMEOUT",
                "exited": "TUI_EXITED_BEFORE_EPISODE_RESULT",
            }.get(state, "TUI_SCRIPTED_DRIVER_STATE_INVALID")
            _scripted_tui_driver_evidence(
                paths,
                status="blocked" if state == "human_input_required" else "failed",
                reason_code=reason_code,
            )
            _terminate_scripted_tui_process(process)
            # The caller's normal finally block must still stop MCP/Gazebo and
            # materialize cleanup evidence, so return a failed TUI code rather
            # than bypassing it with an exception.
            return 1
        if process.poll() is None:
            process.stdin.write("/quit\n")
            process.stdin.flush()
        try:
            return int(process.wait(timeout=30))
        except subprocess.TimeoutExpired:
            _scripted_tui_driver_evidence(
                paths,
                status="failed",
                reason_code="TUI_DID_NOT_EXIT_AFTER_QUIT",
            )
            _terminate_scripted_tui_process(process)
            # Return through run_case's ordinary cleanup path. Raising here
            # would run only its inner MCP finally block and skip the durable
            # worker/ROS/Gazebo cleanup evidence assembled afterwards.
            return 1
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()


def _provider_endpoint_id(api_base: str) -> str:
    """Return a credential-, path-, and query-free provider identifier."""

    parsed = urlparse(str(api_base or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "unconfigured"
    try:
        port_value = parsed.port
    except ValueError:
        return "invalid"
    port = f":{port_value}" if port_value is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _root_provider_config(
    repo: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> PlannerProviderConfig:
    """Resolve the repository provider configuration before changing into a case.

    The PTY and MCP children deliberately use the case directory as their
    working directory, where the repository `.env` is not visible.  Resolve
    it here using the normal application precedence instead of copying a
    credential-bearing configuration file into a case artifact directory.
    """

    return load_planner_provider_config(
        env=environ,
        dotenv_path=repo / ".env",
        apikey_path=repo / "apikey.md",
    )


def _resolved_provider_environment(config: PlannerProviderConfig) -> dict[str, str]:
    """Return the primary, resolved provider config for a case-local TUI.

    This function intentionally returns only the primary.  A root-configured
    fallback is not transferred automatically, so an unavailable primary
    cannot turn into an unrecorded switch during a scripted acceptance case.
    A formal profile may add its explicitly evaluated fallback through
    :func:`_tui_provider_environment`.  The mapping stays in process memory.
    """

    values = {
        "OPENETA_LLM_PROVIDER": config.provider,
        "OPENETA_LLM_MODEL": config.model,
        "OPENETA_LLM_API_BASE": config.api_base,
        "OPENETA_LLM_API_KEY": config.api_key,
        "OPENETA_LLM_TIMEOUT_S": str(config.timeout_s),
        "OPENETA_LLM_MAX_ATTEMPTS": str(config.max_attempts),
        "OPENETA_LLM_RETRY_BACKOFF_S": str(config.retry_backoff_s),
        "OPENETA_LLM_MAX_TOKENS": str(config.max_tokens),
    }
    if config.context_window_tokens is not None:
        values["OPENETA_LLM_CONTEXT_WINDOW_TOKENS"] = str(config.context_window_tokens)
    vision = config.metadata.get("enable_vision")
    if isinstance(vision, bool):
        values["OPENETA_LLM_ENABLE_VISION"] = "true" if vision else "false"
    return values


def _tui_provider_environment(
    config: PlannerProviderConfig,
    *,
    profile_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve one provider identity plus profile-scoped request bounds.

    The root configuration remains authoritative for primary identity and all
    credentials.  Acceptance profiles may tune bounded request handling and
    name one evaluated fallback model hosted by that exact provider/endpoint.
    Keeping this merge explicit prevents a profile from silently redirecting
    requests or replacing the primary model being evaluated.
    """

    values = _resolved_provider_environment(config)
    if profile_environment:
        values.update(
            {
                str(key): str(value)
                for key, value in profile_environment.items()
                if str(key) in _PROVIDER_PROFILE_KEYS
            }
        )
        fallback_model = str(
            profile_environment.get("OPENETA_LLM_FALLBACK_MODEL") or ""
        ).strip()
        if fallback_model:
            # Derive endpoint identity and credentials from the selected
            # primary rather than accepting duplicates from a profile.
            values.update(
                {
                    "OPENETA_LLM_FALLBACK_PROVIDER": config.provider,
                    "OPENETA_LLM_FALLBACK_MODEL": fallback_model,
                    "OPENETA_LLM_FALLBACK_API_BASE": config.api_base,
                    "OPENETA_LLM_FALLBACK_API_KEY": config.api_key,
                    "OPENETA_LLM_FALLBACK_TIMEOUT_S": str(
                        profile_environment.get("OPENETA_LLM_FALLBACK_TIMEOUT_S")
                        or values["OPENETA_LLM_TIMEOUT_S"]
                    ),
                }
            )
    return values


def _without_provider_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Copy an environment without inherited provider or fallback settings."""

    return {
        key: value
        for key, value in environ.items()
        if not key.startswith(_PROVIDER_ENV_PREFIX)
    }


def _provider_preflight_failure_code(exc: Exception, *, stage: str) -> tuple[str, str]:
    """Classify a provider failure without materializing provider text.

    Provider error bodies are untrusted and can contain echoed credentials or
    request headers.  Durable acceptance evidence records only this bounded
    code and error class, never the raw body.
    """

    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        matched = re.search(r"\bHTTP\s+(\d{3})\b", str(exc))
        status = int(matched.group(1)) if matched else None
    if status in {401, 403}:
        return "blocked", "PROVIDER_AUTH_FAILED"
    if status is not None:
        return "blocked", f"PROVIDER_HTTP_{status}"
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return "blocked", "PROVIDER_NETWORK_OR_TIMEOUT"
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        return "failed", "PROVIDER_RESPONSE_JSON_INCOMPATIBLE"
    if stage == "models":
        return "blocked", "PROVIDER_MODEL_LIST_UNAVAILABLE"
    return "blocked", "PROVIDER_PLANNER_SMOKE_UNAVAILABLE"


def _provider_preflight_result(
    repo: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the no-Gazebo, primary-only scripted-provider preflight.

    This checks the exact selected model through `/v1/models`, then makes one
    constrained planner request which can only return an `ask_human` response.
    It neither starts the MCP server nor offers a simulator tool, and it never
    persists provider request/response payloads.
    """

    started = time.monotonic()
    try:
        config = _root_provider_config(repo, environ=environ)
    except Exception as exc:  # noqa: BLE001 - unreadable local config is a gate result.
        return {
            "schema_version": "openeta.provider_preflight.v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "blocked",
            "provider": "openai-compatible",
            "model": "",
            "endpoint_id": "unconfigured",
            "vision_enabled": False,
            "fallback_used": False,
            "reason_code": "PROVIDER_CONFIG_UNREADABLE",
            "error_type": type(exc).__name__,
            "model_list": {"status": "not_run"},
            "planner_smoke": {"status": "not_run"},
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }
    backend_config = OpenAICompatiblePlannerBackendConfig.from_provider_config(config)
    # A preflight is evidence for the selected primary endpoint only.  Do not
    # retry or activate a configured fallback and accidentally validate a
    # different provider/model than the TUI will be allowed to use.
    backend_config.max_attempts = 1
    backend_config.retry_backoff_s = 0.0
    backend_config.fallback = None
    result: dict[str, Any] = {
        "schema_version": "openeta.provider_preflight.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "blocked",
        "provider": config.provider,
        "model": config.model,
        "endpoint_id": _provider_endpoint_id(config.api_base),
        # This is the resolved configuration after normal environment > .env
        # precedence, not a documentation/default value.
        "max_tokens": config.max_tokens,
        "vision_enabled": backend_config.enable_vision,
        "fallback_used": False,
        "model_list": {"status": "not_run"},
        "planner_smoke": {"status": "not_run"},
    }
    missing = [
        field
        for field, value in (
            ("model", config.model),
            ("api_base", config.api_base),
            ("api_key", config.api_key),
        )
        if not value
    ]
    if missing:
        result.update(
            {
                "reason_code": "PROVIDER_CONFIG_MISSING",
                "missing_fields": missing,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }
        )
        return result

    models_started = time.monotonic()
    try:
        models = list_openai_compatible_models(backend_config)
    except Exception as exc:  # noqa: BLE001 - provider failures are evidence, not crashes.
        status, reason_code = _provider_preflight_failure_code(exc, stage="models")
        result.update(
            {
                "status": status,
                "reason_code": reason_code,
                "error_type": type(exc).__name__,
                "model_list": {
                    "status": "failed",
                    "latency_ms": round((time.monotonic() - models_started) * 1000, 3),
                },
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }
        )
        return result
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
        result.update(
            {
                "status": "failed",
                "reason_code": "PROVIDER_MODEL_LIST_RESPONSE_INCOMPATIBLE",
                "model_list": {
                    "status": "failed",
                    "latency_ms": round((time.monotonic() - models_started) * 1000, 3),
                },
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }
        )
        return result
    if config.model not in models:
        result.update(
            {
                "status": "failed",
                "reason_code": "PROVIDER_MODEL_NOT_FOUND",
                "model_list": {
                    "status": "failed",
                    "latency_ms": round((time.monotonic() - models_started) * 1000, 3),
                    "selected_model_found": False,
                },
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }
        )
        return result
    result["model_list"] = {
        "status": "passed",
        "latency_ms": round((time.monotonic() - models_started) * 1000, 3),
        "selected_model_found": True,
    }

    smoke_started = time.monotonic()
    try:
        smoke = OpenAICompatiblePlannerBackend(backend_config).decide(
            PlannerBackendRequest(
                tool_context={"acceptance_preflight": True, "available_tools": []},
                system_prompt=(
                    "This is a provider connectivity preflight with no tools and no "
                    "world access. Return only this JSON object: "
                    '{"kind":"response","name":"ask_human",'
                    '"parameters":{"message":"provider preflight"},'
                    '"reasoning":"structured provider preflight"}.'
                ),
                metadata={"isolated_context": True, "purpose": "provider_preflight"},
            )
        )
    except Exception as exc:  # noqa: BLE001 - retain a redacted, bounded failure only.
        status, reason_code = _provider_preflight_failure_code(exc, stage="planner")
        result.update(
            {
                "status": status,
                "reason_code": reason_code,
                "error_type": type(exc).__name__,
                "planner_smoke": {
                    "status": "failed",
                    "latency_ms": round((time.monotonic() - smoke_started) * 1000, 3),
                },
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }
        )
        return result

    if smoke.status != PipelineStatus.PLANNED:
        # The planner backend normally converts provider exceptions into a
        # failed, structured ask_human result.  Classify that bounded failure
        # before schema validation; otherwise an HTTP billing/auth error is
        # misleadingly reported as a structured-response incompatibility.
        details = smoke.details if isinstance(smoke.details, Mapping) else {}
        error_text = str(details.get("error") or "")
        status, reason_code = _provider_preflight_failure_code(
            RuntimeError(error_text), stage="planner"
        )
        result.update(
            {
                "status": status,
                "reason_code": reason_code,
                "error_type": str(details.get("error_type") or "PlannerBackendFailed"),
                "planner_smoke": {
                    "status": "failed",
                    "latency_ms": round((time.monotonic() - smoke_started) * 1000, 3),
                },
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }
        )
        return result

    payload: Any = smoke.payload
    try:
        if isinstance(payload, str):
            payload = json.loads(payload)
        structured = (
            smoke.status == PipelineStatus.PLANNED
            and isinstance(payload, Mapping)
            and payload.get("kind") == "response"
            and payload.get("name") == "ask_human"
            and isinstance(payload.get("parameters"), Mapping)
            and isinstance(payload.get("reasoning"), str)
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        structured = False
    if not structured:
        result.update(
            {
                "status": "failed",
                "reason_code": "PROVIDER_STRUCTURED_RESPONSE_INCOMPATIBLE",
                "planner_smoke": {
                    "status": "failed",
                    "latency_ms": round((time.monotonic() - smoke_started) * 1000, 3),
                },
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }
        )
        return result
    result.update(
        {
            "status": "passed",
            "reason_code": "PROVIDER_PREFLIGHT_PASSED",
            "planner_smoke": {
                "status": "passed",
                "latency_ms": round((time.monotonic() - smoke_started) * 1000, 3),
                "response_schema": "response/ask_human",
            },
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }
    )
    return result


def run_case(
    repo: Path,
    paths: CasePaths,
    allocation: Allocation,
    *,
    calibration_profile: Path | None = None,
    extra_environment: Mapping[str, str] | None = None,
) -> int:
    scripted = paths.root.name == SCRIPTED_TUI
    # The simulator MCP server never needs provider credentials.  Start from
    # a scrubbed copy so process inheritance cannot make them appear in MCP or
    # worker diagnostics; only the actual TUI receives the resolved primary
    # provider configuration below.
    env = _without_provider_environment(os.environ)
    # Jazzy deprecates ROS_LOCALHOST_ONLY in favour of discovery-range
    # controls. In this multi-process topology the legacy switch can prevent
    # the bridge and camera subscriber from discovering each other. Keep DDS
    # on this host; the allocated ROS domain isolates individual cases.
    env.pop("ROS_LOCALHOST_ONLY", None)
    env.pop("ROS_STATIC_PEERS", None)
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
            # Gazebo launches Python adapters outside the bench worker.  Pin
            # those imports to this worktree instead of any editable install
            # recorded in the shared application virtualenv.
            "OPENETA_GAZEBO_SOURCE_ROOT": str(repo),
            "ROS_DOMAIN_ID": str(allocation.ros_domain_id),
            "GZ_PARTITION": allocation.gz_partition,
            "MCP_PORT": str(allocation.port),
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
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
            "OPENETA_CONTROL_ONLY": "0",
            "RMW_IMPLEMENTATION": env.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        }
    )
    if extra_environment:
        # Provider settings belong only to the TUI child.  Even harmless
        # request bounds are excluded from the simulator MCP environment so
        # the process boundary stays obvious in diagnostics.
        env.update(
            {
                str(key): str(value)
                for key, value in extra_environment.items()
                if not str(key).startswith(_PROVIDER_ENV_PREFIX)
            }
        )
    tui_env = dict(env)
    # Resolve root `.env`/`apikey.md` before the child changes into its
    # isolated case directory.  Both human-gated and scripted TUI cases use
    # the same provider boundary; otherwise a human-gated case silently falls
    # back to its empty case-directory defaults.  Keep this mapping solely in
    # the child process environment; never copy a config file into evidence.
    tui_env.update(
        _tui_provider_environment(
            _root_provider_config(repo),
            profile_environment=extra_environment,
        )
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
    operator_gui_process: subprocess.Popen[Any] | None = None
    operator_gui_pgid = -1
    operator_gui_log: Any = None
    operator_gui_evidence: dict[str, Any] = {
        "requested": _operator_gui_requested(env),
        "started": False,
    }
    try:
        _wait_ready(allocation.port, process)
        if operator_gui_evidence["requested"]:
            (
                operator_gui_process,
                operator_gui_pgid,
                operator_gui_log,
            ) = _launch_operator_gui(repo, paths, env)
            operator_gui_evidence.update(
                {
                    "started": True,
                    "pid": operator_gui_process.pid,
                    "pgid": operator_gui_pgid,
                    "display": str(env.get("OPENETA_GAZEBO_DISPLAY") or ":3"),
                    "gz_partition": allocation.gz_partition,
                    "ros_domain_id": allocation.ros_domain_id,
                    "log": "operator-gui.log",
                }
            )
        print(f"\n=== {paths.root.name} ===")
        print(paths.instructions.read_text(encoding="utf-8"))
        command = f"{shlex.quote(str(python))} -m agent.cli.openeta_cli"
        if calibration_profile is not None:
            command += (
                " --calibration-profile "
                + shlex.quote(str(calibration_profile.resolve()))
            )
        if shutil.which("script") is None:
            raise AcceptanceError("TUI_NOT_READY: util-linux script is missing")
        if scripted:
            tui_code = _run_scripted_tui(command, paths, tui_env)
        else:
            completed = subprocess.run(
                ["script", "--flush", "--return", "--command", command, str(paths.transcript)],
                cwd=paths.root,
                env=tui_env,
                check=False,
            )
            tui_code = int(completed.returncode)
    finally:
        if operator_gui_process is not None:
            operator_gui_evidence["live_until_case_cleanup"] = (
                operator_gui_process.poll() is None
            )
            termination_error = ""
            try:
                _terminate_owned_group(
                    operator_gui_process,
                    operator_gui_pgid,
                    label="operator GUI",
                )
            except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence.
                termination_error = f"{type(exc).__name__}: {exc}"
            finally:
                if operator_gui_log is not None:
                    operator_gui_log.close()
            operator_gui_evidence.update(
                {
                    "termination_error": termination_error,
                    "group_exited": _process_group_exited(operator_gui_pgid),
                    "returncode": operator_gui_process.poll(),
                }
            )
            operator_gui_evidence["lifecycle_ok"] = bool(
                operator_gui_evidence.get("live_until_case_cleanup") is True
                and operator_gui_evidence.get("group_exited") is True
                and not termination_error
            )
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
    residual_groups = _terminate_owned_residual_groups(
        run_id=allocation.run_id,
        before=receipt["preexisting_processes"],
        candidates=after,
    )
    after = _process_snapshot()
    owned_residuals = _owned_process_residuals(after, run_id=allocation.run_id)
    residual_deadline = time.monotonic() + 15.0
    while owned_residuals and time.monotonic() < residual_deadline:
        time.sleep(0.25)
        after = _process_snapshot()
        owned_residuals = _owned_process_residuals(after, run_id=allocation.run_id)
    # DDS and Gazebo discovery can briefly retain endpoints after every owned
    # process has exited. Give their leases one bounded settle interval before
    # recording the authoritative graph/partition cleanup evidence.
    if not owned_residuals:
        time.sleep(2.0)
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
    process_continuity = _preexisting_process_continuity(
        receipt["preexisting_processes"], after
    )
    cleanup = {
        "mcp_group_exited": process.poll() is not None,
        "mcp_termination_error": mcp_termination_error,
        "port_free": _wait_for_free_port(allocation.port),
        "owned_worker_groups": worker_groups,
        "owned_worker_groups_exited": all(
            item.get("group_exited") is True for item in worker_groups
        ),
        "owned_residual_groups": residual_groups,
        "owned_residual_groups_exited": all(
            item.get("group_exited") is True for item in residual_groups
        ),
        "owned_process_residuals": owned_residuals,
        "operator_gui": operator_gui_evidence,
        **process_continuity,
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
    # Repeated motions are counted exactly once rather than once per trace
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
        if (profile == SCRIPTED_TUI or source == SCRIPTED_TUI) and (
            item.get("allowed") is True or item.get("approved") is True
        ):
            return True
    return False


def _human_approved(call: Mapping[str, Any]) -> bool:
    """Require durable human-gated provenance for a world mutation."""

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
            profile == "human_gated"
            and source == "human"
            and (item.get("allowed") is True or item.get("approved") is True)
        ):
            return True
    return False


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
        unmanaged_continuity = cleanup.get(
            "preexisting_unmanaged_process_snapshot_unchanged"
        )
        if unmanaged_continuity is False or (
            unmanaged_continuity is None
            and cleanup.get("preexisting_process_snapshot_unchanged") is not True
        ):
            errors.append("preexisting unmanaged process snapshot changed")
        if cleanup.get("protected_ros_graphs_unchanged") is not True:
            errors.append("protected ROS domain 42/100 graph changed")
        operator_gui = cleanup.get("operator_gui")
        if (
            isinstance(operator_gui, Mapping)
            and operator_gui.get("requested") is True
            and operator_gui.get("lifecycle_ok") is not True
        ):
            errors.append("case-owned Gazebo operator GUI lifecycle proof failed")
        for name in ("ros_graph", "gz_partition"):
            state = str((cleanup.get(name) or {}).get("state") or "INCONCLUSIVE")
            if state == "FAILED":
                errors.append(f"cleanup {name} is not empty")
            elif state != "PASSED":
                errors.append(f"cleanup {name} is inconclusive")
    if not events:
        errors.append("trace.jsonl is missing or empty")
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
    """Validate every required simulator MCP call.

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
            # Keep the durable response itself unchanged, but retain the
            # already-validated RPC association for semantic verifiers. They
            # must never borrow a receipt from another action.
            annotated = dict(value)
            annotated[_MCP_EVIDENCE_REQUEST_ID] = request_id
            annotated[_MCP_EVIDENCE_AGENT_TOOL] = agent_tool
            payloads.append(annotated)
    if not paths.mcp_log.is_file() or not paths.mcp_log.read_text(encoding="utf-8", errors="replace").strip():
        errors.append("MCP server log is missing or empty")
    return payloads, errors
