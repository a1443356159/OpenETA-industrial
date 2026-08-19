#!/usr/bin/env python3
"""Gazebo M0--M5 acceptance and control-preflight coordinator.

The coordinator owns isolation, evidence locations, process groups and report
assembly. Formal modes use the real PTY TUI: the default profile is
``human_gated`` and ``scripted_tui`` records automation approval rather than
impersonating a human operator. The separate ``control_only`` mode exercises
the same AgentTool -> MCP/SSE -> Gazebo boundary without a planner, provider,
or PTY and cannot produce a formal acceptance report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, UTC
import hashlib
import json
import math
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


SCHEMA_VERSION = "openeta.tui_gazebo_acceptance.v2"
MILESTONES = ("m0", "m1", "m2", "m3", "m4")
M5_MILESTONE = "m5"
M2_GRIPPER_SEQUENCE = (1, 0, 1, 1, 0, 1)
DETERMINISTIC = "deterministic"
AUTONOMY = "planner_autonomy"
SCRIPTED_TUI = "scripted_tui"
# A deliberately separate execution mode for exercising the simulator control
# boundary without loading a planner, provider, or the PTY TUI.  Its report is
# never named ``acceptance-report.json`` and cannot satisfy formal acceptance.
CONTROL_ONLY = "control_only"
CONTROL_REPORT_FILENAME = "control-acceptance-report.json"
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
    # M5 deliberately reuses M3's exact physical scene and native
    # DetachableJoint rules; only the preceding perception gate changes.
    "m5": "openeta/gazebo_rm75_robotiq2f85_pickplace-v0",
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
        "ROS_CAMERA_TOPICS_NOT_READY",
        "SAM3_UNAVAILABLE",
        "SAM3_TOOL_UNAVAILABLE",
        "SAM3_INFERENCE_UNAVAILABLE",
    }
)
PROVIDER_BILLING_EXHAUSTED = "PROVIDER_BILLING_EXHAUSTED"
PROVIDER_PREFLIGHT_FILENAME = "provider-preflight.json"
_PROVIDER_ENV_PREFIX = "OPENETA_LLM_"
# These keys exist only on in-memory copies returned by
# ``_mcp_response_payloads``.  They preserve the exact action/RPC association
# while keeping the materialized simulator response byte-for-byte intact on
# disk.
_MCP_EVIDENCE_REQUEST_ID = "__openeta_acceptance_mcp_request_id"
_MCP_EVIDENCE_AGENT_TOOL = "__openeta_acceptance_agent_tool"


class AcceptanceError(RuntimeError):
    pass


class M5ControlError(AcceptanceError):
    """A recorded M5 gate result, distinguished from simulator infrastructure."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


class M5BlockedError(M5ControlError):
    """An unavailable external SAM3 service; M3 motion must not start."""


class M5FailedError(M5ControlError):
    """A malformed or ambiguous perception result; M3 motion must not start."""


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
    if allocation.candidate_domain_preflight is not None:
        payload["candidate_domain_preflight"] = dict(
            allocation.candidate_domain_preflight
        )
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


def instructions_for(milestone: str, mode: str) -> str:
    if mode == CONTROL_ONLY:
        return (
            "No-provider control-layer exercise. Execute only the repository's "
            "AgentTool → simulator MCP/SSE path; do not start a planner, invoke "
            "a model/provider, or claim PTY/TUI formal acceptance. "
            f"Milestone environment: {ENV_IDS[milestone]}.\n"
        )
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
        "m2": f"""第一步且唯一的环境创建必须是 AgentTool create_simulator_env，env_id 精确为 `{ENV_IDS["m2"]}`。
禁止调用 python_exec；禁止以任何其他 env_id（包括 libero）创建环境，也不得用其他方式创建环境。
创建后仅使用真实 MCP/MoveIt 回执，严格按以下合同执行：四次成功 move_to，目标依次为 A、B、A、B（A 与 B 必须不同且碰撞安全）；随后六次成功 gripper_control，position 精确依次为 [1, 0, 1, 1, 0, 1]；随后仅一次不可达 move_to，且它必须返回 MOTION_PLAN_FAILED；随后 observe；最后唯一一次 close_simulator_env。
将创建回执中的初始 EEF xyz 作为唯一基准：A 为该 xyz 加 validated vertical_low 偏移，B 为该 xyz 加 validated vertical_high 偏移。四次 A/B move_to 的 target_pose 只提交 frame=world 和这两个固定 xyz，不得提交 quat_xyzw；第三次必须逐字节复用第一次的 target_pose，第四次必须逐字节复用第二次的 target_pose，不得从后续观测重算。
六个夹爪位置是六个独立原子动作；第 3 个 position=1 成功后，仍必须单独调用第 4 个 position=1 并等待其独立回执，不得合并、去重或省略相邻的相同参数。
每个 move_to 或 gripper_control 只可在严格回执 ok=true、reached_goal/reached_target=true 且 stalled=false 后进入下一步。stalled、超时、未到目标或任何失败都不是成功。某一夹爪步骤首次失败时，必须先 fresh observe，再以完全相同的 position 仅重试一次；该重试再次失败时，立即 close_simulator_env 并让 M2 失败，绝不可继续或进行第二次重试。不得绕过 MoveIt、跳步、替换 A/B，或把失败动作当作成功。逐一批准所有 human_gated 提示，确认 close 后用 /quit 退出。
""",
        "m3": f"""第一步且唯一的环境创建必须是 AgentTool create_simulator_env，env_id 精确为 `{ENV_IDS["m3"]}`。
禁止调用 python_exec；禁止以任何其他 env_id（包括 libero）创建环境，也不得用其他方式创建环境。
创建后先接近，再执行一次真实 close。只有回执显示双垫 native contact 与 attached ACK 后才能执行第一段 lift；记录 child-link 的 >=80 mm lift 和 <=10 mm 相对位移；然后 open、detach ACK，最后唯一一次 close_simulator_env。
close 回执中 native_contact_gate.accepted=true（source=gazebo_native_contacts）且 detachable_joint.state=attached 时，physical_verification.reason_code=NATIVE_GRASP_ATTACH_ACKED_UNPROVEN 是已通过双垫接触与 attach ACK、等待唯一一次 lift proof 的预期中间态。此时 grasp_confirmed=false 与 verdict=UNKNOWN 仅表示 lift proof 尚未执行，不是 attach 失败；不得用 observe 等待它自行变化，必须立即执行 validated lift [0.1552,-0.1000,0.5976]，再以 lift 回执判定 >=80 mm 与 <=10 mm 物理证明。
""",
        "m4": f"""第一步且唯一的环境创建必须是 AgentTool create_simulator_env，env_id 精确为 `{ENV_IDS["m4"]}`。
禁止调用 python_exec；禁止以任何其他 env_id（包括 libero）创建环境，也不得用其他方式创建环境。
创建后实际调用 oracle_perceive 并记录 perception_source=gazebo_oracle 与 contractual fake candidate；它不能替代 M3：随后执行真实 close、native contact、attached ACK 和 child-link lift 证明，然后 open、detach ACK，最后唯一一次 close_simulator_env。
close 回执中 native_contact_gate.accepted=true（source=gazebo_native_contacts）且 detachable_joint.state=attached 时，physical_verification.reason_code=NATIVE_GRASP_ATTACH_ACKED_UNPROVEN 是已通过双垫接触与 attach ACK、等待唯一一次 lift proof 的预期中间态。此时 grasp_confirmed=false 与 verdict=UNKNOWN 仅表示 lift proof 尚未执行，不是 attach 失败；不得用 observe 等待它自行变化，必须立即执行 validated lift [0.1552,-0.1000,0.5976]，再以 lift 回执判定 >=80 mm 与 <=10 mm 物理证明。
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


def _control_trace_path(paths: CasePaths) -> Path:
    """Return the durable raw-command trace for one no-provider case."""

    return paths.trace_root / "sessions" / "control" / "trace.jsonl"


def _append_control_trace(paths: CasePaths, event: Mapping[str, Any]) -> None:
    trace = _control_trace_path(paths)
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True))
        stream.write("\n")


def _tool_result_record(result: Any) -> dict[str, Any]:
    """Return a JSON-safe, full-fidelity AgentTool result for the raw trace."""

    return {
        "success": bool(getattr(result, "success", False)),
        "content": str(getattr(result, "content", "")),
        "details": dict(getattr(result, "details", {}) or {}),
    }


def _control_response_payload(result: Any) -> Mapping[str, Any]:
    """Load the materialized response belonging to one proxied AgentTool call."""

    details = getattr(result, "details", {})
    if not isinstance(details, Mapping):
        return {}
    outputs = details.get("outputs")
    if not isinstance(outputs, Mapping):
        return {}
    response = outputs.get("response")
    if not isinstance(response, Mapping):
        return {}
    raw_path = response.get("response_path")
    if not isinstance(raw_path, str) or not raw_path:
        return {}
    try:
        payload = _json_load(Path(raw_path))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _control_response_has(result: Any, key: str, expected: Any) -> bool:
    """Read an exact result field from the call's materialized MCP response.

    AgentTool's compact result intentionally summarizes a failed motion as a
    generic target-not-reached diagnostic.  The durable response is the
    authoritative controller receipt and retains the precise MoveIt failure
    code.  A missing or unreadable response remains a failure here.
    """

    return _contains(_control_response_payload(result), key, expected)


def _control_eef_xyz(result: Any) -> tuple[float, float, float]:
    """Read an actual MCP observation; never infer an M2/M3 pose geometrically."""

    payload = _control_response_payload(result)
    observation = payload.get("observation", payload)
    robot = observation.get("robot") if isinstance(observation, Mapping) else None
    pose = robot.get("end_effector_pose") if isinstance(robot, Mapping) else None
    xyz = pose.get("xyz") if isinstance(pose, Mapping) else None
    if (
        not isinstance(xyz, Sequence)
        or isinstance(xyz, (str, bytes, bytearray))
        or len(xyz) < 3
        or any(type(value) not in {int, float} for value in xyz[:3])
    ):
        raise AcceptanceError("CONTROL_EEF_STATE_MISSING")
    return tuple(float(value) for value in xyz[:3])


def _control_rgb_path(result: Any) -> str:
    """Return one actual case-local RGB artifact for M4's Gazebo oracle."""

    details = getattr(result, "details", {})
    for item in _walk(details):
        if not isinstance(item, Mapping) or item.get("kind") != "rgb":
            continue
        value = item.get("path")
        if isinstance(value, str) and Path(value).is_file():
            return value
    raise AcceptanceError("CONTROL_RGB_ARTIFACT_MISSING")


def _control_observation(result: Any) -> Any | None:
    """Build a current typed observation only for the local M4 Oracle tool."""

    payload = _control_response_payload(result)
    raw = payload.get("observation", payload)
    if not isinstance(raw, Mapping):
        return None
    try:
        from adapter.protocol import EnvObservation

        observation = EnvObservation.from_dict(dict(raw), task=str(raw.get("task") or ""))
    except (ImportError, TypeError, ValueError):
        return None
    details = getattr(result, "details", {})
    artifacts = [
        dict(item)
        for item in _walk(details)
        if isinstance(item, Mapping) and item.get("kind") in {"rgb", "depth"}
    ]
    if artifacts:
        observation.metadata["image_artifacts"] = artifacts
    return observation


class _ControlToolRunner:
    """Direct AgentTool executor used only by the no-provider control path.

    It deliberately does not construct an agent runtime, planner, provider, or
    PTY.  Every call is still a normal registered AgentTool backed by the SSE
    proxy, and every full result is written as a raw command trace so the same
    receipt verifier can validate the simulator boundary.
    """

    def __init__(
        self,
        *,
        paths: CasePaths,
        allocation: Allocation,
        registry: Any,
    ) -> None:
        self.paths = paths
        self.allocation = allocation
        self.registry = registry
        self.metadata = {
            "execution_id": f"control-{allocation.run_id}",
            "session_id": f"control-{allocation.run_id}",
            "agent_session_id": f"control-{allocation.run_id}",
            "execution_profile": CONTROL_ONLY,
            "planner_invoked": False,
            "provider_invoked": False,
        }
        self.closed = False

    def invoke(
        self,
        name: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        observation: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        normalized_parameters = dict(parameters or {})
        call_metadata = {**self.metadata, **dict(metadata or {})}
        result = self.registry.call(
            name,
            normalized_parameters,
            observation=observation,
            metadata=call_metadata,
        )
        _append_control_trace(
            self.paths,
            {
                "event_type": "action",
                "payload": {
                    "execution_profile": CONTROL_ONLY,
                    "planner_invoked": False,
                    "provider_invoked": False,
                    "command": {
                        "tool_calls": [
                            {
                                "kind": "tool_call",
                                "name": name,
                                "parameters": normalized_parameters,
                                "status": "executed" if getattr(result, "success", False) else "failed",
                                "result": (
                                    _m5_redact_external_value(_tool_result_record(result))
                                    if name == "sam3"
                                    else _tool_result_record(result)
                                ),
                            }
                        ]
                    },
                },
            },
        )
        if name == "close_simulator_env" and getattr(result, "success", False):
            self.closed = True
        return result

    def require_success(
        self,
        name: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        observation: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        result = self.invoke(
            name,
            parameters,
            observation=observation,
            metadata=metadata,
        )
        if not getattr(result, "success", False):
            raise AcceptanceError(f"CONTROL_TOOL_FAILED:{name}")
        return result

    def close_if_open(self) -> None:
        if not self.closed:
            self.invoke("close_simulator_env")


def _bind_control_tools(*, paths: CasePaths, allocation: Allocation, milestone: str) -> _ControlToolRunner:
    """Bind only simulator-owned tools; this path has no planner/provider dependency."""

    from agent.tools.handlers import build_oracle_perceive_segmenter, build_sam3_handler
    from agent.tools.registry import build_default_tool_registry
    from agent.tools.sim_mcp import (
        SimulatorMcpToolProxyConfig,
        SseSimulatorMcpTransport,
        bind_simulator_mcp_tool_handlers,
    )

    transport = SseSimulatorMcpTransport(f"http://127.0.0.1:{allocation.port}/sse")
    catalog = transport.list_tools(timeout_s=30.0)
    _json_dump(paths.root / "mcp-tool-catalog.json", catalog, exclusive=True)
    proxy_config = SimulatorMcpToolProxyConfig(
        session_id=f"control-{allocation.run_id}",
        timeout_s=180.0,
        image_output_root=paths.root / "mcp-images",
        text_output_root=paths.root / "mcp-text",
        response_output_root=paths.root / "mcp-responses",
    )
    registry = build_default_tool_registry(
        perception_profile="oracle" if milestone == "m4" else "sam3"
    )
    bind_simulator_mcp_tool_handlers(
        registry,
        transport=transport,
        config=proxy_config,
        replace=True,
    )
    if milestone == "m4":
        # Reuse the exact production M4 Oracle evidence wrapper, but do not
        # assemble the broader provider-backed runtime just to exercise it.
        from agent.runtime.runtime_assembly import (
            _OracleMcpEvidence,
            _with_contractual_fake_candidate,
        )

        evidence = _OracleMcpEvidence(
            proxy_config=proxy_config,
            response_output_root=Path(proxy_config.response_output_root),
        )
        oracle_handler = build_sam3_handler(
            build_oracle_perceive_segmenter(
                transport,
                handle_provider=lambda: proxy_config.handle,
                session_id_provider=lambda: proxy_config.session_id,
                response_callback=evidence.record,
            ),
            tool_name="oracle_perceive",
            output_root=paths.root / "oracle-images",
            result_output_root=paths.root / "oracle-results",
        )
        registry.bind_handler(
            "oracle_perceive",
            _with_contractual_fake_candidate(oracle_handler, mcp_evidence=evidence),
            replace=True,
        )
    return _ControlToolRunner(paths=paths, allocation=allocation, registry=registry)


def _run_m0_control(runner: _ControlToolRunner) -> None:
    runner.require_success(
        "create_simulator_env",
        {"env_id": ENV_IDS["m0"], "seed": 0, "task": "M0 control connectivity"},
    )
    runner.require_success("observe")


def _run_m1_control(runner: _ControlToolRunner) -> None:
    runner.require_success(
        "create_simulator_env",
        {"env_id": ENV_IDS["m1"], "seed": 0, "task": "M1 RGB-D control connectivity"},
    )
    runner.require_success("observe")
    # Require a later live sample rather than accepting the same initial
    # image twice. Gazebo continues running in M1 after the first observe.
    time.sleep(0.15)
    runner.require_success("observe")


def _m2_control_motion(runner: _ControlToolRunner) -> None:
    observed = runner.require_success("observe")
    x, y, z = _control_eef_xyz(observed)
    # Keep both repeated A/B targets below the spawn pose.  Returning the
    # redundant RM75 wrist all the way to its initial z after a 40 mm descent
    # can select a hard joint-limit IK branch, even though the Cartesian goal
    # itself is reachable.  This is the same real MoveIt probe topology used
    # by the M2 live acceptance: 40 mm down, then 20 mm up, repeated.  It does
    # not suppress an error or add a recovery retry; every one of the four
    # required A<->B motions must still complete through the production MCP
    # controller.
    b = {
        "x": x,
        "y": y,
        "z": z - 0.040,
        "velocity_scaling": 0.1,
        "acceleration_scaling": 0.1,
    }
    a = {
        "x": x,
        "y": y,
        "z": z - 0.020,
        "velocity_scaling": 0.1,
        "acceleration_scaling": 0.1,
    }
    for target in (b, a, b, a):
        runner.require_success("move_to", target)
    for position in M2_GRIPPER_SEQUENCE:
        runner.require_success("gripper_control", {"position": position})
    unreachable = runner.invoke(
        "move_to",
        {
            "x": 99.0,
            "y": 99.0,
            "z": 99.0,
            "velocity_scaling": 0.1,
            "acceleration_scaling": 0.1,
        },
    )
    if getattr(unreachable, "success", False) or not _control_response_has(
        unreachable, "error_code", "MOTION_PLAN_FAILED"
    ):
        raise AcceptanceError("CONTROL_M2_UNREACHABLE_DID_NOT_FAIL_CLOSED")
    runner.require_success("observe")


def _m3_motion(runner: _ControlToolRunner) -> list[Any]:
    """Issue fixed fixture waypoints; native receipts alone prove a grasp.

    These M3 control-preflight poses are static world-frame values for the
    repository-owned target block and RM75/2F-85 fixture.  They are neither a
    geometry/contact predicate nor a TF, distance, or perception-derived
    target estimate.  In particular, no movement is allowed after close until
    native bilateral contact and the stock attachment ACK have succeeded in
    ``GazeboDirectEnv``.
    """

    # The gripper's fixture-specific contact centre is offset from the block
    # centre.  Specify the calibrated *mount* pose explicitly, including the
    # horizontal closing-axis orientation, rather than aiming the mount at the
    # target model origin.  All values are fixed scene configuration; contact
    # admission remains exclusively the native two-pad Gazebo stream.
    common = {
        "target_pose": {
            "frame": "world",
            "euler_xyz_deg": [115.0, 0.0, 90.0],
        },
        "velocity_scaling": 0.1,
        "acceleration_scaling": 0.1,
        # Match the profile-owned stable motion contract.  A looser goal
        # window lets the first pad push the 40 mm target away before the
        # opposite native contact stream begins.
        "tolerance": 0.0002,
        "ori_tolerance": 0.002,
    }
    approach = {
        **common,
        "target_pose": {**common["target_pose"], "xyz": [0.1552, -0.1000, 0.5686]},
    }
    capture = {
        **common,
        "target_pose": {**common["target_pose"], "xyz": [0.1552, -0.1000, 0.4976]},
    }
    lift = {
        **common,
        # Keep the first post-attach motion 100 mm above capture.  The M3
        # verifier requires a measured lift of at least 80 mm, so reusing the
        # 71 mm approach clearance here could never satisfy the native
        # child-link proof even if the stock joint held perfectly.
        "target_pose": {**common["target_pose"], "xyz": [0.1552, -0.1000, 0.5976]},
    }
    return [
        runner.require_success("move_to", approach),
        runner.require_success("move_to", capture),
        runner.require_success("gripper_control", {"position": 0}),
        runner.require_success("move_to", lift),
        runner.require_success("gripper_control", {"position": 1}),
    ]


def _m5_endpoint_id(url: str) -> str:
    """Return a credential/query-free identifier for M5 durable evidence."""

    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise AcceptanceError("--sam3-url must be an absolute http(s) SSE MCP URL")
    try:
        port_value = parsed.port
    except ValueError as exc:
        raise AcceptanceError("--sam3-url has an invalid port") from exc
    port = f":{port_value}" if port_value is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


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

    The acceptance run intentionally has one selected model.  In particular,
    it does not transfer a configured fallback endpoint, so an unavailable
    primary cannot turn into an unrecorded model/provider switch during a
    scripted acceptance case.  The mapping is held only in process memory.
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


def _m5_case_artifact(paths: CasePaths, value: str | Path) -> dict[str, Any]:
    """Reference one existing case-local file with a durable content hash."""

    path = Path(value).resolve()
    try:
        relative = path.relative_to(paths.root.resolve())
    except ValueError as exc:
        raise M5FailedError("M5_ARTIFACT_OUTSIDE_CASE") from exc
    if not path.is_file():
        raise M5FailedError("M5_ARTIFACT_MISSING")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(relative),
        "sha256": digest,
        "byte_size": path.stat().st_size,
    }


def _m5_optional_case_artifact(paths: CasePaths, value: Any) -> dict[str, Any] | None:
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    return _m5_case_artifact(paths, value)


def _m5_scrub_json_artifact(paths: CasePaths, value: str | Path) -> None:
    """Remove credential-like values before linking an external SAM3 artifact."""

    path = Path(value).resolve()
    _m5_case_artifact(paths, path)
    try:
        decoded = _json_load(path)
    except (OSError, ValueError) as exc:
        raise M5FailedError("M5_SAM3_ARTIFACT_UNREADABLE") from exc

    _json_dump(path, _m5_redact_external_value(decoded))


def _m5_redact_external_value(value: Any) -> Any:
    """Recursively remove payload forms that must not enter M5 evidence."""

    secret_markers = (
        "api_key",
        "apikey",
        "token",
        "secret",
        "credential",
        "password",
        "authorization",
    )
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            normalized = key_text.lower().replace("-", "_")
            if "base64" in normalized or any(marker in normalized for marker in secret_markers):
                cleaned[f"{key_text}_omitted"] = True
            else:
                cleaned[key_text] = _m5_redact_external_value(child)
        return cleaned
    if isinstance(value, list):
        return [_m5_redact_external_value(item) for item in value]
    return value


def _m5_tool_catalog_names(catalog: Mapping[str, Any]) -> set[str]:
    tools = catalog.get("tools")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return set()
    return {
        str(item.get("name") or "")
        for item in tools
        if isinstance(item, Mapping) and str(item.get("name") or "")
    }


def _m5_current_camera(observe_result: Any, paths: CasePaths) -> dict[str, Any]:
    """Select the one numeric-extrinsic top RGB-D camera from this observe call."""

    payload = _control_response_payload(observe_result)
    observation = payload.get("observation", payload)
    cameras = observation.get("cameras") if isinstance(observation, Mapping) else None
    if not isinstance(cameras, Sequence) or isinstance(cameras, (str, bytes, bytearray)):
        raise M5FailedError("M5_OBSERVE_CAMERAS_MISSING")
    candidates = [
        dict(camera)
        for camera in cameras
        if isinstance(camera, Mapping)
        and camera.get("role") == "scene_primary"
        and isinstance(camera.get("extrinsics"), Mapping)
        and camera["extrinsics"].get("frame_transform") == "camera_to_world"
    ]
    if len(candidates) != 1:
        raise M5FailedError("M5_TOP_RGBD_CAMERA_AMBIGUOUS")
    camera = candidates[0]
    # Validate all paths before sending the RGB upstream.  This is the first
    # M5 gate and prevents a later bridge error from accepting a stale frame.
    _m5_case_artifact(paths, str(camera.get("rgb_path") or ""))
    _m5_case_artifact(paths, str(camera.get("depth_path") or ""))
    if not isinstance(camera.get("intrinsics"), Mapping):
        raise M5FailedError("M5_INTRINSICS_MISSING")
    if not str(camera.get("frame_id") or ""):
        raise M5FailedError("M5_SOURCE_FRAME_MISSING")
    return camera


def _m5_observation_for_sam3(camera: Mapping[str, Any], task: str) -> Any:
    """Expose only the current top RGB artifact to SAM3, never Gazebo objects."""

    from adapter.protocol import CameraFrame, EnvObservation, RobotState

    frame_id = str(camera["frame_id"])
    role = str(camera.get("role") or "scene_primary")
    return EnvObservation(
        task=task,
        cameras=[
            CameraFrame(
                frame_id=frame_id,
                role=role,
                rgb=[],
                depth=None,
                intrinsics=dict(camera.get("intrinsics") or {}),
                extrinsics=dict(camera.get("extrinsics") or {}),
            )
        ],
        robot=RobotState(),
        # The SAM3 handler resolves its image solely from these current
        # artifacts.  Do not copy raw observation objects/metadata here: those
        # can contain Gazebo ground truth and are out of M5 control scope.
        metadata={
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": frame_id,
                    "role": role,
                    "path": str(camera["rgb_path"]),
                },
                {
                    "kind": "depth",
                    "frame_id": frame_id,
                    "role": role,
                    "path": str(camera["depth_path"]),
                },
            ]
        },
    )


def _m5_observe_receipt(observe_result: Any, paths: CasePaths) -> dict[str, Any]:
    details = getattr(observe_result, "details", {})
    outputs = details.get("outputs") if isinstance(details, Mapping) else None
    response = outputs.get("response") if isinstance(outputs, Mapping) else None
    mcp_calls = outputs.get("mcp_calls") if isinstance(outputs, Mapping) else None
    receipt: dict[str, Any] = {}
    if isinstance(response, Mapping):
        artifact = _m5_optional_case_artifact(paths, response.get("response_path"))
        if artifact is not None:
            receipt["response"] = artifact
        for key in ("request_id", "tool", "session_id", "handle"):
            if isinstance(response.get(key), str):
                receipt[key] = response[key]
    if isinstance(mcp_calls, Sequence) and not isinstance(mcp_calls, (str, bytes, bytearray)):
        for call in mcp_calls:
            if not isinstance(call, Mapping):
                continue
            environment_receipt = call.get("environment_receipt")
            if isinstance(environment_receipt, Mapping):
                receipt["environment_receipt"] = {
                    key: environment_receipt[key]
                    for key in ("receipt_id", "remote_tool", "mcp_request_id", "handle", "simulator_session_id")
                    if isinstance(environment_receipt.get(key), str)
                }
                break
    return receipt


def _m5_selection_is_forbidden(value: Any) -> bool:
    """Reject Oracle and contractual/fake-candidate contamination before motion."""

    if _contains(value, "perception_source", "gazebo_oracle"):
        return True
    if _contains(value, "provenance", "gazebo_oracle"):
        return True
    if any(isinstance(item, Mapping) for item in _values(value, "fake_grasp_candidate")):
        return True
    return any(
        isinstance(item, Mapping)
        and (
            item.get("kind") == "contractual_fake_grasp_candidate"
            or item.get("schema_version")
            == "openeta.contractual_fake_grasp_candidate.v1"
        )
        for item in _walk(value)
    )


def _m5_require_single_real_candidate(outputs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return M5's sole permissible candidate without score/oracle fallback."""

    if _m5_selection_is_forbidden(outputs):
        raise M5FailedError("M5_ORACLE_OR_FAKE_CANDIDATE")
    candidates = outputs.get("detections")
    if not isinstance(candidates, list) or len(candidates) == 0:
        raise M5FailedError("M5_ZERO_SAM3_CANDIDATES")
    if len(candidates) != 1:
        raise M5FailedError("M5_MULTIPLE_SAM3_CANDIDATES")
    candidate = candidates[0]
    if not isinstance(candidate, Mapping) or not str(candidate.get("id") or ""):
        raise M5FailedError("M5_SAM3_CANDIDATE_MALFORMED")
    score = candidate.get("score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
    ):
        raise M5FailedError("M5_SAM3_CANDIDATE_MALFORMED")
    return candidate


def _m5_m3_receipts(results: Sequence[Any], paths: CasePaths) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for result in results:
        details = getattr(result, "details", {})
        outputs = details.get("outputs") if isinstance(details, Mapping) else None
        response = outputs.get("response") if isinstance(outputs, Mapping) else None
        if not isinstance(response, Mapping):
            continue
        artifact = _m5_optional_case_artifact(paths, response.get("response_path"))
        if artifact is None:
            continue
        receipt: dict[str, Any] = {"response": artifact}
        for key in ("request_id", "tool", "session_id", "handle"):
            if isinstance(response.get(key), str):
                receipt[key] = response[key]
        receipts.append(receipt)
    return receipts


def _m5_post_motion_evaluation(last_result: Any) -> dict[str, Any]:
    """Record bounded Gazebo truth only after M3 has completed.

    This is deliberately an evaluation annotation rather than an input to a
    selection, pose, contact, or motion decision.
    """

    payload = _control_response_payload(last_result)
    observation = payload.get("observation", payload)
    objects = observation.get("objects") if isinstance(observation, Mapping) else None
    entries = objects if isinstance(objects, Sequence) and not isinstance(objects, (str, bytes, bytearray)) else []
    return {
        "available": bool(entries),
        "used_for_control": False,
        "captured_after_m3_motion": True,
        "object_count": len(entries),
    }


def _run_m5_control(
    runner: _ControlToolRunner,
    *,
    paths: CasePaths,
    sam3_url: str,
) -> None:
    """Run real external SAM3 once, then the unchanged M3 physical sequence."""

    from adapter.protocol import EnvAction
    from agent.runtime.planner import BasePlanner
    from agent.runtime.runtime import OpenEtaAgentRuntime
    from agent.tools.handlers import build_sam3_handler, build_sse_sam3_mcp_segmenter
    from agent.tools.sim_mcp import SseSimulatorMcpTransport
    from extensions.gazebo.perception_summary import (
        PerceptionBridgeError,
        build_perception_object_summary,
    )

    evidence_path = paths.root / "m5-perception.json"
    evidence: dict[str, Any] = {
        "schema_version": "openeta.gazebo.m5_perception_evidence.v1",
        "acceptance_scope": "control_only_real_sam3_no_planner_not_formal_tui",
        "status": "started",
        "sam3": {"endpoint_id": _m5_endpoint_id(sam3_url)},
        "ground_truth_evaluation": {
            "available": False,
            "used_for_control": False,
            "captured_after_m3_motion": False,
        },
    }

    def write_evidence() -> None:
        _json_dump(evidence_path, evidence)

    class _SelectionOnlyPlanner(BasePlanner):
        """A guard that makes accidental planner execution fail immediately."""

        def plan(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("M5 control-only selection must not invoke a planner")

    try:
        # This transport belongs to the externally supplied SAM3 URL only.
        # We do not spawn, stop, or otherwise manage that service.
        try:
            sam3_transport = SseSimulatorMcpTransport(sam3_url)
            catalog = sam3_transport.list_tools(timeout_s=30.0)
        except Exception as exc:  # noqa: BLE001 - transport details may contain credentials.
            evidence.update({"status": "blocked", "reason_code": "SAM3_UNAVAILABLE"})
            write_evidence()
            raise M5BlockedError("SAM3_UNAVAILABLE", "external SAM3 service is unavailable") from exc
        if not isinstance(catalog, Mapping) or "segment" not in _m5_tool_catalog_names(catalog):
            evidence.update({"status": "blocked", "reason_code": "SAM3_TOOL_UNAVAILABLE"})
            write_evidence()
            raise M5BlockedError("SAM3_TOOL_UNAVAILABLE", "external SAM3 service lacks segment")
        # The endpoint is external. Persist only the discovered names/count,
        # not arbitrary remote tool descriptions or schemas that could carry
        # deployment-specific secrets.
        catalog_path = paths.root / "m5-sam3-tool-catalog.json"
        tool_names = sorted(_m5_tool_catalog_names(catalog))
        _json_dump(
            catalog_path,
            {"tool_names": tool_names, "tool_count": len(tool_names)},
            exclusive=True,
        )
        evidence["sam3"]["tool_catalog"] = _m5_case_artifact(paths, catalog_path)

        runner.require_success(
            "create_simulator_env",
            {
                "env_id": ENV_IDS[M5_MILESTONE],
                "seed": 0,
                "task": "M5 real SAM3 perception then M3 native joint control",
            },
        )
        current = runner.require_success("observe")
        camera = _m5_current_camera(current, paths)
        evidence["observe"] = {
            "receipt": _m5_observe_receipt(current, paths),
            "frame_id": str(camera["frame_id"]),
            "rgb": _m5_case_artifact(paths, str(camera["rgb_path"])),
            "depth": _m5_case_artifact(paths, str(camera["depth_path"])),
            "intrinsics": dict(camera["intrinsics"]),
            "extrinsics": dict(camera["extrinsics"]),
        }

        # Reuse the production SAM3 materializer so the request/response are
        # scrubbed and its mask artifacts remain case-local.  OpenEtaAgentRuntime
        # is instantiated only to bind the existing selection contract; no
        # planner, provider, PTY, or runtime act loop is invoked.
        sam3_handler = build_sam3_handler(
            build_sse_sam3_mcp_segmenter(url=sam3_url, tool_name="segment"),
            output_root=paths.root / "m5-sam3-images",
            result_output_root=paths.root / "m5-sam3-results",
        )
        runner.registry.bind_handler("sam3", sam3_handler, replace=True)
        selection_runtime = OpenEtaAgentRuntime(
            planner=_SelectionOnlyPlanner(),
            tools=runner.registry,
            rollout_enabled=False,
            default_session_id=str(runner.metadata["session_id"]),
        )
        selection_runtime.start_session(
            task="M5 control-only SAM3 single-candidate selection",
            metadata={"execution_profile": CONTROL_ONLY, "planner_invoked": False},
        )
        sam3_result = runner.invoke(
            "sam3",
            {"image": str(camera["rgb_path"]), "prompt": "red rectangular block"},
            observation=_m5_observation_for_sam3(camera, "M5 SAM3 segmentation"),
        )
        sam3_details = getattr(sam3_result, "details", {})
        sam3_outputs = (
            sam3_details.get("outputs") if isinstance(sam3_details, Mapping) else None
        )
        if not getattr(sam3_result, "success", False):
            reason = (
                str(sam3_outputs.get("reason") or "SAM3_INFERENCE_UNAVAILABLE")
                if isinstance(sam3_outputs, Mapping)
                else "SAM3_INFERENCE_UNAVAILABLE"
            )
            if reason in {
                "inconsistent_detection_outputs",
                "artifact_write_failed",
                "image_not_found",
                "image_encode_failed",
            }:
                evidence.update({"status": "failed", "reason_code": "M5_SAM3_RESPONSE_MALFORMED", "sam3_reason": reason})
                write_evidence()
                raise M5FailedError("M5_SAM3_RESPONSE_MALFORMED")
            evidence.update({"status": "blocked", "reason_code": "SAM3_INFERENCE_UNAVAILABLE", "sam3_reason": reason})
            write_evidence()
            raise M5BlockedError("SAM3_INFERENCE_UNAVAILABLE", "external SAM3 inference is unavailable")
        if not isinstance(sam3_outputs, Mapping):
            evidence.update({"status": "failed", "reason_code": "M5_SAM3_RESPONSE_MALFORMED"})
            write_evidence()
            raise M5FailedError("M5_SAM3_RESPONSE_MALFORMED")
        try:
            candidate = _m5_require_single_real_candidate(sam3_outputs)
        except M5FailedError as exc:
            evidence.update({"status": "failed", "reason_code": exc.code})
            write_evidence()
            raise
        raw_response_ref = sam3_outputs.get("raw_output_ref")
        sam3_run_dir = Path(str(raw_response_ref)).parent if isinstance(raw_response_ref, str) else None
        for artifact_path in (
            sam3_run_dir / "request.json" if sam3_run_dir else None,
            raw_response_ref,
            sam3_run_dir / "tool_result.json" if sam3_run_dir else None,
        ):
            if artifact_path is None:
                continue
            try:
                _m5_scrub_json_artifact(paths, artifact_path)
            except M5FailedError as exc:
                evidence.update({"status": "failed", "reason_code": exc.code})
                write_evidence()
                raise
        sam3_artifacts: dict[str, Any] = {}
        for name, artifact_path in (
            ("request", sam3_run_dir / "request.json" if sam3_run_dir else None),
            ("response", raw_response_ref),
            ("tool_result", sam3_run_dir / "tool_result.json" if sam3_run_dir else None),
            ("mask", candidate.get("mask_ref")),
        ):
            artifact = _m5_optional_case_artifact(paths, artifact_path)
            if artifact is None:
                evidence.update({"status": "failed", "reason_code": "M5_SAM3_ARTIFACT_MISSING"})
                write_evidence()
                raise M5FailedError("M5_SAM3_ARTIFACT_MISSING")
            sam3_artifacts[name] = artifact
        evidence["sam3"].update(
            {
                "tool": "segment",
                "candidate_count": 1,
                "request_response_artifacts": sam3_artifacts,
            }
        )

        # Feed the real handler result into the unchanged pending-selection
        # state machine, then make its one candidate explicit with protected
        # host-only metadata.  There is no score-based selection or Oracle
        # fallback in this branch.
        selection_runtime.memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "tool_calls": [
                        {
                            "name": "sam3",
                            "result": _m5_redact_external_value(
                                _tool_result_record(sam3_result)
                            ),
                        }
                    ]
                },
            )
        )
        selection = runner.invoke(
            "select_sam3_detection",
            {
                "sam3_result_id": str(sam3_outputs.get("result_id") or ""),
                "detection_id": str(candidate["id"]),
                "reason": "control-only exact single SAM3 candidate",
            },
            metadata={
                "_openeta_control_only_perception": True,
                "_openeta_host_selection_source": "scripted_single_candidate",
            },
        )
        selection_outputs = (
            selection.details.get("outputs")
            if isinstance(getattr(selection, "details", {}), Mapping)
            else None
        )
        selected = selection_outputs.get("selected_detection") if isinstance(selection_outputs, Mapping) else None
        if (
            not getattr(selection, "success", False)
            or not isinstance(selected, Mapping)
            or selected.get("selection_source") != "scripted_single_candidate"
            or _m5_selection_is_forbidden(selected)
        ):
            evidence.update({"status": "failed", "reason_code": "M5_SELECTION_REJECTED"})
            write_evidence()
            raise M5FailedError("M5_SELECTION_REJECTED")
        evidence["selection"] = {
            "tool": "select_sam3_detection",
            "result_id": str(selection_outputs.get("result_id") or ""),
            "detection_id": str(selected.get("id") or ""),
            "selection_source": "scripted_single_candidate",
            "reason": "control-only exact single SAM3 candidate",
        }

        try:
            object_summary = build_perception_object_summary(
                detection=selected,
                camera=camera,
                case_root=paths.root,
            )
        except PerceptionBridgeError as exc:
            evidence.update({"status": "failed", "reason_code": exc.code})
            write_evidence()
            raise M5FailedError(exc.code) from exc
        summary_path = paths.root / "m5-object-summary.json"
        _json_dump(summary_path, object_summary, exclusive=True)
        evidence["m5_object_summary"] = {
            "artifact": _m5_case_artifact(paths, summary_path),
            "provenance": "sam3_perception",
            "source_camera": str(camera["frame_id"]),
            "confidence": object_summary["objects"][0].get("confidence"),
        }
        write_evidence()

        m3_results = _m3_motion(runner)
        evidence["m3_receipts"] = _m5_m3_receipts(m3_results, paths)
        evidence["ground_truth_evaluation"] = _m5_post_motion_evaluation(m3_results[-1])
        evidence["status"] = "passed"
        write_evidence()
    except M5ControlError:
        raise
    except Exception as exc:  # noqa: BLE001 - retain no endpoint/credential in artifacts.
        evidence.update({"status": "failed", "reason_code": "M5_UNEXPECTED_FAILURE"})
        write_evidence()
        raise M5FailedError("M5_UNEXPECTED_FAILURE") from exc


def _run_m3_or_m4_control(runner: _ControlToolRunner, *, m4: bool) -> None:
    runner.require_success(
        "create_simulator_env",
        {
            "env_id": ENV_IDS["m4" if m4 else "m3"],
            "seed": 0,
            "task": "M4 Oracle control connectivity" if m4 else "M3 native joint control connectivity",
        },
    )
    current = runner.require_success("observe")
    if m4:
        runner.require_success(
            "oracle_perceive",
            {"image": _control_rgb_path(current), "prompt": "red rectangular target"},
            observation=_control_observation(current),
        )
    _m3_motion(runner)


def _run_control_case(
    paths: CasePaths,
    allocation: Allocation,
    *,
    sam3_url: str = "",
) -> int:
    """Run one no-provider case and retain raw AgentTool/MCP evidence."""

    milestone = paths.root.parent.name
    runner: _ControlToolRunner | None = None
    try:
        runner = _bind_control_tools(paths=paths, allocation=allocation, milestone=milestone)
        if milestone == "m0":
            _run_m0_control(runner)
        elif milestone == "m1":
            _run_m1_control(runner)
        elif milestone == "m2":
            runner.require_success(
                "create_simulator_env",
                {"env_id": ENV_IDS["m2"], "seed": 0, "task": "M2 control connectivity"},
            )
            _m2_control_motion(runner)
        elif milestone == "m3":
            _run_m3_or_m4_control(runner, m4=False)
        elif milestone == "m4":
            _run_m3_or_m4_control(runner, m4=True)
        elif milestone == M5_MILESTONE:
            if not sam3_url:
                raise M5BlockedError("SAM3_UNAVAILABLE", "M5 requires an external SAM3 URL")
            _run_m5_control(runner, paths=paths, sam3_url=sam3_url)
        else:
            raise AcceptanceError(f"unknown control milestone: {milestone}")
    except Exception as exc:  # noqa: BLE001 - record the actual local failure before cleanup.
        error_code = exc.code if isinstance(exc, M5ControlError) else str(exc)
        _append_control_trace(
            paths,
            {
                "event_type": "control_error",
                "payload": {
                    "execution_profile": CONTROL_ONLY,
                    "planner_invoked": False,
                    "provider_invoked": False,
                    "error_code": error_code,
                    "error_type": type(exc).__name__,
                },
            },
        )
        return 1
    finally:
        if runner is not None:
            runner.close_if_open()
    return 0


def run_case(
    repo: Path,
    paths: CasePaths,
    allocation: Allocation,
    *,
    sam3_url: str = "",
    calibration_profile: Path | None = None,
) -> int:
    milestone = paths.root.parent.name
    scripted = paths.root.name == SCRIPTED_TUI
    control_only = paths.root.name == CONTROL_ONLY
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
            "OPENETA_CONTROL_ONLY": "1" if control_only else "0",
            # M4 explicitly selects the existing Gazebo Oracle tool.  The
            # candidate it emits is a contractual fixture, never a model
            # prediction and never an alternative M3 grasp gate.
            "OPENETA_PERCEPTION_PROFILE": "oracle" if milestone == "m4" else "",
            "OPENETA_M4_CONTRACTUAL_FAKE_CANDIDATE": "1" if milestone == "m4" else "0",
            "RMW_IMPLEMENTATION": env.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        }
    )
    tui_env = dict(env)
    # Resolve root `.env`/`apikey.md` before the child changes into its
    # isolated case directory.  Both human-gated and scripted TUI cases use
    # the same provider boundary; otherwise a human-gated case silently falls
    # back to its empty case-directory defaults.  Keep this mapping solely in
    # the child process environment; never copy a config file into evidence.
    tui_env.update(_resolved_provider_environment(_root_provider_config(repo)))
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
        if control_only:
            tui_code = _run_control_case(paths, allocation, sam3_url=sam3_url)
        else:
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
                # Feed these through the actual PTY TUI rather than shortcut mode,
                # so M0's operator-console evidence lands in the transcript.
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


def _case_local_artifact_path(value: Any, root: Path) -> Path | None:
    """Resolve one existing artifact, rejecting path escapes and absent files."""

    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved if resolved.is_file() else None


def _durable_rgbd_pairs(payload: Mapping[str, Any], root: Path) -> list[tuple[str, Path, Path]]:
    """Return only camera pairs whose RGB and depth files are case-local."""

    pairs: list[tuple[str, Path, Path]] = []
    for item in _walk(payload):
        if not isinstance(item, Mapping):
            continue
        if "rgb_path" not in item and "depth_path" not in item:
            continue
        rgb = _case_local_artifact_path(item.get("rgb_path"), root)
        depth = _case_local_artifact_path(item.get("depth_path"), root)
        if rgb is None or depth is None:
            continue
        pairs.append((str(item.get("frame_id") or ""), rgb, depth))
    return pairs


def _m1_camera_frames(
    result: Mapping[str, Any], *, root: Path
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Resolve compact M1 RGB-D refs against durable response artifacts.

    The action trace deliberately omits image payloads.  Its camera summaries
    therefore prove RGB-D only when both refs map to existing, case-local
    materialized files and the durable MCP response records that exact pair.
    """

    outputs = _mapping_with(result, "outputs")
    response = outputs.get("response") if isinstance(outputs, Mapping) else None
    if not isinstance(response, Mapping):
        return [], ["lacks materialized MCP response reference"]
    durable_path = _case_local_artifact_path(response.get("response_path"), root)
    if durable_path is None:
        return [], ["durable MCP response artifact is missing or escapes the case"]
    try:
        durable = _json_load(durable_path)
    except (OSError, ValueError) as exc:
        return [], [f"durable MCP response artifact is unreadable: {exc}"]
    if not isinstance(durable, Mapping):
        return [], ["durable MCP response artifact is not an object"]
    durable_pairs = _durable_rgbd_pairs(durable, root)
    if not durable_pairs:
        return [], ["durable MCP response has no existing RGB-D pair"]
    summaries = response.get("cameras")
    artifacts = response.get("image_artifacts")
    if not isinstance(summaries, Sequence) or isinstance(summaries, (str, bytes, bytearray)):
        return [], ["materialized response lacks camera references"]
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes, bytearray)):
        return [], ["materialized response lacks image artifacts"]
    resolved_frames: list[Mapping[str, Any]] = []
    errors: list[str] = []
    for index, summary in enumerate(summaries, 1):
        if not isinstance(summary, Mapping):
            errors.append(f"camera reference {index} is not an object")
            continue
        rgb_ref = summary.get("rgb_ref")
        depth_ref = summary.get("depth_ref")
        if not isinstance(rgb_ref, str) or not rgb_ref or not isinstance(depth_ref, str) or not depth_ref:
            errors.append(f"camera reference {index} lacks paired rgb_ref/depth_ref")
            continue

        def artifact_path(kind: str, reference: str) -> Path | None:
            matches = [
                item for item in artifacts
                if isinstance(item, Mapping)
                and item.get("kind") == kind
                and item.get("index") == reference
            ]
            if len(matches) != 1:
                return None
            return _case_local_artifact_path(matches[0].get("path"), root)

        rgb = artifact_path("rgb", rgb_ref)
        depth = artifact_path("depth", depth_ref)
        if rgb is None or depth is None:
            errors.append(f"camera reference {index} has missing, ambiguous, or nonlocal RGB-D artifacts")
            continue
        frame_id = str(summary.get("frame_id") or "")
        matching_durable = [
            pair for pair in durable_pairs
            if pair[1] == rgb and pair[2] == depth and (not frame_id or pair[0] == frame_id)
        ]
        if not matching_durable:
            errors.append(f"camera reference {index} does not match its durable RGB-D response")
            continue
        resolved_frames.append({**summary, "rgb_path": str(rgb), "depth_path": str(depth)})
    return resolved_frames, errors


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


def _provider_billing_exhaustion_errors(
    events: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Recognise only durable pre-tool provider billing exhaustion.

    A provider may refuse a planner request before it can select a simulator
    tool.  That is an external blocked run, not evidence that M2 motion
    planning failed.  Do not infer this from planner prose: require the
    structured planner metadata emitted by the runtime, the exact upstream
    ``ProviderHttpError`` type, HTTP 402, and the provider's insufficient
    balance marker.  Any executed tool call, a different HTTP error, or an
    ordinary invalid planner response remains subject to the normal strict
    milestone verifier.
    """

    if calls:
        return []
    for event in events:
        if str(event.get("event_type") or "") not in {
            "pipeline_plan",
            "action",
            "episode_step",
        }:
            continue
        for item in _walk(event):
            if not isinstance(item, Mapping):
                continue
            details = item.get("backend_details")
            if (
                str(item.get("backend_status") or "").lower() != "failed"
                or not isinstance(details, Mapping)
                or str(details.get("error_type") or "") != "ProviderHttpError"
            ):
                continue
            message = str(details.get("error") or "")
            if re.search(r"\bHTTP\s+402\b", message, flags=re.IGNORECASE) and re.search(
                r"\binsufficient\s+balance\b", message, flags=re.IGNORECASE
            ):
                return [
                    "provider billing exhausted before any simulator tool call "
                    "(durable ProviderHttpError HTTP 402 Insufficient Balance)"
                ]
    return []


def _verify_m0(
    calls: Sequence[Mapping[str, Any]],
    paths: CasePaths,
    *,
    control_only: bool = False,
) -> list[str]:
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
    if not control_only:
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


def _verify_m1(calls: Sequence[Mapping[str, Any]], paths: CasePaths) -> list[str]:
    errors: list[str] = []
    observes = [call for call in calls if str(call.get("name") or call.get("tool_name")) == "observe"]
    if len(observes) < 2:
        return ["M1 requires two observe calls"]
    last_timestamp: float | None = None
    for index, call in enumerate(observes[:2], 1):
        result = _result(call)
        cameras, camera_errors = _m1_camera_frames(result, root=paths.root)
        errors.extend(f"M1 observe {index} {error}" for error in camera_errors)
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


def _call_mcp_request_ids(call: Mapping[str, Any]) -> set[str]:
    """Return the explicit RPC ids emitted for one AgentTool action."""

    outputs = _mapping_with(_result(call), "outputs")
    entries = outputs.get("mcp_calls") if isinstance(outputs, Mapping) else None
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        return set()
    request_ids: set[str] = set()
    for entry in entries:
        request = entry.get("request") if isinstance(entry, Mapping) else None
        request_id = request.get("request_id") if isinstance(request, Mapping) else None
        if isinstance(request_id, str) and request_id:
            request_ids.add(request_id)
    return request_ids


def _m2_semantic_nodes(
    call: Mapping[str, Any],
    payloads: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Join one M2 action to its validated durable controller receipt.

    ``payloads`` are only supplied after ``_mcp_response_payloads`` has
    checked request, response and environment-receipt correlation.  Thus this
    does not let an unrelated response satisfy an M2 safety rule.  Direct
    unit callers may omit ``payloads`` to exercise the legacy raw-result
    contract in isolation.
    """

    result = _result(call)
    if payloads is None:
        return [result], []
    request_ids = _call_mcp_request_ids(call)
    linked = [
        payload
        for payload in payloads
        if str(payload.get(_MCP_EVIDENCE_REQUEST_ID) or "") in request_ids
    ]
    name = str(call.get("name") or call.get("tool_name") or "action")
    if len(request_ids) != 1 or len(linked) != 1:
        return [result], [f"M2 {name} lacks one correlated durable MCP response"]
    return [result, linked[0]], []


def _m2_terminal_receipt(nodes: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return the durable controller response for one M2 action when present."""

    # ``_m2_semantic_nodes`` places the validated materialized MCP response
    # last.  Work in reverse for direct unit tests too, whose single node is
    # their synthetic controller receipt.
    for node in reversed(nodes):
        for item in _walk(node):
            if isinstance(item, Mapping) and "ok" in item:
                return item
    return {}


def _m2_strict_action_success(
    nodes: Sequence[Mapping[str, Any]],
    *,
    reached_key: str,
) -> bool:
    """Accept only the terminal M2 receipt's explicit, non-stalled success."""

    receipt = _m2_terminal_receipt(nodes)
    reached = receipt.get(reached_key)
    if reached is None and reached_key == "reached_target":
        reached = receipt.get("reached_goal")
    return (
        receipt.get("ok") is True
        and reached is True
        and receipt.get("stalled") is False
    )


def _m2_gripper_position(call: Mapping[str, Any]) -> int | None:
    """Read exactly the agent-submitted binary command, never a nested echo."""

    parameters = call.get("parameters")
    position = parameters.get("position") if isinstance(parameters, Mapping) else None
    return position if type(position) is int and position in {0, 1} else None


def _m2_move_target_key(
    call: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]
) -> str | None:
    """Canonicalise the submitted A/B target before receipt-normalised poses."""

    candidates: list[Any] = []
    # The AgentTool request is the A/B contract.  A target pose may omit
    # orientation, in which case the simulator intentionally preserves the
    # current orientation and the post-action receipt can differ by tiny
    # numerical amounts between identical requested xyz values.  Do not turn
    # that controller normalisation into a false A/B mismatch.
    parameters = call.get("parameters")
    if isinstance(parameters, Mapping):
        candidates.extend((parameters.get("target_pose"), parameters))
    for node in reversed(nodes):
        if isinstance(node, Mapping):
            candidates.append(node.get("target"))
            candidates.append(node.get("requested_tool_pose"))
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        xyz = candidate.get("xyz")
        if xyz is None and all(key in candidate for key in ("x", "y", "z")):
            xyz = [candidate["x"], candidate["y"], candidate["z"]]
        quat = candidate.get("quat_xyzw")
        if not (
            isinstance(xyz, Sequence)
            and not isinstance(xyz, (str, bytes, bytearray))
            and len(xyz) == 3
        ):
            continue
        try:
            canonical: dict[str, list[float]] = {"xyz": [float(value) for value in xyz]}
            if (
                isinstance(quat, Sequence)
                and not isinstance(quat, (str, bytes, bytearray))
                and len(quat) == 4
            ):
                canonical["quat_xyzw"] = [float(value) for value in quat]
            return json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            continue
    return None


def _m2_successful_observe(
    call: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]
) -> bool:
    """Keep retry recovery tied to a real successful observation action."""

    return (
        str(call.get("name") or call.get("tool_name") or "") == "observe"
        and _successful(call)
        and _contains(nodes, "observation_fresh", True)
    )


def _verify_m2(
    calls: Sequence[Mapping[str, Any]],
    payloads: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Verify M2's ordered, fail-closed motion and gripper contract.

    A mere sequence of requested gripper parameters is not evidence of six
    state transitions.  Every credited action must instead point at its own
    correlated, durable controller response and meet the strict terminal
    success predicate.  The one recovery path is deliberately explicit so a
    planner cannot hide retries or treat stalls as a successful close/open.
    """

    errors: list[str] = []
    semantic_nodes: dict[int, list[Mapping[str, Any]]] = {}
    for call in calls:
        name = str(call.get("name") or call.get("tool_name") or "")
        if name not in {"move_to", "gripper_control", "observe"}:
            continue
        nodes, node_errors = _m2_semantic_nodes(call, payloads)
        semantic_nodes[id(call)] = nodes
        errors.extend(node_errors)

    contract_names = {
        "create_simulator_env",
        "move_to",
        "gripper_control",
        "observe",
        "close_simulator_env",
    }
    contract_calls = [
        call
        for call in calls
        if str(call.get("name") or call.get("tool_name") or "") in contract_names
    ]
    forbidden_trajectory_calls = [
        call
        for call in calls
        if str(call.get("name") or call.get("tool_name") or "") == "follow_eef_trajectory"
    ]
    if forbidden_trajectory_calls:
        errors.append("M2 contract forbids follow_eef_trajectory in place of A/B moves")

    create_calls = [
        call
        for call in contract_calls
        if str(call.get("name") or call.get("tool_name") or "") == "create_simulator_env"
    ]
    if len(create_calls) != 1:
        errors.append("M2 requires exactly one create_simulator_env")
    elif not _contains(create_calls[0], "env_id", ENV_IDS["m2"]):
        errors.append("M2 environment identity is not the required M2 environment")
    if not contract_calls or str(contract_calls[0].get("name") or contract_calls[0].get("tool_name") or "") != "create_simulator_env":
        errors.append("M2 create_simulator_env must precede every M2 action")
        return errors

    cursor = 1
    # The create receipt normally includes its reset observation.  Permit one
    # explicit initial observe for a controller that needs it to derive two
    # safe Cartesian targets; it cannot substitute for the mandatory final
    # observe or for a failed-gripper recovery observe.
    if cursor < len(contract_calls) and _m2_successful_observe(
        contract_calls[cursor], semantic_nodes.get(id(contract_calls[cursor]), [])
    ):
        cursor += 1

    successful_actions: list[Mapping[str, Any]] = []
    move_targets: list[str | None] = []
    for step in range(4):
        if cursor >= len(contract_calls) or str(
            contract_calls[cursor].get("name") or contract_calls[cursor].get("tool_name") or ""
        ) != "move_to":
            errors.append("M2 requires four ordered successful moves A,B,A,B")
            return errors
        call = contract_calls[cursor]
        nodes = semantic_nodes.get(id(call), [])
        if not _m2_strict_action_success(nodes, reached_key="reached_target"):
            errors.append(f"M2 move {step + 1} is not a strict successful receipt")
        else:
            successful_actions.append(call)
        move_targets.append(_m2_move_target_key(call, nodes))
        cursor += 1
    if any(target is None for target in move_targets):
        errors.append("M2 A/B move receipts lack canonical controller targets")
    elif (
        move_targets[0] != move_targets[2]
        or move_targets[1] != move_targets[3]
        or move_targets[0] == move_targets[1]
    ):
        errors.append("M2 moves must be two distinct targets in exact A,B,A,B order")

    retry_used = False
    for step, expected_position in enumerate(M2_GRIPPER_SEQUENCE, 1):
        if cursor >= len(contract_calls) or str(
            contract_calls[cursor].get("name") or contract_calls[cursor].get("tool_name") or ""
        ) != "gripper_control":
            errors.append("M2 requires six ordered gripper steps [1,0,1,1,0,1]")
            return errors
        call = contract_calls[cursor]
        position = _m2_gripper_position(call)
        if position != expected_position:
            errors.append(
                f"M2 gripper step {step} must use position={expected_position}, got {position!r}"
            )
        nodes = semantic_nodes.get(id(call), [])
        strict_success = _m2_strict_action_success(nodes, reached_key="reached_goal")
        cursor += 1
        if strict_success:
            successful_actions.append(call)
            continue

        if retry_used:
            errors.append("M2 permits only one fresh-observe gripper retry")
            return errors
        retry_used = True
        if cursor >= len(contract_calls) or not _m2_successful_observe(
            contract_calls[cursor], semantic_nodes.get(id(contract_calls[cursor]), [])
        ):
            errors.append("M2 failed gripper step requires a fresh observe before retry")
            return errors
        cursor += 1
        if cursor >= len(contract_calls) or str(
            contract_calls[cursor].get("name") or contract_calls[cursor].get("tool_name") or ""
        ) != "gripper_control":
            errors.append("M2 failed gripper step requires exactly one same-position retry")
            return errors
        retry = contract_calls[cursor]
        retry_position = _m2_gripper_position(retry)
        retry_nodes = semantic_nodes.get(id(retry), [])
        retry_success = _m2_strict_action_success(retry_nodes, reached_key="reached_goal")
        if retry_position != expected_position:
            errors.append("M2 gripper retry must use the failed step's same position")
        if not retry_success:
            errors.append("M2 gripper retry did not strictly succeed; close and fail M2")
            return errors
        successful_actions.append(retry)
        cursor += 1

    if cursor >= len(contract_calls) or str(
        contract_calls[cursor].get("name") or contract_calls[cursor].get("tool_name") or ""
    ) != "move_to":
        errors.append("M2 requires one MOTION_PLAN_FAILED move after all gripper steps")
        return errors
    unreachable = contract_calls[cursor]
    unreachable_receipt = _m2_terminal_receipt(semantic_nodes.get(id(unreachable), []))
    if (
        unreachable_receipt.get("ok") is not False
        or unreachable_receipt.get("error_code") != "MOTION_PLAN_FAILED"
        or unreachable_receipt.get("reached_target") is True
        or unreachable_receipt.get("reached_goal") is True
    ):
        errors.append("M2 requires exactly one fail-closed MOTION_PLAN_FAILED target")
    cursor += 1

    if cursor >= len(contract_calls) or not _m2_successful_observe(
        contract_calls[cursor], semantic_nodes.get(id(contract_calls[cursor]), [])
    ):
        errors.append("M2 requires one successful observe after MOTION_PLAN_FAILED")
        return errors
    cursor += 1
    if cursor >= len(contract_calls) or str(
        contract_calls[cursor].get("name") or contract_calls[cursor].get("tool_name") or ""
    ) != "close_simulator_env":
        errors.append("M2 requires one final close_simulator_env")
        return errors
    if not _successful(contract_calls[cursor]):
        errors.append("M2 final close_simulator_env did not succeed")
    cursor += 1
    if cursor != len(contract_calls):
        errors.append("M2 contract contains skipped, duplicate, or out-of-order control actions")

    for call in successful_actions:
        nodes = semantic_nodes[id(call)]
        if len(_camera_frames(nodes)) < 2:
            errors.append("M2 successful action lacks fresh dual RGB-D")
        if not _contains(nodes, "robot") and not _contains(nodes, "joint_positions"):
            errors.append("M2 successful action lacks RobotState")
        # Start-state recovery is a MoveIt arm-motion contract.  The parallel
        # gripper action never plans an arm trajectory, so requiring a motion
        # recovery record on it would reject an otherwise valid native gripper
        # receipt.  It still needs every other post-action freshness barrier.
        if (
            str(call.get("name") or call.get("tool_name") or "") == "move_to"
            and not _contains(nodes, "schema_version", "m2_start_state_recovery_v1")
        ):
            errors.append("M2 successful move lacks recovery receipt")
        if not _contains(nodes, "action_completed_ros_time_s"):
            errors.append("M2 successful action lacks completion barrier")
        if not _contains(nodes, "observation_fresh", True):
            errors.append("M2 successful action lacks fresh-observation receipt")
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
            # Keep the durable response itself unchanged, but retain the
            # already-validated RPC association for semantic verifiers such
            # as M2.  They must never borrow a receipt from another action.
            annotated = dict(value)
            annotated[_MCP_EVIDENCE_REQUEST_ID] = request_id
            annotated[_MCP_EVIDENCE_AGENT_TOOL] = agent_tool
            payloads.append(annotated)
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
    if not any(item.get("reason_code") == "NATIVE_GRASP_TARGET_HELD" and item.get("grasp_confirmed") is True for item in records):
        errors.append("M3 child-link held proof missing")
    held = [item for item in records if item.get("reason_code") == "NATIVE_GRASP_TARGET_HELD"]
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
        and item["evidence"].get("target_id") == "target_object"
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
                == "openeta.contractual_fake_grasp_candidate.v1"
                and candidate.get("kind") == "contractual_fake_grasp_candidate"
                and candidate.get("is_model_prediction") is False
                and candidate.get("perception_source") == "gazebo_oracle"
            ):
                valid_oracle = True
    if not valid_oracle:
        errors.append("M4 Oracle output lacks a correctly labelled contractual fake candidate")
    return errors


def _verify_m5_artifact(
    paths: CasePaths,
    artifact: Any,
    *,
    label: str,
) -> tuple[Path | None, str | None]:
    if not isinstance(artifact, Mapping):
        return None, f"M5 {label} artifact descriptor is missing"
    relative = artifact.get("path")
    digest = artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        return None, f"M5 {label} artifact descriptor is malformed"
    path = (paths.root / relative).resolve()
    try:
        path.relative_to(paths.root.resolve())
    except ValueError:
        return None, f"M5 {label} artifact escapes the case directory"
    if not path.is_file():
        return None, f"M5 {label} artifact is missing"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        return None, f"M5 {label} artifact hash mismatch"
    return path, None


def _verify_m5(
    calls: Sequence[Mapping[str, Any]],
    payloads: Sequence[Mapping[str, Any]],
    paths: CasePaths,
) -> list[str]:
    """Verify the M5 perception gate before accepting the inherited M3 proof."""

    errors: list[str] = []
    evidence_path = paths.root / "m5-perception.json"
    if not evidence_path.is_file():
        return ["M5 perception evidence is missing"]
    try:
        evidence = _json_load(evidence_path)
    except (OSError, ValueError):
        return ["M5 perception evidence is unreadable"]
    if not isinstance(evidence, Mapping):
        return ["M5 perception evidence is malformed"]
    if evidence.get("schema_version") != "openeta.gazebo.m5_perception_evidence.v1":
        errors.append("M5 perception evidence schema is missing")
    if evidence.get("acceptance_scope") != "control_only_real_sam3_no_planner_not_formal_tui":
        errors.append("M5 report scope is not the restricted real-SAM3 control scope")
    endpoint = evidence.get("sam3", {}).get("endpoint_id") if isinstance(evidence.get("sam3"), Mapping) else ""
    if not isinstance(endpoint, str) or not endpoint or any(token in endpoint for token in ("?", "#", "@")):
        errors.append("M5 SAM3 endpoint identifier is not redacted")

    sam3_calls = [
        call for call in calls if str(call.get("name") or call.get("tool_name") or "") == "sam3"
    ]
    if len(sam3_calls) != 1 or not _successful(sam3_calls[0]):
        errors.append("M5 did not complete exactly one real SAM3 segment call")
    if any(_m5_selection_is_forbidden(call) for call in sam3_calls):
        errors.append("M5 SAM3 evidence contains Oracle or fake candidate data")

    evidence_status = str(evidence.get("status") or "")
    if evidence_status != "passed":
        reason = str(evidence.get("reason_code") or "unknown")
        errors.append(f"M5 perception gate did not pass: {reason}")
        if any(
            str(call.get("name") or call.get("tool_name") or "")
            in {"move_to", "follow_eef_trajectory", "gripper_control"}
            for call in calls
        ):
            errors.append("M5 entered M3 motion before its perception gate passed")
        return errors

    sam3 = evidence.get("sam3")
    if not isinstance(sam3, Mapping) or sam3.get("tool") != "segment" or sam3.get("candidate_count") != 1:
        errors.append("M5 SAM3 evidence does not prove one segment candidate")
    artifacts = sam3.get("request_response_artifacts") if isinstance(sam3, Mapping) else None
    if not isinstance(artifacts, Mapping):
        errors.append("M5 SAM3 request/response artifacts are missing")
    else:
        for name in ("request", "response", "tool_result", "mask"):
            _, artifact_error = _verify_m5_artifact(paths, artifacts.get(name), label=f"SAM3 {name}")
            if artifact_error:
                errors.append(artifact_error)

    observe = evidence.get("observe")
    if not isinstance(observe, Mapping):
        errors.append("M5 observe evidence is missing")
    else:
        for name in ("rgb", "depth"):
            _, artifact_error = _verify_m5_artifact(paths, observe.get(name), label=name)
            if artifact_error:
                errors.append(artifact_error)
        if not isinstance(observe.get("frame_id"), str) or not observe["frame_id"]:
            errors.append("M5 observe source frame is missing")
        if not isinstance(observe.get("intrinsics"), Mapping):
            errors.append("M5 observe intrinsics are missing")
        if not isinstance(observe.get("extrinsics"), Mapping):
            errors.append("M5 observe extrinsics are missing")

    selection = evidence.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("tool") != "select_sam3_detection"
        or selection.get("selection_source") != "scripted_single_candidate"
        or not isinstance(selection.get("result_id"), str)
        or not isinstance(selection.get("detection_id"), str)
    ):
        errors.append("M5 does not prove a host-only explicit single-candidate selection")
    select_calls = [
        call
        for call in calls
        if str(call.get("name") or call.get("tool_name") or "") == "select_sam3_detection"
    ]
    if len(select_calls) != 1 or not _successful(select_calls[0]):
        errors.append("M5 select_sam3_detection call is missing or failed")
    elif _m5_selection_is_forbidden(select_calls[0]):
        errors.append("M5 selection contains Oracle or fake candidate data")

    summary_meta = evidence.get("m5_object_summary")
    summary_artifact = summary_meta.get("artifact") if isinstance(summary_meta, Mapping) else None
    summary_path, summary_error = _verify_m5_artifact(
        paths, summary_artifact, label="object summary"
    )
    if summary_error:
        errors.append(summary_error)
    elif summary_path is not None:
        try:
            summary = _json_load(summary_path)
        except (OSError, ValueError):
            errors.append("M5 object summary is unreadable")
        else:
            objects = summary.get("objects") if isinstance(summary, Mapping) else None
            entry = objects[0] if isinstance(objects, list) and len(objects) == 1 and isinstance(objects[0], Mapping) else None
            position = entry.get("position") if isinstance(entry, Mapping) else None
            if (
                not isinstance(entry, Mapping)
                or entry.get("provenance") != "sam3_perception"
                or not isinstance(entry.get("source_camera"), str)
                or not isinstance(entry.get("confidence"), (int, float))
                or isinstance(entry.get("confidence"), bool)
                or not math.isfinite(float(entry.get("confidence")))
                or not isinstance(position, list)
                or len(position) != 3
                or any(
                    not isinstance(item, (int, float))
                    or isinstance(item, bool)
                    or not math.isfinite(float(item))
                    for item in position
                )
            ):
                errors.append("M5 object summary lacks a finite SAM3 world position")

    evaluation = evidence.get("ground_truth_evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("used_for_control") is not False or evaluation.get("captured_after_m3_motion") is not True:
        errors.append("M5 Gazebo truth is not restricted to post-motion evaluation")
    if not isinstance(evidence.get("m3_receipts"), list) or not evidence["m3_receipts"]:
        errors.append("M5 evidence lacks linked M3 receipts")
    errors.extend(_verify_m3(calls, payloads))
    return errors


def _verify_control_trace(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Prove the control-only runner did not silently become a planner run."""

    errors: list[str] = []
    action_events = [
        event
        for event in events
        if str(event.get("event_type") or "") in {"action", "control_error"}
    ]
    if not action_events:
        return ["control-only trace has no action or failure record"]
    for event in action_events:
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            errors.append("control-only trace payload is malformed")
            continue
        if payload.get("execution_profile") != CONTROL_ONLY:
            errors.append("control-only trace lacks its explicit execution profile")
        if payload.get("planner_invoked") is not False:
            errors.append("control-only trace does not prove planner_invoked=false")
        if payload.get("provider_invoked") is not False:
            errors.append("control-only trace does not prove provider_invoked=false")
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
        provider_billing_errors = _provider_billing_exhaustion_errors(events, calls)
        payloads: list[Mapping[str, Any]] = []
        # The provider may fail before selecting any tool.  Its durable HTTP
        # 402 exhaustion record is an external blocked condition, so do not
        # mislabel the unattempted formal tool/M2 requirements as functional
        # failures.  This exception is deliberately narrower than generic
        # planner or provider errors; see _provider_billing_exhaustion_errors.
        if mode in {DETERMINISTIC, SCRIPTED_TUI, CONTROL_ONLY} and not provider_billing_errors:
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
        if provider_billing_errors:
            errors.extend(provider_billing_errors)
        else:
            if mode == CONTROL_ONLY:
                errors.extend(_verify_control_trace(events))
                expected = ENV_IDS[milestone]
                if not any(_contains(call, "env_id", expected) for call in calls):
                    errors.append("control-only case did not create the requested milestone environment")
            else:
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
                errors.extend(_verify_m0(calls, paths, control_only=mode == CONTROL_ONLY))
            elif milestone == "m1":
                errors.extend(_verify_m1(calls, paths))
            elif milestone == "m2":
                errors.extend(_verify_m2(calls, payloads))
            elif milestone == "m3":
                errors.extend(_verify_m3(calls, payloads))
            elif milestone == "m4":
                errors.extend(_verify_m4(calls, payloads))
            elif milestone == M5_MILESTONE:
                errors.extend(_verify_m5(calls, payloads, paths))
        error_codes = {str(value) for event in events for value in _values(event, "error_code")}
        infrastructure_codes = error_codes & INFRA_CODES
        if provider_billing_errors:
            infrastructure_codes.add(PROVIDER_BILLING_EXHAUSTED)
        infra = bool(infrastructure_codes) or any(
            "inconclusive" in error for error in errors
        )
        status = "blocked" if infra else ("failed" if errors else "passed")
        return {
            "status": status,
            "errors": list(dict.fromkeys(errors)),
            "trace_paths": [str(path.resolve()) for path in trace_paths],
            "tool_call_count": len(calls),
            "infrastructure_codes": sorted(infrastructure_codes),
        }
    except (OSError, ValueError, AcceptanceError) as exc:
        return {
            "status": "blocked",
            "errors": [f"evidence unreadable: {type(exc).__name__}: {exc}"],
            "trace_paths": [],
            "tool_call_count": 0,
            "infrastructure_codes": ["EVIDENCE_UNREADABLE"],
        }


def _provider_preflight_summary(run_root: Path) -> dict[str, Any] | None:
    """Load only the safe, reportable subset of provider-preflight evidence."""

    path = run_root / PROVIDER_PREFLIGHT_FILENAME
    if not path.is_file():
        return None
    try:
        raw = _json_load(path)
    except (OSError, ValueError):
        return {
            "status": "blocked",
            "reason_code": "PROVIDER_PREFLIGHT_EVIDENCE_UNREADABLE",
        }
    if not isinstance(raw, Mapping):
        return {
            "status": "blocked",
            "reason_code": "PROVIDER_PREFLIGHT_EVIDENCE_INVALID",
        }

    def bounded_string(value: Any, *, limit: int = 240) -> str:
        return value[:limit] if isinstance(value, str) else ""

    status = str(raw.get("status") or "")
    reason_code = bounded_string(raw.get("reason_code"), limit=120)
    summary: dict[str, Any] = {
        "status": status if status in {"passed", "blocked", "failed"} else "blocked",
        "reason_code": reason_code or "PROVIDER_PREFLIGHT_EVIDENCE_INVALID",
    }
    for key in ("provider", "model", "endpoint_id"):
        value = bounded_string(raw.get(key))
        if value:
            summary[key] = value
    for key in ("vision_enabled", "fallback_used"):
        if isinstance(raw.get(key), bool):
            summary[key] = raw[key]
    for key in ("max_tokens", "elapsed_ms"):
        if isinstance(raw.get(key), (int, float)) and not isinstance(raw.get(key), bool):
            summary[key] = raw[key]
    for stage in ("model_list", "planner_smoke"):
        raw_stage = raw.get(stage)
        if not isinstance(raw_stage, Mapping):
            continue
        stage_summary: dict[str, Any] = {}
        stage_status = bounded_string(raw_stage.get("status"), limit=32)
        if stage_status:
            stage_summary["status"] = stage_status
        latency = raw_stage.get("latency_ms")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            stage_summary["latency_ms"] = latency
        if raw_stage.get("selected_model_found") is not None:
            stage_summary["selected_model_found"] = raw_stage.get("selected_model_found") is True
        schema = bounded_string(raw_stage.get("response_schema"), limit=80)
        if schema:
            stage_summary["response_schema"] = schema
        if stage_summary:
            summary[stage] = stage_summary
    return summary


def _scripted_tui_report_metadata(
    preflight: Mapping[str, Any] | None,
    *,
    milestones: Sequence[str] = MILESTONES,
) -> dict[str, Any]:
    """Keep local scripted evidence distinct from human/remote acceptance."""

    selected = tuple(milestones)
    full_m0_m4 = selected == MILESTONES
    payload: dict[str, Any] = {
        "acceptance_scope": (
            "local_automated_scripted_tui_m0_m4"
            if full_m0_m4
            else "local_automated_scripted_tui_selected_milestones"
        ),
        "selected_milestones": list(selected),
        "full_m0_m4_acceptance": full_m0_m4,
        "human_approval": "not_claimed",
        "remote_clean_clone_acceptance": "not_run",
        "m5_sam3_acceptance": "not_run",
    }
    if preflight is not None:
        payload["provider_preflight"] = dict(preflight)
    return payload


def assemble_report(
    run_root: Path,
    *,
    formal_mode: str = DETERMINISTIC,
    milestones: Sequence[str] = MILESTONES,
) -> dict[str, Any]:
    selected_milestones = tuple(milestones)
    if not selected_milestones or any(item not in MILESTONES for item in selected_milestones):
        raise AcceptanceError("formal report has an invalid selected milestone scope")
    if len(set(selected_milestones)) != len(selected_milestones):
        raise AcceptanceError("formal report selected milestone scope contains duplicates")
    milestones: dict[str, Any] = {}
    stop = False
    overall = "passed"
    scripted_tui = formal_mode == SCRIPTED_TUI
    preflight = _provider_preflight_summary(run_root) if scripted_tui else None

    # A requested provider preflight is a gate before any ROS/MCP process can
    # be started.  Report its own bounded result rather than pretending that a
    # missing M0 trace was a simulator failure.
    if preflight is not None and preflight["status"] != "passed":
        first_status = str(preflight["status"])
        reason_code = str(preflight["reason_code"])
        for milestone in selected_milestones:
            if milestone == "m0":
                backend = {
                    "status": first_status,
                    "errors": [f"provider preflight did not pass: {reason_code}"],
                    "trace_paths": [],
                    "tool_call_count": 0,
                    "infrastructure_codes": [reason_code]
                    if first_status == "blocked"
                    else [],
                }
            else:
                backend = {
                    "status": "not_run",
                    "errors": ["provider preflight gate did not pass"],
                }
            milestones[milestone] = {
                "backend_chain_status": backend,
                "planner_autonomy_status": {
                    "status": "not_applicable",
                    "errors": [],
                    "reason_code": "SCRIPTED_TUI_AUTONOMY_NOT_REQUIRED",
                },
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "run_root": str(run_root.resolve()),
            "overall_status": "inconclusive" if first_status == "blocked" else "failed",
            "milestones": milestones,
            **_scripted_tui_report_metadata(preflight, milestones=selected_milestones),
        }

    def autonomy_not_applicable() -> dict[str, Any]:
        if scripted_tui:
            return {
                "status": "not_applicable",
                "errors": [],
                "reason_code": "SCRIPTED_TUI_AUTONOMY_NOT_REQUIRED",
            }
        return {"status": "not_applicable", "errors": []}

    for milestone in selected_milestones:
        if stop:
            milestones[milestone] = {
                "backend_chain_status": {"status": "not_run", "errors": ["formal predecessor gate did not pass"]},
                "planner_autonomy_status": (
                    autonomy_not_applicable()
                    if scripted_tui
                    else {"status": "not_run", "errors": ["backend gate not passed"]}
                ),
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
            autonomy = (
                autonomy_not_applicable()
                if scripted_tui
                else {"status": "not_run", "errors": ["backend gate not passed"]}
            )
        elif scripted_tui or milestone == "m0":
            autonomy = autonomy_not_applicable()
        else:
            autonomy = verify_case(case_paths(run_root, milestone, AUTONOMY), milestone, AUTONOMY)
        milestones[milestone] = {
            "backend_chain_status": backend,
            "planner_autonomy_status": autonomy,
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_root": str(run_root.resolve()),
        "overall_status": overall,
        "milestones": milestones,
    }
    if scripted_tui:
        report.update(_scripted_tui_report_metadata(preflight, milestones=selected_milestones))
    return report


def assemble_control_report(
    run_root: Path,
    *,
    include_m5: bool = False,
) -> dict[str, Any]:
    """Assemble the no-provider control report without mimicking formal TUI output."""

    milestones: dict[str, Any] = {}
    stop = False
    overall = "passed"
    control_milestones = (*MILESTONES, M5_MILESTONE) if include_m5 else MILESTONES
    for milestone in control_milestones:
        if stop:
            milestones[milestone] = {
                "control_layer_status": {
                    "status": "not_run",
                    "errors": ["control predecessor gate did not pass"],
                },
                "formal_tui_acceptance": "not_run",
            }
            continue
        result = verify_case(
            case_paths(run_root, milestone, CONTROL_ONLY),
            milestone,
            CONTROL_ONLY,
        )
        if result["status"] != "passed":
            stop = True
            overall = "inconclusive" if result["status"] == "blocked" else "failed"
        milestones[milestone] = {
            "control_layer_status": result,
            "formal_tui_acceptance": "not_run",
        }
    scope = (
        "control_only_real_sam3_no_planner_not_formal_tui"
        if include_m5
        else "control_only_no_provider_not_formal_tui"
    )
    return {
        "schema_version": "openeta.gazebo_control_acceptance.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_root": str(run_root.resolve()),
        "acceptance_scope": scope,
        "planner_provider_invoked": False,
        "formal_tui_acceptance": "not_run",
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
    parser.add_argument(
        "--provider-preflight",
        action="store_true",
        help=(
            "For --scripted-tui only, verify the configured primary provider/model "
            "before any Gazebo or MCP case starts."
        ),
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        choices=MILESTONES,
        metavar="MILESTONE",
        help=(
            "For --scripted-tui only, run and report only the selected M0–M4 "
            "milestones. The default remains the complete M0–M4 sequence."
        ),
    )
    parser.add_argument(
        "--control-only",
        action="store_true",
        help=(
            "Exercise M0–M4 AgentTool → MCP/SSE → Gazebo control only; never "
            "start a planner/provider or claim formal PTY/TUI acceptance."
        ),
    )
    parser.add_argument(
        "--include-m5",
        action="store_true",
        help=(
            "After strict M0–M4 control gates, run the opt-in real-SAM3 M5 "
            "perception-to-M3 control check. Valid only with --control-only."
        ),
    )
    parser.add_argument(
        "--sam3-url",
        default="",
        metavar="URL",
        help="Already-running external SAM3 SSE MCP endpoint, required with --include-m5.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.control_only and args.scripted_tui:
        raise AcceptanceError("--control-only and --scripted-tui are mutually exclusive")
    if args.provider_preflight and not args.scripted_tui:
        raise AcceptanceError("--provider-preflight is valid only with --scripted-tui")
    if args.milestones and not args.scripted_tui:
        raise AcceptanceError("--milestones is valid only with --scripted-tui")
    if args.milestones and len(set(args.milestones)) != len(args.milestones):
        raise AcceptanceError("--milestones must not contain duplicates")
    if args.include_m5 and not args.control_only:
        raise AcceptanceError("--include-m5 is valid only with --control-only")
    if args.include_m5 and not args.sam3_url:
        raise AcceptanceError("--include-m5 requires --sam3-url URL")
    if args.sam3_url and not args.include_m5:
        raise AcceptanceError("--sam3-url requires --include-m5")
    if args.include_m5:
        # Validate once before creating a run root.  The original URL remains
        # in process memory only; case evidence records a redacted identifier.
        _m5_endpoint_id(args.sam3_url)
    repo = Path(__file__).resolve().parents[1]
    run_root = _new_run_root(repo, args.run_root)
    if args.control_only:
        if args.verify_only:
            report = assemble_control_report(run_root, include_m5=args.include_m5)
            report_path = run_root / CONTROL_REPORT_FILENAME
            if report_path.exists():
                raise AcceptanceError(f"immutable report already exists: {report_path}")
            _json_dump(report_path, report, exclusive=True)
            report_path.chmod(0o444)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return report_exit_code(report)

        run_root.mkdir(parents=True, exist_ok=False)
        occupied: set[int] = set()
        try:
            control_milestones = (*MILESTONES, M5_MILESTONE) if args.include_m5 else MILESTONES
            for milestone in control_milestones:
                allocation = allocate(
                    f"{milestone}-{CONTROL_ONLY}",
                    occupied,
                    # Preparing an immutable plan must remain side-effect
                    # free and usable on a host without a sourced ROS stack.
                    # Every execution path below still preflights its actual
                    # candidate domain before it can start a worker.
                    preflight=not args.prepare_only,
                )
                occupied.add(allocation.ros_domain_id)
                paths = prepare_case(repo, run_root, milestone, CONTROL_ONLY, allocation)
                if args.prepare_only:
                    continue
                code = run_case(repo, paths, allocation, sam3_url=args.sam3_url)
                if code == 130:
                    return 130
                gate = verify_case(paths, milestone, CONTROL_ONLY)
                _json_dump(paths.root / "verification.json", gate)
                if code != 0 or gate["status"] != "passed":
                    report = assemble_control_report(run_root, include_m5=args.include_m5)
                    _json_dump(run_root / CONTROL_REPORT_FILENAME, report, exclusive=True)
                    return report_exit_code(report)
            if args.prepare_only:
                print(run_root)
                return 0
            report = assemble_control_report(run_root, include_m5=args.include_m5)
            report_path = run_root / CONTROL_REPORT_FILENAME
            _json_dump(report_path, report, exclusive=True)
            report_path.chmod(0o444)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return report_exit_code(report)
        except KeyboardInterrupt:
            return 130

    formal_mode = SCRIPTED_TUI if args.scripted_tui else DETERMINISTIC
    selected_milestones = tuple(args.milestones or MILESTONES)
    if args.verify_only:
        report = assemble_report(
            run_root,
            formal_mode=formal_mode,
            milestones=selected_milestones,
        )
        report_path = run_root / "acceptance-report.json"
        if report_path.exists():
            raise AcceptanceError(f"immutable report already exists: {report_path}")
        _json_dump(report_path, report, exclusive=True)
        report_path.chmod(0o444)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report_exit_code(report)

    run_root.mkdir(parents=True, exist_ok=False)
    if args.scripted_tui and args.provider_preflight:
        preflight = _provider_preflight_result(repo)
        _json_dump(run_root / PROVIDER_PREFLIGHT_FILENAME, preflight, exclusive=True)
        if preflight["status"] != "passed":
            report = assemble_report(
                run_root,
                formal_mode=SCRIPTED_TUI,
                milestones=selected_milestones,
            )
            report_path = run_root / "acceptance-report.json"
            _json_dump(report_path, report, exclusive=True)
            report_path.chmod(0o444)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return report_exit_code(report)
    occupied: set[int] = set()
    try:
        for milestone in selected_milestones:
            modes = (
                (SCRIPTED_TUI,)
                if args.scripted_tui
                else (DETERMINISTIC,)
                if milestone in {"m0", "m3", "m4"}
                else (DETERMINISTIC, AUTONOMY)
            )
            for mode in modes:
                allocation = allocate(
                    f"{milestone}-{mode}",
                    occupied,
                    preflight=not args.prepare_only,
                )
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
                            milestones=selected_milestones,
                        )
                        _json_dump(run_root / "acceptance-report.json", report, exclusive=True)
                        return report_exit_code(report)
        if args.prepare_only:
            print(run_root)
            return 0
        report = assemble_report(
            run_root,
            formal_mode=formal_mode,
            milestones=selected_milestones,
        )
        report_path = run_root / "acceptance-report.json"
        _json_dump(report_path, report, exclusive=True)
        report_path.chmod(0o444)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report_exit_code(report)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
