from __future__ import annotations

import json
import os
import signal
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import scripts.tui_gazebo_acceptance as tui_acceptance
from extensions.gazebo.ros2_ws.acceptance_isolation import _normalise_graph_rows
from scripts.tui_gazebo_acceptance import (
    AUTONOMY,
    DETERMINISTIC,
    ENV_IDS,
    SCHEMA_VERSION,
    SCRIPTED_TUI,
    SIX_SIMULATOR_TOOLS,
    allocate,
    assemble_report,
    case_paths,
    environment_receipt,
    prepare_case,
    report_exit_code,
    run_case,
    scripted_tui_input,
    main,
    verify_case,
    verify_receipt,
)


ROOT = Path(__file__).parents[1]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _tool(name: str, *, parameters=None, outputs=None, receipt=None, profile="human_gated"):
    return {
        "event_type": "action",
        "payload": {
            "command": {
                "tool_calls": [
                    {
                        "name": name,
                        "parameters": parameters or {},
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "supervision": {
                                    "allowed": True,
                                    "profile": profile,
                                    "details": {"profile": profile},
                                },
                                "outputs": outputs or {},
                                **({"environment_receipt": receipt} if receipt else {}),
                            },
                        },
                    }
                ]
            }
        },
    }


def _mcp_outputs(
    paths: Path,
    agent_tool: str,
    responses: list[tuple[str, dict]],
    *,
    handle: str = "test-handle",
    session_id: str = "test-session",
) -> tuple[dict, dict]:
    """Build the same per-RPC evidence contract formal TUI cases require."""

    entries = []
    for index, (remote_tool, payload) in enumerate(responses, 1):
        request_id = f"{agent_tool}-{index}-request"
        arguments = {"session_id": session_id}
        if remote_tool != "create_env":
            arguments["handle"] = handle
        response_path = paths / "responses" / f"{agent_tool}-{index}-{remote_tool}.json"
        _write_json(response_path, payload)
        response = {
            "response_path": str(response_path),
            "response_omitted": True,
            "request_id": request_id,
            "tool": remote_tool,
            "handle": handle,
            "session_id": session_id,
        }
        receipt = {
            "remote_tool": remote_tool,
            "mcp_request_id": request_id,
            "handle": handle,
            "simulator_session_id": session_id,
            "receipt_id": f"receipt-{request_id}",
        }
        entries.append(
            {
                "request": {
                    "request_id": request_id,
                    "tool": remote_tool,
                    "arguments": arguments,
                },
                "response": response,
                "environment_receipt": receipt,
            }
        )
    assert entries
    return (
        {
            "mcp": {
                "tool": entries[0]["request"]["tool"],
                "handle": handle,
                "session_id": session_id,
                "request": entries[0]["request"],
                "response": entries[0]["response"],
            },
            "response": entries[0]["response"],
            "mcp_calls": entries,
        },
        entries[-1]["environment_receipt"],
    )


def _prepare_evidence(tmp_path: Path, milestone: str, mode: str):
    allocation = allocate(f"{milestone}-{mode}")
    paths = prepare_case(ROOT, tmp_path, milestone, mode, allocation)
    _write_json(
        paths.root / "cleanup.json",
        {
            "mcp_group_exited": True,
            "port_free": True,
            "owned_worker_groups": [],
            "owned_worker_groups_exited": True,
            "owned_process_residuals": [],
            "preexisting_process_snapshot_unchanged": True,
            "protected_ros_graphs_unchanged": True,
            "ros_graph": {"state": "PASSED"},
            "gz_partition": {"state": "PASSED"},
        },
    )
    return paths


def _m1_materialized_observe(root: Path, *, timestamp_s: float) -> tuple[dict, Path]:
    """Build one compact observe result plus its durable RGB-D response."""

    image_root = root / "images"
    rgb_path = image_root / "top-rgb.png"
    depth_path = image_root / "top-depth.png"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_path.write_bytes(b"rgb")
    depth_path.write_bytes(b"depth")
    camera = {
        "frame_id": "top_camera",
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "scale": 1000.0},
        "extrinsics": {"frame_transform": "camera_to_world"},
        "timestamp_s": timestamp_s,
    }
    response_path = root / "responses" / "observe.json"
    _write_json(
        response_path,
        {"cameras": [camera], "metadata": {"observation_provenance": "gazebo_ros_live"}},
    )
    return (
        {
            "success": True,
            "details": {
                "outputs": {
                    "response": {
                        "response_path": str(response_path),
                        "cameras": [
                            {
                                "frame_id": "top_camera",
                                "rgb_ref": "top-camera.rgb",
                                "depth_ref": "top-camera.depth",
                                "intrinsics": dict(camera["intrinsics"]),
                                "extrinsics": dict(camera["extrinsics"]),
                                "timestamp_s": timestamp_s,
                            }
                        ],
                        "image_artifacts": [
                            {"index": "top-camera.rgb", "kind": "rgb", "path": str(rgb_path)},
                            {"index": "top-camera.depth", "kind": "depth", "path": str(depth_path)},
                        ],
                        "observation_provenance": "gazebo_ros_live",
                    }
                }
            },
        },
        response_path,
    )


def test_allocation_and_receipt_exclude_protected_domains() -> None:
    allocation = allocate("unit", occupied_domains={80, 81})
    assert allocation.ros_domain_id not in {42, 80, 81, 100}
    receipt = environment_receipt(
        ROOT, allocation, case_name="unit", before=[], capture_protected=False
    )
    assert verify_receipt(receipt) == []
    assert receipt["python_executable"] == str(Path(sys.executable).absolute())
    tampered = dict(receipt, ros_domain_id=42)
    assert verify_receipt(tampered)


def test_protected_graph_rows_are_json_native_before_baseline_comparison() -> None:
    """An unchanged rclpy tuple row must equal its JSON-loaded list form."""

    live = _normalise_graph_rows(
        [("/parameter_events", ["rcl_interfaces/msg/ParameterEvent"])]
    )
    persisted = [["/parameter_events", ["rcl_interfaces/msg/ParameterEvent"]]]

    assert live == persisted


def test_m1_verifier_accepts_paired_materialized_rgbd_refs(tmp_path: Path) -> None:
    first, _ = _m1_materialized_observe(tmp_path / "first", timestamp_s=1.0)
    second, _ = _m1_materialized_observe(tmp_path / "second", timestamp_s=2.0)
    calls = [
        {"name": "observe", "result": first},
        {"name": "observe", "result": second},
    ]

    assert tui_acceptance._verify_m1(calls, SimpleNamespace(root=tmp_path)) == []


def test_m1_verifier_rejects_missing_mismatched_or_nonexistent_rgbd_artifacts(
    tmp_path: Path,
) -> None:
    missing, _ = _m1_materialized_observe(tmp_path / "missing", timestamp_s=1.0)
    missing_response = missing["details"]["outputs"]["response"]
    missing_response["image_artifacts"] = missing_response["image_artifacts"][:1]
    frames, errors = tui_acceptance._m1_camera_frames(missing, root=tmp_path)
    assert not frames and any("missing" in error for error in errors)

    mismatched, response_path = _m1_materialized_observe(tmp_path / "mismatched", timestamp_s=1.0)
    durable = json.loads(response_path.read_text(encoding="utf-8"))
    other_depth = tmp_path / "mismatched" / "images" / "other-depth.png"
    other_depth.write_bytes(b"other-depth")
    durable["cameras"][0]["depth_path"] = str(other_depth)
    _write_json(response_path, durable)
    frames, errors = tui_acceptance._m1_camera_frames(mismatched, root=tmp_path)
    assert not frames and any("does not match" in error for error in errors)

    nonexistent, _ = _m1_materialized_observe(tmp_path / "nonexistent", timestamp_s=1.0)
    nonexistent["details"]["outputs"]["response"]["image_artifacts"][0]["path"] = str(
        tmp_path / "nonexistent" / "images" / "absent.png"
    )
    frames, errors = tui_acceptance._m1_camera_frames(nonexistent, root=tmp_path)
    assert not frames and any("nonlocal" in error for error in errors)


def test_cleanup_waits_briefly_for_mcp_listener_release(monkeypatch) -> None:
    """A just-reaped process must not make cleanup race its socket teardown."""

    checks = iter((False, False, True))
    monkeypatch.setattr(tui_acceptance, "_port_is_free", lambda _port: next(checks))
    monkeypatch.setattr(tui_acceptance.time, "sleep", lambda _seconds: None)

    assert tui_acceptance._wait_for_free_port(45678, timeout_s=1.0) is True


def test_cleanup_port_wait_remains_fail_closed_for_a_bound_listener(monkeypatch) -> None:
    clock = iter((0.0, 0.2))
    monkeypatch.setattr(tui_acceptance, "_port_is_free", lambda _port: False)
    monkeypatch.setattr(tui_acceptance.time, "monotonic", lambda: next(clock))

    assert tui_acceptance._wait_for_free_port(45678, timeout_s=0.1) is False


def test_cleanup_port_probe_rejects_an_active_loopback_listener() -> None:
    """SO_REUSEADDR accepts transient TCP state, never a real listener."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    try:
        assert tui_acceptance._port_is_free(port) is False
    finally:
        listener.close()
    assert tui_acceptance._port_is_free(port) is True


def test_tui_runner_sets_a_case_local_worker_log_directory(tmp_path: Path, monkeypatch) -> None:
    """Gazebo launch diagnostics must survive in the formal case directory."""

    allocation = allocate("m1-worker-log")
    paths = case_paths(tmp_path, "m1", SCRIPTED_TUI)
    paths.root.mkdir(parents=True)
    paths.instructions.write_text("scripted M1 task", encoding="utf-8")
    ros_python_path = "/opt/ros/jazzy/lib/python3.12/site-packages"
    monkeypatch.setenv("PYTHONPATH", ros_python_path)
    _write_json(
        paths.receipt,
        {"preexisting_processes": [], "protected_ros_graphs": {}},
    )
    seen = {}

    class Process:
        pid = 12345
        returncode = 0

        @staticmethod
        def poll():
            return 0

    def popen(*_args, **kwargs):
        seen["environment"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(tui_acceptance.subprocess, "Popen", popen)
    monkeypatch.setattr(
        tui_acceptance.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(tui_acceptance, "_wait_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tui_acceptance, "_process_snapshot", lambda: [])
    monkeypatch.setattr(tui_acceptance, "_terminate_owned_worker_groups", lambda **_kwargs: [])
    monkeypatch.setattr(tui_acceptance, "_wait_for_free_port", lambda _port: True)
    monkeypatch.setattr(tui_acceptance, "_partition_cleanup", lambda _partition: {"state": "PASSED"})
    monkeypatch.setattr(tui_acceptance.shutil, "which", lambda _name: "/usr/bin/script")
    monkeypatch.setattr(tui_acceptance.os, "getpgid", lambda _pid: 12345)
    from extensions.gazebo.ros2_ws import acceptance_isolation

    monkeypatch.setattr(
        acceptance_isolation, "candidate_domain_evidence", lambda _domain: {"state": "PASSED"}
    )
    monkeypatch.setattr(
        acceptance_isolation,
        "probe_ros_graph",
        lambda _domain: {"availability": "AVAILABLE", "nodes": [], "topics": []},
    )

    assert run_case(ROOT, paths, allocation) == 0
    assert seen["environment"]["PYTHONPATH"] == os.pathsep.join((str(ROOT), ros_python_path))
    assert seen["environment"]["OPENETA_WORKER_LOG_DIR"] == str(paths.root / "worker-logs")


def test_owned_worker_cleanup_uses_only_matching_run_process_group(monkeypatch) -> None:
    """A runner must not leave (or signal) a worker outside its own case."""

    rows = [
        {
            "pid": 31001,
            "cmdline": "python sim/bench_worker.py --bench dummy --port 0",
            "openeta_tui_run_id": "this-case",
        },
        {
            "pid": 31002,
            "cmdline": "python sim/bench_worker.py --bench dummy --port 0",
            "openeta_tui_run_id": "other-case",
        },
        {
            "pid": 31003,
            "cmdline": "python sim/bench_worker.py --bench dummy --port 0",
            "openeta_tui_run_id": "this-case",
        },
    ]
    terminated: set[int] = set()

    # The server may stop the worker before the runner acts.  Cleanup must use
    # the pre-shutdown ownership snapshot rather than rediscovering nothing.
    monkeypatch.setattr(tui_acceptance, "_process_snapshot", lambda: [])
    monkeypatch.setattr(tui_acceptance.os, "getpgid", lambda pid: pid)

    def fake_killpg(pgid: int, action: int) -> None:
        if action == signal.SIGTERM:
            terminated.add(pgid)

    monkeypatch.setattr(tui_acceptance.os, "killpg", fake_killpg)
    monkeypatch.setattr(
        tui_acceptance,
        "_process_group_exited",
        lambda pgid: pgid in terminated,
    )

    evidence = tui_acceptance._terminate_owned_worker_groups(
        run_id="this-case",
        before=[{"pid": 31003}],
        candidates=rows,
        timeout_s=0.01,
    )

    assert terminated == {31001}
    assert evidence == [
        {
            "pid": 31001,
            "cmdline": "python sim/bench_worker.py --bench dummy --port 0",
            "run_id": "this-case",
            "owned": True,
            "pgid": 31001,
            "termination_signal": "SIGTERM",
            "group_exited": True,
            "state": "exited",
        }
    ]


def test_formal_verifier_rejects_missing_worker_cleanup_evidence(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m0", DETERMINISTIC)
    cleanup_path = paths.root / "cleanup.json"
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    cleanup.pop("owned_worker_groups_exited")
    _write_json(cleanup_path, cleanup)

    result = verify_case(paths, "m0", DETERMINISTIC)

    assert result["status"] == "failed"
    assert "bench-worker process groups" in " ".join(result["errors"])


def test_tool_call_reader_prefers_raw_command_over_compact_action_summary() -> None:
    """Compact episode summaries must never hide the correlated MCP result."""

    raw = {
        "kind": "tool_call",
        "name": "create_simulator_env",
        "parameters": {"env_id": ENV_IDS["m0"]},
        "status": "executed",
        "result": {
            "success": True,
            "details": {"outputs": {"mcp_calls": [{"request": {"request_id": "raw"}}]}},
        },
    }
    compact = {
        "name": "create_simulator_env",
        "status": "executed",
        "result": {"success": True, "details": {"outputs": {"mcp": {}}}},
    }
    events = [
        {
            "event_type": "action",
            "payload": {
                # The episode-step form is intentionally compact and lacks
                # mcp_calls. It must not be mistaken for evidence.
                "action": {"tool_calls": [compact]},
                "command": {"tool_calls": [raw]},
            },
        },
        {
            "event_type": "tool_execution",
            "payload": {"tool_calls": [raw]},
        },
    ]

    assert tui_acceptance._tool_calls(events) == [raw]


def test_m0_verifier_uses_trace_and_artifacts_not_planner_summary(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m0", DETERMINISTIC)
    paths.mcp_log.write_text("OpenETA MCP server started\n", encoding="utf-8")
    create_outputs, create_receipt = _mcp_outputs(
        paths.root,
        "create_simulator_env",
        [("create_env", {"handle": "test-handle"}), ("reset_env", {"success": True})],
    )
    create_outputs.update(
        {
            "environment": {"env_id": ENV_IDS["m0"]},
            "initial_observation": {"cameras": [{"rgb_path": "rgb.png"}]},
        }
    )
    observe_outputs, observe_receipt = _mcp_outputs(
        paths.root, "observe", [("render_env", {"success": True})]
    )
    close_outputs, close_receipt = _mcp_outputs(
        paths.root, "close_simulator_env", [("close_env", {"success": True})]
    )
    events = [
        _tool(
            "create_simulator_env",
            parameters={"env_id": ENV_IDS["m0"]},
            outputs=create_outputs,
            receipt=create_receipt,
        ),
        _tool("observe", outputs=observe_outputs, receipt=observe_receipt),
        _tool("close_simulator_env", outputs=close_outputs, receipt=close_receipt),
    ]
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    artifacts = paths.trace_root / "sessions/unit/working/artifacts.json"
    _write_json(artifacts, {"rgb": "rgb.png"})
    paths.transcript.write_text(
        "\n".join(
            ["/config", "/tools", "/session", "/memory all --json", *sorted(SIX_SIMULATOR_TOOLS)]
        ),
        encoding="utf-8",
    )

    result = verify_case(paths, "m0", DETERMINISTIC)

    assert result["status"] == "passed", result["errors"]
    assert result["tool_call_count"] == 3


def test_report_keeps_planner_status_separate_and_stops_formal_chain(tmp_path: Path) -> None:
    # Missing M0 evidence is a blocked infrastructure result. Later milestones
    # must be not_run rather than accidentally inferred from planner prose.
    report = assemble_report(tmp_path)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["overall_status"] == "inconclusive"
    assert report["milestones"]["m0"]["backend_chain_status"]["status"] == "blocked"
    assert report["milestones"]["m1"]["backend_chain_status"]["status"] == "not_run"
    assert report_exit_code(report) == 2


def test_scripted_tui_report_never_verifies_unrun_planner_autonomy(
    tmp_path: Path, monkeypatch
) -> None:
    """Scripted PTY runs are backend-only; absent autonomy cases are not failures."""

    seen: list[tuple[str, str]] = []

    def passed_backend(_paths, milestone: str, mode: str):
        seen.append((milestone, mode))
        return {"status": "passed", "errors": []}

    monkeypatch.setattr(tui_acceptance, "verify_case", passed_backend)

    report = assemble_report(tmp_path, formal_mode=SCRIPTED_TUI)

    assert report["overall_status"] == "passed"
    assert seen == [(milestone, SCRIPTED_TUI) for milestone in tui_acceptance.MILESTONES]
    for milestone in tui_acceptance.MILESTONES:
        autonomy = report["milestones"][milestone]["planner_autonomy_status"]
        assert autonomy == {
            "status": "not_applicable",
            "errors": [],
            "reason_code": "SCRIPTED_TUI_AUTONOMY_NOT_REQUIRED",
        }


def test_exact_pre_tool_provider_billing_exhaustion_is_blocked_not_m2_failure(
    tmp_path: Path,
) -> None:
    paths = _prepare_evidence(tmp_path, "m2", SCRIPTED_TUI)
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "event_type": "pipeline_plan",
                "payload": {
                    "metadata": {
                        "planner_metadata": {
                            "backend_status": "failed",
                            "backend_details": {
                                "error_type": "ProviderHttpError",
                                "error": "HTTP 402: Insufficient Balance",
                                "provider_attempts": 1,
                            },
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = verify_case(paths, "m2", SCRIPTED_TUI)

    assert result["status"] == "blocked"
    assert result["infrastructure_codes"] == ["PROVIDER_BILLING_EXHAUSTED"]
    assert "billing exhausted" in " ".join(result["errors"])
    assert not any("M2 requires" in error for error in result["errors"])
    assert not any("formal case has no simulator" in error for error in result["errors"])


def test_non_billing_planner_failure_remains_a_strict_m2_failure(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m2", SCRIPTED_TUI)
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "event_type": "pipeline_plan",
                "payload": {
                    "metadata": {
                        "planner_metadata": {
                            "backend_status": "failed",
                            "backend_details": {
                                "error_type": "ProviderHttpError",
                                "error": "HTTP 400: invalid planner request",
                            },
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = verify_case(paths, "m2", SCRIPTED_TUI)

    assert result["status"] == "failed"
    assert tui_acceptance.PROVIDER_BILLING_EXHAUSTED not in result["infrastructure_codes"]
    assert any("M2 requires" in error for error in result["errors"])


def test_autonomy_failure_cannot_change_backend_result_shape(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m1", AUTONOMY)
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({"event_type": "assistant_message", "payload": {}}) + "\n")
    result = verify_case(paths, "m1", AUTONOMY)
    assert result["status"] == "failed"
    assert "Planner did not create" in " ".join(result["errors"])


def test_scripted_tui_cli_prepares_only_scripted_cases(tmp_path: Path) -> None:
    run_root = tmp_path / "scripted"
    assert main(["--prepare-only", "--scripted-tui", "--run-root", str(run_root)]) == 0
    for milestone in ("m0", "m1", "m2", "m3", "m4"):
        instructions = (run_root / milestone / SCRIPTED_TUI / "operator-instructions.txt").read_text(
            encoding="utf-8"
        )
        assert "automation=scripted_tui" in instructions
        assert "human approval" in instructions
    m0_paths = case_paths(run_root, "m0", SCRIPTED_TUI)
    keys = scripted_tui_input(m0_paths)
    for command in ("/config", "/tools", "/session", "/memory all --json", "/quit"):
        assert command in keys
    submissions = keys.splitlines()
    assert submissions[:4] == ["/config", "/tools", "/session", "/memory all --json"]
    # The first planner prompt after console setup is one complete task, never
    # a standalone scripted_tui prefix that can trigger generic planning.
    assert len(submissions) == 6
    assert submissions[4].startswith("[automation=scripted_tui;")
    assert "openeta/dummy_sim-v0" in submissions[4]
    assert "\n" not in submissions[4]
    assert submissions[5] == "/quit"


def test_m3_verifier_correlates_tui_mcp_responses_ack_and_numeric_proof(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m3", DETERMINISTIC)
    paths.mcp_log.write_text("OpenETA MCP server started\n", encoding="utf-8")

    close = {
        "native_contact_gate": {
            "accepted": True,
            "left_sample_count": 3,
            "right_sample_count": 3,
            "left_span_s": 0.100,
            "right_span_s": 0.101,
            "evidence": {"target_id": "m3_target"},
        },
        "detachable_joint": {"state": "attached"},
        "physical_verification": {
            "schema_version": "openeta.m3.detachable_joint.v1",
            "reason_code": "M3_ATTACH_ACKED_UNPROVEN",
            "grasp_confirmed": False,
        },
    }
    lift = {
        "child_link_proof": {"lift_m": 0.080, "capture_relative_translation_m": 0.010},
        "physical_verification": {
            "schema_version": "openeta.m3.detachable_joint.v1",
            "reason_code": "M3_TARGET_HELD",
            "grasp_confirmed": True,
            "evidence": {"lift_m": 0.080, "capture_relative_translation_m": 0.010},
        },
    }
    opened = {
        "detachable_joint": {"state": "detached"},
        "physical_verification": {
            "schema_version": "openeta.m3.detachable_joint.v1",
            "reason_code": "READY",
            "grasp_confirmed": False,
        },
    }
    create_outputs, create_receipt = _mcp_outputs(
        paths.root,
        "create_simulator_env",
        [("create_env", {"handle": "m3-handle"}), ("reset_env", {"success": True})],
        handle="m3-handle",
        session_id="m3-session",
    )
    create_outputs["initial_observation"] = {"success": True}
    close_outputs, close_receipt = _mcp_outputs(
        paths.root,
        "gripper_control",
        [("gripper_close", close)],
        handle="m3-handle",
        session_id="m3-session",
    )
    lift_outputs, lift_receipt = _mcp_outputs(
        paths.root,
        "move_to",
        [("move_to", lift)],
        handle="m3-handle",
        session_id="m3-session",
    )
    open_outputs, open_receipt = _mcp_outputs(
        paths.root,
        "gripper_control",
        [("gripper_open", opened)],
        handle="m3-handle",
        session_id="m3-session",
    )
    environment_close_outputs, environment_close_receipt = _mcp_outputs(
        paths.root,
        "close_simulator_env",
        [("close_env", {"success": True})],
        handle="m3-handle",
        session_id="m3-session",
    )
    events = [
        _tool(
            "create_simulator_env",
            parameters={"env_id": ENV_IDS["m3"]},
            outputs=create_outputs,
            receipt=create_receipt,
        ),
        _tool(
            "gripper_control",
            outputs=close_outputs,
            receipt=close_receipt,
        ),
        _tool(
            "move_to",
            outputs=lift_outputs,
            receipt=lift_receipt,
        ),
        _tool(
            "gripper_control",
            outputs=open_outputs,
            receipt=open_receipt,
        ),
        _tool(
            "close_simulator_env",
            outputs=environment_close_outputs,
            receipt=environment_close_receipt,
        ),
    ]
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")

    assert verify_case(paths, "m3", DETERMINISTIC)["status"] == "passed"

    lift["physical_verification"]["evidence"]["lift_m"] = 0.079
    lift_path = Path(lift_outputs["response"]["response_path"])
    _write_json(lift_path, lift)
    failed = verify_case(paths, "m3", DETERMINISTIC)
    assert failed["status"] == "failed"
    assert "numeric child-link" in " ".join(failed["errors"])


def test_formal_tui_rejects_missing_or_mismatched_mcp_chain(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m0", SCRIPTED_TUI)
    paths.mcp_log.write_text("OpenETA MCP server started\n", encoding="utf-8")
    outputs, receipt = _mcp_outputs(
        paths.root,
        "create_simulator_env",
        [("create_env", {"handle": "test-handle"}), ("reset_env", {"success": True})],
    )
    outputs["initial_observation"] = {"success": True}
    observe_outputs, observe_receipt = _mcp_outputs(
        paths.root, "observe", [("render_env", {"success": True})]
    )
    close_outputs, close_receipt = _mcp_outputs(
        paths.root, "close_simulator_env", [("close_env", {"success": True})]
    )
    close_outputs["mcp_calls"][0]["environment_receipt"]["mcp_request_id"] = "wrong-request"
    events = [
        _tool("create_simulator_env", parameters={"env_id": ENV_IDS["m0"]}, outputs=outputs, receipt=receipt, profile=SCRIPTED_TUI),
        _tool("observe", outputs=observe_outputs, receipt=observe_receipt, profile=SCRIPTED_TUI),
        _tool("close_simulator_env", outputs=close_outputs, receipt=close_receipt, profile=SCRIPTED_TUI),
    ]
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    _write_json(paths.trace_root / "sessions/unit/working/artifacts.json", {"m": 1})
    paths.transcript.write_text(
        "\n".join(["/config", "/tools", "/session", "/memory all --json", *sorted(SIX_SIMULATOR_TOOLS)]),
        encoding="utf-8",
    )
    result = verify_case(paths, "m0", SCRIPTED_TUI)
    assert result["status"] == "failed"
    assert "not correlated" in " ".join(result["errors"])


def test_m4_requires_actual_oracle_output_and_truthful_fake_candidate(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m4", SCRIPTED_TUI)
    paths.mcp_log.write_text("OpenETA MCP server started\n", encoding="utf-8")
    attached = {
        "native_contact_gate": {
            "accepted": True, "left_sample_count": 3, "right_sample_count": 3,
            "left_span_s": 0.101, "right_span_s": 0.101,
            "evidence": {"target_id": "m3_target"},
        },
        "detachable_joint": {"state": "attached"},
    }
    held = {
        "physical_verification": {
            "schema_version": "openeta.m3.detachable_joint.v1",
            "reason_code": "M3_TARGET_HELD", "grasp_confirmed": True,
            "evidence": {"lift_m": 0.080, "capture_relative_translation_m": 0.010},
        }
    }
    detached = {"detachable_joint": {"state": "detached"}}
    create_outputs, create_receipt = _mcp_outputs(
        paths.root, "create_simulator_env",
        [("create_env", {"handle": "m4-handle"}), ("reset_env", {"success": True})],
        handle="m4-handle", session_id="m4-session",
    )
    gripper_close, close_receipt = _mcp_outputs(
        paths.root, "gripper_control", [("gripper_close", attached)],
        handle="m4-handle", session_id="m4-session",
    )
    move, move_receipt = _mcp_outputs(
        paths.root, "move_to", [("move_to", held)], handle="m4-handle", session_id="m4-session"
    )
    gripper_open, open_receipt = _mcp_outputs(
        paths.root, "gripper_control", [("gripper_open", detached)],
        handle="m4-handle", session_id="m4-session",
    )
    env_close, env_close_receipt = _mcp_outputs(
        paths.root, "close_simulator_env", [("close_env", {"success": True})],
        handle="m4-handle", session_id="m4-session",
    )
    candidate = {
        "schema_version": "openeta.m4.contractual_fake_grasp_candidate.v1",
        "kind": "contractual_fake_grasp_candidate",
        "candidate_id": "m4-contractual-test",
        "perception_source": "gazebo_oracle",
        "is_model_prediction": False,
        "provenance": "oracle_contract_fixture",
    }
    oracle_outputs, oracle_receipt = _mcp_outputs(
        paths.root,
        "oracle_perceive",
        [("oracle_perceive", {"success": True})],
        handle="m4-handle",
        session_id="m4-session",
    )
    oracle_outputs.update(
        {"perception_source": "gazebo_oracle", "fake_grasp_candidate": candidate}
    )
    events = [
        _tool("create_simulator_env", parameters={"env_id": ENV_IDS["m4"]}, outputs=create_outputs, receipt=create_receipt, profile=SCRIPTED_TUI),
        _tool("gripper_control", outputs=gripper_close, receipt=close_receipt, profile=SCRIPTED_TUI),
        _tool("move_to", outputs=move, receipt=move_receipt, profile=SCRIPTED_TUI),
        _tool("oracle_perceive", outputs=oracle_outputs, receipt=oracle_receipt, profile=SCRIPTED_TUI),
        _tool("gripper_control", outputs=gripper_open, receipt=open_receipt, profile=SCRIPTED_TUI),
        _tool("close_simulator_env", outputs=env_close, receipt=env_close_receipt, profile=SCRIPTED_TUI),
    ]
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    verified = verify_case(paths, "m4", SCRIPTED_TUI)
    assert verified["status"] == "passed", verified["errors"]

    candidate["is_model_prediction"] = True
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    failed = verify_case(paths, "m4", SCRIPTED_TUI)
    assert failed["status"] == "failed"
    assert "fake candidate" in " ".join(failed["errors"])

    candidate["is_model_prediction"] = False
    oracle_outputs["mcp_calls"][0]["environment_receipt"]["mcp_request_id"] = "wrong-oracle-request"
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    uncorrelated = verify_case(paths, "m4", SCRIPTED_TUI)
    assert uncorrelated["status"] == "failed"
    assert "not correlated" in " ".join(uncorrelated["errors"])
