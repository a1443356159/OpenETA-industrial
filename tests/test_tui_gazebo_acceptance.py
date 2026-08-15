from __future__ import annotations

import json
import sys
from pathlib import Path

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
            "owned_process_residuals": [],
            "preexisting_process_snapshot_unchanged": True,
            "protected_ros_graphs_unchanged": True,
            "ros_graph": {"state": "PASSED"},
            "gz_partition": {"state": "PASSED"},
        },
    )
    return paths


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
