from __future__ import annotations

import json
from pathlib import Path

from scripts.tui_gazebo_acceptance import (
    AUTONOMY,
    DETERMINISTIC,
    ENV_IDS,
    SCHEMA_VERSION,
    SIX_SIMULATOR_TOOLS,
    allocate,
    assemble_report,
    case_paths,
    environment_receipt,
    prepare_case,
    report_exit_code,
    verify_case,
    verify_receipt,
)


ROOT = Path(__file__).parents[1]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _tool(name: str, *, parameters=None, outputs=None):
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
                                    "details": {"profile": "human_gated"},
                                },
                                "outputs": outputs or {},
                            },
                        },
                    }
                ]
            }
        },
    }


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
    tampered = dict(receipt, ros_domain_id=42)
    assert verify_receipt(tampered)


def test_m0_verifier_uses_trace_and_artifacts_not_planner_summary(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m0", DETERMINISTIC)
    events = [
        _tool(
            "create_simulator_env",
            parameters={"env_id": ENV_IDS["m0"]},
            outputs={
                "environment": {"env_id": ENV_IDS["m0"]},
                "initial_observation": {"cameras": [{"rgb_path": "rgb.png"}]},
            },
        ),
        _tool("observe", outputs={"cameras": [{"rgb_path": "rgb.png"}]}),
        _tool("close_simulator_env"),
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
    report = assemble_report(tmp_path, direct_gates={})
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
