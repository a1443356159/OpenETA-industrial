"""Contracts for the task-neutral operator sorting launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import gazebo_acceptance_runtime as runtime
from scripts import open_sort_gazebo_tui as operator_session


def _receipt(*_args, **_kwargs) -> dict[str, object]:
    return {
        "schema_version": "openeta.gazebo_environment_receipt.v1",
        "trusted": True,
        "preexisting_processes": [],
    }


def test_task_neutral_session_has_no_fixture_assignment(tmp_path, monkeypatch) -> None:
    allocation = runtime.Allocation(123, "openeta-tui-open-sort", 18779, "run-id")
    monkeypatch.setattr(runtime, "_process_snapshot", lambda: [])
    monkeypatch.setattr(runtime, "environment_receipt", _receipt)

    paths = operator_session.prepare_operator_session(
        tmp_path,
        tmp_path / "run",
        allocation,
        {
            "openeta-sam3": "http://127.0.0.1:8773/sse",
            "openeta-graspgenx": "http://127.0.0.1:8778/sse",
            "openeta-anyplace": "http://127.0.0.1:8775/sse",
        },
        scenario="multi_normal",
        grasp_backend="graspgenx",
        qualification_profile="fast_v3",
    )

    assert paths.root == tmp_path / "run" / "open-sort" / "human_tui"
    instructions = paths.instructions.read_text(encoding="utf-8")
    assert "任务中立" in instructions
    assert "黄色活动扳手" not in instructions
    assert "红色六角螺栓" not in instructions
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert receipt["operator_session"] is True
    assert receipt["operator_task_source"] == "human_tui_vlm_authored"
    assert receipt["static_work_order_injected"] is False
    assert "expected_work_order" not in receipt
    assert receipt["scene"]["id"] == "multi_normal"
    config = json.loads(paths.mcp_config.read_text(encoding="utf-8"))["mcpServers"]
    assert set(config) == {
        "openeta-sim",
        "openeta-sam3",
        "openeta-graspgenx",
        "openeta-anyplace",
    }


def test_operator_metadata_carries_runtime_provenance_not_task_semantics() -> None:
    metadata = operator_session.operator_metadata(qualification_profile="fast_v3")

    assert "operator=human_tui" in metadata
    assert "planner_mode=agentic_closed_loop" in metadata
    assert "work_order_source=vlm_conversation" in metadata
    assert "grasp_target=" not in metadata
    assert "placement_region=" not in metadata
    assert "yellow" not in metadata
    with pytest.raises(ValueError, match="unsupported qualification profile"):
        operator_session.operator_metadata(qualification_profile="not-a-profile")


def test_closed_operator_session_is_not_reported_as_task_pass(tmp_path) -> None:
    paths = runtime.case_paths(tmp_path, "open-sort", runtime.HUMAN_TUI)
    paths.root.mkdir(parents=True)
    (paths.root / "host-simulator-lifecycle.json").write_text(
        json.dumps({"status": "closed", "closed": True}), encoding="utf-8"
    )
    (paths.root / "cleanup.json").write_text(
        json.dumps({"port_free": True}), encoding="utf-8"
    )

    report = operator_session.session_report(
        paths,
        run_root=tmp_path,
        scenario="multi_normal",
        tui_exit_code=0,
    )

    assert report["status"] == "closed"
    assert report["formal_acceptance_verifier"] == "not_run"
    assert report["host_environment_closed"] is True
    assert report["work_order_outcome"] == "not_configured_or_unavailable"
    assert "not a fixed acceptance PASS" in report["note"]


def test_operator_session_reports_latest_valid_work_order_progress(tmp_path) -> None:
    paths = runtime.case_paths(tmp_path, "open-sort", runtime.HUMAN_TUI)
    trace = paths.trace_root / "sessions" / "session-1" / "trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "event_type": "tool_result",
                    "payload": {
                        "multi_sort_progress": {
                            "schema_version": "openeta.multi_sort_progress.v1",
                            "assignment_count": 5,
                            "completed_count": 4,
                            "remaining_count": 1,
                            "all_completed": False,
                            "work_order": {"selection_scope": "all_catalog_targets"},
                        }
                    },
                },
                {
                    "event_type": "tool_result",
                    "payload": {
                        "multi_sort_progress": {
                            "schema_version": "openeta.multi_sort_progress.v1",
                            "assignment_count": 5,
                            "completed_count": 5,
                            "remaining_count": 0,
                            "all_completed": True,
                            "work_order": {"selection_scope": "all_catalog_targets"},
                        }
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = operator_session.session_report(
        paths,
        run_root=tmp_path,
        scenario="multi_normal",
        tui_exit_code=0,
    )

    assert report["work_order_outcome"] == "completed"
    assert report["multi_sort_progress"] == {
        "configured": True,
        "all_completed": True,
        "assignment_count": 5,
        "completed_count": 5,
        "remaining_count": 0,
        "selection_scope": "all_catalog_targets",
    }


def test_operator_session_rejects_non_multi_object_scene() -> None:
    with pytest.raises(ValueError, match="multi-object"):
        operator_session.operator_instructions(scenario="normal")


def test_shell_dispatch_only_allows_known_gazebo_runners() -> None:
    root = Path(__file__).resolve().parent.parent
    bootstrap = (root / "scripts" / "run_normal_gazebo_acceptance.sh").read_text(
        encoding="utf-8"
    )
    wrapper = (root / "scripts" / "run_open_sort_gazebo_tui.sh").read_text(
        encoding="utf-8"
    )

    assert "OPENETA_GAZEBO_RUNNER_INVALID" in bootstrap
    assert "open_sort_gazebo_tui.py" in bootstrap
    assert "OPENETA_GAZEBO_RUNNER" in wrapper
