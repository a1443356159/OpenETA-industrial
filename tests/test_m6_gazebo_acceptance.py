from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from scripts import m6_gazebo_acceptance as m6


M6_RUNNER = Path(__file__).resolve().parents[1] / "scripts/run_m6_gazebo_acceptance.sh"


def test_m6_prepare_registers_real_services_and_constraint_prompt(
    tmp_path, monkeypatch
) -> None:
    allocation = m6.base.Allocation(81, "openeta-tui-m6-test", 18765, "run-id")
    monkeypatch.setattr(m6.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        m6.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )
    services = dict(m6.DEFAULT_SERVICES)

    paths = m6.prepare_case(tmp_path, tmp_path / "run", allocation, services)

    config = json.loads(paths.mcp_config.read_text(encoding="utf-8"))["mcpServers"]
    assert set(config) == {
        "openeta-sim",
        "openeta-sam3",
        "openeta-anygrasp",
        "openeta-anyplace",
    }
    prompt = paths.instructions.read_text(encoding="utf-8")
    assert "AnyGrasp" in prompt and "AnyPlace" in prompt
    assert "execution_started=false" in prompt
    assert "最终\n0.5 s 判断稳定" in prompt
    assert "禁止 Oracle" in prompt
    assert "initial observation 不计作这次显式 observe" in prompt
    assert "覆盖完整目标轮廓" in prompt
    assert "红色方块 target_object" in prompt
    assert "独立 placement RGB-D" in prompt
    assert "host_candidate_compilation" in prompt
    assert "compile_placement_seed" not in prompt
    assert "source_grasp_id" in prompt
    assert "不得固定 detection id" in prompt
    assert "不得调用 python_exec" in prompt


def test_m6_order_helper_rejects_anyplace_before_lift() -> None:
    valid = ["observe", "anygrasp", "gripper_control", "move_to", "anyplace"]
    invalid = ["observe", "anygrasp", "anyplace", "gripper_control", "move_to"]

    required = ("observe", "anygrasp", "gripper_control", "move_to", "anyplace")
    assert m6._ordered(valid, required)
    assert not m6._ordered(invalid, required)


def test_m6_canonicalizes_public_grasp_tool_only_with_real_anygrasp_backend() -> None:
    assert m6._name(
        {
            "name": "grasp_pose_estimate",
            "result": {"details": {"backend": "anygrasp_mcp"}},
        }
    ) == "anygrasp"
    assert m6._name({"name": "grasp_pose_estimate"}) == "grasp_pose_estimate"


def test_m6_requires_only_executable_public_grasp_tools() -> None:
    assert "anygrasp" in m6.REQUIRED_REAL_M6_TOOLS
    assert "grasp_pose_estimate" not in m6.REQUIRED_REAL_M6_TOOLS


def test_m6_health_url_preserves_service_root() -> None:
    assert m6._health_url("http://127.0.0.1:8778/sse") == "http://127.0.0.1:8778/"


def test_m6_runtime_preflight_accepts_the_selected_overlay(
    tmp_path, monkeypatch
) -> None:
    overlay = tmp_path / "external-overlay"
    package_prefix = overlay / m6.GAZEBO_SIM_PACKAGE
    monkeypatch.setenv("OPENETA_GAZEBO_OVERLAY", str(overlay))
    monkeypatch.setattr(m6, "_ros_python_import_error", lambda: "")
    monkeypatch.setattr(m6, "_gazebo_package_prefix", lambda _package: package_prefix)
    monkeypatch.setattr(m6.shutil, "which", lambda name: f"/usr/bin/{name}")

    result = m6.gazebo_runtime_preflight(tmp_path)

    assert result["status"] == "passed"
    assert result["expected_overlay"] == str(overlay.resolve())
    assert result["package_prefix"] == str(package_prefix.resolve())
    assert result["reason_codes"] == []


def test_m6_runtime_preflight_rejects_an_overlay_from_another_checkout(
    tmp_path, monkeypatch
) -> None:
    expected = tmp_path / "extensions/gazebo/ros2_ws/install"
    monkeypatch.delenv("OPENETA_GAZEBO_OVERLAY", raising=False)
    monkeypatch.setattr(m6, "_ros_python_import_error", lambda: "")
    monkeypatch.setattr(
        m6,
        "_gazebo_package_prefix",
        lambda _package: (
            tmp_path / "stale-checkout/install" / m6.GAZEBO_SIM_PACKAGE
        ),
    )
    monkeypatch.setattr(m6.shutil, "which", lambda name: f"/usr/bin/{name}")

    result = m6.gazebo_runtime_preflight(tmp_path)

    assert result["status"] == "blocked"
    assert result["expected_overlay"] == str(expected.resolve())
    assert result["reason_codes"] == [
        "OPENETA_GAZEBO_OVERLAY_PACKAGE_MISMATCH"
    ]


def test_m6_canonical_runner_sources_ros_and_executes_m6() -> None:
    source = M6_RUNNER.read_text(encoding="utf-8")

    assert os.access(M6_RUNNER, os.X_OK)
    assert 'source "${SYSTEM_ROS_SETUP}"' in source
    assert 'source "${OVERLAY_SETUP}"' in source
    assert "import rclpy; from rosgraph_msgs.msg import Clock" in source
    assert "ros2 pkg prefix openeta_rm75_robotiq2f85_sim" in source
    assert 'm6_gazebo_acceptance.py" "$@"' in source


def test_m6_verifier_uses_model_raw_count_for_frozen_goal_requalification() -> None:
    call = {
        "result": {
            "details": {
                "outputs": {
                    "model_raw_candidate_count": 96,
                    "raw_candidate_count": 4,
                    "frozen_pregrasp_goal_count": 4,
                }
            }
        }
    }

    assert m6._has_minimum_int_value(call, "model_raw_candidate_count", 96)
    assert not m6._has_minimum_int_value(call, "raw_candidate_count", 96)


def test_m6_rejection_scenario_is_explicit_acceptance_only_fixture(
    tmp_path, monkeypatch
) -> None:
    allocation = m6.base.Allocation(81, "partition", 18765, "run-id")
    monkeypatch.setattr(m6.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        m6.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {"trusted": True},
    )

    scenario = "reject-first"
    paths = m6.prepare_case(
        tmp_path,
        tmp_path / scenario,
        allocation,
        dict(m6.DEFAULT_SERVICES),
        scenario=scenario,
    )
    prompt = paths.instructions.read_text(encoding="utf-8")
    assert scenario in prompt
    assert "execution_started=false" in prompt
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert receipt["m6_scenario"] == scenario
    unhashed = dict(receipt)
    supplied_hash = unhashed.pop("receipt_sha256")
    assert supplied_hash == m6.base.hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m6_failed_fingerprint_check_ignores_receipt_mirrors() -> None:
    event = {
        "execution_started": False,
        "request_fingerprint": "fingerprint-a",
        "result": {
            "details": {
                "environment_receipt": {
                    "execution_started": False,
                    "request_fingerprint": "fingerprint-a",
                },
                "outputs": {
                    "mcp_calls": [
                        {
                            "environment_receipt": {
                                "execution_started": False,
                                "request_fingerprint": "fingerprint-a",
                            }
                        }
                    ]
                },
            }
        },
    }

    assert not m6._repeated_failed_motion_fingerprints([event])
    assert m6._repeated_failed_motion_fingerprints([event, event]) == {
        "fingerprint-a"
    }


def test_m6_qualification_blocks_resolve_relative_to_case_root(tmp_path) -> None:
    artifact = (
        tmp_path
        / ".openeta_memory"
        / "sessions"
        / "session-a"
        / "artifacts"
        / "moveit_qualification"
        / "qualification.json"
    )
    artifact.parent.mkdir(parents=True)
    proof = {
        "results": [
            {
                "candidate_id": "placement_000",
                "verdict": "FAIL",
                "reason": "plan_only_failed",
                "execution_started": False,
                "full_plan_submitted": True,
            }
        ]
    }
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    call = {
        "qualification_artifact": {
            "kind": "json",
            "path": str(artifact.relative_to(tmp_path)),
        }
    }

    assert m6._qualification_blocks(call, artifact_root=tmp_path) == [proof]


def test_m6_qualification_blocks_include_pregrasp_joint_proof(tmp_path) -> None:
    artifact = (
        tmp_path
        / ".openeta_memory"
        / "sessions"
        / "session-a"
        / "artifacts"
        / "moveit_qualification"
        / "pregrasp-joint.json"
    )
    artifact.parent.mkdir(parents=True)
    proof = {
        "purpose": "placement",
        "results": [
            {
                "candidate_id": "pregrasp_pair_grasp_000_placement_000",
                "verdict": "FAIL",
                "reason": "plan_only_failed",
                "execution_started": False,
                "full_plan_submitted": True,
            }
        ],
    }
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    call = {
        "pregrasp_joint_qualification_artifact": {
            "kind": "json",
            "path": str(artifact.relative_to(tmp_path)),
        }
    }

    assert m6._qualification_blocks(call, artifact_root=tmp_path) == [proof]


def test_scripted_tui_quit_timeout_returns_through_cleanup_path(tmp_path, monkeypatch) -> None:
    instructions = tmp_path / "instructions.txt"
    instructions.write_text("task\n", encoding="utf-8")
    paths = SimpleNamespace(
        root=tmp_path,
        transcript=tmp_path / "tui.transcript",
        instructions=instructions,
    )

    class Stdin:
        closed = False

        def write(self, value):
            return len(value)

        def flush(self):
            return None

        def close(self):
            self.closed = True

    class Process:
        pid = 12345
        stdin = Stdin()

        def poll(self):
            return None

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("tui", timeout)

    process = Process()
    terminated = []
    monkeypatch.setattr(m6.base.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(m6.base, "_wait_for_scripted_tui_episode", lambda *args, **kwargs: "completed")
    monkeypatch.setattr(m6.base, "_terminate_scripted_tui_process", lambda value: terminated.append(value))

    assert m6.base._run_scripted_tui("tui", paths, {}) == 1
    assert terminated == [process]
    evidence = json.loads((tmp_path / "scripted-tui-driver.json").read_text())
    assert evidence["reason_code"] == "TUI_DID_NOT_EXIT_AFTER_QUIT"
