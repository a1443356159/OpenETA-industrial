from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from scripts import m6_gazebo_acceptance as m6


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
        "openeta-anyplace",
        "openeta-graspgenx",
    }
    prompt = paths.instructions.read_text(encoding="utf-8")
    assert "GraspGenX" in prompt and "AnyPlace" in prompt
    assert "execution_started=false" in prompt
    assert "稳定 >=0.5 s" in prompt
    assert "禁止 Oracle" in prompt
    assert "initial observation 不计作这次显式 observe" in prompt
    assert "覆盖完整目标轮廓" in prompt
    assert "红色方块 target_object" in prompt
    assert "不得固定 detection id" in prompt


def test_m6_order_helper_rejects_anyplace_before_lift() -> None:
    valid = ["observe", "graspgenx", "gripper_control", "move_to", "anyplace", "compile_grasp_seed"]
    invalid = ["observe", "graspgenx", "anyplace", "gripper_control", "move_to", "compile_grasp_seed"]

    required = ("observe", "graspgenx", "gripper_control", "move_to", "anyplace", "compile_grasp_seed")
    assert m6._ordered(valid, required)
    assert not m6._ordered(invalid, required)


def test_m6_canonicalizes_public_grasp_tool_only_with_real_graspgenx_backend() -> None:
    assert m6._name(
        {
            "name": "grasp_pose_estimate",
            "result": {"details": {"backend": "graspgenx_mcp"}},
        }
    ) == "graspgenx"
    assert m6._name({"name": "grasp_pose_estimate"}) == "grasp_pose_estimate"


def test_m6_requires_only_executable_public_grasp_tools() -> None:
    assert "graspgenx" in m6.REQUIRED_REAL_M6_TOOLS
    assert "list_graspgenx_grippers" not in m6.REQUIRED_REAL_M6_TOOLS


def test_m6_health_url_preserves_service_root() -> None:
    assert m6._health_url("http://127.0.0.1:8778/sse") == "http://127.0.0.1:8778/"


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
