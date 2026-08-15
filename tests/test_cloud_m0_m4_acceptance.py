"""The cloud coordinator requires an immutable SHA before remote TUI work."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cloud_m0_m4_acceptance.py"
TUI_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_tui_gazebo_acceptance.sh"
SPEC = importlib.util.spec_from_file_location("cloud_m0_m4_acceptance", SCRIPT)
assert SPEC and SPEC.loader
cloud = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cloud
SPEC.loader.exec_module(cloud)


def test_cloud_entry_requires_sha_and_stops_the_formal_chain(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    assert cloud.main(["--report", str(report_path)]) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["reason_code"] == "REMOTE_SHA_REQUIRED"
    assert report["milestones"]["m3"]["status"] == "not_run"
    assert report["milestones"]["m4"]["status"] == "not_run"


def test_cloud_plan_only_generates_sha_specific_tui_clean_clone_command(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    base_python = "/root/autodl-tmp/env/ros2_jazzy/bin/python"
    assert cloud.main([
        "--sha", "abc123", "--origin", "https://example.invalid/openeta.git",
        "--branch", "codex/m3-detachable-native-live",
        "--remote-python", base_python, "--report", str(report_path),
    ]) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    command = report["remote_command"]
    assert report["status"] == "not_run"
    assert "ssh" not in command.lower()
    assert "extensions/gazebo/ros2_ws" in command
    assert "colcon build" in command
    assert "--scripted-tui" in command and "--run-root" in command
    assert report["remote"]["run_root"].endswith("/abc123")
    assert report["remote"]["branch"] == "codex/m3-detachable-native-live"
    assert report["remote"]["base_python_executable"] == base_python
    assert report["remote"]["python_executable"].endswith("/venvs/abc123/bin/python")
    assert "venv --system-site-packages" in command
    assert "pip install --no-build-isolation ." in command
    assert f"OPENETA_PYTHON_EXECUTABLE={report['remote']['python_executable']}" in command
    assert "git -c http.version=HTTP/1.1 clone --depth 1 --branch codex/m3-detachable-native-live" in command


def test_cloud_plan_requires_an_absolute_verified_base_python(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    assert cloud.main([
        "--sha", "abc123", "--origin", "https://example.invalid/openeta.git",
        "--branch", "codex/m3-detachable-native-live", "--report", str(report_path),
    ]) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["reason_code"] == "REMOTE_PYTHON_REQUIRED"


def test_cloud_plan_requires_a_safe_branch(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    assert cloud.main([
        "--sha", "abc123", "--origin", "https://example.invalid/openeta.git",
        "--branch", "unsafe..branch", "--remote-python", "/opt/python", "--report", str(report_path),
    ]) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["reason_code"] == "REMOTE_BRANCH_REQUIRED"


def test_tui_runner_rejects_invalid_explicit_interpreter() -> None:
    env = dict(os.environ, OPENETA_PYTHON_EXECUTABLE="relative/python")
    result = subprocess.run(
        ["bash", str(TUI_RUNNER), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 3
    assert "OPENETA_PYTHON_EXECUTABLE_INVALID" in result.stderr


def test_tui_runner_uses_explicit_interpreter_for_help() -> None:
    python = Path(sys.executable).absolute()
    result = subprocess.run(
        ["bash", str(TUI_RUNNER), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ, OPENETA_PYTHON_EXECUTABLE=str(python)),
    )
    assert result.returncode == 0, result.stderr
    assert "--scripted-tui" in result.stdout
