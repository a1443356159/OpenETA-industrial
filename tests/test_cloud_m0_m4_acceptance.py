"""The cloud coordinator requires an immutable SHA before remote TUI work."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cloud_m0_m4_acceptance.py"
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
    assert cloud.main(["--sha", "abc123", "--origin", "https://example.invalid/openeta.git", "--report", str(report_path)]) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    command = report["remote_command"]
    assert report["status"] == "not_run"
    assert "ssh" not in command.lower()
    assert "extensions/gazebo/ros2_ws" in command
    assert "colcon build" in command
    assert "--scripted-tui" in command and "--run-root" in command
    assert report["remote"]["run_root"].endswith("/abc123")
