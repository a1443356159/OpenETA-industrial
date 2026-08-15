"""The former cloud coordinator must remain unable to start M3/M4 work."""

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


def test_cloud_entry_reports_m3_blocked_and_never_schedules_m4(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    assert cloud.main(["--report", str(report_path)]) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["reason_code"] == cloud.UNAVAILABLE_REASON
    assert report["milestones"]["m3"]["status"] == "blocked"
    assert report["milestones"]["m4"]["status"] == "not_run"
