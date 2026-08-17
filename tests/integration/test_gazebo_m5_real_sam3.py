"""Opt-in live M5 control-only acceptance against an external SAM3 SSE MCP."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_m5_real_sam3_control_only_acceptance(tmp_path: Path) -> None:
    """Run only when a caller explicitly supplies real SAM3 and Gazebo access."""

    if os.environ.get("OPENETA_RUN_M5_LIVE") != "1":
        pytest.skip("set OPENETA_RUN_M5_LIVE=1 to run the real M5 integration")
    sam3_url = os.environ.get("OPENETA_SAM3_URL", "").strip()
    if not sam3_url:
        pytest.skip("OPENETA_RUN_M5_LIVE=1 requires OPENETA_SAM3_URL")
    run_root = tmp_path / "m5-live"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "tui_gazebo_acceptance.py"),
            "--control-only",
            "--include-m5",
            "--sam3-url",
            sam3_url,
            "--run-root",
            str(run_root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((run_root / "control-acceptance-report.json").read_text())
    assert report["acceptance_scope"] == "control_only_real_sam3_no_planner_not_formal_tui"
    assert report["milestones"]["m5"]["control_layer_status"]["status"] == "passed"
