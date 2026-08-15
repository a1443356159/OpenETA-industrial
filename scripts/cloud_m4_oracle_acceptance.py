#!/usr/bin/env python3
"""Run the live M4 oracle chain on three isolated, declared seeds.

The test remains an oracle-perception acceptance: it deliberately uses a
contract-shaped fake grasp candidate, not a real SAM3 or GraspNet inference
claim.  What is live is the MCP -> Gazebo M3 -> MoveIt -> contact-verdict
chain, which must reach ``TARGET_HELD`` for every seed.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from cloud_m0_m1_acceptance import (
    AcceptanceError,
    _cleanup,
    _segment_env,
    _start_mcp,
    _write_final,
    _wait_for_mcp,
    allocate,
)


SCHEMA_VERSION = "openeta.cloud_m4_oracle_acceptance.v1"
M3_WORLD = "m3_rm75_robotiq2f85_pickplace"
SEEDS = (301, 302, 303)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def run(*, python: str, artifact_dir: Path) -> dict[str, Any]:
    allocation = allocate("m4", "oracle")
    process = None
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": _now(),
        "oracle_boundary": {
            "perception": "gazebo_oracle",
            "grasp_candidate": "contractual_fake_candidate",
            "not_claimed": ["SAM3 inference", "GraspNet inference"],
        },
        "seeds": list(SEEDS),
        "allocation": allocation.evidence(),
    }
    try:
        env = _segment_env(allocation)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        process = _start_mcp(python, allocation, env, artifact_dir / "mcp.log")
        from agent.tools.sim_mcp import SseSimulatorMcpTransport

        transport = SseSimulatorMcpTransport(f"http://127.0.0.1:{allocation.port}/sse")
        _wait_for_mcp(transport, process)
        test_env = dict(env)
        test_env.update({
            "OPENETA_RUN_LIVE_ROS_TEST": "1",
            "OPENETA_SIM_MCP_URL": f"http://127.0.0.1:{allocation.port}/sse",
            "OPENETA_M4_ORACLE_SEEDS": ",".join(str(seed) for seed in SEEDS),
            "OPENETA_PERCEPTION_PROFILE": "oracle",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        })
        completed = subprocess.run(
            [python, "-m", "pytest", "-q", "tests/test_gazebo_m4_oracle_pick_chain.py"],
            cwd=Path.cwd(), env=test_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, timeout=90 * 60,
        )
        log = artifact_dir / "pytest.log"
        log.write_text(completed.stdout or "", encoding="utf-8")
        report["pytest"] = {"command": [python, "-m", "pytest", "-q", "tests/test_gazebo_m4_oracle_pick_chain.py"], "exit_code": completed.returncode, "stdout": str(log)}
        if completed.returncode:
            raise AcceptanceError(f"M4_ORACLE_CHAIN_FAILED: see {log}")
        report["status"] = "passed"
    finally:
        report["cleanup"] = _cleanup(allocation, process, world=M3_WORLD)
        allocation.close()
        report["finished_at_utc"] = _now()
    if report["cleanup"]["status"] != "passed":
        raise AcceptanceError("M4_CLEANUP_FAILED")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run(python=sys.executable, artifact_dir=args.artifact_dir)
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "started_at_utc": _now(), "finished_at_utc": _now(),
            "status": "blocked" if "ISOLATION_" in str(exc) or "MCP_NOT_READY" in str(exc) else "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_final(args.report, report)
        print(report["error"], file=sys.stderr)
        return 2 if report["status"] == "blocked" else 1
    _write_final(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
