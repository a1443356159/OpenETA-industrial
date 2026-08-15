#!/usr/bin/env python3
"""Prepare, but never execute, the remote clean-clone PTY TUI acceptance.

This local coordinator is deliberately incapable of opening SSH, pushing a
branch, or deciding that a remote command passed.  Its output is an auditable
plan for the authorized remote operator: a SHA-specific detached clone, one
build, then the real PTY TUI M0→M4 chain.  The generated remote command itself
requires the TUI acceptance report to say ``overall_status=passed``.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any


SCHEMA_VERSION = "openeta.cloud_m0_m4_acceptance.v3"
REMOTE = "root@connect.nmb1.seetacloud.com"
REMOTE_PORT = 33584
REMOTE_ROOT = "/home/yyysaiko/.cache/openeta-cloud-acceptance"
MILESTONES = ("m0", "m1", "m2", "m3", "m4")


def _report(
    *,
    sha: str,
    status: str,
    reason_code: str | None = None,
    detail: str = "",
    remote_command: str = "",
    remote_run_root: str = "",
) -> dict[str, Any]:
    milestones: dict[str, Any] = {}
    stopped = False
    for milestone in MILESTONES:
        if stopped:
            milestones[milestone] = {
                "status": "not_run",
                "reason_code": "PREDECESSOR_GATE_NOT_PASSED",
            }
        else:
            milestones[milestone] = {
                "status": status,
                **({"reason_code": reason_code} if reason_code else {}),
            }
            stopped = status != "passed"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sha": sha,
        "status": status,
        "reason_code": reason_code,
        "detail": detail,
        "remote": {
            "host": REMOTE,
            "port": REMOTE_PORT,
            "acceptance_root": REMOTE_ROOT,
            "run_root": remote_run_root,
            "execution": "not_started_by_local_coordinator",
        },
        "remote_command": remote_command,
        "milestones": milestones,
    }


def _write(path: Path | None, report: dict[str, Any]) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(rendered, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"CLOUD_M0_M4_ACCEPTANCE_REPORT={path}")


def _remote_command(sha: str, origin: str) -> tuple[str, str]:
    """Return the command an authorized remote operator may run over SSH."""

    clone = f"{REMOTE_ROOT}/{sha}"
    run_root = f"{REMOTE_ROOT}/runs/{sha}"
    report = f"{run_root}/acceptance-report.json"
    quote = shlex.quote
    command = " && ".join(
        (
            "set -eo pipefail",
            "set +u",
            f"mkdir -p {quote(REMOTE_ROOT)}",
            f"test ! -e {quote(clone)}",
            f"git clone --no-checkout {quote(origin)} {quote(clone)}",
            f"git -C {quote(clone)} checkout --detach {quote(sha)}",
            f"test \"$(git -C {quote(clone)} rev-parse HEAD)\" = {quote(sha)}",
            f"test -z \"$(git -C {quote(clone)} status --porcelain)\"",
            f"cd {quote(clone)}",
            "source /opt/ros/jazzy/setup.bash",
            "set -u",
            "cd extensions/gazebo/ros2_ws",
            "colcon build --symlink-install",
            f"cd {quote(clone)}",
            f"test \"$(git rev-parse HEAD)\" = {quote(sha)}",
            "test -z \"$(git status --porcelain)\"",
            (
                f"OPENETA_CLOUD_ACCEPTANCE_ROOT={quote(REMOTE_ROOT)} "
                f"scripts/run_tui_gazebo_acceptance.sh --scripted-tui --run-root {quote(run_root)}"
            ),
            f"test -f {quote(report)}",
            (
                "python3 -c "
                + quote(
                    "import json,sys; "
                    f"data=json.load(open({report!r}, encoding='utf-8')); "
                    "sys.exit(0 if data.get('overall_status') == 'passed' else 1)"
                )
            ),
        )
    )
    return command, run_root


def _origin(value: str) -> str:
    if value != "origin":
        return value
    try:
        return subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("REMOTE_ORIGIN_UNAVAILABLE") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", default="")
    parser.add_argument("--origin", default="origin")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    sha = args.sha.strip()
    if not sha:
        _write(
            args.report,
            _report(
                sha="",
                status="blocked",
                reason_code="REMOTE_SHA_REQUIRED",
                detail="Push an immutable SHA before preparing remote TUI acceptance.",
            ),
        )
        return 2
    try:
        origin = _origin(args.origin)
    except RuntimeError:
        _write(
            args.report,
            _report(
                sha=sha,
                status="blocked",
                reason_code="REMOTE_ORIGIN_UNAVAILABLE",
                detail="Provide --origin with a cloneable repository URL.",
            ),
        )
        return 2
    command, run_root = _remote_command(sha, origin)
    _write(
        args.report,
        _report(
            sha=sha,
            status="not_run",
            reason_code="REMOTE_TUI_PLAN_PREPARED",
            detail="No SSH or remote command was run. An authorized operator must execute the plan and inspect its TUI report.",
            remote_command=command,
            remote_run_root=run_root,
        ),
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
