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
import re
import shlex
import subprocess
from typing import Any


SCHEMA_VERSION = "openeta.cloud_m0_m4_acceptance.v3"
REMOTE = "root@connect.nmb1.seetacloud.com"
REMOTE_PORT = 33584
REMOTE_ROOT = "/home/yyysaiko/.cache/openeta-cloud-acceptance"
MILESTONES = ("m0", "m1", "m2", "m3", "m4")
# A virtualenv based on a host-managed ROS Python can still expose that
# installation's generated packages and native typesupport libraries.  Gazebo
# workers source the one supported Jazzy stack separately, so acceptance uses
# the OS CPython only.  Both paths resolve to the Ubuntu 24.04 Python 3.12
# interpreter on the approved remote image.
TRUSTED_REMOTE_PYTHONS = frozenset({"/usr/bin/python3", "/usr/bin/python3.12"})


def _report(
    *,
    sha: str,
    status: str,
    reason_code: str | None = None,
    detail: str = "",
    remote_command: str = "",
    remote_run_root: str = "",
    remote_base_python: str = "",
    remote_venv_python: str = "",
    branch: str = "",
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
            "base_python_executable": remote_base_python,
            "python_executable": remote_venv_python,
            "branch": branch,
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


def _remote_command(
    sha: str,
    origin: str,
    remote_python: str,
    branch: str,
) -> tuple[str, str, str]:
    """Return the command an authorized remote operator may run over SSH."""

    clone = f"{REMOTE_ROOT}/{sha}"
    run_root = f"{REMOTE_ROOT}/runs/{sha}"
    venv = f"{REMOTE_ROOT}/venvs/{sha}"
    venv_python = f"{venv}/bin/python"
    report = f"{run_root}/acceptance-report.json"
    quote = shlex.quote
    command = " && ".join(
        (
            "set -eo pipefail",
            "set +u",
            f"mkdir -p {quote(REMOTE_ROOT)}",
            f"test ! -e {quote(clone)}",
            f"test ! -e {quote(venv)}",
            (
                f"git -c http.version=HTTP/1.1 clone --depth 1 --branch {quote(branch)} "
                f"--no-checkout {quote(origin)} {quote(clone)}"
            ),
            f"git -C {quote(clone)} checkout --detach {quote(sha)}",
            f"test \"$(git -C {quote(clone)} rev-parse HEAD)\" = {quote(sha)}",
            f"test -z \"$(git -C {quote(clone)} status --porcelain)\"",
            (
                f"{quote(remote_python)} -c "
                + quote("import sys; assert sys.version_info[:2] == (3, 12)")
            ),
            f"{quote(remote_python)} -m venv {quote(venv)}",
            f"cd {quote(clone)}",
            (
                f"{quote(venv_python)} -m pip install --upgrade "
                f"{quote('pip>=24')} {quote('setuptools>=68')} {quote('wheel>=0.42')}"
            ),
            (
                f"{quote(venv_python)} -m pip install --no-build-isolation . "
                f"{quote('pytest>=8')}"
            ),
            (
                f"{quote(venv_python)} -c "
                + quote("import pytest, gymnasium, numpy, prompt_toolkit, mcp, starlette, uvicorn")
            ),
            "source /opt/ros/jazzy/setup.bash",
            "set -u",
            "cd extensions/gazebo/ros2_ws",
            "colcon build --symlink-install",
            f"cd {quote(clone)}",
            f"test \"$(git rev-parse HEAD)\" = {quote(sha)}",
            "test -z \"$(git status --porcelain)\"",
            (
                f"OPENETA_CLOUD_ACCEPTANCE_ROOT={quote(REMOTE_ROOT)} "
                f"OPENETA_PYTHON_EXECUTABLE={quote(venv_python)} "
                f"scripts/run_tui_gazebo_acceptance.sh --scripted-tui --run-root {quote(run_root)}"
            ),
            f"test -f {quote(report)}",
            (
                f"{quote(venv_python)} -c "
                + quote(
                    "import json,sys; "
                    f"data=json.load(open({report!r}, encoding='utf-8')); "
                    "sys.exit(0 if data.get('overall_status') == 'passed' else 1)"
                )
            ),
        )
    )
    return command, run_root, venv_python


def _origin(value: str) -> str:
    if value != "origin":
        return value
    try:
        return subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("REMOTE_ORIGIN_UNAVAILABLE") from exc


def _safe_branch(value: str) -> bool:
    """Accept only a simple, non-ambiguous Git branch ref for the clone plan."""

    return bool(
        value
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value)
        and ".." not in value
        and "//" not in value
        and "@{" not in value
        and not value.endswith(("/", "."))
    )


def _trusted_remote_python(value: str) -> bool:
    """Accept only the system CPython approved for the remote Jazzy image."""

    return value in TRUSTED_REMOTE_PYTHONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", default="")
    parser.add_argument("--origin", default="origin")
    parser.add_argument("--branch", default="", help="Required branch containing the SHA.")
    parser.add_argument(
        "--remote-python",
        default="",
        help="Verified absolute Python executable on the remote host.",
    )
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
    branch = args.branch.strip()
    if not _safe_branch(branch):
        _write(
            args.report,
            _report(
                sha=sha,
                status="blocked",
                reason_code="REMOTE_BRANCH_REQUIRED",
                detail="Provide a non-empty safe --branch containing the requested SHA.",
            ),
        )
        return 2
    remote_python = args.remote_python.strip()
    if not remote_python or not Path(remote_python).is_absolute():
        _write(
            args.report,
            _report(
                sha=sha,
                status="blocked",
                reason_code="REMOTE_PYTHON_REQUIRED",
                detail=(
                    "Provide --remote-python with the verified absolute interpreter "
                    "for the detached clean clone."
                ),
            ),
        )
        return 2
    if not _trusted_remote_python(remote_python):
        _write(
            args.report,
            _report(
                sha=sha,
                status="blocked",
                reason_code="REMOTE_PYTHON_UNTRUSTED",
                detail=(
                    "Use the remote OS CPython (/usr/bin/python3 or "
                    "/usr/bin/python3.12), not a host-managed ROS Python."
                ),
                branch=branch,
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
    command, run_root, venv_python = _remote_command(
        sha,
        origin,
        remote_python,
        branch,
    )
    _write(
        args.report,
        _report(
            sha=sha,
            status="not_run",
            reason_code="REMOTE_TUI_PLAN_PREPARED",
            detail="No SSH or remote command was run. An authorized operator must execute the plan and inspect its TUI report.",
            remote_command=command,
            remote_run_root=run_root,
            remote_base_python=remote_python,
            remote_venv_python=venv_python,
            branch=branch,
        ),
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
