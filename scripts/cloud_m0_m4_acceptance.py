#!/usr/bin/env python3
"""Non-interactive, clean-checkout acceptance coordinator for Gazebo M0--M4.

This is deliberately an *orchestrator*, not a second implementation of the
milestone drivers.  M2 and M3 continue to own their existing isolated live
acceptance scripts; M0/M1 and M4 have small dedicated drivers because their
acceptance contracts are different.  A formal invocation always runs from a
fresh, detached checkout of a commit that is already reachable from ``origin``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "openeta.cloud_m0_m4_acceptance.v1"
MILESTONES = ("m0", "m1", "m2", "m3", "m4")
SUCCESS, FAILED, BLOCKED = 0, 1, 2


class CloudAcceptanceError(RuntimeError):
    """Expected preflight or acceptance error with a stable reason code."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}{': ' + detail if detail else ''}")


@dataclass(frozen=True, slots=True)
class CleanCheckout:
    source_repo: Path
    checkout: Path
    commit: str
    origin_ref: str


@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path
    logs: Path
    reports: Path
    total_report: Path
    stdout: Path


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    """Write a report exactly once; final evidence must never be overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _run(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None,
    timeout_s: float | None = None, stdout_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded command, teeing complete stdout/stderr to an artifact."""
    merged = os.environ.copy()
    if env:
        merged.update({key: str(value) for key, value in env.items()})
    try:
        completed = subprocess.run(
            list(command), cwd=cwd, env=merged, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise CloudAcceptanceError("COMMAND_UNAVAILABLE", str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if stdout_path is not None:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text(output, encoding="utf-8")
        raise CloudAcceptanceError("COMMAND_TIMEOUT", " ".join(command)) from exc
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    return completed


def _command_text(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    return (completed.stdout or "").strip()


def _git(repo: Path, *args: str) -> str:
    completed = _run(("git", "-C", str(repo), *args), cwd=repo)
    if completed.returncode:
        raise CloudAcceptanceError("GIT_COMMAND_FAILED", completed.stdout.strip()[:1000])
    return completed.stdout.strip()


def _origin_refs(repo: Path) -> dict[str, str]:
    output = _git(repo, "ls-remote", "--refs", "origin")
    refs: dict[str, str] = {}
    for line in output.splitlines():
        sha, _, ref = line.partition("\t")
        if len(sha) == 40 and ref:
            refs[ref] = sha
    return refs


def resolve_clean_commit(source_repo: Path) -> tuple[str, str]:
    """Require a clean source and a commit present in an immutable origin ref."""
    source_repo = source_repo.resolve()
    if not (source_repo / ".git").exists():
        raise CloudAcceptanceError("SOURCE_REPOSITORY_REQUIRED", str(source_repo))
    if _git(source_repo, "status", "--porcelain=v1"):
        raise CloudAcceptanceError("SOURCE_WORKTREE_DIRTY")
    commit = _git(source_repo, "rev-parse", "HEAD")
    origin_refs = _origin_refs(source_repo)
    matching = sorted(ref for ref, sha in origin_refs.items() if sha == commit)
    if not matching:
        raise CloudAcceptanceError("HEAD_NOT_AT_ORIGIN", commit)
    return commit, matching[0]


def _checkout_path(work_root: Path, commit: str) -> Path:
    # Each call gets a new directory rather than deleting an old checkout.  A
    # collision is evidence of an operator retry and must be resolved openly.
    return work_root / f"openeta-cloud-{commit[:12]}-{_utc_now()}"


def create_clean_checkout(source_repo: Path, work_root: Path) -> CleanCheckout:
    """Clone the exact origin-reachable SHA into the designated data disk."""
    commit, origin_ref = resolve_clean_commit(source_repo)
    work_root = work_root.resolve()
    if not work_root.is_absolute():  # pragma: no cover - resolve is absolute; documents invariant
        raise CloudAcceptanceError("DATA_DISK_PATH_MUST_BE_ABSOLUTE")
    source = source_repo.resolve()
    if work_root == source or source in work_root.parents or work_root in source.parents:
        raise CloudAcceptanceError("DATA_DISK_PATH_OVERLAPS_SOURCE", str(work_root))
    work_root.mkdir(parents=True, exist_ok=True)
    checkout = _checkout_path(work_root, commit)
    if checkout.exists():
        raise CloudAcceptanceError("CLEAN_CHECKOUT_PATH_EXISTS", str(checkout))
    origin_url = _git(source_repo, "remote", "get-url", "origin")
    clone = _run(
        ("git", "clone", "--no-checkout", "--origin", "origin", origin_url, str(checkout)),
        cwd=work_root,
    )
    if clone.returncode:
        raise CloudAcceptanceError("CLEAN_CLONE_FAILED", clone.stdout.strip()[:1000])
    _git(checkout, "fetch", "--quiet", "origin", commit)
    _git(checkout, "checkout", "--detach", "--quiet", commit)
    observed = _git(checkout, "rev-parse", "HEAD")
    if observed != commit:
        raise CloudAcceptanceError("CLEAN_CHECKOUT_SHA_MISMATCH", f"expected={commit} actual={observed}")
    if _git(checkout, "status", "--porcelain=v1"):
        raise CloudAcceptanceError("CLEAN_CHECKOUT_DIRTY")
    # Re-query the clone's remote after fetch: this proves the checkout SHA is
    # available from the clone's own origin, not just the initiating worktree.
    clone_refs = _origin_refs(checkout)
    if clone_refs.get(origin_ref) != commit:
        raise CloudAcceptanceError("CLEAN_CHECKOUT_ORIGIN_MISMATCH", origin_ref)
    return CleanCheckout(source_repo.resolve(), checkout, commit, origin_ref)


def collect_host_evidence(checkout: CleanCheckout, *, python: str) -> dict[str, Any]:
    packages = (
        "gz-sim8", "gz-plugin2", "gz-transport13", "ros-jazzy-ros-gz",
        "ros-jazzy-moveit", "ros-jazzy-ros2-control", "ros-jazzy-ros2-controllers",
    )
    apt_versions = {
        package: _command_text(("dpkg-query", "-W", "-f=${Version}", package))
        for package in packages
    }
    overlay = checkout.checkout / "extensions/gazebo/ros2_ws/install/setup.bash"
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": python,
        "ubuntu_release": _command_text(("lsb_release", "-ds")),
        "kernel": _command_text(("uname", "-a")),
        "gpu": _command_text(("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")),
        "disk": _command_text(("df", "-h", str(checkout.checkout))),
        "mount": _command_text(("findmnt", "-T", str(checkout.checkout), "-no", "TARGET,SOURCE,FSTYPE,OPTIONS")),
        "apt_vendor_versions": apt_versions,
        "gz_sim_versions": _command_text(("gz", "sim", "--force-version", "8", "--versions")),
        "overlay": str(overlay),
        "overlay_present_before_build": overlay.is_file(),
        "git": {
            "head": _git(checkout.checkout, "rev-parse", "HEAD"),
            "origin_ref": checkout.origin_ref,
            "origin_sha": _origin_refs(checkout.checkout).get(checkout.origin_ref, ""),
            "clean": not bool(_git(checkout.checkout, "status", "--porcelain=v1")),
        },
    }


def _resolve_python(checkout: Path, supplied: str | None) -> str:
    candidate = Path(supplied).expanduser() if supplied else checkout / ".venv/bin/python"
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise CloudAcceptanceError("PYTHON_NOT_READY", str(candidate))
    # Keep the venv entry point rather than resolving its interpreter symlink.
    # Python derives ``sys.prefix`` from the invoked venv path; resolving it
    # to /usr/bin/python drops the venv's third-party dependencies.
    return str(candidate)


def _checkout_pythonpath(checkout: Path) -> str:
    """Put the checkout first without discarding ROS setup's Python packages."""
    inherited = os.environ.get("PYTHONPATH", "")
    return os.pathsep.join(item for item in (str(checkout), inherited) if item)


def _prepare_paths(checkout: CleanCheckout) -> RunPaths:
    stamp = _utc_now()
    token = hashlib.sha256(f"{checkout.commit}:{stamp}:{os.getpid()}".encode()).hexdigest()[:12]
    root = checkout.checkout / ".cache" / f"cloud-m0-m4-{stamp}-{checkout.commit[:12]}-{token}"
    root.mkdir(parents=True, exist_ok=False)
    logs = root / "logs"
    reports = root / "reports"
    logs.mkdir()
    reports.mkdir()
    stdout = root / "stdout.log"
    stdout.write_text("", encoding="utf-8")
    return RunPaths(root, logs, reports, root / "cloud-m0-m4-report.json", stdout)


def _append_stdout(paths: RunPaths, label: str, source: Path) -> None:
    """Keep one ordered transcript in addition to the per-command full logs."""
    content = source.read_text(encoding="utf-8", errors="replace") if source.is_file() else ""
    with paths.stdout.open("a", encoding="utf-8") as stream:
        stream.write(f"\n===== {label} =====\n")
        stream.write(content)
        if content and not content.endswith("\n"):
            stream.write("\n")


def _build_once(checkout: Path, paths: RunPaths, env: Mapping[str, str]) -> None:
    completed = _run(
        ("bash", "extensions/gazebo/ros2_ws/build.sh"), cwd=checkout, env=env,
        timeout_s=45 * 60, stdout_path=paths.logs / "build.log",
    )
    if completed.returncode:
        raise CloudAcceptanceError("ROS_WORKSPACE_BUILD_FAILED", str(paths.logs / "build.log"))


def _milestone_command(checkout: Path, python: str, name: str, paths: RunPaths) -> tuple[str, ...]:
    if name in {"m0", "m1"}:
        return (
            python, "scripts/cloud_m0_m1_acceptance.py", name,
            "--report", str(paths.reports / f"{name}.json"),
            "--artifact-dir", str(paths.root / name),
        )
    if name == "m2":
        return ("bash", "extensions/gazebo/ros2_ws/run_m2_robotiq2f85_smoke.sh")
    if name == "m3":
        return ("bash", "extensions/gazebo/ros2_ws/run_m3_pickplace_acceptance.sh")
    return (
        python, "scripts/cloud_m4_oracle_acceptance.py",
        "--report", str(paths.reports / "m4.json"),
        "--artifact-dir", str(paths.root / "m4"),
    )


def _report_for_milestone(paths: RunPaths, milestone: str) -> Path | None:
    explicit = paths.reports / f"{milestone}.json"
    if explicit.is_file():
        return explicit
    # M2/M3 retain their stable report formats and names.  The shell runners
    # write them under this common artifact root through the env hook.
    candidates = sorted((paths.root / "reports").glob(f"{milestone}-*.json"))
    return candidates[-1] if candidates else None


def _result_status(report: Path | None) -> str:
    if report is None:
        return "failed"
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "failed"
    status = str(payload.get("status") or payload.get("overall_status") or "").lower()
    if status in {"passed", "pass"}:
        return "passed"
    # Existing M2/M3 reports do not have one top-level status.  Their gates
    # are authoritative and the scripts only return zero after they pass.
    gates = payload.get("gates")
    if isinstance(gates, Mapping) and gates and all(
        isinstance(value, Mapping) and value.get("status") == "passed"
        for value in gates.values()
    ):
        return "passed"
    if status in {"blocked", "inconclusive"}:
        return "blocked"
    return "failed"


def _final_status(records: Mapping[str, Mapping[str, Any]]) -> tuple[str, int]:
    statuses = [str(value.get("status")) for value in records.values()]
    if all(status == "passed" for status in statuses) and len(statuses) == len(MILESTONES):
        return "passed", SUCCESS
    if "blocked" in statuses:
        return "blocked", BLOCKED
    return "failed", FAILED


def orchestrate(*, source_repo: Path, work_root: Path, python_arg: str | None, dry_run: bool = False) -> tuple[int, Path | None]:
    """Build once and run M0--M4 serially, stopping before every unsafe successor."""
    checkout: CleanCheckout | None = None
    paths: RunPaths | None = None
    records: dict[str, dict[str, Any]] = {}
    try:
        checkout = create_clean_checkout(source_repo, work_root)
        python = _resolve_python(checkout.checkout, python_arg)
        paths = _prepare_paths(checkout)
        base_env = {
            # /opt/ros/*/setup.bash exposes rclpy and generated message
            # packages through PYTHONPATH.  The acceptance checkout must win
            # imports, but replacing that inherited path makes every graph
            # isolation probe inconclusive on an otherwise healthy ROS host.
            "PYTHONPATH": _checkout_pythonpath(checkout.checkout),
            "OPENETA_ACCEPTANCE_PYTHON": python,
            "OPENETA_ACCEPTANCE_SKIP_BUILD": "1",
            "OPENETA_ACCEPTANCE_ARTIFACT_DIR": str(paths.root),
            "OPENETA_ACCEPTANCE_RUN_TAG": paths.root.name,
            "OPENETA_CLOUD_RUN_ID": paths.root.name,
            "OPENETA_GAZEBO_OVERLAY": str(
                checkout.checkout / "extensions/gazebo/ros2_ws/install"
            ),
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "checkout": {
                "path": str(checkout.checkout), "commit": checkout.commit,
                "origin_ref": checkout.origin_ref,
            },
            "host": collect_host_evidence(checkout, python=python),
            "milestones": records,
        }
        if dry_run:
            manifest["status"] = "dry_run"
            manifest["commands"] = {
                name: list(_milestone_command(checkout.checkout, python, name, paths))
                for name in MILESTONES
            }
            manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
            _json_write(paths.total_report, manifest)
            return SUCCESS, paths.total_report

        _build_once(checkout.checkout, paths, base_env)
        _append_stdout(paths, "build", paths.logs / "build.log")
        manifest["host"]["overlay_present_after_build"] = (
            checkout.checkout / "extensions/gazebo/ros2_ws/install/setup.bash"
        ).is_file()
        for milestone in MILESTONES:
            command = _milestone_command(checkout.checkout, python, milestone, paths)
            started = datetime.now(UTC).isoformat()
            completed = _run(
                command, cwd=checkout.checkout, env=base_env, timeout_s=90 * 60,
                stdout_path=paths.logs / f"{milestone}.log",
            )
            _append_stdout(paths, milestone, paths.logs / f"{milestone}.log")
            report = _report_for_milestone(paths, milestone)
            status = "passed" if completed.returncode == 0 and _result_status(report) == "passed" else _result_status(report)
            if status == "passed" and completed.returncode != 0:
                status = "failed"
            records[milestone] = {
                "status": status,
                "exit_code": completed.returncode,
                "started_at_utc": started,
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "command": list(command),
                "stdout": str(paths.logs / f"{milestone}.log"),
                "report": str(report) if report else None,
            }
            if status != "passed":
                for later in MILESTONES[MILESTONES.index(milestone) + 1:]:
                    records[later] = {
                        "status": "not_run", "reason_code": "PREDECESSOR_GATE_NOT_PASSED",
                        "predecessor": milestone,
                    }
                break
        status, code = _final_status(records)
        manifest["status"] = status
        manifest["exit_code"] = code
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        _json_write(paths.total_report, manifest)
        return code, paths.total_report
    except CloudAcceptanceError as exc:
        if paths is None:
            print(str(exc), file=sys.stderr)
            return BLOCKED, None
        records["preflight"] = {"status": "blocked", "reason_code": exc.reason_code, "detail": str(exc)}
        _json_write(paths.total_report, {
            "schema_version": SCHEMA_VERSION,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "status": "blocked", "exit_code": BLOCKED,
            "checkout": {"path": str(checkout.checkout), "commit": checkout.commit} if checkout else {},
            "milestones": records,
        })
        return BLOCKED, paths.total_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--work-root", type=Path,
        default=Path(os.environ.get("OPENETA_CLOUD_ACCEPTANCE_ROOT", "/data/openeta-cloud-acceptance")),
        help="empty/dedicated parent on the cloud data disk; a fresh clone is created below it",
    )
    parser.add_argument("--python", dest="python_arg", default=os.environ.get("OPENETA_ACCEPTANCE_PYTHON"))
    parser.add_argument("--dry-run", action="store_true", help="clone and record commands, but do not build or run milestones")
    args = parser.parse_args()
    code, report = orchestrate(
        source_repo=args.source_repo, work_root=args.work_root,
        python_arg=args.python_arg, dry_run=args.dry_run,
    )
    if report is not None:
        print(f"CLOUD_M0_M4_ACCEPTANCE_REPORT={report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
