"""Offline contracts for the clean-SHA cloud acceptance coordinator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cloud_m0_m4_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("cloud_m0_m4_acceptance", _SCRIPT)
assert _SPEC and _SPEC.loader
cloud = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cloud
_SPEC.loader.exec_module(cloud)

_M0_M1_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cloud_m0_m1_acceptance.py"
_M0_M1_SPEC = importlib.util.spec_from_file_location("cloud_m0_m1_acceptance", _M0_M1_SCRIPT)
assert _M0_M1_SPEC and _M0_M1_SPEC.loader
m0_m1 = importlib.util.module_from_spec(_M0_M1_SPEC)
sys.modules[_M0_M1_SPEC.name] = m0_m1
_M0_M1_SPEC.loader.exec_module(m0_m1)


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _origin_backed_repo(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "origin.git"
    source = tmp_path / "source"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", str(source))
    _git(source, "config", "user.email", "acceptance@example.test")
    _git(source, "config", "user.name", "Acceptance Test")
    (source / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "initial")
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "origin", "HEAD:main")
    return source, remote


def test_clean_checkout_is_detached_clean_and_origin_backed(tmp_path: Path) -> None:
    source, _remote = _origin_backed_repo(tmp_path)
    work_root = tmp_path / "data-disk"

    checkout = cloud.create_clean_checkout(source, work_root)

    assert checkout.checkout.is_dir()
    assert cloud._git(checkout.checkout, "rev-parse", "HEAD") == checkout.commit
    assert cloud._git(checkout.checkout, "status", "--porcelain=v1") == ""
    assert cloud._origin_refs(checkout.checkout)[checkout.origin_ref] == checkout.commit


def test_dirty_source_is_rejected_before_clone(tmp_path: Path) -> None:
    source, _remote = _origin_backed_repo(tmp_path)
    (source / "uncommitted.txt").write_text("not clean\n", encoding="utf-8")

    with pytest.raises(cloud.CloudAcceptanceError, match="SOURCE_WORKTREE_DIRTY"):
        cloud.create_clean_checkout(source, tmp_path / "data-disk")


def test_dry_run_writes_one_immutable_sha_specific_total_report(tmp_path: Path) -> None:
    source, _remote = _origin_backed_repo(tmp_path)

    code, report_path = cloud.orchestrate(
        source_repo=source,
        work_root=tmp_path / "data-disk",
        python_arg=sys.executable,
        dry_run=True,
    )

    assert code == cloud.SUCCESS
    assert report_path is not None and report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "dry_run"
    assert report["checkout"]["commit"] == cloud._git(source, "rev-parse", "HEAD")
    assert set(report["commands"]) == set(cloud.MILESTONES)


def test_checkout_pythonpath_preserves_ros_setup_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/jazzy/lib/python3.12/site-packages")

    assert cloud._checkout_pythonpath(tmp_path) == (
        f"{tmp_path}:/opt/ros/jazzy/lib/python3.12/site-packages"
    )


def test_resolve_python_keeps_the_explicit_venv_entrypoint(tmp_path: Path) -> None:
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable))

    assert cloud._resolve_python(tmp_path, str(venv_python)) == str(venv_python)


def test_report_selection_and_existing_gate_status(tmp_path: Path) -> None:
    root = tmp_path / "run"
    paths = cloud.RunPaths(root, root / "logs", root / "reports", root / "total.json", root / "stdout.log")
    paths.reports.mkdir(parents=True)
    m2 = paths.reports / "m2-robotiq2f85-acceptance.json"
    m2.write_text(
        '{"gates":{"direct_live":{"status":"passed"},"mcp_live":{"status":"passed"}}}',
        encoding="utf-8",
    )

    assert cloud._report_for_milestone(paths, "m2") == m2
    assert cloud._result_status(m2) == "passed"
    assert cloud._final_status({"m0": {"status": "passed"}}) == ("failed", cloud.FAILED)
    assert cloud._final_status({"m0": {"status": "blocked"}}) == ("blocked", cloud.BLOCKED)


def test_cloud_commands_keep_m2_m3_drivers_and_use_dedicated_m0_m1_m4_drivers(tmp_path: Path) -> None:
    paths = cloud.RunPaths(tmp_path, tmp_path / "logs", tmp_path / "reports", tmp_path / "total.json", tmp_path / "stdout.log")
    commands = {
        milestone: cloud._milestone_command(tmp_path, "/python", milestone, paths)
        for milestone in cloud.MILESTONES
    }

    assert commands["m2"][-1].endswith("run_m2_robotiq2f85_smoke.sh")
    assert commands["m3"][-1].endswith("run_m3_pickplace_acceptance.sh")
    assert any(item.endswith("cloud_m0_m1_acceptance.py") for item in commands["m0"])
    assert any(item.endswith("cloud_m0_m1_acceptance.py") for item in commands["m1"])
    assert any(item.endswith("cloud_m4_oracle_acceptance.py") for item in commands["m4"])


def test_m1_live_rgbd_validator_rejects_stale_or_nonlive_observations() -> None:
    observation = {
        "metadata": {"observation_provenance": "gazebo_ros_live"},
        "cameras": [{
            "frame_id": "top", "rgb_base64": "rgb", "depth_base64": "depth",
            "timestamp_s": 4.0,
            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5},
            "extrinsics": {"frame_transform": "camera_to_world"},
        }],
    }

    assert m0_m1._validate_m1_observation(observation, previous=None) == {"top": 4.0}
    with pytest.raises(m0_m1.AcceptanceError, match="M1_STALE_RGBD"):
        m0_m1._validate_m1_observation(observation, previous={"top": 4.0})
    observation["metadata"]["observation_provenance"] = "gazebo_oracle"
    with pytest.raises(m0_m1.AcceptanceError, match="M1_NONLIVE_PROVENANCE"):
        m0_m1._validate_m1_observation(observation, previous=None)
