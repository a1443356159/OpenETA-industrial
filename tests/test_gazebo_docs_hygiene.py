"""Offline checks that Gazebo documentation matches the guarded native-grasp status."""

from __future__ import annotations

import re
from pathlib import Path

from extensions.gazebo.native_grasp import NATIVE_GRASP_SCHEMA_VERSION
from extensions.gazebo.profiles import gazebo_profiles


ROOT = Path(__file__).resolve().parent.parent
DESIGN_DOC = (ROOT / "docs" / "gazebo-adapter-design.md").read_text(encoding="utf-8")
INVENTORY_DOC = (ROOT / "docs" / "env-backend-inventory.md").read_text(encoding="utf-8")
REPRODUCTION_DOC = (ROOT / "docs" / "multi-normal-tui-reproduction.md").read_text(
    encoding="utf-8"
)
REGISTRY_SOURCE = (ROOT / "sim" / "env_registry.py").read_text(encoding="utf-8")
GAZEBO_ENV_IDS = (
    "openeta/gazebo_live_rgbd-v0",
    "openeta/gazebo_rm75_robotiq2f85-v0",
    "openeta/gazebo_rm75_robotiq2f85_pickplace-v0",
)


def _registered_gazebo_env_ids() -> tuple[str, ...]:
    match = re.search(r"def _register_gazebo_envs\(\).*?(?=\ndef )", REGISTRY_SOURCE, flags=re.DOTALL)
    assert match is not None, "_register_gazebo_envs not found in sim/env_registry.py"
    return tuple(dict.fromkeys(re.findall(r'"(openeta/gazebo[^"]+)"', match.group(0))))


def test_registered_gazebo_env_ids_match_documented_set() -> None:
    assert _registered_gazebo_env_ids() == GAZEBO_ENV_IDS


def test_design_doc_marks_native_contact_and_joint_guards() -> None:
    for profile_name in gazebo_profiles():
        assert profile_name in DESIGN_DOC
    for symbol in ("GazeboDirectEnv", "GazeboRuntime", "UnifiedEnv", "DetachableJoint"):
        assert symbol in DESIGN_DOC
    assert "Motion-control gripper safeguards" in DESIGN_DOC


def test_inventory_doc_lists_guarded_gazebo_environment_honestly() -> None:
    assert "`gazebo`" in INVENTORY_DOC
    assert "_register_gazebo_envs" in INVENTORY_DOC
    assert NATIVE_GRASP_SCHEMA_VERSION in INVENTORY_DOC
    for env_id in GAZEBO_ENV_IDS:
        assert env_id in INVENTORY_DOC


def test_multi_normal_operator_guide_keeps_reproducible_human_boundary() -> None:
    for required in (
        "--operator-mode human_tui",
        "请先看清工作台",
        "mode=frozen_frontier model_inference=False",
        "$RUN_ROOT/acceptance-report.json",
        "$RUN_ROOT/pick-place/human_tui/cleanup.json",
        "host_dispatch_count",
    ):
        assert required in REPRODUCTION_DOC
