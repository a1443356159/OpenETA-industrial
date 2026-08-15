"""Offline checks that Gazebo documentation matches the disabled M3 status."""

from __future__ import annotations

import re
from pathlib import Path

from extensions.gazebo.m3 import M3_UNAVAILABLE_REASON
from extensions.gazebo.profiles import gazebo_profiles


ROOT = Path(__file__).resolve().parent.parent
DESIGN_DOC = (ROOT / "docs" / "gazebo-adapter-design.md").read_text(encoding="utf-8")
INVENTORY_DOC = (ROOT / "docs" / "env-backend-inventory.md").read_text(encoding="utf-8")
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


def test_design_doc_marks_m3_profile_unavailable_and_preserves_m1_m2_runtime() -> None:
    for profile_name in gazebo_profiles():
        assert profile_name in DESIGN_DOC
    for symbol in ("GazeboDirectEnv", "GazeboRuntime", "UnifiedEnv", M3_UNAVAILABLE_REASON):
        assert symbol in DESIGN_DOC
    assert "M2 gripper safeguards" in DESIGN_DOC


def test_inventory_doc_lists_disabled_gazebo_environment_honestly() -> None:
    assert "`gazebo`" in INVENTORY_DOC
    assert "_register_gazebo_envs" in INVENTORY_DOC
    assert M3_UNAVAILABLE_REASON in INVENTORY_DOC
    for env_id in GAZEBO_ENV_IDS:
        assert env_id in INVENTORY_DOC
