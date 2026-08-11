"""Documentation hygiene checks for the Gazebo adapter docs.

These tests pin the rewritten `docs/gazebo-adapter-design.md` and the
`docs/env-backend-inventory.md` gazebo entry against the code they describe,
so the docs cannot silently drift back to the removed live-boundary classes.
Purely offline: no Gazebo or ROS 2 environment is required.
"""

from __future__ import annotations

import re
from pathlib import Path

from extensions.gazebo.profiles import gazebo_profiles

ROOT = Path(__file__).resolve().parent.parent
DESIGN_DOC = (ROOT / "docs" / "gazebo-adapter-design.md").read_text(encoding="utf-8")
INVENTORY_DOC = (ROOT / "docs" / "env-backend-inventory.md").read_text(encoding="utf-8")
REGISTRY_SOURCE = (ROOT / "sim" / "env_registry.py").read_text(encoding="utf-8")

REMOVED_CLASSES = ("GazeboLiveSession", "GazeboLiveMcpTransport", "GazeboWorkerEnv")
GAZEBO_ENV_IDS = (
    "openeta/gazebo_live_rgbd-v0",
    "openeta/gazebo_rm75_robotiq2f85-v0",
    "openeta/gazebo_rm75_robotiq2f85_pickplace-v0",
)


def _registered_gazebo_env_ids() -> tuple[str, ...]:
    match = re.search(
        r"def _register_gazebo_envs\(\).*?(?=\ndef )",
        REGISTRY_SOURCE,
        flags=re.DOTALL,
    )
    assert match is not None, "_register_gazebo_envs not found in sim/env_registry.py"
    # Each id appears twice (the _register_one key and the EnvSpec id field).
    return tuple(dict.fromkeys(re.findall(r'"(openeta/gazebo[^"]+)"', match.group(0))))


def test_registered_gazebo_env_ids_match_documented_set() -> None:
    assert _registered_gazebo_env_ids() == GAZEBO_ENV_IDS


def test_design_doc_documents_current_profile_runtime_architecture() -> None:
    for profile_name in gazebo_profiles():
        assert profile_name in DESIGN_DOC
    for capability in (
        "fresh_observation",
        "authoritative_camera",
        "control",
        "structured_receipt",
        "physics",
    ):
        assert capability in DESIGN_DOC
    for symbol in ("GazeboDirectEnv", "GazeboRuntime", "profiles.py", "UnifiedEnv"):
        assert symbol in DESIGN_DOC
    assert "bench worker" in DESIGN_DOC


def test_design_doc_mentions_removed_classes_only_as_removed_history() -> None:
    change_note_index = DESIGN_DOC.index("## Change note")
    current_body = DESIGN_DOC[:change_note_index]
    for removed in REMOVED_CLASSES:
        assert removed not in current_body, (
            f"{removed} was removed from the codebase and must not be "
            "described as the current live boundary"
        )
        assert removed in DESIGN_DOC[change_note_index:], (
            f"{removed} should be acknowledged as removed in the change note"
        )


def test_design_doc_records_plan_s6_naming_deviation() -> None:
    assert "## Module naming vs plan.md §6" in DESIGN_DOC
    for module in ("m2.py", "m3.py", "ros_control.py", "ros_physics.py", "runtime.py", "direct_env.py"):
        assert module in DESIGN_DOC


def test_inventory_doc_lists_gazebo_backend() -> None:
    assert "`gazebo`" in INVENTORY_DOC
    assert "_register_gazebo_envs" in INVENTORY_DOC
    for env_id in GAZEBO_ENV_IDS:
        assert env_id in INVENTORY_DOC
