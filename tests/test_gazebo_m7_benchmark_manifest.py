"""Offline contract tests for the M7 industrial benchmark manifest (v0).

The manifest is pure data: these tests pin the schema promised in
docs/gazebo-m7-industrial-benchmark.md and the NativePickPlaceConfig-derived scene
defaults, without starting Gazebo.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.cli.batch_eval import load_parallel_episode_manifest
from extensions.gazebo.native_grasp import NativePickPlaceConfig, PICKPLACE_ENV_ID

MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "gazebo_industrial_benchmark_v0.json"
)

TASK_CLASSES = {
    "single_grasp",
    "pick_place",
    "sort_to_bin",
    "multi_sort",
    "grasp_recovery",
}
OBJECT_ROLES = {"target", "distractor", "clutter"}
OBJECT_KINDS = {"box", "cylinder"}
FAILURE_KINDS = {"grasp_pose_offset", "drop_after_lift"}


def _load_rows() -> list[dict]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return payload["episodes"]


def test_manifest_parses_with_existing_loader() -> None:
    specs = load_parallel_episode_manifest(MANIFEST_PATH)
    assert len(specs) == 10
    assert len({spec.episode_id for spec in specs}) == 10
    assert all(spec.env_id == PICKPLACE_ENV_ID for spec in specs)
    assert all(spec.task for spec in specs)
    assert all(spec.metadata.get("suite") == "gazebo_industrial" for spec in specs)
    assert all(spec.metadata.get("benchmark_version") == "v0" for spec in specs)


def test_five_task_classes_covered() -> None:
    rows = _load_rows()
    by_class: dict[str, int] = {}
    for row in rows:
        task_class = row["metadata"]["task_class"]
        assert task_class in TASK_CLASSES
        by_class[task_class] = by_class.get(task_class, 0) + 1
    assert set(by_class) == TASK_CLASSES
    assert all(count >= 2 for count in by_class.values())


def test_scene_objects_and_goal_references() -> None:
    for row in _load_rows():
        scene = row["metadata"]["scene"]
        object_ids = {item["id"] for item in scene["objects"]}
        destination_ids = {item["id"] for item in scene["destinations"]}
        for item in scene["objects"]:
            assert item["role"] in OBJECT_ROLES
            assert item["kind"] in OBJECT_KINDS
            assert len(item["initial_xyz"]) == 3
            expected_dims = 3 if item["kind"] == "box" else 2
            assert len(item["size_m"]) == expected_dims
        goal = scene["goal"]
        if "assignments" in goal:
            pairs = [(a["target_id"], a["destination_id"]) for a in goal["assignments"]]
        else:
            pairs = [(goal["target_id"], goal["destination_id"])]
        for target_id, destination_id in pairs:
            assert target_id in object_ids
            assert destination_id is None or destination_id in destination_ids
        variation = scene["variation"]
        assert set(variation) == {"occlusion", "lighting", "clutter"}


def test_failure_injection_scoping() -> None:
    for row in _load_rows():
        metadata = row["metadata"]
        injection = metadata["failure_injection"]
        if metadata["task_class"] == "grasp_recovery":
            assert injection is not None
            assert injection["kind"] in FAILURE_KINDS
            assert injection["max_injections"] >= 1
            if injection["kind"] == "grasp_pose_offset":
                assert len(injection["offset_m"]) == 3
        else:
            assert injection is None


def test_m3_scene_defaults_match_m3config() -> None:
    config = NativePickPlaceConfig()
    for row in _load_rows():
        scene = row["metadata"]["scene"]
        objects = {item["id"]: item for item in scene["objects"]}
        if config.target_id in objects:
            target = objects[config.target_id]
            assert target["kind"] == "box"
            assert list(target["size_m"]) == list(config.target_size_m)
            assert target["mass_kg"] == config.target_mass_kg
        if config.distractor_id in objects:
            distractor = objects[config.distractor_id]
            assert distractor["kind"] == "cylinder"
            assert list(distractor["size_m"]) == list(config.distractor_size_m)
            assert distractor["mass_kg"] == config.distractor_mass_kg
        for destination in scene["destinations"]:
            if destination["id"] in {"m3_zone", "bin_a"}:
                assert list(destination["center_xy"]) == list(config.destination_center_xy)
                assert list(destination["size_xy_m"]) == list(config.destination_size_xy_m)
