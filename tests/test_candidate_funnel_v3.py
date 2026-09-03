from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import threading
import time
from pathlib import Path

import pytest

from agent.runtime import qualification_v3 as qualification_v3_module
from agent.runtime.moveit_qualification import (
    QUALIFICATION_SCHEMA_V3,
    MoveItCandidateQualifier,
    MoveItQualificationEngine,
    SAME_RUN_QUALIFICATION_SEED_FIELD,
    SAME_RUN_QUALIFICATION_SEED_PROVENANCE,
    SAME_RUN_QUALIFICATION_SEED_SCHEMA,
)
from agent.runtime.qualification_legality import (
    bind_qualified_placement_goal,
    evaluate_grasp_target_closing_alignment,
    evaluate_grasp_placement_pair_legality,
    evaluate_placement_goal_legality,
)
from agent.runtime.qualification_v3 import (
    CandidateWave,
    candidate_physical_quality_key,
    frozen_frontier_parent_priority,
    parallel_gripper_centering_quality,
    parallel_gripper_centering_variant_priority,
    reprioritize_grasp_frontier,
    schedule_candidate_waves,
    select_grasp_branches,
)
from agent.tools.registry import ToolResult
from tools.candidate_config import DEFAULT_GRASP_WAVES


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate(index: int, *, stages: int = 1, score: float = 0.0):
    return {
        "id": f"c{index}",
        "score": score,
        "qualification_stages": [
            {
                "name": f"c{index}_stage{stage}",
                "xyz": [0.4 + index * 0.001, 0.0, 0.5],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
            for stage in range(stages)
        ],
    }


def test_frozen_frontier_parent_priority_reads_compiled_camera_pose() -> None:
    assert frozen_frontier_parent_priority({}) == 1
    assert (
        frozen_frontier_parent_priority(
            {"compile_parameters": {"camera_pose": {"frozen_frontier_parent_priority": True}}}
        )
        == 0
    )


def test_centering_reserve_variant_enters_the_first_deep_wave_without_deletion() -> None:
    descriptors = []
    for index in range(8):
        candidate = _candidate(index, score=100.0 - index)
        candidate["qualification_stages"][0]["xyz"] = [
            0.4 + index * 0.012,
            0.0,
            0.5,
        ]
        if index == 7:
            candidate["target_closing_alignment"] = {
                "correction_m": 0.001,
                "target_span_m": 0.04,
                "variant_role": "same_approach_centering_reserve",
                "compatible_parent_backend_indices": [10],
            }
        descriptors.append(
            {
                "candidate_id": candidate["id"],
                "candidate": candidate,
                "candidate_pose_sha256": _hash(candidate),
            }
        )

    waves = schedule_candidate_waves(
        descriptors,
        purpose="grasp",
        grasp_waves=[4, 8],
    )

    assert parallel_gripper_centering_variant_priority(descriptors[7]["candidate"]) == 0
    assert "c7" in [row["candidate_id"] for row in waves[0].candidates]
    assert sorted(row["candidate_id"] for wave in waves for row in wave.candidates) == [
        f"c{index}" for index in range(8)
    ]


def test_measured_centering_quality_precedes_weaker_reserve_labels() -> None:
    descriptors = []
    for index in range(12):
        candidate = _candidate(index, score=100.0 - index)
        candidate["qualification_stages"][0]["xyz"] = [
            0.35 + index * 0.015,
            0.0,
            0.5,
        ]
        candidate["target_closing_alignment"] = {
            "correction_m": 0.0032,
            "target_span_m": 0.04,
            "variant_role": "same_approach_centering_reserve",
        }
        if index == 11:
            candidate["target_closing_alignment"] = {
                "correction_m": 0.0002,
                "target_span_m": 0.04,
            }
        descriptors.append(
            {
                "candidate_id": candidate["id"],
                "candidate": candidate,
                "candidate_pose_sha256": _hash(candidate),
            }
        )

    waves = schedule_candidate_waves(
        descriptors,
        purpose="grasp",
        grasp_waves=[4, 12],
    )

    assert "c11" in [row["candidate_id"] for row in waves[0].candidates]


def test_deep_quality_prefers_joint_margin_before_safe_span_tie_breakers() -> None:
    narrow = {
        "endpoint_pass": True,
        "target_closing_alignment": {
            "correction_m": 0.001,
            "target_span_m": 0.02,
        },
        "stages": [{"joint_margin": 0.8, "min_singular_value": 0.8}],
    }
    broad = {
        "endpoint_pass": True,
        "target_closing_alignment": {
            "correction_m": 0.002,
            "target_span_m": 0.04,
        },
        "stages": [{"joint_margin": 0.2, "min_singular_value": 0.2}],
    }

    assert candidate_physical_quality_key(narrow) < candidate_physical_quality_key(broad)


def test_recovery_deep_quality_prefers_joint_margin_before_safe_span_difference() -> None:
    narrow = {
        "endpoint_pass": True,
        "frozen_frontier_parent_priority": True,
        "target_closing_alignment": {
            "correction_m": 0.0002,
            "target_span_m": 0.02,
        },
        "stages": [{"joint_margin": 0.8, "min_singular_value": 0.8}],
    }
    broad = {
        "endpoint_pass": True,
        "frozen_frontier_parent_priority": True,
        "target_closing_alignment": {
            "correction_m": 0.002,
            "target_span_m": 0.04,
        },
        "stages": [{"joint_margin": 0.2, "min_singular_value": 0.2}],
    }

    # Both ratios are below the 0.10 risk boundary and both candidates have
    # already passed the bilateral terminal-state check.  Joint robustness is
    # therefore stronger deep-funnel evidence than another span heuristic.
    assert candidate_physical_quality_key(narrow) < candidate_physical_quality_key(broad)


def test_scheduler_prefers_wider_span_within_safe_centering_tier() -> None:
    narrow = _candidate(0, score=100.0)
    narrow["frozen_frontier_parent_priority"] = True
    narrow["target_closing_alignment"] = {
        "correction_m": 0.0002,
        "target_span_m": 0.02,
    }
    broad = _candidate(1, score=1.0)
    broad["frozen_frontier_parent_priority"] = True
    broad["target_closing_alignment"] = {
        "correction_m": 0.002,
        "target_span_m": 0.04,
    }
    descriptors = [
        {
            "candidate_id": candidate["id"],
            "candidate": candidate,
            "candidate_pose_sha256": _hash(candidate),
        }
        for candidate in (narrow, broad)
    ]

    waves = schedule_candidate_waves(
        descriptors,
        purpose="grasp",
        grasp_waves=[1, 2],
    )

    assert [row["candidate_id"] for row in waves[0].candidates] == ["c1"]


def test_initial_scheduler_keeps_centering_before_span() -> None:
    narrow = _candidate(0, score=1.0)
    narrow["target_closing_alignment"] = {
        "correction_m": 0.0002,
        "target_span_m": 0.02,
    }
    broad = _candidate(1, score=100.0)
    broad["target_closing_alignment"] = {
        "correction_m": 0.002,
        "target_span_m": 0.04,
    }
    descriptors = [
        {
            "candidate_id": candidate["id"],
            "candidate": candidate,
            "candidate_pose_sha256": _hash(candidate),
        }
        for candidate in (narrow, broad)
    ]

    waves = schedule_candidate_waves(
        descriptors,
        purpose="grasp",
        grasp_waves=[1, 2],
    )

    assert [row["candidate_id"] for row in waves[0].candidates] == ["c0"]


def test_pose_diversity_scheduler_caches_pairwise_distances(monkeypatch) -> None:
    descriptors = []
    for index in range(32):
        candidate = _candidate(index, score=32.0 - index)
        candidate["qualification_stages"][0]["xyz"] = [
            0.2 + index * 0.007,
            (index % 4) * 0.013,
            0.5,
        ]
        descriptors.append(
            {
                "candidate_id": candidate["id"],
                "candidate": candidate,
                "candidate_pose_sha256": _hash(candidate),
            }
        )
    calls = 0
    original = qualification_v3_module._descriptor_diversity_distance

    def counted(left, right):
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(
        qualification_v3_module,
        "_descriptor_diversity_distance",
        counted,
    )

    waves = schedule_candidate_waves(
        descriptors,
        purpose="grasp",
        grasp_waves=[4, 8, 16, 32],
    )

    assert sum(len(wave.candidates) for wave in waves) == 32
    assert calls <= 32 * 31 // 2


def test_l5_miss_reorders_untouched_frontier_without_losing_candidates() -> None:
    def descriptor(
        index: int,
        *,
        x: float,
        rotation: list[list[float]],
    ) -> dict[str, object]:
        candidate = _candidate(index, score=10.0 - index)
        candidate["qualification_stages"][0]["xyz"] = [x, 0.0, 0.5]
        candidate["compile_parameters"] = {
            "camera_pose": {
                "rotation_matrix": rotation,
                "target_closing_alignment": {
                    "closing_axis": "graspnet_local_y",
                    "binormal_axis": "graspnet_local_z",
                },
            }
        }
        return {
            "candidate_id": candidate["id"],
            "candidate": candidate,
            "fixed_candidate_index": index,
            "se3_cluster_id": f"se3_{index:04d}",
            "capability_score": {},
        }

    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    # A 180-degree roll around GraspNet's explicit approach axis (local X)
    # is a parallel-jaw equivalent orientation, not a different approach.
    symmetric_roll = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    anchor = descriptor(0, x=0.40, rotation=identity)
    frozen_head = descriptor(1, x=0.80, rotation=identity)
    sibling = descriptor(2, x=0.41, rotation=symmetric_roll)
    nearby = descriptor(3, x=0.42, rotation=identity)
    tail = descriptor(4, x=0.90, rotation=identity)
    waves = [
        CandidateWave(0, 1, (anchor,)),
        CandidateWave(1, 3, (frozen_head, sibling)),
        CandidateWave(2, 5, (nearby, tail)),
    ]

    reordered, evidence = reprioritize_grasp_frontier(
        waves,
        completed_wave_position=0,
        anchors=[anchor],
    )

    assert evidence["applied"] is True
    assert evidence["anchor_candidate_ids"] == ["c0"]
    # Exploit the model sibling, then keep the original exploration head.
    assert [item["candidate_id"] for item in reordered[1].candidates] == ["c2", "c1"]
    assert [len(wave.candidates) for wave in reordered] == [1, 2, 2]
    assert {
        item["candidate_id"] for wave in reordered for item in wave.candidates
    } == {"c0", "c1", "c2", "c3", "c4"}


def test_default_grasp_ladder_reaches_256_before_implicit_pool_exhaustion() -> None:
    descriptors = []
    for index in range(512):
        candidate = _candidate(index, score=512.0 - index)
        candidate["qualification_stages"][0]["xyz"] = [0.4, 0.0, 0.5]
        descriptors.append(
            {
                "candidate_id": candidate["id"],
                "candidate": candidate,
                "candidate_pose_sha256": _hash(candidate),
            }
        )

    waves = schedule_candidate_waves(descriptors, purpose="grasp")

    assert DEFAULT_GRASP_WAVES == (4, 8, 16, 32, 64, 128, 256)
    assert [wave.cumulative_per_branch for wave in waves] == [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]
    assert [len(wave.candidates) for wave in waves] == [
        4,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    ]


def _request(candidates, *, purpose="placement", overrides=None):
    funnel = {
        "qualification_profile": "fast_v3",
        "solver_profile": "kdl_fast",
        "beam_width": 2,
        "ik_seed_count": 8,
        "fast_seed_count": 2,
        "recovery_seed_count": 6,
        "fast_ik_timeout_s": 0.05,
        "recovery_ik_timeout_s": 0.2,
        "max_ik_concurrency": 8,
        "max_state_validity_concurrency": 8,
        "grasp_waves": [4, 8, 16, 32, 64],
        "placement_waves": [4, 8, 16, 32, 96],
        "full_plan_limit": 2,
        "l5_pass_target": 2 if purpose == "grasp" else 1,
    }
    funnel.update(overrides or {})
    return {
        "schema_version": QUALIFICATION_SCHEMA_V3,
        "purpose": purpose,
        "planning_scene_revision": 4,
        "qualification_binding_sha256": "binding",
        "funnel": funnel,
        "source": {
            "joint_limits": {"lower": [-1.0], "upper": [1.0]},
            "home_joint_state": {"names": ["j1"], "positions": [0.8]},
        },
        "candidates": [
            {
                "candidate_id": candidate["id"],
                "candidate_pose_sha256": _hash(candidate),
                "candidate": candidate,
            }
            for candidate in candidates
        ],
    }


def _engine(**overrides):
    callbacks = {
        "current_joint_state": lambda: {
            "names": ["j1"],
            "positions": [0.0],
            "joint_limits": {"lower": [-1.0], "upper": [1.0]},
            "home_joint_state": {"names": ["j1"], "positions": [0.8]},
            "authoritative_scene_sha256": "a" * 64,
            "moveit_world_geometry_sha256": "b" * 64,
            "moveit_attached_geometry_sha256": "c" * 64,
            "moveit_geometry_verified_ids": ["table", "target"],
        },
        "scene_revision": lambda: 4,
        "compute_ik": lambda target, seed, collision: {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": [0.2]},
            "min_singular_value": 0.3,
        },
        "check_state_validity": lambda state: {
            "valid": True,
            "collision_pairs": [],
        },
        "plan_only": lambda target, start, timeout, attempts: {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": [0.2]}],
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        },
    }
    callbacks.update(overrides)
    return MoveItQualificationEngine(**callbacks)


def test_fast_wire_response_omits_rehydratable_compile_parameters():
    candidate = _candidate(0)
    candidate["compile_parameters"] = {
        "camera_pose": {"provider_candidate_index": 17}
    }

    response = _engine().qualify(
        _request(
            [candidate],
            purpose="grasp",
            overrides={"l5_pass_target": 1, "l5_min_pass_target": 1},
        )
    )

    assert response["results"][0]["verdict"] == "PASS"
    assert "compile_parameters" not in response["results"][0]
    assert response["results"][0]["screening_attempt_count"] == 1
    assert response["results"][0]["screening_attempts"] == []


def test_single_pass_grasp_target_keeps_alternate_beam_branch_for_recovery():
    planned_states = []

    def compute_ik(target, seed, collision):
        del target, collision
        position = -0.2 if seed.get("seed_source") == "named_home" else 0.2
        return {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": [position]},
            "min_singular_value": 0.3,
        }

    def plan_only(target, start, timeout, attempts):
        del start, timeout, attempts
        state = target["qualification_goal_joint_state"]
        planned_states.append(state["positions"][0])
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": list(state["positions"])}],
            "end_joint_state": dict(state),
        }

    response = _engine(compute_ik=compute_ik, plan_only=plan_only).qualify(
        _request(
            [_candidate(0)],
            purpose="grasp",
            overrides={"l5_pass_target": 1, "l5_min_pass_target": 1},
        )
    )

    assert response["selected_candidate_ids"] == ["c0"]
    assert response["stop_reason"] == "complete_l5_pass_found"
    assert response["metrics"]["l5_pass_count"] == 1
    assert response["metrics"]["l5_joint_branch_pass_count"] == 1
    assert planned_states == [0.2]
    assert [attempt["joint_branch_index"] for attempt in response["l5_attempts"]] == [0]


def test_fast_ik_request_carries_the_wave_queue_depth_separately_from_solver_budget():
    seen: list[tuple[float, int]] = []
    validity_depths: list[int] = []

    def compute_ik(target, seed, collision):
        seen.append(
            (
                float(target["ik_seed_timeout_s"]),
                int(target["qualification_ik_queue_depth"]),
            )
        )
        return {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": [0.2]},
            "min_singular_value": 0.3,
        }

    def check_state_validity(state):
        validity_depths.append(
            int(state["qualification_state_validity_queue_depth"])
        )
        return {"valid": True, "collision_pairs": []}

    response = _engine(
        compute_ik=compute_ik,
        check_state_validity=check_state_validity,
    ).qualify(
        _request(
            [_candidate(0)],
            overrides={"max_ik_concurrency": 4, "l5_pass_target": 1},
        )
    )

    assert response["selected_candidate_ids"] == ["c0"]
    assert seen
    assert set(seen) == {(0.05, 4)}
    assert validity_depths
    assert set(validity_depths) == {8}


def test_frozen_pair_retains_two_diverse_l5_passes_without_early_cutoff():
    calls = 0

    def plan_only(target, start, timeout, attempts):
        nonlocal calls
        calls += 1
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": [0.2]}],
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    request = _request(
        [_candidate(0), _candidate(1), _candidate(2)],
        overrides={
            "qualification_mode": "frozen_pair",
            "l5_pass_target": 2,
        },
    )

    response = _engine(plan_only=plan_only).qualify(request)

    assert calls == 2
    assert response["metrics"]["l5_attempt_count"] == 2
    assert response["metrics"]["l5_pass_count"] == 2
    assert response["selected_candidate_ids"] == ["c0", "c1"]
    assert response["results"][2]["verdict"] == "NOT_EVALUATED"


def test_l5_rank_preserves_safe_centering_evidence_but_prefers_joint_margin():
    candidates = [_candidate(0), _candidate(1)]
    candidates[0]["target_closing_alignment"] = {
        "correction_m": 0.0036,
        "target_span_m": 0.04,
    }
    candidates[1]["target_closing_alignment"] = {
        "correction_m": 0.0004,
        "target_span_m": 0.04,
    }
    planned: list[str] = []

    def compute_ik(target, seed, collision):
        del seed, collision
        # The better-centered c1 deliberately has the worse joint margin.
        position = 0.8 if str(target["name"]).startswith("c1_") else 0.0
        return {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": [position]},
            "min_singular_value": 0.3,
        }

    def plan_only(target, start, timeout, attempts):
        del start, timeout, attempts
        planned.append(str(target["name"]))
        position = 0.8 if str(target["name"]).startswith("c1_") else 0.0
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": [position]}],
            "end_joint_state": {"names": ["j1"], "positions": [position]},
        }

    response = _engine(compute_ik=compute_ik, plan_only=plan_only).qualify(
        _request(
            candidates,
            purpose="grasp",
            overrides={"grasp_waves": [2], "l5_pass_target": 1},
        )
    )

    assert response["selected_candidate_ids"] == ["c0"]
    assert planned == ["c0_stage0"]
    result = next(row for row in response["results"] if row["candidate_id"] == "c0")
    assert result["target_closing_alignment"] == {
        "correction_m": 0.0036,
        "target_span_m": 0.04,
    }


def test_frozen_pair_l5_order_prefers_distinct_grasp_and_goal_cluster():
    candidates = []
    for index, (grasp_id, score) in enumerate((("g0", 4.0), ("g0", 3.0), ("g1", 2.0), ("g1", 1.0))):
        candidate = _candidate(index, score=score)
        candidate.update(
            {
                "source_grasp_id": grasp_id,
                "source_grasp_equivalence_id": grasp_id,
                "source_object_goal_id": "p0",
                "object_goal_pose": {
                    "frame": "world",
                    "translation_xyz": [0.48, -0.1, 0.43],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            }
        )
        candidates.append(candidate)
    # c0/c1/c2 share one 10 mm cluster. c3 is the lower-scored, but only,
    # candidate that provides both a second grasp and a second goal cluster.
    candidates[3]["qualification_stages"][0]["xyz"][0] = 0.43

    request = _request(
        candidates,
        overrides={
            "qualification_mode": "frozen_pair",
            "l5_pass_target": 2,
        },
    )
    response = _engine(clone_scene=_placement_scene).qualify(request)

    assert response["selected_candidate_ids"] == ["c0", "c3"]
    assert [attempt["source_grasp_id"] for attempt in response["l5_attempts"]] == [
        "g0",
        "g1",
    ]
    assert [attempt["se3_cluster_id"] for attempt in response["l5_attempts"]] == [
        "se3_0000",
        "se3_0001",
    ]


def test_grasp_minimum_two_stops_after_small_wave_without_recovery_chase():
    candidates = [_candidate(index) for index in range(8)]
    for index, candidate in enumerate(candidates):
        candidate["qualification_stages"][0]["xyz"][0] = 0.4 + index * 0.02

    def plan_only(target, start, timeout, attempts):
        del start, timeout, attempts
        candidate_index = int(str(target["name"]).split("_stage", 1)[0][1:])
        passed = candidate_index < 2
        return {
            "ok": passed,
            "execution_started": False,
            "trajectory_points": ([{"positions": [0.2]}] if passed else []),
            "end_joint_state": ({"names": ["j1"], "positions": [0.2]} if passed else None),
        }

    response = _engine(plan_only=plan_only).qualify(
        _request(
            candidates,
            purpose="grasp",
            overrides={
                "l5_pass_target": 4,
                "l5_min_pass_target": 2,
                "grasp_waves": [4, 8],
            },
        )
    )

    assert response["stop_reason"] == ("complete_l5_pass_found_minimum_lookahead")
    assert response["selected_candidate_ids"] == ["c0", "c1"]
    assert len(response["waves"]) == 1
    assert response["waves"][0]["candidate_count"] == 4
    assert response["waves"][0]["recovery_layer"] is False
    assert response["metrics"]["screening_attempt_count"] == 4
    assert sum(row["verdict"] == "NOT_EVALUATED" for row in response["results"]) == 4


def test_same_run_frozen_pair_seed_is_used_as_second_fast_seed():
    candidate = _candidate(0)
    candidate[SAME_RUN_QUALIFICATION_SEED_FIELD] = {
        "schema_version": SAME_RUN_QUALIFICATION_SEED_SCHEMA,
        "provenance": SAME_RUN_QUALIFICATION_SEED_PROVENANCE,
        "source_candidate_id": "frozen_pair_g0_p0",
        "states": [{"names": ["j1"], "positions": [0.35]}],
    }
    sources = []

    def ik(target, seed, collision):
        del target, collision
        sources.append(str(seed.get("seed_source") or ""))
        if seed.get("seed_source") != "frozen_pair_qualified_same_run":
            return {"ok": False}
        return {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": [0.2]},
            "min_singular_value": 0.3,
        }

    response = _engine(compute_ik=ik).qualify(_request([candidate]))

    assert response["selected_candidate_ids"] == ["c0"]
    assert sources == ["current_robot_state", "frozen_pair_qualified_same_run"]


def _placement_scene():
    return {
        "revision": 4,
        "target_id": "target",
        "world_specs": {
            "target": {
                "id": "target",
                "shape": "box",
                "size_xyz": [0.04, 0.04, 0.06],
                "pose_xyz": [0.28, -0.1, 0.43],
                "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "table": {
                "id": "table",
                "shape": "box",
                "size_xyz": [0.7, 0.6, 0.04],
                "pose_xyz": [0.4, 0.0, 0.38],
                "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "attached_specs": {},
        "placement_region": {
            "center_xy": [0.48, -0.1],
            "size_xy_m": [0.12, 0.12],
            "support_z_m": 0.4,
            "support_object_id": "table",
            "support_height_tolerance_m": 0.01,
            "support_penetration_tolerance_m": 0.005,
            "static_penetration_tolerance_m": 0.001,
        },
    }


def _physical_bin_placement_scene():
    scene = _placement_scene()
    scene["world_specs"]["parts_bin"] = {
        "id": "parts_bin",
        "shape": "box",
        "size_xyz": [0.32, 0.36, 0.18],
        "pose_xyz": [0.48, -0.1, 0.0],
        "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "primitives": [
            {
                "shape": "box",
                "size_xyz": [0.32, 0.36, 0.02],
                "pose_xyz": [0.0, 0.0, 0.01],
                "pose_rpy": [0.0, 0.0, 0.0],
            },
            {
                "shape": "box",
                "size_xyz": [0.027, 0.36, 0.18],
                "pose_xyz": [-0.1465, 0.0, 0.09],
                "pose_rpy": [0.0, 0.0, 0.0],
            },
            {
                "shape": "box",
                "size_xyz": [0.027, 0.36, 0.18],
                "pose_xyz": [0.1465, 0.0, 0.09],
                "pose_rpy": [0.0, 0.0, 0.0],
            },
            {
                "shape": "box",
                "size_xyz": [0.32, 0.047, 0.09],
                "pose_xyz": [0.0, -0.1565, 0.045],
                "pose_rpy": [0.0, 0.0, 0.0],
            },
            {
                "shape": "box",
                "size_xyz": [0.32, 0.016, 0.18],
                "pose_xyz": [0.0, 0.172, 0.09],
                "pose_rpy": [0.0, 0.0, 0.0],
            },
        ],
    }
    scene["placement_region"].update(
        {
            "support_object_id": "parts_bin",
            "support_z_m": 0.02,
            "release_z_offset_m": 0.05,
        }
    )
    return scene


def _scene_aligned_grasp_candidate(
    index: int,
    *,
    tip_y_m: float,
    score: float,
) -> dict:
    candidate = _candidate(index, score=score)
    candidate["target_closing_alignment"] = {
        "source": "aligned_selected_mask_depth",
        "correction_m": 0.0,
        "target_span_m": 0.04,
    }
    candidate["compile_parameters"] = {
        "camera_pose": {
            "id": candidate["id"],
            "frame": "camera",
            "camera_frame": "opencv",
            "translation_xyz": [0.28, tip_y_m, 0.43],
            "gripper_tip_position_xyz": [0.28, tip_y_m, 0.43],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "width": 0.04,
            "height": 0.03,
        },
        "camera_extrinsics": {
            "camera_frame": "opencv",
            "camera_to_world": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "max_gripper_width_m": 0.085,
    }
    return candidate


def test_scene_target_closing_alignment_is_exact_ordering_evidence_not_pose_edit():
    candidate = _scene_aligned_grasp_candidate(0, tip_y_m=-0.02, score=100.0)
    original_tip = list(candidate["compile_parameters"]["camera_pose"]["gripper_tip_position_xyz"])

    evidence = evaluate_grasp_target_closing_alignment(
        {"candidate_id": "c0", "candidate": candidate},
        scene=_placement_scene(),
    )

    assert evidence["evaluated"] is True
    assert evidence["section_intersects_target"] is True
    assert evidence["target_span_m"] == pytest.approx(0.04)
    assert evidence["correction_m"] == pytest.approx(-0.08)
    assert evidence["required_aperture_m"] == pytest.approx(0.20)
    assert evidence["aperture_feasible"] is False
    assert evidence["ordering_only"] is True
    assert evidence["pose_modified"] is False
    assert (
        candidate["compile_parameters"]["camera_pose"]["gripper_tip_position_xyz"] == original_tip
    )


def test_grasp_scene_geometry_demotes_one_sided_pose_without_pruning_it():
    risky = _scene_aligned_grasp_candidate(0, tip_y_m=-0.02, score=100.0)
    centered = _scene_aligned_grasp_candidate(1, tip_y_m=-0.1, score=1.0)
    planned: list[str] = []

    def plan_only(target, start, timeout, attempts):
        del start, timeout, attempts
        planned.append(str(target["name"]))
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": [0.2]}],
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    response = _engine(
        clone_scene=_placement_scene,
        plan_only=plan_only,
    ).qualify(
        _request(
            [risky, centered],
            purpose="grasp",
            overrides={"grasp_waves": [2], "l5_pass_target": 1},
        )
    )

    assert response["selected_candidate_ids"] == ["c1"]
    assert planned == ["c1_stage0"]
    assert response["metrics"]["grasp_scene_alignment_evaluated_count"] == 2
    assert response["metrics"]["grasp_scene_alignment_aperture_risk_count"] == 1
    results = {row["candidate_id"]: row for row in response["results"]}
    assert results["c0"]["workspace_pass"] is True
    assert results["c0"]["scene_target_closing_alignment"]["ordering_only"] is True
    assert results["c1"]["scene_target_closing_alignment"]["aperture_feasible"] is True
    assert parallel_gripper_centering_quality(results["c0"])[0] == 2
    assert parallel_gripper_centering_quality(results["c1"])[0] == 0


def test_goal_legality_projects_offset_compound_body_instead_of_centered_outer_box():
    scene = _placement_scene()
    scene["world_specs"]["target"] = {
        "id": "target",
        "shape": "compound",
        "size_xyz": [0.22, 0.062, 0.03],
        "pose_xyz": [0.29, -0.105, 0.015],
        "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "primitives": [
            {
                "shape": "box",
                "size_xyz": [0.165, 0.025, 0.026],
                "pose_xyz": [-0.025, 0.0, -0.002],
                "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "shape": "box",
                "size_xyz": [0.055, 0.062, 0.03],
                "pose_xyz": [0.08, 0.0, 0.0],
                "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        ],
    }
    scene["placement_region"].update(
        {
            "center_xy": [0.7, 0.18],
            "size_xy_m": [0.285, 0.275],
            "support_z_m": 0.02,
        }
    )
    candidate = {
        "id": "placement_compound",
        "object_goal_pose": {
            "translation_xyz": [
                0.66628256313311,
                0.16635623309923503,
                0.05537536084996851,
            ],
            "rotation_matrix": [
                [0.9370705755214195, 0.29557532788619456, 0.185833290219],
                [-0.3181968095037969, 0.942066089145608, 0.106123954058],
                [-0.14369966754420774, -0.158577187005505, 0.976833641529],
            ],
        },
    }

    compound = evaluate_placement_goal_legality(
        {"candidate_id": "placement_compound", "candidate": candidate},
        scene=scene,
    )
    centered_scene = deepcopy(scene)
    centered_scene["world_specs"]["target"].pop("primitives")
    centered = evaluate_placement_goal_legality(
        {"candidate_id": "placement_centered", "candidate": candidate},
        scene=centered_scene,
    )

    assert compound["verdict"] == "PASS"
    assert compound["checks"]["object_bbox"]["geometry_source"] == ("compound_collision_primitives")
    assert compound["checks"]["placement_region"]["minimum_margin_m"] == (
        pytest.approx(0.0015652853126949529)
    )
    support = compound["checks"]["support"]
    region = compound["checks"]["placement_region"]
    expected_sweep = (
        2.0
        * region["conservative_footprint_radius_m"]
        * math.sin(support["support_face_alignment_error_rad"] / 2.0)
    )
    assert support["settling_sweep_translation_bound_m"] == pytest.approx(expected_sweep)
    assert support["settling_sweep_clearance_m"] == pytest.approx(
        region["conservative_minimum_margin_m"] - expected_sweep
    )
    assert support["settling_sweep_role"] == "ordering_only"
    assert centered["reason"] == "goal_footprint_outside_placement_region"


def test_container_goal_legality_keeps_edge_goal_when_geometry_centroid_is_inside():
    scene = _placement_scene()
    scene["placement_region"]["acceptance_semantics"] = "stable_geometry_centroid_inside"
    candidate = {
        "id": "container_edge",
        "object_goal_pose": {
            "translation_xyz": [0.53, -0.1, 0.43],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
    }

    result = evaluate_placement_goal_legality(
        {"candidate_id": "container_edge", "candidate": candidate},
        scene=scene,
    )

    region = result["checks"]["placement_region"]
    assert result["verdict"] == "PASS"
    assert region["complete_footprint_inside"] is False
    assert region["geometry_centroid_inside"] is True
    assert region["complete_footprint_margin_role"] == "ordering_only"


def test_placement_clearance_orders_l5_without_deleting_edge_goals() -> None:
    candidates = [_candidate(0, score=10.0), _candidate(1, score=1.0)]
    for index, candidate in enumerate(candidates):
        candidate.update(
            {
                "source_grasp_id": "g0",
                "source_grasp_equivalence_id": "g0",
                "source_object_goal_id": f"p{index}",
                "object_goal_pose": {
                    "frame": "world",
                    "translation_xyz": [0.515 if index == 0 else 0.48, -0.1, 0.43],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            }
        )
    planned: list[str] = []

    def plan_only(target, start, timeout, attempts):
        del start, timeout, attempts
        planned.append(str(target["name"]))
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": [0.2]}],
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    response = _engine(
        clone_scene=_placement_scene,
        plan_only=plan_only,
    ).qualify(
        _request(
            candidates,
            overrides={"placement_waves": [2], "l5_pass_target": 1},
        )
    )

    assert response["selected_candidate_ids"] == ["c1"]
    assert planned == ["c1_stage0"]
    results = {row["candidate_id"]: row for row in response["results"]}
    assert (
        results["c1"]["placement_robust_clearance_m"]
        > results["c0"]["placement_robust_clearance_m"]
    )
    assert results["c0"]["goal_legality"]["verdict"] == "PASS"
    assert sorted(results) == ["c0", "c1"]


def test_supported_object_height_orders_l5_without_deleting_model_goals() -> None:
    candidates = [_candidate(0, score=100.0), _candidate(1, score=1.0)]
    rotations = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    ]
    for index, candidate in enumerate(candidates):
        candidate.update(
            {
                "source_grasp_id": "g0",
                "source_grasp_equivalence_id": "g0",
                "source_object_goal_id": f"p{index}",
                "object_goal_pose": {
                    "frame": "world",
                    "translation_xyz": [0.48, -0.1, 0.43 if index == 0 else 0.42],
                    "rotation_matrix": rotations[index],
                },
            }
        )
    planned: list[str] = []

    def plan_only(target, start, timeout, attempts):
        del start, timeout, attempts
        planned.append(str(target["name"]))
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": [0.2]}],
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    response = _engine(
        clone_scene=_placement_scene,
        plan_only=plan_only,
    ).qualify(
        _request(
            candidates,
            overrides={"placement_waves": [2], "l5_pass_target": 1},
        )
    )

    assert response["selected_candidate_ids"] == ["c1"]
    assert planned == ["c1_stage0"]
    results = {row["candidate_id"]: row for row in response["results"]}
    assert results["c1"]["placement_vertical_extent_m"] == pytest.approx(0.04)
    assert results["c0"]["placement_vertical_extent_m"] == pytest.approx(0.06)
    assert results["c0"]["goal_legality"]["verdict"] == "PASS"
    assert sorted(results) == ["c0", "c1"]


def test_l4_pass_prefers_positive_settling_clearance_before_height() -> None:
    near_wall = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": -0.005,
        "placement_vertical_extent_m": 0.216,
        "stages": [{"joint_margin": 0.8, "min_singular_value": 0.8}],
    }
    centered = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": 0.003,
        "placement_vertical_extent_m": 0.220,
        "stages": [{"joint_margin": 0.2, "min_singular_value": 0.2}],
    }

    assert candidate_physical_quality_key(centered) < candidate_physical_quality_key(near_wall)


def test_l4_pass_prefers_low_support_energy_within_positive_clearance_tier() -> None:
    low_supported = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": 0.001,
        "placement_vertical_extent_m": 0.03,
        "stages": [{"joint_margin": 0.4, "min_singular_value": 0.4}],
    }
    tall_supported = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": 0.003,
        "placement_vertical_extent_m": 0.22,
        "stages": [{"joint_margin": 0.4, "min_singular_value": 0.4}],
    }

    assert candidate_physical_quality_key(low_supported) < candidate_physical_quality_key(
        tall_supported
    )


def test_subresolution_support_energy_uses_face_alignment_before_float_noise() -> None:
    less_settled = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": 0.007,
        "placement_support_energy_m": 0.1080,
        "placement_support_energy_resolution_m": 0.01,
        "placement_support_face_alignment_error_rad": math.radians(5.0),
        "stages": [{"joint_margin": 0.8, "min_singular_value": 0.8}],
    }
    face_aligned = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": 0.007,
        "placement_support_energy_m": 0.1085,
        "placement_support_energy_resolution_m": 0.01,
        "placement_support_face_alignment_error_rad": math.radians(2.0),
        "stages": [{"joint_margin": 0.4, "min_singular_value": 0.4}],
    }

    assert candidate_physical_quality_key(face_aligned) < candidate_physical_quality_key(
        less_settled
    )


def test_face_alignment_precedes_resolved_support_energy() -> None:
    low_energy = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": 0.007,
        "placement_support_energy_m": 0.055,
        "placement_support_energy_resolution_m": 0.01,
        "placement_support_face_alignment_error_rad": math.radians(20.0),
        "stages": [{"joint_margin": 0.4, "min_singular_value": 0.4}],
    }
    high_energy = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": 0.007,
        "placement_support_energy_m": 0.115,
        "placement_support_energy_resolution_m": 0.01,
        "placement_support_face_alignment_error_rad": 0.0,
        "stages": [{"joint_margin": 0.8, "min_singular_value": 0.8}],
    }

    assert candidate_physical_quality_key(high_energy) < candidate_physical_quality_key(low_energy)


def test_settling_sweep_clearance_precedes_lower_nominal_energy() -> None:
    moves_outside = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": 0.003,
        "placement_settling_sweep_clearance_m": -0.045,
        "placement_support_energy_m": 0.08,
        "placement_support_energy_resolution_m": 0.01,
        "placement_support_face_alignment_error_rad": math.radians(25.0),
        "stages": [{"joint_margin": 0.8, "min_singular_value": 0.8}],
    }
    remains_inside = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": 0.011,
        "placement_settling_sweep_clearance_m": 0.004,
        "placement_support_energy_m": 0.13,
        "placement_support_energy_resolution_m": 0.01,
        "placement_support_face_alignment_error_rad": math.radians(3.0),
        "stages": [{"joint_margin": 0.4, "min_singular_value": 0.4}],
    }

    assert candidate_physical_quality_key(remains_inside) < candidate_physical_quality_key(
        moves_outside
    )


def test_l4_pass_prefers_safer_margin_within_uncertain_clearance_tier() -> None:
    less_exposed = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": -0.003,
        "placement_vertical_extent_m": 0.22,
        "stages": [{"joint_margin": 0.4, "min_singular_value": 0.4}],
    }
    more_exposed = {
        "endpoint_pass": True,
        "placement_robust_clearance_m": -0.015,
        "placement_vertical_extent_m": 0.03,
        "stages": [{"joint_margin": 0.4, "min_singular_value": 0.4}],
    }

    assert candidate_physical_quality_key(less_exposed) < candidate_physical_quality_key(
        more_exposed
    )


def test_joint_scheduler_covers_two_complete_anyplace_branches():
    descriptors = []
    for placement_index in range(96):
        for grasp_index in range(2):
            candidate = _candidate(placement_index * 2 + grasp_index)
            candidate["source_grasp_id"] = f"g{grasp_index}"
            descriptors.append(
                {
                    "candidate_id": candidate["id"],
                    "candidate": candidate,
                    "candidate_pose_sha256": _hash(candidate),
                }
            )

    waves = schedule_candidate_waves(descriptors, purpose="placement")

    assert [wave.cumulative_per_branch for wave in waves] == [4, 8, 16, 32, 96]
    assert [len(wave.candidates) for wave in waves] == [8, 8, 16, 32, 128]
    assert sum(len(wave.candidates) for wave in waves) == 192
    assert [wave.candidates[0]["candidate"]["source_grasp_id"] for wave in waves] == ["g0"] * 5
    assert [
        waves[0].candidates[0]["candidate"]["source_grasp_id"],
        waves[0].candidates[1]["candidate"]["source_grasp_id"],
    ] == ["g0", "g1"]


def test_placement_first_wave_prioritizes_supported_geometry_before_pose_extremes() -> None:
    descriptors = []
    vertical_extents = [0.03, 0.04, 0.05, 0.08, 0.12, 0.22]
    for index, vertical_extent in enumerate(vertical_extents):
        candidate = _candidate(index, score=100.0 if index == 5 else 1.0)
        candidate["source_grasp_id"] = "g0"
        candidate["qualification_stages"][0]["xyz"] = [
            0.35 + index * 0.03,
            0.0,
            0.5,
        ]
        descriptors.append(
            {
                "candidate_id": candidate["id"],
                "candidate": candidate,
                "candidate_pose_sha256": _hash(candidate),
                "placement_vertical_extent_m": vertical_extent,
            }
        )

    waves = schedule_candidate_waves(
        descriptors,
        purpose="placement",
        placement_waves=[2, 6],
    )

    assert [row["candidate_id"] for row in waves[0].candidates] == ["c0", "c1"]
    assert sorted(row["candidate_id"] for wave in waves for row in wave.candidates) == [
        f"c{index}" for index in range(6)
    ]


def test_grasp_first_wave_is_quality_seeded_but_pose_diverse_without_deletion():
    descriptors = []
    for index in range(8):
        candidate = _candidate(index, score=8.0 - index)
        candidate["qualification_stages"][0]["xyz"] = [
            0.4 + index * 0.012,
            0.0,
            0.5,
        ]
        descriptors.append(
            {
                "candidate_id": candidate["id"],
                "candidate": candidate,
                "candidate_pose_sha256": _hash(candidate),
            }
        )

    waves = schedule_candidate_waves(
        descriptors,
        purpose="grasp",
        grasp_waves=[4, 8],
    )

    first_ids = [item["candidate_id"] for item in waves[0].candidates]
    all_ids = [item["candidate_id"] for wave in waves for item in wave.candidates]
    assert first_ids[0] == "c0"
    assert "c7" in first_ids
    assert len(set(first_ids)) == 4
    assert sorted(all_ids) == [f"c{index}" for index in range(8)]


def test_grasp_quality_prefers_centered_model_pose_without_applying_offset():
    centered = {
        "candidate_id": "centered",
        "endpoint_pass": True,
        "generator_score": 0.6,
        "fixed_candidate_index": 1,
        "se3_cluster_id": "se3_0001",
        "grasp_symmetry_family_id": "centered",
        "compile_parameters": {
            "camera_pose": {
                "target_closing_alignment": {
                    "correction_m": 0.003,
                    "target_span_m": 0.04,
                }
            }
        },
        "stages": [
            {
                "joint_margin": 0.05,
                "min_singular_value": 0.02,
                "joint_travel": 0.5,
                "collision_rescues": 0,
            }
        ],
    }
    off_center = {
        "candidate_id": "off-center",
        "endpoint_pass": True,
        "generator_score": 0.9,
        "fixed_candidate_index": 0,
        "se3_cluster_id": "se3_0000",
        "grasp_symmetry_family_id": "off-center",
        "compile_parameters": {
            "camera_pose": {
                "target_closing_alignment": {
                    "correction_m": -0.006,
                    "target_span_m": 0.04,
                }
            }
        },
        "stages": [
            {
                "joint_margin": 0.10,
                "min_singular_value": 0.15,
                "joint_travel": 0.4,
                "collision_rescues": 0,
            }
        ],
    }

    selected = select_grasp_branches(
        [off_center, centered],
        source={"joint_limits": {"lower": [-1.0], "upper": [1.0]}},
        limit=2,
    )

    assert selected == ["centered", "off-center"]
    assert centered["compile_parameters"]["camera_pose"] == {
        "target_closing_alignment": {
            "correction_m": 0.003,
            "target_span_m": 0.04,
        }
    }


def test_joint_scheduler_defers_frozen_reserve_branches_until_primary_exhausts():
    descriptors = []
    for placement_index in range(96):
        for grasp_index in range(4):
            candidate = _candidate(placement_index * 4 + grasp_index)
            candidate.update(
                {
                    "source_grasp_id": f"g{grasp_index}",
                    "frozen_pair_batch_index": grasp_index // 2,
                    "frozen_pair_batch_role": ("primary" if grasp_index < 2 else "reserve"),
                }
            )
            descriptors.append(
                {
                    "candidate_id": candidate["id"],
                    "candidate": candidate,
                    "candidate_pose_sha256": _hash(candidate),
                }
            )

    waves = schedule_candidate_waves(descriptors, purpose="placement")

    assert [wave.frozen_pair_batch_index for wave in waves] == [0] * 5 + [1] * 5
    assert [wave.cumulative_per_branch for wave in waves] == [
        4,
        8,
        16,
        32,
        96,
        4,
        8,
        16,
        32,
        96,
    ]
    assert [len(wave.candidates) for wave in waves] == [
        8,
        8,
        16,
        32,
        128,
        8,
        8,
        16,
        32,
        128,
    ]
    assert all(
        int(descriptor["candidate"]["source_grasp_id"][1:]) < 2
        for wave in waves[:5]
        for descriptor in wave.candidates
    )
    assert all(
        int(descriptor["candidate"]["source_grasp_id"][1:]) >= 2
        for wave in waves[5:]
        for descriptor in wave.candidates
    )


def test_goal_legality_barrier_evaluates_one_goal_once_and_rejects_before_ik():
    ik_calls = 0
    candidates = []
    for grasp_id in ("g0", "g1"):
        candidate = _candidate(len(candidates))
        candidate.update(
            {
                "source_grasp_id": grasp_id,
                "source_grasp_equivalence_id": grasp_id,
                "source_object_goal_id": "p0",
                "object_goal_pose": {
                    "frame": "world",
                    "translation_xyz": [0.62, -0.1, 0.43],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            }
        )
        candidates.append(candidate)

    def ik(*_args):
        nonlocal ik_calls
        ik_calls += 1
        return {"ok": False}

    response = _engine(
        clone_scene=_placement_scene,
        compute_ik=ik,
    ).qualify(_request(candidates))

    assert ik_calls == 0
    assert response["metrics"]["goal_legality_unique_count"] == 1
    assert response["metrics"]["goal_legality_reject_count"] == 1
    assert response["metrics"]["pair_legality_evaluation_count"] == 0
    assert [item["reason"] for item in response["results"]] == [
        "goal_footprint_outside_placement_region",
        "goal_footprint_outside_placement_region",
    ]
    assert all(
        item["pair_legality"]["reason"] == "goal_legality_rejected" for item in response["results"]
    )


def test_parallel_gripper_symmetry_shares_pair_gate_but_keeps_two_evidence_copies():
    candidates = []
    for index, grasp_id in enumerate(("g0", "g0_sym180")):
        candidate = _candidate(index)
        candidate.update(
            {
                "source_grasp_id": grasp_id,
                "source_grasp_equivalence_id": "g0",
                "source_grasp_symmetry_equivalent": index == 1,
                "source_object_goal_id": "p0",
                "object_goal_pose": {
                    "frame": "world",
                    "translation_xyz": [0.48, -0.1, 0.43],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            }
        )
        candidates.append(candidate)

    response = _engine(clone_scene=_placement_scene).qualify(_request(candidates))

    assert response["metrics"]["goal_legality_unique_count"] == 1
    assert response["metrics"]["pair_legality_evaluation_count"] == 1
    assert response["metrics"]["pair_legality_shared_count"] == 1
    first, second = response["results"]
    assert first["pair_legality"]["screening_reused"] is False
    assert second["pair_legality"]["screening_reused"] is True
    assert second["pair_legality"]["shared_from_candidate_id"] == "c0"
    assert [first["candidate_id"], second["candidate_id"]] == ["c0", "c1"]


def test_pair_legality_runs_only_for_the_current_small_deep_wave():
    candidates = []
    for goal_index in range(10):
        for grasp_index in range(2):
            candidate = _candidate(goal_index * 2 + grasp_index)
            candidate.update(
                {
                    "source_grasp_id": f"g{grasp_index}",
                    "source_grasp_equivalence_id": f"g{grasp_index}",
                    "source_object_goal_id": f"p{goal_index}",
                    "object_goal_pose": {
                        "frame": "world",
                        "translation_xyz": [0.48, -0.1, 0.43],
                        "rotation_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    },
                }
            )
            candidates.append(candidate)

    response = _engine(clone_scene=_placement_scene).qualify(
        _request(
            candidates,
            overrides={
                "l5_pass_target": 1,
                "placement_waves": [4, 8, 10],
            },
        )
    )

    assert response["metrics"]["goal_legality_unique_count"] == 10
    assert response["metrics"]["pair_legality_reached_count"] == 8
    assert response["metrics"]["pair_legality_evaluation_count"] == 8
    assert response["metrics"]["pair_legality_pending_count"] == 12
    assert len(response["waves"]) == 1
    assert response["waves"][0]["candidate_count"] == 8
    assert response["waves"][0]["deep_candidate_count"] == 8


def test_frozen_goal_binds_anyplace_motion_to_scene_box_not_pointcloud_centroid():
    candidate = _candidate(0)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            # The visible point-cloud centroid can be well above the physical
            # collision-box origin and must never be used as the box center.
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.50],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "object_motion_world_transform": {
                "frame": "world",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.20],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        }
    )

    response = _engine(clone_scene=_placement_scene).qualify(_request([candidate]))

    assert response["selected_candidate_ids"] == ["c0"]
    binding = response["results"][0]["goal_legality"]["checks"]["object_frame_binding"]
    assert binding["pointcloud_goal_translation_xyz"][2] == 0.50
    assert binding["collision_goal_translation_xyz"] == pytest.approx([0.48, -0.1, 0.435])
    assert binding["collision_goal_pose"]["translation_xyz"] == pytest.approx(
        [0.48, -0.1, 0.435]
    )
    assert response["results"][0]["goal_legality"]["verdict"] == "PASS"


def test_attached_frozen_goal_ignores_stale_predicted_object_motion():
    scene = _placement_scene()
    attached = scene["world_specs"].pop("target")
    attached["pose_xyz"] = [0.0, 0.02, 0.15]
    scene["attached_specs"]["target"] = attached
    candidate = _candidate(0)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "world_object_goal_pose": {
                "convention": "T_world_object_goal",
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.43],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            # This remains as provenance on frozen predicted PASS goals. It
            # must not be multiplied by the attached object's EEF-local pose.
            "object_motion_world_transform": {
                "frame": "world",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.2],
                    [0.0, 1.0, 0.0, 0.1],
                    [0.0, 0.0, 1.0, 0.02],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        }
    )

    response = _engine(clone_scene=lambda: scene).qualify(_request([candidate]))

    assert response["selected_candidate_ids"] == ["c0"]
    binding = response["results"][0]["goal_legality"]["checks"]["object_frame_binding"]
    assert binding["method"] == "direct_physical_object_goal"
    assert binding["target_is_attached"] is True
    assert binding["collision_goal_translation_xyz"] == pytest.approx([0.48, -0.1, 0.43])


def test_support_contact_uncertainty_is_separate_from_static_collision_tolerance():
    within_contact_band = _candidate(0)
    within_contact_band.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.426],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        }
    )
    definite_penetration = json.loads(json.dumps(within_contact_band))
    definite_penetration["id"] = "c1"
    definite_penetration["source_object_goal_id"] = "p1"
    definite_penetration["object_goal_pose"]["translation_xyz"][2] = 0.423

    response = _engine(clone_scene=_placement_scene).qualify(
        _request([within_contact_band, definite_penetration])
    )

    assert response["selected_candidate_ids"] == ["c0"]
    assert (
        response["results"][0]["goal_legality"]["checks"]["support"]["tolerance_basis"]
        == "sensor_and_model_support_contact_uncertainty"
    )
    assert response["results"][1]["reason"] == "goal_support_surface_penetration"


def test_partial_pointcloud_goal_is_bound_to_exact_physical_support_before_pair_ik():
    candidate = _candidate(0)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.443],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "object_motion_world_transform": {
                "frame": "world",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.2],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -0.0115],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "frozen_contact_pose": {
                "frame": "world",
                "xyz": [0.3, 0.0, 0.5],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "initial_scene_transition": "virtual_attach",
            "initial_scene_transition_pose": {
                "frame": "world",
                "xyz": [0.3, 0.0, 0.5],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "predicted_attachment_transform": {
                "translation_xyz": [-0.018, -0.1, -0.045],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.5, 0.0, 0.4885],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ],
        }
    )
    descriptor = {"candidate_id": "c0", "candidate": candidate}

    legality = evaluate_placement_goal_legality(
        descriptor,
        scene=_placement_scene(),
    )

    assert legality["verdict"] == "PASS"
    binding = legality["checks"]["object_frame_binding"]
    reconciliation = binding["support_contact_reconciliation"]
    assert reconciliation["applied"] is True
    assert reconciliation["release_clearance_m"] == pytest.approx(0.005)
    assert reconciliation["required_translation_z_m"] == pytest.approx(0.0165)
    assert reconciliation["qualified_bottom_z_m"] == pytest.approx(0.405)
    assert binding["collision_goal_pose"]["translation_xyz"] == pytest.approx(
        [0.48, -0.1, 0.435]
    )
    assert binding["pointcloud_goal_translation_xyz"] == pytest.approx([0.48, -0.1, 0.443])

    bind_qualified_placement_goal(descriptor, legality)

    bound = descriptor["candidate"]
    assert bound["qualification_stages"][0]["xyz"] == pytest.approx([0.5, 0.0, 0.505])
    assert bound["object_motion_world_transform"]["transform_matrix"][2][3] == (
        pytest.approx(0.005)
    )
    assert bound["physical_scene_attachment_required"] is True
    pair = evaluate_grasp_placement_pair_legality(
        descriptor,
        scene=_placement_scene(),
        workspace_filter=None,
    )
    assert pair["verdict"] == "PASS"


def test_release_z_offset_translates_only_the_terminal_not_the_settled_goal():
    scene = _placement_scene()
    scene["placement_region"]["release_z_offset_m"] = 0.20
    candidate = _candidate(0)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.43],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.48, -0.1, 0.43],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ],
        }
    )
    descriptor = {"candidate_id": "c0", "candidate": candidate}

    legality = evaluate_placement_goal_legality(descriptor, scene=scene)
    bind_qualified_placement_goal(descriptor, legality)

    binding = legality["checks"]["object_frame_binding"]
    assert legality["verdict"] == "PASS"
    assert binding["collision_goal_pose"]["translation_xyz"] == pytest.approx(
        [0.48, -0.1, 0.43]
    )
    assert binding["release_collision_goal_pose"]["translation_xyz"] == pytest.approx(
        [0.48, -0.1, 0.63]
    )
    stage = descriptor["candidate"]["qualification_stages"][0]
    assert stage["xyz"] == pytest.approx([0.48, -0.1, 0.63])
    assert stage["placement_release_z_offset_m"] == pytest.approx(0.20)


@pytest.mark.parametrize(
    "container_quat",
    [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
    ],
    ids=["axis_aligned", "yaw_rotated"],
)
def test_container_release_clears_lowest_entry_without_moving_settled_goal(
    container_quat,
):
    scene = _physical_bin_placement_scene()
    scene["world_specs"]["parts_bin"]["pose_quat_xyzw"] = container_quat
    candidate = _candidate(0)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.05],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.48, -0.1, 0.25],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ],
        }
    )
    descriptor = {"candidate_id": "c0", "candidate": candidate}

    legality = evaluate_placement_goal_legality(descriptor, scene=scene)
    bind_qualified_placement_goal(descriptor, legality)

    assert legality["verdict"] == "PASS"
    binding = legality["checks"]["object_frame_binding"]
    selection = binding["release_offset_selection"]
    assert selection["source"] == "container_exterior_entry_clearance"
    assert selection["configured_drop_height_m"] == pytest.approx(0.05)
    assert selection["support_collision_maximum_z_m"] == pytest.approx(0.18)
    assert selection["support_collision_height_above_surface_m"] == pytest.approx(0.16)
    assert selection["support_entry_minimum_z_m"] == pytest.approx(0.09)
    assert selection["support_entry_height_above_surface_m"] == pytest.approx(0.07)
    assert selection["entry_clearance_above_edge_m"] == pytest.approx(0.005)
    assert selection["support_barrier_count"] == 4
    assert selection["support_exterior_entry_barrier_count"] == 4
    assert selection["support_unclassified_barrier_count"] == 0
    assert selection["support_entry_geometry_proven"] is True
    assert selection["container_clearance_m"] == pytest.approx(0.005)
    assert selection["effective_offset_m"] == pytest.approx(0.075)
    assert binding["collision_goal_pose"]["translation_xyz"] == pytest.approx(
        [0.48, -0.1, 0.05]
    )
    assert binding["release_collision_goal_pose"]["translation_xyz"] == pytest.approx(
        [0.48, -0.1, 0.125]
    )
    stage = descriptor["candidate"]["qualification_stages"][0]
    assert stage["xyz"] == pytest.approx([0.48, -0.1, 0.325])
    assert stage["placement_release_z_offset_m"] == pytest.approx(0.075)


@pytest.mark.parametrize(
    "extra_primitive",
    [
        {
            "shape": "box",
            "size_xyz": [0.02, 0.30, 0.04],
            "pose_xyz": [0.08, 0.0, 0.04],
            "pose_rpy": [0.0, 0.0, 0.0],
        },
        {
            "shape": "box",
            "size_xyz": [0.32, 0.02, 0.02],
            "pose_xyz": [0.0, -0.18, 0.06],
            "pose_rpy": [0.0, 0.0, 0.0],
        },
    ],
    ids=["internal_divider", "suspended_edge_ledge"],
)
def test_container_entry_height_ignores_unproven_obstacles(extra_primitive):
    scene = _physical_bin_placement_scene()
    scene["world_specs"]["parts_bin"]["primitives"].append(extra_primitive)
    candidate = _candidate(0)
    candidate.update(
        {
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.05],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        }
    )

    legality = evaluate_placement_goal_legality(
        {"candidate_id": "c0", "candidate": candidate},
        scene=scene,
    )

    selection = legality["checks"]["object_frame_binding"][
        "release_offset_selection"
    ]
    assert legality["verdict"] == "PASS"
    assert selection["support_entry_minimum_z_m"] == pytest.approx(0.09)
    assert selection["effective_offset_m"] == pytest.approx(0.075)
    assert selection["support_exterior_entry_barrier_count"] == 4
    assert selection["support_unclassified_barrier_count"] == 1


def test_unclassified_compound_obstacles_do_not_override_configured_drop_height():
    scene = _physical_bin_placement_scene()
    base = scene["world_specs"]["parts_bin"]["primitives"][0]
    scene["world_specs"]["parts_bin"]["primitives"] = [
        base,
        {
            "shape": "box",
            "size_xyz": [0.02, 0.30, 0.12],
            "pose_xyz": [0.08, 0.0, 0.06],
            "pose_rpy": [0.0, 0.0, 0.0],
        },
    ]
    candidate = _candidate(0)
    candidate.update(
        {
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.05],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        }
    )

    legality = evaluate_placement_goal_legality(
        {"candidate_id": "c0", "candidate": candidate},
        scene=scene,
    )

    selection = legality["checks"]["object_frame_binding"][
        "release_offset_selection"
    ]
    assert legality["verdict"] == "PASS"
    assert selection["source"] == "configured_drop_height"
    assert selection["support_entry_geometry_proven"] is False
    assert selection["support_exterior_entry_barrier_count"] == 0
    assert selection["support_unclassified_barrier_count"] == 1
    assert selection["effective_offset_m"] == pytest.approx(0.05)


def _rotated_container_goal() -> tuple[dict, dict]:
    scene = _physical_bin_placement_scene()
    scene["world_specs"]["target"]["pose_xyz"] = [0.24, -0.19, 0.05]
    candidate = {
        "id": "placement_rotated",
        "object_goal_pose": {
            "frame": "world",
            "translation_xyz": [0.48, -0.1, 0.05],
            "rotation_matrix": [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        "world_object_goal_pose": {
            "frame": "world",
            "translation_xyz": [0.48, -0.1, 0.05],
            "rotation_matrix": [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        "object_motion_world_transform": {
            "frame": "world",
            "convention": "T_world_motion_applied_left",
            "transform_matrix": [
                [0.0, -1.0, 0.0, 0.29],
                [1.0, 0.0, 0.0, -0.34],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    }
    return scene, candidate


def test_container_drop_preserves_anyplace_se3_and_adds_release_height():
    scene, candidate = _rotated_container_goal()
    descriptor = {"candidate_id": candidate["id"], "candidate": candidate}

    legality = evaluate_placement_goal_legality(descriptor, scene=scene)
    bind_qualified_placement_goal(descriptor, legality)

    binding = legality["checks"]["object_frame_binding"]
    settled = binding["collision_goal_pose"]
    release = binding["release_collision_goal_pose"]
    assert legality["verdict"] == "PASS"
    assert binding["release_orientation_policy"] == "model_settled_orientation"
    assert binding["container_drop"]["model_destination_xy_preserved"] is True
    assert binding["container_drop"]["model_destination_se3_preserved"] is True
    assert settled["translation_xyz"][:2] == pytest.approx([0.48, -0.1])
    assert settled["rotation_matrix"] == [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert release["translation_xyz"][:2] == pytest.approx([0.48, -0.1])
    assert release["rotation_matrix"] == [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    bound = descriptor["candidate"]
    assert bound["goal_legality_prebound"] is True
    assert bound["container_drop_release_prebound"] is True
    assert bound["model_object_motion_world_transform"] == (
        candidate["object_motion_world_transform"]
    )
    release_motion = bound["object_motion_world_transform"]["transform_matrix"]
    assert [row[:3] for row in release_motion[:3]] == [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_goal_prebind_rpc_freezes_container_release_before_pair_compilation():
    scene, candidate = _rotated_container_goal()
    engine = _engine(clone_scene=lambda: scene)
    rpc_timeouts: list[float] = []

    def rpc(_name, request, timeout):
        rpc_timeouts.append(timeout)
        return engine.qualify(request)

    qualifier = MoveItCandidateQualifier(
        rpc,
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
    )

    goals, summary = qualifier.prebind_placement_goals(
        [candidate],
        scene_epoch=1,
        planning_scene_revision=4,
    )

    assert summary["frozen_goal_legality_screen_complete"] is True
    assert summary["frozen_goal_legality_pass_count"] == 1
    assert summary["frozen_goal_legality_reason_counts"] == {
        "goal_legality_qualified": 1
    }
    assert summary["frozen_goal_legality_elapsed_s"] >= 0.0
    assert summary["frozen_goal_legality_rpc_timeout_s"] == rpc_timeouts[0]
    assert rpc_timeouts[0] > 30.0
    assert len(goals) == 1
    assert goals[0]["container_drop_release_prebound"] is True
    assert goals[0]["qualified_release_pointcloud_object_goal_pose"][
        "rotation_matrix"
    ] == [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_goal_prebind_can_materialize_configured_height_recovery_variant():
    scene, candidate = _rotated_container_goal()
    engine = _engine(clone_scene=lambda: scene)
    qualifier = MoveItCandidateQualifier(
        lambda _name, request, _timeout: engine.qualify(request),
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
    )

    primary, _ = qualifier.prebind_placement_goals(
        [candidate],
        scene_epoch=1,
        planning_scene_revision=4,
    )
    fallback, summary = qualifier.prebind_placement_goals(
        primary,
        scene_epoch=1,
        planning_scene_revision=4,
        release_height_variant="configured_drop_height_fallback",
    )

    primary_selection = primary[0]["placement_release_offset_selection"]
    fallback_selection = fallback[0]["placement_release_offset_selection"]
    assert primary_selection["effective_offset_m"] == pytest.approx(0.075)
    assert fallback_selection["source"] == "configured_drop_height_fallback"
    assert fallback_selection["primary_effective_offset_m"] == pytest.approx(0.075)
    assert fallback_selection["effective_offset_m"] == pytest.approx(0.05)
    assert fallback_selection["fallback_activated"] is True
    assert fallback[0]["qualified_release_object_goal_pose"]["translation_xyz"][
        2
    ] == pytest.approx(
        primary[0]["qualified_release_object_goal_pose"]["translation_xyz"][2]
        - 0.025
    )
    assert summary["frozen_goal_release_height_variant"] == (
        "configured_drop_height_fallback"
    )


def test_goal_prebind_can_raise_release_above_every_exterior_barrier():
    scene, candidate = _rotated_container_goal()
    engine = _engine(clone_scene=lambda: scene)
    qualifier = MoveItCandidateQualifier(
        lambda _name, request, _timeout: engine.qualify(request),
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
    )

    primary, _ = qualifier.prebind_placement_goals(
        [candidate],
        scene_epoch=1,
        planning_scene_revision=4,
    )
    cleared, summary = qualifier.prebind_placement_goals(
        primary,
        scene_epoch=1,
        planning_scene_revision=4,
        release_height_variant="full_barrier_clearance",
    )

    primary_selection = primary[0]["placement_release_offset_selection"]
    cleared_selection = cleared[0]["placement_release_offset_selection"]
    assert primary_selection["effective_offset_m"] == pytest.approx(0.075)
    assert primary_selection["full_barrier_clearance_offset_m"] == pytest.approx(
        0.165
    )
    assert cleared_selection["source"] == "container_full_barrier_clearance"
    assert cleared_selection["primary_effective_offset_m"] == pytest.approx(0.075)
    assert cleared_selection["effective_offset_m"] == pytest.approx(0.165)
    assert cleared_selection["fallback_activated"] is True
    assert cleared[0]["qualified_release_object_goal_pose"]["translation_xyz"][
        2
    ] == pytest.approx(
        primary[0]["qualified_release_object_goal_pose"]["translation_xyz"][2]
        + 0.09
    )
    assert summary["frozen_goal_release_height_variant"] == (
        "full_barrier_clearance"
    )


def test_full_barrier_release_clears_gripper_from_tall_container_wall():
    scene, candidate = _rotated_container_goal()
    scene["gripper_collision_boxes"] = [
        {
            "id": "mount_plate",
            "shape": "box",
            "size_xyz": [0.08, 0.08, 0.012],
            "pose_xyz": [0.14, 0.0, 0.006],
            "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    ]
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": candidate["id"],
            "compile_parameters": {
                "attachment_transform": {
                    "parent_frame": "eef",
                    "child_frame": "object",
                    "translation_xyz": [0.0, 0.0, 0.0],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            },
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.48, -0.1, 0.05],
                    "rotation_matrix": [
                        [0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            ],
        }
    )
    engine = _engine(clone_scene=lambda: scene)
    qualifier = MoveItCandidateQualifier(
        lambda _name, request, _timeout: engine.qualify(request),
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
    )
    primary, _ = qualifier.prebind_placement_goals(
        [candidate], scene_epoch=1, planning_scene_revision=4
    )
    cleared, _ = qualifier.prebind_placement_goals(
        primary,
        scene_epoch=1,
        planning_scene_revision=4,
        release_height_variant="full_barrier_clearance",
    )

    primary_pair = evaluate_grasp_placement_pair_legality(
        {"candidate_id": "primary", "candidate": primary[0]},
        scene=scene,
        workspace_filter=None,
    )
    cleared_pair = evaluate_grasp_placement_pair_legality(
        {"candidate_id": "cleared", "candidate": cleared[0]},
        scene=scene,
        workspace_filter=None,
    )

    assert primary_pair["verdict"] == "FAIL"
    assert primary_pair["reason"] == "gripper_static_collision"
    assert cleared_pair["verdict"] == "PASS"
    assert cleared_pair["checks"]["eef_chain"]["pass"] is True


def test_container_goal_prebind_is_idempotent_after_pair_compilation():
    scene, candidate = _rotated_container_goal()
    descriptor = {"candidate_id": candidate["id"], "candidate": candidate}
    first = evaluate_placement_goal_legality(descriptor, scene=scene)
    bind_qualified_placement_goal(descriptor, first)
    bound = descriptor["candidate"]
    first_release = bound["qualified_release_pointcloud_object_goal_pose"]

    # This is the same representation change made by the placement compiler:
    # the executable release goal becomes public while the model goal remains
    # immutable private evidence.
    bound["world_object_goal_pose"] = dict(first_release)
    bound["object_goal_pose"] = dict(first_release)
    second = evaluate_placement_goal_legality(descriptor, scene=scene)
    bind_qualified_placement_goal(descriptor, second)

    second_binding = second["checks"]["object_frame_binding"]
    assert second["verdict"] == "PASS"
    assert second["checks"]["se3"]["source"] == (
        "immutable_model_pointcloud_goal"
    )
    assert second_binding["collision_goal_pose"]["rotation_matrix"] == [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert descriptor["candidate"][
        "qualified_release_pointcloud_object_goal_pose"
    ] == first_release


def test_prebound_container_release_stage_is_not_corrected_twice():
    scene, candidate = _rotated_container_goal()
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "frozen_contact_pose": {
                "frame": "world",
                "xyz": [0.3, 0.0, 0.5],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        }
    )
    descriptor = {"candidate_id": candidate["id"], "candidate": candidate}

    first = evaluate_placement_goal_legality(descriptor, scene=scene)
    bind_qualified_placement_goal(descriptor, first)
    bound = descriptor["candidate"]
    release_motion = bound["object_motion_world_transform"]["transform_matrix"]
    contact_xyz = bound["frozen_contact_pose"]["xyz"]
    release_xyz = [
        sum(release_motion[row][column] * contact_xyz[column] for column in range(3))
        + release_motion[row][3]
        for row in range(3)
    ]
    bound["qualification_stages"] = [
        {
            "name": "release",
            "xyz": release_xyz,
            "rotation_matrix": [row[:3] for row in release_motion[:3]],
        }
    ]

    second = evaluate_placement_goal_legality(descriptor, scene=scene)
    bind_qualified_placement_goal(descriptor, second)

    stage = descriptor["candidate"]["qualification_stages"][0]
    assert stage["xyz"] == pytest.approx(release_xyz)
    assert stage["release_target_translation_correction_application"] == (
        "already_materialized_in_compiled_terminal"
    )
    pair = evaluate_grasp_placement_pair_legality(
        descriptor,
        scene=scene,
        workspace_filter=None,
    )
    assert pair["checks"]["eef_chain"]["pass"] is True


def test_flat_support_keeps_configured_release_minimum():
    scene = _placement_scene()
    scene["placement_region"]["release_z_offset_m"] = 0.05
    candidate = _candidate(0)
    candidate.update(
        {
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.43],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        }
    )

    legality = evaluate_placement_goal_legality(
        {"candidate_id": "c0", "candidate": candidate},
        scene=scene,
    )

    selection = legality["checks"]["object_frame_binding"]["release_offset_selection"]
    assert legality["verdict"] == "PASS"
    assert selection["support_geometry_available"] is True
    assert selection["source"] == "configured_drop_height"
    assert selection["support_barrier_count"] == 0
    assert selection["container_clearance_m"] == pytest.approx(0.0)
    assert selection["effective_offset_m"] == pytest.approx(0.05)


def test_artificial_placement_waypoints_are_rejected_before_ik():
    ik_calls = 0
    candidate = _candidate(0, stages=3)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.43],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "compile_parameters": {
                "attachment_transform": {
                    "parent_frame": "eef",
                    "child_frame": "object",
                    "translation_xyz": [0.0, 0.0, 0.0],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
                "placement_candidate": {
                    "object_goal_pose": {
                        "frame": "world",
                        "translation_xyz": [0.48, -0.1, 0.50],
                        "rotation_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    }
                },
            },
        }
    )
    candidate["qualification_stages"] = [
        {"name": "hover", "xyz": [0.48, -0.1, 0.60], "quat_xyzw": [0, 0, 0, 1]},
        {"name": "release", "xyz": [0.49, -0.1, 0.50], "quat_xyzw": [0, 0, 0, 1]},
        {"name": "retreat", "xyz": [0.49, -0.1, 0.60], "quat_xyzw": [0, 0, 0, 1]},
    ]

    def ik(*_args):
        nonlocal ik_calls
        ik_calls += 1
        return {"ok": False}

    response = _engine(
        clone_scene=_placement_scene,
        compute_ik=ik,
    ).qualify(_request([candidate]))

    assert ik_calls == 0
    assert response["results"][0]["reason"] == "pair_artificial_waypoint_forbidden"
    assert response["metrics"]["pair_legality_reject_count"] == 1


def test_pair_chain_accepts_fixed_precision_equivalent_rotations():
    # Regression for live frozen-pair artifacts: the motion/contact composition
    # and the serialized release pose differ only through fixed-precision
    # matrix arithmetic.  A trace/acos distance falsely reported ~2 mrad.
    rotation = [
        [-0.489227965089, -0.106757288545, 0.865596042682],
        [-0.866254593951, -0.055771671978, -0.496478946107],
        [0.101278178226, -0.992719284323, -0.065193324362],
    ]
    candidate = _candidate(0)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.43],
                "rotation_matrix": rotation,
            },
            "compile_parameters": {
                "attachment_transform": {
                    "parent_frame": "eef",
                    "child_frame": "object",
                    "translation_xyz": [0.0, 0.0, 0.0],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            },
        }
    )
    candidate["qualification_stages"] = [
        {"name": "release", "xyz": [0.48, -0.1, 0.43], "rotation_matrix": rotation},
    ]

    response = _engine(clone_scene=_placement_scene).qualify(_request([candidate]))

    assert response["selected_candidate_ids"] == ["c0"]
    chain = response["results"][0]["pair_legality"]["checks"]["eef_chain"]
    assert chain["orientation_error_rad"] == pytest.approx(0.0, abs=1e-10)


def test_attached_pair_chain_derives_eef_goal_from_measured_attachment():
    candidate = _candidate(0, stages=3)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "object_goal_pose": {
                "convention": "T_world_object_goal",
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.43],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "compile_parameters": {
                "attachment_transform": {
                    "parent_frame": "eef",
                    "child_frame": "object",
                    "translation_xyz": [0.0, 0.0, -0.1],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
                "placement_candidate": {
                    # This is an object pose, not a direct EEF pose.
                    "object_goal_pose": {
                        "convention": "T_world_object_goal",
                        "frame": "world",
                        "translation_xyz": [0.48, -0.1, 0.43],
                        "rotation_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    }
                },
            },
        }
    )
    candidate["qualification_stages"] = [
        {"name": "release", "xyz": [0.48, -0.1, 0.53], "quat_xyzw": [0, 0, 0, 1]},
    ]

    response = _engine(clone_scene=_placement_scene).qualify(_request([candidate]))

    assert response["selected_candidate_ids"] == ["c0"]
    chain = response["results"][0]["pair_legality"]["checks"]["eef_chain"]
    assert chain["pass"] is True
    assert chain["translation_error_m"] == pytest.approx(0.0)
    assert response["results"][0]["pair_legality"]["checks"]["stage_se3"]["stage_count"] == 1


def test_rebased_pair_uses_qualified_object_goal_instead_of_stale_motion():
    candidate = _candidate(0)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "frozen_contact_pose": {
                "xyz": [0.2, 0.1, 0.5],
                "quat_xyzw": [0, 0, 0, 1],
            },
            # This transform belonged to the object pose before a failed close
            # moved it and must not override the newly bound physical goal.
            "object_motion_world_transform": {
                "transform_matrix": [
                    [1, 0, 0, 0.2],
                    [0, 1, 0, 0.0],
                    [0, 0, 1, 0.0],
                    [0, 0, 0, 1],
                ]
            },
            "qualified_world_collision_object_goal_pose": {
                "xyz": [0.48, -0.1, 0.43],
                "quat_xyzw": [0, 0, 0, 1],
            },
            "frozen_object_motion_rebase": {
                "schema_version": "openeta.frozen_object_motion_rebase.v1",
                "model_inference_invoked": False,
            },
            "compile_parameters": {
                "attachment_transform": {
                    "xyz": [0.0, 0.0, -0.1],
                    "quat_xyzw": [0, 0, 0, 1],
                }
            },
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.48, -0.1, 0.53],
                    "quat_xyzw": [0, 0, 0, 1],
                }
            ],
        }
    )

    evidence = evaluate_grasp_placement_pair_legality(
        {"candidate_id": "c0", "candidate": candidate},
        scene={},
        workspace_filter=None,
    )

    assert evidence["verdict"] == "PASS"
    assert evidence["checks"]["eef_chain"]["translation_error_m"] == pytest.approx(0.0)


def test_object_goal_static_collision_is_rejected_by_target_gate():
    scene = _placement_scene()
    scene["world_specs"]["distractor"] = {
        "id": "distractor",
        "shape": "box",
        "size_xyz": [0.05, 0.05, 0.08],
        "pose_xyz": [0.28, 0.12, 0.44],
        "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gazebo_static": True,
    }
    scene["placement_region"]["center_xy"] = [0.28, 0.12]
    scene["placement_region"]["size_xy_m"] = [0.12, 0.12]
    candidate = _candidate(0)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.28, 0.12, 0.44],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.28, 0.12, 0.44],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ],
        }
    )

    response = _engine(clone_scene=lambda: scene).qualify(_request([candidate]))

    assert response["results"][0]["reason"] == "goal_static_obstacle_penetration"
    collisions = response["results"][0]["goal_legality"]["checks"]["static_scene_collision"][
        "collision_ids"
    ]
    assert collisions == ["distractor"]
    assert (
        "distractor"
        in response["results"][0]["goal_legality"]["checks"]["static_scene_collision"][
            "evaluated_obstacle_ids"
        ]
    )
    assert response["metrics"]["screening_attempt_count"] == 0


def test_object_goal_dynamic_overlap_is_evidence_not_a_hard_reject():
    scene = _placement_scene()
    scene["world_specs"]["settled_payload"] = {
        "id": "settled_payload",
        "shape": "box",
        "size_xyz": [0.05, 0.05, 0.08],
        "pose_xyz": [0.28, 0.12, 0.44],
        "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gazebo_static": False,
    }
    scene["placement_region"].update(
        {
            "center_xy": [0.28, 0.12],
            "size_xy_m": [0.12, 0.12],
            "acceptance_semantics": "stable_geometry_centroid_inside",
        }
    )
    candidate = _candidate(0)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.28, 0.12, 0.44],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        }
    )
    descriptor = {"candidate_id": "p0", "candidate": candidate}

    evidence = evaluate_placement_goal_legality(descriptor, scene=scene)
    bind_qualified_placement_goal(descriptor, evidence)

    assert evidence["verdict"] == "PASS"
    collision = evidence["checks"]["static_scene_collision"]
    assert collision["collision_ids"] == []
    assert collision["dynamic_overlap_ids"] == ["settled_payload"]
    assert descriptor["candidate"]["qualified_settled_dynamic_overlap_ids"] == [
        "settled_payload"
    ]


def test_compound_bin_exempts_only_support_floor_not_collision_wall() -> None:
    scene = _placement_scene()
    scene["world_specs"]["table"] = {
        "id": "table",
        "shape": "compound",
        "size_xyz": [0.12, 0.12, 0.1],
        "pose_xyz": [0.0, 0.0, 0.0],
        "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "primitives": [
            {
                "shape": "box",
                "size_xyz": [0.12, 0.12, 0.02],
                "pose_xyz": [0.0, 0.0, 0.39],
                "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "shape": "box",
                "size_xyz": [0.01, 0.12, 0.1],
                "pose_xyz": [0.035, 0.0, 0.45],
                "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        ],
    }
    scene["placement_region"].update(
        {
            "center_xy": [0.0, 0.0],
            "size_xy_m": [0.12, 0.12],
            "support_z_m": 0.4,
            "static_penetration_tolerance_m": 0.0,
        }
    )
    candidate = _candidate(0)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.03, 0.0, 0.43],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        }
    )

    response = _engine(clone_scene=lambda: scene).qualify(_request([candidate]))

    result = response["results"][0]
    assert result["reason"] == "goal_static_obstacle_penetration"
    collision = result["goal_legality"]["checks"]["static_scene_collision"]
    assert collision["collision_ids"] == ["table"]
    assert collision["support_contact_primitive_count"] == 1
    assert collision["support_barrier_primitive_count"] == 1


def test_exact_gripper_collision_primitive_is_rejected_by_pair_gate():
    scene = _placement_scene()
    scene["gripper_collision_boxes"] = [
        {
            "id": "mount_plate",
            "shape": "box",
            "size_xyz": [0.08, 0.08, 0.012],
            "pose_xyz": [0.0, 0.0, 0.006],
            "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    ]
    candidate = _candidate(0)
    candidate.update(
        {
            "source_grasp_id": "g0",
            "source_object_goal_id": "p0",
            "object_goal_pose": {
                "frame": "world",
                "translation_xyz": [0.48, -0.1, 0.43],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "qualification_stages": [
                {
                    "name": "release",
                    "xyz": [0.48, -0.1, 0.39],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ],
            "compile_parameters": {
                "attachment_transform": {
                    "parent_frame": "eef",
                    "child_frame": "object",
                    "translation_xyz": [0.0, 0.0, 0.04],
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            },
        }
    )

    response = _engine(clone_scene=lambda: scene).qualify(_request([candidate]))

    assert response["results"][0]["reason"] == "gripper_static_collision"
    collision = response["results"][0]["pair_legality"]["checks"]["static_scene_collision"][
        "collisions"
    ][0]
    assert collision["body"] == "mount_plate"
    assert collision["obstacle"] == "table"
    assert (
        "table"
        in response["results"][0]["pair_legality"]["checks"]["static_scene_collision"][
            "evaluated_obstacle_ids"
        ]
    )


def test_symmetry_twin_does_not_consume_second_grasp_branch_slot():
    base = _candidate(0, score=3.0)
    twin = _candidate(1, score=2.0)
    twin["symmetry_parent_id"] = "c0"
    twin["qualification_stages"][0]["quat_xyzw"] = [1.0, 0.0, 0.0, 0.0]
    independent = _candidate(20, score=1.0)

    response = _engine().qualify(_request([base, twin, independent], purpose="grasp"))

    assert response["selected_candidate_ids"] == ["c0", "c20"]
    assert len(response["l5_attempts"]) == 3
    assert response["stop_reason"] == "complete_l5_pass_found"


def test_grasp_profile_publishes_one_l5_pose_only_after_search_exhaustion():
    response = _engine().qualify(_request([_candidate(0)], purpose="grasp"))

    assert response["results"][0]["verdict"] == "PASS"
    assert response["selected_candidate_ids"] == ["c0"]
    assert response["stop_reason"] == ("complete_l5_pass_found_single_branch_exhaustive_fallback")
    assert response["metrics"]["l5_pass_count"] == 1
    assert response["metrics"]["l5_joint_branch_pass_count"] == 1
    assert response["search_exhaustion"] == {
        "fast_wave_count_expected": 1,
        "fast_wave_count_completed": 1,
        "fast_pool_exhausted": True,
        "recovery_wave_count_expected": 0,
        "recovery_wave_count_completed": 0,
        "recovery_pool_exhausted": True,
        "preferred_grasp_branch_target": 2,
        "published_grasp_branch_count": 1,
        "redundancy_degraded": True,
    }


def test_grasp_profile_proves_farthest_second_beam_branch_after_pool_exhaustion():
    planned_states = []

    def compute_ik(target, seed, collision):
        del target, collision
        position = -0.2 if seed.get("seed_source") == "named_home" else 0.2
        return {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": [position]},
            "min_singular_value": 0.3,
        }

    def plan_only(target, start, timeout, attempts):
        del start, timeout, attempts
        state = target["qualification_goal_joint_state"]
        planned_states.append(state["positions"][0])
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": list(state["positions"])}],
            "end_joint_state": dict(state),
        }

    response = _engine(compute_ik=compute_ik, plan_only=plan_only).qualify(
        _request([_candidate(0)], purpose="grasp")
    )

    assert response["selected_candidate_ids"] == ["c0"]
    assert response["stop_reason"] == "complete_l5_pass_found_joint_branch_fallback"
    assert response["metrics"]["l5_pass_count"] == 1
    assert response["metrics"]["l5_joint_branch_pass_count"] == 2
    assert planned_states == [0.2, -0.2]
    assert [attempt["joint_branch_index"] for attempt in response["l5_attempts"]] == [0, 1]
    assert response["l5_attempts"][1]["joint_branch_normalized_distance"] >= 0.05
    assert {proof["joint_branch_index"] for proof in response["selected_joint_branches"]} == {0, 1}
    assert (
        len(
            {
                proof["selected_ik_joint_state_sha256"]
                for proof in response["selected_joint_branches"]
            }
        )
        == 2
    )


def test_grasp_branch_selection_preserves_screen_quality_if_plan_omits_margin():
    response = _engine().qualify(_request([_candidate(0), _candidate(1)], purpose="grasp"))

    assert response["selected_candidate_ids"] == ["c0", "c1"]
    assert response["stop_reason"] == "complete_l5_pass_found_joint_space_fallback"
    for result in response["results"]:
        assert all(isinstance(stage.get("joint_margin"), float) for stage in result["stages"])


def test_valid_pure_ik_never_calls_collision_aware_ik():
    collision_flags = []

    def ik(target, seed, collision):
        collision_flags.append(collision)
        return {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    response = _engine(compute_ik=ik).qualify(_request([_candidate(0)]))

    assert response["selected_candidate_ids"] == ["c0"]
    assert collision_flags == [False, False]
    stage = response["results"][0]["stages"][0]
    assert stage["collision_ik_called"] is False
    assert stage["collision_ik"] is True


def test_staged_virtual_transition_clones_scene_before_contact():
    candidate = _candidate(0, stages=2)
    candidate["qualification_stages"][0]["scene_transition"] = "virtual_attach"
    clones = []
    transitions = []

    def clone_scene():
        scene = {"revision": 4, "transitions": []}
        clones.append(scene)
        return scene

    def transition(scene, name, target):
        assert any(scene is clone for clone in clones)
        transitions.append((name, target["name"]))
        return {
            "ok": True,
            "planning_scene_diff": {"attached_objects": [{"id": "target"}]},
        }

    response = _engine(
        clone_scene=clone_scene,
        apply_scene_transition=transition,
    ).qualify(_request([candidate]))

    assert response["selected_candidate_ids"] == ["c0"]
    assert len(clones) == 2
    assert transitions == [
        ("virtual_attach", "c0_stage0"),
        ("virtual_attach", "c0_stage0"),
    ]
    assert response["results"][0]["stages"][0]["scene_transition"]["ok"] is True


def test_post_detach_open_hand_collision_is_rejected_before_l5() -> None:
    candidate = _candidate(0)
    candidate["qualification_stages"][0].update(
        {
            "scene_transition": "virtual_detach",
            "qualification_post_transition_gripper_state": "open",
        }
    )
    planned = []
    validity_requests = []

    def validity(state):
        validity_requests.append(dict(state))
        open_hand = state.get("qualification_gripper_state") == "open"
        return {
            "valid": not open_hand,
            "collision_pairs": (
                [["green_bin_wall_left", "robotiq_85_left_finger_tip_link"]] if open_hand else []
            ),
        }

    response = _engine(
        clone_scene=lambda: {"revision": 4, "transitions": []},
        apply_scene_transition=lambda scene, transition, target: {
            "ok": True,
            "planning_scene_diff": {"remove_attached_ids": ["target"]},
        },
        check_state_validity=validity,
        plan_only=lambda *args: planned.append(args),
    ).qualify(_request([candidate]))

    result = response["results"][0]
    assert result["verdict"] == "FAIL"
    assert result["reason"] == "post_transition_gripper_state_invalid"
    assert planned == []
    assert any(
        request.get("qualification_gripper_state") == "open"
        and request.get("qualification_scene_diff") == {"remove_attached_ids": ["target"]}
        for request in validity_requests
    )
    post_checks = result["stages"][0]["post_transition_gripper_state_checks"]
    assert post_checks[0]["collision_pairs"] == [
        ["green_bin_wall_left", "robotiq_85_left_finger_tip_link"]
    ]


def test_l5_rechecks_open_hand_at_actual_planned_endpoint() -> None:
    candidate = _candidate(0)
    candidate["qualification_stages"][0].update(
        {
            "scene_transition": "virtual_detach",
            "qualification_post_transition_gripper_state": "open",
        }
    )
    open_checks = 0

    def validity(state):
        nonlocal open_checks
        if state.get("qualification_gripper_state") != "open":
            return {"valid": True, "collision_pairs": []}
        open_checks += 1
        return {
            # Screen Beam endpoint passes, but the exact L5 trajectory endpoint
            # exposes a terminal collision and must still fail closed.
            "valid": open_checks % 2 == 1,
            "collision_pairs": ([] if open_checks % 2 == 1 else [["bin_wall", "right_finger_tip"]]),
        }

    response = _engine(
        clone_scene=lambda: {"revision": 4, "transitions": []},
        apply_scene_transition=lambda scene, transition, target: {
            "ok": True,
            "planning_scene_diff": {"remove_attached_ids": ["target"]},
        },
        check_state_validity=validity,
    ).qualify(_request([candidate]))

    result = response["results"][0]
    # The primary and fixed-seed recovery layers both prove their screen
    # endpoint and independently reject their actual L5 endpoint.
    assert open_checks == 4
    assert result["verdict"] == "FAIL"
    assert result["reason"] == "post_transition_gripper_state_invalid"
    l5_open = result["stages"][0]["post_transition_gripper_state_validity"]
    assert l5_open["requested_gripper_state"] == "open"
    assert l5_open["joint_state_sha256"] == _hash({"names": ["j1"], "positions": [0.2]})
    assert l5_open["valid"] is False
    assert l5_open["retry_count"] == 0
    assert l5_open["elapsed_s"] >= 0.0
    assert l5_open["collision_pairs"] == [["bin_wall", "right_finger_tip"]]


def test_grasp_contact_allows_only_request_local_target_touch_links():
    candidate = _candidate(0)
    candidate["qualification_stages"][0].update(
        {
            "grasp_stage": "contact",
            "scene_transition": "virtual_attach",
        }
    )
    expected = {"target_object": ["left_tip", "right_tip"]}
    validity_policies = []
    planning_policies = []

    def clone_scene():
        return {
            "revision": 4,
            "target_id": "target_object",
            "target_touch_links": ["right_tip", "left_tip"],
            "transitions": [],
        }

    def validity(state):
        validity_policies.append(state.get("qualification_allowed_collisions"))
        return {"valid": state.get("qualification_allowed_collisions") == expected}

    def plan_only(target, start, timeout, attempts):
        planning_policies.append(target.get("qualification_allowed_collisions"))
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": [0.2]}],
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    response = _engine(
        clone_scene=clone_scene,
        apply_scene_transition=lambda scene, transition, target: {"ok": True},
        check_state_validity=validity,
        plan_only=plan_only,
    ).qualify(_request([candidate], purpose="grasp"))

    assert response["selected_candidate_ids"] == ["c0"]
    assert validity_policies and all(policy == expected for policy in validity_policies)
    assert planning_policies == [expected]


def test_grasp_contact_rejects_static_collision_during_l5_close_sweep() -> None:
    candidate = _candidate(0)
    candidate["qualification_stages"][0].update(
        {
            "grasp_stage": "contact",
            "scene_transition": "virtual_attach",
            "qualification_terminal_gripper_state": "closing_sweep",
        }
    )
    planned = []
    sweep_requests = []

    def validity(state):
        if state.get("qualification_gripper_state") == "closing_sweep":
            sweep_requests.append(dict(state))
            return {
                "valid": False,
                "collision_pairs": [["robotiq_85_left_finger_tip_link", "work_table"]],
                "qualification_gripper_sweep_checks": [
                    {
                        "sample": "near_open",
                        "active_joint_position_rad": 0.05,
                        "valid": False,
                        "collision_pairs": [
                            [
                                "robotiq_85_left_finger_tip_link",
                                "work_table",
                            ]
                        ],
                    }
                ],
                "qualification_seed_independent_static_collision": True,
            }
        return {"valid": True, "collision_pairs": []}

    def plan_only(target, start, timeout, attempts):
        planned.append(target)
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": [0.2]}],
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    response = _engine(
        clone_scene=lambda: {
            "revision": 4,
            "target_id": "target_object",
            "target_touch_links": ["left_tip", "right_tip"],
            "transitions": [],
        },
        apply_scene_transition=lambda scene, transition, target: {"ok": True},
        check_state_validity=validity,
        plan_only=plan_only,
    ).qualify(_request([candidate], purpose="grasp"))

    result = response["results"][0]
    assert result["verdict"] == "FAIL"
    assert result["reason"] == "terminal_gripper_state_invalid"
    assert planned == []
    assert sweep_requests
    assert sweep_requests[0]["qualification_allowed_collisions"] == {
        "target_object": ["left_tip", "right_tip"]
    }
    terminal = result["stages"][0]["terminal_gripper_state_validity"]
    assert terminal["requested_gripper_state"] == "closing_sweep"
    assert terminal["preplan_endpoint_check"] is True
    assert terminal["seed_independent_static_collision"] is True
    assert terminal["collision_pairs"] == [["robotiq_85_left_finger_tip_link", "work_table"]]


def test_grasp_contact_rejects_seed_independent_one_sided_contact_geometry() -> None:
    candidate = _candidate(0)
    candidate["qualification_stages"][0].update(
        {
            "grasp_stage": "contact",
            "scene_transition": "virtual_attach",
            "qualification_terminal_gripper_state": "closing_sweep",
        }
    )
    planned = []

    def validity(state):
        if state.get("qualification_gripper_state") == "closing_sweep":
            return {
                "valid": False,
                "reason": "qualification_bilateral_target_contact_not_predicted",
                "collision_pairs": [["robotiq_85_left_finger_tip_link", "target_object"]],
                "qualification_gripper_sweep_checks": [
                    {
                        "sample": "close_2",
                        "active_joint_position_rad": 0.4,
                        "valid": True,
                        "collision_pairs": [
                            [
                                "robotiq_85_left_finger_tip_link",
                                "target_object",
                            ]
                        ],
                        "target_contact_links": ["robotiq_85_left_finger_tip_link"],
                        "bilateral_target_contact": False,
                    }
                ],
                "qualification_seed_independent_static_collision": False,
                "qualification_bilateral_target_contact_required": True,
                "qualification_bilateral_target_contact_predicted": False,
                "qualification_seed_independent_contact_geometry_failure": True,
            }
        return {"valid": True, "collision_pairs": []}

    def plan_only(target, start, timeout, attempts):
        planned.append(target)
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{"positions": [0.2]}],
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    response = _engine(
        clone_scene=lambda: {
            "revision": 4,
            "target_id": "target_object",
            "target_touch_links": ["left_tip", "right_tip"],
            "transitions": [],
        },
        apply_scene_transition=lambda scene, transition, target: {"ok": True},
        check_state_validity=validity,
        plan_only=plan_only,
    ).qualify(_request([candidate], purpose="grasp"))

    result = response["results"][0]
    assert result["verdict"] == "FAIL"
    assert result["reason"] == "terminal_gripper_state_invalid"
    assert planned == []
    terminal = result["stages"][0]["terminal_gripper_state_validity"]
    assert terminal["bilateral_target_contact_required"] is True
    assert terminal["bilateral_target_contact_predicted"] is False
    assert terminal["seed_independent_contact_geometry_failure"] is True
    assert terminal["reason"] == ("qualification_bilateral_target_contact_not_predicted")


def test_nonfinite_transform_is_hard_rejected_before_ik():
    candidate = _candidate(0)
    candidate["qualification_stages"][0]["xyz"][0] = float("nan")

    response = _engine().qualify(_request([candidate]))

    assert response["results"][0]["verdict"] == "FAIL"
    assert response["results"][0]["reason"] == "invalid_target_transform"


def test_colliding_pure_solution_gets_exactly_one_rescue_for_its_seed():
    collision_flags = []

    def ik(target, seed, collision):
        collision_flags.append(collision)
        return {
            "ok": True,
            "joint_state": {
                "names": ["j1"],
                "positions": [0.6 if collision else 0.2],
            },
        }

    response = _engine(
        compute_ik=ik,
        check_state_validity=lambda state: {
            "valid": state["positions"] == [0.6],
            "collision_pairs": [] if state["positions"] == [0.6] else [["table", "tool"]],
        },
    ).qualify(
        _request(
            [_candidate(0)],
            overrides={
                "beam_width": 1,
                "ik_seed_count": 1,
                "fast_seed_count": 1,
                "recovery_seed_count": 0,
            },
        )
    )

    assert response["selected_candidate_ids"] == ["c0"]
    assert collision_flags == [False, True]
    assert len(response["results"][0]["stages"][0]["collision_ik_attempts"]) == 1


def test_beam_two_propagates_parent_solutions_to_the_next_stage():
    second_stage_seeds = []

    def ik(target, seed, collision):
        assert collision is False
        if target["name"].endswith("stage1"):
            second_stage_seeds.append(list(seed["positions"]))
        return {
            "ok": True,
            "joint_state": {
                "names": ["j1"],
                "positions": [float(seed["positions"][0]) + 0.2],
            },
            "min_singular_value": 0.2,
        }

    response = _engine(compute_ik=ik).qualify(_request([_candidate(0, stages=2)]))

    assert response["selected_candidate_ids"] == ["c0"]
    assert second_stage_seeds == [[0.2], [1.0]]
    assert response["results"][0]["stages"][0]["beam_width"] == 2


def test_recovery_uses_six_fixed_seeds_only_at_first_chain_stage():
    by_stage = {"stage0": [], "stage1": []}

    def ik(target, seed, collision):
        del collision
        stage = "stage1" if target["name"].endswith("stage1") else "stage0"
        by_stage[stage].append(str(seed.get("seed_source") or ""))
        return {
            "ok": True,
            "joint_state": {
                "names": ["j1"],
                "positions": [float(seed["positions"][0]) * 0.5],
            },
        }

    plan_calls = 0

    def plan(target, start, timeout, attempts):
        nonlocal plan_calls
        plan_calls += 1
        return {
            "ok": plan_calls > 1,
            "execution_started": False,
            "trajectory_points": ([{}] if plan_calls > 1 else []),
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    response = _engine(compute_ik=ik, plan_only=plan).qualify(_request([_candidate(0, stages=2)]))

    assert response["selected_candidate_ids"] == ["c0"]
    assert sum(source.startswith("fixed_recovery") for source in by_stage["stage0"]) == 6
    assert all(not source.startswith("fixed_recovery") for source in by_stage["stage1"])
    # Two fast parents and two recovery parents: dependent stages never widen
    # to the six-seed recovery budget.
    assert len(by_stage["stage1"]) == 4


def test_initial_seed_uses_next_cache_state_when_nearest_is_duplicate():
    engine = _engine()
    start = {"names": ["j1"], "positions": [0.0]}

    seeds = engine._fast_stage_seeds(
        start,
        previous_beam=[],
        batch_cache=[
            {"names": ["j1"], "positions": [0.0]},
            {"names": ["j1"], "positions": [0.4]},
        ],
        current_state=start,
        source={"joint_limits": {"lower": [-1.0], "upper": [1.0]}},
        count=2,
        recovery=False,
        initial_seed_source="current_robot_state",
    )

    assert [seed["positions"] for seed in seeds] == [[0.0], [0.4]]
    assert [seed["seed_source"] for seed in seeds] == [
        "current_robot_state",
        "batch_cache",
    ]


def test_wave_barrier_makes_out_of_order_completion_deterministic():
    candidates = [_candidate(index, score=float(10 - index)) for index in range(6)]

    def ik(target, seed, collision):
        time.sleep((5 - int(target["name"].split("_")[0][1:])) * 0.001)
        return {
            "ok": True,
            "joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    first = _engine(compute_ik=ik).qualify(_request(candidates))
    second = _engine(compute_ik=ik).qualify(_request(candidates))

    assert first["selected_candidate_ids"] == second["selected_candidate_ids"] == ["c0"]
    assert [
        {key: value for key, value in attempt.items() if key != "elapsed_s"}
        for attempt in first["l5_attempts"]
    ] == [
        {key: value for key, value in attempt.items() if key != "elapsed_s"}
        for attempt in second["l5_attempts"]
    ]


def test_l5_failure_continues_to_next_quality_ranked_candidate():
    planned = []
    candidates = [
        _candidate(0, score=3.0),
        _candidate(1, score=2.0),
        _candidate(2, score=1.0),
    ]

    def plan(target, start, timeout, attempts):
        planned.append(target["name"])
        ok = not target["name"].startswith("c0_")
        return {
            "ok": ok,
            "execution_started": False,
            "trajectory_points": ([{}] if ok else []),
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    response = _engine(plan_only=plan).qualify(_request(candidates))

    assert planned == ["c0_stage0", "c1_stage0"]
    assert response["selected_candidate_ids"] == ["c1"]
    assert [attempt["verdict"] for attempt in response["l5_attempts"]] == [
        "FAIL",
        "PASS",
    ]
    assert all(attempt["elapsed_s"] >= 0.0 for attempt in response["l5_attempts"])
    assert all(
        stage["elapsed_s"] >= 0.0 for result in response["results"] for stage in result["stages"]
    )


def test_l5_plan_only_is_constrained_to_selected_beam_joint_branch():
    goals = []

    def plan(target, start, timeout, attempts):
        goals.append(target["qualification_goal_joint_state"])
        return {
            "ok": True,
            "execution_started": False,
            "trajectory_points": [{}],
            "end_joint_state": dict(target["qualification_goal_joint_state"]),
        }

    response = _engine(plan_only=plan).qualify(_request([_candidate(0)]))

    assert goals == [{"names": ["j1"], "positions": [0.2]}]
    assert response["results"][0]["stages"][0]["selected_ik_end_joint_state_sha256"]


def test_l5_failure_is_replanned_with_fixed_recovery_branch():
    plan_calls = 0

    def plan(target, start, timeout, attempts):
        nonlocal plan_calls
        plan_calls += 1
        ok = plan_calls == 2
        return {
            "ok": ok,
            "execution_started": False,
            "trajectory_points": ([{}] if ok else []),
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    response = _engine(plan_only=plan).qualify(_request([_candidate(0)]))

    assert response["selected_candidate_ids"] == ["c0"]
    assert plan_calls == 2
    assert [attempt["recovery_layer"] for attempt in response["l5_attempts"]] == [
        False,
        True,
    ]
    assert [
        attempt["recovery_layer"] for attempt in response["results"][0]["screening_attempts"]
    ] == [False, True]
    assert all(
        attempt["stages"][0]["pure_ik_attempts"]
        for attempt in response["results"][0]["screening_attempts"]
    )


def test_recovery_barrier_does_not_publish_unsubmitted_endpoint_as_pass():
    plan_calls = 0

    def plan(target, start, timeout, attempts):
        nonlocal plan_calls
        plan_calls += 1
        # Both candidates fail in the fast layer.  The first recovery L5
        # succeeds, so the second recovery screen is intentionally never sent
        # to L5 even though the wave barrier has already completed it.
        ok = plan_calls == 3
        return {
            "ok": ok,
            "execution_started": False,
            "trajectory_points": ([{}] if ok else []),
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    response = _engine(plan_only=plan).qualify(
        _request([_candidate(0, score=2.0), _candidate(1, score=1.0)])
    )

    assert response["selected_candidate_ids"] == ["c0"]
    assert plan_calls == 3
    deferred = next(item for item in response["results"] if item["candidate_id"] == "c1")
    assert deferred["verdict"] == "NOT_EVALUATED"
    assert deferred["reason"] == "l5_not_submitted_after_success"
    assert deferred["endpoint_pass"] is True
    assert deferred["full_plan_submitted"] is False


def test_pick_ik_uses_global_mode_only_for_recovery_and_restores_local():
    modes = []
    plan_calls = 0

    def set_mode(mode):
        modes.append(mode)
        return {"ok": True}

    def plan(target, start, timeout, attempts):
        nonlocal plan_calls
        plan_calls += 1
        ok = plan_calls == 2
        return {
            "ok": ok,
            "execution_started": False,
            "trajectory_points": ([{}] if ok else []),
            "end_joint_state": {"names": ["j1"], "positions": [0.2]},
        }

    request = _request([_candidate(0)], overrides={"solver_profile": "pick_ik_local"})
    response = _engine(
        plan_only=plan,
        set_solver_mode=set_mode,
    ).qualify(request)

    assert response["selected_candidate_ids"] == ["c0"]
    assert modes == ["local", "global", "local"]
    recovery_result = response["results"][0]
    assert recovery_result["recovery_layer"] is True
    assert recovery_result["stages"][0]["pure_ik_attempts"][0]["solver"] == ("pick_ik_global")


def test_recovery_seeds_start_only_after_complete_fast_pool_failure():
    sources = []
    lock = threading.Lock()

    def ik(target, seed, collision):
        with lock:
            sources.append(str(seed.get("seed_source") or ""))
        return {"ok": False}

    response = _engine(compute_ik=ik).qualify(_request([_candidate(0), _candidate(1)]))

    first_recovery = next(
        index for index, source in enumerate(sources) if source.startswith("fixed_recovery")
    )
    assert first_recovery == 4
    assert all(source.startswith("fixed_recovery") for source in sources[first_recovery:])
    assert len(sources[first_recovery:]) == 12
    assert response["stop_reason"] == "candidate_and_recovery_exhausted"


def test_frozen_pair_can_defer_fixed_recovery_after_complete_fast_pool():
    sources = []
    lock = threading.Lock()

    def ik(target, seed, collision):
        with lock:
            sources.append(str(seed.get("seed_source") or ""))
        return {"ok": False}

    request = _request(
        [_candidate(0), _candidate(1)],
        overrides={
            "qualification_mode": "frozen_pair",
            "defer_recovery": True,
            "l5_pass_target": 1,
            "l5_min_pass_target": 1,
        },
    )
    response = _engine(compute_ik=ik).qualify(request)

    assert len(sources) == 4
    assert not any(source.startswith("fixed_recovery") for source in sources)
    assert response["selected_candidate_ids"] == []
    assert response["stop_reason"] == "fast_pool_exhausted_recovery_deferred"
    assert response["search_exhaustion"]["fast_pool_exhausted"] is True
    assert response["search_exhaustion"]["recovery_deferred"] is True
    assert response["search_exhaustion"]["recovery_pool_exhausted"] is False


def test_repeated_service_timeout_aborts_as_infrastructure_error():
    calls = 0

    def ik(target, seed, collision):
        nonlocal calls
        calls += 1
        raise TimeoutError("service unavailable")

    response = _engine(compute_ik=ik).qualify(_request([_candidate(0)]))

    assert calls == 2
    assert response["stop_reason"] == "infrastructure_error"
    assert response["results"][0]["verdict"] == "UNKNOWN"
    assert response["results"][0]["reason"] == "qualification_service_error"


def test_repeated_malformed_l5_evidence_aborts_as_infrastructure_error():
    calls = 0

    def plan(target, start, timeout, attempts):
        nonlocal calls
        calls += 1
        return {
            "ok": True,
            "execution_started": None,
            "trajectory_points": [],
        }

    response = _engine(plan_only=plan).qualify(_request([_candidate(0)]))

    assert calls == 2
    assert response["stop_reason"] == "infrastructure_error"
    assert response["results"][0]["verdict"] == "UNKNOWN"
    assert response["results"][0]["infrastructure_error"] is True


def test_missing_runtime_jacobian_is_a_configuration_error_not_unreachable():
    response = _engine(
        current_joint_state=lambda: {
            "names": ["j1"],
            "positions": [0.0],
            "joint_limits": {"lower": [-1.0], "upper": [1.0]},
            "jacobian_quality_available": False,
            "jacobian_quality_error": "model parse failed",
        }
    ).qualify(_request([_candidate(0)]))

    assert response["stop_reason"] == "configuration_error"
    assert response["infrastructure_error"] is True
    assert response["results"][0]["verdict"] == "UNKNOWN"
    assert response["results"][0]["reason"] == "jacobian_quality_unavailable"


def test_shadow_keeps_legacy_diversity_subset_authoritative():
    request = _request([_candidate(0), _candidate(1)])
    request["funnel"]["qualification_profile"] = "shadow"
    request["funnel"]["shadow_legacy_candidate_ids"] = ["c1"]

    response = _engine().qualify(request)

    assert response["qualification_profile"] == "shadow"
    assert [item["verdict"] for item in response["results"]] == [
        "NOT_EVALUATED",
        "PASS",
    ]
    assert response["results"][0]["reason"] == "shadow_legacy_diversity_not_selected"
    assert len(response["shadow_fast_v3"]["results"]) == 2
    assert response["shadow_fast_v3"]["artifact_schema_version"] == (
        "openeta.moveit_candidate_qualification.v3"
    )
    assert response["shadow_fast_v3"]["robot_model_sha256"]
    assert "capability_map_id" in response["shadow_fast_v3"]


def test_fast_profile_preserves_exact_duplicate_anyplace_results():
    captured = {}
    candidates = [_candidate(0), _candidate(1)]
    for candidate in candidates:
        candidate["object_goal_pose"] = {
            "translation_xyz": [0.4, 0.0, 0.5],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        }

    def rpc(_name, request, _timeout):
        captured.update(request)
        return _engine().qualify(request)

    MoveItCandidateQualifier(
        rpc,
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
    ).qualify_result(
        ToolResult(True, "ok", {"placement_candidates": candidates}),
        purpose="placement",
        scene_epoch=1,
        planning_scene_revision=4,
    )

    assert [item["candidate_id"] for item in captured["candidates"]] == [
        "c0",
        "c1",
    ]


def test_fast_grasp_lookahead_expands_l5_capacity_without_model_rerun():
    captured = {}
    candidates = [_candidate(index) for index in range(5)]
    for index, candidate in enumerate(candidates):
        candidate["qualification_stages"][0]["xyz"][0] = 0.4 + (index % 3) * 0.02

    def rpc(_name, request, _timeout):
        captured.update(request)
        return _engine().qualify(request)

    result = MoveItCandidateQualifier(
        rpc,
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
    ).qualify_result(
        ToolResult(True, "one frozen model output", {"grasp_candidates": candidates}),
        purpose="grasp",
        scene_epoch=1,
        planning_scene_revision=4,
        l5_pass_target=4,
    )

    assert captured["funnel"]["full_plan_limit"] == 4
    assert captured["funnel"]["endpoint_pass_target"] == 4
    assert captured["funnel"]["l5_pass_target"] == 4
    assert result.details["candidate_count"] == 4
    assert len(result.details["grasp_candidates"]) == 4
    assert len({item["id"] for item in result.details["grasp_candidates"]}) == 4
    assert result.details["ranking"] == "moveit_physical_quality"
    assert [
        item["moveit_physical_quality_rank"] for item in result.details["grasp_candidates"]
    ] == [0, 1, 2, 3]
    assert all(item["moveit_l5_qualified"] is True for item in result.details["grasp_candidates"])


def test_shadow_preserves_complete_fast_pool_but_legacy_uses_old_dedup_subset():
    captured = {}
    candidates = [_candidate(0), _candidate(1)]
    for candidate in candidates:
        candidate["object_goal_pose"] = {
            "translation_xyz": [0.4, 0.0, 0.5],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        }

    def rpc(_name, request, _timeout):
        captured.update(request)
        return _engine().qualify(request)

    MoveItCandidateQualifier(
        rpc,
        qualification_profile="shadow",
        solver_profile="kdl_fast",
    ).qualify_result(
        ToolResult(True, "ok", {"placement_candidates": candidates}),
        purpose="placement",
        scene_epoch=1,
        planning_scene_revision=4,
    )

    assert len(captured["candidates"]) == 2
    assert captured["funnel"]["shadow_legacy_candidate_ids"] == ["c0"]


def test_fast_qualifier_writes_v3_artifact_and_selected_candidate(tmp_path: Path):
    engine = _engine()
    qualifier = MoveItCandidateQualifier(
        lambda _name, request, _timeout: engine.qualify(request),
        artifact_root=tmp_path,
        compile_candidate=lambda candidate, *_args: {
            "qualification_stages": candidate["qualification_stages"]
        },
        qualification_profile="fast_v3",
        solver_profile="kdl_fast",
    )

    result = qualifier.qualify_result(
        ToolResult(True, "ok", {"placement_candidates": [_candidate(0)]}),
        purpose="placement",
        scene_epoch=1,
        planning_scene_revision=4,
    )

    assert [item["id"] for item in result.details["placement_candidates"]] == ["c0"]
    artifact = json.loads(Path(result.details["qualification_artifact"]["path"]).read_text())
    assert artifact["schema_version"] == QUALIFICATION_SCHEMA_V3
    assert artifact["qualification_profile"] == "fast_v3"
    assert artifact["selected_candidate_ids"] == ["c0"]
    assert artifact["waves"]
    assert artifact["l5_attempts"]
    assert artifact["legality_screening"]["goal_legality_unique_count"] == 1
    assert artifact["legality_screening"]["pair_legality_evaluation_count"] == 1
    assert artifact["stop_reason"] == "complete_l5_pass_found_minimum_lookahead"
    assert artifact["authoritative_scene_sha256"] == "a" * 64
    assert artifact["moveit_world_geometry_sha256"] == "b" * 64
    assert artifact["moveit_attached_geometry_sha256"] == "c" * 64
    assert artifact["moveit_geometry_verified_ids"] == ["table", "target"]
