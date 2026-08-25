from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from agent.runtime.moveit_qualification import (
    QUALIFICATION_SCHEMA_V3,
    MoveItCandidateQualifier,
    MoveItQualificationEngine,
    SAME_RUN_QUALIFICATION_SEED_FIELD,
    SAME_RUN_QUALIFICATION_SEED_PROVENANCE,
    SAME_RUN_QUALIFICATION_SEED_SCHEMA,
)
from agent.runtime.qualification_v3 import schedule_candidate_waves
from agent.tools.registry import ToolResult


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
        "grasp_waves": [16, 32, 64],
        "placement_waves": [12, 24, 48, 96],
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


def test_frozen_pair_l5_order_prefers_distinct_grasp_and_goal_cluster():
    candidates = []
    for index, (grasp_id, score) in enumerate(
        (("g0", 4.0), ("g0", 3.0), ("g1", 2.0), ("g1", 1.0))
    ):
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

    assert [wave.cumulative_per_branch for wave in waves] == [12, 24, 48, 96]
    assert [len(wave.candidates) for wave in waves] == [24, 24, 48, 96]
    assert sum(len(wave.candidates) for wave in waves) == 192
    assert [
        wave.candidates[0]["candidate"]["source_grasp_id"] for wave in waves
    ] == ["g0"] * 4
    assert [
        waves[0].candidates[0]["candidate"]["source_grasp_id"],
        waves[0].candidates[1]["candidate"]["source_grasp_id"],
    ] == ["g0", "g1"]


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
        item["pair_legality"]["reason"] == "goal_legality_rejected"
        for item in response["results"]
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
    binding = response["results"][0]["goal_legality"]["checks"][
        "object_frame_binding"
    ]
    assert binding["pointcloud_goal_translation_xyz"][2] == 0.50
    assert binding["collision_goal_translation_xyz"] == pytest.approx(
        [0.48, -0.1, 0.43]
    )
    assert binding["collision_goal_pose"]["translation_xyz"] == pytest.approx(
        [0.48, -0.1, 0.43]
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
    binding = response["results"][0]["goal_legality"]["checks"][
        "object_frame_binding"
    ]
    assert binding["method"] == "direct_physical_object_goal"
    assert binding["target_is_attached"] is True
    assert binding["collision_goal_translation_xyz"] == pytest.approx(
        [0.48, -0.1, 0.43]
    )


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
    assert response["results"][0]["goal_legality"]["checks"]["support"][
        "tolerance_basis"
    ] == "sensor_and_model_support_contact_uncertainty"
    assert response["results"][1]["reason"] == "goal_support_surface_penetration"


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
                }
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
    assert response["results"][0]["pair_legality"]["checks"]["stage_se3"][
        "stage_count"
    ] == 1


def test_object_goal_static_collision_is_rejected_by_target_gate():
    scene = _placement_scene()
    scene["world_specs"]["distractor"] = {
        "id": "distractor",
        "shape": "box",
        "size_xyz": [0.05, 0.05, 0.08],
        "pose_xyz": [0.28, 0.12, 0.44],
        "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
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
    collisions = response["results"][0]["goal_legality"]["checks"][
        "static_scene_collision"
    ]["collision_ids"]
    assert collisions == ["distractor"]
    assert response["metrics"]["screening_attempt_count"] == 0


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
    collision = response["results"][0]["pair_legality"]["checks"][
        "static_scene_collision"
    ]["collisions"][0]
    assert collision["body"] == "mount_plate"
    assert collision["obstacle"] == "table"


def test_symmetry_twin_does_not_consume_second_grasp_branch_slot():
    base = _candidate(0, score=3.0)
    twin = _candidate(1, score=2.0)
    twin["symmetry_parent_id"] = "c0"
    twin["qualification_stages"][0]["quat_xyzw"] = [1.0, 0.0, 0.0, 0.0]
    independent = _candidate(20, score=1.0)

    response = _engine().qualify(
        _request([base, twin, independent], purpose="grasp")
    )

    assert response["selected_candidate_ids"] == ["c0", "c20"]
    assert len(response["l5_attempts"]) == 3
    assert response["stop_reason"] == "complete_l5_pass_found"


def test_grasp_profile_does_not_publish_only_one_qualified_branch():
    response = _engine().qualify(_request([_candidate(0)], purpose="grasp"))

    assert response["results"][0]["verdict"] == "PASS"
    assert response["selected_candidate_ids"] == []
    assert response["stop_reason"] == "candidate_and_recovery_exhausted"


def test_grasp_branch_selection_preserves_screen_quality_if_plan_omits_margin():
    response = _engine().qualify(
        _request([_candidate(0), _candidate(1)], purpose="grasp")
    )

    assert response["selected_candidate_ids"] == ["c0", "c1"]
    assert response["stop_reason"] == "complete_l5_pass_found_joint_space_fallback"
    for result in response["results"]:
        assert all(
            isinstance(stage.get("joint_margin"), float)
            for stage in result["stages"]
        )


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

    response = _engine(compute_ik=ik).qualify(
        _request([_candidate(0, stages=2)])
    )

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

    response = _engine(compute_ik=ik, plan_only=plan).qualify(
        _request([_candidate(0, stages=2)])
    )

    assert response["selected_candidate_ids"] == ["c0"]
    assert sum(source.startswith("fixed_recovery") for source in by_stage["stage0"]) == 6
    assert all(
        not source.startswith("fixed_recovery") for source in by_stage["stage1"]
    )
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
        stage["elapsed_s"] >= 0.0
        for result in response["results"]
        for stage in result["stages"]
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
    assert response["results"][0]["stages"][0][
        "selected_ik_end_joint_state_sha256"
    ]


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
        attempt["recovery_layer"]
        for attempt in response["results"][0]["screening_attempts"]
    ] == [False, True]
    assert all(
        attempt["stages"][0]["pure_ik_attempts"]
        for attempt in response["results"][0]["screening_attempts"]
    )


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

    request = _request(
        [_candidate(0)], overrides={"solver_profile": "pick_ik_local"}
    )
    response = _engine(
        plan_only=plan,
        set_solver_mode=set_mode,
    ).qualify(request)

    assert response["selected_candidate_ids"] == ["c0"]
    assert modes == ["local", "global", "local"]
    recovery_result = response["results"][0]
    assert recovery_result["recovery_layer"] is True
    assert recovery_result["stages"][0]["pure_ik_attempts"][0]["solver"] == (
        "pick_ik_global"
    )


def test_recovery_seeds_start_only_after_complete_fast_pool_failure():
    sources = []
    lock = threading.Lock()

    def ik(target, seed, collision):
        with lock:
            sources.append(str(seed.get("seed_source") or ""))
        return {"ok": False}

    response = _engine(compute_ik=ik).qualify(
        _request([_candidate(0), _candidate(1)])
    )

    first_recovery = next(
        index for index, source in enumerate(sources) if source.startswith("fixed_recovery")
    )
    assert first_recovery == 4
    assert all(source.startswith("fixed_recovery") for source in sources[first_recovery:])
    assert len(sources[first_recovery:]) == 12
    assert response["stop_reason"] == "candidate_and_recovery_exhausted"


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
    artifact = json.loads(
        Path(result.details["qualification_artifact"]["path"]).read_text()
    )
    assert artifact["schema_version"] == QUALIFICATION_SCHEMA_V3
    assert artifact["qualification_profile"] == "fast_v3"
    assert artifact["selected_candidate_ids"] == ["c0"]
    assert artifact["waves"]
    assert artifact["l5_attempts"]
    assert artifact["legality_screening"]["goal_legality_unique_count"] == 1
    assert artifact["legality_screening"]["pair_legality_evaluation_count"] == 1
    assert artifact["stop_reason"] == "complete_l5_pass_found"
