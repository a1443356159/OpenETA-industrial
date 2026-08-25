from __future__ import annotations

import json

from agent.runtime.qualification_bakeoff import (
    evaluate_solver_bakeoff,
    gpu_upgrade_required,
    read_qualification_artifacts,
    standard_bakeoff_matrix,
)


def _artifact(configuration, repetition, latency, *, candidate="a04", case="normal"):
    return {
        "schema_version": "openeta.moveit_candidate_funnel.v3",
        "solver_configuration_id": configuration,
        "case_id": case,
        "repetition": repetition,
        "robot_model_sha256": "robot",
        "first_l5_pass_s": latency,
        "stop_reason": "complete_l5_pass_found",
        "selected_candidate_ids": [candidate],
        "results": [
            {
                "candidate_id": candidate,
                "verdict": "PASS",
                "reason": "qualified",
                "stages": [
                    {
                        "end_joint_state": {
                            "names": ["j1"],
                            "positions": [0.2],
                        }
                    }
                ],
            }
        ],
    }


def test_bakeoff_requires_recall_and_uses_kdl_within_five_percent_tie():
    artifacts = []
    for repetition in range(10):
        artifacts.extend(
            [
                _artifact("kdl_legacy", repetition, 80.0),
                _artifact("kdl_fast@50ms", repetition, 50.0),
                _artifact("trac_ik_speed@50ms", repetition, 48.0),
                _artifact("pick_ik_local@50ms", repetition, 45.0, candidate="other"),
            ]
        )

    selection = evaluate_solver_bakeoff(artifacts)

    assert selection.selected_configuration == "kdl_fast@50ms"
    assert selection.report["configurations"]["pick_ik_local@50ms"]["eligible"] is False


def test_bakeoff_rejects_nondeterministic_failure_classification():
    artifacts = [_artifact("kdl_legacy", repetition, 80.0) for repetition in range(10)]
    candidate = [_artifact("trac_ik_speed", repetition, 40.0) for repetition in range(10)]
    candidate[-1]["results"][0]["reason"] = "different"
    artifacts.extend(candidate)

    selection = evaluate_solver_bakeoff(artifacts)

    assert selection.report["configurations"]["trac_ik_speed"]["gates"][
        "deterministic_repetitions"
    ] is False


def test_bakeoff_rejects_configuration_without_first_pass_latency():
    artifacts = [_artifact("kdl_legacy", repetition, 80.0) for repetition in range(10)]
    candidate = [_artifact("trac_ik_speed", repetition, 40.0) for repetition in range(10)]
    for run in candidate:
        run.pop("first_l5_pass_s")
        run["selected_candidate_ids"] = []
        run["results"][0]["verdict"] = "FAIL"
    artifacts.extend(candidate)

    selection = evaluate_solver_bakeoff(artifacts)

    assert selection.report["configurations"]["trac_ik_speed"]["gates"][
        "first_l5_pass_latency_available"
    ] is False


def test_bakeoff_rejects_configuration_that_omits_a_legacy_failure_case():
    artifacts = []
    for repetition in range(10):
        artifacts.append(_artifact("kdl_legacy", repetition, 80.0))
        hard = _artifact(
            "kdl_legacy",
            repetition,
            80.0,
            candidate="none",
            case="hard-no-pass",
        )
        hard["first_l5_pass_s"] = None
        hard["selected_candidate_ids"] = []
        hard["stop_reason"] = "candidate_and_recovery_exhausted"
        hard["results"][0].update(
            {"verdict": "FAIL", "reason": "kinematic_ik_failed"}
        )
        artifacts.append(hard)
        artifacts.append(_artifact("kdl_fast@50ms", repetition, 40.0))

    selection = evaluate_solver_bakeoff(artifacts)

    assert selection.report["configurations"]["kdl_fast@50ms"]["gates"][
        "complete_case_coverage"
    ] is False


def test_gpu_upgrade_gate_requires_30_cold_and_warm_runs():
    assert gpu_upgrade_required(
        [61.0] * 30,
        [50.0] * 30,
        pass_recall=1.0,
        legacy_pass_recall=1.0,
    )
    assert not gpu_upgrade_required(
        [50.0] * 30,
        [40.0] * 30,
        pass_recall=1.0,
        legacy_pass_recall=1.0,
    )
    assert gpu_upgrade_required(
        [61.0, 61.0, *([50.0] * 28)],
        [40.0] * 30,
        pass_recall=1.0,
        legacy_pass_recall=1.0,
    )


def test_standard_bakeoff_matrix_has_baseline_and_all_timeout_profiles():
    matrix = standard_bakeoff_matrix()

    assert len(matrix) == 17
    assert matrix[0]["solver_configuration_id"] == "kdl_legacy"
    assert {
        row["solver_configuration_id"] for row in matrix[1:]
    } == {
        f"{solver}@{timeout}ms/c8"
        for solver in (
            "kdl_fast",
            "trac_ik_speed",
            "trac_ik_distance",
            "pick_ik_local",
        )
        for timeout in (20, 50, 100, 200)
    }


def test_replay_reader_accepts_legacy_schemas_and_filters_model_hash(tmp_path):
    for index, schema in enumerate(
        (
            "openeta.moveit_candidate_qualification.v1",
            "openeta.moveit_candidate_funnel.v2",
            "openeta.moveit_candidate_qualification.v3",
        )
    ):
        (tmp_path / f"artifact-{index}.json").write_text(
            json.dumps(
                {
                    "schema_version": schema,
                    "robot_model_sha256": "active" if index < 2 else "old",
                    "results": [],
                }
            ),
            encoding="utf-8",
        )

    artifacts = read_qualification_artifacts(
        [tmp_path], robot_model_sha256="active"
    )

    assert [artifact["schema_version"] for artifact in artifacts] == [
        "openeta.moveit_candidate_qualification.v1",
        "openeta.moveit_candidate_funnel.v2",
    ]


def test_replay_reader_expands_shadow_fast_evidence(tmp_path):
    artifact = _artifact("kdl_legacy", 0, 80.0)
    artifact["shadow_fast_v3"] = {
        **_artifact("kdl_fast@50ms/c8", 0, 40.0),
        "solver_profile": "kdl_fast",
        "solver_configuration_id": "kdl_fast@50ms/c8",
    }
    path = tmp_path / "shadow.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    loaded = read_qualification_artifacts([path])

    assert len(loaded) == 2
    assert loaded[1]["qualification_profile"] == "shadow_fast_v3"
    assert loaded[1]["solver_configuration_id"] == "kdl_fast@50ms/c8"
