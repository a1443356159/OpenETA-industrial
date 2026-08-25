"""Replay-based CPU IK solver bake-off and deterministic promotion gates."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict


SUPPORTED_ARTIFACT_SCHEMAS = {
    "openeta.moveit_candidate_qualification.v1",
    "openeta.moveit_candidate_funnel.v2",
    "openeta.moveit_candidate_funnel.v3",
    "openeta.moveit_candidate_qualification.v3",
}
SOLVER_TIE_PRIORITY = ("kdl", "trac_ik", "pick_ik")
STANDARD_FAST_SOLVERS = (
    "kdl_fast",
    "trac_ik_speed",
    "trac_ik_distance",
    "pick_ik_local",
)
STANDARD_FAST_TIMEOUTS_MS = (20, 50, 100, 200)


def standard_bakeoff_matrix() -> list[JsonDict]:
    """Return the frozen CPU comparison matrix used by replay/shadow jobs."""

    rows: list[JsonDict] = [
        {
            "solver_configuration_id": "kdl_legacy",
            "qualification_profile": "legacy",
            "solver_profile": "kdl_legacy",
            "legacy_ik_timeout_ms": 200,
            "ik_seed_count": 8,
            "max_ik_concurrency": 1,
            "determinism_repetitions": 10,
        }
    ]
    rows.extend(
        {
            "solver_configuration_id": f"{solver}@{timeout_ms}ms/c8",
            "qualification_profile": "fast_v3",
            "solver_profile": solver,
            "fast_ik_timeout_ms": timeout_ms,
            "recovery_ik_timeout_ms": 200,
            "fast_seed_count": 2,
            "recovery_seed_count": 6,
            "max_ik_concurrency": 8,
            "max_state_validity_concurrency": 8,
            "determinism_repetitions": 10,
        }
        for solver in STANDARD_FAST_SOLVERS
        for timeout_ms in STANDARD_FAST_TIMEOUTS_MS
    )
    return rows


def read_qualification_artifacts(
    roots: Sequence[str | Path],
    *,
    robot_model_sha256: str = "",
) -> list[JsonDict]:
    """Read old and v3 artifacts; optionally retain only the active model hash."""

    paths: list[Path] = []
    for raw in roots:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            paths.append(path)
    artifacts: list[JsonDict] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        schema = str(
            payload.get("artifact_schema_version")
            or payload.get("schema_version")
            or ""
        )
        if schema not in SUPPORTED_ARTIFACT_SCHEMAS:
            continue
        if not isinstance(payload.get("results"), list):
            continue
        artifact_hash = str(payload.get("robot_model_sha256") or "")
        if robot_model_sha256 and artifact_hash != robot_model_sha256:
            continue
        payload["_artifact_path"] = str(path)
        artifacts.append(payload)
        shadow = payload.get("shadow_fast_v3")
        if isinstance(shadow, Mapping) and isinstance(shadow.get("results"), list):
            expanded = dict(payload)
            expanded.update(dict(shadow))
            expanded["qualification_profile"] = "shadow_fast_v3"
            expanded["_artifact_path"] = f"{path}#shadow_fast_v3"
            artifacts.append(expanded)
    return artifacts


def _configuration_id(artifact: Mapping[str, Any]) -> str:
    explicit = artifact.get("solver_configuration_id")
    if explicit:
        return str(explicit)
    solver = str(artifact.get("solver_profile") or "unknown")
    funnel = artifact.get("funnel")
    funnel = funnel if isinstance(funnel, Mapping) else {}
    timeout = funnel.get("fast_ik_timeout_s")
    concurrency = funnel.get("max_ik_concurrency")
    return f"{solver}@{timeout or 'default'}s/c{concurrency or 'default'}"


def _case_id(artifact: Mapping[str, Any]) -> str:
    return str(
        artifact.get("case_id")
        or artifact.get("scenario_id")
        or artifact.get("qualification_case_sha256")
        or artifact.get("qualification_binding_sha256")
        or artifact.get("_artifact_path")
        or "unknown"
    )


def _passed_candidate_ids(artifact: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("candidate_id") or "")
        for item in artifact.get("results", [])
        if isinstance(item, Mapping)
        and item.get("verdict") == "PASS"
        and str(item.get("candidate_id") or "")
    }


def _run_signature(artifact: Mapping[str, Any]) -> str:
    results = artifact.get("results")
    results = results if isinstance(results, list) else []
    selected = list(artifact.get("selected_candidate_ids") or [])
    branches = []
    classifications = []
    for item in results:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("candidate_id") or "") in selected:
            stages = item.get("stages")
            branches.append(
                [
                    stage.get("end_joint_state")
                    for stage in stages or []
                    if isinstance(stage, Mapping)
                ]
            )
        classifications.append(
            [
                str(item.get("candidate_id") or ""),
                str(item.get("verdict") or "UNKNOWN"),
                str(item.get("reason") or ""),
            ]
        )
    return json.dumps(
        {
            "selected": selected,
            "branches": branches,
            "classifications": classifications,
            "stop_reason": artifact.get("stop_reason"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _first_pass_latency(artifact: Mapping[str, Any]) -> float | None:
    explicit = artifact.get("first_l5_pass_s")
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
        value = float(explicit)
        return value if math.isfinite(value) and value >= 0.0 else None
    metrics = artifact.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    value = metrics.get("first_l5_pass_s")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed >= 0.0 else None
    if artifact.get("selected_candidate_ids"):
        total = metrics.get("total_elapsed_s")
        if isinstance(total, (int, float)) and not isinstance(total, bool):
            parsed = float(total)
            return parsed if math.isfinite(parsed) and parsed >= 0.0 else None
    return None


def _percentile95(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.inf
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _solver_family(configuration_id: str) -> str:
    lowered = configuration_id.lower()
    if "kdl" in lowered:
        return "kdl"
    if "trac" in lowered:
        return "trac_ik"
    if "pick" in lowered:
        return "pick_ik"
    return "other"


@dataclass(frozen=True, slots=True)
class BakeoffSelection:
    selected_configuration: str | None
    report: JsonDict


def evaluate_solver_bakeoff(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    legacy_configuration: str = "kdl_legacy",
    determinism_repetitions: int = 10,
    tie_fraction: float = 0.05,
    required_configurations: Sequence[str] = (),
) -> BakeoffSelection:
    """Apply recall, ten-run determinism, latency, and dependency tie gates."""

    by_configuration: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        by_configuration[_configuration_id(artifact)].append(artifact)
        by_case[_case_id(artifact)].append(artifact)
    required_cases = set(by_case)
    known_by_case: dict[str, set[str]] = {}
    for case_id, runs in by_case.items():
        known: set[str] = set()
        for run in runs:
            configuration = _configuration_id(run)
            if (
                run.get("long_budget") is True
                or configuration == legacy_configuration
                or configuration.startswith(f"{legacy_configuration}@")
            ):
                known.update(_passed_candidate_ids(run))
        known_by_case[case_id] = known

    def task_recall(runs: Sequence[Mapping[str, Any]]) -> float:
        cases: dict[str, bool] = {case: False for case in required_cases}
        for run in runs:
            case = _case_id(run)
            cases[case] = cases.get(case, False) or bool(_passed_candidate_ids(run))
        return sum(cases.values()) / len(cases) if cases else 0.0

    legacy_runs = [
        run
        for config, runs in by_configuration.items()
        if config == legacy_configuration or config.startswith(f"{legacy_configuration}@")
        for run in runs
    ]
    legacy_recall = task_recall(legacy_runs)
    rows: dict[str, JsonDict] = {}
    eligible: list[str] = []
    for configuration, runs in sorted(by_configuration.items()):
        successes_by_case: dict[str, set[str]] = defaultdict(set)
        signatures_by_case: dict[str, set[str]] = defaultdict(set)
        run_count_by_case: dict[str, int] = defaultdict(int)
        latencies: list[float] = []
        infrastructure_errors = 0
        for run in runs:
            case = _case_id(run)
            successes_by_case[case].update(_passed_candidate_ids(run))
            signatures_by_case[case].add(_run_signature(run))
            run_count_by_case[case] += 1
            latency = _first_pass_latency(run)
            if latency is not None:
                latencies.append(latency)
            infrastructure_errors += int(
                run.get("infrastructure_error") is True
                or run.get("stop_reason") == "infrastructure_error"
            )
        known_pass_retained = all(
            known.issubset(successes_by_case.get(case, set()))
            for case, known in known_by_case.items()
        )
        deterministic = bool(run_count_by_case) and all(
            count >= determinism_repetitions
            and len(signatures_by_case[case]) == 1
            for case, count in run_count_by_case.items()
        )
        recall = task_recall(runs)
        first_pass_p95 = _percentile95(latencies) if latencies else None
        gates = {
            "complete_case_coverage": set(run_count_by_case) == required_cases,
            "known_l5_pass_recall_100pct": known_pass_retained,
            "task_recall_not_below_legacy": recall >= legacy_recall,
            "deterministic_repetitions": deterministic,
            "no_infrastructure_errors": infrastructure_errors == 0,
            "first_l5_pass_latency_available": bool(latencies),
        }
        row = {
            "configuration": configuration,
            "solver_family": _solver_family(configuration),
            "run_count": len(runs),
            "case_count": len(run_count_by_case),
            "task_recall": recall,
            "legacy_task_recall": legacy_recall,
            "known_pass_count": sum(len(value) for value in known_by_case.values()),
            "first_l5_pass_p95_s": first_pass_p95,
            "infrastructure_error_count": infrastructure_errors,
            "gates": gates,
            "eligible": all(gates.values()),
        }
        rows[configuration] = row
        if row["eligible"]:
            eligible.append(configuration)

    missing_configurations = sorted(
        set(required_configurations).difference(by_configuration)
    )
    selected: str | None = None
    if eligible and not missing_configurations:
        fastest = min(float(rows[name]["first_l5_pass_p95_s"]) for name in eligible)
        near_tie = [
            name
            for name in eligible
            if float(rows[name]["first_l5_pass_p95_s"])
            <= fastest * (1.0 + tie_fraction)
        ]
        priority = {family: index for index, family in enumerate(SOLVER_TIE_PRIORITY)}
        selected = min(
            near_tie,
            key=lambda name: (
                priority.get(str(rows[name]["solver_family"]), len(priority)),
                float(rows[name]["first_l5_pass_p95_s"]),
                name,
            ),
        )
    report = {
        "schema_version": "openeta.qualification_solver_bakeoff.v1",
        "legacy_configuration": legacy_configuration,
        "determinism_repetitions": determinism_repetitions,
        "known_pass_cases": {
            case: sorted(values) for case, values in sorted(known_by_case.items())
        },
        "required_cases": sorted(required_cases),
        "configurations": rows,
        "required_configurations": list(required_configurations),
        "missing_configurations": missing_configurations,
        "selected_configuration": selected,
        "selection_reason": (
            "lowest_eligible_p95_with_5pct_dependency_tie_break"
            if selected
            else "required_configuration_evidence_missing"
            if missing_configurations
            else "no_configuration_passed_all_gates"
        ),
    }
    return BakeoffSelection(selected, report)


def gpu_upgrade_required(
    cold_first_pass_seconds: Sequence[float],
    warm_first_pass_seconds: Sequence[float],
    *,
    pass_recall: float,
    legacy_pass_recall: float,
    threshold_s: float = 60.0,
    required_runs_each: int = 30,
) -> bool:
    """Apply the post-shadow CPU gate before authorizing a cuRobo sidecar."""

    if (
        len(cold_first_pass_seconds) < required_runs_each
        or len(warm_first_pass_seconds) < required_runs_each
    ):
        raise ValueError("CPU shadow gate requires 30 cold and 30 warm runs")
    if (
        any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in [*cold_first_pass_seconds, *warm_first_pass_seconds]
        )
        or not 0.0 <= pass_recall <= 1.0
        or not 0.0 <= legacy_pass_recall <= 1.0
    ):
        raise ValueError("CPU shadow gate evidence is invalid")
    # Cold-start regressions must not be hidden by pooling them with faster
    # warm runs. Both independently collected 30-run populations must clear
    # the latency gate before CPU is promoted without a GPU sidecar.
    latency_failed = max(
        _percentile95(cold_first_pass_seconds),
        _percentile95(warm_first_pass_seconds),
    ) > threshold_s
    return latency_failed or pass_recall < legacy_pass_recall
