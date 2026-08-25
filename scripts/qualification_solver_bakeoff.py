#!/usr/bin/env python3
"""Select the fastest recall-safe deterministic CPU qualification profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.runtime.qualification_bakeoff import (
    evaluate_solver_bakeoff,
    read_qualification_artifacts,
    standard_bakeoff_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="*", type=Path)
    parser.add_argument("--robot-model-sha256", default="")
    parser.add_argument("--legacy-configuration", default="kdl_legacy")
    parser.add_argument("--determinism-repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-matrix", action="store_true")
    parser.add_argument("--allow-partial-matrix", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = standard_bakeoff_matrix()
    if args.print_matrix:
        print(json.dumps(matrix, indent=2, sort_keys=True))
        return 0
    if not args.artifacts:
        raise SystemExit("at least one artifact path is required")
    artifacts = read_qualification_artifacts(
        args.artifacts, robot_model_sha256=args.robot_model_sha256
    )
    selection = evaluate_solver_bakeoff(
        artifacts,
        legacy_configuration=args.legacy_configuration,
        determinism_repetitions=args.determinism_repetitions,
        required_configurations=(
            ()
            if args.allow_partial_matrix
            else tuple(row["solver_configuration_id"] for row in matrix)
        ),
    )
    rendered = json.dumps(selection.report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if selection.selected_configuration else 2


if __name__ == "__main__":
    raise SystemExit(main())
