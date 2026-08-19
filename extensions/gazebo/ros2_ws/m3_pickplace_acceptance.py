#!/usr/bin/env python3
"""Validate M3 evidence files without claiming a Direct/MCP acceptance run.

Formal M3 acceptance is intentionally performed by the repository's PTY TUI
coordinator.  This helper is limited to offline schema and threshold checks so
it cannot become a second, easier acceptance route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from extensions.gazebo.native_grasp import NativePickPlaceConfig, NATIVE_GRASP_SCHEMA_VERSION


def _read(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError("M3 evidence must be a JSON object")
    return value


def validate_evidence(value: Mapping[str, Any]) -> list[str]:
    """Validate one real DirectEnv receipt shape, not an acceptance aggregate.

    M3 emits the attached ACK on the close receipt, numerical child-link proof
    on the configured lift receipt, and detached ACK on the open receipt.  A
    caller may validate each of those actual wire receipts here, but only the
    PTY TUI coordinator correlates all three into formal acceptance evidence.
    """

    config = NativePickPlaceConfig()
    errors: list[str] = []
    physical = value.get("physical_verification")
    if not isinstance(physical, Mapping):
        physical = value
    if physical.get("schema_version") != NATIVE_GRASP_SCHEMA_VERSION:
        errors.append("NATIVE_GRASP_SCHEMA_VERSION_MISMATCH")
    ack = value.get("detachable_joint")
    proof = value.get("child_link_proof")
    if not isinstance(proof, Mapping):
        proof = physical.get("evidence")
    is_attached_receipt = isinstance(ack, Mapping) and ack.get("state") == "attached"
    is_detached_receipt = isinstance(ack, Mapping) and ack.get("state") == "detached"
    is_lift_receipt = physical.get("reason_code") == "NATIVE_GRASP_TARGET_HELD"
    if is_attached_receipt:
        gate = value.get("native_contact_gate")
        if not isinstance(gate, Mapping) or gate.get("accepted") is not True:
            errors.append("NATIVE_GRASP_NATIVE_CONTACT_GATE_MISSING")
        elif (
            int(gate.get("left_sample_count", 0)) < config.contact_samples_required
            or int(gate.get("right_sample_count", 0)) < config.contact_samples_required
        ):
            errors.append("NATIVE_GRASP_NATIVE_CONTACT_SAMPLE_COUNT")
    elif is_lift_receipt:
        if not isinstance(proof, Mapping):
            errors.append("NATIVE_GRASP_CHILD_LINK_PROOF_MISSING")
        else:
            try:
                lift_m = float(proof.get("lift_m", -1.0))
                relative_m = float(proof.get("capture_relative_translation_m", float("inf")))
            except (TypeError, ValueError):
                errors.append("NATIVE_GRASP_CHILD_LINK_PROOF_MALFORMED")
                return errors
            if lift_m < config.minimum_lift_m:
                errors.append("NATIVE_GRASP_TARGET_NOT_LIFTED")
            if relative_m > config.maximum_capture_relative_translation_m:
                errors.append("NATIVE_GRASP_CAPTURE_RELATIVE_TRANSLATION_EXCEEDED")
    elif is_detached_receipt:
        return errors
    else:
        errors.append("NATIVE_GRASP_DIRECT_RECEIPT_ROLE_UNKNOWN")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    errors = validate_evidence(_read(args.evidence))
    print(json.dumps({"formal_acceptance": "not_run", "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
