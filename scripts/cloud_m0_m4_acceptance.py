#!/usr/bin/env python3
"""Report that the retired M3/M4 cloud acceptance path is fail-closed.

This is intentionally not an acceptance coordinator.  The prior coordinator
could reach an invalid manipulation implementation, so it must not build,
launch, connect to MCP, or execute any milestone.  A future approved
DetachableJoint design will replace this status-only entry point.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any


SCHEMA_VERSION = "openeta.cloud_m0_m4_acceptance.v1"
UNAVAILABLE_REASON = "DETACHABLE_JOINT_UNIMPLEMENTED_OR_UNAPPROVED"


def blocked_report() -> dict[str, Any]:
    """Return the only honest status before the approved implementation exists."""

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "blocked",
        "reason_code": UNAVAILABLE_REASON,
        "detail": "M3/M4 manipulation acceptance is unavailable pending approved DetachableJoint assets.",
        "milestones": {
            "m3": {"status": "blocked", "reason_code": UNAVAILABLE_REASON},
            "m4": {
                "status": "not_run",
                "reason_code": "PREDECESSOR_GATE_NOT_PASSED",
                "predecessor": "m3",
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = blocked_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
        print(f"CLOUD_M0_M4_ACCEPTANCE_REPORT={args.report}")
    else:
        print(rendered, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
