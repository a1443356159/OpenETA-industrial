"""Shared startup-only candidate-count configuration."""

from __future__ import annotations

import argparse
from typing import Any


DEFAULT_CANDIDATE_COUNT = 10
MIN_CANDIDATE_COUNT = 1
MAX_CANDIDATE_COUNT = 20


def candidate_count(value: Any) -> int:
    """Parse one strict candidate count in the supported startup range."""

    if isinstance(value, bool):
        raise ValueError("candidate count must be an integer in [1, 20]")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate count must be an integer in [1, 20]") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("candidate count must be an integer in [1, 20]")
    if isinstance(value, str) and value.strip() != str(parsed):
        raise ValueError("candidate count must be an integer in [1, 20]")
    if not MIN_CANDIDATE_COUNT <= parsed <= MAX_CANDIDATE_COUNT:
        raise ValueError("candidate count must be an integer in [1, 20]")
    return parsed


def argparse_candidate_count(value: str) -> int:
    try:
        return candidate_count(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
