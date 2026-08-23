"""Shared startup-only candidate-count configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


DEFAULT_GRASP_RAW_POOL_SIZE = 200
DEFAULT_ANYPLACE_RAW_POOL_SIZE = 96
DEFAULT_GRASP_DIVERSITY_POOL_SIZE = 64
DEFAULT_ANYPLACE_DIVERSITY_POOL_SIZE = 96
DEFAULT_GRASP_FULL_PLAN_LIMIT = 4
DEFAULT_ANYPLACE_FULL_PLAN_LIMIT = 4
DEFAULT_MOVEIT_IK_SEED_COUNT = 8
DEFAULT_ANYPLACE_MAX_QUALIFICATION_ROUNDS = 2

def bounded_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    """Parse a strict (non-boolean) integer bounded at startup."""

    message = f"{name} must be an integer in [{minimum}, {maximum}]"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(message)
    if isinstance(value, str) and value.strip() != str(parsed):
        raise ValueError(message)
    if not minimum <= parsed <= maximum:
        raise ValueError(message)
    return parsed


def raw_pool_size(value: Any, *, placement: bool = False) -> int:
    return bounded_int(
        value,
        name="raw pool size",
        minimum=10,
        maximum=256 if placement else 512,
    )


def pool_limit(value: Any, *, name: str, maximum: int = 512) -> int:
    return bounded_int(value, name=name, minimum=1, maximum=maximum)


def argparse_raw_pool_size(*, placement: bool = False):
    def parse(value: str) -> int:
        try:
            return raw_pool_size(value, placement=placement)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    return parse


@dataclass(frozen=True, slots=True)
class CandidateFunnelConfig:
    """Immutable host configuration for reserve pools and bounded planning."""

    graspgenx_raw_pool_size: int = DEFAULT_GRASP_RAW_POOL_SIZE
    anygrasp_raw_pool_size: int = DEFAULT_GRASP_RAW_POOL_SIZE
    anyplace_raw_pool_size: int = DEFAULT_ANYPLACE_RAW_POOL_SIZE
    grasp_diversity_pool_size: int = DEFAULT_GRASP_DIVERSITY_POOL_SIZE
    anyplace_diversity_pool_size: int = DEFAULT_ANYPLACE_DIVERSITY_POOL_SIZE
    grasp_full_plan_limit: int = DEFAULT_GRASP_FULL_PLAN_LIMIT
    anyplace_full_plan_limit: int = DEFAULT_ANYPLACE_FULL_PLAN_LIMIT
    moveit_ik_seed_count: int = DEFAULT_MOVEIT_IK_SEED_COUNT
    anyplace_max_qualification_rounds: int = DEFAULT_ANYPLACE_MAX_QUALIFICATION_ROUNDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "graspgenx_raw_pool_size", raw_pool_size(self.graspgenx_raw_pool_size))
        object.__setattr__(self, "anygrasp_raw_pool_size", raw_pool_size(self.anygrasp_raw_pool_size))
        object.__setattr__(self, "anyplace_raw_pool_size", raw_pool_size(self.anyplace_raw_pool_size, placement=True))
        for field_name, maximum in (
            ("grasp_diversity_pool_size", 512),
            ("anyplace_diversity_pool_size", 256),
            ("grasp_full_plan_limit", 512),
            ("anyplace_full_plan_limit", 256),
            ("moveit_ik_seed_count", 64),
            ("anyplace_max_qualification_rounds", 16),
        ):
            object.__setattr__(
                self,
                field_name,
                bounded_int(getattr(self, field_name), name=field_name, minimum=1, maximum=maximum),
            )
        self._validate_chain(
            "GraspGenX",
            self.graspgenx_raw_pool_size,
            self.grasp_diversity_pool_size,
            self.grasp_full_plan_limit,
        )
        self._validate_chain(
            "AnyGrasp",
            self.anygrasp_raw_pool_size,
            self.grasp_diversity_pool_size,
            self.grasp_full_plan_limit,
        )
        self._validate_chain(
            "AnyPlace",
            self.anyplace_raw_pool_size,
            self.anyplace_diversity_pool_size,
            self.anyplace_full_plan_limit,
        )

    @staticmethod
    def _validate_chain(name: str, raw: int, diversity: int, full_plan: int) -> None:
        if not raw >= diversity >= full_plan:
            raise ValueError(
                f"{name} requires raw_pool >= diversity_pool >= full_plan_limit"
            )
