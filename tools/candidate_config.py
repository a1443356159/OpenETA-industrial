"""Shared startup-only candidate-count configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_GRASP_RAW_POOL_SIZE = 200
DEFAULT_ANYPLACE_RAW_POOL_SIZE = 96
DEFAULT_GRASP_DIVERSITY_POOL_SIZE = 64
DEFAULT_ANYPLACE_DIVERSITY_POOL_SIZE = 96
DEFAULT_GRASP_FULL_PLAN_LIMIT = 2
DEFAULT_ANYPLACE_FULL_PLAN_LIMIT = 2
DEFAULT_FROZEN_PAIR_GRASP_BRANCH_LIMIT = 4
DEFAULT_FROZEN_PAIR_FULL_PLAN_LIMIT = 2
DEFAULT_MOVEIT_IK_SEED_COUNT = 8
DEFAULT_ANYPLACE_MAX_QUALIFICATION_ROUNDS = 2

QUALIFICATION_PROFILES = ("legacy", "fast_v3", "shadow")
SOLVER_PROFILES = (
    "auto",
    "kdl_legacy",
    "kdl_fast",
    "trac_ik_speed",
    "trac_ik_distance",
    "pick_ik_local",
)
DEFAULT_QUALIFICATION_PROFILE = "legacy"
DEFAULT_SOLVER_PROFILE = "auto"
DEFAULT_FAST_BEAM_WIDTH = 2
DEFAULT_GRASP_WAVES = (16, 32, 64)
DEFAULT_PLACEMENT_WAVES = (12, 24, 48, 96)
DEFAULT_QUALIFICATION_MAX_CONCURRENCY = 8
DEFAULT_FAST_IK_SEED_COUNT = 2
DEFAULT_RECOVERY_IK_SEED_COUNT = 6
DEFAULT_FAST_IK_TIMEOUT_MS = 50
DEFAULT_RECOVERY_IK_TIMEOUT_MS = 200

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


def enum_value(value: Any, *, name: str, choices: Iterable[str]) -> str:
    """Parse one startup-only enumerated value without silent normalization."""

    parsed = str(value).strip()
    allowed = tuple(choices)
    if parsed not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")
    return parsed


def cumulative_waves(
    value: Any,
    *,
    name: str,
    maximum: int,
) -> tuple[int, ...]:
    """Parse a strictly increasing cumulative-wave sequence.

    Environment variables use comma-separated integers while Python callers
    may pass any non-string iterable.  The final exhaustive wave is implicit
    for grasp pools and explicit (96) for the AnyPlace pool.
    """

    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    else:
        try:
            raw = list(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a comma-separated integer sequence") from exc
    if not raw:
        raise ValueError(f"{name} must contain at least one wave")
    parsed = tuple(
        bounded_int(item, name=name, minimum=1, maximum=maximum) for item in raw
    )
    if any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return parsed


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
    frozen_pair_grasp_branch_limit: int = DEFAULT_FROZEN_PAIR_GRASP_BRANCH_LIMIT
    frozen_pair_full_plan_limit: int = DEFAULT_FROZEN_PAIR_FULL_PLAN_LIMIT
    moveit_ik_seed_count: int = DEFAULT_MOVEIT_IK_SEED_COUNT
    anyplace_max_qualification_rounds: int = DEFAULT_ANYPLACE_MAX_QUALIFICATION_ROUNDS
    qualification_profile: str = DEFAULT_QUALIFICATION_PROFILE
    solver_profile: str = DEFAULT_SOLVER_PROFILE
    fast_beam_width: int = DEFAULT_FAST_BEAM_WIDTH
    grasp_waves: tuple[int, ...] = DEFAULT_GRASP_WAVES
    placement_waves: tuple[int, ...] = DEFAULT_PLACEMENT_WAVES
    max_ik_concurrency: int = DEFAULT_QUALIFICATION_MAX_CONCURRENCY
    max_state_validity_concurrency: int = DEFAULT_QUALIFICATION_MAX_CONCURRENCY
    fast_ik_seed_count: int = DEFAULT_FAST_IK_SEED_COUNT
    recovery_ik_seed_count: int = DEFAULT_RECOVERY_IK_SEED_COUNT
    fast_ik_timeout_ms: int = DEFAULT_FAST_IK_TIMEOUT_MS
    recovery_ik_timeout_ms: int = DEFAULT_RECOVERY_IK_TIMEOUT_MS
    capability_map_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "graspgenx_raw_pool_size", raw_pool_size(self.graspgenx_raw_pool_size))
        object.__setattr__(self, "anygrasp_raw_pool_size", raw_pool_size(self.anygrasp_raw_pool_size))
        object.__setattr__(self, "anyplace_raw_pool_size", raw_pool_size(self.anyplace_raw_pool_size, placement=True))
        for field_name, maximum in (
            ("grasp_diversity_pool_size", 512),
            ("anyplace_diversity_pool_size", 256),
            ("grasp_full_plan_limit", 512),
            ("anyplace_full_plan_limit", 256),
            ("frozen_pair_grasp_branch_limit", 4),
            ("frozen_pair_full_plan_limit", 512),
            ("moveit_ik_seed_count", 64),
            ("anyplace_max_qualification_rounds", 16),
            ("fast_beam_width", 8),
            ("max_ik_concurrency", 64),
            ("max_state_validity_concurrency", 64),
            ("fast_ik_seed_count", 8),
            ("recovery_ik_seed_count", 56),
            ("fast_ik_timeout_ms", 2_000),
            ("recovery_ik_timeout_ms", 10_000),
        ):
            object.__setattr__(
                self,
                field_name,
                bounded_int(getattr(self, field_name), name=field_name, minimum=1, maximum=maximum),
            )
        object.__setattr__(
            self,
            "qualification_profile",
            enum_value(
                self.qualification_profile,
                name="qualification_profile",
                choices=QUALIFICATION_PROFILES,
            ),
        )
        object.__setattr__(
            self,
            "solver_profile",
            enum_value(
                self.solver_profile,
                name="solver_profile",
                choices=SOLVER_PROFILES,
            ),
        )
        grasp_waves_value = self.grasp_waves
        placement_waves_value = self.placement_waves
        placement_uses_default = placement_waves_value == DEFAULT_PLACEMENT_WAVES or (
            isinstance(placement_waves_value, str)
            and placement_waves_value.replace(" ", "") == "12,24,48,96"
        )
        if placement_uses_default and self.anyplace_raw_pool_size != DEFAULT_ANYPLACE_RAW_POOL_SIZE:
            placement_waves_value = tuple(
                [
                    value
                    for value in DEFAULT_PLACEMENT_WAVES
                    if value < self.anyplace_raw_pool_size
                ]
                + [self.anyplace_raw_pool_size]
            )
        object.__setattr__(
            self,
            "grasp_waves",
            cumulative_waves(
                grasp_waves_value,
                name="grasp_waves",
                maximum=max(self.graspgenx_raw_pool_size, self.anygrasp_raw_pool_size),
            ),
        )
        object.__setattr__(
            self,
            "placement_waves",
            cumulative_waves(
                placement_waves_value,
                name="placement_waves",
                maximum=self.anyplace_raw_pool_size,
            ),
        )
        capability_map_id = str(self.capability_map_id).strip()
        if capability_map_id and (
            len(capability_map_id) != 64
            or any(character not in "0123456789abcdef" for character in capability_map_id)
        ):
            raise ValueError("capability_map_id must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "capability_map_id", capability_map_id)
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
        if self.frozen_pair_full_plan_limit > (
            self.frozen_pair_grasp_branch_limit * self.anyplace_raw_pool_size
        ):
            raise ValueError(
                "frozen-pair full-plan limit cannot exceed its constructed pair ceiling"
            )
        if self.fast_beam_width > self.fast_ik_seed_count:
            raise ValueError("fast_beam_width cannot exceed fast_ik_seed_count")
        if self.fast_ik_seed_count + self.recovery_ik_seed_count != self.moveit_ik_seed_count:
            raise ValueError(
                "fast and recovery IK seed counts must equal moveit_ik_seed_count"
            )
        if self.placement_waves[-1] != self.anyplace_raw_pool_size:
            raise ValueError(
                "placement_waves must end at anyplace_raw_pool_size for exhaustive coverage"
            )
        if self.fast_ik_timeout_ms > self.recovery_ik_timeout_ms:
            raise ValueError("fast IK timeout cannot exceed recovery IK timeout")

    @staticmethod
    def _validate_chain(name: str, raw: int, diversity: int, full_plan: int) -> None:
        if not raw >= diversity >= full_plan:
            raise ValueError(
                f"{name} requires raw_pool >= diversity_pool >= full_plan_limit"
            )
