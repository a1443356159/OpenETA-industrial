from __future__ import annotations

import inspect

import pytest

from scripts import openeta_mcp_services as services
from tools.anygrasp_core import AnyGraspBackend
from tools.anyplace_core import AnyPlaceBackend
from tools.candidate_config import (
    DEFAULT_ANYPLACE_RAW_POOL_SIZE,
    DEFAULT_GRASP_RAW_POOL_SIZE,
    CandidateFunnelConfig,
)
from tools.graspgenx_core import GraspGenXBackend


def test_candidate_services_have_only_reserve_defaults(tmp_path):
    assert DEFAULT_GRASP_RAW_POOL_SIZE == 200
    assert DEFAULT_ANYPLACE_RAW_POOL_SIZE == 96
    assert AnyGraspBackend(
        sdk_root=tmp_path, checkpoint_path=tmp_path
    ).raw_pool_size == 200
    assert AnyPlaceBackend(
        anyplace_root=tmp_path, config_path=tmp_path
    ).raw_pool_size == 96
    assert inspect.signature(GraspGenXBackend).parameters["raw_pool_size"].default == 200
    assert not hasattr(CandidateFunnelConfig(), "graspgenx_exposure_limit")


def test_removed_exposure_cli_is_not_registered(tmp_path):
    with pytest.raises(SystemExit):
        services.build_parser().parse_args(
            [
                "status",
                "anygrasp",
                "--state-dir",
                str(tmp_path),
                "--anygrasp-max-candidates",
                "13",
            ]
        )


def test_raw_pool_cli_precedes_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENETA_ANYGRASP_RAW_POOL_SIZE", "120")
    args = services.build_parser().parse_args(
        [
            "status",
            "anygrasp",
            "--state-dir",
            str(tmp_path),
            "--anygrasp-raw-pool-size",
            "180",
        ]
    )
    command = services._build_configs(args)[0].command
    assert command[command.index("--raw-pool-size") + 1] == "180"
    assert "--max-candidates" not in command


def test_raw_pool_environment_precedes_default(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENETA_ANYPLACE_RAW_POOL_SIZE", "80")
    monkeypatch.setenv("OPENETA_ANYPLACE_DIVERSITY_POOL_SIZE", "80")
    args = services.build_parser().parse_args(
        ["status", "anyplace", "--state-dir", str(tmp_path)]
    )
    command = services._build_configs(args)[0].command
    assert command[command.index("--raw-pool-size") + 1] == "80"
    assert "--candidate-count" not in command


def test_invalid_raw_pool_environment_fails_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENETA_GRASPGENX_RAW_POOL_SIZE", "513")
    args = services.build_parser().parse_args(
        ["status", "graspgenx", "--state-dir", str(tmp_path)]
    )
    with pytest.raises(services.ConfigError):
        services._build_configs(args)
