from __future__ import annotations

import pytest
import inspect

from scripts import openeta_mcp_services as services
from tools.anygrasp_core import AnyGraspBackend
from tools.anyplace_core import AnyPlaceBackend
from tools.candidate_config import DEFAULT_CANDIDATE_COUNT, candidate_count
from tools.graspgenx_core import GraspGenXBackend


def test_all_candidate_defaults_are_ten(tmp_path):
    assert DEFAULT_CANDIDATE_COUNT == 10
    assert AnyGraspBackend(sdk_root=tmp_path, checkpoint_path=tmp_path).max_candidates == 10
    assert AnyPlaceBackend(anyplace_root=tmp_path, config_path=tmp_path).candidate_count == 10
    assert inspect.signature(GraspGenXBackend).parameters["max_candidates"].default == 10


@pytest.mark.parametrize("value", [True, False, 0, -1, 21, "true", "1.0", 1.5])
def test_candidate_count_is_type_and_range_strict(value):
    with pytest.raises(ValueError):
        candidate_count(value)


def test_service_cli_precedes_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENETA_ANYGRASP_MAX_CANDIDATES", "7")
    args = services.build_parser().parse_args(
        [
            "status",
            "anygrasp",
            "--state-dir",
            str(tmp_path),
            "--anygrasp-max-candidates",
            "13",
        ]
    )
    command = services._build_configs(args)[0].command
    assert command[command.index("--max-candidates") + 1] == "13"


def test_service_environment_precedes_default(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENETA_ANYPLACE_CANDIDATE_COUNT", "6")
    args = services.build_parser().parse_args(
        ["status", "anyplace", "--state-dir", str(tmp_path)]
    )
    command = services._build_configs(args)[0].command
    assert command[command.index("--candidate-count") + 1] == "6"


def test_invalid_service_environment_fails_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENETA_GRASPGENX_MAX_CANDIDATES", "21")
    args = services.build_parser().parse_args(
        ["status", "graspgenx", "--state-dir", str(tmp_path)]
    )
    with pytest.raises(services.ConfigError):
        services._build_configs(args)
