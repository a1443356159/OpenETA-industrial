from __future__ import annotations

from pathlib import Path

import yaml

from deploy.ubuntu.load_provider_env import provider_values
from deploy.ubuntu.prepare_tui_workspace import mcp_registry


REPO_ROOT = Path(__file__).resolve().parents[1]
UBUNTU_DEPLOY = REPO_ROOT / "deploy" / "ubuntu"
HPC_DEPLOY = REPO_ROOT / "deploy" / "HPC"
RELEASE_BUNDLE = REPO_ROOT / "scripts" / "package_final_dev_bundle.sh"


def _compose_service() -> dict[str, object]:
    document = yaml.safe_load(
        (UBUNTU_DEPLOY / "compose.yaml").read_text(encoding="utf-8")
    )
    return document["services"]["openeta"]


def test_ubuntu_compose_uses_single_canonical_runtime_image() -> None:
    service = _compose_service()

    assert service["build"] == {
        "context": "../..",
        "dockerfile": "deploy/ubuntu/Dockerfile",
        "args": {"OPENETA_REVISION": "${OPENETA_REVISION:-local}"},
    }
    assert service["entrypoint"] == [
        "/opt/openeta/src/deploy/ubuntu/container_entrypoint.sh"
    ]
    assert service["network_mode"] == "host"
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert "privileged" not in service
    assert "env_file" not in service
    assert service["secrets"] == [
        {"source": "openeta_provider_env", "target": "openeta_provider_env"}
    ]
    devices = service["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]


def test_ubuntu_compose_keeps_models_read_only_and_state_writable() -> None:
    volumes = _compose_service()["volumes"]
    by_target = {volume["target"]: volume for volume in volumes}

    assert by_target["/srv/openeta/models"]["read_only"] is True
    assert "read_only" not in by_target["/srv/openeta/state"]
    assert by_target["/srv/openeta/state"]["source"].endswith(
        ".cache/docker/state}"
    )


def test_container_entrypoints_use_graspgenx_and_two_run_default() -> None:
    compose = (UBUNTU_DEPLOY / "compose.yaml").read_text(encoding="utf-8")
    normal = (UBUNTU_DEPLOY / "run_normal.sh").read_text(encoding="utf-8")
    open_sort = (UBUNTU_DEPLOY / "run_open_sort.sh").read_text(encoding="utf-8")
    entrypoint = (UBUNTU_DEPLOY / "container_entrypoint.sh").read_text(
        encoding="utf-8"
    )
    services = (UBUNTU_DEPLOY / "model_services.sh").read_text(encoding="utf-8")

    assert "OPENETA_ACCEPTANCE_RUNS: ${OPENETA_ACCEPTANCE_RUNS:-2}" in compose
    assert "OPENETA_GRASPGENX_RAW_POOL_SIZE: ${OPENETA_GRASPGENX_RAW_POOL_SIZE:-512}" in compose
    assert "OPENETA_SCRIPTED_TUI_FOLLOW_UP_TASKS:" in compose
    assert "--grasp-backend graspgenx" in normal
    assert 'scenario="multi_normal"' in normal
    assert "multi_normal_random_12345" in normal
    assert "--task-variant" in normal
    assert 'scripts/run_pick_place_acceptance.sh' in normal
    assert "run_m6_gazebo_acceptance.sh" not in normal
    assert "for target in sam3 anyplace graspgenx" in services
    assert "start anygrasp" not in services
    assert 'open-sort)' in entrypoint
    assert "run_open_sort.sh" in entrypoint
    assert "run_open_sort_gazebo_tui.sh" in open_sort
    assert "--grasp-backend graspgenx" in open_sort
    assert "operator-session-report.json" in (
        REPO_ROOT / "docs" / "final-dev-delivery.md"
    ).read_text(encoding="utf-8")
    assert "deploy/ubuntu/openeta.sh open-sort" in (
        REPO_ROOT / "docs" / "ubuntu-docker-deployment.md"
    ).read_text(encoding="utf-8")


def test_docker_example_matches_final_graspgenx_reserve_default() -> None:
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "OPENETA_GRASPGENX_RAW_POOL_SIZE=512" in example


def test_tui_registry_contains_only_container_started_services() -> None:
    registry = mcp_registry(
        sim_port=8765,
        sam3_port=8773,
        anyplace_port=8775,
        graspgenx_port=8778,
    )

    assert registry == {
        "mcpServers": {
            "openeta-sim": {"url": "http://127.0.0.1:8765/sse"},
            "openeta-sam3": {"url": "http://127.0.0.1:8773/sse"},
            "openeta-anyplace": {"url": "http://127.0.0.1:8775/sse"},
            "openeta-graspgenx": {"url": "http://127.0.0.1:8778/sse"},
        }
    }
    tui_launcher = (UBUNTU_DEPLOY / "run_tui.sh").read_text(encoding="utf-8")
    assert '"/dev/tcp/127.0.0.1/${SIM_PORT}"' in tui_launcher


def test_docker_context_excludes_local_credentials_and_runtime_state() -> None:
    ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    for protected in (
        ".env",
        ".mcp.json",
        "apikey.md",
        ".openeta_memory",
        "checkpoint_detection.tar*",
        "license_YuanyiYan.zip*",
    ):
        assert protected in ignored


def test_provider_secret_loader_has_a_narrow_allowlist(tmp_path: Path) -> None:
    secret = tmp_path / "provider.env"
    secret.write_text(
        "OPENETA_LLM_API_KEY=secret with spaces\n"
        "OPENETA_LLM_MODEL=vision-model\n"
        "OPENETA_GRASP_BACKEND=anygrasp\n"
        "PATH=/untrusted\n",
        encoding="utf-8",
    )

    assert provider_values(secret) == {
        "OPENETA_LLM_API_KEY": "secret with spaces",
        "OPENETA_LLM_MODEL": "vision-model",
    }


def test_platform_specific_launchers_are_separated() -> None:
    launcher = (UBUNTU_DEPLOY / "openeta.sh").read_text(encoding="utf-8")

    assert (UBUNTU_DEPLOY / "openeta.sh").is_file()
    assert (UBUNTU_DEPLOY / "compose.yaml").is_file()
    assert (HPC_DEPLOY / "run_smoke_normal.sbatch").is_file()
    assert (HPC_DEPLOY / "import_oci_image.sh").is_file()
    assert not (HPC_DEPLOY / "compose.yaml").exists()
    assert "--env-file /dev/null" in launcher
    assert "/srv/openeta/model-download:rw" in launcher
    assert "OPENETA_MODEL_ROOT=/srv/openeta/model-download" in launcher


def test_release_bundle_script_archives_only_a_pinned_git_revision() -> None:
    source = RELEASE_BUNDLE.read_text(encoding="utf-8")

    assert RELEASE_BUNDLE.is_file()
    assert "git -C \"${repo_root}\" archive" in source
    assert "rev-parse --verify \"${revision}^{commit}\"" in source
    assert "sha256sum" in source
    assert "refusing to overwrite" in source
