from __future__ import annotations

import os
from pathlib import Path
import subprocess

from deploy.hepo.prepare_assets import (
    ANYPLACE_RELEASE,
    EXPECTED_FILES,
    SAM3_REVISION,
    asset_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hepo_asset_layout_is_revisioned_and_complete(tmp_path: Path) -> None:
    paths = asset_paths(tmp_path)

    assert paths["sam3/sam3.pt"] == tmp_path / f"sam3/{SAM3_REVISION}/sam3.pt"
    assert paths["anyplace/anyplace_multitask/model.pth"] == (
        tmp_path
        / f"anyplace/release-{ANYPLACE_RELEASE}/anyplace_ckpts/anyplace_multitask/model.pth"
    )
    assert set(paths) == set(EXPECTED_FILES)


def test_hepo_container_keeps_model_services_in_separate_venvs() -> None:
    dockerfile = (REPO_ROOT / "deploy/hepo/Dockerfile").read_text(encoding="utf-8")

    for environment in ("openeta", "sam3", "anyplace", "graspgenx"):
        assert f"/opt/openeta/venvs/{environment}" in dockerfile
    assert "COPY . /opt/openeta/src" in dockerfile
    assert "COPY checkpoint" not in dockerfile
    assert "unset PIP_CONSTRAINT" in dockerfile
    assert "pip==25.2" in dockerfile
    assert "torch-cluster==1.6.1+pt113cu117" in dockerfile
    assert "mktemp -d /tmp/openeta-anyplace-build.XXXXXX" in dockerfile
    assert 'touch "${anyplace_build}/anyplace/__init__.py"' in dockerfile
    assert 'LD_LIBRARY_PATH="${anyplace_torch_lib}' in dockerfile
    assert "from graspgenx.grasp_server import" in dockerfile
    assert "from anyplace.model.transformer.policy import" in dockerfile
    assert "-e /opt/openeta/third_party" not in dockerfile


def test_hepo_slurm_job_leaves_cluster_resources_to_submit_wrapper() -> None:
    sbatch = (REPO_ROOT / "deploy/hepo/run_smoke_normal.sbatch").read_text(
        encoding="utf-8"
    )
    submit = (REPO_ROOT / "deploy/hepo/submit_smoke_normal.sh").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --cpus-per-task=12" in sbatch
    assert "#SBATCH --partition" not in sbatch
    assert "#SBATCH --gres" not in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert "command -v apptainer || command -v singularity" in sbatch
    assert "OPENETA_SLURM_PARTITION" in submit
    assert "OPENETA_SLURM_GRES:-gpu:1" in submit
    assert "OPENETA_SLURM_MEMORY" in submit


def test_slurm_runtime_scripts_do_not_embed_site_identity() -> None:
    runtime_scripts = (
        "fetch_models.sh",
        "import_oci_image.sh",
        "run_smoke_normal.sbatch",
        "submit_smoke_normal.sh",
    )

    for name in runtime_scripts:
        content = (REPO_ROOT / "deploy/hepo" / name).read_text(encoding="utf-8")
        assert "/home/yyy" not in content
        assert "hepnodes" not in content
        assert "gpu:L40" not in content


def test_docker_context_excludes_local_protected_assets() -> None:
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "checkpoint_detection.tar*" in dockerignore
    assert "checkpoint_tracking.tar*" in dockerignore
    assert "license_YuanyiYan.zip*" in dockerignore


def test_hepo_workflow_keeps_digest_provenance_without_oversized_sbom() -> None:
    workflow = (REPO_ROOT / ".github/workflows/hepo-container.yml").read_text(
        encoding="utf-8"
    )

    assert "provenance: mode=max" in workflow
    assert "sbom: false" in workflow
    assert "digest=${{ steps.build.outputs.digest }}" in workflow
    assert "cache-from: type=gha" in workflow
    assert "cache-to:" not in workflow


def test_image_import_retries_in_place_with_http1_fallback(tmp_path: Path) -> None:
    fake_runtime = tmp_path / "singularity"
    attempt_file = tmp_path / "attempts"
    godebug_file = tmp_path / "retry-godebug"
    fake_runtime.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  build)
    attempt=0
    [[ ! -f "${ATTEMPT_FILE}" ]] || attempt="$(cat "${ATTEMPT_FILE}")"
    attempt=$((attempt + 1))
    printf '%s\n' "${attempt}" > "${ATTEMPT_FILE}"
    if (( attempt == 1 )); then
      exit 17
    fi
    printf '%s\n' "${GODEBUG:-}" > "${GODEBUG_FILE}"
    : > "$2"
    ;;
  inspect)
    printf '{}\n'
    ;;
  *)
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    fake_runtime.chmod(0o755)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    target = tmp_path / "images" / "openeta.sif"
    current = tmp_path / "images" / "current.sif"
    environment = os.environ | {
        "ATTEMPT_FILE": str(attempt_file),
        "GODEBUG_FILE": str(godebug_file),
        "OPENETA_CONTAINER_RUNTIME": str(fake_runtime),
        "OPENETA_IMAGE_IMPORT_RETRY_DELAY_SECONDS": "0",
        "SLURM_JOB_ID": "1234",
        "SLURM_TMPDIR": str(scratch),
    }

    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "deploy/hepo/import_oci_image.sh"),
            "ghcr.io/example/openeta@sha256:deadbeef",
            str(target),
            str(current),
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert attempt_file.read_text(encoding="utf-8").strip() == "2"
    assert "http2client=0" in godebug_file.read_text(encoding="utf-8")
    assert target.is_file()
    assert current.resolve() == target.resolve()
