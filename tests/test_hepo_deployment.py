from __future__ import annotations

from pathlib import Path

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
