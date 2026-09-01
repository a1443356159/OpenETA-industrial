#!/usr/bin/env python3
"""Validate frozen OpenETA model assets and create the SAM3 offline cache view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "openeta.hepo.model_assets.v1"
SAM3_REVISION = "96f3e1b404ba14f2cfac60ee6ae87c269a7b7923"
ANYPLACE_RELEASE = "669f1b0ebcbe2ae3a72970ff31e911e8af73b2d6"
GRASPGENX_MODEL_REVISION = "7c834043c11a11417e31d6d5ea9355801e40a2c1"
GRIPPER_REVISION = "19a03c00d19aeaf052d0f6801f0041982d676e8a"

EXPECTED_FILES = {
    "sam3/config.json": "4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f",
    "sam3/sam3.pt": "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
    "anyplace/anyplace_multitask/model.pth": "d3d33f0a279633c25f252960a208d4b4447a756f0cff8e94be0faadc20dc5be5",
    "graspgenx/gen/epoch_736.pth": "8b55f31cdb8340a573b4df27b027c15cff326bd6debcb389bf631d2aaab7ac44",
    "graspgenx/dis/epoch_1056.pth": "cbf3f3bdb2e4c03fca8486ed24de0e6a8a859e6bd22bce2f1434a610335abd3e",
    "graspgenx/robotiq_2f_85/config.json": "098a69c968b05dc0f712b26c7043cf888290e08e1b67a1778e7bfa4825163165",
    "graspgenx/robotiq_2f_85/gripper.urdf": "39bb45ebe636d11b20eb171cae453c5fd0f5901e35dab1a536d2e8e5eb2728ef",
    "graspgenx/robotiq_2f_85/points.json": "cc5d6d867c9f77a61d1659cb37df8270aead11aeb3cd348fd38719938ad1e0d8",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={path}",
            "-C",
            str(path),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _relative_symlink(path: Path, target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink cache entry: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target)
    os.replace(temporary, path)


def asset_paths(model_root: Path) -> dict[str, Path]:
    gripper = (
        model_root
        / "graspgenx/gripper_descriptions/gripper_descriptions/assets/x_grippers/robotiq_2f_85"
    )
    return {
        "sam3/config.json": model_root / f"sam3/{SAM3_REVISION}/config.json",
        "sam3/sam3.pt": model_root / f"sam3/{SAM3_REVISION}/sam3.pt",
        "anyplace/anyplace_multitask/model.pth": (
            model_root
            / f"anyplace/release-{ANYPLACE_RELEASE}/anyplace_ckpts/anyplace_multitask/model.pth"
        ),
        "graspgenx/gen/epoch_736.pth": (
            model_root / "graspgenx/GraspGenXModel/release/gen/epoch_736.pth"
        ),
        "graspgenx/dis/epoch_1056.pth": (
            model_root / "graspgenx/GraspGenXModel/release/dis/epoch_1056.pth"
        ),
        "graspgenx/robotiq_2f_85/config.json": gripper / "config.json",
        "graspgenx/robotiq_2f_85/gripper.urdf": gripper / "gripper.urdf",
        "graspgenx/robotiq_2f_85/points.json": gripper / "points.json",
    }


def prepare_sam3_cache_view(model_root: Path, sam3_hf_home: Path) -> Path:
    """Create a writable HF cache view backed by immutable SAM3 model files."""

    resolved_home = sam3_hf_home.resolve()
    snapshot = (
        resolved_home / f"hub/models--facebook--sam3/snapshots/{SAM3_REVISION}"
    )
    sam3_source = model_root.resolve() / f"sam3/{SAM3_REVISION}"
    _relative_symlink(
        snapshot / "config.json",
        os.path.relpath(sam3_source / "config.json", snapshot),
    )
    _relative_symlink(
        snapshot / "sam3.pt",
        os.path.relpath(sam3_source / "sam3.pt", snapshot),
    )
    _atomic_text(
        resolved_home / "hub/models--facebook--sam3/refs/main",
        SAM3_REVISION,
    )
    return resolved_home


def prepare_assets(
    model_root: Path,
    source_root: Path,
    *,
    sam3_hf_home: Path | None = None,
) -> dict[str, Any]:
    paths = asset_paths(model_root)
    files: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing model asset: {path}")
        actual = _sha256(path)
        if actual != EXPECTED_FILES[name]:
            raise RuntimeError(
                f"model asset digest mismatch for {name}: {actual} != {EXPECTED_FILES[name]}"
            )
        files[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}

    revisions = {
        "anyplace_source": _git_revision(source_root / "anyplace"),
        "graspgenx_source": _git_revision(source_root / "GraspGenX"),
        "sam3_source": _git_revision(source_root / "sam3"),
        "graspgenx_model": _git_revision(model_root / "graspgenx/GraspGenXModel"),
        "gripper_descriptions": _git_revision(
            model_root / "graspgenx/gripper_descriptions"
        ),
    }
    expected_revisions = {
        "anyplace_source": "3049f78ad226ba0d9e54c63e2ca7ad7bbcfaa45e",
        "graspgenx_source": "b9429097728cb1c430dd78b92edf17ba318aad03",
        "sam3_source": "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da",
        "graspgenx_model": GRASPGENX_MODEL_REVISION,
        "gripper_descriptions": GRIPPER_REVISION,
    }
    if revisions != expected_revisions:
        raise RuntimeError(f"model/source revision mismatch: {revisions} != {expected_revisions}")

    sam3_hf_home = prepare_sam3_cache_view(
        model_root,
        sam3_hf_home or model_root / "sam3/hf",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "model_root": str(model_root),
        "files": files,
        "revisions": revisions,
        "runtime": {
            "sam3_hf_home": str(sam3_hf_home),
            "anyplace_config": "/opt/openeta/config/anyplace-normal.yaml",
            "anyplace_root": str(source_root / "anyplace"),
            "graspgenx_root": str(source_root / "GraspGenX"),
            "graspgenx_checkpoint_root": str(
                model_root / "graspgenx/GraspGenXModel/release"
            ),
            "gripper_descriptions_root": str(
                model_root / "graspgenx/gripper_descriptions"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("/srv/openeta/models"))
    parser.add_argument(
        "--source-root", type=Path, default=Path("/opt/openeta/third_party")
    )
    parser.add_argument(
        "--sam3-hf-home",
        type=Path,
        help=(
            "Writable Hugging Face cache view for SAM3. Defaults under the model "
            "root for backward compatibility; container runtimes should place it "
            "in their writable state volume so model assets can stay read-only."
        ),
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = prepare_assets(
        args.model_root.resolve(),
        args.source_root.resolve(),
        sam3_hf_home=(args.sam3_hf_home.resolve() if args.sam3_hf_home else None),
    )
    if args.manifest:
        _atomic_text(
            args.manifest.resolve(),
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
