"""Offline integrity and relocatability checks for the embedded RM75 closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

FORBIDDEN = (
    # Match any copied vendor workspace path without embedding a developer's
    # home directory (the repository must remain relocatable).
    "workstation",
    "OPENETA_" + "RM75_" + "MODEL_PATH",
)
PACKAGE_RE = re.compile(r"package://([^/]+)/([^\"'<>)\s]+)")


def validate_asset_root(root: Path) -> dict:
    root = root.resolve()
    manifest_path = root / "asset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("asset_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in manifest.get("files", [])}
    # Check directory links too; pathlib does not necessarily recurse through
    # a symlinked directory, so validating only ``is_file()`` entries would
    # leave an external asset tree undetected.
    for link in root.rglob("*"):
        if link.is_symlink() and not link.resolve().is_relative_to(root):
            raise ValueError(f"external symlink: {link.relative_to(root)}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "asset_manifest.json"
    }
    if actual != set(expected):
        raise ValueError(f"manifest file set mismatch: missing={set(expected)-actual}, extra={actual-set(expected)}")
    package_roots = {
        "openeta_rm75_v_description": root,
        "openeta_rm75_parallel_sim": root,
        "openeta_rm75_robotiq2f85_sim": root,
        # Kept for validating unmodified upstream descriptions if present.
        "rm_description": root,
    }
    combined = ""
    for rel, item in expected.items():
        path = root / rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"digest mismatch: {rel}")
        if path.suffix.lower() in {".urdf", ".xacro", ".srdf", ".yaml", ".json", ".md"} or path.name in {"NOTICE", "LICENSE"}:
            text = path.read_text(encoding="utf-8", errors="strict")
            combined += "\n" + text
            for forbidden in FORBIDDEN:
                if forbidden in text:
                    raise ValueError(f"forbidden runtime path in {rel}")
            for package, package_rel in PACKAGE_RE.findall(text):
                candidate = package_roots.get(package, root / "__missing_package__") / package_rel
                if not candidate.is_file():
                    raise ValueError(f"unresolved package URI in {rel}: package://{package}/{package_rel}")
    if manifest.get("model_id") == "rm75_robotiq_2f85_sim_v1":
        required = ["gripper_mount_link", "gripper_left_finger_joint", "gripper_right_finger_joint"]
        if any(token not in combined for token in required):
            raise ValueError("Robotiq gripper contract is incomplete")
    elif manifest.get("description_id") == "RM75-6FB-V":
        required = [
            *(f"joint_{i}" for i in range(1, 8)),
            *(f"link_{i}" for i in range(1, 8)),
            "base_link", "camera_rolink", "camera_link",
            "wrist_camera_optical_frame", "openeta_wrist_rgbd",
        ]
        if any(token not in combined for token in required):
            raise ValueError("required RM75-6FB-V arm/camera description is incomplete")
        if manifest.get("joint_names") != [f"joint_{i}" for i in range(1, 8)]:
            raise ValueError("manifest joint_names mismatch")
        if manifest.get("terminal_link") != "link_7":
            raise ValueError("manifest planning contract mismatch")
        if re.search(r'\b(?:Link[1-7]|joint[1-7])\b', combined):
            raise ValueError("legacy RM75 names found in V description")
    else:
        raise ValueError("unknown asset manifest kind")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset_root", nargs="*", type=Path)
    args = parser.parse_args()
    roots = args.asset_root or [
        Path(__file__).parent / "assets" / "rm75_6fb_v_vendor",
        Path(__file__).parent / "assets" / "robotiq_2f85_vendor",
    ]
    for root in roots:
        try:
            manifest = validate_asset_root(root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"MODEL_ASSET_NOT_FOUND: {root}: {exc}", file=sys.stderr)
            return 2
        asset_id = manifest.get("model_id", manifest.get("description_id", "unknown"))
        print(f"validated {asset_id} ({len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
