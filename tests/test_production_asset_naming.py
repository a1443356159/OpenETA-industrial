"""Keep development milestone labels out of production asset identities."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_ROOTS = (
    ROOT / "agent",
    ROOT / "extensions" / "gazebo",
    ROOT / "sim",
    ROOT / "tools",
    ROOT / "config",
    ROOT / "scripts",
    ROOT / ".github" / "workflows",
)
MILESTONE_PATH = re.compile(r"(?:^|[/_.-])m[0-9]+(?:[/_.-]|$)", re.IGNORECASE)
MILESTONE_INTERFACE = re.compile(
    r"extensions\.gazebo\.m[0-9]+\b"
    r"|from \.m[0-9]+ import"
    r"|\bM[0-9]+(?:Config|Controller|Verifier|ControlResult|_ENV_ID|_SCHEMA_VERSION)\b"
    r"|openeta\.m[0-9]+\."
    r"|OPENETA_M[0-9]+_"
    r"|['\"]/m[0-9]+/"
    r"|['\"]m[0-9]+_(?:target|distractor|table)['\"]",
)


def _is_acceptance_asset(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix().lower()
    name = path.name.lower()
    return (
        "acceptance" in relative
        or name.startswith("test_")
        or name.startswith("run_m")
        or name.startswith("cloud_m")
    )


def _production_files() -> list[Path]:
    files: list[Path] = []
    for root in PRODUCTION_ROOTS:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not {"__pycache__", "build", "install", "log"}.intersection(path.parts)
            and path.suffix != ".log"
            and not _is_acceptance_asset(path)
        )
    return files


def test_production_paths_do_not_encode_development_milestones() -> None:
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in _production_files()
        if MILESTONE_PATH.search(path.relative_to(ROOT).as_posix())
    ]
    assert offenders == []


def test_production_interfaces_do_not_encode_development_milestones() -> None:
    offenders: list[str] = []
    for path in _production_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if MILESTONE_INTERFACE.search(text):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_semantic_gazebo_modules_replace_milestone_modules() -> None:
    gazebo = ROOT / "extensions" / "gazebo"
    assert (gazebo / "robot_control.py").is_file()
    assert (gazebo / "native_grasp.py").is_file()
    assert not (gazebo / "m2.py").exists()
    assert not (gazebo / "m3.py").exists()
