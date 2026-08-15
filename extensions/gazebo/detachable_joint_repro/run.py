#!/usr/bin/env python3
"""Physical DetachableJoint reproductions for the M3 attachment hard gate.

The plugin state topic is recorded only as a diagnostic.  A row passes only
when three attach -> parent-motion -> detach cycles measured from Gazebo's
world ``Pose_V`` stream show less than 1 mm / 0.5 degree relative drift while
attached, then either a fall or more than 50 mm relative separation detached.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
MAX_ATTACHED_TRANSLATION_M = 0.001
MAX_ATTACHED_ROTATION_RAD = math.radians(0.5)
MIN_DETACHED_SEPARATION_M = 0.050
# A wrist-roll probe can rotate its parent substantially while translating a
# nearby payload only a few millimetres.  Detachment is therefore a Pose_V
# condition: either its relative position or its relative orientation must
# change decisively.  This is deliberately far above the attached 0.5 degree
# drift budget.
MIN_DETACHED_ROTATION_RAD = 0.050


class ReproError(RuntimeError):
    pass


def _quat_normalized(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ReproError("Pose_V contained a zero quaternion")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _quat_inverse(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = _quat_normalized(values)
    return (-x, -y, -z, w)


def _quat_multiply(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _rotate(
    quaternion: tuple[float, float, float, float], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    q_vector = (vector[0], vector[1], vector[2], 0.0)
    rotated = _quat_multiply(_quat_multiply(quaternion, q_vector), _quat_inverse(quaternion))
    return rotated[:3]


def _relative_pose(
    parent: dict[str, tuple[float, ...]], child: dict[str, tuple[float, ...]]
) -> dict[str, tuple[float, ...]]:
    parent_position = parent["position"]
    child_position = child["position"]
    parent_orientation = parent["orientation"]
    child_orientation = child["orientation"]
    inverse = _quat_inverse(parent_orientation)  # type: ignore[arg-type]
    translation = _rotate(
        inverse,
        tuple(child_position[index] - parent_position[index] for index in range(3)),  # type: ignore[arg-type]
    )
    return {
        "position": translation,
        "orientation": _quat_normalized(_quat_multiply(inverse, child_orientation)),  # type: ignore[arg-type]
    }


def _pose_delta(left: dict[str, tuple[float, ...]], right: dict[str, tuple[float, ...]]) -> tuple[float, float]:
    translation = math.dist(left["position"], right["position"])
    dot = abs(sum(a * b for a, b in zip(left["orientation"], right["orientation"])))
    return translation, 2.0 * math.acos(min(1.0, max(-1.0, dot)))


class ReproCase:
    def __init__(self, *, name: str, cycles: int, output_dir: Path) -> None:
        layouts = {
            "single": {
                "world": "detachable_single_link_repro",
                "world_file": ROOT / "single_link.sdf",
                "parent": "single_parent",
                "child": "single_child",
                "attach": "/repro/single/attach",
                "detach": "/repro/single/detach",
                "state": "/repro/single/state",
                "child_start": (0.15, 0.0, 0.80),
            },
            "articulated": {
                "world": "detachable_articulated_repro",
                "world_file": ROOT / "articulated.sdf",
                "parent": "articulated_parent",
                "child": "articulated_child",
                "attach": "/repro/articulated/attach",
                "detach": "/repro/articulated/detach",
                "state": "/repro/articulated/state",
                "child_start": (0.25, 0.0, 0.80),
            },
            "dynamic": {
                "world": "detachable_dynamic_articulated_repro",
                "world_file": ROOT / "dynamic_articulated.sdf",
                "parent": "terminal_link",
                "parent_model": "dynamic_parent",
                "child": "dynamic_child",
                "attach": "/repro/dynamic/attach",
                "detach": "/repro/dynamic/detach",
                "state": "/repro/dynamic/state",
                "child_start": (0.35, 0.0, 0.80),
                "command": "/repro/dynamic/command",
            },
            "dynamic_fixed_terminal": {
                "world": "detachable_dynamic_fixed_terminal_repro",
                "world_file": ROOT / "dynamic_fixed_terminal.sdf",
                "parent": "mount_link",
                "parent_model": "fixed_terminal_parent",
                "child": "fixed_terminal_child",
                "attach": "/repro/fixed_terminal/attach",
                "detach": "/repro/fixed_terminal/detach",
                "state": "/repro/fixed_terminal/state",
                "child_start": (0.45, 0.0, 0.80),
                "command": "/repro/fixed_terminal/command",
            },
        }
        if name not in layouts:
            raise ValueError(f"unsupported repro case: {name}")
        self.name = name
        self.layout = layouts[name]
        self.cycles = cycles
        self.output_dir = output_dir
        self.partition = f"openeta_detachable_repro_{name}_{os.getpid()}"
        self.environment = dict(os.environ, GZ_PARTITION=self.partition)
        self.gz = shutil.which("gz")
        self.process: subprocess.Popen[str] | None = None
        self.diagnostics: list[dict[str, Any]] = []

    def _run(self, *arguments: str, timeout_s: float = 15.0) -> subprocess.CompletedProcess[str]:
        if not self.gz:
            raise ReproError("gz is unavailable; source /opt/ros/jazzy/setup.bash first")
        return subprocess.run(
            [self.gz, *arguments],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=self.environment,
            check=False,
        )

    def _service(self, name: str, request_type: str, reply_type: str, request: str) -> None:
        result = self._run(
            "service", "-s", f"/world/{self.layout['world']}/{name}",
            "--reqtype", request_type, "--reptype", reply_type,
            "--timeout", "3000", "--req", request,
        )
        if result.returncode != 0 or "data: true" not in result.stdout.lower():
            raise ReproError(f"{name} service failed: {(result.stdout + result.stderr)[-800:]}")

    def _set_pose(self, model: str, xyz: tuple[float, float, float]) -> None:
        self._service(
            "set_pose", "gz.msgs.Pose", "gz.msgs.Boolean",
            f'name: "{model}", position: {{x: {xyz[0]}, y: {xyz[1]}, z: {xyz[2]}}}, orientation: {{w: 1.0}}',
        )

    def _pause(self, paused: bool) -> None:
        self._service(
            "control", "gz.msgs.WorldControl", "gz.msgs.Boolean",
            f"pause: {'true' if paused else 'false'}",
        )

    def _publish(self, topic: str) -> None:
        result = self._run("topic", "-t", topic, "-m", "gz.msgs.Empty", "-p", "")
        if result.returncode != 0:
            raise ReproError(f"publish to {topic} failed: {result.stderr[-800:]}")

    def _move_parent(self, index: int) -> None:
        command = self.layout.get("command")
        if command:
            result = self._run(
                "topic", "-t", command, "-m", "gz.msgs.Double", "-p", "data: 0.35"
            )
            if result.returncode != 0:
                raise ReproError(f"dynamic parent command failed: {result.stderr[-800:]}")
            return
        self._set_pose(self.layout["parent"], (0.12, 0.04 * (index + 1), 0.86))

    def _return_parent(self) -> None:
        command = self.layout.get("command")
        if command:
            result = self._run(
                "topic", "-t", command, "-m", "gz.msgs.Double", "-p", "data: 0.0"
            )
            if result.returncode != 0:
                raise ReproError(f"dynamic parent reset failed: {result.stderr[-800:]}")

    @staticmethod
    def _field(block: str, name: str, *, default: float) -> float:
        found = re.search(rf"\b{name}:\s*([-+0-9.eE]+)", block)
        return float(found.group(1)) if found else default

    def _world_poses(self) -> dict[str, dict[str, tuple[float, ...]]]:
        result = self._run(
            "topic", "-e", "-n", "1", "-t", f"/world/{self.layout['world']}/pose/info",
            timeout_s=10.0,
        )
        if result.returncode != 0:
            raise ReproError(f"Pose_V read failed: {result.stderr[-800:]}")
        poses: dict[str, dict[str, tuple[float, ...]]] = {}
        blocks: list[str] = []
        current: list[str] | None = None
        depth = 0
        for line in result.stdout.splitlines():
            if current is None:
                if line.strip() == "pose {":
                    current = [line]
                    depth = 1
                continue
            current.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0:
                blocks.append("\n".join(current))
                current = None
        for block in blocks:
            named = re.search(r'\bname:\s*"([^"]+)"', block)
            if not named:
                continue
            position_block = re.search(r"position\s*\{(.*?)\}", block, re.DOTALL)
            orientation_block = re.search(r"orientation\s*\{(.*?)\}", block, re.DOTALL)
            if not position_block or not orientation_block:
                continue
            poses[named.group(1)] = {
                "position": tuple(self._field(position_block.group(1), axis, default=0.0) for axis in ("x", "y", "z")),
                "orientation": _quat_normalized(tuple(self._field(orientation_block.group(1), axis, default=1.0 if axis == "w" else 0.0) for axis in ("x", "y", "z", "w"))),
            }
        missing = {self.layout["parent"], self.layout["child"]} - set(poses)
        if missing:
            raise ReproError(f"Pose_V missing model poses: {sorted(missing)}")
        return poses

    def _relative_from_world(self) -> dict[str, tuple[float, ...]]:
        poses = self._world_poses()
        return _relative_pose(poses[self.layout["parent"]], poses[self.layout["child"]])

    def _state_diagnostic(self, action: str) -> None:
        """Best-effort state capture; deliberately excluded from the verdict."""
        if not self.gz:
            return
        echo = subprocess.Popen(
            [self.gz, "topic", "-e", "-n", "1", "-t", self.layout["state"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.environment,
        )
        try:
            time.sleep(0.25)
            self._publish(self.layout[action])
            stdout, stderr = echo.communicate(timeout=3.0)
            self.diagnostics.append({"action": action, "state_topic": stdout.strip(), "stderr": stderr.strip()})
        except (subprocess.TimeoutExpired, ReproError) as exc:
            self.diagnostics.append({"action": action, "state_topic_error": str(exc)})
        finally:
            if echo.poll() is None:
                echo.kill()
                echo.wait()

    def _wait_for_models(self) -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            result = self._run("model", "--list", timeout_s=5.0)
            parent_model = self.layout.get("parent_model", self.layout["parent"])
            if parent_model in result.stdout and self.layout["child"] in result.stdout:
                return
            time.sleep(0.25)
        raise ReproError("models did not appear before deadline")

    def _start(self) -> None:
        if not self.gz:
            raise ReproError("gz is unavailable; source /opt/ros/jazzy/setup.bash first")
        log = self.output_dir / f"{self.name}.gz.log"
        self.process = subprocess.Popen(
            [self.gz, "sim", "-s", "-r", str(self.layout["world_file"]), "--physics-engine", "gz-physics-dartsim-plugin"],
            stdout=log.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            env=self.environment,
            start_new_session=True,
            text=True,
        )
        self._wait_for_models()

    def _stop(self) -> None:
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=8.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
        finally:
            self.process = None

    def run(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "case": self.name,
            "partition": self.partition,
            "criteria": {
                "attached_relative_translation_m_lt": MAX_ATTACHED_TRANSLATION_M,
                "attached_relative_rotation_rad_lt": MAX_ATTACHED_ROTATION_RAD,
                "detached_relative_separation_m_gt": MIN_DETACHED_SEPARATION_M,
                "detached_relative_rotation_rad_gt": MIN_DETACHED_ROTATION_RAD,
                "pose_source": f"/world/{self.layout['world']}/pose/info (gz.msgs.Pose_V)",
            },
            "cycles": [],
        }
        try:
            self._start()
            # Plugins request an attach at spawn.  Begin every measured cycle
            # detached; no initial plugin state is accepted as proof.
            self._state_diagnostic("detach")
            time.sleep(1.0)
            for index in range(self.cycles):
                child_start = self.layout["child_start"]
                self._pause(True)
                if "command" not in self.layout:
                    self._set_pose(self.layout["parent"], (0.0, 0.0, 0.80))
                self._set_pose(self.layout["child"], child_start)
                if "command" in self.layout:
                    self._return_parent()
                self._state_diagnostic("attach")
                self._pause(False)
                time.sleep(0.6)
                before = self._relative_from_world()

                self._move_parent(index)
                time.sleep(0.8)
                after = self._relative_from_world()
                translation_drift, rotation_drift = _pose_delta(before, after)

                self._state_diagnostic("detach")
                self._return_parent()
                time.sleep(1.0)
                detached = self._relative_from_world()
                detached_separation, detached_rotation = _pose_delta(after, detached)
                cycle = {
                    "index": index + 1,
                    "relative_before_motion": before,
                    "relative_after_motion": after,
                    "relative_after_detach": detached,
                    "attached_translation_drift_m": translation_drift,
                    "attached_rotation_drift_rad": rotation_drift,
                    "detached_relative_separation_m": detached_separation,
                    "detached_relative_rotation_rad": detached_rotation,
                    "passed": (
                        translation_drift < MAX_ATTACHED_TRANSLATION_M
                        and rotation_drift < MAX_ATTACHED_ROTATION_RAD
                        and (
                            detached_separation > MIN_DETACHED_SEPARATION_M
                            or detached_rotation > MIN_DETACHED_ROTATION_RAD
                        )
                    ),
                }
                report["cycles"].append(cycle)
            report["passed"] = bool(report["cycles"]) and all(cycle["passed"] for cycle in report["cycles"])
        except Exception as exc:
            report["passed"] = False
            report["error_type"] = type(exc).__name__
            report["error"] = str(exc)
        finally:
            self._stop()
        report["state_topic_diagnostics"] = self.diagnostics
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("single", "articulated", "dynamic", "dynamic_fixed_terminal", "all"), default="all")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args(argv)
    if arguments.cycles < 3:
        parser.error("--cycles must be at least 3 for the M3 hard gate")
    output_dir = arguments.output_dir or Path(tempfile.mkdtemp(prefix="openeta-detachable-repro-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("single", "articulated", "dynamic", "dynamic_fixed_terminal") if arguments.case == "all" else (arguments.case,)
    reports = [ReproCase(name=name, cycles=arguments.cycles, output_dir=output_dir).run() for name in names]
    summary = {"output_dir": str(output_dir), "reports": reports, "passed": all(item["passed"] for item in reports)}
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
