"""Validate one ros2 topic echo JointState sample for the 2F-85 mimic contract."""

from __future__ import annotations

import argparse
from pathlib import Path


def _yaml_sequence(lines: list[str], key: str, next_key: str) -> list[str]:
    start = lines.index(f"{key}:") + 1
    try:
        end = lines.index(f"{next_key}:", start)
    except ValueError:
        end = len(lines)
    return [line.strip()[2:].strip().strip("'\"") for line in lines[start:end] if line.strip().startswith("-")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("state", choices=("open", "closed"))
    args = parser.parse_args()
    lines = args.sample.read_text(encoding="utf-8").splitlines()
    names = _yaml_sequence(lines, "name", "position")
    raw_positions = _yaml_sequence(lines, "position", "velocity")
    if len(names) != len(raw_positions):
        raise SystemExit("JOINT_STATE_TIMEOUT: malformed name/position arrays")
    values = dict(zip(names, map(float, raw_positions), strict=True))
    multipliers = {
        "gripper_right_finger_joint": -1.0,
        "gripper_left_inner_knuckle_joint": 1.0,
        "gripper_right_inner_knuckle_joint": -1.0,
        "gripper_left_finger_tip_joint": -1.0,
        "gripper_right_finger_tip_joint": 1.0,
    }
    active_name = "gripper_left_finger_joint"
    required = {*(f"joint_{index}" for index in range(1, 8)), active_name, *multipliers}
    if not required.issubset(values):
        raise SystemExit(f"JOINT_STATE_TIMEOUT: missing {sorted(required - values.keys())}")
    active = values[active_name]
    target = 0.0 if args.state == "open" else 0.7929
    if abs(active - target) > 0.035:
        raise SystemExit(f"GRIPPER_FAILED: {args.state} active angle {active:.5f}, target {target:.5f}")
    for name, multiplier in multipliers.items():
        if abs(values[name] - multiplier * active) > 0.035:
            raise SystemExit(f"GRIPPER_FAILED: mimic mismatch {name}={values[name]:.5f}")
    aperture_mm = 85.0 if args.state == "open" else 0.0
    print(f"OK GRIPPER_{args.state.upper()} active_angle_rad={active:.5f} aperture_mm~={aperture_mm:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
