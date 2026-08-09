from __future__ import annotations

from pathlib import Path

import pytest

from extensions.gazebo.ros2_ws import m3_pickplace_acceptance as acceptance


ROOT = Path(__file__).resolve().parents[1]


def test_acceptance_uses_frozen_fingertip_collision_mesh_centers() -> None:
    mesh_root = (
        ROOT
        / "extensions/gazebo/assets/robotiq_2f85_vendor/meshes/collision/2f_85"
    )
    left = acceptance._stl_center(mesh_root / "left_finger_tip.stl")
    right = acceptance._stl_center(mesh_root / "right_finger_tip.stl")
    assert left == pytest.approx((-0.00965179, -0.00080010, 0.02252109))
    assert right == pytest.approx((0.00965179, 0.00005010, 0.02252109))


def test_acceptance_compaction_keeps_physics_timestamps_but_drops_images() -> None:
    compacted = acceptance._compact(
        {
            "camera": {"rgb": [[1, 2]], "depth": [[0.5]]},
            "stream_timestamps_s": {"rgb": 10.2, "depth": 10.2},
            "rgb_base64": "payload",
        }
    )
    assert compacted["camera"] == {}
    assert compacted["stream_timestamps_s"] == {"rgb": 10.2, "depth": 10.2}
    assert "rgb_base64" not in compacted


def test_acceptance_mount_pose_places_live_collision_center_at_requested_point() -> None:
    orientation = acceptance._q_euler(3.141592653589793, 1.0471975511965976, 3.141592653589793)
    offset = (0.0, -0.000375, 0.1268473)
    center = (0.28, -0.10, 0.51)
    pose = acceptance._mount_pose(center, orientation, offset)
    recovered_offset = acceptance._q_rotate(pose["quat_xyzw"], offset)
    assert [pose["xyz"][index] + recovered_offset[index] for index in range(3)] == pytest.approx(center)


def test_acceptance_shell_owns_isolation_without_broad_process_kills() -> None:
    script = (
        ROOT / "extensions/gazebo/ros2_ws/run_m3_pickplace_acceptance.sh"
    ).read_text(encoding="utf-8")
    assert "GZ_PARTITION=" in script
    assert "ROS_DOMAIN_ID" in script
    assert "flock -n" in script
    assert "kill -TERM -- \"-${pgid}\"" in script
    assert '"${pgid}" != "${current_pgid}"' in script
    assert "pkill" not in script


def test_partition_scan_never_returns_an_ancestor_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    own_group = 700
    independent_group = 900
    monkeypatch.setattr(acceptance, "_ancestors", lambda: {10, 20})
    monkeypatch.setattr(acceptance, "_ancestor_process_groups", lambda: {own_group})
    monkeypatch.setattr(
        acceptance.Path,
        "iterdir",
        lambda _path: iter((Path("/proc/10"), Path("/proc/30"), Path("/proc/40"))),
    )
    monkeypatch.setattr(
        acceptance.Path,
        "read_bytes",
        lambda path: b"GZ_PARTITION=test-partition\0" if path.name == "environ" else b"",
    )
    monkeypatch.setattr(
        acceptance,
        "_process_row",
        lambda pid: {
            "pid": pid,
            "pgid": own_group if pid == 30 else independent_group,
            "command": "helper",
        },
    )

    rows = acceptance._isolated_processes("test-partition")

    assert [row["pid"] for row in rows] == [40]
