from __future__ import annotations

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

from extensions.gazebo.ros2_ws import m3_pickplace_acceptance as acceptance
from extensions.gazebo.ros2_ws import m2_robotiq2f85_acceptance as m2_acceptance
from extensions.gazebo.ros2_ws import acceptance_isolation


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "raw",
    (
        [("worker", "/openeta"), ("camera", "/")],
        (["worker", "camera"], ["/openeta", "/"]),
    ),
)
def test_isolation_node_snapshot_accepts_rclpy_shapes(raw: object) -> None:
    assert acceptance_isolation._normalise_node_names(raw) == [
        {"name": "camera", "namespace": "/"},
        {"name": "worker", "namespace": "/openeta"},
    ]


def test_isolation_snapshot_removes_only_its_own_probe_instance() -> None:
    nodes = [
        {"name": "openeta_acceptance_probe", "namespace": "/"},
        {"name": "openeta_acceptance_probe", "namespace": "/"},
        {"name": "worker", "namespace": "/openeta"},
    ]
    assert acceptance_isolation._remove_own_node(
        nodes, name="openeta_acceptance_probe", namespace="/"
    ) == [
        {"name": "openeta_acceptance_probe", "namespace": "/"},
        {"name": "worker", "namespace": "/openeta"},
    ]


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


def test_candidate_selection_continues_after_reachable_pregrasp_fails_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Controller:
        @staticmethod
        def plan_pose(_pose, *, timeout_s):
            assert timeout_s == 12.0
            return {"ok": True}

    class Environment:
        controller = Controller()

        def __init__(self) -> None:
            self.attempt = 0
            self.moves = 0

        def reset(self):
            self.attempt += 1
            self.moves = 0
            return observation("READY"), {}

    def observation(reason: str):
        return {
            "objects": [
                {"id": "m3_target", "position": [0.28, -0.10, 0.43]},
                {"id": "m3_distractor", "position": [0.28, 0.12, 0.44]},
            ],
            "metadata": {
                "physical_verification": {
                    "schema_version": "m3_physical_verification_v1",
                    "reason_code": reason,
                }
            },
        }

    environment = Environment()

    def fake_step(env, action, gate):
        ok = True
        reason = "READY"
        if action["action_type"] == "move_to":
            env.moves += 1
            ok = not (env.attempt == 1 and env.moves == 2)
            if env.attempt == 2 and env.moves == 3:
                reason = "TARGET_HELD"
        elif action["action_type"] == "gripper_close":
            reason = "LIFT_REQUIRED"
        gate["actions"].append({"action": dict(action), "receipt": {"ok": ok}})
        return observation(reason)

    monkeypatch.setattr(acceptance, "_step", fake_step)
    gate = {"actions": [], "plan_only_candidates": []}
    selected = acceptance._select_candidate(environment, observation("READY"), (0, 0, 0), gate)

    assert (selected["pitch_degrees"], selected["yaw_degrees"]) == (70, 0)
    assert gate["plan_only_candidates"][0]["blocker_stage"] == "contact_execute"
    assert gate["plan_only_candidates"][1]["status"] == "passed"


def test_acceptance_step_reads_direct_env_namespaced_receipt() -> None:
    observation = {"robot": {"end_effector_pose": {"xyz": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]}}}
    receipt = {"ok": True, "observation": observation}

    class Environment:
        @staticmethod
        def step(_action):
            return observation, 0.0, False, False, {"_openeta_receipt": receipt}

    gate = {"actions": []}
    returned = acceptance._step(Environment(), {"action_type": "gripper_close"}, gate)

    assert returned is observation
    assert gate["actions"][0]["receipt"]["ok"] is True


def test_direct_acceptance_restarts_after_held_candidate_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instances = []

    class Environment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.openeta_control_spec = {"physical_verification": True}
            self.runtime = SimpleNamespace(grasp_center_offset_m=(0.0, 0.0, 0.1))
            self.closed = 0
            instances.append(self)

        def reset(self, *, seed):
            return {"objects": []}, {}

        def close(self):
            self.closed += 1

    module = SimpleNamespace(GazeboDirectEnv=Environment)
    monkeypatch.setitem(sys.modules, "extensions.gazebo.direct_env", module)
    monkeypatch.setattr(acceptance, "_base", lambda _path: {"gates": {}})
    monkeypatch.setattr(acceptance, "_write", lambda *_args: None)
    monkeypatch.setattr(acceptance, "_joint_inventory", lambda: {"sha256": "stable"})
    monkeypatch.setattr(acceptance, "_physical", lambda _observation: {})
    monkeypatch.setattr(
        acceptance,
        "_select_candidate",
        lambda *_args: {"orientation": (0.0, 0.0, 0.0, 1.0)},
    )
    monkeypatch.setattr(acceptance, "_positive_round", lambda *_args: None)
    monkeypatch.setattr(acceptance, "_negative_cases", lambda *_args: None)

    acceptance.run_direct(tmp_path / "m3.json")

    assert [instance.kwargs["seed"] for instance in instances] == [31, 32]
    assert instances[0].closed == 1
    assert instances[1].closed == 1


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


def test_m2_and_m3_share_locks_and_m2_records_actual_test_world() -> None:
    m2_script = (
        ROOT / "extensions/gazebo/ros2_ws/run_m2_robotiq2f85_smoke.sh"
    ).read_text(encoding="utf-8")
    m3_script = (
        ROOT / "extensions/gazebo/ros2_ws/run_m3_pickplace_acceptance.sh"
    ).read_text(encoding="utf-8")

    assert 'LOCK_DIR="/tmp/openeta-acceptance-locks"' in m2_script
    assert 'LOCK_DIR="/tmp/openeta-acceptance-locks"' in m3_script
    assert '--world "${OPENETA_GAZEBO_WORLD}"' in m2_script
    assert 'OPENETA_GAZEBO_WORLD="m2_rm75_robotiq2f85_z_test"' in m2_script
    assert '"${pgid}" != "${current_pgid}"' in m2_script
    assert m2_script.count(
        "env -u OPENETA_GAZEBO_WORLD -u OPENETA_GAZEBO_LAUNCH_ARGUMENTS"
    ) == 2
    assert "export ROS2CLI_DISABLE_DAEMON=1" in m2_script


@pytest.mark.parametrize("driver", (m2_acceptance, acceptance))
def test_finalized_acceptance_report_is_immutable(
    tmp_path: Path, driver: object
) -> None:
    report = tmp_path / "report.json"
    original = {"finished_at_utc": "2026-08-09T00:00:00Z", "sentinel": 1}
    report.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(RuntimeError, match="REPORT_ALREADY_FINALIZED"):
        driver.record_gate(report, "late_gate", "passed", "must be rejected")

    assert json.loads(report.read_text(encoding="utf-8")) == original


def test_m2_finalize_rejects_world_mismatch_before_cleanup(tmp_path: Path) -> None:
    report = tmp_path / "m2.json"
    report.write_text(
        json.dumps(
            {
                "isolation": {
                    "ros_domain_id": 88,
                    "gz_partition": "partition",
                    "mcp_port": 18800,
                    "world": "m2_rm75_robotiq2f85_z_test",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="FINALIZE_ARGUMENT_MISMATCH"):
        m2_acceptance.finalize_isolation_report(
            report,
            domain=88,
            partition="partition",
            port=18800,
            world="m2_rm75_robotiq2f85",
            exit_code=0,
        )


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
