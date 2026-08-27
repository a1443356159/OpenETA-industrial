from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from extensions.gazebo import GazeboProcess, Ros2LaunchProcess
from extensions.gazebo.native_grasp import NativePickPlaceConfig, NativeContactSample
from extensions.gazebo import process as gazebo_process


def test_gazebo_process_lifecycle_when_gz_is_installed() -> None:
    gz = shutil.which("gz") or ("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz" if os.path.exists("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz") else None)
    if gz is None:
        pytest.skip("Gazebo Sim executable is not installed")
    process = GazeboProcess(world="extensions/gazebo/worlds/industrial_fixture.sdf", gz_executable=gz)
    try:
        pid = process.start()
        assert pid > 0
        assert process.running
        process.wait_for_topics(())
    finally:
        process.close()
        process.close()
    assert not process.running


def test_ros_launch_close_waits_for_every_process_group_member() -> None:
    child_program = (
        "import signal,sys,time;"
        "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(0.4), sys.exit(0)));"
        "print('ready', flush=True);"
        "time.sleep(60)"
    )
    leader_program = (
        "import subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_program!r}],"
        "stdout=subprocess.PIPE,text=True);"
        "p.stdout.readline();"
        "print(p.pid, flush=True);"
        "time.sleep(60)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_program],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )
    assert leader.stdout is not None
    child_pid = int(leader.stdout.readline().strip())
    launch = Ros2LaunchProcess(package="unused", launch_file="unused")
    launch._process = leader

    launch.close()

    assert leader.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_ros_launch_inherits_worker_streams_for_case_local_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker log must retain Gazebo's own launch diagnostics."""

    captured = {}

    class Process:
        pid = 12345

        @staticmethod
        def poll():
            return None

    def popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(gazebo_process.subprocess, "Popen", popen)
    launch = Ros2LaunchProcess(
        package="openeta_rm75_robotiq2f85_sim",
        launch_file="gazebo_rgbd.launch.py",
        ros2_executable=sys.executable,
    )

    assert launch.start() == 12345
    assert captured["kwargs"]["stdout"] is None
    assert captured["kwargs"]["stderr"] is None


def test_detachable_joint_wait_ready_requires_the_stock_endpoint_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native grasp must not publish its one-shot detach before plugin endpoints exist."""

    seen = []

    def run(command, **kwargs):
        seen.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(
                [
                    "/openeta/native_grasp/detachable_joint/target/attach",
                    "/openeta/native_grasp/detachable_joint/target/detach",
                    "/openeta/native_grasp/detachable_joint/target/state",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(gazebo_process.subprocess, "run", run)
    control = gazebo_process.GazeboDetachableJointControl(
        gz_executable=sys.executable,
        environment={"GZ_PARTITION": "isolated"},
    )

    control.wait_ready(timeout_s=1.0)

    assert seen[0][0] == [sys.executable, "topic", "-l"]
    assert seen[0][1]["env"] == {"GZ_PARTITION": "isolated"}


def test_detachable_joint_wait_ready_fails_closed_when_an_endpoint_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="/openeta/native_grasp/detachable_joint/target/attach\n",
            stderr="",
        )

    monkeypatch.setattr(gazebo_process.subprocess, "run", run)
    control = gazebo_process.GazeboDetachableJointControl(gz_executable=sys.executable)

    with pytest.raises(gazebo_process.GazeboProcessError, match="NATIVE_GRASP_DETACHABLE_JOINT_NOT_READY"):
        control.wait_ready(timeout_s=0.001)


def test_detachable_joint_command_waits_for_state_listener_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Listener:
        @staticmethod
        def communicate(timeout):
            events.append("ack")
            return ('data: "detached"\n', "")

        @staticmethod
        def poll():
            return 0

    monkeypatch.setattr(
        gazebo_process.subprocess,
        "Popen",
        lambda *_args, **_kwargs: Listener(),
    )

    def run(command, **_kwargs):
        if "-i" in command:
            is_state = command[-1].endswith("/state")
            events.append("state_listener" if is_state else "command_listener")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    (
                        "Publishers [Address, Message Type]:\n"
                        "  tcp://127.0.0.1:12344, gz.msgs.StringMsg\n"
                    )
                    if is_state else "No publishers on topic [command]\n"
                )
                + (
                    "Subscribers [Address, Message Type]:\n"
                    "  tcp://127.0.0.1:12345, google.protobuf.Message\n"
                ),
                stderr="",
            )
        events.append("publish")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(gazebo_process.subprocess, "run", run)
    control = gazebo_process.GazeboDetachableJointControl(
        gz_executable=sys.executable,
    )

    assert control.ensure_detached() == gazebo_process.DetachableJointState.DETACHED
    assert events == ["state_listener", "command_listener", "publish", "ack"]


def test_detachable_joint_proof_uses_the_target_model_world_pose() -> None:
    """The target link is local-to-model in Gazebo Pose_V, not world-space."""

    poses = {
        "gripper_mount_link": (0.10, -0.20, 0.50),
        "target_object": (0.20, -0.20, 0.40),
        "target_link": (0.0, 0.0, 0.0),
    }
    control = gazebo_process.GazeboDetachableJointControl()
    control._state = gazebo_process.DetachableJointState.ATTACHED
    control._world_link_poses = lambda: {
        name: gazebo_process.GazeboNativePose(value, (0.0, 0.0, 0.0, 1.0))
        for name, value in poses.items()
    }  # type: ignore[method-assign]
    control.capture_baseline(settle_duration_s=0.0)

    poses.update(
        {
            "gripper_mount_link": (0.10, -0.20, 0.60),
            "target_object": (0.20, -0.20, 0.50),
        }
    )
    proof = control.child_link_proof()

    assert proof.vertical_displacement_m == pytest.approx(0.10)
    assert proof.capture_relative_translation_m == pytest.approx(0.0)


def test_detachable_joint_baseline_uses_the_final_settled_pose(monkeypatch) -> None:
    control = gazebo_process.GazeboDetachableJointControl()
    control._state = gazebo_process.DetachableJointState.ATTACHED
    readings = iter(
        [
            {"gripper_mount_link": (0.0, 0.0, 0.5), "target_object": (0.0, 0.0, 0.4), "target_link": (0.0, 0.0, 0.0)},
            {"gripper_mount_link": (0.0, 0.0, 0.5), "target_object": (0.0, 0.0, 0.41), "target_link": (0.0, 0.0, 0.0)},
        ]
    )
    control._world_link_poses = lambda: {
        name: gazebo_process.GazeboNativePose(value, (0.0, 0.0, 0.0, 1.0))
        for name, value in next(readings).items()
    }  # type: ignore[method-assign]
    monotonic = iter([0.0, 0.0, 0.02, 0.02])
    monkeypatch.setattr(gazebo_process.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(gazebo_process.time, "sleep", lambda _seconds: None)

    control.capture_baseline(settle_duration_s=0.01, sample_interval_s=0.01)

    assert control._baseline is not None
    assert control._baseline[0] == pytest.approx(0.41)
    assert control._baseline[1] == pytest.approx((0.0, 0.0, -0.09))


def test_native_pose_snapshot_retries_one_incomplete_gazebo_frame() -> None:
    control = gazebo_process.GazeboDetachableJointControl()
    target = gazebo_process.GazeboNativePose(
        (0.2, -0.1, 0.43), (0.0, 0.0, 0.0, 1.0)
    )
    mount = gazebo_process.GazeboNativePose(
        (0.2, -0.1, 0.55), (0.0, 0.0, 0.0, 1.0)
    )
    calls = 0

    def read_pose():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise gazebo_process.GazeboProcessError(
                "NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"
            )
        return target, mount

    control.native_target_mount_poses = read_pose  # type: ignore[method-assign]

    observed_target, observed_mount, attempts = (
        control.native_target_mount_poses_with_retry(max_attempts=2)
    )

    assert (observed_target, observed_mount) == (target, mount)
    assert attempts == 2
    assert control._last_native_pose_read_attempt_count == 2


def test_native_pose_snapshot_stops_after_one_transport_retry() -> None:
    control = gazebo_process.GazeboDetachableJointControl()
    calls = 0

    def read_pose():
        nonlocal calls
        calls += 1
        raise gazebo_process.GazeboProcessError(
            "NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"
        )

    control.native_target_mount_poses = read_pose  # type: ignore[method-assign]

    with pytest.raises(
        gazebo_process.GazeboProcessError,
        match="NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE",
    ):
        control.native_target_mount_poses_with_retry(max_attempts=2)

    assert calls == 2
    assert control._last_native_pose_read_attempt_count == 2


def test_detached_placement_sampling_retries_one_incomplete_pose_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = gazebo_process.GazeboDetachableJointControl()
    control._state = gazebo_process.DetachableJointState.DETACHED
    target = gazebo_process.GazeboNativePose(
        (0.48, -0.10, 0.43), (0.0, 0.0, 0.0, 1.0)
    )
    mount = gazebo_process.GazeboNativePose(
        (0.48, -0.10, 0.55), (0.0, 0.0, 0.0, 1.0)
    )
    reads = 0

    def read_pose():
        nonlocal reads
        reads += 1
        if reads == 1:
            raise gazebo_process.GazeboProcessError(
                "NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"
            )
        return target, mount

    control.native_target_mount_poses = read_pose  # type: ignore[method-assign]
    sampled_times = iter([0.0, 0.02])
    monkeypatch.setattr(
        gazebo_process.time, "monotonic", lambda: next(sampled_times)
    )
    monkeypatch.setattr(gazebo_process.time, "sleep", lambda _seconds: None)

    samples = control.sample_detached_target_poses(
        duration_s=0.01,
        interval_s=0.01,
    )

    assert reads == 3
    assert [sample.xyz for sample in samples] == [target.xyz, target.xyz]


def test_detachable_joint_rigid_rotation_has_zero_relative_translation_drift() -> None:
    """A rigid object rotates with its EEF without being misclassified as slip."""

    control = gazebo_process.GazeboDetachableJointControl()
    control._state = gazebo_process.DetachableJointState.ATTACHED
    poses = {
        "gripper_mount_link": gazebo_process.GazeboNativePose(
            (0.10, -0.20, 0.50), (0.0, 0.0, 0.0, 1.0)
        ),
        "target_object": gazebo_process.GazeboNativePose(
            (0.26, -0.20, 0.50), (0.0, 0.0, 0.0, 1.0)
        ),
        "target_link": gazebo_process.GazeboNativePose(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        ),
    }
    control._world_link_poses = lambda: dict(poses)  # type: ignore[method-assign]
    control.capture_baseline(settle_duration_s=0.0)

    # Rotate the parent and its 160 mm attachment lever arm together by 90°.
    half_sqrt = 2**-0.5
    poses["gripper_mount_link"] = gazebo_process.GazeboNativePose(
        (0.30, -0.10, 0.60), (0.0, 0.0, half_sqrt, half_sqrt)
    )
    poses["target_object"] = gazebo_process.GazeboNativePose(
        (0.30, 0.06, 0.60), (0.0, 0.0, half_sqrt, half_sqrt)
    )

    proof = control.child_link_proof()

    assert proof.capture_relative_translation_m == pytest.approx(0.0, abs=1e-9)


def test_detachable_joint_real_parent_frame_slip_exceeds_ten_mm() -> None:
    """A true 20 mm child displacement in the EEF frame remains detectable."""

    control = gazebo_process.GazeboDetachableJointControl()
    control._state = gazebo_process.DetachableJointState.ATTACHED
    poses = {
        "gripper_mount_link": gazebo_process.GazeboNativePose(
            (0.10, -0.20, 0.50), (0.0, 0.0, 0.0, 1.0)
        ),
        "target_object": gazebo_process.GazeboNativePose(
            (0.10, -0.20, 0.34), (0.0, 0.0, 0.0, 1.0)
        ),
        "target_link": gazebo_process.GazeboNativePose(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        ),
    }
    control._world_link_poses = lambda: dict(poses)  # type: ignore[method-assign]
    control.capture_baseline(settle_duration_s=0.0)
    poses["target_object"] = gazebo_process.GazeboNativePose(
        (0.10, -0.20, 0.32), (0.0, 0.0, 0.0, 1.0)
    )

    proof = control.child_link_proof()

    assert proof.capture_relative_translation_m == pytest.approx(0.02)
    assert (
        proof.capture_relative_translation_m
        > NativePickPlaceConfig().maximum_capture_relative_translation_m
    )


def test_native_pose_parser_preserves_and_normalizes_quaternion() -> None:
    poses = gazebo_process.GazeboDetachableJointControl._pose_records(
        '''
pose {
  name: "target_object"
  position { x: 0.2 y: -0.1 z: 0.43 }
  orientation { x: 0.0 y: 0.0 z: 1.0 w: 1.0 }
}
'''
    )

    assert poses["target_object"].xyz == (0.2, -0.1, 0.43)
    assert poses["target_object"].quat_xyzw == pytest.approx(
        (0.0, 0.0, 2**-0.5, 2**-0.5)
    )


def test_detachable_joint_proof_rejects_a_noncanonical_child_link_frame() -> None:
    control = gazebo_process.GazeboDetachableJointControl()
    control._state = gazebo_process.DetachableJointState.ATTACHED
    control._world_link_poses = lambda: {
        "gripper_mount_link": gazebo_process.GazeboNativePose(
            (0.0, 0.0, 0.50), (0.0, 0.0, 0.0, 1.0)
        ),
        "target_object": gazebo_process.GazeboNativePose(
            (0.0, 0.0, 0.40), (0.0, 0.0, 0.0, 1.0)
        ),
        "target_link": gazebo_process.GazeboNativePose(
            (0.0, 0.0, 0.01), (0.0, 0.0, 0.0, 1.0)
        ),
    }  # type: ignore[method-assign]

    with pytest.raises(gazebo_process.GazeboProcessError, match="NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"):
        control.capture_baseline()


def test_native_contact_window_accepts_terminal_hold_without_post_result_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stable end-of-close hold needs no new post-result sensor message."""

    now = 100.0
    monkeypatch.setattr(gazebo_process.time, "monotonic", lambda: now)
    window = gazebo_process.GazeboNativeContactWindow(timeout_s=0.1)
    window._armed = True
    target = "target_object::target_link::target_collision"
    window._samples.append(
        NativeContactSample(
            "left", 9.4, 90.0,
            ("robot::robotiq_85_left_finger_tip_link::collision", target),
        )
    )
    for side in ("left", "right"):
        tip = f"robot::robotiq_85_{side}_finger_tip_link::collision"
        for timestamp in (9.62, 9.76, 9.91):
            window._samples.append(
                NativeContactSample(side, timestamp, 99.9, (tip, target))
            )

    result = window.evaluate(
        close_completed_sim_time_s=10.0,
        config=NativePickPlaceConfig(contact_freshness_s=2.0),
    )

    assert result.accepted is True
    assert result.left_sample_count == 3
    assert result.right_sample_count == 3
    assert result.evidence["proof_boundary"] == (
        "terminal_bilateral_hold_before_close_result"
    )
    assert result.evidence["close_completed_sim_time_s"] == 10.0
    assert result.evidence["sample_freshness_basis"] == (
        "closed_gazebo_sim_time_window"
    )


def test_native_contact_window_uses_gazebo_time_under_slow_real_time_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU/UI wall time must not invalidate a closed simulator-time proof."""

    monkeypatch.setattr(gazebo_process.time, "monotonic", lambda: 100.0)
    window = gazebo_process.GazeboNativeContactWindow(timeout_s=0.0)
    window._armed = True
    target = "target_object::target_link::target_collision"
    for side in ("left", "right"):
        tip = f"robot::robotiq_85_{side}_finger_tip_link::collision"
        for timestamp in (9.62, 9.76, 9.91):
            window._samples.append(
                NativeContactSample(side, timestamp, 90.0, (tip, target))
            )

    result = window.evaluate(
        close_completed_sim_time_s=10.0,
        config=NativePickPlaceConfig(contact_freshness_s=2.0),
    )

    assert result.accepted is True
    assert result.reason_code.name == "CONTACT_TARGET_CONFIRMED"
    assert result.left_sample_count == 3
    assert result.right_sample_count == 3
    assert result.evidence["sample_freshness_basis"] == (
        "closed_gazebo_sim_time_window"
    )


def test_native_contact_window_rejects_a_transient_terminal_touch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(gazebo_process.time, "monotonic", lambda: now)
    window = gazebo_process.GazeboNativeContactWindow(timeout_s=0.0)
    window._armed = True
    target = "target_object::target_link::target_collision"
    for side in ("left", "right"):
        tip = f"robot::robotiq_85_{side}_finger_tip_link::collision"
        for timestamp in (9.80, 9.82, 9.84):
            window._samples.append(
                NativeContactSample(side, timestamp, 99.9, (tip, target))
            )

    result = window.evaluate(
        close_completed_sim_time_s=10.0,
        config=NativePickPlaceConfig(contact_freshness_s=2.0),
    )

    assert result.accepted is False
    assert result.reason_code.name == "CONTACT_WINDOW_TOO_SHORT"


def test_native_contact_window_proves_a_bounded_post_close_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late stable pad contact may prove itself while close stays commanded."""

    monkeypatch.setattr(gazebo_process.time, "monotonic", lambda: 100.0)
    window = gazebo_process.GazeboNativeContactWindow(timeout_s=0.0)
    window._armed = True
    target = "target_object::target_link::target_collision"
    for side in ("left", "right"):
        tip = f"robot::robotiq_85_{side}_finger_tip_link::collision"
        for timestamp in (9.95, 10.02, 10.08, 10.12):
            window._samples.append(
                NativeContactSample(side, timestamp, 100.0, (tip, target))
            )

    result = window.evaluate(
        close_completed_sim_time_s=10.0,
        config=NativePickPlaceConfig(contact_post_close_hold_s=0.12),
    )

    assert result.accepted is True
    assert result.left_span_s == pytest.approx(0.17)
    assert result.right_span_s == pytest.approx(0.17)
    assert result.evidence["proof_boundary"] == (
        "bounded_post_close_bilateral_hold"
    )
    assert result.evidence["post_close_hold_completed"] is True
