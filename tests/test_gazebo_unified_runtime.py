from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from extensions.gazebo.deployment import GazeboDeploymentConfig
from extensions.gazebo.direct_env import GazeboDirectEnv
from extensions.gazebo.observation import GazeboObservationError
from extensions.gazebo.profiles import gazebo_profile, gazebo_profiles
from extensions.gazebo.runtime import GazeboRuntime
from extensions.gazebo import runtime as gazebo_runtime
from sim.mcp_server.worker_mgr import (
    BenchWorkerHandle,
    BenchWorkerManager,
    _gazebo_ros_abi_environment,
)


def _deployment() -> GazeboDeploymentConfig:
    return GazeboDeploymentConfig(
        ros_domain_id=17, gz_partition="test-partition", ros2_executable="ros2",
        gz_executable="gz", process_environment={"ROS_DOMAIN_ID": "17"},
    )


class _Launch:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = 0
        self.closed = 0

    def start(self):
        self.started += 1

    def close(self):
        self.closed += 1


class _Camera:
    def __init__(self, config, **_kwargs):
        self.config = config
        self.started = 0
        self.closed = 0
        self.sequence = 0

    def start(self):
        self.started += 1

    def capture(self, **_kwargs):
        self.sequence += 1
        return CameraFrame(
            frame_id=self.config.frame_id, role=self.config.role,
            rgb=[[[0, 0, 0]]], depth=[[1.0]],
            intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            extrinsics=dict(self.config.extrinsics), timestamp_s=float(self.sequence),
        )

    def close(self):
        self.closed += 1


class _RecordingCamera(_Camera):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.capture_arguments = []

    def capture(self, **kwargs):
        self.capture_arguments.append(dict(kwargs))
        return super().capture(**kwargs)


class _World:
    def __init__(self):
        self.resets = []

    def reset_all(self, *, seed=None):
        self.resets.append(("all", seed))

    def reset_models(self, *, seed=None):
        self.resets.append(("models", seed))


class _ReadyWorld(_World):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.deadlines = []

    def wait_ready(self, *, timeout_s):
        self.events.append("world_control_ready")
        self.deadlines.append(timeout_s)


class _NativeGraspWorld(_World):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.poses = []

    def set_paused(self, paused):
        self.events.append(("paused", bool(paused)))

    def set_model_pose(self, model_name, xyz):
        self.events.append(("pose", model_name))
        self.poses.append((model_name, xyz))


class _Receipt:
    def __init__(self, payload):
        self.payload = dict(payload)

    def to_dict(self):
        return dict(self.payload)


class _ResetController:
    def __init__(self, receipts):
        self.receipts = list(receipts)
        self.actions = []

    def execute(self, action):
        self.actions.append(dict(action))
        return _Receipt(self.receipts.pop(0))

    @staticmethod
    def state_provider():
        return RobotState()

    def sync_planning_scene_reset(self, config):
        self.actions.append({"action_type": "planning_scene_reset", "target": config.target_id})
        return 1

    def sync_planning_scene_empty(self):
        self.actions.append({"action_type": "planning_scene_empty"})
        return 1


def test_all_profiles_use_the_same_direct_env_type_without_starting_runtime() -> None:
    for profile in gazebo_profiles().values():
        runtime = SimpleNamespace(started=False, close=lambda: None)
        if profile.unavailable_reason:
            with pytest.raises(RuntimeError, match=profile.unavailable_reason):
                GazeboDirectEnv(profile=profile, deployment=_deployment(), runtime=runtime)
            continue
        env = GazeboDirectEnv(profile=profile, deployment=_deployment(), runtime=runtime)
        assert type(env) is GazeboDirectEnv
        assert runtime.started is False


def test_direct_observe_publishes_current_planning_scene_revision() -> None:
    runtime = SimpleNamespace(
        controller=SimpleNamespace(planning_scene=SimpleNamespace(revision=7)),
        observe=lambda: EnvObservation(
            task="test",
            cameras=[],
            robot=RobotState(),
            metadata={"scene_epoch": 3},
        ),
        close=lambda: None,
        started=False,
    )
    env = GazeboDirectEnv(
        profile=gazebo_profile("rm75_robotiq2f85_control"),
        deployment=_deployment(),
        runtime=runtime,
    )

    observation = env.observe()

    assert observation["metadata"]["scene_epoch"] == 3
    assert observation["metadata"]["planning_scene_revision"] == 7


def test_runtime_is_lazy_starts_once_observes_fresh_and_closes_idempotently() -> None:
    made_launch = []
    made_cameras = []

    def launch_factory(**kwargs):
        value = _Launch(**kwargs)
        made_launch.append(value)
        return value

    def camera_factory(config, **kwargs):
        value = _Camera(config, **kwargs)
        made_cameras.append(value)
        return value

    world = _World()
    runtime = GazeboRuntime(
        _deployment(), gazebo_profile("rgbd_observation"), launch_factory=launch_factory,
        camera_factory=camera_factory, world_control=world,
    )
    assert runtime.started is False and not made_launch and not made_cameras
    first = runtime.reset(seed=3)
    second = runtime.reset(seed=4)
    fresh = runtime.observe()
    assert runtime.start_count == 1
    assert world.resets == [("all", 3), ("all", 4)]
    assert first.cameras[0].timestamp_s < second.cameras[0].timestamp_s < fresh.cameras[0].timestamp_s
    runtime.close()
    runtime.close()
    assert made_launch[0].closed == 1
    assert made_cameras[0].closed == 1


def test_runtime_action_observation_uses_ros_completion_barrier_without_callback_race() -> None:
    """A post-action frame may be queued before ``execute`` returns to Python.

    Camera headers and the controller completion receipt use ROS simulation
    time, whereas callback arrival uses host monotonic time.  The latter must
    not reject an otherwise ordered image.
    """
    profile = gazebo_profile("rm75_robotiq2f85_control")
    camera = _RecordingCamera(profile.cameras[0])
    runtime = GazeboRuntime(_deployment(), profile, world_control=_World())
    runtime.started = True
    runtime._cameras = [camera]
    runtime.controller = _ResetController([
        {"ok": True, "action_completed_ros_time_s": 42.0},
    ])

    _observation, receipt = runtime.execute({"action_type": "move_to"})

    assert receipt["action_completed_ros_time_s"] == 42.0
    assert camera.capture_arguments == [{"timeout_s": pytest.approx(30.0), "min_timestamp_s": 42.0, "min_received_monotonic_s": None}]


def test_read_only_motion_qualification_reuses_frozen_observation() -> None:
    profile = gazebo_profile("rm75_robotiq2f85_control")
    camera = _RecordingCamera(profile.cameras[0])
    runtime = GazeboRuntime(_deployment(), profile, world_control=_World())
    runtime.started = True
    runtime._cameras = [camera]
    cached = runtime.observe()
    runtime.controller = _ResetController(
        [{"ok": True, "execution_started": False, "results": []}]
    )

    observation, receipt = runtime.execute(
        {"action_type": "qualify_motion_candidates", "candidates": []}
    )

    assert observation is cached
    assert camera.capture_arguments == [
        {
            "timeout_s": pytest.approx(30.0),
            "min_timestamp_s": None,
            "min_received_monotonic_s": None,
        }
    ]
    assert receipt["observation_reused"] is True
    assert receipt["observation_reuse_reason"] == "read_only_motion_qualification"
    assert receipt["observation_scene_epoch"] == 0


def test_read_only_motion_qualification_without_frozen_observation_fails_closed() -> None:
    profile = gazebo_profile("rm75_robotiq2f85_control")
    runtime = GazeboRuntime(_deployment(), profile, world_control=_World())
    runtime.started = True
    runtime.controller = _ResetController(
        [{"ok": True, "execution_started": False, "results": []}]
    )

    with pytest.raises(
        RuntimeError, match="READ_ONLY_ACTION_OBSERVATION_UNAVAILABLE"
    ):
        runtime.execute(
            {"action_type": "qualify_motion_candidates", "candidates": []}
        )


def test_runtime_samples_fresh_robot_state_before_slow_camera_capture() -> None:
    events = []
    profile = gazebo_profile("rm75_robotiq2f85_control")

    class Camera(_Camera):
        def capture(self, **kwargs):
            events.append("camera")
            return super().capture(**kwargs)

    class Controller(_ResetController):
        def state_provider(self):
            events.append("robot")
            return RobotState()

    runtime = GazeboRuntime(_deployment(), profile, world_control=_World())
    runtime.started = True
    runtime._cameras = [Camera(profile.cameras[0])]
    runtime.controller = Controller(
        [{"ok": True, "action_completed_ros_time_s": 42.0}]
    )

    runtime.execute({"action_type": "move_to"})

    assert events == ["robot", "camera"]


def test_runtime_retries_only_observation_after_camera_transport_timeout() -> None:
    profile = gazebo_profile("rm75_robotiq2f85_control")

    class Camera(_Camera):
        def __init__(self, config):
            super().__init__(config)
            self.attempts = 0

        def capture(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise GazeboObservationError("camera transport timeout")
            return super().capture(**kwargs)

    camera = Camera(profile.cameras[0])
    controller = _ResetController(
        [{"ok": True, "action_completed_ros_time_s": 42.0}]
    )
    runtime = GazeboRuntime(_deployment(), profile, world_control=_World())
    runtime.started = True
    runtime._cameras = [camera]
    runtime.controller = controller

    observation, receipt = runtime.execute({"action_type": "gripper_close"})

    assert observation.cameras[0].timestamp_s == 1.0
    assert camera.attempts == 2
    assert controller.actions == [{"action_type": "gripper_close"}]
    assert receipt["observation_refresh_retry_count"] == 1
    assert receipt["observation_refresh_retry_reason"] == "camera_transport_timeout"


def test_runtime_propagates_second_camera_transport_timeout_without_repeating_action() -> None:
    profile = gazebo_profile("rm75_robotiq2f85_control")

    class Camera(_Camera):
        def __init__(self, config):
            super().__init__(config)
            self.attempts = 0

        def capture(self, **kwargs):
            self.attempts += 1
            raise GazeboObservationError("camera transport timeout")

    camera = Camera(profile.cameras[0])
    controller = _ResetController(
        [{"ok": True, "action_completed_ros_time_s": 42.0}]
    )
    runtime = GazeboRuntime(_deployment(), profile, world_control=_World())
    runtime.started = True
    runtime._cameras = [camera]
    runtime.controller = controller

    with pytest.raises(GazeboObservationError, match="camera transport timeout"):
        runtime.execute({"action_type": "gripper_close"})

    assert camera.attempts == 2
    assert controller.actions == [{"action_type": "gripper_close"}]


def test_runtime_waits_for_world_control_before_its_first_rgbd_reset() -> None:
    events = []

    class Launch(_Launch):
        def start(self):
            super().start()
            events.append("launch")

    world = _ReadyWorld(events)
    runtime = GazeboRuntime(
        _deployment(), gazebo_profile("rgbd_observation"),
        launch_factory=lambda **kwargs: Launch(**kwargs),
        camera_factory=lambda config, **kwargs: _Camera(config, **kwargs),
        world_control=world,
    )

    runtime.reset(seed=9)

    assert events == ["launch", "world_control_ready"]
    assert world.deadlines and world.deadlines[0] > 0
    assert world.resets == [("all", 9)]
    runtime.close()


def test_runtime_subscribes_to_cameras_only_after_launch_bridge_readiness() -> None:
    """Headless RGB-D runtime must not create DDS subscriptions before its publishers."""

    events = []

    class Launch(_Launch):
        def start(self):
            super().start()
            events.append("launch")

    class Camera(_Camera):
        def start(self):
            super().start()
            events.append("camera_subscribe")

    world = _ReadyWorld(events)
    runtime = GazeboRuntime(
        _deployment(),
        gazebo_profile("rgbd_observation"),
        launch_factory=lambda **kwargs: Launch(**kwargs),
        camera_factory=lambda config, **kwargs: Camera(config, **kwargs),
        world_control=world,
    )

    runtime.reset(seed=5)

    assert events == ["launch", "world_control_ready", "camera_subscribe"]
    runtime.close()


def test_runtime_waits_for_all_configured_ros_camera_publishers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = GazeboRuntime(_deployment(), gazebo_profile("rgbd_observation"))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(
                [
                    "/rgbd_camera/image",
                    "/rgbd_camera/depth_image",
                    "/rgbd_camera/camera_info",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(gazebo_runtime.subprocess, "run", run)

    runtime._wait_for_camera_publishers(deadline=gazebo_runtime.time.monotonic() + 1.0)

    assert calls[0][0][-2:] == ["topic", "list"]
    assert calls[0][1]["env"] == {"ROS_DOMAIN_ID": "17"}


def test_runtime_reset_retries_only_one_transient_gripper_timeout() -> None:
    profile = gazebo_profile("rm75_robotiq2f85_control")
    world = _World()
    runtime = GazeboRuntime(_deployment(), profile, world_control=world)
    runtime.started = True
    runtime._cameras = [_Camera(profile.cameras[0])]
    controller = _ResetController([
        {"ok": False, "error_code": "GRIPPER_TIMEOUT"},
        {"ok": True, "action_completed_ros_time_s": 4.0},
    ])
    runtime.controller = controller

    runtime.reset(seed=7)

    assert world.resets == [("models", 7)]
    assert controller.actions == [
        {"action_type": "planning_scene_empty"},
        {"action_type": "gripper_open"},
        {"action_type": "gripper_open"},
    ]


def test_runtime_reset_does_not_retry_non_transient_gripper_failure() -> None:
    profile = gazebo_profile("rm75_robotiq2f85_control")
    runtime = GazeboRuntime(_deployment(), profile, world_control=_World())
    runtime.started = True
    runtime._cameras = [_Camera(profile.cameras[0])]
    controller = _ResetController([{"ok": False, "error_code": "GRIPPER_FAILED"}])
    runtime.controller = controller

    with pytest.raises(RuntimeError, match="GRIPPER_FAILED"):
        runtime.reset(seed=7)
    assert controller.actions == [
        {"action_type": "planning_scene_empty"},
        {"action_type": "gripper_open"},
    ]


def test_native_grasp_runtime_detaches_while_paused_before_controller_ready_and_reset_poses() -> None:
    events = []

    class Attachment:
        state = "unknown"

        def wait_ready(self, *, timeout_s):
            assert timeout_s > 0
            events.append(("attachment_ready",))

        def ensure_detached(self, *, require_ack):
            assert require_ack is True
            self.state = "detached"
            events.append(("detach_ack",))

    class Controller(_ResetController):
        def __init__(self):
            super().__init__([{"ok": True, "action_completed_ros_time_s": 1.0}])

        def wait_ready(self, _timeout):
            events.append(("controller_ready",))

        def close(self):
            events.append(("controller_close",))

    class ControllerFactory:
        def create(self, *_args, **_kwargs):
            events.append(("controller_create",))
            return Controller()

    class Launch(_Launch):
        def start(self):
            super().start()
            events.append(("launch",))

    world = _NativeGraspWorld(events)
    runtime = GazeboRuntime(
        _deployment(),
        gazebo_profile("rm75_robotiq2f85_pickplace"),
        launch_factory=lambda **kwargs: Launch(**kwargs),
        camera_factory=lambda config, **kwargs: _Camera(config, **kwargs),
        controller_factory=ControllerFactory(),
        world_control=world,
        attachment_factory=lambda **_kwargs: Attachment(),
    )

    runtime.reset(seed=11)

    first_detach = events.index(("detach_ack",))
    assert events[:first_detach] == [("launch",), ("attachment_ready",), ("paused", True)]
    assert events[first_detach + 1] == ("paused", False)
    assert events.index(("controller_ready",)) > events.index(("paused", False))
    # The fresh paused launch, rather than model_only reset, restores the SDF
    # object poses and gives the stock plugin a real detached transition.
    assert world.resets == []
    assert world.poses == []
    # A known detached state must not request an impossible no-op ACK at close.
    runtime.close()
    assert events.count(("detach_ack",)) == 1


def test_native_grasp_runtime_recreates_the_paused_world_for_a_second_reset() -> None:
    events = []
    launches = []

    class Attachment:
        state = "unknown"

        def wait_ready(self, *, timeout_s):
            assert timeout_s > 0
            events.append("attachment_ready")

        def ensure_detached(self, *, require_ack):
            assert require_ack is True
            self.state = "detached"
            events.append("detach_ack")

    class Controller(_ResetController):
        def __init__(self):
            super().__init__([{"ok": True, "action_completed_ros_time_s": 1.0}])

        def wait_ready(self, _timeout):
            events.append("controller_ready")

        def close(self):
            events.append("controller_close")

    class ControllerFactory:
        def create(self, *_args, **_kwargs):
            return Controller()

    class Launch(_Launch):
        def start(self):
            super().start()
            events.append("launch")

    def launch_factory(**kwargs):
        launch = Launch(**kwargs)
        launches.append(launch)
        return launch

    world = _NativeGraspWorld(events)
    runtime = GazeboRuntime(
        _deployment(),
        gazebo_profile("rm75_robotiq2f85_pickplace"),
        launch_factory=launch_factory,
        camera_factory=lambda config, **kwargs: _Camera(config, **kwargs),
        controller_factory=ControllerFactory(),
        world_control=world,
        attachment_factory=lambda **_kwargs: Attachment(),
    )

    runtime.reset(seed=11)
    runtime.reset(seed=12)

    assert len(launches) == 2
    assert launches[0].closed == 1
    assert events.count("detach_ack") == 2
    assert world.resets == []
    runtime.close()


def test_deployment_environment_is_snapshotted_and_child_environment_is_explicit() -> None:
    source = {
        "ROS_DOMAIN_ID": "23", "GZ_PARTITION": "locked",
        "OPENETA_GAZEBO_LAUNCH_ARGUMENTS": '["rviz:=false"]',
        "OPENETA_GAZEBO_CAMERA_EXTRINSICS": '{"camera_frame":"opencv"}',
        "OPENETA_ACCEPTANCE_SCENE": "narrow-pick",
    }
    config = GazeboDeploymentConfig.from_environment(source)
    source["ROS_DOMAIN_ID"] = "42"
    assert config.ros_domain_id == 23
    assert config.gz_partition == "locked"
    assert config.launch_arguments == ("rviz:=false",)
    assert config.startup_timeout_s == 45.0
    assert config.process_environment["ROS2CLI_NO_DAEMON"] == "1"
    assert config.process_environment["OPENETA_ACCEPTANCE_SCENE"] == "narrow-pick"


def test_deployment_sanitizes_host_ruby_for_the_vendor_gz_wrapper() -> None:
    source = {
        "ROS_DOMAIN_ID": "24",
        "GZ_PARTITION": "ruby-isolated",
        "GZ_SIM_RESOURCE_PATH": "/opt/ros/jazzy/share",
        "PYTHONPATH": "/opt/ros/jazzy/lib/python3.12/site-packages",
        "PATH": "/root/autodl-tmp/env/ros2_jazzy/bin:/opt/ros/jazzy/bin:/usr/bin",
        "RUBYOPT": "-I/root/autodl-tmp/env/ros2_jazzy/gems",
        "RUBYLIB": "/root/autodl-tmp/env/ros2_jazzy/gems",
        "GEM_HOME": "/root/autodl-tmp/env/ros2_jazzy/gems",
        "BUNDLE_GEMFILE": "/root/autodl-tmp/env/ros2_jazzy/Gemfile",
        "ROS_LOCALHOST_ONLY": "1",
        "ROS_STATIC_PEERS": "127.0.0.1",
    }

    config = GazeboDeploymentConfig.from_environment(source)

    child = config.process_environment
    assert child["PATH"] == (
        "/usr/bin:/root/autodl-tmp/env/ros2_jazzy/bin:/opt/ros/jazzy/bin"
    )
    assert not any(name.startswith(("RUBY", "GEM", "BUNDLE")) for name in child)
    assert child["PYTHONPATH"] == source["PYTHONPATH"]
    assert child["GZ_SIM_RESOURCE_PATH"] == source["GZ_SIM_RESOURCE_PATH"]
    assert child["ROS_DOMAIN_ID"] == "24"
    assert child["GZ_PARTITION"] == "ruby-isolated"
    assert "ROS_LOCALHOST_ONLY" not in child
    assert "ROS_STATIC_PEERS" not in child
    assert child["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"


def test_gazebo_worker_uses_only_the_sourced_ros_python_and_native_libraries(
    tmp_path: Path,
) -> None:
    """A host Python's second ROS build must not mix with /opt ROS ABI libs."""

    active = tmp_path / "opt" / "ros" / "jazzy"
    overlay = tmp_path / "workspace" / "install"
    source_root = tmp_path / "source" / "selected-worktree"
    foreign = tmp_path / "host-python" / "ros2_jazzy"
    active_python = active / "lib" / "python3.12" / "site-packages"
    overlay_python = overlay / "lib" / "python3.12" / "site-packages"
    foreign_python = foreign / "lib" / "python3.12" / "site-packages"
    for path in (
        active_python / "rclpy",
        active_python / "sensor_msgs",
        overlay_python / "rclpy",
        overlay_python / "sensor_msgs",
        foreign_python / "rclpy",
        foreign_python / "sensor_msgs",
        active / "lib",
        overlay / "lib",
        foreign / "lib",
        foreign / "opt" / "rviz_ogre_vendor" / "lib",
        source_root / "extensions",
    ):
        path.mkdir(parents=True, exist_ok=True)
    environment = _gazebo_ros_abi_environment(
        {
            "ROS_DISTRO": "jazzy",
            "OPENETA_GAZEBO_SYSTEM_ROS_PREFIX": str(active),
            "OPENETA_GAZEBO_OVERLAY": str(overlay),
            "OPENETA_GAZEBO_SOURCE_ROOT": str(source_root),
            "PYTHONPATH": os.pathsep.join(
                (
                    str(source_root),
                    str(overlay_python),
                    str(active_python),
                    str(foreign_python),
                )
            ),
            "LD_LIBRARY_PATH": os.pathsep.join(
                (
                    str(foreign / "opt" / "rviz_ogre_vendor" / "lib"),
                    str(foreign / "lib"),
                    str(overlay / "lib"),
                    str(active / "lib"),
                    "/usr/local/cuda/lib64",
                )
            ),
            "AMENT_PREFIX_PATH": os.pathsep.join((str(overlay), str(active), str(foreign))),
        }
    )

    assert environment["PYTHONPATH"] == os.pathsep.join(
        (str(source_root), str(overlay_python), str(active_python))
    )
    assert environment["LD_LIBRARY_PATH"] == os.pathsep.join(
        (str(overlay / "lib"), str(active / "lib"), "/usr/local/cuda/lib64")
    )
    assert environment["AMENT_PREFIX_PATH"] == os.pathsep.join((str(overlay), str(active)))


def test_manager_rejects_second_gazebo_and_retires_worker_on_release() -> None:
    manager = object.__new__(BenchWorkerManager)
    import threading
    manager._lock = threading.RLock()
    stopped = []
    worker = BenchWorkerHandle(
        bench="gazebo", port=1, process=None, base_url="http://worker", env_count=1
    )
    worker.stop = lambda **kwargs: stopped.append(kwargs)  # type: ignore[method-assign]
    manager._pools = {"gazebo": [worker]}
    try:
        manager.acquire_worker("gazebo")
    except RuntimeError as exc:
        assert str(exc) == "GAZEBO_CAPACITY_EXHAUSTED"
    else:
        raise AssertionError("second Gazebo environment was accepted")
    manager.release_worker(worker.base_url)
    assert stopped == [{"wait": True}]
    assert manager._pools["gazebo"] == []


def test_architecture_gate_keeps_generic_worker_and_runtime_free_of_backdoors() -> None:
    root = Path(__file__).parents[1]
    generic = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("sim/bench_worker.py", "sim/mcp_server/server.py")
    )
    for forbidden in (
        '_backend", "") == "gazebo"', "refresh_observation",
        "_openeta_structured_actions",
    ):
        assert forbidden not in generic
    runtime = (root / "extensions/gazebo/runtime.py").read_text(encoding="utf-8")
    assert "os.environ" not in runtime
    assert "os.getenv" not in runtime
    assert "time.sleep" not in runtime
    worker = (root / "extensions/gazebo/worker.py").read_text(encoding="utf-8")
    assert "class Gazebo" not in worker
