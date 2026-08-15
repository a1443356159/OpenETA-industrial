from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from adapter.protocol import CameraFrame, RobotState
from extensions.gazebo.deployment import GazeboDeploymentConfig
from extensions.gazebo.direct_env import GazeboDirectEnv
from extensions.gazebo.profiles import gazebo_profile, gazebo_profiles
from extensions.gazebo.runtime import GazeboRuntime
from sim.mcp_server.worker_mgr import BenchWorkerHandle, BenchWorkerManager


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


class _World:
    def __init__(self):
        self.resets = []

    def reset_all(self, *, seed=None):
        self.resets.append(("all", seed))

    def reset_models(self, *, seed=None):
        self.resets.append(("models", seed))


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


def test_all_profiles_use_the_same_direct_env_type_without_starting_runtime() -> None:
    for profile in gazebo_profiles().values():
        runtime = SimpleNamespace(started=False, close=lambda: None)
        env = GazeboDirectEnv(profile=profile, deployment=_deployment(), runtime=runtime)
        assert type(env) is GazeboDirectEnv
        assert runtime.started is False


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
        _deployment(), gazebo_profile("m1"), launch_factory=launch_factory,
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


def test_runtime_reset_retries_only_one_transient_gripper_timeout() -> None:
    profile = gazebo_profile("m2_robotiq2f85")
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
        {"action_type": "gripper_open"},
        {"action_type": "gripper_open"},
    ]


def test_runtime_reset_does_not_retry_non_transient_gripper_failure() -> None:
    profile = gazebo_profile("m2_robotiq2f85")
    runtime = GazeboRuntime(_deployment(), profile, world_control=_World())
    runtime.started = True
    runtime._cameras = [_Camera(profile.cameras[0])]
    controller = _ResetController([{"ok": False, "error_code": "GRIPPER_FAILED"}])
    runtime.controller = controller

    with pytest.raises(RuntimeError, match="GRIPPER_FAILED"):
        runtime.reset(seed=7)
    assert controller.actions == [{"action_type": "gripper_open"}]


def test_deployment_environment_is_snapshotted_and_child_environment_is_explicit() -> None:
    source = {
        "ROS_DOMAIN_ID": "23", "GZ_PARTITION": "locked",
        "OPENETA_GAZEBO_LAUNCH_ARGUMENTS": '["rviz:=false"]',
        "OPENETA_GAZEBO_CAMERA_EXTRINSICS": '{"camera_frame":"opencv"}',
    }
    config = GazeboDeploymentConfig.from_environment(source)
    source["ROS_DOMAIN_ID"] = "42"
    assert config.ros_domain_id == 23
    assert config.gz_partition == "locked"
    assert config.launch_arguments == ("rviz:=false",)
    assert config.process_environment["ROS2CLI_NO_DAEMON"] == "1"


def test_deployment_rejects_retired_m3_attachment_mode() -> None:
    with pytest.raises(ValueError, match="OPENETA_M3_ATTACHMENT_MODE is retired"):
        GazeboDeploymentConfig.from_environment({
            "OPENETA_M3_ATTACHMENT_MODE": "detachable",
        })


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
