# Gazebo adapter design (M1–M3)

For the M4 oracle perception module built on this adapter (SAM3-shaped
`oracle_perceive` tool, projection core, and perception profile switch), see
`docs/gazebo-m4-oracle-perception.md`.

## Runtime architecture (current)

The production live boundary is profile-driven.  `profiles.py` declares three
immutable repository-owned profiles — `m1`, `m2_robotiq2f85`, and
`m3_pickplace` — each pinning its launch description, world name, RGB-D
camera set, and capability set.  Capabilities are additive:
`fresh_observation` and `authoritative_camera` form the base set; the M2
profile adds `control` and `structured_receipt`; the M3 profile adds
`physics` (which requires `control`).

`GazeboDirectEnv` (`direct_env.py`) is the sole Gym-shaped environment.  Its
`__init__` starts nothing; the first `reset()` is the authoritative
lazy-start boundary.  It delegates process, ROS, readiness, freshness, and
teardown concerns to `GazeboRuntime` (`runtime.py`), which owns one explicit
`rclpy.Context` with a shared executor, the `Ros2LaunchProcess`, the
`RosRgbdCameraSource` cameras, and — when the profile capabilities require
them — the M2 controller (`ros_control.py`) and the M3 physics source
(`ros_physics.py`).  Step observations are ordered after the action
completion barrier; the structured receipt is namespaced inside Gym info and
restored to the established MCP wire fields by the bench worker's generic
control codec.

The deployed call chain is:

`MCP → dedicated gazebo bench worker → UnifiedEnv → GazeboDirectEnv → GazeboRuntime → ROS/Gazebo adapters`

Deployment settings are snapshotted once per worker into an immutable
`GazeboDeploymentConfig` (`deployment.py`); runtime code never reads or
mutates global environment variables.  The earlier per-milestone
session/worker classes were removed in favour of this single profile/runtime
stack; see the change note at the end of this document.

## Lifecycle

`GazeboEnvironment` (`lifecycle.py`) is the deterministic, oracle-only
lifecycle boundary used by contract tests; the production live boundary is
the profile/runtime stack above.  `GazeboEnvironment.create()` establishes
one isolated world and immediately performs a deterministic reset.
`reset(task, seed)` restores the configured robot/object/camera initial
state.  `observe()` returns the latest observation without mutation.
`close()` is idempotent and clears all retained state.  `GazeboProcess` owns
a headless `gz sim -s -r <world>` process and terminates its process group
in `close()`/`finally`.  ROS 2 nodes/executors and sensor bridges remain
separate until their deployment configuration is supplied.
`RosGzBridgeProcess` can own the documented Jazzy `ros_gz_bridge
parameter_bridge` process; M1 tests use only the standard `/clock` mapping.

## Observation mapping

The top camera is the global `scene_primary` camera.  Its RGB and metric
depth, calibration, and explicit frame tags map to `CameraFrame`.  Robot
joint/EEF/gripper values map to `RobotState`; the read-only M1 profile
publishes an empty state because no robot-control process is started.
Oracle objects are compact summaries with `provenance=gazebo_oracle`, never
hidden perception results.

## ROS 2 and camera mapping

Deployment configuration supplies world, robot, MoveIt group, camera
topics/frames, intrinsics, and depth encoding.  The adapter converts ROS
image timestamps and units at the boundary; Planner code never receives ROS
topics or raw Gazebo entity APIs.  Top and wrist roles remain semantic
(`scene_primary`, `wrist_primary`).
The oracle packet records configurable top-camera RGB, metric-depth, and
CameraInfo topic names in metadata; it does not claim those topics are live.
`RosRgbdCameraSource` is the live conversion boundary: standard ROS encodings
are decoded into RGB uint8 and metric float32 depth, `CameraInfo.K` supplies
fx/fy/cx/cy, and explicit camera-to-world extrinsics are mandatory. Missing
calibration, unsupported encodings, or incomplete frames fail closed.
`Ros2LaunchProcess` owns the profile's official ROS 2 launch description.
The opt-in integration test uses the installed
`ros_gz_sim_demos/rgbd_camera_bridge.launch.py` and verifies a real RGB-D +
CameraInfo packet becomes `CameraFrame`.  `GazeboRuntime` composes that
launch, the `RosRgbdCameraSource` cameras, and the official world-control
reset service behind its lazy-start `reset/observe/execute/close` lifecycle.

## Reset, errors, and cleanup

Reset records task/seed and scene epoch.  Invalid lifecycle order raises a
typed `GazeboLifecycleError` on the oracle boundary and `GazeboProcessError`
on the live runtime; transport failures are reported as structured MCP
errors.  Cleanup is idempotent and must run in `finally`, with process/node
termination verified by integration tests.  The runtime tears down in
reverse order — physics source, controller, cameras, ROS graph, then launch
— and cleanup failures do not short-circuit the remaining resources.

## Test strategy

Unit tests cover deterministic reset, metric camera packet conventions,
oracle provenance, lifecycle ordering, and idempotent close.  Contract tests
round-trip through `EnvObservation` and assert the MCP create/reset/render/
close schema.  Integration tests are gated on an installed Gazebo/ROS 2
profile and are not silently replaced by oracle results.

For M1 contract coverage, `GazeboOracleMcpTransport` is an in-process test
transport that feeds the existing `SimulatorMcpEpisodeEnvironment`. It is
explicitly oracle-only and does not claim to start Gazebo or ROS 2. It
remains the only in-process transport; the deployed path is the bench-worker
chain documented below.  Live behavior is covered by the profile/runtime
contract tests, which dependency-inject the launch, camera, controller, and
physics factories into `GazeboRuntime`, and by opt-in integration tests
gated on an installed Gazebo/ROS 2 profile.

The checked-in `worlds/m1_oracle.sdf` is only a deterministic smoke world for
process lifecycle testing; it is not a robot or perception benchmark scene.
`worlds/m1_rgbd.sdf` follows the installed Gazebo RGB-D sensor schema and is
used to prove raw RGB, depth, and CameraInfo topic availability. The bridge
test proves process ownership and raw topic discovery; converting ROS messages
into `CameraFrame` remains the next adapter step.
`GazeboProcess.wait_for_topics()` is the readiness gate between server startup
and bridge startup; process liveness alone is not treated as sensor readiness.
`GazeboWorldControl.reset_all()` uses the official
`/world/<name>/control` service (`gz.msgs.WorldControl` → `gz.msgs.Boolean`)
and checks the trusted `data: true` response. A reset seed is passed through
the documented WorldControl field.  Profiles with the `control` capability
use the model-only `reset_models()` variant plus explicit object-pose
restores, so a reset never rewinds the simulator clock under a running
trajectory controller.

## Worker deployment

The adapter is registered through OpenETA's existing bench-worker path; no
second MCP server is introduced.  `sim/env_registry.py:_register_gazebo_envs`
registers one stable environment ID per profile:

- `openeta/gazebo_live_rgbd-v0` — `m1` profile: read-only live RGB-D
  observation.
- `openeta/gazebo_rm75_robotiq2f85-v0` — `m2_robotiq2f85` profile: RM75 +
  frozen Robotiq 2F-85 control with structured receipts.
- `openeta/gazebo_rm75_robotiq2f85_pickplace-v0` — `m3_pickplace` profile:
  deterministic contact-based pick-place physical verification.

`BenchWorkerManager` resolves the `gazebo` prefix, starts the existing
`sim/bench_worker.py --bench gazebo` subprocess, and the MCP server pins the
returned handle to that worker exactly like the other benches.  Inside the
worker, `gym.make` routes through the registry entry point to
`GazeboDirectEnv` wrapped in `UnifiedEnv`; there are no profile-specific
worker classes.

Deployment-owned camera/world values are supplied through worker environment
variables, parsed once into the immutable `GazeboDeploymentConfig`.
Profiles carry repository-pinned camera extrinsics;
`OPENETA_GAZEBO_CAMERA_EXTRINSICS` overrides the top-camera extrinsics when
a deployment must re-calibrate, and the worker fails closed when explicit
camera-to-world calibration is absent from both profile and deployment.
The remaining `OPENETA_GAZEBO_*` variables select ROS 2/Gazebo executables,
the workspace overlay, extra launch arguments, a world override, and the
startup/observation deadlines. M3's grasp mechanism is fixed to
`bilateral_contact_adhesion_v1`: native left/right Gazebo contact sensors gate
repository-owned kinematic capture. Legacy attachment-mode environment values
are rejected rather than silently selecting a fallback.

## Module naming vs plan.md §6

plan.md §6 sketched a `robot.py` / `cameras.py` / `manipulation.py` /
`verifier.py` / `checker.py` / `mcp_server.py` layout.  The implementation
deliberately deviates in file names while preserving the section's actual
guidance — no unnecessary service layers, Gazebo behind the existing
Tool/MCP/environment boundary:

- `m2.py` / `ros_control.py` fill the `robot.py` / `manipulation.py` role:
  ROS-free control contracts and geometry (`m2.py`) are separated from the
  ROS action/TF adapters (`ros_control.py`) so contract tests never import
  ROS.
- `m3.py` / `ros_physics.py` fill the `verifier.py` / `checker.py` role:
  ROS-free physics snapshot and verification contracts (`m3.py`) are
  separated from the live ROS physics/planning-scene source
  (`ros_physics.py`).
- `runtime.py` / `direct_env.py` are the single lifecycle owner and the sole
  Gym boundary; they exist because the unified-runtime refactor collapsed
  the earlier per-milestone session/worker classes into one stack.
- `observation.py` already covers the `cameras.py` role, and `mcp.py` is an
  in-process contract-test transport rather than the sketched
  `mcp_server.py`, because deployment reuses the existing bench worker
  instead of adding a second MCP server.

## Change note

2026-08-11: rewritten for the unified profile/runtime architecture.  Earlier
revisions described the live boundary as `GazeboLiveSession` composed with a
`GazeboLiveMcpTransport` and adapted into the bench worker by a
`GazeboWorkerEnv`; those classes were removed.  The live boundary is now
`GazeboDirectEnv` + `GazeboRuntime` driven by the immutable profiles in
`profiles.py` (`m1`, `m2_robotiq2f85`, `m3_pickplace`).  The deterministic
oracle boundary (`GazeboEnvironment`, `GazeboOracleMcpTransport`) is
unchanged and remains a contract-test fixture.  See
`docs/gazebo-unified-runtime-acceptance.md` for the refactor acceptance
record.
