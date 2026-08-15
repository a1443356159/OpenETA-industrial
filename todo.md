# Task status

Source of truth: `plan.md` (M0/M1 first assignment). Keep this file updated
when work advances; do not start later milestones early.

## M0 — upstream audit

- [x] Inspect architecture, action pipeline, unified observation, adapter
      protocol, and simulator MCP lifecycle.
- [x] Execute an existing OpenETA simulator/MCP and rollout path: the
      simulator proxy plus rollout recorder regression passes **54 passed**.
- [x] Record extension points and deviations in
      `docs/gazebo-integration-audit.md`.
- [x] Run full upstream baseline in an environment with declared dependencies.
  - Focused M0/M1/MCP set remains green; worker contract tests report
    **57 passed**, and the opt-in live worker episode reports **1 passed**.
  - Sourced regression with CPU `torch` and `omegaconf` installed in the
    isolated declared-dependency environment: **1194 passed, 14 skipped**.

## M1 — Gazebo read-only observation

- [x] Specify lifecycle, ROS/camera mapping, reset, cleanup, errors, and tests
      in `docs/gazebo-adapter-design.md`.
- [x] Add deterministic oracle create/reset/observe/close adapter.
- [x] Emit OpenETA `EnvObservation` with metric depth and explicit frame tags.
- [x] Add lifecycle/serialization/provenance tests.
- [x] Validate configured camera dimensions and oracle object geometry before
      constructing an observation.
- [x] Start and clean up the checked-in headless Gazebo smoke world when the
      ROS 2 Jazzy/Gazebo toolchain is installed (`tests/test_gazebo_process.py`).
- [x] Start and clean up the official ROS-Gazebo `/clock` bridge process using
      the documented `parameter_bridge` syntax.
- [x] Keep top-camera RGB/depth/CameraInfo topic names in configuration and
      expose them as provenance metadata without fabricating live frames.
- [x] Add an official-schema RGB-D smoke world and verify Gazebo publishes the
      configured raw RGB/depth/CameraInfo topics.
- [x] Gate bridge startup on Gazebo raw-topic readiness rather than process
      liveness alone.
- [x] Implement and test deterministic Gazebo world reset through the official
      `/world/<name>/control` WorldControl service.
- [x] Compose live launch, RGB-D observation, and world reset into the
      `GazeboLiveSession` M1 lifecycle facade; opt-in integration passes.
- [x] Exercise the live ROS subscription/executor source against the official
      `ros_gz_sim_demos/rgbd_camera_bridge.launch.py` world with CameraInfo and
      explicit extrinsics (`OPENETA_RUN_LIVE_ROS_TEST=1`).
- [x] Implement the documented conversion boundary and unit-test RGB/BGR,
      16UC1/32FC1 depth, CameraInfo intrinsics, and fail-closed validation.
- [x] Connect the adapter to a real Gazebo/ROS 2 process through the existing
      MCP worker boundary (`openeta/gazebo_live_rgbd-v0`, `GazeboWorkerEnv`,
      and `BenchWorkerManager`; opt-in worker episode passes).
- [x] Verify the OpenETA MCP lifecycle contract with the oracle transport and
      resource cleanup (`tests/test_gazebo_mcp_episode.py`).
- [x] Verify an end-to-end episode against a real Gazebo/ROS 2 deployment
      (opt-in `tests/test_gazebo_live_mcp_episode.py`).

## Deferred by plan

- [x] M2 dependency-light RM75/parallel-gripper configuration, state adapter,
      structured action routing, worker profile registration, and contract tests.
- [x] M2 live ROS action client/launch verification using the repository-owned
      Jazzy/Harmonic RM75 + Robotiq 2F-85 profile and frozen vendor assets.
- [ ] M3/M4 remote formal acceptance remains pending. M3 now has the approved
      native-contact stock `DetachableJoint` implementation; it fails closed
      on missing contact, attach/detach ACK, DART child-link state, or physical
      lift proof. Old soft-adhesion reports remain diagnostic evidence only.
- [ ] M5+ SAM3 integration and industrial benchmark.

## M2 verification

- [x] Preserve the legacy fixture contract: fixed RM75 names, binary 0/1
      gripper mapping, 35 mm active travel / 70 mm aperture, mimic state-only
      behavior, and model metadata.
- [x] Convert tool-frame targets through the inverse fixed mount into `link_7`
      MoveGroup goals with the specified tolerances.
- [x] Build fresh RobotState from complete JointState + TF only and surface
      structured M2 errors at the worker step top level.
- [x] Preserve M1 registration and read-only behavior (focused suite: 6 passed,
      1 skipped).
- [x] Fix OpenETA MCP runtime-task forwarding for Gazebo workers (avoid
      duplicate `task` keyword); sourced live M1 worker MCP lifecycle passes
      (`1 passed`).
- [x] Run the sourced unified Gazebo regression and M2 checkpoint verification
      on the Ubuntu 24.04/Jazzy/Harmonic baseline. The recorded non-optional
      repository regression reports **1202 passed, 14 skipped**; this local
      development evidence does not constitute formal acceptance.
- [x] Locate and preflight the neighbouring workstation's verified Jazzy RM75
      workspace (`/home/yyysaiko/workstation/external/rm75_ros2_ws`); asset
      validation passes for all four required installed packages.
- [x] Integrate the execution-capable repository-owned `RosM2ControllerFactory`
      and pass the real SSE MCP create/reset/gripper/A-B/observe/unreachable/
      idempotent-close lifecycle.
- [ ] Pass the formal Robotiq 2F-85 direct and MCP live acceptance. The
      2026-08-09 local checkpoint covered fresh post-action JointState/RGB-D
      timestamps, structured planning rejection, isolation, and all four
      cleanup paths, but has not been designated a formal acceptance run.

## Formal cloud acceptance

- [ ] M3/M4 remote formal acceptance remains pending. The approved local
      coordinator prepares, but never executes, a SHA-specific detached clean
      clone plan and a separate `venvs/<SHA>` runtime built from an explicitly
      selected base Python; it installs project dependencies, builds the ROS
      workspace, runs the real PTY TUI → MCP/SSE → Gazebo M0→M4 chain, and
      stops on a failed predecessor. The remote plan may report a pass only
      when its `acceptance-report.json` has
      `overall_status=passed`.

## Runtime configuration

- [x] Configure the local OpenAI-compatible planner provider for DeepSeek
      `deepseek-v4-pro`; verify the model through the provider `/models`
      endpoint, validate the OpenETA provider schema, and run a real planner
      smoke through `OpenEtaAgentRuntime` with `max_tokens=4096`. The API key
      remains only in the ignored local `.env` file.
