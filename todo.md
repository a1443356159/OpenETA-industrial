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
  - Focused M0/M1/MCP set remains green; current Gazebo collection is
    **13 passed, 3 skipped** with ROS Jazzy sourced.
  - Sourced regression with CPU `torch` and `omegaconf` installed in the
    isolated declared-dependency environment: **1192 passed, 13 skipped**.

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
- [ ] Connect the adapter to a real Gazebo/ROS 2 process through the existing
      MCP worker boundary (requires documented Gazebo worker registration and
      process transport; generic SSE alone is insufficient).
- [x] Verify the OpenETA MCP lifecycle contract with the oracle transport and
      resource cleanup (`tests/test_gazebo_mcp_episode.py`).
- [x] Verify an end-to-end episode against a real Gazebo/ROS 2 deployment
      (opt-in `tests/test_gazebo_live_mcp_episode.py`).

## Deferred by plan

- [ ] M2 robot control / MoveIt.
- [ ] M3 physical grasp and placement verification.
- [ ] M4 oracle pick/place.
- [ ] M5+ SAM3 integration and industrial benchmark.
