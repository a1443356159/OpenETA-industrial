# Project memory

## 2026-08-08 — M0/M1 baseline

- `plan.md` is the authoritative implementation plan. The first assignment is
  repository audit, adapter design, and a minimal Gazebo create/reset/observe/
  close backend; SAM3, MoveIt, and physical manipulation remain deferred.
- Existing OpenETA contracts are documented in `docs/architecture.md`,
  `docs/agent-action-pipeline.md`, `adapter/protocol.py`, `sim/unified_env.py`,
  and the MCP lifecycle in `sim/mcp_server/server.py`.
- Added `docs/gazebo-integration-audit.md` and
  `docs/gazebo-adapter-design.md`.
- The audit now records the requested README/architecture/action-pipeline/
  simulator/agent/real/protocol source inventory and the existing ToolResult,
  environment-receipt, MCP cleanup, and fresh-observation requirements.
- Added dependency-light oracle adapter under `extensions/gazebo/`. It emits
  `EnvObservation` with metric RGB-D, explicit OpenCV camera conventions, and
  `provenance=gazebo_oracle`; reset is deterministic for a given config/seed
  and close is idempotent.
- M1 adapter is intentionally read-only. `step()` rejects control until M2;
  it does not claim Gazebo or ROS 2 physical execution.
- Added `GazeboOracleMcpTransport`, which reuses the existing
  `SimulatorMcpEpisodeEnvironment` for a create/reset/render/close contract
  test. It is explicitly an in-process oracle transport, not a replacement
  MCP server or a claim of real Gazebo execution.
- Added fail-fast validation for camera dimensions, object SI-unit positions,
  quaternions, labels, and confidence bounds; Gazebo tests now pass `6/6`.
- The environment contains ROS 2 Jazzy/Gazebo Sim. Added
  `GazeboProcess` plus `worlds/m1_oracle.sdf`; with `/opt/ros/jazzy/setup.bash`
  sourced, process lifecycle and oracle tests pass `7/7`. This validates only
  process ownership/cleanup, not camera topics or robot control.
- Added `RosGzBridgeProcess` using the installed Jazzy
  `ros_gz_bridge parameter_bridge` contract. The documented `/clock` mapping
  starts and cleans up successfully; no undocumented camera/controller topic
  is introduced.
- Camera topic names are now explicit configuration (`rgb`, metric `depth`,
  `CameraInfo`) and are emitted as metadata; the oracle still makes no claim
  that those topics are live until a configured sensor world is attached.
- Added `worlds/m1_rgbd.sdf` based on the installed Gazebo RGB-D sensor schema;
  test evidence confirms raw `/top_camera/image`, `/top_camera/depth_image`,
  and `/top_camera/camera_info` topics are published. ROS message conversion
  is intentionally still pending a documented executor/subscription boundary.
- Added `RosRgbdCameraSource` and pure ROS message conversion helpers. RGB/BGR
  encodings, 16UC1 millimetre depth, 32FC1 metre depth, and CameraInfo.K are
  decoded into OpenETA packets; explicit extrinsics are required and invalid
  packets fail closed. Conversion tests pass `7/7` with no ROS runtime needed.
- Added an opt-in live ROS integration test. Its default skip is intentional:
  Jazzy `parameter_bridge` lazy discovery requires a launch/TTY context that
  pytest's captured stdin does not provide; no live CameraFrame claim is made
  until that deployment boundary is supplied.
- Added `GazeboProcess.wait_for_topics()` and gated bridge startup on raw topic
  discovery. This fixes the race where a live Gazebo process existed before
  sensor topics were registered; the stable raw RGB-D bridge test passes.
- Validation: `PYTHONPATH=. pytest -q tests/test_gazebo_lifecycle.py` passes
  (4 tests). Broader simulator tests could not collect in the current shell
  because optional dependency `gymnasium` is not installed.
- Git record: commit `8220bdb` (`feat(gazebo): add M1 lifecycle oracle adapter`).
- `uv` is not installed in the current shell, so the documented `uv run`
  baseline command could not be used.
- Created isolated `/tmp/openeta-plan-venv` from the declared project
  dependencies. The focused M0/M1/MCP regression set passes `62 passed`.
  The broader suite reaches `1154 passed, 11 skipped` but has four unrelated
  pre-existing failures (CLI contact-graspnet binding, object-memory URL
  validation, and two web-fetch resolver expectations); full collection also
  requires optional `torch` for `test_behavior_vector_contract.py`.
- Oracle MCP episode test passes (`5 passed` across the two Gazebo test files)
  and confirms the handle is released in `finally`.

## Open questions / blockers

- A real Gazebo process/ROS 2 transport and MCP registry entry require the
  deployment environment and must be implemented only after its documented
  process/topic contract is available.
