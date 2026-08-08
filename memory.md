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
- The live ROS integration test remains opt-in because it starts a full Gazebo
  demo launch, but it now uses the official launch context rather than a
  captured-stdin bridge process.
- Replaced the custom bridge in the opt-in test with the official ROS 2 launch
  process `ros_gz_sim_demos/rgbd_camera_bridge.launch.py`. With
  `OPENETA_RUN_LIVE_ROS_TEST=1` and ROS Jazzy sourced, a real 320x240 RGB-D +
  CameraInfo stream now converts to `CameraFrame` successfully (`1 passed`).
- Added `GazeboProcess.wait_for_topics()` and gated bridge startup on raw topic
  discovery. This fixes the race where a live Gazebo process existed before
  sensor topics were registered; the stable raw RGB-D bridge test passes.
- Added `GazeboWorldControl.reset_all()` using the installed Gazebo service
  contract. The `m1_rgbd` world reset with seed 7 returns trusted `data: true`
  and the process cleans up successfully.
- Added `GazeboLiveSession`, composing the official RGB-D launch,
  `RosRgbdCameraSource`, and `GazeboWorldControl` behind
  `create/reset/observe/close`. The opt-in live lifecycle test passes (`1
  passed`) and confirms scene epoch/reset metadata and idempotent cleanup.
- Added `GazeboLiveMcpTransport`, preserving the existing OpenETA MCP tool
  names and delegating lifecycle calls to `GazeboLiveSession`. Its opt-in
  end-to-end episode test uses `SimulatorMcpEpisodeEnvironment`, checks the
  trusted fresh-observation receipt, and always closes the live handle.
- Regression evidence: with ROS Jazzy sourced, the complete Gazebo test
  collection passes `12 passed, 3 skipped`; the opt-in live MCP episode passes
  separately under the system ROS Python. Without the ROS environment, the
  process/topic smoke test can time out because Gazebo resources are not
  sourced, so ROS sourcing remains part of the documented test command.
- The deployed worker-pool MCP registration for a Gazebo environment is still
  intentionally open: the repository documents the generic SSE worker
  boundary but provides no Gazebo worker registration/process contract. The
  live MCP facade therefore fails closed and cleans up a session if launch
  creation raises, without inventing a second server or undocumented worker
  protocol.
- Upstream regression follow-up fixed Contact-GraspNet endpoint discovery in
  shared runtime assembly without enabling the planner-facing backend, and
  rejects RFC 5737 documentation IPv4 ranges in cleartext object-memory URLs.
  The sourced baseline now reaches `1166 passed, 14 skipped` when the two
  existing web-fetch resolver tests are deselected; those tests use reserved
  TEST-NET addresses as if globally routable and remain open pending an
  upstream contract decision. The optional torch contract test is not
  collectible in the current venv.
- M0 runtime evidence: the existing OpenETA simulator MCP proxy and rollout
  recorder path passes `54 passed` in `/tmp/openeta-plan-venv`, covering the
  documented create/reset/render/close proxy lifecycle and immutable rollout
  bundle recording without introducing a parallel runtime.
- Web-fetch regression is now green (`22 passed`): RFC 5737 TEST-NET addresses
  are accepted only when both resolver and transport are explicitly injected
  as synthetic test seams; default network fetching still rejects them, and
  private/mixed DNS answers remain blocked. The sourced non-torch upstream
  suite now reaches `1168 passed, 14 skipped`.
- M0 baseline is now green in the isolated `/tmp/openeta-plan-venv` after
  adding the optional test dependencies `torch==2.7.1+cpu` and `omegaconf`:
  ROS Jazzy sourced full collection reports `1192 passed, 13 skipped`.
  These packages were installed only in the test environment; project
  dependency declarations and runtime architecture were not changed.
- The former worker-boundary blocker is resolved using the repository's own
  documented bench protocol: `gazebo` is a worker-manager bench, the stable
  env ID is `openeta/gazebo_live_rgbd-v0`, and `GazeboWorkerEnv` adapts the
  existing `GazeboLiveSession` behind `sim/bench_worker.py`. The live worker
  MCP episode passes with ROS Jazzy sourced; M1 reset skips generic action
  settling because the adapter is explicitly read-only.
- Final post-integration regression: the full sourced suite reports
  `1194 passed, 14 skipped`; worker contract tests report `57 passed`, and
  the opt-in real worker episode reports `1 passed` with no residual Gazebo,
  ROS launch, bridge, or worker processes.
- Local planner provider configuration: `.env` now selects provider
  `deepseek`, model `deepseek-v4-pro`, base `https://api.deepseek.com`, and
  `max_tokens=4096` to leave room for the model's reasoning output.
  The provider `/models` endpoint confirmed that exact model ID, and
  `load_planner_provider_config()` plus
  `OpenAICompatiblePlannerBackendConfig.from_provider_config()` validate
  successfully. The API key is intentionally not recorded in this memory or
  Git history.
- Provider smoke evidence: a real `OpenEtaAgentRuntime` turn using the
  configured DeepSeek backend produced a valid `response` action. The default
  512-token budget had once yielded an empty content with `finish_reason=length`
  on the reasoning model, so `OPENETA_LLM_MAX_TOKENS` was added to the existing
  provider config contract and set locally to 4096; the full regression remains
  green at `1194 passed, 14 skipped`.
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
