# Project memory

## 2026-08-09 — M2 Robotiq 2F-85 local checkpoint verification

- The production M2 profile is now
  `openeta/gazebo_rm75_robotiq2f85-v0` / `rm75_robotiq_2f85_sim_v1`; the old
  70 mm parallel fixture remains a buildable offline compatibility profile.
- The repository-owned Jazzy/Harmonic launch, `RosM2ControllerFactory`, MoveIt
  action path, dual live RGB-D source, worker routing, and real SSE MCP
  lifecycle are integrated. Mutating actions consume strictly newer RGB and
  depth samples and a JointState received after the action result; CameraInfo
  is reusable.
- Gazebo's imported Robotiq linkage is driven by a simulation-only adapter
  that expands the standard active-joint `ParallelGripperCommand` into the six
  vendor multiplier targets. The planner-facing MCP and ROS action schemas did
  not change.
- `run_m2_robotiq2f85_smoke.sh` locks an unused ROS domain in `100..199`, a
  unique Gazebo partition, and an MCP port. Cleanup targets only owned handles,
  PIDs, and process groups; normal, startup-failure, action-failure, and signal
  paths all passed while the pre-existing domain 42 stack remained healthy.
- Local development verification based on commit
  `9bc2a2c67c3881b8c687182de341cc1a8bf7c503` passed build and asset checks,
  `scripts/check_openeta_m2.sh`, 34 focused offline contracts, direct live ROS,
  real MCP live, isolation cleanup, and the non-optional repository regression
  (`1202 passed, 14 skipped`). This evidence is a checkpoint and does not
  constitute formal M2 acceptance. Optional BEHAVIOR/RoboCasa `torch` suites
  are outside the checkpoint gate.
- Direct live evidence: two `open -> close -> open` cycles, maximum active
  error 0.000164 rad, maximum mimic error 0.0166 rad, and apertures near
  85/0/85 mm. Dynamic targets were 60 mm apart; four arm motions remained
  within 3.46 mm and 0.054 rad.
- MCP evidence: `create -> reset -> close/open -> A/B/A/B -> observe ->
  unreachable -> close -> close` passed with a fresh complete observation on
  each world-changing action. The unreachable pose returned
  `MOTION_PLAN_FAILED` with MoveIt code `-27`; the second close returned
  `already_closed=true`.
- The ignored evidence report is
  `.cache/reports/m2-robotiq2f85-acceptance-20260808T180318Z-74215.json`
  (UTC filename, local checkpoint date 2026-08-09). Frozen asset manifest
  digests are `3b920b22...e9b824` for RM75-6FB-V and
  `c80c6db6...31f86` for the Robotiq closure.

## 2026-08-08 — superseded M2 prototype notes

- Registered an early simulation-only gripper prototype without changing the
  M1 ID or read-only class. Its superseded model contract used
  `base_link -> link_7`, group `rm_group`, fixed child
  `gripper_mount_link`, and joints `joint_1..joint_7` plus active/mimic fingers.
- The standard two-finger fixture is simulation-only: active travel 0.035 m,
  total aperture 0.070 m. Only exact integer 0/1 commands are accepted; the
  mimic finger has no command mapping.
- The dependency-light adapter builds link_7 goals by inverting the configured
  fixed mount, and builds RobotState only from complete JointState plus TF.
  Missing state/action dependencies return the planned structured error codes.
- Worker M2 structured results retain StepResult compatibility and additionally
  promote `ok`/`error_code` to the top level. Every successful controller
  result includes a post-action state observation. Unknown motion outcome sets
  `reconciliation_required=true`.
- Official-interface decision: live deployments must use MoveGroup and
  ParallelGripperCommand actions and Gazebo Sim gz_ros2_control plugin/system;
  no direct Gazebo or trajectory-topic LLM control was added.
- Blocker: no explicit vendor RM75 Jazzy asset/package path is configured, so
  a live launch/action client was not guessed from the Humble/Classic user
  material. The SDF checked into M2 is fixture metadata, not a fabricated RM75
  arm model. Live launch/controller verification remains opt-in and blocked.
- Production M2 creation therefore requires both `OPENETA_RM75_MODEL_PATH` and
  an explicit `OPENETA_GAZEBO_M2_CONTROLLER_FACTORY`; absent either, creation
  fails with `MODEL_ASSET_NOT_FOUND` or `ROS_NOT_READY` instead of launching
  the M1 demo under a false RM75 identity.
- OpenETA MCP worker smoke uncovered and fixed duplicate runtime `task`
  forwarding in `sim/env_registry.make_env`; the sourced M1 worker lifecycle
  then passed end-to-end (`1 passed, 11.31s`) with no residual processes.
- Direct M2 MCP create attempt correctly returned `create_env failed:
  MODEL_ASSET_NOT_FOUND`; no move/gripper command was issued against an
  unverified model.
- Focused contract/M1 regression: `6 passed, 1 skipped` in the isolated project
  environment.
- Broader unsourced Gazebo runs reached `16 passed, 2 skipped` and `18 passed,
  4 skipped`; the existing live world-control and RGB-D bridge topic checks
  time out when Gazebo resources are not sourced. This matches the recorded M1
  prerequisite and does not exercise M2 code.

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

## Resolved M2 blockers and deferred scope

- The neighbouring `/home/yyysaiko/workstation` was audited. Its
  `/home/yyysaiko/workstation/external/rm75_ros2_ws` overlay sources cleanly
  under Jazzy and `ros2 pkg prefix` resolves `rm_description`, `rm_75_config`,
  `rm_gazebo`, and `rm_bringup`; pinned source commits are
  `5fc226e...` (`ros2_rm_robot`) and `bdb12c...` (`rm_models`). M2 asset
  validation accepts this explicit workspace path. That backend remains
  plan-only and was not reused. The former `ROS_NOT_READY` blocker is resolved
  by the repository-owned Robotiq profile and `RosM2ControllerFactory`.

- M3 contact, grasp, attachment, and placement physics are now implemented as
  an isolated development checkpoint. Formal M3 acceptance remains pending:
  the first live candidate reached pregrasp but did not establish bilateral
  fingertip contact or a stall in the 32–48 mm aperture band, and Harmonic did
  not publish empty Contact heartbeats for non-contacting fingertips. The
  fail-closed verifier therefore returned `UNKNOWN` rather than claiming a
  grasp. See `docs/gazebo-m3-physical-verification.md`; neither the M2 nor M3
  milestone checkbox is accepted.
