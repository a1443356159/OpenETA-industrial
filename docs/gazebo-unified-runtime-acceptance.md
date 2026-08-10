# Gazebo unified runtime acceptance

Date: 2026-08-10 (Asia/Shanghai)

## Implemented boundary

The production call chain is now:

`MCP → dedicated gazebo bench worker → UnifiedEnv → GazeboDirectEnv → GazeboRuntime → ROS/Gazebo adapters`.

M1, M2 parallel, M2 Robotiq 2F-85 and M3 pick-place are immutable profiles of
the same `GazeboDirectEnv` and `GazeboRuntime`. The former profile-specific
worker classes and the parallel `GazeboLiveSession` / live in-process MCP
transport were removed. The deterministic oracle transport remains a contract
test fixture only.

Deployment settings are snapshotted once at Gazebo worker startup. Runtime
objects receive an immutable config and pass an explicit child-process
environment containing the locked ROS domain, Gazebo partition and
`ROS2CLI_NO_DAEMON=1`. Runtime code does not read or mutate global deployment
variables.

The runtime is lazy, uses one explicit `rclpy.Context` and one shared executor,
and treats camera/action/service data as readiness. Step observations are
ordered after the action completion timestamp and receipt time. Cleanup order
is physics, controller, camera/ROS graph, then launch; manager retirement then
verifies the complete worker PGID has exited.

## Verification performed

- Python compileall: passed.
- Bash syntax for the M2/M3 launch harness scripts: passed.
- `git diff --check`: passed.
- Unified runtime and ROS adapter selection: 36 passed.
- Gazebo offline/contract suite: 104 passed, 4 skipped.
- Full available offline regression: 1268 passed, 13 skipped, 2 live tests
  deselected. The optional BEHAVIOR vector test was not collected because this
  environment does not contain `torch`.

The two deselected live M1 checks were also run directly. Both failed closed
while waiting for `/top_camera/*` Gazebo transport topics. ROS 2 Jazzy/Gazebo
executables are installed, but topic discovery did not become ready within the
15 second gate. No fallback domain, ros2cli daemon, pre-existing ROS graph or
synthetic observation was used. Consequently live M0–M3 acceptance is
**blocked/inconclusive on this host**, and no milestone completion state was
updated.

## Workspace state

No commit was created and nothing was pushed. The workspace already contained
an extensive dirty Gazebo patch; the unified runtime changes intentionally
preserve its fresh-observation barriers, M2 recovery receipts, Robotiq control
validation, M3 physical evidence and cleanup error propagation. The final
dirty diff includes those pre-existing asset, launch, harness and documentation
changes plus the new deployment/profile/runtime/DirectEnv architecture.

WSL2 remains best-effort. A formal WSL2 run must record the WSL version, RMW,
domain, partition, overlay, missing readiness items and worker PGID. Discovery
failure is reported as blocked/inconclusive and must not fall back to domain 42
or a long-lived daemon.
