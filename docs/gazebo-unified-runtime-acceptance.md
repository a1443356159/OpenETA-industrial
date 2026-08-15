# Gazebo unified runtime and formal acceptance

The dated entries below are implementation history. Current formal evidence is
always produced by the clean-SHA cloud M0–M4 coordinator described at the end
of this document; historical local runs cannot mark a milestone complete.

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

## Current M3 and cloud acceptance contract

M3 is a profile of the same runtime, but its carry mechanism is deliberately
deterministic: native Gazebo left/right fingertip contact sensors must provide
fresh evidence for the same known object before `M3AdhesionSystem` captures it.
The physics engine remains responsible for contact and release settling; held
motion is labelled `grasp_mechanism=bilateral_contact_adhesion_v1`. Plugin
state alone cannot prove a grasp: target lift, stable target-relative pose and
non-moving distractor remain verifier gates before MoveIt attachment changes.

The formal entry point is:

```bash
OPENETA_CLOUD_ACCEPTANCE_ROOT=/data/openeta-cloud-acceptance \
  bash scripts/run_cloud_m0_m4_acceptance.sh
```

It rejects a dirty source or a commit not reachable from `origin`, makes a
fresh detached clone on the data disk, records OS/GPU/disk/vendor/Gazebo and
overlay evidence, builds once, then runs M0–M4 serially. Every live segment
receives its own ROS domain, Gazebo partition and MCP port; cleanup targets
only processes carrying that unique partition. The immutable root report,
milestone JSON, stdout, build/launch/MCP logs and cleanup evidence reside in
the SHA-specific `.cache/cloud-m0-m4-<UTC>-<SHA>/` directory. M4 is skipped
after any preceding failure or inconclusive result.
