# Gazebo adapter design

## Current runtime boundary

The executable Gazebo profiles are M1 (live RGB-D observation) and M2 (RM75 +
Robotiq control with structured receipts). `GazeboDirectEnv` and
`GazeboRuntime` own lazy startup, ROS 2 launch, camera freshness, controller
access and reverse-order cleanup. The deployed path remains:

The profile names are `m1`, `m2_robotiq2f85`, and the disabled
`m3_pickplace`.

`MCP → dedicated gazebo bench worker → UnifiedEnv → GazeboDirectEnv → GazeboRuntime → ROS/Gazebo adapters`

M3 is registered only to preserve its stable environment identifier and static
scene dimensions for non-manipulation consumers. Its profile has no launch,
world, control, physics or structured-receipt capability. Constructing it
fails with `DETACHABLE_JOINT_UNIMPLEMENTED_OR_UNAPPROVED` before any process,
MCP call or action can start. M4 manipulation is blocked by the same boundary.
No alternative holding, attachment, geometry, transform or distance mechanism
is present.

## Oracle boundary

`extensions/gazebo/oracle_perception.py` is a pure-Python projection utility.
It can turn known static object declarations and cached observation poses into
SAM3-shaped oracle detections for offline contract tests. It is explicitly
simulator truth, not real visual inference. A fake grasp candidate only tests
parameter shape; neither component authorizes a grasp or proves M4 execution.

M2 gripper safeguards and articulated-handle support remain independent of
this disabled M3/M4 work.

## Acceptance status

The current TUI coordinator covers M0–M2 only. The former cloud entry reports
blocked status without building, launching or connecting to a worker. Historic
M3/M4 results are diagnostic evidence for the removed implementation, not
formal acceptance. A future approved DetachableJoint design must define the
native topology, ACK semantics, evidence chain and remote isolation plan before
new M3/M4 assets are created.
