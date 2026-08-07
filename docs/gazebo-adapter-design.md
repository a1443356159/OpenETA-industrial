# Gazebo adapter design (M1)

## Lifecycle

`GazeboEnvironment.create()` establishes one isolated world and immediately
performs a deterministic reset.  `reset(task, seed)` restores the configured
robot/object/camera initial state.  `observe()` returns the latest observation
without mutation.  `close()` is idempotent and clears all retained state.
The first real transport implementation will own a Gazebo process and ROS 2
node/executor; both are started during create and shut down in close/finally.

## Observation mapping

The top camera is the global `scene_primary` camera.  Its RGB and metric depth,
calibration, and explicit frame tags map to `CameraFrame`.  Robot joint/EEF/
gripper values map to `RobotState`; M1 publishes an empty state because no
robot-control process is started.  Oracle objects are compact summaries with
`provenance=gazebo_oracle`, never hidden perception results.

## ROS 2 and camera mapping

Future configuration supplies world, robot, MoveIt group, camera topics/frames,
intrinsics, and depth encoding.  The adapter converts ROS image timestamps and
units at the boundary; Planner code never receives ROS topics or raw Gazebo
entity APIs.  Top and wrist roles remain semantic (`scene_primary`,
`wrist_primary`).

## Reset, errors, and cleanup

Reset records task/seed and scene epoch.  Invalid lifecycle order raises a
typed `GazeboLifecycleError`; transport failures will be reported as
structured MCP errors.  Cleanup is idempotent and must run in `finally`, with
process/node termination verified by integration tests.

## Test strategy

Unit tests cover deterministic reset, metric camera packet conventions,
oracle provenance, lifecycle ordering, and idempotent close.  Contract tests
round-trip through `EnvObservation` and assert the MCP create/reset/render/
close schema.  Integration tests are gated on an installed Gazebo/ROS 2
profile and are not silently replaced by oracle results.
