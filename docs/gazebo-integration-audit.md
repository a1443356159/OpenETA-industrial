# Gazebo integration audit (M0/M1)

This audit records the existing OpenETA contracts before adding a Gazebo
embodiment.  The adapter is an extension; no OpenETA cognition or memory
runtime is replaced.

## Extension points

* `adapter.protocol.EnvObservation`, `CameraFrame`, `RobotState`, and
  `StepResult` are the typed boundary (`adapter/protocol.py`).
* `adapter.sim.SimulatorAdapter` is the local lifecycle/action interface;
  `sim/adapter.py` maps a `UnifiedEnv` observation into it.
* `sim/unified_env.py` normalises camera RGB-D, proprioception, task text, and
  optional object summaries.  Depth is linear metric metres at this boundary.
* `agent/tools/sim_mcp.py` owns the MCP episode lifecycle and fresh-observation
  obligation.  It calls `create_env`, `reset_env`, `render_env`, and
  `close_env`; control tools are separate.

## MCP lifecycle contract

`sim/mcp_server/server.py` exposes `create_env`, `reset_env`, `observe_env` /
`render_env`, `step_env`, and idempotent `close_env`.  Handles are session
scoped and pinned to a worker.  `close_env` must release the worker/environment
even when rendering or stepping failed.  M1 only needs create/reset/observe/
close; manipulation tools remain out of scope.

## Observation contract

An observation contains task text, a list of camera frames (RGB, optional
metric depth, intrinsics, world-frame extrinsics), robot state, object
summaries, and metadata.  Camera conventions are self-describing; the Gazebo
adapter publishes OpenCV optical frames (+X right, +Y down, +Z forward),
camera-to-world extrinsics, metres, and `top_left` image origin.

## Tool and side-effect semantics

Observation is read-only.  Reset mutates simulator state and starts a fresh
scene epoch.  Any future motion, inspection, gripper, or manipulation tool is
world-mutating and must return execution evidence followed by a fresh
observation.  Planner text cannot establish physical success; only an
environment receipt/checker may do so.

## Minimum upstream changes

None are required for M1.  `extensions/gazebo/` implements the smallest
compatible lifecycle adapter and emits `EnvObservation` directly.  The
existing MCP worker can host it once a registry entry is added in the
ROS/Gazebo deployment milestone; no core `agent/` or `adapter/` rewrite is
justified by the current contracts.

