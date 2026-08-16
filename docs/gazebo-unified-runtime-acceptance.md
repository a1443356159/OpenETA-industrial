# Gazebo unified runtime acceptance

M1 and M2 use the profile-driven runtime for launch, cameras, controller
readiness, fresh observations and cleanup. M3 adds only the guarded stock
DetachableJoint path: paused preflight detach ACK, reset detach ACK, native
dual-pad contact admission, attach ACK and native child-link physical proof.

M3 is fail-closed on contact, state ACK, DART, or proof errors. It has no
compatibility mechanism. M4 Oracle data is marked simulator truth and does
not relax these requirements.

Formal M0–M4 acceptance is PTY TUI → MCP/SSE → Gazebo, with per-case ROS
domain, Gazebo partition, loopback port, process-group logs and cleanup
evidence. This source tree does not assert a completed remote acceptance.

## Current control-layer evidence

On 2026-08-16, SHA `c202f51` completed a remote, no-provider M2 control
diagnostic from a fresh hash-verified detached clone. A clean system-Python
venv passed the ROS ABI preflight and workspace build; real MCP/SSE then
performed create/reset, dual RGB-D observations, gripper open/close/open,
two completed MoveIt motions, and close. The isolated ROS domain, Gazebo
partition, loopback port, and owned worker cleanup were checked afterward.

This is useful control-layer evidence only. It did not invoke a planner or
provider, did not use the formal PTY TUI, and could not use the required
remote `origin` clean clone because of `problem.md` P-007. It therefore does
not mark M0–M4 formal acceptance complete.

On 2026-08-16, the local no-provider coordinator completed a separate,
strictly serial M0–M4 control run at
`.cache/reports/control-local-20260816T054203Z-m2inset` with
`overall_status=passed`. It used AgentTool → MCP/SSE → Gazebo only, records
`planner_provider_invoked=false`, and explicitly records
`formal_tui_acceptance=not_run`. M0 through M4 each used a distinct empty ROS
domain (80, 81, 82, 84, 85), Gazebo partition and loopback port; all five
cleanup records report an empty candidate graph, empty partition, free port
and no owned process residuals. Domain 83 was skipped because a pre-existing
external ros2cli daemon occupied it.

M2 completed its six gripper commands, four real A↔B MoveIt motions and a
real unreachable `MOTION_PLAN_FAILED`. M3 recorded bilateral native contact,
the stock `attached` and `detached` ACKs, a 96.5 mm native child-link lift and
2.51 mm capture-relative translation. M4 ran the same physical sequence
(100.8 mm / 1.04 mm) plus the explicitly labelled `gazebo_oracle` contractual
fake candidate (`is_model_prediction=false`). These are local control-layer
facts, not evidence of an external perception model or a formal remote PTY
TUI run.
