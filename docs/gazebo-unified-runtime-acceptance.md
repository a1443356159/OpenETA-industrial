# Gazebo unified runtime acceptance

Robot control uses the profile-driven runtime for launch, cameras, controller
readiness, fresh observations and cleanup. Native grasp adds the guarded stock
DetachableJoint path: paused preflight detach ACK, reset detach ACK, exact
model-generated contact terminal, native dual-pad contact admission, and
attach ACK. Contact plus ACK directly proves the grasp; no artificial
pregrasp, hover, lift, retreat, or post-close displacement is generated.

The production perception path is RGB-D → SAM3 → GraspGenX/AnyGrasp and
AnyPlace. Simulator ground-truth perception and fabricated grasp candidates
are not exposed by the worker, MCP server, tool registry, or acceptance
runner.

Formal release evidence uses the task-neutral `multi_normal` scene and the PTY
TUI → MCP/SSE → Gazebo chain, with a distinct ROS
domain, Gazebo partition, loopback port, process-group logs, correlated MCP
receipts, and cleanup evidence for each case. Scripted approvals are labelled
`scripted_tui`; interactive operator approvals are labelled `human_gated` with
source `human`. The verifier never reports one as the other.

The no-VLM `smoke_normal` profile calls the same real external model services
and native-contact chain, but is control-layer evidence only. It does not
replace the normal agentic acceptance defined by
`scripts/normal_gazebo_acceptance.py` and the versioned scene contract.

Historical lift-based and simulator-truth reports predate the exact-terminal
contract and are not release evidence.
