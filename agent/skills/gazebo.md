---
name: gazebo
description: Gazebo RM75 Robotiq live_rgbd guidance.
version: v1
editable: true
task_patterns:
  - gazebo
  - gazebo_live_rgbd
  - gazebo_rm75
  - rm75
  - robotiq
  - robotiq2f85
  - live_rgbd
allowed_tools:
  - create_simulator_env
  - close_simulator_env
  - observe
  - move_to
  - gripper_control
---
# Gazebo

Use this skill as text guidance only, not as an executable macro. It applies
only when the current task, active `env_id`, observation, or structured receipt
identifies Gazebo, RM75, Robotiq, or `live_rgbd`. Do not select a different
environment backend: the host supplies the fixed `env_id`, and existing
`openeta/gazebo_*` environments are routed by the configured MCP worker.

Read the current observation and structured receipt before choosing an action.
Their `backend`, profile, capabilities, executable tool references, and error
fields define what can run now. The live tool schema is authoritative; do not
assume a ROS service, Gazebo transport, controller, topic, or parameter exists.
Do not connect to ROS or Gazebo directly and do not construct a second client.

Use the stable, currently executable atomic AgentTools. Observe before control,
perform one control edge, inspect its receipt, and obtain fresh evidence before
dependent control. A successful tool call or gripper-close acknowledgement does
not establish grasp, placement, contact, attachment, or task completion.

For a read-only environment, only observe and report the available evidence.
For a control-capable environment, issue only stable atomic tools exposed in
`tool_context.tool_references`; ask for help or report the missing capability
instead of guessing a low-level command. For a physical-verification profile,
accept success only when the structured environment receipt says the required
verification passed; visual plausibility and actuator acknowledgements are not
substitutes.

When a structured receipt exposes `control_spec.validated_relative_motion`, it
is the authoritative motion envelope for that profile. Record the first fresh
end-effector pose after reset as its stated reference, preserve the observed
orientation unless the receipt says otherwise, and derive any named targets
from that same reference. If a task needs two distinct neutral targets, use two
different receipt-advertised target names rather than guessing a lateral world
coordinate. A rejected motion remains a rejection: observe and report it, and
do not manufacture an unadvertised target to continue.

Use real RGB-D observations as the default perception basis. Oracle or
ground-truth perception is allowed only when an explicit test or debugging
context and the runtime profile expose it. Never switch to Oracle merely because
visual perception failed; inspect the failure and use only the recovery path
advertised by the runtime.

Keep user intent and the host-owned active environment task across turns.
Create or close an environment only through the exposed lifecycle AgentTools,
and use the exact runtime schema and `env_id` supplied by the host.
