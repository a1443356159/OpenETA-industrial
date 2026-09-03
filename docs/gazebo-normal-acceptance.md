# Gazebo `multi_normal` release acceptance

> **Status:** Normative final-release contract. Human operators should follow
> [`multi-normal-tui-reproduction.md`](multi-normal-tui-reproduction.md); broader documentation is
> indexed in [`README.md`](README.md).

Normal acceptance keeps the existing AgentTool surface. A `move_to` call is one MoveIt request
covering IK, collision-aware planning, and execution from the current state to
an exact terminal. The agent never invents a pregrasp, hover, approach,
precontact, lift, carry, descent, clearance, or retreat pose.

## Frozen perception and candidate pools

A release run acquires one calibrated grasp RGB-D observation and performs the
target SAM3 selection once. GraspGenX returns contact EEF poses;
camera/world and configured TCP representation transforms are the only allowed
pose conversions. Scores and capability evidence may order candidates but may
not edit their translation or rotation.

AnyGrasp remains an optional backend for development compatibility, but it is
not part of the final `multi_normal` release gate.

The placement object and placement region each use their own calibrated RGB-D
packet and SAM3 mask. AnyPlace returns up to 96 settled world-object goals.
The host preserves each goal's rotation and horizontal translation, and adds
only the configured vertical gravity-drop distance. A completely proven
exterior container edge may define a higher preferred entry terminal. If that
preferred terminal exhausts the current frozen grasp batch, the same batch is
retried at the configured drop height before any grasp-frontier expansion.
Those goals and the grasp reserve are immutable for the current scene epoch. A
failed motion candidate advances within the frozen qualified pool; it does not
rerun SAM3, the grasp provider, or AnyPlace. Perception/model inference repeats
only after the applicable pool is exhausted or evidence is stale.

## Legality barriers and qualification

Before Beam-2 IK, the host evaluates one complete cheap barrier:

1. Each AnyPlace object goal is checked once for finite valid SE(3), full
   footprint containment, legal support height and bounding box, static-scene
   penetration, and mathematically certain workspace exclusion.

Pair checks are wave-local. When a grasp-goal pair reaches the current deep
wave, it is checked using the exact contact terminal, predicted attachment
transform, and exact release terminal. Attached-object and gripper geometry
must be collision-free at terminal states, and each terminal must remain inside
the conservative analytic reach envelope. The untouched tail remains
`NOT_EVALUATED` and incurs no pair-collision work.

Parallel-jaw symmetric equivalents may share the physical legality
calculation, while retaining separate candidate evidence. No empirical score,
capability-map hole, or heuristic can permanently reject a candidate.

The host freezes 512 GraspGenX candidates and schedules incremental grasp slices at cumulative limits
`4 → 8 → 16 → 32 → 64 → 128 → 256`. The untouched remainder is visited only
as the implicit exhaustive tail when the configured waves cannot produce a
complete pair. Each grasp branch covers AnyPlace waves at cumulative limits
`4 → 8 → 16 → 32 → 96`; a physical failure resumes the next frozen grasp
branch rather than rerunning a model.
Different candidates run concurrently, but each candidate's dependent stages
remain ordered and wave results merge by fixed candidate ID.

Beam-2 propagates at most two distinct joint solutions per terminal. Pure IK
success is checked once with MoveIt state validity. A colliding solution gets
at most one collision-aware rescue; a pure no-solution gets no random fast-path
retry. L5 plan-only attempts are serialized in deterministic physical-quality
order. A candidate L5 failure advances to the next candidate and then the next
wave; the search stops at the first complete grasp/place proof. A later
candidate-linked execution failure resumes the unvisited frozen provider
frontier with `model_inference=false`. The action adapter first waits for
causal post-action joint samples to settle. A settled target state is success;
a settled non-target state is a recoverable `current_state_restart`; absence
of a provable settled state remains unknown and cannot be relabelled as an
unreachable candidate.

Recovery qualification starts from the measured current joint state, not the
pre-action state or named home. If a failed close moved a still-detached
object, the authoritative Gazebo pose is synchronized into the PlanningScene
and the remaining frozen grasp poses are rigidly rebased to that pose. Frozen
placement goals remain physical world goals: stale
`model_object_motion_world_transform` and `object_motion_world_transform`
metadata is removed before pair compilation so the destination is never moved
twice. Candidate IDs and both the original and rebased evidence remain
auditable.
Infrastructure timeout/error is health-checked and retried once, then
terminates the run as infrastructure failure rather than “unreachable”.

## Exact grasp and placement execution

Execution is:

1. MoveIt plans current state → exact provider contact EEF pose.
2. Close the gripper; native bilateral contact plus attach ACK proves the grasp
   and records the measured `T_eef_object_attached`.
3. Bind the frozen AnyPlace rigid motion to the measured collision body, apply
   the host-owned vertical gravity-drop terminal, and compute
   `T_world_eef_release = T_world_object_release × inverse(T_eef_object_attached)`.
4. MoveIt plans the complete attached current state → exact release EEF pose.
5. Detach and open, then let the VLM review the causal post-release RGB-D
   observation. The release transaction does not block on repeated simulator
   pose polling or a fixed stability dwell.

There is no post-close lift waypoint and no agent-authored release offset. The
attached object is present in the MoveIt PlanningScene during placement. Scene revision,
attachment transform, robot state, model hash, or calibration changes
invalidate prior qualification evidence.

## VLM contract

The VLM runs the observe-decide-act loop: it explicitly chooses each semantic,
lifecycle, recovery, and motion tool after inspecting current feedback. Typed
host obligations constrain exact parameters, geometry, joins, and safety but
do not act as a hidden task macro. The VLM identifies the requested target and
placement region, chooses among host-provided calibrated observations, and
calls the existing tools. It is told
explicitly that provider/AnyPlace poses are immutable model outputs and that
MoveIt owns the route. It must not estimate offsets, create waypoint variants,
repeat inference while a frozen pool remains, or treat a generic MoveIt error
as proof of a geometric cause. Ordinary portable-object stages expose only the
single relevant scene image; multi-view reasoning is reserved for an explicit
articulated-object probe.

The normal agent runtime retains its compatibility budgets of 100 planner
turns and 200 concrete tool calls. There is no episode-wide circuit breaker
that converts repeated observe, GraspGenX, or AnyPlace failures into hidden
task control. Typed subsystem retries and evidence freshness rules remain
local; when evidence cannot prove a safe continuation, the agent may observe,
ask the operator, change strategy, or terminate based on the real feedback.

## Final acceptance

The release gate uses one task-neutral physical scene: `multi_normal`. The
operator's natural-language prompt defines the ordered object-to-bin work
order. Prompt fixtures used by automated regression are task variants, not
additional scenes, and the physical scene never injects a static assignment.

The task succeeds at the first complete L5-backed grasp-and-placement PASS for
each ordered item. The
60-second P95 is a performance objective, not a hard deadline; a no-PASS run
must exhaust all 192 pairs and the fixed recovery seed budget. Fixed input must
produce the same selected candidate, joint branch, attempt order, and failure
classification across repeated runs. Dashboard and GUI modes are observers and
must not affect selection.

Run repeatable VLM acceptance through the real PTY TUI with:

```bash
scripts/run_multi_normal_gazebo_acceptance.sh
```

To reproduce the representative work order as a human operator, use the
interactive `human_tui` mode and follow
[the operator procedure](multi-normal-tui-reproduction.md).

The canonical entry point starts a case-owned Gazebo operator GUI on the GPU
VNC desktop by default. Set `OPENETA_GAZEBO_OPERATOR_GUI=0` only for unattended
CI; this does not change the simulator server or qualification executor.
The GUI process inherits the case's `GZ_PARTITION` and waits until that exact
partition exposes a Gazebo world/server service before starting the client.
It is then launched through VirtualGL, while GUI latency remains outside the
qualification executor.

When the planner provider is unavailable, the same perception, frozen-pool,
qualification, MoveIt, and physical chain can be exercised without a VLM:

```bash
scripts/run_normal_gazebo_acceptance.sh \
  --scenario normal \
  --execution-profile smoke_normal
```

`smoke_normal` is deliberately reported as
`control_only_no_vlm_smoke_normal_not_agentic_acceptance`. It requires zero
planner tokens and host-only decision provenance. An ambiguous semantic result
or missing deterministic obligation fails closed; it never falls through to a
configured VLM. This smoke is useful control-chain evidence, but it does not
replace the default `agentic_normal` acceptance.

The wrapper validates Python, ROS, Gazebo, overlay provenance, provider
endpoints, and checkout consistency before starting an isolated run. Use
`--verify-only` to re-verify an existing evidence directory without launching
the environment.

The validated 2026-08-31 release baseline is implementation commit `3a70294`.
Three consecutive GPU-GUI `human_tui` runs of the representative two-item work
order passed with 22 agent-selected tools each and zero host dispatches; exact
paths, timings, token counts, approval procedure, and cleanup checks are
recorded in the [operator reproduction guide](multi-normal-tui-reproduction.md).
