# Gazebo normal acceptance: exact model terminals and frozen-candidate recovery

Normal acceptance keeps the existing AgentTool surface. A `move_to` call is one MoveIt request
covering IK, collision-aware planning, and execution from the current state to
an exact terminal. The agent never invents a pregrasp, hover, approach,
precontact, lift, carry, descent, clearance, or retreat pose.

## Frozen perception and candidate pools

A `normal` run acquires one calibrated grasp RGB-D observation and performs the
target SAM3 selection once. AnyGrasp or GraspGenX returns contact EEF poses;
camera/world and configured TCP representation transforms are the only allowed
pose conversions. Scores and capability evidence may order candidates but may
not edit their translation or rotation.

The placement object and placement region each use their own calibrated RGB-D
packet and SAM3 mask. AnyPlace returns up to 96 exact world object goals. Those
goals and the grasp reserve are immutable for the current scene epoch. A failed
motion candidate advances within the frozen qualified pool; it does not rerun
SAM3, the grasp provider, or AnyPlace. Perception/model inference repeats only
after the applicable pool is exhausted or evidence is stale.

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

The host freezes 512 GraspGenX candidates, retains a primary and a diverse
backup grasp, and schedules incremental grasp slices at cumulative limits
`4 → 8 → 16 → 32 → 64 → 128 → 256`. The untouched remainder is visited only
as the implicit exhaustive tail when the configured waves cannot fill the L5
target. Each branch
covers AnyPlace waves at cumulative limits `4 → 8 → 16 → 32 → 96`, giving a
complete `2 × 96 = 192` pair search when needed.
Different candidates run concurrently, but each candidate's dependent stages
remain ordered and wave results merge by fixed candidate ID.

Beam-2 propagates at most two distinct joint solutions per terminal. Pure IK
success is checked once with MoveIt state validity. A colliding solution gets
at most one collision-aware rescue; a pure no-solution gets no random fast-path
retry. L5 plan-only attempts are serialized in deterministic physical-quality
order. A candidate L5 failure advances to the next candidate and then the next
wave; the search stops as soon as a primary and distinct backup are proven. A
later candidate-linked execution failure uses that backup, then resumes the
unvisited frozen provider frontier with `model_inference=false`.
Infrastructure timeout/error is health-checked and retried once, then
terminates the run as infrastructure failure rather than “unreachable”.

## Exact grasp and placement execution

Execution is:

1. MoveIt plans current state → exact provider contact EEF pose.
2. Close the gripper; native bilateral contact plus attach ACK proves the grasp
   and records the measured `T_eef_object_attached`.
3. For a frozen AnyPlace object goal, compute exactly
   `T_world_eef_release = T_world_object_goal × inverse(T_eef_object_attached)`.
4. MoveIt plans the complete attached current state → exact release EEF pose.
5. Open, detach, and verify object stability in the declared placement region.

There is no post-close lift proof and no release offset. The attached object is
present in the MoveIt PlanningScene during placement. Scene revision,
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

## Acceptance

`normal` succeeds at the first complete L5-backed grasp-and-placement PASS. The
60-second P95 is a performance objective, not a hard deadline; a no-PASS run
must exhaust all 192 pairs and the fixed recovery seed budget. Fixed input must
produce the same selected candidate, joint branch, attempt order, and failure
classification across repeated runs. Dashboard and GUI modes are observers and
must not affect selection.

Run the canonical acceptance entry point with:

```bash
scripts/run_normal_gazebo_acceptance.sh --scenario normal
```

The canonical entry point starts a case-owned Gazebo operator GUI on the GPU
VNC desktop by default. Set `OPENETA_GAZEBO_OPERATOR_GUI=0` only for unattended
CI; this does not change the simulator server or qualification executor.

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
