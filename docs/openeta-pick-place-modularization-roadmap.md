# OpenETA pick-place modularization discussion record

Status: discussion draft. This document records optimization candidates only.
Nothing here authorizes a runtime, ROS, tool-surface, safety-gate, or acceptance
change.

## Purpose and decision boundary

This record captures the current, remotely exercised pick-place architecture
and the modularization opportunities identified after the `v70-frozen-normal`
real-chain PASS. It is intended to support item-by-item review before any
implementation begins.

The following constraints remain authoritative during discussion:

- do not add an AgentTool or expose the private MoveIt qualification RPC;
- do not add a regrasp-specific mode, state, prompt, or tool;
- do not specialize functional behavior for an acceptance scenario;
- do not loosen collision, drift, placement, execution, or evidence gates;
- do not reuse qualification trajectories for execution;
- do not make design-level changes without explicit approval.

## Current baseline

The current ownership split is:

```text
VLM
  task understanding, SAM3 selection, PASS-only grasp/place ID selection

Host runtime
  obligations, immutable candidate pools, frame compilation, qualification,
  proof cache, state/recovery policy, and evidence binding

MoveIt
  IK, state validity, collision-aware plan-only, and fresh execution planning

Gazebo native adapter
  contact evidence, attach/detach ACK, measured attachment, scene synchronization,
  lift/retention proof, and stable-placement verification
```

The normal pick-place sequence is:

```text
pregrasp observe
→ target and destination SAM3 selection
→ host-private 96-goal AnyPlace pool in absolute world object SE(3)
→ GraspGenX reserve pool
→ grasp world-EEF compilation and MoveIt funnel
→ bounded pregrasp grasp-place compatibility qualification
→ host activates and compiles the stable head of the equal-status PASS grasp queue
→ fresh real grasp planning and execution
→ native attach ACK and measured T_eef_object_attached
→ lift/retention gate
→ independent post-attachment placement observation and SAM3 selection
→ measured-attachment requalification of the selected grasp's frozen PASS goals
→ host activates and compiles the stable head of the equal-status PASS placement queue
→ fresh hover/release planning and execution
→ native detach, settling verification, and retreat
```

Default pool and funnel limits are:

| Field | Grasp | Placement |
|---|---:|---:|
| Model reserve/raw pool | 200 | 96 |
| Diversity pool | 64 | 96 |
| Full plan-only submissions | 4 | 4 |
| Deterministic IK seeds per endpoint | 8 | 8 |
| Qualification rounds | 1 | 2 |

Every full plan-only PASS is stored in the qualified candidate queue and
returned as selectable. There is no separate exposure limit or exposure
truncation; with the current planning bound, one run returns at most four PASS
candidates.

For pregrasp compatibility, the host constructs pairs from at most four grasp
PASS candidates and the complete current 96-goal pool. Pair ordering is
round-robin by grasp. Every constructed pair enters the coordinate, workspace,
pure-IK, collision-IK, and endpoint layers, so the default ceiling is
`4 x 96 = 384` screened pairs. Only full plan-only remains bounded to four
fairly interleaved submissions.

After a real attach, only the absolute object goals that passed for the
actually executed grasp are retained. They are recompiled using:

```text
T_world_eef_goal =
    T_world_object_goal × inverse(T_eef_object_attached_measured)
```

and pass through the complete placement qualification chain again. The
pregrasp trajectory is never treated as executable proof.

## Optimization candidates

Each item below is unapproved until it is discussed and explicitly selected.

### O1. Split the pregrasp coordinator

Current issue:

`_PregraspGraspPlaceCoordinator` owns goal materialization, pool lifetime,
grasp-goal pairing, qualification dispatch, PASS aggregation, frozen-goal
lookup, and measured-attachment one-use binding. These responsibilities have
different lifetimes and invariants.

Proposed module boundaries:

```text
AbsoluteObjectGoalPool
  immutable world goals, observation/scene binding, provenance

GraspPlaceCompatibilitySearch
  pair construction, quotas, ordering, qualification, PASS aggregation

MeasuredAttachmentGoalBinding
  executed grasp identity, one-use attachment binding, requalification input
```

Expected benefit: clearer lifecycle, focused tests, less implicit mutable state.

Risk: incorrectly splitting cache invalidation could admit stale goals. Initial
work should therefore be a behavior-preserving extraction with identical
hashes, counters, and fail-closed tests.

### O2. Give pregrasp joint qualification its own budget policy

Status: approved for implementation.

Current issue:

Joint grasp-place qualification implicitly reuses the normal placement
diversity limit. The global 96-pair cap and global top-four plan tail are real,
but per-grasp coverage is not expressed as a first-class contract.

Approved contract:

```text
JointQualificationBudget
  max_grasps = 4
  goals_considered_per_grasp = complete active goal batch (at most 96)
  L1-L4 pair input = every constructed pair (at most 4 × 96)
  L5 full_plan_submission_limit = 4
```

There is no separate pair-screen limit and the ordinary placement diversity
limit is not reused. After L4, at most four candidates are chosen by stable,
deterministic grasp-branch fairness. They are insurance against individual
full-planning failure, not a quality ranking. All four selected submissions are
attempted even if an earlier one passes, and all L5 PASS results have equal
qualification status.

Expected benefit: complete compatibility coverage before the expensive tail,
truthful accounting, and no accidental loss of three grasp branches behind a
global 96-pair cap. Risk: L2-L4 work increases and must be validated before
later performance optimization.

### O3. Remove redundant post-attachment AnyPlace inference — implemented

Status: approved and implemented locally; remote chain acceptance remains a
separate verification step.

The independent post-attachment RGB-D observation and object/region SAM3
selections remain mandatory. The existing public `anyplace` obligation first
validates those exact packets. If the executed grasp has frozen pregrasp PASS
absolute object goals, a host pre-inference hook then skips the AnyPlace MCP
predictor and sends only those frozen goals through measured-attachment
recompilation and the complete placement qualification funnel.

Implemented internal flow, without adding a VLM tool:

```text
independent placement observe and segment
→ validate scene/region binding
→ measured-attachment frozen-goal requalifier
→ only on zero PASS: one new-seed AnyPlace inference
```

The normalized object/placement packets and scene revision remain in
`ToolResult.details.source`, so a zero-PASS first round reuses the same
independent observation through the existing bounded recovery obligation. The
attachment binding is one-use: the second call therefore reaches the real
AnyPlace predictor exactly once, qualifies only that new pool, and never merges
the failed frozen IDs. If there is no verified attachment or no frozen pool for
the executed grasp, the normal model-backed AnyPlace path is unchanged.

The public tool surface, attachment/scene bindings, qualification cache,
plan-only limits, and execution safety gates are unchanged. The original
pregrasp AnyPlace candidate-image artifact is retained with explicit frozen-pool
provenance; it is not presented as a new post-attachment model output.

### O4. Introduce typed geometry and compilation boundaries

Current issue:

Geometry, calibration, strategy selection, identity hashes, clearance policy,
and tool payload generation share dictionary-heavy functions. Frame names and
pose representations (`xyz`, `translation_xyz`, quaternion, rotation matrix)
are validated repeatedly at runtime.

Candidate modules/types:

```text
SE3
WorldObjectGoal
MeasuredAttachment
WorldEEFGoal
CompiledGraspChain
CompiledPlacementChain

FrameCompiler
CalibrationCompiler
GraspStageCompiler
PlacementStageCompiler
ProofBinding
```

Expected benefit: makes frame direction explicit and permits property-based
tests such as composition/inversion round trips. Risk: serialization must stay
byte-for-byte compatible where hashes or external schemas depend on it.

### O5. Decompose the MoveIt qualification engine

Current issue:

`MoveItQualificationEngine` owns workspace rejection, seed generation, pure IK,
state validity, collision IK, endpoint propagation, cloned-scene transitions,
full planning, timeout handling, and verdict/evidence reduction.

Candidate modules:

```text
WorkspaceRejector
IKSeedPolicy
KinematicIKScreen
CollisionIKScreen
EndpointSequenceScreen
VirtualSceneTransition
FullPlanTail
QualificationVerdictReducer
```

All modules should emit a common evidence record containing the input binding,
verdict, reason, elapsed time, MoveIt code, solver, collision pairs, end state,
and `execution_started`.

Expected benefit: solver/planner substitution becomes localized and fail-closed
branches become easier to test. Risk: evidence ordering and scene-diff
propagation are safety-critical and require parity tests before replacement.

### O6. Separate deduplication, diversity, and equivalence-preserving scheduling

Status: approved direction; exact profile extraction remains implementation
work.

Current issue:

Candidate diversity currently influences both coverage and the order in which
the bounded full-plan tail is consumed. For grasp-place pairs, a global SE(3)
selection can obscure per-grasp coverage and falsely imply that the first
planned candidate is semantically better.

Candidate modules:

```text
CandidateDeduplicator
DiversitySelector
PlanSubmissionScheduler
```

For joint candidates, the approved hierarchy is:

```text
all constructed pairs through L1-L4
→ deterministic fair traversal across grasp branches
→ first four eligible pairs submitted to L5
→ every L5 PASS stored with equal qualification status
```

Source scores and diversity remain useful for forming finite model/host pools,
but there is no post-L4 `best` score and no need to optimize the relative rank
of the four submitted candidates. Expected benefit: simpler policy and clearer
proof semantics. Risk: deterministic ordering still needs stable tie-breaking
and branch-fair tests.

### O7. Split runtime memory into explicit reducers

Status: approved only as a host-internal implementation boundary.

Current issue:

`memory.py` combines perception selection, grasp policy, execution state,
attachment state, placement policy, recovery, scene identity, and planner
context. Dictionary-based cross-policy updates make lifecycle mistakes hard to
isolate. A recent passed run also showed the VLM producing an unnecessary,
unexecuted `ask_human` response after the environment had already closed,
illustrating the split between authoritative host state and summarized planner
state.

OpenETA remains a tool-calling agent. O7 must not expose a finite-state-machine
schema, state-transition tool, or workflow-state vocabulary to the VLM. The
reference model is a coding-agent orchestration loop: the model chooses the
next semantic tool action, while the host privately reduces immutable tool
receipts, performs mandatory deterministic transitions between model turns,
and computes the next allowed obligations.

Candidate internal projections:

```text
ToolReceiptReducer
CandidateQueueProjection
PerceptionBindingProjection
AttachmentEvidenceProjection
MotionEvidenceProjection
SceneIdentityProjection
EpisodeLifecycleProjection
ObligationResolver
```

Each private projection implements:

```text
(previous host projection, immutable tool receipt/internal event)
  → new host projection
```

The planner receives only ordinary tool availability, concise evidence facts,
and the next semantic obligation. It never receives or manipulates an explicit
state-machine node. Candidate selection may trigger O12 compilation inside the
host before the next model turn.

Expected benefit: explicit terminal states, easier replay, smaller policy
tests, and fewer prompt/host disagreements. Risk: migration must preserve all
existing obligations and fail-closed stops.

### O8. Standardize candidate-pool accounting

Current issue:

`model_raw_candidate_count`, `raw_candidate_count`, diversity counts, and
compatibility aliases can be overwritten by different layers. The difference
between a 96-candidate model pool and a four-goal frozen requalification pool
has already exposed ambiguity in verification code.

The agreed direction is recorded in
[`openeta-candidate-batch-qualification-contract.md`](openeta-candidate-batch-qualification-contract.md).
The authoritative unit is an immutable candidate batch with multi-parent
lineage, not a handler-owned flat counter set. Source provenance, current-run
qualification, and the qualified PASS queue are separate planes.

The proposed qualification layers are:

```text
L0  source materialization and lineage
L1  identity/frame/TCP/calibration and exact geometry compilation
L2  conservative structural feasibility
L3  endpoint pure-IK, collision-IK, and state validity
L4  ordered chain and cloned-scene transition feasibility
L5  bounded full segmented plan-only
L6  verdict reduction, qualified-candidate storage, and selection authority
```

`FAIL`, `UNKNOWN`, and `NOT_EVALUATED` are distinct. In particular, an endpoint-
PASS candidate outside the full-plan submission limit was not evaluated at L5
and is not a planning failure. Compatibility fields remain during migration
and are derived at one serialization boundary rather than manually overwritten
by handlers.

The same decision record separates startup count/budget parameters, versioned
producer/diversity profiles, derived values, and non-tunable safety or
calibration invariants. The redundant VLM exposure settings are removed. O2
requires every constructed pregrasp pair to enter L1-L4, so there is no
pregrasp pair-screen parameter.

Expected benefit: truthful health metadata, monotonic counters, and simpler
verification without binding later architecture to one producer or scheduling
policy. Runtime implementation is approved under the qualified-queue contract.

### O9. Define a stable planning-backend interface

Current issue:

The qualification core already receives callbacks, but the expected semantics
of each callback are spread across the engine and ROS adapter.

Candidate interface:

```text
PlanningBackend
  current_state()
  compute_ik()
  check_state_validity()
  clone_scene()
  transition_scene()
  plan_segment()
```

Initial implementations would remain the current MoveGroup backend and a
deterministic test backend. A future MTC experiment, if separately approved,
should replace only the bounded full-plan tail first. It should not replace raw
pool generation, diversity, cheap screening, proof cache, or the qualified
candidate queue.

Expected benefit: isolates ROS messages and permits controlled backend
comparison. Risk: backend equivalence requires exact execution and evidence
semantics, not only similar trajectories.

### O10. Introduce an append-only evidence ledger

Current issue:

Evidence is distributed across qualification cache entries, artifacts,
`ToolResult.details`, memory facts, Gazebo receipts, and trace events. Verifiers
must recursively inspect several representations.

Candidate event model:

```text
candidate_generated
candidate_compiled
candidate_screened
candidate_planned
candidate_qualified
candidate_selected
motion_planned
motion_started
motion_completed
attachment_changed
placement_verified
episode_closed
```

Every event should carry applicable candidate ID, scene epoch, PlanningScene
revision, attachment hash, request fingerprint, input/output hashes, and
`execution_started`.

Expected benefit: deterministic replay, simpler verification, and clearer
failure analysis. Risk: the ledger must complement existing evidence until
all consumers migrate; it must not become a second conflicting source of
truth.

### O11. Cache immutable geometry and reduce repeated ROS payload work

Current issue:

Absolute object goals, compiled candidate stages, scene diffs, and repeated
service payloads can be serialized or recomputed across qualification phases.

Possible optimizations:

- cache absolute object-goal materialization by observation/calibration hash;
- cache coordinate/TCP compilation by candidate/profile/scene binding;
- reuse immutable PlanningScene base snapshots while applying candidate-local
  diffs;
- batch or reduce repeated scene serialization around IK/state-validity calls;
- retain deterministic successful-IK reservoirs as an explicit scoped object.

Expected benefit: lower funnel latency without reducing candidate coverage.
Risk: cache keys must include every geometry, calibration, attachment, start
state, scene epoch, and revision dependency. A cache miss is preferable to a
stale hit.

### O12. Move candidate compilation behind host-owned workflow transitions

Current issue:

Before this migration, `compile_grasp_seed` and `compile_placement_seed` were
registered as AgentTools even though the main VLM supplied only one PASS
candidate ID. The
host, rather than the VLM, owns every actual compilation input: the immutable
candidate, camera calibration, TCP profile, measured attachment, current robot
state, scene epoch, PlanningScene revision, and proof hashes. Presenting this
deterministic binding step as a tool makes it appear optional and forces prompts
and planner validators to prevent the VLM from skipping, reordering, modifying,
or repeating it.

The task-level model should treat OpenETA's reasoning loop as a pace and
decision synchronizer, not as the executor of every deterministic transition.
Because the bounded L5 PASS candidates have equal qualification status and no
new selection tool is allowed, candidate activation is a host reducer:

```text
host activates the stable head of the immutable qualified queue
→ host validates its ID against the queue and proof binding
→ host performs the candidate-to-compiled-seed transition
→ host records the compilation proof and next required motion state
→ VLM receives the next semantic decision boundary
```

Approved boundary:

- remove `compile_grasp_seed` and `compile_placement_seed` from the AgentTool
  registry and skill `allowed_tools` surface;
- retain their geometry compilers as host-internal functions with the same
  fail-closed validation, hashes, calibration ownership, and testability;
- represent candidate activation as a hidden workflow reduction carrying only
  the queue-head candidate ID, never as permission for the VLM to provide poses
  or compilation parameters;
- let the host-owned grasp/placement state transition invoke compilation and
  publish an immutable compilation event before any motion can be dispatched;
- keep real motion freshly planned from the latest state. Candidate selection
  and successful compilation must never imply that execution started.

This does not create a monolithic `pick_place` AgentTool. The perception,
model, MoveIt, controller, and verification capabilities remain independently
testable. A pick-place request is one task-level workflow, while leaf tools and
private RPCs remain implementation capabilities coordinated by the host.

Expected benefit: fewer discretionary VLM steps, a smaller public tool surface,
and a clearer ownership boundary between semantic choice and deterministic
robot-state transition. Migration requirement: memory capture, planner
obligations, skills, traces, and acceptance verification must all consume the
equivalent immutable host compilation event before motion is allowed. The
implementation therefore uses the private reducer boundary in O7 and the
minimal evidence vocabulary selected in O10.

## Approved implementation order

The agreed sequencing principle is architecture first, behavior policy second,
and performance optimization last. The currently approved work is ordered by
runtime dependency:

1. **O8 Qualified-queue accounting** — remove the redundant exposure layer and
   make L5 PASS the single selectable candidate set.
2. **O2/O6 joint coverage and scheduling** — screen every constructed pair
   through L1-L4, submit four fairly and deterministically to L5, and treat all
   L5 PASS candidates as equivalent.
3. **O10 minimal internal event vocabulary** — establish the immutable
   candidate-selection and host-compilation evidence needed by O7/O12 while
   retaining legacy evidence during migration.
4. **O7 private projections and obligation resolution** — use coding-agent-like
   host orchestration without exposing a state machine to the VLM.
5. **O12 host-owned candidate compilation** — remove both deterministic compile
   operations from the AgentTool surface after every consumer uses the internal
   event.
6. **O4/O1/O9/O5 structural refactors** — proceed only where they simplify the
   approved implementation without changing behavior or safety semantics.
7. **O11 performance caches and payload reduction** — optimize only after all
    identities, dependencies, transitions, and measurements are explicit.
8. **O9 optional MTC experiment** — evaluate only after the architecture and
   repeated real-chain baseline remain stable.

O3 is omitted from the pending discussion order because its approved runtime
change is already implemented locally; its remote verification remains part of
the later unified verification pass.

## Approval ledger

Use this table during later discussion. `Recorded` means documented only.

| ID | Topic | Status | Decision / constraints |
|---|---|---|---|
| O1 | Split pregrasp coordinator | Recorded | No implementation authorized |
| O2 | Explicit joint qualification budget | Approved | All constructed pairs enter L1-L4; four grasp branches × the complete active goal batch; L5 remains bounded to four |
| O3 | Skip redundant post-attach inference | Implemented locally | Approved flow; independent observation retained, frozen-first qualification, one new-only model round on zero PASS; remote verification pending |
| O4 | Typed geometry layer | Recorded | No implementation authorized |
| O5 | Qualification-stage decomposition | Recorded | No implementation authorized |
| O6 | Separate diversity and scheduling | Approved | No post-L4 quality ranking; deterministic branch-fair selection of four; all L5 PASS candidates have equal status |
| O7 | Private runtime projections | Implemented locally | Coding-agent-style host reduction and obligations only; no VLM-visible FSM, state tool, or explicit workflow-state interface; remote verification pending |
| O8 | Candidate-batch lineage and layered qualification schema | Approved | Remove exposure settings; immutable lineage/current-run accounting; `candidate_count` is the complete L5 PASS queue |
| O9 | Planning backend / optional MTC tail | Recorded | No implementation authorized |
| O10 | Append-only evidence ledger | Supporting refactor approved | Introduce only the minimal internal selection/compilation events needed for O7/O12; retain legacy evidence during migration |
| O11 | Immutable geometry and ROS payload caches | Recorded | No implementation authorized |
| O12 | Host-owned candidate compilation transition | Implemented locally | Both compile operations removed from the AgentTool surface; fail-closed host transitions, evidence, prompts, planner, memory, and verification consumers migrated; remote verification pending |

## Verification expectations for any approved item

Every later implementation proposal should state:

- whether it is a pure refactor or behavior change;
- exact files and interfaces affected;
- preserved safety gates and evidence fields;
- unit and replay tests required;
- remote smoke/acceptance scenarios required;
- rollback boundary;
- expected latency or reliability effect and how it will be measured.

No item should be bundled with another merely because they touch the same
file. Each approved change should remain independently reviewable.
