# OpenETA candidate-batch lineage and layered qualification contract

Status: approved architecture and runtime-migration contract for O2, O6, O7,
O8, and O12. Performance tuning, MTC adoption, and any safety-gate change remain
outside this approval.

## Purpose

This document records the agreed direction for truthful candidate provenance,
layered qualification, and storage of plan-only PASS candidates in the OpenETA
pick-place workflow.

The contract must describe all of these flows without changing the meaning of
a count:

- a fresh GraspGenX, AnyGrasp, or AnyPlace inference batch;
- host-generated parallel-gripper symmetry variants;
- a bounded pregrasp grasp-object-goal compatibility batch with two parents;
- post-attachment requalification of frozen absolute object goals without a
  new AnyPlace inference;
- a genuinely new AnyPlace retry batch after frozen goals produce zero PASS,
  without merging the failed frozen batch;
- candidates that pass endpoint screening but are not submitted to bounded
  full planning;
- candidates that fail conclusively, return UNKNOWN, or are never evaluated.

This is an accounting and module-boundary design record. Runtime changes are
implemented and verified separately; motion safety and execution gates remain
unchanged.

## Core model

The authoritative accounting unit is an immutable **candidate batch**, not a
flat collection of counters attached to whichever handler most recently
touched the result.

The design has three planes:

```text
provenance plane
  where this immutable batch came from and which parent batches contributed

qualification plane
  which checks this batch actually entered and what each check concluded

selection plane
  which complete plan-only PASS candidates are stored and may be activated
```

These planes must not overwrite one another. In particular, the model output
count of an ancestor is not the current frozen-requalification input count, and
an endpoint-PASS candidate omitted by the full-plan budget is not a planning
failure.

## Terms

- **Candidate**: one immutable grasp pose, absolute object goal, compiled
  motion chain, or grasp-object-goal compatibility pair with a stable ID.
- **Candidate batch**: an immutable, typed set of candidates produced by one
  source or one deterministic host transformation.
- **Parent batch**: a batch whose candidate data contributes to a derived
  batch. A derived batch may have more than one parent.
- **Trigger context**: the event that caused a new operation, without implying
  candidate reuse. A new AnyPlace retry may reference the prior zero-PASS run
  as its trigger, but the failed prior batch is not its parent.
- **Qualification run**: one scene-bound traversal of eligible candidates
  through the qualification layers.
- **Qualified candidate**: a candidate that passed every required layer,
  including full segmented plan-only.
- **Qualified candidate queue**: the immutable set of all candidates that
  completed L5 with PASS. There is no second exposure cap or exposure queue.
- **Activated candidate**: the stable head ID chosen by the hidden host reducer
  from an equal-status qualified queue. Activation is not compilation,
  planning, or execution.

## Layered funnel

The funnel is divided into seven layers. The layers define ownership and proof
strength. They do not require one large implementation class.

```text
L0  source materialization and lineage
 ↓
L1  identity, frame, TCP, calibration, and exact geometry compilation
 ↓
L2  conservative structural feasibility
 ↓
L3  endpoint kinematic and collision feasibility
 ↓
L4  ordered task-chain and virtual scene-transition feasibility
 ↓
L5  bounded full segmented plan-only
 ↓
L6  verdict reduction, qualified-candidate storage, and selection authority
```

Deduplication, diversity selection, deterministic ordering, and budget
scheduling are policy operations between layers. Every such operation must
produce a child batch or a scheduling record, but it cannot create a
qualification PASS. Their exact policy belongs to O6 and O2; the accounting
contract records what they did without assigning different status to L5 PASS
candidates.

### L0. Source materialization and lineage

L0 identifies how candidates came into existence and assigns an immutable
batch ID.

Supported source kinds should include:

```text
model_inference
frozen_requalification
deterministic_derivation
constructed_pair_batch
```

Examples include a backend reserve returned to the host, a parallel-gripper
180-degree symmetry variant, a selected frozen-goal subset, and a grasp-goal
pair batch.

L0 records:

- candidate kind and immutable batch ID;
- producer name and producer/schema revision;
- whether this operation invoked inference;
- backend output and backend-returned reserve counts when inference occurred;
- all parent batch IDs, their roles, and the number of parent candidates used;
- an optional trigger context that is explicitly not candidate lineage;
- materialized output count and content digest.

L0 makes no IK, collision, planning, stability, or execution claim.

### L1. Identity and exact geometry compilation

L1 binds a candidate to the robot and observation context and compiles its
exact world-EEF task chain.

It owns:

- candidate identity and schema validation;
- source frame, transform-chain, TCP, calibration, and robot-profile checks;
- exact world-EEF compilation;
- grasp hover, optional precontact, contact, and lift/retract poses;
- placement hover, release, and retreat poses using the active attachment;
- compilation input/output hashes and exact compiled stage poses.

Placement compilation remains:

```text
T_world_eef_goal =
    T_world_object_goal × inverse(T_eef_object_attached)
```

L1 must not project, clamp, rewrite, or otherwise improve a model-provided
SE(3) pose. A missing or inconsistent transform is UNKNOWN and cannot proceed.

### L2. Conservative structural feasibility

L2 rejects only targets that are deterministically impossible from static
robot structure and conservative geometry.

It owns:

- URDF/SRDF and joint-limit consistency;
- conservative workspace reachability;
- fixed-link and TCP envelope checks;
- malformed or structurally impossible compiled chains.

It must not introduce empirical workspace thresholds that can reject a valid
pose. It does not call full motion planning and does not mutate a scene.

### L3. Endpoint kinematic and collision feasibility

L3 evaluates one endpoint at a time under the start-state and cloned-scene
binding supplied by L4, using deterministic multi-seed qualification. Its
counts are aggregated over the candidate chain.

It has two explicit sublayers:

```text
L3a  pure IK with the configured deterministic seed set
L3b  collision-aware IK plus state-validity evidence
```

The seed set may include current state, configured home, previous-stage IK,
the scoped successful-solution reservoir, and candidate-bound legal
pseudorandom seeds. Seeds share the configured endpoint budget. Only failure
of every eligible seed is a conclusive IK FAIL; timeout, missing evidence, or
service ambiguity is UNKNOWN.

L3 proves endpoint feasibility only under the supplied binding. It does not
decide endpoint order or apply scene transitions itself.

### L4. Ordered task-chain and virtual scene-transition feasibility

L4 orchestrates repeated L3 checks as an ordered task chain rather than as
unrelated poses. Each accepted endpoint state becomes the next endpoint's
start state, and declared scene transitions are applied to a cloned
PlanningScene before L3 checks a dependent endpoint. L3 and L4 are therefore
nested responsibilities during a candidate traversal; the layer numbering
must never be interpreted as permission to check every endpoint against one
unchanged scene and apply transitions afterward.

The grasp chain is:

```text
current → hover → optional precontact → contact
→ virtual attach → lift/retract
```

The placement chain is:

```text
current(attached) → hover → release
→ virtual detach → retreat
```

L4 owns transition order, cloned-scene hashes, virtual attach/detach evidence,
collision pairs, and scene-revision checks. It must not call the Gazebo
DetachableJoint, apply the cloned scene to the live PlanningScene, send a
controller goal, or change the real scene revision.

The current implementation must not be assumed to satisfy this target
boundary merely because individual endpoint IK passed. Sequential transition
application during screening requires its own implementation verification.

### L5. Bounded full segmented plan-only

L5 receives only candidates eligible after L4 and submits the scheduler's
bounded selection to complete segmented planning.

It owns:

- the exact set and order of submitted candidates;
- a fresh start state for the qualification run;
- each segment starting from the preceding segment's planned tail;
- the same cloned-scene transitions at their declared boundaries;
- MoveIt error codes, solver, duration, joint margin, collision pairs, and
  non-empty trajectory summaries;
- proof that `execution_started=false` for every plan-only operation.

Plan-only trajectories are discarded after proof summarization. A later real
action always replans from the latest real state.

An L4-PASS candidate not selected by the full-plan limit is
`NOT_EVALUATED_AT_L5`, not FAIL. UNKNOWN, empty trajectory, missing execution
evidence, revision drift, or ambiguous response cannot become PASS.

### L6. Verdict reduction, qualified-candidate storage, and selection authority

L6 reduces immutable evidence into the only queue that grants selection
authority.

It owns:

- required-layer completeness and binding validation;
- storage of every full-plan PASS produced by the bounded L5 run;
- immutable qualified candidate IDs and their proof bindings;
- rejection of failed, UNKNOWN, unsubmitted, stale, private, or otherwise
  unqualified IDs.

`candidate_count` and `full_plan_pass_count` describe the same candidate set.
There is no `exposure_limit`, exposure truncation, or second PASS subset. A
host-activated ID authorizes the next host-owned workflow transition; it does not
imply that compilation, planning, motion, attach/detach, or execution has
occurred.

## Verdict and count semantics

The canonical contract distinguishes four outcomes:

| Outcome | Meaning | May enter the qualified queue |
|---|---|---:|
| `PASS` | The layer produced complete positive evidence | Only after all required layers |
| `FAIL` | Complete evidence proves the candidate ineligible | No |
| `UNKNOWN` | Evidence is missing, ambiguous, stale, or timed out | No |
| `NOT_EVALUATED` | The candidate was eligible but not processed because of a bound or early stop | No |

For a filtering layer that ran:

```text
evaluated_count = pass_count + fail_count + unknown_count
input_count = evaluated_count + not_evaluated_count
output_count = pass_count
```

For a derivation or policy transformation, input and output counts may differ;
the transformation must record both and link the output child batch to its
parent. It must not masquerade as a filter.

Count values follow these rules:

- `null` means the layer did not run or is not applicable to this batch;
- `0` means the layer ran and produced zero candidates in that category;
- upstream failure leaves downstream layer values `null`, not zero;
- candidates skipped by a bound contribute to `not_evaluated_count`, not
  `fail_count`;
- parent counts are never copied into current-run counts.

## Canonical internal shape

The names below are a proposed schema shape, not yet an implementation API:

```text
CandidateBatchAccountingV2
  schema_version
  batch_id_sha256
  candidate_kind

  source
    source_kind
    producer
    producer_revision
    inference_invoked
    inference_run_id
    backend_output_candidate_count
    backend_reserve_candidate_count
    trigger_context_id

  parents[]
    role
    batch_id_sha256
    selected_candidate_count

  materialization
    output_candidate_count
    candidate_set_sha256

  layers[]
    layer_id
    input_count
    evaluated_count
    pass_count
    fail_count
    unknown_count
    not_evaluated_count
    output_count
    input_batch_id_sha256
    output_batch_id_sha256
    binding_ref
    evidence_ref

  qualified_queue
    full_plan_pass_candidate_count
    candidate_count
    qualified_queue_sha256
```

The complete structure is host-private evidence. Agent-visible results should
contain only the stable funnel summary, retry/terminal policy, L5 PASS
candidates, and information needed for the next semantic decision. Parent IDs,
full bindings, seeds, collision details, and non-qualified internal candidate
IDs remain in the artifact.

## Required bindings

Qualification evidence is valid only under its recorded dependencies. The
canonical artifact must reference the applicable values from this set:

```text
observation hash
candidate-set hash
calibration/TCP hash
robot-model hash
scene epoch
PlanningScene revision/hash
attachment-transform hash
qualification start-state hash
planning configuration hash
```

A changed dependency invalidates the affected downstream layers. It must not
silently relabel stale PASS evidence as current.

## Flow examples

| Flow | `inference_invoked` | Source kind | Candidate parents | Current funnel input |
|---|---:|---|---|---|
| Fresh AnyPlace call | true | `model_inference` | none | New returned goal batch |
| Frozen goals after measured attach | false | `frozen_requalification` | selected pregrasp goal batch | Frozen selected goals only |
| New AnyPlace retry after zero PASS | true | `model_inference` | none; prior run is trigger context only | New returned goal batch only |
| Parallel-gripper symmetry expansion | false | `deterministic_derivation` | backend grasp batch | Derived grasp batch |
| Pregrasp grasp-goal compatibility | false | `constructed_pair_batch` | grasp batch and object-goal batch | Constructed pair batch |

The retry example is intentionally not a parent-child candidate relationship:
the prior failure causes the retry but contributes no candidate to it. This
prevents an implementation from accidentally merging previously rejected
candidates into the new round.

## Compatibility projection

Existing flat fields remain during migration and are written once at a final
serialization boundary. They are not the canonical source of lineage.

The intended stable meanings are:

```text
candidate_count
  complete L5 PASS count stored in the qualified queue

submitted_candidate_count
  compatibility alias of full_plan_submitted_count

generated_candidate_count
  compatibility alias of the current funnel/materialized input count
```

`model_raw_candidate_count` and `raw_candidate_count` currently carry
producer-specific historical meanings. During migration they may remain for
compatibility, but V2 verifiers and runtime policy must use the explicit source
and current-batch fields instead:

```text
backend_output_candidate_count
backend_reserve_candidate_count
funnel_input_candidate_count
```

For a frozen requalification, ancestral backend counts may be shown only as
lineage metadata. They must not imply that inference occurred in the current
operation.

## Architectural module boundaries

The layer contract supports these conceptual modules:

```text
CandidateBatchBuilder             L0
CandidateGeometryCompiler         L1
ConservativeFeasibilityGate       L2
EndpointKinematicQualifier        L3
SequentialSceneQualifier          L4
FullPlanQualifier                 L5
QualificationVerdictReducer       L6
CandidateAccountingSerializer     compatibility/public projection
```

These are responsibility boundaries, not approved class or file names. O4
defines their typed geometry, O5 defines their callable interfaces, O6/O2 own
policy and budgets, O10 consumes their immutable evidence events, and O12 later
moves deterministic candidate compilation behind host-owned workflow
transitions. O11 performance work remains last and must not change this
contract.

## Hyperparameter ownership model

Layering makes the required parameter families explicit, but it does not mean
that every number should become an environment variable. Parameters belong to
four different control classes:

```text
startup control-plane parameters
  bounded counts and operational budgets; immutable for the process lifetime

versioned policy profiles
  correlated diversity/ordering values changed and validated as one policy

derived values
  computed from immutable batches and other parameters; never configured twice

safety/calibration invariants
  embodiment or verification contract values; not general tuning knobs
```

Every resolved startup configuration and every selected policy profile must
have a schema version and content hash. A qualification batch binds the hash
before L1. Configuration cannot change during a batch or an episode.

### Existing startup parameters to retain

These existing CLI/environment names remain compatibility interfaces. Their
canonical internal names should reflect the layer-owned meaning rather than
the historical handler name.

| Existing setting | Canonical meaning | Layer | Current default | Valid domain |
|---|---|---|---:|---|
| `OPENETA_GRASPGENX_RAW_POOL_SIZE` | GraspGenX backend reserve limit returned to L0 | L0 | 200 | integer `10..512` |
| `OPENETA_ANYGRASP_RAW_POOL_SIZE` | AnyGrasp backend reserve limit returned to L0 | L0 | 200 | integer `10..512` |
| `OPENETA_ANYPLACE_RAW_POOL_SIZE` | AnyPlace object-goal reserve limit returned to L0 | L0 | 96 | integer `10..256` |
| `OPENETA_GRASP_DIVERSITY_POOL_SIZE` | Maximum compiled grasp batch selected by the current diversity policy | L1→L2 policy | 64 | integer `1..prepared grasp count` |
| `OPENETA_ANYPLACE_DIVERSITY_POOL_SIZE` | Maximum compiled placement batch selected by the current diversity policy | L1→L2 policy | 96 | integer `1..prepared placement count` |
| `OPENETA_MOVEIT_IK_SEED_COUNT` | Deterministic seed count per endpoint | L3 | 8 | integer `1..64` |
| `OPENETA_GRASP_FULL_PLAN_LIMIT` | Grasp full-plan submission limit per qualification run | L5 | 4 | integer `1..L4 eligible count` |
| `OPENETA_ANYPLACE_FULL_PLAN_LIMIT` | Placement full-plan submission limit per qualification run | L5 | 4 | integer `1..L4 eligible count` |
| `OPENETA_ANYPLACE_MAX_QUALIFICATION_ROUNDS` | Total post-attach placement qualification-run cap | workflow policy | 2 | proposed integer `1..2` |

The redundant startup settings `OPENETA_GRASPGENX_MAX_CANDIDATES`,
`OPENETA_ANYGRASP_MAX_CANDIDATES`, and `OPENETA_ANYPLACE_CANDIDATE_COUNT` are
removed by the approved design. The full-plan submission limit already bounds
the PASS candidate queue, and every L5 PASS is retained. Migration must remove
their CLI arguments, service metadata, host registration checks, and tests as
one change rather than silently repurposing the names.

The current placement-round setting needs a clearer canonical projection after
O3:

```text
frozen_requalification_run_limit = 1
new_anyplace_retry_limit = total_qualification_run_limit - 1
```

Thus the default value 2 means one frozen-goal run followed, only after a
conclusive zero PASS, by at most one genuinely new AnyPlace inference. It does
not authorize merging failed pools or any special regrasp workflow.

### Architecture-required parameters currently hidden or coupled

The following parameters are required by the layered ownership model but are
currently hard-coded or borrowed from a differently scoped setting. Names are
proposals; adding a CLI/environment interface requires separate approval.

| Proposed canonical setting | Owner | Current effective value | Proposed domain | Reason it must be separate |
|---|---|---:|---|---|
| `pregrasp_joint.grasp_branch_limit` | pregrasp scheduler | 4 | integer `1..4` | Controls how many qualified grasp modes receive goal compatibility checks |
| `pregrasp_joint.full_plan_submission_limit` | L5 pregrasp-pair scheduler | 4 | integer `1..constructed_pair_count` | Joint proof budgeting is distinct from post-attach placement planning |
| `endpoint.pure_ik_budget_s` | L3a | 2.0 s | finite positive duration | Total shared budget across pure-IK seeds for one endpoint |
| `endpoint.collision_ik_budget_s` | L3b | 2.0 s | finite positive duration | Total shared budget across collision-IK seeds for one endpoint |
| `endpoint.state_validity_timeout_s` | L3b | 2.0 s | finite positive duration | Bounds each state-validity service operation without changing its verdict semantics |
| `full_plan.segment_timeout_s` | L5 | 30.0 s | finite positive duration | Planning budget belongs to one segment, not to the entire batch |
| `full_plan.attempt_count` | L5 | 3 | integer `1..10` | Attempts belong to each submitted segment |

The two `pregrasp_joint` values are approved for runtime migration. The
remaining timing values are recorded only so later refactoring cannot silently
change behavior; they are not approved as new configuration in this phase.

O2 is decided: every constructed pregrasp pair enters L1-L4. With four grasp
branches and a 96-goal batch this is at most `4 × 96 = 384` pair traversals.
There is no separate pair-screen limit. Only L5 remains bounded to four
deterministically and fairly selected pairs.

### Derived values that must not become independent knobs

The following values are computed from the batch graph and resolved config:

| Derived value | Definition |
|---|---|
| `backend_output_candidate_count` | Actual producer output before its returned-reserve cap |
| `backend_reserve_candidate_count` | Actual immutable batch returned to L0 |
| `host_prepared_candidate_count` | Result after approved deterministic derivations and exact deduplication |
| `pregrasp_goal_count_per_grasp` | Number of goals selected from the active immutable goal batch for that grasp branch |
| `constructed_pair_count` | Sum of materialized pair counts across included grasp branches |
| `qualification_rpc_timeout_s` | Derived outer transport allowance covering all configured inner budgets plus transport grace |
| deterministic random seeds | Hash-derived from batch/candidate/stage identity and recorded, never selected by the VLM |

In particular, the RPC timeout must not be an independently tuned competing
deadline that can expire before the budgets it encloses. The host derives it
from candidate count, endpoint count, L3 budgets, L5 submission count, segment
budget, and a deployment-owned transport grace.

### Producer-owned model profiles

L0 deliberately distinguishes model-internal output from the reserve returned
to the host. The host reserve limit is a startup parameter, but the settings
that shape model-internal output belong to a versioned producer profile.

A `GraspGenXProducerProfile` should bind:

- diffusion draw count and candidates requested per draw;
- diffusion/OBB union and OBB sampling policy;
- model-visible scene-collision prefilter configuration;
- backend-local source-aware ordering before the reserve cap;
- model, checkpoint, gripper-description, and producer revisions.

The current implementation uses four stochastic inference draws and requests
200 grasps per draw before union/filtering. These are inventory values, not new
global CLI settings and not approved tuning targets.

An `AnyGraspProducerProfile` should bind its SDK/model revision, requested raw
generation behavior, backend-local score/filter settings, and returned-reserve
contract. An `AnyPlaceProducerProfile` should bind the model/config revision,
`init_k` or equivalent raw generation request, and inference-seed derivation.
With the current design, the AnyPlace raw request is 96.

Producer profiles and their hashes are reported by service health and copied
into L0 lineage. They are never VLM arguments. No checkpoint, vendor asset, or
backend safety behavior is changed merely by defining this profile boundary.

### Versioned diversity and scheduling profiles, not CLI flag collections

Deduplication, diversity, and deterministic ordering have correlated parameters. Exposing
each weight as an environment variable would make the proof binding difficult
to audit and would invite acceptance-specific tuning. They should be stored as
versioned, hashed profiles selected by candidate kind and embodiment.

A `GraspDiversityPolicy` needs fields for:

- backend/source coverage policy;
- translation distance scale;
- approach-direction distance scale;
- complete wrist-rotation/roll distance scale;
- source, approach, and wrist diversity weights;
- formal novelty thresholds for translation, approach, and wrist rotation;
- treatment of declared parallel-gripper symmetry variants;
- deterministic tie-breaking and ordering revision.

A `PlacementDiversityPolicy` needs fields for:

- absolute object-goal translation and rotation distance scales;
- stable contact-face coverage;
- destination-region spatial coverage;
- compiled world-EEF translation and rotation distance scales;
- footprint-aware separation policy;
- deterministic tie-breaking and ordering revision.

A `QualificationSchedulingPolicy` needs fields for:

- deterministic source and grasp-branch fairness;
- stable full-plan submission order for at most four L4-PASS candidates;
- treatment of L4-PASS candidates that remain `NOT_EVALUATED_AT_L5`.

It does not assign a semantic quality rank to the selected four. They are
submitted as insurance against individual planning failure, and every L5 PASS
has equal qualification status. All selected submissions are attempted; a PASS
does not stop evaluation of the remaining selected submissions.

The existing GraspGenX MMR scales, weights, minimum separations, source floor,
the host farthest-first distance formula, and Poisson-disk settings are an
implementation inventory, not yet the canonical profiles. Their exact values
must be reviewed under O6 before being copied into the new contract.

### Task-chain and recovery profiles

Geometry and recovery settings are not generic funnel counts.

The versioned task-chain profile owns, per embodiment and task kind:

- hover, precontact, contact, lift/retract, release, and retreat construction;
- optional-stage rules;
- required virtual attach/detach transition positions;
- any profile-owned release clearance.

The placement recovery profile owns:

- whether the frozen measured-attachment run is required;
- the maximum genuinely new AnyPlace retry count;
- inference-seed derivation by run identity;
- any conservative footprint-erosion and Poisson-disk policy;
- permission to use only explicitly declared object symmetry groups.

There is no regrasp-specific parameter, mode, state, prompt, or tool in this
design.

### Values that are not funnel hyperparameters

The following remain model, calibration, collision, or physical-verification
contracts and must not be exposed as general candidate-funnel tuning knobs:

- coordinate transforms, TCP, robot identity, URDF/SRDF, and joint limits;
- collision geometry and non-negotiable collision margins;
- declared gripper/object symmetry groups;
- the 100 mm planned lift/retract and retreat chain geometry unless a reviewed
  embodiment task-chain profile supplies the equivalent requirement;
- native bilateral-contact and attach/detach ACK requirements;
- at least 80 mm physical lift and at most 10 mm capture-relative drift;
- the final 0.5 s placement-stability window and at most 5 mm terminal drift;
- height, footprint-containment, and execution-evidence gates.

The natural settling observation horizon is an adapter/verifier operational
parameter, not a qualification-funnel parameter. It may be longer than the
final judgment window, but it cannot shorten or relax the final 0.5 s / 5 mm
gate.

### Resolution, validation, and evidence rules

Startup control-plane parameters retain this precedence:

```text
CLI → environment → compiled default
```

All values are parsed strictly. Booleans are not integers, integer strings must
be canonical, floats must be finite, and all bounds are checked before any
service or ROS node starts.

Cross-field validation must include:

```text
configured qualification-input limit <= corresponding configured reserve ceiling
full-plan submission limit <= its scheduler input limit
pregrasp joint full-plan limit <= constructed pregrasp pair count at runtime
placement total run limit = 1 frozen run + 0 or 1 new-model retry
```

If a producer actually returns fewer candidates than its configured ceiling,
the runtime input is `min(configured_limit, materialized_count)`; that is a
truthfully counted smaller batch, not a startup configuration failure.

Health and qualification evidence should report:

- config schema version and resolved config hash;
- each resolved startup value and whether it came from CLI, environment, or
  default;
- selected policy-profile IDs and hashes;
- the actual derived batch limits used by each layer;
- configured limits separately from actual input/evaluated/PASS counts.

No handler may re-read environment variables after startup or locally replace
one resolved field. The complete resolved object is injected into L0-L6 and
the final compatibility serializer.

## Safety and behavior invariants

- Only complete L5 PASS candidates may enter the qualified candidate queue.
- FAIL, UNKNOWN, NOT_EVALUATED, private, stale, and unqualified IDs cannot be
  selected or compiled for execution.
- Frozen requalification never claims a new model inference.
- A new retry pool never merges a failed earlier pool.
- Pair batches preserve both grasp and object-goal parents.
- Virtual scene transitions affect only a cloned PlanningScene.
- Qualification never sends controller goals or changes the live scene.
- Every plan-only record proves `execution_started=false`.
- Real actions always replan from the latest state.
- No field or layer may contain acceptance-scenario-specific behavior.

## Deferred decisions

The approved migration does not require tuning these deferred items:

1. numeric changes to producer/diversity profiles under O6;
2. later removal of legacy counters unrelated to the deleted exposure layer;
3. performance caches, payload batching, or qualification concurrency;
4. any optional MTC experiment.
