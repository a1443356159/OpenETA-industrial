# CODEX_INSTRUCTIONS.md

# OpenETA Industrial Gazebo Extension — Engineering Construction Instructions

> Status: FINAL BASELINE
> Date: 2026-08-08
> Upstream: `OpenMOSS/OpenETA`
>
> This document is the authoritative engineering instruction for the first implementation phase.
>
> If existing repository code conflicts with this document, do not silently redesign the architecture. Record the conflict, preserve the OpenETA contract whenever possible, and implement the smallest compatible extension.

---

# 0. Mission

Fork OpenETA and extend it to support an industrial robotic manipulation embodiment based on:

* OpenETA
* Gazebo
* ROS 2
* MoveIt / MoveIt Pro
* SAM3
* OpenETA grasp / placement tools
* Gazebo-grounded physical verification

The project is **not** a new embodied-agent framework.

The project is:

> An industrial Gazebo embodiment extension for OpenETA.

The intended architecture is:

```text
User
  │
  ▼
OpenETA Agent
  │
  ├── LLM Planner
  ├── Working Memory
  ├── Tool / Skill system
  ├── Closed-loop execution
  ├── Rollout / Replay
  └── Evaluation
  │
  ▼
Tool / MCP Boundary
  │
  ├──────────── Perception / Manipulation Reasoning Tools
  │                 │
  │                 ├── SAM3
  │                 ├── grasp_pose_estimate
  │                 │     ├── AnyGrasp
  │                 │     └── GraspGenX
  │                 └── AnyPlace
  │
  └──────────── Gazebo Embodiment Tools
                    │
                    ├── observe
                    ├── inspect
                    ├── pick_place
                    └── oracle/debug tools
                    │
                    ▼
             ROS 2 / MoveIt Pro
                    │
                    ▼
                  Gazebo
                    │
                    ▼
          Gazebo Physical Verifier
                    │
                    ▼
             Fresh Observation
                    │
                    ▼
              OpenETA Agent
```

The OpenETA causal loop must remain:

```text
observe
→ reason
→ act
→ verify world change
→ observe again
→ adapt
```

---

# 1. Non-Negotiable Architecture Rules

## 1.1 Do not build another Agent runtime

Do NOT implement replacements for:

* OpenETA Planner
* Agent loop
* Working Memory
* ToolRegistry
* ToolResult
* Skills
* session management
* rollout recording
* replay
* evaluation runtime
* supervision framework
* provider abstraction

Reuse OpenETA.

Do not introduce:

```text
LangGraph
custom workflow engine
custom task state machine
custom transition policy
custom lease system
custom callback store
custom event bus
custom execution database
```

unless a concrete upstream limitation is demonstrated and documented.

---

## 1.2 Preserve OpenETA's cognition / execution boundary

The LLM proposes intent.

The host and embodiment layer own execution authority.

The LLM must never directly control:

* ROS topics
* joint positions
* raw controller commands
* Gazebo entity APIs
* arbitrary XYZ trajectories
* collision parameters
* attachment state

The LLM interacts through registered OpenETA tools.

---

## 1.3 Preserve OpenETA's evidence boundary

These are different facts:

```text
tool completed
≠
world changed
≠
task succeeded
```

Never mark manipulation success only because:

```text
MoveIt returned success
```

or:

```text
gripper close returned success
```

A physical/environment checker must establish the relevant world consequence.

---

## 1.4 Preserve fresh-observation semantics

Every world-mutating operation must follow OpenETA's fresh-observation rule.

Conceptually:

```text
Observation N
   ↓
World-mutating Tool
   ↓
Execution result
   ↓
Fresh Observation N+1
   ↓
Planner may reason again
```

Do not allow:

```text
world mutation
→ world mutation
→ world mutation
```

using stale physical observations.

---

## 1.5 No SceneGraph in V1

Do NOT create:

* SceneGraph server
* SceneGraph database
* graph memory
* relation inference subsystem
* world-model service

Use OpenETA's existing Observation + Working Memory model.

World information exposed to the planner should remain a compact object-oriented summary.

Example:

```json
{
  "objects": [
    {
      "id": "screw_17",
      "label": "silver screw",
      "confidence": 0.96,
      "position": [0.42, -0.11, 0.03],
      "visibility": "clear"
    },
    {
      "id": "bin_3",
      "label": "storage bin",
      "position": [0.61, 0.20, 0.08]
    }
  ]
}
```

Only add additional structure when an actual task requires it.

---

## 1.6 Do not add a VLM Planner in V1

Planner input should be primarily:

* user instruction
* current OpenETA observation
* object summary
* Working Memory
* ToolResult
* available Tool schemas
* selected Skills

Perception of RGB images is handled by SAM3 and other explicit perception tools.

Do not introduce VLM-based scene reasoning unless later experiments establish a concrete need.

---

# 2. Upstream OpenETA Components to Reuse

Treat the following as upstream infrastructure.

Reuse rather than rewrite.

## Agent

Use:

* `ToolCallingPlanner`
* OpenAI-compatible Planner backend
* Working Memory / `AgentMemory`
* ToolRegistry
* ToolResult
* SkillRegistry
* episode runner
* supervision
* rollout recorder

---

## Existing semantic tools

Reuse OpenETA's existing capabilities where applicable:

### Perception

```text
SAM3
UniDepth where necessary
```

For Gazebo RGB-D, prefer simulator-provided metric depth instead of monocular depth inference.

---

### Grasp

Use the OpenETA grasp abstraction rather than writing a new grasp model.

Expected backend path:

```text
grasp_pose_estimate
    │
    ├── AnyGrasp
    └── GraspGenX
```

Do not expose model-backend selection unnecessarily to the Planner.

---

### Placement

Use:

```text
AnyPlace
```

Do not implement a new placement network.

---

# 3. Final Responsibility Split

The system must preserve the following ownership.

| Component                | Responsibility                                                           |
| ------------------------ | ------------------------------------------------------------------------ |
| OpenETA Planner          | Decide what to do next                                                   |
| OpenETA Working Memory   | Maintain session-local task state, candidates, failures and artifacts    |
| SAM3                     | Identify / segment target objects and destination regions                |
| AnyGrasp / GraspGenX     | Generate grasp candidates                                                |
| AnyPlace                 | Generate placement candidates                                            |
| MoveIt / MoveIt Pro      | Robot motion planning and execution                                      |
| Gazebo                   | Physical simulation                                                      |
| Gazebo Adapter           | Translate OpenETA environment/tool contracts into Gazebo/ROS 2 execution |
| Gazebo Physical Verifier | Establish simulated physical consequences                                |
| Industrial SAM3 work     | Improve perception for industrial objects                                |

---

# 4. Working Memory Rules

Do NOT build a new memory subsystem.

Use OpenETA Working Memory.

Examples of information that belongs in Working Memory:

```text
current task
selected target
selected destination
selected SAM3 detection
grasp candidates
active grasp candidate
placement candidates
active placement candidate
last failure
retry state
current execution state
scene epoch / observation provenance
```

Large binary/numeric data should remain artifacts:

```text
RGB image
Depth image
mask
point cloud
grasp candidate payload
placement candidate payload
trajectory
```

The Planner should receive bounded summaries and references, not giant raw payloads.

---

# 5. Manipulation Design

## 5.1 Cognitive tools remain explicit

The embodied brain should determine whether sufficient information exists to execute manipulation.

Expected reasoning flow:

```text
observe
   ↓
identify target
   ↓
is grasp known?
   ├── no → grasp_pose_estimate
   └── yes
   ↓
is destination known?
   ├── no → perceive / inspect
   └── yes
   ↓
is placement known?
   ├── no → AnyPlace
   └── yes
   ↓
pick_place
```

The LLM decides which semantic tool is needed next.

---

## 5.2 `pick_place` must be execution-only

Do NOT implement `pick_place` as:

```text
SAM3
→ AnyGrasp
→ AnyPlace
→ MoveIt
→ verify
```

hidden inside one tool.

That would bypass the OpenETA embodied reasoning loop.

Instead:

```text
pick_place(target, destination)
```

is an execution capability.

Before execution, the host must verify that Working Memory already contains valid manipulation information.

Example readiness gate:

```text
target resolved
AND
active grasp candidate exists
AND
destination resolved
AND
active placement candidate exists
AND
candidate provenance is compatible with current observation / scene epoch
```

If not ready:

```text
pick_place
→ NOT_READY
→ structured missing-information feedback
→ Planner decides next Tool
```

---

## 5.3 LLM must not rewrite geometric candidates

Do not require the LLM to copy complete 6-DoF poses.

Prefer host-owned candidate references.

Conceptually:

```text
grasp candidate ID
placement candidate ID
```

with actual pose data retained in Working Memory / artifacts.

The Host resolves them before calling MoveIt.

---

# 6. Gazebo Embodiment Adapter

This is the main new engineering component.

Follow OpenETA's simulator separation philosophy.

Prefer implementing Gazebo behind the same Tool/MCP/environment boundary rather than adding direct ROS access to the Planner.

Suggested package:

```text
extensions/gazebo/
```

Possible structure:

```text
extensions/gazebo/
├── observation.py
├── lifecycle.py
├── robot.py
├── cameras.py
├── manipulation.py
├── verifier.py
├── checker.py
├── mcp_server.py
└── config.py
```

Do not create unnecessary service layers.

---

# 7. Gazebo Observation Contract

Gazebo observation must provide enough information for OpenETA and perception tools.

Minimum required information:

```text
task description
camera frames
RGB
metric depth
camera intrinsics
camera extrinsics
robot joint state
EEF pose
gripper state
optional simulator object summary
metadata
```

Respect OpenETA's existing camera and depth conventions wherever possible.

Metric units:

```text
position → metres
depth → metres
orientation → documented convention
```

Every camera packet must explicitly identify frame conventions.

Never infer coordinate conventions from backend names.

---

# 8. Camera Configuration

V1 uses two camera roles.

## Top camera

Role:

```text
global scene observation
```

Used primarily by:

```text
observe
SAM3 global perception
destination detection
```

---

## Wrist / palm camera

Role:

```text
target refinement
active perception
close-range grasp perception
```

Used by:

```text
inspect(target)
```

`inspect` may cause robot motion.

Therefore it is a world-mutating embodied operation and must obey OpenETA fresh-observation semantics.

---

# 9. Gazebo Lifecycle

Implement:

```text
create
reset
observe
step / execute
close
```

where required by the chosen OpenETA environment boundary.

Reset must be deterministic given:

```text
task
seed
configuration
```

Reset responsibilities include:

```text
robot home state
gripper state
object poses
task state
camera readiness
physics settling
verification state
```

Every created Gazebo environment/session must be cleanly destroyed.

Resource leakage is a test failure.

---

# 10. ROS 2 / MoveIt Integration

The embodiment layer owns robot execution.

Recommended boundary:

```text
OpenETA
    ↓
Gazebo Tool / MCP Adapter
    ↓
ROS 2
    ↓
MoveIt / MoveIt Pro
    ↓
ros2_control
    ↓
Gazebo
```

Do not send raw ROS commands from the LLM.

MoveIt owns:

```text
IK
collision checking
motion planning
trajectory execution
joint constraints
robot geometry
```

MoveIt completion is execution evidence, not physical task-success evidence.

---

# 11. Gazebo Attachment / Grasp Verification

This is a required implementation.

## 11.1 Problem

This is invalid:

```text
gripper_control(close) == success
therefore
object grasped == true
```

The two claims are different.

---

## 11.2 V1 grasp verification

Implement a Gazebo-grounded attachment / lift checker.

After closing the gripper:

```text
close
↓
small lift probe
↓
observe simulator state
```

Evaluate evidence such as:

```text
target moved upward
target remains spatially coupled with EEF
target does not remain on support surface
gripper state is compatible with holding an object
optional Gazebo contact / attachment signal
```

Return three-state semantics when possible:

```text
SUCCESS
FAILURE
UNKNOWN
```

Do not force uncertainty into success/failure.

---

## 11.3 Bilateral-contact kinematic adhesion

M3 uses repository-owned kinematic adhesion after Gazebo's two real fingertip
contact sensors prove the same known object is simultaneously present at both
pads. The physics engine remains authoritative for contact and post-release
settling, not for holding the object.

```text
fresh left-pad contact window
+
fresh right-pad contact window
+
same target or distractor identity
→ capture with bilateral_contact_adhesion_v1

gripper open
→ open, release, restore dynamics, settle
```

The plugin is isolated in the Gazebo embodiment layer and its state alone is
never a task-success proof: `TARGET_HELD` still requires stable relative pose,
actual target lift, and no distractor co-motion. Do not leak this
simulator-only mechanism into Planner logic.

---

# 12. Placement Verification

Placement success is not:

```text
gripper opened successfully
```

V1 placement verification should check:

```text
target no longer attached
AND
target is inside / on intended destination
AND
target has settled sufficiently
```

Possible evidence:

```text
target pose
destination volume / region
linear velocity
angular velocity
attachment state
```

Use deterministic geometry for simulation.

---

# 13. Trusted Environment Result

Physical verification must feed OpenETA through trusted host/environment semantics.

Do not let Planner-generated text establish success.

A successful physical task result should originate from:

```text
Gazebo truth
+
checker
+
environment receipt / equivalent trusted result
```

OpenETA Working Memory can then record the result.

---

# 14. Oracle / Debug Capability

A simulator-only oracle is allowed.

Example capability:

```text
get_oracle_pose(target)
```

Purpose:

```text
debugging
ablation
perception isolation
baseline
```

Rules:

* mark clearly as simulator-only
* do not expose in formal perception profile
* never mix oracle pose silently into SAM3 perception results
* preserve provenance

Profiles should include at least:

```text
oracle
perception
```

Optional:

```text
hybrid
```

---

# 15. Industrial SAM3

This is the main learning/model adaptation work.

Do NOT fork SAM3 architecture unnecessarily.

Reuse upstream model implementation.

Develop:

```text
industrial dataset
training configuration
fine-tuned checkpoints
evaluation
OpenETA SAM3 adapter compatibility
```

Initial industrial object families may include:

```text
screw
nut
roller
bearing
fixture
bin
tool
industrial component
```

---

# 16. SAM3 Synthetic Data Pipeline

Gazebo should also serve as a synthetic-data generator where useful.

Generate:

```text
RGB
depth
segmentation ground truth
object identity
camera calibration
object pose
scene metadata
```

Randomization may include:

```text
object pose
object count
lighting
camera noise
textures
background
occlusion
```

Do not optimize synthetic diversity before the basic system works.

---

# 17. Object Summary

No SceneGraph in V1.

Perception should produce compact object-oriented summaries.

Example:

```json
{
  "objects": [
    {
      "id": "screw_17",
      "label": "silver screw",
      "confidence": 0.96,
      "visibility": "partial",
      "position": [0.42, -0.11, 0.03],
      "source_camera": "top"
    }
  ]
}
```

Exact 6-DoF pose is optional when not required by the current Tool.

Do not invent precise orientation just to fill a schema.

Use `unknown` / missing fields where appropriate.

---

# 18. `observe`

Global observation tool.

Expected implementation:

```text
top RGB-D
↓
SAM3 / object extraction
↓
object summary
↓
OpenETA Observation / Working Memory
```

It is primarily perception/read-only.

---

# 19. `inspect(target)`

Local active-perception capability.

Expected implementation:

```text
current target estimate
↓
MoveIt view motion
↓
wrist RGB-D
↓
SAM3
↓
updated target observation
```

Do not expose arbitrary `move_camera_xyz` to the Planner.

The Planner asks for semantic intent:

```text
inspect target
```

The Host determines safe robot motion.

---

# 20. Grasp Tool

Reuse OpenETA's grasp façade.

Planner-facing semantics:

```text
grasp_pose_estimate(target)
```

Possible backend order remains host-owned.

Do not make the LLM manually compare incompatible backend scores.

Store:

```text
candidate provenance
backend
score
scene epoch
camera
target
```

---

# 21. Placement Tool

Reuse AnyPlace.

Planner-facing semantics should remain close to upstream OpenETA.

Preserve the requirement that placement geometry must be grounded in compatible observation data.

Do not fabricate destination poses.

---

# 22. `pick_place`

Execution-only tool.

Conceptual API:

```text
pick_place(
    target_id,
    destination_id
)
```

Before execution:

```text
resolve target
resolve selected grasp from Working Memory
resolve destination
resolve selected placement from Working Memory
validate provenance/freshness
```

Then execute through MoveIt.

If prerequisites are missing:

```text
return NOT_READY
```

with structured missing fields.

Example:

```json
{
  "status": "NOT_READY",
  "missing": [
    "active_placement_candidate"
  ]
}
```

The Planner then selects another Tool.

---

# 23. Repository Modification Policy

Prefer minimal changes to upstream OpenETA.

Before editing upstream core code:

1. verify extension cannot be implemented through existing interfaces;
2. document why;
3. isolate patch;
4. add regression tests;
5. keep patch small enough for future upstream rebase.

Do not casually rewrite:

```text
agent/runtime/
agent/backends/
agent/tools/registry.py
agent/runtime/memory.py
```

Use their APIs first.

---

# 24. Branch Strategy

Recommended:

```text
main
develop
feature/gazebo-observation
feature/gazebo-control
feature/gazebo-verifier
feature/moveit-integration
feature/industrial-sam3
```

Configure upstream:

```text
origin   → our fork
upstream → OpenMOSS/OpenETA
```

Regularly:

```text
fetch upstream
merge/rebase consciously
run full regression
```

Do not let the fork diverge unnecessarily.

---

# 25. Commit Policy

Use small scoped commits.

Examples:

```text
feat(gazebo): add RGB-D observation adapter

feat(gazebo): add simulator lifecycle MCP tools

feat(moveit): add manipulation execution backend

feat(verify): add grasp lift-probe checker

feat(perception): add industrial SAM3 adapter

test(gazebo): add deterministic reset test

docs(industrial): document Gazebo embodiment contract
```

One commit should represent one coherent engineering change.

---

# 26. Testing Pyramid

## Unit tests

Required for:

```text
coordinate transforms
camera packets
depth conversion
object summary serialization
candidate provenance
attachment verifier
placement geometry
ToolResult normalization
```

---

## Contract tests

Verify Gazebo output satisfies OpenETA-facing contracts.

Test:

```text
observation schema
tool schema
result schema
side-effect classification
fresh-observation behavior
resource cleanup
```

---

## Integration tests

At minimum:

```text
OpenETA → Gazebo observe

OpenETA → ROS2/MoveIt → Gazebo motion

SAM3 → grasp candidate

AnyPlace → placement candidate

pick_place → Gazebo verifier
```

---

## End-to-end tests

Initial benchmark tasks:

```text
pick one object
pick and place one object
sort one object into specified bin
sort multiple objects
recover after failed grasp
```

---

# 27. Metrics

Do not report only task success.

Track separately:

### Agent

```text
task success
tool calls
planner turns
recovery count
```

### Perception

```text
detection recall
mask IoU
wrong-target rate
```

### Grasp

```text
candidate generation success
reachable grasp rate
physical grasp success
```

### Placement

```text
placement generation success
physical place success
```

### Verification

```text
true positive
true negative
false success
false failure
unknown rate
```

False-success rate is especially important.

---

# 28. Development Order

Follow this order.

Do NOT begin all subsystems simultaneously.

---

## M0 — Upstream Baseline

Goal:

```text
unmodified OpenETA runs
```

Tasks:

* install dependencies
* run tests
* run one existing simulator example
* inspect ToolRegistry
* inspect Working Memory
* inspect rollout output

Acceptance:

```text
OpenETA baseline green
```

---

## M1 — Gazebo Read-Only Observation

Goal:

```text
OpenETA can create/reset/observe/close Gazebo environment
```

Use oracle simulator object summaries initially.

No SAM3 requirement yet.

Acceptance:

```text
Planner receives valid Gazebo observation
```

---

## M2 — Gazebo Robot Control

Goal:

```text
OpenETA Tool
→ ROS2 / MoveIt
→ Gazebo robot
```

Implement:

```text
robot motion
gripper open
gripper close
```

Acceptance:

```text
repeatable controlled motion
```

---

## M3 — Physical Verification

Implement:

```text
grasp verification
placement verification
trusted environment result
```

Acceptance:

Positive and negative test scenes must both pass.

Explicitly test:

```text
successful grasp
empty gripper close
wrong-object grasp
object dropped during lift
successful placement
object outside destination
```

---

## M4 — Full Oracle Pick/Place

Use Gazebo ground truth perception.

Goal:

```text
LLM
→ grasp planning
→ placement planning
→ MoveIt
→ Gazebo
→ verifier
→ fresh observation
```

Do not proceed to SAM3 until this path is reliable.

---

## M5 — Generic SAM3 Integration

Replace oracle object discovery with upstream SAM3.

Keep Gazebo ground truth only for evaluation.

Acceptance:

```text
same manipulation flow works with SAM3 perception
```

---

## M6 — Industrial SAM3 Fine-Tuning

Build dataset and fine-tune.

Compare:

```text
upstream SAM3
vs
industrial SAM3
```

using identical Gazebo tasks.

---

## M7 — Industrial Benchmark

Build reproducible task manifests.

Include:

```text
multiple object identities
occlusion
different placements
lighting variation
clutter
failure injection
```

Use OpenETA rollout/evaluation infrastructure instead of building a second evaluator.

---

# 29. Engineering Governance

Every new subsystem must have:

```text
owner
interface
tests
failure semantics
logging
configuration
documentation
```

Every design decision must answer:

1. Is this already provided by OpenETA?
2. Is this already provided by ROS2 / MoveIt / Gazebo?
3. Is this truly industrial-domain-specific?
4. Can this remain an adapter instead of a new framework?

If the first or second answer is yes, reuse it.

---

# 30. Configuration Policy

No important robot/simulator values hard-coded in Python.

Use configuration for:

```text
robot name
MoveIt group
gripper
camera topics
camera frames
world
verification thresholds
object assets
SAM3 checkpoint
grasp backend
placement backend
```

Keep environment-specific parameters outside Planner prompts.

---

# 31. Logging and Reproducibility

Each episode must retain enough provenance to reproduce:

```text
OpenETA commit
fork commit
Gazebo world version
robot config
task seed
SAM3 checkpoint
grasp backend
placement backend
MoveIt configuration
verification configuration
Planner model
```

Reuse OpenETA rollout infrastructure.

Do not build a second trajectory recorder.

---

# 32. Error Semantics

Do not collapse everything into boolean success.

Use structured errors such as:

```text
TARGET_NOT_FOUND
AMBIGUOUS_TARGET
NO_GRASP_CANDIDATE
NO_PLACEMENT_CANDIDATE
STALE_CANDIDATE
IK_FAILED
MOTION_PLAN_FAILED
GRIPPER_FAILED
GRASP_NOT_VERIFIED
OBJECT_DROPPED
PLACE_NOT_VERIFIED
ENVIRONMENT_TIMEOUT
UNKNOWN_PHYSICAL_OUTCOME
```

`UNKNOWN` is not `FAILURE`.

---

# 33. First Codex Assignment

Codex must begin only with **M0 and M1**.

Do not start SAM3 fine-tuning or MoveIt manipulation in the first implementation pass.

First tasks:

### Task A — Repository audit

Inspect current OpenETA:

```text
README.md
docs/architecture.md
docs/agent-action-pipeline.md
sim/README.md
agent/README.md
real/README.md
adapter/
agent/tools/sim_mcp.py
```

Produce:

```text
docs/gazebo-integration-audit.md
```

Document:

* exact extension points
* existing environment contracts
* MCP lifecycle contract
* observation schema
* ToolResult requirements
* world-mutating tool semantics
* cleanup requirements
* minimum upstream files requiring modification

Do not change architecture during this task.

---

### Task B — Gazebo adapter design

Produce:

```text
docs/gazebo-adapter-design.md
```

It must specify:

```text
Gazebo process lifecycle
ROS2 lifecycle
observation mapping
camera mapping
robot-state mapping
MCP tools
reset semantics
resource cleanup
error semantics
test strategy
```

---

### Task C — Minimal Gazebo backend

Implement only:

```text
create/reset
observe
close
```

with oracle object summaries.

Do not implement perception yet.

Do not implement physical manipulation yet.

Acceptance test:

```text
OpenETA episode
→ Gazebo environment created
→ observation returned
→ reset works deterministically
→ environment closes without leak
```

---

# 34. Stop Conditions

Codex must stop and document before proceeding if any of these occur:

```text
OpenETA core contract must apparently be changed

Gazebo observation cannot map cleanly to existing observation contract

ROS2/Gazebo process lifecycle cannot be isolated

coordinate-frame convention is ambiguous

camera depth is not metric or calibration is missing

MoveIt integration would require Planner-visible raw trajectories

physical verification cannot distinguish SUCCESS / FAILURE / UNKNOWN
```

Do not resolve such problems through large speculative rewrites.

---

# 35. Explicitly Forbidden Scope Expansion

Do not add in V1:

```text
SceneGraph
VLM Planner
World Model
VLA
RL training
multi-agent coordination
distributed robot scheduler
custom orchestration runtime
new memory database
new event system
new web gateway
new dashboard
general physical verification framework
digital twin platform
```

These are out of scope unless separately approved.

---

# 36. Definition of Done — V1

V1 is complete when the following scenario works reproducibly:

User:

```text
Put the silver screw into bin three.
```

System:

```text
OpenETA LLM Planner
        ↓
observe Gazebo
        ↓
SAM3 identifies object / destination
        ↓
grasp_pose_estimate
        ↓
AnyGrasp / GraspGenX candidate
        ↓
AnyPlace placement candidate
        ↓
pick_place
        ↓
MoveIt / MoveIt Pro
        ↓
Gazebo
        ↓
physical grasp/place verification
        ↓
fresh observation
        ↓
OpenETA determines task completion
```

And:

```text
the episode is reproducible
the rollout is recorded
failure is diagnosable
oracle mode can isolate perception failures
no SceneGraph exists
no custom Agent runtime exists
no duplicate Memory system exists
```

---

# 37. Final Engineering Principle

The project must remain conceptually:

```text
OpenETA
    = embodied brain and agent harness

SAM3
    = industrial visual perception

AnyGrasp / GraspGenX
    = grasp reasoning

AnyPlace
    = placement reasoning

MoveIt / MoveIt Pro
    = robot motion execution

Gazebo
    = simulated physical world

Gazebo Verifier
    = physical consequence authority
```

Our contribution is the integration and industrial adaptation:

```text
OpenETA
+
Gazebo embodiment
+
industrial SAM3
+
physics-grounded verification
```

Do not turn this project back into a new general-purpose agent framework.

---

# 38. Immediate Instruction to Codex

Start by auditing the current upstream repository.

Then implement the smallest possible Gazebo observation/lifecycle integration.

Do not implement later milestones early.

At every milestone:

```text
read existing OpenETA code
reuse existing contracts
write tests first where practical
implement minimal adapter
run upstream regression tests
document deviations
commit narrowly
```

Architecture simplification is a feature.

Upstream compatibility is a requirement.

Physical truth belongs to the embodiment/environment, not to the LLM.
