# OpenETA qualification funnel v3

`fast_v3` is an opt-in CPU-first qualification profile. The shipped default is
`legacy`; `shadow` keeps legacy results authoritative while recording private
v3 evidence. Promotion to `fast_v3` is an operational decision made only after
the replay and shadow gates below pass.

## Candidate and scheduling contract

- AnyPlace's returned pool is already model-filtered. v3 does not apply a
  second hard Top-K. Each frozen grasp branch may pair with all 96 goals, but
  those pairs are materialized lazily and the first complete pair is executed.
- Before IK, placement qualification executes one complete cheap deterministic
  barrier. It evaluates each unique AnyPlace object goal exactly once:
  proper finite SE(3), complete footprint inside the declared placement
  region, exact scene-object bounding box/support height, static-box
  penetration, and the conservative analytic workspace envelope. Pair-level
  geometry is evaluated only when a candidate reaches its small deep wave:
  exact model-derived release
  consistency, terminal analytic reach, attached-object collision, and exact
  URDF primitive collision for the gripper mount. Untouched pairs remain
  explicitly `NOT_EVALUATED`, so an early solution does not pay for all pair
  checks. No hover, lift, descent
  offset, clearance pose, or retreat is compiled. These analytic proofs are
  prefilters only; every selected state still requires MoveIt state validity
  and L5 plan-only proof.
- A parallel-jaw 180-degree roll twin keeps its own candidate/result evidence,
  but reuses the pair-legality proof of its explicitly marked physical family.
  Equivalence changes ordering and evidence reuse, never the retained model
  candidate set.
- GraspGenX freezes 512 model candidates once. Grasp waves are incremental
  slices with cumulative limits `4 → 8 → 16 → 32 → 64 → 128 → 256`; only if
  those waves cannot produce a complete grasp/place branch does the
  deterministic implicit exhaustion wave visit the remaining frozen pool.
  Placement waves use cumulative limits
  `4 → 8 → 16 → 32 → 96` per grasp branch. Candidate IDs, branch IDs,
  and 10 mm / 10 degree SE(3) cluster IDs determine a stable round-robin order.
- Capability-map density, joint margin, singular value, and generator score
  alter ordering only. A missing cell has zero confidence and cannot reject a
  candidate. The map content hash is bound to URDF, SRDF, planning group, TCP,
  and gripper; a missing configured file or cross-model map fails as
  configuration evidence rather than producing an unreachable verdict.
- Within a wave, each candidate proceeds directly through pair legality,
  Beam-2 IK, state validity, and then the serial L5 tail. Each wave ends at a
  deterministic barrier. Up to eight IK and eight state-validity calls run for
  different candidates, while each candidate's stage chain remains ordered.
  Results are merged by fixed candidate index, and the run-local seed cache is
  updated only after that merge.
- L5 is serial after each barrier and follows physical-quality ordering. A
  plan-only failure advances to the next L4 candidate and then the next wave.
  Sixty seconds is measured as a P95 objective, never used as a search cutoff.

## Beam and recovery contract

Every endpoint propagates at most two non-duplicate joint solutions. The first
stage starts from the real state plus the nearest prior-wave success, or named
home when the cache is empty. Later stages start from parent Beam solutions and
use the wave cache only to fill a missing second seed.

A successful pure IK state is checked with MoveIt state validity. If valid, it
is accepted without a redundant collision-aware IK call. If it collides, that
seed receives one collision-aware rescue and another state-validity check. A
pure no-solution result receives no random retry. Solutions closer than 0.05 in
normalized joint space are deduplicated, then ordered by validity, normalized
joint-limit margin, minimum Jacobian singular value, cumulative travel, rescue
count, generator score, and fixed candidate index.

The live ROS adapter computes that singular value for the concrete returned
joint branch with a geometric Jacobian parsed from the same expanded URDF. A
missing or malformed runtime chain is a configuration error; it is never
silently replaced by a reachability verdict.

The first grasp that completes its paired placement L5 proof is executed.
Untouched candidates remain in the same frozen provider frontier. A
candidate-linked execution failure resumes that frontier through an explicit
`model_inference=false` request and proves the next complete pair. SAM3,
AnyPlace, and the grasp model are not rerun unless the scene actually changed
or the frozen pool was exhausted.

Only after every applicable fast wave has failed does v3 use six fixed low-discrepancy
recovery seeds, completing the existing eight-seed budget. Runtime RNG and
uncontrolled `${HOME}/.ros` caches are not inputs.

## Profiles and evidence

The startup profile is selected with `OPENETA_QUALIFICATION_PROFILE`:

- `legacy`: current v2 scheduling and proof semantics;
- `shadow`: legacy result plus private `shadow_fast_v3` comparison evidence;
- `fast_v3`: v3 scheduling is authoritative.

MoveIt loads the solver selected by `OPENETA_QUALIFICATION_SOLVER_PROFILE`.
Available profiles are `kdl_legacy`, `kdl_fast`, `trac_ik_speed`,
`trac_ik_distance`, and `pick_ik_local`. For `pick_ik_local`, the completed
fast-wave barrier dynamically changes the plugin to `global` only while the
fixed recovery seeds run, then restores `local`. `auto` resolves to
`kdl_legacy` for `legacy`/`shadow` and `kdl_fast` for `fast_v3`. A
request/startup mismatch or mode-switch failure is an infrastructure
configuration error, not an unreachable candidate.

Artifacts use the v3 funnel protocol and declare
`openeta.moveit_candidate_qualification.v3` as their artifact schema. They
record robot/scene/capability-map hashes, provider and solver versions,
candidate/cluster/wave/seed lineage, endpoint timing and rescue evidence,
joint-quality values, L5 attempt order, concurrency/cache metrics, and the
final stop reason. Goal and pair evidence additionally records unique goal
count, pair-reached/evaluation/reuse/pending counts, hard-rejection reasons,
and goal/pair latencies. The reader continues to accept v1 and v2 artifacts.
The read-only Dashboard keeps first-L5 latency history partitioned by solver
configuration and displays its P50/P95 alongside wave, concurrency, cache,
rescue, layer-count, and failure-reason metrics. It is not an input to the
qualification executor.

## CPU bake-off and promotion

Generate a capability map with:

```bash
PYTHONPATH=. python3 scripts/generate_capability_map.py \
  --urdf XACRO_EXPANDED_ROBOT.urdf --srdf ROBOT.srdf \
  --planning-group rm_group --tcp link_7 --gripper robotiq_2f85 \
  --joint-lower=-3.14,-3.14,-3.14,-3.14,-3.14,-3.14,-3.14 \
  --joint-upper=3.14,3.14,3.14,3.14,3.14,3.14,3.14 \
  --output capability.json
```

The default provider parses the expanded URDF directly. The optional
`--kinematics-plugin package.module:factory` override remains available for a
separately audited MoveIt or Pinocchio provider.

Set `OPENETA_CAPABILITY_MAP_ID` to the printed content hash. Store the file as
`config/capability_maps/<MAP_ID>.json`, or set the explicit
`OPENETA_CAPABILITY_MAP_PATH`. The ROS qualification process loads it from
this controlled path; it never reads `${HOME}/.ros` caches.

Run the replay selection over model-matched artifacts with:

```bash
PYTHONPATH=. python3 scripts/qualification_solver_bakeoff.py --print-matrix
PYTHONPATH=. python3 scripts/qualification_solver_bakeoff.py ARTIFACT_DIR \
  --robot-model-sha256 MODEL_HASH --output solver-selection.json
```

The emitted frozen matrix contains the Legacy KDL baseline plus Fast KDL,
TRAC-IK Speed/Distance, and pick_ik local at 20/50/100/200 ms, all with ten
deterministic repetitions and concurrency eight. Selection fails closed when
any matrix configuration is missing; `--allow-partial-matrix` is diagnostic
only and must not be used for promotion.

The selector rejects a configuration unless it retains every known full L5
PASS, has task recall no lower than legacy, has no infrastructure errors, and
produces one identical selection/joint-branch/failure signature across ten
runs per case. It then chooses the lowest first-PASS P95; within five percent,
the dependency order is KDL, TRAC-IK, then pick_ik.

After CPU shadow, collect at least 30 cold and 30 warm `normal` runs. A cuRobo
sidecar is considered if either population's P95 is still above 60 seconds or
full-PASS recall is below legacy. Keeping the populations separate prevents
fast warm runs from masking a cold-start regression. Any future GPU result
remains provisional until MoveIt state validity and segmented L5 plan-only
both pass.
