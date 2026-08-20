# Gazebo M6: real GraspGenX/AnyPlace constraint-correct placement and recovery

M6 keeps the existing AgentTool surface. `move_to` remains one MoveIt request
that performs IK, collision-aware planning, and execution. There is no separate
IK preview, route planner, trajectory tool, fixed world wrist orientation, or
goal-region extension.

## Observed failure and semantics

The remote run returned a generic MoveIt failure for the old fixed-orientation
transport request. That proves only that the current joint state, current
planning scene, exact target, and tolerances did not produce a plan. A generic
numeric result such as `99999` is not evidence of a specific IK or collision
cause, and a failed request does not make a spatial coordinate permanently
unreachable.

Motion failure receipts include MoveIt code, planned point count,
`execution_started`, scene revision, and a fingerprint over start joints,
target, tolerances, and scene. A planning failure with
`execution_started=false` rejects only the active placement candidate. The same
fingerprint is not retried. Unknown execution outcome stops for reconciliation
or human handling.

## Perception and compilation boundary

Before the first grasp motion, one RGB-D packet is frozen and feeds target SAM3,
GraspGenX for the `robotiq_2f_85` embodiment, and placement-region SAM3. No
AnyPlace inference, placement candidate
selection, placement compilation, or transport planning occurs yet. After
close, Gazebo attach acknowledgement, and the unchanged M3 lift gate pass,
AnyPlace runs against that frozen packet and retained source grasp. Every one of
its ten generated candidates retain that source grasp, receive MoveIt qualification, and carry a
projection/region-clearance summary plus a candidate image attachment.

Only then does the main VLM call `compile_grasp_seed` with:

```json
{"purpose":"placement","placement_candidate_id":"placement_002"}
```

The host resolves that id against retained working memory and binds the full
camera-frame pose, source grasp, original camera extrinsics, scene epoch, and
embodiment `T_grasp_eef`. The output is a world-frame EEF hover/release pair
with the candidate's full rotation. Grasp-strategy orientation clamps are not
applied to placement. Raw AnyPlace and grasp-estimator poses fail closed at the
motion proxy.

## Motion and scene constraints

M3 approach/capture/lift and its `0.0002 m / 0.002 rad` gate are unchanged.
After lift, placement uses:

- direct MoveIt planning to candidate release XY at least 100 mm above release;
- full compiled wrist rotation, without direction path constraints;
- `0.002 m / 0.05 rad` goal tolerances and `0.1` velocity/acceleration scaling;
- release 5 mm above AnyPlace's low reference, then open and detach.

The adapter applies and reads back a MoveIt planning scene. Reset contains the
table, distractor, and target. Only target contact with the two fingertip touch
links is allowed. After Gazebo attach ACK, the native target/mount state moves
the target from world collision object to an attached object on
`gripper_mount_link`; after detach ACK, the latest native target pose is added
back to world. Apply failure, readback mismatch, or set/attachment mismatch
marks the scene unavailable and blocks motion. The attached payload therefore
participates in table and distractor collision checking.

## Recovery and acceptance

Candidate rejection retains the grasp and asks the main VLM to select another
retained candidate. If all fail, return through the verified source hover and
capture geometry, then open/detach, reobserve, select a new GraspGenX candidate,
and rerun AnyPlace. Unsafe return or uncertain motion stops for human handling.

## Remote real-model deployment

GraspGenX and AnyPlace source, isolated environments, gripper assets, and
checkpoints belong under `/root/autodl-tmp/openeta-services/` on the approved
RTX 4090 service node, beside but isolated from the existing SAM3 service.
Nothing below `third_party/` in a developer checkout is an accepted model
deployment. GraspGenX and AnyPlace must use separate environments because their
official PyTorch stacks are incompatible.

The pinned revisions are GraspGenX `b9429097`, model repository `7c834043`,
gripper assets `19a03c00`, and AnyPlace `3049f78a`. The canonical working tree
is `/root/autodl-tmp/OpenETA-industrial`; model sources, environments, assets,
and checkpoints are deployed separately under
`/root/autodl-tmp/openeta-services/m6`. Checkpoint SHA-256 values and real MCP
smoke results must be recorded from that server. Local workstation artifacts
are never M6 evidence.

Live acceptance requires real SAM3, official GraspGenX checkpoints and
`robotiq_2f_85` gripper assets, official AnyPlace, a
main-VLM selection with no Oracle, native bilateral contact, attach ACK, at
least 80 mm lift, at most 10 mm relative drift, end-to-end candidate/calibration/
scene-revision provenance, fresh receipts, no repeated fingerprint, detach ACK,
and stable marked-zone placement. Any missing or incompatible GraspGenX,
AnyPlace, SAM3, MoveIt, or Gazebo dependency blocks live M6 rather than being
substituted with AnyGrasp, mocks, fixed candidates, or Oracle state.
