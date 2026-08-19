---
name: place
description: Guidance for placing a held object on or inside a target receptacle.
version: v1
editable: true
task_patterns:
  - place <object> on <target>
  - put <object> into <target>
  - place <object> in <target>
allowed_tools:
  - observe
  - sam3
  - select_sam3_detection
  - reject_sam3_detections
  - anyplace
  - compile_grasp_seed
  - move_to
  - gripper_control
---
# Place

Use this skill as text guidance only. Do not treat `place` as an executable
macro. After each tool result, inspect the returned observation or tool output
before choosing the next tool call.

## Recommended Tool Sequence

1. For a combined pick-and-place task, freeze one aligned pre-grasp RGB-D
   observation, but do not infer or plan placement before the grasp succeeds.
   Use the frozen packet for the object and placement-region evidence so it
   remains valid after the object has moved.
2. Retain the targeted `grasp_pose_estimate` result used for pickup, including its selected
   candidate and `details.outputs.source`. On the same original RGB image, call
   `sam3` for the basket, bin, or other placement region and resolve its
   selection obligation with `select_sam3_detection`. Do not segment or select
   the placement region before targeted grasp estimation succeeds: the runtime has one
   active SAM3 selection slot, and doing so would overwrite the selected object
   mask. After object selection, use the RGB, depth, intrinsics, and mask from
   that aligned observation directly; do not call `observe` merely to refresh
   unchanged artifact paths.
3. Complete the pickup using the selected grasp. After closing the gripper,
   require the native attach acknowledgement, fixed lift, and attachment PASS.
   AnyPlace, candidate selection, placement compilation, and transport planning
   remain blocked until this gate completes.
4. Call `anyplace` only now, using the exact host-provided
   `placement_obligation.required_parameters`. These join the original RGB,
   depth, intrinsics, object mask, placement-region artifact, and
   `selected_grasp={candidate, source}` from the successful grasp.
   Never run grasp estimation on the receptacle as a substitute for AnyPlace.
5. Select one retained AnyPlace candidate id using its projection, region
   clearance, score, and candidate image. Call `compile_grasp_seed` with only
   `purpose=placement` and that `placement_candidate_id`. The host binds the
   full pose, source grasp, original camera extrinsics, scene revision, and
   calibrated grasp-to-EEF transform. Never send a raw AnyPlace pose to
   `camera_pose_to_world` or `move_to`.
6. Move directly to the compiled pre-place hover, then descend to the compiled
   release pose. Preserve its full wrist rotation and use the placement motion
   profile; MoveIt computes the joint path under the complete planning scene.
7. Inspect fresh evidence after transport. The earlier lift-probe
   PASS is stale after motion: continue only when the target is still co-located
   with the gripper and its source location remains vacant. If the target is
   visible elsewhere and the closed-gripper openness has collapsed to the empty
   threshold, follow the `attachment_lost` recovery action so the current grasp
   candidate is rejected before regrasping.
8. Descend only to the compiled `placement_motion_guidance.release_pose`, whose
   profile-owned clearance is 5 mm above the AnyPlace low reference.
9. Call `gripper_control` with `position=1` only after the vertical placement
   motion succeeds and fresh evidence still supports attachment over the
   receptacle.
10. Retreat with `move_to`, then call `observe` to verify the object was released
    in the intended place and check the official task reward.

## Recovery Notes

- If the target receptacle or surface is ambiguous, call `ask_human` before
  moving.
- If an already-held object has no retained targeted grasp-estimation provenance or
  pre-grasp aligned placement mask, do not fabricate AnyPlace inputs. Ask for a
  new supported plan or use an explicit task-provided release pose.
- If the target is occluded, observe from another camera or request a broader
  scene query before choosing a release pose.
- If MoveIt rejects a plan before execution starts, reject only that candidate
  and select another retained candidate. Never blindly retry the same request
  fingerprint or describe the coordinate as permanently unreachable.
- If the object remains in the gripper after opening, retry `gripper_control`
  once, observe, then ask for help or replan.
- Never release an object from stale perception. Observe again after every
  world-mutating tool call.

For explicit clearance or controller-profile discovery, use the
`embodiment_explore` skill outside the benchmark episode. Do not copy a
successful value from another robot or environment into this task.
