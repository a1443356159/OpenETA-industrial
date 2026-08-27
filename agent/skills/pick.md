---
name: pick
description: Guidance for acquiring a target object with atomic tools.
version: v1
editable: true
task_patterns:
  - pick <object>
  - grasp <object>
  - take <object>
  - 抓取 <object>
  - 抓起来 <object>
  - 拿起 <object>
allowed_tools:
  - observe
  - retrieve_asset_reference
  - sam3
  - select_sam3_detection
  - estimate_depth_prior
  - enhance_depth
  - grasp_pose_estimate
  - active_observe
  - reject_sam3_detections
  - activate_final_grasp_candidate
  - camera_pose_to_world
  - obstacle_avoidance
  - move_to
  - gripper_control
---
# Pick

This is decision guidance, not an executable macro. The planner chooses every
semantic, lifecycle, and motion tool from fresh feedback. The host owns exact
geometry, qualification evidence, candidate joins, and safety gates; a typed
host obligation constrains the next choice but does not execute the task for
the planner.

## Normal flow

1. Observe once and choose the complete RGB-D view that best shows the named
   target. Run `sam3` with one concise English target phrase, inspect its
   contact sheet, and resolve the exact detection through
   `select_sam3_detection`. Overlay colors are synthetic and do not establish
   object identity.
2. Reuse that selected RGB/depth/mask packet. Do not rerun SAM3 merely because
   one grasp later fails. In combined pick-place, first let the host create the
   private 96-goal AnyPlace pool for look-ahead; those object goals are not
   executable motions.
   If the selected target packet is too small, depth-sparse, border-clipped, or
   occluded, call `active_observe` once with its exact SAM3 result ID before
   grasp generation. A normal top view that already passes quality needs no
   active-vision call or wrist motion.
3. Call `grasp_pose_estimate` only with
   `targeted_grasp_obligation.required_parameters`. Backend choice and fallback
   are host-owned. A provider pose is the exact terminal EEF contact pose after
   only calibrated camera/world and TCP representation transforms.
4. The host must not center, translate, rotate, mirror, reverse, symmetrize, or
   otherwise create pose variants. It may hard-reject only malformed/non-finite
   transforms, illegal gripper width, mathematically certain workspace
   violations, or deterministic collision/scene illegality. Scores and
   capability maps may reorder candidates but never permanently delete a valid
   model pose.
5. Qualification runs the complete cheap structural/workspace breadth check
   once, then expands the frozen model pool in deterministic incremental
   `4 -> 8 -> 16 -> 32 -> 64 -> all` waves. Each wave takes its candidates
   straight through Beam-2, state validity, and L5 plan-only before the next
   wave begins. Stop as soon as two qualified grasps from different SE(3) and
   physical symmetry families are available. Beam-2 propagates at most two
   joint solutions for each exact contact target; the planner never copies or
   edits the pose.
6. Explicitly choose each constrained execution edge after inspecting the
   latest receipt: open only if needed, one `move_to` to the exact contact
   pose, then `gripper_control position=0`. There is no pregrasp, hover, precontact,
   approach offset, or fixed lift.
7. Portable attachment PASS requires native bilateral target contact plus the
   DetachableJoint attach ACK. Gripper acknowledgement, openness, a static
   image, or a lift displacement is not proof. Attached transport is rechecked
   for relative-pose drift on every MoveIt receipt.
8. If contact motion or close fails with a known candidate-linked outcome,
   consume only that candidate. Reopen once, use an already-qualified distinct
   backup first, then explicitly request the host's `frozen_frontier`
   obligation when it is offered. That request has `model_inference=false` and
   qualifies the next unvisited wave from the original provider output. Do not rerun SAM3 or grasp inference while that queue remains; do not rerun AnyPlace while either frozen queue remains. Only a
   genuinely changed scene or complete frozen-pool exhaustion permits fresh
   perception/model inference.
9. A timeout, service exception, missing calibration, OOM, or unavailable model
   is infrastructure failure, not candidate unreachability. Reconcile an
   unknown motion outcome on the same environment and retry infrastructure only
   according to its bounded host policy.
10. Articulated handles retain their separate frozen 5 cm attachment probe and
    assessor. Never apply that probe to portable pick-and-place objects.

Never move from stale scene/PlanningScene revisions, invent a waypoint, or
repeat an unchanged failed request fingerprint.
