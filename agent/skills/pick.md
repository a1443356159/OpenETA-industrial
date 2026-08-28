---
name: pick
description: Closed-loop guidance for selecting, qualifying, executing, and recovering model-generated grasps.
version: v2
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

This is compact decision guidance, not an executable macro. Choose each tool
from fresh feedback. The host owns geometry, candidate joins, qualification,
and safety gates; typed obligations constrain the next choice without making it.

## Normal flow

1. Observe once, choose the complete RGB-D view that best shows the target, and
   run `sam3` with one concise English target phrase. Inspect the contact sheet
   and select the exact detection; overlay colors do not prove identity.
2. Reuse the selected RGB/depth/mask packet. In combined pick-place, let the
   host create the private AnyPlace look-ahead pool before grasp generation.
   If the packet is genuinely small, sparse, clipped, or occluded, call
   `active_observe` once with its SAM3 result ID. A good top view needs no wrist
   motion.
3. Call `grasp_pose_estimate` only with the targeted obligation. Backend choice
   and fallback are host-owned. A provider pose is the exact terminal EEF contact pose
   after calibrated frame/TCP representation transforms only.
4. Never copy, edit, center, mirror, offset, or invent a pose. Cheap legality is
   evaluated once; the frozen provider pool then advances through deterministic
   small waves, Beam-2, state validity, and L5 plan-only. Scores only order the
   search. Use the first complete grasp/place proof and retain the untouched
   tail.
5. Inspect each receipt, open only if required, call one `move_to` to contact,
   then `gripper_control position=0`. There is no pregrasp, hover, precontact,
   approach offset, or fixed lift.
6. Portable attachment PASS requires native bilateral target contact and attach
   ACK. A successful command, static image, or lift is not proof; attached
   transport must retain the object within the native drift bound.
7. For a candidate-linked motion failure, follow the typed recovery obligation:
   restore the exact pre-attempt EEF anchor before another candidate. For a
   failed close, reopen only when requested. Then request `frozen_frontier` with
   `model_inference=false` and continue the next unvisited wave.
   Do not rerun SAM3 or grasp inference while that queue remains, and do not
   rerun AnyPlace while its frozen pool remains. Fresh inference requires a
   genuinely changed scene or exhausted frozen pools.
8. Timeout, service error, missing calibration, OOM, or unavailable deployment
   is infrastructure failure, not candidate unreachability. Reconcile unknown
   motion state on the same handle and follow the bounded host retry policy.
9. Only articulated handles use the separate frozen 5 cm attachment probe.

Never move from stale scene/PlanningScene revisions, invent a waypoint, or
repeat an unchanged failed request fingerprint.
