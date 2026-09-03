---
name: place
description: Closed-loop guidance for qualifying AnyPlace goals and releasing an attached object in a receptacle.
version: v2
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
  - move_to
  - gripper_control
---
# Place

This is compact decision guidance, not an executable macro. Choose every tool
from fresh feedback. AnyPlace predicts object goals; the host proves geometry
and MoveIt owns complete robot paths.

## Normal flow

1. In combined pick-place, select the target object and the flat support inside
   the receptacle once from calibrated RGB-D. Do not segment the whole bin, rim,
   or walls as its support. Call AnyPlace once and keep all 96 object goals
   private; they are not VLM waypoints.
2. The host checks each goal once for finite SE(3), valid rotation, legal
   support/height/footprint, static penetration, and certain workspace bounds.
   Only pairs entering the current small wave pay the deeper geometry cost.
3. For a flat support, derive the exact release EEF pose as
   `object_goal * inverse(measured_attachment)`. For a collision-backed
   container, AnyPlace selects the destination while the host derives a
   collision-backed gravity-drop terminal from exterior entry geometry. Its Z
   is host evidence, not an agent waypoint; gravity owns motion after detach.
   If a wave exhausts IK/L5, retry the configured height before expanding the
   frozen grasp frontier. Pair legality, Beam-2, state validity, and L5
   plan-only still gate every pair; use the first complete pair.
4. After native attach PASS, recompile the frozen AnyPlace pool from the
   measured attachment and current PlanningScene. Do not rerun AnyPlace or the
   grasp model because one pair fails; continue the frozen pair/grasp frontiers.
5. Execute one `move_to` to the exact host-qualified release EEF pose with its
   full compiled orientation.
   There is no carry lift, hover, descent waypoint, agent-authored offset,
   near-target shortcut, or retreat. Every receipt must retain attachment and
   satisfy the native drift bound.
6. At the release terminal, call `gripper_control position=1` once. Release
   completion requires native detach ACK and gripper-open ACK. Opening overlaps
   PlanningScene detach synchronization; repeated simulator pose polling and a
   fixed stability dwell are not part of this action.
7. Review the causal post-release RGB-D observation with the VLM. Continue the
   next assignment or finish the task when the part is visibly acceptable in
   its container; observe again if ambiguous. A pre-execution planning
   rejection consumes only that pair. Resume a known safe state from the frozen
   frontier; observe unknown transport state on the same handle. Infrastructure
   errors are never candidate failures.

Never hand-edit model goals, add agent-authored offsets, retry an unchanged failed fingerprint, or
claim success without native state validity, complete MoveIt plan proof, the
ordered native release receipt, and VLM review of a causal post-release image.
