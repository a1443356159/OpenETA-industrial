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
  - close_simulator_env
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
3. For each pair, derive the exact release EEF pose as
   `object_goal * inverse(measured_attachment)`. Pair legality, Beam-2, state
   validity, and L5 plan-only must all pass. Stable wave barriers preserve
   deterministic order; use the first complete pair.
4. After native attach PASS, recompile the frozen AnyPlace pool from the
   measured attachment and current PlanningScene. Do not rerun AnyPlace or the
   grasp model because one pair fails; continue the frozen pair/grasp frontiers.
5. Execute one `move_to` to the exact release EEF pose with its full orientation.
   There is no carry lift, hover, descend offset, rim-clearance waypoint,
   near-target shortcut, or retreat. Every receipt must retain attachment and
   satisfy the native drift bound.
6. At exact release, call `gripper_control position=1` once. PASS requires
   detach ACK and native stable placement. A successful open, empty gripper,
   visual proximity, or reward alone is not proof.
7. On retained placement PASS, close the environment once. If planning rejects
   before execution, consume only that pair. If execution leaves a known safe
   state, follow the typed frozen-frontier recovery. Unknown transport state
   requires observation on the same handle. Infrastructure errors are never
   candidate failures.

Never edit model goals, add offsets, retry an unchanged failed fingerprint, or
claim success without native state validity, complete MoveIt plan proof, and
stable in-zone release evidence.
