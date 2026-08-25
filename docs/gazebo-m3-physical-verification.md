# Gazebo M3 native-contact verification

M3 verifies a grasp at the model-generated contact terminal. MoveIt plans the
entire route from the current robot state to that exact terminal; OpenETA does
not synthesize a pregrasp, hover, approach, lift, or retreat waypoint.

The fixture uses Gazebo Sim's stock `DetachableJoint` between
`gripper_mount_link` and `m3_target/target_link`, with
`/m3/detachable_joint/target/{attach,detach,state}`. Runtime starts from a
confirmed detached state, then executes only:

1. `move_to(exact model contact EEF pose)`;
2. `gripper_control(position=0)`;
3. `gripper_control(position=1)` for cleanup.

Before close, M3 arms the native left- and right-pad contact streams. The close
is a PASS only when both pads provide fresh, unambiguous target contact and the
detachable joint returns an attached ACK. The verifier records the measured
`T_eef_object_attached` transform and reports
`NATIVE_GRASP_ATTACHMENT_CONFIRMED`. An ACK without bilateral contact, stale or
mixed contact evidence, a different target, or an invalid transform fails
closed.

There is deliberately no displacement or minimum-lift threshold. Such a test
would measure an artificial post-grasp waypoint rather than the validity of the
model terminal. Opening must complete before detach and requires a detached ACK.
Plugin/transport failures are infrastructure errors, not candidate
unreachability.
