from __future__ import annotations

from extensions.gazebo.native_grasp import (
    ChildLinkProof,
    NativeGraspVerifier,
    NativeContactSample,
    ReasonCode,
    Verdict,
    confirm_native_bilateral_contact,
)


def _sample(side: str, stamp: float) -> NativeContactSample:
    return NativeContactSample(
        side=side, timestamp_s=stamp, received_monotonic_s=20.0,
        collision_names=(f"rm75::robotiq_85_{side}_finger_tip_link", "target_object::target_link::target_collision"),
    )


def test_m3_verifier_requires_native_bilateral_target_contact_and_child_link_proof() -> None:
    gate = confirm_native_bilateral_contact(
        [*(_sample("left", stamp) for stamp in (10.01, 10.07, 10.12)), *(_sample("right", stamp) for stamp in (10.02, 10.08, 10.13))],
        close_completed_sim_time_s=10.0, now_monotonic_s=20.1,
    )
    assert gate.accepted and gate.reason_code is ReasonCode.CONTACT_TARGET_CONFIRMED
    verifier = NativeGraspVerifier()
    assert verifier.close_result(gate, attach_acked=True).reason_code is ReasonCode.ATTACH_ACKED_UNPROVEN
    record = verifier.prove_lift(ChildLinkProof(0.43, 0.51, 0.009))
    assert (record.verdict, record.reason_code, record.grasp_confirmed) == (Verdict.PASS, ReasonCode.TARGET_HELD, True)


def test_m3_verifier_rejects_unknown_mixed_or_distractor_native_contact() -> None:
    result = confirm_native_bilateral_contact(
        [
            NativeContactSample("left", 10.01, 20.0, ("left_tip", "distractor_object::distractor_link")),
            *(_sample("left", stamp) for stamp in (10.07, 10.12)),
            *(_sample("right", stamp) for stamp in (10.02, 10.08, 10.13)),
        ], close_completed_sim_time_s=10.0, now_monotonic_s=20.1,
    )
    assert not result.accepted and result.reason_code is ReasonCode.CONTACT_DISTRACTOR


def test_pregrasp_open_stays_ready_without_detach_ack() -> None:
    verifier = NativeGraspVerifier()

    record = verifier.pregrasp_open_result()

    assert record.verdict is Verdict.UNKNOWN
    assert record.reason_code is ReasonCode.READY
    assert record.phase == "ready"
    assert verifier.attached is False


def test_m3_verifier_fails_closed_for_preclose_contact_missing_ack_and_bad_proof() -> None:
    preclose = confirm_native_bilateral_contact(
        [*(_sample("left", stamp) for stamp in (9.90, 10.01, 10.12)), *(_sample("right", stamp) for stamp in (10.02, 10.08, 10.13))],
        close_completed_sim_time_s=10.0,
        now_monotonic_s=20.1,
    )
    assert preclose.reason_code is ReasonCode.CONTACT_SAMPLE_BEFORE_CLOSE

    verifier = NativeGraspVerifier()
    assert verifier.close_result(preclose, attach_acked=False).reason_code is ReasonCode.CONTACT_SAMPLE_BEFORE_CLOSE
    assert verifier.prove_lift(ChildLinkProof(0.43, 0.60, 0.0)).reason_code is ReasonCode.ATTACH_ACK_MISSING

    gate = confirm_native_bilateral_contact(
        [*(_sample("left", stamp) for stamp in (10.01, 10.07, 10.12)), *(_sample("right", stamp) for stamp in (10.02, 10.08, 10.13))],
        close_completed_sim_time_s=10.0,
        now_monotonic_s=20.1,
    )
    verifier.close_result(gate, attach_acked=True)
    assert verifier.prove_lift(ChildLinkProof(0.43, 0.50, 0.0)).reason_code is ReasonCode.TARGET_NOT_LIFTED
    verifier.close_result(gate, attach_acked=True)
    assert verifier.prove_lift(ChildLinkProof(0.43, 0.52, 0.011)).reason_code is ReasonCode.RELATIVE_POSE_DRIFT


def test_transport_retention_does_not_reapply_lift_height_threshold() -> None:
    gate = confirm_native_bilateral_contact(
        [
            *(_sample("left", stamp) for stamp in (10.01, 10.07, 10.12)),
            *(_sample("right", stamp) for stamp in (10.02, 10.08, 10.13)),
        ],
        close_completed_sim_time_s=10.0,
        now_monotonic_s=20.1,
    )
    verifier = NativeGraspVerifier()
    verifier.close_result(gate, attach_acked=True)
    assert verifier.prove_lift(ChildLinkProof(0.43, 0.52, 0.002)).verdict is Verdict.PASS

    lowered = verifier.prove_retention(ChildLinkProof(0.43, 0.45, 0.003))

    assert lowered.verdict is Verdict.PASS
    assert lowered.evidence["minimum_lift_required"] is False
