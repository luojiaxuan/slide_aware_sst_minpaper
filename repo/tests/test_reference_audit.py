from slidesst.data.reference_audit import audit_reference_item, summarize_reference_audits
from slidesst.data.schema import ChallengeItem, ReferenceInfo, VisualContext


def test_reference_audit_rejects_visual_placeholder_and_cjk_leakage():
    item = ChallengeItem(
        id="clip_1",
        lecture_id="lecture",
        source_transcript="这会让个人对自己的身体形象有更积极的认识。",
        reference=ReferenceInfo(
            translation="This improves self-perception. Say something. 长期坚持",
            status="llm_generated",
        ),
        visual_context=VisualContext(ocr_text=["说点什么"]),
    )

    audit = audit_reference_item(item)

    assert audit.severity == "reject"
    assert "visual_placeholder=say something" in audit.flags
    assert any(flag.startswith("target_cjk_chars=") for flag in audit.flags)


def test_reference_audit_summary_counts_flag_families():
    good = ChallengeItem(
        id="good",
        lecture_id="lecture",
        source_transcript="我们讨论运动对健康的影响。",
        reference=ReferenceInfo(translation="We discuss the health effects of exercise."),
    )
    bad = ChallengeItem(
        id="bad",
        lecture_id="lecture",
        source_transcript="我们讨论运动。",
        reference=ReferenceInfo(translation="The slide says exercise improves health."),
    )

    summary = summarize_reference_audits([audit_reference_item(good), audit_reference_item(bad)])

    assert summary["n"] == 2
    assert summary["severity_counts"]["pass"] == 1
    assert summary["severity_counts"]["review"] == 1
    assert summary["flag_counts"]["evidence_source_mention"] == 1
