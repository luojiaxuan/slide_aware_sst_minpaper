from scripts import build_mcif_strict_beyond_ocr_researcher_prescreen as prescreen


def visual(candidate, ocr, fields):
    return {
        "candidate_source_en": candidate,
        "current_slide_r0_text": ocr,
        "proposed_evidence_origins": [
            {"descriptor_field": field, "descriptor_text": "arrow on the left"}
            for field in fields
        ],
    }


def mapping(candidate_id, talk, segment, lead=10.0):
    return {
        "candidate_id": candidate_id,
        "talk_id": talk,
        "segment_id": segment,
        "lead_lower_bound_sec": lead,
    }


def test_token_overlap_uses_tokens_not_substrings():
    result = prescreen.token_overlap("nouns", "Pronouns and verbs")
    assert result["coverage"] == 0
    assert result["matched_tokens"] == []


def test_priority_prefers_relation_and_zero_overlap():
    relation = prescreen.priority_score(
        visual("three bubbles", "dataset methodology", ["spatial_relations"]),
        mapping("one", "talk", "segment"),
    )
    text = prescreen.priority_score(
        visual("dataset methodology", "dataset methodology", ["scene_summary"]),
        mapping("two", "talk", "segment"),
    )
    assert relation > text


def test_priority_selection_caps_talk_and_segment():
    rows = [
        {
            "mapping": mapping("a", "talk-1", "segment-1"),
            "visual": visual("a", "", ["spatial_relations"]),
            "_priority_score": 5,
        },
        {
            "mapping": mapping("b", "talk-1", "segment-1"),
            "visual": visual("b", "", ["spatial_relations"]),
            "_priority_score": 4,
        },
        {
            "mapping": mapping("c", "talk-1", "segment-2"),
            "visual": visual("c", "", ["spatial_relations"]),
            "_priority_score": 3,
        },
        {
            "mapping": mapping("d", "talk-2", "segment-3"),
            "visual": visual("d", "", ["spatial_relations"]),
            "_priority_score": 2,
        },
    ]
    assert prescreen.choose_priority(rows, max_per_talk=1) == {"a", "d"}
