from scripts.analyze_acl6060_ocr_anticipation import (
    analyze_talk,
    collapse_nested_candidates,
    normalized_tokens,
    observation_candidates,
    seed_coverage,
)


def observation(observation_id, start, end, text):
    words = text.split()
    return {
        "talk_id": "talk",
        "observation_id": observation_id,
        "observed_at_sec": start,
        "availability_end_sec": end,
        "frame_path": f"{observation_id}.jpg",
        "frame_sha256": observation_id,
        "tokens": [
            {"text": word, "confidence": 95.0, "line_key": "1"} for word in words
        ],
        "lines": [{"text": text, "mean_confidence": 95.0}],
    }


def segment(index, start, text):
    return {
        "talk_segment_index": index,
        "segment_id": f"talk:S{index:03d}",
        "offset_sec": start,
        "source_text": text,
    }


def test_normalization_and_observation_candidates():
    assert normalized_tokens("Model's task-oriented score") == (
        "model",
        "task-oriented",
        "score",
    )
    candidates = observation_candidates(observation("F0", 0, 10, "ACL semantic parsing"), 3)
    assert ("token", "acl") in candidates
    assert ("phrase", "semantic parsing") in candidates


def test_analyze_talk_requires_content_in_current_frame_and_tracks_contiguous_lead():
    observations = [
        observation("F0", 0, 5, "semantic parsing"),
        observation("F1", 5, 10, "semantic parsing"),
        observation("F2", 10, 20, "different content"),
    ]
    segments = [
        segment(0, 1, "An introduction."),
        segment(1, 8, "We now describe semantic parsing."),
        segment(2, 12, "Different content appears."),
    ]
    rows = analyze_talk("talk", observations, segments, max_ngram=3)
    phrase = next(row for row in rows if row["normalized_text"] == "semantic parsing")
    assert phrase["current_observation_id"] == "F1"
    assert phrase["earliest_contiguous_observation_id"] == "F0"
    assert phrase["lead_lower_bound_sec"] == 8.0
    assert not any(row["normalized_text"] == "introduction" for row in rows)


def test_seed_coverage_only_counts_future_content_in_same_observation():
    opportunities = [
        {
            "current_observation_id": "F1",
            "candidate_kind": "phrase",
            "normalized_text": "semantic parsing",
            "first_spoken_segment_start_sec": 8.0,
            "lead_lower_bound_sec": 8.0,
        }
    ]
    seeds = [
        {
            "packet_id": "A1",
            "talk_id": "talk",
            "observation_id": "F1",
            "selection_stratum": "random",
            "t_evidence_sec": 5.0,
            "suggested_audio_window_end_sec": 10.0,
        },
        {
            "packet_id": "A2",
            "talk_id": "talk",
            "observation_id": "F0",
            "selection_stratum": "random",
            "t_evidence_sec": 0.0,
            "suggested_audio_window_end_sec": 10.0,
        },
    ]
    rows = seed_coverage(seeds, opportunities)
    assert rows[0]["automatic_future_phrase_count"] == 1
    assert rows[1]["automatic_future_candidate_count"] == 0


def test_collapse_nested_candidates_keeps_maximal_phrase_and_uncovered_token():
    base = {
        "talk_id": "talk",
        "first_spoken_segment_id": "talk:S001",
        "current_observation_id": "F1",
        "first_spoken_segment_start_sec": 8.0,
    }
    rows = [
        {**base, "candidate_kind": "phrase", "normalized_text": "semantic parsing"},
        {**base, "candidate_kind": "phrase", "normalized_text": "semantic parsing model"},
        {**base, "candidate_kind": "token", "normalized_text": "semantic"},
        {**base, "candidate_kind": "token", "normalized_text": "latency"},
    ]
    collapsed = collapse_nested_candidates(rows)
    assert [row["normalized_text"] for row in collapsed] == [
        "semantic parsing model",
        "latency",
    ]
    assert {row["automatic_event_id"] for row in collapsed} == {"talk:E001"}
