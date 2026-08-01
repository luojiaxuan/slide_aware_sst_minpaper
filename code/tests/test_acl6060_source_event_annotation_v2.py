import hashlib
import json
import os
from pathlib import Path
import wave

import pytest

os.environ.setdefault("ACL6060_V2_BLINDING_SECRET", "unit-test-secret-0123456789abcdef")

from scripts.acl6060_source_event_annotation_v2 import (
    audio_cohort_lock_sha256,
    apply_adjudications,
    build_agreement_report,
    canonical_hash,
    canonical_validator_answer,
    estimate_stratified_prevalence,
    expected_prefix_grid,
    frame_binding_hmac,
    first_stable_correct,
    freeze_audio_rows,
    freeze_adjudication_rows,
    freeze_author_audio_reviews,
    freeze_author_rows,
    freeze_frame_annotation,
    make_audio_validator_rows,
    make_frame_validator_rows,
    materialize_author_audio_review_view,
    materialize_audio_validator_view,
    materialize_frame_validator_view,
    prepare_author_rows,
    prepare_adjudication_rows,
    question_only_lock_payload,
)


CONFIG = json.loads(
    (Path(__file__).parents[1] / "configs" / "acl6060_source_event_annotation_v2.json").read_text()
)


def packet():
    return {
        "packet_id": "talk:A001",
        "talk_id": "talk",
        "selection_stratum": "hash_random",
        "t_evidence_sec": 5.0,
        "workspace_frame_path": "packets/talk__A001/frame.jpg",
        "workspace_frame_sha256": "framehash",
        "workspace_audio_path": "packets/talk__A001/audio.wav",
        "workspace_audio_sha256": "audiohash",
        "audio_id": "talk.wav",
        "audio_sha256": "fullaudiohash",
        "suggested_audio_window_end_sec": 65.0,
        "clip_start_sec": 0.0,
        "clip_end_sec": 65.0,
    }


def completed_author():
    row = prepare_author_rows([packet()], "author")[0]
    row.update(
        {
            "authoring_status": "candidate",
            "source_question": "Which method has lower latency?",
            "source_options": ["Method A", "Method B", "Neither"],
            "canonical_answer_index": 1,
            "evidence_subtypes": ["chart_relation"],
            "evidence_region": {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3},
            "term_or_entity": "Method B",
            "authoring_completed_at_utc": "2026-08-01T12:00:00Z",
            "question_locked_at_utc": "2026-08-01T12:00:00Z",
        }
    )
    return row


def completed_review():
    row = freeze_author_rows([completed_author()], [packet()], CONFIG)[0]
    row.update(
        {
            "audio_sha256": "b" * 64,
            "speech_relevance_status": "addressed",
            "full_audio_answer_index": row["canonical_answer_index"],
            "frame_current_for_question": True,
            "speech_reviewed_at_utc": "2026-08-01T12:10:00Z",
        }
    )
    return freeze_author_audio_reviews([row], [packet()], CONFIG)[0]


def complete_audio(row, stable_option_id):
    output = dict(row)
    output["audio_sha256"] = "a" * 64
    output.update(
        {
            "question_only_status": "not_answerable",
            "question_only_submitted_at_utc": "2026-08-01T12:20:00Z",
        }
    )
    output["question_only_lock_sha256"] = canonical_hash(question_only_lock_payload(output))
    grid = expected_prefix_grid(output)
    output["prefix_judgments"] = [
        {
            "step_index": index,
            "prefix_end_sec": time_sec,
            "status": "insufficient" if index < 2 else "option",
            "option_id": None if index < 2 else stable_option_id,
        }
        for index, time_sec in enumerate(grid)
    ]
    output.update(
        {
            "audio_annotation_status": "complete",
            "audio_submitted_at_utc": "2026-08-01T13:00:00Z",
            "interaction_log_tail_sha256": "c" * 64,
            "interaction_log_sha256": "d" * 64,
            "sequential_delivery_backend": "acl6060_audio_gate_v1",
        }
    )
    return output


def complete_frame(row, answer_option_id):
    output = dict(row)
    output.update(
        {
            "frame_support_status": "supported",
            "frame_answer_option_id": answer_option_id,
            "evidence_subtypes": ["chart_relation"],
            "frame_confidence": "high",
            "frame_submitted_at_utc": "2026-08-01T14:00:00Z",
        }
    )
    return freeze_frame_annotation(output, CONFIG)


def test_stage_views_enforce_blinding_and_question_lock():
    author = prepare_author_rows([packet()], "author")[0]
    assert "audio_path" not in author
    assert "talk_id" not in author
    assert "t_evidence_sec" not in author
    assert "frame_sha256" not in author
    review = completed_review()
    audio = make_audio_validator_rows([review], "validator_a", CONFIG)[0]
    assert "frame_path" not in audio
    assert "talk_id" not in audio
    assert "canonical_answer_index" not in audio
    assert "question_lock_sha256" in audio
    with pytest.raises(ValueError, match="authors and audio validators"):
        make_audio_validator_rows([review], review["author_id"], CONFIG)


def test_question_lock_detects_post_lock_change():
    review = completed_review()
    review["source_question"] = "Changed after lock?"
    with pytest.raises(ValueError, match="lock mismatch"):
        make_audio_validator_rows([review], "validator_a", CONFIG)


def test_audio_grid_validation_and_frame_release():
    review = completed_review()
    audio_a = make_audio_validator_rows([review], "audio_validator_a", CONFIG)[0]
    audio_b = make_audio_validator_rows([review], "audio_validator_b", CONFIG)[0]
    canonical_a = canonical_validator_answer(review, audio_a)
    canonical_b = canonical_validator_answer(review, audio_b)
    frozen_a = freeze_audio_rows(
        [complete_audio(audio_a, canonical_a)], [review], [packet()], CONFIG
    )
    frozen_b = freeze_audio_rows(
        [complete_audio(audio_b, canonical_b)], [review], [packet()], CONFIG
    )
    assert frozen_a[0]["audio_annotation_lock_sha256"]
    cohort_lock = audio_cohort_lock_sha256(frozen_a, frozen_b)
    frames = make_frame_validator_rows(
        [review], [packet()], "frame_validator_a", cohort_lock, CONFIG
    )
    assert "audio_path" not in frames[0]
    assert frames[0]["frame_path"].endswith("frame.jpg")
    bad = complete_audio(audio_a, canonical_a)
    bad["prefix_judgments"][2]["prefix_end_sec"] = 11.0
    with pytest.raises(ValueError, match="off grid"):
        freeze_audio_rows([bad], [review], [packet()], CONFIG)
    changed_contract = complete_audio(audio_a, canonical_a)
    changed_contract["prefix_step_sec"] = 1.0
    changed_contract["prefix_judgments"] = [
        {
            "step_index": index,
            "prefix_end_sec": time_sec,
            "status": "insufficient" if index < 2 else "option",
            "option_id": None if index < 2 else canonical_a,
        }
        for index, time_sec in enumerate(expected_prefix_grid(changed_contract))
    ]
    with pytest.raises(ValueError, match="Audio task field changed"):
        freeze_audio_rows(
            [changed_contract], [review], [packet()], CONFIG
        )


def test_report_has_defined_answer_and_boundary_agreement():
    review = completed_review()
    audio_a = make_audio_validator_rows([review], "audio_validator_a", CONFIG)[0]
    audio_b = make_audio_validator_rows([review], "audio_validator_b", CONFIG)[0]
    canonical_a = canonical_validator_answer(review, audio_a)
    canonical_b = canonical_validator_answer(review, audio_b)
    frozen_a = freeze_audio_rows(
        [complete_audio(audio_a, canonical_a)], [review], [packet()], CONFIG
    )
    frozen_b = freeze_audio_rows(
        [complete_audio(audio_b, canonical_b)], [review], [packet()], CONFIG
    )
    cohort_lock = audio_cohort_lock_sha256(frozen_a, frozen_b)
    frame_a = make_frame_validator_rows(
        [review], [packet()], "frame_validator_a", cohort_lock, CONFIG
    )
    frame_b = make_frame_validator_rows(
        [review], [packet()], "frame_validator_b", cohort_lock, CONFIG
    )
    frame_canonical_a = canonical_validator_answer(review, frame_a[0])
    frame_canonical_b = canonical_validator_answer(review, frame_b[0])
    rows, summary = build_agreement_report(
        [review],
        [packet()],
        frozen_a,
        frozen_b,
        [complete_frame(frame_a[0], frame_canonical_a)],
        [complete_frame(frame_b[0], frame_canonical_b)],
        CONFIG,
    )
    assert rows[0]["primary_eligible"] is True
    assert rows[0]["frame_answer_exact_agreement"] is True
    assert rows[0]["boundary_gap_steps"] == 0
    assert summary["primary_eligible_count"] == 1
    frozen_a[0]["prefix_judgments"][2]["option_id"] = next(
        option["option_id"]
        for option in frozen_a[0]["source_options"]
        if option["option_id"] != canonical_a
    )
    with pytest.raises(ValueError, match="Audio annotation lock mismatch"):
        build_agreement_report(
            [review],
            [packet()],
            frozen_a,
            frozen_b,
            [complete_frame(frame_a[0], frame_canonical_a)],
            [complete_frame(frame_b[0], frame_canonical_b)],
            CONFIG,
        )


def test_report_keeps_author_negatives_in_sampling_denominator():
    author = prepare_author_rows([packet()], "author")[0]
    author.update(
        {
            "authoring_status": "negative_no_visual_question",
            "negative_labels": ["visual_not_unique"],
            "authoring_note": "No unique visual answer.",
            "authoring_completed_at_utc": "2026-08-01T12:00:00Z",
        }
    )
    review = freeze_author_rows([author], [packet()], CONFIG)[0]
    review = freeze_author_audio_reviews([review], [packet()], CONFIG)[0]
    rows, summary = build_agreement_report(
        [review], [packet()], [], [], [], [], CONFIG
    )
    assert len(rows) == 1
    assert rows[0]["source_inventory_status"] == "negative_no_visual_question"
    assert summary["packet_count"] == 1
    assert summary["validator_packet_count"] == 0
    assert summary["author_negative_or_excluded_count"] == 1
    assert summary["stratified_yield"][0]["packet_count"] == 1

    excluded = prepare_author_rows([packet()], "author")[0]
    excluded.update(
        {
            "authoring_status": "exclude_other",
            "exclusion_labels": ["media_error"],
            "authoring_note": "Frame cannot be decoded.",
            "authoring_completed_at_utc": "2026-08-01T12:00:00Z",
        }
    )
    excluded_review = freeze_author_rows([excluded], [packet()], CONFIG)[0]
    excluded_review = freeze_author_audio_reviews(
        [excluded_review], [packet()], CONFIG
    )[0]
    excluded_rows, _ = build_agreement_report(
        [excluded_review], [packet()], [], [], [], [], CONFIG
    )
    assert excluded_rows[0]["source_excluded"] is True
    assert excluded_rows[0]["primary_eligible"] is None


def test_report_rejects_missing_packet_and_last_point_is_not_stable():
    review = completed_review()
    audio = make_audio_validator_rows([review], "audio_validator_a", CONFIG)[0]
    canonical = canonical_validator_answer(review, audio)
    completed = complete_audio(audio, canonical)
    for judgment in completed["prefix_judgments"]:
        judgment.update({"status": "insufficient", "option_id": None})
    completed["prefix_judgments"][-1].update(
        {"status": "option", "option_id": canonical}
    )
    assert first_stable_correct(completed, canonical, minimum_steps=2) is None

    second_packet = dict(packet())
    second_packet.update(
        {
            "packet_id": "talk:A002",
            "workspace_frame_path": "packets/talk__A002/frame.jpg",
        }
    )
    with pytest.raises(ValueError, match="Author-review and packet ids differ"):
        build_agreement_report(
            [review], [packet(), second_packet], [], [], [], [], CONFIG
        )


def test_stratified_prevalence_uses_frozen_pool_sizes():
    rows = [
        {"talk_id": "t1", "selection_stratum": "s", "primary_eligible": True},
        {"talk_id": "t1", "selection_stratum": "s", "primary_eligible": False},
        {"talk_id": "t2", "selection_stratum": "s", "primary_eligible": True},
        {"talk_id": "t2", "selection_stratum": "s", "primary_eligible": True},
    ]
    design = {
        "total_observation_count": 8,
        "strata": [
            {
                "talk_id": "t1",
                "selection_stratum": "s",
                "pool_count": 4,
                "selected_count": 2,
            },
            {
                "talk_id": "t2",
                "selection_stratum": "s",
                "pool_count": 4,
                "selected_count": 2,
            },
        ],
    }
    estimate = estimate_stratified_prevalence(rows, design)
    assert estimate["estimate"] == pytest.approx(0.75)
    assert estimate["standard_error"] == pytest.approx(2**0.5 / 8)
    rows[0]["primary_eligible"] = None
    assert estimate_stratified_prevalence(rows, design)["status"] == "UNRESOLVED_MISSING_OUTCOME"


def test_adjudication_is_locked_and_resolves_without_overwriting_raw_report():
    report = {
        "packet_id": "ACLDEV-opaque",
        "requires_adjudication": True,
        "primary_eligible": None,
        "adjudication_positive_allowed": True,
        "t_evidence_sec": 10.0,
        "causal_audio_end_sec": 20.0,
        "prefix_step_sec": 1.0,
        "question_author_id": "author",
        "audio_validator_ids": ["audio_a", "audio_b"],
        "frame_validator_ids": ["frame_a", "frame_b"],
    }
    row = prepare_adjudication_rows([report], "adjudicator")[0]
    row.update(
        {
            "adjudication_status": "resolved",
            "adjudicated_primary_eligible": True,
            "adjudicated_boundary_sec": 12.0,
            "reason_labels": ["boundary_disagreement"],
            "note": "Replayed the conflicting causal intervals.",
            "submitted_at_utc": "2026-08-01T15:00:00Z",
        }
    )
    frozen = freeze_adjudication_rows([row], [report], CONFIG)
    applied = apply_adjudications([report], frozen)
    assert report["primary_eligible"] is None
    assert applied[0]["primary_eligible"] is True
    assert applied[0]["adjudication_resolved"] is True
    frozen[0]["adjudicated_primary_eligible"] = False
    with pytest.raises(ValueError, match="Adjudication lock mismatch"):
        apply_adjudications([report], frozen)

    hard_failure = {**report, "adjudication_positive_allowed": False}
    blocked = prepare_adjudication_rows([hard_failure], "adjudicator")[0]
    blocked.update(
        {
            "adjudication_status": "resolved",
            "adjudicated_primary_eligible": True,
            "adjudicated_boundary_sec": 12.0,
            "reason_labels": ["answer_disagreement"],
            "note": "Attempted positive override.",
            "submitted_at_utc": "2026-08-01T15:00:00Z",
        }
    )
    with pytest.raises(ValueError, match="Hard failure"):
        freeze_adjudication_rows([blocked], [hard_failure], CONFIG)

    isolated_tail = prepare_adjudication_rows([report], "adjudicator")[0]
    isolated_tail.update(
        {
            "adjudication_status": "resolved",
            "adjudicated_primary_eligible": True,
            "adjudicated_boundary_sec": 20.0,
            "reason_labels": ["boundary_disagreement"],
            "note": "Only the final grid point was correct.",
            "submitted_at_utc": "2026-08-01T15:00:00Z",
        }
    )
    with pytest.raises(ValueError, match="lacks stable tail"):
        freeze_adjudication_rows([isolated_tail], [report], CONFIG)


def test_materialized_validator_views_are_physically_modality_separated(tmp_path):
    acl_root = tmp_path / "acl"
    full_audio = acl_root / "dev" / "full_wavs" / "talk.wav"
    full_audio.parent.mkdir(parents=True)
    with wave.open(str(full_audio), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(10)
        audio.writeframes(b"\x00\x00" * 100)
    local_packet = packet()
    local_packet.update(
        {
            "split": "dev",
            "audio_sha256": hashlib.sha256(full_audio.read_bytes()).hexdigest(),
            "suggested_audio_window_end_sec": 5.0,
        }
    )
    author = prepare_author_rows([local_packet], "author")[0]
    author.update(completed_author())
    review = freeze_author_rows([author], [local_packet], CONFIG)[0]
    review.update(
        {
            "audio_sha256": "b" * 64,
            "speech_relevance_status": "addressed",
            "full_audio_answer_index": review["canonical_answer_index"],
            "frame_current_for_question": True,
            "speech_reviewed_at_utc": "2026-08-01T12:10:00Z",
        }
    )
    review = freeze_author_audio_reviews([review], [local_packet], CONFIG)[0]
    audio_root = tmp_path / "audio_view"
    author_audio_root = tmp_path / "author_audio_view"
    unlocked_review = dict(review)
    unlocked_review["author_review_lock_sha256"] = None
    materialized_review = materialize_author_audio_review_view(
        [unlocked_review], [local_packet], acl_root, author_audio_root
    )
    assert list(author_audio_root.rglob("*.wav"))
    assert not list(author_audio_root.rglob("*.jpg"))
    assert materialized_review[0]["audio_sha256"]

    materialized_review = freeze_author_audio_reviews(
        materialized_review, [local_packet], CONFIG
    )
    audio_rows = make_audio_validator_rows(materialized_review, "validator_a", CONFIG)
    materialized_audio = materialize_audio_validator_view(
        audio_rows, [local_packet], acl_root, audio_root
    )
    assert list(audio_root.rglob("*.wav"))
    assert not list(audio_root.rglob("*.jpg"))
    assert materialized_audio[0]["audio_duration_sec"] == 5.0

    workspace_root = tmp_path / "workspace"
    frame_source = workspace_root / local_packet["workspace_frame_path"]
    frame_source.parent.mkdir(parents=True)
    frame_source.write_bytes(b"frame")
    frame_hash = hashlib.sha256(frame_source.read_bytes()).hexdigest()
    local_packet["workspace_frame_sha256"] = frame_hash
    frame_rows = [
        {
            "packet_id": review["packet_id"],
            "frame_path": f'packets/{review["packet_id"]}/frame.jpg',
            "frame_binding_hmac": frame_binding_hmac(review["packet_id"], frame_hash),
        }
    ]
    frame_root = tmp_path / "frame_view"
    materialize_frame_validator_view(frame_rows, [local_packet], workspace_root, frame_root)
    assert list(frame_root.rglob("*.jpg"))
    assert not list(frame_root.rglob("*.wav"))
