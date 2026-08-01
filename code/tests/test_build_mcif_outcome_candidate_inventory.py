import copy
import json
from pathlib import Path
import zipfile

import pytest
import yaml

from scripts.build_mcif_outcome_candidate_inventory import (
    REFERENCE_MEMBERS,
    SEGMENT_MEMBER,
    build_bundle,
    build_candidates,
    load_archive,
    parse_reference_lines,
    parse_segment_metadata,
    validate_source_inputs,
)
from scripts.build_mcif_visual_token_controls import canonical_sha256, file_sha256


def segment_source() -> list[dict]:
    return [
        {
            "wav": "talk-a.wav",
            "offset": 5.0,
            "duration": 2.0,
            "speaker_id": "speaker-a",
        },
        {
            "wav": "talk-a.wav",
            "offset": 12.0,
            "duration": 3.0,
            "speaker_id": "speaker-a",
        },
        {
            "wav": "talk-b.wav",
            "offset": 1.0,
            "duration": 2.5,
            "speaker_id": "speaker-b",
        },
    ]


def write_archive(path: Path) -> None:
    references = {
        "en": '"Hello ""world\ncontinued"\nPlain row\n',
        "zh": "你好世界\n继续\n普通行\n",
        "de": "Hallo Welt\nweiter\neinfache Zeile\n",
        "it": '"Ciao mondo\ncontinua"\n"riga semplice"\n',
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(SEGMENT_MEMBER, yaml.safe_dump(segment_source()))
        for language, member in REFERENCE_MEMBERS.items():
            archive.writestr(member, references[language])


def reference_segment(
    index: int,
    *,
    offset: float,
    duration: float,
    source: str,
) -> dict:
    row = {
        "schema_version": "mcif_iwslt2026_reference_segment_v1",
        "global_segment_index": index,
        "talk_id": "talk-a",
        "talk_segment_index": index,
        "speaker_id": "speaker-a",
        "offset_sec": offset,
        "duration_sec": duration,
        "end_sec": offset + duration,
        "segment_id": f"mcif:talk-a:SEG{index:03d}",
        "source_reference_en": source,
        "target_reference_zh": f"zh-{index}",
        "target_reference_de": f"de-{index}",
        "target_reference_it": f"it-{index}",
        "official_reference_consumed": True,
        "model_output_consumed": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def ladder_row(
    state_id: int,
    *,
    start: float,
    end: float,
    text: str,
) -> dict:
    row = {
        "schema_version": "mcif_source_evidence_ladder_v1",
        "id": f"mcif:talk-a:S{state_id:03d}",
        "lecture_id": "talk-a",
        "state_id": state_id,
        "availability_start_sec": start,
        "availability_end_sec": end,
        "r0_flat_ocr": {"model_input_text": text},
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def inference_row(segments: list[dict]) -> dict:
    return {
        "talk_id": "talk-a",
        "segments": [
            {
                "segment_id": row["talk_segment_index"],
                "offset_sec": row["offset_sec"],
                "duration_sec": row["duration_sec"],
            }
            for row in segments
        ],
    }


def test_load_archive_preserves_physical_rows_and_decodes_talk_wrappers(tmp_path):
    archive = tmp_path / "mcif.zip"
    write_archive(archive)

    rows, raw_members, report = load_archive(archive)

    assert len(rows) == 3
    assert rows[0]["source_reference_en"] == 'Hello "world'
    assert rows[1]["source_reference_en"] == "continued"
    assert rows[2]["source_reference_en"] == "Plain row"
    assert rows[2]["target_reference_it"] == "riga semplice"
    assert report["talks"] == 2
    assert report["reference_parsing"]["en"]["wrapped_talks"] == 1
    assert report["reference_parsing"]["it"]["wrapped_talks"] == 2
    assert set(raw_members) == {SEGMENT_MEMBER, *REFERENCE_MEMBERS.values()}
    assert all(row["official_reference_consumed"] is True for row in rows)
    assert all(row["model_output_consumed"] is False for row in rows)


def test_reference_parser_rejects_count_empty_and_unbalanced_rows():
    segments = parse_segment_metadata(yaml.safe_dump(segment_source()).encode())
    with pytest.raises(ValueError, match="count differs"):
        parse_reference_lines(b"one\ntwo\n", language="en", segments=segments)
    with pytest.raises(ValueError, match="empty reference"):
        parse_reference_lines(b"one\n\nthree\n", language="en", segments=segments)
    with pytest.raises(ValueError, match="unbalanced"):
        parse_reference_lines(b'"one\ntwo\nthree\n', language="en", segments=segments)


def test_segment_parser_rejects_noncontiguous_talks():
    rows = segment_source() + [
        {
            "wav": "talk-a.wav",
            "offset": 20.0,
            "duration": 1.0,
            "speaker_id": "speaker-a",
        }
    ]
    with pytest.raises(ValueError, match="not contiguous"):
        parse_segment_metadata(yaml.safe_dump(rows).encode())


def test_validate_source_inputs_locks_talks_timing_hashes_and_source_only_boundary():
    segments = [
        reference_segment(0, offset=5.0, duration=2.0, source="first source"),
        reference_segment(1, offset=12.0, duration=3.0, source="second source"),
    ]
    ladder = [
        ladder_row(0, start=0.0, end=10.0, text="first source"),
        ladder_row(1, start=10.0, end=20.0, text="second source"),
    ]
    inference = [inference_row(segments)]

    validate_source_inputs(
        segments,
        ladder,
        inference,
        expected_segments=2,
        expected_talks=1,
    )

    bad_timing = copy.deepcopy(inference)
    bad_timing[0]["segments"][1]["offset_sec"] = 13.0
    with pytest.raises(ValueError, match="timing differs"):
        validate_source_inputs(
            segments,
            ladder,
            bad_timing,
            expected_segments=2,
            expected_talks=1,
        )

    bad_ladder = copy.deepcopy(ladder)
    bad_ladder[0]["target_or_reference_consumed"] = True
    bad_ladder[0]["row_sha256"] = canonical_sha256(
        {key: value for key, value in bad_ladder[0].items() if key != "row_sha256"}
    )
    with pytest.raises(ValueError, match="consumed outcome data"):
        validate_source_inputs(
            segments,
            bad_ladder,
            inference,
            expected_segments=2,
            expected_talks=1,
        )


def test_validate_source_inputs_rejects_state_contract_drift():
    segments = [reference_segment(0, offset=5.0, duration=2.0, source="first source")]
    inference = [inference_row(segments)]
    invalid_cases = [
        [ladder_row(1, start=0.0, end=10.0, text="first source")],
        [ladder_row(0, start=2.0, end=2.0, text="first source")],
        [
            ladder_row(0, start=2.0, end=4.0, text="first source"),
            ladder_row(1, start=1.0, end=3.0, text="first source"),
        ],
    ]
    for ladder in invalid_cases:
        with pytest.raises(ValueError, match="state ids|interval|times"):
            validate_source_inputs(
                segments,
                ladder,
                inference,
                expected_segments=1,
                expected_talks=1,
            )


def test_candidates_are_first_occurrence_causal_maximal_and_not_gold():
    segments = [
        reference_segment(
            0,
            offset=5.0,
            duration=2.0,
            source="Neural machine translation arrives",
        ),
        reference_segment(
            1,
            offset=12.0,
            duration=3.0,
            source="Neural machine translation and visual context",
        ),
        reference_segment(
            2,
            offset=25.0,
            duration=2.0,
            source="Outside interval evidence",
        ),
    ]
    ladder = [
        ladder_row(
            0,
            start=0.0,
            end=10.0,
            text="Neural machine translation arrives",
        ),
        ladder_row(
            1,
            start=10.0,
            end=20.0,
            text="Neural machine translation and visual context",
        ),
    ]

    candidates = build_candidates(segments, ladder, max_ngram=3)

    assert {
        row["normalized_source_candidate"] for row in candidates
    } == {
        "machine translation arrives",
        "neural machine translation",
        "and visual context",
        "machine translation and",
        "translation and visual",
    }
    first_segment = [
        row for row in candidates if row["segment_id"] == segments[0]["segment_id"]
    ]
    second_segment = [
        row for row in candidates if row["segment_id"] == segments[1]["segment_id"]
    ]
    assert len(first_segment) == 2
    assert all(
        row["earliest_contiguous_state_id"] == "mcif:talk-a:S000"
        and row["lead_lower_bound_sec"] == 5.0
        for row in first_segment
    )
    assert len(second_segment) == 3
    assert all(
        row["current_state_id"] == "mcif:talk-a:S001"
        and row["lead_lower_bound_sec"] == 2.0
        for row in second_segment
    )
    assert all(
        row["normalized_source_candidate"] != "neural machine translation"
        for row in second_segment
    )
    assert all("outside" not in row["normalized_source_candidate"] for row in candidates)
    for row in candidates:
        assert row["status"] == "AUTOMATIC_REFERENCE_AWARE_CANDIDATE_NOT_GOLD_EVENT"
        assert row["candidate_eligibility"] is None
        assert row["acceptable_target_realizations_it"] == []
        assert row["official_reference_consumed"] is True
        assert row["model_output_consumed"] is False
        assert row["row_sha256"] == canonical_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )


def test_candidate_continuity_stops_across_evidence_gap():
    segments = [
        reference_segment(0, offset=12.0, duration=2.0, source="persistent visual context")
    ]
    ladder = [
        ladder_row(0, start=0.0, end=8.0, text="persistent visual context"),
        ladder_row(1, start=10.0, end=20.0, text="persistent visual context"),
    ]
    candidates = build_candidates(segments, ladder, max_ngram=3)
    assert len(candidates) == 1
    assert candidates[0]["earliest_contiguous_state_id"] == "mcif:talk-a:S001"
    assert candidates[0]["lead_lower_bound_sec"] == 2.0


def test_bundle_is_create_once_checksummed_and_reproducible(tmp_path):
    segments = [reference_segment(0, offset=5.0, duration=2.0, source="visual context")]
    candidates = build_candidates(
        segments,
        [ladder_row(0, start=0.0, end=10.0, text="visual context")],
        max_ngram=2,
    )
    raw_members = {
        SEGMENT_MEMBER: b"segments",
        **{member: language.encode() for language, member in REFERENCE_MEMBERS.items()},
    }
    kwargs = {
        "segments": segments,
        "candidates": candidates,
        "raw_members": raw_members,
        "archive_report": {"segments": 1, "talks": 1},
        "archive_sha256": "1" * 64,
        "ladder_sha256": "2" * 64,
        "inference_manifest_sha256": "3" * 64,
        "builder_git_commit": "4" * 40,
        "max_ngram": 2,
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = build_bundle(first, **kwargs)
    build_bundle(second, **kwargs)

    assert report["status"] == "PRIVATE_REFERENCE_AWARE_CANDIDATES_PENDING_HUMAN_FREEZE"
    assert report["human_event_labels_complete"] is False
    assert report["audio_sufficiency_labels_complete"] is False
    assert {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    for line in (first / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert file_sha256(first / relative) == expected
    with pytest.raises(FileExistsError, match="must not already exist"):
        build_bundle(first, **kwargs)


def test_archive_rejects_missing_required_member(tmp_path):
    archive = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(SEGMENT_MEMBER, yaml.safe_dump(segment_source()))
    with pytest.raises(ValueError, match="lacks required"):
        load_archive(archive)
