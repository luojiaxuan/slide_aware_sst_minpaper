import copy
import json
from pathlib import Path

import pytest

from scripts.build_mcif_target_event_author_workspace import AUTHOR_SCHEMA, MAPPING_SCHEMA
from scripts.build_mcif_visual_token_controls import canonical_sha256, file_sha256
from scripts.mcif_target_event_annotation import (
    WORKING_SCHEMA,
    freeze_annotations,
    initialize_working_rows,
    validate_input_rows,
    validate_mapping_rows,
    validate_working_row,
    validate_working_rows,
    write_jsonl_atomic,
)


def input_row(index: int, *, options: int = 2) -> dict:
    row = {
        "schema_version": AUTHOR_SCHEMA,
        "status": "PENDING_HUMAN_EVENT_AUTHORING",
        "item_id": f"MCIF-ZH-A{index:04d}",
        "source_reference_en": f"English source {index}",
        "target_reference_zh": f"中文参考{index}",
        "current_slide": {
            "path": f"media/{index:02d}.png",
            "sha256": f"{index:064x}",
            "width": 24,
            "height": 12,
        },
        "current_slide_r0_text": f"slide text {index}",
        "current_slide_r1_text": f"structured slide text {index}",
        "candidate_options": [
            {
                "option_id": f"O{option:02d}",
                "source_candidate_en": f"candidate {index} {option}",
                "candidate_kind": "phrase",
                "token_count": 2,
                "lead_lower_bound_sec": 10.0 + option,
                "lead_bin": "10_to_lt30",
            }
            for option in range(1, options + 1)
        ],
        "annotation_status": "pending",
        "selected_option_id": None,
        "canonical_source_event_en": "",
        "acceptable_target_realizations_zh": [],
        "forbidden_target_realizations_zh": [],
        "target_reference_alignment": None,
        "slide_evidence_status": None,
        "annotation_note": "",
        "annotator_id": None,
        "locked_at_utc": None,
        "official_reference_consumed": True,
        "model_output_consumed": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def mapping_row(source: dict, index: int) -> dict:
    row = {
        "schema_version": MAPPING_SCHEMA,
        "item_id": source["item_id"],
        "talk_id": "talk-a",
        "segment_id": f"mcif:talk-a:SEG{index:03d}",
        "talk_segment_index": index,
        "source_segment_offset_sec": 10.0 + index,
        "source_segment_end_sec": 12.0 + index,
        "reference_segment_row_sha256": "1" * 64,
        "current_state_id": "mcif:talk-a:S000",
        "current_state_row_sha256": "2" * 64,
        "current_evidence_available_sec": 0.5,
        "option_mapping": [
            {
                "option_id": option["option_id"],
                "candidate_id": f"mcif:talk-a:C{index:03d}-{option['option_id']}",
                "candidate_row_sha256": f"{index + int(option['option_id'][1:]):064x}",
                "earliest_contiguous_state_id": "mcif:talk-a:S000",
                "earliest_contiguous_state_row_sha256": "2" * 64,
                "earliest_contiguous_evidence_sec": 0.5,
            }
            for option in source["candidate_options"]
        ],
        "author_row_sha256": source["row_sha256"],
        "official_reference_consumed": True,
        "model_output_consumed": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def labeled(row: dict, **updates) -> dict:
    result = copy.deepcopy(row)
    result.update(updates)
    result["row_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "row_sha256"}
    )
    return result


def fixture():
    source = [input_row(1), input_row(2, options=1)]
    working = initialize_working_rows(
        source,
        annotator_id="target-author-01",
        expected_items=2,
    )
    mappings = [mapping_row(row, index) for index, row in enumerate(source)]
    return source, working, mappings


def test_initialize_working_rows_preserves_only_bound_inputs_and_blank_labels():
    source, working, _ = fixture()
    assert len(working) == 2
    assert all(row["schema_version"] == WORKING_SCHEMA for row in working)
    assert all(row["annotator_id"] == "target-author-01" for row in working)
    assert all(row["annotation_status"] == "pending" for row in working)
    assert all(row["source_author_row_sha256"] == original["row_sha256"] for row, original in zip(working, source, strict=True))
    assert all("talk_id" not in row and "segment_id" not in row for row in working)
    assert all(row["row_sha256"] == canonical_sha256({key: value for key, value in row.items() if key != "row_sha256"}) for row in working)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hash", "row hash mismatch"),
        ("premature", "premature human labels"),
        ("duplicate", "duplicate item id"),
        ("option", "duplicate option ids"),
        ("boundary", "invalid data boundary"),
    ],
)
def test_input_validation_rejects_nonblank_or_drifted_workspace(mutation, message):
    rows = [input_row(1), input_row(2)]
    if mutation == "hash":
        rows[0]["source_reference_en"] = "changed"
    elif mutation == "premature":
        rows[0]["annotation_status"] = "eligible"
        rows[0]["row_sha256"] = canonical_sha256(
            {key: value for key, value in rows[0].items() if key != "row_sha256"}
        )
    elif mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "option":
        rows[0]["candidate_options"][1]["option_id"] = "O01"
        rows[0]["row_sha256"] = canonical_sha256(
            {key: value for key, value in rows[0].items() if key != "row_sha256"}
        )
    elif mutation == "boundary":
        rows[0]["model_output_consumed"] = True
        rows[0]["row_sha256"] = canonical_sha256(
            {key: value for key, value in rows[0].items() if key != "row_sha256"}
        )
    with pytest.raises(ValueError, match=message):
        validate_input_rows(rows, expected_items=2)


def test_eligible_row_requires_complete_scoring_definition():
    source, working, _ = fixture()
    valid = labeled(
        working[0],
        annotation_status="eligible",
        selected_option_id="O01",
        canonical_source_event_en="candidate one",
        acceptable_target_realizations_zh=["候选一", "第一个候选"],
        forbidden_target_realizations_zh=["候选二"],
        target_reference_alignment="explicit",
        slide_evidence_status="supported",
    )
    validate_working_row(
        valid,
        source[0],
        annotator_id="target-author-01",
        allow_pending=False,
    )

    mutations = [
        {"selected_option_id": "bad"},
        {"canonical_source_event_en": ""},
        {"acceptable_target_realizations_zh": []},
        {"target_reference_alignment": "omitted"},
        {"slide_evidence_status": "ambiguous"},
        {"forbidden_target_realizations_zh": ["候选一"]},
    ]
    for mutation in mutations:
        invalid = labeled(valid, **mutation)
        with pytest.raises(ValueError):
            validate_working_row(
                invalid,
                source[0],
                annotator_id="target-author-01",
                allow_pending=False,
            )


def test_noneligible_rows_must_clear_answers_but_keep_diagnostic_labels():
    source, working, _ = fixture()
    valid = labeled(
        working[0],
        annotation_status="no_target_alignment",
        target_reference_alignment="omitted",
        slide_evidence_status="supported",
        annotation_note="The Chinese reference omits this concept.",
    )
    validate_working_row(
        valid,
        source[0],
        annotator_id="target-author-01",
        allow_pending=False,
    )
    invalid = labeled(valid, selected_option_id="O01")
    with pytest.raises(ValueError, match="retains scoring answers"):
        validate_working_row(
            invalid,
            source[0],
            annotator_id="target-author-01",
            allow_pending=False,
        )


def test_pending_rows_cannot_contain_partial_labels():
    source, working, _ = fixture()
    validate_working_row(
        working[0],
        source[0],
        annotator_id="target-author-01",
        allow_pending=True,
    )
    partial = labeled(working[0], annotation_note="started")
    with pytest.raises(ValueError, match="partial labels"):
        validate_working_row(
            partial,
            source[0],
            annotator_id="target-author-01",
            allow_pending=True,
        )
    with pytest.raises(ValueError, match="remains pending"):
        validate_working_row(
            working[0],
            source[0],
            annotator_id="target-author-01",
            allow_pending=False,
        )


@pytest.mark.parametrize("mutation", ["immutable", "source_binding", "annotator", "hash"])
def test_working_validation_rejects_identity_drift(mutation):
    source, working, _ = fixture()
    row = copy.deepcopy(working[0])
    if mutation == "immutable":
        row["source_reference_en"] = "changed"
    elif mutation == "source_binding":
        row["source_author_row_sha256"] = "0" * 64
    elif mutation == "annotator":
        row["annotator_id"] = "another-author"
    elif mutation == "hash":
        row["annotation_note"] = "changed"
    if mutation != "hash":
        row["row_sha256"] = canonical_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
    with pytest.raises(ValueError):
        validate_working_row(
            row,
            source[0],
            annotator_id="target-author-01",
            allow_pending=True,
        )


@pytest.mark.parametrize("mutation", ["hash", "author", "options", "boundary"])
def test_mapping_validation_rejects_drift(mutation):
    source, _, mappings = fixture()
    rows = copy.deepcopy(mappings)
    if mutation == "hash":
        rows[0]["talk_id"] = "changed"
    elif mutation == "author":
        rows[0]["author_row_sha256"] = "0" * 64
    elif mutation == "options":
        rows[0]["option_mapping"].pop()
    elif mutation == "boundary":
        rows[0]["model_output_consumed"] = True
    if mutation != "hash":
        rows[0]["row_sha256"] = canonical_sha256(
            {key: value for key, value in rows[0].items() if key != "row_sha256"}
        )
    with pytest.raises(ValueError):
        validate_mapping_rows(rows, {row["item_id"]: row for row in source})


def completed_rows(working: list[dict]) -> list[dict]:
    return [
        labeled(
            working[0],
            annotation_status="eligible",
            selected_option_id="O02",
            canonical_source_event_en="candidate one two",
            acceptable_target_realizations_zh=["目标术语"],
            forbidden_target_realizations_zh=[],
            target_reference_alignment="paraphrased",
            slide_evidence_status="supported",
            annotation_note="",
        ),
        labeled(
            working[1],
            annotation_status="generic_or_unscorable",
            target_reference_alignment="uncertain",
            slide_evidence_status="supported",
            annotation_note="Generic phrase.",
        ),
    ]


def test_freeze_emits_only_authored_events_and_keeps_audio_fields_empty(tmp_path):
    source, working, mappings = fixture()
    working = completed_rows(working)
    output = tmp_path / "frozen"
    report = freeze_annotations(
        output,
        input_rows=source,
        working_rows=working,
        mapping_rows=mappings,
        input_sha256="1" * 64,
        working_sha256="2" * 64,
        mapping_sha256="3" * 64,
        annotator_id="target-author-01",
        locked_at_utc="2026-08-01T12:00:00Z",
        expected_items=2,
    )
    frozen = [json.loads(line) for line in (output / "frozen_author_annotations.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (output / "authored_target_events_private.jsonl").read_text().splitlines()]
    assert report["items"] == 2
    assert report["authored_events"] == 1
    assert report["status_distribution"] == {
        "eligible": 1,
        "generic_or_unscorable": 1,
    }
    assert len(frozen) == 2
    assert len(events) == 1
    assert events[0]["candidate_id"].endswith("O02")
    assert events[0]["audio_insufficient_until_sec"] is None
    assert events[0]["audio_first_sufficient_sec"] is None
    assert events[0]["primary_eligible"] is None
    assert events[0]["status"] == "TARGET_EVENT_AUTHORED_PENDING_AUDIO_SUFFICIENCY"
    assert all(row["model_output_consumed"] is False for row in frozen + events)
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert file_sha256(output / relative) == expected
    with pytest.raises(FileExistsError, match="must not already exist"):
        freeze_annotations(
            output,
            input_rows=source,
            working_rows=working,
            mapping_rows=mappings,
            input_sha256="1" * 64,
            working_sha256="2" * 64,
            mapping_sha256="3" * 64,
            annotator_id="target-author-01",
            locked_at_utc="2026-08-01T12:00:00Z",
            expected_items=2,
        )


def test_freeze_is_reproducible_and_rejects_pending_or_bad_timestamp(tmp_path):
    source, working, mappings = fixture()
    completed = completed_rows(working)
    kwargs = {
        "input_rows": source,
        "working_rows": completed,
        "mapping_rows": mappings,
        "input_sha256": "1" * 64,
        "working_sha256": "2" * 64,
        "mapping_sha256": "3" * 64,
        "annotator_id": "target-author-01",
        "locked_at_utc": "2026-08-01T12:00:00Z",
        "expected_items": 2,
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    freeze_annotations(first, **kwargs)
    freeze_annotations(second, **kwargs)
    assert {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="remains pending"):
        freeze_annotations(tmp_path / "pending", **{**kwargs, "working_rows": working})
    with pytest.raises(ValueError, match="timestamp"):
        freeze_annotations(tmp_path / "timestamp", **{**kwargs, "locked_at_utc": "bad"})


def test_atomic_working_writer_sets_private_permissions(tmp_path):
    source, working, _ = fixture()
    path = tmp_path / "working.jsonl"
    write_jsonl_atomic(path, working)
    assert path.stat().st_mode & 0o777 == 0o600
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    validate_working_rows(
        loaded,
        source,
        annotator_id="target-author-01",
        expected_items=2,
        allow_pending=True,
    )
