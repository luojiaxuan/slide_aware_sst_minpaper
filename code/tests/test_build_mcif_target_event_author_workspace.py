import copy
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.build_mcif_outcome_candidate_inventory import build_candidates
from scripts.build_mcif_target_event_author_workspace import (
    AUTHOR_SCHEMA,
    build_bundle,
    validate_inputs,
)
from scripts.build_mcif_visual_token_controls import canonical_sha256, file_sha256


def reference_row(
    talk_id: str,
    index: int,
    global_index: int,
    *,
    offset: float,
    source: str,
) -> dict:
    row = {
        "schema_version": "mcif_iwslt2026_reference_segment_v1",
        "global_segment_index": global_index,
        "talk_id": talk_id,
        "talk_segment_index": index,
        "speaker_id": f"speaker-{talk_id}",
        "offset_sec": offset,
        "duration_sec": 2.0,
        "end_sec": offset + 2.0,
        "segment_id": f"mcif:{talk_id}:SEG{index:03d}",
        "source_reference_en": source,
        "target_reference_zh": f"zh-{global_index}",
        "target_reference_de": f"de-{global_index}",
        "target_reference_it": f"it-{global_index}",
        "official_reference_consumed": True,
        "model_output_consumed": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def ladder_row(
    source_root: Path,
    talk_id: str,
    state_id: int,
    *,
    start: float,
    end: float,
    text: str,
    color: tuple[int, int, int],
) -> dict:
    relative = f"native_causal_v1/{talk_id}/state_{state_id:03d}.png"
    path = source_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 12), color=color).save(path)
    row = {
        "schema_version": "mcif_source_evidence_ladder_v1",
        "id": f"mcif:{talk_id}:S{state_id:03d}",
        "lecture_id": talk_id,
        "state_id": state_id,
        "availability_start_sec": start,
        "availability_end_sec": end,
        "r0_flat_ocr": {"model_input_text": text},
        "r1_structured_text": {"model_input_text": f"structured: {text}"},
        "r2_raw_image": {
            "source_media_path": relative,
            "source_media_sha256": file_sha256(path),
            "width": 24,
            "height": 12,
        },
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def fixture(tmp_path: Path):
    source_root = tmp_path / "source"
    references = [
        reference_row(
            "talk-a",
            0,
            0,
            offset=5.0,
            source="Neural machine translation arrives",
        ),
        reference_row(
            "talk-a",
            1,
            1,
            offset=7.0,
            source="Visual context",
        ),
        reference_row(
            "talk-b",
            0,
            2,
            offset=3.0,
            source="Language model",
        ),
    ]
    ladder = [
        ladder_row(
            source_root,
            "talk-a",
            0,
            start=0.0,
            end=10.0,
            text="Neural machine translation arrives Visual context",
            color=(10, 20, 30),
        ),
        ladder_row(
            source_root,
            "talk-b",
            0,
            start=0.0,
            end=10.0,
            text="Language model",
            color=(40, 50, 60),
        ),
    ]
    candidates = build_candidates(references, ladder, max_ngram=3)
    assert len(candidates) == 4
    kwargs = {
        "candidates": candidates,
        "references": references,
        "ladder": ladder,
        "source_root": source_root,
        "candidate_inventory_sha256": "1" * 64,
        "reference_segments_sha256": "2" * 64,
        "ladder_sha256": "3" * 64,
        "candidate_inventory_hf_revision": "4" * 40,
        "source_ladder_hf_revision": "5" * 40,
        "builder_git_commit": "6" * 40,
        "max_ngram": 3,
        "expected_candidates": 4,
        "expected_segments": 3,
        "expected_items": 3,
        "expected_talks": 2,
    }
    return source_root, references, ladder, candidates, kwargs


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_builds_exhaustive_segment_items_with_separate_scorer_mapping(tmp_path):
    _, _, _, _, kwargs = fixture(tmp_path)
    output = tmp_path / "workspace"

    report = build_bundle(output, **kwargs)

    author_rows = read_jsonl(output / "author_view" / "annotation_items.jsonl")
    mapping_rows = read_jsonl(output / "scorer_private" / "item_mapping.jsonl")
    assert report["items"] == 3
    assert report["candidate_options"] == 4
    assert report["unique_media_files"] == 2
    assert report["selection"] == (
        "exhaustive_one_author_item_per_candidate_bearing_segment"
    )
    assert len(author_rows) == len(mapping_rows) == 3
    assert {row["item_id"] for row in author_rows} == {
        row["item_id"] for row in mapping_rows
    }
    assert sorted(len(row["candidate_options"]) for row in author_rows) == [1, 1, 2]
    assert all(row["schema_version"] == AUTHOR_SCHEMA for row in author_rows)
    assert all(row["annotation_status"] == "pending" for row in author_rows)
    assert all(row["selected_option_id"] is None for row in author_rows)
    assert all(row["acceptable_target_realizations_zh"] == [] for row in author_rows)
    assert all(row["model_output_consumed"] is False for row in author_rows)
    assert all("talk_id" not in row and "segment_id" not in row for row in author_rows)
    assert all("target_reference_de" not in row for row in author_rows)
    assert all("target_reference_it" not in row for row in author_rows)
    assert all("talk_id" in row and "segment_id" in row for row in mapping_rows)
    assert all(
        row["row_sha256"]
        == canonical_sha256({key: value for key, value in row.items() if key != "row_sha256"})
        for row in author_rows + mapping_rows
    )


def test_media_are_hash_bound_deduplicated_and_author_checksums_pass(tmp_path):
    _, _, _, _, kwargs = fixture(tmp_path)
    output = tmp_path / "workspace"
    build_bundle(output, **kwargs)

    author_rows = read_jsonl(output / "author_view" / "annotation_items.jsonl")
    media_paths = {
        output / "author_view" / row["current_slide"]["path"] for row in author_rows
    }
    assert len(media_paths) == 2
    for row in author_rows:
        media = output / "author_view" / row["current_slide"]["path"]
        assert file_sha256(media) == row["current_slide"]["sha256"]
    for checksum in (
        output / "author_view" / "SHA256SUMS",
        output / "scorer_private" / "SHA256SUMS",
        output / "SHA256SUMS",
    ):
        root = checksum.parent
        for line in checksum.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            assert file_sha256(root / relative) == expected


def test_build_is_byte_reproducible_and_create_once(tmp_path):
    _, _, _, _, kwargs = fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_bundle(first, **kwargs)
    build_bundle(second, **kwargs)
    assert {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError, match="must not already exist"):
        build_bundle(first, **kwargs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("candidate_hash", "candidate row hash mismatch"),
        ("reference_content", "reference content differs"),
        ("first_occurrence", "not the first source occurrence"),
        ("state_hash", "evidence hash binding differs"),
        ("causality", "not causal at segment start"),
        ("premature_label", "premature human labels"),
        ("model_output", "invalid data boundary"),
    ],
)
def test_validation_fails_closed_on_contract_drift(tmp_path, mutation, message):
    _, references, ladder, candidates, _ = fixture(tmp_path)
    candidates = copy.deepcopy(candidates)
    references = copy.deepcopy(references)
    ladder = copy.deepcopy(ladder)
    row = candidates[0]
    if mutation == "candidate_hash":
        row["normalized_source_candidate"] = "changed"
    elif mutation == "reference_content":
        row["target_reference_zh"] = "changed"
        row["row_sha256"] = canonical_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
    elif mutation == "first_occurrence":
        row["normalized_source_candidate"] = "visual context"
        row["candidate_kind"] = "phrase"
        row["candidate_token_count"] = 2
        row["row_sha256"] = canonical_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
    elif mutation == "state_hash":
        row["current_state_row_sha256"] = "0" * 64
        row["row_sha256"] = canonical_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
    elif mutation == "causality":
        state = next(item for item in ladder if item["id"] == row["current_state_id"])
        state["availability_end_sec"] = 4.0
        state["row_sha256"] = canonical_sha256(
            {key: value for key, value in state.items() if key != "row_sha256"}
        )
        for candidate in candidates:
            if candidate["current_state_id"] == state["id"]:
                candidate["current_state_row_sha256"] = state["row_sha256"]
                if candidate["earliest_contiguous_state_id"] == state["id"]:
                    candidate["earliest_contiguous_state_row_sha256"] = state[
                        "row_sha256"
                    ]
                candidate["row_sha256"] = canonical_sha256(
                    {key: value for key, value in candidate.items() if key != "row_sha256"}
                )
    elif mutation == "premature_label":
        row["candidate_eligibility"] = True
        row["row_sha256"] = canonical_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
    elif mutation == "model_output":
        row["model_output_consumed"] = True
        row["row_sha256"] = canonical_sha256(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
    with pytest.raises(ValueError, match=message):
        validate_inputs(
            candidates,
            references,
            ladder,
            max_ngram=3,
            expected_candidates=4,
            expected_segments=3,
            expected_talks=2,
        )


def test_media_byte_drift_is_rejected_without_partial_output(tmp_path):
    source_root, _, _, _, kwargs = fixture(tmp_path)
    image = next(source_root.rglob("*.png"))
    Image.new("RGB", (24, 12), color=(200, 0, 0)).save(image)
    output = tmp_path / "workspace"
    with pytest.raises(ValueError, match="native image bytes changed"):
        build_bundle(output, **kwargs)
    assert not output.exists()


def test_item_count_contract_prevents_partial_author_view(tmp_path):
    _, _, _, _, kwargs = fixture(tmp_path)
    kwargs["expected_items"] = 4
    output = tmp_path / "workspace"
    with pytest.raises(ValueError, match="candidate-bearing segment count"):
        build_bundle(output, **kwargs)
    assert not output.exists()
