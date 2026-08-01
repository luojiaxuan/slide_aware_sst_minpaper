import copy
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.build_mcif_beyond_ocr_candidate_inventory import (
    build_candidates,
    validate_ladder_rows,
    validate_reference_rows,
    validate_vlm_rows,
)
from scripts.build_mcif_beyond_ocr_validation_workspace import (
    TARGET_SCHEMA,
    VISUAL_SCHEMA,
    build_bundle,
)
from scripts.build_mcif_visual_token_controls import canonical_sha256, file_sha256


def hashed(row):
    result = dict(row)
    result["row_sha256"] = canonical_sha256(result)
    return result


def reference_row():
    return hashed(
        {
            "schema_version": "mcif_iwslt2026_reference_segment_v1",
            "global_segment_index": 0,
            "talk_id": "talk",
            "talk_segment_index": 0,
            "speaker_id": "speaker",
            "offset_sec": 11.0,
            "duration_sec": 2.0,
            "end_sec": 13.0,
            "segment_id": "mcif:talk:SEG000",
            "source_reference_en": "We compare vector orchard and semantic bridge.",
            "target_reference_zh": "我们比较向量果园和语义桥。",
            "target_reference_de": "Wir vergleichen.",
            "target_reference_it": "Confrontiamo.",
            "official_reference_consumed": True,
            "model_output_consumed": False,
        }
    )


def ladder_row(source_root: Path, state_id: int):
    start = 0.5 + 10 * state_id
    relative = f"native_causal_v1/talk/state_{state_id:03d}.png"
    image_path = source_root / relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 18), color=(20 * state_id, 40, 80)).save(image_path)
    blocks = [
        {
            "bbox_norm": [0.0, 0.0, 1.0, 0.5],
            "block_id": 0,
            "content": "| concept | relation |\n|---|---|\n| vector | orchard |",
            "content_kind": "chart_markdown",
            "label": "chart",
            "provider_index": 0,
            "reading_order": 0,
        }
    ]
    return hashed(
        {
            "automatic_source_evidence_not_annotation": True,
            "availability_end_sec": start + 10.0,
            "availability_start_sec": start,
            "evidence_timestamp_sec": start,
            "id": f"mcif:talk:S{state_id:03d}",
            "lecture_id": "talk",
            "r0_flat_ocr": {
                "item_count": 1,
                "model_input_text": "flat baseline",
                "ordering": "provider",
                "source_items_sha256": "a" * 64,
            },
            "r1_structured_text": {
                "block_count": 1,
                "blocks": blocks,
                "model_input_text": "serialized wrapper must not drive candidates",
                "ordering": "reading_order",
                "source_blocks_sha256": "b" * 64,
            },
            "r2_raw_image": {
                "height": 18,
                "width": 32,
                "source_media_path": relative,
                "source_media_sha256": file_sha256(image_path),
            },
            "schema_version": "mcif_source_evidence_ladder_v1",
            "source_binding_sha256": "c" * 64,
            "source_transcript_consumed": False,
            "state_id": state_id,
            "target_or_reference_consumed": False,
        }
    )


def vlm_row(state_id: int):
    item_id = f"mcif:talk:S{state_id:03d}"
    visual_fields = {
        "ocr_text": ["ignored OCR"],
        "scene_summary": "A semantic bridge connects two nodes.",
        "objects": ["node diagram"],
        "actions": [],
        "spatial_relations": ["semantic bridge is above the nodes"],
    }
    return {
        "ambiguous_items": [],
        "annotation": {"verified": False},
        "background_docs": [],
        "evidence": [],
        "glossary": [],
        "hard_labels": [],
        "id": item_id,
        "lecture_id": "talk",
        "reference": {"alternatives": [], "status": "missing", "translation": None},
        "reference_translation": None,
        "slides": {"matched_slide_image": None, "matched_slide_text": None},
        "source_lang": "en",
        "source_transcript": "",
        "streaming_units": [],
        "target_lang": "zh",
        "video": {
            "end_sec": 10.0 + 10 * state_id,
            "frame_paths": [f"talks/talk/frames/{state_id}.jpg"],
            "start_sec": 10.0 * state_id,
        },
        "visual_context": {
            **visual_fields,
            "clip_id": item_id,
            "frame_ids": [f"frame-{state_id}"],
            "metadata": {
                "availability_start_sec": 10.0 * state_id,
                "availability_end_sec": 10.0 + 10 * state_id,
                "context_enrichment": {
                    "batch_size": 2,
                    "frame_path": f"talks/talk/frames/{state_id}.jpg",
                    "model_id": "model",
                    "model_revision": "revision",
                    "prompt_id": "prompt",
                    "prompt_sha256": "d" * 64,
                    "provider": "qwen_vl",
                    "raw_output": json.dumps(visual_fields),
                },
                "evidence_frame_sha256": "e" * 64,
                "screen_role": "private_source_only_prescreen_not_annotation",
                "state_id": state_id,
            },
            "video_id": "talk",
        },
    }


def fixture(tmp_path: Path):
    source_root = tmp_path / "source"
    references = [reference_row()]
    ladder = [ladder_row(source_root, 0), ladder_row(source_root, 1)]
    vlm_rows = [vlm_row(0), vlm_row(1)]
    references_by_talk = validate_reference_rows(
        references, expected_rows=1, expected_talks=1
    )
    ladder_by_id, ladder_by_talk = validate_ladder_rows(
        ladder, expected_rows=2, expected_talks=1
    )
    vlm_by_id = validate_vlm_rows(
        vlm_rows,
        ladder_by_id=ladder_by_id,
        expected_rows=2,
        expected_talks=1,
        expected_model_id="model",
        expected_model_revision="revision",
        allowed_prompts={"prompt": "d" * 64},
    )
    r1, r2 = build_candidates(
        references_by_talk,
        ladder_by_talk,
        vlm_by_id,
        max_ngram=2,
        vlm_output_sha256="1" * 64,
    )
    assert len(r1) == len(r2) == 1
    kwargs = {
        "references": references,
        "ladder": ladder,
        "vlm_rows": vlm_rows,
        "r1_candidates": r1,
        "r2_candidates": r2,
        "source_root": source_root,
        "reference_sha256": "2" * 64,
        "ladder_sha256": "3" * 64,
        "vlm_output_sha256": "1" * 64,
        "r1_candidates_sha256": "4" * 64,
        "r2_candidates_sha256": "5" * 64,
        "candidate_inventory_hf_revision": "6" * 40,
        "source_ladder_hf_revision": "7" * 40,
        "expected_reference_rows": 1,
        "expected_state_rows": 2,
        "expected_talks": 1,
        "expected_r1_candidates": 1,
        "expected_r2_candidates": 1,
        "expected_model_id": "model",
        "expected_model_revision": "revision",
        "allowed_prompts": {"prompt": "d" * 64},
        "max_ngram": 2,
        "builder_git_commit": "8" * 40,
    }
    return source_root, kwargs


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_builds_physically_separate_visual_and_target_views(tmp_path):
    _, kwargs = fixture(tmp_path)
    output = tmp_path / "workspace"
    report = build_bundle(output, **kwargs)
    visual = read_jsonl(output / "visual_validator_view" / "validation_items.jsonl")
    target = read_jsonl(output / "target_author_view" / "annotation_items.jsonl")
    mapping = read_jsonl(output / "scorer_private" / "item_mapping.jsonl")

    assert report["visual_validation_items"] == report["target_author_items"] == 2
    assert report["unique_media_files"] == 1
    assert len(visual) == len(target) == len(mapping) == 2
    assert all(row["schema_version"] == VISUAL_SCHEMA for row in visual)
    assert all(row["schema_version"] == TARGET_SCHEMA for row in target)
    assert all("source_reference_en" not in row for row in visual)
    assert all("target_reference_zh" not in row for row in visual)
    assert all("current_slide" not in row for row in target)
    assert all("current_slide_r0_text" not in row for row in target)
    assert all("evidence_channel" not in row for row in target)
    assert all("proposed_evidence_origins" not in row for row in target)
    assert all("talk_id" not in row and "segment_id" not in row for row in visual)
    assert all("talk_id" not in row and "segment_id" not in row for row in target)
    assert {row["item_id"] for row in visual}.isdisjoint(
        {row["item_id"] for row in target}
    )
    assert all(row["candidate_eligibility"] is None for row in target)
    assert all(row["visual_evidence_correct"] is None for row in visual)


def test_visual_view_exposes_model_description_only_for_r2(tmp_path):
    _, kwargs = fixture(tmp_path)
    output = tmp_path / "workspace"
    build_bundle(output, **kwargs)
    visual = read_jsonl(output / "visual_validator_view" / "validation_items.jsonl")
    by_channel = {row["evidence_channel"]: row for row in visual}
    r1 = by_channel["structure_preserving_text"]
    r2 = by_channel["raw_visual_semantics"]
    assert r1["generative_model_output_exposed"] is False
    assert r1["requires_r1_insufficiency_judgment"] is False
    assert r2["generative_model_output_exposed"] is True
    assert r2["requires_r1_insufficiency_judgment"] is True
    assert r2["proposed_evidence_origins"][0]["descriptor_field"] in {
        "scene_summary",
        "spatial_relations",
    }


def test_media_and_all_checksum_manifests_are_valid(tmp_path):
    _, kwargs = fixture(tmp_path)
    output = tmp_path / "workspace"
    build_bundle(output, **kwargs)
    visual = read_jsonl(output / "visual_validator_view" / "validation_items.jsonl")
    paths = {output / "visual_validator_view" / row["current_slide"]["path"] for row in visual}
    assert len(paths) == 1
    for row in visual:
        path = output / "visual_validator_view" / row["current_slide"]["path"]
        assert file_sha256(path) == row["current_slide"]["sha256"]
    for checksum in output.rglob("SHA256SUMS"):
        for line in checksum.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            assert file_sha256(checksum.parent / relative) == expected


def test_workspace_is_byte_reproducible_and_create_once(tmp_path):
    _, kwargs = fixture(tmp_path)
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


def test_candidate_drift_is_rejected_before_workspace_creation(tmp_path):
    _, kwargs = fixture(tmp_path)
    kwargs = copy.deepcopy(kwargs)
    kwargs["r2_candidates"][0]["normalized_source_candidate"] = "changed"
    output = tmp_path / "workspace"
    with pytest.raises(ValueError, match="deterministic replay"):
        build_bundle(output, **kwargs)
    assert not output.exists()


def test_vlm_provenance_drift_is_rejected_before_workspace_creation(tmp_path):
    _, kwargs = fixture(tmp_path)
    kwargs = copy.deepcopy(kwargs)
    kwargs["vlm_rows"][0]["visual_context"]["metadata"]["context_enrichment"][
        "model_revision"
    ] = "drift"
    output = tmp_path / "workspace"
    with pytest.raises(ValueError, match="model provenance"):
        build_bundle(output, **kwargs)
    assert not output.exists()


def test_media_byte_drift_is_rejected_without_partial_output(tmp_path):
    source_root, kwargs = fixture(tmp_path)
    image = next(source_root.rglob("state_001.png"))
    Image.new("RGB", (32, 18), color=(255, 0, 0)).save(image)
    output = tmp_path / "workspace"
    with pytest.raises(ValueError, match="native image bytes changed"):
        build_bundle(output, **kwargs)
    assert not output.exists()
