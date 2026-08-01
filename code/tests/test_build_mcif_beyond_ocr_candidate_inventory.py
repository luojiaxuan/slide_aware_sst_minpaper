import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "build_mcif_beyond_ocr_candidate_inventory.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_mcif_beyond_ocr_candidate_inventory", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def hashed(module, row):
    result = dict(row)
    result["row_sha256"] = module.canonical_sha256(result)
    return result


def reference_row(module):
    return hashed(
        module,
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
            "source_reference_en": (
                "We compare vector orchard and semantic bridge while hiddenoracle "
                "serializationonly and baseline remain unavailable."
            ),
            "target_reference_zh": "我们比较向量果园和语义桥。",
            "target_reference_de": "Wir vergleichen.",
            "target_reference_it": "Confrontiamo.",
            "official_reference_consumed": True,
            "model_output_consumed": False,
        },
    )


def ladder_row(module, state_id):
    start = 0.5 + 10 * state_id
    blocks = [
        {
            "bbox_norm": [0.0, 0.0, 0.5, 0.5],
            "block_id": 0,
            "content": "| concept | relation |\n|---|---|\n| vector | orchard |",
            "content_kind": "chart_markdown",
            "label": "chart",
            "provider_index": 0,
            "reading_order": 0,
        },
        {
            "bbox_norm": [0.5, 0.0, 1.0, 0.5],
            "block_id": 1,
            "content": "visibleword",
            "content_kind": "text",
            "label": "text",
            "provider_index": 1,
            "reading_order": 1,
        },
        {
            "bbox_norm": [0.0, 0.5, 1.0, 1.0],
            "block_id": 2,
            "content": "<div style='layoutword'>markupword</div>",
            "content_kind": "table_html",
            "label": "table",
            "provider_index": 2,
            "reading_order": 2,
        },
    ]
    row = {
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
            "block_count": len(blocks),
            "blocks": blocks,
            "model_input_text": (
                'Structured blocks: {"content":"serializationonly",'
                '"content_kind":"chart_markdown","label":"chart"}'
            ),
            "ordering": "reading_order",
            "source_blocks_sha256": "b" * 64,
        },
        "r2_raw_image": {
            "height": 10,
            "width": 10,
            "source_media_path": f"frames/{state_id}.png",
            "source_media_sha256": "c" * 64,
        },
        "schema_version": "mcif_source_evidence_ladder_v1",
        "source_binding_sha256": "d" * 64,
        "source_transcript_consumed": False,
        "state_id": state_id,
        "target_or_reference_consumed": False,
    }
    return hashed(module, row)


def vlm_row(state_id, *, model_id="model", model_revision="revision"):
    item_id = f"mcif:talk:S{state_id:03d}"
    visual_fields = {
        "ocr_text": ["hiddenoracle"],
        "scene_summary": "A semantic bridge connects visibleword nodes.",
        "objects": ["visibleword diagram", "baseline indicator"],
        "actions": [],
        "spatial_relations": ["semantic bridge lies above the nodes"],
    }
    visual = {
        "actions": visual_fields["actions"],
        "clip_id": item_id,
        "frame_ids": [f"frame-{state_id}"],
        "metadata": {
            "availability_start_sec": 100.0 + state_id,
            "availability_end_sec": 101.0 + state_id,
            "context_enrichment": {
                "batch_size": 2,
                "frame_path": f"talks/talk/frames/{state_id}.jpg",
                "model_id": model_id,
                "model_revision": model_revision,
                "prompt_id": "prompt",
                "prompt_sha256": "e" * 64,
                "provider": "qwen_vl",
                "raw_output": json.dumps(visual_fields),
            },
            "evidence_frame_sha256": "f" * 64,
            "screen_role": "private_source_only_prescreen_not_annotation",
            "state_id": state_id,
        },
        "objects": visual_fields["objects"],
        "ocr_text": visual_fields["ocr_text"],
        "scene_summary": visual_fields["scene_summary"],
        "spatial_relations": visual_fields["spatial_relations"],
        "video_id": "talk",
    }
    return {
        "ambiguous_items": [],
        "annotation": {"annotator": None, "notes": None, "verified": False},
        "audio": None,
        "background_docs": [],
        "evidence": [],
        "glossary": [],
        "hard_labels": [],
        "id": item_id,
        "lecture_id": "talk",
        "reference": {
            "alternatives": [],
            "status": "missing",
            "translation": None,
        },
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
        "visual_context": visual,
    }


def validated_fixture(module):
    references = [reference_row(module)]
    ladder = [ladder_row(module, 0), ladder_row(module, 1)]
    vlm = [vlm_row(0), vlm_row(1)]
    references_by_talk = module.validate_reference_rows(
        references, expected_rows=1, expected_talks=1
    )
    ladder_by_id, ladder_by_talk = module.validate_ladder_rows(
        ladder, expected_rows=2, expected_talks=1
    )
    vlm_by_id = module.validate_vlm_rows(
        vlm,
        ladder_by_id=ladder_by_id,
        expected_rows=2,
        expected_talks=1,
        expected_model_id="model",
        expected_model_revision="revision",
        allowed_prompts={"prompt": "e" * 64},
    )
    return references, references_by_talk, ladder_by_talk, vlm_by_id


def test_builds_strict_r1_and_semantic_r2_candidates_with_ladder_timing():
    module = load_module()
    _, references_by_talk, ladder_by_talk, vlm_by_id = validated_fixture(module)
    r1, r2 = module.build_candidates(
        references_by_talk,
        ladder_by_talk,
        vlm_by_id,
        max_ngram=2,
        vlm_output_sha256="1" * 64,
    )

    assert {row["normalized_source_candidate"] for row in r1} == {"vector orchard"}
    assert {row["normalized_source_candidate"] for row in r2} == {"semantic bridge"}
    assert r1[0]["earliest_contiguous_evidence_sec"] == 0.5
    assert r2[0]["earliest_contiguous_evidence_sec"] == 0.5
    assert r1[0]["current_evidence_available_sec"] == 10.5
    assert r2[0]["current_evidence_available_sec"] == 10.5
    assert r2[0]["lead_lower_bound_sec"] == 10.5
    assert r2[0]["generative_model_output_consumed"] is True
    assert r1[0]["generative_model_output_consumed"] is False
    assert all(row["candidate_eligibility"] is None for row in [*r1, *r2])


def test_excludes_serialized_labels_markup_r1_text_and_vlm_ocr_text():
    module = load_module()
    _, references_by_talk, ladder_by_talk, vlm_by_id = validated_fixture(module)
    r1, r2 = module.build_candidates(
        references_by_talk,
        ladder_by_talk,
        vlm_by_id,
        max_ngram=2,
        vlm_output_sha256="1" * 64,
    )
    all_candidates = {
        row["normalized_source_candidate"] for row in [*r1, *r2]
    }
    assert "serializationonly" not in all_candidates
    assert "hiddenoracle" not in all_candidates
    assert "visibleword" not in all_candidates
    assert "baseline" not in all_candidates
    assert "layoutword" not in all_candidates
    assert "markupword" not in all_candidates
    assert r1[0]["current_evidence_origins"][0]["content_kind"] == "chart_markdown"
    assert all(
        origin["descriptor_field"] != "ocr_text"
        for origin in r2[0]["current_evidence_origins"]
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda row: row.__setitem__("source_transcript", "leak"), "source transcript"),
        (
            lambda row: row["annotation"].__setitem__("verified", True),
            "promoted to annotation",
        ),
        (
            lambda row: row["reference"].__setitem__("translation", "leak"),
            "reference outcomes",
        ),
        (
            lambda row: row["visual_context"]["metadata"]["context_enrichment"].__setitem__(
                "model_revision", "drift"
            ),
            "model provenance",
        ),
        (
            lambda row: row["visual_context"]["metadata"]["context_enrichment"].__setitem__(
                "prompt_sha256", "0" * 64
            ),
            "prompt provenance",
        ),
        (
            lambda row: row["visual_context"]["objects"].append("invented object"),
            "structured field differs",
        ),
    ],
)
def test_vlm_validation_fails_closed_on_provenance_or_outcome_leak(mutation, match):
    module = load_module()
    ladder = [ladder_row(module, 0)]
    ladder_by_id, _ = module.validate_ladder_rows(
        ladder, expected_rows=1, expected_talks=1
    )
    row = vlm_row(0)
    mutation(row)
    with pytest.raises(ValueError, match=match):
        module.validate_vlm_rows(
            [row],
            ladder_by_id=ladder_by_id,
            expected_rows=1,
            expected_talks=1,
            expected_model_id="model",
            expected_model_revision="revision",
            allowed_prompts={"prompt": "e" * 64},
        )


def test_bundle_is_create_once_and_reproducible(tmp_path):
    module = load_module()
    references, references_by_talk, ladder_by_talk, vlm_by_id = validated_fixture(
        module
    )
    r1, r2 = module.build_candidates(
        references_by_talk,
        ladder_by_talk,
        vlm_by_id,
        max_ngram=2,
        vlm_output_sha256="1" * 64,
    )
    segments = module.build_candidate_segments([*r1, *r2], references)
    kwargs = {
        "references": references,
        "r1_candidates": r1,
        "r2_candidates": r2,
        "candidate_segments": segments,
        "reference_sha256": "2" * 64,
        "ladder_sha256": "3" * 64,
        "vlm_output_sha256": "1" * 64,
        "model_id": "model",
        "model_revision": "revision",
        "allowed_prompts": {"prompt": "e" * 64},
        "builder_git_commit": "4" * 40,
        "max_ngram": 2,
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = module.build_bundle(first, **kwargs)
    module.build_bundle(second, **kwargs)
    assert report["r1_strict"]["candidates"] == 1
    assert report["r2_semantic"]["candidates"] == 1
    assert (first / "SHA256SUMS").read_bytes() == (second / "SHA256SUMS").read_bytes()
    for relative in (
        "r1_strict_candidates.jsonl",
        "r2_semantic_candidates.jsonl",
        "candidate_segments.jsonl",
        "report.json",
        "README.md",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
    with pytest.raises(FileExistsError):
        module.build_bundle(first, **kwargs)


def test_allowed_prompt_parser_rejects_invalid_values():
    module = load_module()
    assert module.parse_allowed_prompts([f"prompt={'a' * 64}"]) == {
        "prompt": "a" * 64
    }
    with pytest.raises(ValueError):
        module.parse_allowed_prompts(["prompt=short"])
