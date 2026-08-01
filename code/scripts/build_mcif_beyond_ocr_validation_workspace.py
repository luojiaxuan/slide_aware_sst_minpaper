#!/usr/bin/env python3
"""Build isolated MCIF beyond-OCR visual and target validation views."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from PIL import Image

from scripts.build_mcif_beyond_ocr_candidate_inventory import (
    build_candidates,
    parse_allowed_prompts,
    validate_ladder_rows,
    validate_reference_rows,
    validate_vlm_rows,
)
from scripts.build_mcif_visual_token_controls import (
    canonical_sha256,
    file_sha256,
    git_head_clean,
    load_jsonl,
    resolve_regular_file,
)


ORDERING_SEED = "mcif-beyond-ocr-validation-workspace-v1-20260801"
VISUAL_SCHEMA = "mcif_beyond_ocr_visual_validation_item_v1"
TARGET_SCHEMA = "mcif_beyond_ocr_target_author_item_v1"
MAPPING_SCHEMA = "mcif_beyond_ocr_validation_mapping_v1"


def deterministic_key(namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"{ORDERING_SEED}\0{namespace}\0{value}".encode()
    ).hexdigest()


def compare_replayed_candidates(
    supplied: list[dict[str, Any]],
    replayed: list[dict[str, Any]],
    *,
    tier: str,
    expected_count: int,
) -> None:
    if len(supplied) != expected_count or len(replayed) != expected_count:
        raise ValueError(f"MCIF {tier} candidate count differs from contract")
    supplied_by_id = {row.get("candidate_id"): row for row in supplied}
    replayed_by_id = {row.get("candidate_id"): row for row in replayed}
    if len(supplied_by_id) != len(supplied) or None in supplied_by_id:
        raise ValueError(f"MCIF {tier} candidates contain duplicate or missing ids")
    if supplied_by_id != replayed_by_id:
        raise ValueError(f"MCIF {tier} candidates differ from deterministic replay")


def validate_and_replay(
    *,
    references: list[dict[str, Any]],
    ladder: list[dict[str, Any]],
    vlm_rows: list[dict[str, Any]],
    r1_candidates: list[dict[str, Any]],
    r2_candidates: list[dict[str, Any]],
    expected_reference_rows: int,
    expected_state_rows: int,
    expected_talks: int,
    expected_r1_candidates: int,
    expected_r2_candidates: int,
    expected_model_id: str,
    expected_model_revision: str,
    allowed_prompts: dict[str, str],
    max_ngram: int,
    vlm_output_sha256: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    references_by_talk = validate_reference_rows(
        references,
        expected_rows=expected_reference_rows,
        expected_talks=expected_talks,
    )
    reference_by_id = {row["segment_id"]: row for row in references}
    ladder_by_id, ladder_by_talk = validate_ladder_rows(
        ladder,
        expected_rows=expected_state_rows,
        expected_talks=expected_talks,
    )
    if set(references_by_talk) != set(ladder_by_talk):
        raise ValueError("MCIF reference and ladder talk inventories differ")
    vlm_by_id = validate_vlm_rows(
        vlm_rows,
        ladder_by_id=ladder_by_id,
        expected_rows=expected_state_rows,
        expected_talks=expected_talks,
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
        allowed_prompts=allowed_prompts,
    )
    replayed_r1, replayed_r2 = build_candidates(
        references_by_talk,
        ladder_by_talk,
        vlm_by_id,
        max_ngram=max_ngram,
        vlm_output_sha256=vlm_output_sha256,
    )
    compare_replayed_candidates(
        r1_candidates,
        replayed_r1,
        tier="R1 strict",
        expected_count=expected_r1_candidates,
    )
    compare_replayed_candidates(
        r2_candidates,
        replayed_r2,
        tier="R2 semantic",
        expected_count=expected_r2_candidates,
    )
    return reference_by_id, ladder_by_id, [*r1_candidates, *r2_candidates]


def verify_and_copy_media(
    temporary: Path,
    *,
    source_root: Path,
    states: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    state_order = sorted(
        states,
        key=lambda row: deterministic_key("media", row["id"]),
    )
    output = {}
    for index, state in enumerate(state_order, start=1):
        media = state["r2_raw_image"]
        source = resolve_regular_file(source_root, media["source_media_path"])
        if file_sha256(source) != media["source_media_sha256"]:
            raise ValueError(f"MCIF native image bytes changed: {state['id']}")
        with Image.open(source) as image:
            image.verify()
        with Image.open(source) as image:
            size = image.size
        if size != (media["width"], media["height"]):
            raise ValueError(f"MCIF native image dimensions changed: {state['id']}")
        media_id = f"M{index:04d}"
        relative = Path("media") / f"{media_id}.png"
        target = temporary / "visual_validator_view" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_sha256(target) != media["source_media_sha256"]:
            raise ValueError(f"MCIF copied image bytes changed: {state['id']}")
        output[state["id"]] = {
            "media_id": media_id,
            "path": relative.as_posix(),
            "sha256": media["source_media_sha256"],
            "width": media["width"],
            "height": media["height"],
        }
    return output


def clean_r1_blocks(state: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for block in state["r1_structured_text"]["blocks"]:
        output.append(
            {
                "content_kind": block.get("content_kind"),
                "label": block.get("label"),
                "content": block.get("content"),
                "bbox_norm": block.get("bbox_norm"),
                "reading_order": block.get("reading_order"),
            }
        )
    return output


def build_view_rows(
    candidates: list[dict[str, Any]],
    *,
    reference_by_id: dict[str, dict[str, Any]],
    ladder_by_id: dict[str, dict[str, Any]],
    media_by_state: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    visual_order = sorted(
        candidates,
        key=lambda row: deterministic_key("visual", row["candidate_id"]),
    )
    target_order = sorted(
        candidates,
        key=lambda row: deterministic_key("target", row["candidate_id"]),
    )
    visual_id_by_candidate = {
        row["candidate_id"]: f"MCIF-BOV-V{index:04d}"
        for index, row in enumerate(visual_order, start=1)
    }
    target_id_by_candidate = {
        row["candidate_id"]: f"MCIF-BOV-T{index:04d}"
        for index, row in enumerate(target_order, start=1)
    }
    visual_rows = []
    for candidate in visual_order:
        state = ladder_by_id[candidate["current_state_id"]]
        tier = candidate["evidence_tier"]
        row = {
            "schema_version": VISUAL_SCHEMA,
            "status": "PENDING_INDEPENDENT_VISUAL_VALIDATION",
            "item_id": visual_id_by_candidate[candidate["candidate_id"]],
            "candidate_source_en": candidate["normalized_source_candidate"],
            "candidate_kind": candidate["candidate_kind"],
            "candidate_token_count": candidate["candidate_token_count"],
            "evidence_channel": (
                "structure_preserving_text"
                if tier == "r1_strict"
                else "raw_visual_semantics"
            ),
            "current_slide": media_by_state[state["id"]],
            "current_slide_r0_text": state["r0_flat_ocr"]["model_input_text"],
            "current_slide_r1_blocks": clean_r1_blocks(state),
            "proposed_evidence_origins": candidate["current_evidence_origins"],
            "requires_r1_insufficiency_judgment": tier == "r2_semantic",
            "annotation_status": "pending",
            "visual_evidence_correct": None,
            "candidate_supported_by_visual_evidence": None,
            "r0_insufficient": None,
            "r1_insufficient": None,
            "reason_codes": [],
            "annotation_note": "",
            "annotator_id": None,
            "locked_at_utc": None,
            "official_reference_consumed": True,
            "source_reference_exposed": False,
            "target_reference_exposed": False,
            "generative_model_output_exposed": tier == "r2_semantic",
        }
        row["row_sha256"] = canonical_sha256(row)
        visual_rows.append(row)
    target_rows = []
    for candidate in target_order:
        reference = reference_by_id[candidate["segment_id"]]
        row = {
            "schema_version": TARGET_SCHEMA,
            "status": "PENDING_INDEPENDENT_TARGET_EVENT_AUTHORING",
            "item_id": target_id_by_candidate[candidate["candidate_id"]],
            "candidate_source_en": candidate["normalized_source_candidate"],
            "candidate_kind": candidate["candidate_kind"],
            "candidate_token_count": candidate["candidate_token_count"],
            "source_reference_en": reference["source_reference_en"],
            "target_reference_zh": reference["target_reference_zh"],
            "annotation_status": "pending",
            "candidate_eligibility": None,
            "canonical_source_event_en": "",
            "acceptable_target_realizations_zh": [],
            "forbidden_target_realizations_zh": [],
            "target_reference_alignment": None,
            "reason_codes": [],
            "annotation_note": "",
            "annotator_id": None,
            "locked_at_utc": None,
            "official_reference_consumed": True,
            "slide_or_ocr_exposed": False,
            "visual_evidence_origin_exposed": False,
            "generative_model_output_exposed": False,
        }
        row["row_sha256"] = canonical_sha256(row)
        target_rows.append(row)
    visual_by_id = {row["item_id"]: row for row in visual_rows}
    target_by_id = {row["item_id"]: row for row in target_rows}
    mapping_rows = []
    for candidate in sorted(candidates, key=lambda row: row["candidate_id"]):
        visual_item_id = visual_id_by_candidate[candidate["candidate_id"]]
        target_item_id = target_id_by_candidate[candidate["candidate_id"]]
        reference = reference_by_id[candidate["segment_id"]]
        state = ladder_by_id[candidate["current_state_id"]]
        mapping = {
            "schema_version": MAPPING_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "candidate_row_sha256": candidate["row_sha256"],
            "evidence_tier": candidate["evidence_tier"],
            "talk_id": candidate["talk_id"],
            "segment_id": candidate["segment_id"],
            "talk_segment_index": candidate["talk_segment_index"],
            "source_segment_offset_sec": candidate["source_segment_offset_sec"],
            "source_segment_end_sec": candidate["source_segment_end_sec"],
            "reference_segment_row_sha256": reference["row_sha256"],
            "current_state_id": state["id"],
            "current_state_row_sha256": state["row_sha256"],
            "current_evidence_available_sec": candidate[
                "current_evidence_available_sec"
            ],
            "earliest_contiguous_state_id": candidate[
                "earliest_contiguous_state_id"
            ],
            "earliest_contiguous_state_row_sha256": candidate[
                "earliest_contiguous_state_row_sha256"
            ],
            "earliest_contiguous_evidence_sec": candidate[
                "earliest_contiguous_evidence_sec"
            ],
            "lead_lower_bound_sec": candidate["lead_lower_bound_sec"],
            "visual_item_id": visual_item_id,
            "visual_item_row_sha256": visual_by_id[visual_item_id]["row_sha256"],
            "target_item_id": target_item_id,
            "target_item_row_sha256": target_by_id[target_item_id]["row_sha256"],
            "media_sha256": media_by_state[state["id"]]["sha256"],
            "official_reference_consumed": True,
            "human_labels_complete": False,
        }
        mapping["row_sha256"] = canonical_sha256(mapping)
        mapping_rows.append(mapping)
    return visual_rows, target_rows, mapping_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def write_checksums(root: Path) -> tuple[int, str]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum = root / "SHA256SUMS"
    checksum.write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return len(paths), file_sha256(checksum)


def build_bundle(
    output_root: Path,
    *,
    references: list[dict[str, Any]],
    ladder: list[dict[str, Any]],
    vlm_rows: list[dict[str, Any]],
    r1_candidates: list[dict[str, Any]],
    r2_candidates: list[dict[str, Any]],
    source_root: Path,
    reference_sha256: str,
    ladder_sha256: str,
    vlm_output_sha256: str,
    r1_candidates_sha256: str,
    r2_candidates_sha256: str,
    candidate_inventory_hf_revision: str,
    source_ladder_hf_revision: str,
    expected_reference_rows: int,
    expected_state_rows: int,
    expected_talks: int,
    expected_r1_candidates: int,
    expected_r2_candidates: int,
    expected_model_id: str,
    expected_model_revision: str,
    allowed_prompts: dict[str, str],
    max_ngram: int,
    builder_git_commit: str,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF beyond-OCR workspace must not already exist")
    reference_by_id, ladder_by_id, candidates = validate_and_replay(
        references=references,
        ladder=ladder,
        vlm_rows=vlm_rows,
        r1_candidates=r1_candidates,
        r2_candidates=r2_candidates,
        expected_reference_rows=expected_reference_rows,
        expected_state_rows=expected_state_rows,
        expected_talks=expected_talks,
        expected_r1_candidates=expected_r1_candidates,
        expected_r2_candidates=expected_r2_candidates,
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
        allowed_prompts=allowed_prompts,
        max_ngram=max_ngram,
        vlm_output_sha256=vlm_output_sha256,
    )
    current_state_ids = sorted({row["current_state_id"] for row in candidates})
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        for directory in (
            "visual_validator_view",
            "target_author_view",
            "scorer_private",
        ):
            (temporary / directory).mkdir()
        media_by_state = verify_and_copy_media(
            temporary,
            source_root=source_root,
            states=[ladder_by_id[state_id] for state_id in current_state_ids],
        )
        visual_rows, target_rows, mapping_rows = build_view_rows(
            candidates,
            reference_by_id=reference_by_id,
            ladder_by_id=ladder_by_id,
            media_by_state=media_by_state,
        )
        visual_path = temporary / "visual_validator_view" / "validation_items.jsonl"
        target_path = temporary / "target_author_view" / "annotation_items.jsonl"
        mapping_path = temporary / "scorer_private" / "item_mapping.jsonl"
        write_jsonl(visual_path, visual_rows)
        write_jsonl(target_path, target_rows)
        write_jsonl(mapping_path, mapping_rows)
        (temporary / "visual_validator_view" / "README.md").write_text(
            "# MCIF Beyond-OCR Visual-validator View V1\n\n"
            "Judge whether the displayed slide supports the proposed candidate and "
            "evidence description, and whether R0/R1 are insufficient. This view has no "
            "source segment, target reference, talk/state id, timing, lead, or scorer mapping.\n",
            encoding="utf-8",
        )
        (temporary / "target_author_view" / "README.md").write_text(
            "# MCIF Beyond-OCR Target-author View V1\n\n"
            "Judge target-event eligibility and Chinese realizations from the candidate, "
            "English segment, and Chinese reference. This view has no slide, OCR, evidence "
            "tier, model description, timing, lead, or scorer mapping.\n",
            encoding="utf-8",
        )
        report = {
            "schema_version": "mcif_beyond_ocr_validation_workspace_report_v1",
            "status": "TWO_ROLE_VIEWS_READY_NO_HUMAN_LABELS",
            "builder_git_commit": builder_git_commit,
            "ordering_seed_sha256": hashlib.sha256(ORDERING_SEED.encode()).hexdigest(),
            "reference_segments_sha256": reference_sha256,
            "source_evidence_ladder_sha256": ladder_sha256,
            "source_only_vlm_output_sha256": vlm_output_sha256,
            "r1_strict_candidates_sha256": r1_candidates_sha256,
            "r2_semantic_candidates_sha256": r2_candidates_sha256,
            "candidate_inventory_hf_revision": candidate_inventory_hf_revision,
            "source_ladder_hf_revision": source_ladder_hf_revision,
            "candidates": len(candidates),
            "candidate_segments": len({row["segment_id"] for row in candidates}),
            "candidate_talks": len({row["talk_id"] for row in candidates}),
            "r1_strict_candidates": len(r1_candidates),
            "r2_semantic_candidates": len(r2_candidates),
            "visual_validation_items": len(visual_rows),
            "target_author_items": len(target_rows),
            "current_states": len(current_state_ids),
            "unique_media_files": len(media_by_state),
            "visual_labels_complete": False,
            "target_event_labels_complete": False,
            "audio_sufficiency_labels_complete": False,
            "visual_validator_source_or_target_references_exposed": False,
            "target_author_visual_or_model_evidence_exposed": False,
            "official_reference_consumed": True,
            "visual_items_sha256": file_sha256(visual_path),
            "target_items_sha256": file_sha256(target_path),
            "scorer_mapping_sha256": file_sha256(mapping_path),
            "interpretation": (
                "The two physically separate views support independent validation of "
                "visual/OCR claims and target-event realizations. They contain no human "
                "labels and do not establish that pixels outperform OCR."
            ),
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# MCIF Beyond-OCR Validation Workspace V1\n\n"
            "Private outcome-side workspace with physically separate visual-validator, "
            "target-author, and scorer-private subtrees. Never give one annotator another "
            "role's subtree or expose scorer-private data. No labels are complete.\n",
            encoding="utf-8",
        )
        visual_entries, visual_checksum = write_checksums(
            temporary / "visual_validator_view"
        )
        target_entries, target_checksum = write_checksums(
            temporary / "target_author_view"
        )
        scorer_entries, scorer_checksum = write_checksums(
            temporary / "scorer_private"
        )
        root_entries, root_checksum = write_checksums(temporary)
        os.rename(temporary, output_root)
        return {
            **report,
            "visual_checksum_entries": visual_entries,
            "visual_checksum_sha256": visual_checksum,
            "target_checksum_entries": target_entries,
            "target_checksum_sha256": target_checksum,
            "scorer_checksum_entries": scorer_entries,
            "scorer_checksum_sha256": scorer_checksum,
            "root_checksum_entries": root_entries,
            "root_checksum_sha256": root_checksum,
            "bundle_files": sum(path.is_file() for path in output_root.rglob("*")),
            "bundle_bytes": sum(
                path.stat().st_size for path in output_root.rglob("*") if path.is_file()
            ),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--expected-references-sha256", required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--expected-ladder-sha256", required=True)
    parser.add_argument("--vlm-output", type=Path, required=True)
    parser.add_argument("--expected-vlm-output-sha256", required=True)
    parser.add_argument("--r1-candidates", type=Path, required=True)
    parser.add_argument("--expected-r1-candidates-sha256", required=True)
    parser.add_argument("--r2-candidates", type=Path, required=True)
    parser.add_argument("--expected-r2-candidates-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-model-id", required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--allowed-prompt", action="append", default=[])
    parser.add_argument("--candidate-inventory-hf-revision", required=True)
    parser.add_argument("--source-ladder-hf-revision", required=True)
    parser.add_argument("--code-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--expected-reference-rows", type=int, default=919)
    parser.add_argument("--expected-state-rows", type=int, default=304)
    parser.add_argument("--expected-talks", type=int, default=21)
    parser.add_argument("--expected-r1-candidates", type=int, default=2)
    parser.add_argument("--expected-r2-candidates", type=int, default=150)
    args = parser.parse_args()
    for path, expected, label in (
        (args.references, args.expected_references_sha256, "references"),
        (args.ladder, args.expected_ladder_sha256, "ladder"),
        (args.vlm_output, args.expected_vlm_output_sha256, "VLM output"),
        (args.r1_candidates, args.expected_r1_candidates_sha256, "R1 candidates"),
        (args.r2_candidates, args.expected_r2_candidates_sha256, "R2 candidates"),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"MCIF {label} hash differs from the frozen input")
    if args.max_ngram < 1:
        raise ValueError("max-ngram must be positive")
    allowed_prompts = parse_allowed_prompts(args.allowed_prompt)
    builder_git_commit = git_head_clean(args.code_repo)
    report = build_bundle(
        args.output_root,
        references=load_jsonl(args.references),
        ladder=load_jsonl(args.ladder),
        vlm_rows=load_jsonl(args.vlm_output),
        r1_candidates=load_jsonl(args.r1_candidates),
        r2_candidates=load_jsonl(args.r2_candidates),
        source_root=args.source_root,
        reference_sha256=args.expected_references_sha256,
        ladder_sha256=args.expected_ladder_sha256,
        vlm_output_sha256=args.expected_vlm_output_sha256,
        r1_candidates_sha256=args.expected_r1_candidates_sha256,
        r2_candidates_sha256=args.expected_r2_candidates_sha256,
        candidate_inventory_hf_revision=args.candidate_inventory_hf_revision,
        source_ladder_hf_revision=args.source_ladder_hf_revision,
        expected_reference_rows=args.expected_reference_rows,
        expected_state_rows=args.expected_state_rows,
        expected_talks=args.expected_talks,
        expected_r1_candidates=args.expected_r1_candidates,
        expected_r2_candidates=args.expected_r2_candidates,
        expected_model_id=args.expected_model_id,
        expected_model_revision=args.expected_model_revision,
        allowed_prompts=allowed_prompts,
        max_ngram=args.max_ngram,
        builder_git_commit=builder_git_commit,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
