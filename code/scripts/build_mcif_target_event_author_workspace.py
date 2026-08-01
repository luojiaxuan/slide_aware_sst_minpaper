#!/usr/bin/env python3
"""Build an exhaustive En-to-Zh MCIF target-event authoring workspace."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from PIL import Image

from scripts.analyze_acl6060_ocr_anticipation import normalized_tokens, source_candidates
from scripts.build_mcif_visual_token_controls import (
    canonical_sha256,
    file_sha256,
    git_head_clean,
    load_jsonl,
    resolve_regular_file,
)


ORDERING_SEED = "mcif-target-event-author-v1-20260801"
AUTHOR_SCHEMA = "mcif_target_event_author_item_v1"
MAPPING_SCHEMA = "mcif_target_event_author_mapping_v1"


def row_hash_valid(row: dict[str, Any]) -> bool:
    return row.get("row_sha256") == canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )


def unique_rows(
    rows: list[dict[str, Any]],
    *,
    key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} contains an invalid {key}")
        if value in output:
            raise ValueError(f"{label} contains duplicate {key}: {value}")
        output[value] = row
    return output


def deterministic_key(value: str) -> str:
    return hashlib.sha256(f"{ORDERING_SEED}\0{value}".encode()).hexdigest()


def lead_bin(value: float) -> str:
    if value < 5:
        return "lt5"
    if value < 10:
        return "5_to_lt10"
    if value < 30:
        return "10_to_lt30"
    return "ge30"


def validate_inputs(
    candidates: list[dict[str, Any]],
    references: list[dict[str, Any]],
    ladder: list[dict[str, Any]],
    *,
    max_ngram: int,
    expected_candidates: int,
    expected_segments: int,
    expected_talks: int,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if len(candidates) != expected_candidates:
        raise ValueError("MCIF candidate count differs from contract")
    if len(references) != expected_segments:
        raise ValueError("MCIF reference segment count differs from contract")
    candidate_by_id = unique_rows(
        candidates,
        key="candidate_id",
        label="MCIF target-event candidates",
    )
    reference_by_id = unique_rows(
        references,
        key="segment_id",
        label="MCIF reference segments",
    )
    ladder_by_id = unique_rows(ladder, key="id", label="MCIF evidence ladder")
    talks = {row.get("talk_id") for row in references}
    if len(talks) != expected_talks or None in talks:
        raise ValueError("MCIF reference talk count differs from contract")
    if {row.get("talk_id") for row in candidates} != talks:
        raise ValueError("MCIF candidate and reference talk sets differ")
    if {row.get("lecture_id") for row in ladder} != talks:
        raise ValueError("MCIF evidence and reference talk sets differ")

    references_by_talk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in references:
        if row.get("schema_version") != "mcif_iwslt2026_reference_segment_v1":
            raise ValueError("Unexpected MCIF reference segment schema")
        if not row_hash_valid(row):
            raise ValueError(f"MCIF reference row hash mismatch: {row.get('segment_id')}")
        if row.get("official_reference_consumed") is not True or row.get(
            "model_output_consumed"
        ) is not False:
            raise ValueError("MCIF reference segment has an invalid data boundary")
        references_by_talk[row["talk_id"]].append(row)
    first_occurrence: dict[str, dict[tuple[str, str], str]] = {}
    for talk_id, rows in references_by_talk.items():
        rows.sort(key=lambda row: row["talk_segment_index"])
        if [row["talk_segment_index"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"MCIF reference segment ids are not contiguous for {talk_id}")
        first: dict[tuple[str, str], str] = {}
        for row in rows:
            for candidate in source_candidates(
                normalized_tokens(row["source_reference_en"]), max_ngram
            ):
                first.setdefault(candidate, row["segment_id"])
        first_occurrence[talk_id] = first

    for row in ladder:
        if row.get("schema_version") != "mcif_source_evidence_ladder_v1":
            raise ValueError("Unexpected MCIF source evidence schema")
        if not row_hash_valid(row):
            raise ValueError(f"MCIF evidence row hash mismatch: {row.get('id')}")
        if row.get("source_transcript_consumed") is not False or row.get(
            "target_or_reference_consumed"
        ) is not False:
            raise ValueError("MCIF evidence ladder consumed outcome data")

    for row in candidates:
        if row.get("schema_version") != "mcif_target_event_candidate_v1":
            raise ValueError("Unexpected MCIF target-event candidate schema")
        if row.get("status") != "AUTOMATIC_REFERENCE_AWARE_CANDIDATE_NOT_GOLD_EVENT":
            raise ValueError("MCIF automatic candidate was promoted before human freeze")
        if not row_hash_valid(row):
            raise ValueError(f"MCIF candidate row hash mismatch: {row.get('candidate_id')}")
        if row.get("official_reference_consumed") is not True or row.get(
            "model_output_consumed"
        ) is not False:
            raise ValueError("MCIF candidate has an invalid data boundary")
        reference = reference_by_id.get(row["segment_id"])
        if reference is None or reference["talk_id"] != row["talk_id"]:
            raise ValueError("MCIF candidate reference binding differs")
        expected_reference_fields = {
            "source_reference_en": reference["source_reference_en"],
            "target_reference_zh": reference["target_reference_zh"],
            "target_reference_de": reference["target_reference_de"],
            "target_reference_it": reference["target_reference_it"],
            "talk_segment_index": reference["talk_segment_index"],
            "source_segment_offset_sec": reference["offset_sec"],
            "source_segment_end_sec": reference["end_sec"],
        }
        if any(row.get(key) != value for key, value in expected_reference_fields.items()):
            raise ValueError("MCIF candidate reference content differs")
        identity = (row["candidate_kind"], row["normalized_source_candidate"])
        if first_occurrence[row["talk_id"]].get(identity) != row["segment_id"]:
            raise ValueError("MCIF candidate is not the first source occurrence")
        current = ladder_by_id.get(row["current_state_id"])
        earliest = ladder_by_id.get(row["earliest_contiguous_state_id"])
        if current is None or earliest is None:
            raise ValueError("MCIF candidate references an absent evidence state")
        if current["lecture_id"] != row["talk_id"] or earliest["lecture_id"] != row["talk_id"]:
            raise ValueError("MCIF candidate evidence comes from another talk")
        if (
            row["current_state_row_sha256"] != current["row_sha256"]
            or row["earliest_contiguous_state_row_sha256"] != earliest["row_sha256"]
        ):
            raise ValueError("MCIF candidate evidence hash binding differs")
        source_offset = float(row["source_segment_offset_sec"])
        if not (
            float(current["availability_start_sec"])
            <= source_offset
            < float(current["availability_end_sec"])
        ):
            raise ValueError("MCIF candidate current state is not causal at segment start")
        if row["current_evidence_available_sec"] != current["availability_start_sec"]:
            raise ValueError("MCIF candidate current evidence timing differs")
        if row["earliest_contiguous_evidence_sec"] != earliest["availability_start_sec"]:
            raise ValueError("MCIF candidate earliest evidence timing differs")
        expected_lead = round(
            source_offset - float(earliest["availability_start_sec"]), 6
        )
        if float(row["lead_lower_bound_sec"]) != expected_lead or expected_lead < 0:
            raise ValueError("MCIF candidate lead differs from causal evidence timing")
        if identity not in source_candidates(
            normalized_tokens(current["r0_flat_ocr"]["model_input_text"]), max_ngram
        ):
            raise ValueError("MCIF candidate is not visible in current R0 evidence")
        if any(
            row.get(key) not in (None, [], "")
            for key in (
                "candidate_eligibility",
                "acceptable_target_realizations_zh",
                "forbidden_target_realizations_zh",
                "acceptable_target_realizations_de",
                "forbidden_target_realizations_de",
                "acceptable_target_realizations_it",
                "forbidden_target_realizations_it",
                "audio_insufficient_until_sec",
                "audio_first_sufficient_sec",
                "annotator_id",
                "annotation_note",
            )
        ):
            raise ValueError("MCIF candidate contains premature human labels")
    return candidate_by_id, reference_by_id, ladder_by_id


def verify_and_copy_media(
    temporary: Path,
    *,
    source_root: Path,
    states: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for state in states:
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
        relative = Path("media") / media["source_media_sha256"][:2] / (
            media["source_media_sha256"] + ".png"
        )
        target = temporary / "author_view" / relative
        if target.exists():
            if file_sha256(target) != media["source_media_sha256"]:
                raise ValueError("MCIF deduplicated author media hash collision")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        output[state["id"]] = {
            "path": relative.as_posix(),
            "sha256": media["source_media_sha256"],
            "width": media["width"],
            "height": media["height"],
        }
    return output


def build_rows(
    candidates: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    ladder: dict[str, dict[str, Any]],
    media_by_state: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["segment_id"]].append(row)
    ordered_segments = sorted(grouped, key=deterministic_key)
    author_rows = []
    mapping_rows = []
    for item_index, segment_id in enumerate(ordered_segments, start=1):
        group = grouped[segment_id]
        current_state_ids = {row["current_state_id"] for row in group}
        if len(current_state_ids) != 1:
            raise ValueError("MCIF candidates in one segment bind different current states")
        current_state_id = next(iter(current_state_ids))
        state = ladder[current_state_id]
        reference = references[segment_id]
        item_id = f"MCIF-ZH-A{item_index:04d}"
        candidate_order = sorted(
            group,
            key=lambda row: deterministic_key(row["candidate_id"]),
        )
        options = []
        option_mapping = []
        for option_index, candidate in enumerate(candidate_order, start=1):
            option_id = f"O{option_index:02d}"
            options.append(
                {
                    "option_id": option_id,
                    "source_candidate_en": candidate["normalized_source_candidate"],
                    "candidate_kind": candidate["candidate_kind"],
                    "token_count": candidate["candidate_token_count"],
                    "lead_lower_bound_sec": candidate["lead_lower_bound_sec"],
                    "lead_bin": lead_bin(float(candidate["lead_lower_bound_sec"])),
                }
            )
            option_mapping.append(
                {
                    "option_id": option_id,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_row_sha256": candidate["row_sha256"],
                    "earliest_contiguous_state_id": candidate[
                        "earliest_contiguous_state_id"
                    ],
                    "earliest_contiguous_state_row_sha256": candidate[
                        "earliest_contiguous_state_row_sha256"
                    ],
                    "earliest_contiguous_evidence_sec": candidate[
                        "earliest_contiguous_evidence_sec"
                    ],
                }
            )
        author = {
            "schema_version": AUTHOR_SCHEMA,
            "status": "PENDING_HUMAN_EVENT_AUTHORING",
            "item_id": item_id,
            "source_reference_en": reference["source_reference_en"],
            "target_reference_zh": reference["target_reference_zh"],
            "current_slide": media_by_state[current_state_id],
            "current_slide_r0_text": state["r0_flat_ocr"]["model_input_text"],
            "current_slide_r1_text": state["r1_structured_text"]["model_input_text"],
            "candidate_options": options,
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
        author["row_sha256"] = canonical_sha256(author)
        mapping = {
            "schema_version": MAPPING_SCHEMA,
            "item_id": item_id,
            "talk_id": reference["talk_id"],
            "segment_id": segment_id,
            "talk_segment_index": reference["talk_segment_index"],
            "source_segment_offset_sec": reference["offset_sec"],
            "source_segment_end_sec": reference["end_sec"],
            "reference_segment_row_sha256": reference["row_sha256"],
            "current_state_id": current_state_id,
            "current_state_row_sha256": state["row_sha256"],
            "current_evidence_available_sec": state["availability_start_sec"],
            "option_mapping": option_mapping,
            "author_row_sha256": author["row_sha256"],
            "official_reference_consumed": True,
            "model_output_consumed": False,
        }
        mapping["row_sha256"] = canonical_sha256(mapping)
        author_rows.append(author)
        mapping_rows.append(mapping)
    return author_rows, mapping_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_checksums(root: Path) -> tuple[int, str]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_path = root / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
        ),
        encoding="utf-8",
    )
    return len(paths), file_sha256(checksum_path)


def build_bundle(
    output_root: Path,
    *,
    candidates: list[dict[str, Any]],
    references: list[dict[str, Any]],
    ladder: list[dict[str, Any]],
    source_root: Path,
    candidate_inventory_sha256: str,
    reference_segments_sha256: str,
    ladder_sha256: str,
    candidate_inventory_hf_revision: str,
    source_ladder_hf_revision: str,
    builder_git_commit: str,
    max_ngram: int,
    expected_candidates: int,
    expected_segments: int,
    expected_items: int,
    expected_talks: int,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF target-event author workspace must not already exist")
    _, reference_by_id, ladder_by_id = validate_inputs(
        candidates,
        references,
        ladder,
        max_ngram=max_ngram,
        expected_candidates=expected_candidates,
        expected_segments=expected_segments,
        expected_talks=expected_talks,
    )
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["segment_id"]].append(row)
    if len(grouped) != expected_items:
        raise ValueError("MCIF candidate-bearing segment count differs from contract")
    current_states = sorted(
        {row["current_state_id"] for row in candidates},
        key=lambda state_id: (
            ladder_by_id[state_id]["lecture_id"],
            ladder_by_id[state_id]["state_id"],
        ),
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        (temporary / "author_view").mkdir()
        (temporary / "scorer_private").mkdir()
        media_by_state = verify_and_copy_media(
            temporary,
            source_root=source_root,
            states=[ladder_by_id[state_id] for state_id in current_states],
        )
        author_rows, mapping_rows = build_rows(
            candidates,
            reference_by_id,
            ladder_by_id,
            media_by_state,
        )
        author_path = temporary / "author_view" / "annotation_items.jsonl"
        mapping_path = temporary / "scorer_private" / "item_mapping.jsonl"
        write_jsonl(author_path, author_rows)
        write_jsonl(mapping_path, mapping_rows)
        talk_item_counts = Counter(row["talk_id"] for row in mapping_rows)
        option_counts = [len(row["candidate_options"]) for row in author_rows]
        lead_bins = Counter(
            option["lead_bin"]
            for row in author_rows
            for option in row["candidate_options"]
        )
        report = {
            "schema_version": "mcif_target_event_author_workspace_report_v1",
            "status": "AUTHOR_VIEW_READY_NO_HUMAN_LABELS",
            "target_language": "zh",
            "selection": "exhaustive_one_author_item_per_candidate_bearing_segment",
            "maximum_final_events_per_segment": 1,
            "ordering_seed_sha256": hashlib.sha256(ORDERING_SEED.encode()).hexdigest(),
            "builder_git_commit": builder_git_commit,
            "candidate_inventory_sha256": candidate_inventory_sha256,
            "reference_segments_sha256": reference_segments_sha256,
            "ladder_sha256": ladder_sha256,
            "candidate_inventory_hf_revision": candidate_inventory_hf_revision,
            "source_ladder_hf_revision": source_ladder_hf_revision,
            "items": len(author_rows),
            "talks": len(talk_item_counts),
            "candidate_options": len(candidates),
            "current_states": len(current_states),
            "unique_media_files": len({row["current_slide"]["sha256"] for row in author_rows}),
            "talk_item_count_min": min(talk_item_counts.values()),
            "talk_item_count_max": max(talk_item_counts.values()),
            "options_per_item_min": min(option_counts),
            "options_per_item_max": max(option_counts),
            "lead_bin_distribution": dict(sorted(lead_bins.items())),
            "author_labels_complete": False,
            "audio_sufficiency_labels_complete": False,
            "official_reference_consumed": True,
            "model_output_consumed": False,
            "author_items_sha256": file_sha256(author_path),
            "scorer_mapping_sha256": file_sha256(mapping_path),
            "interpretation": (
                "This is an exhaustive En-to-Zh human-authoring workspace over automatic "
                "R0 lexical candidates. It is not a gold event set, audio boundary, model "
                "result, or evidence that raw pixels outperform OCR."
            ),
        }
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "author_view" / "README.md").write_text(
            "# MCIF En-to-Zh Target-event Author View V1\n\n"
            "This view contains 355 candidate-bearing speech segments in deterministic "
            "shuffled order. For each item, inspect the current slide, English source "
            "segment, Chinese reference, R0 OCR, R1 structure text, and candidate options. "
            "At most one option may become a final event. All rows are pending and must "
            "be frozen before any audio-only validation or model inference.\n",
            encoding="utf-8",
        )
        author_checksum_entries, author_checksum_sha256 = write_checksums(
            temporary / "author_view"
        )
        scorer_checksum_entries, scorer_checksum_sha256 = write_checksums(
            temporary / "scorer_private"
        )
        root_checksum_entries, root_checksum_sha256 = write_checksums(temporary)
        os.rename(temporary, output_root)
        return {
            **report,
            "author_checksum_entries": author_checksum_entries,
            "author_checksum_sha256": author_checksum_sha256,
            "scorer_checksum_entries": scorer_checksum_entries,
            "scorer_checksum_sha256": scorer_checksum_sha256,
            "root_checksum_entries": root_checksum_entries,
            "root_checksum_sha256": root_checksum_sha256,
            "bundle_files": sum(1 for path in output_root.rglob("*") if path.is_file()),
            "bundle_bytes": sum(
                path.stat().st_size for path in output_root.rglob("*") if path.is_file()
            ),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--expected-candidates-sha256", required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--expected-references-sha256", required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--expected-ladder-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-inventory-hf-revision", required=True)
    parser.add_argument("--source-ladder-hf-revision", required=True)
    parser.add_argument("--code-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--expected-candidates", type=int, default=954)
    parser.add_argument("--expected-segments", type=int, default=919)
    parser.add_argument("--expected-items", type=int, default=355)
    parser.add_argument("--expected-talks", type=int, default=21)
    args = parser.parse_args()
    for path, expected, label in (
        (args.candidates, args.expected_candidates_sha256, "candidate inventory"),
        (args.references, args.expected_references_sha256, "reference segments"),
        (args.ladder, args.expected_ladder_sha256, "source evidence ladder"),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"MCIF {label} hash differs from the frozen input")
    if args.max_ngram < 1:
        raise ValueError("max-ngram must be positive")
    builder_git_commit = git_head_clean(args.code_repo)
    report = build_bundle(
        args.output_root,
        candidates=load_jsonl(args.candidates),
        references=load_jsonl(args.references),
        ladder=load_jsonl(args.ladder),
        source_root=args.source_root,
        candidate_inventory_sha256=args.expected_candidates_sha256,
        reference_segments_sha256=args.expected_references_sha256,
        ladder_sha256=args.expected_ladder_sha256,
        candidate_inventory_hf_revision=args.candidate_inventory_hf_revision,
        source_ladder_hf_revision=args.source_ladder_hf_revision,
        builder_git_commit=builder_git_commit,
        max_ngram=args.max_ngram,
        expected_candidates=args.expected_candidates,
        expected_segments=args.expected_segments,
        expected_items=args.expected_items,
        expected_talks=args.expected_talks,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
