#!/usr/bin/env python3
"""Build private MCIF R1/R2 target-event candidates beyond flat OCR."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable

from scripts.analyze_acl6060_ocr_anticipation import normalized_tokens, source_candidates
from scripts.build_mcif_outcome_candidate_inventory import maximal_candidates
from scripts.build_mcif_visual_token_controls import (
    canonical_sha256,
    file_sha256,
    git_head_clean,
    load_jsonl,
)
from scripts.finalize_mcif_visual_context_screen import parse_raw_output


REFERENCE_SCHEMA = "mcif_iwslt2026_reference_segment_v1"
LADDER_SCHEMA = "mcif_source_evidence_ladder_v1"
CANDIDATE_SCHEMA = "mcif_beyond_ocr_candidate_v1"
SEGMENT_SCHEMA = "mcif_beyond_ocr_candidate_segment_v1"
STRICT_R1_CONTENT_KINDS = frozenset(
    {"chart_markdown", "table_html", "formula_latex"}
)
R2_FIELDS = ("scene_summary", "objects", "actions", "spatial_relations")
VLM_EMPTY_LIST_FIELDS = (
    "streaming_units",
    "ambiguous_items",
    "hard_labels",
    "glossary",
    "background_docs",
    "evidence",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+\*?")


class _VisibleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def row_hash_valid(row: dict[str, Any]) -> bool:
    return row.get("row_sha256") == canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )


def unique_rows(
    rows: Iterable[dict[str, Any]], *, key: str, label: str
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


def visible_block_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    kind = block.get("content_kind")
    if not isinstance(content, str):
        raise ValueError("MCIF R1 block has non-string content")
    if kind == "visual_placeholder":
        return ""
    if kind == "table_html" or (kind == "text" and "<" in content):
        parser = _VisibleHTML()
        parser.feed(content)
        parser.close()
        return " ".join(parser.parts)
    if kind == "formula_latex":
        return LATEX_COMMAND_RE.sub(" ", content)
    return content


def candidate_set(text: str, max_ngram: int) -> set[tuple[str, str]]:
    return source_candidates(normalized_tokens(text), max_ngram)


def validate_reference_rows(
    rows: list[dict[str, Any]], *, expected_rows: int, expected_talks: int
) -> dict[str, list[dict[str, Any]]]:
    if len(rows) != expected_rows:
        raise ValueError("MCIF reference count differs from contract")
    unique_rows(rows, key="segment_id", label="MCIF reference segments")
    by_talk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("schema_version") != REFERENCE_SCHEMA or not row_hash_valid(row):
            raise ValueError(f"Invalid MCIF reference row: {row.get('segment_id')}")
        if row.get("official_reference_consumed") is not True or row.get(
            "model_output_consumed"
        ) is not False:
            raise ValueError("MCIF reference row has an invalid data boundary")
        by_talk[str(row.get("talk_id"))].append(row)
    if len(by_talk) != expected_talks or "None" in by_talk:
        raise ValueError("MCIF reference talk inventory differs from contract")
    for talk_id, talk_rows in by_talk.items():
        talk_rows.sort(key=lambda row: row["talk_segment_index"])
        if [row["talk_segment_index"] for row in talk_rows] != list(
            range(len(talk_rows))
        ):
            raise ValueError(f"MCIF reference segment ids are not contiguous for {talk_id}")
        for row in talk_rows:
            offset = float(row["offset_sec"])
            end = float(row["end_sec"])
            if offset < 0 or end <= offset:
                raise ValueError(f"MCIF reference timing is invalid for {row['segment_id']}")
    return by_talk


def validate_ladder_rows(
    rows: list[dict[str, Any]], *, expected_rows: int, expected_talks: int
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if len(rows) != expected_rows:
        raise ValueError("MCIF ladder count differs from contract")
    by_id = unique_rows(rows, key="id", label="MCIF evidence ladder")
    by_talk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("schema_version") != LADDER_SCHEMA or not row_hash_valid(row):
            raise ValueError(f"Invalid MCIF ladder row: {row.get('id')}")
        if row.get("source_transcript_consumed") is not False or row.get(
            "target_or_reference_consumed"
        ) is not False:
            raise ValueError("MCIF evidence ladder consumed forbidden outcome data")
        if not isinstance(row.get("r0_flat_ocr", {}).get("model_input_text"), str):
            raise ValueError("MCIF ladder has invalid R0 text")
        blocks = row.get("r1_structured_text", {}).get("blocks")
        if not isinstance(blocks, list):
            raise ValueError("MCIF ladder has invalid R1 blocks")
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError("MCIF ladder has a non-object R1 block")
            visible_block_text(block)
        by_talk[str(row.get("lecture_id"))].append(row)
    if len(by_talk) != expected_talks or "None" in by_talk:
        raise ValueError("MCIF ladder talk inventory differs from contract")
    for talk_id, talk_rows in by_talk.items():
        talk_rows.sort(key=lambda row: row["state_id"])
        if [row["state_id"] for row in talk_rows] != list(range(len(talk_rows))):
            raise ValueError(f"MCIF ladder states are not contiguous for {talk_id}")
        previous_start = None
        for row in talk_rows:
            start = float(row["availability_start_sec"])
            end = float(row["availability_end_sec"])
            if start < 0 or end <= start or (
                previous_start is not None and start <= previous_start
            ):
                raise ValueError(f"MCIF ladder timing is invalid for {row['id']}")
            previous_start = start
    return by_id, by_talk


def _require_empty_vlm_outcomes(row: dict[str, Any]) -> None:
    if row.get("source_transcript") != "":
        raise ValueError("MCIF VLM screen contains a source transcript")
    if row.get("reference_translation") not in (None, ""):
        raise ValueError("MCIF VLM screen contains a reference translation")
    reference = row.get("reference") or {}
    if reference.get("status") != "missing":
        raise ValueError("MCIF VLM screen reference status is not missing")
    if reference.get("translation") not in (None, "") or reference.get(
        "alternatives"
    ) not in (None, []):
        raise ValueError("MCIF VLM screen contains reference outcomes")
    if any(row.get(key) not in (None, []) for key in VLM_EMPTY_LIST_FIELDS):
        raise ValueError("MCIF VLM screen contains outcome-side fields")
    if (row.get("annotation") or {}).get("verified") is not False:
        raise ValueError("MCIF VLM screen was promoted to annotation")
    slides = row.get("slides") or {}
    if slides.get("matched_slide_text") not in (None, "") or slides.get(
        "matched_slide_image"
    ) not in (None, ""):
        raise ValueError("MCIF VLM screen contains matched-slide outcomes")


def validate_vlm_rows(
    rows: list[dict[str, Any]],
    *,
    ladder_by_id: dict[str, dict[str, Any]],
    expected_rows: int,
    expected_talks: int,
    expected_model_id: str,
    expected_model_revision: str,
    allowed_prompts: dict[str, str],
) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_rows:
        raise ValueError("MCIF VLM screen count differs from contract")
    by_id = unique_rows(rows, key="id", label="MCIF VLM screen")
    if set(by_id) != set(ladder_by_id):
        raise ValueError("MCIF VLM screen and ladder state ids differ")
    talks = set()
    for item_id, row in by_id.items():
        ladder = ladder_by_id[item_id]
        _require_empty_vlm_outcomes(row)
        if row.get("lecture_id") != ladder["lecture_id"]:
            raise ValueError(f"MCIF VLM talk binding differs for {item_id}")
        if row.get("source_lang") != "en" or row.get("target_lang") != "zh":
            raise ValueError(f"MCIF VLM language binding differs for {item_id}")
        visual = row.get("visual_context")
        if not isinstance(visual, dict):
            raise ValueError(f"MCIF VLM visual context is missing for {item_id}")
        metadata = visual.get("metadata") or {}
        enrichment = metadata.get("context_enrichment") or {}
        if (
            visual.get("video_id") != ladder["lecture_id"]
            or visual.get("clip_id") != item_id
            or metadata.get("state_id") != ladder["state_id"]
            or metadata.get("screen_role")
            != "private_source_only_prescreen_not_annotation"
        ):
            raise ValueError(f"MCIF VLM state binding differs for {item_id}")
        if (
            enrichment.get("provider") != "qwen_vl"
            or enrichment.get("model_id") != expected_model_id
            or enrichment.get("model_revision") != expected_model_revision
        ):
            raise ValueError(f"MCIF VLM model provenance differs for {item_id}")
        prompt_id = enrichment.get("prompt_id")
        if allowed_prompts.get(prompt_id) != enrichment.get("prompt_sha256"):
            raise ValueError(f"MCIF VLM prompt provenance differs for {item_id}")
        frame_sha = metadata.get("evidence_frame_sha256")
        if not isinstance(frame_sha, str) or SHA256_RE.fullmatch(frame_sha) is None:
            raise ValueError(f"MCIF VLM frame hash is invalid for {item_id}")
        raw_output = enrichment.get("raw_output")
        if not isinstance(raw_output, str):
            raise ValueError(f"MCIF VLM raw output is missing for {item_id}")
        parsed = parse_raw_output(raw_output, item_id)
        if not isinstance(visual.get("scene_summary"), str):
            raise ValueError(f"MCIF VLM scene summary is invalid for {item_id}")
        if any(
            not isinstance(visual.get(field), list)
            or any(not isinstance(value, str) for value in visual[field])
            for field in ("ocr_text", "objects", "actions", "spatial_relations")
        ):
            raise ValueError(f"MCIF VLM list field is invalid for {item_id}")
        if parsed.get("scene_summary") != visual["scene_summary"]:
            raise ValueError(f"MCIF VLM scene summary differs from raw output: {item_id}")
        for field in ("ocr_text", "objects", "actions", "spatial_relations"):
            raw_values = parsed.get(field)
            if raw_values is None and not visual[field]:
                raw_values = []
            if (
                not isinstance(raw_values, list)
                or any(not isinstance(value, str) for value in raw_values)
                or not set(visual[field]).issubset(set(raw_values))
            ):
                raise ValueError(
                    f"MCIF VLM structured field differs from raw output: {item_id}"
                )
        talks.add(row["lecture_id"])
    if len(talks) != expected_talks:
        raise ValueError("MCIF VLM talk inventory differs from contract")
    return by_id


def first_source_occurrences(
    references_by_talk: dict[str, list[dict[str, Any]]], *, max_ngram: int
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    output = {}
    for talk_id, rows in references_by_talk.items():
        first = {}
        for row in rows:
            for candidate in candidate_set(row["source_reference_en"], max_ngram):
                first.setdefault(candidate, row)
        output[talk_id] = first
    return output


def r1_state_evidence(
    row: dict[str, Any], *, max_ngram: int
) -> tuple[
    set[tuple[str, str]],
    set[tuple[str, str]],
    dict[tuple[str, str], list[dict[str, Any]]],
]:
    r0 = candidate_set(row["r0_flat_ocr"]["model_input_text"], max_ngram)
    all_r1: set[tuple[str, str]] = set()
    strict: set[tuple[str, str]] = set()
    origins: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for block in row["r1_structured_text"]["blocks"]:
        text = visible_block_text(block)
        candidates = candidate_set(text, max_ngram)
        all_r1.update(candidates)
        if block.get("content_kind") not in STRICT_R1_CONTENT_KINDS:
            continue
        for candidate in candidates - r0:
            strict.add(candidate)
            origins[candidate].append(
                {
                    "block_id": block.get("block_id"),
                    "content_kind": block.get("content_kind"),
                    "label": block.get("label"),
                    "content": block.get("content"),
                    "content_sha256": canonical_sha256(block.get("content")),
                }
            )
    return r0 | all_r1, strict, dict(origins)


def r2_state_evidence(
    row: dict[str, Any],
    *,
    excluded_r1: set[tuple[str, str]],
    max_ngram: int,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], list[dict[str, Any]]]]:
    visual = row["visual_context"]
    candidates: set[tuple[str, str]] = set()
    origins: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for field in R2_FIELDS:
        values = [visual[field]] if field == "scene_summary" else visual[field]
        for index, text in enumerate(values):
            for candidate in candidate_set(text, max_ngram) - excluded_r1:
                candidates.add(candidate)
                origins[candidate].append(
                    {
                        "descriptor_field": field,
                        "descriptor_index": index,
                        "descriptor_text": text,
                        "descriptor_sha256": canonical_sha256(text),
                    }
                )
    return candidates, dict(origins)


def _blank_annotation_fields() -> dict[str, Any]:
    return {
        "candidate_eligibility": None,
        "visual_evidence_correct": None,
        "ocr_insufficient": None,
        "acceptable_target_realizations_zh": [],
        "forbidden_target_realizations_zh": [],
        "audio_insufficient_until_sec": None,
        "audio_first_sufficient_sec": None,
        "annotator_id": None,
        "annotation_note": "",
    }


def build_candidates(
    references_by_talk: dict[str, list[dict[str, Any]]],
    ladder_by_talk: dict[str, list[dict[str, Any]]],
    vlm_by_id: dict[str, dict[str, Any]],
    *,
    max_ngram: int,
    vlm_output_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first_source = first_source_occurrences(
        references_by_talk, max_ngram=max_ngram
    )
    r1_output = []
    r2_output = []
    for talk_id in sorted(references_by_talk):
        states = ladder_by_talk[talk_id]
        state_times = [float(row["availability_start_sec"]) for row in states]
        r1_sets = []
        r1_origins = []
        r2_sets = []
        r2_origins = []
        for state in states:
            all_r1, strict_r1, strict_origins = r1_state_evidence(
                state, max_ngram=max_ngram
            )
            semantic_r2, semantic_origins = r2_state_evidence(
                vlm_by_id[state["id"]],
                excluded_r1=all_r1,
                max_ngram=max_ngram,
            )
            r1_sets.append(strict_r1)
            r1_origins.append(strict_origins)
            r2_sets.append(semantic_r2)
            r2_origins.append(semantic_origins)
        tier_specs = (
            (
                "r1_strict",
                r1_sets,
                r1_origins,
                "AUTOMATIC_REFERENCE_AWARE_R1_STRICT_CANDIDATE_NOT_GOLD",
                r1_output,
            ),
            (
                "r2_semantic",
                r2_sets,
                r2_origins,
                "AUTOMATIC_REFERENCE_AWARE_UNVERIFIED_VLM_CANDIDATE_NOT_GOLD",
                r2_output,
            ),
        )
        tier_counts = Counter()
        for segment in references_by_talk[talk_id]:
            state_index = bisect_right(state_times, float(segment["offset_sec"])) - 1
            if state_index < 0 or float(segment["offset_sec"]) >= float(
                states[state_index]["availability_end_sec"]
            ):
                continue
            for tier, evidence_sets, origins, status, output in tier_specs:
                intersections = {
                    candidate
                    for candidate in evidence_sets[state_index]
                    if first_source[talk_id].get(candidate, {}).get("segment_id")
                    == segment["segment_id"]
                }
                for candidate in maximal_candidates(intersections):
                    earliest_index = state_index
                    while (
                        earliest_index > 0
                        and candidate in evidence_sets[earliest_index - 1]
                        and float(states[earliest_index - 1]["availability_end_sec"])
                        >= float(states[earliest_index]["availability_start_sec"])
                    ):
                        earliest_index -= 1
                    current_state = states[state_index]
                    earliest_state = states[earliest_index]
                    tier_counts[tier] += 1
                    prefix = "R1C" if tier == "r1_strict" else "R2C"
                    row = {
                        "schema_version": CANDIDATE_SCHEMA,
                        "status": status,
                        "candidate_id": f"mcif:{talk_id}:{prefix}{tier_counts[tier]:03d}",
                        "evidence_tier": tier,
                        "talk_id": talk_id,
                        "segment_id": segment["segment_id"],
                        "talk_segment_index": segment["talk_segment_index"],
                        "source_segment_offset_sec": segment["offset_sec"],
                        "source_segment_end_sec": segment["end_sec"],
                        "source_reference_en": segment["source_reference_en"],
                        "target_reference_zh": segment["target_reference_zh"],
                        "candidate_kind": candidate[0],
                        "normalized_source_candidate": candidate[1],
                        "candidate_token_count": len(candidate[1].split()),
                        "current_state_id": current_state["id"],
                        "current_state_row_sha256": current_state["row_sha256"],
                        "current_evidence_available_sec": current_state[
                            "availability_start_sec"
                        ],
                        "current_evidence_origins": origins[state_index][candidate],
                        "earliest_contiguous_state_id": earliest_state["id"],
                        "earliest_contiguous_state_row_sha256": earliest_state[
                            "row_sha256"
                        ],
                        "earliest_contiguous_evidence_sec": earliest_state[
                            "availability_start_sec"
                        ],
                        "lead_lower_bound_sec": round(
                            float(segment["offset_sec"])
                            - float(earliest_state["availability_start_sec"]),
                            6,
                        ),
                        "current_r0_candidate_absent": True,
                        "current_r1_candidate_absent": tier == "r2_semantic",
                        "source_only_vlm_output_sha256": (
                            vlm_output_sha256 if tier == "r2_semantic" else None
                        ),
                        "current_visual_context_sha256": (
                            canonical_sha256(vlm_by_id[current_state["id"]]["visual_context"])
                            if tier == "r2_semantic"
                            else None
                        ),
                        **_blank_annotation_fields(),
                        "official_reference_consumed": True,
                        "automatic_extractor_output_consumed": tier == "r1_strict",
                        "generative_model_output_consumed": tier == "r2_semantic",
                        "model_output_role": (
                            "source_only_candidate_proposal"
                            if tier == "r2_semantic"
                            else None
                        ),
                    }
                    row["row_sha256"] = canonical_sha256(row)
                    output.append(row)
    return r1_output, r2_output


def build_candidate_segments(
    candidates: list[dict[str, Any]], references: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    reference_by_id = {row["segment_id"]: row for row in references}
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_segment[row["segment_id"]].append(row)
    output = []
    for segment_id in sorted(
        by_segment,
        key=lambda value: (
            reference_by_id[value]["talk_id"],
            reference_by_id[value]["talk_segment_index"],
        ),
    ):
        reference = reference_by_id[segment_id]
        rows = sorted(by_segment[segment_id], key=lambda row: row["candidate_id"])
        item = {
            "schema_version": SEGMENT_SCHEMA,
            "segment_id": segment_id,
            "talk_id": reference["talk_id"],
            "talk_segment_index": reference["talk_segment_index"],
            "source_segment_offset_sec": reference["offset_sec"],
            "source_reference_en": reference["source_reference_en"],
            "target_reference_zh": reference["target_reference_zh"],
            "candidate_ids": [row["candidate_id"] for row in rows],
            "r1_strict_candidate_count": sum(
                row["evidence_tier"] == "r1_strict" for row in rows
            ),
            "r2_semantic_candidate_count": sum(
                row["evidence_tier"] == "r2_semantic" for row in rows
            ),
            "official_reference_consumed": True,
            "human_labels_complete": False,
        }
        item["row_sha256"] = canonical_sha256(item)
        output.append(item)
    return output


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
    checksum_path = root / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return len(paths), file_sha256(checksum_path)


def _tier_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    leads = [float(row["lead_lower_bound_sec"]) for row in rows]
    return {
        "candidates": len(rows),
        "talks": len({row["talk_id"] for row in rows}),
        "segments": len({row["segment_id"] for row in rows}),
        "candidate_kind_distribution": dict(
            sorted(Counter(row["candidate_kind"] for row in rows).items())
        ),
        "lead_ge_5_sec": sum(value >= 5 for value in leads),
        "lead_ge_10_sec": sum(value >= 10 for value in leads),
        "lead_max_sec": max(leads, default=None),
    }


def build_bundle(
    output_root: Path,
    *,
    references: list[dict[str, Any]],
    r1_candidates: list[dict[str, Any]],
    r2_candidates: list[dict[str, Any]],
    candidate_segments: list[dict[str, Any]],
    reference_sha256: str,
    ladder_sha256: str,
    vlm_output_sha256: str,
    model_id: str,
    model_revision: str,
    allowed_prompts: dict[str, str],
    builder_git_commit: str,
    max_ngram: int,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF beyond-OCR output root must not already exist")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        paths = {
            "r1": temporary / "r1_strict_candidates.jsonl",
            "r2": temporary / "r2_semantic_candidates.jsonl",
            "segments": temporary / "candidate_segments.jsonl",
        }
        write_jsonl(paths["r1"], r1_candidates)
        write_jsonl(paths["r2"], r2_candidates)
        write_jsonl(paths["segments"], candidate_segments)
        report = {
            "schema_version": "mcif_beyond_ocr_candidate_inventory_report_v1",
            "status": "PRIVATE_REFERENCE_AWARE_CANDIDATES_PENDING_INDEPENDENT_HUMAN_VALIDATION",
            "builder_git_commit": builder_git_commit,
            "reference_segments_sha256": reference_sha256,
            "source_evidence_ladder_sha256": ladder_sha256,
            "source_only_vlm_output_sha256": vlm_output_sha256,
            "vlm_model_id": model_id,
            "vlm_model_revision": model_revision,
            "allowed_vlm_prompts": dict(sorted(allowed_prompts.items())),
            "reference_segments": len(references),
            "reference_talks": len({row["talk_id"] for row in references}),
            "candidate_segments": len(candidate_segments),
            "r1_strict": _tier_stats(r1_candidates),
            "r2_semantic": _tier_stats(r2_candidates),
            "max_ngram": max_ngram,
            "strict_r1_content_kinds": sorted(STRICT_R1_CONTENT_KINDS),
            "r2_descriptor_fields": list(R2_FIELDS),
            "r2_ocr_text_excluded": True,
            "r2_candidates_present_in_current_r1_excluded": True,
            "vlm_nominal_timing_ignored": True,
            "official_reference_consumed": True,
            "human_event_labels_complete": False,
            "visual_correctness_labels_complete": False,
            "ocr_insufficiency_labels_complete": False,
            "audio_sufficiency_labels_complete": False,
            "interpretation": (
                "R1 and R2 rows are high-recall proposal candidates, not gold events or "
                "evidence that pixels outperform OCR. R2 descriptions are unverified model "
                "outputs and require independent visual-correctness and OCR-insufficiency labels."
            ),
        }
        report.update(
            {
                "r1_strict_candidates_sha256": file_sha256(paths["r1"]),
                "r2_semantic_candidates_sha256": file_sha256(paths["r2"]),
                "candidate_segments_sha256": file_sha256(paths["segments"]),
            }
        )
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# MCIF Beyond-OCR Candidate Inventory V1\n\n"
            "Private reference-aware proposal pool for independent R1/R2 human "
            "validation. R1 uses only strict chart/table/formula block content beyond "
            "flat OCR. R2 uses source-only VLM descriptions excluding `ocr_text` and "
            "terms already present in current R1 blocks. The VLM timing fields are not "
            "used; causal timing comes from the corrected source-evidence ladder.\n\n"
            "This bundle must not be mounted into inference, shown to the R0 target-event "
            "author, used to remove any R2 condition, or reported as gold labels/results.\n",
            encoding="utf-8",
        )
        checksum_entries, checksum_manifest_sha256 = write_checksums(temporary)
        os.rename(temporary, output_root)
        return {
            **report,
            "checksum_entries": checksum_entries,
            "checksum_manifest_sha256": checksum_manifest_sha256,
            "bundle_files": sum(path.is_file() for path in output_root.rglob("*")),
            "bundle_bytes": sum(
                path.stat().st_size for path in output_root.rglob("*") if path.is_file()
            ),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_allowed_prompts(values: list[str]) -> dict[str, str]:
    output = {}
    for value in values:
        prompt_id, separator, digest = value.partition("=")
        if not separator or not prompt_id or SHA256_RE.fullmatch(digest) is None:
            raise ValueError("Allowed prompts must use PROMPT_ID=SHA256")
        if prompt_id in output and output[prompt_id] != digest:
            raise ValueError(f"Conflicting hashes for prompt {prompt_id}")
        output[prompt_id] = digest
    if not output:
        raise ValueError("At least one allowed prompt is required")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--expected-references-sha256", required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--expected-ladder-sha256", required=True)
    parser.add_argument("--vlm-output", type=Path, required=True)
    parser.add_argument("--expected-vlm-output-sha256", required=True)
    parser.add_argument("--expected-model-id", required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--allowed-prompt", action="append", default=[])
    parser.add_argument("--code-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--expected-reference-rows", type=int, default=919)
    parser.add_argument("--expected-state-rows", type=int, default=304)
    parser.add_argument("--expected-talks", type=int, default=21)
    args = parser.parse_args()
    for path, expected, label in (
        (args.references, args.expected_references_sha256, "references"),
        (args.ladder, args.expected_ladder_sha256, "ladder"),
        (args.vlm_output, args.expected_vlm_output_sha256, "VLM output"),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"MCIF {label} hash differs from the frozen input")
    if args.max_ngram < 1:
        raise ValueError("max-ngram must be positive")
    allowed_prompts = parse_allowed_prompts(args.allowed_prompt)
    builder_git_commit = git_head_clean(args.code_repo)
    references = load_jsonl(args.references)
    ladder = load_jsonl(args.ladder)
    vlm_rows = load_jsonl(args.vlm_output)
    references_by_talk = validate_reference_rows(
        references,
        expected_rows=args.expected_reference_rows,
        expected_talks=args.expected_talks,
    )
    ladder_by_id, ladder_by_talk = validate_ladder_rows(
        ladder,
        expected_rows=args.expected_state_rows,
        expected_talks=args.expected_talks,
    )
    if set(references_by_talk) != set(ladder_by_talk):
        raise ValueError("MCIF reference and ladder talk inventories differ")
    vlm_by_id = validate_vlm_rows(
        vlm_rows,
        ladder_by_id=ladder_by_id,
        expected_rows=args.expected_state_rows,
        expected_talks=args.expected_talks,
        expected_model_id=args.expected_model_id,
        expected_model_revision=args.expected_model_revision,
        allowed_prompts=allowed_prompts,
    )
    r1_candidates, r2_candidates = build_candidates(
        references_by_talk,
        ladder_by_talk,
        vlm_by_id,
        max_ngram=args.max_ngram,
        vlm_output_sha256=args.expected_vlm_output_sha256,
    )
    candidate_segments = build_candidate_segments(
        [*r1_candidates, *r2_candidates], references
    )
    report = build_bundle(
        args.output_root,
        references=references,
        r1_candidates=r1_candidates,
        r2_candidates=r2_candidates,
        candidate_segments=candidate_segments,
        reference_sha256=args.expected_references_sha256,
        ladder_sha256=args.expected_ladder_sha256,
        vlm_output_sha256=args.expected_vlm_output_sha256,
        model_id=args.expected_model_id,
        model_revision=args.expected_model_revision,
        allowed_prompts=allowed_prompts,
        builder_git_commit=builder_git_commit,
        max_ngram=args.max_ngram,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
