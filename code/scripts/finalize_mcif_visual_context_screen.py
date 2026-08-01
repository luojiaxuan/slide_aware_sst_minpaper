#!/usr/bin/env python3
"""Finalize and audit a portable MCIF source-only VLM screen artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VISUAL_LIST_FIELDS = ("ocr_text", "objects", "actions", "spatial_relations")
EMPTY_OUTCOME_FIELDS = (
    "streaming_units",
    "ambiguous_items",
    "hard_labels",
    "glossary",
    "background_docs",
    "evidence",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def parse_raw_output(text: str, item_id: str) -> dict[str, Any]:
    try:
        parsed = json.loads(extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid raw VLM JSON for {item_id}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Raw VLM output is not an object for {item_id}")
    return parsed


def require_unique_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    ids = [row.get("id") for row in rows]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError(f"{label} contains a missing id")
    duplicate_ids = [item_id for item_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"{label} contains duplicate ids: {duplicate_ids[:3]}")
    return {row["id"]: row for row in rows}


def overlay_replacements(
    base_rows: list[dict[str, Any]], replacement_groups: list[list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[int]]:
    base_by_id = require_unique_rows(base_rows, "base shards")
    ordered_ids = [row["id"] for row in base_rows]
    replacement_counts = []
    for index, rows in enumerate(replacement_groups):
        replacements = require_unique_rows(rows, f"replacement shard group {index}")
        extra = sorted(set(replacements) - set(base_by_id))
        if extra:
            raise ValueError(f"Replacement shard contains unknown ids: {extra[:3]}")
        base_by_id.update(replacements)
        replacement_counts.append(len(replacements))
    return [base_by_id[item_id] for item_id in ordered_ids], replacement_counts


def validate_source_only(input_row: dict[str, Any], output_row: dict[str, Any]) -> None:
    item_id = input_row["id"]
    for key in ("id", "lecture_id", "source_lang", "target_lang", "source_transcript"):
        if output_row.get(key) != input_row.get(key):
            raise ValueError(f"Protected field {key} changed for {item_id}")
    if output_row.get("source_transcript") != "":
        raise ValueError(f"Source transcript leaked into {item_id}")

    input_video = input_row.get("video") or {}
    output_video = output_row.get("video") or {}
    for key in ("start_sec", "end_sec", "frame_paths"):
        if output_video.get(key) != input_video.get(key):
            raise ValueError(f"Protected video field {key} changed for {item_id}")

    if output_row.get("reference_translation") not in (None, ""):
        raise ValueError(f"Reference translation leaked into {item_id}")
    reference = output_row.get("reference") or {}
    if reference.get("translation") not in (None, ""):
        raise ValueError(f"Nested reference translation leaked into {item_id}")
    if reference.get("alternatives") not in (None, []):
        raise ValueError(f"Reference alternatives leaked into {item_id}")
    for key in EMPTY_OUTCOME_FIELDS:
        if output_row.get(key) not in (None, []):
            raise ValueError(f"Outcome field {key} is populated for {item_id}")

    slides = output_row.get("slides") or {}
    if any(
        slides.get(key) not in (None, "")
        for key in ("matched_slide_text", "matched_slide_image")
    ):
        raise ValueError(f"Matched slide outcome leaked into {item_id}")


def validate_visual_binding(
    input_row: dict[str, Any],
    output_row: dict[str, Any],
    *,
    expected_raw_model_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item_id = input_row["id"]
    input_visual = input_row.get("visual_context") or {}
    output_visual = output_row.get("visual_context") or {}
    if output_visual.get("video_id") != input_visual.get("video_id"):
        raise ValueError(f"Visual video id changed for {item_id}")
    if output_visual.get("clip_id") != input_visual.get("clip_id"):
        raise ValueError(f"Visual clip id changed for {item_id}")

    input_meta = input_visual.get("metadata") or {}
    output_meta = output_visual.get("metadata") or {}
    for key in (
        "screen_role",
        "state_id",
        "availability_start_sec",
        "availability_end_sec",
        "evidence_frame_sha256",
    ):
        if output_meta.get(key) != input_meta.get(key):
            raise ValueError(f"Visual binding field {key} changed for {item_id}")

    enrichment = output_meta.get("context_enrichment")
    if not isinstance(enrichment, dict):
        raise ValueError(f"Missing context enrichment metadata for {item_id}")
    if enrichment.get("provider") != "qwen_vl":
        raise ValueError(f"Unexpected VLM provider for {item_id}")
    if enrichment.get("model_id") != expected_raw_model_id:
        raise ValueError(f"Unexpected raw model binding for {item_id}")

    relative_frame = (input_row.get("video") or {}).get("frame_paths", [None])[0]
    if not isinstance(relative_frame, str) or not relative_frame:
        raise ValueError(f"Missing portable frame path for {item_id}")
    raw_frame = enrichment.get("frame_path")
    if not isinstance(raw_frame, str):
        raise ValueError(f"Missing raw frame path for {item_id}")
    relative_parts = Path(relative_frame).parts
    if Path(raw_frame).parts[-len(relative_parts) :] != relative_parts:
        raise ValueError(f"Raw frame path does not match the input binding for {item_id}")

    raw_output = enrichment.get("raw_output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError(f"Missing raw VLM output for {item_id}")
    parsed = parse_raw_output(raw_output, item_id)
    return enrichment, parsed


def finalize_rows(
    input_rows: list[dict[str, Any]],
    shard_rows: list[dict[str, Any]],
    *,
    expected_raw_model_id: str,
    canonical_model_id: str,
    model_revision: str,
    default_prompt_id: str,
    default_prompt_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_by_id = require_unique_rows(input_rows, "input")
    shard_by_id = require_unique_rows(shard_rows, "shards")
    if set(input_by_id) != set(shard_by_id):
        missing = sorted(set(input_by_id) - set(shard_by_id))
        extra = sorted(set(shard_by_id) - set(input_by_id))
        raise ValueError(f"Shard id set mismatch; missing={missing[:3]} extra={extra[:3]}")

    field_row_counts = Counter()
    field_term_counts = Counter()
    talk_counts = Counter()
    batch_size_counts = Counter()
    prompt_id_counts = Counter()
    prompt_sha256_counts = Counter()
    raw_output_counts = Counter()
    finalized = []
    for input_row in input_rows:
        item_id = input_row["id"]
        output_row = shard_by_id[item_id]
        validate_source_only(input_row, output_row)
        enrichment, _ = validate_visual_binding(
            input_row,
            output_row,
            expected_raw_model_id=expected_raw_model_id,
        )

        portable = copy.deepcopy(output_row)
        portable_enrichment = portable["visual_context"]["metadata"]["context_enrichment"]
        portable_enrichment["model_id"] = canonical_model_id
        portable_enrichment["model_revision"] = model_revision
        portable_enrichment["frame_path"] = input_row["video"]["frame_paths"][0]
        output_metadata = portable["visual_context"]["metadata"]
        prompt_id = output_metadata.get("screen_prompt_id", default_prompt_id)
        prompt_sha256 = output_metadata.get(
            "screen_prompt_sha256", default_prompt_sha256
        )
        portable_enrichment["prompt_id"] = prompt_id
        portable_enrichment["prompt_sha256"] = prompt_sha256
        batch_size_counts[str(portable_enrichment.get("batch_size"))] += 1
        prompt_id_counts[str(prompt_id)] += 1
        prompt_sha256_counts[str(prompt_sha256)] += 1
        raw_output_counts[portable_enrichment["raw_output"]] += 1

        visual = portable["visual_context"]
        if visual.get("scene_summary"):
            field_row_counts["scene_summary"] += 1
        for field in VISUAL_LIST_FIELDS:
            terms = visual.get(field) or []
            if terms:
                field_row_counts[field] += 1
            field_term_counts[field] += len(terms)
        talk_counts[portable["lecture_id"]] += 1
        finalized.append(portable)

    duplicate_raw_groups = sum(count > 1 for count in raw_output_counts.values())
    stats = {
        "rows": len(finalized),
        "unique_ids": len({row["id"] for row in finalized}),
        "talk_count": len(talk_counts),
        "talk_state_counts": dict(sorted(talk_counts.items())),
        "field_nonempty_row_counts": dict(sorted(field_row_counts.items())),
        "field_term_counts": dict(sorted(field_term_counts.items())),
        "rows_with_any_non_ocr_visual_description": sum(
            bool(
                (row["visual_context"].get("objects") or [])
                or (row["visual_context"].get("actions") or [])
                or (row["visual_context"].get("spatial_relations") or [])
            )
            for row in finalized
        ),
        "rows_with_spatial_relation_candidates": sum(
            bool(row["visual_context"].get("spatial_relations")) for row in finalized
        ),
        "raw_json_parse_failures": 0,
        "duplicate_raw_output_groups": duplicate_raw_groups,
        "batch_size_counts": dict(sorted(batch_size_counts.items())),
        "prompt_id_counts": dict(sorted(prompt_id_counts.items())),
        "prompt_sha256_counts": dict(sorted(prompt_sha256_counts.items())),
    }
    return finalized, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--replacement-shard", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-talks", type=int, required=True)
    parser.add_argument("--expected-raw-model-id", required=True)
    parser.add_argument("--canonical-model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--default-prompt-id", required=True)
    parser.add_argument("--default-prompt-sha256", required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Final MCIF screen artifacts must be created once")

    input_rows = load_jsonl(args.input)
    base_shard_rows = [row for path in args.shard for row in load_jsonl(path)]
    replacement_groups = [load_jsonl(path) for path in args.replacement_shard]
    shard_rows, replacement_counts = overlay_replacements(
        base_shard_rows, replacement_groups
    )
    finalized, stats = finalize_rows(
        input_rows,
        shard_rows,
        expected_raw_model_id=args.expected_raw_model_id,
        canonical_model_id=args.canonical_model_id,
        model_revision=args.model_revision,
        default_prompt_id=args.default_prompt_id,
        default_prompt_sha256=args.default_prompt_sha256,
    )
    if stats["rows"] != args.expected_rows or stats["unique_ids"] != args.expected_rows:
        raise ValueError("Final MCIF screen row count does not match the frozen contract")
    if stats["talk_count"] != args.expected_talks:
        raise ValueError("Final MCIF screen talk count does not match the frozen contract")

    write_jsonl(args.output, finalized)
    report = {
        "artifact": "mcif_source_only_visual_context_screen_v1",
        "status": "PRIVATE_SOURCE_ONLY_PRESCREEN_NOT_ANNOTATION",
        "git_commit": args.git_commit,
        "input_sha256": sha256_file(args.input),
        "shard_sha256": {path.name: sha256_file(path) for path in args.shard},
        "replacement_shard_sha256": {
            path.name: sha256_file(path) for path in args.replacement_shard
        },
        "replacement_group_row_counts": replacement_counts,
        "output_sha256": sha256_file(args.output),
        "model_id": args.canonical_model_id,
        "model_revision": args.model_revision,
        **stats,
        "interpretation": (
            "Counts describe reference-free VLM fields over the complete frozen state "
            "inventory. They are not human eligibility labels, image-needed labels, "
            "translation results, or evidence that pixels outperform OCR."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
