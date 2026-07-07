#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from slidesst.data.io import read_jsonl
from slidesst.data.schema import ChallengeItem, EvidenceItem


VISUAL_FIELDS = ("ocr_text", "objects", "actions", "spatial_relations")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit enriched visual-context JSONL quality.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--sample-stats", default=None)
    parser.add_argument("--failures-output", default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    report = audit_challenge(Path(args.input), max_examples=args.max_examples)
    if args.failures_output:
        report["failures_output"] = write_failure_rows(Path(args.input), Path(args.failures_output))
    if args.baseline:
        report["baseline_comparison"] = compare_to_baseline(
            report,
            audit_challenge(Path(args.baseline), max_examples=0),
        )
    if args.evidence:
        report["evidence"] = audit_evidence(Path(args.evidence), report["expected_evidence_source_counts"])
    if args.sample_stats:
        report["diagnostic_sample_stats"] = json.loads(Path(args.sample_stats).read_text(encoding="utf-8"))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote visual context QA report to {output_path}")


def audit_challenge(path: Path, *, max_examples: int) -> dict[str, Any]:
    items = read_jsonl(path, ChallengeItem)
    ids = [item.id for item in items]
    duplicate_ids = [item_id for item_id, count in Counter(ids).items() if count > 1]
    field_counts: dict[str, list[int]] = {field: [] for field in VISUAL_FIELDS}
    field_char_counts: dict[str, list[int]] = {field: [] for field in VISUAL_FIELDS}
    scene_summary_chars: list[int] = []
    raw_output_chars: list[int] = []
    raw_parse_failure_chars: list[int] = []
    batch_size_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    raw_parse_failures = 0
    raw_parse_failure_no_ocr = 0
    raw_parse_failure_with_fallback_summary = 0
    missing_raw_output = 0
    missing_visual_context = 0
    missing_enrichment_metadata = 0
    empty_context = 0
    no_ocr_with_summary = 0
    long_scene_summary = 0
    long_raw_output = 0
    long_term_rows = 0
    examples: dict[str, list[dict[str, Any]]] = {
        "empty_context": [],
        "missing_raw_output": [],
        "raw_parse_failure": [],
        "long_scene_summary": [],
        "long_raw_output": [],
        "long_term": [],
    }
    scene_summary_counts: Counter[str] = Counter()
    raw_parse_failure_ids: set[str] = set()
    no_ocr_with_summary_ids: set[str] = set()

    for item in items:
        visual = item.visual_context
        if visual is None:
            missing_visual_context += 1
            add_example(examples["empty_context"], item, "missing visual_context", max_examples)
            continue

        meta = visual.metadata.get("context_enrichment") if visual.metadata else None
        if not isinstance(meta, dict):
            missing_enrichment_metadata += 1
            meta = {}
        batch_size_counts.update([str(meta.get("batch_size"))])
        provider_counts.update([str(meta.get("provider"))])
        model_counts.update([str(meta.get("model_id"))])

        summary = visual.scene_summary or ""
        raw_output = str(meta.get("raw_output") or "")
        raw_output_chars.append(len(raw_output))
        if not raw_output:
            missing_raw_output += 1
            add_example(examples["missing_raw_output"], item, "empty context_enrichment.raw_output", max_examples)
        elif not is_json_like(raw_output):
            raw_parse_failures += 1
            raw_parse_failure_ids.add(item.id)
            raw_parse_failure_chars.append(len(raw_output))
            if not visual.ocr_text:
                raw_parse_failure_no_ocr += 1
            if is_fallback_scene_summary(summary):
                raw_parse_failure_with_fallback_summary += 1
            add_example(examples["raw_parse_failure"], item, raw_output[:240], max_examples)
        if len(raw_output) > 3000:
            long_raw_output += 1
            add_example(examples["long_raw_output"], item, raw_output[:240], max_examples)

        scene_summary_chars.append(len(summary))
        if summary:
            scene_summary_counts.update([summary])
        if len(summary) > 240:
            long_scene_summary += 1
            add_example(examples["long_scene_summary"], item, summary[:240], max_examples)

        has_any_terms = False
        row_has_long_term = False
        for field in VISUAL_FIELDS:
            terms = list(getattr(visual, field) or [])
            field_counts[field].append(len(terms))
            field_char_counts[field].append(sum(len(term) for term in terms))
            if terms:
                has_any_terms = True
            if any(len(term) > 120 for term in terms):
                row_has_long_term = True
        if row_has_long_term:
            long_term_rows += 1
            add_example(examples["long_term"], item, first_long_term(visual), max_examples)

        if not summary and not has_any_terms:
            empty_context += 1
            add_example(examples["empty_context"], item, "no visual fields populated", max_examples)
        if not visual.ocr_text and summary:
            no_ocr_with_summary += 1
            no_ocr_with_summary_ids.add(item.id)

    expected_evidence_source_counts = {
        "video_action": sum(field_counts["actions"]),
        "video_object": sum(field_counts["objects"]),
        "video_ocr": sum(field_counts["ocr_text"]),
        "video_scene": sum(1 for value in scene_summary_chars if value > 0),
        "video_spatial": sum(field_counts["spatial_relations"]),
    }
    no_ocr_parse_overlap = raw_parse_failure_ids & no_ocr_with_summary_ids

    return {
        "path": str(path),
        "rows": len(items),
        "unique_ids": len(set(ids)),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_id_examples": duplicate_ids[:max_examples],
        "missing_visual_context": missing_visual_context,
        "missing_enrichment_metadata": missing_enrichment_metadata,
        "empty_context": empty_context,
        "no_ocr_with_summary": no_ocr_with_summary,
        "raw_parse_failures": raw_parse_failures,
        "raw_parse_failure_no_ocr": raw_parse_failure_no_ocr,
        "raw_parse_failure_with_fallback_summary": raw_parse_failure_with_fallback_summary,
        "missing_raw_output": missing_raw_output,
        "raw_parse_failure_no_ocr_with_summary_overlap": len(no_ocr_parse_overlap),
        "no_ocr_with_summary_not_parse_failure": len(no_ocr_with_summary_ids - raw_parse_failure_ids),
        "long_scene_summary_rows": long_scene_summary,
        "long_raw_output_rows": long_raw_output,
        "long_term_rows": long_term_rows,
        "provider_counts": dict(sorted(provider_counts.items())),
        "model_counts": dict(sorted(model_counts.items())),
        "batch_size_counts": dict(sorted(batch_size_counts.items(), key=lambda kv: kv[0])),
        "scene_summary_chars": describe_numbers(scene_summary_chars),
        "raw_output_chars": describe_numbers(raw_output_chars),
        "raw_parse_failure_chars": describe_numbers(raw_parse_failure_chars),
        "field_term_counts": {field: describe_numbers(values) for field, values in field_counts.items()},
        "field_char_counts": {field: describe_numbers(values) for field, values in field_char_counts.items()},
        "expected_evidence_source_counts": expected_evidence_source_counts,
        "top_repeated_scene_summaries": [
            {"count": count, "scene_summary": text}
            for text, count in scene_summary_counts.most_common(10)
            if count > 1
        ],
        "examples": examples,
    }


def write_failure_rows(input_path: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            item = ChallengeItem.model_validate(json.loads(line))
            if is_failure_item(item):
                dst.write(line if line.endswith("\n") else line + "\n")
                rows += 1
    return {"path": str(output_path), "rows": rows}


def is_failure_item(item: ChallengeItem) -> bool:
    visual = item.visual_context
    if visual is None:
        return True
    meta = visual.metadata.get("context_enrichment") if visual.metadata else None
    raw_output = str((meta or {}).get("raw_output") or "") if isinstance(meta, dict) else ""
    if not raw_output or not is_json_like(raw_output):
        return True
    return is_fallback_scene_summary(visual.scene_summary or "")


def audit_evidence(path: Path, expected_source_counts: dict[str, int] | None = None) -> dict[str, Any]:
    evidence = read_jsonl(path, EvidenceItem)
    source_counts = Counter(item.source_type for item in evidence)
    modality_counts = Counter(str(item.modality) for item in evidence)
    visual_only_counts = Counter(str(item.visual_only) for item in evidence)
    consistency = {}
    if expected_source_counts:
        consistency = {
            source_type: {
                "challenge_expected": expected,
                "evidence_actual": source_counts.get(source_type, 0),
                "delta": source_counts.get(source_type, 0) - expected,
            }
            for source_type, expected in sorted(expected_source_counts.items())
        }
    return {
        "path": str(path),
        "rows": len(evidence),
        "unique_evidence_ids": len({item.evidence_id for item in evidence}),
        "source_type_counts": dict(sorted(source_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "visual_only_counts": dict(sorted(visual_only_counts.items())),
        "challenge_consistency": consistency,
    }


def compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rows",
        "empty_context",
        "missing_raw_output",
        "no_ocr_with_summary",
        "raw_parse_failures",
        "long_scene_summary_rows",
        "long_raw_output_rows",
        "long_term_rows",
    ]
    numeric_delta = {
        key: {
            "current": current.get(key),
            "baseline": baseline.get(key),
            "delta": current.get(key, 0) - baseline.get(key, 0),
        }
        for key in keys
    }
    return {
        "baseline_path": baseline.get("path"),
        "numeric_delta": numeric_delta,
        "current_batch_size_counts": current.get("batch_size_counts"),
        "baseline_batch_size_counts": baseline.get("batch_size_counts"),
        "current_model_counts": current.get("model_counts"),
        "baseline_model_counts": baseline.get("model_counts"),
        "current_field_term_counts": current.get("field_term_counts"),
        "baseline_field_term_counts": baseline.get("field_term_counts"),
    }


def describe_numbers(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    values_sorted = sorted(values)
    return {
        "count": len(values),
        "min": values_sorted[0],
        "p50": percentile(values_sorted, 50),
        "p90": percentile(values_sorted, 90),
        "p99": percentile(values_sorted, 99),
        "max": values_sorted[-1],
        "mean": round(sum(values_sorted) / len(values_sorted), 3),
    }


def percentile(values_sorted: list[int], pct: int) -> int:
    idx = round((len(values_sorted) - 1) * pct / 100)
    return values_sorted[idx]


def is_json_like(text: str) -> bool:
    payload = extract_json_object(text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    fenced = re.match(r"```(?:json)?\s*(.*?)\s*```\s*$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        if "{" in text[end + 1 :] or "}" in text[:start]:
            return text
        return text[start : end + 1]
    return text


def is_fallback_scene_summary(text: str) -> bool:
    return text.startswith("Slide first frame for Chinese-LiPS topic ") and "OCR not provided" in text


def add_example(bucket: list[dict[str, Any]], item: ChallengeItem, reason: str, max_examples: int) -> None:
    if len(bucket) >= max_examples:
        return
    visual = item.visual_context
    bucket.append(
        {
            "id": item.id,
            "lecture_id": item.lecture_id,
            "reason": reason,
            "source_transcript": item.source_transcript[:160],
            "scene_summary": (visual.scene_summary if visual else "") or "",
            "ocr_text": list((visual.ocr_text if visual else []) or [])[:5],
        }
    )


def first_long_term(visual: Any) -> str:
    for field in VISUAL_FIELDS:
        for term in getattr(visual, field) or []:
            if len(term) > 120:
                return f"{field}: {term[:240]}"
    return ""


if __name__ == "__main__":
    main()
