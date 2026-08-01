#!/usr/bin/env python3
"""Analyze MCIF flat OCR and structured-text evidence without reading outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any


MACHINE_READABLE_LABELS = ("chart", "table", "formula")
HIERARCHY_LABELS = {
    "aside_text",
    "doc_title",
    "figure_title",
    "footer",
    "header",
    "number",
    "paragraph_title",
    "reference_content",
    "vision_footnote",
}
MARKDOWN_TABLE_SEPARATOR = re.compile(r"\|\s*:?-{2,}:?\s*\|")
HTML_TAG = re.compile(r"<[^>]+>")
TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def serialization_mode(block: dict[str, Any]) -> str:
    label = block.get("label")
    content = block.get("content")
    if not isinstance(content, str):
        raise ValueError("Structured block content must be a string")
    lower = content.lower()
    if label == "chart" and MARKDOWN_TABLE_SEPARATOR.search(content):
        return "chart_markdown_table"
    if label == "table" and "<table" in lower:
        return "table_html"
    if label == "formula" and (
        "$$" in content or "\\begin" in content or "\\frac" in content
    ):
        return "formula_latex"
    if label == "table" and "<img" in lower:
        return "table_image_placeholder"
    return "unstructured"


def tokens(text: str) -> list[str]:
    plain = html.unescape(HTML_TAG.sub(" ", text)).lower()
    return TOKEN.findall(plain)


def overlap_rates(flat_text: str, structured_text: str) -> tuple[float, float]:
    flat = Counter(tokens(flat_text))
    structured = Counter(tokens(structured_text))
    overlap = sum((flat & structured).values())
    flat_coverage = overlap / sum(flat.values()) if flat else 0.0
    structure_coverage = overlap / sum(structured.values()) if structured else 0.0
    return flat_coverage, structure_coverage


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def validate_row(row: dict[str, Any]) -> None:
    item_id = row.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("PP-Structure row is missing an id")
    if not isinstance(row.get("lecture_id"), str) or not row["lecture_id"]:
        raise ValueError(f"PP-Structure row is missing a lecture id: {item_id}")
    if row.get("source_transcript_consumed") is not False:
        raise ValueError(f"Source transcript was consumed for {item_id}")
    if row.get("target_or_reference_consumed") is not False:
        raise ValueError(f"Target or reference was consumed for {item_id}")
    forbidden = {
        "audio",
        "model_output",
        "reference",
        "reference_translation",
        "source_transcript",
        "target_translation",
    }
    if forbidden & set(row):
        raise ValueError(f"Outcome-bearing field found for {item_id}")
    frame = row.get("frame") or {}
    frame_path = frame.get("path")
    if not isinstance(frame_path, str) or not frame_path:
        raise ValueError(f"Frame path is missing for {item_id}")
    if Path(frame_path).is_absolute() or ".." in Path(frame_path).parts:
        raise ValueError(f"Frame path is not portable for {item_id}")


def row_categories(row: dict[str, Any]) -> set[str]:
    blocks = (row.get("structured_text") or {}).get("blocks") or []
    labels = {block.get("label") for block in blocks}
    modes = {serialization_mode(block) for block in blocks}
    categories = set()
    if "chart_markdown_table" in modes:
        categories.add("machine_readable_chart")
    if "table_html" in modes:
        categories.add("machine_readable_table")
    if "formula_latex" in modes:
        categories.add("machine_readable_formula")
    if "table" in labels and "machine_readable_table" not in categories:
        categories.add("table_detection_only")
    if labels & HIERARCHY_LABELS:
        categories.add("layout_hierarchy")
    if "image" in labels:
        categories.add("image_region")
    if categories & {
        "machine_readable_chart",
        "machine_readable_table",
        "machine_readable_formula",
    }:
        categories.add("machine_readable_nonflat_structure")
    return categories


def analyze(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, set[str]]]:
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("PP-Structure screen contains duplicate ids")

    category_rows = Counter()
    category_talks: dict[str, set[str]] = {}
    mode_blocks = Counter()
    label_rows = Counter()
    label_blocks = Counter()
    fallback_rows = Counter()
    flat_coverage: list[float] = []
    structure_coverage: list[float] = []
    categories_by_id: dict[str, set[str]] = {}
    talk_ids = set()
    for row in rows:
        validate_row(row)
        item_id = row["id"]
        talk_id = row["lecture_id"]
        talk_ids.add(talk_id)
        categories = row_categories(row)
        categories_by_id[item_id] = categories
        for category in categories:
            category_rows[category] += 1
            category_talks.setdefault(category, set()).add(talk_id)

        blocks = row["structured_text"]["blocks"]
        row_labels = set()
        for block in blocks:
            label = block["label"]
            label_blocks[label] += 1
            row_labels.add(label)
            mode_blocks[serialization_mode(block)] += 1
        label_rows.update(row_labels)

        fallback = row.get("inference_fallback")
        fallback_rows["none" if fallback is None else fallback["strategy"]] += 1
        flat_rate, structure_rate = overlap_rates(
            row["flat_ocr"]["text"], row["structured_text"]["compact_text"]
        )
        flat_coverage.append(flat_rate)
        structure_coverage.append(structure_rate)

    def distribution(values: list[float]) -> dict[str, float]:
        return {
            "mean": round(sum(values) / len(values), 6) if values else 0.0,
            "p10": round(percentile(values, 0.1), 6),
            "p50": round(percentile(values, 0.5), 6),
            "p90": round(percentile(values, 0.9), 6),
        }

    report = {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "talk_count": len(talk_ids),
        "category_row_counts": dict(sorted(category_rows.items())),
        "category_talk_counts": {
            category: len(talks)
            for category, talks in sorted(category_talks.items())
        },
        "serialization_mode_block_counts": dict(sorted(mode_blocks.items())),
        "structured_label_row_counts": dict(sorted(label_rows.items())),
        "structured_label_block_counts": dict(sorted(label_blocks.items())),
        "inference_fallback_row_counts": dict(sorted(fallback_rows.items())),
        "token_overlap": {
            "flat_ocr_tokens_covered_by_structure": distribution(flat_coverage),
            "structure_tokens_recoverable_from_flat_ocr": distribution(
                structure_coverage
            ),
        },
    }
    return report, categories_by_id


def build_qa_inventory(
    rows: list[dict[str, Any]],
    categories_by_id: dict[str, set[str]],
    *,
    per_category: int,
    seed: str,
) -> list[dict[str, Any]]:
    strata = (
        "machine_readable_chart",
        "machine_readable_table",
        "machine_readable_formula",
        "table_detection_only",
        "layout_hierarchy_without_machine_structure",
        "plain_or_image_only",
    )
    candidates: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in strata}
    for row in rows:
        categories = categories_by_id[row["id"]]
        assigned = set(categories)
        if "layout_hierarchy" in categories and "machine_readable_nonflat_structure" not in categories:
            assigned.add("layout_hierarchy_without_machine_structure")
        if not categories & {
            "layout_hierarchy",
            "machine_readable_nonflat_structure",
            "table_detection_only",
        }:
            assigned.add("plain_or_image_only")
        for stratum in strata:
            if stratum in assigned:
                candidates[stratum].append(row)

    inventory = []
    for stratum in strata:
        ranked = sorted(
            candidates[stratum],
            key=lambda row: hashlib.sha256(
                f"{seed}\0{stratum}\0{row['id']}".encode()
            ).hexdigest(),
        )[:per_category]
        for row in ranked:
            inventory.append(
                {
                    "qa_stratum": stratum,
                    "id": row["id"],
                    "lecture_id": row["lecture_id"],
                    "state_id": row["state_id"],
                    "frame": row["frame"],
                    "categories": sorted(categories_by_id[row["id"]]),
                    "structured_labels": sorted(
                        {block["label"] for block in row["structured_text"]["blocks"]}
                    ),
                    "flat_ocr_text": row["flat_ocr"]["text"],
                    "structured_compact_text": row["structured_text"]["compact_text"],
                    "inference_fallback": row.get("inference_fallback"),
                    "source_only_automatic_qa_not_annotation": True,
                }
            )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--qa-inventory", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, default=304)
    parser.add_argument("--expected-talks", type=int, default=21)
    parser.add_argument("--qa-per-category", type=int, default=8)
    parser.add_argument("--qa-seed", default="mcif-ppstructure-source-screen-v1")
    args = parser.parse_args()
    if args.report.exists() or args.qa_inventory.exists():
        raise FileExistsError("PP-Structure analysis outputs must be created once")
    if args.qa_per_category < 1:
        raise ValueError("QA sample size must be positive")
    input_sha256 = sha256_file(args.input)
    if input_sha256 != args.expected_input_sha256:
        raise ValueError("PP-Structure input SHA256 differs from the frozen contract")

    rows = load_jsonl(args.input)
    report, categories_by_id = analyze(rows)
    if report["rows"] != args.expected_rows or report["talk_count"] != args.expected_talks:
        raise ValueError("PP-Structure inventory differs from the frozen contract")
    inventory = build_qa_inventory(
        rows,
        categories_by_id,
        per_category=args.qa_per_category,
        seed=args.qa_seed,
    )
    report = {
        "artifact": "mcif_ppstructurev3_source_screen_analysis_v1",
        "status": "SOURCE_ONLY_AUTOMATIC_DIAGNOSTIC_NOT_LABELS_OR_ST_RESULT",
        "input_sha256": input_sha256,
        **report,
        "qa_inventory_rows": len(inventory),
        "qa_per_category": args.qa_per_category,
        "qa_seed": args.qa_seed,
        "interpretation": (
            "Machine-readable chart/table/formula counts describe automatic "
            "serializations over native causal frames. They do not establish visual "
            "truth, event eligibility, translation utility, or pixels beyond OCR. "
            "Table image placeholders are detection-only and are excluded from the "
            "machine-readable table tier."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.qa_inventory.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.qa_inventory.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in inventory
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
