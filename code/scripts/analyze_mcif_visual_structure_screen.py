#!/usr/bin/env python3
"""Run a lexical, non-labeling triage over MCIF VLM visual descriptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RELATION_PATTERNS = {
    "table_association": re.compile(r"\b(table|rows?|columns?|cells?|matrix)\b", re.I),
    "chart_quantitative": re.compile(
        r"\b(chart|bars?|axes?|axis|curves?|plots?|legend|trend|higher|lower|"
        r"increase|decrease|largest|smallest|taller|shorter|outperform)\b",
        re.I,
    ),
    "connectivity_process": re.compile(
        r"\b(arrows?|connect(?:s|ed)?|flows?|pipeline|nodes?|edges?|graph|"
        r"sequence|feeds? into|points? to|links?|linked)\b",
        re.I,
    ),
    "formula_grouping": re.compile(
        r"\b(formulas?|equations?|fractions?|numerator|denominator|brackets?|"
        r"parentheses|subscripts?|superscripts?)\b",
        re.I,
    ),
    "visual_emphasis": re.compile(
        r"\b(highlight(?:ed)?|bold(?:ed)?|colou?r(?:-coded)?|red|blue|green|"
        r"orange|purple|shaded|dashed|solid)\b",
        re.I,
    ),
    "label_mapping": re.compile(
        r"\b(maps?|corresponds?|associated|association|represents?|indicates?)\b",
        re.I,
    ),
}
SIMPLE_LAYOUT_PATTERN = re.compile(
    r"\b(above|below|top|bottom|left|right|positioned|aligned|center(?:ed)?|"
    r"horizontal(?:ly)?|vertical(?:ly)?|next to|beside)\b",
    re.I,
)


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


def validate_source_only(row: dict[str, Any]) -> None:
    item_id = row.get("id")
    if row.get("source_transcript") != "":
        raise ValueError(f"Source transcript is populated for {item_id}")
    if row.get("reference_translation") not in (None, ""):
        raise ValueError(f"Reference translation is populated for {item_id}")
    reference = row.get("reference") or {}
    if reference.get("translation") not in (None, ""):
        raise ValueError(f"Nested reference translation is populated for {item_id}")
    if reference.get("alternatives") not in (None, []):
        raise ValueError(f"Reference alternatives are populated for {item_id}")


def relation_categories(relations: list[str]) -> set[str]:
    text = "\n".join(relations)
    return {
        category for category, pattern in RELATION_PATTERNS.items() if pattern.search(text)
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row.get("id") for row in rows]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError("Visual screen contains a missing id")
    if len(ids) != len(set(ids)):
        raise ValueError("Visual screen contains duplicate ids")

    talk_rows = Counter()
    category_rows = Counter()
    category_talks: dict[str, set[str]] = defaultdict(set)
    relation_rows = 0
    structural_rows = 0
    simple_layout_only_rows = 0
    relation_without_known_pattern_rows = 0
    no_relation_rows = 0
    for row in rows:
        validate_source_only(row)
        talk_id = str(row.get("lecture_id") or "")
        if not talk_id:
            raise ValueError(f"Missing lecture id for {row['id']}")
        talk_rows[talk_id] += 1
        visual = row.get("visual_context") or {}
        relations = visual.get("spatial_relations") or []
        if not isinstance(relations, list) or any(
            not isinstance(value, str) for value in relations
        ):
            raise ValueError(f"Invalid spatial relation list for {row['id']}")
        if not relations:
            no_relation_rows += 1
            continue
        relation_rows += 1
        categories = relation_categories(relations)
        for category in categories:
            category_rows[category] += 1
            category_talks[category].add(talk_id)
        has_simple_layout = SIMPLE_LAYOUT_PATTERN.search("\n".join(relations)) is not None
        if categories:
            structural_rows += 1
        elif has_simple_layout:
            simple_layout_only_rows += 1
        else:
            relation_without_known_pattern_rows += 1

    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "talk_count": len(talk_rows),
        "talk_state_counts": dict(sorted(talk_rows.items())),
        "rows_with_relations": relation_rows,
        "rows_without_relations": no_relation_rows,
        "rows_with_lexical_structural_relation_candidates": structural_rows,
        "talks_with_lexical_structural_relation_candidates": len(
            {
                row["lecture_id"]
                for row in rows
                if relation_categories(
                    (row.get("visual_context") or {}).get("spatial_relations") or []
                )
            }
        ),
        "simple_layout_only_rows": simple_layout_only_rows,
        "relation_without_known_pattern_rows": relation_without_known_pattern_rows,
        "relation_category_row_counts": dict(sorted(category_rows.items())),
        "relation_category_talk_counts": {
            category: len(talks) for category, talks in sorted(category_talks.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-talks", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("MCIF lexical triage output must be created once")
    actual_hash = sha256_file(args.input)
    if actual_hash != args.expected_input_sha256:
        raise ValueError("MCIF visual screen SHA256 does not match the frozen input")

    report = analyze(load_jsonl(args.input))
    if report["rows"] != args.expected_rows or report["talk_count"] != args.expected_talks:
        raise ValueError("MCIF lexical triage inventory does not match the contract")
    report = {
        "artifact": "mcif_visual_structure_lexical_triage_v1",
        "status": "MODEL_OUTPUT_LEXICAL_DIAGNOSTIC_NOT_LABELS",
        "input_sha256": actual_hash,
        **report,
        "interpretation": (
            "Keyword matches summarize what the source-only VLM wrote. They do not "
            "verify image content, event eligibility, OCR insufficiency, translation "
            "utility, or pixels-beyond-OCR value and cannot filter the inventory."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
