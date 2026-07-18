#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from slidesst.data.io import read_jsonl
from slidesst.data.schema import ChallengeItem, EvidenceItem, ModelOutput


DEFAULT_CONDITIONS = (
    "V0_no_context",
    "V2_ocr_only",
    "V3_visual_caption_only",
    "V4_ocr_plus_visual",
    "V5_naive_all_visual",
    "V6_policy_visual",
    "V8_wrong_visual",
)

BASE_COLUMNS = [
    "id",
    "lecture_id",
    "zh_transcript",
    "candidate_reference_en",
    "reference_audit_severity",
    "reference_audit_flags",
    "visual_summary",
    "ocr_text",
    "v4_evidence_packet",
    "v6_evidence_packet",
    "human_reference_en",
    "reference_quality",
    "requires_visual",
    "requires_ocr",
    "supporting_evidence_ids",
    "hallucination_conditions",
    "reviewer_notes",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", nargs="*", default=list(DEFAULT_CONDITIONS))
    args = parser.parse_args()

    items = read_jsonl(args.input, ChallengeItem)
    audit = _read_audit(Path(args.audit_csv))
    outputs = _read_outputs(Path(args.run_dir), args.conditions)
    export_sheet(Path(args.output), items, audit, outputs, args.conditions)
    print(f"Wrote {len(items)} diagnostic review rows to {args.output}")


def export_sheet(
    path: Path,
    items: list[ChallengeItem],
    audit: dict[str, dict[str, str]],
    outputs: dict[str, dict[str, ModelOutput]],
    conditions: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*BASE_COLUMNS, *[f"hyp_{condition}" for condition in conditions]]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in items:
            row = _base_row(item, audit.get(item.id, {}), outputs)
            for condition in conditions:
                out = outputs.get(condition, {}).get(item.id)
                row[f"hyp_{condition}"] = out.hypothesis if out else ""
            writer.writerow(row)


def _base_row(
    item: ChallengeItem,
    audit_row: dict[str, str],
    outputs: dict[str, dict[str, ModelOutput]],
) -> dict[str, str]:
    visual = item.visual_context
    return {
        "id": item.id,
        "lecture_id": item.lecture_id,
        "zh_transcript": item.source_transcript,
        "candidate_reference_en": item.reference.translation or item.reference_translation or "",
        "reference_audit_severity": audit_row.get("severity", ""),
        "reference_audit_flags": audit_row.get("flags", ""),
        "visual_summary": visual.scene_summary if visual else "",
        "ocr_text": " | ".join(visual.ocr_text) if visual else "",
        "v4_evidence_packet": _packet_brief(outputs, "V4_ocr_plus_visual", item.id),
        "v6_evidence_packet": _packet_brief(outputs, "V6_policy_visual", item.id),
        "human_reference_en": "",
        "reference_quality": "",
        "requires_visual": "",
        "requires_ocr": "",
        "supporting_evidence_ids": "",
        "hallucination_conditions": "",
        "reviewer_notes": "",
    }


def _packet_brief(outputs: dict[str, dict[str, ModelOutput]], condition: str, item_id: str) -> str:
    out = outputs.get(condition, {}).get(item_id)
    if not out:
        return ""
    return " || ".join(_evidence_brief(ev) for ev in out.evidence_packet)


def _evidence_brief(ev: EvidenceItem) -> str:
    text = " ".join(ev.text.split())
    if len(text) > 160:
        text = f"{text[:157]}..."
    return f"{ev.evidence_id} [{ev.source_type}]: {text}"


def _read_audit(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def _read_outputs(run_dir: Path, conditions: list[str]) -> dict[str, dict[str, ModelOutput]]:
    outputs: dict[str, dict[str, ModelOutput]] = {}
    for condition in conditions:
        path = run_dir / "matched" / condition / "outputs.jsonl"
        if not path.exists():
            path = run_dir / condition / "outputs.jsonl"
        if path.exists():
            outputs[condition] = {out.id: out for out in read_jsonl(path, ModelOutput)}
    return outputs


if __name__ == "__main__":
    main()
