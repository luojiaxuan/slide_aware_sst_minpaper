#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from slidesst.data.io import read_jsonl
from slidesst.data.schema import ChallengeItem, EvidenceItem
from slidesst.streaming.simulator import StreamState
from slidesst.translation.adapters import build_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--evidence-jsonl", default=None)
    parser.add_argument("--out-sft", required=True)
    parser.add_argument("--out-rejected", required=True)
    parser.add_argument("--accepted-severities", nargs="+", default=["pass"])
    parser.add_argument("--condition", default="reference_generation")
    args = parser.parse_args()

    items = read_jsonl(args.references, ChallengeItem)
    audit = _read_audit(args.audit_csv)
    evidence_by_item = _read_evidence(args.evidence_jsonl)
    accepted = set(args.accepted_severities)

    sft_rows = []
    rejected_rows = []
    for item in items:
        row = audit.get(item.id, {})
        severity = row.get("severity", "unknown")
        flags = row.get("flags", "")
        evidence = evidence_by_item.get(item.id, item.evidence)
        prompt = build_prompt(_final_state(item), evidence, args.condition)
        translation = item.reference.translation or item.reference_translation or ""
        metadata = {
            "id": item.id,
            "audit_severity": severity,
            "audit_flags": flags,
            "teacher_models": item.reference.teacher_models,
            "prompt_version": item.reference.prompt_version,
        }
        if severity in accepted:
            sft_rows.append(
                {
                    "id": item.id,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": translation},
                    ],
                    "metadata": metadata,
                }
            )
        else:
            rejected_rows.append(
                {
                    "id": item.id,
                    "prompt": prompt,
                    "rejected": translation,
                    "source_transcript": item.source_transcript,
                    "metadata": metadata,
                }
            )

    _write_jsonl(args.out_sft, sft_rows)
    _write_jsonl(args.out_rejected, rejected_rows)
    print(f"Wrote {len(sft_rows)} SFT rows to {args.out_sft}")
    print(f"Wrote {len(rejected_rows)} rejected/review rows to {args.out_rejected}")


def _read_audit(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def _read_evidence(path: str | None) -> dict[str, list[EvidenceItem]]:
    if not path:
        return {}
    evidence_by_item: dict[str, list[EvidenceItem]] = {}
    for ev in read_jsonl(path, EvidenceItem):
        evidence_by_item.setdefault(ev.item_id or "", []).append(ev)
    return evidence_by_item


def _final_state(item: ChallengeItem) -> StreamState:
    return StreamState(
        item_id=item.id,
        t=item.video.end_sec if item.video else 0.0,
        partial_transcript=item.source_transcript,
        video_time_sec=item.video.end_sec if item.video else None,
    )


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
