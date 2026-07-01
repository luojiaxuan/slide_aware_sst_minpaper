#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from slidesst.data.io import read_jsonl, write_jsonl
from slidesst.data.schema import ChallengeItem, EvidenceItem
from slidesst.vision.evidence_builder import build_visual_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mismatch", default="matched")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    input_path = Path(args.input or _first_existing(cfg["paths"].get("challenge_jsonl"), cfg["paths"].get("candidate_jsonl")))
    output_path = Path(args.output or cfg["paths"]["context_index_jsonl"])
    items = read_jsonl(input_path, ChallengeItem)
    evidence: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in items:
        for ev in [*item.evidence, *build_visual_evidence(item, mismatch=args.mismatch)]:
            if ev.evidence_id in seen:
                continue
            seen.add(ev.evidence_id)
            evidence.append(ev)
    write_jsonl(output_path, evidence)
    print(f"Wrote {len(evidence)} visual evidence items to {output_path}")


def _first_existing(*paths: str | None) -> str:
    for path in paths:
        if path and Path(path).exists():
            return path
    for path in paths:
        if path:
            return path
    raise FileNotFoundError("No input path configured")


if __name__ == "__main__":
    main()
