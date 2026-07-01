#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from slidesst.data.io import read_jsonl, write_jsonl
from slidesst.data.reference_generation import PROMPT_VERSION, generate_reference
from slidesst.data.schema import ChallengeItem, EvidenceItem
from slidesst.translation.adapters import build_translator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/vision_zh_en.yaml")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}
    translation_cfg = cfg.get("translation", {"provider": "mock", "model": "mock-translator"})
    if args.provider:
        translation_cfg["provider"] = args.provider
    if args.model:
        translation_cfg["model"] = args.model
    translation_cfg["prompt_version"] = args.prompt_version
    translator = build_translator(translation_cfg)
    items = read_jsonl(args.input, ChallengeItem)
    if args.offset:
        items = items[args.offset :]
    if args.limit:
        items = items[: args.limit]
    evidence = _load_evidence(cfg)
    evidence_by_item: dict[str, list[EvidenceItem]] = {}
    for ev in evidence:
        evidence_by_item.setdefault(ev.item_id or "", []).append(ev)
    generated = [
        generate_reference(
            item,
            translator,
            evidence_by_item.get(item.id, item.evidence),
            translation_cfg.get("model", "unknown"),
            translation_cfg.get("prompt_version", PROMPT_VERSION),
        )
        for item in items
    ]
    write_jsonl(args.output, generated)
    print(f"Wrote {len(generated)} reference items to {args.output}")


def _load_evidence(cfg: dict) -> list[EvidenceItem]:
    path = cfg.get("paths", {}).get("context_index_jsonl")
    if path and Path(path).exists():
        return read_jsonl(path, EvidenceItem)
    return []


if __name__ == "__main__":
    main()
