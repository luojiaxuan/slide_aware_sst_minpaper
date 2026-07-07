#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Iterator
from tqdm import tqdm
import yaml

from slidesst.data.io import read_jsonl, write_jsonl
from slidesst.data.reference_generation import PROMPT_VERSION, attach_reference, generate_reference
from slidesst.data.schema import ChallengeItem, EvidenceItem
from slidesst.streaming.simulator import StreamState
from slidesst.translation.adapters import build_translator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/vision_zh_en.yaml")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}
    translation_cfg = _translation_config(cfg, args.provider, args.model, args.prompt_version)
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
    batch_size = args.batch_size or int(translation_cfg.get("batch_size", 1))
    batch_translate = getattr(translator, "translate_batch", None)
    if batch_size > 1 and callable(batch_translate):
        generated = _generate_batched(
            items,
            batch_translate,
            evidence_by_item,
            translation_cfg.get("model", "unknown"),
            translation_cfg.get("prompt_version", PROMPT_VERSION),
            batch_size,
        )
    else:
        generated = (
            generate_reference(
                item,
                translator,
                evidence_by_item.get(item.id, item.evidence),
                translation_cfg.get("model", "unknown"),
                translation_cfg.get("prompt_version", PROMPT_VERSION),
            )
            for item in tqdm(items, desc="Generating references")
        )
    write_jsonl(args.output, generated)
    print(f"Wrote {len(items)} reference items to {args.output}")


def _load_evidence(cfg: dict) -> list[EvidenceItem]:
    path = cfg.get("paths", {}).get("context_index_jsonl")
    if path and Path(path).exists():
        return read_jsonl(path, EvidenceItem)
    return []


def _translation_config(cfg: dict, provider: str | None, model: str | None, prompt_version: str | None) -> dict:
    translation_cfg = dict(cfg.get("translation", {"provider": "mock", "model": "mock-translator"}))
    if provider:
        translation_cfg["provider"] = provider
    if model:
        translation_cfg["model"] = model
    if prompt_version:
        translation_cfg["prompt_version"] = prompt_version
    return translation_cfg


def _generate_batched(
    items: list[ChallengeItem],
    batch_translate,
    evidence_by_item: dict[str, list[EvidenceItem]],
    model_name: str,
    prompt_version: str,
    batch_size: int,
) -> Iterator[ChallengeItem]:
    total = (len(items) + batch_size - 1) // batch_size
    for batch in tqdm(_chunks(items, batch_size), total=total, desc="Generating reference batches"):
        evidence_packets = [evidence_by_item.get(item.id, item.evidence) for item in batch]
        states = [_reference_state(item) for item in batch]
        outputs = batch_translate(states, evidence_packets, "reference_generation")
        for item, result, evidence in zip(batch, outputs, evidence_packets, strict=True):
            yield attach_reference(item, result.text, evidence, model_name, prompt_version)


def _chunks(items: list[ChallengeItem], batch_size: int) -> Iterator[list[ChallengeItem]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _reference_state(item: ChallengeItem) -> StreamState:
    return StreamState(
        item_id=item.id,
        t=item.video.end_sec if item.video else 0.0,
        partial_transcript=item.source_transcript,
        video_time_sec=item.video.end_sec if item.video else None,
    )


if __name__ == "__main__":
    main()
