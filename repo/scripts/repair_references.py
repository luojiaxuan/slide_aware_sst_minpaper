#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm
import yaml

from slidesst.data.io import read_jsonl, write_jsonl
from slidesst.data.reference_audit import audit_reference_item
from slidesst.data.reference_generation import PROMPT_VERSION, attach_reference
from slidesst.data.schema import ChallengeItem
from slidesst.translation.adapters import build_translator


DEFAULT_TARGET_FLAGS = ("target_cjk_chars", "length_ratio_high")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/vision_zh_en.yaml")
    parser.add_argument("--prompt-version", default=f"{PROMPT_VERSION}_repair_no_cjk")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-flag", action="append", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}
    translation_cfg = cfg.get("translation", {"provider": "mock", "model": "mock-translator"})
    batch_size = args.batch_size or int(translation_cfg.get("batch_size", 1))
    target_flags = tuple(args.target_flag or DEFAULT_TARGET_FLAGS)
    translator = build_translator(translation_cfg)
    complete_prompts = getattr(translator, "complete_prompts", None)
    if not callable(complete_prompts):
        raise RuntimeError("Reference repair requires a translator with complete_prompts support")

    items = read_jsonl(args.input, ChallengeItem)
    repair_indices = [
        index
        for index, item in enumerate(items)
        if _needs_repair(audit_reference_item(item).flags, target_flags)
    ]
    repaired = 0
    for index_batch in tqdm(list(_chunks(repair_indices, batch_size)), desc="Repairing reference batches"):
        prompts = [_repair_prompt(items[index]) for index in index_batch]
        outputs = complete_prompts(prompts)
        for index, text in zip(index_batch, outputs, strict=True):
            item = items[index]
            teacher_models = list(item.reference.teacher_models)
            attach_reference(
                item,
                text,
                item.evidence,
                translation_cfg.get("model", "unknown"),
                args.prompt_version,
            )
            item.reference.teacher_models = _dedupe(teacher_models + item.reference.teacher_models)
            repaired += 1

    write_jsonl(args.output, items)
    print(json.dumps({"n": len(items), "repaired": repaired, "target_flags": target_flags}, ensure_ascii=False, indent=2))


def _needs_repair(flags: list[str], target_flags: tuple[str, ...]) -> bool:
    return any(_flag_name(flag) in target_flags for flag in flags)


def _flag_name(flag: str) -> str:
    for sep in ("=", ":"):
        if sep in flag:
            return flag.split(sep, 1)[0]
    return flag


def _repair_prompt(item: ChallengeItem) -> str:
    draft = item.reference.translation or item.reference_translation or ""
    return f"""Repair an English reference translation for Chinese-to-English speech translation.
Use the Chinese source transcript as the authority and the current draft only as a draft.
Rewrite as one fluent, concise English translation of only the spoken source transcript.
Fix any Chinese characters, half-translated terms, or explanatory text in the draft.
Keep proper names in Latin letters when appropriate.

Chinese source transcript:
{item.source_transcript}

Current English draft:
{draft}

Return only the corrected English translation:"""


def _chunks(items: list[int], batch_size: int) -> Iterator[list[int]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


if __name__ == "__main__":
    main()
