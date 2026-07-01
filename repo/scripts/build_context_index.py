#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from slidesst.context.retriever import zh_pinyin
from slidesst.data.io import read_jsonl, write_jsonl
from slidesst.data.schema import ChallengeItem, EvidenceItem


def build_evidence(items: list[ChallengeItem], mismatch: str = "matched") -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in items:
        for ev in item.evidence:
            normalized = _normalize_existing_evidence(ev, item)
            _append_unique(evidence, seen, normalized)

        if item.slides.matched_slide_text:
            _append_unique(
                evidence,
                seen,
                EvidenceItem(
                    evidence_id=f"{item.id}:slide:matched",
                    item_id=item.id,
                    lecture_id=item.lecture_id,
                    source_type="slide_ocr",
                    text=item.slides.matched_slide_text,
                    target_hint=_best_slide_hint(item),
                    slide_id=item.slides.matched_slide_id,
                    time_distance_sec=0.0,
                    pinyin=zh_pinyin(item.slides.matched_slide_text),
                    support_label="matched_slide",
                    is_supporting=True,
                ),
            )

        for idx, entry in enumerate(item.glossary):
            _append_unique(
                evidence,
                seen,
                EvidenceItem(
                    evidence_id=f"{item.id}:glossary:{idx}",
                    item_id=item.id,
                    lecture_id=item.lecture_id,
                    source_type="glossary",
                    text=entry.src,
                    target_hint=entry.tgt,
                    pinyin=zh_pinyin(entry.src),
                    support_label=entry.source,
                    is_supporting=entry.source != "distractor",
                    metadata={"desc": entry.desc},
                ),
            )

        for idx, doc in enumerate(item.background_docs):
            _append_unique(
                evidence,
                seen,
                EvidenceItem(
                    evidence_id=f"{item.id}:background:{idx}",
                    item_id=item.id,
                    lecture_id=item.lecture_id,
                    source_type="background",
                    text=doc.text,
                    target_hint=None,
                    pinyin=zh_pinyin(doc.text),
                    support_label=doc.doc_id,
                    is_supporting=None,
                ),
            )

    if mismatch != "matched":
        for ev in _make_mismatch_evidence(items, mismatch):
            _append_unique(evidence, seen, ev)
    return evidence


def _normalize_existing_evidence(ev: EvidenceItem, item: ChallengeItem) -> EvidenceItem:
    data = ev.model_dump(mode="json")
    data["item_id"] = data.get("item_id") or item.id
    data["lecture_id"] = data.get("lecture_id") or item.lecture_id
    data["pinyin"] = data.get("pinyin") or zh_pinyin(data.get("text") or "")
    return EvidenceItem.model_validate(data)


def _append_unique(evidence: list[EvidenceItem], seen: set[str], item: EvidenceItem) -> None:
    if item.evidence_id in seen:
        return
    seen.add(item.evidence_id)
    evidence.append(item)


def _best_slide_hint(item: ChallengeItem) -> str | None:
    if item.ambiguous_items:
        return "; ".join(item.ambiguous_items[0].correct_target)
    return None


def _make_mismatch_evidence(items: list[ChallengeItem], mismatch: str) -> list[EvidenceItem]:
    wrong: list[EvidenceItem] = []
    for idx, item in enumerate(items):
        donor = _choose_donor(items, item, idx, mismatch)
        if donor is None or not donor.slides.matched_slide_text:
            continue
        text = donor.slides.matched_slide_text
        source_type = "wrong_slide"
        if mismatch == "noisy_ocr":
            text = _noisy_text(item.slides.matched_slide_text or text)
            source_type = "slide_ocr"
        wrong.append(
            EvidenceItem(
                evidence_id=f"{item.id}:mismatch:{mismatch}",
                item_id=item.id,
                lecture_id=item.lecture_id,
                source_type=source_type,
                text=text,
                target_hint=_best_slide_hint(donor),
                slide_id=donor.slides.matched_slide_id,
                time_distance_sec=30.0 if mismatch == "next_slide" else -30.0,
                pinyin=zh_pinyin(text),
                support_label=mismatch,
                is_supporting=mismatch == "noisy_ocr",
                metadata={"donor_item_id": donor.id, "mismatch_mode": mismatch},
            )
        )
    return wrong


def _choose_donor(items: list[ChallengeItem], item: ChallengeItem, idx: int, mismatch: str) -> ChallengeItem | None:
    if mismatch == "previous_slide" and idx > 0:
        return items[idx - 1]
    if mismatch == "next_slide" and idx + 1 < len(items):
        return items[idx + 1]
    if mismatch == "random_same_lecture":
        for offset in range(1, len(items)):
            candidate = items[(idx + offset) % len(items)]
            if candidate.id != item.id and candidate.lecture_id == item.lecture_id:
                return candidate
    if mismatch == "noisy_ocr":
        return item
    for candidate in items:
        if candidate.id != item.id:
            return candidate
    return None


def _noisy_text(text: str) -> str:
    chars = [ch for i, ch in enumerate(text) if i % 5 != 0]
    return "".join(chars) or text

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mismatch", default="matched")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    items = read_jsonl(cfg["paths"]["challenge_jsonl"], ChallengeItem)
    evidence = build_evidence(items, mismatch=args.mismatch)
    out = Path(cfg["paths"]["context_index_jsonl"])
    write_jsonl(out, evidence)
    print(f"Wrote {len(evidence)} evidence items to {out}")


if __name__ == "__main__":
    main()
