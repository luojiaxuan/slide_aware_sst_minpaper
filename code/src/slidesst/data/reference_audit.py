from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from slidesst.data.schema import ChallengeItem


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
PLACEHOLDER_PHRASES = ("say something",)
EVIDENCE_SOURCE_PHRASES = (
    "visual evidence",
    "ocr",
    "on-screen text",
    "onscreen text",
    "slide shows",
    "slide says",
    "the slide",
)


@dataclass(frozen=True)
class ReferenceAudit:
    item_id: str
    severity: str
    flags: list[str]
    source_chars: int
    candidate_chars: int
    target_cjk_chars: int
    length_ratio: float


def audit_reference_item(
    item: ChallengeItem,
    *,
    review_length_ratio: float = 5.0,
    reject_length_ratio: float = 8.0,
) -> ReferenceAudit:
    candidate = (item.reference.translation or item.reference_translation or "").strip()
    source = item.source_transcript.strip()
    flags: list[str] = []
    if not candidate:
        flags.append("empty_translation")

    target_cjk_chars = len(CJK_RE.findall(candidate))
    if target_cjk_chars:
        flags.append(f"target_cjk_chars={target_cjk_chars}")

    lower_candidate = candidate.lower()
    for phrase in PLACEHOLDER_PHRASES:
        if phrase in lower_candidate:
            flags.append(f"visual_placeholder={phrase}")
    evidence_source_phrase = next((phrase for phrase in EVIDENCE_SOURCE_PHRASES if phrase in lower_candidate), None)
    if evidence_source_phrase:
        flags.append(f"evidence_source_mention={evidence_source_phrase}")

    for text in _visual_texts(item):
        if len(text) >= 2 and _contains_visual_text(candidate, text):
            flags.append(f"copied_visual_text={_shorten(text)}")

    source_chars = len(source)
    candidate_chars = len(candidate)
    length_ratio = candidate_chars / max(1, source_chars)
    if length_ratio >= reject_length_ratio:
        flags.append(f"length_ratio_high={length_ratio:.2f}")
    elif length_ratio >= review_length_ratio:
        flags.append(f"length_ratio_review={length_ratio:.2f}")

    severity = _severity(flags)
    return ReferenceAudit(
        item_id=item.id,
        severity=severity,
        flags=flags,
        source_chars=source_chars,
        candidate_chars=candidate_chars,
        target_cjk_chars=target_cjk_chars,
        length_ratio=length_ratio,
    )


def audit_reference_items(items: list[ChallengeItem]) -> list[ReferenceAudit]:
    return [audit_reference_item(item) for item in items]


def write_reference_audit_csv(path: str | Path, items: list[ChallengeItem], audits: list[ReferenceAudit]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit_by_id = {audit.item_id: audit for audit in audits}
    fieldnames = [
        "id",
        "severity",
        "flags",
        "source_chars",
        "candidate_chars",
        "length_ratio",
        "target_cjk_chars",
        "zh_transcript",
        "candidate_en",
        "ocr_text",
        "visual_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in items:
            audit = audit_by_id[item.id]
            writer.writerow(
                {
                    "id": item.id,
                    "severity": audit.severity,
                    "flags": " | ".join(audit.flags),
                    "source_chars": audit.source_chars,
                    "candidate_chars": audit.candidate_chars,
                    "length_ratio": f"{audit.length_ratio:.2f}",
                    "target_cjk_chars": audit.target_cjk_chars,
                    "zh_transcript": item.source_transcript,
                    "candidate_en": item.reference.translation or item.reference_translation or "",
                    "ocr_text": " | ".join(item.visual_context.ocr_text) if item.visual_context else "",
                    "visual_summary": item.visual_context.scene_summary if item.visual_context else "",
                }
            )


def write_reference_audit_summary(path: str | Path, audits: list[ReferenceAudit]) -> dict:
    summary = summarize_reference_audits(audits)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def summarize_reference_audits(audits: list[ReferenceAudit]) -> dict:
    severity_counts = Counter(audit.severity for audit in audits)
    flag_counts: Counter[str] = Counter()
    for audit in audits:
        for flag in audit.flags:
            flag_counts[_flag_name(flag)] += 1
    return {
        "n": len(audits),
        "severity_counts": dict(sorted(severity_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
    }


def _severity(flags: list[str]) -> str:
    reject_prefixes = (
        "empty_translation",
        "visual_placeholder=",
        "length_ratio_high=",
    )
    if any(flag.startswith(reject_prefixes) for flag in flags):
        return "reject"
    review_prefixes = (
        "target_cjk_chars=",
        "evidence_source_mention=",
        "copied_visual_text=",
        "length_ratio_review=",
    )
    if any(flag.startswith(review_prefixes) for flag in flags):
        return "review"
    return "pass"


def _visual_texts(item: ChallengeItem) -> list[str]:
    texts: list[str] = []
    if item.visual_context:
        texts.extend(item.visual_context.ocr_text)
        texts.extend(item.visual_context.objects)
        texts.extend(item.visual_context.actions)
        texts.extend(item.visual_context.spatial_relations)
    for label in item.hard_labels:
        texts.extend(label.gold_en)
        texts.extend(label.unspoken_visual_distractors)
    unique: list[str] = []
    seen: set[str] = set()
    for text in texts:
        text = text.strip() if text else ""
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def _shorten(text: str, limit: int = 32) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _contains_visual_text(candidate: str, visual_text: str) -> bool:
    text = re.sub(r"\s+", " ", visual_text).strip()
    if not text:
        return False
    if CJK_RE.search(text):
        return text in candidate
    normalized_candidate = re.sub(r"\s+", " ", candidate.lower())
    normalized_text = text.lower()
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_text)}(?![a-z0-9])"
    return re.search(pattern, normalized_candidate) is not None


def _flag_name(flag: str) -> str:
    for sep in ("=", ":"):
        if sep in flag:
            return flag.split(sep, 1)[0]
    return flag
