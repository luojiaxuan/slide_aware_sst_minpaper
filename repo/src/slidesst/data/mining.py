from __future__ import annotations

import csv
from pathlib import Path

from slidesst.context.retriever import zh_pinyin
from slidesst.data.io import read_jsonl, write_jsonl
from slidesst.data.schema import AmbiguousItem, Annotation, ChallengeItem


DEFAULT_TERM_SEEDS = {
    "线程": {
        "correct_target": ["thread", "threads"],
        "distractor_targets": ["ready-made", "existing"],
        "category": ["homophone", "technical_term"],
    },
    "现成": {
        "correct_target": ["ready-made", "existing"],
        "distractor_targets": ["thread", "threads"],
        "category": ["homophone"],
    },
    "基音": {
        "correct_target": ["fundamental frequency", "fundamental tone"],
        "distractor_targets": ["gene"],
        "category": ["homophone", "technical_term"],
    },
    "基因": {
        "correct_target": ["gene"],
        "distractor_targets": ["fundamental frequency", "fundamental tone"],
        "category": ["homophone", "technical_term"],
    },
    "卷积": {
        "correct_target": ["convolution"],
        "distractor_targets": ["volume"],
        "category": ["technical_term"],
    },
    "卷积核": {
        "correct_target": ["convolution kernel", "kernel"],
        "distractor_targets": ["volume core"],
        "category": ["technical_term"],
    },
    "消息": {
        "correct_target": ["message", "messages"],
        "distractor_targets": ["stream", "creek"],
        "category": ["homophone", "technical_term"],
    },
    "源域": {
        "correct_target": ["source domain"],
        "distractor_targets": ["original domain"],
        "category": ["technical_term"],
    },
    "法向": {
        "correct_target": ["normal", "normal direction", "surface normal"],
        "distractor_targets": ["dharma appearance"],
        "category": ["homophone", "technical_term"],
    },
    "标识": {
        "correct_target": ["identifier", "identification"],
        "distractor_targets": ["sign", "symbol"],
        "category": ["technical_term"],
    },
}


CSV_COLUMNS = [
    "id",
    "lecture_id",
    "verified",
    "source_transcript",
    "source_token",
    "pinyin",
    "correct_target",
    "distractor_targets",
    "category",
    "reference_translation",
    "slide_text",
    "notes",
]


def mine_hard_examples(items: list[ChallengeItem], limit: int | None = None) -> list[ChallengeItem]:
    mined: list[ChallengeItem] = []
    for item in items:
        ambiguous = list(item.ambiguous_items)
        transcript = item.source_transcript or ""
        for token, spec in DEFAULT_TERM_SEEDS.items():
            if token not in transcript:
                continue
            if any(existing.source_token == token or token in existing.source_token or existing.source_token in token for existing in ambiguous):
                continue
            ambiguous.append(
                AmbiguousItem(
                    source_token=token,
                    pinyin=zh_pinyin(token),
                    correct_target=spec["correct_target"],
                    distractor_targets=spec["distractor_targets"],
                    category=spec["category"],
                )
            )
        if ambiguous:
            item.ambiguous_items = ambiguous
            mined.append(item)
            if limit and len(mined) >= limit:
                break
    return mined


def write_annotation_csv(path: str | Path, items: list[ChallengeItem]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for item in items:
            for amb in item.ambiguous_items:
                writer.writerow(
                    {
                        "id": item.id,
                        "lecture_id": item.lecture_id,
                        "verified": str(item.annotation.verified).lower(),
                        "source_transcript": item.source_transcript,
                        "source_token": amb.source_token,
                        "pinyin": amb.pinyin,
                        "correct_target": "|".join(amb.correct_target),
                        "distractor_targets": "|".join(amb.distractor_targets),
                        "category": "|".join(amb.category),
                        "reference_translation": item.reference_translation or "",
                        "slide_text": item.slides.matched_slide_text or "",
                        "notes": item.annotation.notes or "",
                    }
                )


def build_verified_items(candidate_jsonl: str | Path, annotation_csv: str | Path) -> list[ChallengeItem]:
    candidates = {item.id: item for item in read_jsonl(candidate_jsonl, ChallengeItem)}
    rows_by_id: dict[str, list[dict[str, str]]] = {}
    with Path(annotation_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if _truthy(row.get("verified")):
                rows_by_id.setdefault(row["id"], []).append(row)

    verified: list[ChallengeItem] = []
    for item_id, rows in rows_by_id.items():
        item = candidates[item_id]
        item.ambiguous_items = [
            AmbiguousItem(
                source_token=row["source_token"],
                pinyin=row.get("pinyin") or zh_pinyin(row["source_token"]),
                correct_target=_split_terms(row.get("correct_target")),
                distractor_targets=_split_terms(row.get("distractor_targets")),
                category=_split_terms(row.get("category")),
            )
            for row in rows
        ]
        first = rows[0]
        if first.get("reference_translation"):
            item.reference_translation = first["reference_translation"]
        item.annotation = Annotation(verified=True, annotator=first.get("annotator") or None, notes=first.get("notes") or None)
        verified.append(item)
    return verified


def copy_items(path: str | Path, items: list[ChallengeItem]) -> None:
    write_jsonl(path, items)


def _split_terms(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.replace(";", "|").split("|") if part.strip()]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "verified"}
