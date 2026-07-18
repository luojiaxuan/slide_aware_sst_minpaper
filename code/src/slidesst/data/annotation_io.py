from __future__ import annotations

import csv
from pathlib import Path

from slidesst.data.schema import ChallengeItem, ReferenceInfo


REVIEW_COLUMNS = [
    "id",
    "zh_transcript",
    "visual_summary",
    "ocr_text",
    "candidate_en",
    "corrected_en",
    "reference_status",
    "hard_label_gold_en",
    "hard_label_distractors",
    "requires_visual",
    "hallucination_flag",
    "reviewer_notes",
]


def export_review_sheet(path: str | Path, items: list[ChallengeItem]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "id": item.id,
                    "zh_transcript": item.source_transcript,
                    "visual_summary": item.visual_context.scene_summary if item.visual_context else "",
                    "ocr_text": " | ".join(item.visual_context.ocr_text) if item.visual_context else "",
                    "candidate_en": item.reference.translation or item.reference_translation or "",
                    "corrected_en": "",
                    "reference_status": item.reference.status,
                    "hard_label_gold_en": " | ".join(term for label in item.hard_labels for term in label.gold_en),
                    "hard_label_distractors": " | ".join(term for label in item.hard_labels for term in label.distractor_en),
                    "requires_visual": str(any(label.requires_visual for label in item.hard_labels)).lower(),
                    "hallucination_flag": item.reference.verification_notes or "",
                    "reviewer_notes": "",
                }
            )


def import_review_sheet(items: list[ChallengeItem], review_path: str | Path) -> list[ChallengeItem]:
    by_id = {item.id: item for item in items}
    reviewed: list[ChallengeItem] = []
    with Path(review_path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            item = by_id.get(row["id"])
            if item is None:
                continue
            corrected = row.get("corrected_en") or row.get("candidate_en") or item.reference.translation
            status = row.get("reference_status") or "human_verified"
            if status == "llm_generated":
                status = "human_verified"
            item.reference = ReferenceInfo(
                translation=corrected,
                status=status,
                teacher_models=item.reference.teacher_models,
                prompt_version=item.reference.prompt_version,
                verification_notes=row.get("reviewer_notes") or row.get("hallucination_flag") or None,
                alternatives=item.reference.alternatives,
            )
            item.reference_translation = corrected
            reviewed.append(item)
    return reviewed
