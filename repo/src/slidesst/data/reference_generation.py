from __future__ import annotations

import re

from slidesst.data.schema import ChallengeItem, EvidenceItem, ReferenceInfo
from slidesst.streaming.simulator import StreamState
from slidesst.translation.adapters import Translator


PROMPT_VERSION = "v1_visual_faithful"


def generate_reference(item: ChallengeItem, translator: Translator, evidence: list[EvidenceItem], model_name: str) -> ChallengeItem:
    state = StreamState(
        item_id=item.id,
        t=item.video.end_sec if item.video else 0.0,
        partial_transcript=item.source_transcript,
        video_time_sec=item.video.end_sec if item.video else None,
    )
    result = translator.translate(state, evidence, "reference_generation")
    item.reference = ReferenceInfo(
        translation=result.text,
        status="llm_generated",
        teacher_models=[model_name],
        prompt_version=PROMPT_VERSION,
        verification_notes="; ".join(run_reference_checks(item, result.text, evidence)),
    )
    item.reference_translation = result.text
    return item


def run_reference_checks(item: ChallengeItem, translation: str, evidence: list[EvidenceItem]) -> list[str]:
    flags: list[str] = []
    if item.source_transcript and translation:
        ratio = len(translation) / max(1, len(item.source_transcript))
        if ratio < 0.2 or ratio > 8.0:
            flags.append(f"length_ratio={ratio:.2f}")
    src_numbers = set(re.findall(r"\d+(?:\.\d+)?", item.source_transcript))
    hyp_numbers = set(re.findall(r"\d+(?:\.\d+)?", translation))
    missing_numbers = sorted(src_numbers - hyp_numbers)
    if missing_numbers:
        flags.append("missing_numbers=" + ",".join(missing_numbers))
    for label in item.hard_labels:
        if label.gold_en and not _contains_any(translation, label.gold_en):
            flags.append(f"missing_hard_label={label.label_id}")
    visual_only_terms = []
    for ev in evidence:
        if ev.visual_only and ev.target_hint:
            visual_only_terms.extend(_split_hint(ev.target_hint))
    for term in visual_only_terms:
        if term and _contains_any(translation, [term]):
            flags.append(f"possible_visual_hallucination={term}")
    return flags


def _contains_any(text: str, terms: list[str]) -> bool:
    norm = text.lower()
    return any(term.lower() in norm for term in terms if term)


def _split_hint(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;|,]", text) if part.strip()]
