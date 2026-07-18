from __future__ import annotations

from dataclasses import dataclass

from slidesst.data.schema import ChallengeItem, ModelOutput
from slidesst.eval.metrics import contains_any


@dataclass
class VisualMetrics:
    vga: float
    dra: float
    oaa: float
    oga: float
    vhr: float
    wvar: float
    evidence_precision: float
    evidence_recall: float
    visual_label_n: int


def evaluate_visual_cases(items: list[ChallengeItem], outputs: list[ModelOutput]) -> VisualMetrics:
    by_id = {out.id: out for out in outputs}
    visual_hits = visual_total = 0
    deixis_hits = deixis_total = 0
    object_action_hits = object_action_total = 0
    ocr_hits = ocr_total = 0
    hallucination_hits = hallucination_total = 0

    selected_supporting = selected_total = supporting_total = 0

    for item in items:
        out = by_id.get(item.id)
        if out is None:
            continue
        selected_ids = set(out.used_evidence_ids)
        supporting_ids = {eid for label in item.hard_labels for eid in label.supporting_evidence_ids}
        selected_total += len(selected_ids)
        selected_supporting += len(selected_ids & supporting_ids)
        supporting_total += len(supporting_ids)

        for label in item.hard_labels:
            ok = contains_any(out.hypothesis, label.gold_en) and not contains_any(out.hypothesis, label.distractor_en)
            if label.requires_visual:
                visual_total += 1
                visual_hits += int(ok)
            if label.label_type == "visual_deixis":
                deixis_total += 1
                deixis_hits += int(ok)
            if label.label_type in {"object", "action"}:
                object_action_total += 1
                object_action_hits += int(ok)
            if label.requires_ocr:
                ocr_total += 1
                ocr_hits += int(ok)
            if label.unspoken_visual_distractors:
                hallucination_total += 1
                hallucination_hits += int(contains_any(out.hypothesis, label.unspoken_visual_distractors))

    return VisualMetrics(
        vga=_rate(visual_hits, visual_total),
        dra=_rate(deixis_hits, deixis_total),
        oaa=_rate(object_action_hits, object_action_total),
        oga=_rate(ocr_hits, ocr_total),
        vhr=_rate(hallucination_hits, hallucination_total),
        wvar=_wrong_visual_adoption_rate(outputs),
        evidence_precision=_rate(selected_supporting, selected_total),
        evidence_recall=_rate(selected_supporting, supporting_total),
        visual_label_n=visual_total,
    )


def _wrong_visual_adoption_rate(outputs: list[ModelOutput]) -> float:
    total = 0
    hits = 0
    for out in outputs:
        wrong_terms = []
        for ev in out.evidence_packet:
            if ev.source_type in {"wrong_video", "wrong_clip", "negative_visual"}:
                total += 1
                if ev.target_hint:
                    wrong_terms.extend(_split_hint(ev.target_hint))
                wrong_terms.append(ev.text)
        if wrong_terms and contains_any(out.hypothesis, wrong_terms):
            hits += 1
    return _rate(hits, total)


def _split_hint(text: str) -> list[str]:
    return [part.strip() for part in text.replace("|", ";").split(";") if part.strip()]


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0
