from __future__ import annotations

import re
from dataclasses import dataclass

from slidesst.data.schema import ChallengeItem, ModelOutput


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_any(text: str, terms: list[str]) -> bool:
    text_n = normalize(text)
    return any(normalize(t) in text_n for t in terms if t)


@dataclass
class HardCaseMetrics:
    hda: float
    term_precision: float
    term_recall: float
    term_f1: float
    context_overuse_rate: float
    wrong_slide_adoption_rate: float
    bleu: float
    avg_wall_time_sec: float
    n: int


def evaluate_hard_cases(items: list[ChallengeItem], outputs: list[ModelOutput], aliases: dict[str, list[str]] | None = None) -> HardCaseMetrics:
    aliases = aliases or {}
    by_id = {o.id: o for o in outputs}
    hda_hits = 0
    hda_total = 0
    term_hits = 0
    term_total = 0
    overuse_hits = 0
    overuse_total = 0

    for item in items:
        out = by_id.get(item.id)
        if out is None:
            continue
        hyp = out.hypothesis
        for amb in item.ambiguous_items:
            hda_total += 1
            correct_terms = _expand_aliases(amb.correct_target, aliases)
            distractor_terms = _expand_aliases(amb.distractor_targets, aliases)
            if contains_any(hyp, correct_terms):
                hda_hits += 1
            term_total += 1
            if contains_any(hyp, correct_terms):
                term_hits += 1
            unsupported_terms = [*distractor_terms, *_unsupported_evidence_terms(item, out)]
            if contains_any(hyp, unsupported_terms):
                overuse_hits += 1
            overuse_total += 1
    precision_den = term_hits + overuse_hits
    term_precision = term_hits / precision_den if precision_den else 0.0
    term_recall = term_hits / term_total if term_total else 0.0
    term_f1 = 2 * term_precision * term_recall / (term_precision + term_recall) if term_precision + term_recall else 0.0
    return HardCaseMetrics(
        hda=hda_hits / hda_total if hda_total else 0.0,
        term_precision=term_precision,
        term_recall=term_recall,
        term_f1=term_f1,
        context_overuse_rate=overuse_hits / overuse_total if overuse_total else 0.0,
        wrong_slide_adoption_rate=_wrong_slide_adoption_rate(outputs),
        bleu=_corpus_bleu(items, outputs),
        avg_wall_time_sec=_avg_wall_time(outputs),
        n=hda_total,
    )


def _expand_aliases(terms: list[str], aliases: dict[str, list[str]]) -> list[str]:
    expanded = list(terms)
    for term in terms:
        expanded.extend(aliases.get(term, []))
    return expanded


def _unsupported_evidence_terms(item: ChallengeItem, out: ModelOutput) -> list[str]:
    evidence = [*item.evidence, *out.evidence_packet]
    terms: list[str] = []
    for ev in evidence:
        if ev.is_supporting is False:
            if ev.target_hint:
                terms.append(ev.target_hint)
            terms.append(ev.text)
    return terms


def _wrong_slide_adoption_rate(outputs: list[ModelOutput]) -> float:
    total = 0
    hits = 0
    for out in outputs:
        wrong_terms = []
        for ev in out.evidence_packet:
            if ev.source_type == "wrong_slide" or ev.support_label in {"previous_slide", "next_slide", "random_same_lecture"}:
                total += 1
                if ev.target_hint:
                    wrong_terms.append(ev.target_hint)
                wrong_terms.append(ev.text)
        if wrong_terms and contains_any(out.hypothesis, wrong_terms):
            hits += 1
    return hits / total if total else 0.0


def _corpus_bleu(items: list[ChallengeItem], outputs: list[ModelOutput]) -> float:
    by_id = {o.id: o for o in outputs}
    hyps = []
    refs = []
    for item in items:
        out = by_id.get(item.id)
        if out and item.reference_translation:
            hyps.append(out.hypothesis)
            refs.append(item.reference_translation)
    if not hyps:
        return 0.0
    try:
        import sacrebleu  # type: ignore
    except Exception:
        return 0.0
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)


def _avg_wall_time(outputs: list[ModelOutput]) -> float:
    values = [out.latency.get("wall_time_sec", 0.0) for out in outputs]
    return sum(values) / len(values) if values else 0.0
