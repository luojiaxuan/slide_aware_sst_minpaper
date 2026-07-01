from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

try:
    from pypinyin import lazy_pinyin  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal smoke tests
    def lazy_pinyin(text: str, errors: str = "ignore") -> list[str]:
        return list(text)

try:
    import jieba  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal smoke tests
    jieba = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal smoke tests
    class BM25Okapi:  # type: ignore
        def __init__(self, corpus: list[list[str]]):
            self.corpus = corpus
        def get_scores(self, query_tokens: list[str]) -> list[float]:
            q = set(query_tokens)
            scores = []
            for doc in self.corpus:
                d = set(doc)
                scores.append(float(len(q & d)) / max(1, len(q | d)))
            return scores

from slidesst.data.schema import EvidenceItem


def zh_pinyin(text: str) -> str:
    return " ".join(lazy_pinyin(text, errors="ignore"))


@dataclass
class RetrievedEvidence:
    item: EvidenceItem
    score: float
    features: dict


class EvidenceRetriever:
    """BM25 + pinyin + temporal-prior retriever.

    Future versions can add embedding retrieval for background documents.
    """

    def __init__(self, evidence: List[EvidenceItem], weights: dict | None = None):
        self.evidence = evidence
        self.weights = weights or {"bm25": 0.35, "pinyin": 0.35, "temporal": 0.20, "source_trust": 0.10}
        tokenized = [self._tokenize(e.text + " " + (e.target_hint or "")) for e in evidence]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def retrieve(
        self,
        query: str,
        current_slide_id: str | None = None,
        top_m: int = 20,
        *,
        current_time_sec: float | None = None,
        visual_availability: str = "offline_context",
        allowed_lookahead_sec: float = 0.0,
    ) -> list[RetrievedEvidence]:
        if not self.evidence:
            return []
        q_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens) if self.bm25 is not None else [0.0] * len(self.evidence)
        q_pinyin = zh_pinyin(query)
        results: list[RetrievedEvidence] = []
        for item, bm25_score in zip(self.evidence, bm25_scores):
            if not self._is_available(item, current_time_sec, visual_availability, allowed_lookahead_sec):
                continue
            evidence_pinyin = item.pinyin or zh_pinyin(item.text)
            pinyin_score = self._pinyin_overlap(q_pinyin, evidence_pinyin)
            temporal_score = self._temporal_score(item, current_slide_id, current_time_sec)
            source_trust = self._source_trust(item.source_type)
            anchor_score = self._anchor_score(query, item)
            confidence_score = item.confidence if item.confidence is not None else 0.0
            score = (
                self.weights.get("bm25", 0.0) * float(bm25_score)
                + self.weights.get("pinyin", 0.0) * pinyin_score
                + self.weights.get("temporal", 0.0) * temporal_score
                + self.weights.get("source_trust", 0.0) * source_trust
                + self.weights.get("anchor", 0.0) * anchor_score
                + self.weights.get("confidence", 0.0) * confidence_score
            )
            results.append(RetrievedEvidence(item, score, {
                "bm25": float(bm25_score),
                "pinyin": pinyin_score,
                "temporal": temporal_score,
                "source_trust": source_trust,
                "anchor": anchor_score,
                "confidence": confidence_score,
            }))
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_m]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower().replace("/", " ")
        ascii_tokens = re.findall(r"[a-z0-9_+-]+", text)
        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        zh_text = "".join(zh_chars)
        zh_tokens: list[str] = []
        if zh_text:
            if jieba is not None:
                zh_tokens.extend(tok for tok in jieba.lcut(zh_text) if tok.strip())
            zh_tokens.extend(zh_chars)
            zh_tokens.extend(zh_text[i : i + 2] for i in range(max(0, len(zh_text) - 1)))
        return [tok for tok in [*ascii_tokens, *zh_tokens] if tok]

    @staticmethod
    def _pinyin_overlap(a: str, b: str) -> float:
        aset, bset = set(a.split()), set(b.split())
        if not aset or not bset:
            return 0.0
        return len(aset & bset) / max(1, len(aset | bset))

    @staticmethod
    def _source_trust(source_type: str) -> float:
        return {
            "glossary": 0.95,
            "slide_ocr": 0.85,
            "slide_vlm": 0.70,
            "video_ocr": 0.82,
            "video_object": 0.72,
            "video_action": 0.72,
            "video_spatial": 0.64,
            "video_scene": 0.52,
            "video_vlm_frame": 0.68,
            "video_vlm_clip": 0.62,
            "background": 0.55,
            "history": 0.60,
            "wrong_slide": 0.20,
            "wrong_video": 0.10,
            "wrong_clip": 0.10,
            "negative_visual": 0.05,
        }.get(source_type, 0.50)

    @staticmethod
    def _temporal_score(item: EvidenceItem, current_slide_id: str | None, current_time_sec: float | None) -> float:
        if current_slide_id and item.slide_id == current_slide_id:
            return 1.0
        if current_time_sec is None or item.temporal_start_sec is None:
            return 0.0
        distance = abs(item.temporal_start_sec - current_time_sec)
        return max(0.0, 1.0 - min(distance / 30.0, 1.0))

    @staticmethod
    def _anchor_score(query: str, item: EvidenceItem) -> float:
        if item.spoken_anchor and item.spoken_anchor in query:
            return 1.0
        if item.target_hint and item.target_hint.lower() in query.lower():
            return 0.7
        if item.spoken_anchor:
            return EvidenceRetriever._pinyin_overlap(zh_pinyin(query), zh_pinyin(item.spoken_anchor))
        return 0.0

    @staticmethod
    def _is_available(item: EvidenceItem, current_time_sec: float | None, visual_availability: str, allowed_lookahead_sec: float) -> bool:
        if visual_availability == "offline_context":
            return True
        if item.modality not in {"image", "video", "mixed"}:
            return True
        if current_time_sec is None or item.temporal_start_sec is None:
            return True
        return item.temporal_start_sec <= current_time_sec + allowed_lookahead_sec
