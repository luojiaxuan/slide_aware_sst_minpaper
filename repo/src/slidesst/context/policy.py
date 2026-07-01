from __future__ import annotations

from dataclasses import dataclass
from typing import List

from slidesst.context.retriever import RetrievedEvidence


@dataclass
class PolicyDecision:
    evidence_id: str
    decision: str  # use | ignore | delay | downweight
    score: float
    reason: str


class EvidencePolicy:
    def __init__(self, use_threshold: float = 0.65, delay_threshold: float = 0.45, top_k: int = 5, penalties: dict | None = None):
        self.use_threshold = use_threshold
        self.delay_threshold = delay_threshold
        self.top_k = top_k
        self.penalties = penalties or {"time_distance": 0.05, "conflict": 0.30, "unsupported": 0.25, "supporting": 0.05, "visual_only": 0.30, "broad_scene": 0.15}

    def select(self, retrieved: List[RetrievedEvidence]) -> tuple[list, list[PolicyDecision]]:
        decisions: list[PolicyDecision] = []
        usable = []
        for r in retrieved:
            score = r.score
            reason_parts = []
            if r.item.time_distance_sec is not None:
                penalty = self.penalties.get("time_distance", 0.0) * min(abs(r.item.time_distance_sec) / 30.0, 2.0)
                score -= penalty
                if penalty:
                    reason_parts.append(f"time_distance_penalty={penalty:.3f}")
            if r.item.source_type in {"wrong_slide", "wrong_video", "wrong_clip", "negative_visual"} or r.item.support_label in {"wrong_slide", "previous_clip", "future_clip", "random_video", "negative_visual"}:
                score -= self.penalties.get("conflict", 0.0)
                reason_parts.append("source_type suggests mismatch")
            if r.item.visual_only and r.features.get("anchor", 0.0) <= 0.0:
                score -= self.penalties.get("visual_only", 0.0)
                reason_parts.append("visual_only_without_spoken_anchor")
            if r.item.source_type == "video_scene" and r.item.is_supporting is not True:
                score -= self.penalties.get("broad_scene", 0.15)
                reason_parts.append("broad_scene_without_label_support")
            if r.item.is_supporting is True:
                score += self.penalties.get("supporting", 0.05)
                reason_parts.append("marked supporting")
            if r.item.is_supporting is False:
                score -= self.penalties.get("unsupported", 0.0)
                reason_parts.append("marked non-supporting")
            if r.features.get("pinyin", 0.0) <= 0.0 and r.item.source_type in {"glossary", "slide_ocr"}:
                reason_parts.append("no pinyin overlap")
            if score >= self.use_threshold:
                decision = "use"
                usable.append((score, r.item))
            elif score >= self.delay_threshold:
                decision = "delay"
            else:
                decision = "ignore"
            decisions.append(PolicyDecision(r.item.evidence_id, decision, score, "; ".join(reason_parts) or "threshold rule"))
        usable_items = [item for _, item in sorted(usable, key=lambda x: x[0], reverse=True)[: self.top_k]]
        return usable_items, decisions
