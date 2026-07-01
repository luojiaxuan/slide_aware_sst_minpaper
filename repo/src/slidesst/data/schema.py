from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AudioSpan(BaseModel):
    path: Optional[str] = None
    start_sec: float = 0.0
    end_sec: float = 0.0


class StreamingUnit(BaseModel):
    t: float
    partial_transcript: str


class AmbiguousItem(BaseModel):
    source_token: str
    pinyin: str
    correct_target: List[str]
    distractor_targets: List[str] = Field(default_factory=list)
    category: List[str] = Field(default_factory=list)


class GlossaryEntry(BaseModel):
    src: str
    tgt: str
    desc: Optional[str] = None
    source: str = "manual"


class BackgroundDoc(BaseModel):
    doc_id: str
    text: str


class EvidenceItem(BaseModel):
    evidence_id: str
    source_type: str
    text: str
    item_id: Optional[str] = None
    lecture_id: Optional[str] = None
    target_hint: Optional[str] = None
    slide_id: Optional[str] = None
    time_distance_sec: Optional[float] = None
    pinyin: Optional[str] = None
    score: Optional[float] = None
    support_label: Optional[str] = None
    is_supporting: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SlideInfo(BaseModel):
    matched_slide_id: Optional[str] = None
    previous_slide_id: Optional[str] = None
    next_slide_id: Optional[str] = None
    matched_slide_text: Optional[str] = None
    matched_slide_image: Optional[str] = None


class Annotation(BaseModel):
    verified: bool = False
    annotator: Optional[str] = None
    notes: Optional[str] = None


class ChallengeItem(BaseModel):
    id: str
    lecture_id: str
    source_lang: str = "zh"
    target_lang: str = "en"
    audio: Optional[AudioSpan] = None
    source_transcript: str
    reference_translation: Optional[str] = None
    streaming_units: List[StreamingUnit] = Field(default_factory=list)
    ambiguous_items: List[AmbiguousItem] = Field(default_factory=list)
    slides: SlideInfo = Field(default_factory=SlideInfo)
    glossary: List[GlossaryEntry] = Field(default_factory=list)
    background_docs: List[BackgroundDoc] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    annotation: Annotation = Field(default_factory=Annotation)


class ModelOutput(BaseModel):
    id: str
    condition: str
    hypothesis: str
    used_evidence_ids: List[str] = Field(default_factory=list)
    evidence_packet: List[EvidenceItem] = Field(default_factory=list)
    prompt: Optional[str] = None
    latency: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
