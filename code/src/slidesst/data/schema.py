from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AudioSpan(BaseModel):
    path: Optional[str] = None
    start_sec: float = 0.0
    end_sec: float = 0.0


class VideoSpan(BaseModel):
    path: Optional[str] = None
    start_sec: float = 0.0
    end_sec: float = 0.0
    frame_paths: List[str] = Field(default_factory=list)
    fps: Optional[float] = None


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


class ReferenceInfo(BaseModel):
    translation: Optional[str] = None
    status: str = "missing"
    teacher_models: List[str] = Field(default_factory=list)
    prompt_version: Optional[str] = None
    verification_notes: Optional[str] = None
    alternatives: List[str] = Field(default_factory=list)


class VisualContext(BaseModel):
    video_id: Optional[str] = None
    clip_id: Optional[str] = None
    scene_summary: Optional[str] = None
    ocr_text: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    spatial_relations: List[str] = Field(default_factory=list)
    frame_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HardLabel(BaseModel):
    label_id: str
    label_type: str
    source_span: str
    gold_en: List[str]
    distractor_en: List[str] = Field(default_factory=list)
    requires_visual: bool = False
    requires_ocr: bool = False
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    unspoken_visual_distractors: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


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
    modality: Optional[str] = None
    visual_scope: Optional[str] = None
    video_id: Optional[str] = None
    frame_id: Optional[str] = None
    temporal_start_sec: Optional[float] = None
    temporal_end_sec: Optional[float] = None
    confidence: Optional[float] = None
    spoken_anchor: Optional[str] = None
    visual_only: bool = False
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
    video: Optional[VideoSpan] = None
    source_transcript: str
    reference_translation: Optional[str] = None
    reference: ReferenceInfo = Field(default_factory=ReferenceInfo)
    streaming_units: List[StreamingUnit] = Field(default_factory=list)
    ambiguous_items: List[AmbiguousItem] = Field(default_factory=list)
    hard_labels: List[HardLabel] = Field(default_factory=list)
    slides: SlideInfo = Field(default_factory=SlideInfo)
    visual_context: Optional[VisualContext] = None
    glossary: List[GlossaryEntry] = Field(default_factory=list)
    background_docs: List[BackgroundDoc] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    annotation: Annotation = Field(default_factory=Annotation)

    def model_post_init(self, __context: Any) -> None:
        if self.reference.translation and not self.reference_translation:
            self.reference_translation = self.reference.translation
        elif self.reference_translation and not self.reference.translation:
            self.reference.translation = self.reference_translation
            if self.reference.status == "missing":
                self.reference.status = "legacy"


class ModelOutput(BaseModel):
    id: str
    condition: str
    hypothesis: str
    used_evidence_ids: List[str] = Field(default_factory=list)
    evidence_packet: List[EvidenceItem] = Field(default_factory=list)
    prompt: Optional[str] = None
    latency: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
