from __future__ import annotations

from slidesst.context.retriever import zh_pinyin
from slidesst.data.schema import ChallengeItem, EvidenceItem
from slidesst.vision.ocr import OCREntry
from slidesst.vision.vlm import VisualDescription


def build_visual_evidence(
    item: ChallengeItem,
    ocr_entries: list[OCREntry] | None = None,
    frame_descriptions: list[VisualDescription] | None = None,
    clip_description: VisualDescription | None = None,
    mismatch: str = "matched",
) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    visual = item.visual_context
    video = item.video
    video_id = visual.video_id if visual else item.lecture_id
    clip_id = visual.clip_id if visual else item.id
    start = video.start_sec if video else None
    end = video.end_sec if video else None

    if visual and visual.scene_summary:
        evidence.append(
            _ev(item, "video_scene", f"{item.id}_scene_0", visual.scene_summary, None, video_id, clip_id, start, end, confidence=0.70)
        )

    if visual:
        for idx, text in enumerate(visual.ocr_text):
            evidence.append(
                _ev(item, "video_ocr", f"{item.id}_ocr_{idx}", text, text, video_id, clip_id, start, end, confidence=0.85, visual_only=False)
            )
        for idx, obj in enumerate(visual.objects):
            visual_only = not _anchored(item.source_transcript, obj, item.hard_labels)
            evidence.append(
                _ev(item, "video_object", f"{item.id}_obj_{idx}", f"Visible object: {obj}", obj, video_id, clip_id, start, end, confidence=0.80, visual_only=visual_only)
            )
        for idx, action in enumerate(visual.actions):
            visual_only = not _anchored(item.source_transcript, action, item.hard_labels)
            evidence.append(
                _ev(item, "video_action", f"{item.id}_action_{idx}", f"Visible action: {action}", action, video_id, clip_id, start, end, confidence=0.80, visual_only=visual_only)
            )
        for idx, relation in enumerate(visual.spatial_relations):
            evidence.append(
                _ev(item, "video_spatial", f"{item.id}_spatial_{idx}", f"Spatial relation: {relation}", relation, video_id, clip_id, start, end, confidence=0.75)
            )

    for idx, entry in enumerate(ocr_entries or []):
        evidence.append(
            _ev(item, "video_ocr", f"{item.id}_ocr_extra_{idx}", entry.text, entry.text, video_id, clip_id, start, end, frame_id=entry.frame_id, confidence=entry.confidence)
        )

    for idx, desc in enumerate(frame_descriptions or []):
        if desc.text:
            evidence.append(
                _ev(item, "video_vlm_frame", f"{item.id}_vlm_frame_{idx}", desc.text, "; ".join([*desc.objects, *desc.actions]), video_id, clip_id, start, end, frame_id=desc.frame_id, confidence=desc.confidence)
            )

    if clip_description and clip_description.text:
        evidence.append(
            _ev(item, "video_vlm_clip", f"{item.id}_vlm_clip_0", clip_description.text, "; ".join([*clip_description.objects, *clip_description.actions]), video_id, clip_id, start, end, confidence=clip_description.confidence)
        )

    for label in item.hard_labels:
        for idx, term in enumerate(label.unspoken_visual_distractors):
            evidence.append(
                _ev(item, "negative_visual", f"{item.id}_neg_{label.label_id}_{idx}", f"Unspoken visible distractor: {term}", term, video_id, clip_id, start, end, confidence=0.90, visual_only=True, is_supporting=False)
            )

    if mismatch != "matched":
        evidence.extend(make_wrong_visual_evidence(item, evidence, mismatch))
    supporting_ids = {eid for label in item.hard_labels for eid in label.supporting_evidence_ids}
    for ev in evidence:
        if ev.evidence_id in supporting_ids:
            ev.is_supporting = True
            ev.visual_only = False
    return evidence


def make_wrong_visual_evidence(item: ChallengeItem, source: list[EvidenceItem], mismatch: str) -> list[EvidenceItem]:
    wrong = []
    for idx, ev in enumerate(source[:3]):
        if ev.source_type == "negative_visual":
            continue
        data = ev.model_dump(mode="json")
        data["evidence_id"] = f"{item.id}_wrong_{mismatch}_{idx}"
        data["source_type"] = "wrong_clip" if "clip" in mismatch or "previous" in mismatch or "future" in mismatch else "wrong_video"
        data["support_label"] = mismatch
        data["is_supporting"] = False
        data["visual_only"] = True
        wrong.append(EvidenceItem.model_validate(data))
    return wrong


def _ev(
    item: ChallengeItem,
    source_type: str,
    evidence_id: str,
    text: str,
    target_hint: str | None,
    video_id: str | None,
    clip_id: str | None,
    start: float | None,
    end: float | None,
    *,
    frame_id: str | None = None,
    confidence: float | None = None,
    visual_only: bool = False,
    is_supporting: bool | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        source_type=source_type,
        text=text,
        item_id=item.id,
        lecture_id=item.lecture_id,
        target_hint=target_hint,
        pinyin=zh_pinyin(text),
        modality="video",
        visual_scope="frame" if frame_id else "clip",
        video_id=video_id,
        frame_id=frame_id,
        temporal_start_sec=start,
        temporal_end_sec=end,
        confidence=confidence,
        spoken_anchor=_spoken_anchor(item, target_hint or text),
        visual_only=visual_only,
        is_supporting=is_supporting,
        metadata={"clip_id": clip_id},
    )


def _spoken_anchor(item: ChallengeItem, hint: str) -> str | None:
    for label in item.hard_labels:
        if any(term.lower() in hint.lower() for term in label.gold_en):
            return label.source_span
    return None


def _anchored(transcript: str, term: str, labels) -> bool:
    if term and term.lower() in transcript.lower():
        return True
    return any(term.lower() in " ".join(label.gold_en).lower() for label in labels)
