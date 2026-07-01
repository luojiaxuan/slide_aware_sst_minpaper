from pathlib import Path

from slidesst.context.retriever import EvidenceRetriever
from slidesst.data.io import read_jsonl
from slidesst.data.schema import ChallengeItem, EvidenceItem, ModelOutput
from slidesst.eval.visual_metrics import evaluate_visual_cases
from slidesst.vision.evidence_builder import build_visual_evidence


def test_visual_toy_schema_loads():
    items = read_jsonl(Path("examples/toy_visual_challenge.jsonl"), ChallengeItem)
    assert len(items) == 3
    assert items[0].video is not None
    assert items[0].visual_context is not None
    assert items[0].hard_labels[0].requires_visual is True
    assert items[0].reference_translation == "Here we tighten this nut."


def test_build_visual_evidence_includes_negative_visual():
    item = read_jsonl(Path("examples/toy_visual_challenge.jsonl"), ChallengeItem)[2]
    evidence = build_visual_evidence(item)
    assert any(ev.source_type == "negative_visual" and ev.target_hint == "knife" for ev in evidence)


def test_future_visual_evidence_hidden_in_past_only_mode():
    ev = EvidenceItem(
        evidence_id="future",
        source_type="video_object",
        text="Visible object: nut",
        target_hint="nut",
        modality="video",
        temporal_start_sec=10.0,
    )
    got = EvidenceRetriever([ev]).retrieve("这个", current_time_sec=2.0, visual_availability="past_only")
    assert got == []


def test_visual_hallucination_metric_catches_unspoken_distractor():
    item = read_jsonl(Path("examples/toy_visual_challenge.jsonl"), ChallengeItem)[2]
    out = ModelOutput(id=item.id, condition="x", hypothesis="Now we look at the knife on the left.")
    metrics = evaluate_visual_cases([item], [out])
    assert metrics.vhr == 1.0
