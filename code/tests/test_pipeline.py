from slidesst.context.policy import EvidencePolicy
from slidesst.context.retriever import EvidenceRetriever
from slidesst.data.mining import mine_hard_examples
from slidesst.data.schema import ChallengeItem, EvidenceItem, SlideInfo, StreamingUnit, VideoSpan, VisualContext
from slidesst.translation.adapters import OpenAICompatibleTranslator, _build_messages, _chat_template_kwargs
from scripts.generate_references import _translation_config
from scripts.run_batched_reference_experiments import _condition_evidence, _final_state, _packet_for_condition


def test_mine_hard_examples_adds_homophone_item():
    item = ChallengeItem(id="x", lecture_id="l", source_transcript="这里讨论线程调度。")
    mined = mine_hard_examples([item])
    assert len(mined) == 1
    assert mined[0].ambiguous_items[0].source_token == "线程"
    assert "thread" in mined[0].ambiguous_items[0].correct_target


def test_policy_prefers_supporting_current_slide_over_wrong_slide():
    evidence = [
        EvidenceItem(
            evidence_id="ok",
            source_type="slide_ocr",
            text="Thread scheduling / 线程调度",
            target_hint="thread scheduling",
            slide_id="s1",
            is_supporting=True,
        ),
        EvidenceItem(
            evidence_id="bad",
            source_type="wrong_slide",
            text="现成模板",
            target_hint="ready-made template",
            slide_id="s2",
            is_supporting=False,
        ),
    ]
    retrieved = EvidenceRetriever(evidence).retrieve("线程调度", current_slide_id="s1", top_m=2)
    packet, decisions = EvidencePolicy(use_threshold=0.30, delay_threshold=0.10, top_k=2).select(retrieved)
    assert [item.evidence_id for item in packet] == ["ok"]
    assert {decision.evidence_id: decision.decision for decision in decisions}["bad"] != "use"


def test_openai_compatible_translator_chat_url():
    translator = OpenAICompatibleTranslator({"model": "m", "base_url": "http://127.0.0.1:8000/v1"})
    assert translator._chat_url() == "http://127.0.0.1:8000/v1/chat/completions"


def test_chat_template_kwargs_supports_qwen3_thinking_toggle():
    kwargs = _chat_template_kwargs(
        {
            "enable_thinking": False,
            "chat_template_kwargs": {"tokenize": False},
        }
    )

    assert kwargs == {"tokenize": False, "enable_thinking": False}


def test_build_messages_includes_optional_system_prompt():
    assert _build_messages("Translate this.", "Return English only.") == [
        {"role": "system", "content": "Return English only."},
        {"role": "user", "content": "Translate this."},
    ]


def test_generate_references_keeps_config_prompt_version_by_default():
    cfg = {"translation": {"provider": "hf_transformers", "model": "m", "prompt_version": "config_v"}}

    assert _translation_config(cfg, None, None, None, None)["prompt_version"] == "config_v"
    assert _translation_config(cfg, None, None, None, "cli_v")["prompt_version"] == "cli_v"
    assert _translation_config(cfg, None, None, "cuda:1", None)["device"] == "cuda:1"


def test_batched_runner_uses_final_streaming_state():
    item = ChallengeItem(
        id="x",
        lecture_id="l",
        source_transcript="这里讨论线程调度。",
        streaming_units=[
            StreamingUnit(t=1.0, partial_transcript="这里讨论"),
            StreamingUnit(t=2.0, partial_transcript="这里讨论线程调度。"),
        ],
        slides=SlideInfo(matched_slide_id="s1"),
    )

    state = _final_state(item)

    assert state.partial_transcript == item.source_transcript
    assert state.current_slide_id == "s1"
    assert state.video_time_sec == 2.0


def test_batched_runner_v8_packet_is_wrong_visual_only_and_does_not_leak():
    item = ChallengeItem(
        id="x",
        lecture_id="l",
        source_transcript="这里讨论线程调度。",
        video=VideoSpan(start_sec=0.0, end_sec=2.0),
        slides=SlideInfo(matched_slide_id="s1"),
        visual_context=VisualContext(
            video_id="v",
            clip_id="c",
            scene_summary="A lecture slide about thread scheduling.",
            ocr_text=["Thread scheduling"],
            objects=["diagram"],
        ),
    )
    indexed = [
        EvidenceItem(
            evidence_id="correct_ocr",
            item_id="x",
            source_type="video_ocr",
            modality="video",
            text="Thread scheduling",
        )
    ]
    cfg = {"context": {"packet_top_k": 5}}
    policy = EvidencePolicy(use_threshold=0.3, delay_threshold=0.2, top_k=5)
    state = _final_state(item)

    item.evidence = _condition_evidence(item, {"x": indexed}, "V8_wrong_visual", "matched")
    packet, policy_log = _packet_for_condition(item, state, "V8_wrong_visual", policy, cfg)

    assert policy_log == []
    assert packet
    assert {ev.source_type for ev in packet} <= {"wrong_video", "wrong_clip", "negative_visual"}
    assert "correct_ocr" not in {ev.evidence_id for ev in packet}

    next_evidence = _condition_evidence(item, {"x": indexed}, "V2_ocr_only", "matched")
    assert [ev.evidence_id for ev in next_evidence] == ["correct_ocr"]
