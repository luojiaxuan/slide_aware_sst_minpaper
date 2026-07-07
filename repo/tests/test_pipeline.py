from slidesst.context.policy import EvidencePolicy
from slidesst.context.retriever import EvidenceRetriever
from slidesst.data.mining import mine_hard_examples
from slidesst.data.schema import ChallengeItem, EvidenceItem
from slidesst.translation.adapters import OpenAICompatibleTranslator, _build_messages, _chat_template_kwargs
from scripts.generate_references import _translation_config


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

    assert _translation_config(cfg, None, None, None)["prompt_version"] == "config_v"
    assert _translation_config(cfg, None, None, "cli_v")["prompt_version"] == "cli_v"
