from slidesst.data.schema import ChallengeItem, ModelOutput
from slidesst.eval.metrics import evaluate_hard_cases


def test_hard_case_metrics():
    item = ChallengeItem.model_validate({
        "id": "x",
        "lecture_id": "l",
        "source_transcript": "线程",
        "ambiguous_items": [{"source_token": "线程", "pinyin": "xiancheng", "correct_target": ["thread"], "distractor_targets": ["ready-made"]}],
    })
    out = ModelOutput(id="x", condition="test", hypothesis="thread scheduling")
    m = evaluate_hard_cases([item], [out])
    assert m.hda == 1.0
    assert m.context_overuse_rate == 0.0
