import json
from types import SimpleNamespace

import pytest

from scripts import analyze_mcif_beyond_ocr_exploratory_screen as analyze
from scripts import build_mcif_beyond_ocr_exploratory_screen as build


def candidate(candidate_id, talk_id="talk", segment_id="seg"):
    return {
        "candidate_id": candidate_id,
        "talk_id": talk_id,
        "segment_id": segment_id,
        "lead_lower_bound_sec": 6.0,
        "source_segment_offset_sec": 10.0,
        "source_segment_end_sec": 15.0,
        "current_r0_candidate_absent": True,
        "normalized_source_candidate": "bounding box",
        "source_reference_en": "The model predicts a bounding box.",
    }


def test_validate_selection_enforces_bounds_and_unique_segments():
    rows = [candidate(f"c{i}", f"t{i // 2}", f"s{i}") for i in range(3)]
    config = {
        "selection": {
            "candidate_ids": ["c0", "c1", "c2"],
            "minimum_candidates": 3,
            "maximum_candidates": 5,
            "minimum_lead_sec": 5.0,
            "minimum_segment_duration_sec": 3.0,
            "maximum_segment_duration_sec": 24.0,
            "maximum_candidates_per_talk": 2,
            "maximum_candidates_per_segment": 1,
        }
    }
    assert build.validate_selection(rows, config) == rows
    rows[2]["segment_id"] = "s1"
    with pytest.raises(ValueError, match="segment exceeds"):
        build.validate_selection(rows, config)


def test_inference_firewall_rejects_reference_like_keys():
    build.assert_inference_safe({"audio": "a.wav", "ocr_text": "safe"})
    with pytest.raises(ValueError, match="Forbidden inference field"):
        build.assert_inference_safe({"target_reference": "secret"})


def test_prefix_auc_uses_all_prefixes():
    row = {
        "id": "x",
        "condition": "raw_image",
        "n_chunks": 2,
        "prefix_hypotheses": [
            {"step": 1, "audio_time_sec": 1.0, "hypothesis": "目标"},
            {"step": 2, "audio_time_sec": 2.0, "hypothesis": "目标文本"},
        ],
    }
    score = analyze.prefix_auc(row, "目标文本")
    assert 0.0 < score < 100.0


def test_analyzer_rejects_reference_in_model_output(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "runs_shard_0.jsonl").write_text(
        json.dumps(
            {
                "screen_id": "s",
                "acoustic_condition": "clean",
                "condition": "audio_only",
                "model_revision": "a" * 40,
                "reference": "leaked",
                "id": "s-clean",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Reference was exposed"):
        analyze.load_matrix(run_root, "a" * 40)


def test_analyzer_end_to_end_applies_positive_gate(tmp_path):
    revision = "a" * 40
    run_root = tmp_path / "run"
    run_root.mkdir()
    rows = []
    hypotheses = {
        "audio_only": "差",
        "ocr": "较差",
        "raw_image": "目标文本",
        "wrong_image": "错误",
    }
    for acoustic in ("clean", "babble_p5_s0"):
        for condition in analyze.CONDITIONS:
            hypothesis = hypotheses[condition]
            rows.append(
                {
                    "id": f"s-{acoustic}",
                    "screen_id": "s",
                    "acoustic_condition": acoustic,
                    "condition": condition,
                    "model_revision": revision,
                    "reference": "",
                    "n_chunks": 1,
                    "prefix_hypotheses": [
                        {"step": 1, "audio_time_sec": 1.0, "hypothesis": hypothesis}
                    ],
                    "hypothesis": hypothesis,
                }
            )
    (run_root / "runs_shard_0.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "screen_id": "s",
                "candidate_id": "c",
                "candidate_text": "target",
                "talk_id": "talk",
                "segment_id": "segment",
                "source_reference_en": "target text",
                "target_reference_zh": "目标文本",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "scope": "test",
                "inference": {"model_revision": revision},
                "acoustic_conditions": [
                    {"condition_id": "clean"},
                    {"condition_id": "babble_p5_s0"},
                ],
                "decision_rule": {"primary_metric": "prefix_auc_sentence_chrf"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    summary = analyze.analyze(
        SimpleNamespace(
            run_root=run_root,
            selection=selection,
            config=config,
            output_root=output,
            bootstrap_samples=100,
            seed=0,
        )
    )
    assert summary["result_count"] == 8
    assert summary["positive_candidate_acoustic_pairs"] == 2
    assert summary["unique_positive_candidates"] == 1
    assert summary["positive_in_both_acoustic_conditions"] == 1
    assert summary["positive_in_exactly_one_acoustic_condition"] == 0
    assert len((output / "positive_candidates.jsonl").read_text().splitlines()) == 2
