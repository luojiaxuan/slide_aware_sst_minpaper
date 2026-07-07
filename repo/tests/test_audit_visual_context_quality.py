import importlib.util
import json
import subprocess
import sys


def test_audit_visual_context_quality_outputs_counts(tmp_path):
    input_path = tmp_path / "items.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    output_path = tmp_path / "qa.json"
    failures_path = tmp_path / "failures.jsonl"

    rows = [
        {
            "id": "clip_1",
            "lecture_id": "lec",
            "source_transcript": "我们讨论模型。",
            "video": {"frame_paths": ["/frames/a.jpg"]},
            "visual_context": {
                "ocr_text": ["模型"],
                "scene_summary": "A slide about models.",
                "objects": ["chart"],
                "actions": [],
                "spatial_relations": [],
                "metadata": {
                    "context_enrichment": {
                        "provider": "qwen_vl",
                        "model_id": "fake/qwen",
                        "batch_size": 2,
                        "raw_output": '{"ocr_text":["模型"],"scene_summary":"A slide about models."}',
                    }
                },
            },
        },
        {
            "id": "clip_2",
            "lecture_id": "lec",
            "source_transcript": "继续讨论。",
            "video": {"frame_paths": ["/frames/b.jpg"]},
            "visual_context": {
                "ocr_text": [],
                "scene_summary": "Slide first frame for Chinese-LiPS topic TEST; OCR not provided in source metadata.",
                "metadata": {
                    "context_enrichment": {
                        "provider": "qwen_vl",
                        "model_id": "fake/qwen",
                        "batch_size": 1,
                        "raw_output": "not-json",
                    }
                },
            },
        },
        {
            "id": "clip_3",
            "lecture_id": "lec",
            "source_transcript": "空输出元数据。",
            "video": {"frame_paths": ["/frames/c.jpg"]},
            "visual_context": {
                "ocr_text": ["空输出"],
                "scene_summary": "A slide with parsed fields but missing raw output.",
                "metadata": {
                    "context_enrichment": {
                        "provider": "qwen_vl",
                        "model_id": "fake/qwen",
                        "batch_size": 1,
                        "raw_output": "",
                    }
                },
            },
        },
        {
            "id": "clip_4",
            "lecture_id": "lec",
            "source_transcript": "看这幅山水图。",
            "video": {"frame_paths": ["/frames/d.jpg"]},
            "visual_context": {
                "ocr_text": [],
                "scene_summary": "A clean image-only slide.",
                "objects": ["mountains"],
                "actions": [],
                "spatial_relations": [],
                "metadata": {
                    "context_enrichment": {
                        "provider": "qwen_vl",
                        "model_id": "fake/qwen",
                        "batch_size": 2,
                        "raw_output": '{"ocr_text":[],"scene_summary":"A clean image-only slide.","objects":["mountains"]}',
                    }
                },
            },
        },
    ]
    input_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    baseline_path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    evidence_path.write_text(
        "\n".join(
            [
                json.dumps({"evidence_id": "e1", "source_type": "video_ocr", "text": "模型", "modality": "video"}),
                json.dumps(
                    {
                        "evidence_id": "e2",
                        "source_type": "video_object",
                        "text": "chart",
                        "modality": "video",
                        "visual_only": True,
                    }
                ),
                json.dumps(
                    {
                        "evidence_id": "e3",
                        "source_type": "video_scene",
                        "text": "A slide about models.",
                        "modality": "video",
                    }
                ),
                json.dumps(
                    {
                        "evidence_id": "e4",
                        "source_type": "video_scene",
                        "text": "Slide first frame for Chinese-LiPS topic TEST; OCR not provided in source metadata.",
                        "modality": "video",
                    }
                ),
                json.dumps({"evidence_id": "e5", "source_type": "video_ocr", "text": "空输出", "modality": "video"}),
                json.dumps(
                    {
                        "evidence_id": "e6",
                        "source_type": "video_scene",
                        "text": "A slide with parsed fields but missing raw output.",
                        "modality": "video",
                    }
                ),
                json.dumps(
                    {
                        "evidence_id": "e7",
                        "source_type": "video_object",
                        "text": "mountains",
                        "modality": "video",
                    }
                ),
                json.dumps(
                    {
                        "evidence_id": "e8",
                        "source_type": "video_scene",
                        "text": "A clean image-only slide.",
                        "modality": "video",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_visual_context_quality.py",
            "--input",
            str(input_path),
            "--baseline",
            str(baseline_path),
            "--evidence",
            str(evidence_path),
            "--output-json",
            str(output_path),
            "--failures-output",
            str(failures_path),
        ],
        check=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["rows"] == 4
    assert report["unique_ids"] == 4
    assert report["empty_context"] == 0
    assert report["missing_raw_output"] == 1
    assert report["no_ocr_with_summary"] == 2
    assert report["raw_parse_failures"] == 1
    assert report["raw_parse_failure_no_ocr"] == 1
    assert report["raw_parse_failure_with_fallback_summary"] == 1
    assert report["raw_parse_failure_no_ocr_with_summary_overlap"] == 1
    assert report["no_ocr_with_summary_not_parse_failure"] == 1
    assert report["batch_size_counts"] == {"1": 2, "2": 2}
    assert report["evidence"]["rows"] == 8
    assert report["evidence"]["challenge_consistency"]["video_ocr"]["delta"] == 0
    assert report["evidence"]["challenge_consistency"]["video_object"]["delta"] == 0
    assert report["evidence"]["challenge_consistency"]["video_scene"]["delta"] == 0
    assert report["baseline_comparison"]["numeric_delta"]["rows"]["delta"] == 3
    assert report["baseline_comparison"]["numeric_delta"]["missing_raw_output"]["delta"] == 1
    assert report["failures_output"]["rows"] == 2
    input_lines = input_path.read_text(encoding="utf-8").splitlines()
    failure_lines = failures_path.read_text(encoding="utf-8").splitlines()
    assert failure_lines[0] == input_lines[1]
    failures = [json.loads(line) for line in failure_lines]
    assert [row["id"] for row in failures] == ["clip_2", "clip_3"]


def test_audit_visual_context_quality_json_detection():
    spec = importlib.util.spec_from_file_location(
        "audit_visual_context_quality_test",
        "scripts/audit_visual_context_quality.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    assert module.is_json_like('```json\n{"ocr_text":[]}\n```')
    assert module.is_json_like('```json\n{"ocr_text":["x}y"],"scene_summary":"ok"}\n```')
    assert module.is_json_like('{"ocr_text": []}\ntrailing prose')
    assert not module.is_json_like('{"ocr_text": ["理想的睡眠时间"')
    assert not module.is_json_like('{"ok": true} trailing {"ocr_text": ["理')
    assert not module.is_json_like("no json here")
