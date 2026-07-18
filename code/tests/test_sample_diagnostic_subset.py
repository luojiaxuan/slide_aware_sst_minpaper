import csv
import json
import subprocess
import sys
from pathlib import Path


def test_sample_diagnostic_subset_labels_and_exports(tmp_path):
    input_path = tmp_path / "items.jsonl"
    rows = [
        {
            "id": "ocr_1",
            "lecture_id": "lec",
            "source_transcript": "我们讨论卷积模型。",
            "video": {"frame_paths": ["/frames/ocr_1.jpg"]},
            "visual_context": {"ocr_text": ["卷积 模型"], "scene_summary": "slide"},
        },
        {
            "id": "visual_1",
            "lecture_id": "lec",
            "source_transcript": "我们看这个流程。",
            "video": {"frame_paths": ["/frames/visual_1.jpg"]},
            "visual_context": {"ocr_text": [], "scene_summary": "diagram"},
        },
        {
            "id": "distractor_1",
            "lecture_id": "lec",
            "source_transcript": "今天先介绍问题。",
            "video": {"frame_paths": ["/frames/distractor_1.jpg"]},
            "visual_context": {"ocr_text": ["ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4], "scene_summary": "dense text"},
        },
    ]
    input_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    out_jsonl = tmp_path / "sample.jsonl"
    out_csv = tmp_path / "sample.csv"
    stats = tmp_path / "stats.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/sample_diagnostic_subset.py",
            "--input",
            str(input_path),
            "--output-jsonl",
            str(out_jsonl),
            "--output-csv",
            str(out_csv),
            "--stats-json",
            str(stats),
            "--target-size",
            "3",
            "--per-slice",
            "2",
        ],
        check=True,
    )

    sampled = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    by_id = {row["id"]: row for row in sampled}
    assert "ocr_support" in by_id["ocr_1"]["visual_context"]["metadata"]["diagnostic_slices"]
    assert "term_homophone" in by_id["ocr_1"]["visual_context"]["metadata"]["diagnostic_slices"]
    assert "visual_non_ocr" in by_id["visual_1"]["visual_context"]["metadata"]["diagnostic_slices"]
    assert "latency_critical" in by_id["visual_1"]["visual_context"]["metadata"]["diagnostic_slices"]

    csv_rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8", newline="")))
    assert len(csv_rows) == 3
    assert any("distractor_risk" in row["diagnostic_slices"] for row in csv_rows)
    stats_data = json.loads(stats.read_text(encoding="utf-8"))
    assert stats_data["selected_items"] == 3
