import csv
import json
import subprocess
import sys


def test_export_diagnostic_review_sheet_includes_outputs_and_blank_labels(tmp_path):
    challenge = tmp_path / "challenge.jsonl"
    challenge.write_text(
        json.dumps(
            {
                "id": "x",
                "lecture_id": "l",
                "source_transcript": "这里讨论线程调度。",
                "reference_translation": "This discusses thread scheduling.",
                "visual_context": {
                    "scene_summary": "A slide about thread scheduling.",
                    "ocr_text": ["Thread scheduling"],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    audit = tmp_path / "audit.csv"
    audit.write_text("id,severity,flags\nx,review,length_ratio_review=5.0\n", encoding="utf-8")
    run_dir = tmp_path / "runs"
    out_dir = run_dir / "matched" / "V4_ocr_plus_visual"
    out_dir.mkdir(parents=True)
    (out_dir / "outputs.jsonl").write_text(
        json.dumps(
            {
                "id": "x",
                "condition": "V4_ocr_plus_visual",
                "hypothesis": "This discusses thread scheduling.",
                "evidence_packet": [
                    {
                        "evidence_id": "ev1",
                        "source_type": "video_ocr",
                        "text": "Thread scheduling",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    sheet = tmp_path / "sheet.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/export_diagnostic_review_sheet.py",
            "--input",
            str(challenge),
            "--audit-csv",
            str(audit),
            "--run-dir",
            str(run_dir),
            "--output",
            str(sheet),
            "--conditions",
            "V4_ocr_plus_visual",
        ],
        check=True,
    )

    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["reference_audit_flags"] == "length_ratio_review=5.0"
    assert rows[0]["hyp_V4_ocr_plus_visual"] == "This discusses thread scheduling."
    assert rows[0]["v4_evidence_packet"] == "ev1 [video_ocr]: Thread scheduling"
    assert rows[0]["human_reference_en"] == ""
    assert rows[0]["supporting_evidence_ids"] == ""
