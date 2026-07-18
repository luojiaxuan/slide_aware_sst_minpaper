import csv
import json
import subprocess
import sys
from pathlib import Path


def test_build_training_data_filters_by_audit(tmp_path):
    refs = tmp_path / "refs.jsonl"
    refs.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "pass_1",
                        "lecture_id": "lec",
                        "source_transcript": "你好。",
                        "reference_translation": "Hello.",
                    }
                ),
                json.dumps(
                    {
                        "id": "reject_1",
                        "lecture_id": "lec",
                        "source_transcript": "世界。",
                        "reference_translation": "World. Say something.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    audit = tmp_path / "audit.csv"
    with audit.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "severity", "flags"])
        writer.writeheader()
        writer.writerow({"id": "pass_1", "severity": "pass", "flags": ""})
        writer.writerow({"id": "reject_1", "severity": "reject", "flags": "visual_placeholder=say something"})

    out_sft = tmp_path / "sft.jsonl"
    out_rejected = tmp_path / "rejected.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_training_data.py",
            "--references",
            str(refs),
            "--audit-csv",
            str(audit),
            "--out-sft",
            str(out_sft),
            "--out-rejected",
            str(out_rejected),
        ],
        check=True,
    )

    sft_rows = [json.loads(line) for line in out_sft.read_text(encoding="utf-8").splitlines()]
    rejected_rows = [json.loads(line) for line in out_rejected.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in sft_rows] == ["pass_1"]
    assert [row["id"] for row in rejected_rows] == ["reject_1"]
    assert sft_rows[0]["messages"][1]["content"] == "Hello."
