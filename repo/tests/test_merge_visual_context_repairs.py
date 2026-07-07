import json
import subprocess
import sys


def test_merge_visual_context_repairs_preserves_order_and_overrides(tmp_path):
    base = tmp_path / "base.jsonl"
    repair_1 = tmp_path / "repair_1.jsonl"
    repair_2 = tmp_path / "repair_2.jsonl"
    output = tmp_path / "merged.jsonl"
    log_json = tmp_path / "merge_log.json"

    base.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "value": "old-a"}),
                json.dumps({"id": "b", "value": "old-b"}),
                json.dumps({"id": "c", "value": "old-c"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repair_1.write_text(
        "\n".join(
            [
                json.dumps({"id": "b", "value": "repair-b-1"}),
                json.dumps({"id": "c", "value": "repair-c"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repair_2.write_text(json.dumps({"id": "b", "value": "repair-b-2"}) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/merge_visual_context_repairs.py",
            "--base",
            str(base),
            "--repair",
            str(repair_1),
            "--repair",
            str(repair_2),
            "--output",
            str(output),
            "--expected-rows",
            "3",
            "--expected-replacements",
            "2",
            "--log-json",
            str(log_json),
        ],
        check=True,
    )

    merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in merged] == ["a", "b", "c"]
    assert [row["value"] for row in merged] == ["old-a", "repair-b-2", "repair-c"]
    log = json.loads(log_json.read_text(encoding="utf-8"))
    assert log["rows"] == 3
    assert log["replaced_rows"] == 2
    assert log["repair_unique_ids"] == 2
    assert log["repair_overridden_ids"] == 1


def test_merge_visual_context_repairs_rejects_unused_repair_id(tmp_path):
    base = tmp_path / "base.jsonl"
    repair = tmp_path / "repair.jsonl"
    output = tmp_path / "merged.jsonl"

    base.write_text(json.dumps({"id": "a", "value": "old-a"}) + "\n", encoding="utf-8")
    repair.write_text(json.dumps({"id": "missing", "value": "repair"}) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/merge_visual_context_repairs.py",
            "--base",
            str(base),
            "--repair",
            str(repair),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not output.exists()
    assert "unused_repair_ids" in result.stderr


def test_merge_visual_context_repairs_supports_in_place_output(tmp_path):
    base = tmp_path / "base.jsonl"
    repair = tmp_path / "repair.jsonl"

    base.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "value": "old-a"}),
                json.dumps({"id": "b", "value": "old-b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repair.write_text(json.dumps({"id": "b", "value": "repair-b"}) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/merge_visual_context_repairs.py",
            "--base",
            str(base),
            "--repair",
            str(repair),
            "--output",
            str(base),
            "--expected-rows",
            "2",
            "--expected-replacements",
            "1",
        ],
        check=True,
    )

    merged = [json.loads(line) for line in base.read_text(encoding="utf-8").splitlines()]
    assert [row["value"] for row in merged] == ["old-a", "repair-b"]


def test_merge_visual_context_repairs_keeps_existing_output_on_validation_failure(tmp_path):
    base = tmp_path / "base.jsonl"
    repair = tmp_path / "repair.jsonl"
    output = tmp_path / "merged.jsonl"

    base.write_text(json.dumps({"id": "a", "value": "old-a"}) + "\n", encoding="utf-8")
    repair.write_text(json.dumps({"id": "a", "value": "repair-a"}) + "\n", encoding="utf-8")
    output.write_text("keep me\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/merge_visual_context_repairs.py",
            "--base",
            str(base),
            "--repair",
            str(repair),
            "--output",
            str(output),
            "--expected-rows",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert output.read_text(encoding="utf-8") == "keep me\n"
    assert not output.with_suffix(output.suffix + ".tmp").exists()
