import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.analyze_mcif_ppstructure_screen import (
    analyze,
    build_qa_inventory,
    row_categories,
    serialization_mode,
)


def block(label: str, content: str) -> dict:
    return {"label": label, "content": content}


def row(item_id: str, talk_id: str, blocks: list[dict]) -> dict:
    return {
        "id": item_id,
        "lecture_id": talk_id,
        "state_id": int(item_id[-1]),
        "frame": {
            "path": f"talks/{talk_id}/frames/state_{item_id[-1]}.png",
            "sha256": f"sha-{item_id}",
            "width": 100,
            "height": 50,
        },
        "flat_ocr": {"text": "Title A 10", "items": []},
        "structured_text": {
            "compact_text": "\n".join(item["content"] for item in blocks),
            "blocks": blocks,
        },
        "inference_fallback": None,
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
    }


def sample_rows() -> list[dict]:
    detected_table = row("a4", "talk-2", [block("table", '<img src="table.jpg" />')])
    detected_table["inference_fallback"] = {"strategy": "disable_table_recognition"}
    return [
        row("a0", "talk-1", [block("chart", "| A | 10 |\n|---|---|")]),
        row("a1", "talk-1", [block("table", "<table><tr><td>A</td></tr></table>")]),
        row("a2", "talk-1", [block("formula", "$$A=10$$")]),
        row("a3", "talk-2", [block("doc_title", "# Title")]),
        detected_table,
        row("a5", "talk-2", [block("text", "A")]),
    ]


def test_serialization_modes_distinguish_detection_from_content():
    assert serialization_mode(block("chart", "| A | 1 |\n|---|---|")) == "chart_markdown_table"
    assert serialization_mode(block("table", "<table></table>")) == "table_html"
    assert serialization_mode(block("formula", "$$x$$")) == "formula_latex"
    assert serialization_mode(block("table", '<img src="x.jpg" />')) == "table_image_placeholder"


def test_analyze_builds_strict_evidence_tiers():
    report, categories = analyze(sample_rows())
    assert report["rows"] == 6
    assert report["talk_count"] == 2
    assert report["category_row_counts"]["machine_readable_nonflat_structure"] == 3
    assert report["category_row_counts"]["machine_readable_table"] == 1
    assert report["category_row_counts"]["table_detection_only"] == 1
    assert report["serialization_mode_block_counts"]["table_image_placeholder"] == 1
    assert categories["a4"] == {"table_detection_only"}
    assert row_categories(sample_rows()[3]) == {"layout_hierarchy"}


def test_qa_inventory_is_deterministic_and_stratified():
    rows = sample_rows()
    _, categories = analyze(rows)
    first = build_qa_inventory(rows, categories, per_category=2, seed="seed")
    second = build_qa_inventory(rows, categories, per_category=2, seed="seed")
    assert first == second
    assert {item["qa_stratum"] for item in first} == {
        "layout_hierarchy_without_machine_structure",
        "machine_readable_chart",
        "machine_readable_formula",
        "machine_readable_table",
        "plain_or_image_only",
        "table_detection_only",
    }
    assert all(item["source_only_automatic_qa_not_annotation"] for item in first)


@pytest.mark.parametrize("mutation", ["duplicate", "transcript", "reference", "path"])
def test_analysis_fails_closed(mutation):
    rows = [row("a0", "talk-1", [block("text", "A")])]
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "transcript":
        rows[0]["source_transcript_consumed"] = True
    elif mutation == "reference":
        rows[0]["reference_translation"] = "target"
    elif mutation == "path":
        rows[0]["frame"]["path"] = "/absolute/frame.png"
    with pytest.raises(ValueError):
        analyze(rows)


def test_cli_binds_hash_and_create_once(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "analyze_mcif_ppstructure_screen.py"
    input_path = tmp_path / "input.jsonl"
    report_path = tmp_path / "report.json"
    inventory_path = tmp_path / "inventory.jsonl"
    input_path.write_text(
        "".join(json.dumps(item) + "\n" for item in sample_rows()), encoding="utf-8"
    )
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    command = [
        sys.executable,
        str(script),
        "--input",
        str(input_path),
        "--report",
        str(report_path),
        "--qa-inventory",
        str(inventory_path),
        "--expected-input-sha256",
        digest,
        "--expected-rows",
        "6",
        "--expected-talks",
        "2",
        "--qa-per-category",
        "1",
    ]
    subprocess.run(command, check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "SOURCE_ONLY_AUTOMATIC_DIAGNOSTIC_NOT_LABELS_OR_ST_RESULT"
    assert report["qa_inventory_rows"] == 6
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command, check=True)
