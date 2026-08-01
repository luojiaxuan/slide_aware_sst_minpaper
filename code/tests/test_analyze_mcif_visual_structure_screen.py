import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analyze_mcif_visual_structure_screen import analyze, relation_categories


def row(item_id: str, talk_id: str, relations: list[str]) -> dict:
    return {
        "id": item_id,
        "lecture_id": talk_id,
        "source_transcript": "",
        "reference_translation": None,
        "reference": {"translation": None, "alternatives": []},
        "visual_context": {"spatial_relations": relations},
    }


def test_relation_categories_are_nonexclusive():
    categories = relation_categories(
        ["A blue arrow connects the table row to a higher bar in the chart."]
    )
    assert categories == {
        "chart_quantitative",
        "connectivity_process",
        "table_association",
        "visual_emphasis",
    }


def test_video_feed_is_not_a_connectivity_process():
    assert relation_categories(["A video feed is located in the bottom right corner."]) == set()


def test_analyze_distinguishes_structural_and_simple_layout_rows():
    report = analyze(
        [
            row("a", "talk-1", ["The title is above the bullet list."]),
            row("b", "talk-1", ["An arrow connects table rows to chart bars."]),
            row("c", "talk-2", []),
        ]
    )
    assert report["rows"] == 3
    assert report["talk_count"] == 2
    assert report["rows_with_relations"] == 2
    assert report["rows_with_lexical_structural_relation_candidates"] == 1
    assert report["talks_with_lexical_structural_relation_candidates"] == 1
    assert report["simple_layout_only_rows"] == 1
    assert report["rows_without_relations"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "transcript", "reference"])
def test_analyze_fails_closed(mutation):
    first = row("a", "talk-1", [])
    rows = [first]
    if mutation == "duplicate":
        rows.append(first)
    elif mutation == "transcript":
        first["source_transcript"] = "speech"
    elif mutation == "reference":
        first["reference"]["translation"] = "target"
    with pytest.raises(ValueError):
        analyze(rows)


def test_cli_binds_hash_and_create_once(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "analyze_mcif_visual_structure_screen.py"
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(row("a", "talk-1", [])) + "\n", encoding="utf-8")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    command = [
        sys.executable,
        str(script),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--expected-input-sha256",
        digest,
        "--expected-rows",
        "1",
        "--expected-talks",
        "1",
    ]
    subprocess.run(command, check=True)
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "MODEL_OUTPUT_LEXICAL_DIAGNOSTIC_NOT_LABELS"
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command, check=True)
