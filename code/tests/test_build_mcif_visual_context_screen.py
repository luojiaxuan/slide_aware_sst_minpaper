import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_mcif_visual_context_screen import build_rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest() -> dict:
    return {
        "dataset": "mcif",
        "subset": "iwslt2026_translation_21",
        "talk_count": 1,
        "reference_files_extracted": False,
        "reference_content_inspected": False,
        "upstream": {"revision": "a" * 40},
        "talks": [{"talk_id": "talk-a", "video_duration_sec": 10.0}],
    }


def _states(root: Path) -> list[dict]:
    frame_a = root / "talks" / "talk-a" / "frames" / "frame_00001.jpg"
    frame_b = root / "talks" / "talk-a" / "frames" / "frame_00006.jpg"
    frame_a.parent.mkdir(parents=True, exist_ok=True)
    frame_a.write_bytes(b"frame-a")
    frame_b.write_bytes(b"frame-b")
    return [
        {
            "talk_id": "talk-a",
            "state_id": 0,
            "availability_start_sec": 0.0,
            "availability_end_sec": 5.0,
            "evidence_frame_path": str(frame_a),
            "evidence_frame_sha256": _sha256(frame_a),
        },
        {
            "talk_id": "talk-a",
            "state_id": 1,
            "availability_start_sec": 5.0,
            "availability_end_sec": 10.0,
            "evidence_frame_path": str(frame_b),
            "evidence_frame_sha256": _sha256(frame_b),
        },
    ]


def test_build_mcif_screen_is_complete_hash_bound_and_source_only(tmp_path):
    root = tmp_path / "states"
    rows = build_rows(_states(root), _source_manifest(), root)
    assert [row["id"] for row in rows] == [
        "mcif:talk-a:S000",
        "mcif:talk-a:S001",
    ]
    assert all(row["source_transcript"] == "" for row in rows)
    assert all(row["source_lang"] == "en" for row in rows)
    assert all(row["target_lang"] == "zh" for row in rows)
    assert all(len(row["video"]["frame_paths"]) == 1 for row in rows)
    assert all(not Path(row["video"]["frame_paths"][0]).is_absolute() for row in rows)
    assert all(
        row["visual_context"]["metadata"]["screen_role"]
        == "private_source_only_prescreen_not_annotation"
        for row in rows
    )
    serialized = json.dumps(rows).lower()
    assert "reference_translation" not in serialized
    assert "model_output" not in serialized


def test_build_mcif_screen_rejects_reference_unseal_and_hash_drift(tmp_path):
    root = tmp_path / "states"
    states = _states(root)
    source = _source_manifest()
    source["reference_files_extracted"] = True
    with pytest.raises(ValueError, match="must remain unextracted"):
        build_rows(states, source, root)

    source["reference_files_extracted"] = False
    source["reference_content_inspected"] = True
    with pytest.raises(ValueError, match="must remain uninspected"):
        build_rows(states, source, root)

    source["reference_content_inspected"] = False
    states[0]["evidence_frame_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        build_rows(states, source, root)


def test_build_mcif_screen_rejects_path_escape_and_timeline_gap(tmp_path):
    root = tmp_path / "states"
    states = _states(root)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    states[0]["evidence_frame_path"] = str(outside)
    states[0]["evidence_frame_sha256"] = _sha256(outside)
    with pytest.raises(ValueError, match="escapes the state root"):
        build_rows(states, _source_manifest(), root)

    states = _states(root)
    states[1]["availability_start_sec"] = 6.0
    with pytest.raises(ValueError, match="Non-contiguous MCIF state intervals"):
        build_rows(states, _source_manifest(), root)


def test_build_mcif_screen_rejects_symlink_frame(tmp_path):
    root = tmp_path / "states"
    states = _states(root)
    actual = Path(states[0]["evidence_frame_path"])
    linked = actual.with_name("linked.jpg")
    linked.symlink_to(actual)
    states[0]["evidence_frame_path"] = str(linked)
    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        build_rows(states, _source_manifest(), root)


def test_build_mcif_screen_rejects_incomplete_or_duplicate_inventory(tmp_path):
    root = tmp_path / "states"
    states = _states(root)
    source = _source_manifest()
    source["talk_count"] = 2
    with pytest.raises(ValueError, match="talk ids are not unique"):
        build_rows(states, source, root)

    source = _source_manifest()
    source["talk_count"] = 2
    source["talks"].append({"talk_id": "talk-b", "video_duration_sec": 10.0})
    with pytest.raises(ValueError, match="talk ids differ"):
        build_rows(states, source, root)

    with pytest.raises(ValueError, match="duplicate identifiers"):
        build_rows([*states, dict(states[0])], _source_manifest(), root)


def test_build_mcif_screen_cli_is_create_once_and_hashes_output(tmp_path):
    root = tmp_path / "states"
    states = _states(root)
    state_path = tmp_path / "states.jsonl"
    state_path.write_text(
        "".join(json.dumps(row) + "\n" for row in states), encoding="utf-8"
    )
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(_source_manifest()), encoding="utf-8")
    output = tmp_path / "input.jsonl"
    summary = tmp_path / "summary.json"
    command = [
        sys.executable,
        "scripts/build_mcif_visual_context_screen.py",
        "--causal-states",
        str(state_path),
        "--source-manifest",
        str(source_path),
        "--state-root",
        str(root),
        "--output",
        str(output),
        "--summary-out",
        str(summary),
        "--portable-output-label",
        "portable/input.jsonl",
    ]
    subprocess.run(command, check=True)
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["screen_input_sha256"] == _sha256(output)
    assert report["source_transcript_consumed"] is False
    assert report["target_or_reference_consumed"] is False
    assert report["model_output_consumed"] is False
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    assert subprocess.run(command, check=False, capture_output=True).returncode != 0
