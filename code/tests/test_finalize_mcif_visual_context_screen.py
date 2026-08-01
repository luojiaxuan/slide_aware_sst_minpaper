import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "finalize_mcif_visual_context_screen.py"
    spec = importlib.util.spec_from_file_location("finalize_mcif_visual_context_screen", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def input_row(item_id: str, talk: str, state_id: int) -> dict:
    frame = f"talks/{talk}/frames/frame_{state_id:05d}.jpg"
    return {
        "id": item_id,
        "lecture_id": talk,
        "source_lang": "en",
        "target_lang": "zh",
        "source_transcript": "",
        "video": {"start_sec": float(state_id), "end_sec": float(state_id + 1), "frame_paths": [frame]},
        "visual_context": {
            "video_id": talk,
            "clip_id": item_id,
            "metadata": {
                "screen_role": "private_source_only_prescreen_not_annotation",
                "state_id": state_id,
                "availability_start_sec": float(state_id),
                "availability_end_sec": float(state_id + 1),
                "evidence_frame_sha256": f"sha-{state_id}",
            },
        },
    }


def output_row(source: dict, model_path: str, batch_size: int = 16) -> dict:
    row = json.loads(json.dumps(source))
    row.update(
        {
            "reference_translation": None,
            "reference": {"translation": None, "alternatives": []},
            "streaming_units": [],
            "ambiguous_items": [],
            "hard_labels": [],
            "glossary": [],
            "background_docs": [],
            "evidence": [],
            "slides": {"matched_slide_text": None, "matched_slide_image": None},
        }
    )
    visual = row["visual_context"]
    visual.update(
        {
            "scene_summary": "A chart compares system A and system B.",
            "ocr_text": ["System A", "System B"],
            "objects": ["bar chart"],
            "actions": ["System A increases"],
            "spatial_relations": ["System A is above System B"],
        }
    )
    raw = {
        "scene_summary": visual["scene_summary"],
        "ocr_text": visual["ocr_text"],
        "objects": visual["objects"],
        "actions": visual["actions"],
        "spatial_relations": visual["spatial_relations"],
    }
    relative_frame = source["video"]["frame_paths"][0]
    visual["metadata"]["context_enrichment"] = {
        "provider": "qwen_vl",
        "model_id": model_path,
        "frame_path": f"/remote/state_root/{relative_frame}",
        "batch_size": batch_size,
        "raw_output": json.dumps(raw),
    }
    return row


def test_finalize_reorders_and_sanitizes_rows():
    module = load_module()
    model_path = "/cache/model/snapshots/rev"
    first = input_row("mcif:a:S000", "a", 0)
    second = input_row("mcif:b:S000", "b", 0)
    finalized, report = module.finalize_rows(
        [first, second],
        [output_row(second, model_path, 32), output_row(first, model_path, 16)],
        expected_raw_model_id=model_path,
        canonical_model_id="Qwen/Qwen3-VL-32B-Instruct",
        model_revision="rev",
        default_prompt_id="source_v1",
        default_prompt_sha256="prompt-sha",
    )

    assert [row["id"] for row in finalized] == [first["id"], second["id"]]
    metadata = finalized[0]["visual_context"]["metadata"]["context_enrichment"]
    assert metadata["model_id"] == "Qwen/Qwen3-VL-32B-Instruct"
    assert metadata["model_revision"] == "rev"
    assert metadata["frame_path"] == first["video"]["frame_paths"][0]
    assert metadata["prompt_id"] == "source_v1"
    assert metadata["prompt_sha256"] == "prompt-sha"
    assert report["rows"] == 2
    assert report["talk_count"] == 2
    assert report["raw_json_parse_failures"] == 0
    assert report["rows_with_spatial_relation_candidates"] == 2
    assert report["batch_size_counts"] == {"16": 1, "32": 1}
    assert report["prompt_id_counts"] == {"source_v1": 2}


@pytest.mark.parametrize("failure", ["duplicate", "missing", "reference", "raw_json"])
def test_finalize_fails_closed(failure):
    module = load_module()
    model_path = "/cache/model/snapshots/rev"
    source = input_row("mcif:a:S000", "a", 0)
    output = output_row(source, model_path)
    input_rows = [source]
    output_rows = [output]
    if failure == "duplicate":
        output_rows.append(output)
    elif failure == "missing":
        output_rows.clear()
    elif failure == "reference":
        output["reference"]["translation"] = "target text"
    elif failure == "raw_json":
        output["visual_context"]["metadata"]["context_enrichment"]["raw_output"] = "not json"

    with pytest.raises(ValueError):
        module.finalize_rows(
            input_rows,
            output_rows,
            expected_raw_model_id=model_path,
            canonical_model_id="Qwen/Qwen3-VL-32B-Instruct",
            model_revision="rev",
            default_prompt_id="source_v1",
            default_prompt_sha256="prompt-sha",
        )


def test_overlay_replacements_preserves_base_order_and_rejects_unknown_ids():
    module = load_module()
    first = {"id": "a", "value": 1}
    second = {"id": "b", "value": 2}
    overlaid, counts = module.overlay_replacements(
        [first, second], [[{"id": "b", "value": 3}]]
    )
    assert overlaid == [first, {"id": "b", "value": 3}]
    assert counts == [1]
    with pytest.raises(ValueError, match="unknown ids"):
        module.overlay_replacements([first], [[{"id": "c"}]])


def test_cli_create_once_and_write_hashes(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "finalize_mcif_visual_context_screen.py"
    model_path = "/cache/model/snapshots/rev"
    source = input_row("mcif:a:S000", "a", 0)
    output = output_row(source, model_path)
    input_path = tmp_path / "input.jsonl"
    shard_path = tmp_path / "shard.jsonl"
    final_path = tmp_path / "final.jsonl"
    report_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    shard_path.write_text(json.dumps(output) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(script),
        "--input",
        str(input_path),
        "--shard",
        str(shard_path),
        "--output",
        str(final_path),
        "--report",
        str(report_path),
        "--expected-rows",
        "1",
        "--expected-talks",
        "1",
        "--expected-raw-model-id",
        model_path,
        "--canonical-model-id",
        "Qwen/Qwen3-VL-32B-Instruct",
        "--model-revision",
        "rev",
        "--default-prompt-id",
        "source_v1",
        "--default-prompt-sha256",
        "prompt-sha",
        "--git-commit",
        "abc123",
    ]
    subprocess.run(command, check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["rows"] == 1
    assert len(report["output_sha256"]) == 64
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command, check=True)
