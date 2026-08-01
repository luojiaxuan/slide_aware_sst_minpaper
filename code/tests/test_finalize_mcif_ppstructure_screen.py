import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    script = (
        Path(__file__).parents[1] / "scripts" / "finalize_mcif_ppstructure_screen.py"
    )
    spec = importlib.util.spec_from_file_location(
        "finalize_mcif_ppstructure_screen", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def source(item_id: str, talk: str, state_id: int) -> dict:
    return {
        "id": item_id,
        "lecture_id": talk,
        "state_id": state_id,
        "availability_start_sec": state_id + 0.5,
        "availability_end_sec": state_id + 1.5,
        "evidence_timestamp_sec": state_id + 0.5,
        "frame_path": f"talks/{talk}/frames/state_{state_id:03d}.png",
        "frame_sha256": f"sha-{state_id}",
        "frame_width": 100,
        "frame_height": 50,
    }


def config() -> dict:
    return {
        "models": {"text_detection": "PP-OCRv6_medium_det"},
    }


def output(row: dict, config_sha: str, input_sha: str, *, label: str = "chart") -> dict:
    return {
        "id": row["id"],
        "lecture_id": row["lecture_id"],
        "state_id": row["state_id"],
        "availability_start_sec": row["availability_start_sec"],
        "availability_end_sec": row["availability_end_sec"],
        "evidence_timestamp_sec": row["evidence_timestamp_sec"],
        "frame": {
            "path": row["frame_path"],
            "sha256": row["frame_sha256"],
            "width": row["frame_width"],
            "height": row["frame_height"],
        },
        "provenance": {
            "provider": "PaddleOCR.PPStructureV3",
            "package_versions": {
                "paddleocr": "3.7.0",
                "paddlex": "3.7.0",
                "paddlepaddle": "3.3.0",
            },
            "config_sha256": config_sha,
            "input_manifest_sha256": input_sha,
            "models": config()["models"],
            "model_manifest_sha256": "model-sha",
            "shard_index": row["state_id"] % 2,
        },
        "flat_ocr": {
            "item_count": 1,
            "text": "A",
            "items": [{"text": "A", "bbox_out_of_bounds": False}],
        },
        "structured_text": {
            "block_count": 1,
            "compact_text": f"[{label}] A",
            "blocks": [
                {"label": label, "content": "A", "bbox_out_of_bounds": False}
            ],
        },
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
    }


def test_finalize_orders_rows_and_counts_structure():
    module = load_module()
    rows = [source("a", "talk-a", 0), source("b", "talk-b", 1)]
    config_sha = module.canonical_hash(config())
    outputs = [
        output(rows[1], config_sha, "input-sha", label="formula"),
        output(rows[0], config_sha, "input-sha", label="chart"),
    ]
    finalized, report = module.finalize(
        rows,
        outputs,
        config=config(),
        input_manifest_sha256="input-sha",
        model_manifest_sha256="model-sha",
    )
    assert [row["id"] for row in finalized] == ["a", "b"]
    assert report["rows"] == 2
    assert report["talk_count"] == 2
    assert report["flat_ocr"]["rows_with_text"] == 2
    assert report["structured_text"]["rows_with_chart"] == 1
    assert report["structured_text"]["rows_with_formula"] == 1
    assert report["shard_counts"] == {"0": 1, "1": 1}


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "missing", "frame", "config", "model", "reference", "version"],
)
def test_finalize_fails_closed(mutation):
    module = load_module()
    row = source("a", "talk-a", 0)
    config_sha = module.canonical_hash(config())
    result = output(row, config_sha, "input-sha")
    outputs = [result]
    if mutation == "duplicate":
        outputs.append(result)
    elif mutation == "missing":
        outputs.clear()
    elif mutation == "frame":
        result["frame"]["sha256"] = "wrong"
    elif mutation == "config":
        result["provenance"]["config_sha256"] = "wrong"
    elif mutation == "model":
        result["provenance"]["model_manifest_sha256"] = "wrong"
    elif mutation == "reference":
        result["reference_translation"] = "target"
    elif mutation == "version":
        result["provenance"]["package_versions"]["paddleocr"] = "3.6.0"
    with pytest.raises(ValueError):
        module.finalize(
            [row],
            outputs,
            config=config(),
            input_manifest_sha256="input-sha",
            model_manifest_sha256="model-sha",
        )


def test_nonempty_failure_file_is_rejected(tmp_path):
    module = load_module()
    empty = tmp_path / "empty.jsonl"
    failed = tmp_path / "failed.jsonl"
    empty.write_text("", encoding="utf-8")
    failed.write_text(json.dumps({"id": "a", "error": "boom"}) + "\n", encoding="utf-8")
    module.assert_no_failures([empty, tmp_path / "missing.jsonl"])
    with pytest.raises(ValueError, match="Nonempty"):
        module.assert_no_failures([failed])
