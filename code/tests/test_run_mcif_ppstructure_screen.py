import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "run_mcif_ppstructure_screen.py"
    spec = importlib.util.spec_from_file_location("run_mcif_ppstructure_screen", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_row(frame_path: str, frame_sha: str) -> dict:
    return {
        "id": "mcif:talk-a:S000",
        "lecture_id": "talk-a",
        "state_id": 0,
        "availability_start_sec": 0.5,
        "availability_end_sec": 2.0,
        "evidence_timestamp_sec": 0.5,
        "frame_path": frame_path,
        "frame_sha256": frame_sha,
        "frame_width": 100,
        "frame_height": 50,
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
    }


def provider_result() -> dict:
    return {
        "input_path": "/remote/frame.png",
        "width": 100,
        "height": 50,
        "model_settings": {
            "use_table_recognition": True,
            "use_formula_recognition": True,
            "use_chart_recognition": True,
        },
        "overall_ocr_res": {
            "rec_texts": ["Title", "A > B"],
            "rec_scores": [0.99, 0.8],
            "rec_boxes": [[10, 2, 90, 12], [20, 20, 80, 35]],
            "rec_polys": [
                [[10, 2], [90, 2], [90, 12], [10, 12]],
                [[20, 20], [80, 20], [80, 35], [20, 35]],
            ],
        },
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": "A > B",
                "block_bbox": [20, 20, 80, 35],
                "block_id": 1,
                "block_order": 1,
            },
            {
                "block_label": "doc_title",
                "block_content": "Title",
                "block_bbox": [10, 2, 90, 12],
                "block_id": 0,
                "block_order": 0,
            },
        ],
        "layout_det_res": {
            "boxes": [
                {
                    "label": "doc_title",
                    "score": 0.95,
                    "coordinate": [10, 2, 90, 12],
                }
            ]
        },
    }


def config() -> dict:
    models = {name: name for name in load_module().EXPECTED_MODELS}
    models["text_detection"] = "PP-OCRv6_medium_det"
    models["text_recognition"] = "PP-OCRv6_medium_rec"
    return {
        "schema_version": 1,
        "packages": {
            "paddleocr": "3.7.0",
            "paddlex": "3.7.2",
            "paddlepaddle_gpu": "3.3.0",
        },
        "upstream": {"paddleocr_git_tag": "v3.7.0", "paddleocr_git_commit": "commit"},
        "inference_engine": "paddle_dynamic",
        "models": models,
        "modules": {
            "use_table_recognition": True,
            "use_formula_recognition": True,
            "use_chart_recognition": True,
            "use_region_detection": True,
            "format_block_content": True,
        },
    }


def test_parse_flat_and_structured_share_complete_provider_result():
    module = load_module()
    result = provider_result()
    flat = module.parse_flat_ocr(result, 100, 50)
    structured = module.parse_structured_text(result, 100, 50)
    assert flat["text"] == "Title\nA > B"
    assert flat["item_count"] == 2
    assert flat["items"][0]["bbox_norm"] == [0.1, 0.04, 0.9, 0.24]
    assert structured["compact_text"] == "[doc_title] Title\n[text] A > B"
    assert structured["label_counts"] == {"doc_title": 1, "text": 1}
    assert structured["blocks"][0]["provider_index"] == 0


def test_build_output_row_is_portable_and_source_only():
    module = load_module()
    row = input_row("talks/talk-a/frames/state_000.png", "frame-sha")
    output = module.build_output_row(
        row,
        provider_result(),
        config=config(),
        config_sha256="config-sha",
        input_manifest_sha256="input-sha",
        package_versions={"paddleocr": "3.7.0", "paddlex": "3.7.2", "paddlepaddle": "3.3.0"},
        model_manifest_sha256="model-sha",
        shard_index=0,
    )
    assert output["frame"]["path"] == row["frame_path"]
    assert output["provenance"]["config_sha256"] == "config-sha"
    assert output["provenance"]["model_manifest_sha256"] == "model-sha"
    assert "/remote/frame.png" not in json.dumps(output)
    assert not output["source_transcript_consumed"]
    assert not output["target_or_reference_consumed"]


@pytest.mark.parametrize("mutation", ["length", "score", "bbox", "polygon"])
def test_flat_ocr_fails_closed(mutation):
    module = load_module()
    result = provider_result()
    ocr = result["overall_ocr_res"]
    if mutation == "length":
        ocr["rec_scores"].pop()
    elif mutation == "score":
        ocr["rec_scores"][0] = 2.0
    elif mutation == "bbox":
        ocr["rec_boxes"][0] = [1, 1, 0, 0]
    elif mutation == "polygon":
        ocr["rec_polys"][0] = [[1, 2]]
    with pytest.raises(ValueError):
        module.parse_flat_ocr(result, 100, 50)


def test_validate_input_rows_checks_hash_dimensions_and_no_outcomes(tmp_path):
    module = load_module()
    frame = tmp_path / "talks" / "talk-a" / "frames" / "state_000.png"
    frame.parent.mkdir(parents=True)
    Image.new("RGB", (100, 50), "white").save(frame)
    row = input_row(frame.relative_to(tmp_path).as_posix(), sha256(frame))
    assert module.validate_input_rows([row], tmp_path)[row["id"]] == frame
    row["reference"] = {"translation": "target"}
    with pytest.raises(ValueError, match="forbidden"):
        module.validate_input_rows([row], tmp_path)


def test_config_rejects_document_vlm_or_disabled_structure():
    module = load_module()
    good = config()
    module.validate_config(good)
    bad = json.loads(json.dumps(good))
    bad["models"]["chart_recognition"] = "PaddleOCR-VL-1.6"
    with pytest.raises(ValueError, match="document VLM|OCR baseline"):
        module.validate_config(bad)
    bad = json.loads(json.dumps(good))
    bad["modules"]["use_chart_recognition"] = False
    with pytest.raises(ValueError, match="modules"):
        module.validate_config(bad)


def test_resume_rejects_changed_frame_binding(tmp_path):
    module = load_module()
    source = input_row("talks/talk-a/frames/state_000.png", "frame-sha")
    output = tmp_path / "shard.jsonl"
    output.write_text(
        json.dumps({"id": source["id"], "frame": {"sha256": "wrong"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="binding changed"):
        module.load_resume_rows(output, {source["id"]: source}, resume=True)


def test_model_manifest_must_bind_config_and_every_model():
    module = load_module()
    frozen = config()
    manifest = {
        "schema_version": 1,
        "config_sha256": module.canonical_hash(frozen),
        "models": frozen["models"],
        "unique_models": [
            {
                "requested_name": name,
                "resolved_name": f"{name}_safetensors",
                "tree_sha256": "a" * 64,
                "file_count": 1,
            }
            for name in sorted(set(frozen["models"].values()))
        ],
        "runtime_validations": {
            "chart_tied_embeddings": {"same_parameter": True}
        },
    }
    module.validate_model_manifest(manifest, frozen, module.canonical_hash(frozen))
    manifest["unique_models"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        module.validate_model_manifest(manifest, frozen, module.canonical_hash(frozen))
