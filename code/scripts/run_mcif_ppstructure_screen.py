#!/usr/bin/env python3
"""Run source-only flat OCR and structured text extraction over MCIF frames."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

from PIL import Image


EXPECTED_MODELS = {
    "layout_detection",
    "region_detection",
    "text_detection",
    "text_recognition",
    "textline_orientation",
    "table_classification",
    "wired_table_structure_recognition",
    "wireless_table_structure_recognition",
    "wired_table_cells_detection",
    "wireless_table_cells_detection",
    "table_orientation_classification",
    "formula_recognition",
    "chart_recognition",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported PP-Structure screen config schema")
    packages = config.get("packages") or {}
    if packages.get("paddleocr") != "3.7.0":
        raise ValueError("PaddleOCR must be pinned to 3.7.0")
    if packages.get("paddlex") != "3.7.2":
        raise ValueError("PaddleX must be pinned to 3.7.2")
    if packages.get("paddlepaddle_gpu") != "3.3.0":
        raise ValueError("PaddlePaddle GPU must be pinned to 3.3.0")
    models = config.get("models") or {}
    if set(models) != EXPECTED_MODELS:
        raise ValueError("PP-Structure model inventory is incomplete")
    if models.get("text_detection") != "PP-OCRv6_medium_det":
        raise ValueError("Flat and structured text must share PP-OCRv6 medium detection")
    if models.get("text_recognition") != "PP-OCRv6_medium_rec":
        raise ValueError("Flat and structured text must share PP-OCRv6 medium recognition")
    if any("PaddleOCR-VL" in str(value) for value in models.values()):
        raise ValueError("PaddleOCR-VL is not an OCR baseline")
    if config.get("inference_engine") != "paddle_dynamic":
        raise ValueError("The frozen screen uses paddle_dynamic for chart compatibility")
    modules = config.get("modules") or {}
    required_modules = {
        "use_table_recognition",
        "use_formula_recognition",
        "use_chart_recognition",
        "use_region_detection",
        "format_block_content",
    }
    if not all(modules.get(name) is True for name in required_modules):
        raise ValueError("Structured text modules must remain enabled")


def validate_model_manifest(
    manifest: dict[str, Any], config: dict[str, Any], config_sha256: str
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported PaddleX model manifest schema")
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError("PaddleX model manifest config binding changed")
    if manifest.get("models") != config["models"]:
        raise ValueError("PaddleX model manifest inventory changed")
    records = manifest.get("unique_models") or []
    names = [record.get("name") for record in records]
    if len(names) != len(set(names)) or set(names) != set(config["models"].values()):
        raise ValueError("PaddleX model manifest is incomplete")
    for record in records:
        if not isinstance(record.get("tree_sha256"), str) or len(record["tree_sha256"]) != 64:
            raise ValueError("PaddleX model manifest contains an invalid tree hash")
        if not isinstance(record.get("file_count"), int) or record["file_count"] <= 0:
            raise ValueError("PaddleX model manifest contains an empty model")


def validate_input_rows(
    rows: list[dict[str, Any]], frame_root: Path
) -> dict[str, Path]:
    ids = [row.get("id") for row in rows]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError("Native evidence manifest contains a missing id")
    duplicate_ids = [item_id for item_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Native evidence manifest contains duplicate ids: {duplicate_ids[:3]}")
    if any(row.get("source_transcript_consumed") is not False for row in rows):
        raise ValueError("Native evidence consumed source transcript")
    if any(row.get("target_or_reference_consumed") is not False for row in rows):
        raise ValueError("Native evidence consumed target or reference")
    forbidden = {
        "source_transcript",
        "reference",
        "reference_translation",
        "target_translation",
        "model_output",
        "audio",
    }
    if any(forbidden & set(row) for row in rows):
        raise ValueError("Native evidence manifest contains forbidden outcome fields")

    resolved_root = frame_root.resolve(strict=True)
    paths = {}
    for row in rows:
        relative = row.get("frame_path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"Invalid portable frame path for {row['id']}")
        path = (resolved_root / relative).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(resolved_root):
            raise ValueError(f"Frame path escapes the native evidence root for {row['id']}")
        if sha256_file(path) != row.get("frame_sha256"):
            raise ValueError(f"Native frame hash mismatch for {row['id']}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != (int(row["frame_width"]), int(row["frame_height"])):
                raise ValueError(f"Native frame dimensions mismatch for {row['id']}")
        if float(row["availability_start_sec"]) != float(row["evidence_timestamp_sec"]):
            raise ValueError(f"Visual evidence is backdated for {row['id']}")
        paths[row["id"]] = path
    return paths


def normalize_bbox(
    bbox: object, width: int, height: int, *, label: str
) -> tuple[list[float], list[float], bool]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid {label} bbox")
    values = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite {label} bbox")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Non-positive {label} bbox")
    out_of_bounds = x1 < 0 or y1 < 0 or x2 > width or y2 > height
    normalized = [
        round(min(max(x1 / width, 0.0), 1.0), 6),
        round(min(max(y1 / height, 0.0), 1.0), 6),
        round(min(max(x2 / width, 0.0), 1.0), 6),
        round(min(max(y2 / height, 0.0), 1.0), 6),
    ]
    return values, normalized, out_of_bounds


def normalize_polygon(polygon: object, *, label: str) -> list[list[float]]:
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 4:
        raise ValueError(f"Invalid {label} polygon")
    points = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Invalid {label} polygon point")
        values = [float(value) for value in point]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite {label} polygon point")
        points.append(values)
    return points


def compact_content(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    value = value.replace("\x00", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def parse_flat_ocr(result: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    ocr = result.get("overall_ocr_res") or {}
    texts = ocr.get("rec_texts") or []
    scores = ocr.get("rec_scores") or []
    boxes = ocr.get("rec_boxes") or []
    polygons = ocr.get("rec_polys") or []
    lengths = {len(texts), len(scores), len(boxes), len(polygons)}
    if len(lengths) != 1:
        raise ValueError("PP-OCR result field lengths differ")
    items = []
    for index, (text, score, bbox, polygon) in enumerate(
        zip(texts, scores, boxes, polygons, strict=True)
    ):
        text = compact_content(text)
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Invalid PP-OCR recognition score")
        bbox_px, bbox_norm, out_of_bounds = normalize_bbox(
            bbox, width, height, label="OCR"
        )
        items.append(
            {
                "provider_order": index,
                "text": text,
                "score": round(score, 8),
                "bbox_px": bbox_px,
                "bbox_norm": bbox_norm,
                "polygon_px": normalize_polygon(polygon, label="OCR"),
                "bbox_out_of_bounds": out_of_bounds,
            }
        )
    nonempty = [item["text"] for item in items if item["text"]]
    return {
        "ordering": "paddleocr_provider_order",
        "item_count": len(items),
        "nonempty_item_count": len(nonempty),
        "text": "\n".join(nonempty),
        "items": items,
    }


def parse_structured_text(
    result: dict[str, Any], width: int, height: int
) -> dict[str, Any]:
    raw_blocks = result.get("parsing_res_list") or []
    blocks = []
    for provider_index, raw in enumerate(raw_blocks):
        if not isinstance(raw, dict):
            raise ValueError("Invalid PP-Structure block")
        bbox_px, bbox_norm, out_of_bounds = normalize_bbox(
            raw.get("block_bbox"), width, height, label="structure"
        )
        order = raw.get("block_order")
        if order is not None:
            order = int(order)
        block_id = raw.get("block_id")
        blocks.append(
            {
                "provider_index": provider_index,
                "block_id": provider_index if block_id is None else int(block_id),
                "reading_order": order,
                "label": str(raw.get("block_label", "unknown")),
                "content": compact_content(raw.get("block_content")),
                "bbox_px": bbox_px,
                "bbox_norm": bbox_norm,
                "bbox_out_of_bounds": out_of_bounds,
            }
        )
    ordered = sorted(
        blocks,
        key=lambda block: (
            block["reading_order"] is None,
            block["reading_order"] if block["reading_order"] is not None else 0,
            block["provider_index"],
        ),
    )
    compact_lines = [
        f"[{block['label']}] {block['content']}"
        for block in ordered
        if block["content"]
    ]

    detections = []
    for index, raw in enumerate((result.get("layout_det_res") or {}).get("boxes") or []):
        bbox_px, bbox_norm, out_of_bounds = normalize_bbox(
            raw.get("coordinate"), width, height, label="layout detection"
        )
        score = float(raw.get("score", 0.0))
        if not math.isfinite(score):
            raise ValueError("Invalid layout score")
        detections.append(
            {
                "provider_index": index,
                "label": str(raw.get("label", "unknown")),
                "score": round(score, 8),
                "bbox_px": bbox_px,
                "bbox_norm": bbox_norm,
                "bbox_out_of_bounds": out_of_bounds,
            }
        )
    label_counts = dict(sorted(Counter(block["label"] for block in blocks).items()))
    return {
        "ordering": "ppstructurev3_reading_order_then_provider_order",
        "block_count": len(blocks),
        "nonempty_block_count": sum(bool(block["content"]) for block in blocks),
        "label_counts": label_counts,
        "compact_text": "\n".join(compact_lines),
        "blocks": blocks,
        "layout_detections": detections,
    }


def build_output_row(
    input_row: dict[str, Any],
    result: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha256: str,
    input_manifest_sha256: str,
    package_versions: dict[str, str],
    model_manifest_sha256: str,
    shard_index: int,
) -> dict[str, Any]:
    width = int(input_row["frame_width"])
    height = int(input_row["frame_height"])
    if int(result.get("width", -1)) != width or int(result.get("height", -1)) != height:
        raise ValueError(f"Provider image dimensions differ for {input_row['id']}")
    return {
        "id": input_row["id"],
        "lecture_id": input_row["lecture_id"],
        "state_id": input_row["state_id"],
        "availability_start_sec": input_row["availability_start_sec"],
        "availability_end_sec": input_row["availability_end_sec"],
        "evidence_timestamp_sec": input_row["evidence_timestamp_sec"],
        "frame": {
            "path": input_row["frame_path"],
            "sha256": input_row["frame_sha256"],
            "width": width,
            "height": height,
        },
        "provenance": {
            "provider": "PaddleOCR.PPStructureV3",
            "package_versions": package_versions,
            "paddleocr_git_tag": config["upstream"]["paddleocr_git_tag"],
            "paddleocr_git_commit": config["upstream"]["paddleocr_git_commit"],
            "inference_engine": config["inference_engine"],
            "models": config["models"],
            "model_manifest_sha256": model_manifest_sha256,
            "config_sha256": config_sha256,
            "input_manifest_sha256": input_manifest_sha256,
            "shard_index": shard_index,
        },
        "model_settings": result.get("model_settings") or {},
        "flat_ocr": parse_flat_ocr(result, width, height),
        "structured_text": parse_structured_text(result, width, height),
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
    }


def load_resume_rows(
    output_path: Path, selected: dict[str, dict[str, Any]], *, resume: bool
) -> set[str]:
    if not output_path.exists():
        return set()
    if not resume:
        raise FileExistsError(output_path)
    rows = load_jsonl(output_path)
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Resume output contains duplicate ids")
    if not set(ids) <= set(selected):
        raise ValueError("Resume output contains ids outside its shard")
    for row in rows:
        source = selected[row["id"]]
        frame = row.get("frame") or {}
        if frame.get("sha256") != source.get("frame_sha256"):
            raise ValueError(f"Resume frame binding changed for {row['id']}")
    return set(ids)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def create_pipeline(config: dict[str, Any], device: str):
    import paddle
    import paddleocr
    import paddlex
    from paddleocr import PPStructureV3

    expected = config["packages"]
    versions = {
        "paddleocr": paddleocr.__version__,
        "paddlex": paddlex.__version__,
        "paddlepaddle": paddle.__version__,
    }
    if versions["paddleocr"] != expected["paddleocr"]:
        raise ValueError(f"Unexpected PaddleOCR version: {versions['paddleocr']}")
    if versions["paddlex"] != expected["paddlex"]:
        raise ValueError(f"Unexpected PaddleX version: {versions['paddlex']}")
    if versions["paddlepaddle"] != expected["paddlepaddle_gpu"]:
        raise ValueError(f"Unexpected PaddlePaddle version: {versions['paddlepaddle']}")
    models = config["models"]
    modules = config["modules"]
    batch_sizes = config["batch_sizes"]
    pipeline = PPStructureV3(
        layout_detection_model_name=models["layout_detection"],
        region_detection_model_name=models["region_detection"],
        text_detection_model_name=models["text_detection"],
        textline_orientation_model_name=models["textline_orientation"],
        textline_orientation_batch_size=batch_sizes["textline_orientation"],
        text_recognition_model_name=models["text_recognition"],
        text_recognition_batch_size=batch_sizes["text_recognition"],
        table_classification_model_name=models["table_classification"],
        wired_table_structure_recognition_model_name=models[
            "wired_table_structure_recognition"
        ],
        wireless_table_structure_recognition_model_name=models[
            "wireless_table_structure_recognition"
        ],
        wired_table_cells_detection_model_name=models["wired_table_cells_detection"],
        wireless_table_cells_detection_model_name=models[
            "wireless_table_cells_detection"
        ],
        table_orientation_classify_model_name=models[
            "table_orientation_classification"
        ],
        formula_recognition_model_name=models["formula_recognition"],
        formula_recognition_batch_size=batch_sizes["formula_recognition"],
        chart_recognition_model_name=models["chart_recognition"],
        chart_recognition_batch_size=batch_sizes["chart_recognition"],
        text_rec_score_thresh=config["thresholds"]["text_recognition_score"],
        use_doc_orientation_classify=modules["use_doc_orientation_classify"],
        use_doc_unwarping=modules["use_doc_unwarping"],
        use_textline_orientation=modules["use_textline_orientation"],
        use_seal_recognition=modules["use_seal_recognition"],
        use_table_recognition=modules["use_table_recognition"],
        use_formula_recognition=modules["use_formula_recognition"],
        use_chart_recognition=modules["use_chart_recognition"],
        use_region_detection=modules["use_region_detection"],
        format_block_content=modules["format_block_content"],
        markdown_ignore_labels=modules["markdown_ignore_labels"],
        device=device,
        engine=config["inference_engine"],
    )
    return pipeline, versions


def result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict) and "error" in result:
        raise ValueError(f"PP-Structure returned an error: {result['error']}")
    payload = result.json["res"]
    if not isinstance(payload, dict):
        raise ValueError("PP-Structure result JSON is not an object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--resolved-config-out", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard assignment")

    config = load_json(args.config)
    validate_config(config)
    config_sha256 = canonical_hash(config)
    model_manifest = load_json(args.model_manifest)
    validate_model_manifest(model_manifest, config, config_sha256)
    model_manifest_sha256 = sha256_file(args.model_manifest)
    if os.environ.get("PADDLE_PDX_MODEL_SOURCE", "huggingface").lower() != config[
        "model_source"
    ]:
        raise ValueError("PaddleX model source differs from the frozen config")
    input_rows = load_jsonl(args.input_manifest)
    paths = validate_input_rows(input_rows, args.frame_root)
    selected_rows = [
        row for index, row in enumerate(input_rows) if index % args.num_shards == args.shard_index
    ]
    selected = {row["id"]: row for row in selected_rows}
    completed_ids = load_resume_rows(args.output, selected, resume=args.resume)
    pending = [row for row in selected_rows if row["id"] not in completed_ids]
    if args.failures.exists() and not args.resume:
        raise FileExistsError(args.failures)

    pipeline, package_versions = create_pipeline(config, args.device)
    if args.resolved_config_out.exists() and not args.resume:
        raise FileExistsError(args.resolved_config_out)
    args.resolved_config_out.parent.mkdir(parents=True, exist_ok=True)
    pipeline.export_paddlex_config_to_yaml(str(args.resolved_config_out))

    input_manifest_sha256 = sha256_file(args.input_manifest)
    batch_size = int(config["batch_sizes"]["input"])
    succeeded = len(completed_ids)
    failures = 0
    for batch in chunks(pending, batch_size):
        batch_paths = [str(paths[row["id"]]) for row in batch]
        try:
            results = pipeline.predict(batch_paths)
            payloads = [result_payload(result) for result in results]
            if len(payloads) != len(batch):
                raise ValueError("PP-Structure batch result count differs from input")
            by_path = {str(Path(payload["input_path"]).resolve()): payload for payload in payloads}
            for row, path in zip(batch, batch_paths, strict=True):
                payload = by_path.get(str(Path(path).resolve()))
                if payload is None:
                    raise ValueError(f"Missing PP-Structure output for {row['id']}")
                output_row = build_output_row(
                    row,
                    payload,
                    config=config,
                    config_sha256=config_sha256,
                    input_manifest_sha256=input_manifest_sha256,
                    package_versions=package_versions,
                    model_manifest_sha256=model_manifest_sha256,
                    shard_index=args.shard_index,
                )
                append_jsonl(args.output, output_row)
                succeeded += 1
        except Exception as exc:
            failures += len(batch)
            for row in batch:
                append_jsonl(
                    args.failures,
                    {
                        "id": row["id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
        print(
            json.dumps(
                {
                    "shard_index": args.shard_index,
                    "succeeded": succeeded,
                    "failed": failures,
                    "selected": len(selected_rows),
                }
            ),
            flush=True,
        )
    pipeline.close()
    if failures:
        raise RuntimeError(f"PP-Structure screen recorded {failures} failed rows")
    if succeeded != len(selected_rows):
        raise RuntimeError("PP-Structure screen did not complete its shard")


if __name__ == "__main__":
    main()
