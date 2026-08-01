#!/usr/bin/env python3
"""Finalize and audit MCIF flat OCR and structured text shard outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


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


def unique_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    ids = [row.get("id") for row in rows]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError(f"{label} contains a missing id")
    duplicates = [item_id for item_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"{label} contains duplicate ids: {duplicates[:3]}")
    return {row["id"]: row for row in rows}


def assert_no_failures(paths: list[Path]) -> None:
    for path in paths:
        if path.exists() and load_jsonl(path):
            raise ValueError(f"Nonempty PP-Structure failure file: {path}")


def validate_row(
    source: dict[str, Any],
    output: dict[str, Any],
    *,
    expected_config_sha256: str,
    expected_input_manifest_sha256: str,
    expected_model_manifest_sha256: str,
    expected_models: dict[str, str],
) -> None:
    item_id = source["id"]
    protected = (
        "id",
        "lecture_id",
        "state_id",
        "availability_start_sec",
        "availability_end_sec",
        "evidence_timestamp_sec",
    )
    for key in protected:
        if output.get(key) != source.get(key):
            raise ValueError(f"Protected field {key} changed for {item_id}")
    frame = output.get("frame") or {}
    expected_frame = {
        "path": source["frame_path"],
        "sha256": source["frame_sha256"],
        "width": source["frame_width"],
        "height": source["frame_height"],
    }
    if frame != expected_frame:
        raise ValueError(f"Frame binding changed for {item_id}")
    if Path(frame["path"]).is_absolute() or ".." in Path(frame["path"]).parts:
        raise ValueError(f"Nonportable frame path for {item_id}")
    if output.get("source_transcript_consumed") is not False:
        raise ValueError(f"Source transcript was consumed for {item_id}")
    if output.get("target_or_reference_consumed") is not False:
        raise ValueError(f"Target or reference was consumed for {item_id}")
    forbidden = {
        "source_transcript",
        "reference",
        "reference_translation",
        "target_translation",
        "model_output",
        "audio",
    }
    if forbidden & set(output):
        raise ValueError(f"Forbidden outcome field in {item_id}")

    provenance = output.get("provenance") or {}
    if provenance.get("provider") != "PaddleOCR.PPStructureV3":
        raise ValueError(f"Unexpected provider for {item_id}")
    if provenance.get("config_sha256") != expected_config_sha256:
        raise ValueError(f"Config binding changed for {item_id}")
    if provenance.get("input_manifest_sha256") != expected_input_manifest_sha256:
        raise ValueError(f"Input manifest binding changed for {item_id}")
    if provenance.get("models") != expected_models:
        raise ValueError(f"Model inventory changed for {item_id}")
    if provenance.get("model_manifest_sha256") != expected_model_manifest_sha256:
        raise ValueError(f"Model file binding changed for {item_id}")
    versions = provenance.get("package_versions") or {}
    if versions.get("paddleocr") != "3.7.0":
        raise ValueError(f"Unexpected PaddleOCR version for {item_id}")
    if versions.get("paddlepaddle") != "3.3.0":
        raise ValueError(f"Unexpected PaddlePaddle version for {item_id}")

    flat = output.get("flat_ocr") or {}
    structured = output.get("structured_text") or {}
    if flat.get("item_count") != len(flat.get("items") or []):
        raise ValueError(f"Flat OCR item count mismatch for {item_id}")
    if structured.get("block_count") != len(structured.get("blocks") or []):
        raise ValueError(f"Structured block count mismatch for {item_id}")
    if any(not isinstance(item.get("text"), str) for item in flat.get("items") or []):
        raise ValueError(f"Invalid flat OCR item in {item_id}")
    if any(
        not isinstance(block.get("content"), str)
        for block in structured.get("blocks") or []
    ):
        raise ValueError(f"Invalid structure block in {item_id}")


def finalize(
    input_rows: list[dict[str, Any]],
    shard_rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    input_manifest_sha256: str,
    model_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_by_id = unique_by_id(input_rows, "input")
    output_by_id = unique_by_id(shard_rows, "shards")
    if set(input_by_id) != set(output_by_id):
        missing = sorted(set(input_by_id) - set(output_by_id))
        extra = sorted(set(output_by_id) - set(input_by_id))
        raise ValueError(f"Shard id set mismatch; missing={missing[:3]} extra={extra[:3]}")

    config_sha256 = canonical_hash(config)
    label_rows = Counter()
    label_blocks = Counter()
    flat_item_count = 0
    structured_block_count = 0
    rows_with_flat_text = 0
    rows_with_structured_text = 0
    flat_bbox_out_of_bounds = 0
    structure_bbox_out_of_bounds = 0
    shard_counts = Counter()
    package_version_counts = Counter()
    finalized = []
    for source in input_rows:
        output = output_by_id[source["id"]]
        validate_row(
            source,
            output,
            expected_config_sha256=config_sha256,
            expected_input_manifest_sha256=input_manifest_sha256,
            expected_model_manifest_sha256=model_manifest_sha256,
            expected_models=config["models"],
        )
        flat = output["flat_ocr"]
        structured = output["structured_text"]
        rows_with_flat_text += bool(flat.get("text"))
        rows_with_structured_text += bool(structured.get("compact_text"))
        flat_item_count += int(flat["item_count"])
        structured_block_count += int(structured["block_count"])
        flat_bbox_out_of_bounds += sum(
            bool(item.get("bbox_out_of_bounds")) for item in flat["items"]
        )
        structure_bbox_out_of_bounds += sum(
            bool(block.get("bbox_out_of_bounds")) for block in structured["blocks"]
        )
        row_labels = set()
        for block in structured["blocks"]:
            label = block["label"]
            label_blocks[label] += 1
            row_labels.add(label)
        label_rows.update(row_labels)
        provenance = output["provenance"]
        shard_counts[str(provenance["shard_index"])] += 1
        package_version_counts[
            json.dumps(provenance["package_versions"], sort_keys=True)
        ] += 1
        finalized.append(output)

    report = {
        "dataset": "mcif",
        "artifact": "flat_ocr_and_structured_text_source_screen_v1",
        "status": "COMPLETE_SOURCE_ONLY_AUTOMATIC_DIAGNOSTIC_NOT_TRANSLATION_RESULT",
        "rows": len(finalized),
        "unique_ids": len(output_by_id),
        "talk_count": len({row["lecture_id"] for row in finalized}),
        "input_manifest_sha256": input_manifest_sha256,
        "config_sha256": config_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "flat_ocr": {
            "rows_with_text": rows_with_flat_text,
            "item_count": flat_item_count,
            "bbox_out_of_bounds_count": flat_bbox_out_of_bounds,
        },
        "structured_text": {
            "rows_with_text": rows_with_structured_text,
            "block_count": structured_block_count,
            "bbox_out_of_bounds_count": structure_bbox_out_of_bounds,
            "row_counts_by_label": dict(sorted(label_rows.items())),
            "block_counts_by_label": dict(sorted(label_blocks.items())),
            "rows_with_table": label_rows["table"],
            "rows_with_formula": label_rows["formula"],
            "rows_with_chart": label_rows["chart"],
        },
        "shard_counts": dict(sorted(shard_counts.items())),
        "package_version_counts": dict(sorted(package_version_counts.items())),
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
        "interpretation": (
            "The artifact provides matched flat PP-OCRv6 text and PP-StructureV3 "
            "document structure over identical native causal frames. It is automatic "
            "input evidence, not a label, translation result, or proof that pixels are needed."
        ),
    }
    return finalized, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--failure-file", type=Path, action="append", default=[])
    parser.add_argument("--resolved-config", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=304)
    parser.add_argument("--expected-talks", type=int, default=21)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Final PP-Structure outputs must be created once")
    assert_no_failures(args.failure_file)
    if args.resolved_config:
        hashes = {sha256_file(path) for path in args.resolved_config}
        if len(hashes) != 1:
            raise ValueError("Resolved PaddleX configs differ across shards")

    input_rows = load_jsonl(args.input_manifest)
    shard_rows = []
    for path in args.shard:
        shard_rows.extend(load_jsonl(path))
    finalized, report = finalize(
        input_rows,
        shard_rows,
        config=load_json(args.config),
        input_manifest_sha256=sha256_file(args.input_manifest),
        model_manifest_sha256=sha256_file(args.model_manifest),
    )
    if len(finalized) != args.expected_rows:
        raise ValueError("Final PP-Structure row count differs from expectation")
    if report["talk_count"] != args.expected_talks:
        raise ValueError("Final PP-Structure talk count differs from expectation")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in finalized
        ),
        encoding="utf-8",
    )
    report["output_sha256"] = sha256_file(args.output)
    report["git_commit"] = args.git_commit
    if args.resolved_config:
        report["resolved_config_sha256"] = sha256_file(args.resolved_config[0])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
