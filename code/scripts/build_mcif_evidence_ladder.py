#!/usr/bin/env python3
"""Build the source-only MCIF R0/R1/R2 evidence ladder."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

from PIL import Image


GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_TAG_RE = re.compile(r"<img\b", re.IGNORECASE)
NATIVE_FRAME_SUBDIR = "frames"
FORBIDDEN_KEYS = {
    "audio",
    "model_output",
    "reference",
    "reference_text",
    "reference_translation",
    "source_transcript",
    "target",
    "target_text",
    "target_translation",
    "translation",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_head_clean(repo: Path) -> str:
    resolved = repo.resolve(strict=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolved,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=resolved,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not GIT_SHA_RE.fullmatch(commit) or status:
        raise ValueError("Ladder builder requires a clean Git checkout at a full commit")
    return commit


def unique_by_id(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{label} contains a missing id")
        if item_id in result:
            raise ValueError(f"{label} contains duplicate id {item_id}")
        result[item_id] = row
    return result


def find_forbidden_keys(value: object, path: str = "$") -> list[str]:
    findings = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key.casefold() in FORBIDDEN_KEYS:
                findings.append(child_path)
            findings.extend(find_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_forbidden_keys(child, f"{path}[{index}]"))
    return findings


def resolve_regular_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Evidence frame path must be canonical and relative")
    if root.is_symlink():
        raise ValueError("Evidence root cannot be a symlink")
    resolved_root = root.resolve(strict=True)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Evidence frame path traverses a symlink")
    resolved = (root / relative).resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
        raise ValueError("Evidence frame path escapes the native artifact root")
    return resolved


def validate_provenance_manifests(
    native_provenance: dict[str, Any],
    ppstructure_provenance: dict[str, Any],
    *,
    native_manifest_sha256: str,
    ppstructure_output_sha256: str,
) -> dict[str, Any]:
    native_hf = native_provenance.get("hugging_face") or {}
    ppstructure_hf = ppstructure_provenance.get("hugging_face") or {}
    for label, manifest in (
        ("native", native_provenance),
        ("PPStructure", ppstructure_provenance),
    ):
        inventory = manifest.get("inventory") or {}
        if inventory.get("source_transcript_consumed") is not False or inventory.get(
            "target_or_reference_consumed"
        ) is not False:
            raise ValueError(f"{label} provenance consumed a forbidden source or outcome")
    if native_provenance.get("quality_audit", {}).get("manifest_sha256") != (
        native_manifest_sha256
    ):
        raise ValueError("Native provenance does not bind the supplied manifest")
    if ppstructure_provenance.get("checksums", {}).get("output_sha256") != (
        ppstructure_output_sha256
    ):
        raise ValueError("PPStructure provenance does not bind the supplied output")
    if ppstructure_provenance.get("upstream", {}).get(
        "native_evidence_manifest_sha256"
    ) != native_manifest_sha256:
        raise ValueError("PPStructure provenance binds a different native manifest")
    if native_hf.get("repo") != ppstructure_hf.get("repo"):
        raise ValueError("Native and PPStructure artifacts use different HF repos")
    for label, value in (("native", native_hf), ("PPStructure", ppstructure_hf)):
        artifact_path = Path(str(value.get("path", "")))
        if (
            value.get("repo_type") != "dataset"
            or value.get("private_verified") is not True
            or not GIT_SHA_RE.fullmatch(str(value.get("revision", "")))
            or not isinstance(value.get("path"), str)
            or artifact_path.is_absolute()
            or ".." in artifact_path.parts
            or artifact_path.as_posix() != value.get("path")
        ):
            raise ValueError(f"{label} HF provenance is not immutable and private")
    if native_provenance.get("upstream", {}).get("revision") != (
        ppstructure_provenance.get("upstream", {}).get("revision")
    ):
        raise ValueError("Native and PPStructure artifacts use different MCIF revisions")
    return {
        "dataset_repo": native_hf["repo"],
        "native_revision": native_hf["revision"],
        "native_path": native_hf["path"],
        "ppstructure_revision": ppstructure_hf["revision"],
        "ppstructure_path": ppstructure_hf["path"],
        "upstream_revision": native_provenance["upstream"]["revision"],
        "upstream_license": native_provenance["upstream"]["license"],
    }


def validate_native_inventory(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    expected_talks: int,
) -> None:
    if len(rows) != expected_rows:
        raise ValueError("Native manifest row count differs from the frozen inventory")
    if len({row.get("lecture_id") for row in rows}) != expected_talks:
        raise ValueError("Native manifest talk count differs from the frozen inventory")
    observed_order = [(row.get("lecture_id"), row.get("state_id")) for row in rows]
    if observed_order != sorted(observed_order):
        raise ValueError("Native manifest is not in canonical talk/state order")
    by_talk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_talk[str(row.get("lecture_id"))].append(row)
    for talk_id, talk_rows in by_talk.items():
        if [row.get("state_id") for row in talk_rows] != list(range(len(talk_rows))):
            raise ValueError(f"Non-contiguous native state ids for {talk_id}")
        if float(talk_rows[0].get("availability_start_sec", -1)) != 0.5:
            raise ValueError(f"Native evidence does not preserve the initial gap for {talk_id}")
        for previous, current in zip(talk_rows, talk_rows[1:]):
            if abs(
                float(previous["availability_end_sec"])
                - float(current["availability_start_sec"])
            ) > 1e-6:
                raise ValueError(f"Non-contiguous native availability for {talk_id}")
        for row in talk_rows:
            if float(row["availability_start_sec"]) != float(
                row["evidence_timestamp_sec"]
            ):
                raise ValueError(f"Native evidence is backdated for {row.get('id')}")
            if float(row["availability_end_sec"]) <= float(
                row["availability_start_sec"]
            ):
                raise ValueError(f"Non-positive native interval for {row.get('id')}")


def normalize_block(block: dict[str, Any]) -> dict[str, Any]:
    label = block.get("label")
    content = block.get("content")
    bbox = block.get("bbox_norm")
    if not isinstance(label, str) or not label:
        raise ValueError("Structured block has no label")
    if not isinstance(content, str):
        raise ValueError("Structured block content is not text")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(not isinstance(value, (int, float)) for value in bbox)
        or any(float(value) < 0 or float(value) > 1 for value in bbox)
    ):
        raise ValueError("Structured block has an invalid normalized box")
    is_visual_placeholder = bool(IMAGE_TAG_RE.search(content))
    if is_visual_placeholder:
        normalized_content = (
            "[unserialized table visual region]"
            if label == "table"
            else "[non-text visual region]"
        )
        content_kind = "visual_placeholder"
    elif label == "chart" and "|" in content:
        normalized_content = content.strip()
        content_kind = "chart_markdown"
    elif label == "table" and "<table" in content.casefold():
        normalized_content = content.strip()
        content_kind = "table_html"
    elif label == "formula":
        normalized_content = content.strip()
        content_kind = "formula_latex"
    else:
        normalized_content = content.strip()
        content_kind = "text"
    return {
        "block_id": block.get("block_id"),
        "provider_index": block.get("provider_index"),
        "reading_order": block.get("reading_order"),
        "label": label,
        "bbox_norm": [round(float(value), 6) for value in bbox],
        "content_kind": content_kind,
        "content": normalized_content,
    }


def structured_order_key(block: dict[str, Any]) -> tuple[int, int, int]:
    reading_order = block.get("reading_order")
    provider_index = block.get("provider_index")
    if reading_order is not None and (
        not isinstance(reading_order, int) or reading_order < 0
    ):
        raise ValueError("Structured block has an invalid reading order")
    if not isinstance(provider_index, int) or provider_index < 0:
        raise ValueError("Structured block has an invalid provider index")
    return (
        int(reading_order is None),
        provider_index if reading_order is None else reading_order,
        provider_index,
    )


def length_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty length inventory")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    return {
        "min": ordered[0],
        "median": median,
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def render_structured_text(
    blocks: list[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> str:
    heading = f"Slide canvas: {width}x{height}\nStructured blocks (JSONL):"
    if not blocks:
        return heading + "\n[no structured text extracted]"
    return heading + "\n" + "\n".join(
        json.dumps(block, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for block in blocks
    )


def validate_and_build_row(
    native: dict[str, Any],
    ppstructure: dict[str, Any],
    *,
    native_root: Path,
    native_hf_path: str,
    source_binding_sha256: str,
    expected_native_manifest_sha256: str,
) -> dict[str, Any]:
    item_id = native["id"]
    protected = (
        "id",
        "lecture_id",
        "state_id",
        "availability_start_sec",
        "availability_end_sec",
        "evidence_timestamp_sec",
    )
    if any(ppstructure.get(key) != native.get(key) for key in protected):
        raise ValueError(f"PPStructure timing/identity differs from native row {item_id}")
    if native.get("source_transcript_consumed") is not False or native.get(
        "target_or_reference_consumed"
    ) is not False:
        raise ValueError(f"Native row consumed a forbidden source or outcome for {item_id}")
    if ppstructure.get("source_transcript_consumed") is not False or ppstructure.get(
        "target_or_reference_consumed"
    ) is not False:
        raise ValueError(f"PPStructure row consumed a forbidden source or outcome for {item_id}")
    forbidden = find_forbidden_keys(native) + find_forbidden_keys(ppstructure)
    if forbidden:
        raise ValueError(f"Forbidden source/outcome fields in {item_id}: {forbidden[:3]}")

    frame = ppstructure.get("frame") or {}
    expected_frame = {
        "path": native.get("frame_path"),
        "sha256": native.get("frame_sha256"),
        "width": native.get("frame_width"),
        "height": native.get("frame_height"),
    }
    if frame != expected_frame:
        raise ValueError(f"PPStructure frame binding differs from native row {item_id}")
    frame_path = resolve_regular_file(native_root, native["frame_path"])
    if file_sha256(frame_path) != native["frame_sha256"]:
        raise ValueError(f"Native image bytes changed for {item_id}")
    with Image.open(frame_path) as image:
        image.verify()
    with Image.open(frame_path) as image:
        observed_size = image.size
    if observed_size != (native["frame_width"], native["frame_height"]):
        raise ValueError(f"Native image dimensions changed for {item_id}")

    provenance = ppstructure.get("provenance") or {}
    if provenance.get("input_manifest_sha256") != expected_native_manifest_sha256:
        raise ValueError(f"PPStructure row binds a different native manifest for {item_id}")
    if provenance.get("provider") != "PaddleOCR.PPStructureV3":
        raise ValueError(f"Unexpected PPStructure provider for {item_id}")
    flat = ppstructure.get("flat_ocr") or {}
    structured = ppstructure.get("structured_text") or {}
    flat_items = flat.get("items") or []
    source_blocks = structured.get("blocks") or []
    if flat.get("item_count") != len(flat_items):
        raise ValueError(f"Flat OCR count mismatch for {item_id}")
    if structured.get("block_count") != len(source_blocks):
        raise ValueError(f"Structured block count mismatch for {item_id}")
    if not isinstance(flat.get("text"), str):
        raise ValueError(f"Flat OCR text is invalid for {item_id}")
    ordered_source_blocks = sorted(source_blocks, key=structured_order_key)
    normalized_blocks = [normalize_block(block) for block in ordered_source_blocks]
    structured_text = render_structured_text(
        normalized_blocks,
        width=native["frame_width"],
        height=native["frame_height"],
    )
    row = {
        "schema_version": "mcif_source_evidence_ladder_v1",
        "id": item_id,
        "lecture_id": native["lecture_id"],
        "state_id": native["state_id"],
        "availability_start_sec": native["availability_start_sec"],
        "availability_end_sec": native["availability_end_sec"],
        "evidence_timestamp_sec": native["evidence_timestamp_sec"],
        "r0_flat_ocr": {
            "model_input_text": flat["text"],
            "item_count": flat["item_count"],
            "ordering": flat.get("ordering"),
            "source_items_sha256": canonical_sha256(flat_items),
        },
        "r1_structured_text": {
            "model_input_text": structured_text,
            "block_count": len(normalized_blocks),
            "ordering": structured.get("ordering"),
            "blocks": normalized_blocks,
            "source_blocks_sha256": canonical_sha256(source_blocks),
        },
        "r2_raw_image": {
            "source_media_path": (
                Path(native_hf_path) / NATIVE_FRAME_SUBDIR / native["frame_path"]
            ).as_posix(),
            "source_media_sha256": native["frame_sha256"],
            "width": native["frame_width"],
            "height": native["frame_height"],
        },
        "source_binding_sha256": source_binding_sha256,
        "source_transcript_consumed": False,
        "target_or_reference_consumed": False,
        "automatic_source_evidence_not_annotation": True,
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def build_ladder(
    native_rows: list[dict[str, Any]],
    ppstructure_rows: list[dict[str, Any]],
    *,
    native_root: Path,
    native_provenance: dict[str, Any],
    ppstructure_provenance: dict[str, Any],
    native_manifest_sha256: str,
    ppstructure_output_sha256: str,
    native_provenance_sha256: str,
    ppstructure_provenance_sha256: str,
    builder_git_commit: str,
    expected_rows: int,
    expected_talks: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not GIT_SHA_RE.fullmatch(builder_git_commit):
        raise ValueError("Builder Git revision must be a full commit")
    validate_native_inventory(
        native_rows,
        expected_rows=expected_rows,
        expected_talks=expected_talks,
    )
    native_by_id = unique_by_id(native_rows, "native manifest")
    ppstructure_by_id = unique_by_id(ppstructure_rows, "PPStructure output")
    if set(native_by_id) != set(ppstructure_by_id):
        missing = sorted(set(native_by_id) - set(ppstructure_by_id))
        extra = sorted(set(ppstructure_by_id) - set(native_by_id))
        raise ValueError(f"Evidence id matrix mismatch: missing={missing[:3]} extra={extra[:3]}")
    hf = validate_provenance_manifests(
        native_provenance,
        ppstructure_provenance,
        native_manifest_sha256=native_manifest_sha256,
        ppstructure_output_sha256=ppstructure_output_sha256,
    )
    source_binding = {
        "native_manifest_sha256": native_manifest_sha256,
        "ppstructure_output_sha256": ppstructure_output_sha256,
        "native_provenance_sha256": native_provenance_sha256,
        "ppstructure_provenance_sha256": ppstructure_provenance_sha256,
        "native_hf_revision": hf["native_revision"],
        "ppstructure_hf_revision": hf["ppstructure_revision"],
    }
    source_binding_sha256 = canonical_sha256(source_binding)
    rows = [
        validate_and_build_row(
            native,
            ppstructure_by_id[native["id"]],
            native_root=native_root,
            native_hf_path=hf["native_path"],
            source_binding_sha256=source_binding_sha256,
            expected_native_manifest_sha256=native_manifest_sha256,
        )
        for native in native_rows
    ]
    content_kinds = Counter(
        block["content_kind"]
        for row in rows
        for block in row["r1_structured_text"]["blocks"]
    )
    r0_lengths = [len(row["r0_flat_ocr"]["model_input_text"]) for row in rows]
    r1_lengths = [len(row["r1_structured_text"]["model_input_text"]) for row in rows]
    report = {
        "schema_version": "mcif_source_evidence_ladder_report_v1",
        "status": "SOURCE_ONLY_R0_R1_R2_INPUT_NOT_ANNOTATION_OR_ST_RESULT",
        "builder_git_commit": builder_git_commit,
        "rows": len(rows),
        "talks": len({row["lecture_id"] for row in rows}),
        "r0_flat_ocr_items": sum(
            row["r0_flat_ocr"]["item_count"] for row in rows
        ),
        "r0_rows_with_text": sum(
            bool(row["r0_flat_ocr"]["model_input_text"].strip()) for row in rows
        ),
        "r1_structured_blocks": sum(
            row["r1_structured_text"]["block_count"] for row in rows
        ),
        "r1_content_kind_counts": dict(sorted(content_kinds.items())),
        "model_input_character_lengths": {
            "r0_flat_ocr": length_summary(r0_lengths),
            "r1_structured_text": length_summary(r1_lengths),
        },
        "r2_raw_image_rows": len(rows),
        "source_binding": source_binding,
        "source_binding_sha256": source_binding_sha256,
        "hugging_face_sources": hf,
        "upstream_source_transcript_consumed": False,
        "target_or_reference_consumed": False,
        "interpretation": (
            "R0 is flat provider-order OCR text without boxes. R1 is a text-only "
            "serialization of PP-Structure labels, normalized boxes, reading order, "
            "and machine-readable chart/table/formula content; image tags are replaced "
            "by explicit non-text placeholders. R2 references the identical native "
            "causal PNG. No tier is an image-needed label or translation result."
        ),
    }
    return rows, report


def write_bundle(
    output_root: Path,
    *,
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    native_provenance_path: Path,
    ppstructure_provenance_path: Path,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Evidence ladder output root must not already exist")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        provenance_root = temporary / "provenance"
        provenance_root.mkdir()
        ladder_path = temporary / "ladder.jsonl"
        ladder_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        final_report = dict(report)
        final_report["ladder_sha256"] = file_sha256(ladder_path)
        final_report["row_binding_set_sha256"] = canonical_sha256(
            [{"id": row["id"], "row_sha256": row["row_sha256"]} for row in rows]
        )
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(
            native_provenance_path,
            provenance_root / "mcif_native_causal_evidence_v1_20260801.json",
        )
        shutil.copyfile(
            ppstructure_provenance_path,
            provenance_root / "mcif_ppstructurev3_source_screen_v1_20260801.json",
        )
        readme = temporary / "README.md"
        readme.write_text(
            "# MCIF Source Evidence Ladder V1\n\n"
            "Private source-only automatic input artifact. It contains 304 matched "
            "R0 flat-OCR, R1 structured-text, and R2 raw-image references. It contains "
            "no transcript, audio, target, reference, annotation, or ST output.\n\n"
            f"- rows / talks: {final_report['rows']} / {final_report['talks']}\n"
            f"- ladder SHA256: `{final_report['ladder_sha256']}`\n"
            f"- source binding: `{final_report['source_binding_sha256']}`\n"
            f"- raw images: `{final_report['hugging_face_sources']['dataset_repo']}@"
            f"{final_report['hugging_face_sources']['native_revision']}/"
            f"{final_report['hugging_face_sources']['native_path']}`\n\n"
            "R1 preserves labels, normalized boxes, reading order, and recognized "
            "chart/table/formula serialization. Image tags are replaced by explicit "
            "non-text placeholders. These automatic tiers must not filter states or "
            "serve as image-needed labels.\n",
            encoding="utf-8",
        )
        checksum_paths = sorted(
            path
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        checksum_path = temporary / "SHA256SUMS"
        checksum_path.write_text(
            "".join(
                f"{file_sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in checksum_paths
            ),
            encoding="utf-8",
        )
        os.rename(temporary, output_root)
        return {
            **final_report,
            "checksum_manifest_sha256": file_sha256(output_root / "SHA256SUMS"),
            "checksum_entries": len(checksum_paths),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--native-provenance", type=Path, required=True)
    parser.add_argument("--ppstructure-output", type=Path, required=True)
    parser.add_argument("--ppstructure-provenance", type=Path, required=True)
    parser.add_argument("--code-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=304)
    parser.add_argument("--expected-talks", type=int, default=21)
    args = parser.parse_args()
    if args.expected_rows <= 0 or args.expected_talks <= 0:
        raise ValueError("Expected inventory sizes must be positive")
    builder_git_commit = git_head_clean(args.code_repo)
    native_manifest_sha256 = file_sha256(args.native_manifest)
    ppstructure_output_sha256 = file_sha256(args.ppstructure_output)
    rows, report = build_ladder(
        load_jsonl(args.native_manifest),
        load_jsonl(args.ppstructure_output),
        native_root=args.native_root,
        native_provenance=load_json(args.native_provenance),
        ppstructure_provenance=load_json(args.ppstructure_provenance),
        native_manifest_sha256=native_manifest_sha256,
        ppstructure_output_sha256=ppstructure_output_sha256,
        native_provenance_sha256=file_sha256(args.native_provenance),
        ppstructure_provenance_sha256=file_sha256(args.ppstructure_provenance),
        builder_git_commit=builder_git_commit,
        expected_rows=args.expected_rows,
        expected_talks=args.expected_talks,
    )
    final_report = write_bundle(
        args.output_root,
        rows=rows,
        report=report,
        native_provenance_path=args.native_provenance,
        ppstructure_provenance_path=args.ppstructure_provenance,
    )
    print(json.dumps(final_report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
