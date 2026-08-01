#!/usr/bin/env python3
"""Materialize processor-matched MCIF wrong-image media without target data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable

from PIL import Image, __version__ as pillow_version

from scripts.build_mcif_visual_token_controls import (
    canonical_sha256,
    file_sha256,
    git_head_clean,
    load_jsonl,
    prepare_control_image,
    processor_file_manifest,
    qwen_image_messages,
    resolve_regular_file,
    row_values,
    unique_by_id,
    verify_frozen_inputs,
)


BUNDLE_PATH_PREFIX = "visual_control_media_v1"
CONTROL_FAMILIES = ("same_talk_stale", "cross_talk_wrong")


def validate_control_inputs(
    inventory: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    processor_manifest: dict[str, Any],
    *,
    expected_rows: int,
    expected_talks: int,
) -> None:
    inventory_by_id = unique_by_id(inventory, "visual-token inventory")
    controls_by_id = unique_by_id(controls, "wrong-image controls")
    if (
        len(inventory) != expected_rows
        or len(controls) != expected_rows
        or set(inventory_by_id) != set(controls_by_id)
    ):
        raise ValueError("Visual-token inventory and controls differ from frozen rows")
    if len({row.get("lecture_id") for row in inventory}) != expected_talks:
        raise ValueError("Visual-token inventory talk count differs from contract")
    if [row["id"] for row in inventory] != [row["id"] for row in controls]:
        raise ValueError("Visual-token inventory and controls are not canonically aligned")
    processor_binding = processor_manifest.get("processor_binding_sha256")
    if not isinstance(processor_binding, str):
        raise ValueError("Processor manifest has no binding")
    for source, control in zip(inventory, controls, strict=True):
        if source.get("schema_version") != "mcif_qwen3_omni_visual_token_inventory_v1":
            raise ValueError("Unexpected visual-token inventory schema")
        if control.get("schema_version") != "mcif_qwen3_omni_wrong_image_candidates_v1":
            raise ValueError("Unexpected wrong-image control schema")
        if source.get("row_sha256") != canonical_sha256(
            {key: value for key, value in source.items() if key != "row_sha256"}
        ):
            raise ValueError(f"Visual-token inventory row hash mismatch: {source.get('id')}")
        if control.get("row_sha256") != canonical_sha256(
            {key: value for key, value in control.items() if key != "row_sha256"}
        ):
            raise ValueError(f"Wrong-image control row hash mismatch: {control.get('id')}")
        if source.get("processor_binding_sha256") != processor_binding:
            raise ValueError("Inventory processor binding differs from manifest")
        if control.get("inventory_row_sha256") != source["row_sha256"]:
            raise ValueError("Wrong-image control is not bound to its source inventory row")
        if source.get("source_transcript_consumed") is not False or source.get(
            "target_or_reference_consumed"
        ) is not False:
            raise ValueError("Visual-token inventory consumed forbidden data")
        if control.get("source_transcript_consumed") is not False or control.get(
            "target_or_reference_consumed"
        ) is not False:
            raise ValueError("Wrong-image control consumed forbidden data")
        for family in CONTROL_FAMILIES:
            candidate = control.get(family)
            if candidate is None:
                if family == "cross_talk_wrong":
                    raise ValueError("Every source state requires a cross-talk control")
                continue
            candidate_source = inventory_by_id.get(candidate.get("id"))
            if candidate_source is None:
                raise ValueError(f"Control candidate is absent from inventory: {source['id']}")
            expected_candidate = {
                "id": candidate_source["id"],
                "lecture_id": candidate_source["lecture_id"],
                "state_id": candidate_source["state_id"],
                "source_media_path": candidate_source["source_media_path"],
                "source_media_sha256": candidate_source["source_media_sha256"],
                "width": candidate_source["width"],
                "height": candidate_source["height"],
                "image_grid_thw": candidate_source["image_grid_thw"],
                "visual_token_count": candidate_source["visual_token_count"],
                "inventory_row_sha256": candidate_source["row_sha256"],
            }
            observed_candidate = {
                key: candidate.get(key) for key in expected_candidate
            }
            if observed_candidate != expected_candidate:
                raise ValueError(f"Control candidate identity drift: {source['id']}:{family}")
            processor_input = candidate.get("processor_input") or {}
            if (
                processor_input.get("target_width") != source["width"]
                or processor_input.get("target_height") != source["height"]
                or processor_input.get("expected_image_grid_thw")
                != source["image_grid_thw"]
                or processor_input.get("expected_visual_token_count")
                != source["visual_token_count"]
            ):
                raise ValueError(f"Control processor target drift: {source['id']}:{family}")
            if family == "same_talk_stale" and not (
                candidate["lecture_id"] == source["lecture_id"]
                and candidate["state_id"] < source["state_id"]
            ):
                raise ValueError("Same-talk stale control is not causally prior")
            if family == "cross_talk_wrong" and (
                candidate["lecture_id"] == source["lecture_id"]
            ):
                raise ValueError("Cross-talk control comes from the source talk")


def media_path_for(source: dict[str, Any], family: str) -> str:
    lecture_id = source.get("lecture_id")
    state_id = source.get("state_id")
    if not isinstance(lecture_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", lecture_id) is None:
        raise ValueError("Source lecture id is not safe for a media path")
    if not isinstance(state_id, int) or isinstance(state_id, bool) or state_id < 0:
        raise ValueError("Source state id is not safe for a media path")
    return (
        f"{BUNDLE_PATH_PREFIX}/media/{family}/{lecture_id}/"
        f"state_{state_id:03d}.png"
    )


def materialize_candidate(
    temporary: Path,
    *,
    source_root: Path,
    source: dict[str, Any],
    candidate: dict[str, Any],
    family: str,
) -> tuple[dict[str, Any], Path]:
    original_path = resolve_regular_file(source_root, candidate["source_media_path"])
    if file_sha256(original_path) != candidate["source_media_sha256"]:
        raise ValueError(f"Control candidate source bytes changed: {source['id']}:{family}")
    processor_input = candidate["processor_input"]
    if processor_input["mode"] == "identity":
        final_path = original_path
        location = "canonical_native_source"
        final_relative_path = candidate["source_media_path"]
    else:
        final_relative_path = media_path_for(source, family)
        internal_relative = Path(final_relative_path).relative_to(BUNDLE_PATH_PREFIX)
        final_path = temporary / internal_relative
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(original_path) as image:
            prepared = prepare_control_image(image, processor_input)
        try:
            prepared.save(
                final_path,
                format="PNG",
                optimize=False,
                compress_level=9,
            )
        finally:
            prepared.close()
        location = "control_media_bundle"
    with Image.open(final_path) as image:
        image.verify()
    with Image.open(final_path) as image:
        final_size = image.size
    expected_size = (
        (candidate["width"], candidate["height"])
        if processor_input["mode"] == "identity"
        else (source["width"], source["height"])
    )
    if final_size != expected_size:
        raise ValueError(f"Final control dimensions differ from contract: {source['id']}:{family}")
    final = {
        "location": location,
        "source_media_path": final_relative_path,
        "source_media_sha256": file_sha256(final_path),
        "width": final_size[0],
        "height": final_size[1],
        "image_grid_thw": source["image_grid_thw"],
        "visual_token_count": source["visual_token_count"],
    }
    result = {
        "candidate_id": candidate["id"],
        "candidate_inventory_row_sha256": candidate["inventory_row_sha256"],
        "candidate_original_media_path": candidate["source_media_path"],
        "candidate_original_media_sha256": candidate["source_media_sha256"],
        "processor_input": processor_input,
        "final_media": final,
    }
    return result, final_path


def verify_processor_outputs(
    rows: list[dict[str, Any]],
    final_paths: dict[tuple[str, str], Path],
    *,
    processor: Any,
    batch_size: int,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("Processor batch size must be positive")
    records = []
    for row in rows:
        for family in CONTROL_FAMILIES:
            if row[family] is not None:
                records.append((row, family, final_paths[(row["id"], family)]))
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        images = []
        texts = []
        try:
            for _, _, path in batch:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
                texts.append(
                    processor.apply_chat_template(
                        qwen_image_messages(path, include_audio=False),
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                )
            encoded = processor(
                text=texts,
                images=images,
                videos=None,
                return_tensors="pt",
                padding=True,
            )
        finally:
            for image in images:
                image.close()
        for (row, family, _), token_row, grid_row in zip(
            batch,
            encoded["input_ids"],
            encoded["image_grid_thw"],
            strict=True,
        ):
            observed_count = sum(
                token_id == image_token_id for token_id in row_values(token_row)
            )
            observed_grid = row_values(grid_row)
            expected = row[family]["final_media"]
            if (
                observed_count != expected["visual_token_count"]
                or observed_grid != expected["image_grid_thw"]
            ):
                raise ValueError(f"Materialized control processor mismatch: {row['id']}:{family}")
    return {
        "records_verified": len(records),
        "bundle_media_verified": sum(
            row[family] is not None
            and row[family]["final_media"]["location"] == "control_media_bundle"
            for row in rows
            for family in CONTROL_FAMILIES
        ),
    }


def write_checksums(root: Path) -> tuple[int, str]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_path = root / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
        ),
        encoding="utf-8",
    )
    return len(paths), file_sha256(checksum_path)


def build_bundle(
    output_root: Path,
    *,
    inventory: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    processor_manifest: dict[str, Any],
    source_root: Path,
    processor: Any,
    builder_git_commit: str,
    source_inventory_sha256: str,
    source_controls_sha256: str,
    source_processor_manifest_sha256: str,
    source_controls_hf_revision: str,
    batch_size: int,
    expected_rows: int,
    expected_talks: int,
    frozen_input_revalidator: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Visual-control media output root must not already exist")
    validate_control_inputs(
        inventory,
        controls,
        processor_manifest,
        expected_rows=expected_rows,
        expected_talks=expected_talks,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        rows = []
        final_paths = {}
        for source, control in zip(inventory, controls, strict=True):
            correct_path = resolve_regular_file(source_root, source["source_media_path"])
            if file_sha256(correct_path) != source["source_media_sha256"]:
                raise ValueError(f"Correct source image bytes changed: {source['id']}")
            row = {
                "schema_version": "mcif_qwen3_omni_visual_control_media_v1",
                "id": source["id"],
                "lecture_id": source["lecture_id"],
                "state_id": source["state_id"],
                "availability_start_sec": source["availability_start_sec"],
                "source_inventory_row_sha256": source["row_sha256"],
                "source_control_row_sha256": control["row_sha256"],
                "processor_binding_sha256": source["processor_binding_sha256"],
                "correct_image": {
                    "location": "canonical_native_source",
                    "source_media_path": source["source_media_path"],
                    "source_media_sha256": source["source_media_sha256"],
                    "width": source["width"],
                    "height": source["height"],
                    "image_grid_thw": source["image_grid_thw"],
                    "visual_token_count": source["visual_token_count"],
                },
                "same_talk_stale": None,
                "cross_talk_wrong": None,
                "source_transcript_consumed": False,
                "target_or_reference_consumed": False,
            }
            for family in CONTROL_FAMILIES:
                candidate = control[family]
                if candidate is None:
                    continue
                materialized, final_path = materialize_candidate(
                    temporary,
                    source_root=source_root,
                    source=source,
                    candidate=candidate,
                    family=family,
                )
                row[family] = materialized
                final_paths[(source["id"], family)] = final_path
            row["row_sha256"] = canonical_sha256(row)
            rows.append(row)
        processor_audit = verify_processor_outputs(
            rows,
            final_paths,
            processor=processor,
            batch_size=batch_size,
        )
        manifest_path = temporary / "control_media_manifest.jsonl"
        manifest_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            ),
            encoding="utf-8",
        )
        media_paths = [path for path in temporary.rglob("*.png")]
        mode_counts = Counter(
            row[family]["processor_input"]["mode"]
            for row in rows
            for family in CONTROL_FAMILIES
            if row[family] is not None
        )
        report = {
            "schema_version": "mcif_qwen3_omni_visual_control_media_report_v1",
            "status": "SOURCE_ONLY_STATE_CONTROL_MEDIA_NOT_EVENT_PACKET_OR_ST_RESULT",
            "builder_git_commit": builder_git_commit,
            "source_inventory_sha256": source_inventory_sha256,
            "source_controls_sha256": source_controls_sha256,
            "source_processor_manifest_sha256": source_processor_manifest_sha256,
            "source_controls_hf_revision": source_controls_hf_revision,
            "processor_binding_sha256": processor_manifest["processor_binding_sha256"],
            "model_id": processor_manifest["model_id"],
            "model_revision": processor_manifest["model_revision"],
            "transformers_version": processor_manifest["transformers_version"],
            "pillow_version": pillow_version,
            "rows": len(rows),
            "talks": len({row["lecture_id"] for row in rows}),
            "same_talk_stale_coverage": sum(
                row["same_talk_stale"] is not None for row in rows
            ),
            "cross_talk_wrong_coverage": sum(
                row["cross_talk_wrong"] is not None for row in rows
            ),
            "processor_input_mode_distribution": dict(sorted(mode_counts.items())),
            "materialized_media_files": len(media_paths),
            "materialized_media_bytes": sum(path.stat().st_size for path in media_paths),
            "processor_audit": processor_audit,
            "control_media_manifest_sha256": file_sha256(manifest_path),
            "source_transcript_consumed": False,
            "target_or_reference_consumed": False,
            "interpretation": (
                "This bundle materializes source-only state control media. It does not "
                "define target events, event timing, labels, translations, or ST effects."
            ),
        }
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        readme_path = temporary / "README.md"
        readme_path.write_text(
            "# MCIF Qwen3-Omni Visual Control Media V1\n\n"
            "Source-only, state-level control media for the frozen 304-state MCIF "
            "visual evidence inventory. Transformed images are materialized only when "
            "the selected unrelated or stale source image does not naturally match the "
            "correct image processor shape. Every final control image was replayed through "
            "the frozen processor. This is not an event packet, annotation, translation "
            "output, or paper result.\n\n"
            f"- rows / talks: {report['rows']} / {report['talks']}\n"
            f"- materialized media: {report['materialized_media_files']} files\n"
            f"- manifest SHA256: `{report['control_media_manifest_sha256']}`\n",
            encoding="utf-8",
        )
        if frozen_input_revalidator is not None:
            frozen_input_revalidator()
        checksum_entries, checksum_sha256 = write_checksums(temporary)
        os.rename(temporary, output_root)
        return {
            **report,
            "checksum_entries": checksum_entries,
            "checksum_manifest_sha256": checksum_sha256,
            "bundle_files": sum(1 for path in output_root.rglob("*") if path.is_file()),
            "bundle_bytes": sum(
                path.stat().st_size for path in output_root.rglob("*") if path.is_file()
            ),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--expected-controls-sha256", required=True)
    parser.add_argument("--processor-manifest", type=Path, required=True)
    parser.add_argument("--expected-processor-manifest-sha256", required=True)
    parser.add_argument("--source-controls-hf-revision", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--processor-root", type=Path, required=True)
    parser.add_argument("--code-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-rows", type=int, default=304)
    parser.add_argument("--expected-talks", type=int, default=21)
    args = parser.parse_args()
    for path, expected, label in (
        (args.inventory, args.expected_inventory_sha256, "inventory"),
        (args.controls, args.expected_controls_sha256, "controls"),
        (
            args.processor_manifest,
            args.expected_processor_manifest_sha256,
            "processor manifest",
        ),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen {label} hash mismatch")
    if (
        len(args.source_controls_hf_revision) != 40
        or any(
            character not in "0123456789abcdef"
            for character in args.source_controls_hf_revision
        )
    ):
        raise ValueError("Source controls require a full immutable HF revision")
    builder_git_commit = git_head_clean(args.code_repo)
    inventory = load_jsonl(args.inventory)
    controls = load_jsonl(args.controls)
    processor_manifest = json.loads(args.processor_manifest.read_text(encoding="utf-8"))
    expected_processor_files = processor_manifest.get("processor_files")
    if processor_file_manifest(args.processor_root) != expected_processor_files:
        raise ValueError("Local processor files differ from the frozen control manifest")
    from transformers import Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.processor_root,
        local_files_only=True,
        trust_remote_code=True,
    )
    def revalidate_frozen_inputs() -> None:
        verify_frozen_inputs(
            inventory,
            source_root=args.source_root,
            processor_root=args.processor_root,
            expected_processor_files=expected_processor_files,
        )

    report = build_bundle(
        args.output_root,
        inventory=inventory,
        controls=controls,
        processor_manifest=processor_manifest,
        source_root=args.source_root,
        processor=processor,
        builder_git_commit=builder_git_commit,
        source_inventory_sha256=args.expected_inventory_sha256,
        source_controls_sha256=args.expected_controls_sha256,
        source_processor_manifest_sha256=args.expected_processor_manifest_sha256,
        source_controls_hf_revision=args.source_controls_hf_revision,
        batch_size=args.batch_size,
        expected_rows=args.expected_rows,
        expected_talks=args.expected_talks,
        frozen_input_revalidator=revalidate_frozen_inputs,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
