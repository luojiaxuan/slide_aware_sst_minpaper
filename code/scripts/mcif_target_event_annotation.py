#!/usr/bin/env python3
"""Initialize, validate, and freeze MCIF target-event author annotations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from scripts.build_mcif_target_event_author_workspace import AUTHOR_SCHEMA, MAPPING_SCHEMA
from scripts.build_mcif_visual_token_controls import canonical_sha256, file_sha256, load_jsonl


WORKING_SCHEMA = "mcif_target_event_author_working_v1"
FROZEN_SCHEMA = "mcif_target_event_author_frozen_v1"
AUTHORED_EVENT_SCHEMA = "mcif_authored_target_event_v1"
ANNOTATION_STATUSES = {
    "pending",
    "eligible",
    "no_target_alignment",
    "generic_or_unscorable",
    "visual_mismatch",
    "exclude_quality",
}
TARGET_ALIGNMENTS = {"explicit", "paraphrased", "omitted", "uncertain"}
SLIDE_EVIDENCE_STATUSES = {"supported", "ambiguous", "not_supported"}
UTC_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
IMMUTABLE_FIELDS = (
    "item_id",
    "source_reference_en",
    "target_reference_zh",
    "current_slide",
    "current_slide_r0_text",
    "current_slide_r1_text",
    "candidate_options",
    "official_reference_consumed",
    "model_output_consumed",
)


def row_hash_valid(row: dict[str, Any]) -> bool:
    return row.get("row_sha256") == canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )


def validate_input_rows(
    rows: list[dict[str, Any]],
    *,
    expected_items: int,
) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_items:
        raise ValueError("MCIF author input item count differs from contract")
    output = {}
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in output:
            raise ValueError("MCIF author input contains an invalid or duplicate item id")
        if row.get("schema_version") != AUTHOR_SCHEMA:
            raise ValueError("Unexpected MCIF author input schema")
        if row.get("status") != "PENDING_HUMAN_EVENT_AUTHORING":
            raise ValueError("MCIF author input is not an untouched pending workspace")
        if not row_hash_valid(row):
            raise ValueError(f"MCIF author input row hash mismatch: {item_id}")
        if row.get("annotation_status") != "pending" or any(
            row.get(key) not in (None, [], "")
            for key in (
                "selected_option_id",
                "canonical_source_event_en",
                "acceptable_target_realizations_zh",
                "forbidden_target_realizations_zh",
                "target_reference_alignment",
                "slide_evidence_status",
                "annotation_note",
                "annotator_id",
                "locked_at_utc",
            )
        ):
            raise ValueError("MCIF author input contains premature human labels")
        if row.get("official_reference_consumed") is not True or row.get(
            "model_output_consumed"
        ) is not False:
            raise ValueError("MCIF author input has an invalid data boundary")
        options = row.get("candidate_options")
        if not isinstance(options, list) or not options:
            raise ValueError("MCIF author input has no candidate options")
        option_ids = [option.get("option_id") for option in options]
        if any(not isinstance(value, str) or not value for value in option_ids):
            raise ValueError("MCIF author input has an invalid option id")
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("MCIF author input has duplicate option ids")
        output[item_id] = row
    return output


def initialize_working_rows(
    input_rows: list[dict[str, Any]],
    *,
    annotator_id: str,
    expected_items: int,
) -> list[dict[str, Any]]:
    if not annotator_id.strip():
        raise ValueError("MCIF target-event annotator id must not be empty")
    validate_input_rows(input_rows, expected_items=expected_items)
    output = []
    for source in input_rows:
        row = {
            "schema_version": WORKING_SCHEMA,
            "source_author_row_sha256": source["row_sha256"],
            **{field: source[field] for field in IMMUTABLE_FIELDS},
            "annotation_status": "pending",
            "selected_option_id": None,
            "canonical_source_event_en": "",
            "acceptable_target_realizations_zh": [],
            "forbidden_target_realizations_zh": [],
            "target_reference_alignment": None,
            "slide_evidence_status": None,
            "annotation_note": "",
            "annotator_id": annotator_id,
        }
        row["row_sha256"] = canonical_sha256(row)
        output.append(row)
    return output


def clean_realizations(values: Any, *, label: str) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"MCIF {label} must be a list of strings")
    cleaned = [value.strip() for value in values]
    if any(not value for value in cleaned):
        raise ValueError(f"MCIF {label} contains an empty realization")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"MCIF {label} contains duplicate realizations")
    return cleaned


def validate_working_row(
    row: dict[str, Any],
    source: dict[str, Any],
    *,
    annotator_id: str,
    allow_pending: bool,
) -> None:
    item_id = source["item_id"]
    if row.get("schema_version") != WORKING_SCHEMA:
        raise ValueError(f"Unexpected MCIF working schema: {item_id}")
    if row.get("source_author_row_sha256") != source["row_sha256"]:
        raise ValueError(f"MCIF working row source binding differs: {item_id}")
    if any(row.get(field) != source[field] for field in IMMUTABLE_FIELDS):
        raise ValueError(f"MCIF working row changed immutable input: {item_id}")
    if row.get("annotator_id") != annotator_id:
        raise ValueError(f"MCIF working row annotator differs: {item_id}")
    if not row_hash_valid(row):
        raise ValueError(f"MCIF working row hash mismatch: {item_id}")
    status = row.get("annotation_status")
    if status not in ANNOTATION_STATUSES:
        raise ValueError(f"MCIF working row has an invalid status: {item_id}")
    selected = row.get("selected_option_id")
    canonical = row.get("canonical_source_event_en")
    if not isinstance(canonical, str):
        raise ValueError(f"MCIF canonical source event is not text: {item_id}")
    acceptable = clean_realizations(
        row.get("acceptable_target_realizations_zh"),
        label="acceptable target realizations",
    )
    forbidden = clean_realizations(
        row.get("forbidden_target_realizations_zh"),
        label="forbidden target realizations",
    )
    if set(acceptable) & set(forbidden):
        raise ValueError(f"MCIF acceptable and forbidden realizations overlap: {item_id}")
    alignment = row.get("target_reference_alignment")
    slide_status = row.get("slide_evidence_status")
    note = row.get("annotation_note")
    if not isinstance(note, str):
        raise ValueError(f"MCIF annotation note is not text: {item_id}")
    if status == "pending":
        if not allow_pending:
            raise ValueError(f"MCIF working row remains pending: {item_id}")
        if any(
            value not in (None, "", [])
            for value in (selected, canonical, acceptable, forbidden, alignment, slide_status, note)
        ):
            raise ValueError(f"MCIF pending row contains partial labels: {item_id}")
        return
    if status == "eligible":
        option_ids = {option["option_id"] for option in source["candidate_options"]}
        if selected not in option_ids:
            raise ValueError(f"MCIF eligible row has no valid selected option: {item_id}")
        if not canonical.strip():
            raise ValueError(f"MCIF eligible row has no canonical source event: {item_id}")
        if not acceptable:
            raise ValueError(f"MCIF eligible row has no acceptable target realization: {item_id}")
        if alignment not in {"explicit", "paraphrased"}:
            raise ValueError(f"MCIF eligible row lacks target alignment: {item_id}")
        if slide_status != "supported":
            raise ValueError(f"MCIF eligible row lacks supported slide evidence: {item_id}")
        return
    if any(value not in (None, "", []) for value in (selected, canonical, acceptable, forbidden)):
        raise ValueError(f"MCIF non-eligible row retains scoring answers: {item_id}")
    if alignment not in TARGET_ALIGNMENTS:
        raise ValueError(f"MCIF non-eligible row lacks target alignment: {item_id}")
    if slide_status not in SLIDE_EVIDENCE_STATUSES:
        raise ValueError(f"MCIF non-eligible row lacks slide evidence status: {item_id}")


def validate_working_rows(
    working_rows: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
    *,
    annotator_id: str,
    expected_items: int,
    allow_pending: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    source_by_id = validate_input_rows(input_rows, expected_items=expected_items)
    if len(working_rows) != expected_items:
        raise ValueError("MCIF working sheet item count differs from contract")
    working_by_id = {}
    for row in working_rows:
        item_id = row.get("item_id")
        if item_id not in source_by_id or item_id in working_by_id:
            raise ValueError("MCIF working sheet has an absent or duplicate item id")
        validate_working_row(
            row,
            source_by_id[item_id],
            annotator_id=annotator_id,
            allow_pending=allow_pending,
        )
        working_by_id[item_id] = row
    if set(working_by_id) != set(source_by_id):
        raise ValueError("MCIF working and input item sets differ")
    return source_by_id, working_by_id


def validate_mapping_rows(
    rows: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(rows) != len(source_by_id):
        raise ValueError("MCIF scorer mapping item count differs")
    output = {}
    for row in rows:
        item_id = row.get("item_id")
        if item_id not in source_by_id or item_id in output:
            raise ValueError("MCIF scorer mapping has an absent or duplicate item id")
        if row.get("schema_version") != MAPPING_SCHEMA or not row_hash_valid(row):
            raise ValueError(f"MCIF scorer mapping schema/hash differs: {item_id}")
        if row.get("author_row_sha256") != source_by_id[item_id]["row_sha256"]:
            raise ValueError(f"MCIF scorer mapping author binding differs: {item_id}")
        if row.get("official_reference_consumed") is not True or row.get(
            "model_output_consumed"
        ) is not False:
            raise ValueError("MCIF scorer mapping has an invalid data boundary")
        option_ids = {option["option_id"] for option in source_by_id[item_id]["candidate_options"]}
        mapping_ids = {option.get("option_id") for option in row.get("option_mapping", [])}
        if option_ids != mapping_ids:
            raise ValueError(f"MCIF scorer option mapping differs: {item_id}")
        output[item_id] = row
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_checksums(root: Path) -> tuple[int, str]:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum = root / "SHA256SUMS"
    checksum.write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
        ),
        encoding="utf-8",
    )
    return len(paths), file_sha256(checksum)


def freeze_annotations(
    output_root: Path,
    *,
    input_rows: list[dict[str, Any]],
    working_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    input_sha256: str,
    working_sha256: str,
    mapping_sha256: str,
    annotator_id: str,
    locked_at_utc: str,
    expected_items: int,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF frozen author annotation output must not already exist")
    if UTC_PATTERN.fullmatch(locked_at_utc) is None:
        raise ValueError("MCIF author lock timestamp must use YYYY-MM-DDTHH:MM:SSZ")
    source_by_id, working_by_id = validate_working_rows(
        working_rows,
        input_rows,
        annotator_id=annotator_id,
        expected_items=expected_items,
        allow_pending=False,
    )
    mapping_by_id = validate_mapping_rows(mapping_rows, source_by_id)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        frozen_rows = []
        event_rows = []
        for source in input_rows:
            item_id = source["item_id"]
            working = working_by_id[item_id]
            mapping = mapping_by_id[item_id]
            annotation = {
                key: working[key]
                for key in (
                    "annotation_status",
                    "selected_option_id",
                    "canonical_source_event_en",
                    "acceptable_target_realizations_zh",
                    "forbidden_target_realizations_zh",
                    "target_reference_alignment",
                    "slide_evidence_status",
                    "annotation_note",
                    "annotator_id",
                )
            }
            annotation_sha256 = canonical_sha256(annotation)
            frozen = {
                "schema_version": FROZEN_SCHEMA,
                "status": "HUMAN_EVENT_AUTHORING_FROZEN",
                "item_id": item_id,
                "source_author_row_sha256": source["row_sha256"],
                "working_row_sha256": working["row_sha256"],
                "scorer_mapping_row_sha256": mapping["row_sha256"],
                **annotation,
                "annotation_sha256": annotation_sha256,
                "locked_at_utc": locked_at_utc,
                "official_reference_consumed": True,
                "model_output_consumed": False,
            }
            frozen["row_sha256"] = canonical_sha256(frozen)
            frozen_rows.append(frozen)
            if working["annotation_status"] != "eligible":
                continue
            selected = working["selected_option_id"]
            candidate = next(
                option for option in mapping["option_mapping"] if option["option_id"] == selected
            )
            event = {
                "schema_version": AUTHORED_EVENT_SCHEMA,
                "status": "TARGET_EVENT_AUTHORED_PENDING_AUDIO_SUFFICIENCY",
                "event_id": (
                    f"mcif:{mapping['talk_id']}:R0E{mapping['talk_segment_index']:03d}"
                ),
                "item_id": item_id,
                "talk_id": mapping["talk_id"],
                "segment_id": mapping["segment_id"],
                "talk_segment_index": mapping["talk_segment_index"],
                "source_segment_offset_sec": mapping["source_segment_offset_sec"],
                "source_segment_end_sec": mapping["source_segment_end_sec"],
                "candidate_id": candidate["candidate_id"],
                "candidate_row_sha256": candidate["candidate_row_sha256"],
                "current_state_id": mapping["current_state_id"],
                "current_state_row_sha256": mapping["current_state_row_sha256"],
                "current_evidence_available_sec": mapping["current_evidence_available_sec"],
                "earliest_contiguous_state_id": candidate["earliest_contiguous_state_id"],
                "earliest_contiguous_state_row_sha256": candidate[
                    "earliest_contiguous_state_row_sha256"
                ],
                "evidence_available_sec": candidate["earliest_contiguous_evidence_sec"],
                "canonical_source_event_en": working["canonical_source_event_en"].strip(),
                "acceptable_target_realizations_zh": clean_realizations(
                    working["acceptable_target_realizations_zh"],
                    label="acceptable target realizations",
                ),
                "forbidden_target_realizations_zh": clean_realizations(
                    working["forbidden_target_realizations_zh"],
                    label="forbidden target realizations",
                ),
                "target_reference_alignment": working["target_reference_alignment"],
                "slide_evidence_status": working["slide_evidence_status"],
                "annotation_sha256": annotation_sha256,
                "audio_insufficient_until_sec": None,
                "audio_first_sufficient_sec": None,
                "primary_eligible": None,
                "official_reference_consumed": True,
                "model_output_consumed": False,
            }
            event["row_sha256"] = canonical_sha256(event)
            event_rows.append(event)
        frozen_path = temporary / "frozen_author_annotations.jsonl"
        event_path = temporary / "authored_target_events_private.jsonl"
        write_jsonl(frozen_path, frozen_rows)
        write_jsonl(event_path, event_rows)
        status_counts = Counter(row["annotation_status"] for row in working_rows)
        event_talks = {row["talk_id"] for row in event_rows}
        report = {
            "schema_version": "mcif_target_event_author_freeze_report_v1",
            "status": "HUMAN_EVENT_AUTHORING_FROZEN_AUDIO_SUFFICIENCY_PENDING",
            "input_sha256": input_sha256,
            "working_sha256": working_sha256,
            "mapping_sha256": mapping_sha256,
            "annotator_id": annotator_id,
            "locked_at_utc": locked_at_utc,
            "items": len(frozen_rows),
            "status_distribution": dict(sorted(status_counts.items())),
            "authored_events": len(event_rows),
            "authored_event_talks": len(event_talks),
            "frozen_author_annotations_sha256": file_sha256(frozen_path),
            "authored_target_events_sha256": file_sha256(event_path),
            "audio_sufficiency_labels_complete": False,
            "primary_eligibility_complete": False,
            "official_reference_consumed": True,
            "model_output_consumed": False,
            "interpretation": (
                "Authored target events remain provisional until independent audio-only "
                "sufficiency validation. They are not yet SourceEventTiming rows or ST results."
            ),
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument("--input-sheet", type=Path, required=True)
    initialize.add_argument("--expected-input-sha256", required=True)
    initialize.add_argument("--working-sheet", type=Path, required=True)
    initialize.add_argument("--annotator-id", required=True)
    initialize.add_argument("--expected-items", type=int, default=355)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--input-sheet", type=Path, required=True)
    freeze.add_argument("--expected-input-sha256", required=True)
    freeze.add_argument("--working-sheet", type=Path, required=True)
    freeze.add_argument("--expected-working-sha256", required=True)
    freeze.add_argument("--scorer-mapping", type=Path, required=True)
    freeze.add_argument("--expected-mapping-sha256", required=True)
    freeze.add_argument("--annotator-id", required=True)
    freeze.add_argument("--locked-at-utc", required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument("--expected-items", type=int, default=355)
    args = parser.parse_args()
    if file_sha256(args.input_sheet) != args.expected_input_sha256:
        raise ValueError("MCIF author input sheet hash differs from contract")
    input_rows = load_jsonl(args.input_sheet)
    if args.command == "init":
        if args.working_sheet.exists() or args.working_sheet.is_symlink():
            raise FileExistsError("MCIF author working sheet must not already exist")
        rows = initialize_working_rows(
            input_rows,
            annotator_id=args.annotator_id,
            expected_items=args.expected_items,
        )
        write_jsonl_atomic(args.working_sheet, rows)
        print(
            json.dumps(
                {
                    "status": "WORKING_SHEET_INITIALIZED",
                    "items": len(rows),
                    "working_sheet_sha256": file_sha256(args.working_sheet),
                },
                sort_keys=True,
            )
        )
        return
    for path, expected, label in (
        (args.working_sheet, args.expected_working_sha256, "working sheet"),
        (args.scorer_mapping, args.expected_mapping_sha256, "scorer mapping"),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"MCIF {label} hash differs from contract")
    report = freeze_annotations(
        args.output_root,
        input_rows=input_rows,
        working_rows=load_jsonl(args.working_sheet),
        mapping_rows=load_jsonl(args.scorer_mapping),
        input_sha256=args.expected_input_sha256,
        working_sha256=args.expected_working_sha256,
        mapping_sha256=args.expected_mapping_sha256,
        annotator_id=args.annotator_id,
        locked_at_utc=args.locked_at_utc,
        expected_items=args.expected_items,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
