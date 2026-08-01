#!/usr/bin/env python3
"""Initialize, validate, freeze, and join MCIF beyond-OCR human labels."""

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

from scripts.build_mcif_beyond_ocr_validation_workspace import (
    MAPPING_SCHEMA,
    TARGET_SCHEMA,
    VISUAL_SCHEMA,
)
from scripts.build_mcif_visual_token_controls import (
    canonical_sha256,
    file_sha256,
    load_jsonl,
)


VISUAL_WORKING_SCHEMA = "mcif_beyond_ocr_visual_validation_working_v1"
TARGET_WORKING_SCHEMA = "mcif_beyond_ocr_target_author_working_v1"
VISUAL_FROZEN_SCHEMA = "mcif_beyond_ocr_visual_validation_frozen_v1"
TARGET_FROZEN_SCHEMA = "mcif_beyond_ocr_target_author_frozen_v1"
JOINED_SCHEMA = "mcif_beyond_ocr_visual_target_join_v1"
JUDGMENTS = {"yes", "no", "uncertain"}
TARGET_ALIGNMENTS = {"explicit", "paraphrased", "omitted", "unsupported", "uncertain"}
UTC_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
VISUAL_IMMUTABLE_FIELDS = (
    "item_id",
    "candidate_source_en",
    "candidate_kind",
    "candidate_token_count",
    "evidence_channel",
    "current_slide",
    "current_slide_r0_text",
    "current_slide_r1_blocks",
    "proposed_evidence_origins",
    "requires_r1_insufficiency_judgment",
    "official_reference_consumed",
    "source_reference_exposed",
    "target_reference_exposed",
    "generative_model_output_exposed",
)
TARGET_IMMUTABLE_FIELDS = (
    "item_id",
    "candidate_source_en",
    "candidate_kind",
    "candidate_token_count",
    "source_reference_en",
    "target_reference_zh",
    "official_reference_consumed",
    "slide_or_ocr_exposed",
    "visual_evidence_origin_exposed",
    "generative_model_output_exposed",
)


def row_hash_valid(row: dict[str, Any]) -> bool:
    return row.get("row_sha256") == canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "schema_version": "mcif_beyond_ocr_validation_config_v1",
        "target_language": "zh",
        "candidate_scope": "r1_strict_and_r2_semantic_first_source_occurrence",
        "roles": ["visual_validator", "target_author"],
        "judgments": ["yes", "no", "uncertain"],
        "target_reference_alignments": [
            "explicit",
            "paraphrased",
            "omitted",
            "unsupported",
            "uncertain",
        ],
        "visual_gate": {
            "visual_evidence_correct": "yes",
            "candidate_supported_by_visual_evidence": "yes",
            "r0_insufficient": "yes",
            "r1_insufficient_for_r2": "yes",
        },
        "target_gate": {
            "candidate_eligibility": "yes",
            "canonical_source_event_required": True,
            "acceptable_target_realization_required": True,
            "target_reference_alignment_allowed": ["explicit", "paraphrased"],
        },
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("MCIF beyond-OCR validation config differs from code contract")
    boundary = config.get("access_boundary") or {}
    if set(boundary.get("visual_validator_forbidden") or []) != {
        "source_reference",
        "target_reference",
        "talk_id",
        "segment_id",
        "state_id",
        "timing",
        "lead",
        "scorer_mapping",
        "target_author_labels",
    }:
        raise ValueError("MCIF beyond-OCR visual-validator firewall differs")
    if set(boundary.get("target_author_forbidden") or []) != {
        "slide",
        "r0_ocr",
        "r1_structured_text",
        "evidence_tier",
        "vlm_description",
        "talk_id",
        "segment_id",
        "state_id",
        "timing",
        "lead",
        "scorer_mapping",
        "visual_validator_labels",
    }:
        raise ValueError("MCIF beyond-OCR target-author firewall differs")
    if set(boundary.get("audio_validator_forbidden") or []) != {
        "slide",
        "r0_ocr",
        "r1_structured_text",
        "source_reference",
        "target_reference",
        "candidate",
        "evidence_tier",
        "vlm_description",
        "scorer_mapping",
        "visual_validator_labels",
        "target_author_labels",
    }:
        raise ValueError("MCIF beyond-OCR audio-validator firewall differs")


def load_frozen_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise ValueError("MCIF beyond-OCR validation config hash differs")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("MCIF beyond-OCR validation config is not an object")
    validate_config(config)
    return config


def role_contract(role: str) -> tuple[str, str, tuple[str, ...]]:
    if role == "visual":
        return VISUAL_SCHEMA, VISUAL_WORKING_SCHEMA, VISUAL_IMMUTABLE_FIELDS
    if role == "target":
        return TARGET_SCHEMA, TARGET_WORKING_SCHEMA, TARGET_IMMUTABLE_FIELDS
    raise ValueError("MCIF beyond-OCR role must be visual or target")


def validate_input_rows(
    rows: list[dict[str, Any]], *, role: str, expected_items: int
) -> dict[str, dict[str, Any]]:
    input_schema, _, _ = role_contract(role)
    expected_status = (
        "PENDING_INDEPENDENT_VISUAL_VALIDATION"
        if role == "visual"
        else "PENDING_INDEPENDENT_TARGET_EVENT_AUTHORING"
    )
    if len(rows) != expected_items:
        raise ValueError(f"MCIF beyond-OCR {role} input count differs from contract")
    output = {}
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in output:
            raise ValueError(f"MCIF beyond-OCR {role} input has invalid or duplicate id")
        if row.get("schema_version") != input_schema or row.get("status") != expected_status:
            raise ValueError(f"MCIF beyond-OCR {role} input schema/status differs")
        if not row_hash_valid(row):
            raise ValueError(f"MCIF beyond-OCR {role} input row hash mismatch: {item_id}")
        common = (
            row.get("annotation_status") == "pending"
            and row.get("reason_codes") == []
            and row.get("annotation_note") == ""
            and row.get("annotator_id") is None
            and row.get("locked_at_utc") is None
        )
        if role == "visual":
            labels_blank = all(
                row.get(key) is None
                for key in (
                    "visual_evidence_correct",
                    "candidate_supported_by_visual_evidence",
                    "r0_insufficient",
                    "r1_insufficient",
                )
            )
            boundary_valid = (
                row.get("source_reference_exposed") is False
                and row.get("target_reference_exposed") is False
            )
        else:
            labels_blank = (
                row.get("candidate_eligibility") is None
                and row.get("canonical_source_event_en") == ""
                and row.get("acceptable_target_realizations_zh") == []
                and row.get("forbidden_target_realizations_zh") == []
                and row.get("target_reference_alignment") is None
            )
            boundary_valid = (
                row.get("slide_or_ocr_exposed") is False
                and row.get("visual_evidence_origin_exposed") is False
                and row.get("generative_model_output_exposed") is False
            )
        if not common or not labels_blank:
            raise ValueError(f"MCIF beyond-OCR {role} input contains premature labels")
        if row.get("official_reference_consumed") is not True or not boundary_valid:
            raise ValueError(f"MCIF beyond-OCR {role} input boundary differs")
        output[item_id] = row
    return output


def initialize_working_rows(
    input_rows: list[dict[str, Any]],
    *,
    role: str,
    annotator_id: str,
    expected_items: int,
) -> list[dict[str, Any]]:
    if not annotator_id.strip():
        raise ValueError("MCIF beyond-OCR annotator id must not be empty")
    validate_input_rows(input_rows, role=role, expected_items=expected_items)
    _, working_schema, immutable_fields = role_contract(role)
    output = []
    for source in input_rows:
        row = {
            "schema_version": working_schema,
            "source_input_row_sha256": source["row_sha256"],
            **{field: source[field] for field in immutable_fields},
            "annotation_status": "pending",
        }
        if role == "visual":
            row.update(
                {
                    "visual_evidence_correct": None,
                    "candidate_supported_by_visual_evidence": None,
                    "r0_insufficient": None,
                    "r1_insufficient": None,
                }
            )
        else:
            row.update(
                {
                    "candidate_eligibility": None,
                    "canonical_source_event_en": "",
                    "acceptable_target_realizations_zh": [],
                    "forbidden_target_realizations_zh": [],
                    "target_reference_alignment": None,
                }
            )
        row.update(
            {
                "reason_codes": [],
                "annotation_note": "",
                "annotator_id": annotator_id,
            }
        )
        row["row_sha256"] = canonical_sha256(row)
        output.append(row)
    return output


def clean_string_list(values: Any, *, label: str) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"MCIF beyond-OCR {label} must be a list of strings")
    cleaned = [value.strip() for value in values]
    if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
        raise ValueError(f"MCIF beyond-OCR {label} contains empty or duplicate values")
    return cleaned


def validate_working_row(
    row: dict[str, Any],
    source: dict[str, Any],
    *,
    role: str,
    annotator_id: str,
    allow_pending: bool,
) -> None:
    _, working_schema, immutable_fields = role_contract(role)
    item_id = source["item_id"]
    if row.get("schema_version") != working_schema:
        raise ValueError(f"MCIF beyond-OCR {role} working schema differs: {item_id}")
    if row.get("source_input_row_sha256") != source["row_sha256"]:
        raise ValueError(f"MCIF beyond-OCR {role} source binding differs: {item_id}")
    if any(row.get(field) != source[field] for field in immutable_fields):
        raise ValueError(f"MCIF beyond-OCR {role} immutable input changed: {item_id}")
    if row.get("annotator_id") != annotator_id or not row_hash_valid(row):
        raise ValueError(f"MCIF beyond-OCR {role} annotator/hash differs: {item_id}")
    status = row.get("annotation_status")
    if status not in {"pending", "completed"}:
        raise ValueError(f"MCIF beyond-OCR {role} working status differs: {item_id}")
    reason_codes = clean_string_list(row.get("reason_codes"), label="reason codes")
    note = row.get("annotation_note")
    if not isinstance(note, str):
        raise ValueError(f"MCIF beyond-OCR {role} note is not text: {item_id}")
    if role == "visual":
        values = [
            row.get("visual_evidence_correct"),
            row.get("candidate_supported_by_visual_evidence"),
            row.get("r0_insufficient"),
        ]
        r1_value = row.get("r1_insufficient")
        if source["requires_r1_insufficiency_judgment"]:
            values.append(r1_value)
        elif r1_value is not None:
            raise ValueError(f"MCIF beyond-OCR R1 item has an R1-insufficiency label: {item_id}")
        partial_values = values
    else:
        eligibility = row.get("candidate_eligibility")
        canonical = row.get("canonical_source_event_en")
        if not isinstance(canonical, str):
            raise ValueError(f"MCIF beyond-OCR canonical event is not text: {item_id}")
        acceptable = clean_string_list(
            row.get("acceptable_target_realizations_zh"), label="acceptable realizations"
        )
        forbidden = clean_string_list(
            row.get("forbidden_target_realizations_zh"), label="forbidden realizations"
        )
        if set(acceptable) & set(forbidden):
            raise ValueError(f"MCIF beyond-OCR target realizations overlap: {item_id}")
        alignment = row.get("target_reference_alignment")
        partial_values = [eligibility, canonical, acceptable, forbidden, alignment]
    if status == "pending":
        if not allow_pending:
            raise ValueError(f"MCIF beyond-OCR {role} row remains pending: {item_id}")
        if any(value not in (None, "", []) for value in [*partial_values, reason_codes, note]):
            raise ValueError(f"MCIF beyond-OCR {role} pending row contains partial labels: {item_id}")
        return
    if role == "visual":
        if any(value not in JUDGMENTS for value in values):
            raise ValueError(f"MCIF beyond-OCR visual row lacks a judgment: {item_id}")
        if any(value != "yes" for value in values) and not reason_codes:
            raise ValueError(f"MCIF beyond-OCR rejected visual row lacks a reason: {item_id}")
        return
    if eligibility not in JUDGMENTS or alignment not in TARGET_ALIGNMENTS:
        raise ValueError(f"MCIF beyond-OCR target row lacks eligibility/alignment: {item_id}")
    if eligibility == "yes":
        if not canonical.strip() or not acceptable:
            raise ValueError(f"MCIF beyond-OCR eligible target row lacks scoring text: {item_id}")
        if alignment not in {"explicit", "paraphrased"}:
            raise ValueError(f"MCIF beyond-OCR eligible target row lacks alignment: {item_id}")
    else:
        if any(value not in ("", []) for value in (canonical, acceptable, forbidden)):
            raise ValueError(f"MCIF beyond-OCR rejected target row retains scoring text: {item_id}")
        if not reason_codes:
            raise ValueError(f"MCIF beyond-OCR rejected target row lacks a reason: {item_id}")


def validate_working_rows(
    working_rows: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
    *,
    role: str,
    annotator_id: str,
    expected_items: int,
    allow_pending: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    source_by_id = validate_input_rows(input_rows, role=role, expected_items=expected_items)
    if len(working_rows) != expected_items:
        raise ValueError(f"MCIF beyond-OCR {role} working count differs")
    working_by_id = {}
    for row in working_rows:
        item_id = row.get("item_id")
        if item_id not in source_by_id or item_id in working_by_id:
            raise ValueError(f"MCIF beyond-OCR {role} working ids differ")
        validate_working_row(
            row,
            source_by_id[item_id],
            role=role,
            annotator_id=annotator_id,
            allow_pending=allow_pending,
        )
        working_by_id[item_id] = row
    if set(working_by_id) != set(source_by_id):
        raise ValueError(f"MCIF beyond-OCR {role} input/working sets differ")
    return source_by_id, working_by_id


def validate_mapping_rows(
    rows: list[dict[str, Any]],
    visual_by_id: dict[str, dict[str, Any]],
    target_by_id: dict[str, dict[str, Any]],
    *,
    expected_items: int,
) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_items:
        raise ValueError("MCIF beyond-OCR scorer mapping count differs")
    output = {}
    used_visual = set()
    used_target = set()
    for row in rows:
        candidate_id = row.get("candidate_id")
        visual_id = row.get("visual_item_id")
        target_id = row.get("target_item_id")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in output
            or visual_id not in visual_by_id
            or target_id not in target_by_id
            or visual_id in used_visual
            or target_id in used_target
        ):
            raise ValueError("MCIF beyond-OCR scorer mapping ids differ")
        if row.get("schema_version") != MAPPING_SCHEMA or not row_hash_valid(row):
            raise ValueError(f"MCIF beyond-OCR scorer mapping schema/hash differs: {candidate_id}")
        if (
            row.get("visual_item_row_sha256") != visual_by_id[visual_id]["row_sha256"]
            or row.get("target_item_row_sha256") != target_by_id[target_id]["row_sha256"]
        ):
            raise ValueError(f"MCIF beyond-OCR scorer role binding differs: {candidate_id}")
        if row.get("human_labels_complete") is not False:
            raise ValueError("MCIF beyond-OCR scorer mapping contains human labels")
        output[candidate_id] = row
        used_visual.add(visual_id)
        used_target.add(target_id)
    if used_visual != set(visual_by_id) or used_target != set(target_by_id):
        raise ValueError("MCIF beyond-OCR scorer mapping role sets differ")
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
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


def visual_gate_passed(row: dict[str, Any], source: dict[str, Any]) -> bool:
    values = [
        row["visual_evidence_correct"],
        row["candidate_supported_by_visual_evidence"],
        row["r0_insufficient"],
    ]
    if source["requires_r1_insufficiency_judgment"]:
        values.append(row["r1_insufficient"])
    return all(value == "yes" for value in values)


def target_gate_passed(row: dict[str, Any]) -> bool:
    return (
        row["candidate_eligibility"] == "yes"
        and bool(row["canonical_source_event_en"].strip())
        and bool(row["acceptable_target_realizations_zh"])
        and row["target_reference_alignment"] in {"explicit", "paraphrased"}
    )


def freeze_role(
    output_root: Path,
    *,
    role: str,
    input_rows: list[dict[str, Any]],
    working_rows: list[dict[str, Any]],
    input_sha256: str,
    working_sha256: str,
    config_sha256: str,
    annotator_id: str,
    locked_at_utc: str,
    expected_items: int,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF beyond-OCR frozen role output must not already exist")
    if UTC_PATTERN.fullmatch(locked_at_utc) is None:
        raise ValueError("MCIF beyond-OCR lock timestamp must use YYYY-MM-DDTHH:MM:SSZ")
    source_by_id, working_by_id = validate_working_rows(
        working_rows,
        input_rows,
        role=role,
        annotator_id=annotator_id,
        expected_items=expected_items,
        allow_pending=False,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        frozen_rows = []
        gate_passed = 0
        for source in input_rows:
            working = working_by_id[source["item_id"]]
            if role == "visual":
                labels = {
                    key: working[key]
                    for key in (
                        "visual_evidence_correct",
                        "candidate_supported_by_visual_evidence",
                        "r0_insufficient",
                        "r1_insufficient",
                        "reason_codes",
                        "annotation_note",
                        "annotator_id",
                    )
                }
                passed = visual_gate_passed(working, source)
                schema = VISUAL_FROZEN_SCHEMA
            else:
                labels = {
                    key: working[key]
                    for key in (
                        "candidate_eligibility",
                        "canonical_source_event_en",
                        "acceptable_target_realizations_zh",
                        "forbidden_target_realizations_zh",
                        "target_reference_alignment",
                        "reason_codes",
                        "annotation_note",
                        "annotator_id",
                    )
                }
                passed = target_gate_passed(working)
                schema = TARGET_FROZEN_SCHEMA
            gate_passed += passed
            frozen = {
                "schema_version": schema,
                "status": f"HUMAN_{role.upper()}_ANNOTATION_FROZEN",
                "item_id": source["item_id"],
                "source_input_row_sha256": source["row_sha256"],
                "working_row_sha256": working["row_sha256"],
                **labels,
                "role_gate_passed": passed,
                "annotation_sha256": canonical_sha256(labels),
                "locked_at_utc": locked_at_utc,
                "official_reference_consumed": True,
            }
            frozen["row_sha256"] = canonical_sha256(frozen)
            frozen_rows.append(frozen)
        frozen_path = temporary / f"frozen_{role}_annotations.jsonl"
        write_jsonl(frozen_path, frozen_rows)
        report = {
            "schema_version": f"mcif_beyond_ocr_{role}_freeze_report_v1",
            "status": f"HUMAN_{role.upper()}_ANNOTATION_FROZEN",
            "role": role,
            "items": len(frozen_rows),
            "gate_passed": gate_passed,
            "gate_rejected_or_uncertain": len(frozen_rows) - gate_passed,
            "input_sha256": input_sha256,
            "working_sha256": working_sha256,
            "config_sha256": config_sha256,
            "annotator_id": annotator_id,
            "locked_at_utc": locked_at_utc,
            "frozen_rows_sha256": file_sha256(frozen_path),
            "audio_sufficiency_complete": False,
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        entries, checksum_sha256 = write_checksums(temporary)
        os.rename(temporary, output_root)
        return {
            **report,
            "checksum_entries": entries,
            "checksum_sha256": checksum_sha256,
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_frozen_rows(
    rows: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    *,
    role: str,
) -> dict[str, dict[str, Any]]:
    schema = VISUAL_FROZEN_SCHEMA if role == "visual" else TARGET_FROZEN_SCHEMA
    if len(rows) != len(source_by_id):
        raise ValueError(f"MCIF beyond-OCR frozen {role} count differs")
    output = {}
    for row in rows:
        item_id = row.get("item_id")
        if item_id not in source_by_id or item_id in output:
            raise ValueError(f"MCIF beyond-OCR frozen {role} ids differ")
        if row.get("schema_version") != schema or not row_hash_valid(row):
            raise ValueError(f"MCIF beyond-OCR frozen {role} schema/hash differs: {item_id}")
        if row.get("source_input_row_sha256") != source_by_id[item_id]["row_sha256"]:
            raise ValueError(f"MCIF beyond-OCR frozen {role} source binding differs: {item_id}")
        if row.get("status") != f"HUMAN_{role.upper()}_ANNOTATION_FROZEN" or (
            UTC_PATTERN.fullmatch(str(row.get("locked_at_utc"))) is None
        ):
            raise ValueError(f"MCIF beyond-OCR frozen {role} status/time differs: {item_id}")
        if row.get("official_reference_consumed") is not True or not isinstance(
            row.get("annotator_id"), str
        ) or not str(row.get("annotator_id")).strip():
            raise ValueError(f"MCIF beyond-OCR frozen {role} boundary/annotator differs: {item_id}")
        reason_codes = clean_string_list(row.get("reason_codes"), label="reason codes")
        if not isinstance(row.get("annotation_note"), str):
            raise ValueError(f"MCIF beyond-OCR frozen {role} note differs: {item_id}")
        if role == "visual":
            label_keys = (
                "visual_evidence_correct",
                "candidate_supported_by_visual_evidence",
                "r0_insufficient",
                "r1_insufficient",
                "reason_codes",
                "annotation_note",
                "annotator_id",
            )
            values = [
                row.get("visual_evidence_correct"),
                row.get("candidate_supported_by_visual_evidence"),
                row.get("r0_insufficient"),
            ]
            if source_by_id[item_id]["requires_r1_insufficiency_judgment"]:
                values.append(row.get("r1_insufficient"))
            elif row.get("r1_insufficient") is not None:
                raise ValueError(f"MCIF beyond-OCR frozen R1 label differs: {item_id}")
            if any(value not in JUDGMENTS for value in values):
                raise ValueError(f"MCIF beyond-OCR frozen visual judgment differs: {item_id}")
            expected_gate = all(value == "yes" for value in values)
            if not expected_gate and not reason_codes:
                raise ValueError(f"MCIF beyond-OCR frozen visual reason differs: {item_id}")
        else:
            label_keys = (
                "candidate_eligibility",
                "canonical_source_event_en",
                "acceptable_target_realizations_zh",
                "forbidden_target_realizations_zh",
                "target_reference_alignment",
                "reason_codes",
                "annotation_note",
                "annotator_id",
            )
            if row.get("candidate_eligibility") not in JUDGMENTS or row.get(
                "target_reference_alignment"
            ) not in TARGET_ALIGNMENTS:
                raise ValueError(f"MCIF beyond-OCR frozen target judgment differs: {item_id}")
            canonical = row.get("canonical_source_event_en")
            if not isinstance(canonical, str):
                raise ValueError(f"MCIF beyond-OCR frozen target event differs: {item_id}")
            acceptable = clean_string_list(
                row.get("acceptable_target_realizations_zh"),
                label="acceptable realizations",
            )
            forbidden = clean_string_list(
                row.get("forbidden_target_realizations_zh"),
                label="forbidden realizations",
            )
            if set(acceptable) & set(forbidden):
                raise ValueError(f"MCIF beyond-OCR frozen target overlap differs: {item_id}")
            if row["candidate_eligibility"] == "yes":
                if (
                    not canonical.strip()
                    or not acceptable
                    or row["target_reference_alignment"] not in {"explicit", "paraphrased"}
                ):
                    raise ValueError(f"MCIF beyond-OCR frozen target gate text differs: {item_id}")
            elif any(value not in ("", []) for value in (canonical, acceptable, forbidden)):
                raise ValueError(f"MCIF beyond-OCR frozen rejected target retains text: {item_id}")
            elif not reason_codes:
                raise ValueError(f"MCIF beyond-OCR frozen target reason differs: {item_id}")
            expected_gate = target_gate_passed(row)
        labels = {key: row.get(key) for key in label_keys}
        if row.get("annotation_sha256") != canonical_sha256(labels):
            raise ValueError(f"MCIF beyond-OCR frozen {role} annotation hash differs: {item_id}")
        if row.get("role_gate_passed") is not expected_gate:
            raise ValueError(f"MCIF beyond-OCR frozen {role} gate differs: {item_id}")
        output[item_id] = row
    return output


def join_role_freezes(
    output_root: Path,
    *,
    visual_input_rows: list[dict[str, Any]],
    target_input_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    visual_frozen_rows: list[dict[str, Any]],
    target_frozen_rows: list[dict[str, Any]],
    visual_input_sha256: str,
    target_input_sha256: str,
    mapping_sha256: str,
    visual_frozen_sha256: str,
    target_frozen_sha256: str,
    config_sha256: str,
    joined_at_utc: str,
    expected_items: int,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF beyond-OCR joined output must not already exist")
    if UTC_PATTERN.fullmatch(joined_at_utc) is None:
        raise ValueError("MCIF beyond-OCR join timestamp must use YYYY-MM-DDTHH:MM:SSZ")
    visual_by_id = validate_input_rows(
        visual_input_rows, role="visual", expected_items=expected_items
    )
    target_by_id = validate_input_rows(
        target_input_rows, role="target", expected_items=expected_items
    )
    mapping_by_candidate = validate_mapping_rows(
        mapping_rows, visual_by_id, target_by_id, expected_items=expected_items
    )
    visual_frozen_by_id = validate_frozen_rows(
        visual_frozen_rows, visual_by_id, role="visual"
    )
    target_frozen_by_id = validate_frozen_rows(
        target_frozen_rows, target_by_id, role="target"
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        joined_rows = []
        for candidate_id in sorted(mapping_by_candidate):
            mapping = mapping_by_candidate[candidate_id]
            visual = visual_frozen_by_id[mapping["visual_item_id"]]
            target = target_frozen_by_id[mapping["target_item_id"]]
            visual_passed = visual.get("role_gate_passed") is True
            target_passed = target.get("role_gate_passed") is True
            joint_passed = visual_passed and target_passed
            joined = {
                "schema_version": JOINED_SCHEMA,
                "status": (
                    "BEYOND_OCR_VISUAL_TARGET_VALIDATED_PENDING_AUDIO_SUFFICIENCY"
                    if joint_passed
                    else "BEYOND_OCR_CANDIDATE_REJECTED_BEFORE_AUDIO"
                ),
                "candidate_id": candidate_id,
                "candidate_row_sha256": mapping["candidate_row_sha256"],
                "evidence_tier": mapping["evidence_tier"],
                "talk_id": mapping["talk_id"],
                "segment_id": mapping["segment_id"],
                "talk_segment_index": mapping["talk_segment_index"],
                "source_segment_offset_sec": mapping["source_segment_offset_sec"],
                "source_segment_end_sec": mapping["source_segment_end_sec"],
                "current_state_id": mapping["current_state_id"],
                "current_state_row_sha256": mapping["current_state_row_sha256"],
                "current_evidence_available_sec": mapping[
                    "current_evidence_available_sec"
                ],
                "earliest_contiguous_state_id": mapping[
                    "earliest_contiguous_state_id"
                ],
                "earliest_contiguous_state_row_sha256": mapping[
                    "earliest_contiguous_state_row_sha256"
                ],
                "evidence_available_sec": mapping["earliest_contiguous_evidence_sec"],
                "lead_lower_bound_sec": mapping["lead_lower_bound_sec"],
                "visual_item_id": mapping["visual_item_id"],
                "visual_frozen_row_sha256": visual["row_sha256"],
                "visual_gate_passed": visual_passed,
                "visual_labels": {
                    key: visual[key]
                    for key in (
                        "visual_evidence_correct",
                        "candidate_supported_by_visual_evidence",
                        "r0_insufficient",
                        "r1_insufficient",
                        "reason_codes",
                    )
                },
                "target_item_id": mapping["target_item_id"],
                "target_frozen_row_sha256": target["row_sha256"],
                "target_gate_passed": target_passed,
                "target_labels": {
                    key: target[key]
                    for key in (
                        "candidate_eligibility",
                        "canonical_source_event_en",
                        "acceptable_target_realizations_zh",
                        "forbidden_target_realizations_zh",
                        "target_reference_alignment",
                        "reason_codes",
                    )
                },
                "joint_gate_passed": joint_passed,
                "audio_insufficient_until_sec": None,
                "audio_first_sufficient_sec": None,
                "primary_eligible": None,
                "joined_at_utc": joined_at_utc,
                "official_reference_consumed": True,
            }
            joined["row_sha256"] = canonical_sha256(joined)
            joined_rows.append(joined)
        joined_path = temporary / "joined_candidate_decisions_private.jsonl"
        write_jsonl(joined_path, joined_rows)
        joint_passed_rows = [row for row in joined_rows if row["joint_gate_passed"]]
        report = {
            "schema_version": "mcif_beyond_ocr_visual_target_join_report_v1",
            "status": "VISUAL_TARGET_LABELS_JOINED_PENDING_AUDIO_SUFFICIENCY",
            "items": len(joined_rows),
            "visual_gate_passed": sum(row["visual_gate_passed"] for row in joined_rows),
            "target_gate_passed": sum(row["target_gate_passed"] for row in joined_rows),
            "joint_gate_passed": len(joint_passed_rows),
            "joint_gate_passed_by_tier": dict(
                sorted(Counter(row["evidence_tier"] for row in joint_passed_rows).items())
            ),
            "visual_input_sha256": visual_input_sha256,
            "target_input_sha256": target_input_sha256,
            "mapping_sha256": mapping_sha256,
            "visual_frozen_sha256": visual_frozen_sha256,
            "target_frozen_sha256": target_frozen_sha256,
            "config_sha256": config_sha256,
            "joined_at_utc": joined_at_utc,
            "joined_rows_sha256": file_sha256(joined_path),
            "audio_sufficiency_complete": False,
            "primary_eligibility_complete": False,
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        entries, checksum_sha256 = write_checksums(temporary)
        os.rename(temporary, output_root)
        return {**report, "checksum_entries": entries, "checksum_sha256": checksum_sha256}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def require_hash(path: Path, expected: str, label: str) -> None:
    if file_sha256(path) != expected:
        raise ValueError(f"MCIF beyond-OCR {label} hash differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--role", choices=("visual", "target"), required=True)
    init.add_argument("--input", type=Path, required=True)
    init.add_argument("--expected-input-sha256", required=True)
    init.add_argument("--working", type=Path, required=True)
    init.add_argument("--annotator-id", required=True)
    init.add_argument("--expected-items", type=int, default=152)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--role", choices=("visual", "target"), required=True)
    freeze.add_argument("--input", type=Path, required=True)
    freeze.add_argument("--expected-input-sha256", required=True)
    freeze.add_argument("--working", type=Path, required=True)
    freeze.add_argument("--expected-working-sha256", required=True)
    freeze.add_argument("--annotator-id", required=True)
    freeze.add_argument("--locked-at-utc", required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument("--expected-items", type=int, default=152)
    join = subparsers.add_parser("join")
    for name in (
        "visual-input",
        "target-input",
        "mapping",
        "visual-frozen",
        "target-frozen",
    ):
        join.add_argument(f"--{name}", type=Path, required=True)
        join.add_argument(f"--expected-{name}-sha256", required=True)
    join.add_argument("--joined-at-utc", required=True)
    join.add_argument("--output-root", type=Path, required=True)
    join.add_argument("--expected-items", type=int, default=152)
    args = parser.parse_args()
    load_frozen_config(args.config, args.expected_config_sha256)
    if args.command == "init":
        require_hash(args.input, args.expected_input_sha256, f"{args.role} input")
        if args.working.exists() or args.working.is_symlink():
            raise FileExistsError("MCIF beyond-OCR working sheet must not already exist")
        rows = initialize_working_rows(
            load_jsonl(args.input),
            role=args.role,
            annotator_id=args.annotator_id,
            expected_items=args.expected_items,
        )
        write_jsonl_atomic(args.working, rows)
        print(json.dumps({"role": args.role, "items": len(rows), "sha256": file_sha256(args.working)}))
        return
    if args.command == "freeze":
        require_hash(args.input, args.expected_input_sha256, f"{args.role} input")
        require_hash(args.working, args.expected_working_sha256, f"{args.role} working")
        report = freeze_role(
            args.output_root,
            role=args.role,
            input_rows=load_jsonl(args.input),
            working_rows=load_jsonl(args.working),
            input_sha256=args.expected_input_sha256,
            working_sha256=args.expected_working_sha256,
            config_sha256=args.expected_config_sha256,
            annotator_id=args.annotator_id,
            locked_at_utc=args.locked_at_utc,
            expected_items=args.expected_items,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return
    paths = {
        name.replace("-", "_"): getattr(args, name.replace("-", "_"))
        for name in (
            "visual-input",
            "target-input",
            "mapping",
            "visual-frozen",
            "target-frozen",
        )
    }
    expected = {
        name.replace("-", "_"): getattr(args, f"expected_{name.replace('-', '_')}_sha256")
        for name in (
            "visual-input",
            "target-input",
            "mapping",
            "visual-frozen",
            "target-frozen",
        )
    }
    for name, path in paths.items():
        require_hash(path, expected[name], name)
    report = join_role_freezes(
        args.output_root,
        visual_input_rows=load_jsonl(paths["visual_input"]),
        target_input_rows=load_jsonl(paths["target_input"]),
        mapping_rows=load_jsonl(paths["mapping"]),
        visual_frozen_rows=load_jsonl(paths["visual_frozen"]),
        target_frozen_rows=load_jsonl(paths["target_frozen"]),
        visual_input_sha256=expected["visual_input"],
        target_input_sha256=expected["target_input"],
        mapping_sha256=expected["mapping"],
        visual_frozen_sha256=expected["visual_frozen"],
        target_frozen_sha256=expected["target_frozen"],
        config_sha256=args.expected_config_sha256,
        joined_at_utc=args.joined_at_utc,
        expected_items=args.expected_items,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
