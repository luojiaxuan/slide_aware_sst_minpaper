#!/usr/bin/env python3
"""Run leak-resistant MCIF beyond-OCR reliability-v2 annotation stages."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile
from typing import Any

from scripts.build_mcif_beyond_ocr_reliability_workspace import (
    MAPPING_SCHEMA,
    PRIVATE_TARGET_SCHEMA,
    PRIVATE_VISUAL_SCHEMA,
    TARGET_AUTHOR_SCHEMA,
    TARGET_VALIDATOR_STAGE1_SCHEMA,
    VISUAL_R0_SCHEMA,
)
from scripts.build_mcif_visual_token_controls import canonical_sha256, file_sha256, load_jsonl
from slidesst.data.reliability import cluster_bootstrap_percentile_ci, reliability_report


VISUAL_STAGES = ("r0", "r1", "pixels", "descriptor")
VISUAL_FIELDS = {
    "r0": "r0_support",
    "r1": "r1_support",
    "pixels": "pixel_support",
    "descriptor": "descriptor_fidelity",
}
VISUAL_INPUT_SCHEMAS = {
    "r0": VISUAL_R0_SCHEMA,
    "r1": "mcif_beyond_ocr_visual_r1_item_v2",
    "pixels": "mcif_beyond_ocr_visual_pixels_item_v2",
    "descriptor": "mcif_beyond_ocr_visual_descriptor_item_v2",
}
TARGET_VALIDATOR_STAGE2_SCHEMA = "mcif_beyond_ocr_target_validator_stage2_item_v2"
VISUAL_ADJUDICATION_INPUT_SCHEMA = "mcif_beyond_ocr_visual_adjudication_item_v2"
TARGET_ADJUDICATION_INPUT_SCHEMA = "mcif_beyond_ocr_target_adjudication_item_v2"
EVENT_SCHEMA = "mcif_beyond_ocr_annotation_event_v2"
FROZEN_SCHEMA = "mcif_beyond_ocr_annotation_frozen_v2"
VISUAL_ADJUDICATION_SCHEMA = "mcif_beyond_ocr_visual_adjudication_v2"
TARGET_ADJUDICATION_SCHEMA = "mcif_beyond_ocr_target_adjudication_v2"
UTC_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
JUDGMENTS = {"yes", "no", "uncertain"}
ALIGNMENTS = {"explicit", "paraphrased", "omitted", "unsupported", "uncertain"}
TARGET_STAGE2_DECISIONS = {"accept", "edit", "reject"}
HEX_64 = re.compile(r"[0-9a-f]{64}")


def row_hash_valid(row: dict[str, Any]) -> bool:
    return row.get("row_sha256") == canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )


def load_config(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("MCIF reliability-v2 config hash differs")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "mcif_beyond_ocr_reliability_config_v2":
        raise ValueError("MCIF reliability-v2 config schema differs")
    if [stage.get("name") for stage in config["visual"]["stages"]] != list(VISUAL_STAGES):
        raise ValueError("MCIF reliability-v2 visual stage order differs")
    if set(config["visual"]["judgments"]) != JUDGMENTS:
        raise ValueError("MCIF reliability-v2 visual judgments differ")
    return config


def normalize_identity(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("MCIF reliability-v2 identity must not be empty")
    return " ".join(value.casefold().split())


def require_disjoint_identities(role_ids: dict[str, str]) -> None:
    normalized: dict[str, str] = {}
    for role, value in role_ids.items():
        identity = normalize_identity(value)
        if identity in normalized:
            raise ValueError(
                f"MCIF reliability-v2 roles must be disjoint: {normalized[identity]} and {role}"
            )
        normalized[identity] = role


def load_hmac_key(path: Path) -> bytes:
    key = path.read_bytes()
    if len(key) < 32:
        raise ValueError("MCIF reliability-v2 HMAC key must contain at least 32 bytes")
    return key


def create_hmac_key(path: Path) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("MCIF reliability-v2 HMAC key must not already exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, secrets.token_bytes(32))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_sha256(path)


def signed_payload(payload: dict[str, Any], key: bytes, signature_field: str) -> dict[str, Any]:
    if signature_field in payload:
        raise ValueError(f"Signature field already present: {signature_field}")
    output = dict(payload)
    output[signature_field] = hmac.new(
        key, canonical_sha256(payload).encode(), hashlib.sha256
    ).hexdigest()
    return output


def signature_valid(row: dict[str, Any], key: bytes, signature_field: str) -> bool:
    supplied = row.get(signature_field)
    if not isinstance(supplied, str):
        return False
    payload = {name: value for name, value in row.items() if name != signature_field}
    expected = hmac.new(key, canonical_sha256(payload).encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(supplied, expected)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def exact_keys(row: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(row)
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(f"MCIF reliability-v2 {label} keys differ; extra={extra}, missing={missing}")


def clean_string_list(values: Any, *, label: str) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"MCIF reliability-v2 {label} must be a list of strings")
    cleaned = [value.strip() for value in values]
    if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
        raise ValueError(f"MCIF reliability-v2 {label} contains empty or duplicate values")
    return cleaned


def input_contract(row: dict[str, Any]) -> tuple[str, str]:
    role = row.get("role")
    if role in {"visual_a", "visual_b"}:
        stage = row.get("stage")
        if stage not in VISUAL_STAGES or row.get("schema_version") != VISUAL_INPUT_SCHEMAS[stage]:
            raise ValueError("MCIF reliability-v2 visual input schema/stage differs")
        return role, stage
    if role == "target_author" and row.get("schema_version") == TARGET_AUTHOR_SCHEMA:
        return role, "author"
    if role == "target_validator":
        if row.get("schema_version") == TARGET_VALIDATOR_STAGE1_SCHEMA:
            return role, "independent_alignment"
        if row.get("schema_version") == TARGET_VALIDATOR_STAGE2_SCHEMA:
            return role, "author_text_review"
    if role == "visual_adjudicator" and row.get(
        "schema_version"
    ) == VISUAL_ADJUDICATION_INPUT_SCHEMA:
        return role, "visual_resolution"
    if role == "target_adjudicator" and row.get(
        "schema_version"
    ) == TARGET_ADJUDICATION_INPUT_SCHEMA:
        return role, "target_resolution"
    raise ValueError("MCIF reliability-v2 input role/schema differs")


def input_expected_keys(role: str, stage: str) -> set[str]:
    common = {
        "schema_version",
        "status",
        "role",
        "item_id",
        "candidate_source_en",
        "candidate_kind",
        "candidate_token_count",
        "annotation_status",
        "reason_codes",
        "annotation_note",
        "annotator_id",
        "locked_at_utc",
        "timing_exposed",
        "row_sha256",
    }
    if role in {"visual_a", "visual_b"}:
        keys = {
            *common,
            "stage",
            "r0_text",
            VISUAL_FIELDS[stage],
            "r1_exposed",
            "pixels_exposed",
            "descriptor_exposed",
            "reference_exposed",
        }
        if stage != "r0":
            keys |= {
                "locked_judgments",
                "prior_stage",
                "prior_input_row_sha256",
                "prior_freeze_hmac_sha256",
                "prior_cohort_lock_sha256",
            }
        if stage in {"r1", "pixels", "descriptor"}:
            keys.add("r1_blocks")
        if stage in {"pixels", "descriptor"}:
            keys.add("current_slide")
        if stage == "descriptor":
            keys.add("proposed_evidence_origins")
        return keys
    if role == "visual_adjudicator":
        return {
            *common,
            "stage",
            "primitive_field",
            "released_evidence",
            "visual_a_raw",
            "visual_b_raw",
            "pre_adjudication_row_sha256",
            "adjudicated_judgment",
            "reference_exposed",
        }
    if role == "target_adjudicator":
        return {
            *common,
            "stage",
            "released_source",
            "target_raw",
            "author_scoring_text",
            "validator_edits",
            "pre_adjudication_row_sha256",
            "adjudication_decision",
            "final_canonical_source_event_en",
            "final_acceptable_target_realizations_zh",
            "final_forbidden_target_realizations_zh",
            "slide_or_visual_exposed",
        }
    target_common = {
        *common,
        "source_reference_en",
        "target_reference_zh",
        "slide_or_visual_exposed",
    }
    if role == "target_author":
        return {
            *target_common,
            "candidate_eligibility",
            "canonical_source_event_en",
            "acceptable_target_realizations_zh",
            "forbidden_target_realizations_zh",
            "target_reference_alignment",
        }
    if stage == "independent_alignment":
        return {
            *target_common,
            "stage",
            "candidate_eligibility",
            "target_reference_alignment",
            "author_identity_exposed",
            "author_labels_exposed",
            "author_scoring_text_exposed",
        }
    return {
        *target_common,
        "stage",
        "locked_candidate_eligibility",
        "locked_target_reference_alignment",
        "author_candidate_eligibility",
        "author_canonical_source_event_en",
        "author_acceptable_target_realizations_zh",
        "author_forbidden_target_realizations_zh",
        "author_target_reference_alignment",
        "author_source_input_row_sha256",
        "author_freeze_hmac_sha256",
        "validator_stage1_input_row_sha256",
        "validator_stage1_freeze_hmac_sha256",
        "review_decision",
        "edited_canonical_source_event_en",
        "edited_acceptable_target_realizations_zh",
        "edited_forbidden_target_realizations_zh",
        "author_identity_exposed",
    }


def validate_input_rows(rows: list[dict[str, Any]], expected_items: int) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_items:
        raise ValueError("MCIF reliability-v2 input item count differs")
    output = {}
    contract = None
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in output:
            raise ValueError("MCIF reliability-v2 input id is absent or duplicate")
        current = input_contract(row)
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError("MCIF reliability-v2 input mixes role/stage contracts")
        exact_keys(row, input_expected_keys(*current), "input")
        if not row_hash_valid(row):
            raise ValueError(f"MCIF reliability-v2 input hash differs: {item_id}")
        if row.get("annotation_status") != "pending" or row.get("annotator_id") is not None:
            raise ValueError(f"MCIF reliability-v2 input contains a label: {item_id}")
        output[item_id] = row
    return output


def blank_annotation(role: str, stage: str) -> dict[str, Any]:
    common = {"reason_codes": [], "annotation_note": ""}
    if role in {"visual_a", "visual_b"}:
        return {VISUAL_FIELDS[stage]: None, **common}
    if role == "target_author":
        return {
            "candidate_eligibility": None,
            "canonical_source_event_en": "",
            "acceptable_target_realizations_zh": [],
            "forbidden_target_realizations_zh": [],
            "target_reference_alignment": None,
            **common,
        }
    if role == "visual_adjudicator":
        return {"adjudicated_judgment": None, **common}
    if role == "target_adjudicator":
        return {
            "adjudication_decision": None,
            "final_canonical_source_event_en": "",
            "final_acceptable_target_realizations_zh": [],
            "final_forbidden_target_realizations_zh": [],
            **common,
        }
    if stage == "independent_alignment":
        return {
            "candidate_eligibility": None,
            "target_reference_alignment": None,
            **common,
        }
    return {
        "review_decision": None,
        "edited_canonical_source_event_en": "",
        "edited_acceptable_target_realizations_zh": [],
        "edited_forbidden_target_realizations_zh": [],
        **common,
    }


def event_expected_keys(role: str, stage: str) -> set[str]:
    return {
        "schema_version",
        "role",
        "stage",
        "item_id",
        "source_input_row_sha256",
        "annotator_id",
        "event_index",
        "previous_event_hmac",
        "annotation_status",
        "submitted_at_utc",
        *blank_annotation(role, stage),
        "event_hmac_sha256",
    }


def make_event(
    *,
    source: dict[str, Any],
    annotator_id: str,
    event_index: int,
    previous_event_hmac: str | None,
    annotation_status: str,
    annotation: dict[str, Any],
    submitted_at_utc: str | None,
    key: bytes,
) -> dict[str, Any]:
    role, stage = input_contract(source)
    payload = {
        "schema_version": EVENT_SCHEMA,
        "role": role,
        "stage": stage,
        "item_id": source["item_id"],
        "source_input_row_sha256": source["row_sha256"],
        "annotator_id": annotator_id,
        "event_index": event_index,
        "previous_event_hmac": previous_event_hmac,
        "annotation_status": annotation_status,
        "submitted_at_utc": submitted_at_utc,
        **annotation,
    }
    return signed_payload(payload, key, "event_hmac_sha256")


def validate_annotation(
    annotation: dict[str, Any],
    *,
    role: str,
    stage: str,
    status: str,
    config: dict[str, Any],
) -> None:
    expected = set(blank_annotation(role, stage))
    if set(annotation) != expected:
        raise ValueError("MCIF reliability-v2 annotation fields differ")
    reason_codes = clean_string_list(annotation["reason_codes"], label="reason codes")
    if not isinstance(annotation["annotation_note"], str):
        raise ValueError("MCIF reliability-v2 annotation note must be text")
    allowed_reasons = set(
        config["visual"]["reason_codes"]
        if role in {"visual_a", "visual_b", "visual_adjudicator"}
        else config["target"]["reason_codes"]
    )
    if not set(reason_codes) <= allowed_reasons:
        raise ValueError("MCIF reliability-v2 annotation reason code differs")
    if status == "pending":
        if annotation != blank_annotation(role, stage):
            raise ValueError("MCIF reliability-v2 pending event contains labels")
        return
    if status not in {"draft", "completed"}:
        raise ValueError("MCIF reliability-v2 annotation status differs")
    if role in {"visual_a", "visual_b"}:
        judgment = annotation[VISUAL_FIELDS[stage]]
        if judgment not in JUDGMENTS:
            raise ValueError("MCIF reliability-v2 visual judgment differs")
        if judgment != "yes" and not reason_codes:
            raise ValueError("MCIF reliability-v2 non-yes visual judgment needs a reason")
        return
    if role == "visual_adjudicator":
        if annotation["adjudicated_judgment"] not in {"yes", "no", "unresolvable"}:
            raise ValueError("MCIF reliability-v2 visual adjudication judgment differs")
        if not reason_codes:
            raise ValueError("MCIF reliability-v2 visual adjudication needs a reason")
        return
    if role == "target_adjudicator":
        decision = annotation["adjudication_decision"]
        if decision not in {"accept", "edit", "reject", "unresolvable"}:
            raise ValueError("MCIF reliability-v2 target adjudication decision differs")
        canonical = annotation["final_canonical_source_event_en"]
        acceptable = clean_string_list(
            annotation["final_acceptable_target_realizations_zh"],
            label="adjudicated acceptable realizations",
        )
        forbidden = clean_string_list(
            annotation["final_forbidden_target_realizations_zh"],
            label="adjudicated forbidden realizations",
        )
        if not isinstance(canonical, str) or set(acceptable) & set(forbidden):
            raise ValueError("MCIF reliability-v2 target adjudication scoring text differs")
        if decision in {"accept", "edit"}:
            if not canonical.strip() or not acceptable:
                raise ValueError("MCIF reliability-v2 positive target adjudication lacks scoring text")
        elif canonical or acceptable or forbidden:
            raise ValueError("MCIF reliability-v2 non-positive target adjudication retains scoring text")
        if not reason_codes:
            raise ValueError("MCIF reliability-v2 target adjudication needs a reason")
        return
    if role == "target_author":
        eligibility = annotation["candidate_eligibility"]
        alignment = annotation["target_reference_alignment"]
        canonical = annotation["canonical_source_event_en"]
        acceptable = clean_string_list(
            annotation["acceptable_target_realizations_zh"], label="acceptable realizations"
        )
        forbidden = clean_string_list(
            annotation["forbidden_target_realizations_zh"], label="forbidden realizations"
        )
        if not isinstance(canonical, str) or set(acceptable) & set(forbidden):
            raise ValueError("MCIF reliability-v2 target author scoring text differs")
        if eligibility not in JUDGMENTS or alignment not in ALIGNMENTS:
            raise ValueError("MCIF reliability-v2 target author judgment differs")
        if eligibility == "yes":
            if not canonical.strip() or not acceptable or alignment not in {"explicit", "paraphrased"}:
                raise ValueError("MCIF reliability-v2 eligible target author row lacks scoring text")
        elif canonical or acceptable or forbidden or not reason_codes:
            raise ValueError("MCIF reliability-v2 non-eligible target author row retains scoring text")
        return
    if stage == "independent_alignment":
        if (
            annotation["candidate_eligibility"] not in JUDGMENTS
            or annotation["target_reference_alignment"] not in ALIGNMENTS
        ):
            raise ValueError("MCIF reliability-v2 target validator stage1 judgment differs")
        if (
            annotation["candidate_eligibility"] != "yes"
            or annotation["target_reference_alignment"] not in {"explicit", "paraphrased"}
        ) and not reason_codes:
            raise ValueError("MCIF reliability-v2 target validator stage1 rejection needs a reason")
        return
    decision = annotation["review_decision"]
    if decision not in TARGET_STAGE2_DECISIONS:
        raise ValueError("MCIF reliability-v2 target validator stage2 decision differs")
    canonical = annotation["edited_canonical_source_event_en"]
    acceptable = clean_string_list(
        annotation["edited_acceptable_target_realizations_zh"], label="edited acceptable realizations"
    )
    forbidden = clean_string_list(
        annotation["edited_forbidden_target_realizations_zh"], label="edited forbidden realizations"
    )
    if not isinstance(canonical, str) or set(acceptable) & set(forbidden):
        raise ValueError("MCIF reliability-v2 edited scoring text differs")
    if decision == "edit":
        if not canonical.strip() or not acceptable or not reason_codes:
            raise ValueError("MCIF reliability-v2 target edit lacks scoring text/reason")
    elif canonical or acceptable or forbidden:
        raise ValueError("MCIF reliability-v2 non-edit target review retains edited text")
    if decision == "reject" and not reason_codes:
        raise ValueError("MCIF reliability-v2 target rejection lacks a reason")


def initialize_events(
    input_rows: list[dict[str, Any]],
    *,
    annotator_id: str,
    expected_items: int,
    key: bytes,
) -> list[dict[str, Any]]:
    normalize_identity(annotator_id)
    validate_input_rows(input_rows, expected_items)
    return [
        make_event(
            source=source,
            annotator_id=annotator_id,
            event_index=0,
            previous_event_hmac=None,
            annotation_status="pending",
            annotation=blank_annotation(*input_contract(source)),
            submitted_at_utc=None,
            key=key,
        )
        for source in input_rows
    ]


def validate_event_log(
    events: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
    *,
    annotator_id: str,
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    source_by_id = validate_input_rows(input_rows, expected_items)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        item_id = event.get("item_id")
        if item_id not in source_by_id:
            raise ValueError("MCIF reliability-v2 event item differs")
        grouped[item_id].append(event)
    if set(grouped) != set(source_by_id):
        raise ValueError("MCIF reliability-v2 event/input item sets differ")
    for item_id, item_events in grouped.items():
        source = source_by_id[item_id]
        role, stage = input_contract(source)
        item_events.sort(key=lambda row: row.get("event_index", -1))
        previous = None
        completed = False
        for index, event in enumerate(item_events):
            exact_keys(event, event_expected_keys(role, stage), "event")
            if event["schema_version"] != EVENT_SCHEMA or not signature_valid(
                event, key, "event_hmac_sha256"
            ):
                raise ValueError(f"MCIF reliability-v2 event signature differs: {item_id}")
            if (
                event["role"] != role
                or event["stage"] != stage
                or event["source_input_row_sha256"] != source["row_sha256"]
                or event["annotator_id"] != annotator_id
                or event["event_index"] != index
                or event["previous_event_hmac"] != previous
            ):
                raise ValueError(f"MCIF reliability-v2 event chain differs: {item_id}")
            if completed:
                raise ValueError(f"MCIF reliability-v2 completed event was extended: {item_id}")
            status = event["annotation_status"]
            annotation = {name: event[name] for name in blank_annotation(role, stage)}
            validate_annotation(annotation, role=role, stage=stage, status=status, config=config)
            timestamp = event["submitted_at_utc"]
            if status == "pending":
                if index != 0 or timestamp is not None:
                    raise ValueError(f"MCIF reliability-v2 pending event position differs: {item_id}")
            elif not isinstance(timestamp, str) or UTC_PATTERN.fullmatch(timestamp) is None:
                raise ValueError(f"MCIF reliability-v2 event timestamp differs: {item_id}")
            completed = status == "completed"
            previous = event["event_hmac_sha256"]
    return grouped


def append_annotation_event(
    events: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
    *,
    item_id: str,
    expected_event_index: int,
    annotation_status: str,
    annotation: dict[str, Any],
    submitted_at_utc: str,
    annotator_id: str,
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped = validate_event_log(
        events,
        input_rows,
        annotator_id=annotator_id,
        expected_items=expected_items,
        key=key,
        config=config,
    )
    if item_id not in grouped:
        raise ValueError("MCIF reliability-v2 submitted item differs")
    previous = grouped[item_id][-1]
    if previous["event_index"] != expected_event_index:
        raise ValueError("MCIF reliability-v2 stale event version")
    if previous["annotation_status"] == "completed":
        raise ValueError("MCIF reliability-v2 completed annotation is immutable")
    source = next(row for row in input_rows if row["item_id"] == item_id)
    role, stage = input_contract(source)
    validate_annotation(annotation, role=role, stage=stage, status=annotation_status, config=config)
    event = make_event(
        source=source,
        annotator_id=annotator_id,
        event_index=expected_event_index + 1,
        previous_event_hmac=previous["event_hmac_sha256"],
        annotation_status=annotation_status,
        annotation=annotation,
        submitted_at_utc=submitted_at_utc,
        key=key,
    )
    return [*events, event]


def freeze_annotations(
    output_root: Path,
    *,
    input_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    annotator_id: str,
    expected_items: int,
    locked_at_utc: str,
    config_sha256: str,
    key: bytes,
    config: dict[str, Any],
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF reliability-v2 freeze output must not already exist")
    if UTC_PATTERN.fullmatch(locked_at_utc) is None:
        raise ValueError("MCIF reliability-v2 freeze timestamp differs")
    grouped = validate_event_log(
        events,
        input_rows,
        annotator_id=annotator_id,
        expected_items=expected_items,
        key=key,
        config=config,
    )
    source_by_id = {row["item_id"]: row for row in input_rows}
    frozen_rows = []
    for source in input_rows:
        role, stage = input_contract(source)
        final_event = grouped[source["item_id"]][-1]
        if final_event["annotation_status"] != "completed":
            raise ValueError(f"MCIF reliability-v2 annotation remains incomplete: {source['item_id']}")
        annotation = {name: final_event[name] for name in blank_annotation(role, stage)}
        payload = {
            "schema_version": FROZEN_SCHEMA,
            "status": "HUMAN_ANNOTATION_FROZEN",
            "role": role,
            "stage": stage,
            "item_id": source["item_id"],
            "source_input_row_sha256": source["row_sha256"],
            "annotator_id": annotator_id,
            "final_event_index": final_event["event_index"],
            "final_event_hmac_sha256": final_event["event_hmac_sha256"],
            **annotation,
            "locked_at_utc": locked_at_utc,
            "config_sha256": config_sha256,
        }
        frozen_rows.append(signed_payload(payload, key, "freeze_hmac_sha256"))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        frozen_path = temporary / "frozen_annotations.jsonl"
        write_jsonl(frozen_path, frozen_rows)
        role, stage = input_contract(next(iter(source_by_id.values())))
        report_payload = {
            "schema_version": "mcif_beyond_ocr_freeze_report_v2",
            "status": "HUMAN_ANNOTATION_FROZEN",
            "role": role,
            "stage": stage,
            "items": len(frozen_rows),
            "annotator_id": annotator_id,
            "input_sha256": canonical_sha256(input_rows),
            "event_log_sha256": canonical_sha256(events),
            "frozen_rows_sha256": file_sha256(frozen_path),
            "config_sha256": config_sha256,
            "locked_at_utc": locked_at_utc,
            "audio_release_allowed": False,
            "inference_release_allowed": False,
        }
        report = signed_payload(report_payload, key, "freeze_report_hmac_sha256")
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_root)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_frozen_rows(
    rows: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
    *,
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    source_by_id = validate_input_rows(input_rows, expected_items)
    if len(rows) != expected_items:
        raise ValueError("MCIF reliability-v2 frozen item count differs")
    output = {}
    for row in rows:
        item_id = row.get("item_id")
        if item_id not in source_by_id or item_id in output:
            raise ValueError("MCIF reliability-v2 frozen id differs")
        source = source_by_id[item_id]
        role, stage = input_contract(source)
        expected_keys = {
            "schema_version",
            "status",
            "role",
            "stage",
            "item_id",
            "source_input_row_sha256",
            "annotator_id",
            "final_event_index",
            "final_event_hmac_sha256",
            *blank_annotation(role, stage),
            "locked_at_utc",
            "config_sha256",
            "freeze_hmac_sha256",
        }
        exact_keys(row, expected_keys, "frozen row")
        if (
            row["schema_version"] != FROZEN_SCHEMA
            or row["status"] != "HUMAN_ANNOTATION_FROZEN"
            or row["role"] != role
            or row["stage"] != stage
            or row["source_input_row_sha256"] != source["row_sha256"]
            or not signature_valid(row, key, "freeze_hmac_sha256")
        ):
            raise ValueError(f"MCIF reliability-v2 frozen binding/signature differs: {item_id}")
        annotation = {name: row[name] for name in blank_annotation(role, stage)}
        validate_annotation(annotation, role=role, stage=stage, status="completed", config=config)
        output[item_id] = row
    return output


def cohort_lock_sha256(
    frozen_a: dict[str, dict[str, Any]], frozen_b: dict[str, dict[str, Any]]
) -> str:
    return canonical_sha256(
        {
            "visual_a": [frozen_a[key]["freeze_hmac_sha256"] for key in sorted(frozen_a)],
            "visual_b": [frozen_b[key]["freeze_hmac_sha256"] for key in sorted(frozen_b)],
        }
    )


def validate_private_rows(
    rows: list[dict[str, Any]], *, schema: str, expected_items: int, label: str
) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_items:
        raise ValueError(f"MCIF reliability-v2 private {label} item count differs")
    output = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in output:
            raise ValueError(f"MCIF reliability-v2 private {label} id differs")
        if row.get("schema_version") != schema or not row_hash_valid(row):
            raise ValueError(f"MCIF reliability-v2 private {label} schema/hash differs")
        output[candidate_id] = row
    return output


def validate_private_mapping(
    rows: list[dict[str, Any]], *, expected_items: int
) -> dict[str, dict[str, Any]]:
    output = validate_private_rows(
        rows, schema=MAPPING_SCHEMA, expected_items=expected_items, label="mapping"
    )
    for row in output.values():
        if row.get("human_labels_complete") is not False:
            raise ValueError("MCIF reliability-v2 private mapping contains human labels")
        if row.get("audio_release_allowed") is not False or row.get(
            "inference_release_allowed"
        ) is not False:
            raise ValueError("MCIF reliability-v2 private mapping release firewall differs")
    return output


def visual_item_candidate_index(
    mapping_by_candidate: dict[str, dict[str, Any]], role: str
) -> dict[str, str]:
    field = f"{role}_item_id"
    output = {}
    for candidate_id, row in mapping_by_candidate.items():
        item_id = row[field]
        if item_id in output:
            raise ValueError("MCIF reliability-v2 visual mapping is not one-to-one")
        output[item_id] = candidate_id
    return output


def build_visual_release_row(
    *,
    material: dict[str, Any],
    role: str,
    stage: str,
    prior_input: dict[str, Any],
    prior_frozen: dict[str, Any],
    cohort_lock: str,
    media: dict[str, Any] | None,
) -> dict[str, Any]:
    prior_stage = prior_input["stage"]
    locked_judgments = dict(prior_input.get("locked_judgments", {}))
    locked_judgments[VISUAL_FIELDS[prior_stage]] = prior_frozen[VISUAL_FIELDS[prior_stage]]
    row = {
        "schema_version": VISUAL_INPUT_SCHEMAS[stage],
        "status": f"{stage.upper()}_RELEASED_AFTER_FULL_COHORT_FREEZE",
        "role": role,
        "stage": stage,
        "item_id": material[f"{role}_item_id"],
        "candidate_source_en": material["candidate_source_en"],
        "candidate_kind": material["candidate_kind"],
        "candidate_token_count": material["candidate_token_count"],
        "r0_text": material["r0_text"],
        "locked_judgments": locked_judgments,
        "prior_stage": prior_stage,
        "prior_input_row_sha256": prior_input["row_sha256"],
        "prior_freeze_hmac_sha256": prior_frozen["freeze_hmac_sha256"],
        "prior_cohort_lock_sha256": cohort_lock,
        "annotation_status": "pending",
        VISUAL_FIELDS[stage]: None,
        "reason_codes": [],
        "annotation_note": "",
        "annotator_id": None,
        "locked_at_utc": None,
        "r1_exposed": stage in {"r1", "pixels", "descriptor"},
        "pixels_exposed": stage in {"pixels", "descriptor"},
        "descriptor_exposed": stage == "descriptor",
        "reference_exposed": False,
        "timing_exposed": False,
    }
    if row["r1_exposed"]:
        row["r1_blocks"] = material["r1_blocks"]
    if row["pixels_exposed"]:
        if media is None:
            raise ValueError("MCIF reliability-v2 pixel release lacks media")
        row["current_slide"] = media
    if row["descriptor_exposed"]:
        row["proposed_evidence_origins"] = material["proposed_evidence_origins"]
    row["row_sha256"] = canonical_sha256(row)
    return row


def release_visual_stage(
    output_root: Path,
    *,
    workspace_root: Path,
    private_visual_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    prior_input_a: list[dict[str, Any]],
    prior_input_b: list[dict[str, Any]],
    prior_frozen_a: list[dict[str, Any]],
    prior_frozen_b: list[dict[str, Any]],
    next_stage: str,
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF reliability-v2 visual release must not already exist")
    if next_stage not in VISUAL_STAGES[1:]:
        raise ValueError("MCIF reliability-v2 next visual stage differs")
    prior_stage = VISUAL_STAGES[VISUAL_STAGES.index(next_stage) - 1]
    input_a = validate_input_rows(prior_input_a, expected_items)
    input_b = validate_input_rows(prior_input_b, expected_items)
    if {input_contract(row) for row in input_a.values()} != {("visual_a", prior_stage)}:
        raise ValueError("MCIF reliability-v2 visual A prior stage differs")
    if {input_contract(row) for row in input_b.values()} != {("visual_b", prior_stage)}:
        raise ValueError("MCIF reliability-v2 visual B prior stage differs")
    frozen_a = validate_frozen_rows(
        prior_frozen_a, prior_input_a, expected_items=expected_items, key=key, config=config
    )
    frozen_b = validate_frozen_rows(
        prior_frozen_b, prior_input_b, expected_items=expected_items, key=key, config=config
    )
    id_a = next(iter({row["annotator_id"] for row in frozen_a.values()}))
    id_b = next(iter({row["annotator_id"] for row in frozen_b.values()}))
    require_disjoint_identities({"visual_a": id_a, "visual_b": id_b})
    materials = validate_private_rows(
        private_visual_rows,
        schema=PRIVATE_VISUAL_SCHEMA,
        expected_items=expected_items,
        label="visual material",
    )
    mapping = validate_private_mapping(mapping_rows, expected_items=expected_items)
    if set(materials) != set(mapping):
        raise ValueError("MCIF reliability-v2 visual material/mapping candidate sets differ")
    indices = {
        "visual_a": visual_item_candidate_index(mapping, "visual_a"),
        "visual_b": visual_item_candidate_index(mapping, "visual_b"),
    }
    if set(input_a) != set(indices["visual_a"]) or set(input_b) != set(indices["visual_b"]):
        raise ValueError("MCIF reliability-v2 visual release would change the full cohort")
    cohort_lock = cohort_lock_sha256(frozen_a, frozen_b)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        released: dict[str, list[dict[str, Any]]] = {"visual_a": [], "visual_b": []}
        for role, source_by_id, frozen_by_id in (
            ("visual_a", input_a, frozen_a),
            ("visual_b", input_b, frozen_b),
        ):
            view = temporary / f"{role}_{next_stage}_view"
            view.mkdir()
            for item_id in sorted(source_by_id):
                candidate_id = indices[role][item_id]
                material = materials[candidate_id]
                media = None
                if next_stage in {"pixels", "descriptor"}:
                    private_media = material["private_media"]
                    source_media = workspace_root / "scorer_private" / private_media["private_path"]
                    if file_sha256(source_media) != private_media["sha256"]:
                        raise ValueError("MCIF reliability-v2 private media hash differs")
                    media_name = hashlib.sha256(
                        f"{role}\0{next_stage}\0{candidate_id}".encode()
                    ).hexdigest()[:20] + ".png"
                    target_media = view / "media" / media_name
                    target_media.parent.mkdir(exist_ok=True)
                    shutil.copyfile(source_media, target_media)
                    media = {
                        "path": f"media/{media_name}",
                        "sha256": private_media["sha256"],
                        "width": private_media["width"],
                        "height": private_media["height"],
                    }
                released[role].append(
                    build_visual_release_row(
                        material=material,
                        role=role,
                        stage=next_stage,
                        prior_input=source_by_id[item_id],
                        prior_frozen=frozen_by_id[item_id],
                        cohort_lock=cohort_lock,
                        media=media,
                    )
                )
            write_jsonl(view / "items.jsonl", released[role])
            (view / "README.md").write_text(
                f"# MCIF Beyond-OCR Visual {role} {next_stage}\n\n"
                "This release was created only after both visual cohorts completed the prior stage.\n",
                encoding="utf-8",
            )
        report = {
            "schema_version": "mcif_beyond_ocr_visual_stage_release_report_v2",
            "status": "FULL_COHORT_NEXT_STAGE_RELEASED",
            "prior_stage": prior_stage,
            "released_stage": next_stage,
            "items_per_cohort": expected_items,
            "visual_a_annotator_id": id_a,
            "visual_b_annotator_id": id_b,
            "prior_cohort_lock_sha256": cohort_lock,
            "visual_a_items_sha256": canonical_sha256(released["visual_a"]),
            "visual_b_items_sha256": canonical_sha256(released["visual_b"]),
            "candidate_set_sha256": canonical_sha256(sorted(materials)),
            "audio_release_allowed": False,
            "inference_release_allowed": False,
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_root)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def release_target_validator_stage2(
    output_root: Path,
    *,
    private_target_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    author_input_rows: list[dict[str, Any]],
    author_frozen_rows: list[dict[str, Any]],
    validator_input_rows: list[dict[str, Any]],
    validator_frozen_rows: list[dict[str, Any]],
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF reliability-v2 target stage2 release must not already exist")
    authors = validate_input_rows(author_input_rows, expected_items)
    validators = validate_input_rows(validator_input_rows, expected_items)
    if {input_contract(row) for row in authors.values()} != {("target_author", "author")}:
        raise ValueError("MCIF reliability-v2 target author input contract differs")
    if {input_contract(row) for row in validators.values()} != {
        ("target_validator", "independent_alignment")
    }:
        raise ValueError("MCIF reliability-v2 target validator stage1 input contract differs")
    frozen_authors = validate_frozen_rows(
        author_frozen_rows, author_input_rows, expected_items=expected_items, key=key, config=config
    )
    frozen_validators = validate_frozen_rows(
        validator_frozen_rows,
        validator_input_rows,
        expected_items=expected_items,
        key=key,
        config=config,
    )
    author_id = next(iter({row["annotator_id"] for row in frozen_authors.values()}))
    validator_id = next(iter({row["annotator_id"] for row in frozen_validators.values()}))
    require_disjoint_identities({"target_author": author_id, "target_validator": validator_id})
    private = validate_private_rows(
        private_target_rows,
        schema=PRIVATE_TARGET_SCHEMA,
        expected_items=expected_items,
        label="target material",
    )
    mapping = validate_private_mapping(mapping_rows, expected_items=expected_items)
    if set(private) != set(mapping):
        raise ValueError("MCIF reliability-v2 target material/mapping candidate sets differ")
    author_to_candidate = {
        row["target_author_item_id"]: candidate_id for candidate_id, row in mapping.items()
    }
    validator_to_candidate = {
        row["target_validator_item_id"]: candidate_id for candidate_id, row in mapping.items()
    }
    if set(authors) != set(author_to_candidate) or set(validators) != set(validator_to_candidate):
        raise ValueError("MCIF reliability-v2 target release would change the full cohort")
    author_by_candidate = {author_to_candidate[item_id]: row for item_id, row in authors.items()}
    frozen_author_by_candidate = {
        author_to_candidate[item_id]: row for item_id, row in frozen_authors.items()
    }
    validator_by_candidate = {
        validator_to_candidate[item_id]: row for item_id, row in validators.items()
    }
    frozen_validator_by_candidate = {
        validator_to_candidate[item_id]: row for item_id, row in frozen_validators.items()
    }
    rows = []
    for candidate_id in sorted(mapping, key=lambda value: mapping[value]["target_validator_item_id"]):
        author_source = author_by_candidate[candidate_id]
        author = frozen_author_by_candidate[candidate_id]
        validator_source = validator_by_candidate[candidate_id]
        validator = frozen_validator_by_candidate[candidate_id]
        row = {
            "schema_version": TARGET_VALIDATOR_STAGE2_SCHEMA,
            "status": "AUTHOR_TEXT_RELEASED_AFTER_INDEPENDENT_STAGE1_FREEZE",
            "role": "target_validator",
            "stage": "author_text_review",
            "item_id": validator_source["item_id"],
            "candidate_source_en": validator_source["candidate_source_en"],
            "candidate_kind": validator_source["candidate_kind"],
            "candidate_token_count": validator_source["candidate_token_count"],
            "source_reference_en": validator_source["source_reference_en"],
            "target_reference_zh": validator_source["target_reference_zh"],
            "locked_candidate_eligibility": validator["candidate_eligibility"],
            "locked_target_reference_alignment": validator["target_reference_alignment"],
            "author_candidate_eligibility": author["candidate_eligibility"],
            "author_canonical_source_event_en": author["canonical_source_event_en"],
            "author_acceptable_target_realizations_zh": author[
                "acceptable_target_realizations_zh"
            ],
            "author_forbidden_target_realizations_zh": author[
                "forbidden_target_realizations_zh"
            ],
            "author_target_reference_alignment": author["target_reference_alignment"],
            "author_source_input_row_sha256": author_source["row_sha256"],
            "author_freeze_hmac_sha256": author["freeze_hmac_sha256"],
            "validator_stage1_input_row_sha256": validator_source["row_sha256"],
            "validator_stage1_freeze_hmac_sha256": validator["freeze_hmac_sha256"],
            "annotation_status": "pending",
            "review_decision": None,
            "edited_canonical_source_event_en": "",
            "edited_acceptable_target_realizations_zh": [],
            "edited_forbidden_target_realizations_zh": [],
            "reason_codes": [],
            "annotation_note": "",
            "annotator_id": None,
            "locked_at_utc": None,
            "author_identity_exposed": False,
            "slide_or_visual_exposed": False,
            "timing_exposed": False,
        }
        row["row_sha256"] = canonical_sha256(row)
        rows.append(row)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        write_jsonl(temporary / "items.jsonl", rows)
        report = {
            "schema_version": "mcif_beyond_ocr_target_stage2_release_report_v2",
            "status": "AUTHOR_TEXT_RELEASED_AFTER_INDEPENDENT_STAGE1_FREEZE",
            "items": len(rows),
            "author_id": author_id,
            "target_validator_id": validator_id,
            "items_sha256": canonical_sha256(rows),
            "audio_release_allowed": False,
            "inference_release_allowed": False,
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# MCIF Beyond-OCR Target Validator Stage 2\n\n"
            "Author scoring text was released only after the validator's independent stage-1 freeze.\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_root)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def one_annotator_id(rows: dict[str, dict[str, Any]], label: str) -> str:
    values = {row["annotator_id"] for row in rows.values()}
    if len(values) != 1:
        raise ValueError(f"MCIF reliability-v2 {label} must use one annotator identity")
    return next(iter(values))


def validate_visual_chains(
    *,
    visual_inputs: dict[str, dict[str, list[dict[str, Any]]]],
    visual_frozen: dict[str, dict[str, list[dict[str, Any]]]],
    mapping: dict[str, dict[str, Any]],
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, dict[str, dict[str, Any]]]], dict[str, str]]:
    output: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        stage: {} for stage in VISUAL_STAGES
    }
    identities: dict[str, str] = {}
    prior_inputs_by_role: dict[str, dict[str, dict[str, Any]]] = {}
    prior_frozen_by_role: dict[str, dict[str, dict[str, Any]]] = {}
    prior_cohort_lock = None
    for stage in VISUAL_STAGES:
        current_inputs_by_role: dict[str, dict[str, dict[str, Any]]] = {}
        current_frozen_by_role: dict[str, dict[str, dict[str, Any]]] = {}
        for role in ("visual_a", "visual_b"):
            item_to_candidate = visual_item_candidate_index(mapping, role)
            if stage not in visual_inputs or role not in visual_inputs[stage]:
                raise ValueError(f"MCIF reliability-v2 missing visual input: {role}/{stage}")
            if stage not in visual_frozen or role not in visual_frozen[stage]:
                raise ValueError(f"MCIF reliability-v2 missing visual freeze: {role}/{stage}")
            input_rows = visual_inputs[stage][role]
            source_by_id = validate_input_rows(input_rows, expected_items)
            if {input_contract(row) for row in source_by_id.values()} != {(role, stage)}:
                raise ValueError(f"MCIF reliability-v2 visual role/stage differs: {role}/{stage}")
            if set(source_by_id) != set(item_to_candidate):
                raise ValueError("MCIF reliability-v2 visual chain changed the full cohort")
            frozen_by_id = validate_frozen_rows(
                visual_frozen[stage][role],
                input_rows,
                expected_items=expected_items,
                key=key,
                config=config,
            )
            identity = one_annotator_id(frozen_by_id, f"{role}/{stage}")
            if role in identities and normalize_identity(identities[role]) != normalize_identity(identity):
                raise ValueError(f"MCIF reliability-v2 {role} identity changed across stages")
            identities[role] = identity
            if stage != "r0":
                if role not in prior_inputs_by_role or role not in prior_frozen_by_role or prior_cohort_lock is None:
                    raise AssertionError("visual predecessor state missing")
                for item_id, row in source_by_id.items():
                    if (
                        row["prior_stage"] != VISUAL_STAGES[VISUAL_STAGES.index(stage) - 1]
                        or row["prior_input_row_sha256"]
                        != prior_inputs_by_role[role][item_id]["row_sha256"]
                        or row["prior_freeze_hmac_sha256"]
                        != prior_frozen_by_role[role][item_id]["freeze_hmac_sha256"]
                        or row["prior_cohort_lock_sha256"] != prior_cohort_lock
                    ):
                        raise ValueError(
                            f"MCIF reliability-v2 visual predecessor lock differs: {role}/{stage}/{item_id}"
                        )
            output[stage][role] = {
                item_to_candidate[item_id]: row for item_id, row in frozen_by_id.items()
            }
            current_inputs_by_role[role] = source_by_id
            current_frozen_by_role[role] = frozen_by_id
        frozen_a_by_id = {
            mapping[candidate_id]["visual_a_item_id"]: row
            for candidate_id, row in output[stage]["visual_a"].items()
        }
        frozen_b_by_id = {
            mapping[candidate_id]["visual_b_item_id"]: row
            for candidate_id, row in output[stage]["visual_b"].items()
        }
        prior_cohort_lock = cohort_lock_sha256(frozen_a_by_id, frozen_b_by_id)
        prior_inputs_by_role = current_inputs_by_role
        prior_frozen_by_role = current_frozen_by_role
    return output, identities


def metric_with_cluster_interval(
    pairs: list[tuple[str, str, str]],
    categories: list[str],
    *,
    config: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    point = reliability_report([(label_a, label_b) for _, label_a, label_b in pairs], categories)
    interval = cluster_bootstrap_percentile_ci(
        pairs,
        categories,
        n_resamples=config["reliability"]["bootstrap_samples"],
        seed=config["reliability"]["bootstrap_seed"] + seed_offset,
    )
    return {**point, "talk_cluster_bootstrap_ci95": interval}


def raw_candidate_status(row: dict[str, Any]) -> str:
    if row["requires_adjudication"]:
        return "missing_pending_adjudication"
    visual = row["visual_raw"]
    target = row["target_raw"]
    tier = row["evidence_tier"]
    if tier == "r2_semantic":
        visual_pass = (
            visual["r0_support"]["visual_a"] == "no"
            and visual["r1_support"]["visual_a"] == "no"
            and visual["pixel_support"]["visual_a"] == "yes"
            and visual["descriptor_fidelity"]["visual_a"] == "yes"
        )
    else:
        visual_pass = (
            visual["r0_support"]["visual_a"] == "no"
            and visual["r1_support"]["visual_a"] == "yes"
            and visual["pixel_support"]["visual_a"] == "yes"
            and visual["descriptor_fidelity"]["visual_a"] == "yes"
        )
    target_pass = (
        target["candidate_eligibility"]["author"] == "yes"
        and target["target_reference_alignment"]["author"] in {"explicit", "paraphrased"}
        and target["stage2_review_decision"] == "accept"
    )
    return "eligible_raw" if visual_pass and target_pass else "rejected_raw"


def build_pre_adjudication_report(
    *,
    visual_inputs: dict[str, dict[str, list[dict[str, Any]]]],
    visual_frozen: dict[str, dict[str, list[dict[str, Any]]]],
    target_author_inputs: list[dict[str, Any]],
    target_author_frozen: list[dict[str, Any]],
    target_validator_stage1_inputs: list[dict[str, Any]],
    target_validator_stage1_frozen: list[dict[str, Any]],
    target_validator_stage2_inputs: list[dict[str, Any]],
    target_validator_stage2_frozen: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = validate_private_mapping(mapping_rows, expected_items=expected_items)
    visual, identities = validate_visual_chains(
        visual_inputs=visual_inputs,
        visual_frozen=visual_frozen,
        mapping=mapping,
        expected_items=expected_items,
        key=key,
        config=config,
    )
    author_sources = validate_input_rows(target_author_inputs, expected_items)
    validator1_sources = validate_input_rows(target_validator_stage1_inputs, expected_items)
    validator2_sources = validate_input_rows(target_validator_stage2_inputs, expected_items)
    authors = validate_frozen_rows(
        target_author_frozen,
        target_author_inputs,
        expected_items=expected_items,
        key=key,
        config=config,
    )
    validators1 = validate_frozen_rows(
        target_validator_stage1_frozen,
        target_validator_stage1_inputs,
        expected_items=expected_items,
        key=key,
        config=config,
    )
    validators2 = validate_frozen_rows(
        target_validator_stage2_frozen,
        target_validator_stage2_inputs,
        expected_items=expected_items,
        key=key,
        config=config,
    )
    identities["target_author"] = one_annotator_id(authors, "target author")
    identities["target_validator"] = one_annotator_id(validators1, "target validator stage1")
    validator2_id = one_annotator_id(validators2, "target validator stage2")
    if normalize_identity(validator2_id) != normalize_identity(identities["target_validator"]):
        raise ValueError("MCIF reliability-v2 target validator identity changed across stages")
    require_disjoint_identities(identities)
    author_to_candidate = {
        row["target_author_item_id"]: candidate_id for candidate_id, row in mapping.items()
    }
    validator_to_candidate = {
        row["target_validator_item_id"]: candidate_id for candidate_id, row in mapping.items()
    }
    if (
        set(author_sources) != set(author_to_candidate)
        or set(validator1_sources) != set(validator_to_candidate)
        or set(validator2_sources) != set(validator_to_candidate)
    ):
        raise ValueError("MCIF reliability-v2 target report changed the full cohort")
    author_by_candidate = {author_to_candidate[item_id]: row for item_id, row in authors.items()}
    validator1_by_candidate = {
        validator_to_candidate[item_id]: row for item_id, row in validators1.items()
    }
    validator2_by_candidate = {
        validator_to_candidate[item_id]: row for item_id, row in validators2.items()
    }
    for candidate_id in mapping:
        stage2_source = validator2_sources[mapping[candidate_id]["target_validator_item_id"]]
        author_item_id = mapping[candidate_id]["target_author_item_id"]
        validator_item_id = mapping[candidate_id]["target_validator_item_id"]
        if (
            stage2_source["author_source_input_row_sha256"]
            != author_sources[author_item_id]["row_sha256"]
            or stage2_source["author_freeze_hmac_sha256"]
            != authors[author_item_id]["freeze_hmac_sha256"]
            or stage2_source["validator_stage1_input_row_sha256"]
            != validator1_sources[validator_item_id]["row_sha256"]
            or stage2_source["validator_stage1_freeze_hmac_sha256"]
            != validators1[validator_item_id]["freeze_hmac_sha256"]
        ):
            raise ValueError(
                f"MCIF reliability-v2 target stage2 predecessor lock differs: {candidate_id}"
            )
    rows = []
    metric_pairs: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for candidate_id, mapping_row in sorted(mapping.items()):
        disagreements = []
        visual_raw = {}
        for stage in VISUAL_STAGES:
            field = VISUAL_FIELDS[stage]
            value_a = visual[stage]["visual_a"][candidate_id][field]
            value_b = visual[stage]["visual_b"][candidate_id][field]
            visual_raw[field] = {
                "visual_a": value_a,
                "visual_b": value_b,
                "visual_a_reason_codes": visual[stage]["visual_a"][candidate_id][
                    "reason_codes"
                ],
                "visual_b_reason_codes": visual[stage]["visual_b"][candidate_id][
                    "reason_codes"
                ],
                "visual_a_note": visual[stage]["visual_a"][candidate_id][
                    "annotation_note"
                ],
                "visual_b_note": visual[stage]["visual_b"][candidate_id][
                    "annotation_note"
                ],
            }
            metric_pairs[field].append((mapping_row["talk_id"], value_a, value_b))
            if value_a != value_b:
                disagreements.append(f"visual:{field}:disagreement")
            if "uncertain" in {value_a, value_b}:
                disagreements.append(f"visual:{field}:uncertain")
        author = author_by_candidate[candidate_id]
        validator1 = validator1_by_candidate[candidate_id]
        validator2 = validator2_by_candidate[candidate_id]
        target_raw = {
            "candidate_eligibility": {
                "author": author["candidate_eligibility"],
                "validator": validator1["candidate_eligibility"],
            },
            "target_reference_alignment": {
                "author": author["target_reference_alignment"],
                "validator": validator1["target_reference_alignment"],
            },
            "stage2_review_decision": validator2["review_decision"],
            "author_reason_codes": author["reason_codes"],
            "validator_stage1_reason_codes": validator1["reason_codes"],
            "validator_stage2_reason_codes": validator2["reason_codes"],
        }
        for field in ("candidate_eligibility", "target_reference_alignment"):
            value_a = target_raw[field]["author"]
            value_b = target_raw[field]["validator"]
            metric_pairs[field].append((mapping_row["talk_id"], value_a, value_b))
            if value_a != value_b:
                disagreements.append(f"target:{field}:disagreement")
            if "uncertain" in {value_a, value_b}:
                disagreements.append(f"target:{field}:uncertain")
        if validator2["review_decision"] != "accept":
            disagreements.append(f"target:stage2:{validator2['review_decision']}")
        row = {
            "schema_version": "mcif_beyond_ocr_pre_adjudication_candidate_v2",
            "candidate_id": candidate_id,
            "evidence_tier": mapping_row["evidence_tier"],
            "talk_id": mapping_row["talk_id"],
            "segment_id": mapping_row["segment_id"],
            "current_state_id": mapping_row["current_state_id"],
            "visual_raw": visual_raw,
            "target_raw": target_raw,
            "target_author_scoring_text": {
                "canonical_source_event_en": author["canonical_source_event_en"],
                "acceptable_target_realizations_zh": author[
                    "acceptable_target_realizations_zh"
                ],
                "forbidden_target_realizations_zh": author[
                    "forbidden_target_realizations_zh"
                ],
            },
            "target_validator_edits": {
                "canonical_source_event_en": validator2[
                    "edited_canonical_source_event_en"
                ],
                "acceptable_target_realizations_zh": validator2[
                    "edited_acceptable_target_realizations_zh"
                ],
                "forbidden_target_realizations_zh": validator2[
                    "edited_forbidden_target_realizations_zh"
                ],
            },
            "adjudication_reasons": disagreements,
            "requires_adjudication": bool(disagreements),
            "raw_candidate_status": None,
            "adjudication_applied": False,
            "final_candidate_status": None,
        }
        row["raw_candidate_status"] = raw_candidate_status(row)
        row["row_sha256"] = canonical_sha256(row)
        rows.append(row)
    categories = {
        **{field: list(config["visual"]["judgments"]) for field in VISUAL_FIELDS.values()},
        "candidate_eligibility": list(config["target"]["eligibility_judgments"]),
        "target_reference_alignment": list(config["target"]["reference_alignments"]),
    }
    metrics = {
        field: metric_with_cluster_interval(
            metric_pairs[field], categories[field], config=config, seed_offset=index
        )
        for index, field in enumerate(
            [*VISUAL_FIELDS.values(), "candidate_eligibility", "target_reference_alignment"]
        )
    }
    adjudication_rate = sum(row["requires_adjudication"] for row in rows) / len(rows)
    load_bearing = config["reliability"]["load_bearing_fields"]
    field_gate = {
        field: (
            metrics[field]["exact_agreement"]["value"] is not None
            and metrics[field]["exact_agreement"]["value"]
            >= config["reliability"]["minimum_exact_agreement"]
            and metrics[field]["gwet_ac1"]["value"] is not None
            and metrics[field]["gwet_ac1"]["value"]
            >= config["reliability"]["minimum_gwet_ac1"]
        )
        for field in load_bearing
    }
    gate_passed = all(field_gate.values()) and (
        adjudication_rate <= config["reliability"]["maximum_adjudication_rate"]
    )
    summary = {
        "schema_version": "mcif_beyond_ocr_pre_adjudication_reliability_report_v2",
        "status": (
            "PASS_ADJUDICATION_MAY_BEGIN"
            if gate_passed
            else "FAIL_REVISE_GUIDELINE_AND_RELABEL_ALL"
        ),
        "items": len(rows),
        "talks": len({row["talk_id"] for row in rows}),
        "role_ids": identities,
        "metrics": metrics,
        "load_bearing_field_gate": field_gate,
        "pre_adjudication_composite_exact_agreement": sum(
            not row["requires_adjudication"] for row in rows
        )
        / len(rows),
        "requires_adjudication_count": sum(row["requires_adjudication"] for row in rows),
        "adjudication_rate": adjudication_rate,
        "maximum_adjudication_rate": config["reliability"]["maximum_adjudication_rate"],
        "instrument_gate_passed": gate_passed,
        "failure_action": config["reliability"]["failure_action"],
        "raw_eligible_count": sum(row["raw_candidate_status"] == "eligible_raw" for row in rows),
        "raw_rejected_count": sum(row["raw_candidate_status"] == "rejected_raw" for row in rows),
        "raw_missing_count": sum(
            row["raw_candidate_status"] == "missing_pending_adjudication" for row in rows
        ),
        "adjudication_applied": False,
        "audio_release_allowed": False,
        "inference_release_allowed": False,
    }
    return rows, summary


def adjudication_item_id(namespace: str, candidate_id: str, field: str = "") -> str:
    digest = hashlib.sha256(f"{namespace}\0{candidate_id}\0{field}".encode()).hexdigest()
    prefix = "MCIF-BOR-VA" if namespace == "visual" else "MCIF-BOR-TJ"
    return f"{prefix}-{digest[:16]}"


def prepare_adjudication_release(
    output_root: Path,
    *,
    pre_adjudication_rows: list[dict[str, Any]],
    pre_adjudication_summary: dict[str, Any],
    private_visual_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    target_validator_stage2_inputs: list[dict[str, Any]],
    workspace_root: Path,
    visual_adjudicator_id: str,
    target_adjudicator_id: str,
    expected_items: int,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF reliability-v2 adjudication release must not already exist")
    if pre_adjudication_summary.get("instrument_gate_passed") is not True or pre_adjudication_summary.get(
        "status"
    ) != "PASS_ADJUDICATION_MAY_BEGIN":
        raise ValueError("MCIF reliability-v2 failed instrument cannot enter adjudication")
    if pre_adjudication_summary.get("adjudication_applied") is not False:
        raise ValueError("MCIF reliability-v2 adjudication must consume raw report only")
    if len(pre_adjudication_rows) != expected_items:
        raise ValueError("MCIF reliability-v2 pre-adjudication row count differs")
    report_by_candidate = {}
    for row in pre_adjudication_rows:
        candidate_id = row.get("candidate_id")
        if candidate_id in report_by_candidate or not row_hash_valid(row):
            raise ValueError("MCIF reliability-v2 pre-adjudication row hash/id differs")
        if row.get("adjudication_applied") is not False:
            raise ValueError("MCIF reliability-v2 pre-adjudication row was already modified")
        report_by_candidate[candidate_id] = row
    mapping = validate_private_mapping(mapping_rows, expected_items=expected_items)
    materials = validate_private_rows(
        private_visual_rows,
        schema=PRIVATE_VISUAL_SCHEMA,
        expected_items=expected_items,
        label="visual material",
    )
    if set(report_by_candidate) != set(mapping) or set(materials) != set(mapping):
        raise ValueError("MCIF reliability-v2 adjudication candidate sets differ")
    target_sources = validate_input_rows(target_validator_stage2_inputs, expected_items)
    target_to_candidate = {
        row["target_validator_item_id"]: candidate_id for candidate_id, row in mapping.items()
    }
    if set(target_sources) != set(target_to_candidate):
        raise ValueError("MCIF reliability-v2 target adjudication source set differs")
    target_by_candidate = {
        target_to_candidate[item_id]: row for item_id, row in target_sources.items()
    }
    role_ids = dict(pre_adjudication_summary["role_ids"])
    role_ids.update(
        {
            "visual_adjudicator": visual_adjudicator_id,
            "target_adjudicator": target_adjudicator_id,
        }
    )
    require_disjoint_identities(role_ids)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        visual_root = temporary / "visual_adjudicator_view"
        target_root = temporary / "target_adjudicator_view"
        visual_root.mkdir()
        target_root.mkdir()
        visual_rows = []
        target_rows = []
        for candidate_id, report in sorted(report_by_candidate.items()):
            material = materials[candidate_id]
            for reason in report["adjudication_reasons"]:
                if not reason.startswith("visual:"):
                    continue
                field = reason.split(":", 2)[1]
                if any(row.get("primitive_field") == field and row["pre_adjudication_row_sha256"] == report["row_sha256"] for row in visual_rows):
                    continue
                stage = next(stage for stage, name in VISUAL_FIELDS.items() if name == field)
                evidence: dict[str, Any] = {"r0_text": material["r0_text"]}
                if stage in {"r1", "pixels", "descriptor"}:
                    evidence["r1_blocks"] = material["r1_blocks"]
                if stage in {"pixels", "descriptor"}:
                    private_media = material["private_media"]
                    source_media = workspace_root / "scorer_private" / private_media["private_path"]
                    if file_sha256(source_media) != private_media["sha256"]:
                        raise ValueError("MCIF reliability-v2 adjudication media hash differs")
                    media_name = hashlib.sha256(
                        f"adjudication\0{candidate_id}\0{field}".encode()
                    ).hexdigest()[:20] + ".png"
                    target_media = visual_root / "media" / media_name
                    target_media.parent.mkdir(exist_ok=True)
                    shutil.copyfile(source_media, target_media)
                    evidence["current_slide"] = {
                        "path": f"media/{media_name}",
                        "sha256": private_media["sha256"],
                        "width": private_media["width"],
                        "height": private_media["height"],
                    }
                if stage == "descriptor":
                    evidence["proposed_evidence_origins"] = material[
                        "proposed_evidence_origins"
                    ]
                raw = report["visual_raw"][field]
                row = {
                    "schema_version": VISUAL_ADJUDICATION_INPUT_SCHEMA,
                    "status": "PENDING_APPEND_ONLY_VISUAL_ADJUDICATION",
                    "role": "visual_adjudicator",
                    "stage": "visual_resolution",
                    "item_id": adjudication_item_id("visual", candidate_id, field),
                    "candidate_source_en": material["candidate_source_en"],
                    "candidate_kind": material["candidate_kind"],
                    "candidate_token_count": material["candidate_token_count"],
                    "primitive_field": field,
                    "released_evidence": evidence,
                    "visual_a_raw": {
                        "judgment": raw["visual_a"],
                        "reason_codes": raw["visual_a_reason_codes"],
                        "note": raw["visual_a_note"],
                    },
                    "visual_b_raw": {
                        "judgment": raw["visual_b"],
                        "reason_codes": raw["visual_b_reason_codes"],
                        "note": raw["visual_b_note"],
                    },
                    "pre_adjudication_row_sha256": report["row_sha256"],
                    "annotation_status": "pending",
                    "adjudicated_judgment": None,
                    "reason_codes": [],
                    "annotation_note": "",
                    "annotator_id": None,
                    "locked_at_utc": None,
                    "reference_exposed": False,
                    "timing_exposed": False,
                }
                row["row_sha256"] = canonical_sha256(row)
                visual_rows.append(row)
            if any(reason.startswith("target:") for reason in report["adjudication_reasons"]):
                source = target_by_candidate[candidate_id]
                row = {
                    "schema_version": TARGET_ADJUDICATION_INPUT_SCHEMA,
                    "status": "PENDING_APPEND_ONLY_TARGET_ADJUDICATION",
                    "role": "target_adjudicator",
                    "stage": "target_resolution",
                    "item_id": adjudication_item_id("target", candidate_id),
                    "candidate_source_en": source["candidate_source_en"],
                    "candidate_kind": source["candidate_kind"],
                    "candidate_token_count": source["candidate_token_count"],
                    "released_source": {
                        "source_reference_en": source["source_reference_en"],
                        "target_reference_zh": source["target_reference_zh"],
                    },
                    "target_raw": report["target_raw"],
                    "author_scoring_text": report["target_author_scoring_text"],
                    "validator_edits": report["target_validator_edits"],
                    "pre_adjudication_row_sha256": report["row_sha256"],
                    "annotation_status": "pending",
                    "adjudication_decision": None,
                    "final_canonical_source_event_en": "",
                    "final_acceptable_target_realizations_zh": [],
                    "final_forbidden_target_realizations_zh": [],
                    "reason_codes": [],
                    "annotation_note": "",
                    "annotator_id": None,
                    "locked_at_utc": None,
                    "slide_or_visual_exposed": False,
                    "timing_exposed": False,
                }
                row["row_sha256"] = canonical_sha256(row)
                target_rows.append(row)
        visual_rows.sort(key=lambda row: row["item_id"])
        target_rows.sort(key=lambda row: row["item_id"])
        write_jsonl(visual_root / "items.jsonl", visual_rows)
        write_jsonl(target_root / "items.jsonl", target_rows)
        report = {
            "schema_version": "mcif_beyond_ocr_adjudication_release_report_v2",
            "status": "ROLE_SPECIFIC_ADJUDICATION_RELEASED_AFTER_INSTRUMENT_PASS",
            "visual_items": len(visual_rows),
            "target_items": len(target_rows),
            "visual_adjudicator_id": visual_adjudicator_id,
            "target_adjudicator_id": target_adjudicator_id,
            "pre_adjudication_rows_sha256": canonical_sha256(pre_adjudication_rows),
            "raw_metrics_recomputed": False,
            "audio_release_allowed": False,
            "inference_release_allowed": False,
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_root)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def apply_adjudications(
    *,
    pre_adjudication_rows: list[dict[str, Any]],
    pre_adjudication_summary: dict[str, Any],
    visual_adjudication_inputs: list[dict[str, Any]],
    visual_adjudication_frozen: list[dict[str, Any]],
    target_adjudication_inputs: list[dict[str, Any]],
    target_adjudication_frozen: list[dict[str, Any]],
    key: bytes,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if pre_adjudication_summary.get("instrument_gate_passed") is not True:
        raise ValueError("MCIF reliability-v2 failed instrument cannot apply adjudication")
    report_by_sha = {}
    for row in pre_adjudication_rows:
        if not row_hash_valid(row) or row.get("adjudication_applied") is not False:
            raise ValueError("MCIF reliability-v2 adjudication input is not a raw report row")
        report_by_sha[row["row_sha256"]] = row
    visual_sources = validate_input_rows(
        visual_adjudication_inputs, len(visual_adjudication_inputs)
    ) if visual_adjudication_inputs else {}
    target_sources = validate_input_rows(
        target_adjudication_inputs, len(target_adjudication_inputs)
    ) if target_adjudication_inputs else {}
    visual_frozen = validate_frozen_rows(
        visual_adjudication_frozen,
        visual_adjudication_inputs,
        expected_items=len(visual_adjudication_inputs),
        key=key,
        config=config,
    ) if visual_adjudication_inputs else {}
    target_frozen = validate_frozen_rows(
        target_adjudication_frozen,
        target_adjudication_inputs,
        expected_items=len(target_adjudication_inputs),
        key=key,
        config=config,
    ) if target_adjudication_inputs else {}
    if bool(visual_sources) != bool(visual_frozen) or bool(target_sources) != bool(target_frozen):
        raise ValueError("MCIF reliability-v2 adjudication input/freeze presence differs")
    role_ids = dict(pre_adjudication_summary["role_ids"])
    if visual_frozen:
        role_ids["visual_adjudicator"] = one_annotator_id(
            visual_frozen, "visual adjudicator"
        )
    if target_frozen:
        role_ids["target_adjudicator"] = one_annotator_id(
            target_frozen, "target adjudicator"
        )
    require_disjoint_identities(role_ids)
    visual_by_key = {}
    for item_id, source in visual_sources.items():
        key_tuple = (source["pre_adjudication_row_sha256"], source["primitive_field"])
        if key_tuple in visual_by_key or key_tuple[0] not in report_by_sha:
            raise ValueError("MCIF reliability-v2 visual adjudication task binding differs")
        visual_by_key[key_tuple] = visual_frozen[item_id]
    target_by_sha = {}
    for item_id, source in target_sources.items():
        row_sha = source["pre_adjudication_row_sha256"]
        if row_sha in target_by_sha or row_sha not in report_by_sha:
            raise ValueError("MCIF reliability-v2 target adjudication task binding differs")
        target_by_sha[row_sha] = target_frozen[item_id]
    expected_visual = set()
    expected_target = set()
    for row in pre_adjudication_rows:
        for reason in row["adjudication_reasons"]:
            if reason.startswith("visual:"):
                expected_visual.add((row["row_sha256"], reason.split(":", 2)[1]))
            elif reason.startswith("target:"):
                expected_target.add(row["row_sha256"])
    if set(visual_by_key) != expected_visual or set(target_by_sha) != expected_target:
        raise ValueError("MCIF reliability-v2 adjudication tasks do not exactly cover triggers")
    output = []
    for raw in pre_adjudication_rows:
        final_visual = {}
        unresolved = False
        for field, pair in raw["visual_raw"].items():
            task = visual_by_key.get((raw["row_sha256"], field))
            if task is None:
                final_visual[field] = pair["visual_a"]
            elif task["adjudicated_judgment"] == "unresolvable":
                final_visual[field] = None
                unresolved = True
            else:
                final_visual[field] = task["adjudicated_judgment"]
        target_task = target_by_sha.get(raw["row_sha256"])
        if target_task is None:
            target_decision = "accept"
            final_scoring_text = dict(raw["target_author_scoring_text"])
        else:
            target_decision = target_task["adjudication_decision"]
            if target_decision in {"accept", "edit"}:
                final_scoring_text = {
                    "canonical_source_event_en": target_task[
                        "final_canonical_source_event_en"
                    ],
                    "acceptable_target_realizations_zh": target_task[
                        "final_acceptable_target_realizations_zh"
                    ],
                    "forbidden_target_realizations_zh": target_task[
                        "final_forbidden_target_realizations_zh"
                    ],
                }
            else:
                final_scoring_text = None
        if target_decision == "unresolvable":
            unresolved = True
        if unresolved:
            final_status = "missing_unresolvable"
        elif target_decision == "reject":
            final_status = "rejected_adjudicated"
        else:
            if raw["evidence_tier"] == "r2_semantic":
                visual_pass = (
                    final_visual["r0_support"] == "no"
                    and final_visual["r1_support"] == "no"
                    and final_visual["pixel_support"] == "yes"
                    and final_visual["descriptor_fidelity"] == "yes"
                )
            else:
                visual_pass = (
                    final_visual["r0_support"] == "no"
                    and final_visual["r1_support"] == "yes"
                    and final_visual["pixel_support"] == "yes"
                    and final_visual["descriptor_fidelity"] == "yes"
                )
            final_status = "eligible_adjudicated" if visual_pass else "rejected_adjudicated"
        row = {
            **{key_name: value for key_name, value in raw.items() if key_name != "row_sha256"},
            "adjudication_applied": True,
            "final_visual_judgments": final_visual,
            "final_target_decision": target_decision,
            "final_scoring_text": final_scoring_text,
            "final_candidate_status": final_status,
            "raw_row_sha256": raw["row_sha256"],
        }
        row["row_sha256"] = canonical_sha256(row)
        output.append(row)
    summary = {
        **pre_adjudication_summary,
        "status": "ADJUDICATION_APPLIED_RAW_RELIABILITY_UNCHANGED",
        "adjudication_applied": True,
        "raw_metrics_recomputed": False,
        "resolved_eligible_count": sum(
            row["final_candidate_status"] == "eligible_adjudicated" for row in output
        ),
        "resolved_rejected_count": sum(
            row["final_candidate_status"] == "rejected_adjudicated" for row in output
        ),
        "unresolvable_count": sum(
            row["final_candidate_status"] == "missing_unresolvable" for row in output
        ),
        "audio_release_allowed": False,
        "inference_release_allowed": False,
    }
    return output, summary


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def command_init_key(args: argparse.Namespace) -> None:
    print(json.dumps({"key_sha256": create_hmac_key(args.output)}))


def command_init_events(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.input)
    events = initialize_events(
        rows,
        annotator_id=args.annotator_id,
        expected_items=args.expected_items,
        key=load_hmac_key(args.hmac_key),
    )
    write_jsonl_atomic(args.output, events)
    print(json.dumps({"items": len(rows), "events": len(events)}))


def command_append_event(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.input)
    events = load_jsonl(args.event_log)
    config = load_config(args.config, args.expected_config_sha256)
    updated = append_annotation_event(
        events,
        rows,
        item_id=args.item_id,
        expected_event_index=args.expected_event_index,
        annotation_status=args.annotation_status,
        annotation=load_json(args.annotation_json),
        submitted_at_utc=args.submitted_at_utc,
        annotator_id=args.annotator_id,
        expected_items=args.expected_items,
        key=load_hmac_key(args.hmac_key),
        config=config,
    )
    write_jsonl_atomic(args.event_log, updated)
    print(json.dumps({"events": len(updated), "item_id": args.item_id}))


def command_freeze(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.expected_config_sha256)
    report = freeze_annotations(
        args.output_root,
        input_rows=load_jsonl(args.input),
        events=load_jsonl(args.event_log),
        annotator_id=args.annotator_id,
        expected_items=args.expected_items,
        locked_at_utc=args.locked_at_utc,
        config_sha256=args.expected_config_sha256,
        key=load_hmac_key(args.hmac_key),
        config=config,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def command_release_visual(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.expected_config_sha256)
    report = release_visual_stage(
        args.output_root,
        workspace_root=args.workspace_root,
        private_visual_rows=load_jsonl(args.private_visual),
        mapping_rows=load_jsonl(args.mapping),
        prior_input_a=load_jsonl(args.prior_input_a),
        prior_input_b=load_jsonl(args.prior_input_b),
        prior_frozen_a=load_jsonl(args.prior_frozen_a),
        prior_frozen_b=load_jsonl(args.prior_frozen_b),
        next_stage=args.next_stage,
        expected_items=args.expected_items,
        key=load_hmac_key(args.hmac_key),
        config=config,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def command_release_target_stage2(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.expected_config_sha256)
    report = release_target_validator_stage2(
        args.output_root,
        private_target_rows=load_jsonl(args.private_target),
        mapping_rows=load_jsonl(args.mapping),
        author_input_rows=load_jsonl(args.author_input),
        author_frozen_rows=load_jsonl(args.author_frozen),
        validator_input_rows=load_jsonl(args.validator_stage1_input),
        validator_frozen_rows=load_jsonl(args.validator_stage1_frozen),
        expected_items=args.expected_items,
        key=load_hmac_key(args.hmac_key),
        config=config,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def load_visual_run_manifest(path: Path) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, dict[str, list[dict[str, Any]]]],
]:
    manifest = load_json(path)
    inputs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    frozen: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for stage in VISUAL_STAGES:
        inputs[stage] = {}
        frozen[stage] = {}
        for role in ("visual_a", "visual_b"):
            entry = manifest.get(stage, {}).get(role, {})
            if set(entry) != {"input", "frozen"}:
                raise ValueError(f"Visual run manifest entry differs: {stage}/{role}")
            inputs[stage][role] = load_jsonl(Path(entry["input"]))
            frozen[stage][role] = load_jsonl(Path(entry["frozen"]))
    return inputs, frozen


def command_report(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.expected_config_sha256)
    visual_inputs, visual_frozen = load_visual_run_manifest(args.visual_run_manifest)
    rows, summary = build_pre_adjudication_report(
        visual_inputs=visual_inputs,
        visual_frozen=visual_frozen,
        target_author_inputs=load_jsonl(args.target_author_input),
        target_author_frozen=load_jsonl(args.target_author_frozen),
        target_validator_stage1_inputs=load_jsonl(args.target_validator_stage1_input),
        target_validator_stage1_frozen=load_jsonl(args.target_validator_stage1_frozen),
        target_validator_stage2_inputs=load_jsonl(args.target_validator_stage2_input),
        target_validator_stage2_frozen=load_jsonl(args.target_validator_stage2_frozen),
        mapping_rows=load_jsonl(args.mapping),
        expected_items=args.expected_items,
        key=load_hmac_key(args.hmac_key),
        config=config,
    )
    write_jsonl(args.output, rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def command_prepare_adjudication(args: argparse.Namespace) -> None:
    report = prepare_adjudication_release(
        args.output_root,
        pre_adjudication_rows=load_jsonl(args.pre_adjudication_rows),
        pre_adjudication_summary=load_json(args.pre_adjudication_summary),
        private_visual_rows=load_jsonl(args.private_visual),
        mapping_rows=load_jsonl(args.mapping),
        target_validator_stage2_inputs=load_jsonl(args.target_validator_stage2_input),
        workspace_root=args.workspace_root,
        visual_adjudicator_id=args.visual_adjudicator_id,
        target_adjudicator_id=args.target_adjudicator_id,
        expected_items=args.expected_items,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def command_apply_adjudication(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.expected_config_sha256)
    visual_inputs = load_jsonl(args.visual_adjudication_input)
    target_inputs = load_jsonl(args.target_adjudication_input)
    rows, summary = apply_adjudications(
        pre_adjudication_rows=load_jsonl(args.pre_adjudication_rows),
        pre_adjudication_summary=load_json(args.pre_adjudication_summary),
        visual_adjudication_inputs=visual_inputs,
        visual_adjudication_frozen=(
            load_jsonl(args.visual_adjudication_frozen) if visual_inputs else []
        ),
        target_adjudication_inputs=target_inputs,
        target_adjudication_frozen=(
            load_jsonl(args.target_adjudication_frozen) if target_inputs else []
        ),
        key=load_hmac_key(args.hmac_key),
        config=config,
    )
    write_jsonl(args.output, rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)


def add_key_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hmac-key", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_key = subparsers.add_parser("init-key")
    init_key.add_argument("--output", type=Path, required=True)
    init_key.set_defaults(handler=command_init_key)

    init_events = subparsers.add_parser("init-events")
    init_events.add_argument("--input", type=Path, required=True)
    init_events.add_argument("--output", type=Path, required=True)
    init_events.add_argument("--annotator-id", required=True)
    init_events.add_argument("--expected-items", type=int, required=True)
    add_key_argument(init_events)
    init_events.set_defaults(handler=command_init_events)

    append_event = subparsers.add_parser("append-event")
    append_event.add_argument("--input", type=Path, required=True)
    append_event.add_argument("--event-log", type=Path, required=True)
    append_event.add_argument("--annotation-json", type=Path, required=True)
    append_event.add_argument("--item-id", required=True)
    append_event.add_argument("--expected-event-index", type=int, required=True)
    append_event.add_argument("--annotation-status", choices=["draft", "completed"], required=True)
    append_event.add_argument("--submitted-at-utc", required=True)
    append_event.add_argument("--annotator-id", required=True)
    append_event.add_argument("--expected-items", type=int, required=True)
    add_config_arguments(append_event)
    add_key_argument(append_event)
    append_event.set_defaults(handler=command_append_event)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--input", type=Path, required=True)
    freeze.add_argument("--event-log", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument("--annotator-id", required=True)
    freeze.add_argument("--expected-items", type=int, required=True)
    freeze.add_argument("--locked-at-utc", required=True)
    add_config_arguments(freeze)
    add_key_argument(freeze)
    freeze.set_defaults(handler=command_freeze)

    release_visual = subparsers.add_parser("release-visual")
    release_visual.add_argument("--workspace-root", type=Path, required=True)
    release_visual.add_argument("--private-visual", type=Path, required=True)
    release_visual.add_argument("--mapping", type=Path, required=True)
    release_visual.add_argument("--prior-input-a", type=Path, required=True)
    release_visual.add_argument("--prior-input-b", type=Path, required=True)
    release_visual.add_argument("--prior-frozen-a", type=Path, required=True)
    release_visual.add_argument("--prior-frozen-b", type=Path, required=True)
    release_visual.add_argument("--next-stage", choices=list(VISUAL_STAGES[1:]), required=True)
    release_visual.add_argument("--expected-items", type=int, required=True)
    release_visual.add_argument("--output-root", type=Path, required=True)
    add_config_arguments(release_visual)
    add_key_argument(release_visual)
    release_visual.set_defaults(handler=command_release_visual)

    release_target = subparsers.add_parser("release-target-stage2")
    release_target.add_argument("--private-target", type=Path, required=True)
    release_target.add_argument("--mapping", type=Path, required=True)
    release_target.add_argument("--author-input", type=Path, required=True)
    release_target.add_argument("--author-frozen", type=Path, required=True)
    release_target.add_argument("--validator-stage1-input", type=Path, required=True)
    release_target.add_argument("--validator-stage1-frozen", type=Path, required=True)
    release_target.add_argument("--expected-items", type=int, required=True)
    release_target.add_argument("--output-root", type=Path, required=True)
    add_config_arguments(release_target)
    add_key_argument(release_target)
    release_target.set_defaults(handler=command_release_target_stage2)

    report = subparsers.add_parser("report")
    report.add_argument("--visual-run-manifest", type=Path, required=True)
    report.add_argument("--target-author-input", type=Path, required=True)
    report.add_argument("--target-author-frozen", type=Path, required=True)
    report.add_argument("--target-validator-stage1-input", type=Path, required=True)
    report.add_argument("--target-validator-stage1-frozen", type=Path, required=True)
    report.add_argument("--target-validator-stage2-input", type=Path, required=True)
    report.add_argument("--target-validator-stage2-frozen", type=Path, required=True)
    report.add_argument("--mapping", type=Path, required=True)
    report.add_argument("--expected-items", type=int, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--summary-out", type=Path, required=True)
    add_config_arguments(report)
    add_key_argument(report)
    report.set_defaults(handler=command_report)

    prepare_adjudication = subparsers.add_parser("prepare-adjudication")
    prepare_adjudication.add_argument("--pre-adjudication-rows", type=Path, required=True)
    prepare_adjudication.add_argument("--pre-adjudication-summary", type=Path, required=True)
    prepare_adjudication.add_argument("--private-visual", type=Path, required=True)
    prepare_adjudication.add_argument("--mapping", type=Path, required=True)
    prepare_adjudication.add_argument("--target-validator-stage2-input", type=Path, required=True)
    prepare_adjudication.add_argument("--workspace-root", type=Path, required=True)
    prepare_adjudication.add_argument("--visual-adjudicator-id", required=True)
    prepare_adjudication.add_argument("--target-adjudicator-id", required=True)
    prepare_adjudication.add_argument("--expected-items", type=int, required=True)
    prepare_adjudication.add_argument("--output-root", type=Path, required=True)
    prepare_adjudication.set_defaults(handler=command_prepare_adjudication)

    apply_adjudication = subparsers.add_parser("apply-adjudication")
    apply_adjudication.add_argument("--pre-adjudication-rows", type=Path, required=True)
    apply_adjudication.add_argument("--pre-adjudication-summary", type=Path, required=True)
    apply_adjudication.add_argument("--visual-adjudication-input", type=Path, required=True)
    apply_adjudication.add_argument("--visual-adjudication-frozen", type=Path, required=True)
    apply_adjudication.add_argument("--target-adjudication-input", type=Path, required=True)
    apply_adjudication.add_argument("--target-adjudication-frozen", type=Path, required=True)
    apply_adjudication.add_argument("--output", type=Path, required=True)
    apply_adjudication.add_argument("--summary-out", type=Path, required=True)
    add_config_arguments(apply_adjudication)
    add_key_argument(apply_adjudication)
    apply_adjudication.set_defaults(handler=command_apply_adjudication)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
