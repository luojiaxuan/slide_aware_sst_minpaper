#!/usr/bin/env python3
"""Run leak-resistant MCIF beyond-OCR reliability-v2 annotation stages."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.build_mcif_beyond_ocr_reliability_workspace import (
    MAPPING_SCHEMA,
    PRIVATE_TARGET_SCHEMA,
    PRIVATE_VISUAL_SCHEMA,
    RUN_CONTRACT_SCHEMA,
    TARGET_AUTHOR_SCHEMA,
    TARGET_VALIDATOR_STAGE1_SCHEMA,
    VISUAL_R0_SCHEMA,
    validate_role_access_token_hashes,
)
from scripts.build_mcif_visual_token_controls import (
    canonical_sha256,
    file_sha256,
    load_jsonl,
)
from slidesst.data.reliability import (
    cluster_bootstrap_percentile_ci,
    reliability_report,
)

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
EVENT_HEAD_CHECKPOINT_SCHEMA = "mcif_beyond_ocr_event_head_checkpoint_v2"
FROZEN_SCHEMA = "mcif_beyond_ocr_annotation_frozen_v2"
VISUAL_ADJUDICATION_SCHEMA = "mcif_beyond_ocr_visual_adjudication_v2"
TARGET_ADJUDICATION_SCHEMA = "mcif_beyond_ocr_target_adjudication_v2"
IDENTITY_REGISTRY_SCHEMA = "mcif_beyond_ocr_identity_registry_v2"
RELEASE_SIGNATURE_FIELD = "release_hmac_sha256"
UTC_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
JUDGMENTS = {"yes", "no", "uncertain"}
ALIGNMENTS = {"explicit", "paraphrased", "omitted", "unsupported", "uncertain"}
TARGET_STAGE2_DECISIONS = {"accept", "edit", "reject"}
HEX_64 = re.compile(r"[0-9a-f]{64}")
R1_BLOCK_KEYS = {"content_kind", "label", "content", "bbox_norm", "reading_order"}
EVIDENCE_ORIGIN_BLOCK_KEYS = {
    "block_id",
    "content",
    "content_kind",
    "content_sha256",
    "label",
}
EVIDENCE_ORIGIN_DESCRIPTOR_KEYS = {
    "descriptor_field",
    "descriptor_index",
    "descriptor_sha256",
    "descriptor_text",
}
PRIVATE_MEDIA_KEYS = {"private_media_id", "private_path", "sha256", "width", "height"}
PUBLIC_MEDIA_KEYS = {"path", "sha256", "width", "height"}


def row_hash_valid(row: dict[str, Any]) -> bool:
    return row.get("row_sha256") == canonical_sha256(
        {
            key: value
            for key, value in row.items()
            if key not in {"row_sha256", RELEASE_SIGNATURE_FIELD}
        }
    )


def validate_r1_blocks(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError("MCIF reliability-v2 R1 blocks must be a list")
    output = []
    for value in values:
        if not isinstance(value, dict) or set(value) != R1_BLOCK_KEYS:
            raise ValueError("MCIF reliability-v2 R1 block keys differ")
        bbox = value["bbox_norm"]
        if (
            not isinstance(value["content_kind"], str)
            or not isinstance(value["label"], str)
            or not isinstance(value["content"], str)
            or (
                value["reading_order"] is not None
                and (
                    not isinstance(value["reading_order"], int)
                    or isinstance(value["reading_order"], bool)
                )
            )
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                not isinstance(number, (int, float)) or isinstance(number, bool)
                for number in bbox
            )
        ):
            raise ValueError("MCIF reliability-v2 R1 block value differs")
        output.append({name: value[name] for name in R1_BLOCK_KEYS})
    return output


def validate_evidence_origins(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError("MCIF reliability-v2 evidence origins must be a list")
    output = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("MCIF reliability-v2 evidence origin keys differ")
        keys = set(value)
        if keys == EVIDENCE_ORIGIN_DESCRIPTOR_KEYS:
            valid = (
                isinstance(value["descriptor_field"], str)
                and isinstance(value["descriptor_index"], int)
                and not isinstance(value["descriptor_index"], bool)
                and isinstance(value["descriptor_text"], str)
                and isinstance(value["descriptor_sha256"], str)
                and HEX_64.fullmatch(value["descriptor_sha256"]) is not None
                and value["descriptor_sha256"]
                == canonical_sha256(value["descriptor_text"])
            )
        elif keys == EVIDENCE_ORIGIN_BLOCK_KEYS:
            valid = (
                isinstance(value["block_id"], int)
                and not isinstance(value["block_id"], bool)
                and all(
                    isinstance(value[name], str)
                    for name in ("content", "content_kind", "content_sha256", "label")
                )
                and HEX_64.fullmatch(value["content_sha256"]) is not None
                and value["content_sha256"] == canonical_sha256(value["content"])
            )
        else:
            raise ValueError("MCIF reliability-v2 evidence origin keys differ")
        if not valid:
            raise ValueError("MCIF reliability-v2 evidence origin value differs")
        output.append({name: value[name] for name in keys})
    return output


def validate_media_descriptor(value: Any, *, private: bool) -> dict[str, Any]:
    expected_keys = PRIVATE_MEDIA_KEYS if private else PUBLIC_MEDIA_KEYS
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("MCIF reliability-v2 media descriptor keys differ")
    path_field = "private_path" if private else "path"
    if (
        not isinstance(value[path_field], str)
        or not value[path_field]
        or not isinstance(value["sha256"], str)
        or HEX_64.fullmatch(value["sha256"]) is None
        or any(
            not isinstance(value[field], int)
            or isinstance(value[field], bool)
            or value[field] <= 0
            for field in ("width", "height")
        )
        or (private and not isinstance(value["private_media_id"], str))
    ):
        raise ValueError("MCIF reliability-v2 media descriptor value differs")
    return {name: value[name] for name in expected_keys}


def load_config(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("MCIF reliability-v2 config hash differs")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "mcif_beyond_ocr_reliability_config_v2":
        raise ValueError("MCIF reliability-v2 config schema differs")
    if [stage.get("name") for stage in config["visual"]["stages"]] != list(
        VISUAL_STAGES
    ):
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


def identity_registry_payload(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value for name, value in registry.items() if name != "registry_sha256"
    }


def validate_identity_registry(
    registry: dict[str, Any], config: dict[str, Any]
) -> dict[str, str]:
    exact_keys(
        registry,
        {"schema_version", "people", "role_assignments", "registry_sha256"},
        "identity registry",
    )
    if registry["schema_version"] != IDENTITY_REGISTRY_SCHEMA or registry[
        "registry_sha256"
    ] != canonical_sha256(identity_registry_payload(registry)):
        raise ValueError("MCIF reliability-v2 identity registry hash/schema differs")
    people = registry["people"]
    if not isinstance(people, list) or not people:
        raise ValueError("MCIF reliability-v2 identity registry has no people")
    people_by_id = {}
    alias_owner = {}
    for person in people:
        if not isinstance(person, dict):
            raise ValueError("MCIF reliability-v2 identity registry person differs")
        exact_keys(person, {"person_id", "aliases"}, "identity registry person")
        person_id = person["person_id"]
        aliases = person["aliases"]
        if (
            not isinstance(person_id, str)
            or not person_id.strip()
            or person_id in people_by_id
            or not isinstance(aliases, list)
            or not aliases
        ):
            raise ValueError("MCIF reliability-v2 identity registry person id differs")
        normalized_aliases = {
            normalize_identity(value) for value in [person_id, *aliases]
        }
        if len(normalized_aliases) != len([person_id, *aliases]):
            raise ValueError("MCIF reliability-v2 identity registry aliases duplicate")
        for alias in normalized_aliases:
            if alias in alias_owner:
                raise ValueError(
                    "MCIF reliability-v2 identity registry alias maps to multiple people"
                )
            alias_owner[alias] = person_id
        people_by_id[person_id] = person
    assignments = registry["role_assignments"]
    required_roles = set(config["identity"]["required_disjoint_roles"])
    if not isinstance(assignments, dict) or set(assignments) != required_roles:
        raise ValueError("MCIF reliability-v2 identity registry role set differs")
    if any(person_id not in people_by_id for person_id in assignments.values()):
        raise ValueError("MCIF reliability-v2 identity registry assignment differs")
    require_disjoint_identities(assignments)
    return dict(assignments)


def registered_annotator_id(
    registry: dict[str, Any], config: dict[str, Any], *, role: str
) -> str:
    assignments = validate_identity_registry(registry, config)
    if role not in assignments:
        raise ValueError(f"MCIF reliability-v2 identity registry lacks role: {role}")
    return assignments[role]


def load_identity_registry(
    path: Path, expected_sha256: str, config: dict[str, Any]
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("MCIF reliability-v2 identity registry file hash differs")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("MCIF reliability-v2 identity registry must be an object")
    validate_identity_registry(registry, config)
    return registry


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


def create_access_token(path: Path) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("MCIF reliability-v2 access token must not already exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, secrets.token_urlsafe(32).encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_sha256(path)


def signed_payload(
    payload: dict[str, Any], key: bytes, signature_field: str
) -> dict[str, Any]:
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
    expected = hmac.new(
        key, canonical_sha256(payload).encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(supplied, expected)


def sign_release_row(row: dict[str, Any], key: bytes) -> dict[str, Any]:
    if not row_hash_valid(row):
        raise ValueError("MCIF reliability-v2 release row hash differs before signing")
    return signed_payload(row, key, RELEASE_SIGNATURE_FIELD)


def release_signature_valid(row: dict[str, Any], key: bytes) -> bool:
    return signature_valid(row, key, RELEASE_SIGNATURE_FIELD)


RUN_CONTRACT_KEYS = {
    "schema_version",
    "status",
    "builder_git_commit",
    "config_payload_sha256",
    "config_file_sha256",
    "identity_registry_sha256",
    "release_key_sha256",
    "source_workspace_hf_revision",
    "source_visual_sha256",
    "source_target_sha256",
    "source_mapping_sha256",
    "private_visual_rows_sha256",
    "private_target_rows_sha256",
    "private_mapping_rows_sha256",
    "role_access_token_sha256",
    "expected_items",
    "required_disjoint_roles",
    "audio_release_allowed",
    "inference_release_allowed",
    "contract_hmac_sha256",
}


def validate_run_contract(
    contract: dict[str, Any],
    *,
    key: bytes,
    config: dict[str, Any],
    identity_registry: dict[str, Any] | None = None,
) -> str:
    exact_keys(contract, RUN_CONTRACT_KEYS, "run contract")
    hash_fields = (
        "config_payload_sha256",
        "config_file_sha256",
        "identity_registry_sha256",
        "release_key_sha256",
        "source_visual_sha256",
        "source_target_sha256",
        "source_mapping_sha256",
        "private_visual_rows_sha256",
        "private_target_rows_sha256",
        "private_mapping_rows_sha256",
        "contract_hmac_sha256",
    )
    if any(
        not isinstance(contract[field], str)
        or HEX_64.fullmatch(contract[field]) is None
        for field in hash_fields
    ):
        raise ValueError("MCIF reliability-v2 run contract hash differs")
    if any(
        not isinstance(contract[field], str)
        or re.fullmatch(r"[0-9a-f]{40}", contract[field]) is None
        for field in ("builder_git_commit", "source_workspace_hf_revision")
    ):
        raise ValueError("MCIF reliability-v2 run contract revision differs")
    if (
        contract["schema_version"] != RUN_CONTRACT_SCHEMA
        or contract["status"] != "PRE_ANNOTATION_RUN_CONTRACT_FROZEN"
        or not signature_valid(contract, key, "contract_hmac_sha256")
        or contract["release_key_sha256"] != hashlib.sha256(key).hexdigest()
        or contract["config_payload_sha256"] != canonical_sha256(config)
        or contract["required_disjoint_roles"]
        != sorted(config["identity"]["required_disjoint_roles"])
        or contract["audio_release_allowed"] is not False
        or contract["inference_release_allowed"] is not False
        or not isinstance(contract["expected_items"], int)
        or isinstance(contract["expected_items"], bool)
        or contract["expected_items"] <= 0
    ):
        raise ValueError("MCIF reliability-v2 run contract binding differs")
    validate_role_access_token_hashes(
        contract["role_access_token_sha256"],
        config["identity"]["required_disjoint_roles"],
    )
    if identity_registry is not None:
        validate_identity_registry(identity_registry, config)
        if contract["identity_registry_sha256"] != identity_registry["registry_sha256"]:
            raise ValueError("MCIF reliability-v2 run contract registry differs")
    return canonical_sha256(contract)


def validate_contract_stage_item_count(
    contract: dict[str, Any], *, expected_items: int, role: str
) -> None:
    contract_items = contract["expected_items"]
    if role in {"visual_adjudicator", "target_adjudicator"}:
        if expected_items <= 0 or expected_items > contract_items:
            raise ValueError("MCIF reliability-v2 adjudication item count differs")
    elif expected_items != contract_items:
        raise ValueError("MCIF reliability-v2 run contract item count differs")


def validate_private_bundle_binding(
    rows: list[dict[str, Any]],
    contract: dict[str, Any],
    *,
    contract_field: str,
    label: str,
) -> None:
    if canonical_sha256(rows) != contract[contract_field]:
        raise ValueError(f"MCIF reliability-v2 private {label} contract differs")


def load_run_contract(
    path: Path,
    expected_file_sha256: str,
    *,
    key: bytes,
    config: dict[str, Any],
    identity_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or file_sha256(path) != expected_file_sha256
    ):
        raise ValueError("MCIF reliability-v2 run contract file hash differs")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("MCIF reliability-v2 run contract must be an object")
    validate_run_contract(
        contract, key=key, config=config, identity_registry=identity_registry
    )
    return contract


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
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


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if os.fstat(descriptor).st_mode & 0o777 != 0o600:
            raise ValueError(
                "MCIF reliability-v2 event-head ledger permissions must be 0600"
            )
        payload = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("MCIF reliability-v2 event-head append failed")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def event_log_lock(event_log: Path):
    lock_path = event_log.with_name(f".{event_log.name}.lock")
    if lock_path.is_symlink():
        raise ValueError("MCIF reliability-v2 event lock cannot be a symlink")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def exact_keys(row: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(row)
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(
            f"MCIF reliability-v2 {label} keys differ; extra={extra}, missing={missing}"
        )


def clean_string_list(values: Any, *, label: str) -> list[str]:
    if not isinstance(values, list) or any(
        not isinstance(value, str) for value in values
    ):
        raise ValueError(f"MCIF reliability-v2 {label} must be a list of strings")
    cleaned = [value.strip() for value in values]
    if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
        raise ValueError(
            f"MCIF reliability-v2 {label} contains empty or duplicate values"
        )
    return cleaned


def input_contract(row: dict[str, Any]) -> tuple[str, str]:
    role = row.get("role")
    if role in {"visual_a", "visual_b"}:
        stage = row.get("stage")
        if (
            stage not in VISUAL_STAGES
            or row.get("schema_version") != VISUAL_INPUT_SCHEMAS[stage]
        ):
            raise ValueError("MCIF reliability-v2 visual input schema/stage differs")
        return role, stage
    if role == "target_author" and row.get("schema_version") == TARGET_AUTHOR_SCHEMA:
        return role, "author"
    if role == "target_validator":
        if row.get("schema_version") == TARGET_VALIDATOR_STAGE1_SCHEMA:
            return role, "independent_alignment"
        if row.get("schema_version") == TARGET_VALIDATOR_STAGE2_SCHEMA:
            return role, "author_text_review"
    if (
        role == "visual_adjudicator"
        and row.get("schema_version") == VISUAL_ADJUDICATION_INPUT_SCHEMA
    ):
        return role, "visual_resolution"
    if (
        role == "target_adjudicator"
        and row.get("schema_version") == TARGET_ADJUDICATION_INPUT_SCHEMA
    ):
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
        RELEASE_SIGNATURE_FIELD,
        "run_contract_sha256",
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


def expected_input_status(role: str, stage: str) -> str:
    if role in {"visual_a", "visual_b"}:
        if stage == "r0":
            return "R0_RELEASED_NO_LATER_VISUAL_EVIDENCE"
        return f"{stage.upper()}_RELEASED_AFTER_FULL_COHORT_FREEZE"
    if role == "target_author":
        return "PENDING_INDEPENDENT_TARGET_AUTHORING"
    if role == "target_validator":
        if stage == "independent_alignment":
            return "STAGE1_RELEASED_NO_AUTHOR_TEXT"
        return "AUTHOR_TEXT_RELEASED_AFTER_INDEPENDENT_STAGE1_FREEZE"
    if role == "visual_adjudicator":
        return "PENDING_APPEND_ONLY_VISUAL_ADJUDICATION"
    return "PENDING_APPEND_ONLY_TARGET_ADJUDICATION"


def validate_input_semantics(
    row: dict[str, Any],
    *,
    role: str,
    stage: str,
    key: bytes,
    run_contract_sha256: str,
) -> None:
    item_id = row["item_id"]
    if row["status"] != expected_input_status(role, stage):
        raise ValueError(f"MCIF reliability-v2 input release status differs: {item_id}")
    if row["annotation_status"] != "pending" or row["annotator_id"] is not None:
        raise ValueError(f"MCIF reliability-v2 input contains a label: {item_id}")
    if row["locked_at_utc"] is not None or row["timing_exposed"] is not False:
        raise ValueError(
            f"MCIF reliability-v2 input release firewall differs: {item_id}"
        )
    expected_blank = blank_annotation(role, stage)
    if any(row[name] != value for name, value in expected_blank.items()):
        raise ValueError(
            f"MCIF reliability-v2 input response fields are not blank: {item_id}"
        )
    if not release_signature_valid(row, key):
        raise ValueError(
            f"MCIF reliability-v2 input release signature differs: {item_id}"
        )
    if row["run_contract_sha256"] != run_contract_sha256:
        raise ValueError(f"MCIF reliability-v2 input run contract differs: {item_id}")

    if role in {"visual_a", "visual_b"}:
        expected_flags = {
            "r1_exposed": stage in {"r1", "pixels", "descriptor"},
            "pixels_exposed": stage in {"pixels", "descriptor"},
            "descriptor_exposed": stage == "descriptor",
            "reference_exposed": False,
        }
        if any(row[name] is not value for name, value in expected_flags.items()):
            raise ValueError(
                f"MCIF reliability-v2 visual exposure firewall differs: {item_id}"
            )
        if stage != "r0":
            prior_stages = VISUAL_STAGES[: VISUAL_STAGES.index(stage)]
            expected_locked = {VISUAL_FIELDS[name] for name in prior_stages}
            locked = row["locked_judgments"]
            if not isinstance(locked, dict) or set(locked) != expected_locked:
                raise ValueError(
                    f"MCIF reliability-v2 locked visual fields differ: {item_id}"
                )
            if any(value not in JUDGMENTS for value in locked.values()):
                raise ValueError(
                    f"MCIF reliability-v2 locked visual judgment differs: {item_id}"
                )
        if stage in {"r1", "pixels", "descriptor"}:
            validate_r1_blocks(row["r1_blocks"])
        if stage in {"pixels", "descriptor"}:
            validate_media_descriptor(row["current_slide"], private=False)
        if stage == "descriptor":
            validate_evidence_origins(row["proposed_evidence_origins"])
        return

    if role == "target_author":
        if row["slide_or_visual_exposed"] is not False:
            raise ValueError(
                f"MCIF reliability-v2 target visual firewall differs: {item_id}"
            )
        return
    if role == "target_validator":
        if (
            row["slide_or_visual_exposed"] is not False
            or row["author_identity_exposed"] is not False
        ):
            raise ValueError(
                f"MCIF reliability-v2 target validator firewall differs: {item_id}"
            )
        if stage == "independent_alignment" and (
            row["author_labels_exposed"] is not False
            or row["author_scoring_text_exposed"] is not False
        ):
            raise ValueError(
                f"MCIF reliability-v2 target stage1 firewall differs: {item_id}"
            )
        return
    if role == "visual_adjudicator":
        if row["reference_exposed"] is not False:
            raise ValueError(
                f"MCIF reliability-v2 visual adjudication firewall differs: {item_id}"
            )
        evidence = row["released_evidence"]
        expected_evidence_keys = {
            "r0_support": {"r0_text"},
            "r1_support": {"r0_text", "r1_blocks"},
            "pixel_support": {"r0_text", "r1_blocks", "current_slide"},
            "descriptor_fidelity": {
                "r0_text",
                "r1_blocks",
                "current_slide",
                "proposed_evidence_origins",
            },
        }
        if (
            not isinstance(evidence, dict)
            or row["primitive_field"] not in expected_evidence_keys
            or set(evidence) != expected_evidence_keys[row["primitive_field"]]
            or not isinstance(evidence["r0_text"], str)
        ):
            raise ValueError(
                f"MCIF reliability-v2 visual adjudication evidence differs: {item_id}"
            )
        if "r1_blocks" in evidence:
            validate_r1_blocks(evidence["r1_blocks"])
        if "current_slide" in evidence:
            validate_media_descriptor(evidence["current_slide"], private=False)
        if "proposed_evidence_origins" in evidence:
            validate_evidence_origins(evidence["proposed_evidence_origins"])
        for field in ("visual_a_raw", "visual_b_raw"):
            raw = row[field]
            if not isinstance(raw, dict) or set(raw) != {
                "judgment",
                "reason_codes",
                "note",
            }:
                raise ValueError(
                    f"MCIF reliability-v2 visual adjudication raw row differs: {item_id}"
                )
            if raw["judgment"] not in JUDGMENTS or not isinstance(raw["note"], str):
                raise ValueError(
                    f"MCIF reliability-v2 visual adjudication raw value differs: {item_id}"
                )
            clean_string_list(raw["reason_codes"], label="visual adjudication reasons")
        return
    if row["slide_or_visual_exposed"] is not False:
        raise ValueError(
            f"MCIF reliability-v2 target adjudication firewall differs: {item_id}"
        )
    if not isinstance(row["released_source"], dict) or set(row["released_source"]) != {
        "source_reference_en",
        "target_reference_zh",
    }:
        raise ValueError(
            f"MCIF reliability-v2 target adjudication source differs: {item_id}"
        )
    target_raw_keys = {
        "candidate_eligibility",
        "target_reference_alignment",
        "stage2_review_decision",
        "author_reason_codes",
        "validator_stage1_reason_codes",
        "validator_stage2_reason_codes",
    }
    scoring_keys = {
        "canonical_source_event_en",
        "acceptable_target_realizations_zh",
        "forbidden_target_realizations_zh",
    }
    if (
        not isinstance(row["target_raw"], dict)
        or set(row["target_raw"]) != target_raw_keys
        or any(
            not isinstance(row[field], dict) or set(row[field]) != scoring_keys
            for field in ("author_scoring_text", "validator_edits")
        )
    ):
        raise ValueError(
            f"MCIF reliability-v2 target adjudication nested row differs: {item_id}"
        )


def validate_input_rows(
    rows: list[dict[str, Any]],
    expected_items: int,
    *,
    key: bytes,
    run_contract_sha256: str,
) -> dict[str, dict[str, Any]]:
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
        validate_input_semantics(
            row,
            role=current[0],
            stage=current[1],
            key=key,
            run_contract_sha256=run_contract_sha256,
        )
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
        "run_contract_sha256",
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
        "run_contract_sha256": source["run_contract_sha256"],
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
            raise ValueError(
                "MCIF reliability-v2 non-yes visual judgment needs a reason"
            )
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
            raise ValueError(
                "MCIF reliability-v2 target adjudication scoring text differs"
            )
        if decision in {"accept", "edit"}:
            if not canonical.strip() or not acceptable:
                raise ValueError(
                    "MCIF reliability-v2 positive target adjudication lacks scoring text"
                )
        elif canonical or acceptable or forbidden:
            raise ValueError(
                "MCIF reliability-v2 non-positive target adjudication retains scoring text"
            )
        if not reason_codes:
            raise ValueError("MCIF reliability-v2 target adjudication needs a reason")
        return
    if role == "target_author":
        eligibility = annotation["candidate_eligibility"]
        alignment = annotation["target_reference_alignment"]
        canonical = annotation["canonical_source_event_en"]
        acceptable = clean_string_list(
            annotation["acceptable_target_realizations_zh"],
            label="acceptable realizations",
        )
        forbidden = clean_string_list(
            annotation["forbidden_target_realizations_zh"],
            label="forbidden realizations",
        )
        if not isinstance(canonical, str) or set(acceptable) & set(forbidden):
            raise ValueError("MCIF reliability-v2 target author scoring text differs")
        if eligibility not in JUDGMENTS or alignment not in ALIGNMENTS:
            raise ValueError("MCIF reliability-v2 target author judgment differs")
        if eligibility == "yes":
            if (
                not canonical.strip()
                or not acceptable
                or alignment not in {"explicit", "paraphrased"}
            ):
                raise ValueError(
                    "MCIF reliability-v2 eligible target author row lacks scoring text"
                )
        elif canonical or acceptable or forbidden or not reason_codes:
            raise ValueError(
                "MCIF reliability-v2 non-eligible target author row retains scoring text"
            )
        return
    if stage == "independent_alignment":
        if (
            annotation["candidate_eligibility"] not in JUDGMENTS
            or annotation["target_reference_alignment"] not in ALIGNMENTS
        ):
            raise ValueError(
                "MCIF reliability-v2 target validator stage1 judgment differs"
            )
        if (
            annotation["candidate_eligibility"] != "yes"
            or annotation["target_reference_alignment"]
            not in {"explicit", "paraphrased"}
        ) and not reason_codes:
            raise ValueError(
                "MCIF reliability-v2 target validator stage1 rejection needs a reason"
            )
        return
    decision = annotation["review_decision"]
    if decision not in TARGET_STAGE2_DECISIONS:
        raise ValueError("MCIF reliability-v2 target validator stage2 decision differs")
    canonical = annotation["edited_canonical_source_event_en"]
    acceptable = clean_string_list(
        annotation["edited_acceptable_target_realizations_zh"],
        label="edited acceptable realizations",
    )
    forbidden = clean_string_list(
        annotation["edited_forbidden_target_realizations_zh"],
        label="edited forbidden realizations",
    )
    if not isinstance(canonical, str) or set(acceptable) & set(forbidden):
        raise ValueError("MCIF reliability-v2 edited scoring text differs")
    if decision == "edit":
        if not canonical.strip() or not acceptable or not reason_codes:
            raise ValueError(
                "MCIF reliability-v2 target edit lacks scoring text/reason"
            )
    elif canonical or acceptable or forbidden:
        raise ValueError(
            "MCIF reliability-v2 non-edit target review retains edited text"
        )
    if decision == "reject" and not reason_codes:
        raise ValueError("MCIF reliability-v2 target rejection lacks a reason")


def initialize_events(
    input_rows: list[dict[str, Any]],
    *,
    annotator_id: str,
    expected_items: int,
    key: bytes,
    run_contract_sha256: str,
) -> list[dict[str, Any]]:
    normalize_identity(annotator_id)
    validate_input_rows(
        input_rows,
        expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
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
    run_contract_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    source_by_id = validate_input_rows(
        input_rows,
        expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
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
                raise ValueError(
                    f"MCIF reliability-v2 event signature differs: {item_id}"
                )
            if (
                event["role"] != role
                or event["stage"] != stage
                or event["source_input_row_sha256"] != source["row_sha256"]
                or event["run_contract_sha256"] != run_contract_sha256
                or event["annotator_id"] != annotator_id
                or event["event_index"] != index
                or event["previous_event_hmac"] != previous
            ):
                raise ValueError(f"MCIF reliability-v2 event chain differs: {item_id}")
            if completed:
                raise ValueError(
                    f"MCIF reliability-v2 completed event was extended: {item_id}"
                )
            status = event["annotation_status"]
            annotation = {name: event[name] for name in blank_annotation(role, stage)}
            validate_annotation(
                annotation, role=role, stage=stage, status=status, config=config
            )
            timestamp = event["submitted_at_utc"]
            if status == "pending":
                if index != 0 or timestamp is not None:
                    raise ValueError(
                        f"MCIF reliability-v2 pending event position differs: {item_id}"
                    )
            elif (
                not isinstance(timestamp, str)
                or UTC_PATTERN.fullmatch(timestamp) is None
            ):
                raise ValueError(
                    f"MCIF reliability-v2 event timestamp differs: {item_id}"
                )
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
    run_contract_sha256: str,
) -> list[dict[str, Any]]:
    grouped = validate_event_log(
        events,
        input_rows,
        annotator_id=annotator_id,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
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
    validate_annotation(
        annotation, role=role, stage=stage, status=annotation_status, config=config
    )
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


EVENT_HEAD_CHECKPOINT_KEYS = {
    "schema_version",
    "status",
    "role",
    "stage",
    "annotator_id",
    "source_input_rows_sha256",
    "run_contract_sha256",
    "checkpoint_index",
    "event_count",
    "event_log_head_sha256",
    "previous_checkpoint_hmac",
    "checkpoint_hmac_sha256",
}


def make_event_head_checkpoint(
    events: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
    *,
    annotator_id: str,
    run_contract_sha256: str,
    checkpoint_index: int,
    previous_checkpoint_hmac: str | None,
    key: bytes,
) -> dict[str, Any]:
    contracts = {input_contract(row) for row in input_rows}
    if len(contracts) != 1:
        raise ValueError("MCIF reliability-v2 event-head ledger mixes roles/stages")
    role, stage = next(iter(contracts))
    payload = {
        "schema_version": EVENT_HEAD_CHECKPOINT_SCHEMA,
        "status": "SCORER_PRIVATE_EVENT_HEAD_CHECKPOINT",
        "role": role,
        "stage": stage,
        "annotator_id": annotator_id,
        "source_input_rows_sha256": canonical_sha256(input_rows),
        "run_contract_sha256": run_contract_sha256,
        "checkpoint_index": checkpoint_index,
        "event_count": len(events),
        "event_log_head_sha256": canonical_sha256(events),
        "previous_checkpoint_hmac": previous_checkpoint_hmac,
    }
    return signed_payload(payload, key, "checkpoint_hmac_sha256")


def build_event_head_ledger(
    events: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
    *,
    annotator_id: str,
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
    run_contract_sha256: str,
) -> list[dict[str, Any]]:
    validate_event_log(
        events,
        input_rows,
        annotator_id=annotator_id,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    if len(events) < expected_items:
        raise ValueError("MCIF reliability-v2 event-head ledger event count differs")
    checkpoints = []
    previous = None
    for checkpoint_index, event_count in enumerate(
        range(expected_items, len(events) + 1)
    ):
        checkpoint = make_event_head_checkpoint(
            events[:event_count],
            input_rows,
            annotator_id=annotator_id,
            run_contract_sha256=run_contract_sha256,
            checkpoint_index=checkpoint_index,
            previous_checkpoint_hmac=previous,
            key=key,
        )
        checkpoints.append(checkpoint)
        previous = checkpoint["checkpoint_hmac_sha256"]
    return checkpoints


def validate_event_head_ledger(
    checkpoints: list[dict[str, Any]],
    events: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
    *,
    annotator_id: str,
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    validate_event_log(
        events,
        input_rows,
        annotator_id=annotator_id,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    if not checkpoints:
        raise ValueError("MCIF reliability-v2 event-head ledger is empty")
    contracts = {input_contract(row) for row in input_rows}
    if len(contracts) != 1:
        raise ValueError("MCIF reliability-v2 event-head ledger mixes roles/stages")
    role, stage = next(iter(contracts))
    source_sha256 = canonical_sha256(input_rows)
    previous = None
    for index, checkpoint in enumerate(checkpoints):
        exact_keys(checkpoint, EVENT_HEAD_CHECKPOINT_KEYS, "event-head checkpoint")
        event_count = expected_items + index
        if (
            checkpoint["schema_version"] != EVENT_HEAD_CHECKPOINT_SCHEMA
            or checkpoint["status"] != "SCORER_PRIVATE_EVENT_HEAD_CHECKPOINT"
            or checkpoint["role"] != role
            or checkpoint["stage"] != stage
            or checkpoint["annotator_id"] != annotator_id
            or checkpoint["source_input_rows_sha256"] != source_sha256
            or checkpoint["run_contract_sha256"] != run_contract_sha256
            or not isinstance(checkpoint["checkpoint_index"], int)
            or isinstance(checkpoint["checkpoint_index"], bool)
            or checkpoint["checkpoint_index"] != index
            or not isinstance(checkpoint["event_count"], int)
            or isinstance(checkpoint["event_count"], bool)
            or checkpoint["event_count"] != event_count
            or event_count > len(events)
            or checkpoint["event_log_head_sha256"]
            != canonical_sha256(events[:event_count])
            or checkpoint["previous_checkpoint_hmac"] != previous
            or not signature_valid(checkpoint, key, "checkpoint_hmac_sha256")
        ):
            raise ValueError("MCIF reliability-v2 event-head checkpoint differs")
        previous = checkpoint["checkpoint_hmac_sha256"]
    if checkpoints[-1]["event_count"] != len(events):
        raise ValueError("MCIF reliability-v2 event-head ledger is not current")
    return checkpoints[-1]


def initialize_event_log(
    event_log: Path,
    head_ledger: Path,
    input_rows: list[dict[str, Any]],
    *,
    annotator_id: str,
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
    run_contract_sha256: str,
) -> list[dict[str, Any]]:
    if event_log.resolve(strict=False) == head_ledger.resolve(strict=False):
        raise ValueError("MCIF reliability-v2 event log and ledger must be distinct")
    events = initialize_events(
        input_rows,
        annotator_id=annotator_id,
        expected_items=expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
    checkpoints = build_event_head_ledger(
        events,
        input_rows,
        annotator_id=annotator_id,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    with event_log_lock(event_log):
        if (
            event_log.exists()
            or event_log.is_symlink()
            or head_ledger.exists()
            or head_ledger.is_symlink()
        ):
            raise FileExistsError(
                "MCIF reliability-v2 event log/ledger must not already exist"
            )
        write_jsonl_exclusive(event_log, events)
        try:
            write_jsonl_exclusive(head_ledger, checkpoints)
        except Exception:
            event_log.unlink(missing_ok=True)
            raise
    return events


def append_event_log(
    event_log: Path,
    head_ledger: Path,
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
    run_contract_sha256: str,
) -> list[dict[str, Any]]:
    if event_log.resolve(strict=False) == head_ledger.resolve(strict=False):
        raise ValueError("MCIF reliability-v2 event log and ledger must be distinct")
    with event_log_lock(event_log):
        if (
            event_log.is_symlink()
            or not event_log.is_file()
            or head_ledger.is_symlink()
            or not head_ledger.is_file()
        ):
            raise ValueError(
                "MCIF reliability-v2 event log/ledger must be regular files"
            )
        current = load_jsonl(event_log)
        checkpoints = load_jsonl(head_ledger)
        latest_checkpoint = validate_event_head_ledger(
            checkpoints,
            current,
            input_rows,
            annotator_id=annotator_id,
            expected_items=expected_items,
            key=key,
            config=config,
            run_contract_sha256=run_contract_sha256,
        )
        updated = append_annotation_event(
            current,
            input_rows,
            item_id=item_id,
            expected_event_index=expected_event_index,
            annotation_status=annotation_status,
            annotation=annotation,
            submitted_at_utc=submitted_at_utc,
            annotator_id=annotator_id,
            expected_items=expected_items,
            key=key,
            config=config,
            run_contract_sha256=run_contract_sha256,
        )
        checkpoint = make_event_head_checkpoint(
            updated,
            input_rows,
            annotator_id=annotator_id,
            run_contract_sha256=run_contract_sha256,
            checkpoint_index=len(checkpoints),
            previous_checkpoint_hmac=latest_checkpoint["checkpoint_hmac_sha256"],
            key=key,
        )
        write_jsonl_atomic(event_log, updated)
        append_jsonl_row(head_ledger, checkpoint)
        return updated


def freeze_annotations(
    output_root: Path,
    *,
    input_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    head_checkpoints: list[dict[str, Any]],
    annotator_id: str,
    expected_items: int,
    locked_at_utc: str,
    key: bytes,
    config: dict[str, Any],
    identity_registry: dict[str, Any],
    run_contract: dict[str, Any],
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(
            "MCIF reliability-v2 freeze output must not already exist"
        )
    if UTC_PATTERN.fullmatch(locked_at_utc) is None:
        raise ValueError("MCIF reliability-v2 freeze timestamp differs")
    input_roles = {input_contract(row)[0] for row in input_rows}
    if len(input_roles) != 1:
        raise ValueError("MCIF reliability-v2 freeze mixes roles")
    role_for_registry = next(iter(input_roles))
    expected_annotator_id = registered_annotator_id(
        identity_registry, config, role=role_for_registry
    )
    if annotator_id != expected_annotator_id:
        raise ValueError("MCIF reliability-v2 annotator differs from identity registry")
    identity_registry_sha256 = identity_registry["registry_sha256"]
    config_sha256 = canonical_sha256(config)
    run_contract_sha256 = validate_run_contract(
        run_contract,
        key=key,
        config=config,
        identity_registry=identity_registry,
    )
    validate_contract_stage_item_count(
        run_contract, expected_items=expected_items, role=role_for_registry
    )
    grouped = validate_event_log(
        events,
        input_rows,
        annotator_id=annotator_id,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    latest_checkpoint = validate_event_head_ledger(
        head_checkpoints,
        events,
        input_rows,
        annotator_id=annotator_id,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    source_by_id = {row["item_id"]: row for row in input_rows}
    event_log_head_sha256 = canonical_sha256(events)
    frozen_rows = []
    for source in input_rows:
        role, stage = input_contract(source)
        final_event = grouped[source["item_id"]][-1]
        if final_event["annotation_status"] != "completed":
            raise ValueError(
                f"MCIF reliability-v2 annotation remains incomplete: {source['item_id']}"
            )
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
            "event_log_head_sha256": event_log_head_sha256,
            "event_head_checkpoint_hmac_sha256": latest_checkpoint[
                "checkpoint_hmac_sha256"
            ],
            **annotation,
            "locked_at_utc": locked_at_utc,
            "config_sha256": config_sha256,
            "identity_registry_sha256": identity_registry_sha256,
            "run_contract_sha256": run_contract_sha256,
        }
        frozen_rows.append(signed_payload(payload, key, "freeze_hmac_sha256"))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
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
            "event_head_checkpoint_hmac_sha256": latest_checkpoint[
                "checkpoint_hmac_sha256"
            ],
            "frozen_rows_sha256": file_sha256(frozen_path),
            "config_sha256": config_sha256,
            "identity_registry_sha256": identity_registry_sha256,
            "run_contract_sha256": run_contract_sha256,
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
    run_contract_sha256: str,
) -> dict[str, dict[str, Any]]:
    source_by_id = validate_input_rows(
        input_rows,
        expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
    if len(rows) != expected_items:
        raise ValueError("MCIF reliability-v2 frozen item count differs")
    event_log_heads = {row.get("event_log_head_sha256") for row in rows}
    event_head_checkpoints = {
        row.get("event_head_checkpoint_hmac_sha256") for row in rows
    }
    identity_registry_hashes = {row.get("identity_registry_sha256") for row in rows}
    event_log_head_sha256 = next(iter(event_log_heads), None)
    if (
        len(event_log_heads) != 1
        or not isinstance(event_log_head_sha256, str)
        or HEX_64.fullmatch(event_log_head_sha256) is None
    ):
        raise ValueError("MCIF reliability-v2 frozen rows mix event-log heads")
    event_head_checkpoint = next(iter(event_head_checkpoints), None)
    if (
        len(event_head_checkpoints) != 1
        or not isinstance(event_head_checkpoint, str)
        or HEX_64.fullmatch(event_head_checkpoint) is None
    ):
        raise ValueError("MCIF reliability-v2 frozen rows mix event-head checkpoints")
    identity_registry_sha256 = next(iter(identity_registry_hashes), None)
    if (
        len(identity_registry_hashes) != 1
        or not isinstance(identity_registry_sha256, str)
        or HEX_64.fullmatch(identity_registry_sha256) is None
    ):
        raise ValueError("MCIF reliability-v2 frozen rows mix identity registries")
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
            "event_log_head_sha256",
            "event_head_checkpoint_hmac_sha256",
            "identity_registry_sha256",
            "run_contract_sha256",
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
            or row["config_sha256"] != canonical_sha256(config)
            or row["event_log_head_sha256"] != event_log_head_sha256
            or row["event_head_checkpoint_hmac_sha256"] != event_head_checkpoint
            or row["identity_registry_sha256"] != identity_registry_sha256
            or row["run_contract_sha256"] != run_contract_sha256
            or not signature_valid(row, key, "freeze_hmac_sha256")
        ):
            raise ValueError(
                f"MCIF reliability-v2 frozen binding/signature differs: {item_id}"
            )
        annotation = {name: row[name] for name in blank_annotation(role, stage)}
        validate_annotation(
            annotation, role=role, stage=stage, status="completed", config=config
        )
        output[item_id] = row
    return output


def cohort_lock_sha256(
    frozen_a: dict[str, dict[str, Any]], frozen_b: dict[str, dict[str, Any]]
) -> str:
    return canonical_sha256(
        {
            "visual_a": [
                frozen_a[key]["freeze_hmac_sha256"] for key in sorted(frozen_a)
            ],
            "visual_b": [
                frozen_b[key]["freeze_hmac_sha256"] for key in sorted(frozen_b)
            ],
        }
    )


def validate_private_rows(
    rows: list[dict[str, Any]], *, schema: str, expected_items: int, label: str
) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_items:
        raise ValueError(f"MCIF reliability-v2 private {label} item count differs")
    output = {}
    expected_by_schema = {
        PRIVATE_VISUAL_SCHEMA: {
            "schema_version",
            "candidate_id",
            "visual_a_item_id",
            "visual_b_item_id",
            "candidate_source_en",
            "candidate_kind",
            "candidate_token_count",
            "evidence_tier",
            "r0_text",
            "r1_blocks",
            "private_media",
            "proposed_evidence_origins",
            "v1_visual_item_id",
            "v1_visual_row_sha256",
            "row_sha256",
        },
        PRIVATE_TARGET_SCHEMA: {
            "schema_version",
            "candidate_id",
            "target_author_item_id",
            "target_validator_item_id",
            "v1_target_item_id",
            "v1_target_row_sha256",
            "row_sha256",
        },
        MAPPING_SCHEMA: {
            "schema_version",
            "candidate_id",
            "evidence_tier",
            "talk_id",
            "segment_id",
            "current_state_id",
            "lead_lower_bound_sec",
            "visual_a_item_id",
            "visual_b_item_id",
            "target_author_item_id",
            "target_validator_item_id",
            "v1_mapping_row_sha256",
            "human_labels_complete",
            "audio_release_allowed",
            "inference_release_allowed",
            "row_sha256",
        },
    }
    for row in rows:
        exact_keys(row, expected_by_schema[schema], f"private {label}")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in output:
            raise ValueError(f"MCIF reliability-v2 private {label} id differs")
        if row.get("schema_version") != schema or not row_hash_valid(row):
            raise ValueError(f"MCIF reliability-v2 private {label} schema/hash differs")
        if schema == PRIVATE_VISUAL_SCHEMA:
            validate_r1_blocks(row["r1_blocks"])
            validate_evidence_origins(row["proposed_evidence_origins"])
            validate_media_descriptor(row["private_media"], private=True)
        output[candidate_id] = row
    return output


def resolve_private_media(workspace_root: Path, descriptor: dict[str, Any]) -> Path:
    validate_media_descriptor(descriptor, private=True)
    if workspace_root.is_symlink() or not workspace_root.is_dir():
        raise ValueError("MCIF reliability-v2 workspace root must be a real directory")
    private_root = (workspace_root.resolve(strict=True) / "scorer_private").resolve(
        strict=True
    )
    relative = PurePosixPath(descriptor["private_path"])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != descriptor["private_path"]
        or not relative.parts
        or relative.parts[0] != "media"
    ):
        raise ValueError("MCIF reliability-v2 private media path differs")
    current = private_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(
                "MCIF reliability-v2 private media cannot traverse a symlink"
            )
    path = current.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(private_root):
        raise ValueError("MCIF reliability-v2 private media escapes workspace")
    if file_sha256(path) != descriptor["sha256"]:
        raise ValueError("MCIF reliability-v2 private media hash differs")
    return path


def validate_private_mapping(
    rows: list[dict[str, Any]], *, expected_items: int
) -> dict[str, dict[str, Any]]:
    output = validate_private_rows(
        rows, schema=MAPPING_SCHEMA, expected_items=expected_items, label="mapping"
    )
    for row in output.values():
        if row.get("human_labels_complete") is not False:
            raise ValueError(
                "MCIF reliability-v2 private mapping contains human labels"
            )
        if (
            row.get("audio_release_allowed") is not False
            or row.get("inference_release_allowed") is not False
        ):
            raise ValueError(
                "MCIF reliability-v2 private mapping release firewall differs"
            )
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
    key: bytes,
) -> dict[str, Any]:
    prior_stage = prior_input["stage"]
    locked_judgments = dict(prior_input.get("locked_judgments", {}))
    locked_judgments[VISUAL_FIELDS[prior_stage]] = prior_frozen[
        VISUAL_FIELDS[prior_stage]
    ]
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
        "run_contract_sha256": prior_input["run_contract_sha256"],
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
    return sign_release_row(row, key)


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
    run_contract: dict[str, Any],
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(
            "MCIF reliability-v2 visual release must not already exist"
        )
    if next_stage not in VISUAL_STAGES[1:]:
        raise ValueError("MCIF reliability-v2 next visual stage differs")
    run_contract_sha256 = validate_run_contract(run_contract, key=key, config=config)
    if run_contract["expected_items"] != expected_items:
        raise ValueError("MCIF reliability-v2 run contract item count differs")
    validate_private_bundle_binding(
        private_visual_rows,
        run_contract,
        contract_field="private_visual_rows_sha256",
        label="visual material",
    )
    validate_private_bundle_binding(
        mapping_rows,
        run_contract,
        contract_field="private_mapping_rows_sha256",
        label="mapping",
    )
    prior_stage = VISUAL_STAGES[VISUAL_STAGES.index(next_stage) - 1]
    input_a = validate_input_rows(
        prior_input_a,
        expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
    input_b = validate_input_rows(
        prior_input_b,
        expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
    if {input_contract(row) for row in input_a.values()} != {("visual_a", prior_stage)}:
        raise ValueError("MCIF reliability-v2 visual A prior stage differs")
    if {input_contract(row) for row in input_b.values()} != {("visual_b", prior_stage)}:
        raise ValueError("MCIF reliability-v2 visual B prior stage differs")
    frozen_a = validate_frozen_rows(
        prior_frozen_a,
        prior_input_a,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    frozen_b = validate_frozen_rows(
        prior_frozen_b,
        prior_input_b,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    id_a = one_annotator_id(frozen_a, "visual A prior stage")
    id_b = one_annotator_id(frozen_b, "visual B prior stage")
    registry_a = one_identity_registry_sha256(frozen_a, "visual A prior stage")
    registry_b = one_identity_registry_sha256(frozen_b, "visual B prior stage")
    if registry_a != registry_b:
        raise ValueError(
            "MCIF reliability-v2 visual cohorts use different identity registries"
        )
    if registry_a != run_contract["identity_registry_sha256"]:
        raise ValueError(
            "MCIF reliability-v2 visual release registry differs from contract"
        )
    require_disjoint_identities({"visual_a": id_a, "visual_b": id_b})
    materials = validate_private_rows(
        private_visual_rows,
        schema=PRIVATE_VISUAL_SCHEMA,
        expected_items=expected_items,
        label="visual material",
    )
    mapping = validate_private_mapping(mapping_rows, expected_items=expected_items)
    if set(materials) != set(mapping):
        raise ValueError(
            "MCIF reliability-v2 visual material/mapping candidate sets differ"
        )
    indices = {
        "visual_a": visual_item_candidate_index(mapping, "visual_a"),
        "visual_b": visual_item_candidate_index(mapping, "visual_b"),
    }
    if set(input_a) != set(indices["visual_a"]) or set(input_b) != set(
        indices["visual_b"]
    ):
        raise ValueError(
            "MCIF reliability-v2 visual release would change the full cohort"
        )
    cohort_lock = cohort_lock_sha256(frozen_a, frozen_b)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
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
                    source_media = resolve_private_media(workspace_root, private_media)
                    media_name = (
                        hashlib.sha256(
                            f"{role}\0{next_stage}\0{candidate_id}".encode()
                        ).hexdigest()[:20]
                        + ".png"
                    )
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
                        key=key,
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
            "identity_registry_sha256": registry_a,
            "run_contract_sha256": run_contract_sha256,
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
    run_contract: dict[str, Any],
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(
            "MCIF reliability-v2 target stage2 release must not already exist"
        )
    run_contract_sha256 = validate_run_contract(run_contract, key=key, config=config)
    if run_contract["expected_items"] != expected_items:
        raise ValueError("MCIF reliability-v2 run contract item count differs")
    validate_private_bundle_binding(
        private_target_rows,
        run_contract,
        contract_field="private_target_rows_sha256",
        label="target material",
    )
    validate_private_bundle_binding(
        mapping_rows,
        run_contract,
        contract_field="private_mapping_rows_sha256",
        label="mapping",
    )
    authors = validate_input_rows(
        author_input_rows,
        expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
    validators = validate_input_rows(
        validator_input_rows,
        expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
    if {input_contract(row) for row in authors.values()} != {
        ("target_author", "author")
    }:
        raise ValueError("MCIF reliability-v2 target author input contract differs")
    if {input_contract(row) for row in validators.values()} != {
        ("target_validator", "independent_alignment")
    }:
        raise ValueError(
            "MCIF reliability-v2 target validator stage1 input contract differs"
        )
    frozen_authors = validate_frozen_rows(
        author_frozen_rows,
        author_input_rows,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    frozen_validators = validate_frozen_rows(
        validator_frozen_rows,
        validator_input_rows,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    author_id = one_annotator_id(frozen_authors, "target author")
    validator_id = one_annotator_id(
        frozen_validators, "target validator independent stage"
    )
    author_registry = one_identity_registry_sha256(frozen_authors, "target author")
    validator_registry = one_identity_registry_sha256(
        frozen_validators, "target validator independent stage"
    )
    if author_registry != validator_registry:
        raise ValueError(
            "MCIF reliability-v2 target roles use different identity registries"
        )
    if author_registry != run_contract["identity_registry_sha256"]:
        raise ValueError(
            "MCIF reliability-v2 target release registry differs from contract"
        )
    require_disjoint_identities(
        {"target_author": author_id, "target_validator": validator_id}
    )
    private = validate_private_rows(
        private_target_rows,
        schema=PRIVATE_TARGET_SCHEMA,
        expected_items=expected_items,
        label="target material",
    )
    mapping = validate_private_mapping(mapping_rows, expected_items=expected_items)
    if set(private) != set(mapping):
        raise ValueError(
            "MCIF reliability-v2 target material/mapping candidate sets differ"
        )
    author_to_candidate = {
        row["target_author_item_id"]: candidate_id
        for candidate_id, row in mapping.items()
    }
    validator_to_candidate = {
        row["target_validator_item_id"]: candidate_id
        for candidate_id, row in mapping.items()
    }
    if set(authors) != set(author_to_candidate) or set(validators) != set(
        validator_to_candidate
    ):
        raise ValueError(
            "MCIF reliability-v2 target release would change the full cohort"
        )
    author_by_candidate = {
        author_to_candidate[item_id]: row for item_id, row in authors.items()
    }
    frozen_author_by_candidate = {
        author_to_candidate[item_id]: row for item_id, row in frozen_authors.items()
    }
    validator_by_candidate = {
        validator_to_candidate[item_id]: row for item_id, row in validators.items()
    }
    frozen_validator_by_candidate = {
        validator_to_candidate[item_id]: row
        for item_id, row in frozen_validators.items()
    }
    rows = []
    for candidate_id in sorted(
        mapping, key=lambda value: mapping[value]["target_validator_item_id"]
    ):
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
            "locked_target_reference_alignment": validator[
                "target_reference_alignment"
            ],
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
            "run_contract_sha256": run_contract_sha256,
        }
        row["row_sha256"] = canonical_sha256(row)
        rows.append(sign_release_row(row, key))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        write_jsonl(temporary / "items.jsonl", rows)
        report = {
            "schema_version": "mcif_beyond_ocr_target_stage2_release_report_v2",
            "status": "AUTHOR_TEXT_RELEASED_AFTER_INDEPENDENT_STAGE1_FREEZE",
            "items": len(rows),
            "author_id": author_id,
            "target_validator_id": validator_id,
            "identity_registry_sha256": author_registry,
            "run_contract_sha256": run_contract_sha256,
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


def one_identity_registry_sha256(rows: dict[str, dict[str, Any]], label: str) -> str:
    values = {row["identity_registry_sha256"] for row in rows.values()}
    value = next(iter(values), None)
    if (
        len(values) != 1
        or not isinstance(value, str)
        or HEX_64.fullmatch(value) is None
    ):
        raise ValueError(f"MCIF reliability-v2 {label} must bind one identity registry")
    return value


def validate_visual_chains(
    *,
    visual_inputs: dict[str, dict[str, list[dict[str, Any]]]],
    visual_frozen: dict[str, dict[str, list[dict[str, Any]]]],
    mapping: dict[str, dict[str, Any]],
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
    run_contract_sha256: str,
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
                raise ValueError(
                    f"MCIF reliability-v2 missing visual input: {role}/{stage}"
                )
            if stage not in visual_frozen or role not in visual_frozen[stage]:
                raise ValueError(
                    f"MCIF reliability-v2 missing visual freeze: {role}/{stage}"
                )
            input_rows = visual_inputs[stage][role]
            source_by_id = validate_input_rows(
                input_rows,
                expected_items,
                key=key,
                run_contract_sha256=run_contract_sha256,
            )
            if {input_contract(row) for row in source_by_id.values()} != {
                (role, stage)
            }:
                raise ValueError(
                    f"MCIF reliability-v2 visual role/stage differs: {role}/{stage}"
                )
            if set(source_by_id) != set(item_to_candidate):
                raise ValueError(
                    "MCIF reliability-v2 visual chain changed the full cohort"
                )
            frozen_by_id = validate_frozen_rows(
                visual_frozen[stage][role],
                input_rows,
                expected_items=expected_items,
                key=key,
                config=config,
                run_contract_sha256=run_contract_sha256,
            )
            identity = one_annotator_id(frozen_by_id, f"{role}/{stage}")
            if role in identities and normalize_identity(
                identities[role]
            ) != normalize_identity(identity):
                raise ValueError(
                    f"MCIF reliability-v2 {role} identity changed across stages"
                )
            identities[role] = identity
            if stage != "r0":
                if (
                    role not in prior_inputs_by_role
                    or role not in prior_frozen_by_role
                    or prior_cohort_lock is None
                ):
                    raise AssertionError("visual predecessor state missing")
                for item_id, row in source_by_id.items():
                    if (
                        row["prior_stage"]
                        != VISUAL_STAGES[VISUAL_STAGES.index(stage) - 1]
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
    point = reliability_report(
        [(label_a, label_b) for _, label_a, label_b in pairs], categories
    )
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
        and target["target_reference_alignment"]["author"]
        in {"explicit", "paraphrased"}
        and target["stage2_review_decision"] == "accept"
    )
    return "eligible_raw" if visual_pass and target_pass else "rejected_raw"


def freeze_manifest_sha256(
    *,
    visual_frozen: dict[str, dict[str, list[dict[str, Any]]]],
    target_author_frozen: list[dict[str, Any]],
    target_validator_stage1_frozen: list[dict[str, Any]],
    target_validator_stage2_frozen: list[dict[str, Any]],
) -> str:
    manifest = {
        "visual": {
            stage: {
                role: sorted(
                    row["freeze_hmac_sha256"] for row in visual_frozen[stage][role]
                )
                for role in ("visual_a", "visual_b")
            }
            for stage in VISUAL_STAGES
        },
        "target_author": sorted(
            row["freeze_hmac_sha256"] for row in target_author_frozen
        ),
        "target_validator_stage1": sorted(
            row["freeze_hmac_sha256"] for row in target_validator_stage1_frozen
        ),
        "target_validator_stage2": sorted(
            row["freeze_hmac_sha256"] for row in target_validator_stage2_frozen
        ),
    }
    return canonical_sha256(manifest)


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
    run_contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_contract_sha256 = validate_run_contract(run_contract, key=key, config=config)
    if run_contract["expected_items"] != expected_items:
        raise ValueError("MCIF reliability-v2 run contract item count differs")
    validate_private_bundle_binding(
        mapping_rows,
        run_contract,
        contract_field="private_mapping_rows_sha256",
        label="mapping",
    )
    mapping = validate_private_mapping(mapping_rows, expected_items=expected_items)
    visual, identities = validate_visual_chains(
        visual_inputs=visual_inputs,
        visual_frozen=visual_frozen,
        mapping=mapping,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    author_sources = validate_input_rows(
        target_author_inputs,
        expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
    validator1_sources = validate_input_rows(
        target_validator_stage1_inputs,
        expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
    validator2_sources = validate_input_rows(
        target_validator_stage2_inputs,
        expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
    authors = validate_frozen_rows(
        target_author_frozen,
        target_author_inputs,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    validators1 = validate_frozen_rows(
        target_validator_stage1_frozen,
        target_validator_stage1_inputs,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    validators2 = validate_frozen_rows(
        target_validator_stage2_frozen,
        target_validator_stage2_inputs,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract_sha256=run_contract_sha256,
    )
    identities["target_author"] = one_annotator_id(authors, "target author")
    identities["target_validator"] = one_annotator_id(
        validators1, "target validator stage1"
    )
    validator2_id = one_annotator_id(validators2, "target validator stage2")
    if normalize_identity(validator2_id) != normalize_identity(
        identities["target_validator"]
    ):
        raise ValueError(
            "MCIF reliability-v2 target validator identity changed across stages"
        )
    require_disjoint_identities(identities)
    registry_hashes = {
        row["identity_registry_sha256"]
        for stage in VISUAL_STAGES
        for role in ("visual_a", "visual_b")
        for row in visual_frozen[stage][role]
    }
    registry_hashes.update(
        row["identity_registry_sha256"]
        for rows_for_role in (
            target_author_frozen,
            target_validator_stage1_frozen,
            target_validator_stage2_frozen,
        )
        for row in rows_for_role
    )
    if len(registry_hashes) != 1:
        raise ValueError("MCIF reliability-v2 report mixes identity registries")
    identity_registry_sha256 = next(iter(registry_hashes))
    if identity_registry_sha256 != run_contract["identity_registry_sha256"]:
        raise ValueError("MCIF reliability-v2 report registry differs from contract")
    author_to_candidate = {
        row["target_author_item_id"]: candidate_id
        for candidate_id, row in mapping.items()
    }
    validator_to_candidate = {
        row["target_validator_item_id"]: candidate_id
        for candidate_id, row in mapping.items()
    }
    if (
        set(author_sources) != set(author_to_candidate)
        or set(validator1_sources) != set(validator_to_candidate)
        or set(validator2_sources) != set(validator_to_candidate)
    ):
        raise ValueError("MCIF reliability-v2 target report changed the full cohort")
    author_by_candidate = {
        author_to_candidate[item_id]: row for item_id, row in authors.items()
    }
    validator1_by_candidate = {
        validator_to_candidate[item_id]: row for item_id, row in validators1.items()
    }
    validator2_by_candidate = {
        validator_to_candidate[item_id]: row for item_id, row in validators2.items()
    }
    for candidate_id in mapping:
        stage2_source = validator2_sources[
            mapping[candidate_id]["target_validator_item_id"]
        ]
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
        **{
            field: list(config["visual"]["judgments"])
            for field in VISUAL_FIELDS.values()
        },
        "candidate_eligibility": list(config["target"]["eligibility_judgments"]),
        "target_reference_alignment": list(config["target"]["reference_alignments"]),
    }
    metrics = {
        field: metric_with_cluster_interval(
            metric_pairs[field], categories[field], config=config, seed_offset=index
        )
        for index, field in enumerate(
            [
                *VISUAL_FIELDS.values(),
                "candidate_eligibility",
                "target_reference_alignment",
            ]
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
    summary_payload = {
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
        "requires_adjudication_count": sum(
            row["requires_adjudication"] for row in rows
        ),
        "adjudication_rate": adjudication_rate,
        "maximum_adjudication_rate": config["reliability"]["maximum_adjudication_rate"],
        "instrument_gate_passed": gate_passed,
        "failure_action": config["reliability"]["failure_action"],
        "raw_eligible_count": sum(
            row["raw_candidate_status"] == "eligible_raw" for row in rows
        ),
        "raw_rejected_count": sum(
            row["raw_candidate_status"] == "rejected_raw" for row in rows
        ),
        "raw_missing_count": sum(
            row["raw_candidate_status"] == "missing_pending_adjudication"
            for row in rows
        ),
        "adjudication_applied": False,
        "audio_release_allowed": False,
        "inference_release_allowed": False,
        "pre_adjudication_rows_sha256": canonical_sha256(rows),
        "config_sha256": canonical_sha256(config),
        "mapping_rows_sha256": canonical_sha256(mapping_rows),
        "freeze_manifest_sha256": freeze_manifest_sha256(
            visual_frozen=visual_frozen,
            target_author_frozen=target_author_frozen,
            target_validator_stage1_frozen=target_validator_stage1_frozen,
            target_validator_stage2_frozen=target_validator_stage2_frozen,
        ),
        "identity_registry_sha256": identity_registry_sha256,
        "run_contract_sha256": run_contract_sha256,
    }
    summary = signed_payload(summary_payload, key, "report_hmac_sha256")
    return rows, summary


PRE_ADJUDICATION_ROW_KEYS = {
    "schema_version",
    "candidate_id",
    "evidence_tier",
    "talk_id",
    "segment_id",
    "current_state_id",
    "visual_raw",
    "target_raw",
    "target_author_scoring_text",
    "target_validator_edits",
    "adjudication_reasons",
    "requires_adjudication",
    "raw_candidate_status",
    "adjudication_applied",
    "final_candidate_status",
    "row_sha256",
}
PRE_ADJUDICATION_SUMMARY_KEYS = {
    "schema_version",
    "status",
    "items",
    "talks",
    "role_ids",
    "metrics",
    "load_bearing_field_gate",
    "pre_adjudication_composite_exact_agreement",
    "requires_adjudication_count",
    "adjudication_rate",
    "maximum_adjudication_rate",
    "instrument_gate_passed",
    "failure_action",
    "raw_eligible_count",
    "raw_rejected_count",
    "raw_missing_count",
    "adjudication_applied",
    "audio_release_allowed",
    "inference_release_allowed",
    "pre_adjudication_rows_sha256",
    "config_sha256",
    "mapping_rows_sha256",
    "freeze_manifest_sha256",
    "identity_registry_sha256",
    "run_contract_sha256",
    "report_hmac_sha256",
}


def validate_pre_adjudication_rows(
    rows: list[dict[str, Any]], *, expected_items: int
) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_items:
        raise ValueError("MCIF reliability-v2 pre-adjudication row count differs")
    output = {}
    raw_visual_keys = {
        "visual_a",
        "visual_b",
        "visual_a_reason_codes",
        "visual_b_reason_codes",
        "visual_a_note",
        "visual_b_note",
    }
    target_raw_keys = {
        "candidate_eligibility",
        "target_reference_alignment",
        "stage2_review_decision",
        "author_reason_codes",
        "validator_stage1_reason_codes",
        "validator_stage2_reason_codes",
    }
    scoring_keys = {
        "canonical_source_event_en",
        "acceptable_target_realizations_zh",
        "forbidden_target_realizations_zh",
    }
    for row in rows:
        exact_keys(row, PRE_ADJUDICATION_ROW_KEYS, "pre-adjudication row")
        candidate_id = row["candidate_id"]
        if (
            row["schema_version"] != "mcif_beyond_ocr_pre_adjudication_candidate_v2"
            or not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in output
            or not row_hash_valid(row)
            or row["adjudication_applied"] is not False
            or row["final_candidate_status"] is not None
        ):
            raise ValueError("MCIF reliability-v2 pre-adjudication row hash/id differs")
        if not isinstance(row["visual_raw"], dict) or set(row["visual_raw"]) != set(
            VISUAL_FIELDS.values()
        ):
            raise ValueError(
                "MCIF reliability-v2 pre-adjudication visual fields differ"
            )
        for raw in row["visual_raw"].values():
            if not isinstance(raw, dict) or set(raw) != raw_visual_keys:
                raise ValueError(
                    "MCIF reliability-v2 pre-adjudication visual row differs"
                )
        target_raw = row["target_raw"]
        if not isinstance(target_raw, dict) or set(target_raw) != target_raw_keys:
            raise ValueError("MCIF reliability-v2 pre-adjudication target row differs")
        for field in ("candidate_eligibility", "target_reference_alignment"):
            if not isinstance(target_raw[field], dict) or set(target_raw[field]) != {
                "author",
                "validator",
            }:
                raise ValueError(
                    "MCIF reliability-v2 pre-adjudication target pair differs"
                )
        if (
            set(row["target_author_scoring_text"]) != scoring_keys
            or set(row["target_validator_edits"]) != scoring_keys
        ):
            raise ValueError(
                "MCIF reliability-v2 pre-adjudication scoring text differs"
            )
        output[candidate_id] = row
    return output


def validate_pre_adjudication_bundle(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    expected_items: int,
    key: bytes,
    config: dict[str, Any],
    run_contract: dict[str, Any],
    mapping_rows: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    run_contract_sha256 = validate_run_contract(run_contract, key=key, config=config)
    report_by_candidate = validate_pre_adjudication_rows(
        rows, expected_items=expected_items
    )
    exact_keys(summary, PRE_ADJUDICATION_SUMMARY_KEYS, "pre-adjudication summary")
    if (
        summary["schema_version"]
        != "mcif_beyond_ocr_pre_adjudication_reliability_report_v2"
        or not signature_valid(summary, key, "report_hmac_sha256")
        or summary["pre_adjudication_rows_sha256"] != canonical_sha256(rows)
        or summary["config_sha256"] != canonical_sha256(config)
        or summary["run_contract_sha256"] != run_contract_sha256
        or summary["identity_registry_sha256"]
        != run_contract["identity_registry_sha256"]
        or summary["items"] != expected_items
        or summary["adjudication_applied"] is not False
        or summary["audio_release_allowed"] is not False
        or summary["inference_release_allowed"] is not False
    ):
        raise ValueError("MCIF reliability-v2 pre-adjudication report binding differs")
    for field in (
        "mapping_rows_sha256",
        "freeze_manifest_sha256",
        "identity_registry_sha256",
    ):
        if (
            not isinstance(summary[field], str)
            or HEX_64.fullmatch(summary[field]) is None
        ):
            raise ValueError(
                f"MCIF reliability-v2 pre-adjudication hash differs: {field}"
            )
    if mapping_rows is not None and summary["mapping_rows_sha256"] != canonical_sha256(
        mapping_rows
    ):
        raise ValueError("MCIF reliability-v2 pre-adjudication mapping binding differs")
    expected_role_ids = {"visual_a", "visual_b", "target_author", "target_validator"}
    if (
        not isinstance(summary["role_ids"], dict)
        or set(summary["role_ids"]) != expected_role_ids
        or any(
            not isinstance(value, str) or not value.strip()
            for value in summary["role_ids"].values()
        )
    ):
        raise ValueError("MCIF reliability-v2 pre-adjudication role ids differ")
    require_disjoint_identities(summary["role_ids"])
    passed = summary["instrument_gate_passed"] is True
    expected_status = (
        "PASS_ADJUDICATION_MAY_BEGIN"
        if passed
        else "FAIL_REVISE_GUIDELINE_AND_RELABEL_ALL"
    )
    if summary["status"] != expected_status:
        raise ValueError("MCIF reliability-v2 pre-adjudication gate status differs")
    return report_by_candidate


def adjudication_item_id(namespace: str, candidate_id: str, field: str = "") -> str:
    digest = hashlib.sha256(
        f"{namespace}\0{candidate_id}\0{field}".encode()
    ).hexdigest()
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
    key: bytes,
    config: dict[str, Any],
    identity_registry: dict[str, Any],
    run_contract: dict[str, Any],
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(
            "MCIF reliability-v2 adjudication release must not already exist"
        )
    mapping = validate_private_mapping(mapping_rows, expected_items=expected_items)
    report_by_candidate = validate_pre_adjudication_bundle(
        pre_adjudication_rows,
        pre_adjudication_summary,
        expected_items=expected_items,
        key=key,
        config=config,
        run_contract=run_contract,
        mapping_rows=mapping_rows,
    )
    run_contract_sha256 = validate_run_contract(
        run_contract,
        key=key,
        config=config,
        identity_registry=identity_registry,
    )
    validate_private_bundle_binding(
        private_visual_rows,
        run_contract,
        contract_field="private_visual_rows_sha256",
        label="visual material",
    )
    validate_private_bundle_binding(
        mapping_rows,
        run_contract,
        contract_field="private_mapping_rows_sha256",
        label="mapping",
    )
    assignments = validate_identity_registry(identity_registry, config)
    if (
        identity_registry["registry_sha256"]
        != pre_adjudication_summary["identity_registry_sha256"]
    ):
        raise ValueError("MCIF reliability-v2 adjudication identity registry differs")
    if any(
        pre_adjudication_summary["role_ids"].get(role) != assignments[role]
        for role in ("visual_a", "visual_b", "target_author", "target_validator")
    ):
        raise ValueError("MCIF reliability-v2 pre-adjudication role registry differs")
    if (
        visual_adjudicator_id != assignments["visual_adjudicator"]
        or target_adjudicator_id != assignments["target_adjudicator"]
    ):
        raise ValueError(
            "MCIF reliability-v2 adjudicator differs from identity registry"
        )
    if (
        pre_adjudication_summary.get("instrument_gate_passed") is not True
        or pre_adjudication_summary.get("status") != "PASS_ADJUDICATION_MAY_BEGIN"
    ):
        raise ValueError(
            "MCIF reliability-v2 failed instrument cannot enter adjudication"
        )
    materials = validate_private_rows(
        private_visual_rows,
        schema=PRIVATE_VISUAL_SCHEMA,
        expected_items=expected_items,
        label="visual material",
    )
    if set(report_by_candidate) != set(mapping) or set(materials) != set(mapping):
        raise ValueError("MCIF reliability-v2 adjudication candidate sets differ")
    target_sources = validate_input_rows(
        target_validator_stage2_inputs,
        expected_items,
        key=key,
        run_contract_sha256=run_contract_sha256,
    )
    target_to_candidate = {
        row["target_validator_item_id"]: candidate_id
        for candidate_id, row in mapping.items()
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
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
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
                if any(
                    row.get("primitive_field") == field
                    and row["pre_adjudication_row_sha256"] == report["row_sha256"]
                    for row in visual_rows
                ):
                    continue
                stage = next(
                    stage for stage, name in VISUAL_FIELDS.items() if name == field
                )
                evidence: dict[str, Any] = {"r0_text": material["r0_text"]}
                if stage in {"r1", "pixels", "descriptor"}:
                    evidence["r1_blocks"] = material["r1_blocks"]
                if stage in {"pixels", "descriptor"}:
                    private_media = material["private_media"]
                    source_media = resolve_private_media(workspace_root, private_media)
                    media_name = (
                        hashlib.sha256(
                            f"adjudication\0{candidate_id}\0{field}".encode()
                        ).hexdigest()[:20]
                        + ".png"
                    )
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
                    "run_contract_sha256": run_contract_sha256,
                }
                row["row_sha256"] = canonical_sha256(row)
                visual_rows.append(sign_release_row(row, key))
            if any(
                reason.startswith("target:")
                for reason in report["adjudication_reasons"]
            ):
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
                    "run_contract_sha256": run_contract_sha256,
                }
                row["row_sha256"] = canonical_sha256(row)
                target_rows.append(sign_release_row(row, key))
        visual_rows.sort(key=lambda row: row["item_id"])
        target_rows.sort(key=lambda row: row["item_id"])
        write_jsonl(visual_root / "items.jsonl", visual_rows)
        write_jsonl(target_root / "items.jsonl", target_rows)
        report_payload = {
            "schema_version": "mcif_beyond_ocr_adjudication_release_report_v2",
            "status": "ROLE_SPECIFIC_ADJUDICATION_RELEASED_AFTER_INSTRUMENT_PASS",
            "visual_items": len(visual_rows),
            "target_items": len(target_rows),
            "visual_adjudicator_id": visual_adjudicator_id,
            "target_adjudicator_id": target_adjudicator_id,
            "pre_adjudication_rows_sha256": canonical_sha256(pre_adjudication_rows),
            "pre_adjudication_report_hmac_sha256": pre_adjudication_summary[
                "report_hmac_sha256"
            ],
            "config_sha256": canonical_sha256(config),
            "mapping_rows_sha256": canonical_sha256(mapping_rows),
            "identity_registry_sha256": identity_registry["registry_sha256"],
            "run_contract_sha256": run_contract_sha256,
            "raw_metrics_recomputed": False,
            "audio_release_allowed": False,
            "inference_release_allowed": False,
        }
        report = signed_payload(report_payload, key, "release_report_hmac_sha256")
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_root)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


ADJUDICATION_RELEASE_REPORT_KEYS = {
    "schema_version",
    "status",
    "visual_items",
    "target_items",
    "visual_adjudicator_id",
    "target_adjudicator_id",
    "pre_adjudication_rows_sha256",
    "pre_adjudication_report_hmac_sha256",
    "config_sha256",
    "mapping_rows_sha256",
    "identity_registry_sha256",
    "run_contract_sha256",
    "raw_metrics_recomputed",
    "audio_release_allowed",
    "inference_release_allowed",
    "release_report_hmac_sha256",
}


def apply_adjudications(
    *,
    pre_adjudication_rows: list[dict[str, Any]],
    pre_adjudication_summary: dict[str, Any],
    adjudication_release_report: dict[str, Any],
    visual_adjudication_inputs: list[dict[str, Any]],
    visual_adjudication_frozen: list[dict[str, Any]],
    target_adjudication_inputs: list[dict[str, Any]],
    target_adjudication_frozen: list[dict[str, Any]],
    key: bytes,
    config: dict[str, Any],
    identity_registry: dict[str, Any],
    run_contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_contract_sha256 = validate_run_contract(
        run_contract,
        key=key,
        config=config,
        identity_registry=identity_registry,
    )
    report_by_candidate = validate_pre_adjudication_bundle(
        pre_adjudication_rows,
        pre_adjudication_summary,
        expected_items=len(pre_adjudication_rows),
        key=key,
        config=config,
        run_contract=run_contract,
    )
    exact_keys(
        adjudication_release_report,
        ADJUDICATION_RELEASE_REPORT_KEYS,
        "adjudication release report",
    )
    assignments = validate_identity_registry(identity_registry, config)
    if (
        adjudication_release_report["schema_version"]
        != "mcif_beyond_ocr_adjudication_release_report_v2"
        or adjudication_release_report["status"]
        != "ROLE_SPECIFIC_ADJUDICATION_RELEASED_AFTER_INSTRUMENT_PASS"
        or not signature_valid(
            adjudication_release_report, key, "release_report_hmac_sha256"
        )
        or adjudication_release_report["pre_adjudication_rows_sha256"]
        != canonical_sha256(pre_adjudication_rows)
        or adjudication_release_report["pre_adjudication_report_hmac_sha256"]
        != pre_adjudication_summary["report_hmac_sha256"]
        or adjudication_release_report["config_sha256"] != canonical_sha256(config)
        or adjudication_release_report["identity_registry_sha256"]
        != identity_registry["registry_sha256"]
        or adjudication_release_report["run_contract_sha256"] != run_contract_sha256
        or adjudication_release_report["mapping_rows_sha256"]
        != pre_adjudication_summary["mapping_rows_sha256"]
        or adjudication_release_report["visual_adjudicator_id"]
        != assignments["visual_adjudicator"]
        or adjudication_release_report["target_adjudicator_id"]
        != assignments["target_adjudicator"]
        or adjudication_release_report["raw_metrics_recomputed"] is not False
        or adjudication_release_report["audio_release_allowed"] is not False
        or adjudication_release_report["inference_release_allowed"] is not False
        or any(
            not isinstance(adjudication_release_report[field], int)
            or isinstance(adjudication_release_report[field], bool)
            or adjudication_release_report[field] < 0
            for field in ("visual_items", "target_items")
        )
    ):
        raise ValueError("MCIF reliability-v2 adjudication release report differs")
    if any(
        pre_adjudication_summary["role_ids"].get(role) != assignments[role]
        for role in ("visual_a", "visual_b", "target_author", "target_validator")
    ):
        raise ValueError("MCIF reliability-v2 pre-adjudication role registry differs")
    if pre_adjudication_summary["instrument_gate_passed"] is not True:
        raise ValueError(
            "MCIF reliability-v2 failed instrument cannot apply adjudication"
        )
    report_by_sha = {row["row_sha256"]: row for row in report_by_candidate.values()}
    visual_sources = (
        validate_input_rows(
            visual_adjudication_inputs,
            len(visual_adjudication_inputs),
            key=key,
            run_contract_sha256=run_contract_sha256,
        )
        if visual_adjudication_inputs
        else {}
    )
    target_sources = (
        validate_input_rows(
            target_adjudication_inputs,
            len(target_adjudication_inputs),
            key=key,
            run_contract_sha256=run_contract_sha256,
        )
        if target_adjudication_inputs
        else {}
    )
    visual_frozen = (
        validate_frozen_rows(
            visual_adjudication_frozen,
            visual_adjudication_inputs,
            expected_items=len(visual_adjudication_inputs),
            key=key,
            config=config,
            run_contract_sha256=run_contract_sha256,
        )
        if visual_adjudication_inputs
        else {}
    )
    target_frozen = (
        validate_frozen_rows(
            target_adjudication_frozen,
            target_adjudication_inputs,
            expected_items=len(target_adjudication_inputs),
            key=key,
            config=config,
            run_contract_sha256=run_contract_sha256,
        )
        if target_adjudication_inputs
        else {}
    )
    if bool(visual_sources) != bool(visual_frozen) or bool(target_sources) != bool(
        target_frozen
    ):
        raise ValueError(
            "MCIF reliability-v2 adjudication input/freeze presence differs"
        )
    if adjudication_release_report["visual_items"] != len(
        visual_sources
    ) or adjudication_release_report["target_items"] != len(target_sources):
        raise ValueError("MCIF reliability-v2 adjudication release item count differs")
    expected_registry_sha256 = pre_adjudication_summary["identity_registry_sha256"]
    adjudication_registry_hashes = {
        row["identity_registry_sha256"]
        for rows_for_role in (visual_frozen, target_frozen)
        for row in rows_for_role.values()
    }
    if adjudication_registry_hashes and adjudication_registry_hashes != {
        expected_registry_sha256
    }:
        raise ValueError("MCIF reliability-v2 adjudication identity registry differs")
    role_ids = dict(pre_adjudication_summary["role_ids"])
    if visual_frozen:
        role_ids["visual_adjudicator"] = one_annotator_id(
            visual_frozen, "visual adjudicator"
        )
    if target_frozen:
        role_ids["target_adjudicator"] = one_annotator_id(
            target_frozen, "target adjudicator"
        )
    if any(
        role in role_ids and role_ids[role] != assignments[role]
        for role in ("visual_adjudicator", "target_adjudicator")
    ):
        raise ValueError("MCIF reliability-v2 adjudication role registry differs")
    require_disjoint_identities(role_ids)
    visual_by_key = {}
    for item_id, source in visual_sources.items():
        key_tuple = (source["pre_adjudication_row_sha256"], source["primitive_field"])
        if key_tuple in visual_by_key or key_tuple[0] not in report_by_sha:
            raise ValueError(
                "MCIF reliability-v2 visual adjudication task binding differs"
            )
        visual_by_key[key_tuple] = visual_frozen[item_id]
    target_by_sha = {}
    for item_id, source in target_sources.items():
        row_sha = source["pre_adjudication_row_sha256"]
        if row_sha in target_by_sha or row_sha not in report_by_sha:
            raise ValueError(
                "MCIF reliability-v2 target adjudication task binding differs"
            )
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
        raise ValueError(
            "MCIF reliability-v2 adjudication tasks do not exactly cover triggers"
        )
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
            final_status = (
                "eligible_adjudicated" if visual_pass else "rejected_adjudicated"
            )
        row = {
            **{
                key_name: value
                for key_name, value in raw.items()
                if key_name != "row_sha256"
            },
            "adjudication_applied": True,
            "final_visual_judgments": final_visual,
            "final_target_decision": target_decision,
            "final_scoring_text": final_scoring_text,
            "final_candidate_status": final_status,
            "raw_row_sha256": raw["row_sha256"],
        }
        row["row_sha256"] = canonical_sha256(row)
        output.append(row)
    summary_payload = {
        **{
            name: value
            for name, value in pre_adjudication_summary.items()
            if name != "report_hmac_sha256"
        },
        "schema_version": "mcif_beyond_ocr_post_adjudication_report_v2",
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
    summary = signed_payload(summary_payload, key, "report_hmac_sha256")
    return output, summary


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def command_init_key(args: argparse.Namespace) -> None:
    print(json.dumps({"key_sha256": create_hmac_key(args.output)}))


def command_init_access_token(args: argparse.Namespace) -> None:
    print(json.dumps({"access_token_sha256": create_access_token(args.output)}))


def command_init_events(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.input)
    config = load_config(args.config, args.expected_config_sha256)
    registry = load_identity_registry(
        args.identity_registry, args.expected_identity_registry_sha256, config
    )
    key = load_hmac_key(args.hmac_key)
    contract = load_run_contract(
        args.run_contract,
        args.expected_run_contract_file_sha256,
        key=key,
        config=config,
        identity_registry=registry,
    )
    contract_sha256 = canonical_sha256(contract)
    roles = {input_contract(row)[0] for row in rows}
    if len(roles) != 1 or args.annotator_id != registered_annotator_id(
        registry, config, role=next(iter(roles), "")
    ):
        raise ValueError(
            "MCIF reliability-v2 event annotator differs from identity registry"
        )
    events = initialize_event_log(
        args.output,
        args.head_ledger,
        rows,
        annotator_id=args.annotator_id,
        expected_items=args.expected_items,
        key=key,
        config=config,
        run_contract_sha256=contract_sha256,
    )
    print(json.dumps({"items": len(rows), "events": len(events)}))


def command_append_event(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.input)
    config = load_config(args.config, args.expected_config_sha256)
    registry = load_identity_registry(
        args.identity_registry, args.expected_identity_registry_sha256, config
    )
    roles = {input_contract(row)[0] for row in rows}
    if len(roles) != 1 or args.annotator_id != registered_annotator_id(
        registry, config, role=next(iter(roles), "")
    ):
        raise ValueError(
            "MCIF reliability-v2 event annotator differs from identity registry"
        )
    key = load_hmac_key(args.hmac_key)
    contract = load_run_contract(
        args.run_contract,
        args.expected_run_contract_file_sha256,
        key=key,
        config=config,
        identity_registry=registry,
    )
    updated = append_event_log(
        args.event_log,
        args.head_ledger,
        rows,
        item_id=args.item_id,
        expected_event_index=args.expected_event_index,
        annotation_status=args.annotation_status,
        annotation=load_json(args.annotation_json),
        submitted_at_utc=args.submitted_at_utc,
        annotator_id=args.annotator_id,
        expected_items=args.expected_items,
        key=key,
        config=config,
        run_contract_sha256=canonical_sha256(contract),
    )
    print(json.dumps({"events": len(updated), "item_id": args.item_id}))


def command_freeze(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.expected_config_sha256)
    registry = load_identity_registry(
        args.identity_registry, args.expected_identity_registry_sha256, config
    )
    key = load_hmac_key(args.hmac_key)
    contract = load_run_contract(
        args.run_contract,
        args.expected_run_contract_file_sha256,
        key=key,
        config=config,
        identity_registry=registry,
    )
    with event_log_lock(args.event_log):
        if (
            args.event_log.is_symlink()
            or not args.event_log.is_file()
            or args.head_ledger.is_symlink()
            or not args.head_ledger.is_file()
        ):
            raise ValueError(
                "MCIF reliability-v2 event log/ledger must be regular files"
            )
        report = freeze_annotations(
            args.output_root,
            input_rows=load_jsonl(args.input),
            events=load_jsonl(args.event_log),
            head_checkpoints=load_jsonl(args.head_ledger),
            annotator_id=args.annotator_id,
            expected_items=args.expected_items,
            locked_at_utc=args.locked_at_utc,
            key=key,
            config=config,
            identity_registry=registry,
            run_contract=contract,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def command_release_visual(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.expected_config_sha256)
    key = load_hmac_key(args.hmac_key)
    contract = load_run_contract(
        args.run_contract,
        args.expected_run_contract_file_sha256,
        key=key,
        config=config,
    )
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
        key=key,
        config=config,
        run_contract=contract,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def command_release_target_stage2(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.expected_config_sha256)
    key = load_hmac_key(args.hmac_key)
    contract = load_run_contract(
        args.run_contract,
        args.expected_run_contract_file_sha256,
        key=key,
        config=config,
    )
    report = release_target_validator_stage2(
        args.output_root,
        private_target_rows=load_jsonl(args.private_target),
        mapping_rows=load_jsonl(args.mapping),
        author_input_rows=load_jsonl(args.author_input),
        author_frozen_rows=load_jsonl(args.author_frozen),
        validator_input_rows=load_jsonl(args.validator_stage1_input),
        validator_frozen_rows=load_jsonl(args.validator_stage1_frozen),
        expected_items=args.expected_items,
        key=key,
        config=config,
        run_contract=contract,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def load_visual_run_manifest(
    path: Path,
) -> tuple[
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
    key = load_hmac_key(args.hmac_key)
    contract = load_run_contract(
        args.run_contract,
        args.expected_run_contract_file_sha256,
        key=key,
        config=config,
    )
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
        key=key,
        config=config,
        run_contract=contract,
    )
    write_jsonl(args.output, rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def command_prepare_adjudication(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.expected_config_sha256)
    registry = load_identity_registry(
        args.identity_registry, args.expected_identity_registry_sha256, config
    )
    key = load_hmac_key(args.hmac_key)
    contract = load_run_contract(
        args.run_contract,
        args.expected_run_contract_file_sha256,
        key=key,
        config=config,
        identity_registry=registry,
    )
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
        key=key,
        config=config,
        identity_registry=registry,
        run_contract=contract,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def command_apply_adjudication(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.expected_config_sha256)
    registry = load_identity_registry(
        args.identity_registry, args.expected_identity_registry_sha256, config
    )
    key = load_hmac_key(args.hmac_key)
    contract = load_run_contract(
        args.run_contract,
        args.expected_run_contract_file_sha256,
        key=key,
        config=config,
        identity_registry=registry,
    )
    visual_inputs = load_jsonl(args.visual_adjudication_input)
    target_inputs = load_jsonl(args.target_adjudication_input)
    rows, summary = apply_adjudications(
        pre_adjudication_rows=load_jsonl(args.pre_adjudication_rows),
        pre_adjudication_summary=load_json(args.pre_adjudication_summary),
        adjudication_release_report=load_json(args.adjudication_release_report),
        visual_adjudication_inputs=visual_inputs,
        visual_adjudication_frozen=(
            load_jsonl(args.visual_adjudication_frozen) if visual_inputs else []
        ),
        target_adjudication_inputs=target_inputs,
        target_adjudication_frozen=(
            load_jsonl(args.target_adjudication_frozen) if target_inputs else []
        ),
        key=key,
        config=config,
        identity_registry=registry,
        run_contract=contract,
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


def add_identity_registry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--identity-registry", type=Path, required=True)
    parser.add_argument("--expected-identity-registry-sha256", required=True)


def add_run_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--expected-run-contract-file-sha256", required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_key = subparsers.add_parser("init-key")
    init_key.add_argument("--output", type=Path, required=True)
    init_key.set_defaults(handler=command_init_key)

    init_access_token = subparsers.add_parser("init-access-token")
    init_access_token.add_argument("--output", type=Path, required=True)
    init_access_token.set_defaults(handler=command_init_access_token)

    init_events = subparsers.add_parser("init-events")
    init_events.add_argument("--input", type=Path, required=True)
    init_events.add_argument("--output", type=Path, required=True)
    init_events.add_argument("--head-ledger", type=Path, required=True)
    init_events.add_argument("--annotator-id", required=True)
    init_events.add_argument("--expected-items", type=int, required=True)
    add_key_argument(init_events)
    add_config_arguments(init_events)
    add_identity_registry_arguments(init_events)
    add_run_contract_arguments(init_events)
    init_events.set_defaults(handler=command_init_events)

    append_event = subparsers.add_parser("append-event")
    append_event.add_argument("--input", type=Path, required=True)
    append_event.add_argument("--event-log", type=Path, required=True)
    append_event.add_argument("--head-ledger", type=Path, required=True)
    append_event.add_argument("--annotation-json", type=Path, required=True)
    append_event.add_argument("--item-id", required=True)
    append_event.add_argument("--expected-event-index", type=int, required=True)
    append_event.add_argument(
        "--annotation-status", choices=["draft", "completed"], required=True
    )
    append_event.add_argument("--submitted-at-utc", required=True)
    append_event.add_argument("--annotator-id", required=True)
    append_event.add_argument("--expected-items", type=int, required=True)
    add_config_arguments(append_event)
    add_key_argument(append_event)
    add_identity_registry_arguments(append_event)
    add_run_contract_arguments(append_event)
    append_event.set_defaults(handler=command_append_event)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--input", type=Path, required=True)
    freeze.add_argument("--event-log", type=Path, required=True)
    freeze.add_argument("--head-ledger", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument("--annotator-id", required=True)
    freeze.add_argument("--expected-items", type=int, required=True)
    freeze.add_argument("--locked-at-utc", required=True)
    add_config_arguments(freeze)
    add_key_argument(freeze)
    add_identity_registry_arguments(freeze)
    add_run_contract_arguments(freeze)
    freeze.set_defaults(handler=command_freeze)

    release_visual = subparsers.add_parser("release-visual")
    release_visual.add_argument("--workspace-root", type=Path, required=True)
    release_visual.add_argument("--private-visual", type=Path, required=True)
    release_visual.add_argument("--mapping", type=Path, required=True)
    release_visual.add_argument("--prior-input-a", type=Path, required=True)
    release_visual.add_argument("--prior-input-b", type=Path, required=True)
    release_visual.add_argument("--prior-frozen-a", type=Path, required=True)
    release_visual.add_argument("--prior-frozen-b", type=Path, required=True)
    release_visual.add_argument(
        "--next-stage", choices=list(VISUAL_STAGES[1:]), required=True
    )
    release_visual.add_argument("--expected-items", type=int, required=True)
    release_visual.add_argument("--output-root", type=Path, required=True)
    add_config_arguments(release_visual)
    add_key_argument(release_visual)
    add_run_contract_arguments(release_visual)
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
    add_run_contract_arguments(release_target)
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
    add_run_contract_arguments(report)
    report.set_defaults(handler=command_report)

    prepare_adjudication = subparsers.add_parser("prepare-adjudication")
    prepare_adjudication.add_argument(
        "--pre-adjudication-rows", type=Path, required=True
    )
    prepare_adjudication.add_argument(
        "--pre-adjudication-summary", type=Path, required=True
    )
    prepare_adjudication.add_argument("--private-visual", type=Path, required=True)
    prepare_adjudication.add_argument("--mapping", type=Path, required=True)
    prepare_adjudication.add_argument(
        "--target-validator-stage2-input", type=Path, required=True
    )
    prepare_adjudication.add_argument("--workspace-root", type=Path, required=True)
    prepare_adjudication.add_argument("--visual-adjudicator-id", required=True)
    prepare_adjudication.add_argument("--target-adjudicator-id", required=True)
    prepare_adjudication.add_argument("--expected-items", type=int, required=True)
    prepare_adjudication.add_argument("--output-root", type=Path, required=True)
    add_config_arguments(prepare_adjudication)
    add_key_argument(prepare_adjudication)
    add_identity_registry_arguments(prepare_adjudication)
    add_run_contract_arguments(prepare_adjudication)
    prepare_adjudication.set_defaults(handler=command_prepare_adjudication)

    apply_adjudication = subparsers.add_parser("apply-adjudication")
    apply_adjudication.add_argument("--pre-adjudication-rows", type=Path, required=True)
    apply_adjudication.add_argument(
        "--pre-adjudication-summary", type=Path, required=True
    )
    apply_adjudication.add_argument(
        "--adjudication-release-report", type=Path, required=True
    )
    apply_adjudication.add_argument(
        "--visual-adjudication-input", type=Path, required=True
    )
    apply_adjudication.add_argument(
        "--visual-adjudication-frozen", type=Path, required=True
    )
    apply_adjudication.add_argument(
        "--target-adjudication-input", type=Path, required=True
    )
    apply_adjudication.add_argument(
        "--target-adjudication-frozen", type=Path, required=True
    )
    apply_adjudication.add_argument("--output", type=Path, required=True)
    apply_adjudication.add_argument("--summary-out", type=Path, required=True)
    add_config_arguments(apply_adjudication)
    add_key_argument(apply_adjudication)
    add_identity_registry_arguments(apply_adjudication)
    add_run_contract_arguments(apply_adjudication)
    apply_adjudication.set_defaults(handler=command_apply_adjudication)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
