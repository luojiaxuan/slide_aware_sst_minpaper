#!/usr/bin/env python3
"""Build the leak-resistant MCIF beyond-OCR reliability-v2 base workspace."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from scripts.build_mcif_visual_token_controls import (
    canonical_sha256,
    file_sha256,
    load_jsonl,
)
from scripts.mcif_beyond_ocr_validation import (
    validate_input_rows as validate_v1_input_rows,
)
from scripts.mcif_beyond_ocr_validation import (
    validate_mapping_rows as validate_v1_mapping_rows,
)

ORDERING_SEED = "mcif-beyond-ocr-reliability-v2-20260801"
VISUAL_R0_SCHEMA = "mcif_beyond_ocr_visual_r0_item_v2"
TARGET_AUTHOR_SCHEMA = "mcif_beyond_ocr_target_author_item_v2"
TARGET_VALIDATOR_STAGE1_SCHEMA = "mcif_beyond_ocr_target_validator_stage1_item_v2"
PRIVATE_VISUAL_SCHEMA = "mcif_beyond_ocr_visual_private_material_v2"
PRIVATE_TARGET_SCHEMA = "mcif_beyond_ocr_target_private_material_v2"
MAPPING_SCHEMA = "mcif_beyond_ocr_reliability_mapping_v2"
RUN_CONTRACT_SCHEMA = "mcif_beyond_ocr_run_contract_v2"
IDENTITY_REGISTRY_SCHEMA = "mcif_beyond_ocr_identity_registry_v2"
ACCESS_TOKEN_MANIFEST_SCHEMA = "mcif_beyond_ocr_access_token_manifest_v2"


def validate_role_access_token_hashes(
    values: Any, required_roles: list[str]
) -> dict[str, str]:
    if (
        not isinstance(values, dict)
        or set(values) != set(required_roles)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in values.values()
        )
        or len(set(values.values())) != len(values)
    ):
        raise ValueError("MCIF reliability-v2 role access-token hashes differ")
    return {role: values[role] for role in sorted(values)}


def validate_access_token_manifest(
    manifest: Any, required_roles: list[str]
) -> dict[str, str]:
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "role_access_token_sha256"}
        or manifest["schema_version"] != ACCESS_TOKEN_MANIFEST_SCHEMA
    ):
        raise ValueError("MCIF reliability-v2 access-token manifest differs")
    return validate_role_access_token_hashes(
        manifest["role_access_token_sha256"], required_roles
    )


def deterministic_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{ORDERING_SEED}\0{namespace}\0{value}".encode()).hexdigest()


def opaque_ids(candidates: list[str], namespace: str, prefix: str) -> dict[str, str]:
    ordered = sorted(candidates, key=lambda value: deterministic_key(namespace, value))
    return {
        candidate_id: f"{prefix}{index:04d}"
        for index, candidate_id in enumerate(ordered, 1)
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def write_checksums(root: Path) -> tuple[int, str]:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum = root / "SHA256SUMS"
    checksum.write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return len(paths), file_sha256(checksum)


def hashed(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["row_sha256"] = canonical_sha256(output)
    return output


def signed_release(row: dict[str, Any], key: bytes) -> dict[str, Any]:
    if len(key) < 32:
        raise ValueError(
            "MCIF reliability-v2 release key must contain at least 32 bytes"
        )
    output = hashed(row)
    output["release_hmac_sha256"] = hmac.new(
        key, canonical_sha256(output).encode(), hashlib.sha256
    ).hexdigest()
    return output


def validate_identity_registry(
    registry: dict[str, Any], required_roles: list[str]
) -> dict[str, str]:
    if set(registry) != {
        "schema_version",
        "people",
        "role_assignments",
        "registry_sha256",
    }:
        raise ValueError("MCIF reliability-v2 identity registry keys differ")
    payload = {
        name: value for name, value in registry.items() if name != "registry_sha256"
    }
    if registry["schema_version"] != IDENTITY_REGISTRY_SCHEMA or registry[
        "registry_sha256"
    ] != canonical_sha256(payload):
        raise ValueError("MCIF reliability-v2 identity registry hash/schema differs")
    people = registry["people"]
    if not isinstance(people, list) or not people:
        raise ValueError("MCIF reliability-v2 identity registry has no people")
    people_by_id: dict[str, dict[str, Any]] = {}
    alias_owners: dict[str, str] = {}
    for person in people:
        if not isinstance(person, dict) or set(person) != {"person_id", "aliases"}:
            raise ValueError("MCIF reliability-v2 identity registry person differs")
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
        normalized = []
        for value in [person_id, *aliases]:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("MCIF reliability-v2 identity alias differs")
            normalized.append(" ".join(value.casefold().split()))
        if len(set(normalized)) != len(normalized):
            raise ValueError("MCIF reliability-v2 identity registry aliases duplicate")
        for alias in normalized:
            if alias in alias_owners:
                raise ValueError(
                    "MCIF reliability-v2 identity registry alias maps to multiple people"
                )
            alias_owners[alias] = person_id
        people_by_id[person_id] = person
    assignments = registry["role_assignments"]
    if not isinstance(assignments, dict) or set(assignments) != set(required_roles):
        raise ValueError("MCIF reliability-v2 identity registry role set differs")
    if any(person_id not in people_by_id for person_id in assignments.values()):
        raise ValueError("MCIF reliability-v2 identity registry assignment differs")
    if len(set(assignments.values())) != len(assignments):
        raise ValueError("MCIF reliability-v2 identity registry roles must be disjoint")
    return dict(assignments)


def build_run_contract(
    *,
    config: dict[str, Any],
    config_file_sha256: str,
    identity_registry: dict[str, Any],
    release_key: bytes,
    expected_items: int,
    expected_visual_sha256: str,
    expected_target_sha256: str,
    expected_mapping_sha256: str,
    private_visual_rows_sha256: str,
    private_target_rows_sha256: str,
    private_mapping_rows_sha256: str,
    role_access_token_sha256: dict[str, str],
    source_hf_revision: str,
    builder_git_commit: str,
) -> dict[str, Any]:
    if len(release_key) < 32:
        raise ValueError(
            "MCIF reliability-v2 release key must contain at least 32 bytes"
        )
    hashes = (
        config_file_sha256,
        expected_visual_sha256,
        expected_target_sha256,
        expected_mapping_sha256,
        private_visual_rows_sha256,
        private_target_rows_sha256,
        private_mapping_rows_sha256,
    )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise ValueError("MCIF reliability-v2 run contract hash differs")
    if any(
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
        for value in (source_hf_revision, builder_git_commit)
    ):
        raise ValueError("MCIF reliability-v2 run contract revision differs")
    if (
        not isinstance(expected_items, int)
        or isinstance(expected_items, bool)
        or expected_items <= 0
    ):
        raise ValueError("MCIF reliability-v2 run contract item count differs")
    required_roles = config["identity"]["required_disjoint_roles"]
    validate_identity_registry(identity_registry, required_roles)
    role_access_token_sha256 = validate_role_access_token_hashes(
        role_access_token_sha256, required_roles
    )
    payload = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "status": "PRE_ANNOTATION_RUN_CONTRACT_FROZEN",
        "builder_git_commit": builder_git_commit,
        "config_payload_sha256": canonical_sha256(config),
        "config_file_sha256": config_file_sha256,
        "identity_registry_sha256": identity_registry["registry_sha256"],
        "release_key_sha256": hashlib.sha256(release_key).hexdigest(),
        "source_workspace_hf_revision": source_hf_revision,
        "source_visual_sha256": expected_visual_sha256,
        "source_target_sha256": expected_target_sha256,
        "source_mapping_sha256": expected_mapping_sha256,
        "private_visual_rows_sha256": private_visual_rows_sha256,
        "private_target_rows_sha256": private_target_rows_sha256,
        "private_mapping_rows_sha256": private_mapping_rows_sha256,
        "role_access_token_sha256": role_access_token_sha256,
        "expected_items": expected_items,
        "required_disjoint_roles": sorted(required_roles),
        "audio_release_allowed": False,
        "inference_release_allowed": False,
    }
    output = dict(payload)
    output["contract_hmac_sha256"] = hmac.new(
        release_key, canonical_sha256(payload).encode(), hashlib.sha256
    ).hexdigest()
    return output


def pending_fields(judgment_field: str) -> dict[str, Any]:
    return {
        "annotation_status": "pending",
        judgment_field: None,
        "reason_codes": [],
        "annotation_note": "",
        "annotator_id": None,
        "locked_at_utc": None,
    }


def validate_source_workspace(
    root: Path,
    *,
    expected_items: int,
    expected_visual_sha256: str,
    expected_target_sha256: str,
    expected_mapping_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    visual_path = root / "visual_validator_view" / "validation_items.jsonl"
    target_path = root / "target_author_view" / "annotation_items.jsonl"
    mapping_path = root / "scorer_private" / "item_mapping.jsonl"
    for path, expected in (
        (visual_path, expected_visual_sha256),
        (target_path, expected_target_sha256),
        (mapping_path, expected_mapping_sha256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"Superseded v1 source hash differs: {path.name}")
    visual = load_jsonl(visual_path)
    target = load_jsonl(target_path)
    mapping = load_jsonl(mapping_path)
    visual_by_id = validate_v1_input_rows(
        visual, role="visual", expected_items=expected_items
    )
    target_by_id = validate_v1_input_rows(
        target, role="target", expected_items=expected_items
    )
    validate_v1_mapping_rows(
        mapping, visual_by_id, target_by_id, expected_items=expected_items
    )
    return visual, target, mapping


def candidate_indices(
    visual: list[dict[str, Any]],
    target: list[dict[str, Any]],
    mapping: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    visual_by_id = {row["item_id"]: row for row in visual}
    target_by_id = {row["item_id"]: row for row in target}
    visual_by_candidate: dict[str, dict[str, Any]] = {}
    target_by_candidate: dict[str, dict[str, Any]] = {}
    for row in mapping:
        candidate_id = row["candidate_id"]
        visual_by_candidate[candidate_id] = visual_by_id[row["visual_item_id"]]
        target_by_candidate[candidate_id] = target_by_id[row["target_item_id"]]
    return visual_by_candidate, target_by_candidate


def copy_private_media(
    temporary: Path,
    source_root: Path,
    visual_by_candidate: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    media_by_sha: dict[str, dict[str, Any]] = {}
    media = sorted(
        {
            row["current_slide"]["sha256"]: row["current_slide"]
            for row in visual_by_candidate.values()
        }.values(),
        key=lambda value: deterministic_key("private-media", value["sha256"]),
    )
    for index, descriptor in enumerate(media, 1):
        source = source_root / "visual_validator_view" / descriptor["path"]
        if file_sha256(source) != descriptor["sha256"]:
            raise ValueError(
                f"Superseded v1 media hash differs: {descriptor['media_id']}"
            )
        private_id = f"PM{index:04d}"
        relative = Path("media") / f"{private_id}.png"
        target = temporary / "scorer_private" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_sha256(target) != descriptor["sha256"]:
            raise ValueError(f"Copied private media hash differs: {private_id}")
        media_by_sha[descriptor["sha256"]] = {
            "private_media_id": private_id,
            "private_path": relative.as_posix(),
            "sha256": descriptor["sha256"],
            "width": descriptor["width"],
            "height": descriptor["height"],
        }
    return media_by_sha


def build_rows(
    visual: list[dict[str, Any]],
    target: list[dict[str, Any]],
    mapping: list[dict[str, Any]],
    media_by_sha: dict[str, dict[str, Any]],
    release_key: bytes,
    run_contract_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    visual_by_candidate, target_by_candidate = candidate_indices(
        visual, target, mapping
    )
    candidate_ids = [row["candidate_id"] for row in mapping]
    ids = {
        "visual_a": opaque_ids(candidate_ids, "visual-a", "MCIF-BOR-A"),
        "visual_b": opaque_ids(candidate_ids, "visual-b", "MCIF-BOR-B"),
        "target_author": opaque_ids(candidate_ids, "target-author", "MCIF-BOR-TA"),
        "target_validator": opaque_ids(
            candidate_ids, "target-validator", "MCIF-BOR-TV"
        ),
    }
    visual_releases: dict[str, list[dict[str, Any]]] = {"visual_a": [], "visual_b": []}
    private_visual = []
    target_author = []
    target_validator = []
    private_target = []
    private_mapping = []
    mapping_by_candidate = {row["candidate_id"]: row for row in mapping}
    for candidate_id in candidate_ids:
        visual_source = visual_by_candidate[candidate_id]
        target_source = target_by_candidate[candidate_id]
        mapping_source = mapping_by_candidate[candidate_id]
        for cohort in ("visual_a", "visual_b"):
            row = signed_release(
                {
                    "schema_version": VISUAL_R0_SCHEMA,
                    "status": "R0_RELEASED_NO_LATER_VISUAL_EVIDENCE",
                    "role": cohort,
                    "stage": "r0",
                    "item_id": ids[cohort][candidate_id],
                    "candidate_source_en": visual_source["candidate_source_en"],
                    "candidate_kind": visual_source["candidate_kind"],
                    "candidate_token_count": visual_source["candidate_token_count"],
                    "r0_text": visual_source["current_slide_r0_text"],
                    **pending_fields("r0_support"),
                    "r1_exposed": False,
                    "pixels_exposed": False,
                    "descriptor_exposed": False,
                    "reference_exposed": False,
                    "timing_exposed": False,
                    "run_contract_sha256": run_contract_sha256,
                },
                release_key,
            )
            visual_releases[cohort].append(row)
        media = media_by_sha[visual_source["current_slide"]["sha256"]]
        private_visual.append(
            hashed(
                {
                    "schema_version": PRIVATE_VISUAL_SCHEMA,
                    "candidate_id": candidate_id,
                    "visual_a_item_id": ids["visual_a"][candidate_id],
                    "visual_b_item_id": ids["visual_b"][candidate_id],
                    "candidate_source_en": visual_source["candidate_source_en"],
                    "candidate_kind": visual_source["candidate_kind"],
                    "candidate_token_count": visual_source["candidate_token_count"],
                    "evidence_tier": mapping_source["evidence_tier"],
                    "r0_text": visual_source["current_slide_r0_text"],
                    "r1_blocks": visual_source["current_slide_r1_blocks"],
                    "private_media": media,
                    "proposed_evidence_origins": visual_source[
                        "proposed_evidence_origins"
                    ],
                    "v1_visual_item_id": visual_source["item_id"],
                    "v1_visual_row_sha256": visual_source["row_sha256"],
                }
            )
        )
        common_target = {
            "candidate_source_en": target_source["candidate_source_en"],
            "candidate_kind": target_source["candidate_kind"],
            "candidate_token_count": target_source["candidate_token_count"],
            "source_reference_en": target_source["source_reference_en"],
            "target_reference_zh": target_source["target_reference_zh"],
            "slide_or_visual_exposed": False,
            "timing_exposed": False,
        }
        target_author.append(
            signed_release(
                {
                    "schema_version": TARGET_AUTHOR_SCHEMA,
                    "status": "PENDING_INDEPENDENT_TARGET_AUTHORING",
                    "role": "target_author",
                    "item_id": ids["target_author"][candidate_id],
                    **common_target,
                    "annotation_status": "pending",
                    "candidate_eligibility": None,
                    "canonical_source_event_en": "",
                    "acceptable_target_realizations_zh": [],
                    "forbidden_target_realizations_zh": [],
                    "target_reference_alignment": None,
                    "reason_codes": [],
                    "annotation_note": "",
                    "annotator_id": None,
                    "locked_at_utc": None,
                    "run_contract_sha256": run_contract_sha256,
                },
                release_key,
            )
        )
        target_validator.append(
            signed_release(
                {
                    "schema_version": TARGET_VALIDATOR_STAGE1_SCHEMA,
                    "status": "STAGE1_RELEASED_NO_AUTHOR_TEXT",
                    "role": "target_validator",
                    "stage": "independent_alignment",
                    "item_id": ids["target_validator"][candidate_id],
                    **common_target,
                    "annotation_status": "pending",
                    "candidate_eligibility": None,
                    "target_reference_alignment": None,
                    "reason_codes": [],
                    "annotation_note": "",
                    "annotator_id": None,
                    "locked_at_utc": None,
                    "author_identity_exposed": False,
                    "author_labels_exposed": False,
                    "author_scoring_text_exposed": False,
                    "run_contract_sha256": run_contract_sha256,
                },
                release_key,
            )
        )
        private_target.append(
            hashed(
                {
                    "schema_version": PRIVATE_TARGET_SCHEMA,
                    "candidate_id": candidate_id,
                    "target_author_item_id": ids["target_author"][candidate_id],
                    "target_validator_item_id": ids["target_validator"][candidate_id],
                    "v1_target_item_id": target_source["item_id"],
                    "v1_target_row_sha256": target_source["row_sha256"],
                }
            )
        )
        private_mapping.append(
            hashed(
                {
                    "schema_version": MAPPING_SCHEMA,
                    "candidate_id": candidate_id,
                    "evidence_tier": mapping_source["evidence_tier"],
                    "talk_id": mapping_source["talk_id"],
                    "segment_id": mapping_source["segment_id"],
                    "current_state_id": mapping_source["current_state_id"],
                    "lead_lower_bound_sec": mapping_source["lead_lower_bound_sec"],
                    "visual_a_item_id": ids["visual_a"][candidate_id],
                    "visual_b_item_id": ids["visual_b"][candidate_id],
                    "target_author_item_id": ids["target_author"][candidate_id],
                    "target_validator_item_id": ids["target_validator"][candidate_id],
                    "v1_mapping_row_sha256": mapping_source["row_sha256"],
                    "human_labels_complete": False,
                    "audio_release_allowed": False,
                    "inference_release_allowed": False,
                }
            )
        )
    for rows in (
        visual_releases["visual_a"],
        visual_releases["visual_b"],
        target_author,
        target_validator,
    ):
        rows.sort(key=lambda row: row["item_id"])
    private_visual.sort(key=lambda row: row["candidate_id"])
    private_target.sort(key=lambda row: row["candidate_id"])
    private_mapping.sort(key=lambda row: row["candidate_id"])
    return {
        "visual_a": visual_releases["visual_a"],
        "visual_b": visual_releases["visual_b"],
        "target_author": target_author,
        "target_validator": target_validator,
        "private_visual": private_visual,
        "private_target": private_target,
        "private_mapping": private_mapping,
    }


def build_bundle(
    output_root: Path,
    *,
    source_root: Path,
    expected_items: int,
    expected_visual_sha256: str,
    expected_target_sha256: str,
    expected_mapping_sha256: str,
    source_hf_revision: str,
    config_sha256: str,
    config: dict[str, Any],
    identity_registry: dict[str, Any],
    builder_git_commit: str,
    release_key: bytes,
    role_access_token_sha256: dict[str, str],
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("MCIF reliability-v2 workspace must not already exist")
    visual, target, mapping = validate_source_workspace(
        source_root,
        expected_items=expected_items,
        expected_visual_sha256=expected_visual_sha256,
        expected_target_sha256=expected_target_sha256,
        expected_mapping_sha256=expected_mapping_sha256,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        for directory in (
            "visual_a_r0_view",
            "visual_b_r0_view",
            "target_author_view",
            "target_validator_stage1_view",
            "scorer_private",
        ):
            (temporary / directory).mkdir()
        media_by_sha = copy_private_media(
            temporary, source_root, candidate_indices(visual, target, mapping)[0]
        )
        precontract_rows = build_rows(
            visual,
            target,
            mapping,
            media_by_sha,
            release_key,
            "0" * 64,
        )
        run_contract = build_run_contract(
            config=config,
            config_file_sha256=config_sha256,
            identity_registry=identity_registry,
            release_key=release_key,
            expected_items=expected_items,
            expected_visual_sha256=expected_visual_sha256,
            expected_target_sha256=expected_target_sha256,
            expected_mapping_sha256=expected_mapping_sha256,
            private_visual_rows_sha256=canonical_sha256(
                precontract_rows["private_visual"]
            ),
            private_target_rows_sha256=canonical_sha256(
                precontract_rows["private_target"]
            ),
            private_mapping_rows_sha256=canonical_sha256(
                precontract_rows["private_mapping"]
            ),
            role_access_token_sha256=role_access_token_sha256,
            source_hf_revision=source_hf_revision,
            builder_git_commit=builder_git_commit,
        )
        run_contract_sha256 = canonical_sha256(run_contract)
        rows = build_rows(
            visual,
            target,
            mapping,
            media_by_sha,
            release_key,
            run_contract_sha256,
        )
        for name in ("private_visual", "private_target", "private_mapping"):
            if rows[name] != precontract_rows[name]:
                raise ValueError(
                    f"MCIF reliability-v2 private rows depend on run contract: {name}"
                )
        paths = {
            "visual_a": temporary / "visual_a_r0_view" / "items.jsonl",
            "visual_b": temporary / "visual_b_r0_view" / "items.jsonl",
            "target_author": temporary / "target_author_view" / "items.jsonl",
            "target_validator": temporary
            / "target_validator_stage1_view"
            / "items.jsonl",
            "private_visual": temporary / "scorer_private" / "visual_material.jsonl",
            "private_target": temporary / "scorer_private" / "target_material.jsonl",
            "private_mapping": temporary / "scorer_private" / "item_mapping.jsonl",
        }
        for key, path in paths.items():
            write_jsonl(path, rows[key])
        (temporary / "scorer_private" / "run_contract.json").write_text(
            json.dumps(run_contract, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        run_contract_path = temporary / "scorer_private" / "run_contract.json"
        readmes = {
            "visual_a_r0_view": "Visual validator A: R0-only stage. No R1, pixels, descriptor, reference, identity mapping, or timing is present.\n",
            "visual_b_r0_view": "Visual validator B: R0-only stage. No R1, pixels, descriptor, reference, identity mapping, or timing is present.\n",
            "target_author_view": "Target author view. No visual evidence, timing, visual labels, or validator labels is present.\n",
            "target_validator_stage1_view": "Independent target alignment view. Author identity, labels, and scoring text are not present.\n",
            "scorer_private": "Private release material and identity mapping. Never distribute this subtree to annotators.\n",
        }
        for directory, text in readmes.items():
            (temporary / directory / "README.md").write_text(
                f"# MCIF Beyond-OCR Reliability V2\n\n{text}", encoding="utf-8"
            )
        report = {
            "schema_version": "mcif_beyond_ocr_reliability_workspace_report_v2",
            "status": "INITIAL_R0_AND_TARGET_STAGE1_RELEASES_READY_NO_HUMAN_LABELS",
            "builder_git_commit": builder_git_commit,
            "config_sha256": config_sha256,
            "release_key_sha256": hashlib.sha256(release_key).hexdigest(),
            "run_contract_sha256": run_contract_sha256,
            "identity_registry_sha256": identity_registry["registry_sha256"],
            "ordering_seed_sha256": hashlib.sha256(ORDERING_SEED.encode()).hexdigest(),
            "source_workspace_hf_revision": source_hf_revision,
            "source_visual_sha256": expected_visual_sha256,
            "source_target_sha256": expected_target_sha256,
            "source_mapping_sha256": expected_mapping_sha256,
            "items": expected_items,
            "talks": len({row["talk_id"] for row in mapping}),
            "segments": len({row["segment_id"] for row in mapping}),
            "states": len({row["current_state_id"] for row in mapping}),
            "visual_a_r0_items": len(rows["visual_a"]),
            "visual_b_r0_items": len(rows["visual_b"]),
            "target_author_items": len(rows["target_author"]),
            "target_validator_stage1_items": len(rows["target_validator"]),
            "private_media_files": len(media_by_sha),
            "human_labels": 0,
            "r1_released": False,
            "pixels_released": False,
            "descriptor_released": False,
            "author_text_released_to_validator": False,
            "audio_release_allowed": False,
            "inference_release_allowed": False,
            "file_sha256": {key: file_sha256(path) for key, path in paths.items()},
            "run_contract_file_sha256": file_sha256(run_contract_path),
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# MCIF Beyond-OCR Reliability Workspace V2\n\n"
            "Only one role subtree may be distributed to each human. Later visual stages and "
            "target author text remain scorer-private until prior complete freezes are verified.\n",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-items", type=int, required=True)
    parser.add_argument("--expected-visual-sha256", required=True)
    parser.add_argument("--expected-target-sha256", required=True)
    parser.add_argument("--expected-mapping-sha256", required=True)
    parser.add_argument("--source-hf-revision", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    parser.add_argument("--identity-registry", type=Path, required=True)
    parser.add_argument("--expected-identity-registry-file-sha256", required=True)
    parser.add_argument("--access-token-manifest", type=Path, required=True)
    parser.add_argument("--expected-access-token-manifest-file-sha256", required=True)
    args = parser.parse_args()
    if file_sha256(args.config) != args.expected_config_sha256:
        raise ValueError("MCIF reliability-v2 config hash differs")
    if args.hmac_key.is_symlink() or not args.hmac_key.is_file():
        raise ValueError("MCIF reliability-v2 HMAC key must be a regular file")
    if (
        args.identity_registry.is_symlink()
        or not args.identity_registry.is_file()
        or file_sha256(args.identity_registry)
        != args.expected_identity_registry_file_sha256
    ):
        raise ValueError("MCIF reliability-v2 identity registry file hash differs")
    if (
        args.access_token_manifest.is_symlink()
        or not args.access_token_manifest.is_file()
        or file_sha256(args.access_token_manifest)
        != args.expected_access_token_manifest_file_sha256
    ):
        raise ValueError("MCIF reliability-v2 access-token manifest hash differs")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    identity_registry = json.loads(args.identity_registry.read_text(encoding="utf-8"))
    access_token_manifest = json.loads(
        args.access_token_manifest.read_text(encoding="utf-8")
    )
    role_access_token_sha256 = validate_access_token_manifest(
        access_token_manifest, config["identity"]["required_disjoint_roles"]
    )
    report = build_bundle(
        args.output_root,
        source_root=args.source_root,
        expected_items=args.expected_items,
        expected_visual_sha256=args.expected_visual_sha256,
        expected_target_sha256=args.expected_target_sha256,
        expected_mapping_sha256=args.expected_mapping_sha256,
        source_hf_revision=args.source_hf_revision,
        config_sha256=args.expected_config_sha256,
        config=config,
        identity_registry=identity_registry,
        builder_git_commit=args.builder_git_commit,
        release_key=args.hmac_key.read_bytes(),
        role_access_token_sha256=role_access_token_sha256,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
