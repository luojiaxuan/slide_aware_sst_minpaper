import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.build_mcif_beyond_ocr_reliability_workspace import (
    ACCESS_TOKEN_MANIFEST_SCHEMA,
    TARGET_AUTHOR_SCHEMA,
    TARGET_VALIDATOR_STAGE1_SCHEMA,
    VISUAL_R0_SCHEMA,
    build_bundle,
    validate_access_token_manifest,
    validate_identity_registry,
)
from scripts.build_mcif_visual_token_controls import (
    canonical_sha256,
    file_sha256,
    load_jsonl,
)

CONFIG_PATH = (
    Path(__file__).parents[1] / "configs" / "mcif_beyond_ocr_reliability_v2.json"
)
IDENTITY_REGISTRY_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "templates"
    / "mcif_beyond_ocr_identity_registry_v2.example.json"
)
ACCESS_TOKEN_MANIFEST_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "templates"
    / "mcif_beyond_ocr_access_token_manifest_v2.example.json"
)


def identity_registry_fixture():
    assignments = {
        "visual_a": "Visual A",
        "visual_b": "Visual B",
        "target_author": "Target Author",
        "target_validator": "Target Validator",
        "visual_adjudicator": "Visual Adjudicator",
        "target_adjudicator": "Target Adjudicator",
    }
    registry = {
        "schema_version": "mcif_beyond_ocr_identity_registry_v2",
        "people": [
            {
                "person_id": person_id,
                "aliases": [f"{role.replace('_', '-')}@example.test"],
            }
            for role, person_id in assignments.items()
        ],
        "role_assignments": assignments,
    }
    registry["registry_sha256"] = canonical_sha256(registry)
    return registry


def test_example_identity_registry_is_schema_valid():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    registry = json.loads(IDENTITY_REGISTRY_PATH.read_text(encoding="utf-8"))
    assignments = validate_identity_registry(
        registry, config["identity"]["required_disjoint_roles"]
    )
    assert assignments == registry["role_assignments"]


def test_example_access_token_manifest_is_schema_valid():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(ACCESS_TOKEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == ACCESS_TOKEN_MANIFEST_SCHEMA
    assert (
        validate_access_token_manifest(
            manifest, config["identity"]["required_disjoint_roles"]
        )
        == manifest["role_access_token_sha256"]
    )


def hashed(row):
    result = dict(row)
    result["row_sha256"] = canonical_sha256(result)
    return result


def source_workspace(tmp_path: Path):
    root = tmp_path / "source-v1"
    visual_root = root / "visual_validator_view"
    target_root = root / "target_author_view"
    mapping_root = root / "scorer_private"
    media_root = visual_root / "media"
    for directory in (media_root, target_root, mapping_root):
        directory.mkdir(parents=True)
    visual = []
    target = []
    mapping = []
    for index, tier in enumerate(("r1_strict", "r2_semantic"), 1):
        image_path = media_root / f"M{index:04d}.png"
        Image.new("RGB", (32, 18), color=(20 * index, 40, 80)).save(image_path)
        visual_id = f"MCIF-BOV-V{index:04d}"
        target_id = f"MCIF-BOV-T{index:04d}"
        visual_row = hashed(
            {
                "schema_version": "mcif_beyond_ocr_visual_validation_item_v1",
                "status": "PENDING_INDEPENDENT_VISUAL_VALIDATION",
                "item_id": visual_id,
                "candidate_source_en": f"candidate {index}",
                "candidate_kind": "phrase",
                "candidate_token_count": 2,
                "evidence_channel": "structure_preserving_text"
                if index == 1
                else "raw_visual_semantics",
                "current_slide": {
                    "media_id": f"M{index:04d}",
                    "path": f"media/M{index:04d}.png",
                    "sha256": file_sha256(image_path),
                    "width": 32,
                    "height": 18,
                },
                "current_slide_r0_text": f"flat OCR {index}",
                "current_slide_r1_blocks": [
                    {
                        "content_kind": "chart_markdown",
                        "label": "chart",
                        "content": f"structured {index}",
                        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                        "reading_order": 0,
                    }
                ],
                "proposed_evidence_origins": []
                if index == 1
                else [
                    {
                        "descriptor_field": "scene_summary",
                        "descriptor_index": 0,
                        "descriptor_sha256": canonical_sha256("A relation is visible."),
                        "descriptor_text": "A relation is visible.",
                    }
                ],
                "requires_r1_insufficiency_judgment": index == 2,
                "annotation_status": "pending",
                "visual_evidence_correct": None,
                "candidate_supported_by_visual_evidence": None,
                "r0_insufficient": None,
                "r1_insufficient": None,
                "reason_codes": [],
                "annotation_note": "",
                "annotator_id": None,
                "locked_at_utc": None,
                "official_reference_consumed": True,
                "source_reference_exposed": False,
                "target_reference_exposed": False,
                "generative_model_output_exposed": index == 2,
            }
        )
        target_row = hashed(
            {
                "schema_version": "mcif_beyond_ocr_target_author_item_v1",
                "status": "PENDING_INDEPENDENT_TARGET_EVENT_AUTHORING",
                "item_id": target_id,
                "candidate_source_en": f"candidate {index}",
                "candidate_kind": "phrase",
                "candidate_token_count": 2,
                "source_reference_en": f"source reference {index}",
                "target_reference_zh": f"目标参考 {index}",
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
                "official_reference_consumed": True,
                "slide_or_ocr_exposed": False,
                "visual_evidence_origin_exposed": False,
                "generative_model_output_exposed": False,
            }
        )
        visual.append(visual_row)
        target.append(target_row)
        mapping.append(
            hashed(
                {
                    "schema_version": "mcif_beyond_ocr_validation_mapping_v1",
                    "candidate_id": f"candidate-id-{index}",
                    "evidence_tier": tier,
                    "talk_id": f"talk-{index}",
                    "segment_id": f"segment-{index}",
                    "current_state_id": f"state-{index}",
                    "lead_lower_bound_sec": 5.0 * index,
                    "visual_item_id": visual_id,
                    "visual_item_row_sha256": visual_row["row_sha256"],
                    "target_item_id": target_id,
                    "target_item_row_sha256": target_row["row_sha256"],
                    "human_labels_complete": False,
                }
            )
        )
    paths = {
        "visual": visual_root / "validation_items.jsonl",
        "target": target_root / "annotation_items.jsonl",
        "mapping": mapping_root / "item_mapping.jsonl",
    }
    for name, rows in (("visual", visual), ("target", target), ("mapping", mapping)):
        paths[name].write_text(
            "".join(
                __import__("json").dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
    return root, paths


def build_kwargs(tmp_path: Path):
    root, paths = source_workspace(tmp_path)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "source_root": root,
        "expected_items": 2,
        "expected_visual_sha256": file_sha256(paths["visual"]),
        "expected_target_sha256": file_sha256(paths["target"]),
        "expected_mapping_sha256": file_sha256(paths["mapping"]),
        "source_hf_revision": "1" * 40,
        "config_sha256": file_sha256(CONFIG_PATH),
        "config": config,
        "identity_registry": identity_registry_fixture(),
        "builder_git_commit": "3" * 40,
        "release_key": b"release-key-for-tests-must-be-32b",
        "role_access_token_sha256": {
            role: hashlib.sha256(f"access-token-{role}".encode()).hexdigest()
            for role in config["identity"]["required_disjoint_roles"]
        },
    }


def test_initial_releases_physically_exclude_future_evidence_and_author_text(tmp_path):
    kwargs = build_kwargs(tmp_path)
    output = tmp_path / "v2"
    report = build_bundle(output, **kwargs)
    contract = json.loads(
        (output / "scorer_private" / "run_contract.json").read_text(encoding="utf-8")
    )
    assert contract["private_visual_rows_sha256"] == canonical_sha256(
        load_jsonl(output / "scorer_private" / "visual_material.jsonl")
    )
    assert contract["private_target_rows_sha256"] == canonical_sha256(
        load_jsonl(output / "scorer_private" / "target_material.jsonl")
    )
    assert contract["private_mapping_rows_sha256"] == canonical_sha256(
        load_jsonl(output / "scorer_private" / "item_mapping.jsonl")
    )
    assert report["run_contract_file_sha256"] == file_sha256(
        output / "scorer_private" / "run_contract.json"
    )
    visual_a = load_jsonl(output / "visual_a_r0_view" / "items.jsonl")
    visual_b = load_jsonl(output / "visual_b_r0_view" / "items.jsonl")
    author = load_jsonl(output / "target_author_view" / "items.jsonl")
    validator = load_jsonl(output / "target_validator_stage1_view" / "items.jsonl")

    assert report["human_labels"] == 0
    assert (
        report["release_key_sha256"]
        == hashlib.sha256(kwargs["release_key"]).hexdigest()
    )
    assert len(visual_a) == len(visual_b) == len(author) == len(validator) == 2
    assert all(row["schema_version"] == VISUAL_R0_SCHEMA for row in visual_a + visual_b)
    assert all(row["schema_version"] == TARGET_AUTHOR_SCHEMA for row in author)
    assert all(
        row["schema_version"] == TARGET_VALIDATOR_STAGE1_SCHEMA for row in validator
    )
    assert {row["item_id"] for row in visual_a}.isdisjoint(
        {row["item_id"] for row in visual_b}
    )
    forbidden_visual = {
        "r1_blocks",
        "current_slide_r1_blocks",
        "current_slide",
        "private_media",
        "proposed_evidence_origins",
        "source_reference_en",
        "target_reference_zh",
        "talk_id",
        "segment_id",
    }
    assert all(forbidden_visual.isdisjoint(row) for row in visual_a + visual_b)
    forbidden_validator = {
        "canonical_source_event_en",
        "acceptable_target_realizations_zh",
        "forbidden_target_realizations_zh",
        "author_id",
        "author_labels",
    }
    assert all(forbidden_validator.isdisjoint(row) for row in validator)
    assert not list((output / "visual_a_r0_view").rglob("*.png"))
    assert not list((output / "visual_b_r0_view").rglob("*.png"))
    assert len(list((output / "scorer_private" / "media").glob("*.png"))) == 2


def test_workspace_is_reproducible_create_once_and_source_hash_bound(tmp_path):
    kwargs = build_kwargs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_bundle(first, **kwargs)
    build_bundle(second, **kwargs)
    first_bytes = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_bytes = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_bytes == second_bytes
    with pytest.raises(FileExistsError, match="must not already exist"):
        build_bundle(first, **kwargs)
    drifted = copy.deepcopy(kwargs)
    drifted["expected_visual_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source hash differs"):
        build_bundle(tmp_path / "drifted", **drifted)


def test_workspace_rejects_missing_or_reused_role_access_tokens(tmp_path):
    kwargs = build_kwargs(tmp_path)
    missing = copy.deepcopy(kwargs)
    missing["role_access_token_sha256"].pop("visual_a")
    with pytest.raises(ValueError, match="role access-token hashes differ"):
        build_bundle(tmp_path / "missing-token", **missing)

    reused = copy.deepcopy(kwargs)
    reused["role_access_token_sha256"]["visual_a"] = reused["role_access_token_sha256"][
        "visual_b"
    ]
    with pytest.raises(ValueError, match="role access-token hashes differ"):
        build_bundle(tmp_path / "reused-token", **reused)


def test_v1_label_or_media_drift_is_rejected_without_partial_output(tmp_path):
    kwargs = build_kwargs(tmp_path)
    visual_path = (
        kwargs["source_root"] / "visual_validator_view" / "validation_items.jsonl"
    )
    rows = load_jsonl(visual_path)
    rows[0]["annotation_status"] = "completed"
    rows[0]["row_sha256"] = canonical_sha256(
        {key: value for key, value in rows[0].items() if key != "row_sha256"}
    )
    visual_path.write_text(
        "".join(
            __import__("json").dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    kwargs["expected_visual_sha256"] = file_sha256(visual_path)
    output = tmp_path / "labeled"
    with pytest.raises(ValueError, match="premature labels"):
        build_bundle(output, **kwargs)
    assert not output.exists()
