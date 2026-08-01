import copy
import json
from pathlib import Path

import pytest

from scripts.build_mcif_beyond_ocr_validation_workspace import (
    MAPPING_SCHEMA,
    TARGET_SCHEMA,
    VISUAL_SCHEMA,
)
from scripts.build_mcif_visual_token_controls import canonical_sha256, file_sha256
from scripts.mcif_beyond_ocr_validation import (
    TARGET_FROZEN_SCHEMA,
    TARGET_WORKING_SCHEMA,
    VISUAL_FROZEN_SCHEMA,
    VISUAL_WORKING_SCHEMA,
    freeze_role,
    initialize_working_rows,
    join_role_freezes,
    load_frozen_config,
    validate_frozen_rows,
    validate_input_rows,
    validate_mapping_rows,
    validate_working_row,
    write_jsonl_atomic,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "mcif_beyond_ocr_validation_v1.json"


def hashed(row):
    result = dict(row)
    result["row_sha256"] = canonical_sha256(result)
    return result


def visual_input(index: int, *, r2: bool) -> dict:
    return hashed(
        {
            "schema_version": VISUAL_SCHEMA,
            "status": "PENDING_INDEPENDENT_VISUAL_VALIDATION",
            "item_id": f"MCIF-BOV-V{index:04d}",
            "candidate_source_en": f"candidate {index}",
            "candidate_kind": "token",
            "candidate_token_count": 1,
            "evidence_channel": "raw_visual_semantics" if r2 else "structure_preserving_text",
            "current_slide": {
                "media_id": f"M{index:04d}",
                "path": f"media/M{index:04d}.png",
                "sha256": f"{index:064x}",
                "width": 32,
                "height": 18,
            },
            "current_slide_r0_text": "flat OCR",
            "current_slide_r1_blocks": [
                {
                    "content_kind": "chart_markdown",
                    "label": "chart",
                    "content": "candidate",
                    "bbox_norm": [0, 0, 1, 1],
                    "reading_order": 0,
                }
            ],
            "proposed_evidence_origins": [
                (
                    {
                        "descriptor_field": "scene_summary",
                        "descriptor_index": 0,
                        "descriptor_text": "candidate relation",
                        "descriptor_sha256": "a" * 64,
                    }
                    if r2
                    else {
                        "block_id": 0,
                        "content_kind": "chart_markdown",
                        "label": "chart",
                        "content": "candidate",
                        "content_sha256": "b" * 64,
                    }
                )
            ],
            "requires_r1_insufficiency_judgment": r2,
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
            "generative_model_output_exposed": r2,
        }
    )


def target_input(index: int) -> dict:
    return hashed(
        {
            "schema_version": TARGET_SCHEMA,
            "status": "PENDING_INDEPENDENT_TARGET_EVENT_AUTHORING",
            "item_id": f"MCIF-BOV-T{index:04d}",
            "candidate_source_en": f"candidate {index}",
            "candidate_kind": "token",
            "candidate_token_count": 1,
            "source_reference_en": f"English candidate {index}",
            "target_reference_zh": f"中文候选{index}",
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


def mapping_row(index: int, visual: dict, target: dict, *, tier: str) -> dict:
    return hashed(
        {
            "schema_version": MAPPING_SCHEMA,
            "candidate_id": f"mcif:talk:{'R2C' if tier == 'r2_semantic' else 'R1C'}{index:03d}",
            "candidate_row_sha256": f"{index + 10:064x}",
            "evidence_tier": tier,
            "talk_id": "talk",
            "segment_id": f"mcif:talk:SEG{index:03d}",
            "talk_segment_index": index,
            "source_segment_offset_sec": 10.0 + index,
            "source_segment_end_sec": 12.0 + index,
            "reference_segment_row_sha256": "c" * 64,
            "current_state_id": f"mcif:talk:S{index:03d}",
            "current_state_row_sha256": "d" * 64,
            "current_evidence_available_sec": 0.5,
            "earliest_contiguous_state_id": "mcif:talk:S000",
            "earliest_contiguous_state_row_sha256": "e" * 64,
            "earliest_contiguous_evidence_sec": 0.5,
            "lead_lower_bound_sec": 10.0,
            "visual_item_id": visual["item_id"],
            "visual_item_row_sha256": visual["row_sha256"],
            "target_item_id": target["item_id"],
            "target_item_row_sha256": target["row_sha256"],
            "media_sha256": visual["current_slide"]["sha256"],
            "official_reference_consumed": True,
            "human_labels_complete": False,
        }
    )


def labeled(row: dict, **updates) -> dict:
    result = copy.deepcopy(row)
    result.update(updates)
    result["row_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "row_sha256"}
    )
    return result


def fixture():
    visual = [visual_input(1, r2=False), visual_input(2, r2=True)]
    target = [target_input(1), target_input(2)]
    mapping = [
        mapping_row(1, visual[0], target[0], tier="r1_strict"),
        mapping_row(2, visual[1], target[1], tier="r2_semantic"),
    ]
    visual_working = initialize_working_rows(
        visual, role="visual", annotator_id="visual-01", expected_items=2
    )
    target_working = initialize_working_rows(
        target, role="target", annotator_id="target-01", expected_items=2
    )
    return visual, target, mapping, visual_working, target_working


def complete_visual(rows):
    return [
        labeled(
            rows[0],
            annotation_status="completed",
            visual_evidence_correct="yes",
            candidate_supported_by_visual_evidence="yes",
            r0_insufficient="yes",
        ),
        labeled(
            rows[1],
            annotation_status="completed",
            visual_evidence_correct="yes",
            candidate_supported_by_visual_evidence="yes",
            r0_insufficient="yes",
            r1_insufficient="no",
            reason_codes=["r1_already_sufficient"],
        ),
    ]


def complete_target(rows):
    return [
        labeled(
            rows[0],
            annotation_status="completed",
            candidate_eligibility="yes",
            canonical_source_event_en="candidate one",
            acceptable_target_realizations_zh=["候选一"],
            forbidden_target_realizations_zh=[],
            target_reference_alignment="explicit",
        ),
        labeled(
            rows[1],
            annotation_status="completed",
            candidate_eligibility="yes",
            canonical_source_event_en="candidate two",
            acceptable_target_realizations_zh=["候选二"],
            forbidden_target_realizations_zh=[],
            target_reference_alignment="paraphrased",
        ),
    ]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_frozen_config_matches_code_and_rejects_drift(tmp_path):
    config = load_frozen_config(CONFIG_PATH, file_sha256(CONFIG_PATH))
    assert config["roles"] == ["visual_validator", "target_author"]
    with pytest.raises(ValueError, match="config hash"):
        load_frozen_config(CONFIG_PATH, "0" * 64)
    changed = tmp_path / "changed.json"
    payload = copy.deepcopy(config)
    payload["visual_gate"]["r0_insufficient"] = "uncertain"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from code contract"):
        load_frozen_config(changed, file_sha256(changed))


def test_initialization_preserves_role_firewalls_and_blank_labels():
    visual, target, _, visual_working, target_working = fixture()
    assert all(row["schema_version"] == VISUAL_WORKING_SCHEMA for row in visual_working)
    assert all(row["schema_version"] == TARGET_WORKING_SCHEMA for row in target_working)
    assert all("source_reference_en" not in row for row in visual_working)
    assert all("current_slide" not in row for row in target_working)
    assert all(row["source_input_row_sha256"] == source["row_sha256"] for row, source in zip(visual_working, visual, strict=True))
    assert all(row["source_input_row_sha256"] == source["row_sha256"] for row, source in zip(target_working, target, strict=True))


def test_visual_completion_requires_all_tier_specific_judgments_and_reasons():
    visual, _, _, working, _ = fixture()
    valid = complete_visual(working)
    validate_working_row(
        valid[0], visual[0], role="visual", annotator_id="visual-01", allow_pending=False
    )
    validate_working_row(
        valid[1], visual[1], role="visual", annotator_id="visual-01", allow_pending=False
    )
    r2_missing = labeled(valid[1], r1_insufficient=None)
    with pytest.raises(ValueError, match="lacks a judgment"):
        validate_working_row(
            r2_missing,
            visual[1],
            role="visual",
            annotator_id="visual-01",
            allow_pending=False,
        )
    r1_extra = labeled(valid[0], r1_insufficient="yes")
    with pytest.raises(ValueError, match="R1 item"):
        validate_working_row(
            r1_extra,
            visual[0],
            role="visual",
            annotator_id="visual-01",
            allow_pending=False,
        )
    no_reason = labeled(valid[1], reason_codes=[])
    with pytest.raises(ValueError, match="lacks a reason"):
        validate_working_row(
            no_reason,
            visual[1],
            role="visual",
            annotator_id="visual-01",
            allow_pending=False,
        )


def test_target_completion_requires_scoring_text_only_for_eligible_items():
    _, target, _, _, working = fixture()
    valid = complete_target(working)[0]
    validate_working_row(
        valid, target[0], role="target", annotator_id="target-01", allow_pending=False
    )
    missing = labeled(valid, acceptable_target_realizations_zh=[])
    with pytest.raises(ValueError, match="lacks scoring text"):
        validate_working_row(
            missing,
            target[0],
            role="target",
            annotator_id="target-01",
            allow_pending=False,
        )
    rejected = labeled(
        working[0],
        annotation_status="completed",
        candidate_eligibility="no",
        target_reference_alignment="unsupported",
        reason_codes=["generic"],
    )
    validate_working_row(
        rejected, target[0], role="target", annotator_id="target-01", allow_pending=False
    )
    retained = labeled(rejected, canonical_source_event_en="should be empty")
    with pytest.raises(ValueError, match="retains scoring text"):
        validate_working_row(
            retained,
            target[0],
            role="target",
            annotator_id="target-01",
            allow_pending=False,
        )


@pytest.mark.parametrize("role", ["visual", "target"])
def test_pending_rows_reject_partial_labels(role):
    visual, target, _, visual_working, target_working = fixture()
    source = visual[0] if role == "visual" else target[0]
    working = visual_working[0] if role == "visual" else target_working[0]
    partial = labeled(working, annotation_note="started")
    with pytest.raises(ValueError, match="partial labels"):
        validate_working_row(
            partial,
            source,
            role=role,
            annotator_id=f"{role}-01",
            allow_pending=True,
        )


def test_mapping_validation_rejects_role_binding_drift():
    visual, target, mapping, _, _ = fixture()
    visual_by_id = validate_input_rows(visual, role="visual", expected_items=2)
    target_by_id = validate_input_rows(target, role="target", expected_items=2)
    validate_mapping_rows(mapping, visual_by_id, target_by_id, expected_items=2)
    changed = copy.deepcopy(mapping)
    changed[0]["target_item_row_sha256"] = "0" * 64
    changed[0]["row_sha256"] = canonical_sha256(
        {key: value for key, value in changed[0].items() if key != "row_sha256"}
    )
    with pytest.raises(ValueError, match="role binding"):
        validate_mapping_rows(changed, visual_by_id, target_by_id, expected_items=2)


def freeze_fixture(tmp_path):
    visual, target, mapping, visual_working, target_working = fixture()
    visual_output = tmp_path / "visual-frozen"
    target_output = tmp_path / "target-frozen"
    visual_report = freeze_role(
        visual_output,
        role="visual",
        input_rows=visual,
        working_rows=complete_visual(visual_working),
        input_sha256="1" * 64,
        working_sha256="2" * 64,
        config_sha256="3" * 64,
        annotator_id="visual-01",
        locked_at_utc="2026-08-01T20:00:00Z",
        expected_items=2,
    )
    target_report = freeze_role(
        target_output,
        role="target",
        input_rows=target,
        working_rows=complete_target(target_working),
        input_sha256="4" * 64,
        working_sha256="5" * 64,
        config_sha256="3" * 64,
        annotator_id="target-01",
        locked_at_utc="2026-08-01T20:01:00Z",
        expected_items=2,
    )
    return visual, target, mapping, visual_output, target_output, visual_report, target_report


def test_role_freeze_and_join_emit_only_joint_passes_pending_audio(tmp_path):
    visual, target, mapping, visual_root, target_root, visual_report, target_report = freeze_fixture(tmp_path)
    assert visual_report["gate_passed"] == 1
    assert target_report["gate_passed"] == 2
    visual_frozen_path = visual_root / "frozen_visual_annotations.jsonl"
    target_frozen_path = target_root / "frozen_target_annotations.jsonl"
    visual_frozen = read_jsonl(visual_frozen_path)
    target_frozen = read_jsonl(target_frozen_path)
    assert all(row["schema_version"] == VISUAL_FROZEN_SCHEMA for row in visual_frozen)
    assert all(row["schema_version"] == TARGET_FROZEN_SCHEMA for row in target_frozen)
    output = tmp_path / "joined"
    report = join_role_freezes(
        output,
        visual_input_rows=visual,
        target_input_rows=target,
        mapping_rows=mapping,
        visual_frozen_rows=visual_frozen,
        target_frozen_rows=target_frozen,
        visual_input_sha256="1" * 64,
        target_input_sha256="4" * 64,
        mapping_sha256="6" * 64,
        visual_frozen_sha256=file_sha256(visual_frozen_path),
        target_frozen_sha256=file_sha256(target_frozen_path),
        config_sha256="3" * 64,
        joined_at_utc="2026-08-01T20:02:00Z",
        expected_items=2,
    )
    joined = read_jsonl(output / "joined_candidate_decisions_private.jsonl")
    assert report["joint_gate_passed"] == 1
    assert report["joint_gate_passed_by_tier"] == {"r1_strict": 1}
    assert sum(row["joint_gate_passed"] for row in joined) == 1
    assert all(row["audio_first_sufficient_sec"] is None for row in joined)
    assert all(row["primary_eligible"] is None for row in joined)
    with pytest.raises(FileExistsError, match="must not already exist"):
        join_role_freezes(
            output,
            visual_input_rows=visual,
            target_input_rows=target,
            mapping_rows=mapping,
            visual_frozen_rows=visual_frozen,
            target_frozen_rows=target_frozen,
            visual_input_sha256="1" * 64,
            target_input_sha256="4" * 64,
            mapping_sha256="6" * 64,
            visual_frozen_sha256=file_sha256(visual_frozen_path),
            target_frozen_sha256=file_sha256(target_frozen_path),
            config_sha256="3" * 64,
            joined_at_utc="2026-08-01T20:02:00Z",
            expected_items=2,
        )


def test_join_rejects_frozen_gate_tampering_even_with_rehashed_row(tmp_path):
    visual, _, _, visual_root, _, _, _ = freeze_fixture(tmp_path)
    frozen = read_jsonl(visual_root / "frozen_visual_annotations.jsonl")
    frozen[0]["role_gate_passed"] = False
    frozen[0]["row_sha256"] = canonical_sha256(
        {key: value for key, value in frozen[0].items() if key != "row_sha256"}
    )
    source_by_id = validate_input_rows(visual, role="visual", expected_items=2)
    with pytest.raises(ValueError, match="gate differs"):
        validate_frozen_rows(frozen, source_by_id, role="visual")


def test_join_rejects_rejected_target_with_retained_scoring_text(tmp_path):
    _, target, _, _, target_root, _, _ = freeze_fixture(tmp_path)
    frozen = read_jsonl(target_root / "frozen_target_annotations.jsonl")
    frozen[0]["candidate_eligibility"] = "no"
    frozen[0]["reason_codes"] = ["generic"]
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
    frozen[0]["annotation_sha256"] = canonical_sha256(
        {key: frozen[0][key] for key in label_keys}
    )
    frozen[0]["role_gate_passed"] = False
    frozen[0]["row_sha256"] = canonical_sha256(
        {key: value for key, value in frozen[0].items() if key != "row_sha256"}
    )
    source_by_id = validate_input_rows(target, role="target", expected_items=2)
    with pytest.raises(ValueError, match="retains text"):
        validate_frozen_rows(frozen, source_by_id, role="target")


def test_atomic_writer_sets_private_permissions(tmp_path):
    _, _, _, working, _ = fixture()
    path = tmp_path / "working.jsonl"
    write_jsonl_atomic(path, working)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert len(read_jsonl(path)) == 2
