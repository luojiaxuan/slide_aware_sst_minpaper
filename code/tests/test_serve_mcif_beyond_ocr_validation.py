import copy
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.build_mcif_beyond_ocr_validation_workspace import TARGET_SCHEMA, VISUAL_SCHEMA
from scripts.build_mcif_visual_token_controls import canonical_sha256, file_sha256
from scripts.serve_mcif_beyond_ocr_validation import (
    BeyondOcrValidationSession,
    page_html,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "mcif_beyond_ocr_validation_v1.json"


def hashed(row):
    result = dict(row)
    result["row_sha256"] = canonical_sha256(result)
    return result


def visual_row(media_sha256: str):
    return hashed(
        {
            "schema_version": VISUAL_SCHEMA,
            "status": "PENDING_INDEPENDENT_VISUAL_VALIDATION",
            "item_id": "MCIF-BOV-V0001",
            "candidate_source_en": "semantic bridge",
            "candidate_kind": "phrase",
            "candidate_token_count": 2,
            "evidence_channel": "raw_visual_semantics",
            "current_slide": {
                "media_id": "M0001",
                "path": "media/M0001.png",
                "sha256": media_sha256,
                "width": 32,
                "height": 18,
            },
            "current_slide_r0_text": "flat OCR",
            "current_slide_r1_blocks": [],
            "proposed_evidence_origins": [
                {
                    "descriptor_field": "scene_summary",
                    "descriptor_index": 0,
                    "descriptor_text": "A semantic bridge connects nodes.",
                    "descriptor_sha256": "a" * 64,
                }
            ],
            "requires_r1_insufficiency_judgment": True,
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
            "generative_model_output_exposed": True,
        }
    )


def target_row():
    return hashed(
        {
            "schema_version": TARGET_SCHEMA,
            "status": "PENDING_INDEPENDENT_TARGET_EVENT_AUTHORING",
            "item_id": "MCIF-BOV-T0001",
            "candidate_source_en": "semantic bridge",
            "candidate_kind": "phrase",
            "candidate_token_count": 2,
            "source_reference_en": "The semantic bridge connects nodes.",
            "target_reference_zh": "语义桥连接节点。",
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


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def visual_session(tmp_path: Path):
    workspace = tmp_path / "visual"
    media = workspace / "media" / "M0001.png"
    media.parent.mkdir(parents=True)
    Image.new("RGB", (32, 18), color=(10, 20, 30)).save(media)
    input_path = workspace / "validation_items.jsonl"
    write_jsonl(input_path, [visual_row(file_sha256(media))])
    working = tmp_path / "working" / "visual.jsonl"
    session = BeyondOcrValidationSession(
        role="visual",
        input_sheet=input_path,
        expected_input_sha256=file_sha256(input_path),
        workspace_root=workspace,
        working_sheet=working,
        annotator_id="visual-01",
        config_path=CONFIG_PATH,
        expected_config_sha256=file_sha256(CONFIG_PATH),
        expected_items=1,
    )
    return session, input_path, working, media


def target_session(tmp_path: Path):
    workspace = tmp_path / "target"
    input_path = workspace / "annotation_items.jsonl"
    write_jsonl(input_path, [target_row()])
    working = tmp_path / "working" / "target.jsonl"
    session = BeyondOcrValidationSession(
        role="target",
        input_sheet=input_path,
        expected_input_sha256=file_sha256(input_path),
        workspace_root=workspace,
        working_sheet=working,
        annotator_id="target-01",
        config_path=CONFIG_PATH,
        expected_config_sha256=file_sha256(CONFIG_PATH),
        expected_items=1,
    )
    return session, input_path, working


def test_visual_session_initializes_private_working_sheet_and_saves(tmp_path):
    session, _, working, media = visual_session(tmp_path)
    assert session.state(0)["progress"] == {"completed": 0, "total": 1}
    assert session.media("media/M0001.png") == media
    assert oct(working.stat().st_mode & 0o777) == "0o600"
    saved = session.save(
        {
            "index": 0,
            "annotation_status": "completed",
            "visual_evidence_correct": "yes",
            "candidate_supported_by_visual_evidence": "yes",
            "r0_insufficient": "yes",
            "r1_insufficient": "yes",
            "reason_codes": [],
            "annotation_note": "",
        }
    )
    assert saved["annotation_status"] == "completed"
    assert session.progress() == {"completed": 1, "total": 1}
    assert json.loads(working.read_text().strip())["r1_insufficient"] == "yes"


def test_target_session_saves_complete_target_definition(tmp_path):
    session, _, working = target_session(tmp_path)
    saved = session.save(
        {
            "index": 0,
            "annotation_status": "completed",
            "candidate_eligibility": "yes",
            "canonical_source_event_en": "semantic bridge",
            "acceptable_target_realizations_zh": ["语义桥"],
            "forbidden_target_realizations_zh": [],
            "target_reference_alignment": "explicit",
            "reason_codes": [],
            "annotation_note": "",
        }
    )
    assert saved["candidate_eligibility"] == "yes"
    assert oct(working.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(FileNotFoundError):
        session.media("media/M0001.png")


def test_save_rejects_cross_role_payload_and_partial_completion(tmp_path):
    visual, _, _, _ = visual_session(tmp_path / "visual-case")
    with pytest.raises(ValueError, match="forbidden fields"):
        visual.save({"index": 0, "source_reference_en": "leak"})
    with pytest.raises(ValueError, match="lacks a judgment"):
        visual.save(
            {
                "index": 0,
                "annotation_status": "completed",
                "visual_evidence_correct": "yes",
            }
        )
    target, _, _ = target_session(tmp_path / "target-case")
    with pytest.raises(ValueError, match="forbidden fields"):
        target.save({"index": 0, "current_slide": {}})


def test_session_resumes_valid_working_sheet_and_rejects_annotator_drift(tmp_path):
    session, input_path, working, _ = visual_session(tmp_path)
    resumed = BeyondOcrValidationSession(
        role="visual",
        input_sheet=input_path,
        expected_input_sha256=file_sha256(input_path),
        workspace_root=input_path.parent,
        working_sheet=working,
        annotator_id="visual-01",
        config_path=CONFIG_PATH,
        expected_config_sha256=file_sha256(CONFIG_PATH),
        expected_items=1,
    )
    assert resumed.progress()["completed"] == 0
    with pytest.raises(ValueError, match="annotator/hash differs"):
        BeyondOcrValidationSession(
            role="visual",
            input_sheet=input_path,
            expected_input_sha256=file_sha256(input_path),
            workspace_root=input_path.parent,
            working_sheet=working,
            annotator_id="other",
            config_path=CONFIG_PATH,
            expected_config_sha256=file_sha256(CONFIG_PATH),
            expected_items=1,
        )


def test_session_rejects_input_hash_media_hash_and_symlink(tmp_path):
    session, input_path, working, media = visual_session(tmp_path)
    with pytest.raises(ValueError, match="input hash differs"):
        BeyondOcrValidationSession(
            role="visual",
            input_sheet=input_path,
            expected_input_sha256="0" * 64,
            workspace_root=input_path.parent,
            working_sheet=working,
            annotator_id="visual-01",
            config_path=CONFIG_PATH,
            expected_config_sha256=file_sha256(CONFIG_PATH),
            expected_items=1,
        )
    Image.new("RGB", (32, 18), color=(255, 0, 0)).save(media)
    working.unlink()
    with pytest.raises(ValueError, match="media hash differs"):
        BeyondOcrValidationSession(
            role="visual",
            input_sheet=input_path,
            expected_input_sha256=file_sha256(input_path),
            workspace_root=input_path.parent,
            working_sheet=working,
            annotator_id="visual-01",
            config_path=CONFIG_PATH,
            expected_config_sha256=file_sha256(CONFIG_PATH),
            expected_items=1,
        )
    link = tmp_path / "workspace-link"
    link.symlink_to(input_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        BeyondOcrValidationSession(
            role="visual",
            input_sheet=input_path,
            expected_input_sha256=file_sha256(input_path),
            workspace_root=link,
            working_sheet=working,
            annotator_id="visual-01",
            config_path=CONFIG_PATH,
            expected_config_sha256=file_sha256(CONFIG_PATH),
            expected_items=1,
        )


def test_role_pages_render_distinct_surfaces_without_nested_layouts():
    visual = page_html("visual")
    target = page_html("target")
    assert 'id="slide"' in visual and 'id="source"' not in visual
    assert 'id="source"' in target and 'id="slide"' not in target
    assert "visual-layout" in visual
    assert "target-layout" in target
    assert "border-radius:0" in visual
    assert "split(/\\n/)" in visual
    assert "join('\\n')" in target
