import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.build_mcif_target_event_author_workspace import AUTHOR_SCHEMA
from scripts.build_mcif_visual_token_controls import canonical_sha256, file_sha256
from scripts.serve_mcif_target_event_authoring import TargetEventAuthoringSession


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "mcif_target_event_annotation_v1.json"


def config_kwargs() -> dict:
    return {
        "config_path": CONFIG_PATH,
        "expected_config_sha256": file_sha256(CONFIG_PATH),
    }


def source_row(media_path: Path, workspace: Path) -> dict:
    relative = media_path.relative_to(workspace).as_posix()
    row = {
        "schema_version": AUTHOR_SCHEMA,
        "status": "PENDING_HUMAN_EVENT_AUTHORING",
        "item_id": "MCIF-ZH-A0001",
        "source_reference_en": "Neural machine translation",
        "target_reference_zh": "神经机器翻译",
        "current_slide": {
            "path": relative,
            "sha256": file_sha256(media_path),
            "width": 24,
            "height": 12,
        },
        "current_slide_r0_text": "Neural machine translation",
        "current_slide_r1_text": "title: Neural machine translation",
        "candidate_options": [
            {
                "option_id": "O01",
                "source_candidate_en": "neural machine translation",
                "candidate_kind": "phrase",
                "token_count": 3,
                "lead_lower_bound_sec": 12.5,
                "lead_bin": "10_to_lt30",
            },
            {
                "option_id": "O02",
                "source_candidate_en": "machine translation",
                "candidate_kind": "phrase",
                "token_count": 2,
                "lead_lower_bound_sec": 12.5,
                "lead_bin": "10_to_lt30",
            },
        ],
        "annotation_status": "pending",
        "selected_option_id": None,
        "canonical_source_event_en": "",
        "acceptable_target_realizations_zh": [],
        "forbidden_target_realizations_zh": [],
        "target_reference_alignment": None,
        "slide_evidence_status": None,
        "annotation_note": "",
        "annotator_id": None,
        "locked_at_utc": None,
        "official_reference_consumed": True,
        "model_output_consumed": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def build_session(tmp_path: Path):
    workspace = tmp_path / "author_view"
    media = workspace / "media" / "slide.png"
    media.parent.mkdir(parents=True)
    Image.new("RGB", (24, 12), color=(10, 20, 30)).save(media)
    row = source_row(media, workspace)
    input_sheet = workspace / "annotation_items.jsonl"
    input_sheet.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    working_sheet = tmp_path / "working" / "target-author-01.jsonl"
    session = TargetEventAuthoringSession(
        input_sheet=input_sheet,
        expected_input_sha256=file_sha256(input_sheet),
        workspace_root=workspace,
        working_sheet=working_sheet,
        annotator_id="target-author-01",
        **config_kwargs(),
        expected_items=1,
    )
    return session, input_sheet, workspace, working_sheet


def eligible_payload() -> dict:
    return {
        "index": 0,
        "item_id": "MCIF-ZH-A0001",
        "annotation_status": "eligible",
        "selected_option_id": "O01",
        "canonical_source_event_en": "Neural machine translation",
        "acceptable_target_realizations_zh": ["神经机器翻译", "神经网络机器翻译"],
        "forbidden_target_realizations_zh": ["机器翻译"],
        "target_reference_alignment": "explicit",
        "slide_evidence_status": "supported",
        "annotation_note": "Exact target realization.",
    }


def test_session_saves_eligible_row_and_resumes(tmp_path):
    session, input_sheet, workspace, working_sheet = build_session(tmp_path)
    state = session.save(eligible_payload())
    assert state["completed_count"] == 1
    assert state["annotation_status"] == "eligible"
    assert state["acceptable_target_realizations_zh"] == [
        "神经机器翻译",
        "神经网络机器翻译",
    ]
    assert working_sheet.stat().st_mode & 0o777 == 0o600
    row = json.loads(working_sheet.read_text())
    assert row["selected_option_id"] == "O01"
    assert row["row_sha256"] == canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )

    resumed = TargetEventAuthoringSession(
        input_sheet=input_sheet,
        expected_input_sha256=file_sha256(input_sheet),
        workspace_root=workspace,
        working_sheet=working_sheet,
        annotator_id="target-author-01",
        **config_kwargs(),
        expected_items=1,
    )
    assert resumed.state(0)["canonical_source_event_en"] == (
        "Neural machine translation"
    )


def test_session_saves_negative_and_clears_scoring_answers(tmp_path):
    session, _, _, working_sheet = build_session(tmp_path)
    session.save(eligible_payload())
    state = session.save(
        {
            "index": 0,
            "item_id": "MCIF-ZH-A0001",
            "annotation_status": "no_target_alignment",
            "selected_option_id": "O01",
            "canonical_source_event_en": "should clear",
            "acceptable_target_realizations_zh": ["应清除"],
            "forbidden_target_realizations_zh": [],
            "target_reference_alignment": "omitted",
            "slide_evidence_status": "supported",
            "annotation_note": "Target omits the event.",
        }
    )
    assert state["annotation_status"] == "no_target_alignment"
    row = json.loads(working_sheet.read_text())
    assert row["selected_option_id"] is None
    assert row["canonical_source_event_en"] == ""
    assert row["acceptable_target_realizations_zh"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"selected_option_id": None},
        {"acceptable_target_realizations_zh": []},
        {"target_reference_alignment": "omitted"},
        {"slide_evidence_status": "ambiguous"},
        {"item_id": "wrong"},
    ],
)
def test_session_rejects_incomplete_or_misdirected_save(tmp_path, mutation):
    session, _, _, _ = build_session(tmp_path)
    payload = eligible_payload()
    payload.update(mutation)
    with pytest.raises(ValueError):
        session.save(payload)


def test_session_rejects_wrong_input_hash_and_media_byte_drift(tmp_path):
    session, input_sheet, workspace, working_sheet = build_session(tmp_path)
    with pytest.raises(ValueError, match="input sheet hash"):
        TargetEventAuthoringSession(
            input_sheet=input_sheet,
            expected_input_sha256="0" * 64,
            workspace_root=workspace,
            working_sheet=tmp_path / "wrong-hash.jsonl",
            annotator_id="target-author-01",
            **config_kwargs(),
            expected_items=1,
        )
    media = session.media_path(0)
    Image.new("RGB", (24, 12), color=(200, 0, 0)).save(media)
    with pytest.raises(ValueError, match="media hash"):
        TargetEventAuthoringSession(
            input_sheet=input_sheet,
            expected_input_sha256=file_sha256(input_sheet),
            workspace_root=workspace,
            working_sheet=working_sheet,
            annotator_id="target-author-01",
            **config_kwargs(),
            expected_items=1,
        )


def test_session_rejects_symlink_media_and_path_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (24, 12), color=(1, 2, 3)).save(outside)
    media = workspace / "slide.png"
    media.symlink_to(outside)
    row = source_row(outside, tmp_path)
    row["current_slide"]["path"] = "slide.png"
    row["row_sha256"] = canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )
    input_sheet = workspace / "input.jsonl"
    input_sheet.write_text(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        TargetEventAuthoringSession(
            input_sheet=input_sheet,
            expected_input_sha256=file_sha256(input_sheet),
            workspace_root=workspace,
            working_sheet=tmp_path / "working.jsonl",
            annotator_id="target-author-01",
            **config_kwargs(),
            expected_items=1,
        )

    media.unlink()
    row["current_slide"]["path"] = "../outside.png"
    row["row_sha256"] = canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )
    input_sheet.write_text(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="canonical and relative"):
        TargetEventAuthoringSession(
            input_sheet=input_sheet,
            expected_input_sha256=file_sha256(input_sheet),
            workspace_root=workspace,
            working_sheet=tmp_path / "escaped-working.jsonl",
            annotator_id="target-author-01",
            **config_kwargs(),
            expected_items=1,
        )


def test_resume_rejects_annotator_or_immutable_drift(tmp_path):
    session, input_sheet, workspace, working_sheet = build_session(tmp_path)
    with pytest.raises(ValueError, match="annotator differs"):
        TargetEventAuthoringSession(
            input_sheet=input_sheet,
            expected_input_sha256=file_sha256(input_sheet),
            workspace_root=workspace,
            working_sheet=working_sheet,
            annotator_id="target-author-02",
            **config_kwargs(),
            expected_items=1,
        )
    row = json.loads(working_sheet.read_text())
    row["target_reference_zh"] = "changed"
    row["row_sha256"] = canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )
    working_sheet.write_text(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="immutable input"):
        TargetEventAuthoringSession(
            input_sheet=input_sheet,
            expected_input_sha256=file_sha256(input_sheet),
            workspace_root=workspace,
            working_sheet=working_sheet,
            annotator_id="target-author-01",
            **config_kwargs(),
            expected_items=1,
        )


def test_next_pending_skips_completed_items(tmp_path):
    session, _, _, _ = build_session(tmp_path)
    assert session.next_pending(0) == 0
    session.save(eligible_payload())
    assert session.next_pending(0) == 0
