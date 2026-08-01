import json
from pathlib import Path

import pytest

from scripts.serve_acl6060_authoring import AuthoringSession


def _config(path: Path) -> Path:
    payload = {
        "question_authoring": {
            "authoring_status": [
                "pending",
                "candidate",
                "negative_no_visual_question",
                "exclude_other",
            ],
            "forced_choice_min_options": 2,
            "forced_choice_max_options": 4,
        },
        "evidence_subtypes": ["ocr_sufficient", "chart_relation"],
        "negative_labels": ["visual_not_unique", "no_future_spoken_answer"],
        "exclusion_labels": ["media_error", "other"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(packet_id: str) -> dict:
    return {
        "schema_version": "acl6060_source_event_annotation_v2",
        "packet_id": packet_id,
        "frame_path": f"packets/{packet_id}/frame.jpg",
        "frame_binding_hmac": "FRAME-" + "a" * 64,
        "author_id": "author_pending",
        "stage": "frame_only_question_authoring",
        "authoring_status": "pending",
        "source_question": None,
        "source_options": [],
        "canonical_answer_index": None,
        "evidence_subtypes": [],
        "evidence_region": None,
        "term_or_entity": None,
        "negative_labels": [],
        "exclusion_labels": [],
        "authoring_note": "",
        "authoring_completed_at_utc": None,
        "question_locked_at_utc": None,
        "question_lock_sha256": None,
        "authoring_lock_sha256": None,
    }


def _session(tmp_path: Path) -> tuple[AuthoringSession, Path]:
    workspace = tmp_path / "workspace"
    frame = workspace / "packets" / "packet-1" / "frame.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"fixture-jpeg")
    input_sheet = workspace / "authoring.jsonl"
    input_sheet.write_text(json.dumps(_row("packet-1")) + "\n", encoding="utf-8")
    working_sheet = tmp_path / "working.jsonl"
    session = AuthoringSession(
        input_sheet=input_sheet,
        workspace_root=workspace,
        working_sheet=working_sheet,
        config_path=_config(tmp_path / "config.json"),
        author_id="author-1",
    )
    return session, working_sheet


def test_authoring_session_saves_candidate_and_resumes(tmp_path):
    session, working_sheet = _session(tmp_path)
    state = session.save(
        {
            "index": 0,
            "packet_id": "packet-1",
            "authoring_status": "candidate",
            "source_question": "Which model is shown?",
            "source_options": ["Model A", "Model B"],
            "canonical_answer_index": 1,
            "evidence_subtypes": ["ocr_sufficient"],
            "evidence_region": {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3},
            "term_or_entity": "Model B",
            "negative_labels": ["visual_not_unique"],
            "exclusion_labels": ["media_error"],
            "authoring_note": "Readable title",
        }
    )
    assert state["completed_count"] == 1
    row = json.loads(working_sheet.read_text(encoding="utf-8"))
    assert row["author_id"] == "author-1"
    assert row["canonical_answer_index"] == 1
    assert row["negative_labels"] == []
    assert row["question_locked_at_utc"].endswith("Z")
    assert working_sheet.stat().st_mode & 0o777 == 0o600

    resumed = AuthoringSession(
        input_sheet=session.workspace_root / "authoring.jsonl",
        workspace_root=session.workspace_root,
        working_sheet=working_sheet,
        config_path=tmp_path / "config.json",
        author_id="author-1",
    )
    assert resumed.state(0)["source_question"] == "Which model is shown?"


def test_authoring_session_saves_negative_and_rejects_invalid_candidate(tmp_path):
    session, working_sheet = _session(tmp_path)
    with pytest.raises(ValueError, match="canonical answer"):
        session.save(
            {
                "index": 0,
                "packet_id": "packet-1",
                "authoring_status": "candidate",
                "source_question": "Question",
                "source_options": ["A", "B"],
                "canonical_answer_index": None,
                "evidence_subtypes": ["ocr_sufficient"],
                "evidence_region": {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3},
                "term_or_entity": "A",
                "authoring_note": "",
            }
        )
    session.save(
        {
            "index": 0,
            "packet_id": "packet-1",
            "authoring_status": "negative_no_visual_question",
            "negative_labels": ["visual_not_unique"],
            "exclusion_labels": [],
            "authoring_note": "No unique visual answer",
        }
    )
    row = json.loads(working_sheet.read_text(encoding="utf-8"))
    assert row["source_question"] is None
    assert row["negative_labels"] == ["visual_not_unique"]


def test_authoring_session_rejects_task_drift_and_frame_escape(tmp_path):
    session, working_sheet = _session(tmp_path)
    rows = [json.loads(working_sheet.read_text(encoding="utf-8"))]
    rows[0]["packet_id"] = "changed"
    working_sheet.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable authoring task"):
        AuthoringSession(
            input_sheet=session.workspace_root / "authoring.jsonl",
            workspace_root=session.workspace_root,
            working_sheet=working_sheet,
            config_path=tmp_path / "config.json",
            author_id="author-1",
        )

    symlink_root = tmp_path / "symlink-workspace"
    (symlink_root / "packets" / "packet-1").mkdir(parents=True)
    (symlink_root / "actual.jpg").write_bytes(b"actual")
    (symlink_root / "packets" / "packet-1" / "frame.jpg").symlink_to(
        symlink_root / "actual.jpg"
    )
    symlink_input = symlink_root / "authoring.jsonl"
    symlink_input.write_text(json.dumps(_row("packet-1")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        AuthoringSession(
            input_sheet=symlink_input,
            workspace_root=symlink_root,
            working_sheet=tmp_path / "symlink-working.jsonl",
            config_path=tmp_path / "config.json",
            author_id="author-1",
        )

    escaped = _row("packet-1")
    escaped["frame_path"] = "../outside.jpg"
    (tmp_path / "outside.jpg").write_bytes(b"outside")
    escaped_input = session.workspace_root / "escaped.jsonl"
    escaped_input.write_text(json.dumps(escaped) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical and relative"):
        AuthoringSession(
            input_sheet=escaped_input,
            workspace_root=session.workspace_root,
            working_sheet=tmp_path / "escaped-working.jsonl",
            config_path=tmp_path / "config.json",
            author_id="author-1",
        )
