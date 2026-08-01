import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.acl6060_source_event_annotation_v2 import (
    canonical_hash,
    frame_task_lock_payload,
)
from scripts.serve_acl6060_frame_validation import FrameValidationSession, make_handler


def _config(path: Path) -> Path:
    payload = {
        "frame_validation": {
            "frame_support_status": [
                "pending",
                "supported",
                "unsupported",
                "ambiguous",
            ]
        },
        "evidence_subtypes": ["ocr_sufficient", "chart_relation"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(packet_id: str) -> dict:
    row = {
        "schema_version": "acl6060_source_event_annotation_v2",
        "stage": "validator_frame_only_answer_pass",
        "packet_id": packet_id,
        "question_lock_sha256": "question-lock",
        "source_question": "Which model has the highest score?",
        "source_options": [
            {"option_id": "option-a", "text": "Model A"},
            {"option_id": "option-b", "text": "Model B"},
        ],
        "option_order_sha256": "option-order",
        "frame_path": f"packets/{packet_id}/frame.jpg",
        "frame_binding_hmac": "FRAME-" + "a" * 64,
        "validator_id": "frame-validator-1",
        "frame_support_status": "pending",
        "frame_answer_option_id": None,
        "evidence_subtypes": [],
        "frame_confidence": None,
        "frame_note": "",
        "frame_submitted_at_utc": None,
        "frame_annotation_lock_sha256": None,
        "prior_stage_cohort_lock_sha256": "cohort-lock",
    }
    row["frame_task_lock_sha256"] = canonical_hash(frame_task_lock_payload(row))
    return row


def _session(tmp_path: Path) -> tuple[FrameValidationSession, Path]:
    workspace = tmp_path / "workspace"
    frame = workspace / "packets" / "packet-1" / "frame.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"fixture-jpeg")
    input_sheet = workspace / "frame_validation.jsonl"
    input_sheet.write_text(json.dumps(_row("packet-1")) + "\n", encoding="utf-8")
    working_sheet = tmp_path / "working.jsonl"
    session = FrameValidationSession(
        input_sheet=input_sheet,
        workspace_root=workspace,
        working_sheet=working_sheet,
        config_path=_config(tmp_path / "config.json"),
    )
    return session, working_sheet


def _request(
    server: ThreadingHTTPServer,
    method: str,
    path: str,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = HTTPConnection(*server.server_address, timeout=2)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


@pytest.fixture
def frame_server(tmp_path):
    session, working_sheet = _session(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, working_sheet
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_frame_validation_saves_supported_and_resumes(tmp_path):
    session, working_sheet = _session(tmp_path)
    state = session.save(
        {
            "index": 0,
            "packet_id": "packet-1",
            "frame_support_status": "supported",
            "frame_answer_option_id": "option-b",
            "evidence_subtypes": ["chart_relation"],
            "frame_confidence": "high",
            "frame_note": "  Clear   comparison. ",
        }
    )
    assert state["completed_count"] == 1
    row = json.loads(working_sheet.read_text(encoding="utf-8"))
    assert row["frame_answer_option_id"] == "option-b"
    assert row["frame_note"] == "Clear comparison."
    assert row["frame_submitted_at_utc"].endswith("Z")
    assert row["frame_annotation_lock_sha256"] is None
    assert working_sheet.stat().st_mode & 0o777 == 0o600

    resumed = FrameValidationSession(
        input_sheet=session.workspace_root / "frame_validation.jsonl",
        workspace_root=session.workspace_root,
        working_sheet=working_sheet,
        config_path=tmp_path / "config.json",
    )
    assert resumed.state(0)["frame_answer_option_id"] == "option-b"


def test_frame_validation_rejects_invalid_supported_and_clears_unsupported(tmp_path):
    session, working_sheet = _session(tmp_path)
    with pytest.raises(ValueError, match="Invalid frame answer"):
        session.save(
            {
                "index": 0,
                "packet_id": "packet-1",
                "frame_support_status": "supported",
                "frame_answer_option_id": None,
                "evidence_subtypes": ["chart_relation"],
                "frame_confidence": "medium",
                "frame_note": "",
            }
        )
    session.save(
        {
            "index": 0,
            "packet_id": "packet-1",
            "frame_support_status": "unsupported",
            "frame_answer_option_id": "option-a",
            "evidence_subtypes": ["ocr_sufficient"],
            "frame_confidence": "low",
            "frame_note": "No unique answer",
        }
    )
    row = json.loads(working_sheet.read_text(encoding="utf-8"))
    assert row["frame_answer_option_id"] is None
    assert row["evidence_subtypes"] == []


def test_frame_validation_rejects_task_drift_and_frame_escape(tmp_path):
    session, working_sheet = _session(tmp_path)
    row = json.loads(working_sheet.read_text(encoding="utf-8"))
    row["source_question"] = "Changed after task lock?"
    working_sheet.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable frame task"):
        FrameValidationSession(
            input_sheet=session.workspace_root / "frame_validation.jsonl",
            workspace_root=session.workspace_root,
            working_sheet=working_sheet,
            config_path=tmp_path / "config.json",
        )

    symlink_root = tmp_path / "symlink-workspace"
    (symlink_root / "packets" / "packet-1").mkdir(parents=True)
    (symlink_root / "actual.jpg").write_bytes(b"actual")
    (symlink_root / "packets" / "packet-1" / "frame.jpg").symlink_to(
        symlink_root / "actual.jpg"
    )
    symlink_input = symlink_root / "frame_validation.jsonl"
    symlink_input.write_text(json.dumps(_row("packet-1")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        FrameValidationSession(
            input_sheet=symlink_input,
            workspace_root=symlink_root,
            working_sheet=tmp_path / "symlink-working.jsonl",
            config_path=tmp_path / "config.json",
        )

    escaped = _row("packet-1")
    escaped["frame_path"] = "../outside.jpg"
    escaped["frame_task_lock_sha256"] = canonical_hash(
        frame_task_lock_payload(escaped)
    )
    (tmp_path / "outside.jpg").write_bytes(b"outside")
    escaped_input = session.workspace_root / "escaped.jsonl"
    escaped_input.write_text(json.dumps(escaped) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical and relative"):
        FrameValidationSession(
            input_sheet=escaped_input,
            workspace_root=session.workspace_root,
            working_sheet=tmp_path / "escaped-working.jsonl",
            config_path=tmp_path / "config.json",
        )


def test_frame_validation_http_routes_and_save(frame_server):
    server, working_sheet = frame_server
    status, headers, body = _request(server, "GET", "/api/item?index=0")
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    state = json.loads(body)
    assert state["packet_id"] == "packet-1"
    assert not ({"frame_path", "canonical_answer", "audio_path"} & state.keys())

    status, headers, body = _request(server, "GET", "/api/frame?index=0")
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert body == b"fixture-jpeg"

    payload = {
        "index": 0,
        "packet_id": "packet-1",
        "frame_support_status": "supported",
        "frame_answer_option_id": "option-a",
        "evidence_subtypes": ["ocr_sufficient"],
        "frame_confidence": "medium",
        "frame_note": "Visible in the title",
    }
    status, _, body = _request(
        server, "POST", "/api/item", json.dumps(payload).encode()
    )
    assert status == 200
    assert json.loads(body)["completed_count"] == 1
    assert json.loads(working_sheet.read_text(encoding="utf-8"))[
        "frame_answer_option_id"
    ] == "option-a"

    assert _request(server, "GET", "/api/item?index=9")[0] == 400
    assert _request(server, "GET", "/missing")[0] == 404
    assert _request(server, "POST", "/missing", b"{}")[0] == 404


def test_frame_validation_http_rejects_malformed_bodies(frame_server):
    server, _ = frame_server
    status, _, body = _request(server, "POST", "/api/item", b"{")
    assert status == 400
    assert "error" in json.loads(body)

    status, _, body = _request(server, "POST", "/api/item", b"[]")
    assert status == 400
    assert json.loads(body)["error"] == "Request body must be a JSON object"

    status, _, body = _request(server, "POST", "/api/item", b"x" * 65537)
    assert status == 400
    assert json.loads(body)["error"] == "Request body is too large"
