import hashlib
import copy
import json
from pathlib import Path
import wave

import pytest

from scripts.acl6060_source_event_annotation_v2 import (
    validate_audio_annotation,
    verify_sequential_interaction_log,
)
from scripts.serve_acl6060_audio_annotation import (
    SequentialAudioSession,
    read_events,
)


def make_task(tmp_path: Path) -> tuple[dict, Path]:
    audio_root = tmp_path / "audio"
    audio_path = audio_root / "packets" / "opaque" / "causal_audio.wav"
    audio_path.parent.mkdir(parents=True)
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(10)
        output.writeframes(b"\x00\x00" * 30)
    task = {
        "schema_version": "acl6060_source_event_annotation_v2",
        "stage": "validator_audio_only_boundary_pass",
        "packet_id": "ACLDEV-opaque",
        "talk_id": "talk",
        "question_lock_sha256": "q" * 64,
        "source_question": "Which method is faster?",
        "source_options": [
            {"option_id": "VOPT-a", "text": "A"},
            {"option_id": "VOPT-b", "text": "B"},
        ],
        "option_order_sha256": "o" * 64,
        "audio_path": "packets/opaque/causal_audio.wav",
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "audio_context_start_sec": 0.0,
        "causal_audio_end_sec": 2.0,
        "t_evidence_sec": 1.0,
        "prefix_step_sec": 0.5,
        "validator_id": "audio_validator_a",
        "question_only_status": "pending",
        "question_only_answer_option_id": None,
        "question_only_submitted_at_utc": None,
        "question_only_lock_sha256": None,
        "audio_annotation_status": "pending",
        "prefix_judgments": [],
        "audio_note": "",
        "audio_submitted_at_utc": None,
        "interaction_log_tail_sha256": None,
        "interaction_log_sha256": None,
        "sequential_delivery_backend": None,
        "audio_annotation_lock_sha256": None,
    }
    return task, audio_root


def test_sequential_session_gates_audio_and_exports_auditable_sheet(tmp_path):
    task, audio_root = make_task(tmp_path)
    event_log = tmp_path / "events.jsonl"
    output = tmp_path / "completed.jsonl"
    session = SequentialAudioSession([task], audio_root, event_log, output)

    with pytest.raises(ValueError, match="Audio is locked"):
        session.current_audio(task["packet_id"])
    session.submit_question(
        {
            "packet_id": task["packet_id"],
            "status": "not_answerable",
            "option_id": None,
        }
    )
    audio = session.current_audio(task["packet_id"])
    clipped = tmp_path / "clipped.wav"
    clipped.write_bytes(audio)
    with wave.open(str(clipped), "rb") as source:
        assert source.getnframes() == 10

    with pytest.raises(ValueError, match="out of order"):
        session.submit_prefix(
            {
                "packet_id": task["packet_id"],
                "step_index": 1,
                "status": "insufficient",
                "option_id": None,
            }
        )
    while not session.state()["complete"]:
        state = session.state()
        if state["next_step_index"] > 0:
            session.current_audio(task["packet_id"])
        session.submit_prefix(
            {
                "packet_id": task["packet_id"],
                "step_index": state["next_step_index"],
                "status": "insufficient",
                "option_id": None,
            }
        )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["sequential_delivery_backend"] == "acl6060_audio_gate_v1"
    assert len(rows[0]["prefix_judgments"]) == 3
    validate_audio_annotation(rows[0])
    verify_sequential_interaction_log(rows, event_log)
    assert len(read_events(event_log)) == 9
    tampered = copy.deepcopy(rows)
    tampered[0]["prefix_judgments"][0].update(
        {"status": "option", "option_id": "VOPT-a"}
    )
    validate_audio_annotation(tampered[0])
    with pytest.raises(ValueError, match="Invalid interaction step"):
        verify_sequential_interaction_log(tampered, event_log)

    lines = event_log.read_text().splitlines()
    event_log.write_text("\n".join(lines[:-1]) + "\n")
    output.unlink()
    SequentialAudioSession([task], audio_root, event_log, output)
    recovered_events = read_events(event_log)
    assert recovered_events[-1]["event_type"] == "item_completed"
    assert recovered_events[-1]["recovered_after_restart"] is True
    assert output.is_file()


def test_event_chain_rejects_tampering(tmp_path):
    task, audio_root = make_task(tmp_path)
    event_log = tmp_path / "events.jsonl"
    SequentialAudioSession([task], audio_root, event_log, tmp_path / "out.jsonl")
    row = json.loads(event_log.read_text().splitlines()[0])
    row["validator_id"] = "changed"
    event_log.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Event hash mismatch"):
        read_events(event_log)
