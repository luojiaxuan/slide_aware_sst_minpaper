import hashlib
import json
from pathlib import Path
import wave

from scripts.materialize_acl6060_source_event_workspace import materialize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialize_workspace_clips_audio_and_keeps_annotation_source_only(tmp_path):
    portable_root = tmp_path / "portable"
    acl_root = tmp_path / "acl"
    frame_path = portable_root / "frames" / "frame.jpg"
    audio_path = acl_root / "dev" / "full_wavs" / "talk.wav"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"jpeg fixture")
    audio_path.parent.mkdir(parents=True)
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(10)
        audio.writeframes(b"\x01\x00" * 100)
    seed = {
        "packet_id": "talk:A001",
        "talk_id": "talk",
        "split": "dev",
        "frame_path": "frames/frame.jpg",
        "frame_sha256": sha256(frame_path),
        "audio_id": "talk.wav",
        "audio_sha256": sha256(audio_path),
        "suggested_audio_window_start_sec": 2.0,
        "suggested_audio_window_end_sec": 7.0,
        "t_evidence_sec": 3.0,
        "event_status": "pending",
        "annotator_id": None,
        "source_question": None,
        "source_options": [],
        "source_answer_index": None,
        "evidence_subtypes": [],
        "evidence_region": None,
        "term_or_entity": None,
        "negative_labels": [],
        "annotation_note": "",
    }
    output_root = tmp_path / "workspace"
    rows, summary = materialize(
        seed_rows=[seed],
        acl_root=acl_root,
        portable_root=portable_root,
        output_root=output_root,
    )
    assert summary["packet_count"] == 1
    assert summary["audio_duration_sec"] == 5.0
    assert summary["source_transcript_included"] is False
    assert rows[0]["clip_t_evidence_sec"] == 1.0
    with wave.open(str(output_root / rows[0]["workspace_audio_path"]), "rb") as clip:
        assert clip.getnframes() == 50
    annotations = [
        json.loads(line)
        for line in (output_root / "annotations" / "annotator_a.jsonl")
        .read_text()
        .splitlines()
    ]
    assert annotations[0]["annotator_id"] == "annotator_a"
    assert not any(
        forbidden in key.lower()
        for key in annotations[0]
        for forbidden in ("reference", "target", "translation", "transcript")
    )
