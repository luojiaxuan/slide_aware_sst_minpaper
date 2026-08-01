import hashlib
import json
import os
import struct
import threading
from pathlib import Path

import pytest

from slidesst.eval.causal_audio import (
    CausalAudioBroker,
    CausalAudioObservationCommitRequest,
    CausalAudioRequest,
    ThreadedUnixAudioServer,
    commit_causal_audio_observation,
    finalize_release_log,
    request_causal_audio_prefix,
    verify_causal_audio_source_bytes,
)
from slidesst.eval.event_timing import CausalAudioBrokerAudit, CausalAudioSchedule


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule_fixture(tmp_path: Path) -> tuple[CausalAudioSchedule, Path, Path]:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    pcm_path = audio_root / "talk-native.f32le"
    pcm = struct.pack("<8f", *[float(index) / 8.0 for index in range(8)])
    pcm_path.write_bytes(pcm)
    provenance_path = audio_root / "provenance.json"
    provenance_path.write_text('{"condition":"native"}\n', encoding="utf-8")
    schedule = CausalAudioSchedule.model_validate(
        {
            "schema_version": "acl6060_causal_audio_schedule_v3",
            "run_id": "run-1",
            "expected_conditions": ["audio_only"],
            "source_audio_roots": [str(audio_root)],
            "sources": [
                {
                    "source_id": "source:t1:native",
                    "talk_id": "t1",
                    "acoustic_condition": "native",
                    "source_pcm_path": str(pcm_path),
                    "source_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "pcm_format": "float32le_mono",
                    "sample_rate": 4,
                    "total_sample_count": 8,
                    "materialization_kind": "native",
                    "upstream_audio_sha256": "3" * 64,
                    "materializer_git_commit": "1" * 40,
                    "materializer_entrypoint_sha256": "2" * 64,
                    "source_provenance_path": str(provenance_path),
                    "source_provenance_sha256": file_sha256(provenance_path),
                }
            ],
            "prefixes": [
                {
                    "source_id": "source:t1:native",
                    "event_id": "e1",
                    "acoustic_condition": "native",
                    "sequence_index": index,
                    "audio_time_sec": float(index + 1),
                    "prefix_id": f"prefix:e1:native:{index}",
                    "prefix_pcm_sha256": hashlib.sha256(pcm[: (index + 1) * 16]).hexdigest(),
                    "sample_rate": 4,
                    "sample_count": (index + 1) * 4,
                }
                for index in range(2)
            ],
        }
    )
    return schedule, pcm_path, provenance_path


def request(
    sequence_index: int,
    request_id: str,
    *,
    condition: str = "audio_only",
    acoustic_condition: str = "native",
) -> CausalAudioRequest:
    return CausalAudioRequest(
        run_id="run-1",
        session_id=f"session:{condition}:{acoustic_condition}",
        request_id=request_id,
        event_id="e1",
        condition=condition,
        acoustic_condition=acoustic_condition,
        sequence_index=sequence_index,
    )


def commit(
    sequence_index: int,
    request_id: str,
    *,
    condition: str = "audio_only",
    acoustic_condition: str = "native",
) -> CausalAudioObservationCommitRequest:
    return CausalAudioObservationCommitRequest(
        run_id="run-1",
        session_id=f"session:{condition}:{acoustic_condition}",
        request_id=request_id,
        event_id="e1",
        condition=condition,
        acoustic_condition=acoustic_condition,
        sequence_index=sequence_index,
        observation_sha256=hashlib.sha256(f"observation-{sequence_index}".encode()).hexdigest(),
    )


def test_unix_broker_enforces_monotonic_prefixes_and_writes_chained_log(tmp_path):
    schedule, _, _ = schedule_fixture(tmp_path)
    socket_path = Path("/tmp") / f"slidesst-{os.getpid()}-{tmp_path.name[-6:]}.sock"
    release_events_path = tmp_path / "release-events.jsonl"
    broker = CausalAudioBroker(schedule, release_events_path)
    server = ThreadedUnixAudioServer(str(socket_path), broker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        header0, pcm0 = request_causal_audio_prefix(socket_path, request(0, "request-0"))
        assert header0["sample_count"] == 4
        assert len(pcm0) == 16
        with pytest.raises(RuntimeError, match="next monotonic prefix"):
            request_causal_audio_prefix(socket_path, request(0, "request-duplicate"))
        with pytest.raises(RuntimeError, match="prior hypothesis"):
            request_causal_audio_prefix(socket_path, request(1, "request-too-early"))
        commit0 = commit_causal_audio_observation(socket_path, commit(0, "commit-0"))
        assert commit0["record_type"] == "observation_commit"
        header1, pcm1 = request_causal_audio_prefix(socket_path, request(1, "request-1"))
        assert header1["sample_count"] == 8
        assert len(pcm1) == 32
        commit_causal_audio_observation(socket_path, commit(1, "commit-1"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        broker.close()
        socket_path.unlink(missing_ok=True)

    rows = [json.loads(line) for line in release_events_path.read_text().splitlines()]
    assert [row["server_ordinal"] for row in rows] == [0, 1, 2, 3]
    assert [row["record_type"] for row in rows] == [
        "prefix_release",
        "observation_commit",
        "prefix_release",
        "observation_commit",
    ]
    assert rows[0]["previous_record_sha256"] == hashlib.sha256(b"").hexdigest()
    assert rows[1]["previous_record_sha256"] == rows[0]["record_sha256"]


def test_source_verification_and_release_finalization_bind_exact_bytes(tmp_path):
    schedule, pcm_path, _ = schedule_fixture(tmp_path)
    verify_causal_audio_source_bytes(schedule)
    corrupt = schedule.model_dump()
    corrupt["prefixes"][0]["audio_time_sec"] = 0.5
    with pytest.raises(ValueError, match="sample boundary"):
        CausalAudioSchedule.model_validate(corrupt)

    release_events_path = tmp_path / "release-events.jsonl"
    broker = CausalAudioBroker(schedule, release_events_path)
    broker.release(request(0, "request-0"))
    broker.commit_observation(commit(0, "commit-0"))
    broker.release(request(1, "request-1"))
    broker.commit_observation(commit(1, "commit-1"))
    broker.close()
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(schedule.model_dump_json(indent=2) + "\n", encoding="utf-8")
    audit = CausalAudioBrokerAudit(
        schema_version="acl6060_causal_audio_broker_audit_v2",
        run_id="run-1",
        schedule_sha256=file_sha256(schedule_path),
        broker_git_commit="1" * 40,
        broker_repo_path=str(tmp_path),
        broker_entrypoint_path=str(tmp_path / "broker.py"),
        broker_entrypoint_sha256="2" * 64,
        broker_command="python broker.py --run-id run-1",
        broker_pid=123,
        socket_path=str(tmp_path / "broker.sock"),
        release_events_path=str(release_events_path),
        source_audio_roots=schedule.source_audio_roots,
        delivery_protocol="length_prefixed_unix_socket_v1",
        captured_at_utc="2026-08-01T12:00:00Z",
    )
    audit_path = tmp_path / "broker-audit.json"
    audit_path.write_text(audit.model_dump_json(indent=2) + "\n", encoding="utf-8")
    output_path = tmp_path / "release-log.json"
    release_log = finalize_release_log(
        schedule_path=schedule_path,
        broker_audit_path=audit_path,
        release_events_path=release_events_path,
        output_path=output_path,
    )
    assert len(release_log.releases) == 2
    assert len(release_log.observation_commits) == 2

    pcm_path.write_bytes(b"\0" * pcm_path.stat().st_size)
    with pytest.raises(ValueError, match="source hash mismatch"):
        verify_causal_audio_source_bytes(schedule)


def test_broker_blocks_cross_condition_future_audio_until_frontier_commits(tmp_path):
    schedule, _, _ = schedule_fixture(tmp_path)
    payload = schedule.model_dump()
    payload["expected_conditions"] = ["audio_only", "ocr"]
    schedule = CausalAudioSchedule.model_validate(payload)
    broker = CausalAudioBroker(schedule, tmp_path / "cross-condition-events.jsonl")
    try:
        broker.release(request(0, "audio-release-0"))
        with pytest.raises(ValueError, match="same-time talk observation"):
            broker.release(request(0, "ocr-release-0-too-early", condition="ocr"))
        broker.commit_observation(commit(0, "audio-commit-0"))
        with pytest.raises(ValueError, match="synchronized talk frontier"):
            broker.release(request(1, "audio-release-1-too-early"))
        broker.release(request(0, "ocr-release-0", condition="ocr"))
        broker.commit_observation(commit(0, "ocr-commit-0", condition="ocr"))
        broker.release(request(1, "audio-release-1"))
    finally:
        broker.close()


def test_broker_serializes_same_time_acoustic_streams(tmp_path):
    schedule, pcm_path, _ = schedule_fixture(tmp_path)
    payload = schedule.model_dump()
    noisy_pcm_path = pcm_path.with_name("talk-noisy.f32le")
    noisy_pcm = bytes(reversed(pcm_path.read_bytes()))
    noisy_pcm_path.write_bytes(noisy_pcm)
    noisy_source = dict(payload["sources"][0])
    noisy_source.update(
        {
            "source_id": "source:t1:noisy",
            "acoustic_condition": "noisy",
            "source_pcm_path": str(noisy_pcm_path),
            "source_pcm_sha256": hashlib.sha256(noisy_pcm).hexdigest(),
            "materialization_kind": "generic_noise",
        }
    )
    payload["sources"].append(noisy_source)
    for prefix in list(payload["prefixes"]):
        noisy_prefix = dict(prefix)
        noisy_prefix.update(
            {
                "source_id": "source:t1:noisy",
                "acoustic_condition": "noisy",
                "prefix_id": prefix["prefix_id"].replace(":native:", ":noisy:"),
                "prefix_pcm_sha256": hashlib.sha256(
                    noisy_pcm[: prefix["sample_count"] * 4]
                ).hexdigest(),
            }
        )
        payload["prefixes"].append(noisy_prefix)
    schedule = CausalAudioSchedule.model_validate(payload)
    broker = CausalAudioBroker(schedule, tmp_path / "multi-acoustic-events.jsonl")
    try:
        broker.release(request(0, "native-release-0"))
        with pytest.raises(ValueError, match="same-time talk observation"):
            broker.release(
                request(
                    0,
                    "noisy-release-0-too-early",
                    acoustic_condition="noisy",
                )
            )
        broker.commit_observation(commit(0, "native-commit-0"))
        broker.release(
            request(0, "noisy-release-0", acoustic_condition="noisy")
        )
        broker.commit_observation(
            commit(0, "noisy-commit-0", acoustic_condition="noisy")
        )
        broker.release(request(1, "native-release-1"))
    finally:
        broker.close()
